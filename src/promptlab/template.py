"""A small, dependency-free template engine for prompts.

The engine treats a prompt like source code: variables are explicit, missing
inputs fail loudly, and partials let you share a common block (a safety clause,
an output contract) across many prompts.

Syntax
------
- Variable:      ``{{ name }}`` or ``{{ user.name }}`` (dotted access)
- Comment:       ``{{! ignored }}``
- Partial:       ``{{> partial_name }}``
- If block:      ``{{#if flag}} ... {{else}} ... {{/if}}``
- Unless block:  ``{{#unless flag}} ... {{/unless}}``
- Each block:    ``{{#each items}} ... {{ this }} / {{ field }} ... {{/each}}``

Design decisions
----------------
- Interpolating a variable that is *absent from the context* raises
  ``MissingVariableError`` in strict mode (the default). Renaming or dropping a
  variable then fails immediately instead of silently rendering an empty string
  and shipping a broken prompt. See ``find_variables`` / ``missing_variables``
  for the static side of the same check.
- A variable that is present but ``None`` renders as an empty string (it was
  provided on purpose, so we do not treat it as missing).
- Block conditions treat a missing variable as falsy; iterating a missing or
  empty collection yields nothing. That is what makes optional sections work.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = [
    "Template",
    "render",
    "few_shot",
    "find_variables",
    "missing_variables",
    "variable_references",
    "VariableRef",
    "TemplateError",
    "TemplateSyntaxError",
    "MissingVariableError",
    "PartialNotFoundError",
]


class TemplateError(Exception):
    """Base class for every template error."""


class TemplateSyntaxError(TemplateError):
    """Raised for malformed template syntax (unbalanced or unknown blocks)."""


class MissingVariableError(TemplateError):
    """Raised in strict mode when an interpolated variable is absent."""

    def __init__(self, path):
        self.path = path
        super().__init__("missing template variable: {!r}".format(path))


class PartialNotFoundError(TemplateError):
    """Raised when ``{{> name }}`` references a partial that was not supplied."""

    def __init__(self, name):
        self.name = name
        super().__init__("unknown partial: {!r}".format(name))


_MISSING = object()
_TAG_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def _classify(inner):
    s = inner.strip()
    if not s:
        raise TemplateSyntaxError("empty tag {{}}")
    if s.startswith("!"):
        return ("comment",)
    if s.startswith(">"):
        name = s[1:].strip()
        if not name:
            raise TemplateSyntaxError("partial tag requires a name")
        return ("partial", name)
    if s.startswith("#"):
        body = s[1:].strip()
        parts = body.split(None, 1)
        kind = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if kind not in ("if", "unless", "each"):
            raise TemplateSyntaxError("unknown block type: {!r}".format(kind))
        if not arg:
            raise TemplateSyntaxError("block #{} requires an argument".format(kind))
        return ("open", kind, arg)
    if s.startswith("/"):
        kind = s[1:].strip()
        if kind not in ("if", "unless", "each"):
            raise TemplateSyntaxError("unknown closing block: {!r}".format(kind))
        return ("close", kind)
    if s == "else":
        return ("else",)
    return ("var", s)


def _tokenize(source):
    tokens = []
    pos = 0
    for m in _TAG_RE.finditer(source):
        if m.start() > pos:
            tokens.append(("text", source[pos:m.start()]))
        tokens.append(_classify(m.group(1)))
        pos = m.end()
    if pos < len(source):
        tokens.append(("text", source[pos:]))
    return tokens


# --------------------------------------------------------------------------- #
# Parser -> tree of tuples and block dicts
# --------------------------------------------------------------------------- #
def _parse(tokens):
    root = []
    current = root
    stack = []  # entries: (node_dict, parent_list)
    for token in tokens:
        typ = token[0]
        if typ == "text":
            current.append(("text", token[1]))
        elif typ == "var":
            current.append(("var", token[1]))
        elif typ == "partial":
            current.append(("partial", token[1]))
        elif typ == "comment":
            continue
        elif typ == "open":
            _kind, _arg = token[1], token[2]
            node = {
                "type": "block",
                "kind": _kind,
                "arg": _arg,
                "children": [],
                "else": None,
            }
            current.append(node)
            stack.append((node, current))
            current = node["children"]
        elif typ == "else":
            if not stack:
                raise TemplateSyntaxError("{{else}} outside of a block")
            node = stack[-1][0]
            if node["else"] is not None:
                raise TemplateSyntaxError("duplicate {{else}} in block")
            node["else"] = []
            current = node["else"]
        elif typ == "close":
            if not stack:
                raise TemplateSyntaxError("unexpected {{/%s}}" % token[1])
            node, parent = stack.pop()
            if node["kind"] != token[1]:
                raise TemplateSyntaxError(
                    "mismatched block: opened #{} but closed /{}".format(
                        node["kind"], token[1]
                    )
                )
            current = parent
    if stack:
        raise TemplateSyntaxError(
            "unclosed block #{}".format(stack[-1][0]["kind"])
        )
    return root


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _resolve(path, ctx_stack):
    path = path.strip()
    if path in (".", "this"):
        return ctx_stack[-1][1]
    parts = path.split(".")
    first = parts[0]
    if first == "this":
        base = ctx_stack[-1][1]
        rest = parts[1:]
    else:
        base = _MISSING
        for frame, _item in reversed(ctx_stack):
            if isinstance(frame, Mapping) and first in frame:
                base = frame[first]
                break
        rest = parts[1:]
    if base is _MISSING:
        return _MISSING
    for key in rest:
        if isinstance(base, Mapping) and key in base:
            base = base[key]
        elif hasattr(base, key):
            base = getattr(base, key)
        else:
            return _MISSING
    return base


def _truthy(value):
    if value is _MISSING or value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    return bool(value)


def _stringify(value, path, strict, collector=None):
    if value is _MISSING:
        if strict:
            if collector is not None:
                collector.append(path)
                return ""
            raise MissingVariableError(path)
        return ""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def _render_nodes(nodes, ctx_stack, partials, strict, out, seen, collector=None):
    """Render ``nodes`` into ``out``.

    With ``collector`` (a list), a missing variable in strict mode is appended
    to it instead of raising -- a dry run that finds *every* variable the real
    render would fail on, following exactly the same branches and loop items.
    """
    for node in nodes:
        if isinstance(node, tuple):
            kind = node[0]
            if kind == "text":
                out.append(node[1])
            elif kind == "var":
                out.append(_stringify(_resolve(node[1], ctx_stack), node[1], strict, collector))
            elif kind == "partial":
                name = node[1]
                if name not in partials:
                    raise PartialNotFoundError(name)
                if name in seen:
                    raise TemplateError("partial recursion detected: {!r}".format(name))
                subtree = _parse(_tokenize(partials[name]))
                _render_nodes(subtree, ctx_stack, partials, strict, out, seen | {name}, collector)
            continue

        # block node (dict)
        kind = node["kind"]
        value = _resolve(node["arg"], ctx_stack)
        if kind == "if":
            branch = node["children"] if _truthy(value) else (node["else"] or [])
            _render_nodes(branch, ctx_stack, partials, strict, out, seen, collector)
        elif kind == "unless":
            branch = node["children"] if not _truthy(value) else (node["else"] or [])
            _render_nodes(branch, ctx_stack, partials, strict, out, seen, collector)
        elif kind == "each":
            items = value if _truthy(value) else []
            if isinstance(items, Mapping):
                items = list(items.values())
            if items and not isinstance(items, (str, bytes)):
                for item in items:
                    frame = item if isinstance(item, Mapping) else {}
                    ctx_stack.append((frame, item))
                    _render_nodes(node["children"], ctx_stack, partials, strict, out, seen, collector)
                    ctx_stack.pop()
            else:
                _render_nodes(node["else"] or [], ctx_stack, partials, strict, out, seen, collector)


# --------------------------------------------------------------------------- #
# Static analysis (required-variable extraction)
# --------------------------------------------------------------------------- #
def _root(path):
    head = path.strip().split(".")[0]
    if head in ("", "this"):
        return None
    return head


def _collect(nodes, roots, partials, in_each, seen_partials):
    for node in nodes:
        if isinstance(node, tuple):
            kind = node[0]
            if kind == "var" and in_each == 0:
                r = _root(node[1])
                if r:
                    roots.add(r)
            elif kind == "partial" and partials:
                name = node[1]
                if name in partials and name not in seen_partials:
                    subtree = _parse(_tokenize(partials[name]))
                    _collect(subtree, roots, partials, in_each, seen_partials | {name})
            continue

        # block node
        if in_each == 0:
            r = _root(node["arg"])
            if r:
                roots.add(r)
        child_in_each = in_each + (1 if node["kind"] == "each" else 0)
        _collect(node["children"], roots, partials, child_in_each, seen_partials)
        if node["else"]:
            # The {{else}} of an #each renders when the list is empty, i.e. with
            # the *outer* context -- its variables are not loop-local.
            else_in_each = in_each if node["kind"] == "each" else child_in_each
            _collect(node["else"], roots, partials, else_in_each, seen_partials)


def find_variables(source, partials=None):
    """Return the set of root variable names the outer context must provide.

    Variables that only appear *inside* an ``#each`` block are considered
    loop-local (they come from the iterated items) and are not reported. The
    argument of an ``#each`` or ``#if`` at the top level *is* reported, because
    it is resolved against the outer context. When ``partials`` is supplied,
    variables referenced by included partials are included too.

    This is a purely static answer. Whether a name used inside a loop comes
    from the items or from the outer context depends on the data, so
    :func:`missing_variables` also dry-runs the template against the context.
    """
    tree = _parse(_tokenize(source))
    roots = set()
    _collect(tree, roots, partials or {}, 0, frozenset())
    return roots


def _dry_run_missing(tree, context, partials):
    """Paths a strict render of ``tree`` with ``context`` would fail on."""
    collector = []
    ctx = context or {}
    try:
        _render_nodes(tree, [(ctx, ctx)], partials or {}, True, [], frozenset(), collector)
    except TemplateError:
        # Unknown or recursive partials are reported by render() itself; this
        # function only answers "which variables are missing".
        pass
    return collector


def _report_name(path, context):
    """Report the root when it is absent, else the full dotted path."""
    root = _root(path) or path.strip()
    return root if root not in (context or {}) else path.strip()


def missing_variables(source, context, partials=None):
    """Return the sorted variables a strict render with ``context`` would miss.

    Combines two checks:

    - the static one: every root variable :func:`find_variables` reports must
      be a key of ``context`` (conservative: it also covers branches that this
      particular context would not render);
    - a dry run of the real render, which catches what static analysis cannot
      see: outer variables used inside ``{{#each}}`` (``{{#each rules}}{{ prefix
      }}{{/each}}`` needs ``prefix`` when the items are strings) and dotted paths
      whose root exists but whose leaf does not (reported as ``user.email``).
    """
    ctx = context or {}
    missing = {v for v in find_variables(source, partials) if v not in ctx}
    for path in _dry_run_missing(_parse(_tokenize(source)), ctx, partials):
        missing.add(_report_name(path, ctx))
    return sorted(missing)


class VariableRef:
    """One variable reference found by :func:`variable_references`.

    Attributes
    ----------
    root / path:
        ``"user"`` / ``"user.name"``.
    kind:
        ``"var"`` for ``{{ x }}`` interpolation, ``"block"`` for the argument of
        ``#if`` / ``#unless`` / ``#each``.
    loops:
        Arguments of the enclosing ``#each`` blocks, outermost first. Empty
        when the reference is resolved against the outer context.
    guards:
        Roots of the enclosing conditions that must be truthy for this
        reference to render (``#if`` then-branch, ``#unless`` else-branch).
    partial:
        Name of the partial the reference comes from, or ``None``.
    """

    __slots__ = ("root", "path", "kind", "loops", "guards", "partial")

    def __init__(self, root, path, kind, loops=(), guards=frozenset(), partial=None):
        self.root = root
        self.path = path
        self.kind = kind
        self.loops = tuple(loops)
        self.guards = frozenset(guards)
        self.partial = partial

    @property
    def in_loop(self):
        return bool(self.loops)

    def __repr__(self):
        return "VariableRef({!r}, kind={!r}, loops={}, guards={})".format(
            self.path, self.kind, list(self.loops), sorted(self.guards)
        )


def _references(nodes, partials, loops, guards, partial, seen, out, unknown_partials):
    for node in nodes:
        if isinstance(node, tuple):
            kind = node[0]
            if kind == "var":
                r = _root(node[1])
                if r:
                    out.append(VariableRef(r, node[1].strip(), "var", loops, guards, partial))
            elif kind == "partial":
                name = node[1]
                if name not in partials:
                    unknown_partials.append(name)
                elif name not in seen:
                    subtree = _parse(_tokenize(partials[name]))
                    _references(subtree, partials, loops, guards, name, seen | {name}, out, unknown_partials)
            continue

        arg = node["arg"].strip()
        r = _root(arg)
        if r:
            out.append(VariableRef(r, arg, "block", loops, guards, partial))
        kind = node["kind"]
        if kind == "each":
            _references(node["children"], partials, loops + (arg,), guards, partial, seen, out, unknown_partials)
            if node["else"]:
                _references(node["else"], partials, loops, guards, partial, seen, out, unknown_partials)
        else:
            then_guards = guards | {r} if (r and kind == "if") else guards
            else_guards = guards | {r} if (r and kind == "unless") else guards
            _references(node["children"], partials, loops, then_guards, partial, seen, out, unknown_partials)
            if node["else"]:
                _references(node["else"], partials, loops, else_guards, partial, seen, out, unknown_partials)


def variable_references(source, partials=None):
    """Every variable reference in ``source`` with its loop and guard context.

    Returns ``(references, unknown_partials)`` where ``references`` is a list of
    :class:`VariableRef` (partials expanded) and ``unknown_partials`` lists the
    ``{{> name }}`` includes that are not in ``partials``. Used by
    ``promptlab lint`` to check a prompt's declared inputs against its body.
    Raises ``TemplateSyntaxError`` for malformed templates.
    """
    tree = _parse(_tokenize(source))
    refs, unknown = [], []
    _references(tree, partials or {}, (), frozenset(), None, frozenset(), refs, unknown)
    return refs, unknown


# --------------------------------------------------------------------------- #
# Few-shot helper
# --------------------------------------------------------------------------- #
def few_shot(
    examples,
    *,
    input_key="input",
    output_key="output",
    input_label="Input",
    output_label="Output",
    separator="\n\n",
):
    """Format a list of examples into a labelled few-shot block.

    ``examples`` items may be mappings (``{"input": ..., "output": ...}``) or
    two-tuples ``(input, output)``. The returned string is ready to drop into a
    prompt template as a single variable.
    """
    blocks = []
    for ex in examples:
        if isinstance(ex, Mapping):
            i = ex.get(input_key, "")
            o = ex.get(output_key, "")
        else:
            i, o = ex
        blocks.append("{}: {}\n{}: {}".format(input_label, i, output_label, o))
    return separator.join(blocks)


# --------------------------------------------------------------------------- #
# Public objects
# --------------------------------------------------------------------------- #
class Template:
    """A compiled, reusable prompt template."""

    def __init__(self, source, partials=None, name=None):
        self.source = source
        self.name = name
        self.partials = dict(partials or {})
        self._tree = _parse(_tokenize(source))

    @property
    def variables(self):
        """Root variable names the context must provide (see ``find_variables``)."""
        return find_variables(self.source, self.partials)

    def missing(self, context):
        """Variables a strict render with ``context`` would miss (see ``missing_variables``)."""
        return missing_variables(self.source, context, self.partials)

    def render(self, context=None, *, strict=True):
        out = []
        ctx = context or {}
        _render_nodes(self._tree, [(ctx, ctx)], self.partials, strict, out, frozenset())
        return "".join(out)

    def __repr__(self):
        return "Template(name={!r}, variables={})".format(self.name, sorted(self.variables))


def render(source, context=None, *, partials=None, strict=True):
    """One-shot convenience: compile ``source`` and render it with ``context``."""
    return Template(source, partials=partials).render(context or {}, strict=strict)
