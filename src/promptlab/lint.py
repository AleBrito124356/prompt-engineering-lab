"""Static checks for a prompt library -- ``promptlab lint``.

"Treat prompts like code" needs the equivalent of a compiler warning: a check
that the contract a prompt declares in its front-matter still matches what its
body actually uses, run in CI before a drifted prompt ships.

Checks (``code``: severity):

- ``invalid-meta`` (error): ``meta.yaml`` is not valid YAML / not a mapping.
- ``bad-latest`` (error): ``meta.yaml`` pins ``latest`` to a version that does
  not exist.
- ``missing-changelog`` (warning): a version has no ``changelog`` entry, or the
  changelog names a version that does not exist.
- ``invalid-front-matter`` (error): the version's front-matter does not parse.
- ``missing-purpose`` (error): no ``purpose`` in the front-matter.
- ``invalid-inputs`` (error): the ``inputs`` contract is malformed (unknown
  keys, duplicate names, a default outside its enum, ...).
- ``syntax`` (error): the template (or a partial) does not parse.
- ``unknown-partial`` (error): ``{{> name }}`` with no ``_partials/name.md``.
- ``undeclared-input`` (error): the body uses a variable the contract does not
  declare.
- ``unused-input`` (error): the contract declares an input the body never uses.
- ``optional-mismatch`` (warning): an input's description says "optional" (or
  "defaults to") but the contract makes it required.
- ``unguarded-optional`` (error): an optional input without a non-empty default
  is interpolated outside an ``{{#if name}}`` guard, so omitting it renders an
  empty hole ("a  backend engineer").
- ``outer-var-in-loop`` (warning): a declared input is used inside
  ``{{#each}}``. It resolves against the outer context only when the current
  item has no key of the same name, which static analysis cannot know.
- ``render-failed`` (error): a strict render with placeholder values for the
  required inputs raises.
- ``not-a-prompt`` (warning): a folder in the library root has no ``v<N>.md``.

``lint_library`` returns a list of :class:`Finding`; the CLI exits 1 when any
error is present (or any warning with ``--strict``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .frontmatter import load_file
from .prompt import ContractError, PromptError, PromptLibrary, parse_inputs
from .template import TemplateError, Template, variable_references

__all__ = ["Finding", "lint_library", "format_text", "format_json", "has_failures"]

_VERSION_RE = re.compile(r"^v(\d+)$")
_OPTIONAL_PROSE_RE = re.compile(r"\boptional(?:ly)?\b|\bdefaults?\s+to\b|\bif\s+omitted\b", re.IGNORECASE)


class Finding:
    """One lint result."""

    __slots__ = ("prompt", "code", "severity", "message", "file")

    def __init__(self, prompt, code, severity, message, file=None):
        self.prompt = prompt
        self.code = code
        self.severity = severity
        self.message = message
        self.file = file

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
        }

    def __repr__(self):
        return "Finding({!r}, {!r}, {!r})".format(self.prompt, self.code, self.severity)


def _rel(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve().parent).as_posix()
    except ValueError:
        return str(path)


def _version_files(folder):
    files = {}
    for f in folder.glob("v*.md"):
        if _VERSION_RE.match(f.stem):
            files[f.stem] = f
    return dict(sorted(files.items(), key=lambda kv: int(kv[0][1:])))


class _Placeholder:
    """Stand-in value for the smoke render: renders as ``<name>``, supports
    dotted access (``{{ user.name }}``) and iterates as an empty list."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(key)
        return _Placeholder("{}.{}".format(self._name, key))

    def __iter__(self):
        return iter(())

    def __str__(self):
        return "<{}>".format(self._name)


def _placeholder(spec):
    if spec.enum:
        return spec.enum[0]
    return _Placeholder(spec.name)


def _lint_version(name, version, path, partials, root, out):
    ref = "{}@{}".format(name, version)
    rel = _rel(path, root)

    def add(code, severity, message):
        out.append(Finding(ref, code, severity, message, rel))

    try:
        fm, body = load_file(path)
    except (ValueError, yaml.YAMLError) as exc:
        add("invalid-front-matter", "error", str(exc).splitlines()[0])
        return

    if not str(fm.get("purpose") or "").strip():
        add("missing-purpose", "error", "front-matter has no 'purpose' (it is what `promptlab list` shows)")

    try:
        specs = parse_inputs(fm.get("inputs"), where=ref)
    except ContractError as exc:
        add("invalid-inputs", "error", str(exc))
        specs = None

    try:
        refs, unknown_partials = variable_references(body, partials)
    except TemplateError as exc:
        add("syntax", "error", str(exc))
        return
    for partial in sorted(set(unknown_partials)):
        add("unknown-partial", "error", "{{{{> {} }}}} has no _partials/{}.md".format(partial, partial))

    if specs is None:
        return

    declared = {s.name: s for s in specs}
    for spec in specs:
        if spec.required and _OPTIONAL_PROSE_RE.search(spec.description):
            add(
                "optional-mismatch",
                "warning",
                "input {!r} is described as optional but the contract makes it required; "
                "add 'required: false' (or a default) or fix the description".format(spec.name),
            )
    outer_roots = {r.root for r in refs if not r.in_loop}
    all_roots = {r.root for r in refs}

    for root_name in sorted(outer_roots - set(declared)):
        add(
            "undeclared-input",
            "error",
            "the body uses {{{{ {0} }}}} but 'inputs' does not declare {0!r}".format(root_name),
        )
    for input_name in sorted(set(declared) - all_roots):
        add(
            "unused-input",
            "error",
            "input {!r} is declared but the body never uses it".format(input_name),
        )

    reported = set()
    for r in refs:
        spec = declared.get(r.root)
        if spec is None:
            continue
        if (
            r.kind == "var"
            and not spec.required
            and spec.fallback in ("", None)
            and r.root not in r.guards
            and r.root not in reported
        ):
            reported.add(r.root)
            add(
                "unguarded-optional",
                "error",
                "optional input {0!r} is interpolated outside an {{{{#if {0}}}}} guard, so omitting "
                "it leaves an empty hole in the prompt; guard it or give it a non-empty default".format(r.root),
            )
        if r.in_loop and ("loop", r.root) not in reported:
            reported.add(("loop", r.root))
            add(
                "outer-var-in-loop",
                "warning",
                "input {!r} is used inside {{{{#each {}}}}}; an item with a {!r} key would shadow it "
                "(checked at render time, not statically)".format(r.root, r.loops[-1], r.root),
            )

    # Smoke render: required inputs get placeholders, optional ones their defaults.
    context = {s.name: _placeholder(s) for s in specs if s.required}
    # Loop collections and undeclared names get a neutral value so the smoke
    # render tests the template itself, not the drift already reported above.
    for root_name in outer_roots - set(declared):
        context.setdefault(root_name, _Placeholder(root_name))
    try:
        ctx = {s.name: s.fallback for s in specs if not s.required}
        ctx.update(context)
        Template(body, partials=partials, name=ref).render(ctx, strict=True)
    except TemplateError as exc:
        if not unknown_partials:
            add("render-failed", "error", "strict render with placeholder inputs failed: {}".format(exc))


def _lint_prompt(folder, partials, root, out):
    name = folder.name
    files = _version_files(folder)
    if not files:
        out.append(
            Finding(name, "not-a-prompt", "warning", "folder has no v<N>.md version files", _rel(folder, root))
        )
        return

    meta = {}
    meta_path = folder / "meta.yaml"
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if not isinstance(meta, dict):
                raise ValueError("meta.yaml must be a mapping")
        except (ValueError, yaml.YAMLError) as exc:
            out.append(
                Finding(name, "invalid-meta", "error", str(exc).splitlines()[0], _rel(meta_path, root))
            )
            meta = {}

    latest = meta.get("latest")
    if latest:
        key = str(latest).strip()
        key = key if key.startswith("v") else "v" + key
        if key not in files:
            out.append(
                Finding(
                    name,
                    "bad-latest",
                    "error",
                    "meta.yaml pins latest: {} but only {} exist".format(latest, ", ".join(files)),
                    _rel(meta_path, root),
                )
            )

    changelog = meta.get("changelog") or {}
    if not isinstance(changelog, dict):
        changelog = {}
    logged = {str(k) if str(k).startswith("v") else "v" + str(k) for k in changelog}
    for version in files:
        if version not in logged:
            out.append(
                Finding(
                    "{}@{}".format(name, version),
                    "missing-changelog",
                    "warning",
                    "meta.yaml changelog has no entry for {}".format(version),
                    _rel(meta_path, root),
                )
            )
    for version in sorted(logged - set(files)):
        out.append(
            Finding(
                name,
                "missing-changelog",
                "warning",
                "meta.yaml changelog mentions {} but there is no {}.md".format(version, version),
                _rel(meta_path, root),
            )
        )

    for version, path in files.items():
        _lint_version(name, version, path, partials, root, out)


def lint_library(root, names=None):
    """Lint every prompt under ``root`` (or only ``names``). Returns findings.

    Raises :class:`PromptError` when ``root`` does not exist or a requested name
    is not a folder in it.
    """
    library = PromptLibrary(root)  # validates the root
    root = library.root
    out = []
    partials = {}
    pdir = root / "_partials"
    if pdir.is_dir():
        for f in sorted(pdir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            partials[f.stem] = text
            try:
                variable_references(text)
            except TemplateError as exc:
                out.append(Finding("_partials/" + f.stem, "syntax", "error", str(exc), _rel(f, root)))

    if names:
        folders = []
        for name in names:
            folder = root / name
            if not folder.is_dir():
                raise PromptError("no prompt {!r} in {}".format(name, root))
            folders.append(folder)
    else:
        folders = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith((".", "_"))]

    for folder in folders:
        _lint_prompt(folder, partials, root, out)
    return out


def has_failures(findings, strict=False):
    """True when the findings should fail CI (errors, or any finding if strict)."""
    return any(f.severity == "error" or strict for f in findings)


def _summary(findings, checked):
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    return errors, warnings, checked


def format_text(findings, checked):
    """Human-readable report, one finding per line, then a summary line."""
    lines = []
    if findings:
        w_ref = max(len(f.prompt) for f in findings)
        w_code = max(len(f.code) for f in findings)
        for f in findings:
            lines.append(
                "{ref:<{wr}}  {sev:<7}  {code:<{wc}}  {msg}".format(
                    ref=f.prompt, wr=w_ref, sev=f.severity, code=f.code, wc=w_code, msg=f.message
                )
            )
        lines.append("")
    errors, warnings, count = _summary(findings, checked)
    lines.append(
        "{status}: {n} prompt(s) checked, {e} error(s), {w} warning(s)".format(
            status="ok" if not errors else "failed", n=count, e=errors, w=warnings
        )
    )
    return "\n".join(lines)


def format_json(findings, checked):
    """Machine-readable report for CI annotations."""
    errors, warnings, count = _summary(findings, checked)
    return json.dumps(
        {
            "checked": count,
            "errors": errors,
            "warnings": warnings,
            "findings": [f.to_dict() for f in findings],
        },
        indent=2,
    )
