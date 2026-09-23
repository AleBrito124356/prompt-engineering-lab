"""Versioned, folder-backed prompts and the library that holds them.

A prompt is a directory:

    library/coding-assistant/
        meta.yaml       # name, title, latest pointer, tags, changelog
        v1.md           # front-matter (purpose, inputs, model_tips) + body
        v2.md

Versions are ``v<N>.md`` files. ``meta.yaml`` may pin ``latest``; otherwise the
highest-numbered version wins. Because versions are plain files, you can pin an
exact one in production and ``diff`` two of them to review a change before it
ships -- the same discipline you apply to code.

Each version's front-matter declares its **input contract**::

    inputs:
      - name: role                  # required by default
        description: The role being interviewed for.
      - name: level
        required: false             # optional: filled with "" when omitted
      - name: style
        default: pragmatic REST     # a default makes an input optional
      - name: target_language
        enum: [Spanish, English]    # values outside the list are rejected

``Prompt.render`` fills defaults, rejects values outside an ``enum`` and, in
strict mode, refuses to render when a required input is missing.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

import yaml

from .frontmatter import load_file
from .template import Template

__all__ = [
    "Prompt",
    "PromptLibrary",
    "PromptError",
    "ContractError",
    "InputSpec",
    "parse_inputs",
    "apply_contract",
    "BUNDLED_LIBRARY",
    "default_library_root",
]

_VERSION_RE = re.compile(r"^v(\d+)$")
_INPUT_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
_INPUT_KEYS = {"name", "description", "required", "default", "enum"}

#: The prompt library shipped inside the package (``promptlab/library``).
BUNDLED_LIBRARY = Path(__file__).resolve().parent / "library"


def default_library_root():
    """``$PROMPTLAB_LIBRARY`` when set, else the library bundled with the package."""
    env = os.environ.get("PROMPTLAB_LIBRARY")
    return Path(env) if env else BUNDLED_LIBRARY


class PromptError(Exception):
    """Raised for missing prompts, versions, or malformed prompt folders."""


class ContractError(PromptError):
    """Raised when inputs violate a prompt's declared contract, or it is malformed."""


class _NoDefault:
    def __repr__(self):
        return "<no default>"


_NO_DEFAULT = _NoDefault()


class InputSpec:
    """One declared input from a version's front-matter ``inputs`` list."""

    __slots__ = ("name", "description", "required", "default", "enum")

    def __init__(self, name, description="", required=True, default=_NO_DEFAULT, enum=None):
        self.name = name
        self.description = description
        self.required = required
        self.default = default
        self.enum = list(enum) if enum is not None else None

    @property
    def has_default(self):
        return self.default is not _NO_DEFAULT

    @property
    def fallback(self):
        """Value used when the caller omits this optional input."""
        return self.default if self.has_default else ""

    def to_dict(self):
        out = {"name": self.name, "description": self.description, "required": self.required}
        if self.has_default:
            out["default"] = self.default
        if self.enum is not None:
            out["enum"] = list(self.enum)
        return out

    def __repr__(self):
        return "InputSpec({!r}, required={}, default={!r}, enum={!r})".format(
            self.name, self.required, self.default, self.enum
        )


def parse_inputs(raw, where="front-matter"):
    """Validate a front-matter ``inputs`` list into :class:`InputSpec` objects.

    Raises :class:`ContractError` naming ``where`` for anything malformed:
    non-list, entries without a valid ``name``, unknown keys (catches typos
    such as ``requierd``), duplicates, a non-bool ``required``, ``required:
    true`` together with a ``default``, a non-list or empty ``enum``, or a
    default outside its ``enum``.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ContractError("{}: 'inputs' must be a list, got {}".format(where, type(raw).__name__))
    specs, seen = [], set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ContractError("{}: inputs[{}] must be a mapping with a 'name'".format(where, i))
        name = item.get("name")
        if not isinstance(name, str) or not _INPUT_NAME_RE.match(name):
            raise ContractError("{}: inputs[{}] has an invalid name {!r}".format(where, i, name))
        unknown = sorted(set(item) - _INPUT_KEYS)
        if unknown:
            raise ContractError(
                "{}: input {!r} has unknown key(s) {} (allowed: {})".format(
                    where, name, ", ".join(unknown), ", ".join(sorted(_INPUT_KEYS))
                )
            )
        if name in seen:
            raise ContractError("{}: input {!r} is declared twice".format(where, name))
        seen.add(name)
        default = item["default"] if "default" in item else _NO_DEFAULT
        required = item.get("required", default is _NO_DEFAULT)
        if not isinstance(required, bool):
            raise ContractError("{}: input {!r}: 'required' must be true or false".format(where, name))
        if required and default is not _NO_DEFAULT:
            raise ContractError(
                "{}: input {!r} is required but has a default; drop one of them".format(where, name)
            )
        enum = item.get("enum")
        if enum is not None:
            if not isinstance(enum, list) or not enum:
                raise ContractError("{}: input {!r}: 'enum' must be a non-empty list".format(where, name))
            if default not in (_NO_DEFAULT, None, "") and str(default) not in {str(e) for e in enum}:
                raise ContractError(
                    "{}: input {!r}: default {!r} is not one of {}".format(where, name, default, enum)
                )
        description = item.get("description") or ""
        specs.append(InputSpec(name, str(description), required, default, enum))
    return specs


def apply_contract(specs, context, *, strict=True, where="prompt"):
    """Return ``context`` merged with the defaults from ``specs``.

    Optional inputs the caller omitted get their ``default`` (or ``""``).
    Values outside a declared ``enum`` raise :class:`ContractError`. In strict
    mode a missing required input raises too; with ``strict=False`` it is left
    out so the template renders it as empty.
    """
    merged = dict(context or {})
    missing = []
    for spec in specs:
        if spec.name not in merged:
            if spec.required:
                missing.append(spec.name)
                continue
            merged[spec.name] = spec.fallback
        value = merged[spec.name]
        if spec.enum is not None and value not in (None, ""):
            allowed = {str(e) for e in spec.enum}
            if str(value) not in allowed:
                raise ContractError(
                    "{}: invalid value {!r} for input {!r}; expected one of: {}".format(
                        where, value, spec.name, ", ".join(str(e) for e in spec.enum)
                    )
                )
    if strict and missing:
        raise ContractError(
            "{}: missing required input(s): {} (pass --var key=value)".format(where, ", ".join(missing))
        )
    return merged


class Prompt:
    """A single versioned prompt loaded from a directory."""

    def __init__(self, path, partials=None):
        self.path = Path(path)
        if not self.path.is_dir():
            raise PromptError("not a prompt directory: {}".format(self.path))
        self.name = self.path.name
        self.partials = dict(partials or {})
        self.meta = self._load_meta()
        self._versions = self._discover_versions()

    # -- loading ---------------------------------------------------------- #
    def _load_meta(self):
        meta_path = self.path / "meta.yaml"
        if not meta_path.exists():
            return {}
        try:
            data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PromptError("invalid YAML in {}: {}".format(meta_path, exc)) from None
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise PromptError("meta.yaml must be a mapping in {}".format(self.path))
        return data

    def _discover_versions(self):
        versions = {}
        for f in self.path.glob("v*.md"):
            m = _VERSION_RE.match(f.stem)
            if m:
                versions[f.stem] = f
        if not versions:
            raise PromptError("no version files (v1.md, v2.md, ...) in {}".format(self.path))
        return dict(sorted(versions.items(), key=lambda kv: int(kv[0][1:])))

    # -- version resolution ---------------------------------------------- #
    @property
    def versions(self):
        return list(self._versions)

    def _normalize(self, spec):
        s = str(spec).strip()
        key = s if s.startswith("v") else "v" + s
        if key not in self._versions:
            raise PromptError(
                "unknown version {!r} for {!r}; available: {}".format(
                    spec, self.name, self.versions
                )
            )
        return key

    @property
    def latest(self):
        pinned = self.meta.get("latest")
        if pinned:
            return self._normalize(pinned)
        return self.versions[-1]

    def resolve_version(self, spec):
        if spec in (None, "", "latest"):
            return self.latest
        return self._normalize(spec)

    # -- content ---------------------------------------------------------- #
    def version_path(self, version=None):
        """Path of the resolved version's ``v<N>.md`` file."""
        return self._versions[self.resolve_version(version)]

    def read(self, version=None):
        """Return ``(front_matter, body)`` for the resolved version."""
        path = self.version_path(version)
        try:
            return load_file(path)
        except (ValueError, yaml.YAMLError) as exc:
            raise PromptError("invalid front-matter in {}: {}".format(path, exc)) from None

    def body(self, version=None):
        return self.read(version)[1]

    def ref(self, version=None):
        """``name@vN`` for the resolved version."""
        return "{}@{}".format(self.name, self.resolve_version(version))

    def inputs(self, version=None):
        """The version's declared input contract as a list of :class:`InputSpec`."""
        fm, _ = self.read(version)
        return parse_inputs(fm.get("inputs"), where=self.ref(version))

    def resolve_context(self, context=None, *, version=None, strict=True):
        """``context`` with defaults filled and enums checked (see :func:`apply_contract`)."""
        return apply_contract(
            self.inputs(version), context or {}, strict=strict, where=self.ref(version)
        )

    def missing(self, context=None, *, version=None):
        """Every name a strict render with ``context`` would be missing.

        Required inputs from the contract plus template variables (after
        defaults are applied), sorted. Enum violations still raise
        :class:`ContractError`.
        """
        specs = self.inputs(version)
        merged = apply_contract(specs, context or {}, strict=False, where=self.ref(version))
        required = {s.name for s in specs if s.required and s.name not in merged}
        return sorted(required | set(self.template(version).missing(merged)))

    def metadata(self, version=None):
        """Merge folder ``meta.yaml`` with the version's front-matter."""
        fm, _ = self.read(version)
        merged = dict(self.meta)
        merged.update(fm)
        merged["name"] = self.name
        merged["version"] = self.resolve_version(version)
        merged["versions"] = self.versions
        return merged

    def template(self, version=None):
        v = self.resolve_version(version)
        return Template(
            self.body(v), partials=self.partials, name="{}@{}".format(self.name, v)
        )

    def render(self, context=None, *, version=None, strict=True):
        """Render a version with ``context`` merged over its declared defaults.

        Strict mode (the default) raises :class:`ContractError` for missing
        required inputs and ``MissingVariableError`` for any other variable the
        template cannot resolve.
        """
        ctx = self.resolve_context(context, version=version, strict=strict)
        return self.template(version).render(ctx, strict=strict)

    def variables(self, version=None):
        return self.template(version).variables

    def diff(self, a, b):
        """Unified diff between two versions' bodies."""
        va, vb = self.resolve_version(a), self.resolve_version(b)
        body_a = self.body(va).splitlines(keepends=True)
        body_b = self.body(vb).splitlines(keepends=True)
        lines = difflib.unified_diff(
            body_a,
            body_b,
            fromfile="{}@{}".format(self.name, va),
            tofile="{}@{}".format(self.name, vb),
        )
        return "".join(lines)

    def __repr__(self):
        return "Prompt(name={!r}, versions={}, latest={!r})".format(
            self.name, self.versions, self.latest
        )


class PromptLibrary:
    """Discovers versioned prompts under a root directory and shared partials."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise PromptError("library root does not exist: {}".format(self.root))

    def partials(self):
        """Load shared partials from ``<root>/_partials/*.md`` (stem -> text)."""
        out = {}
        pdir = self.root / "_partials"
        if pdir.is_dir():
            for f in sorted(pdir.glob("*.md")):
                out[f.stem] = f.read_text(encoding="utf-8")
        return out

    def names(self):
        out = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and not p.name.startswith("_") and any(p.glob("v*.md")):
                out.append(p.name)
        return out

    def get(self, name):
        path = self.root / name
        if not path.is_dir():
            raise PromptError("no prompt {!r} in {}".format(name, self.root))
        return Prompt(path, partials=self.partials())

    def render(self, name, context=None, *, version=None, strict=True):
        return self.get(name).render(context or {}, version=version, strict=strict)

    def summaries(self):
        rows = []
        for name in self.names():
            p = self.get(name)
            md = p.metadata()
            rows.append(
                {
                    "name": name,
                    "title": md.get("title", name),
                    "purpose": md.get("purpose", ""),
                    "versions": p.versions,
                    "latest": p.latest,
                    "tags": md.get("tags", []),
                }
            )
        return rows
