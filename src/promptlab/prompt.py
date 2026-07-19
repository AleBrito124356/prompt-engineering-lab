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
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import yaml

from .frontmatter import load_file
from .template import Template

__all__ = ["Prompt", "PromptLibrary", "PromptError"]

_VERSION_RE = re.compile(r"^v(\d+)$")


class PromptError(Exception):
    """Raised for missing prompts, versions, or malformed prompt folders."""


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
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
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
    def read(self, version=None):
        """Return ``(front_matter, body)`` for the resolved version."""
        v = self.resolve_version(version)
        return load_file(self._versions[v])

    def body(self, version=None):
        return self.read(version)[1]

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
        return self.template(version).render(context or {}, strict=strict)

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
