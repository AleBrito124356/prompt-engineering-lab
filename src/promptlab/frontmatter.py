"""YAML front-matter parsing for prompt files.

A prompt version file may begin with a ``---`` fenced YAML block:

    ---
    purpose: Answer coding questions with runnable examples.
    inputs:
      - name: language
        description: Target programming language.
    model_tips: Works best at temperature 0.2 for deterministic code.
    ---
    You are a senior {{ language }} engineer...

``split_frontmatter`` returns ``(metadata_dict, body_str)``. Files without a
front-matter block return ``({}, original_text)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

__all__ = ["split_frontmatter", "load_file"]

# Opening fence on its own line, lazy body, closing fence at the start of a line.
# MULTILINE lets ``^---`` match the closing fence (and handles an empty block);
# DOTALL lets the body span multiple lines. An optional leading BOM is allowed.
_FRONTMATTER_RE = re.compile(
    r"^﻿?---[ \t]*\r?\n(.*?)^---[ \t]*\r?\n?",
    re.DOTALL | re.MULTILINE,
)


def split_frontmatter(text):
    """Split ``text`` into ``(metadata, body)``.

    Raises ``ValueError`` when a front-matter block is present but does not
    parse to a mapping.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    meta = yaml.safe_load(raw) if raw.strip() else {}
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("front-matter must be a YAML mapping, got {}".format(type(meta).__name__))
    body = text[match.end():]
    return meta, body


def load_file(path):
    """Read a file and split its front-matter. Returns ``(metadata, body)``."""
    text = Path(path).read_text(encoding="utf-8")
    return split_frontmatter(text)
