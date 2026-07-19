"""Tests for front-matter parsing."""

import pytest

from promptlab.frontmatter import split_frontmatter


def test_no_frontmatter_returns_empty_meta():
    meta, body = split_frontmatter("Just a body.\n")
    assert meta == {}
    assert body == "Just a body.\n"


def test_basic_frontmatter():
    text = "---\npurpose: greet\ntags: [a, b]\n---\nHello {{ name }}\n"
    meta, body = split_frontmatter(text)
    assert meta["purpose"] == "greet"
    assert meta["tags"] == ["a", "b"]
    assert body == "Hello {{ name }}\n"


def test_frontmatter_with_bom():
    text = "﻿---\nk: v\n---\nbody"
    meta, body = split_frontmatter(text)
    assert meta == {"k": "v"}
    assert body == "body"


def test_empty_frontmatter_block():
    meta, body = split_frontmatter("---\n---\nbody")
    assert meta == {}
    assert body == "body"


def test_non_mapping_frontmatter_raises():
    with pytest.raises(ValueError):
        split_frontmatter("---\n- just\n- a\n- list\n---\nbody")


def test_crlf_frontmatter():
    text = "---\r\nk: v\r\n---\r\nbody"
    meta, body = split_frontmatter(text)
    assert meta == {"k": "v"}
    assert body == "body"
