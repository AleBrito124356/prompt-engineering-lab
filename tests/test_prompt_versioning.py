"""Tests for versioned, folder-backed prompts and the library loader."""

from pathlib import Path

import pytest

from promptlab.prompt import Prompt, PromptError, PromptLibrary

FIXTURES = Path(__file__).parent / "fixtures" / "library"


@pytest.fixture
def library():
    return PromptLibrary(FIXTURES)


def test_library_lists_prompts(library):
    assert "greeting" in library.names()


def test_versions_discovered_and_sorted(library):
    prompt = library.get("greeting")
    assert prompt.versions == ["v1", "v2"]


def test_latest_honors_meta_pin(library):
    assert library.get("greeting").latest == "v2"


@pytest.mark.parametrize(
    "spec,expected",
    [(None, "v2"), ("latest", "v2"), ("v1", "v1"), ("1", "v1"), ("2", "v2")],
)
def test_resolve_version(library, spec, expected):
    assert library.get("greeting").resolve_version(spec) == expected


def test_unknown_version_raises(library):
    with pytest.raises(PromptError):
        library.get("greeting").resolve_version("v9")


def test_render_pins_version(library):
    prompt = library.get("greeting")
    assert prompt.render({"name": "Ada"}, version="v1").strip() == "Hello Ada, welcome aboard."


def test_render_latest_with_conditional_and_partial(library):
    prompt = library.get("greeting")
    out = prompt.render({"name": "Ada", "formal": True, "team": "Ops"})
    assert "Good day, Ada." in out
    assert "-- The Ops team" in out


def test_metadata_merges_meta_and_frontmatter(library):
    meta = library.get("greeting").metadata("v2")
    assert meta["title"] == "Greeting"      # from meta.yaml
    assert meta["purpose"].startswith("Greet")  # from front-matter
    assert meta["version"] == "v2"


def test_diff_between_versions(library):
    diff = library.get("greeting").diff("v1", "v2")
    assert "greeting@v1" in diff
    assert "greeting@v2" in diff
    assert "Good day" in diff


def test_missing_prompt_raises(library):
    with pytest.raises(PromptError):
        library.get("does-not-exist")


def test_bundled_library_loads():
    """The shipped library must load and expose the documented prompts."""
    root = Path(__file__).resolve().parents[1] / "library"
    lib = PromptLibrary(root)
    names = lib.names()
    assert len(names) >= 20
    for expected in ("coding-assistant", "sql-expert", "json-responder"):
        assert expected in names
    # Every shipped prompt must render its latest version with declared inputs absent
    # only when it has no required variables; at minimum it must load cleanly.
    for name in names:
        prompt = lib.get(name)
        assert prompt.versions
        assert prompt.metadata().get("purpose")
