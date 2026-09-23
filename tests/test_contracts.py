"""Tests for prompt input contracts (required / default / enum) and packaging."""

import fnmatch
import sys
from pathlib import Path

import pytest

from promptlab.prompt import (
    BUNDLED_LIBRARY,
    ContractError,
    PromptLibrary,
    apply_contract,
    default_library_root,
    parse_inputs,
)
from promptlab.template import MissingVariableError, find_variables, missing_variables, render


# -- parse_inputs ----------------------------------------------------------- #
def test_parse_inputs_defaults_and_optional():
    specs = parse_inputs(
        [
            {"name": "a", "description": "req"},
            {"name": "b", "required": False},
            {"name": "c", "default": "x"},
            {"name": "d", "enum": ["one", "two"]},
        ]
    )
    a, b, c, d = specs
    assert a.required and not a.has_default
    assert not b.required and b.fallback == ""
    assert not c.required and c.fallback == "x"  # a default implies optional
    assert d.enum == ["one", "two"]
    assert c.to_dict() == {"name": "c", "description": "", "required": False, "default": "x"}


def test_parse_inputs_none_is_empty():
    assert parse_inputs(None) == []


@pytest.mark.parametrize(
    "raw,message",
    [
        ({"name": "a"}, "must be a list"),
        (["a"], "must be a mapping"),
        ([{"name": "1bad"}], "invalid name"),
        ([{"name": "a", "requierd": False}], "unknown key"),
        ([{"name": "a"}, {"name": "a"}], "declared twice"),
        ([{"name": "a", "required": "no"}], "true or false"),
        ([{"name": "a", "required": True, "default": "x"}], "required but has a default"),
        ([{"name": "a", "enum": "x"}], "non-empty list"),
        ([{"name": "a", "enum": ["x"], "default": "y"}], "not one of"),
    ],
)
def test_parse_inputs_rejects_malformed_contracts(raw, message):
    with pytest.raises(ContractError, match=message):
        parse_inputs(raw, where="t@v1")


# -- apply_contract --------------------------------------------------------- #
def test_apply_contract_fills_defaults_without_overriding_caller():
    specs = parse_inputs([{"name": "a"}, {"name": "b", "default": "B"}, {"name": "c", "required": False}])
    assert apply_contract(specs, {"a": 1}) == {"a": 1, "b": "B", "c": ""}
    assert apply_contract(specs, {"a": 1, "b": "mine"})["b"] == "mine"


def test_apply_contract_missing_required_raises_only_when_strict():
    specs = parse_inputs([{"name": "a"}, {"name": "b"}])
    with pytest.raises(ContractError, match="missing required input"):
        apply_contract(specs, {}, where="p@v1")
    assert apply_contract(specs, {}, strict=False) == {}


def test_apply_contract_enum_validation():
    specs = parse_inputs([{"name": "lang", "enum": ["Spanish", "English"]}, {"name": "n", "enum": [1, 2]}])
    assert apply_contract(specs, {"lang": "Spanish", "n": "2"})["n"] == "2"  # CLI strings match ints
    with pytest.raises(ContractError, match="expected one of: Spanish, English"):
        apply_contract(specs, {"lang": "Klingon", "n": 1})


# -- Prompt integration ----------------------------------------------------- #
@pytest.fixture
def lib():
    return PromptLibrary(BUNDLED_LIBRARY)


def test_interviewer_level_is_really_optional(lib):
    """Regression: 'Optional seniority' used to be required."""
    out = lib.render("interviewer", {"role": "backend engineer"})
    assert "for a backend engineer position" in out
    assert "a  backend" not in out
    assert "senior backend engineer" in lib.render("interviewer", {"role": "backend engineer", "level": "senior"})


def test_interviewer_v1_still_requires_level(lib):
    prompt = lib.get("interviewer")
    assert prompt.missing({"role": "x"}, version="v1") == ["level"]
    with pytest.raises(ContractError):
        prompt.render({"role": "x"}, version="v1")


def test_translator_enum_is_enforced(lib):
    with pytest.raises(ContractError, match="Klingon"):
        lib.render("translator", {"target_language": "Klingon"})
    assert "into English" in lib.render("translator", {"target_language": "English"})


def test_optional_inputs_of_summarizer_translator_api_designer(lib):
    assert lib.get("summarizer").missing({"length": "x"}) == []
    assert lib.get("translator").missing({"target_language": "Spanish"}) == []
    assert lib.get("api-designer").missing({}) == []


def test_every_shipped_prompt_renders_with_only_required_inputs(lib):
    """Each version renders strictly when given just its required inputs."""
    for name in lib.names():
        prompt = lib.get(name)
        for version in prompt.versions:
            ctx = {
                s.name: (s.enum[0] if s.enum else "<{}>".format(s.name))
                for s in prompt.inputs(version)
                if s.required
            }
            assert prompt.missing(ctx, version=version) == [], (name, version)
            text = prompt.render(ctx, version=version)
            assert text.strip(), (name, version)


def test_missing_combines_required_inputs_and_template_variables(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "v1.md").write_text(
        "---\npurpose: t\ninputs:\n  - name: a\n  - name: unused\n---\n{{ a }} {{ b }}\n", encoding="utf-8"
    )
    prompt = PromptLibrary(tmp_path).get("p")
    assert prompt.missing({}) == ["a", "b", "unused"]


# -- template soundness (outer variables in loops, dotted paths) ------------ #
def test_missing_variables_sees_outer_variable_inside_each():
    """Regression: the static check reported nothing, then render raised."""
    src = "{{#each rules}}- {{ prefix }}: {{ this }}\n{{/each}}"
    ctx = {"rules": ["a", "b"]}
    assert missing_variables(src, ctx) == ["prefix"]
    with pytest.raises(MissingVariableError):
        render(src, ctx)
    assert missing_variables(src, {"rules": [{"prefix": "p"}]}) == []
    assert missing_variables(src, {"rules": ["a"], "prefix": "p"}) == []


def test_missing_variables_reports_missing_leaf_of_dotted_path():
    assert missing_variables("{{ user.email }}", {"user": {"name": "x"}}) == ["user.email"]


def test_each_else_branch_uses_outer_context():
    src = "{{#each items}}{{ this }}{{else}}Nothing for {{ who }}{{/each}}"
    assert "who" in find_variables(src)
    assert missing_variables(src, {"items": []}) == ["who"]


def test_missing_variables_ignores_unknown_partials():
    assert missing_variables("{{> nope }}{{ a }}", {}) == ["a"]


# -- packaging -------------------------------------------------------------- #
def test_library_lives_inside_the_package():
    import promptlab

    assert BUNDLED_LIBRARY == Path(promptlab.__file__).resolve().parent / "library"
    assert (BUNDLED_LIBRARY / "coding-assistant" / "v2.md").is_file()


def test_default_library_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PROMPTLAB_LIBRARY", raising=False)
    assert default_library_root() == BUNDLED_LIBRARY
    monkeypatch.setenv("PROMPTLAB_LIBRARY", str(tmp_path))
    assert default_library_root() == tmp_path


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_package_data_globs_cover_every_library_file():
    """Regression: a non-editable wheel shipped without the prompt library."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    globs = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]["promptlab"]
    files = [p for p in BUNDLED_LIBRARY.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files
    package_dir = BUNDLED_LIBRARY.parent
    for f in files:
        rel = f.relative_to(package_dir).as_posix()
        assert any(fnmatch.fnmatch(rel, g) for g in globs), rel
