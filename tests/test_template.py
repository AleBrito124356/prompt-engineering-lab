"""Tests for the template engine: interpolation, blocks, partials, validation."""

import pytest

from promptlab.template import (
    MissingVariableError,
    PartialNotFoundError,
    Template,
    TemplateError,
    TemplateSyntaxError,
    few_shot,
    find_variables,
    missing_variables,
    render,
)


def test_simple_variable():
    assert render("Hello {{ name }}!", {"name": "Ada"}) == "Hello Ada!"


def test_dotted_path():
    assert render("{{ user.name }}", {"user": {"name": "Grace"}}) == "Grace"


def test_missing_variable_is_strict_by_default():
    with pytest.raises(MissingVariableError):
        render("{{ missing }}", {})


def test_missing_variable_non_strict_renders_empty():
    assert render("[{{ missing }}]", {}, strict=False) == "[]"


def test_none_value_renders_empty_and_does_not_raise():
    assert render("[{{ x }}]", {"x": None}) == "[]"


def test_boolean_rendering():
    assert render("{{ flag }}", {"flag": True}) == "true"
    assert render("{{ flag }}", {"flag": False}) == "false"


def test_comments_are_ignored():
    assert render("a{{! hidden }}b", {}) == "ab"


def test_if_else_block():
    tmpl = "{{#if admin}}yes{{else}}no{{/if}}"
    assert render(tmpl, {"admin": True}) == "yes"
    assert render(tmpl, {"admin": False}) == "no"
    assert render(tmpl, {}) == "no"  # missing is falsy in conditions


def test_unless_block():
    tmpl = "{{#unless done}}pending{{/unless}}"
    assert render(tmpl, {"done": False}) == "pending"
    assert render(tmpl, {"done": True}) == ""


def test_each_over_dicts_with_this_and_bare_field():
    tmpl = "{{#each items}}{{this.name}}={{value}};{{/each}}"
    ctx = {"items": [{"name": "a", "value": 1}, {"name": "b", "value": 2}]}
    assert render(tmpl, ctx) == "a=1;b=2;"


def test_each_over_scalars_with_dot():
    assert render("{{#each nums}}[{{.}}]{{/each}}", {"nums": [1, 2, 3]}) == "[1][2][3]"


def test_each_empty_uses_else():
    tmpl = "{{#each items}}{{.}}{{else}}none{{/each}}"
    assert render(tmpl, {"items": []}) == "none"
    assert render(tmpl, {}) == "none"


def test_partials():
    tmpl = Template("Intro. {{> footer }}", partials={"footer": "Bye {{ name }}."})
    assert tmpl.render({"name": "Sam"}) == "Intro. Bye Sam."


def test_partial_not_found():
    with pytest.raises(PartialNotFoundError):
        render("{{> nope }}", {}, partials={})


def test_partial_recursion_detected():
    with pytest.raises(TemplateError):
        Template("{{> loop }}", partials={"loop": "x{{> loop }}"}).render({})


def test_unbalanced_block_is_syntax_error():
    with pytest.raises(TemplateSyntaxError):
        Template("{{#if a}}no close")


def test_mismatched_close_is_syntax_error():
    with pytest.raises(TemplateSyntaxError):
        Template("{{#if a}}x{{/each}}")


def test_find_variables_excludes_loop_locals():
    src = "Hi {{ name }} {{#each items}}{{ x }}{{/each}} {{#if flag}}{{ y }}{{/if}}"
    assert find_variables(src) == {"name", "items", "flag", "y"}


def test_find_variables_includes_partial_vars_when_supplied():
    assert find_variables("{{> p }}", partials={"p": "{{ z }}"}) == {"z"}


def test_missing_variables_reports_absent_roots():
    src = "{{ a }} {{ b }}"
    assert missing_variables(src, {"a": 1}) == ["b"]


def test_template_variables_and_missing():
    tmpl = Template("{{ a }}{{ b }}")
    assert tmpl.variables == {"a", "b"}
    assert tmpl.missing({"a": 1}) == ["b"]


def test_few_shot_from_dicts_and_tuples():
    block = few_shot([{"input": "hi", "output": "hola"}, ("bye", "adios")])
    assert block == "Input: hi\nOutput: hola\n\nInput: bye\nOutput: adios"


def test_few_shot_custom_labels():
    block = few_shot([{"input": "2+2", "output": "4"}], input_label="Q", output_label="A")
    assert block == "Q: 2+2\nA: 4"
