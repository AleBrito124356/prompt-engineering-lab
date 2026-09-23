"""Tests for ``promptlab lint``: each check on a deliberately broken library."""

import json
from pathlib import Path

import pytest

from promptlab.cli import main
from promptlab.lint import format_json, format_text, has_failures, lint_library
from promptlab.prompt import BUNDLED_LIBRARY, PromptError


def _write(root, rel, text):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def broken_library(tmp_path):
    root = tmp_path / "lib"
    _write(root, "_partials/sig.md", "-- {{ team }}")
    # drifted: body uses `dialect`, contract declares `language`
    _write(root, "drift/meta.yaml", "latest: v1\nchangelog:\n  v1: first\n")
    _write(
        root,
        "drift/v1.md",
        "---\npurpose: p\ninputs:\n  - name: language\n---\nWrite {{ dialect }} code.\n",
    )
    # optional input interpolated without a guard
    _write(root, "hole/meta.yaml", "changelog:\n  v1: first\n")
    _write(
        root,
        "hole/v1.md",
        "---\npurpose: p\ninputs:\n  - name: role\n  - name: level\n    required: false\n---\n"
        "Interview a {{ level }} {{ role }}.\n",
    )
    # latest pinned to a version that does not exist + missing changelog entry
    _write(root, "badpin/meta.yaml", "latest: v3\nchangelog:\n  v1: first\n")
    _write(root, "badpin/v1.md", "---\npurpose: p\ninputs: []\n---\nHello.\n")
    _write(root, "badpin/v2.md", "---\npurpose: p\ninputs: []\n---\nHello again.\n")
    # unknown partial + missing purpose
    _write(root, "nopartial/v1.md", "---\ninputs: []\n---\nBody {{> missing-partial }}\n")
    # syntax error
    _write(root, "syntax/v1.md", "---\npurpose: p\ninputs: []\n---\n{{#if x}}never closed\n")
    # malformed contract
    _write(root, "contract/v1.md", "---\npurpose: p\ninputs:\n  - name: a\n    requierd: false\n---\n{{ a }}\n")
    # declared input used inside a loop (warning)
    _write(root, "loop/meta.yaml", "changelog:\n  v1: first\n")
    _write(
        root,
        "loop/v1.md",
        "---\npurpose: p\ninputs:\n  - name: rules\n  - name: prefix\n---\n"
        "{{#each rules}}- {{ prefix }} {{ this }}\n{{/each}}",
    )
    # description says optional but contract says required (warning)
    _write(
        root,
        "prose/v1.md",
        "---\npurpose: p\ninputs:\n  - name: tone\n    description: Optional tone.\n---\n"
        "{{#if tone}}Use a {{ tone }} tone.{{/if}}\n",
    )
    # folder without versions
    (root / "stray").mkdir()
    return root


def _codes(findings):
    return {(f.prompt, f.code) for f in findings}


def test_lint_reports_each_problem(broken_library):
    findings = lint_library(broken_library)
    codes = _codes(findings)
    assert ("drift@v1", "undeclared-input") in codes
    assert ("drift@v1", "unused-input") in codes
    assert ("hole@v1", "unguarded-optional") in codes
    assert ("badpin", "bad-latest") in codes
    assert ("badpin@v2", "missing-changelog") in codes
    assert ("nopartial@v1", "unknown-partial") in codes
    assert ("nopartial@v1", "missing-purpose") in codes
    assert ("syntax@v1", "syntax") in codes
    assert ("contract@v1", "invalid-inputs") in codes
    assert ("loop@v1", "outer-var-in-loop") in codes
    assert ("prose@v1", "optional-mismatch") in codes
    assert ("stray", "not-a-prompt") in codes
    severities = {f.code: f.severity for f in findings}
    assert severities["outer-var-in-loop"] == "warning"
    assert severities["unguarded-optional"] == "error"
    assert has_failures(findings)


def test_lint_messages_are_specific(broken_library):
    findings = {(f.prompt, f.code): f for f in lint_library(broken_library)}
    assert "dialect" in findings[("drift@v1", "undeclared-input")].message
    assert "language" in findings[("drift@v1", "unused-input")].message
    assert "latest: v3" in findings[("badpin", "bad-latest")].message
    assert findings[("hole@v1", "unguarded-optional")].file.endswith("hole/v1.md")


def test_guarded_optional_and_defaulted_optional_are_clean(tmp_path):
    root = tmp_path / "lib"
    _write(root, "ok/meta.yaml", "changelog:\n  v1: first\n")
    _write(
        root,
        "ok/v1.md",
        "---\npurpose: p\ninputs:\n  - name: level\n    required: false\n"
        "  - name: style\n    default: pragmatic REST\n---\n"
        "{{#if level}}Level {{ level }}.{{/if}} Style: {{ style }}.\n",
    )
    assert lint_library(root) == []


def test_lint_single_prompt_and_unknown_name(broken_library):
    assert {f.prompt.split("@")[0] for f in lint_library(broken_library, ["hole"])} == {"hole"}
    with pytest.raises(PromptError):
        lint_library(broken_library, ["does-not-exist"])


def test_bundled_library_is_lint_clean_even_in_strict_mode():
    findings = lint_library(BUNDLED_LIBRARY)
    assert findings == [], format_text(findings, 0)


def test_format_json_and_text(broken_library):
    findings = lint_library(broken_library)
    data = json.loads(format_json(findings, 9))
    assert data["checked"] == 9
    assert data["errors"] == sum(1 for f in findings if f.severity == "error")
    assert {"prompt", "code", "severity", "message", "file"} <= set(data["findings"][0])
    assert format_text(findings, 9).splitlines()[-1].startswith("failed: 9 prompt(s) checked")
    assert format_text([], 3) == "ok: 3 prompt(s) checked, 0 error(s), 0 warning(s)"


def test_cli_lint_exit_codes(broken_library, capsys):
    assert main(["--library", str(broken_library), "lint"]) == 1
    out = capsys.readouterr().out
    assert "unguarded-optional" in out
    assert main(["lint"]) == 0
    assert main(["lint", "--strict", "interviewer", "summarizer"]) == 0
    assert "ok: 2 prompt(s) checked" in capsys.readouterr().out


def test_cli_lint_strict_fails_on_warnings(broken_library, capsys):
    assert main(["--library", str(broken_library), "lint", "loop"]) == 0
    assert main(["--library", str(broken_library), "lint", "loop", "--strict"]) == 1
    capsys.readouterr()
    assert main(["--library", str(broken_library), "lint", "loop", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["warnings"] == 1 and data["errors"] == 0
