"""CLI tests: every command through ``main(argv)``, plus clean error handling."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import promptlab.client as client_mod
from promptlab.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "library"


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    """No test may reach NIM: no key, and no .env loading."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTLAB_BACKEND", raising=False)
    monkeypatch.delenv("PROMPTLAB_LIBRARY", raising=False)
    monkeypatch.setattr(client_mod, "_load_dotenv", lambda: None)


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# -- offline commands ------------------------------------------------------- #
def test_list(capsys):
    code, out, _ = run_cli(capsys, "list")
    assert code == 0
    assert "coding-assistant" in out and "[v1,v2]" in out
    assert len(out.strip().splitlines()) >= 22


def test_list_empty_library(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "--library", str(tmp_path), "list")
    assert code == 0 and "No prompts found" in out


def test_show_prints_contract(capsys):
    code, out, _ = run_cli(capsys, "show", "translator")
    assert code == 0
    assert "target_language (required; one of: Spanish, English)" in out
    assert "register (optional)" in out
    assert "You are a professional translator" in out


def test_render_with_vars_and_vars_json(tmp_path, capsys):
    vars_file = tmp_path / "vars.json"
    vars_file.write_text(json.dumps({"dialect": "SQLite"}), encoding="utf-8")
    code, out, _ = run_cli(capsys, "render", "sql-expert", "--vars-json", str(vars_file))
    assert code == 0 and "SQLite" in out
    code, out, _ = run_cli(capsys, "render", "sql-expert", "--vars-json", str(vars_file), "--var", "dialect=MySQL")
    assert "MySQL" in out  # --var overrides --vars-json


def test_render_no_strict(capsys):
    code, out, _ = run_cli(capsys, "render", "sql-expert", "--no-strict")
    assert code == 0 and "You are" in out


def test_render_optional_inputs_regression(capsys):
    code, out, _ = run_cli(capsys, "render", "interviewer", "--var", "role=backend engineer")
    assert code == 0 and "backend engineer position" in out


def test_diff(capsys):
    code, out, _ = run_cli(capsys, "diff", "coding-assistant", "--a", "v1", "--b", "v2")
    assert code == 0 and "coding-assistant@v1" in out and "+Answer in this order:" in out
    code, out, _ = run_cli(capsys, "diff", "coding-assistant", "--a", "v2", "--b", "latest")
    assert "No differences" in out


def test_techniques(capsys):
    code, out, _ = run_cli(capsys, "techniques")
    assert code == 0 and "self_consistency" in out and "--offline" in out


def test_custom_library_flag(capsys):
    code, out, _ = run_cli(capsys, "--library", str(FIXTURES), "render", "greeting", "--var", "name=Ada",
                           "--var", "team=Ops")
    assert code == 0 and "Hey Ada!" in out and "-- The Ops team" in out


# -- clean errors ----------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv,message",
    [
        (["render", "sql-expert"], "missing variables: dialect (pass --var key=value or --no-strict)"),
        (["run", "sql-expert", "--user", "x"], "sql-expert@v2: missing variables: dialect"),
        (
            ["compare", "coding-assistant@v1", "coding-assistant@v2", "--inputs", "examples/coding-questions.txt"],
            "coding-assistant@v1: missing variables: language",
        ),
        (["render", "translator", "--var", "target_language=Klingon"], "expected one of: Spanish, English"),
        (["render", "nope"], "no prompt 'nope'"),
        (["render", "coding-assistant", "-V", "v9", "--var", "language=x"], "unknown version 'v9'"),
        (["render", "sql-expert", "--var", "novalue"], "--var expects key=value"),
        (["render", "sql-expert", "--vars-json", "missing.json"], "--vars-json file not found"),
        (["--library", "does/not/exist", "list"], "library root does not exist"),
        (
            ["compare", "sql-expert", "sql-expert@v1", "--var", "dialect=x", "--inputs", "nope.txt"],
            "inputs file not found: nope.txt",
        ),
    ],
)
def test_expected_errors_exit_2_with_one_line(argv, message, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    code, out, err = run_cli(capsys, *argv)
    assert code == 2
    assert err.startswith("error: ")
    assert message in err
    assert "Traceback" not in err + out


@pytest.mark.parametrize(
    "content,message",
    [("[1, 2]", "expected a JSON object of variables, got list"), ("{oops", "invalid JSON")],
)
def test_bad_vars_json(tmp_path, capsys, content, message):
    path = tmp_path / "v.json"
    path.write_text(content, encoding="utf-8")
    code, _, err = run_cli(capsys, "render", "sql-expert", "--vars-json", str(path))
    assert code == 2 and message in err


@pytest.mark.parametrize(
    "content,message",
    [("", "contains no inputs"), ("\n  \n", "contains no inputs"), ('{"inputs": 3}', "expected a JSON array"),
     ("[1, ", "invalid JSON")],
)
def test_bad_inputs_file(tmp_path, capsys, content, message):
    path = tmp_path / "in.txt"
    path.write_text(content, encoding="utf-8")
    code, _, err = run_cli(
        capsys, "compare", "sql-expert@v1", "sql-expert@v2", "--var", "dialect=x", "--inputs", str(path)
    )
    assert code == 2 and message in err


def test_run_without_key_is_a_clean_error(capsys):
    code, out, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=PostgreSQL", "--user", "top 5")
    assert code == 2
    assert err.startswith("error: NVIDIA_API_KEY is not set.")
    assert "build.nvidia.com" in err
    assert "Traceback" not in err


def test_backend_error_exits_1(monkeypatch, capsys):
    class _SDKError(Exception):
        pass

    _SDKError.__module__ = "openai._exceptions"

    class _Completions:
        def create(self, **kwargs):
            raise _SDKError("401 Unauthorized")

    class _Fake:
        class chat:  # noqa: N801 - mimics the SDK attribute layout
            completions = _Completions()

    monkeypatch.setattr(client_mod, "get_openai_client", lambda: _Fake())
    code, _, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=x", "--user", "q")
    assert code == 1
    assert "NIM request to model" in err and "401 Unauthorized" in err


def test_real_process_has_no_traceback_without_key():
    """Regression: `python cli.py run ...` printed a full traceback first."""
    env = {k: v for k, v in __import__("os").environ.items() if k != "PROMPTLAB_BACKEND"}
    # Present-but-empty: python-dotenv never overrides it, so even a stray .env
    # up the tree cannot turn this test into a live NIM call.
    env["NVIDIA_API_KEY"] = ""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), "run", "sql-expert", "--var", "dialect=PostgreSQL", "--user", "x"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),  # no .env here
        env=env,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr + proc.stdout
    assert proc.stderr.startswith("error: NVIDIA_API_KEY is not set.")


# -- offline / record / replay ---------------------------------------------- #
def test_run_offline(capsys):
    code, out, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=PostgreSQL", "--user", "top 5", "--offline")
    assert code == 0
    assert out.startswith("[offline mock response")
    assert "You are a SQL expert writing PostgreSQL." in out
    assert "note: offline mode" in err


def test_run_offline_via_env(monkeypatch, capsys):
    monkeypatch.setenv("PROMPTLAB_BACKEND", "mock")
    code, out, _ = run_cli(capsys, "run", "interviewer", "--var", "role=backend engineer", "--user", "hi")
    assert code == 0 and "backend engineer position" in out


def test_compare_offline_end_to_end(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    code, out, err = run_cli(
        capsys, "compare", "coding-assistant@v1", "coding-assistant@v2", "--var", "language=Python",
        "--inputs", "examples/coding-questions.txt", "--offline",
    )
    assert code == 0
    assert "Offline heuristic judge" in out
    assert "note: offline mode" in err


def test_run_record_then_replay_gives_identical_output(tmp_path, monkeypatch, capsys):
    import promptlab.backends as backends

    cassette = tmp_path / "run.jsonl"
    monkeypatch.setattr(backends, "NIMClient", lambda model=None: backends.ScriptedClient(["live answer"], model="m"))
    args = ["run", "sql-expert", "--var", "dialect=SQLite", "--user", "count rows"]
    code, recorded, _ = run_cli(capsys, *args, "--record", str(cassette))
    assert code == 0 and recorded.strip() == "live answer"
    code, replayed, _ = run_cli(capsys, *args, "--replay", str(cassette))
    assert code == 0 and replayed == recorded
    # a changed prompt is a clear cassette miss, not a silent live call
    code, _, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=MySQL", "--user", "count rows",
                           "--replay", str(cassette))
    assert code == 1
    assert "no recorded response" in err and "MySQL" in err and "SQLite" in err


def test_backend_flag_errors(capsys, monkeypatch):
    code, _, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=x", "--user", "q", "--replay", "nope.jsonl")
    assert code == 2 and "cassette not found" in err
    monkeypatch.setenv("PROMPTLAB_BACKEND", "bogus")
    code, _, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=x", "--user", "q")
    assert code == 2 and "unknown backend 'bogus'" in err
    with pytest.raises(SystemExit):  # argparse: the flags are mutually exclusive
        main(["run", "sql-expert", "--offline", "--replay", "x.jsonl"])


def test_key_error_mentions_offline(capsys):
    code, _, err = run_cli(capsys, "run", "sql-expert", "--var", "dialect=x", "--user", "q")
    assert code == 2 and "--offline" in err


# -- runner ----------------------------------------------------------------- #
def test_runner_render_and_run_with_backends():
    from promptlab.backends import ScriptedClient
    from promptlab.runner import default_library, render_prompt, run_prompt

    assert "coding-assistant" in default_library().names()
    assert "PostgreSQL" in render_prompt("sql-expert", {"dialect": "PostgreSQL"})
    client = ScriptedClient(["ok"])
    assert run_prompt("sql-expert", {"dialect": "x"}, user_input="q", client=client) == "ok"
    assert client.calls[0]["messages"][0]["content"].startswith("You are a SQL expert writing x.")
    assert client.calls[0]["messages"][1]["content"] == "q"
    out = run_prompt("interviewer", {"role": "analyst"}, backend="mock")
    assert "Request: Begin." in out and "analyst position" in out
