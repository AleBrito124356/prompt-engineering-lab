"""Tests for the offline backends: scripted, record/replay, mock model, selection."""

import json
import re

import pytest

import promptlab.backends as backends
from promptlab.backends import (
    CassetteMissError,
    RecordingClient,
    ReplayClient,
    Rule,
    ScriptedClient,
    ScriptExhaustedError,
    make_client,
    mock_model,
    request_key,
)
from promptlab.client import DEFAULT_MODEL, BackendError, ChatClient, NIMClient
from promptlab.compare import JUDGE_SYSTEM, _JUDGE_TEMPLATE


def _msgs(user, system="S"):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -- ScriptedClient --------------------------------------------------------- #
def test_scripted_queue_in_order_then_exhausted():
    client = ScriptedClient(["one", "two"])
    assert client.complete("a") == "one"
    assert client.chat(_msgs("b")) == "two"
    with pytest.raises(ScriptExhaustedError, match="call #3"):
        client.complete("c")
    assert [c["response"] for c in client.calls] == ["one", "two"]
    assert client.calls[0]["params"]["temperature"] == 0.7


def test_scripted_rules_substring_regex_predicate_and_callable():
    client = ScriptedClient(
        rules=[
            Rule("weather", "sunny"),
            Rule(re.compile(r"\d+ \+ \d+"), lambda messages, params: "math:" + messages[-1]["content"]),
            (lambda messages: messages[0]["content"] == "JUDGE", ["Winner: 1", "Winner: 2"]),
        ],
        default="fallback",
    )
    assert client.complete("what's the weather?") == "sunny"
    assert client.complete("2 + 2") == "math:2 + 2"
    assert client.complete("x", system="JUDGE") == "Winner: 1"
    assert client.complete("x", system="JUDGE") == "Winner: 2"
    assert client.complete("x", system="JUDGE") == "Winner: 1"  # cycles
    assert client.complete("anything else") == "fallback"


def test_scripted_non_cycling_rule_falls_through_to_queue():
    client = ScriptedClient(["queued"], rules=[Rule("hi", "once", cycle=False)])
    assert client.complete("hi") == "once"
    assert client.complete("hi") == "queued"


def test_scripted_honours_stop_sequences_like_a_real_endpoint():
    client = ScriptedClient(["Action: calculator\nAction Input: 2+2\nObservation: 5 (hallucinated)"])
    out = client.chat(_msgs("q"), stop=["Observation:"])
    assert out == "Action: calculator\nAction Input: 2+2\n"


def test_scripted_n_and_sample():
    client = ScriptedClient(default="same")
    assert client.chat(_msgs("q"), n=3) == ["same"] * 3
    assert client.sample(_msgs("q"), 2) == ["same", "same"]
    assert len(client.calls) == 5


def test_rule_requires_a_response():
    with pytest.raises(ValueError):
        Rule("x", [])


def test_every_backend_is_a_chat_client():
    assert issubclass(ScriptedClient, ChatClient)
    assert issubclass(NIMClient, ChatClient)
    assert issubclass(ReplayClient, ChatClient)
    assert issubclass(RecordingClient, ChatClient)


# -- record / replay -------------------------------------------------------- #
def test_record_then_replay_round_trip(tmp_path):
    cassette = tmp_path / "run.jsonl"
    inner = ScriptedClient(["first", "sample A", "sample B", "judge"], model="nim-model")
    recorder = RecordingClient(inner, cassette)
    got = [
        recorder.complete("q1", system="sys", temperature=0.2),
        *recorder.sample(_msgs("same prompt"), 2, temperature=0.9),  # identical requests
        recorder.chat(_msgs("judge me"), temperature=0.0, max_tokens=300, stop=["END"]),
    ]
    lines = cassette.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    entry = json.loads(lines[0])
    assert set(entry) == {"key", "model", "messages", "params", "response"}
    assert entry["model"] == "nim-model" and entry["params"]["temperature"] == 0.2

    replay = ReplayClient(cassette)
    assert replay.model == "nim-model"  # the only model in the cassette
    replayed = [
        replay.complete("q1", system="sys", temperature=0.2),
        *replay.sample(_msgs("same prompt"), 2, temperature=0.9),
        replay.chat(_msgs("judge me"), temperature=0.0, max_tokens=300, stop=["END"]),
    ]
    assert replayed == got == ["first", "sample A", "sample B", "judge"]


def test_replay_changed_prompt_raises_miss_with_diff(tmp_path):
    cassette = tmp_path / "c.jsonl"
    RecordingClient(ScriptedClient(["ok"], model="m"), cassette).complete("Summarize in 3 bullets", system="v1")
    replay = ReplayClient(cassette)
    with pytest.raises(CassetteMissError) as err:
        replay.complete("Summarize in 5 bullets", system="v1")
    message = str(err.value)
    assert "no recorded response" in message
    assert "-" in message and "3 bullets" in message and "5 bullets" in message
    assert isinstance(err.value, BackendError)


def test_replay_params_are_part_of_the_key(tmp_path):
    cassette = tmp_path / "c.jsonl"
    RecordingClient(ScriptedClient(["ok"], model="m"), cassette).complete("q", temperature=0.2)
    with pytest.raises(CassetteMissError, match="temperature"):
        ReplayClient(cassette).complete("q", temperature=0.7)


def test_replay_more_calls_than_recorded(tmp_path):
    cassette = tmp_path / "c.jsonl"
    RecordingClient(ScriptedClient(["only one"], model="m"), cassette).complete("q")
    replay = ReplayClient(cassette)
    assert replay.complete("q") == "only one"
    with pytest.raises(CassetteMissError, match="made 2 times"):
        replay.complete("q")


def test_replay_model_resolution(tmp_path, monkeypatch):
    cassette = tmp_path / "c.jsonl"
    rec = RecordingClient(ScriptedClient(default="x", model="a"), cassette)
    rec.complete("q")
    rec.inner.model = rec.model = "b"
    rec.complete("q")
    monkeypatch.delenv("NIM_MODEL", raising=False)
    assert ReplayClient(cassette).model == DEFAULT_MODEL  # two models: fall back
    assert ReplayClient(cassette, model="b").complete("q") == "x"
    monkeypatch.setenv("NIM_MODEL", "a")
    assert ReplayClient(cassette).model == "a"


@pytest.mark.parametrize("content,message", [("{not json", "invalid JSON"), ('{"a": 1}', "not a recorded")])
def test_replay_rejects_malformed_cassettes(tmp_path, content, message):
    cassette = tmp_path / "c.jsonl"
    cassette.write_text(content + "\n", encoding="utf-8")
    with pytest.raises(BackendError, match=message):
        ReplayClient(cassette)


def test_replay_missing_file():
    with pytest.raises(BackendError, match="cassette not found"):
        ReplayClient("does-not-exist.jsonl")


def test_request_key_is_stable_and_order_insensitive_for_params():
    a = request_key("m", _msgs("q"), {"temperature": 0.1, "n": 1})
    b = request_key("m", _msgs("q"), {"n": 1, "temperature": 0.1})
    assert a == b and len(a) == 64
    assert a != request_key("m2", _msgs("q"), {"n": 1, "temperature": 0.1})


# -- mock_model ------------------------------------------------------------- #
def test_mock_model_answer_echoes_request_and_system_checklist():
    system = "You are a helper.\n\nRules:\n- Be brief.\n- **Cite** sources.\n1. Numbered rule."
    out = mock_model(_msgs("How do I parse CSV?", system))
    assert out.startswith("[offline mock response")
    assert "Request: How do I parse CSV?" in out
    assert "Role from the system prompt: You are a helper." in out
    assert "- Cite sources." in out and "- Numbered rule." in out


def _judge_messages(request, first, second):
    user = _JUDGE_TEMPLATE.format(request=request, criteria="c", first=first, second=second)
    return _msgs(user, JUDGE_SYSTEM)


def test_mock_judge_is_position_independent_and_handles_blank_lines():
    good = "Use csv.reader to parse the CSV file row by row.\n\n- stream it\n- handle quoting"
    weak = "Try pandas."
    request = "How do I parse a large CSV file?"
    v1 = mock_model(_judge_messages(request, good, weak))
    v2 = mock_model(_judge_messages(request, weak, good))
    assert v1.endswith("Winner: 1") and v2.endswith("Winner: 2")
    assert "not an LLM" in v1
    assert mock_model(_judge_messages(request, good, good)).endswith("Winner: tie")


# -- make_client ------------------------------------------------------------ #
def test_make_client_specs(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMPTLAB_BACKEND", raising=False)
    assert isinstance(make_client(), NIMClient)
    assert isinstance(make_client("nim", model="x"), NIMClient)
    mock = make_client("mock")
    assert isinstance(mock, ScriptedClient) and mock.complete("hi").startswith("[offline mock")
    rec = make_client("record:" + str(tmp_path / "r.jsonl"), model="m")
    assert isinstance(rec, RecordingClient) and isinstance(rec.inner, NIMClient)
    cassette = tmp_path / "c.jsonl"
    RecordingClient(ScriptedClient(["hi"], model="m"), cassette).complete("q")
    assert make_client("replay:" + str(cassette)).complete("q") == "hi"
    monkeypatch.setenv("PROMPTLAB_BACKEND", "mock")
    assert isinstance(make_client(), ScriptedClient)
    for bad in ("bogus", "replay", "record:", "mock:x"):
        with pytest.raises(ValueError, match="unknown backend"):
            make_client(bad)


def test_backend_from_env(monkeypatch):
    monkeypatch.setenv("PROMPTLAB_BACKEND", "  ")
    assert backends.backend_from_env() is None
    monkeypatch.setenv("PROMPTLAB_BACKEND", "replay:x.jsonl")
    assert backends.backend_from_env() == "replay:x.jsonl"
