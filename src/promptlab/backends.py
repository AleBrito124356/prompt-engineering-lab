"""Offline and record/replay backends that share ``NIMClient``'s interface.

Three clients, all subclasses of :class:`~promptlab.client.ChatClient` (so
``chat``, ``complete`` and ``sample`` work everywhere a ``NIMClient`` does):

``ScriptedClient``
    Deterministic fake model. Serves canned responses in order, or by
    matching rules (substring, regex or predicate over the messages), or from
    a default responder. Honors ``stop`` sequences like a real endpoint and
    records every call in ``.calls``. Used by tests and by every technique
    demo's ``--offline`` mode.

``RecordingClient``
    Wraps a real client and appends ``{key, model, messages, params,
    response}`` to a JSONL *cassette* for every call.

``ReplayClient``
    Serves responses from a cassette by request hash, in recorded order for
    repeated identical requests (self-consistency samples). A request that is
    not in the cassette raises :class:`CassetteMissError` with a unified diff
    against the closest recorded request, so a changed prompt is obvious.

``make_client(spec)`` picks one from a backend spec -- ``nim`` (default),
``mock``, ``record:FILE`` or ``replay:FILE`` -- which the CLI takes from
``--offline`` / ``--record`` / ``--replay`` or ``$PROMPTLAB_BACKEND``.

Nothing here touches the network except ``RecordingClient``'s inner client.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path

from .client import DEFAULT_MODEL, BackendError, ChatClient, NIMClient

__all__ = [
    "Rule",
    "ScriptedClient",
    "ScriptExhaustedError",
    "RecordingClient",
    "ReplayClient",
    "CassetteMissError",
    "request_key",
    "mock_model",
    "make_client",
    "backend_from_env",
    "OFFLINE_BANNER",
]

#: Printed whenever scripted responses stand in for a live model.
OFFLINE_BANNER = (
    "[offline] Scripted responses, not a live model: the technique code (parsing, "
    "tool calls, voting, search, validation) runs for real on canned model output."
)


class ScriptExhaustedError(BackendError):
    """A ScriptedClient was called more times than it has responses for."""


class CassetteMissError(BackendError):
    """A ReplayClient got a request its cassette does not contain."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _params(temperature, max_tokens, n, kwargs):
    params = {"temperature": temperature, "max_tokens": max_tokens, "n": n}
    params.update(kwargs)
    return params


def request_key(model, messages, params):
    """Stable SHA-256 of a request (model, messages, sampling params)."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_stop(text, stop):
    """Truncate ``text`` at the first stop sequence, like a real endpoint."""
    if not stop or not isinstance(text, str):
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = len(text)
    for s in stops:
        if s:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
    return text[:cut]


def _all_text(messages):
    return "\n".join(str(m.get("content", "")) for m in messages)


# --------------------------------------------------------------------------- #
# ScriptedClient
# --------------------------------------------------------------------------- #
class Rule:
    """Serve ``responses`` when ``match`` fits the request.

    ``match`` is a substring (searched in every message's content), a compiled
    regex, or a callable ``match(messages) -> bool``. ``responses`` is one
    response or a list; each response is a string or a callable
    ``response(messages, params) -> str``. With ``cycle=True`` (default) the
    list repeats; otherwise the rule stops matching once it is used up.
    """

    def __init__(self, match, responses, *, cycle=True, name=None):
        self.match = match
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else [responses]
        if not self.responses:
            raise ValueError("a Rule needs at least one response")
        self.cycle = cycle
        self.name = name or (match if isinstance(match, str) else getattr(match, "pattern", "rule"))
        self.used = 0

    def matches(self, messages):
        if not self.cycle and self.used >= len(self.responses):
            return False
        if callable(self.match) and not hasattr(self.match, "search"):
            return bool(self.match(messages))
        text = _all_text(messages)
        if hasattr(self.match, "search"):
            return self.match.search(text) is not None
        return str(self.match) in text

    def next(self):
        response = self.responses[self.used % len(self.responses)]
        self.used += 1
        return response


class ScriptedClient(ChatClient):
    """A deterministic fake model for tests and offline demos.

    Resolution order for each call: the first matching :class:`Rule`, then the
    ``responses`` queue (in order), then ``default``. When none applies,
    :class:`ScriptExhaustedError` is raised. ``calls`` records every request.
    """

    def __init__(self, responses=(), *, rules=(), default=None, model="scripted"):
        self.model = model
        self.queue = list(responses)
        self.rules = [r if isinstance(r, Rule) else Rule(*r) for r in rules]
        self.default = default
        self.calls = []

    def _one(self, messages, params):
        response = None
        for rule in self.rules:
            if rule.matches(messages):
                response = rule.next()
                break
        else:
            if self.queue:
                response = self.queue.pop(0)
            elif self.default is not None:
                response = self.default
            else:
                last = messages[-1]["content"] if messages else ""
                raise ScriptExhaustedError(
                    "ScriptedClient has no response for call #{} (last message: {!r})".format(
                        len(self.calls) + 1, str(last)[:80]
                    )
                )
        if callable(response):
            response = response(messages, params)
        response = _apply_stop(response, params.get("stop"))
        self.calls.append({"messages": messages, "params": params, "response": response})
        return response

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, n=1, **kwargs):
        params = _params(temperature, max_tokens, n, kwargs)
        if n == 1:
            return self._one(messages, params)
        return [self._one(messages, params) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Record / replay
# --------------------------------------------------------------------------- #
class RecordingClient(ChatClient):
    """Pass every call to ``inner`` and append it to the JSONL ``path``."""

    def __init__(self, inner, path):
        self.inner = inner
        self.path = Path(path)
        self.model = getattr(inner, "model", DEFAULT_MODEL)

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, n=1, **kwargs):
        params = _params(temperature, max_tokens, n, kwargs)
        response = self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, n=n, **kwargs)
        entry = {
            "key": request_key(self.model, messages, params),
            "model": self.model,
            "messages": messages,
            "params": params,
            "response": response,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return response


def _load_cassette(path):
    path = Path(path)
    if not path.is_file():
        raise BackendError("cassette not found: {}".format(path))
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BackendError("cassette {} line {}: invalid JSON ({})".format(path, lineno, exc)) from None
        if not isinstance(entry, dict) or not {"model", "messages", "params", "response"} <= set(entry):
            raise BackendError("cassette {} line {}: not a recorded request".format(path, lineno))
        entry["key"] = request_key(entry["model"], entry["messages"], entry["params"])
        entries.append(entry)
    return entries


def _pretty_request(model, messages, params):
    return json.dumps(
        {"model": model, "params": params, "messages": messages}, indent=2, sort_keys=True, ensure_ascii=False
    ).splitlines(keepends=True)


class ReplayClient(ChatClient):
    """Serve responses recorded by :class:`RecordingClient`, never the network.

    ``model`` defaults like ``NIMClient`` (``$NIM_MODEL``), else to the only
    model in the cassette, else ``DEFAULT_MODEL`` -- so a cassette recorded
    with the defaults replays with the defaults.
    """

    def __init__(self, path, model=None):
        self.path = Path(path)
        self.entries = _load_cassette(self.path)
        models = {e["model"] for e in self.entries}
        env_model = os.environ.get("NIM_MODEL")
        self.model = model or env_model or (models.pop() if len(models) == 1 else DEFAULT_MODEL)
        self._by_key = {}
        for entry in self.entries:
            self._by_key.setdefault(entry["key"], []).append(entry["response"])
        self._served = {}

    def _miss(self, messages, params, key):
        recorded = self._by_key.get(key)
        if recorded:
            return CassetteMissError(
                "cassette {} has {} recorded response(s) for this request, but it was made {} times; "
                "re-record to capture more samples".format(self.path, len(recorded), self._served[key] + 1)
            )
        wanted = "".join(_pretty_request(self.model, messages, params))
        best, best_ratio = None, -1.0
        for entry in self.entries:
            candidate = "".join(_pretty_request(entry["model"], entry["messages"], entry["params"]))
            ratio = difflib.SequenceMatcher(None, candidate, wanted, autojunk=False).quick_ratio()
            if ratio > best_ratio:
                best, best_ratio = entry, ratio
        message = "no recorded response in cassette {} ({} entries) for this request".format(
            self.path, len(self.entries)
        )
        if best is not None:
            diff = difflib.unified_diff(
                _pretty_request(best["model"], best["messages"], best["params"]),
                _pretty_request(self.model, messages, params),
                fromfile="closest recorded request",
                tofile="this request",
                n=2,
            )
            message += ":\n" + "".join(diff).rstrip()
        return CassetteMissError(message)

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, n=1, **kwargs):
        params = _params(temperature, max_tokens, n, kwargs)
        key = request_key(self.model, messages, params)
        served = self._served.get(key, 0)
        recorded = self._by_key.get(key, [])
        if served >= len(recorded):
            self._served.setdefault(key, served)
            raise self._miss(messages, params, key)
        self._served[key] = served + 1
        return recorded[served]


# --------------------------------------------------------------------------- #
# Generic offline "model" for the CLI (run / compare --offline)
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "about after again also back been before being both could does doing down each from have having here "
    "into just more most much only other over same should some such than that their them then there these "
    "they this those through under until very were what when where which while will with would your".split()
)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)")
# Matches promptlab.compare's judge prompt (anchored on its explicit markers,
# so responses containing blank lines are captured whole).
_JUDGE_RE = re.compile(
    r"Request:\n(?P<request>.*?)\n\nCriteria:.*?--- Response 1 ---\n(?P<r1>.*?)\n\n--- Response 2 ---\n"
    r"(?P<r2>.*?)\n\n--- End of responses ---",
    re.DOTALL,
)


def _clip(text, width):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _keywords(text):
    return {w for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()) if w not in _STOPWORDS}


def _mock_judge(request, first, second):
    keys = _keywords(request)

    def score(response):
        words = _keywords(response)
        items = sum(1 for line in response.splitlines() if _LIST_LINE_RE.match(line))
        return len(keys & words), items

    s1, s2 = score(first), score(second)
    verdict = "1" if s1 > s2 else "2" if s2 > s1 else "tie"
    return (
        "Offline heuristic judge (not an LLM): Response 1 covers {}/{} request keywords with {} "
        "checklist items; Response 2 covers {}/{} with {}.\nWinner: {}".format(
            s1[0], len(keys), s1[1], s2[0], len(keys), s2[1], verdict
        )
    )


def mock_model(messages, params=None):
    """Deterministic stand-in for a chat model, used by ``--offline``.

    - For a pairwise-judge request it scores both responses with a transparent,
      position-independent heuristic (request-keyword coverage, then checklist
      length) and ends with ``Winner: 1|2|tie``.
    - For anything else it returns a labelled answer that echoes the request
      and the first instructions of the system prompt, so different prompt
      versions produce visibly different outputs.
    """
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    judge = _JUDGE_RE.search(user)
    if judge:
        return _mock_judge(judge.group("request"), judge.group("r1"), judge.group("r2"))
    first_line = next((line for line in system.splitlines() if line.strip()), "(no system prompt)")
    rules = []
    for line in system.splitlines():
        m = _LIST_LINE_RE.match(line)
        if m:
            rules.append(_clip(m.group(1).replace("**", ""), 90))
    lines = [
        "[offline mock response -- scripted, not a live model]",
        "Request: {}".format(_clip(user, 100)),
        "Role from the system prompt: {}".format(_clip(first_line, 100)),
    ]
    if rules:
        lines.append("Checklist taken from the system prompt:")
        lines.extend("- {}".format(r) for r in rules[:6])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def backend_from_env():
    """``$PROMPTLAB_BACKEND`` or ``None``."""
    value = os.environ.get("PROMPTLAB_BACKEND", "").strip()
    return value or None


def make_client(spec=None, *, model=None):
    """Build a client from a backend spec.

    ``spec`` is ``None``/``"nim"`` (live NVIDIA NIM), ``"mock"`` (the generic
    :func:`mock_model`), ``"record:FILE"`` (NIM, recorded to FILE) or
    ``"replay:FILE"`` (served from FILE). ``None`` falls back to
    ``$PROMPTLAB_BACKEND`` and then to NIM. Raises ``ValueError`` for an
    unknown spec.
    """
    spec = spec or backend_from_env() or "nim"
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "nim" and not arg:
        return NIMClient(model=model)
    if kind == "mock" and not arg:
        return ScriptedClient(default=mock_model, model=model or "offline-mock")
    if kind == "replay" and arg:
        return ReplayClient(arg, model=model)
    if kind == "record" and arg:
        return RecordingClient(NIMClient(model=model), arg)
    raise ValueError(
        "unknown backend {!r}; use nim, mock, record:FILE or replay:FILE".format(spec)
    )


def is_scripted(client):
    return isinstance(client, ScriptedClient)
