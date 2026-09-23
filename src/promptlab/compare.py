"""A/B compare two prompt variants over a small input set, judged by an LLM.

The point is to replace "the new prompt feels better" with a number. For each
input we run both variants, then ask a judge model which answer is better for
that input. To blunt position bias, the two answers are shown to the judge in a
random order and mapped back afterwards.

The scoring and table rendering are pure functions (``parse_verdict``,
``tally``, ``render_table``) so they can be unit-tested without any network.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .client import NIMClient

__all__ = [
    "ComparisonResult",
    "run_comparison",
    "parse_verdict",
    "tally",
    "render_table",
    "JUDGE_SYSTEM",
    "DEFAULT_CRITERIA",
]

DEFAULT_CRITERIA = "helpfulness, factual correctness, and how well it follows the instruction"

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of AI responses. You compare two "
    "candidate answers to the same request and decide which is better. Judge "
    "only on the stated criteria. Ignore length and style unless they affect "
    "quality. You must end with a single line 'Winner: 1', 'Winner: 2', or "
    "'Winner: tie'."
)

_JUDGE_TEMPLATE = """Request:
{request}

Criteria: {criteria}

--- Response 1 ---
{first}

--- Response 2 ---
{second}

Briefly explain which response is better and why (2-3 sentences), then finish
with exactly one line: 'Winner: 1', 'Winner: 2', or 'Winner: tie'."""


@dataclass
class ComparisonResult:
    """One input's outcome: the two answers, the resolved winner, and a reason."""

    input: str
    response_a: str
    response_b: str
    winner: str  # "A", "B", or "tie"
    reason: str = ""
    meta: dict = field(default_factory=dict)


# Markdown emphasis / code / heading characters judges like to wrap verdicts in.
_MARKDOWN_RE = re.compile(r"[*_`#>]+")
# "Winner: ...", "Final verdict - ...", "Better response = ..." (after markdown is stripped).
_VERDICT_LINE_RE = re.compile(
    r"^\s*(?:final\s+)?(?:winner|verdict|better\s+response|preferred\s+response)\s*(?:is)?\s*[:=\-]\s*(.+)$",
    re.IGNORECASE,
)
# Last-resort prose signal: "the winner is Response 2", "winner: 1". Digits only:
# a bare letter in prose ("the winner is a close call") is not a verdict.
_VERDICT_PROSE_RE = re.compile(
    r"\bwinner\s*(?:is|:)\s*(?:response|answer|option|candidate)?\s*#?\s*([12])\b",
    re.IGNORECASE,
)
_SLOT_RE = re.compile(
    r"^(?:response|answer|option|candidate|assistant)?\s*#?\s*\(?\s*([12ab])\s*\)?(?![\w.])",
    re.IGNORECASE,
)
_TIE_RE = re.compile(r"^(?:it'?s\s+a\s+)?(?:tie|draw|equal|even|neither|both|none)\b", re.IGNORECASE)
_ORDINAL = {"first": "1", "second": "2"}
_LETTER = {"1": "1", "2": "2", "a": "1", "b": "2"}


def _parse_token(token):
    """Map the text after 'Winner:' to '1', '2', 'tie', or None."""
    token = token.strip().strip(" .!:;,'\"()[]").strip()
    if not token:
        return None
    if _TIE_RE.match(token):
        return "tie"
    m = _SLOT_RE.match(token)
    if m:
        return _LETTER[m.group(1).lower()]
    head = token.split()[0].lower()
    if head in _ORDINAL:
        return _ORDINAL[head]
    return None


def parse_verdict(text, default="tie"):
    """Extract '1', '2', or 'tie' from a judge response.

    Accepts the phrasings judges actually produce: ``Winner: 2``, ``Winner:
    Response 2``, ``**Winner:** 2``, ``Winner: **2**``, ``Winner: B`` (A/B map to
    1/2), ``Winner: first`` and ``Winner: tie`` / ``draw`` / ``equal`` /
    ``neither``. The last verdict line wins. When nothing parses, ``default`` is
    returned -- pass ``default=None`` to tell "unparseable" apart from a real tie.
    """
    lines = (text or "").strip().splitlines()
    for line in reversed(lines):
        clean = _MARKDOWN_RE.sub("", line)
        m = _VERDICT_LINE_RE.match(clean)
        if m:
            token = _parse_token(m.group(1))
            if token is not None:
                return token
    # Fallback: a clear prose signal anywhere ("... so the winner is Response 2").
    matches = list(_VERDICT_PROSE_RE.finditer(_MARKDOWN_RE.sub("", text or "")))
    if matches:
        return _LETTER[matches[-1].group(1).lower()]
    return default


def _first_line_reason(text):
    for line in text.strip().splitlines():
        s = line.strip()
        if s and not s.lower().startswith("winner:"):
            return s
    return ""


def judge_pair(client, request, response_a, response_b, *, criteria=DEFAULT_CRITERIA, rng=None):
    """Ask the judge which of A/B is better; returns ('A'|'B'|'tie', reason).

    A/B are shuffled before being shown to the judge and mapped back, so the
    label the judge sees ('Response 1') is decoupled from which variant it is.
    """
    rng = rng or random
    a_is_first = rng.random() < 0.5
    if a_is_first:
        first, second = response_a, response_b
    else:
        first, second = response_b, response_a
    prompt = _JUDGE_TEMPLATE.format(
        request=request, criteria=criteria, first=first, second=second
    )
    raw = client.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=300)
    verdict = parse_verdict(raw)
    reason = _first_line_reason(raw)
    if verdict == "tie":
        return "tie", reason
    picked_first = verdict == "1"
    winner_is_a = picked_first == a_is_first
    return ("A" if winner_is_a else "B"), reason


def run_comparison(
    system_a,
    system_b,
    inputs,
    *,
    client=None,
    judge=None,
    label_a="A",
    label_b="B",
    criteria=DEFAULT_CRITERIA,
    temperature=0.3,
    max_tokens=800,
    seed=None,
):
    """Run both system prompts over ``inputs`` and judge each pair.

    ``system_a`` / ``system_b`` are rendered system-prompt strings. ``inputs``
    is an iterable of user-turn strings. Returns a list of ``ComparisonResult``.
    """
    nim = client or NIMClient()
    judge_client = judge or nim
    rng = random.Random(seed)
    results = []
    for item in inputs:
        ra = nim.complete(item, system=system_a, temperature=temperature, max_tokens=max_tokens)
        rb = nim.complete(item, system=system_b, temperature=temperature, max_tokens=max_tokens)
        winner, reason = judge_pair(judge_client, item, ra, rb, criteria=criteria, rng=rng)
        results.append(
            ComparisonResult(
                input=item, response_a=ra, response_b=rb, winner=winner, reason=reason
            )
        )
    return results


def tally(results, label_a="A", label_b="B"):
    """Count wins/ties across results. Returns a dict with counts and a verdict."""
    a = sum(1 for r in results if r.winner == "A")
    b = sum(1 for r in results if r.winner == "B")
    ties = sum(1 for r in results if r.winner == "tie")
    if a > b:
        overall = label_a
    elif b > a:
        overall = label_b
    else:
        overall = "tie"
    return {
        "label_a": label_a,
        "label_b": label_b,
        label_a: a,
        label_b: b,
        "ties": ties,
        "total": len(results),
        "overall": overall,
    }


def _truncate(text, width):
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def render_table(results, label_a="A", label_b="B", *, input_width=32, reason_width=44):
    """Render an ASCII win table plus a summary line. Pure string output."""
    header = ("#", "input", "winner", "why")
    rows = []
    for i, r in enumerate(results, 1):
        winner = {"A": label_a, "B": label_b, "tie": "tie"}[r.winner]
        rows.append(
            (
                str(i),
                _truncate(r.input, input_width),
                winner,
                _truncate(r.reason, reason_width),
            )
        )

    widths = [
        max(len(header[0]), *(len(row[0]) for row in rows)) if rows else len(header[0]),
        max(len(header[1]), *(len(row[1]) for row in rows)) if rows else len(header[1]),
        max(len(header[2]), *(len(row[2]) for row in rows)) if rows else len(header[2]),
        max(len(header[3]), *(len(row[3]) for row in rows)) if rows else len(header[3]),
    ]

    def fmt(cols):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt(header), sep]
    lines.extend(fmt(row) for row in rows)

    counts = tally(results, label_a, label_b)
    summary = "\n{a}: {av}   {b}: {bv}   ties: {t}   ->  winner: {overall}".format(
        a=label_a,
        av=counts[label_a],
        b=label_b,
        bv=counts[label_b],
        t=counts["ties"],
        overall=counts["overall"],
    )
    return "\n".join(lines) + "\n" + summary
