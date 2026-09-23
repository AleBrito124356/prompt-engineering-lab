"""A/B compare two prompt variants over an input set, judged by an LLM.

The point is to replace "the new prompt feels better" with evidence. For each
input both variants answer, then a judge model decides which answer is better.
Two things make that verdict trustworthy:

**Position-bias control.** LLM judges favour whichever answer they see first
(or second). By default every pair is judged *twice*, once in each order. A
pair only counts as a win when both orders agree; when they disagree it is a
tie and counts toward the reported *position-bias rate*. ``--single-order``
restores the old behaviour (one judgement in a seeded random order).

**Statistics instead of a raw count.** Wins exclude ties; the report gives
the win rate with a Wilson confidence interval and a two-sided exact sign-test
p-value (standard library only). The verdict is a variant name only when
``p < alpha``; otherwise it is "no significant difference" -- a 4-3 split, or
even a 5-0 sweep, is not evidence at ``alpha = 0.05``.

With ``repeats > 1`` each variant is sampled that many times per input; each
sample pair is judged and the input's winner is the majority across repeats,
so the sign test still counts independent inputs, not correlated samples.

The scoring, statistics and rendering are pure functions (``parse_verdict``,
``tally``, ``sign_test_p``, ``wilson_interval``, ``render_table``,
``render_markdown``) so they can be tested without any network.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

__all__ = [
    "ComparisonResult",
    "PairResult",
    "Judgement",
    "run_comparison",
    "judge_pair",
    "judge_both_orders",
    "parse_verdict",
    "tally",
    "sign_test_p",
    "wilson_interval",
    "min_decisive_for_significance",
    "render_table",
    "render_markdown",
    "report_dict",
    "write_report",
    "load_report",
    "results_from_report",
    "JUDGE_SYSTEM",
    "DEFAULT_CRITERIA",
    "NO_DIFFERENCE",
]

DEFAULT_CRITERIA = "helpfulness, factual correctness, and how well it follows the instruction"
NO_DIFFERENCE = "no significant difference"

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

--- End of responses ---

Briefly explain which response is better and why (2-3 sentences), then finish
with exactly one line: 'Winner: 1', 'Winner: 2', or 'Winner: tie'."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class Judgement:
    """One judge call. ``order`` is ``"AB"`` when A was shown as Response 1."""

    order: str
    slot: object  # "1", "2", "tie", or None when the verdict could not be parsed
    winner: str  # "A", "B" or "tie" (an unparsed verdict counts as a tie)
    raw: str

    def to_dict(self):
        return {"order": self.order, "slot": self.slot, "winner": self.winner, "raw": self.raw}


@dataclass
class PairResult:
    """One sample of A against one sample of B, judged in one or both orders."""

    response_a: str
    response_b: str
    winner: str
    judgements: list = field(default_factory=list)
    consistent: object = None  # True/False with both orders, None for a single order
    position_bias: bool = False  # the judge picked the same *slot* in both orders
    reason: str = ""

    @property
    def unparsed(self):
        return any(j.slot is None for j in self.judgements)

    def to_dict(self):
        return {
            "response_a": self.response_a,
            "response_b": self.response_b,
            "winner": self.winner,
            "consistent": self.consistent,
            "position_bias": self.position_bias,
            "reason": self.reason,
            "judgements": [j.to_dict() for j in self.judgements],
        }


@dataclass
class ComparisonResult:
    """One input's outcome: the answers, the resolved winner, and a reason.

    ``response_a``/``response_b`` are the first sample's answers; ``pairs``
    holds every judged sample pair (one per repeat).
    """

    input: str
    response_a: str
    response_b: str
    winner: str  # "A", "B", or "tie"
    reason: str = ""
    meta: dict = field(default_factory=dict)
    pairs: list = field(default_factory=list)

    def to_dict(self):
        return {
            "input": self.input,
            "winner": self.winner,
            "reason": self.reason,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "meta": self.meta,
            "pairs": [p.to_dict() for p in self.pairs],
        }


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #
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
    for line in (text or "").strip().splitlines():
        s = line.strip()
        if s and not _VERDICT_LINE_RE.match(_MARKDOWN_RE.sub("", s)):
            return s
    return ""


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #
def _ask_judge(client, request, first, second, criteria):
    prompt = _JUDGE_TEMPLATE.format(request=request, criteria=criteria, first=first, second=second)
    return client.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=300)


def _judgement(order, raw):
    slot = parse_verdict(raw, default=None)
    if slot in ("1", "2"):
        a_first = order == "AB"
        winner = "A" if (slot == "1") == a_first else "B"
    else:
        winner = "tie"
    return Judgement(order=order, slot=slot, winner=winner, raw=raw)


def judge_pair(client, request, response_a, response_b, *, criteria=DEFAULT_CRITERIA, rng=None):
    """Single-order judging: returns ``('A'|'B'|'tie', reason)``.

    A/B are shown in a random order (from ``rng``) and mapped back. This is
    the ``--single-order`` mode; prefer :func:`judge_both_orders`, which
    detects position bias instead of hoping the shuffle averages it out.
    """
    rng = rng or random
    order = "AB" if rng.random() < 0.5 else "BA"
    first, second = (response_a, response_b) if order == "AB" else (response_b, response_a)
    j = _judgement(order, _ask_judge(client, request, first, second, criteria))
    return j.winner, _first_line_reason(j.raw)


def judge_both_orders(client, request, response_a, response_b, *, criteria=DEFAULT_CRITERIA):
    """Judge A-vs-B with A shown first, then with B shown first.

    The pair is a win only when both orders name the same variant. If they
    disagree -- the signature of position bias -- or either verdict cannot be
    parsed, the pair is a tie and ``consistent`` is ``False``.
    """
    ab = _judgement("AB", _ask_judge(client, request, response_a, response_b, criteria))
    ba = _judgement("BA", _ask_judge(client, request, response_b, response_a, criteria))
    unparsed = ab.slot is None or ba.slot is None
    consistent = (not unparsed) and ab.winner == ba.winner
    winner = ab.winner if consistent else "tie"
    position_bias = (not unparsed) and ab.slot == ba.slot and ab.slot in ("1", "2")
    if unparsed:
        reason = "judge verdict could not be parsed ({}); counted as a tie".format(
            ", ".join("{} first".format(j.order[0]) for j in (ab, ba) if j.slot is None)
        )
    elif consistent:
        reason = _first_line_reason(ab.raw)
    else:
        reason = "orders disagree: {} with A first, {} with B first{}".format(
            ab.winner, ba.winner, " (position bias)" if position_bias else ""
        )
    return PairResult(
        response_a=response_a,
        response_b=response_b,
        winner=winner,
        judgements=[ab, ba],
        consistent=consistent,
        position_bias=position_bias,
        reason=reason,
    )


def _majority(pairs):
    a = sum(1 for p in pairs if p.winner == "A")
    b = sum(1 for p in pairs if p.winner == "B")
    return "A" if a > b else "B" if b > a else "tie"


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
    both_orders=True,
    repeats=1,
    on_progress=None,
):
    """Run both system prompts over ``inputs`` and judge every pair.

    ``system_a`` / ``system_b`` are rendered system-prompt strings; ``inputs``
    an iterable of user turns. ``judge`` is a separate client for judging (a
    different model avoids self-preference); it defaults to ``client``.
    ``both_orders=False`` judges once in a ``seed``-ed random order.
    ``repeats`` samples each variant that many times per input. Returns a list
    of :class:`ComparisonResult`. Raises ``ValueError`` for no inputs or
    ``repeats < 1``.
    """
    from .backends import make_client  # deferred: keeps pure imports light

    inputs = [str(i) for i in inputs]
    if not inputs:
        raise ValueError("no inputs to compare")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    llm = client or make_client()
    judge_client = judge or llm
    rng = random.Random(seed)
    results = []
    for index, item in enumerate(inputs, 1):
        pairs = []
        for _ in range(repeats):
            ra = llm.complete(item, system=system_a, temperature=temperature, max_tokens=max_tokens)
            rb = llm.complete(item, system=system_b, temperature=temperature, max_tokens=max_tokens)
            if both_orders:
                pair = judge_both_orders(judge_client, item, ra, rb, criteria=criteria)
            else:
                order = "AB" if rng.random() < 0.5 else "BA"
                first, second = (ra, rb) if order == "AB" else (rb, ra)
                j = _judgement(order, _ask_judge(judge_client, item, first, second, criteria))
                pair = PairResult(ra, rb, j.winner, [j], None, False, _first_line_reason(j.raw))
            pairs.append(pair)
        winner = _majority(pairs)
        reason = next((p.reason for p in pairs if p.winner == winner), pairs[0].reason)
        results.append(
            ComparisonResult(
                input=item,
                response_a=pairs[0].response_a,
                response_b=pairs[0].response_b,
                winner=winner,
                reason=reason,
                pairs=pairs,
            )
        )
        if on_progress:
            on_progress(index, len(inputs), results[-1])
    return results


# --------------------------------------------------------------------------- #
# Statistics (standard library only)
# --------------------------------------------------------------------------- #
def sign_test_p(wins_a, wins_b):
    """Two-sided exact sign test: P(a split at least this lopsided | no difference).

    Ties are excluded before calling. ``sign_test_p(6, 0) == 0.03125``;
    ``sign_test_p(0, 0) == 1.0``.
    """
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def wilson_interval(successes, n, alpha=0.05):
    """Wilson score interval for a proportion; ``(0.0, 1.0)`` when ``n == 0``."""
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def min_decisive_for_significance(alpha=0.05):
    """Fewest decisive inputs for which even a clean sweep has ``p < alpha``."""
    n = 1
    while sign_test_p(n, 0) >= alpha:
        n += 1
    return n


def tally(results, label_a="A", label_b="B", *, alpha=0.05):
    """Counts, statistics and the verdict for a list of :class:`ComparisonResult`.

    Keys kept from v0.1: ``label_a``, ``label_b``, ``<label_a>``, ``<label_b>``,
    ``ties``, ``total`` and ``overall`` (the raw leader, or ``"tie"`` -- not a
    significance claim). New: ``decisive``, ``win_rate_a`` / ``win_rate_b``
    (ties excluded, ``None`` without decisive inputs), ``ci_a`` / ``ci_b``
    (Wilson intervals), ``p_value`` (sign test), ``alpha``, ``significant``,
    ``verdict`` (a label or ``"no significant difference"``), ``winner_side``
    (``"A"``, ``"B"`` or ``None``), ``min_decisive``, and judge diagnostics over
    all judged pairs: ``pairs``, ``pairs_both_orders``, ``inconsistent``,
    ``position_biased``, ``position_bias_rate``, ``unparsed``.
    """
    a = sum(1 for r in results if r.winner == "A")
    b = sum(1 for r in results if r.winner == "B")
    ties = sum(1 for r in results if r.winner == "tie")
    decisive = a + b
    overall = label_a if a > b else label_b if b > a else "tie"
    p_value = sign_test_p(a, b)
    significant = decisive > 0 and p_value < alpha
    winner_side = ("A" if a > b else "B") if significant else None
    ci_b = wilson_interval(b, decisive, alpha)
    ci_a = (1 - ci_b[1], 1 - ci_b[0]) if decisive else (0.0, 1.0)

    pairs = [p for r in results for p in r.pairs]
    both = [p for p in pairs if len(p.judgements) == 2]
    inconsistent = sum(1 for p in both if p.consistent is False)
    biased = sum(1 for p in both if p.position_bias)
    return {
        "label_a": label_a,
        "label_b": label_b,
        label_a: a,
        label_b: b,
        "ties": ties,
        "total": len(results),
        "overall": overall,
        "decisive": decisive,
        "win_rate_a": a / decisive if decisive else None,
        "win_rate_b": b / decisive if decisive else None,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "p_value": p_value,
        "alpha": alpha,
        "significant": significant,
        "winner_side": winner_side,
        "verdict": {"A": label_a, "B": label_b}.get(winner_side, NO_DIFFERENCE),
        "min_decisive": min_decisive_for_significance(alpha),
        "pairs": len(pairs),
        "pairs_both_orders": len(both),
        "inconsistent": inconsistent,
        "position_biased": biased,
        "position_bias_rate": biased / len(both) if both else None,
        "unparsed": sum(1 for p in pairs if p.unparsed),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _truncate(text, width):
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    # ASCII on purpose: a Windows cp1252 console cannot print U+2026.
    return text[: width - 3] + "..."


def _pct(x):
    return "n/a" if x is None else "{:.0f}%".format(100 * x)


def _orders_cell(result):
    if not result.pairs:
        return "-"
    if len(result.pairs) == 1:
        return "/".join(j.winner if j.slot is not None else "?" for j in result.pairs[0].judgements)
    counts = {w: sum(1 for p in result.pairs if p.winner == w) for w in ("A", "B", "tie")}
    return "A{} B{} T{}".format(counts["A"], counts["B"], counts["tie"])


def summary_lines(counts):
    """The human-readable summary under the table (also used in reports)."""
    la, lb = counts["label_a"], counts["label_b"]
    lines = [
        "{a}: {av}   {b}: {bv}   ties: {t}   ({n} inputs, {d} decisive)".format(
            a=la, av=counts[la], b=lb, bv=counts[lb], t=counts["ties"], n=counts["total"], d=counts["decisive"]
        )
    ]
    if counts["decisive"]:
        lo, hi = counts["ci_b"]
        lines.append(
            "{b} win rate (ties excluded): {r}  [{c:.0f}% CI {lo}-{hi}]   sign test p = {p:.3f}".format(
                b=lb, r=_pct(counts["win_rate_b"]), c=100 * (1 - counts["alpha"]), lo=_pct(lo), hi=_pct(hi),
                p=counts["p_value"],
            )
        )
    if counts["pairs_both_orders"]:
        lines.append(
            "judge consistency: {ok}/{n} pairs agreed across both orders; position-bias rate {rate}{un}".format(
                ok=counts["pairs_both_orders"] - counts["inconsistent"],
                n=counts["pairs_both_orders"],
                rate=_pct(counts["position_bias_rate"]),
                un="; {} unparseable verdict(s)".format(counts["unparsed"]) if counts["unparsed"] else "",
            )
        )
    elif counts["pairs"]:
        lines.append("judge consistency: not measured (--single-order)")
    if counts["significant"]:
        verdict = "->  winner: {} (p = {:.3f} < {})".format(counts["verdict"], counts["p_value"], counts["alpha"])
    else:
        verdict = "->  winner: none -- {} at alpha = {}".format(NO_DIFFERENCE, counts["alpha"])
        if counts["decisive"] < counts["min_decisive"]:
            verdict += " (with {} decisive input(s) no split can be significant; need at least {})".format(
                counts["decisive"], counts["min_decisive"]
            )
    lines.append(verdict)
    return lines


def render_table(results, label_a="A", label_b="B", *, alpha=0.05, input_width=32, reason_width=44):
    """Render an ASCII win table plus the statistical summary. Pure string output."""
    header = ("#", "input", "winner", "orders", "why")
    rows = []
    for i, r in enumerate(results, 1):
        winner = {"A": label_a, "B": label_b, "tie": "tie"}[r.winner]
        rows.append(
            (str(i), _truncate(r.input, input_width), winner, _orders_cell(r), _truncate(r.reason, reason_width))
        )
    widths = [max([len(header[c])] + [len(row[c]) for row in rows]) for c in range(len(header))]

    def fmt(cols):
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols)).rstrip()

    lines = [fmt(header), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    counts = tally(results, label_a, label_b, alpha=alpha)
    return "\n".join(lines) + "\n\n" + "\n".join(summary_lines(counts))


def _fence(text):
    body = str(text).rstrip()
    ticks = "````" if "```" in body else "```"
    return "{t}text\n{b}\n{t}".format(t=ticks, b=body)


def _md_cell(text, width=60):
    return _truncate(text, width).replace("|", "\\|")


def render_markdown(results, label_a="A", label_b="B", *, alpha=0.05, config=None):
    """A self-contained Markdown report: verdict, stats, table, and every answer."""
    counts = tally(results, label_a, label_b, alpha=alpha)
    config = config or {}
    out = ["# A/B comparison: {} vs {}".format(label_a, label_b), ""]
    verdict = counts["verdict"] if counts["significant"] else NO_DIFFERENCE
    out += [
        "**Verdict: {}** (two-sided sign test, alpha = {})".format(verdict, alpha),
        "",
        "```text",
        *summary_lines(counts),
        "```",
        "",
    ]
    if config:
        out += ["| setting | value |", "|---|---|"]
        out += ["| {} | {} |".format(k, _md_cell(v, 80)) for k, v in config.items()]
        out.append("")
    out += ["## Results", "", "| # | input | winner | orders | why |", "|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        winner = {"A": label_a, "B": label_b, "tie": "tie"}[r.winner]
        out.append(
            "| {} | {} | {} | {} | {} |".format(i, _md_cell(r.input), winner, _orders_cell(r), _md_cell(r.reason))
        )
    for i, r in enumerate(results, 1):
        out += ["", "## {}. {}".format(i, _truncate(r.input, 100)), ""]
        out += ["Input:", "", _fence(r.input), ""]
        for k, pair in enumerate(r.pairs or [], 1):
            if len(r.pairs) > 1:
                out += ["### Sample {} -- winner: {}".format(k, pair.winner), ""]
            out += ["**{} (A)**".format(label_a), "", _fence(pair.response_a), ""]
            out += ["**{} (B)**".format(label_b), "", _fence(pair.response_b), ""]
            for j in pair.judgements:
                shown = "A shown first" if j.order == "AB" else "B shown first"
                out += ["Judge, {}: **{}**".format(shown, j.winner if j.slot is not None else "unparsed"), ""]
                out += [_fence(j.raw), ""]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Reports (JSON round trip + Markdown)
# --------------------------------------------------------------------------- #
REPORT_FORMAT = "promptlab.compare/1"


def report_dict(results, label_a="A", label_b="B", *, alpha=0.05, config=None):
    """Everything needed to audit a comparison, as JSON-serialisable data."""
    counts = tally(results, label_a, label_b, alpha=alpha)
    summary = {k: v for k, v in counts.items() if k not in (label_a, label_b)}
    summary["wins_a"], summary["wins_b"] = counts[label_a], counts[label_b]
    summary["ci_a"], summary["ci_b"] = list(counts["ci_a"]), list(counts["ci_b"])
    return {
        "format": REPORT_FORMAT,
        "label_a": label_a,
        "label_b": label_b,
        "config": dict(config or {}),
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }


def write_report(path, results, label_a="A", label_b="B", *, alpha=0.05, config=None):
    """Write a ``.json`` or ``.md`` report (chosen by extension). Returns the path."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = json.dumps(report_dict(results, label_a, label_b, alpha=alpha, config=config), indent=2,
                          ensure_ascii=False)
    elif suffix in (".md", ".markdown"):
        text = render_markdown(results, label_a, label_b, alpha=alpha, config=config)
    else:
        raise ValueError("report must end in .json or .md, got {!r}".format(path.name))
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    return path


def results_from_report(data):
    """Rebuild ``ComparisonResult`` objects from :func:`report_dict` output."""
    if data.get("format") != REPORT_FORMAT:
        raise ValueError("not a promptlab compare report (format {!r})".format(data.get("format")))
    results = []
    for r in data["results"]:
        pairs = [
            PairResult(
                response_a=p["response_a"],
                response_b=p["response_b"],
                winner=p["winner"],
                judgements=[Judgement(**j) for j in p["judgements"]],
                consistent=p["consistent"],
                position_bias=p["position_bias"],
                reason=p["reason"],
            )
            for p in r["pairs"]
        ]
        results.append(
            ComparisonResult(
                input=r["input"],
                response_a=r["response_a"],
                response_b=r["response_b"],
                winner=r["winner"],
                reason=r["reason"],
                meta=r.get("meta", {}),
                pairs=pairs,
            )
        )
    return results


def load_report(path):
    """Load a JSON report: returns ``(data, results)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, results_from_report(data)
