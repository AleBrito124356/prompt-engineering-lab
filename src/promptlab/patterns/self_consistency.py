"""Self-consistency.

Sample several chain-of-thought completions at non-zero temperature, extract
each final answer, and take the majority vote. Different reasoning paths that
converge on the same answer are more trustworthy than any single path.

Why it helps: it turns a noisy single sample into an ensemble. It only applies
when the final answer is comparable (a number, a label, a short string) so that
votes can be counted. Cost scales linearly with the number of samples.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from .chain_of_thought import build_messages, clean_answer, parse_final_answer
from ._common import get_client, run_demo

# A leading number: optional sign and currency, thousands separators, decimals,
# an optional percent. It must be followed by the end, whitespace or a closing
# bracket, so "12:30", "1/2" and "3x+1" are *not* read as numbers.
_NUMBER_RE = re.compile(
    r"^(?P<sign>[-+−])?\s*(?:[$€£¥]\s*)?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?"
    r"(?P<pct>\s*(?:%|percent\b))?(?=$|\s|[)\]])",
    re.IGNORECASE,
)


def _normalize(answer):
    """Vote key for an answer: numbers compare as numbers, text case-folded.

    ``"23"``, ``"23 apples"``, ``"**23**"``, ``"$23.00"`` and ``"23."`` all map
    to ``"23"``; ``"1,000"`` to ``"1000"``; ``"50 percent"`` to ``"50%"``.
    """
    if answer is None:
        return None
    a = clean_answer(str(answer))
    if not a:
        return None
    m = _NUMBER_RE.match(a)
    if m:
        try:
            value = Decimal(m.group("int").replace(",", "") + (m.group("frac") or ""))
        except InvalidOperation:  # pragma: no cover - the regex only admits digits
            value = None
        if value is not None:
            if m.group("sign") in ("-", "−"):
                value = -value
            text = str(int(value)) if value == value.to_integral_value() else format(value.normalize(), "f")
            return text + ("%" if m.group("pct") else "")
    return " ".join(a.lower().split())


def majority_vote(answers):
    """Return (winner, vote_counts) over normalised answers, ignoring Nones."""
    votes = Counter()
    display = {}
    for ans in answers:
        key = _normalize(ans)
        if key is None:
            continue
        votes[key] += 1
        display.setdefault(key, clean_answer(str(ans)))
    if not votes:
        return None, votes
    winner_key, _ = votes.most_common(1)[0]
    return display[winner_key], votes


def run(question, *, n=5, client=None, temperature=0.7):
    client = client or get_client()
    samples = client.sample(build_messages(question), n, temperature=temperature, max_tokens=700)
    answers = [parse_final_answer(s) for s in samples]
    winner, votes = majority_vote(answers)
    return winner, answers, votes


def main():
    def demo():
        q = (
            "I have 5 boxes. Three boxes hold 8 apples each and two hold 5 "
            "apples each. I give away 11 apples. How many apples remain?"
        )
        winner, answers, votes = run(q, n=5)
        print("Sampled answers: {}".format(answers))
        print("Vote counts: {}".format(dict(votes)))
        print("Majority answer: {}".format(winner))

    run_demo("Self-consistency (sample N, majority vote)", demo)


if __name__ == "__main__":
    main()
