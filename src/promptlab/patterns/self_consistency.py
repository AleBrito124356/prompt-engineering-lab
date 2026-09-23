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

from ..backends import ScriptedClient
from .chain_of_thought import build_messages, clean_answer, parse_final_answer
from ._common import demo_main, get_client

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


DEMO_QUESTION = (
    "I have 5 boxes. Three boxes hold 8 apples each and two hold 5 "
    "apples each. I give away 11 apples. How many apples remain?"
)


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model).

    Five sampled reasoning paths: four reach 23 (formatted three different
    ways, as real samples are) and one makes an addition slip and says 24.
    Normalisation has to merge the formats for the vote to be right.
    """
    return ScriptedClient(
        [
            "3 boxes x 8 = 24 apples; 2 boxes x 5 = 10 apples; 34 in total.\n34 - 11 = 23.\n"
            "Final answer: 23",
            "Total: 24 + 10 = 34. After giving away 11, 34 - 11 = 23 remain.\n**Final answer:** **23**",
            "8 + 8 + 8 = 24 and 5 + 5 = 10, so 34 apples; minus 11 leaves 23.\nFinal answer: 23 apples",
            "Three boxes of 8 is 24, two boxes of 5 is 10, 24 + 10 = 35, and 35 - 11 = 24.\n"
            "Final answer: 24",
            "There are 34 apples, and 34 - 11 = 23.\nFinal answer: 23.",
        ]
    )


def main(argv=None):
    def demo():
        winner, answers, votes = run(DEMO_QUESTION, n=5)
        print("Sampled answers: {}".format(answers))
        print("Vote counts: {}".format(dict(votes)))
        print("Majority answer: {}".format(winner))

    return demo_main(
        "Self-consistency (sample N, majority vote)",
        demo,
        module="self_consistency",
        demo_client=demo_client,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
