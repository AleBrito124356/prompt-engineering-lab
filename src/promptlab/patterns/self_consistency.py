"""Self-consistency.

Sample several chain-of-thought completions at non-zero temperature, extract
each final answer, and take the majority vote. Different reasoning paths that
converge on the same answer are more trustworthy than any single path.

Why it helps: it turns a noisy single sample into an ensemble. It only applies
when the final answer is comparable (a number, a label, a short string) so that
votes can be counted. Cost scales linearly with the number of samples.
"""

from __future__ import annotations

from collections import Counter

from .chain_of_thought import build_messages, parse_final_answer
from ._common import get_client, run_demo


def _normalize(answer):
    if answer is None:
        return None
    a = answer.strip().rstrip(".").strip()
    # Normalise simple money/number answers so "$4" and "4 dollars" don't split votes.
    stripped = a.replace("$", "").replace(",", "").strip()
    try:
        f = float(stripped)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return a.lower()


def majority_vote(answers):
    """Return (winner, vote_counts) over normalised answers, ignoring Nones."""
    votes = Counter()
    display = {}
    for ans in answers:
        key = _normalize(ans)
        if key is None:
            continue
        votes[key] += 1
        display.setdefault(key, ans.strip() if isinstance(ans, str) else ans)
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
