"""Tree of Thought (bounded breadth and depth).

Instead of one linear chain, expand several candidate "thoughts" at each step,
score them, keep the best few (a beam), and expand again -- a small, budgeted
search over reasoning paths. Bounds on breadth and depth keep the cost finite.

Why it helps: for problems where good and bad partial ideas look similar early
on, exploring and pruning beats committing to the first idea. Here a lightweight
value model (the same LLM asked to score 0-10) guides the search.
"""

from __future__ import annotations

import re

from ._common import get_client, run_demo

PROPOSE_SYSTEM = (
    "You extend a partial plan toward solving a problem. Given the problem and "
    "the steps so far, propose ONE single next step. Output only that step as a "
    "short sentence."
)

VALUE_SYSTEM = (
    "You score how promising a partial plan is for solving the problem, from 0 "
    "(hopeless) to 10 (clearly on track). Output only 'Score: <number>'."
)

ANSWER_SYSTEM = (
    "You are given a problem and a promising chain of steps. Produce the final "
    "answer, using the steps as your reasoning."
)

_SCORE_RE = re.compile(r"score\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text, *, default=0.0):
    """Extract a 0-10 numeric score from a value-model response."""
    m = _SCORE_RE.search(text)
    if not m:
        return default
    try:
        return max(0.0, min(10.0, float(m.group(1))))
    except ValueError:
        return default


def select_best(scored, beam_width):
    """Keep the ``beam_width`` highest-scored (path, score) pairs, stably."""
    ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return ordered[:beam_width]


def _propose(client, problem, path, temperature):
    steps = "\n".join("{}. {}".format(i + 1, s) for i, s in enumerate(path)) or "(none yet)"
    user = "Problem:\n{}\n\nSteps so far:\n{}\n\nPropose the next step.".format(problem, steps)
    return client.chat(
        [{"role": "system", "content": PROPOSE_SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=120,
    ).strip()


def _value(client, problem, path):
    steps = "\n".join("{}. {}".format(i + 1, s) for i, s in enumerate(path))
    user = "Problem:\n{}\n\nPartial plan:\n{}\n\nScore it.".format(problem, steps)
    raw = client.chat(
        [{"role": "system", "content": VALUE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=20,
    )
    return parse_score(raw)


def run(problem, *, client=None, breadth=3, depth=2, beam_width=2, temperature=0.8):
    """Search reasoning paths with the given breadth/depth budget.

    Returns ``(final_answer, best_path, score)``.
    """
    client = client or get_client()
    beam = [([], 0.0)]  # list of (path, score)
    for _ in range(depth):
        candidates = []
        for path, _score in beam:
            for _ in range(breadth):
                step = _propose(client, problem, path, temperature)
                new_path = path + [step]
                candidates.append((new_path, _value(client, problem, new_path)))
        beam = select_best(candidates, beam_width)
    best_path, best_score = beam[0]
    steps = "\n".join("{}. {}".format(i + 1, s) for i, s in enumerate(best_path))
    answer = client.chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": "Problem:\n{}\n\nSteps:\n{}".format(problem, steps)},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return answer, best_path, best_score


def main():
    def demo():
        problem = (
            "Arrange a 3-day trip to a coastal town for a family with a toddler, "
            "balancing rest, one activity per day, and short travel times."
        )
        answer, path, score = run(problem, breadth=3, depth=2, beam_width=2)
        print("BEST PATH (score {:.1f}):".format(score))
        for i, step in enumerate(path, 1):
            print("  {}. {}".format(i, step))
        print("\nFINAL ANSWER:\n{}".format(answer))

    run_demo("Tree of Thought (bounded search)", demo)


if __name__ == "__main__":
    main()
