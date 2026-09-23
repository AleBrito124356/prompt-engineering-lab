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

from ..backends import ScriptedClient
from ._common import demo_main, get_client

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


def run(problem, *, client=None, breadth=3, depth=2, beam_width=2, temperature=0.8, verbose=False):
    """Search reasoning paths with the given breadth/depth budget.

    Makes ``depth * breadth * len(beam)`` proposal calls, as many value calls,
    and one answer call. With ``verbose=True`` every scored candidate and the
    kept beam are printed. Returns ``(final_answer, best_path, score)``.
    """
    client = client or get_client()
    beam = [([], 0.0)]  # list of (path, score)
    for level in range(1, depth + 1):
        candidates = []
        for path, _score in beam:
            for _ in range(breadth):
                step = _propose(client, problem, path, temperature)
                new_path = path + [step]
                candidates.append((new_path, _value(client, problem, new_path)))
        beam = select_best(candidates, beam_width)
        if verbose:
            print("depth {}: scored {} candidate(s)".format(level, len(candidates)))
            for path, score in candidates:
                kept = "kept" if any(path is p for p, _ in beam) else "    "
                print("  [{}] {:>4.1f}  {}".format(kept, score, path[-1]))
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


DEMO_PROBLEM = (
    "Arrange a 3-day trip to a coastal town for a family with a toddler, "
    "balancing rest, one activity per day, and short travel times."
)

# Illustrative proposals (keyed by the parent step) and the scores a value
# model might give them.
_ROOT = ""
_DEMO_STEPS = {
    _ROOT: [
        ("Pick a town within a 2-hour drive so no travel day eats the toddler's nap.", 8),
        ("Book a beach rental with a separate bedroom so naps happen on schedule.", 7),
        ("Plan a day trip to a different nearby city on each of the three days.", 3),
    ],
    "Pick a town within a 2-hour drive so no travel day eats the toddler's nap.": [
        ("Day 1: arrive by noon, nap, beach at 4 pm; Day 2: aquarium at 9 am; "
         "Day 3: tide pools, lunch, drive home during the nap.", 9),
        ("Keep mornings for the single activity and afternoons for rest near the rental.", 7),
        ("Add fireworks and a late dinner downtown every evening.", 2),
    ],
    "Book a beach rental with a separate bedroom so naps happen on schedule.": [
        ("Day 1: settle in and nap; Day 2: beach in the morning; Day 3: playground, then home.", 8),
        ("Rent bikes with a child seat for a coastal ride each morning.", 5),
        ("Stay at the rental pool all three days to avoid driving at all.", 4),
    ],
}
_DEMO_SCORES = {text: score for steps in _DEMO_STEPS.values() for text, score in steps}
_LAST_STEP_RE = re.compile(r"^\d+\.\s*(.+)$", re.MULTILINE)


def _system_is(prompt):
    return lambda messages: messages[0]["content"] == prompt


def _last_step(text, header):
    steps = _LAST_STEP_RE.findall(text.split(header, 1)[-1])
    return steps[-1].strip() if steps else _ROOT


def _scripted_proposer():
    served = {}

    def propose(messages, params):
        parent = _last_step(messages[-1]["content"].split("Propose the next step.")[0], "Steps so far:")
        options = _DEMO_STEPS.get(parent, _DEMO_STEPS[_ROOT])
        i = served.get(parent, 0)
        served[parent] = i + 1
        return options[i % len(options)][0]

    return propose


def _scripted_value(messages, params):
    last = _last_step(messages[-1]["content"].split("Score it.")[0], "Partial plan:")
    return "Score: {}".format(_DEMO_SCORES.get(last, 5))


def demo_client():
    """Scripted proposer, value model and answer for ``--offline``.

    The search itself -- expanding the beam, scoring, pruning with
    ``select_best`` -- runs for real over these canned thoughts.
    """
    return ScriptedClient(
        rules=[
            (_system_is(PROPOSE_SYSTEM), _scripted_proposer()),
            (_system_is(VALUE_SYSTEM), _scripted_value),
            (
                _system_is(ANSWER_SYSTEM),
                "Stay in a coastal town under 2 hours away. Day 1: arrive by noon, nap at the rental, "
                "beach at 4 pm. Day 2: aquarium right after breakfast, afternoon rest. Day 3: tide pools "
                "in the morning, lunch, then drive home during the afternoon nap.",
            ),
        ]
    )


def main(argv=None):
    def demo():
        answer, path, score = run(DEMO_PROBLEM, breadth=3, depth=2, beam_width=2, verbose=True)
        print("\nBEST PATH (score {:.1f}):".format(score))
        for i, step in enumerate(path, 1):
            print("  {}. {}".format(i, step))
        print("\nFINAL ANSWER:\n{}".format(answer))

    return demo_main(
        "Tree of Thought (bounded search)", demo, module="tree_of_thought", demo_client=demo_client, argv=argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
