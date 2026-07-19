"""Least-to-most prompting.

First ask the model to break a hard problem into an ordered list of simpler
sub-problems. Then solve them one at a time, feeding each answer forward, so the
final sub-problem is solved with all the groundwork already in context.

Why it helps: it generalises better than plain chain-of-thought on problems that
require composing several dependent steps, because each step is solved in
isolation with the previous results available.
"""

from __future__ import annotations

import re

from ._common import get_client, run_demo

DECOMPOSE_SYSTEM = (
    "You break a problem into an ordered list of simpler sub-problems that build "
    "on each other. Output a numbered list only, from easiest/first to the final "
    "question. Do not solve them."
)

SOLVE_SYSTEM = (
    "You solve one sub-problem at a time. Use the answers to earlier sub-problems "
    "when they help. Answer only the current sub-problem, concisely."
)

_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$")


def parse_subproblems(text):
    """Extract sub-problems from a numbered list. Returns a list of strings."""
    items = []
    for line in text.splitlines():
        m = _NUMBERED_RE.match(line)
        if m:
            items.append(m.group(1).strip())
    return items


def build_solve(problem, subproblem, solved):
    """Build the solve prompt for one sub-problem, given prior (q, a) pairs."""
    context = "\n".join(
        "Sub-problem: {}\nAnswer: {}".format(q, a) for q, a in solved
    )
    context = context or "(no earlier sub-problems)"
    user = (
        "Overall problem:\n{prob}\n\nEarlier results:\n{ctx}\n\n"
        "Current sub-problem:\n{sp}\n\nAnswer it."
    ).format(prob=problem, ctx=context, sp=subproblem)
    return [
        {"role": "system", "content": SOLVE_SYSTEM},
        {"role": "user", "content": user},
    ]


def run(problem, *, client=None, temperature=0.2, max_subproblems=6):
    client = client or get_client()
    decomposition = client.chat(
        [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    subproblems = parse_subproblems(decomposition)[:max_subproblems]
    solved = []
    for sp in subproblems:
        answer = client.chat(
            build_solve(problem, sp, solved), temperature=temperature, max_tokens=400
        )
        solved.append((sp, answer.strip()))
    final = solved[-1][1] if solved else None
    return {"subproblems": subproblems, "solved": solved, "final": final}


def main():
    def demo():
        problem = (
            "A tank fills at 4 liters/min and drains at 1.5 liters/min. It starts "
            "empty and holds 200 liters. How long until it overflows?"
        )
        out = run(problem)
        print("SUB-PROBLEMS:")
        for i, sp in enumerate(out["subproblems"], 1):
            print("  {}. {}".format(i, sp))
        print("\nSOLUTIONS:")
        for q, a in out["solved"]:
            print("  - {} -> {}".format(q, a))
        print("\nFINAL: {}".format(out["final"]))

    run_demo("Least-to-most prompting", demo)


if __name__ == "__main__":
    main()
