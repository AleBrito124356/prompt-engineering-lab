"""One runnable module per prompt-engineering technique.

Each module documents the technique, exposes pure prompt-building and parsing
functions (unit-tested, no network) and a ``main()`` that runs a live demo on
NVIDIA NIM. The registry below powers ``cli.py techniques`` and keeps the
README cheat-sheet honest.
"""

from __future__ import annotations

__all__ = ["TECHNIQUES", "technique_names"]

# name -> (module, when to use, relative cost in model calls)
TECHNIQUES = [
    {
        "name": "zero_shot",
        "when": "Simple, well-specified tasks a strong model already knows how to do.",
        "cost": "1 call",
    },
    {
        "name": "few_shot",
        "when": "You need a specific format or edge-case behaviour that examples pin down.",
        "cost": "1 call, longer prompt",
    },
    {
        "name": "chain_of_thought",
        "when": "Multi-step reasoning, arithmetic, or logic where showing work helps.",
        "cost": "1 call, more output tokens",
    },
    {
        "name": "self_consistency",
        "when": "CoT tasks with a checkable final answer; trade cost for accuracy.",
        "cost": "N calls (majority vote)",
    },
    {
        "name": "react_mini",
        "when": "The model needs tools (calculator, lookup) to act, not just answer.",
        "cost": "1 call per step, bounded",
    },
    {
        "name": "reflexion",
        "when": "Quality-sensitive drafts that benefit from a self-critique pass.",
        "cost": "3 calls (draft, critique, revise)",
    },
    {
        "name": "tree_of_thought",
        "when": "Search-like problems where several partial ideas must be compared.",
        "cost": "breadth x depth calls",
    },
    {
        "name": "least_to_most",
        "when": "Hard problems that decompose into easier, ordered sub-problems.",
        "cost": "1 + one call per sub-problem",
    },
    {
        "name": "step_back",
        "when": "Questions answered better after stating the general principle first.",
        "cost": "2 calls (principle, then answer)",
    },
    {
        "name": "structured_json",
        "when": "You need machine-readable output that downstream code can parse.",
        "cost": "1 call + local validation",
    },
    {
        "name": "rag_prompt",
        "when": "Answers must be grounded in retrieved context and cite their source.",
        "cost": "1 call (retrieval done upstream)",
    },
    {
        "name": "guardrail_prompt",
        "when": "Untrusted user input; you need scope, safety, and refusal behaviour.",
        "cost": "1 call + local screening",
    },
]


def technique_names():
    return [t["name"] for t in TECHNIQUES]
