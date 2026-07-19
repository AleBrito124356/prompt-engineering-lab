"""Reflexion: draft, self-critique, revise.

Generate a first answer, ask the model to critique it against explicit
criteria, then produce a revised answer that addresses the critique. A separate
critique turn catches problems the model glosses over when it is busy producing.

Why it helps: separating generation from evaluation is more effective than
asking for a perfect answer in one shot. The critique step is where quality
comes from -- make its criteria concrete.
"""

from __future__ import annotations

from ._common import get_client, run_demo

DRAFT_SYSTEM = "You are a capable assistant. Answer the request as well as you can."

CRITIQUE_SYSTEM = (
    "You are a rigorous reviewer. Critique the draft against the criteria. List "
    "concrete, specific problems and what would fix each one. Do not rewrite the "
    "answer; only critique. If the draft is already strong, say so briefly."
)

REVISE_SYSTEM = (
    "You revise a draft using a critique. Apply every valid point from the "
    "critique and produce the improved answer only -- no commentary."
)

DEFAULT_CRITERIA = [
    "Correctness and factual accuracy",
    "Completeness -- nothing important missing",
    "Clarity and concreteness",
    "Follows every explicit instruction in the request",
]


def build_draft(request):
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": request},
    ]


def build_critique(request, draft, criteria=None):
    criteria = criteria or DEFAULT_CRITERIA
    bullets = "\n".join("- {}".format(c) for c in criteria)
    user = (
        "Request:\n{req}\n\nDraft answer:\n{draft}\n\nCriteria:\n{crit}\n\n"
        "Give your critique."
    ).format(req=request, draft=draft, crit=bullets)
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_revision(request, draft, critique):
    user = (
        "Request:\n{req}\n\nOriginal draft:\n{draft}\n\nCritique:\n{crit}\n\n"
        "Write the improved answer."
    ).format(req=request, draft=draft, crit=critique)
    return [
        {"role": "system", "content": REVISE_SYSTEM},
        {"role": "user", "content": user},
    ]


def run(request, *, client=None, criteria=None, temperature=0.4):
    client = client or get_client()
    draft = client.chat(build_draft(request), temperature=temperature, max_tokens=700)
    critique = client.chat(build_critique(request, draft, criteria), temperature=0.2, max_tokens=500)
    revised = client.chat(build_revision(request, draft, critique), temperature=temperature, max_tokens=700)
    return {"draft": draft, "critique": critique, "revised": revised}


def main():
    def demo():
        request = "Write a two-sentence product description for a stainless-steel water bottle aimed at hikers."
        out = run(request)
        print("DRAFT:\n{}\n".format(out["draft"]))
        print("CRITIQUE:\n{}\n".format(out["critique"]))
        print("REVISED:\n{}".format(out["revised"]))

    run_demo("Reflexion (draft, critique, revise)", demo)


if __name__ == "__main__":
    main()
