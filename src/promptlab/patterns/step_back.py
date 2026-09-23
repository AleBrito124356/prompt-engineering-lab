"""Step-back prompting.

Before answering a specific question, ask the model to "step back" and state the
general principle, definition, or law that governs it. Then answer the original
question using that principle as grounding.

Why it helps: retrieving the right abstraction first reduces careless mistakes on
questions that hinge on a rule the model knows but forgets to apply under the
pressure of a concrete case.
"""

from __future__ import annotations

from ..backends import ScriptedClient
from ._common import demo_main, get_client

STEPBACK_SYSTEM = (
    "Given a specific question, state the single most relevant general principle, "
    "rule, or concept needed to answer it. Output only that principle in one or "
    "two sentences. Do not answer the specific question yet."
)

ANSWER_SYSTEM = (
    "Answer the question. First you are given a relevant principle; use it "
    "explicitly to reason to the answer."
)


def build_stepback(question):
    return [
        {"role": "system", "content": STEPBACK_SYSTEM},
        {"role": "user", "content": question},
    ]


def build_answer(question, principle):
    user = "Principle:\n{p}\n\nQuestion:\n{q}\n\nAnswer using the principle.".format(
        p=principle, q=question
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def run(question, *, client=None, temperature=0.2):
    client = client or get_client()
    principle = client.chat(build_stepback(question), temperature=0.2, max_tokens=200).strip()
    answer = client.chat(build_answer(question, principle), temperature=temperature, max_tokens=500)
    return {"principle": principle, "answer": answer}


DEMO_QUESTION = (
    "A gas is compressed to half its volume at constant temperature. What "
    "happens to its pressure?"
)


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model)."""
    return ScriptedClient(
        rules=[
            (
                lambda messages: messages[0]["content"] == STEPBACK_SYSTEM,
                "Boyle's law: at constant temperature, the pressure of a fixed amount of gas is "
                "inversely proportional to its volume, so P1 x V1 = P2 x V2.",
            ),
            (
                lambda messages: messages[0]["content"] == ANSWER_SYSTEM,
                "By Boyle's law, P1 x V1 = P2 x V2. Halving the volume means V2 = V1 / 2, so "
                "P2 = P1 x V1 / (V1 / 2) = 2 x P1. The pressure doubles.",
            ),
        ]
    )


def main(argv=None):
    def demo():
        out = run(DEMO_QUESTION)
        print("STEP-BACK PRINCIPLE:\n{}\n".format(out["principle"]))
        print("ANSWER:\n{}".format(out["answer"]))

    return demo_main("Step-back prompting", demo, module="step_back", demo_client=demo_client, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
