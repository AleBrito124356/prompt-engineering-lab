"""Step-back prompting.

Before answering a specific question, ask the model to "step back" and state the
general principle, definition, or law that governs it. Then answer the original
question using that principle as grounding.

Why it helps: retrieving the right abstraction first reduces careless mistakes on
questions that hinge on a rule the model knows but forgets to apply under the
pressure of a concrete case.
"""

from __future__ import annotations

from ._common import get_client, run_demo

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


def main():
    def demo():
        q = (
            "A gas is compressed to half its volume at constant temperature. What "
            "happens to its pressure?"
        )
        out = run(q)
        print("STEP-BACK PRINCIPLE:\n{}\n".format(out["principle"]))
        print("ANSWER:\n{}".format(out["answer"]))

    run_demo("Step-back prompting", demo)


if __name__ == "__main__":
    main()
