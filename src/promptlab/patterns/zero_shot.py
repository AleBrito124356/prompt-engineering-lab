"""Zero-shot prompting.

Ask directly, with a clear role and instruction, and no examples. This is the
right default for tasks a strong instruction-tuned model already knows how to
do -- a translation, a definition, a short rewrite. Reach for something heavier
only when zero-shot measurably fails.

Why it helps: the shortest prompt that works is the cheapest to run and the
easiest to maintain. Specificity in the instruction (role, audience, format,
length) does most of the work.
"""

from __future__ import annotations

from ._common import get_client, run_demo

SYSTEM = "You are a precise, concise assistant. Follow the instruction exactly."


def build_messages(task, *, role=None, constraints=None):
    """Build chat messages for a zero-shot task.

    ``role`` overrides the default system persona; ``constraints`` is an optional
    list of hard requirements appended to the user turn.
    """
    system = role or SYSTEM
    user = task
    if constraints:
        bullets = "\n".join("- {}".format(c) for c in constraints)
        user = "{}\n\nConstraints:\n{}".format(task, bullets)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run(task, *, client=None, role=None, constraints=None, temperature=0.3):
    client = client or get_client()
    messages = build_messages(task, role=role, constraints=constraints)
    return client.chat(messages, temperature=temperature)


def main():
    def demo():
        task = "Explain what an idempotent HTTP method is to a junior developer."
        print("PROMPT (user turn):")
        print(build_messages(task)[-1]["content"])
        print("\nRESPONSE:")
        print(run(task, constraints=["Under 80 words", "Give one concrete example"]))

    run_demo("Zero-shot prompting", demo)


if __name__ == "__main__":
    main()
