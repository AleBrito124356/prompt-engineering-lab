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

from ..backends import ScriptedClient
from ._common import demo_main, get_client

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


DEMO_TASK = "Explain what an idempotent HTTP method is to a junior developer."
DEMO_CONSTRAINTS = ["Under 80 words", "Give one concrete example"]


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model)."""
    return ScriptedClient(
        [
            "An idempotent HTTP method is one you can repeat without changing the result "
            "after the first call. Example: `PUT /users/42` with the same body leaves user 42 "
            "in the same state whether you send it once or five times. `POST /orders` is not "
            "idempotent: every retry can create another order. That is why clients can safely "
            "retry PUT and DELETE after a timeout."
        ]
    )


def main(argv=None):
    def demo():
        print("PROMPT (user turn):")
        print(build_messages(DEMO_TASK, constraints=DEMO_CONSTRAINTS)[-1]["content"])
        print("\nRESPONSE:")
        print(run(DEMO_TASK, constraints=DEMO_CONSTRAINTS))

    return demo_main("Zero-shot prompting", demo, module="zero_shot", demo_client=demo_client, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
