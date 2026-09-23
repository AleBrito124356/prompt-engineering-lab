"""Few-shot prompting.

Show the model two to five worked examples so it copies a format or a subtle
convention that is hard to describe in prose. Examples are more reliable than
adjectives: to get a specific JSON shape or a specific tone, demonstrate it.

Why it helps: examples reduce ambiguity and anchor the output distribution.
Keep them short, diverse, and representative of the edge cases you care about.
"""

from __future__ import annotations

from ..template import few_shot
from ..backends import ScriptedClient
from ._common import demo_main, get_client

SYSTEM = (
    "You label the sentiment of a product review as positive, negative, or "
    "mixed. Match the format of the examples exactly."
)

DEFAULT_EXAMPLES = [
    {"input": "Battery lasts all day and the screen is gorgeous.", "output": "positive"},
    {"input": "It works, but the app crashes twice a week.", "output": "mixed"},
    {"input": "Stopped charging after a month. Avoid.", "output": "negative"},
]


def build_messages(task, examples=None, *, input_label="Review", output_label="Sentiment"):
    """Compose a few-shot prompt: labelled examples, then the new input."""
    examples = examples if examples is not None else DEFAULT_EXAMPLES
    block = few_shot(
        examples, input_label=input_label, output_label=output_label
    )
    user = "{}\n\n{}: {}\n{}:".format(block, input_label, task, output_label)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def run(task, examples=None, *, client=None, temperature=0.0):
    client = client or get_client()
    return client.chat(build_messages(task, examples), temperature=temperature)


DEMO_TASK = "The design is beautiful but shipping took three weeks."


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model)."""
    return ScriptedClient(["mixed"])


def main(argv=None):
    def demo():
        print("PROMPT (user turn):")
        print(build_messages(DEMO_TASK)[-1]["content"])
        print("\nRESPONSE:")
        print(run(DEMO_TASK))

    return demo_main("Few-shot prompting", demo, module="few_shot", demo_client=demo_client, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
