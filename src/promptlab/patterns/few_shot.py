"""Few-shot prompting.

Show the model two to five worked examples so it copies a format or a subtle
convention that is hard to describe in prose. Examples are more reliable than
adjectives: to get a specific JSON shape or a specific tone, demonstrate it.

Why it helps: examples reduce ambiguity and anchor the output distribution.
Keep them short, diverse, and representative of the edge cases you care about.
"""

from __future__ import annotations

from ..template import few_shot
from ._common import get_client, run_demo

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


def main():
    def demo():
        task = "The design is beautiful but shipping took three weeks."
        print("PROMPT (user turn):")
        print(build_messages(task)[-1]["content"])
        print("\nRESPONSE:")
        print(run(task))

    run_demo("Few-shot prompting", demo)


if __name__ == "__main__":
    main()
