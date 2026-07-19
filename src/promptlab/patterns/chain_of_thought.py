"""Chain-of-thought (CoT) prompting.

Ask the model to reason step by step before committing to an answer, then read
the answer off a fixed marker line. Spending output tokens on intermediate
steps improves accuracy on arithmetic, logic, and multi-hop questions.

Why it helps: it lets the model allocate computation to sub-steps instead of
guessing an answer in one shot. The ``Final answer:`` contract makes the result
machine-readable so you are not scraping prose.
"""

from __future__ import annotations

import re

from ._common import get_client, run_demo

SYSTEM = (
    "You solve problems by reasoning step by step. Think carefully, then state "
    "the result on a final line formatted exactly as 'Final answer: <answer>'."
)

_ANSWER_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def build_messages(question):
    user = "{}\n\nWork through it step by step, then give the final answer.".format(question)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_final_answer(text):
    """Return the text after the last 'Final answer:' marker, or None."""
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def run(question, *, client=None, temperature=0.2):
    client = client or get_client()
    raw = client.chat(build_messages(question), temperature=temperature, max_tokens=700)
    return raw, parse_final_answer(raw)


def main():
    def demo():
        q = (
            "A shop sells pens at 3 for $2. If I buy 18 pens and pay with a "
            "$20 bill, how much change do I get?"
        )
        raw, answer = run(q)
        print("REASONING + ANSWER:\n{}".format(raw))
        print("\nPARSED FINAL ANSWER: {}".format(answer))

    run_demo("Chain-of-thought prompting", demo)


if __name__ == "__main__":
    main()
