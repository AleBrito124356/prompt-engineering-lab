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

from ..backends import ScriptedClient
from ._common import demo_main, get_client

SYSTEM = (
    "You solve problems by reasoning step by step. Think carefully, then state "
    "the result on a final line formatted exactly as 'Final answer: <answer>'."
)

# "Final answer: 42", "**Final answer:** 42", "Final Answer - 42",
# "The final answer is 42", "**Final answer:**\n42" (answer on the next line).
_ANSWER_RE = re.compile(
    r"final\s+answer\s*[*_]*\s*(?:is\b\s*[:\-]?|[:\-])[ \t]*[*_]*\s*(.+)",
    re.IGNORECASE,
)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_LEADIN_RE = re.compile(r"^(?:so\s+|thus\s+|therefore\s+)?(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*", re.IGNORECASE)


def clean_answer(answer):
    """Strip the formatting models wrap around an answer.

    Removes ``**bold**``/``__bold__``/``*italics*``, backticks, ``\\boxed{...}``,
    ``$...$`` or ``\\(...\\)`` math delimiters, a leading "the answer is", and
    trailing punctuation. ``"$8"`` (money) is kept as is. Returns ``None`` for
    ``None`` and ``""`` when nothing is left.
    """
    if answer is None:
        return None
    a = answer.strip()
    boxed = _BOXED_RE.search(a)
    if boxed:
        a = boxed.group(1)
    a = a.replace("**", "").replace("__", "").replace("`", "").strip()
    a = re.sub(r"^[*_]+|[*_]+$", "", a).strip()
    if a.startswith("\\(") and a.endswith("\\)"):
        a = a[2:-2].strip()
    if len(a) >= 2 and a.startswith("$") and a.endswith("$") and a.count("$") == 2:
        a = a[1:-1].strip()
    a = _LEADIN_RE.sub("", a)
    return a.rstrip(".!;,: \t").strip()


def build_messages(question):
    user = "{}\n\nWork through it step by step, then give the final answer.".format(question)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_final_answer(text):
    """Return the cleaned answer after the last 'Final answer:' marker, or None.

    Tolerates markdown around the marker or the answer (``**Final answer:**
    42``, ``Final answer: **42**``, ``\\boxed{42}``) -- see :func:`clean_answer`.
    """
    for match in reversed(list(_ANSWER_RE.finditer(text or ""))):
        cleaned = clean_answer(match.group(1))
        if cleaned:
            return cleaned
    return None


def run(question, *, client=None, temperature=0.2):
    client = client or get_client()
    raw = client.chat(build_messages(question), temperature=temperature, max_tokens=700)
    return raw, parse_final_answer(raw)


DEMO_QUESTION = (
    "A shop sells pens at 3 for $2. If I buy 18 pens and pay with a "
    "$20 bill, how much change do I get?"
)


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model).

    The answer line is wrapped in markdown on purpose: ``parse_final_answer``
    has to strip it, as it must with real model output.
    """
    return ScriptedClient(
        [
            "Pens are sold in groups of 3 for $2.\n"
            "1. 18 pens / 3 pens per group = 6 groups.\n"
            "2. 6 groups x $2 = $12.\n"
            "3. Paying with a $20 bill: $20 - $12 = $8.\n\n"
            "**Final answer:** $8"
        ]
    )


def main(argv=None):
    def demo():
        raw, answer = run(DEMO_QUESTION)
        print("REASONING + ANSWER:\n{}".format(raw))
        print("\nPARSED FINAL ANSWER: {}".format(answer))

    return demo_main(
        "Chain-of-thought prompting", demo, module="chain_of_thought", demo_client=demo_client, argv=argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
