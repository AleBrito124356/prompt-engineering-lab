"""RAG prompting (grounded answering over retrieved context).

Retrieval is out of scope here (see the sibling ``rag-blueprints`` repo); this
module is about the *prompt* that turns retrieved chunks into a grounded,
cited answer. The context is numbered, the model is told to answer only from it,
cite chunk numbers, and say it does not know when the answer is not present.

Why it helps: the failure mode of RAG is a confident answer that the sources do
not support. An explicit grounding instruction plus a citation format makes
unsupported answers visible and hallucinations less likely.
"""

from __future__ import annotations

import re

from ..backends import ScriptedClient
from ._common import demo_main, get_client

SYSTEM = (
    "You answer questions using only the provided context. Cite the sources you "
    "use with bracketed numbers like [1], [2]. If the context does not contain "
    "the answer, reply exactly: 'I don't know based on the provided context.' "
    "Do not use outside knowledge."
)


def format_context(chunks):
    """Number chunks for citation. Accepts strings or {'text','source'} dicts."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("source")
            label = " ({})".format(source) if source else ""
        else:
            text, label = str(chunk), ""
        lines.append("[{}]{} {}".format(i, label, text))
    return "\n".join(lines)


def build_messages(question, chunks):
    context = format_context(chunks)
    user = "Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer with citations.".format(
        ctx=context, q=question
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def run(question, chunks, *, client=None, temperature=0.0):
    client = client or get_client()
    return client.chat(build_messages(question, chunks), temperature=temperature, max_tokens=500)


ABSTAIN = "I don't know based on the provided context."
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]")
# Split after sentence punctuation followed by whitespace (so "82.5" stays whole),
# but not when the next token is a citation ("...long. [2]" cites that sentence).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[\d)")


def _expand(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            numbers.extend(range(lo, hi + 1) if hi >= lo and hi - lo <= 50 else [lo, hi])
        else:
            numbers.append(int(part))
    return numbers


def check_citations(answer, n_chunks):
    """Check a grounded answer's citations against the ``n_chunks`` provided.

    Returns a dict:

    - ``abstained``: the answer is the instructed "I don't know" reply;
    - ``cited``: sorted chunk numbers cited (``[2]``, ``[1, 3]``, ``[1-2]``);
    - ``invalid``: cited numbers that do not exist (``[7]`` with 3 chunks) --
      a sure sign of a fabricated source;
    - ``uncited``: sentences that make a claim without any citation;
    - ``ok``: abstained, or at least one valid citation, no invalid ones and no
      uncited sentences.
    """
    text = (answer or "").strip()
    abstained = ABSTAIN.lower().rstrip(".") in text.lower()
    numbers = [n for m in _CITATION_RE.finditer(text) for n in _expand(m.group(1))]
    cited = sorted({n for n in numbers if 1 <= n <= n_chunks})
    invalid = sorted({n for n in numbers if not 1 <= n <= n_chunks})
    uncited = []
    if not abstained:
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            s = sentence.strip()
            if len(s.split()) >= 4 and not _CITATION_RE.search(s):
                uncited.append(s)
    ok = abstained or (bool(cited) and not invalid and not uncited)
    return {"abstained": abstained, "cited": cited, "invalid": invalid, "uncited": uncited, "ok": ok}


DEMO_CHUNKS = [
    {"text": "The Panama Canal opened to traffic on August 15, 1914.", "source": "history.md"},
    {"text": "The canal is about 82 kilometers long from deep water to deep water.", "source": "facts.md"},
    {"text": "Neopanamax locks entered service in 2016 to handle larger ships.", "source": "expansion.md"},
]
DEMO_QUESTION = "How long is the Panama Canal and when did the larger locks open?"


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model)."""
    return ScriptedClient(
        [
            "The Panama Canal is about 82 kilometers long [2]. The larger Neopanamax locks "
            "entered service in 2016 [3]."
        ]
    )


def main(argv=None):
    def demo():
        print("PROMPT (user turn):")
        print(build_messages(DEMO_QUESTION, DEMO_CHUNKS)[-1]["content"])
        print("\nGROUNDED ANSWER:")
        answer = run(DEMO_QUESTION, DEMO_CHUNKS)
        print(answer)
        report = check_citations(answer, len(DEMO_CHUNKS))
        print(
            "\nCITATION CHECK: ok={ok} cited={cited} invalid={invalid} uncited={n}".format(
                n=len(report["uncited"]), **report
            )
        )

    return demo_main("RAG prompt (grounded + cited)", demo, module="rag_prompt", demo_client=demo_client, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
