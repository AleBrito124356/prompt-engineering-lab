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

from ._common import get_client, run_demo

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


def main():
    def demo():
        chunks = [
            {"text": "The Panama Canal opened to traffic on August 15, 1914.", "source": "history.md"},
            {"text": "The canal is about 82 kilometers long from deep water to deep water.", "source": "facts.md"},
            {"text": "Neopanamax locks entered service in 2016 to handle larger ships.", "source": "expansion.md"},
        ]
        q = "How long is the Panama Canal and when did the larger locks open?"
        print("PROMPT (user turn):")
        print(build_messages(q, chunks)[-1]["content"])
        print("\nGROUNDED ANSWER:")
        print(run(q, chunks))

    run_demo("RAG prompt (grounded + cited)", demo)


if __name__ == "__main__":
    main()
