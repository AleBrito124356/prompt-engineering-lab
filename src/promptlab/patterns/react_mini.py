"""A minimal ReAct loop (Reason + Act).

The model interleaves reasoning and tool calls. It emits ``Thought``,
``Action`` and ``Action Input`` lines; the loop parses the action, runs a real
tool, and feeds the result back as an ``Observation``. It repeats until the
model emits ``Final Answer`` or a step budget is hit.

Why it helps: for anything that needs a fact the model does not have or a
computation it should not do in its head (arithmetic, string ops), giving it
tools beats hoping it guesses right. The safe calculator here uses an AST
evaluator -- never ``eval`` -- so untrusted expressions cannot run code.
"""

from __future__ import annotations

import ast
import operator
import re

from ._common import get_client, run_demo

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculator(expression):
    """Evaluate an arithmetic expression with a whitelisted AST, no ``eval``."""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression element: {}".format(type(node).__name__))

    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def word_count(text):
    """Return the number of whitespace-separated tokens in ``text``."""
    return str(len(text.split()))


TOOLS = {
    "calculator": safe_calculator,
    "word_count": word_count,
}

SYSTEM = """You answer questions by reasoning and using tools. On each turn output:
Thought: <your reasoning>
Action: <one of: {tools}>
Action Input: <the input to the tool>

After you receive an Observation, continue. When you can answer, output:
Thought: <final reasoning>
Final Answer: <the answer>

Use exactly these line prefixes. Only call the listed tools."""

_ACTION_RE = re.compile(r"^Action:\s*(.+?)\s*$", re.MULTILINE)
_INPUT_RE = re.compile(r"^Action Input:\s*(.+?)\s*$", re.MULTILINE)
_FINAL_RE = re.compile(r"^Final Answer:\s*(.+)$", re.MULTILINE | re.DOTALL)


def parse_step(text):
    """Parse a model turn into ('final', answer) or ('action', name, input).

    Returns ('final', text) if a Final Answer line is present, else
    ('action', tool_name, tool_input) if an Action is present, else
    ('none', raw) when neither prefix appears.
    """
    final = _FINAL_RE.search(text)
    if final:
        return ("final", final.group(1).strip())
    action = _ACTION_RE.search(text)
    tool_input = _INPUT_RE.search(text)
    if action:
        name = action.group(1).strip()
        arg = tool_input.group(1).strip() if tool_input else ""
        return ("action", name, arg)
    return ("none", text.strip())


def run(question, *, client=None, tools=None, max_steps=5, temperature=0.0, verbose=False):
    client = client or get_client()
    tools = tools if tools is not None else TOOLS
    system = SYSTEM.format(tools=", ".join(sorted(tools)))
    transcript = "Question: {}".format(question)
    for _ in range(max_steps):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ]
        turn = client.chat(messages, temperature=temperature, max_tokens=400, stop=["Observation:"])
        transcript += "\n" + turn.strip()
        if verbose:
            print(turn.strip())
        kind = parse_step(turn)
        if kind[0] == "final":
            return kind[1], transcript
        if kind[0] == "action":
            name, arg = kind[1], kind[2]
            if name in tools:
                try:
                    observation = tools[name](arg)
                except Exception as exc:  # tool errors are fed back, not fatal
                    observation = "error: {}".format(exc)
            else:
                observation = "error: unknown tool {!r}".format(name)
            transcript += "\nObservation: {}".format(observation)
            if verbose:
                print("Observation: {}".format(observation))
        else:
            # No action and no final answer: nudge once more.
            transcript += "\nObservation: Please either call a tool or give the Final Answer."
    return None, transcript


def main():
    def demo():
        q = (
            "If a recipe needs 3 eggs per cake and I bake 7 cakes, how many "
            "eggs is that, and how many dozen does it round up to?"
        )
        answer, transcript = run(q, verbose=True)
        print("\nFINAL ANSWER: {}".format(answer))

    run_demo("ReAct (reason + act with tools)", demo)


if __name__ == "__main__":
    main()
