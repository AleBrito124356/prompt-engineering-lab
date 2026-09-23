"""A minimal ReAct loop (Reason + Act).

The model interleaves reasoning and tool calls. It emits ``Thought``,
``Action`` and ``Action Input`` lines; the loop parses the action, runs a real
tool, and feeds the result back as an ``Observation``. It repeats until the
model emits ``Final Answer`` or a step budget is hit.

Why it helps: for anything that needs a fact the model does not have or a
computation it should not do in its head (arithmetic, string ops), giving it
tools beats hoping it guesses right. The safe calculator here uses an AST
evaluator -- never ``eval`` -- so untrusted expressions cannot run code, and it
bounds expression length, exponents and intermediate sizes so they cannot hang
the process either (``9**9**9`` is rejected in microseconds).
"""

from __future__ import annotations

import ast
import math
import operator
import re

from ..backends import ScriptedClient
from ._common import demo_main, get_client

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

#: Longest expression the calculator accepts, in characters.
MAX_EXPRESSION_LENGTH = 200
#: Largest absolute exponent allowed in ``a ** b``.
MAX_EXPONENT = 1000
#: Largest integer (in bits, ~1233 decimal digits) any intermediate may reach.
MAX_INT_BITS = 4096


def _check_size(value):
    if isinstance(value, complex):
        raise ValueError("result is not a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError("result too large (over {} bits)".format(MAX_INT_BITS))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not finite")
    return value


def _safe_pow(base, exp):
    if abs(exp) > MAX_EXPONENT:
        raise ValueError("exponent too large (|exponent| > {})".format(MAX_EXPONENT))
    if isinstance(base, int) and isinstance(exp, int) and exp > 0:
        if max(base.bit_length(), 1) * exp > MAX_INT_BITS:
            raise ValueError("result too large (over {} bits)".format(MAX_INT_BITS))
    return base ** exp


def safe_calculator(expression):
    """Evaluate an arithmetic expression with a whitelisted AST, no ``eval``.

    Supports ``+ - * / // % **``, unary ``+``/``-``, parentheses, and int/float
    literals. Anything else -- names, calls, attribute access, strings -- raises
    ``ValueError``. So do inputs that could exhaust CPU or memory: expressions
    longer than ``MAX_EXPRESSION_LENGTH`` characters, ``|exponent| >
    MAX_EXPONENT``, integers beyond ``MAX_INT_BITS`` bits, non-finite floats and
    complex results. Division by zero is a ``ValueError`` too, so a ReAct loop
    can feed every failure back to the model as an observation.
    """
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            "expression too long ({} > {} characters)".format(len(expression), MAX_EXPRESSION_LENGTH)
        )

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return _check_size(node.value)
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow):
                return _check_size(_safe_pow(left, right))
            return _check_size(_ALLOWED_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            raise ValueError("'^' is not supported; use ** for exponentiation")
        raise ValueError("unsupported expression element: {}".format(type(node).__name__))

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression: {}".format(exc.msg)) from None
    try:
        result = _eval(tree)
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except OverflowError:
        raise ValueError("result too large") from None
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


DEMO_QUESTION = (
    "If a recipe needs 3 eggs per cake and I bake 7 cakes, how many "
    "eggs is that, and how many dozen does it round up to?"
)

_OBSERVATION_RE = re.compile(r"^Observation:\s*(.+)$", re.MULTILINE)


def _final_from_observations(messages, params):
    """Scripted last turn that *reads* the real tool results from the transcript."""
    observations = _OBSERVATION_RE.findall(messages[-1]["content"])
    eggs, dozens = (observations + ["?", "?"])[:2]
    return (
        "Thought: The calculator says {eggs} eggs, and rounding {eggs}/12 up gives {dozens}.\n"
        "Final Answer: {eggs} eggs, which rounds up to {dozens} dozen.".format(eggs=eggs, dozens=dozens)
    )


def demo_client():
    """Scripted model turns for ``--offline`` (illustrative, not a live model).

    The model's *actions* are scripted; the observations are computed by the
    real ``safe_calculator`` and the final turn quotes them back.
    """
    return ScriptedClient(
        [
            "Thought: I need the total eggs: 3 eggs per cake times 7 cakes.\n"
            "Action: calculator\nAction Input: 3 * 7",
            "Thought: Now round the eggs up to whole dozens: ceil(21 / 12) = (21 + 11) // 12.\n"
            "Action: calculator\nAction Input: (21 + 11) // 12",
            _final_from_observations,
        ]
    )


def main(argv=None):
    def demo():
        answer, _transcript = run(DEMO_QUESTION, verbose=True)
        print("\nFINAL ANSWER: {}".format(answer))

    return demo_main(
        "ReAct (reason + act with tools)", demo, module="react_mini", demo_client=demo_client, argv=argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
