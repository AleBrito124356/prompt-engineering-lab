"""Structured JSON output.

Ask for JSON that matches a named shape, then parse and validate it locally.
Models drift -- they wrap JSON in prose or code fences -- so the robust move is
to extract the JSON block defensively and validate required keys and types
before trusting it.

Why it helps: downstream code needs data, not paragraphs. Validating locally
turns "usually valid JSON" into "provably valid or a clear error", which is the
difference between a demo and a pipeline.
"""

from __future__ import annotations

import json
import re

from ._common import get_client, run_demo

SYSTEM = (
    "You extract structured data. Respond with a single JSON object only -- no "
    "prose, no markdown fences. Use null for unknown fields."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONValidationError(ValueError):
    """Raised when extracted JSON fails schema validation."""


def extract_json(text):
    """Pull a JSON object out of a model response, tolerating fences and prose.

    Tries, in order: a fenced ```json block, the raw string, and the first
    balanced ``{...}`` span. Raises ``ValueError`` if none parse.
    """
    candidates = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("no valid JSON object found in response")


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_schema(obj, schema):
    """Validate ``obj`` against a tiny schema: {field: {'type', 'required'}}.

    Returns ``obj`` on success; raises ``JSONValidationError`` with all problems.
    """
    if not isinstance(obj, dict):
        raise JSONValidationError("expected a JSON object, got {}".format(type(obj).__name__))
    errors = []
    for field, spec in schema.items():
        required = spec.get("required", False)
        if field not in obj or obj[field] is None:
            if required:
                errors.append("missing required field: {!r}".format(field))
            continue
        expected = spec.get("type")
        if expected and expected in _TYPE_MAP:
            py_type = _TYPE_MAP[expected]
            # bool is a subclass of int; guard against it counting as a number.
            if expected in ("number", "integer") and isinstance(obj[field], bool):
                errors.append("field {!r} should be {}, got boolean".format(field, expected))
            elif not isinstance(obj[field], py_type):
                errors.append(
                    "field {!r} should be {}, got {}".format(
                        field, expected, type(obj[field]).__name__
                    )
                )
    if errors:
        raise JSONValidationError("; ".join(errors))
    return obj


def build_messages(text, schema):
    fields = "\n".join(
        "- {}: {}{}".format(
            name, spec.get("type", "any"), " (required)" if spec.get("required") else ""
        )
        for name, spec in schema.items()
    )
    user = "Extract these fields as JSON:\n{}\n\nText:\n{}".format(fields, text)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def run(text, schema, *, client=None, temperature=0.0):
    client = client or get_client()
    raw = client.chat(build_messages(text, schema), temperature=temperature, max_tokens=400)
    obj = extract_json(raw)
    return validate_schema(obj, schema)


def main():
    def demo():
        schema = {
            "name": {"type": "string", "required": True},
            "role": {"type": "string", "required": True},
            "years_experience": {"type": "integer", "required": False},
            "remote": {"type": "boolean", "required": False},
        }
        text = "Priya Nair is a senior data engineer with 8 years of experience, based remotely."
        result = run(text, schema)
        print("VALIDATED JSON:\n{}".format(json.dumps(result, indent=2)))

    run_demo("Structured JSON (extract + validate)", demo)


if __name__ == "__main__":
    main()
