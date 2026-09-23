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

from ..backends import ScriptedClient
from ._common import demo_main, get_client

SYSTEM = (
    "You extract structured data. Respond with a single JSON object only -- no "
    "prose, no markdown fences. Use null for unknown fields."
)

# ```json ... ```, ``` ... ```, ```JSON{...}``` -- language tag in group 1, body in group 2.
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


class JSONValidationError(ValueError):
    """Raised when extracted JSON fails schema validation."""


def _first_embedded_object(text):
    """Return ``(True, obj)`` for the first ``{`` that starts a valid JSON object.

    Uses ``json.JSONDecoder.raw_decode`` from every ``{`` in turn, so braces in
    surrounding prose ("note: {braces}") and a second object after the first
    one do not break extraction. The outermost object wins because scanning is
    left to right.
    """
    start = text.find("{")
    while start != -1:
        try:
            obj, _end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return True, obj
    return False, None


def extract_json(text):
    """Pull a JSON value out of a model response, tolerating fences and prose.

    Tries, in order:

    1. each fenced code block (```` ```json ```` blocks before untagged ones),
       as a whole and then for an embedded object;
    2. the whole response as JSON;
    3. the first valid ``{...}`` object embedded anywhere in the prose.

    Raises ``ValueError`` if none parse.
    """
    text = text or ""
    fences = sorted(
        _FENCE_RE.finditer(text), key=lambda m: 0 if m.group(1).lower() == "json" else 1
    )
    for fence in fences:
        body = fence.group(2).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            found, obj = _first_embedded_object(body)
            if found:
                return obj
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    found, obj = _first_embedded_object(text)
    if found:
        return obj
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


def run(text, schema, *, client=None, temperature=0.0, verbose=False):
    """Ask for JSON, extract it defensively and validate it against ``schema``.

    Raises ``ValueError`` when no JSON can be extracted and
    ``JSONValidationError`` when it does not match the schema.
    """
    client = client or get_client()
    raw = client.chat(build_messages(text, schema), temperature=temperature, max_tokens=400)
    if verbose:
        print("RAW MODEL OUTPUT:\n{}\n".format(raw))
    obj = extract_json(raw)
    return validate_schema(obj, schema)


DEMO_SCHEMA = {
    "name": {"type": "string", "required": True},
    "role": {"type": "string", "required": True},
    "years_experience": {"type": "integer", "required": False},
    "remote": {"type": "boolean", "required": False},
}
DEMO_TEXT = "Priya Nair is a senior data engineer with 8 years of experience, based remotely."


def demo_client():
    """Scripted model output for ``--offline`` (illustrative, not a live model).

    It ignores the "JSON only" instruction the way models often do: prose
    around the object and a stray ``{braces}`` aside, which a first-``{``-to-
    last-``}`` extractor could not parse.
    """
    return ScriptedClient(
        [
            'Here is the record: {"name": "Priya Nair", "role": "senior data engineer", '
            '"years_experience": 8, "remote": true} -- I left out {location} since only "remotely" '
            "was stated."
        ]
    )


def main(argv=None):
    def demo():
        result = run(DEMO_TEXT, DEMO_SCHEMA, verbose=True)
        print("VALIDATED JSON:\n{}".format(json.dumps(result, indent=2)))

    return demo_main(
        "Structured JSON (extract + validate)", demo, module="structured_json", demo_client=demo_client, argv=argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
