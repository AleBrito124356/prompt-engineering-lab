"""Command-line interface for promptlab.

Commands
--------
  list                        List library prompts with their purpose.
  show NAME [--version V]      Print a prompt's metadata, input contract and body.
  render NAME [vars]           Render a prompt to text (no network).
  run NAME [vars] [--user U]   Render as a system prompt and run it on NIM.
  compare A B --inputs FILE    A/B two prompts over an input set, judge-scored.
  diff NAME --a V1 --b V2      Unified diff between two versions.
  lint [NAME ...]              Check prompts against their declared contracts.
  techniques                   Print the prompt-technique cheat-sheet.

Variables are passed as ``--var key=value`` (repeatable) and/or ``--vars-json
FILE``. ``A``/``B`` for compare accept ``name`` or ``name@version``.

Exit codes: 0 success, 1 lint findings or a failed model call, 2 usage or
configuration errors (unknown prompt, missing variable, missing API key...).
Expected errors print a single ``error: ...`` message, never a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import BackendError, MissingKeyError
from .patterns import TECHNIQUES
from .prompt import PromptError, PromptLibrary, default_library_root
from .template import TemplateError


class CLIError(Exception):
    """A usage error: printed as ``error: <message>`` and exit code 2."""


def _library_root(args):
    return Path(getattr(args, "library", None) or default_library_root())


def _library(args):
    return PromptLibrary(_library_root(args))


def _collect_vars(args):
    variables = {}
    path = getattr(args, "vars_json", None)
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CLIError("--vars-json file not found: {}".format(path)) from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CLIError("--vars-json {}: invalid JSON ({})".format(path, exc)) from None
        if not isinstance(data, dict):
            raise CLIError(
                "--vars-json {}: expected a JSON object of variables, got {}".format(
                    path, type(data).__name__
                )
            )
        variables.update(data)
    for pair in getattr(args, "var", None) or []:
        if "=" not in pair:
            raise CLIError("--var expects key=value, got {!r}".format(pair))
        key, value = pair.split("=", 1)
        variables[key.strip()] = value
    return variables


def _split_ref(ref):
    if "@" in ref:
        name, version = ref.split("@", 1)
        return name.strip(), version.strip()
    return ref.strip(), None


def _read_inputs(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CLIError("inputs file not found: {}".format(path)) from None
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CLIError("inputs file {}: invalid JSON ({})".format(path, exc)) from None
        if isinstance(data, dict):
            data = data.get("inputs", [])
        if not isinstance(data, list):
            raise CLIError("inputs file {}: expected a JSON array or {{\"inputs\": [...]}}".format(path))
        inputs = [str(x) for x in data if str(x).strip()]
    else:
        inputs = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if not inputs:
        raise CLIError("inputs file {} contains no inputs".format(path))
    return inputs


def _render_checked(prompt, variables, version, *, strict=True, hint_no_strict=False):
    """Render with the contract applied, turning missing names into a CLIError."""
    if strict:
        missing = prompt.missing(variables, version=version)
        if missing:
            raise CLIError(
                "{}: missing variables: {} (pass --var key=value{})".format(
                    prompt.ref(version),
                    ", ".join(missing),
                    " or --no-strict" if hint_no_strict else "",
                )
            )
    return prompt.render(variables, version=version, strict=strict)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(args):
    lib = _library(args)
    rows = lib.summaries()
    if not rows:
        print("No prompts found in {}".format(lib.root))
        return 0
    width = max(len(r["name"]) for r in rows)
    vwidth = max(len(",".join(r["versions"])) + 2 for r in rows)
    for r in rows:
        versions = "[{}]".format(",".join(r["versions"]))
        print("{name:<{w}}  {versions:<{vw}}  {purpose}".format(
            name=r["name"], w=width, versions=versions, vw=vwidth, purpose=r["purpose"]
        ))
    return 0


def _describe_input(spec):
    bits = ["required" if spec.required else "optional"]
    if spec.has_default and spec.default not in ("", None):
        bits.append("default {!r}".format(spec.default))
    if spec.enum:
        bits.append("one of: {}".format(", ".join(str(e) for e in spec.enum)))
    return "  - {} ({}){}".format(
        spec.name, "; ".join(bits), "  " + spec.description if spec.description else ""
    )


def cmd_show(args):
    lib = _library(args)
    prompt = lib.get(args.name)
    meta = prompt.metadata(args.version)
    print("# {}  ({})".format(meta.get("title", prompt.name), meta["version"]))
    print("versions : {}".format(", ".join(prompt.versions)))
    if meta.get("purpose"):
        print("purpose  : {}".format(meta["purpose"]))
    if meta.get("tags"):
        print("tags     : {}".format(", ".join(meta["tags"])))
    if meta.get("model_tips"):
        print("model    : {}".format(meta["model_tips"]))
    specs = prompt.inputs(args.version)
    if specs:
        print("inputs   :")
        for spec in specs:
            print(_describe_input(spec))
    print("-" * 72)
    print(prompt.body(args.version), end="")
    return 0


def cmd_render(args):
    lib = _library(args)
    variables = _collect_vars(args)
    prompt = lib.get(args.name)
    print(_render_checked(prompt, variables, args.version, strict=not args.no_strict, hint_no_strict=True))
    return 0


def cmd_run(args):
    from .client import NIMClient

    lib = _library(args)
    variables = _collect_vars(args)
    prompt = lib.get(args.name)
    system = _render_checked(prompt, variables, args.version)
    client = NIMClient(model=args.model)
    user = args.user if args.user is not None else "Begin."
    answer = client.complete(user, system=system, temperature=args.temperature, max_tokens=1024)
    print(answer)
    return 0


def cmd_compare(args):
    from .client import NIMClient
    from .compare import render_table, run_comparison

    lib = _library(args)
    variables = _collect_vars(args)
    name_a, ver_a = _split_ref(args.a)
    name_b, ver_b = _split_ref(args.b)
    system_a = _render_checked(lib.get(name_a), variables, ver_a)
    system_b = _render_checked(lib.get(name_b), variables, ver_b)
    inputs = _read_inputs(args.inputs)
    label_a = args.a
    label_b = args.b
    client = NIMClient(model=args.model)
    results = run_comparison(
        system_a, system_b, inputs, client=client, label_a=label_a, label_b=label_b, seed=args.seed
    )
    print(render_table(results, label_a=label_a, label_b=label_b))
    return 0


def cmd_diff(args):
    lib = _library(args)
    prompt = lib.get(args.name)
    out = prompt.diff(args.a, args.b)
    if not out.strip():
        print("No differences between {}@{} and {}@{}.".format(
            args.name, prompt.resolve_version(args.a), args.name, prompt.resolve_version(args.b)
        ))
    else:
        print(out, end="")
    return 0


def cmd_lint(args):
    from .lint import format_json, format_text, has_failures, lint_library

    root = _library_root(args)
    findings = lint_library(root, args.names or None)
    checked = len(args.names) if args.names else len(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]
    )
    if args.format == "json":
        print(format_json(findings, checked))
    else:
        print(format_text(findings, checked))
    return 1 if has_failures(findings, strict=args.strict) else 0


def cmd_techniques(args):
    width = max(len(t["name"]) for t in TECHNIQUES)
    print("{:<{w}}  {:<28}  {}".format("technique", "cost", "when to use", w=width))
    print("-" * 100)
    for t in TECHNIQUES:
        print("{:<{w}}  {:<28}  {}".format(t["name"], t["cost"], t["when"], w=width))
    print("\nRun a live demo:  python -m promptlab.patterns.<technique>")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(prog="promptlab", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--library",
        help="Path to a prompt library (default: $PROMPTLAB_LIBRARY or the bundled library).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List library prompts.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="Show a prompt's metadata, inputs and body.")
    p.add_argument("name")
    p.add_argument("--version", "-V")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("render", help="Render a prompt to text (no network).")
    p.add_argument("name")
    p.add_argument("--version", "-V")
    p.add_argument("--var", action="append", metavar="KEY=VALUE", help="Repeatable.")
    p.add_argument("--vars-json", metavar="FILE", help="JSON file with an object of variables.")
    p.add_argument("--no-strict", action="store_true", help="Render missing variables as empty.")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("run", help="Render a prompt and run it on NVIDIA NIM.")
    p.add_argument("name")
    p.add_argument("--version", "-V")
    p.add_argument("--user", "-u", help="The user turn to send.")
    p.add_argument("--var", action="append", metavar="KEY=VALUE")
    p.add_argument("--vars-json", metavar="FILE")
    p.add_argument("--model", help="Override NIM_MODEL.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="A/B two prompts, judge-scored.")
    p.add_argument("a", help="Prompt A as name or name@version.")
    p.add_argument("b", help="Prompt B as name or name@version.")
    p.add_argument("--inputs", required=True, metavar="FILE", help="JSON array or newline-delimited inputs.")
    p.add_argument("--var", action="append", metavar="KEY=VALUE")
    p.add_argument("--vars-json", metavar="FILE")
    p.add_argument("--model", help="Override NIM_MODEL.")
    p.add_argument("--seed", type=int, default=None, help="Seed for judge order shuffling.")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("diff", help="Diff two versions of a prompt.")
    p.add_argument("name")
    p.add_argument("--a", required=True, help="From version, e.g. v1.")
    p.add_argument("--b", required=True, help="To version, e.g. v2.")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("lint", help="Check prompts against their declared input contracts.")
    p.add_argument("names", nargs="*", metavar="NAME", help="Prompts to lint (default: all).")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--strict", action="store_true", help="Fail on warnings too.")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("techniques", help="Print the technique cheat-sheet.")
    p.set_defaults(func=cmd_techniques)

    return parser


def _error(message, code):
    print("error: {}".format(message), file=sys.stderr)
    return code


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CLIError, PromptError, TemplateError) as exc:
        return _error(exc, 2)
    except MissingKeyError as exc:
        return _error(exc, 2)
    except BackendError as exc:
        return _error(exc, 1)
    except FileNotFoundError as exc:
        return _error("file not found: {}".format(exc.filename or exc), 2)
    except KeyboardInterrupt:
        return _error("interrupted", 130)


if __name__ == "__main__":
    raise SystemExit(main())
