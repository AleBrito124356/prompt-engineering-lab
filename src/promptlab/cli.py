"""Command-line interface for promptlab.

Commands
--------
  list                        List library prompts with their purpose.
  show NAME [--version V]      Print a prompt's metadata and body.
  render NAME [vars]           Render a prompt to text (no network).
  run NAME [vars] [--user U]   Render as a system prompt and run it on NIM.
  compare A B --inputs FILE    A/B two prompts over an input set, judge-scored.
  diff NAME --a V1 --b V2      Unified diff between two versions.
  techniques                   Print the prompt-technique cheat-sheet.

Variables are passed as ``--var key=value`` (repeatable) and/or ``--vars-json
FILE``. ``A``/``B`` for compare accept ``name`` or ``name@version``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .patterns import TECHNIQUES
from .prompt import PromptError, PromptLibrary

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "library"


def _library(args):
    return PromptLibrary(getattr(args, "library", None) or _DEFAULT_ROOT)


def _collect_vars(args):
    variables = {}
    if getattr(args, "vars_json", None):
        variables.update(json.loads(Path(args.vars_json).read_text(encoding="utf-8")))
    for pair in getattr(args, "var", None) or []:
        if "=" not in pair:
            raise SystemExit("--var expects key=value, got {!r}".format(pair))
        key, value = pair.split("=", 1)
        variables[key.strip()] = value
    return variables


def _split_ref(ref):
    if "@" in ref:
        name, version = ref.split("@", 1)
        return name.strip(), version.strip()
    return ref.strip(), None


def _read_inputs(path):
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("inputs", [])
        return [str(x) for x in data]
    return [line for line in (l.strip() for l in text.splitlines()) if line]


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
    for r in rows:
        versions = ",".join(r["versions"])
        print("{name:<{w}}  [{versions}]  {purpose}".format(
            name=r["name"], w=width, versions=versions, purpose=r["purpose"]
        ))
    return 0


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
    print("-" * 72)
    print(prompt.body(args.version), end="")
    return 0


def cmd_render(args):
    lib = _library(args)
    variables = _collect_vars(args)
    prompt = lib.get(args.name)
    strict = not args.no_strict
    if strict:
        missing = prompt.template(args.version).missing(variables)
        if missing:
            raise SystemExit(
                "missing variables: {}\n(pass --var key=value or --no-strict)".format(
                    ", ".join(missing)
                )
            )
    print(prompt.render(variables, version=args.version, strict=strict))
    return 0


def cmd_run(args):
    from .runner import run_prompt  # deferred: needs openai only for `run`

    variables = _collect_vars(args)
    answer = run_prompt(
        args.name,
        variables,
        user_input=args.user,
        version=args.version,
        root=getattr(args, "library", None) or _DEFAULT_ROOT,
        model=args.model,
        temperature=args.temperature,
    )
    print(answer)
    return 0


def cmd_compare(args):
    from .client import NIMClient
    from .compare import render_table, run_comparison

    lib = _library(args)
    variables = _collect_vars(args)
    name_a, ver_a = _split_ref(args.a)
    name_b, ver_b = _split_ref(args.b)
    system_a = lib.get(name_a).render(variables, version=ver_a)
    system_b = lib.get(name_b).render(variables, version=ver_b)
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
    parser.add_argument("--library", help="Path to the prompt library (default: bundled library/).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List library prompts.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="Show a prompt's metadata and body.")
    p.add_argument("name")
    p.add_argument("--version", "-V")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("render", help="Render a prompt to text (no network).")
    p.add_argument("name")
    p.add_argument("--version", "-V")
    p.add_argument("--var", action="append", metavar="KEY=VALUE", help="Repeatable.")
    p.add_argument("--vars-json", metavar="FILE", help="JSON file of variables.")
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

    p = sub.add_parser("techniques", help="Print the technique cheat-sheet.")
    p.set_defaults(func=cmd_techniques)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PromptError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("error: file not found: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
