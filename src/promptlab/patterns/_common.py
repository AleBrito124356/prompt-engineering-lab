"""Shared helpers for the pattern demos.

Every technique module ends with ``main(argv=None)``, which calls
:func:`demo_main`. That gives each ``python -m promptlab.patterns.<name>`` the
same flags:

``--offline``
    Run the full technique loop against the module's ``demo_client()`` -- a
    :class:`~promptlab.backends.ScriptedClient` holding illustrative model
    output. No key, no network; the technique code itself runs for real.
``--record FILE`` / ``--replay FILE``
    Record a live NIM run to a JSONL cassette, or replay one offline.
``--model NAME``
    Override ``$NIM_MODEL`` for live and recorded runs.

``$PROMPTLAB_BACKEND`` (``mock``, ``record:FILE``, ``replay:FILE``) is honoured
when no flag is given.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager

from ..backends import OFFLINE_BANNER, ScriptedClient, backend_from_env, make_client
from ..client import BackendError, MissingKeyError

__all__ = ["get_client", "use_client", "run_demo", "demo_main", "section"]

_ACTIVE = []


def get_client(model=None):
    """The client a technique's ``run()`` uses when none is passed in.

    Inside :func:`use_client` that is the given client (how ``--offline``
    swaps in scripted output); otherwise the backend from
    ``$PROMPTLAB_BACKEND``, defaulting to live NIM.
    """
    if _ACTIVE:
        return _ACTIVE[-1]
    return make_client(model=model)


@contextmanager
def use_client(client):
    """Make ``client`` the default for :func:`get_client` inside the block."""
    _ACTIVE.append(client)
    try:
        yield client
    finally:
        _ACTIVE.pop()


def section(title):
    bar = "=" * 72
    return "{}\n{}\n{}".format(bar, title, bar)


def run_demo(title, fn):
    """Print a titled demo, converting a missing key into a friendly message.

    Returns 0 on success and 2 when the key is missing (kept for callers of
    the original API; the modules now use :func:`demo_main`).
    """
    print(section(title))
    try:
        fn()
    except MissingKeyError as exc:
        print(str(exc))
        print()
        return 2
    print()
    return 0


def _backend_spec(args):
    if args.offline:
        return "mock"
    if args.record:
        return "record:" + args.record
    if args.replay:
        return "replay:" + args.replay
    return backend_from_env()


def demo_main(title, fn, *, module, demo_client=None, argv=None):
    """Parse the demo flags, pick a backend and run ``fn`` under it.

    ``demo_client`` is a zero-argument factory returning the module's
    :class:`ScriptedClient`; it backs ``--offline`` (and ``mock``). Returns a
    process exit code: 0 on success, 2 for a missing key, 1 for a failed
    model call or cassette miss.
    """
    parser = argparse.ArgumentParser(prog="python -m promptlab.patterns.{}".format(module), description=title)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true", help="Scripted responses: no API key, no network.")
    group.add_argument("--record", metavar="CASSETTE", help="Run live on NIM and record calls to a JSONL file.")
    group.add_argument("--replay", metavar="CASSETTE", help="Replay a recorded JSONL cassette offline.")
    parser.add_argument("--model", help="Override NIM_MODEL for live/recorded runs.")
    args = parser.parse_args(argv)

    spec = _backend_spec(args)
    if spec == "mock" and demo_client is not None:
        client = demo_client()
    else:
        try:
            client = make_client(spec, model=args.model)
        except (ValueError, BackendError) as exc:
            print("error: {}".format(exc))
            return 2

    print(section(title))
    scripted = isinstance(client, ScriptedClient)
    if scripted:
        print(OFFLINE_BANNER)
        print()
    try:
        with use_client(client):
            fn()
    except MissingKeyError as exc:
        print(str(exc))
        print("\nNo key? See the technique run end to end on scripted output:")
        print("  python -m promptlab.patterns.{} --offline".format(module))
        return 2
    except BackendError as exc:
        print("error: {}".format(exc))
        return 1
    if scripted:
        print("\n[offline] {} scripted model call(s) served.".format(len(client.calls)))
    print()
    return 0
