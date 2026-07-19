"""Shared helpers for the pattern demos."""

from __future__ import annotations

from ..client import MissingKeyError, NIMClient

__all__ = ["get_client", "run_demo", "section"]


def get_client(model=None):
    return NIMClient(model=model)


def section(title):
    bar = "=" * 72
    return "{}\n{}\n{}".format(bar, title, bar)


def run_demo(title, fn):
    """Print a titled demo, converting a missing key into a friendly message."""
    print(section(title))
    try:
        fn()
    except MissingKeyError as exc:
        print(str(exc))
    print()
