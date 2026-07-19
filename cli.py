#!/usr/bin/env python3
"""promptlab CLI entry point.

Runs without installation: this shim puts ``src/`` on the path and delegates to
``promptlab.cli``. After ``pip install -e .`` you can also use the ``promptlab``
command directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from promptlab.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
