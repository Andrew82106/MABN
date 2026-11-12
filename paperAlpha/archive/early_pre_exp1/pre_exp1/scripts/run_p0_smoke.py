"""Run the local deterministic P0 smoke test from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pre_exp1.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["run-smoke", *sys.argv[1:]]))

