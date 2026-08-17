"""Source-checkout wrapper for run-p1-local-shakedown."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p1_benign.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main(["run-p1-local-shakedown", *sys.argv[1:]])
    )

