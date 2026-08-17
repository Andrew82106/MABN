"""Source-checkout wrapper for the only P1v2-A writer."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p1v2_benign.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["dry", *sys.argv[1:]]))
