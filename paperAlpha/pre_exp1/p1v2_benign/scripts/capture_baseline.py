"""Source-checkout wrapper for required read-only inventory snapshots."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p1v2_benign.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["capture-baseline", *sys.argv[1:]]))
