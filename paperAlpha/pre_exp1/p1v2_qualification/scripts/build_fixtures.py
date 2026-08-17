"""One-time deterministic builder for the frozen, benign Q1 task-card files."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from p1v2_qualification.fixtures import write_fixture_sets  # noqa: E402


def main() -> int:
    try:
        write_fixture_sets(CODE_ROOT)
    except Exception as exc:  # Safe, terse developer entry point.
        print('{"ok":false,"error":{"code":"fixture_build_failed","message":"fixture build did not complete"}}')
        return 2
    print('{"ok":true,"fixture_build":"complete"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
