"""Public Q0-only runner entry point; it has no real-provider capability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from p1v2_qualification.paths import canonical_json  # noqa: E402
from p1v2_qualification.runner import run_readiness_dry  # noqa: E402


DEFAULT_RUN_ID = "P1V2Q-READINESS-DRY-20260801T010106000000Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    try:
        result = run_readiness_dry(run_id=args.run_id)
    except Exception as exc:
        code = getattr(exc, "code", "readiness_failed")
        print(canonical_json({"ok": False, "error": {"code": code, "message": "Q0 readiness run did not complete"}}))
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
