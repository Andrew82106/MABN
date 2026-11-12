"""Public no-provider replay for one derived Q0 run ID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from p1v2_qualification.paths import canonical_json  # noqa: E402
from p1v2_qualification.replay import replay_run  # noqa: E402


DEFAULT_RUN_ID = "P1V2Q-READINESS-DRY-20260801T010106000000Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    result = replay_run(args.run_id)
    print(canonical_json(result))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
