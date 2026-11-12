"""Explicitly gated entrypoint for the one future Q2-C live probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from q2c_probe.live_runner import execute_released_live_probe


def main(argv: list[str] | None = None, executor=execute_released_live_probe) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live-q2c", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_live_q2c:
        print("LIVE_Q2C_DISABLED explicit_release_flag_required")
        return 2
    paper_alpha = Path(__file__).resolve().parents[3]
    data_root = paper_alpha / "data" / "pre_exp1" / "p1v2_qualification_q2c_probe"
    try:
        run_id, _ = executor(data_root, paper_alpha / ".env")
    except (OSError, RuntimeError, ValueError):
        print("LIVE_Q2C_STOPPED safe_failure")
        return 1
    print(f"LIVE_Q2C_COMPLETED run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
