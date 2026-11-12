"""Public, offline-only replay entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from q2c_probe.replay import replay_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    paper_alpha = Path(__file__).resolve().parents[3]
    data_root = paper_alpha / "data" / "pre_exp1" / "p1v2_qualification_q2c_probe"
    errors = replay_run(data_root, args.run_id)
    if errors:
        print("REPLAY_FAIL codes=" + ",".join(errors))
        return 1
    print(f"REPLAY_PASS run_id={args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
