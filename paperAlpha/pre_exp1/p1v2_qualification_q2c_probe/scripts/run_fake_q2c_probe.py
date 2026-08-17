"""Create one deterministic offline fake Q2-C run; never performs network I/O."""

from __future__ import annotations

from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from q2c_probe.runner import create_offline_fake_run


def main() -> int:
    paper_alpha = Path(__file__).resolve().parents[3]
    data_root = paper_alpha / "data" / "pre_exp1" / "p1v2_qualification_q2c_probe"
    run_id, _ = create_offline_fake_run(data_root)
    print(f"OFFLINE_FAKE_RUN_CREATED run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
