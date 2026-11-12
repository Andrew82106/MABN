"""Public, read-only semantic replay for a repair-derived audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2b_repair.replay import replay_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-id", required=True)
    args = parser.parse_args()
    result = replay_audit(args.audit_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
