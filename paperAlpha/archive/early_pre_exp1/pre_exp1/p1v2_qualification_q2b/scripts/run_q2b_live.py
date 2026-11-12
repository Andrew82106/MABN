"""The sole authorized Q2-B live entry point; it accepts no endpoint or credential arguments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2b.common import safe_error_code
from p1v2_qualification_q2b.errors import ConfigError, ContractViolation, TransportError
from p1v2_qualification_q2b.runner import run_live


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live-q2b", action="store_true")
    args = parser.parse_args()
    if not args.allow_live_q2b:
        print(json.dumps({"decision": "rework", "error": "allow_live_q2b_required"}, sort_keys=True))
        return 2
    try:
        result = run_live()
    except (ConfigError, ContractViolation, TransportError, RuntimeError) as exc:
        print(json.dumps({"decision": "rework", "error": safe_error_code(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
