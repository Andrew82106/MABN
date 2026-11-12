"""Future-only placeholder; no live behavior is implemented in Q2-A."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live-q2", action="store_true")
    args = parser.parse_args()
    if not args.allow_live_q2:
        print(json.dumps({"decision": "rework", "reason": "allow_live_q2_required"}, sort_keys=True))
        return 2
    print(json.dumps({"decision": "rework", "reason": "live_execution_outside_q2a_scope"}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
