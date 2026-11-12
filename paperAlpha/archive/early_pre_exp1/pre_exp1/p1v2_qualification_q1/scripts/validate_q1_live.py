"""Offline public validator for a Q1 run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1v2_qualification_q1.validation import validate_run  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "--run-id":
        print(json.dumps({"passed": False, "errors": ["invalid_arguments"]}, sort_keys=True))
        return 2
    try:
        result = validate_run(argv[1])
    except Exception as exc:
        print(json.dumps({"passed": False, "errors": [f"validator_failed:{type(exc).__name__}"]}, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
