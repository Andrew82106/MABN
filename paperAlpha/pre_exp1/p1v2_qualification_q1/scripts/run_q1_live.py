"""The sole public entry point allowed to perform the Q1 live batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1v2_qualification_q1.runner import run_live  # noqa: E402


def main(argv: list[str]) -> int:
    if argv != ["--allow-live-q1"]:
        error = "live_authorization_required" if not argv else "invalid_arguments"
        print(json.dumps({"passed": False, "error": error, "real_model_calls": 0, "local_loopback_http_calls": 0}, sort_keys=True))
        return 2
    try:
        result = run_live()
    except Exception as exc:
        print(json.dumps({"passed": False, "error": f"runner_failed:{type(exc).__name__}", "real_model_calls": 0}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, "run_id": result["run_id"], "qualification_batch_decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
