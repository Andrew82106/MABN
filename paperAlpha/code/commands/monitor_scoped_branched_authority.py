"""Qualified JSON-lines monitor. Host sends explicit completion, never an Agent.

Control packet: {"control":"finish","completed":true|false}. It closes the
stream; endpoint heads are unavailable at every ordinary event callback.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor
from paperalpha_monitor.qualified_branched_stream import QualifiedBranchedStream


def main(config, models):
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    stream = QualifiedBranchedStream(BranchedAuthorityMonitor(read(config),
        *(read(models / name) for name in ("reference_fit.json", "system_fit.json", "logistic.json"))))
    print(json.dumps({"ready": True, "protocol": "scoped-branched-authority-v1", "future_forecast": False}), flush=True)
    for line in sys.stdin:
        try:
            record = json.loads(line)
            if record.get("control") == "finish":
                if set(record) != {"control", "completed"}:
                    raise ValueError("exact host completion packet required")
                result = stream.finish(completed=record["completed"])
            else:
                result = stream.push(record)
        except Exception as exc:
            print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)
            return 1
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.models))
