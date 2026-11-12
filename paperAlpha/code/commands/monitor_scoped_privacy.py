"""Isolated numerical monitor; stdin has only the allowed current observation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.scoped_privacy_monitor import ScopedPrivacyMonitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    monitor = ScopedPrivacyMonitor(args.models)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            result = {"ok": True, "result": monitor.score(json.loads(line))}
        except Exception as exc:
            result = {"ok": False, "error_type": type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
