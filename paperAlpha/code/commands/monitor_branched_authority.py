"""JSON-lines child. Receives allowlisted public config and runtime records only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor


def main(config, models):
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    monitor = BranchedAuthorityMonitor(read(config), *(read(models / name) for name in ("reference_fit.json", "system_fit.json", "logistic.json")))
    print(json.dumps({"ready": True, "protocol": "branched-authority-v1", "calibrated": False}), flush=True)
    for line in sys.stdin:
        try:
            result = monitor.push(json.loads(line))
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
