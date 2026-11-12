"""Evidence-qualified passive JSON monitor; frozen acquisition CLI is unchanged.

Only configured reference-policy risk is scored. Incomplete observation coverage
is reported separately, never mapped to a fabricated permission or safe label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from monitor_delegation_pilot import make_compiler, restore
from paperalpha_monitor.authority_qualification import QualifiedDelegationStream


def main(config_path, model_path):
    model = restore(model_path)
    compiler = make_compiler(json.loads(config_path.read_text(encoding="utf-8")))
    stream = QualifiedDelegationStream(model, compiler)
    print(json.dumps({"ready": True, "protocol": "evidence-qualified-delegation-v1",
        "parameters": len(model.theta), "fitted_targets": model.fitted_targets,
        "calibrated": False, "automatic_blocking": False}), flush=True)
    for line in sys.stdin:
        try:
            result = stream.push(json.loads(line))
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(main(args.config, args.model))
