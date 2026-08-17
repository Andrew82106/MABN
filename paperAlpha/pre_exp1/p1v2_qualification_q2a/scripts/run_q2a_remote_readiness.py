"""Public entry point for one deterministic, mock-only Q2-A readiness run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2a.runner import run_readiness


if __name__ == "__main__":
    print(json.dumps(run_readiness(), ensure_ascii=False, sort_keys=True))
