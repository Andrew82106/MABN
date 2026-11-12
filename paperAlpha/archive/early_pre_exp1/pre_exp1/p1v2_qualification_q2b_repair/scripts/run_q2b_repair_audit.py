"""Create one offline derived audit; it has no credential or network path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2b_repair.audit import run_derived_audit


if __name__ == "__main__":
    print(json.dumps(run_derived_audit(), ensure_ascii=False, sort_keys=True))
