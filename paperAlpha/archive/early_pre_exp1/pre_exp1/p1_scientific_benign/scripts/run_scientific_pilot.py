from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE_EXP = ROOT.parents[0]
sys.path[:0] = [str(ROOT / "src"), str(PRE_EXP / "p1_benign" / "src"), str(PRE_EXP / "p1_model_qualification" / "src"), str(PRE_EXP / "src")]

from p1_scientific_benign.core import EXECUTION_REAL
from p1_scientific_benign.runner import run_batch

result = run_batch(execution_mode=EXECUTION_REAL, allow_local_pilot=True)
print(json.dumps({"run_id": result["run_id"], "validation": result["validation"], "replay": result["replay"]}, ensure_ascii=False, sort_keys=True))
