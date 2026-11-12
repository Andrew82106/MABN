from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE_EXP = ROOT.parents[0]
sys.path[:0] = [str(ROOT / "src"), str(PRE_EXP / "p1_benign" / "src"), str(PRE_EXP / "src")]

from p1_model_qualification.providers import inspect_local_models

print(json.dumps(inspect_local_models(), ensure_ascii=False, sort_keys=True))
