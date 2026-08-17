from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE_EXP = ROOT.parents[0]
sys.path[:0] = [str(ROOT / "src"), str(PRE_EXP / "p1_benign" / "src"), str(PRE_EXP / "src")]

from p1_benign.providers import DeterministicTestProvider
from p1_model_qualification.core import DEFAULT_CONTEXT
from p1_model_qualification.runner import run_qualification


result = run_qualification(
    context=DEFAULT_CONTEXT,
    provider=DeterministicTestProvider(),
    execution_mode="test_double",
)
print(json.dumps({"run_id": result["run_id"], "validation": result["validation"], "replay": result["replay"]}, ensure_ascii=False, sort_keys=True))
