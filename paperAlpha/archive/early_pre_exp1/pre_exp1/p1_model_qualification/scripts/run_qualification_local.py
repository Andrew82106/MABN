from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE_EXP = ROOT.parents[0]
sys.path[:0] = [str(ROOT / "src"), str(PRE_EXP / "p1_benign" / "src"), str(PRE_EXP / "src")]

from p1_model_qualification.core import CANDIDATE_MODELS, DEFAULT_CONTEXT
from p1_model_qualification.providers import QualificationOllamaProvider
from p1_model_qualification.runner import run_qualification


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=CANDIDATE_MODELS, required=True)
parser.add_argument("--allow-local-qualification", action="store_true")
args = parser.parse_args()
result = run_qualification(
    context=DEFAULT_CONTEXT,
    provider=QualificationOllamaProvider(model_id=args.model),
    execution_mode="local_model_qualification",
    allow_local_qualification=args.allow_local_qualification,
)
print(json.dumps({"run_id": result["run_id"], "validation": result["validation"], "replay": result["replay"]}, ensure_ascii=False, sort_keys=True))
