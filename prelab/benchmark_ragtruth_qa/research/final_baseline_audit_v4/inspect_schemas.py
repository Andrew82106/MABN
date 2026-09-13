import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAG = BASE / "research" / "ragognizer_formal_baseline_v1"

for name in [
    "INFERENCE_PLAN.jsonl",
    "NATIVE_PROBABILITIES.jsonl",
    "SHARED_PROBABILITIES.jsonl",
    "PROVISIONAL_CAL_NATIVE.jsonl",
    "PROVISIONAL_CAL_SHARED.jsonl",
]:
    path = RAG / name
    with path.open(encoding="utf-8") as handle:
        row = json.loads(next(handle))
    print(f"\n{name}")
    for key, value in row.items():
        if isinstance(value, list):
            print(key, "list", len(value), "head", value[:2])
        elif isinstance(value, dict):
            print(key, "dict", list(value)[:10])
        else:
            print(key, repr(value)[:300])
