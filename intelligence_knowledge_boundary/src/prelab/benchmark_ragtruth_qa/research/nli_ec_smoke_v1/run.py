"""Gold-driven error diagnostic only; never a detector score or train feature."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
CASES = ROOT / "research/current_ec_case_review_v1/CASE_REVIEW.json"


def main():
    torch.set_num_threads(4)
    data = json.loads(CASES.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True).eval().cpu()
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    pairs, owners = [], []
    for case in data["cases"]:
        hypothesis = case["gold"]["text"]
        excerpts = [item["text"] for item in case["source_excerpts"]]
        for kind, premise in [("joined", "\n".join(excerpts)), *[(f"excerpt_{i}", x) for i, x in enumerate(excerpts)]]:
            pairs.append((premise, hypothesis))
            owners.append((case["response_id"], case["outcome"], case["interpretation"], kind))
    encoded = tokenizer([p for p, _ in pairs], [h for _, h in pairs], padding=True,
                        truncation=False, return_tensors="pt")
    started = time.perf_counter()
    with torch.inference_mode():
        probability = model(**encoded).logits.float().softmax(-1).numpy()
    rows = []
    for owner, prob in zip(owners, probability):
        rows.append(dict(response_id=owner[0], current_outcome=owner[1], interpretation=owner[2],
                         evidence_view=owner[3], entailment=float(prob[0]), neutral=float(prob[1]),
                         contradiction=float(prob[2])))
    result = {
        "status": "diagnostic_only",
        "warning": "Uses the annotated hallucinated span and hand-selected evidence excerpts. It cannot be reported as detector performance or used as a model feature.",
        "model": "tasksource/ModernBERT-base-nli local frozen checkpoint",
        "cases": len(data["cases"]),
        "pairs": len(rows),
        "seconds": time.perf_counter() - started,
        "rows": rows,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
