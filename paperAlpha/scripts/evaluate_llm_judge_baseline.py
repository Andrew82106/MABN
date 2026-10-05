"""Evaluate predictions from the strict same-information LLM judge adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(predictions_path: Path, labels_path: Path) -> dict:
    predictions = read_jsonl(predictions_path)
    labels = {str(row["episode_id"]): row for row in read_jsonl(labels_path)}
    seen: set[str] = set()
    rows = []
    invalid = []
    for row in predictions:
        episode_id = str(row.get("episode_id", ""))
        if not episode_id or episode_id in seen or episode_id not in labels:
            invalid.append({"episode_id": episode_id, "reason": "missing_duplicate_or_unknown_id"})
            continue
        seen.add(episode_id)
        try:
            probability = float(row["risk_probability"])
        except (KeyError, TypeError, ValueError):
            invalid.append({"episode_id": episode_id, "reason": "invalid_risk_probability"})
            continue
        if not 0.0 <= probability <= 1.0:
            invalid.append({"episode_id": episode_id, "reason": "risk_probability_out_of_range"})
            continue
        rows.append((episode_id, int(labels[episode_id]["label"]), probability))
    y = np.asarray([row[1] for row in rows], dtype=int)
    p = np.asarray([row[2] for row in rows], dtype=float)
    pred = p >= 0.5
    result = {
        "schema": "paperalpha-llm-judge-baseline-evaluation-v1",
        "predictions": str(predictions_path),
        "labels": str(labels_path),
        "n_predicted": int(len(rows)),
        "n_invalid": int(len(invalid)),
        "n_missing_from_predictions": int(len(set(labels) - seen)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "threshold": 0.5,
        "metrics": {
            "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else None,
            "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
            "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "auprc": float(average_precision_score(y, p)) if len(y) else None,
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
        },
        "invalid_rows": invalid,
        "interpretation": "zero-shot same-information API judge; partial smoke runs are diagnostic and are not a main-table result",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("paperAlpha/results/independent_mas_v3/labels.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
