"""Break down independent_mas_v3 predictions by hidden mechanism.

Mechanism labels are evaluation metadata only and are never fed to the model.
This report is used to decide which risk templates are missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=Path('experiments/synthetic_mas/runs/two_layer_independent_mas_v3/independent_predictions.jsonl'))
    ap.add_argument("--labels", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl'))
    ap.add_argument("--out", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/mechanism_error_analysis.json'))
    args = ap.parse_args()
    pred = {r["episode_id"]: r for r in read(args.predictions)}
    labels = {r["episode_id"]: r for r in read(args.labels)}
    rows = [pred[k] | labels[k] for k in pred if k in labels]
    result = {}
    for mechanism in sorted({r.get("mechanism", "unknown") for r in rows}):
        group = [r for r in rows if r.get("mechanism") == mechanism]
        y = np.asarray([int(r["label"]) for r in group])
        p = np.asarray([float(r["risk_probability"]) for r in group])
        result[mechanism] = {
            "n": len(group),
            "positive": int(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else None,
            "f1_at_0_5": float(f1_score(y, p >= .5, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "auprc": float(average_precision_score(y, p)),
            "mean_score_positive": float(p[y == 1].mean()) if (y == 1).any() else None,
            "mean_score_negative": float(p[y == 0].mean()) if (y == 0).any() else None,
        }
    payload = {"source": str(args.predictions), "mechanisms": result, "interpretation": "Use low AUROC/AUPRC mechanisms to extend risk templates; mechanism metadata is not a model feature."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = args.out.with_name("MECHANISM_ERROR_REPORT.md")
    lines = ["# Mechanism error analysis", "", "| mechanism | n | positive rate | F1@0.5 | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in result.items():
        lines.append(f"| {k} | {v['n']} | {v['positive_rate']:.3f} | {v['f1_at_0_5']:.3f} | {v['auroc'] if v['auroc'] is not None else 'n/a'} | {v['auprc']:.3f} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
