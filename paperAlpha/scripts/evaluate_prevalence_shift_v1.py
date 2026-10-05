"""Prevalence-shift diagnostic for the frozen grouped OOF predictions.

This does not refit the monitor and is not an independent test set.  It
reweights the already frozen OOF predictions to low base rates so that the
paper reports how PR/Brier/ECE change when violations are rarer than in the
synthetic development benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(w.sum())
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if not np.any(mask):
            continue
        mass = float(w[mask].sum()) / total
        out += mass * abs(float(np.average(y[mask], weights=w[mask])) - float(np.average(p[mask], weights=w[mask])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prevalence", type=float, nargs="+", default=[0.05, 0.10])
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    observed = float(y.mean())
    methods = [key for key in rows[0] if key not in {"episode_id", "group", "label"}]
    results: dict[str, object] = {"input": str(args.input), "n": len(rows), "observed_prevalence": observed, "diagnostic_only": True, "methods": {}}
    for target in args.prevalence:
        if not 0.0 < target < 1.0:
            raise ValueError("prevalence must be in (0, 1)")
        w = np.where(y == 1, target / observed, (1.0 - target) / (1.0 - observed))
        bucket: dict[str, object] = {}
        for method in methods:
            p = np.asarray([float(row[method]) for row in rows], dtype=float)
            neg = p[y == 0]
            threshold = float(np.quantile(neg, 0.95))
            pred = p >= threshold
            recall = float(pred[y == 1].mean())
            bucket[method] = {
                "target_prevalence": target,
                "auroc": float(roc_auc_score(y, p, sample_weight=w)),
                "auprc": float(average_precision_score(y, p, sample_weight=w)),
                "brier": float(brier_score_loss(y, p, sample_weight=w)),
                "ece_10bin": weighted_ece(y, p, w),
                "pooled_negative_5pct_threshold": threshold,
                "pooled_fixed_5pct_fpr_recall": recall,
            }
        results["methods"][str(target)] = bucket
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
