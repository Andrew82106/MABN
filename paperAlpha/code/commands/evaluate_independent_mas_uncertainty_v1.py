"""Grouped OOF coverage--risk audit for the two-layer monitor.

This is deliberately separate from the headline score.  The point/lower/upper
runtime BN scores are produced from each outer training fold; an episode is
abstained when the observation interval crosses the train-only F1 threshold.
The interval is epistemic missing-provenance uncertainty, not a calibrated
confidence interval.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


journal = load("journal_uncertainty_base", Path(__file__).with_name("evaluate_independent_mas_journal_v1.py"))
two = journal.load_two_layer()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def best_threshold(y, p):
    vals = np.unique(np.r_[0.0, p, 1.0])
    return float(max(vals, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def summarize(y, point, lower, upper, threshold, abstain):
    certain = ~abstain
    pred = point >= threshold
    out = {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "threshold": float(threshold),
        "point": {
            "auroc": float(roc_auc_score(y, point)),
            "auprc": float(average_precision_score(y, point)),
            "brier": float(brier_score_loss(y, point)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        },
        "interval": {
            "mean_width": float(np.mean(upper - lower)),
            "median_width": float(np.median(upper - lower)),
            "abstention_rate": float(abstain.mean()),
            "coverage": float(certain.mean()),
            "certain_error_rate": float(np.mean(pred[certain] != y[certain])) if certain.any() else None,
            "certain_f1": float(f1_score(y[certain], pred[certain], zero_division=0)) if certain.any() else None,
            "certain_positive_rate": float(y[certain].mean()) if certain.any() else None,
        },
    }
    # Coverage--risk points obtained by widening the abstention policy with a
    # deterministic interval-width quantile.  This does not tune the model.
    width = upper - lower
    curve = []
    positive_width = width[width > 0]
    for q in (0.0, 0.10, 0.25, 0.50, 0.75):
        # Quantiles are taken over non-zero widths; otherwise the large mass of
        # fully observed episodes makes every curve point identical.
        cutoff = float(np.quantile(positive_width, q)) if len(positive_width) else 0.0
        # Strictly greater avoids rejecting every episode when the interval is
        # exactly zero (which is expected for fully observed traces).
        reject = width > cutoff
        keep = ~reject
        curve.append({
            "reject_width_quantile": q,
            "cutoff": cutoff,
            "coverage": float(keep.mean()),
            "risk": float(np.mean(pred[keep] != y[keep])) if keep.any() else None,
        })
    out["coverage_risk_curve"] = curve
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / 'paperAlpha/experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl')
    ap.add_argument("--labels", type=Path, default=ROOT / 'paperAlpha/experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl')
    ap.add_argument("--out", type=Path, default=ROOT / 'paperAlpha/experiments/submission_audit/runs/final_eval/uncertainty_coverage_v1')
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    traces = read_jsonl(args.input)
    labels = {r["episode_id"]: r for r in read_jsonl(args.labels)}
    y = np.asarray([int(labels[e["episode_id"]]["label"]) for e in traces], dtype=int)
    groups = np.asarray([labels[e["episode_id"]]["scenario_family"] for e in traces])
    point = np.zeros(len(y)); lower = np.zeros(len(y)); upper = np.zeros(len(y)); abstain = np.zeros(len(y), dtype=bool)
    thresholds = []
    fold_rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups), 1):
        train_eps = [{**traces[i], "label": int(y[i])} for i in train]
        test_eps = [{**traces[i], "label": int(y[i])} for i in test]
        bn = two.fit_bn(train_eps, y[train])
        theta = journal.fit_runtime_bn_inner(two, train_eps, y[train], bn, args.seed + fold)
        p_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
        p = np.asarray([two._propagate(theta, two.event_features(e), bn, "point") for e in test_eps])
        lo = np.asarray([two._propagate(theta, two.event_features(e), bn, "lower") for e in test_eps])
        hi = np.asarray([two._propagate(theta, two.event_features(e), bn, "upper") for e in test_eps])
        threshold = best_threshold(y[train], p_train)
        ab = (lo < threshold) & (hi >= threshold)
        point[test], lower[test], upper[test], abstain[test] = p, lo, hi, ab
        thresholds.append(threshold)
        fold_rows.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "threshold": threshold, "abstention_rate": float(ab.mean())})
    summary = summarize(y, point, lower, upper, float(np.mean(thresholds)), abstain)
    summary.update({"protocol": {"split": "5-fold StratifiedGroupKFold by scenario_family", "label_use": "episode_id join only", "interval": "unknown/redacted edge lower/point/upper semantics", "abstain": "interval crosses train-only fold threshold", "causal_claim": False}, "folds": fold_rows})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "REPORT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as h:
        for i, e in enumerate(traces):
            h.write(json.dumps({"episode_id": e["episode_id"], "group": groups[i], "label": int(y[i]), "point": float(point[i]), "lower": float(lower[i]), "upper": float(upper[i]), "abstain": bool(abstain[i])}, sort_keys=True) + "\n")
    s = summary["interval"]
    lines = ["# Independent MAS uncertainty and coverage-risk audit v1", "", "Grouped OOF runtime-BN interval audit; lower/point/upper is missing-provenance uncertainty, not a calibrated probability interval.", "", f"- n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}", f"- point AUROC/AUPRC/Brier/F1: {summary['point']['auroc']:.3f}/{summary['point']['auprc']:.3f}/{summary['point']['brier']:.3f}/{summary['point']['f1']:.3f}", f"- interval mean/median width: {s['mean_width']:.3f}/{s['median_width']:.3f}", f"- abstention/coverage: {s['abstention_rate']:.3f}/{s['coverage']:.3f}", f"- certain error rate/F1: {s['certain_error_rate']:.3f}/{s['certain_f1']:.3f}", "", "| rejected-width quantile | coverage | risk |", "|---:|---:|---:|"]
    for row in summary["coverage_risk_curve"]:
        lines.append(f"| {row['reject_width_quantile']:.2f} | {row['coverage']:.3f} | {row['risk']:.3f} |" if row["risk"] is not None else f"| {row['reject_width_quantile']:.2f} | {row['coverage']:.3f} | n/a |")
    lines += ["", "The audit is a selective-prediction diagnostic. It does not claim that the black-box API model exposes calibrated uncertainty.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
