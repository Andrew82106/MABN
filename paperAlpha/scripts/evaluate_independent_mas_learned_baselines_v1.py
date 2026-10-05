"""Strict family-disjoint learned MAS baseline comparison.

The three baselines reuse only public trace fields and differ in how much
structure they expose to a train-only logistic head: flat counts, local
per-agent pooling, and graph summaries with connectivity/taint paths.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def load_fair():
    path = Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py")
    spec = importlib.util.spec_from_file_location("fair_baseline_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    values = np.unique(np.r_[0.0, p, 1.0])
    return float(max(values, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target: float = 0.05) -> float:
    negative = p[y == 0]
    return float(np.quantile(negative, 1.0 - target, method="higher")) if len(negative) else 1.0


def metric(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None, "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "positive_predictions": int(pred.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/independent_mas_learned_baselines_v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    fair = load_fair()
    traces = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r["episode_id"]: r for r in (json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    y = np.asarray([int(labels[t["episode_id"]]["label"]) for t in traces], dtype=int)
    groups = np.asarray([labels[t["episode_id"]].get("scenario_family", "unknown") for t in traces])
    features = {
        "aggregate_logistic": np.asarray([fair.aggregate_features(t) for t in traces]),
        "local_only": np.asarray([fair.local_features(t) for t in traces]),
        "graph_features_learned": np.asarray([fair.graph_features(t) for t in traces]),
    }
    names = list(features)
    scores = {name: np.zeros(len(y), dtype=float) for name in names}
    pred = {name: np.zeros(len(y), dtype=bool) for name in names}
    fixed = {name: np.zeros(len(y), dtype=bool) for name in names}
    fold_records = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        fold_records.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "train_groups": int(len(set(groups[train]))), "test_groups": int(len(set(groups[test])))})
        for name, x in features.items():
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed + fold, class_weight=None).fit(x[train], y[train])
            tr = model.predict_proba(x[train])[:, 1]
            te = model.predict_proba(x[test])[:, 1]
            scores[name][test] = te
            pred[name][test] = te >= best_threshold(y[train], tr)
            fixed[name][test] = te >= threshold_at_fpr(y[train], tr)
    result = {name: metric(y, scores[name], pred[name]) for name in names}
    fixed_result = {name: {"fpr": float(fixed[name][y == 0].mean()), "recall": float(fixed[name][y == 1].mean()), "predicted_positive": int(fixed[name].sum())} for name in names}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "independent-mas-learned-baselines-v1", "dataset": {"input": str(args.input), "labels": str(args.labels), "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "groups": int(len(set(groups)))}, "protocol": {"split": "StratifiedGroupKFold by scenario_family", "folds": fold_records, "features_public_only": True, "forbidden_model_fields": ["label", "external_effect", "hidden_fields", "label_source", "scenario_family", "mechanism"]}, "methods": result, "fixed_fpr_5pct": fixed_result}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions_oof.jsonl").open("w", encoding="utf-8") as handle:
        for i, trace in enumerate(traces):
            handle.write(json.dumps({"episode_id": trace["episode_id"], "group": groups[i], "label": int(y[i]), **{name: float(scores[name][i]) for name in names}}, sort_keys=True) + "\n")
    lines = ["# Independent learned MAS baselines v1", "", "Five-fold family-disjoint evaluation on the same public MAS release.", "", f"- n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = result[name]
        auc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        lines.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {auc} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    lines += ["", "## Fixed 5% FPR", "", "| method | empirical FPR | recall |", "|---|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | {fixed_result[name]['fpr']:.3f} | {fixed_result[name]['recall']:.3f} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
