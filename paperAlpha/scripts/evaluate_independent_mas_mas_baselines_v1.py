"""Strict family-disjoint evaluation of MAS-specific monitor baselines.

These are protocol-compatible proxies for common MAS monitors: independent
per-agent pooling, topology-only reachability, edge-count scoring, and
taint-aware runtime reachability.  They receive only the public event stream
and are evaluated with the same five-fold scenario-family holdout used by the
two-line monitor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    from sklearn.model_selection import GroupKFold as StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]


def load_score_episode():
    path = Path(__file__).with_name("evaluate_mas_baselines_v1.py")
    spec = importlib.util.spec_from_file_location("mas_baselines_proxy", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_episode


def metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray, threshold_note: str) -> dict:
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": threshold_note,
        "positive_predictions": int(pred.sum()),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    values = np.unique(np.r_[0.0, p, 1.0])
    return float(max(values, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target: float = 0.05) -> float:
    negatives = p[y == 0]
    if not len(negatives):
        return 1.0
    return float(np.quantile(negatives, 1.0 - target, method="higher"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/independent_mas_mas_baselines_v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    traces = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r["episode_id"]: r for r in (json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    y = np.asarray([int(labels[t["episode_id"]]["label"]) for t in traces], dtype=int)
    groups = np.asarray([labels[t["episode_id"]].get("scenario_family", "unknown") for t in traces])
    score_episode = load_score_episode()
    rows = [score_episode(t) for t in traces]
    names = ["per_agent_max", "per_agent_mean", "topology_only", "edge_count", "no_taint_contribution", "dynamic_taint_path", "trust_reputation_risk"]
    scores = {name: np.asarray([row[name] for row in rows], dtype=float) for name in names}
    oof_pred = {name: np.zeros(len(y), dtype=bool) for name in names}
    oof_fixed = {name: np.zeros(len(y), dtype=bool) for name in names}
    oof_scores = {name: np.zeros(len(y), dtype=float) for name in names}
    folds = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        folds.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "train_groups": int(len(set(groups[train]))), "test_groups": int(len(set(groups[test])))})
        for name in names:
            tr = scores[name][train].copy()
            te = scores[name][test].copy()
            if name == "edge_count":
                scale = max(float(tr.max()), 1.0)
                tr, te = tr / scale, te / scale
            oof_scores[name][test] = te
            threshold = best_threshold(y[train], tr)
            oof_pred[name][test] = te >= threshold
            oof_fixed[name][test] = te >= threshold_at_fpr(y[train], tr)
    result = {name: metrics(y, oof_scores[name], oof_pred[name], "fold-local train-only") for name in names}
    fixed = {}
    for name in names:
        neg = y == 0
        fixed[name] = {"fpr": float(oof_fixed[name][neg].mean()), "recall": float(oof_fixed[name][y == 1].mean()), "predicted_positive": int(oof_fixed[name].sum())}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "independent-mas-mas-baselines-v1",
        "dataset": {"input": str(args.input), "labels": str(args.labels), "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "groups": int(len(set(groups))), "public_fields_only": True},
        "protocol": {"split": "StratifiedGroupKFold by scenario_family", "folds": folds, "forbidden_model_fields": ["label", "external_effect", "hidden_fields", "label_source", "scenario_family", "mechanism"]},
        "methods": result,
        "fixed_fpr_5pct": fixed,
        "definitions": {"per_agent_max": "max independent local scores", "per_agent_mean": "mean independent local scores", "topology_only": "source-to-privileged-sink reachability ignoring data class", "edge_count": "normalized explicit message/delegation edge count", "no_taint_contribution": "shortest-path graph contribution ignoring provenance", "dynamic_taint_path": "source-to-sink reachability over explicitly untrusted edges", "trust_reputation_risk": "decayed public-evidence reputation proxy aligned with CogTrust's agent-level trust question; not a reproduction"},
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions_oof.jsonl").open("w", encoding="utf-8") as handle:
        for i, trace in enumerate(traces):
            handle.write(json.dumps({"episode_id": trace["episode_id"], "group": groups[i], "label": int(y[i]), **{name: float(oof_scores[name][i]) for name in names}}, sort_keys=True) + "\n")
    lines = ["# Independent MAS-specific baselines v1", "", "Five-fold family-disjoint evaluation on the same public MAS release as the two-line monitor.", "", f"- n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:"]
    for name in names:
        m = result[name]
        auc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        lines.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {auc} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    lines += ["", "## Fixed 5% FPR", "", "| method | empirical FPR | recall |", "|---|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | {fixed[name]['fpr']:.3f} | {fixed[name]['recall']:.3f} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
