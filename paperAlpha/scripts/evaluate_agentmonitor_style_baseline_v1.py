"""Evaluate an AgentMonitor-style MAS statistics baseline.

The original AgentMonitor predicts team performance from per-agent input/output
statistics and graph attributes.  This adapter preserves that observable
feature family but changes only the supervised target to this paper's declared
system-risk label.  It is therefore an explicitly named, protocol-compatible
adaptation, not a claim that the original paper solved risk forecasting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRIV = {"payments.write", "filesystem.write", "shell.exec"}


def _stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * 5
    x = np.asarray(values, dtype=float)
    return [float(x.sum()), float(x.mean()), float(x.max()), float(x.std()), float(np.median(x))]


def agentmonitor_features(trace: dict) -> np.ndarray:
    """Counts and degree summaries corresponding to AgentMonitor's statistics."""
    agents = sorted(set(trace.get("agents", [])))
    events = trace.get("events", [])
    per: dict[str, list[float]] = {
        a: [0.0] * 10 for a in agents
    }
    edges: set[tuple[str, str]] = set()
    for event in events:
        actor = event.get("actor")
        target = event.get("target")
        if actor not in per:
            per[actor] = [0.0] * 10
        row = per[actor]
        kind = event.get("event_type")
        if kind == "input_received":
            row[0] += 1
        if kind == "message_sent":
            row[1] += 1
            if target:
                edges.add((actor, target))
        if kind == "delegation":
            row[2] += 1
            if target:
                edges.add((actor, target))
        if kind == "tool_call":
            row[3] += 1
            if event.get("tool") in PRIV:
                row[4] += 1
        if kind in {"message_sent", "delegation"} and event.get("data_class") == "untrusted":
            row[5] += 1
        if kind in {"message_sent", "delegation"} and event.get("data_class") == "unknown":
            row[6] += 1
        if kind in {"message_sent", "delegation"} and event.get("edge_kind") == "unexpected":
            row[7] += 1
        if target in per and kind in {"message_sent", "delegation"}:
            per[target][8] += 1
        if target in per and kind in {"message_sent", "delegation"} and event.get("data_class") == "untrusted":
            per[target][9] += 1
    values = list(per.values())
    features: list[float] = []
    for column in range(10):
        features.extend(_stats([row[column] for row in values]))
    indegree = [sum(v == a for _, v in edges) for a in per]
    outdegree = [sum(u == a for u, _ in edges) for a in per]
    n = max(1, len(per))
    m = len(edges)
    features.extend([
        float(len(events)), float(len(per)), float(m),
        float(m / max(1, n * (n - 1))), float(sum(indegree)),
        float(max(indegree, default=0)), float(np.mean(indegree) if indegree else 0.0),
        float(max(outdegree, default=0)), float(np.mean(outdegree) if outdegree else 0.0),
        float(sum(event.get("event_type") == "tool_call" for event in events)),
    ])
    return np.asarray(features, dtype=float)


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_predictions": int(pred.sum()),
    }


def metric_with_predictions(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_predictions": int(pred.sum()),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, p, 1.0])
    return float(max(candidates, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/submission/development/agentmonitor_style_baseline_20261005")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args(argv)
    traces = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {json.loads(line)["episode_id"]: json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()}
    y = np.asarray([int(labels[row["episode_id"]]["label"]) for row in traces], dtype=int)
    groups = np.asarray([labels[row["episode_id"]].get("scenario_family", "unknown") for row in traces])
    x = np.asarray([agentmonitor_features(row) for row in traces])
    scores = np.zeros(len(y), dtype=float)
    preds = np.zeros(len(y), dtype=bool)
    folds = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed + fold).fit(scaler.transform(x[train]), y[train])
        p_train = model.predict_proba(scaler.transform(x[train]))[:, 1]
        p_test = model.predict_proba(scaler.transform(x[test]))[:, 1]
        threshold = best_threshold(y[train], p_train)
        scores[test] = p_test
        preds[test] = p_test >= threshold
        folds.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "train_groups": int(len(set(groups[train]))), "test_groups": int(len(set(groups[test]))), "threshold": threshold})
    result = metric_with_predictions(y, scores, preds)
    result["threshold_policy"] = "fold-local train-only F1 threshold"
    result["fixed_threshold_0_5_metrics"] = metric(y, scores, 0.5)
    fixed = {"fpr": float((scores[y == 0] >= 0.5).mean()), "recall": float((scores[y == 1] >= 0.5).mean())}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "agentmonitor-style-mas-risk-baseline-v1",
        "dataset": {"n": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "groups": int(len(set(groups))), "input": str(args.input), "labels": str(args.labels)},
        "protocol": {"split": "StratifiedGroupKFold by scenario_family", "folds": folds, "features_public_only": True, "original_method_target": "team task performance and response editing", "adapted_target": "declared system-risk label", "feature_count": int(x.shape[1])},
        "metrics": result,
        "fixed_threshold_0_5": fixed,
        "limitations": ["The original AgentMonitor is not a risk-probability model; this is a transparent statistics-only adaptation.", "Text semantic content is absent from independent_mas_v3, so this is a conservative count/graph projection of its feature family."],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions_oof.jsonl").open("w", encoding="utf-8") as handle:
        for row, p in zip(traces, scores):
            handle.write(json.dumps({"episode_id": row["episode_id"], "group": labels[row["episode_id"]].get("scenario_family"), "label": int(labels[row["episode_id"]]["label"]), "agentmonitor_style": float(p)}, sort_keys=True) + "\n")
    lines = ["# AgentMonitor-style MAS risk baseline", "", "Strict five-fold family-disjoint evaluation. The original AgentMonitor target is team performance; this run adapts its observable per-agent statistics to the declared risk target.", "", f"- n={len(y)}, positives={int(y.sum())}, groups={len(set(groups))}, feature_count={x.shape[1]}", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| AgentMonitor-style statistics + logistic | {result['f1']:.3f} | {result['precision']:.3f} | {result['recall']:.3f} | {result['auroc']:.3f} | {result['auprc']:.3f} | {result['brier']:.3f} |")
    lines += ["", "This is a protocol-compatible adaptation, not a claim that the original AgentMonitor directly predicts this risk label.", f"Fixed threshold 0.5: FPR={fixed['fpr']:.3f}, recall={fixed['recall']:.3f}."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result, "fixed_threshold_0_5": fixed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
