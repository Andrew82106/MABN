"""Journal-neighbor control: reliability-weighted Bayesian fusion.

This is a transparent control inspired by the ESWA multi-agent Bayesian-fusion
pattern. It fits smoothed positive/negative likelihoods on each evidence line,
weights each line by train-only separation from the prior, and combines the
line posteriors in log-odds space. It does not use labels from the test fold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/independent_mas_v3/traces_public.jsonl"
DEFAULT_LABELS = ROOT / "results/independent_mas_v3/labels.jsonl"
DEFAULT_OUT = ROOT / "results/submission/final_eval/reliability_bayes_baseline_v1"


def load_main():
    path = Path(__file__).with_name("evaluate_independent_mas_journal_v1.py")
    spec = importlib.util.spec_from_file_location("journal_eval_for_reliability", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def logit(p: float) -> float:
    return math.log(max(1e-6, min(1 - 1e-6, p)) / max(1e-6, 1 - min(1 - 1e-6, p)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    prior = float((y.sum() + 1.0) / (len(y) + 2.0))
    deltas = []
    for j in range(x.shape[1]):
        on = x[:, j] > 0
        p1 = float((y[on].sum() + 1.0) / (on.sum() + 2.0)) if on.any() else prior
        p0 = float((y[~on].sum() + 1.0) / ((~on).sum() + 2.0)) if (~on).any() else prior
        deltas.append(logit(p1) - logit(p0))
    # Reliability is train-only separation; normalize to keep the fusion from
    # over-counting many correlated specialist signals.
    scale = max(1.0, float(np.sum(np.abs(deltas))))
    weights = np.asarray([abs(d) / scale for d in deltas], dtype=float)
    return {"prior": prior, "deltas": np.asarray(deltas, dtype=float), "weights": weights}


def score_line(x: np.ndarray, model: dict) -> np.ndarray:
    evidence = (x > 0).astype(float) * model["deltas"]
    return sigmoid(logit(model["prior"]) + evidence.sum(axis=1))


def fit_fusion(xw: np.ndarray, xr: np.ndarray, y: np.ndarray) -> dict:
    wm = fit_line(xw, y); rm = fit_line(xr, y)
    # Reliability of each line is the train-only total evidence separation.
    wr = float(np.sum(np.abs(wm["deltas"])))
    rr = float(np.sum(np.abs(rm["deltas"])))
    total = max(1.0, wr + rr)
    return {"workflow": wm, "runtime": rm, "workflow_reliability": wr / total, "runtime_reliability": rr / total}


def score_fusion(xw: np.ndarray, xr: np.ndarray, model: dict) -> np.ndarray:
    pw = score_line(xw, model["workflow"]); pr = score_line(xr, model["runtime"])
    prior = (model["workflow"]["prior"] + model["runtime"]["prior"]) / 2.0
    lw = logit(prior)
    return sigmoid(lw + model["workflow_reliability"] * (np.asarray([logit(p) for p in pw]) - lw) + model["runtime_reliability"] * (np.asarray([logit(p) for p in pr]) - lw))


def best_threshold(y, p):
    vals = np.unique(np.r_[0.0, p, 1.0])
    return float(max(vals, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def metrics(y, p, threshold):
    pred = p >= threshold
    return {"n": int(len(y)), "positives": int(y.sum()), "f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "threshold": float(threshold)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=DEFAULT_INPUT); ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS); ap.add_argument("--out", type=Path, default=DEFAULT_OUT); args = ap.parse_args()
    main_eval = load_main()
    traces = main_eval.read_jsonl(args.input); labels = {x["episode_id"]: x for x in main_eval.read_jsonl(args.labels)}
    y = np.asarray([int(labels[t["episode_id"]]["label"]) for t in traces], dtype=int)
    groups = np.asarray([labels[t["episode_id"]].get("scenario_family", "unknown") for t in traces])
    xw = np.asarray([main_eval.workflow_vector(t) for t in traces], dtype=float)
    xr = np.asarray([main_eval.runtime_vector(t) for t in traces], dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261004)
    oof = np.zeros(len(y)); fold_thresholds = []; fold_records = []
    for fold, (train, test) in enumerate(splitter.split(xw, y, groups), 1):
        model = fit_fusion(xw[train], xr[train], y[train])
        p_train = score_fusion(xw[train], xr[train], model); p_test = score_fusion(xw[test], xr[test], model)
        oof[test] = p_test; threshold = best_threshold(y[train], p_train); fold_thresholds.append(threshold)
        fold_records.append({"fold": fold, "train": int(len(train)), "test": int(len(test)), "workflow_reliability": model["workflow_reliability"], "runtime_reliability": model["runtime_reliability"], "threshold": threshold})
    threshold = float(np.mean(fold_thresholds)); result = metrics(y, oof, threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"method": "reliability_weighted_bayesian_fusion", "dataset": {"n": int(len(y)), "positive": int(y.sum()), "groups": int(len(set(groups)))}, "protocol": "five-fold StratifiedGroupKFold by scenario_family; all likelihoods and line reliabilities fit on train folds only", "result": result, "folds": fold_records, "interpretation": "Transparent journal-neighbor control inspired by reliability-weighted Bayesian fusion; not a reproduction of the ATC task or its specialist prompts."}
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions_oof.jsonl").open("w", encoding="utf-8") as h:
        for t, yy, pp, gg in zip(traces, y, oof, groups): h.write(json.dumps({"episode_id": t["episode_id"], "group": gg, "label": int(yy), "score": float(pp)}, sort_keys=True) + "\n")
    (args.out / "REPORT.md").write_text("\n".join(["# Reliability-weighted Bayesian fusion baseline", "", f"- n={len(y)}, positives={int(y.sum())}, scenario families={len(set(groups))}", f"- AUROC/AUPRC/F1/Brier: {result['auroc']:.3f}/{result['auprc']:.3f}/{result['f1']:.3f}/{result['brier']:.3f}", "- All per-feature likelihoods and line reliabilities are fit on each training fold only.", "- This is a transparent ESWA-neighbor control, not a direct task reproduction.", ""]), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
