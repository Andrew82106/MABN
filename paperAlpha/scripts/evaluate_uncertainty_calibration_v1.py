"""Uncertainty and calibration audit for the learnable MAS graph monitor.

This script evaluates probabilistic quality separately from thresholded F1.
Each random split fits the graph model on the training partition only, fits a
temperature on training logits only, and then reports test-set Brier/ECE and
discrimination metrics. Bootstrap intervals are conditional on each held-out
split; the across-seed summary measures split sensitivity rather than a
population confidence interval.

The compositional benchmark is a controlled mechanism test (reachability
labels), so this report must not be interpreted as deployment calibration or
causal evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss)
from sklearn.model_selection import train_test_split

# Make the sibling evaluator importable when this file is run from the repo
# root.  It contains the canonical model implementation and no side effects.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_learnable_graph_v1 import fit, forward  # noqa: E402
from evaluate_mas_baselines_v1 import score_episode  # noqa: E402


def read_eps(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def ece_score(y, p, n_bins=10):
    """Expected calibration error with fixed, equal-width probability bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi if b == n_bins - 1 else p < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def reliability_bins(y, p, n_bins=10):
    """Return auditable per-bin counts, confidence and empirical accuracy."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi if b == n_bins - 1 else p < hi)
        rows.append({
            "bin": b, "lower": float(lo), "upper": float(hi),
            "n": int(mask.sum()),
            "mean_probability": None if not np.any(mask) else float(p[mask].mean()),
            "empirical_rate": None if not np.any(mask) else float(y[mask].mean()),
        })
    return rows


def best_threshold(y, p):
    vals = np.unique(np.r_[0.0, p, 1.0])
    scores = [f1_score(y, p >= t, zero_division=0) for t in vals]
    # Deterministic tie break: prefer the more conservative (higher) cutoff.
    best = max(zip(scores, vals), key=lambda x: (x[0], x[1]))
    return float(best[1])


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7), labels=[0, 1])),
        "ece_10": ece_score(y, p, 10),
        "mce_10": max((abs(r["mean_probability"] - r["empirical_rate"])
                        for r in reliability_bins(y, p, 10) if r["n"]), default=0.0),
        "threshold": float(threshold),
        "predicted_positive": int(pred.sum()),
    }


def fit_temperature(y, p):
    """Fit scalar T on training logits only, minimizing NLL."""
    y = np.asarray(y, dtype=float)
    z = logit(p)

    def objective(log_t):
        t = float(np.exp(log_t))
        q = sigmoid(z / t)
        return float(log_loss(y, np.clip(q, 1e-7, 1.0 - 1e-7), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded",
                             options={"xatol": 1e-5, "maxiter": 200})
    return float(np.exp(result.x)), bool(result.success), float(result.fun)


def bootstrap_ci(y, p, threshold, seed, n_boot=2000):
    """Percentile 95% CIs for test metrics; preserves paired y/p samples."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {k: [] for k in ("f1", "auroc", "auprc", "brier", "ece_10")}
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        yy, pp = y[ii], p[ii]
        pred = pp >= threshold
        vals["f1"].append(float(f1_score(yy, pred, zero_division=0)))
        vals["auprc"].append(float(average_precision_score(yy, pp)))
        vals["brier"].append(float(brier_score_loss(yy, pp)))
        vals["ece_10"].append(float(ece_score(yy, pp, 10)))
        vals["auroc"].append(float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan)
    out = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        out[key] = {"low": float(np.quantile(a, .025)),
                    "high": float(np.quantile(a, .975)),
                    "n_boot": int(len(a))}
    return out


def run_one_seed(eps, y, seed, test_size, n_boot):
    idx = np.arange(len(eps))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                               stratify=y)
    fitted = fit([eps[i] for i in tr], y[tr], seed)
    theta = fitted.x
    train_p = np.asarray([forward(theta, eps[i]) for i in tr], dtype=float)
    test_p = np.asarray([forward(theta, eps[i]) for i in te], dtype=float)
    threshold = best_threshold(y[tr], train_p)
    temp, temp_ok, temp_train_nll = fit_temperature(y[tr], train_p)
    test_pt = sigmoid(logit(test_p) / temp)
    temp_threshold = best_threshold(y[tr], sigmoid(logit(train_p) / temp))
    raw = classification_metrics(y[te], test_p, threshold)
    calibrated = classification_metrics(y[te], test_pt, temp_threshold)
    baselines = {}
    baseline_rows = [score_episode(eps[i]) for i in range(len(eps))]
    for name in ("per_agent_max", "per_agent_mean", "topology_only",
                 "edge_count", "no_taint_contribution", "dynamic_taint_path"):
        train_s = np.asarray([baseline_rows[i][name] for i in tr], dtype=float)
        test_s = np.asarray([baseline_rows[i][name] for i in te], dtype=float)
        if name == "edge_count":
            scale = max(float(train_s.max()), 1.0)
            train_s, test_s = train_s / scale, test_s / scale
        bt = best_threshold(y[tr], train_s)
        baselines[name] = classification_metrics(y[te], test_s, bt)
    return {
        "seed": int(seed), "train": int(len(tr)), "test": int(len(te)),
        "temperature": {"value": temp, "fit_success": temp_ok,
                         "train_nll": temp_train_nll},
        "raw": raw,
        "temperature_scaled": calibrated,
        "raw_bootstrap_95ci": bootstrap_ci(y[te], test_p, threshold, seed + 100000, n_boot),
        "temperature_scaled_bootstrap_95ci": bootstrap_ci(
            y[te], test_pt, temp_threshold, seed + 200000, n_boot),
        "baselines": baselines,
        "reliability_raw": reliability_bins(y[te], test_p, 10),
        "reliability_temperature_scaled": reliability_bins(y[te], test_pt, 10),
        "parameters": [float(x) for x in theta],
    }


def summarize(runs, variant):
    keys = ("f1", "precision", "recall", "auroc", "auprc", "brier",
            "log_loss", "ece_10", "mce_10")
    out = {}
    for key in keys:
        a = np.asarray([r[variant][key] for r in runs if r[variant][key] is not None], dtype=float)
        out[key] = {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
                    "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("paperAlpha/results/compositional_mas_v1/episodes.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("paperAlpha/results/uncertainty_calibration_v1"))
    ap.add_argument("--test-size", type=float, default=.30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260914, 20260915, 20260916, 20260917, 20260918])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    eps = read_eps(args.input)
    y = np.asarray([int(e["label"]) for e in eps], dtype=int)
    runs = [run_one_seed(eps, y, seed, args.test_size, args.bootstrap) for seed in args.seeds]
    summary = {"raw": summarize(runs, "raw"),
               "temperature_scaled": summarize(runs, "temperature_scaled")}
    # Aggregate calibration comparison for protocol-compatible MAS baselines.
    baseline_summary = {}
    for name in runs[0]["baselines"]:
        rows = [r["baselines"][name] for r in runs]
        baseline_summary[name] = {k: {"mean": float(np.mean([x[k] for x in rows if x[k] is not None])),
                                     "std": float(np.std([x[k] for x in rows if x[k] is not None], ddof=1) if len(rows) > 1 else 0.0)}
                                  for k in ("f1", "auroc", "auprc", "brier", "ece_10")}
    report = {
        "model": "LearnableNoisyORGraph v1",
        "dataset": {"input": str(args.input), "n": int(len(eps)), "positive": int(y.sum()),
                    "negative": int(len(y) - y.sum()),
                    "label_source": "independent structural reachability rule; controlled synthetic mechanism benchmark"},
        "protocol": {"split": "stratified random split; fit and temperature use train only",
                     "test_size": args.test_size, "seeds": args.seeds,
                     "bootstrap_resamples": args.bootstrap,
                     "ece": "10 equal-width probability bins",
                     "bootstrap_ci": "percentile 95% CI conditional on each held-out split"},
        "across_seed_summary": summary,
        "mas_baseline_summary": baseline_summary,
        "runs": runs,
        "caveats": [
            "This benchmark tests a designed reachability mechanism, not all MAS safety risks.",
            "Bootstrap intervals quantify test-sample uncertainty conditional on each split; seed std quantifies split sensitivity.",
            "Temperature scaling changes probability calibration only; it does not add new detection evidence.",
            "No causal-discovery claim is made; explicit edges are observed protocol relations."
        ]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md = ["# Uncertainty and Calibration Audit v1", "",
          "本报告独立评估风险概率的可靠性，不把 F1 当作概率质量。所有模型参数、阈值和温度均只在训练折拟合。", "",
          "## Across-seed summary (mean ± std)", "",
          "| variant | F1 | AUROC | AUPRC | Brier ↓ | ECE-10 ↓ | log loss ↓ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, label in (("raw", "Raw"), ("temperature_scaled", "Temperature-scaled")):
        s = summary[variant]
        def fmt(k): return f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
        md.append(f"| {label} | {fmt('f1')} | {fmt('auroc')} | {fmt('auprc')} | {fmt('brier')} | {fmt('ece_10')} | {fmt('log_loss')} |")
    md += ["", "## MAS baseline calibration comparison", "",
           "| baseline | F1 | AUROC | AUPRC | Brier ↓ | ECE-10 ↓ |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in baseline_summary.items():
        md.append(f"| {name} | {s['f1']['mean']:.3f} ± {s['f1']['std']:.3f} | {s['auroc']['mean']:.3f} ± {s['auroc']['std']:.3f} | {s['auprc']['mean']:.3f} ± {s['auprc']['std']:.3f} | {s['brier']['mean']:.3f} ± {s['brier']['std']:.3f} | {s['ece_10']['mean']:.3f} ± {s['ece_10']['std']:.3f} |")
    md += ["", "## Interpretation", "",
           "温度缩放只校准置信度，不改变排序证据；若 Brier/ECE 改善而 F1 基本不变，说明改进来自概率可靠性而非检测能力。",
           "bootstrap 区间和跨随机种子方差应同时报告。该数据是受控合成机制验证集，不能据此宣称真实部署校准或 SOTA。", "",
           "## Artifacts", "- `metrics.json` contains every seed, bootstrap interval and reliability bin.", "- Raw predictions are regenerated from the canonical model and input; no terminal risk field is used."]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
