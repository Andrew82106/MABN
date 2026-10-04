"""Cluster-bootstrap comparison of the primary fusion and journal-neighbor control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results/independent_mas_journal_v1/predictions_oof.jsonl"
CONTROL = ROOT / "results/submission/final_eval/reliability_bayes_baseline_v1/predictions_oof.jsonl"
OUT = ROOT / "results/submission/final_eval/primary_vs_reliability_v1"


def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    p = {x["episode_id"]: x for x in read(PRIMARY)}
    c = {x["episode_id"]: x for x in read(CONTROL)}
    ids = sorted(set(p) & set(c))
    y = np.asarray([int(p[i]["label"]) for i in ids])
    pp = np.asarray([float(p[i]["hierarchical_fusion"]) for i in ids])
    pc = np.asarray([float(c[i]["score"]) for i in ids])
    groups = np.asarray([p[i]["group"] for i in ids])
    unique = np.asarray(sorted(set(groups)))
    rng = np.random.default_rng(20261004)
    auc_d, pr_d = [], []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        auc_d.append(float(roc_auc_score(y[idx], pp[idx]) - roc_auc_score(y[idx], pc[idx])))
        pr_d.append(float(average_precision_score(y[idx], pp[idx]) - average_precision_score(y[idx], pc[idx])))
    def summary(values):
        arr = np.asarray(values)
        return {"mean": float(arr.mean()), "low": float(np.quantile(arr, .025)), "high": float(np.quantile(arr, .975)), "probability_positive": float(np.mean(arr > 0)), "n_boot": int(len(arr))}
    result = {"n": int(len(ids)), "positive": int(y.sum()), "groups": int(len(unique)), "primary": {"auroc": float(roc_auc_score(y, pp)), "auprc": float(average_precision_score(y, pp))}, "reliability_bayes": {"auroc": float(roc_auc_score(y, pc)), "auprc": float(average_precision_score(y, pc))}, "delta_primary_minus_control": {"auroc": summary(auc_d), "auprc": summary(pr_d)}, "protocol": "paired family-cluster bootstrap over shared OOF episode IDs; 2000 resamples; positive deltas favor the proposed fusion"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# Primary vs reliability-weighted Bayesian control", "", f"- Shared OOF episodes: {len(ids)}; positives: {int(y.sum())}; topology families: {len(unique)}", f"- Primary AUROC/AUPRC: {result['primary']['auroc']:.3f}/{result['primary']['auprc']:.3f}", f"- Control AUROC/AUPRC: {result['reliability_bayes']['auroc']:.3f}/{result['reliability_bayes']['auprc']:.3f}", f"- Cluster-bootstrap ΔAUROC: {result['delta_primary_minus_control']['auroc']['mean']:.3f} [{result['delta_primary_minus_control']['auroc']['low']:.3f}, {result['delta_primary_minus_control']['auroc']['high']:.3f}]; P(Δ>0)={result['delta_primary_minus_control']['auroc']['probability_positive']:.3f}", f"- Cluster-bootstrap ΔAUPRC: {result['delta_primary_minus_control']['auprc']['mean']:.3f} [{result['delta_primary_minus_control']['auprc']['low']:.3f}, {result['delta_primary_minus_control']['auprc']['high']:.3f}]; P(Δ>0)={result['delta_primary_minus_control']['auprc']['probability_positive']:.3f}", "", "Bootstrap samples topology families, retaining all episodes within each sampled family."]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
