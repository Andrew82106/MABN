"""Paired family-cluster bootstrap for the frozen OOF primary comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["episode_id"]): row for row in rows}


def score(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return float(roc_auc_score(y, p))
    return float(average_precision_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["graph_features_learned", "trust_reputation_risk"])
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    primary = read(args.primary)
    baselines = read(args.baselines)
    ids = sorted(set(primary) & set(baselines))
    if len(ids) != len(primary) or len(ids) != len(baselines):
        raise ValueError("prediction files do not contain the same episode IDs")
    y = np.asarray([int(primary[i]["label"]) for i in ids], dtype=int)
    groups = np.asarray([str(primary[i]["group"]) for i in ids])
    unique_groups = np.unique(groups)
    group_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {"n": len(ids), "groups": len(unique_groups), "replicates": args.replicates, "seed": args.seed, "comparisons": {}}

    for method in args.methods:
        if method not in baselines[ids[0]]:
            raise ValueError(f"missing baseline method: {method}")
        p_ours = np.asarray([float(primary[i]["hierarchical_fusion"]) for i in ids])
        p_base = np.asarray([float(baselines[i][method]) for i in ids])
        observed = {metric: score(y, p_ours, metric) - score(y, p_base, metric) for metric in ("auroc", "auprc")}
        draws = {metric: [] for metric in observed}
        for _ in range(args.replicates):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_rows[g] for g in sampled])
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            for metric in draws:
                draws[metric].append(score(yb, p_ours[idx], metric) - score(yb, p_base[idx], metric))
        summary = {"observed_delta": observed, "bootstrap": {}}
        for metric, values in draws.items():
            arr = np.asarray(values, dtype=float)
            summary["bootstrap"][metric] = {"n": int(len(arr)), "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))], "positive_fraction": float(np.mean(arr > 0.0))}
        output["comparisons"][method] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
