"""Paired topology-family bootstrap for the frozen same-information controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def load(path: Path) -> dict[str, dict]:
    return {
        row["episode_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def metric(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray([
        roc_auc_score(y[idx], p[idx]),
        average_precision_score(y[idx], p[idx]),
        brier_score_loss(y[idx], p[idx]),
    ])


def run(main_path: Path, baseline_path: Path, repeats: int, seed: int) -> dict:
    main, baseline = load(main_path), load(baseline_path)
    ids = sorted(set(main).intersection(baseline))
    if not ids:
        raise ValueError("no shared episode IDs")
    y = np.asarray([int(main[i]["label"]) for i in ids])
    groups = np.asarray([main[i]["group"] for i in ids])
    pred = {
        "two_line_bn": np.asarray([float(main[i]["hierarchical_fusion"]) for i in ids]),
        "logistic": np.asarray([float(baseline[i]["logistic_calibrated"]) for i in ids]),
        "histgb": np.asarray([float(baseline[i]["histgb_calibrated"]) for i in ids]),
    }
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    result = {"n": len(ids), "groups": len(unique_groups), "repeats": repeats, "comparisons": {}}
    for name in ("logistic", "histgb"):
        deltas = []
        for _ in range(repeats):
            draw = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == group) for group in draw])
            deltas.append(metric(y, pred["two_line_bn"], idx) - metric(y, pred[name], idx))
        arr = np.asarray(deltas)
        result["comparisons"][name] = {
            key: {
                "mean": float(np.mean(arr[:, col])),
                "low": float(np.quantile(arr[:, col], 0.025)),
                "high": float(np.quantile(arr[:, col], 0.975)),
            }
            for col, key in enumerate(("auroc", "auprc", "brier"))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("paperAlpha/results/independent_mas_journal_v1/predictions_oof.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("paperAlpha/results/submission/development/strong_baselines_canonical_20261006/predictions_oof.jsonl"))
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.main, args.baseline, args.repeats, args.seed)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
