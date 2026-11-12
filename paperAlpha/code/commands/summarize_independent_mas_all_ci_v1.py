"""Cluster-bootstrap intervals for every method in the strict MAS table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ece(y: np.ndarray, p: np.ndarray) -> float:
    out = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def one(y: np.ndarray, p: np.ndarray) -> dict:
    threshold = 0.5
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None, "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "f1_at_0_5": float(f1_score(y, p >= threshold, zero_division=0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/independent_mas_journal_v1/predictions_oof.jsonl"))
    ap.add_argument("--learned", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/independent_mas_learned_baselines_v1/predictions_oof.jsonl"))
    ap.add_argument("--rules", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/independent_mas_mas_baselines_v1/predictions_oof.jsonl"))
    ap.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/submission_audit/runs/final_eval/all_methods_ci_v1"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    rows = {r["episode_id"]: dict(r) for r in read(args.journal)}
    for path in (args.learned, args.rules):
        for row in read(path):
            if row["episode_id"] not in rows or row["label"] != rows[row["episode_id"]]["label"]:
                raise ValueError(f"episode mismatch: {row['episode_id']}")
            rows[row["episode_id"]].update(row)
    ids = sorted(rows)
    groups = np.asarray([rows[i]["group"] for i in ids])
    y = np.asarray([int(rows[i]["label"]) for i in ids])
    methods = [k for k in rows[ids[0]] if k not in {"episode_id", "group", "label"}]
    group_values = np.asarray(sorted(set(groups)))
    rng = np.random.default_rng(args.seed)
    out = {}
    for method in methods:
        p = np.asarray([float(rows[i][method]) for i in ids])
        draws = {metric: [] for metric in ("auroc", "auprc", "brier", "ece", "f1_at_0_5")}
        for _ in range(args.n):
            chosen = rng.choice(group_values, size=len(group_values), replace=True)
            ix = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
            m = one(y[ix], p[ix])
            for metric, value in m.items():
                if value is not None:
                    draws[metric].append(value)
        point = one(y, p)
        out[method] = {"point": point, "cluster_bootstrap_95": {metric: ({"low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)), "n": len(values)} if values else None) for metric, values in draws.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "independent-mas-all-method-ci-v1", "n": len(ids), "groups": len(group_values), "bootstrap": args.n, "seed": args.seed, "methods": out, "scope": "Family-cluster bootstrap over the frozen strict OOF predictions; thresholded F1 is shown at 0.5 only and is not the fold-local F1 from the main report."}
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Independent MAS all-method cluster bootstrap", "", f"n={len(ids)}, groups={len(group_values)}, bootstrap={args.n}", "", "| method | AUROC | AUPRC | Brier | ECE | F1@0.5 |", "|---|---:|---:|---:|---:|---:|"]
    for method, value in out.items():
        point = value["point"]; ci = value["cluster_bootstrap_95"]
        fmt = lambda metric: ("n/a" if ci[metric] is None else f"{point[metric]:.3f} [{ci[metric]['low']:.3f},{ci[metric]['high']:.3f}]")
        lines.append(f"| {method} | {fmt('auroc')} | {fmt('auprc')} | {fmt('brier')} | {fmt('ece')} | {fmt('f1_at_0_5')} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["point"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
