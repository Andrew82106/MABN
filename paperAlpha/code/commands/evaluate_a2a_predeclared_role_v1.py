"""Strict A2ASecBench API audit using predeclared attack/control roles.

Unlike the older adapter score, this audit never uses ``observed_violation``
or outcome metrics.  Features are the event-derived public flags and event
count only; the target is the case role fixed before execution.  It is still
an external benchmark/adapter check rather than human adjudication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["event_count", "anomaly_path", "cross_agent_propagation", "memory_poison", "permission_mismatch", "risk_accumulation", "tool_side_effect", "untrusted_input"]


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def vector(row):
    flags = row.get("features") or {}
    return np.asarray([min(float(row.get("event_count", 0)) / 20.0, 1.0)] + [float(bool(flags.get(k, 0))) for k in FEATURES[1:]], dtype=float)


def cpt_fit(x, y):
    prior = float((y.sum() + 1) / (len(y) + 2))
    probs = []
    for j in range(x.shape[1]):
        active = x[:, j] > 0
        probs.append(float((y[active].sum() + 1) / (active.sum() + 2)) if active.any() else prior)
    return prior, np.asarray(probs)


def bn_score(x, prior, probs):
    out = []
    for row in x:
        q = 1.0
        active = False
        for value, p in zip(row, probs):
            if value > 0:
                active = True; q *= 1.0 - min(.995, max(.001, p))
        out.append(1.0 - q if active else prior)
    return np.asarray(out)


def metric(y, p, threshold=.5):
    pred = p >= threshold
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None, "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "positive_predictions": int(pred.sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=ROOT / 'data/processed/mas_adapter/a2asecbench_api_final_20261002/episodes.jsonl'); ap.add_argument("--out", type=Path, default=ROOT / 'experiments/a2asecbench/runs/submission__a2asecbench_api_final_20261002/strict_role_audit_v1'); args = ap.parse_args()
    rows = read(args.input)
    rows = [r for r in rows if r.get("scenario_role") in {"attack", "control"}]
    x = np.asarray([vector(r) for r in rows]); y = np.asarray([int(r["scenario_role"] == "attack") for r in rows]); families = sorted({str(r.get("attack_id", "unknown")) for r in rows})
    scores = {k: np.zeros(len(rows)) for k in ("layer1_bn", "flat_logistic", "two_line")}; folds = {}
    for fam in families:
        train = np.asarray([i for i, r in enumerate(rows) if str(r.get("attack_id")) != fam]); test = np.asarray([i for i, r in enumerate(rows) if str(r.get("attack_id")) == fam])
        prior, probs = cpt_fit(x[train], y[train]); p1 = bn_score(x[test], prior, probs)
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=0).fit(x[train], y[train]); pl = clf.predict_proba(x[test])[:, 1]
        # The fusion weight is selected on training MSE only; no test labels
        # enter the choice.
        p1tr = bn_score(x[train], prior, probs); pltr = clf.predict_proba(x[train])[:, 1]
        numerator = float(np.dot(p1tr - pltr, y[train] - pltr))
        denominator = max(1e-9, float(np.dot(p1tr - pltr, p1tr - pltr)))
        alpha = max(0.0, min(1.0, numerator / denominator))
        pf = alpha * p1 + (1 - alpha) * pl
        scores["layer1_bn"][test], scores["flat_logistic"][test], scores["two_line"][test] = p1, pl, pf
        folds[fam] = {"train": int(len(train)), "test": int(len(test)), "alpha": alpha, "metrics": {"layer1_bn": metric(y[test], p1), "flat_logistic": metric(y[test], pl), "two_line": metric(y[test], pf)}}
    aggregate = {k: metric(y, v) for k, v in scores.items()}
    result = {"protocol": {"split": "leave-one-attack-family-out", "target": "predeclared scenario_role (attack/control)", "features": "event-derived public flags plus event count only", "excluded": ["observed_violation", "outcome_flags.metrics", "attack_id", "scenario_role", "risk_label"], "interpretation": "External benchmark audit; role labels are fixed by the benchmark case design, not human adjudication."}, "n": len(rows), "positive": int(y.sum()), "families": families, "metrics": aggregate, "folds": folds}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "REPORT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as h:
        for i, r in enumerate(rows): h.write(json.dumps({"episode_id": r["episode_id"], "family": r.get("attack_id"), "role": r["scenario_role"], "label": int(y[i]), **{k: float(v[i]) for k, v in scores.items()}}, sort_keys=True) + "\n")
    lines = ["# A2ASecBench API strict predeclared-role audit v1", "", f"Leave-one-attack-family-out over {len(rows)} episodes; target is the role fixed before execution, not observed_violation.", "", "| method | F1 | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|"]
    for k, m in aggregate.items(): lines.append(f"| {k} | {m['f1']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    lines += ["", "This is a stronger leakage-controlled API benchmark audit, but the role labels are still benchmark-design labels and the adapter is not a human-adjudicated real-world cohort.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8"); print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
