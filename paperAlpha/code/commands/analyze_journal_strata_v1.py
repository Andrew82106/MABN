"""Descriptive missingness/observability strata for the frozen journal OOF."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def metric(y, p):
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() and (1-y).sum() else None,
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    result = root / 'experiments/synthetic_mas/runs/independent_mas_journal_v1'
    pred = {r["episode_id"]: r for r in read_jsonl(result / "predictions_oof.jsonl")}
    traces = read_jsonl(root / 'experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl')
    rows = []
    for trace in traces:
        row = pred[trace["episode_id"]]
        edges = [e for e in trace.get("events", []) if e.get("event_type") in {"message_sent", "delegation"} and e.get("target")]
        unknown = sum(e.get("data_class") in {"unknown", "redacted"} for e in edges)
        unexpected = sum(e.get("edge_kind") == "unexpected" for e in edges)
        workflow = trace.get("workflow_dag", {}).get("edges", [])
        observed = {(e.get("actor"), e.get("target")) for e in edges}
        missing = len(set(map(tuple, workflow)) - observed) / max(1, len(workflow))
        rows.append({"label": int(row["label"]), "score": float(row["hierarchical_fusion"]), "unknown_ratio": unknown / max(1, len(edges)), "unexpected_ratio": unexpected / max(1, len(edges)), "missing_ratio": float(missing)})
    out = {}
    bins = {
        "no_unknown": lambda r: r["unknown_ratio"] == 0,
        "some_unknown": lambda r: 0 < r["unknown_ratio"] <= 0.25,
        "high_unknown": lambda r: r["unknown_ratio"] > 0.25,
        "low_missing": lambda r: r["missing_ratio"] <= 0.5,
        "high_missing": lambda r: r["missing_ratio"] > 0.5,
        "no_unexpected": lambda r: r["unexpected_ratio"] == 0,
        "some_unexpected": lambda r: r["unexpected_ratio"] > 0,
    }
    for name, predicate in bins.items():
        selected = [r for r in rows if predicate(r)]
        y = np.asarray([r["label"] for r in selected], dtype=int); p = np.asarray([r["score"] for r in selected], dtype=float)
        out[name] = metric(y, p)
    payload = {"source": str(result / "predictions_oof.jsonl"), "note": "descriptive OOF strata; no threshold or retraining is selected by these bins", "strata": out}
    (result / "strata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
