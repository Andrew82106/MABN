"""Pre-action/prefix sanity evaluation for independent MAS v3.

This is an early-warning ablation, not the headline full-trace monitor.  It
truncates the public event stream before a selected event fraction (or before
the first privileged tool call) and fits the same transparent two-line
workflow/runtime feature head with a strict family-group split.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PRIV = {"payments.write", "filesystem.write", "shell.exec"}


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def reach(source: set[str], edges: set[tuple[str, str]], sink: set[str]):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    seen = set(source); q = list(source); distance = None
    depth = {u: 0 for u in source}
    while q:
        u = q.pop(0)
        if u in sink:
            distance = depth[u] if distance is None else min(distance, depth[u])
        for v in adj[u]:
            if v not in seen:
                seen.add(v); depth[v] = depth[u] + 1; q.append(v)
    return float(bool(sink & seen)), (1.0 / (1.0 + distance) if distance is not None else 0.0)


def truncate(ep: dict, mode: str) -> dict:
    events = sorted(ep.get("events", []), key=lambda e: e.get("sequence", 0))
    if mode == "pre_action":
        privileged = [e.get("sequence", 0) for e in events if e.get("event_type") == "tool_call" and e.get("tool") in PRIV]
        cut = min(privileged) if privileged else len(events)
    else:
        fraction = float(mode)
        cut = int(np.floor(len(events) * fraction))
    return {**ep, "events": [e for e in events if e.get("sequence", 0) < cut]}


def features(ep: dict) -> np.ndarray:
    events = ep.get("events", [])
    workflow = [tuple(x) for x in ep.get("workflow_dag", {}).get("edges", [])]
    wf = set(workflow)
    observed = [e for e in events if e.get("event_type") in {"message_sent", "delegation"} and e.get("target")]
    observed_edges = {(e.get("actor"), e.get("target")) for e in observed}
    source = {e.get("actor") for e in events if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted"}
    actual_sink = {e.get("actor") for e in events if e.get("event_type") == "tool_call" and e.get("tool") in PRIV}
    nodes = list(ep.get("workflow_dag", {}).get("nodes", ep.get("agents", [])))
    sink = actual_sink or ({nodes[-1]} if nodes else set())
    wf_path, wf_dist = reach(source, wf, sink)
    taint_edges = {(e.get("actor"), e.get("target")) for e in observed if e.get("data_class") == "untrusted"}
    rt_path, rt_dist = reach(source, taint_edges, sink)
    indeg = {a: 0 for a in nodes}; outdeg = {a: 0 for a in nodes}
    for u, v in workflow:
        outdeg[u] = outdeg.get(u, 0) + 1; indeg[v] = indeg.get(v, 0) + 1
    return np.asarray([
        len(nodes), len(workflow), len(source), len(sink), wf_path, wf_dist,
        max(outdeg.values(), default=0), max(indeg.values(), default=0),
        len(wf - observed_edges) / max(1, len(wf)), len(observed),
        sum(e.get("data_class") == "untrusted" for e in observed),
        sum(e.get("data_class") in {"unknown", "redacted"} for e in observed),
        sum(e.get("edge_kind") == "unexpected" for e in observed),
        sum(bool(e.get("permission_mismatch")) for e in events), rt_path, rt_dist,
    ], dtype=float)


def threshold(y, p):
    vals = np.unique(np.r_[0.0, p, 1.0])
    return float(max(vals, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def evaluate(traces, labels, mode, seed):
    y = np.asarray([labels[e["episode_id"]]["label"] for e in traces], dtype=int)
    groups = np.asarray([labels[e["episode_id"]]["scenario_family"] for e in traces])
    X = np.asarray([features(truncate(e, mode)) for e in traces])
    p = np.zeros(len(y)); pred = np.zeros(len(y), dtype=bool)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups), 1):
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed + fold).fit(X[tr], y[tr])
        p_tr = model.predict_proba(X[tr])[:, 1]; p_te = model.predict_proba(X[te])[:, 1]
        p[te] = p_te; pred[te] = p_te >= threshold(y[tr], p_tr)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "positive_predictions": int(pred.sum()),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traces", type=Path, default=Path("paperAlpha/results/independent_mas_v3/traces_public.jsonl")); ap.add_argument("--labels", type=Path, default=Path("paperAlpha/results/independent_mas_v3/labels.jsonl")); ap.add_argument("--out", type=Path, default=Path("paperAlpha/results/independent_mas_prefix_v1")); args = ap.parse_args()
    traces = read_jsonl(args.traces); labels = {r["episode_id"]: r for r in read_jsonl(args.labels)}
    modes = ["0.25", "0.50", "0.75", "1.0", "pre_action"]
    results = {mode: evaluate(traces, labels, mode, 20261020 + i) for i, mode in enumerate(modes)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "metrics.json").write_text(json.dumps({"protocol": {"split": "5-fold StratifiedGroupKFold by scenario_family", "features": "public workflow DAG plus truncated public events", "label_use": "episode_id join only", "pre_action_definition": "events strictly before first privileged tool_call"}, "results": results}, indent=2), encoding="utf-8")
    lines = ["# Independent MAS prefix evaluation v1", "", "Early-warning ablation on public traces; each fold is family-disjoint and thresholds are train-only.", "", "| prefix | F1 | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|"]
    for mode in modes:
        r = results[mode]; lines.append(f"| {mode} | {r['f1']:.3f} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['brier']:.3f} |")
    lines += ["", "`pre_action` excludes the privileged tool-call event; this is a development early-warning check, not a claim that every deployment field is available before action.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8"); print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
