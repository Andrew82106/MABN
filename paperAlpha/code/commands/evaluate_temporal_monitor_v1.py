"""Legacy offline temporal diagnostic; NOT an online early-warning model.

Although edges are traversed in order, source/sink flags are extracted from
the entire episode beforehand. Historical activation times therefore depend
on future events. Retained for reproducibility, not online safety claims.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_two(path: Path):
    spec = importlib.util.spec_from_file_location("two_temporal", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def temporal_forward(theta: np.ndarray, parsed: dict, bn: dict, explain: bool = False):
    names = parsed["names"]
    source = parsed["source"]
    sink = parsed["sink"]
    state = source.copy()
    seen_hard = set(np.where(source > 0)[0])
    hard_distance = {i: 0 for i in seen_hard}
    first_sink_sequence = None
    history = []
    # Preserve the observed runtime order; ties are resolved by input order.
    ordered = sorted(enumerate(parsed["edges"]), key=lambda pair: (pair[1][3].get("sequence") is None, pair[1][3].get("sequence", pair[0]), pair[0]))
    for edge_index, (u, v, feat, info) in ordered:
        ff = feat.copy()
        unknown = ff[4] > 0
        gate = float(sigmoid(np.dot(theta[:7], ff)))
        carrier = float(ff[1] > 0)
        q = gate * carrier * max(0.0, min(1.0, ff[3]))
        state[v] = 1.0 - (1.0 - state[v]) * (1.0 - state[u] * q)
        if ff[1] > 0 and u in seen_hard:
            if v not in seen_hard:
                seen_hard.add(v)
                hard_distance[v] = hard_distance[u] + 1
        if first_sink_sequence is None and sink[v] > 0 and state[v] >= 0.5:
            first_sink_sequence = info.get("sequence")
        history.append({"sequence": info.get("sequence"), "source": info.get("source"), "target": info.get("target"), "edge_score": q, "target_state": float(state[v])})
    path = float(1.0 - np.prod(1.0 - state * sink))
    n = max(1, len(names))
    path_norm = path ** (1.0 / max(1, n - 1))
    local = float(1.0 - np.prod(1.0 - source * sink))
    l1, node_scores = two_module.layer1_score(parsed, bn)
    hard_path = float(any(sink[i] > 0 and i in seen_hard for i in range(n)))
    distance = min((hard_distance[i] for i in hard_distance if sink[i] > 0), default=None)
    distance_evidence = 1.0 / (1.0 + distance) if distance is not None and distance > 0 else 0.0
    score = float(sigmoid(theta[7] + theta[8] * path_norm + theta[9] * local + theta[10] * l1 + theta[11] * hard_path + theta[12] * distance_evidence))
    if not explain:
        return score
    return score, {
        "path_evidence": path_norm,
        "raw_path_evidence": path,
        "local_evidence": local,
        "layer1_score": l1,
        "asserted_path_evidence": hard_path,
        "shortest_asserted_path_evidence": distance_evidence,
        "node_scores": node_scores,
        "propagated_state": {names[i]: float(state[i]) for i in range(n)},
        "first_sink_sequence": first_sink_sequence,
        "temporal_history": history,
    }


def fit_temporal(train_eps, y, bn, seed):
    rng = np.random.default_rng(seed)
    x0 = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3]) + rng.normal(0, .03, 13)

    def objective(theta):
        probs = np.asarray([temporal_forward(theta, two_module.event_features(e), bn) for e in train_eps])
        bce = -np.mean(y * np.log(np.clip(probs, 1e-7, 1 - 1e-7)) + (1 - y) * np.log(np.clip(1 - probs, 1e-7, 1 - 1e-7)))
        return float(bce + 1e-3 * np.sum(theta * theta))

    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(-8., 8.)] * 13, options={"maxiter": 500, "ftol": 1e-10})
    return result.x, {"success": bool(result.success), "iterations": int(result.nit), "train_loss": float(result.fun), "message": str(result.message)}


def metrics(y, p, threshold=.5):
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl'))
    ap.add_argument("--labels", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl'))
    ap.add_argument("--out", type=Path, default=Path('experiments/synthetic_mas/runs/temporal_monitor_v1'))
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    global two_module
    two_module = load_two(Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces = read(args.input)
    labels = {r["episode_id"]: int(r["label"]) for r in read(args.labels)}
    eps = []
    for trace in traces:
        if any(k in trace for k in {"label", "external_effect", "hidden_fields", "label_source", "scenario_family"}):
            raise ValueError("evaluation fields in public trace")
        e = {**trace, "label": labels[trace["episode_id"]]}
        eps.append(e)
    y = np.asarray([e["label"] for e in eps], dtype=int)
    idx = np.arange(len(eps))
    train_idx, test_idx = train_test_split(idx, test_size=.30, stratify=y, random_state=args.seed)
    train_eps = [eps[i] for i in train_idx]
    test_eps = [eps[i] for i in test_idx]
    bn = two_module.fit_bn(train_eps, y[train_idx])
    theta, opt = fit_temporal(train_eps, y[train_idx], bn, args.seed + 1)
    p_train = np.asarray([temporal_forward(theta, two_module.event_features(e), bn) for e in train_eps])
    p_test = np.asarray([temporal_forward(theta, two_module.event_features(e), bn) for e in test_eps])
    threshold = .5
    predictions = []
    for i, e in enumerate(test_eps):
        p, explanation = temporal_forward(theta, two_module.event_features(e), bn, explain=True)
        predictions.append({"episode_id": e["episode_id"], "label": int(y[test_idx[i]]), "risk_probability": p, "explanation": explanation})
    positive_alarms = [r for r in predictions if r["label"] == 1 and r["explanation"].get("first_sink_sequence") is not None]
    lead_times = []
    for row in positive_alarms:
        history = row["explanation"].get("temporal_history", [])
        last_seq = max((h.get("sequence") for h in history if h.get("sequence") is not None), default=None)
        first_seq = row["explanation"].get("first_sink_sequence")
        if last_seq is not None and first_seq is not None:
            lead_times.append(float(last_seq - first_seq))
    legacy_activation = {"positive_count": int(sum(r["label"] == 1 for r in predictions)), "positive_alarm_count": len(positive_alarms), "positive_alarm_rate": float(len(positive_alarms) / max(1, sum(r["label"] == 1 for r in predictions))), "mean_lead_events": float(np.mean(lead_times)) if lead_times else None, "median_lead_events": float(np.median(lead_times)) if lead_times else None,
                         "interpretation": "posthoc activation relative to last message; not alarm recall or harm lead time"}
    early_warning = {"eligible": False, "actual_violation_lead_time": None,
                     "reason": "full-episode source/sink leakage and no independent violation timestamp"}
    result = {"model": {"name": "TemporalEvidenceMonitor-v1", "runtime_only": True, "workflow_dag_used": False, "propagation": "offline sequence traversal with full-episode flags", "online_eligible": False, "causal_claim": False}, "dataset": {"n": len(eps), "positive": int(y.sum()), "input": str(args.input)}, "split": {"train": len(train_idx), "test": len(test_idx), "seed": args.seed}, "optimizer": opt, "metrics": metrics(y[test_idx], p_test, threshold), "train_metrics": metrics(y[train_idx], p_train, threshold), "early_warning": early_warning, "legacy_posthoc_activation": legacy_activation}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    m = result["metrics"]
    (args.out / "REPORT.md").write_text(f"# Legacy Temporal Evidence Monitor v1\n\n仅作离线历史诊断：source/sink来自完整轨迹，不能评价在线预警。\n\n- 离线F1={m['f1']:.3f}\n- 离线AUROC={m['auroc']:.3f}\n- 离线AUPRC={m['auprc']:.3f}\n- 离线Brier={m['brier']:.3f}\n\n早预警资格=false；真实危险提前量不可评估。旧激活统计保留于legacy_posthoc_activation，不得当作预警召回或提前事件数。\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
