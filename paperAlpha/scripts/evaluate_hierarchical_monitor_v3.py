"""Hierarchical MAS monitor v3: normative workflow + runtime graph.

This experiment keeps two graph semantics separate:

* ``workflow_dag`` is the normative collaboration plan (what should happen);
* message/delegation events are the observed runtime graph (what happened).

The local BN and runtime message propagator are reused from v2.  A train-only
workflow-context head then models coverage, missing expected edges, unexpected
edges, and normative-vs-observed reachability.  A final train-only fusion head
combines the local/runtime score and the workflow-context score.  No labels or
hidden outcome fields are used as features.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


FORBIDDEN_TOP = {"label", "external_effect", "hidden_fields", "label_source", "scenario_family"}
FORBIDDEN_EVENT = {"label", "external_effect", "tool_side_effect", "hidden_fields", "label_source"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_public_trace(trace: dict) -> None:
    leaked = FORBIDDEN_TOP.intersection(trace)
    if leaked:
        raise ValueError(f"forbidden top-level fields: {sorted(leaked)}")
    for event in trace.get("events", []):
        leaked_event = FORBIDDEN_EVENT.intersection(event)
        if leaked_event:
            raise ValueError(f"forbidden event fields: {sorted(leaked_event)}")


def _workflow_edges(ep: dict) -> set[tuple[str, str]]:
    dag = ep.get("workflow_dag") or {}
    return {(str(u), str(v)) for u, v in dag.get("edges", []) if u and v and u != v}


def _runtime_edges(ep: dict) -> list[dict]:
    return [
        e for e in ep.get("events", [])
        if e.get("event_type") in {"message_sent", "delegation"}
        and e.get("actor") is not None and e.get("target") is not None
    ]


def _source_sink(ep: dict) -> tuple[set[str], set[str]]:
    sources = {
        e.get("actor") for e in ep.get("events", [])
        if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted"
    }
    sinks = {
        e.get("actor") for e in ep.get("events", [])
        if e.get("event_type") == "tool_call"
        and e.get("tool") in {"payments.write", "filesystem.write", "shell.exec"}
    }
    return sources, sinks


def _reach(sources: set[str], sinks: set[str], edges: set[tuple[str, str]]) -> tuple[bool, int | None]:
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    queue = deque((s, 0) for s in sources)
    seen = set(sources)
    while queue:
        u, depth = queue.popleft()
        if u in sinks:
            return True, depth
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append((v, depth + 1))
    return False, None


def workflow_features(ep: dict) -> np.ndarray:
    """Features that explicitly compare normative and observed graphs."""
    expected = _workflow_edges(ep)
    runtime_rows = _runtime_edges(ep)
    runtime = {(str(e["actor"]), str(e["target"])) for e in runtime_rows}
    overlap = expected.intersection(runtime)
    unexpected = runtime - expected
    missing = expected - runtime
    sources, sinks = _source_sink(ep)
    expected_reach, expected_dist = _reach(sources, sinks, expected)
    observed_reach, observed_dist = _reach(sources, sinks, runtime)
    untrusted = [e for e in runtime_rows if e.get("data_class") == "untrusted"]
    unknown = [e for e in runtime_rows if e.get("data_class") in {"unknown", "redacted"}]
    confidences = [float(e["edge_confidence"]) for e in runtime_rows if isinstance(e.get("edge_confidence"), (int, float))]
    n_agents = max(1, len(ep.get("agents", [])))
    n_expected = max(1, len(expected))
    n_runtime = max(1, len(runtime))
    return np.asarray([
        min(len(ep.get("agents", [])) / 12.0, 1.0),
        min(len(expected) / 20.0, 1.0),
        min(len(runtime) / 20.0, 1.0),
        len(overlap) / n_expected,
        len(missing) / n_expected,
        len(unexpected) / n_runtime,
        len(untrusted) / n_runtime,
        len(unknown) / n_runtime,
        float(np.mean(confidences)) if confidences else 0.0,
        float(np.min(confidences)) if confidences else 0.0,
        min(len(sources) / 4.0, 1.0),
        min(len(sinks) / 4.0, 1.0),
        float(expected_reach),
        float(observed_reach),
        1.0 / (1.0 + (expected_dist if expected_dist is not None else len(ep.get("agents", [])) + 1)),
        1.0 / (1.0 + (observed_dist if observed_dist is not None else len(ep.get("agents", [])) + 1)),
    ], dtype=float)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/independent_mas_v2/traces_public.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("results/independent_mas_v2/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/hierarchical_monitor_v3"))
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--holdout-fraction", type=float, default=.20)
    args = ap.parse_args()

    fair = load_module("fair_hier_v3", Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py"))
    two = load_module("two_hier_v3", Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces = read_jsonl(args.input)
    labels = {row["episode_id"]: {"label": int(row["label"]), "scenario_family": row.get("scenario_family", "unknown")} for row in read_jsonl(args.labels)}
    for trace in traces:
        validate_public_trace(trace)
    eps = [fair.public_episode(trace, labels) for trace in traces]
    y = np.asarray([e["label"] for e in eps], dtype=int)
    train, test, split = fair.group_split(eps, args.holdout_fraction)
    train_eps = [eps[i] for i in train]
    test_eps = [eps[i] for i in test]

    # Existing local/runtime model; all fitted quantities use train only.
    bn = two.fit_bn(train_eps, y[train])
    theta, opt = two.fit_model(train_eps, y[train], bn, args.seed + 11)
    p_local_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
    p_local_test = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in test_eps])

    wf_train = np.asarray([workflow_features(e) for e in train_eps])
    wf_test = np.asarray([workflow_features(e) for e in test_eps])
    wf_model = fair.fit_logistic(wf_train, y[train], args.seed + 12)
    p_wf_train = fair.predict_logistic(wf_model, wf_train)
    p_wf_test = fair.predict_logistic(wf_model, wf_test)

    # Named, train-only fusion.  The head can learn whether workflow mismatch
    # should increase risk or instead signal missing observability.
    fusion_x_train = np.c_[logit(p_local_train), logit(p_wf_train), wf_train[:, 3], wf_train[:, 4], wf_train[:, 5], wf_train[:, 7]]
    fusion_x_test = np.c_[logit(p_local_test), logit(p_wf_test), wf_test[:, 3], wf_test[:, 4], wf_test[:, 5], wf_test[:, 7]]
    fusion_model = fair.fit_logistic(fusion_x_train, y[train], args.seed + 13)
    p_fusion_train = fair.predict_logistic(fusion_model, fusion_x_train)
    p_fusion_test = fair.predict_logistic(fusion_model, fusion_x_test)
    th_local = fair.best_threshold(y[train], p_local_train)
    th_wf = fair.best_threshold(y[train], p_wf_train)
    th_fusion = fair.best_threshold(y[train], p_fusion_train)

    results = {
        "local_message": metric(y[test], p_local_test, th_local),
        "workflow_context": metric(y[test], p_wf_test, th_wf),
        "hierarchical_fusion_v3": metric(y[test], p_fusion_test, th_fusion),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "model": {
            "name": "HierarchicalNormativeRuntimeMonitor-v3",
            "normative_graph": "workflow_dag",
            "runtime_graph": "message_sent/delegation events",
            "local_component": "TwoLayerBNGraphMonitor-v2",
            "workflow_features": [
                "agent_count", "expected_edge_count", "runtime_edge_count", "expected_runtime_overlap",
                "missing_expected_edges", "unexpected_runtime_edges", "untrusted_edge_rate", "unknown_edge_rate",
                "mean_confidence", "min_confidence", "source_count", "sink_count",
                "normative_reachability", "observed_reachability", "normative_distance", "observed_distance",
            ],
            "fusion_inputs": ["logit(local_message)", "logit(workflow_context)", "edge_overlap", "missing_edges", "unexpected_edges", "unknown_rate"],
            "forbidden_features": sorted(FORBIDDEN_TOP),
            "causal_claim": False,
        },
        "dataset": {"n": len(eps), "positive": int(y.sum()), "negative": int(len(y) - y.sum())},
        "split": {**split, "train_count": int(len(train)), "test_count": int(len(test))},
        "metrics": results,
        "optimizer": opt,
        "workflow_feature_count": int(wf_train.shape[1]),
        "fusion_parameters": [float(v) for v in fusion_model[0]],
        "limitations": [
            "Synthetic hidden-effect benchmark; external validity is not established.",
            "workflow_dag is normative context, not a discovered causal graph.",
            "All heads are train-only; threshold is selected on train only.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Hierarchical normative-runtime monitor v3", "",
        "规范 `workflow_dag` 与运行时消息/委派图分开建模；所有参数和阈值仅在训练组上估计。", "",
        "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in results.items():
        lines.append(f"| {name} | {value['f1']:.3f} | {value['precision']:.3f} | {value['recall']:.3f} | {value['auroc']:.3f} | {value['auprc']:.3f} | {value['brier']:.3f} |")
    lines += ["", "## Interpretation", "- workflow_context measures normative/runtime agreement and reachability, not causality.", "- hierarchical_fusion_v3 is an auditable train-only fusion of local risk and graph-context evidence."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, idx in enumerate(test):
            handle.write(json.dumps({
                "episode_id": eps[idx]["episode_id"],
                "label": int(y[idx]),
                "local_message": float(p_local_test[i]),
                "workflow_context": float(p_wf_test[i]),
                "hierarchical_fusion_v3": float(p_fusion_test[i]),
                "workflow_features": wf_test[i].tolist(),
            }, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
