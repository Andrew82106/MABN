"""Fair baseline comparison on the public independent_mas_v2 release.

All methods receive the same public trace rows and the same deterministic
scenario-family group split.  Labels are joined only for fitting/evaluation.
Feature extraction never reads label, external_effect, hidden_fields, or
label_source.  The methods differ only in their inductive bias:

* aggregate_logistic: flat event/count features, no connectivity;
* local_only: per-agent local semantic features pooled without edges;
* graph_features_learned: learned logistic model over local + graph summaries;
* unified_v2: the two-layer local/semantic-gated message-passing model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PRIV_TOOLS = {"payments.write", "filesystem.write", "shell.exec"}
FORBIDDEN = {"label", "external_effect", "hidden_fields", "label_source"}
LOCAL_NAMES = (
    "source_input",
    "privileged_sink",
    "sanitization",
    "unexpected_outgoing",
    "untrusted_outgoing",
    "untrusted_incoming",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def public_episode(trace: dict, labels: dict[str, dict]) -> dict:
    leaked = FORBIDDEN.intersection(trace)
    if leaked:
        raise ValueError(f"public trace contains forbidden fields: {sorted(leaked)}")
    episode_id = trace.get("episode_id")
    if episode_id not in labels:
        raise ValueError(f"missing evaluation label for {episode_id}")
    # Keep the label in the evaluation record, but feature functions below are
    # intentionally trace-only and never access it.
    return {**trace, "label": int(labels[episode_id]["label"]), "scenario_family": labels[episode_id].get("scenario_family", "unknown")}


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def graph_parts(ep: dict):
    events = ep.get("events", [])
    agents = sorted(
        set(ep.get("agents", []))
        | {e.get("actor") for e in events if e.get("actor")}
        | {e.get("target") for e in events if e.get("target")}
    )
    edges = [
        e
        for e in events
        if e.get("event_type") in {"message_sent", "delegation"} and e.get("target")
    ]
    sources = {
        e.get("actor")
        for e in events
        if e.get("event_type") == "input_received"
        and e.get("data_class") == "untrusted"
    }
    sinks = {
        e.get("actor")
        for e in events
        if e.get("event_type") == "tool_call"
        and e.get("tool") in PRIV_TOOLS
    }
    return agents, edges, sources, sinks


def reachable(
    sources: set[str],
    sinks: set[str],
    edges: list[dict],
    taint_only: bool,
) -> tuple[set[str], int | None]:
    adj: dict[str, set[str]] = defaultdict(set)
    for event in edges:
        if not taint_only or event.get("data_class") == "untrusted":
            adj[event["actor"]].add(event["target"])
    seen = set(sources)
    queue = [(source, 0) for source in sources]
    shortest = None
    while queue:
        current, depth = queue.pop(0)
        for nxt in adj[current]:
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, depth + 1))
            if nxt in sources:
                continue
    # Compute shortest source-to-sink distance without changing the observed
    # graph semantics.  This is a graph feature, not a label-derived feature.
    for source in sources:
        todo = [(source, 0)]
        visited = {source}
        while todo:
            current, depth = todo.pop(0)
            if current in sinks:
                shortest = depth if shortest is None else min(shortest, depth)
                break
            for nxt in adj[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    todo.append((nxt, depth + 1))
    return seen, shortest


def local_matrix(ep: dict) -> tuple[list[str], np.ndarray]:
    agents, _, _, _ = graph_parts(ep)
    index = {name: i for i, name in enumerate(agents)}
    matrix = np.zeros((len(agents), len(LOCAL_NAMES)), dtype=float)
    for event in ep.get("events", []):
        actor = event.get("actor")
        if actor not in index:
            continue
        row = matrix[index[actor]]
        kind = event.get("event_type")
        if kind == "input_received" and event.get("data_class") == "untrusted":
            row[0] += 1
        if kind == "tool_call" and event.get("tool") in PRIV_TOOLS:
            row[1] += 1
        if kind == "sanitization":
            row[2] += 1
        if kind in {"message_sent", "delegation"}:
            if event.get("edge_kind") == "unexpected":
                row[3] += 1
            if event.get("data_class") == "untrusted":
                row[4] += 1
            if event.get("data_class") in {"unknown", "redacted"}:
                # Unknown incoming evidence is represented separately by the
                # graph model; it is not silently treated as untrusted here.
                target = event.get("target")
                if target in index:
                    matrix[index[target], 5] += 0
    # Incoming untrusted messages are assigned to the receiving agent.
    for event in ep.get("events", []):
        target = event.get("target")
        if (
            target in index
            and event.get("event_type") in {"message_sent", "delegation"}
            and event.get("data_class") == "untrusted"
        ):
            matrix[index[target], 5] += 1
    return agents, matrix


def aggregate_features(ep: dict) -> np.ndarray:
    """Flat count features; deliberately discards edge connectivity."""
    events = ep.get("events", [])
    agents, edges, sources, sinks = graph_parts(ep)
    messages = [e for e in edges if e.get("event_type") == "message_sent"]
    delegations = [e for e in edges if e.get("event_type") == "delegation"]
    confidences = [
        float(e["edge_confidence"])
        for e in edges
        if isinstance(e.get("edge_confidence"), (int, float))
    ]
    return np.asarray(
        [
            len(events),
            len(agents),
            len(messages),
            len(delegations),
            sum(e.get("event_type") == "tool_call" for e in events),
            len(sinks),
            sum(e.get("event_type") == "input_received" for e in events),
            len(sources),
            sum(e.get("data_class") == "untrusted" for e in events),
            sum(e.get("data_class") == "unknown" for e in events),
            sum(e.get("edge_kind") == "unexpected" for e in edges),
            sum(e.get("data_class") == "clean" for e in edges),
            float(np.mean(confidences)) if confidences else 0.0,
            float(np.min(confidences)) if confidences else 0.0,
            len(PRIV_TOOLS.intersection({e.get("tool") for e in events})),
        ],
        dtype=float,
    )


def local_features(ep: dict) -> np.ndarray:
    _, matrix = local_matrix(ep)
    if matrix.size == 0:
        return np.zeros(2 * len(LOCAL_NAMES), dtype=float)
    # Pool local evidence only. No agent-to-agent edge relation is used.
    return np.r_[matrix.sum(axis=0), matrix.max(axis=0)]


def graph_features(ep: dict) -> np.ndarray:
    agents, edges, sources, sinks = graph_parts(ep)
    all_seen, all_distance = reachable(sources, sinks, edges, taint_only=False)
    taint_seen, taint_distance = reachable(sources, sinks, edges, taint_only=True)
    confidences = [
        float(e["edge_confidence"])
        for e in edges
        if isinstance(e.get("edge_confidence"), (int, float))
    ]
    local = local_features(ep)
    return np.r_[
        local,
        len(agents),
        len(edges),
        sum(e.get("data_class") == "untrusted" for e in edges),
        sum(e.get("data_class") in {"unknown", "redacted"} for e in edges),
        sum(e.get("edge_kind") == "unexpected" for e in edges),
        float(np.mean(confidences)) if confidences else 0.0,
        float(np.min(confidences)) if confidences else 0.0,
        len(sources),
        len(sinks),
        float(bool(sinks.intersection(all_seen))),
        float(bool(sinks.intersection(taint_seen))),
        1.0 / (1.0 + (all_distance if all_distance is not None else len(agents) + 1)),
        1.0 / (1.0 + (taint_distance if taint_distance is not None else len(agents) + 1)),
    ].astype(float)


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    z = (x_train - x_mean) / x_scale
    x_aug = np.c_[np.ones(len(z)), z]
    init = rng.normal(0.0, 0.03, x_aug.shape[1])

    def objective(theta):
        p = np.asarray(sigmoid(x_aug @ theta))
        bce = -np.mean(
            y_train * np.log(np.clip(p, 1e-8, 1 - 1e-8))
            + (1 - y_train) * np.log(np.clip(1 - p, 1e-8, 1 - 1e-8))
        )
        return float(bce + 1e-3 * np.sum(theta[1:] ** 2))

    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0)] * len(init),
        options={"maxiter": 500, "ftol": 1e-11},
    )
    return np.asarray([result.x, x_mean, x_scale], dtype=object)


def predict_logistic(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta, mean, scale = model
    z = (x - mean) / scale
    return np.asarray(sigmoid(np.c_[np.ones(len(z)), z] @ theta), dtype=float)


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    values = np.unique(np.r_[0.0, scores, 1.0])
    return float(max(values, key=lambda threshold: f1_score(y, scores >= threshold, zero_division=0)))


def ece(y: np.ndarray, p: np.ndarray) -> float:
    value = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10bin": ece(y, p),
        "threshold": float(threshold),
        "predicted_positive": int(pred.sum()),
    }


def load_unified():
    path = Path(__file__).with_name("evaluate_unified_monitor_v2.py")
    spec = importlib.util.spec_from_file_location("unified_monitor_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_two_layer():
    path = Path(__file__).with_name("evaluate_two_layer_v2.py")
    spec = importlib.util.spec_from_file_location("two_layer_monitor_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def group_split(eps: list[dict], holdout_fraction: float) -> tuple[np.ndarray, np.ndarray, dict]:
    groups = sorted({e.get("scenario_family", "unknown") for e in eps})
    n_hold = max(1, int(len(groups) * holdout_fraction))
    held = groups[-n_hold:]
    held_set = set(held)
    test = np.asarray([i for i, e in enumerate(eps) if e.get("scenario_family", "unknown") in held_set])
    train = np.asarray([i for i in range(len(eps)) if i not in set(test)])
    return train, test, {
        "type": "scenario_family_group_holdout",
        "seed": None,
        "holdout_fraction": holdout_fraction,
        "groups_total": len(groups),
        "groups_train": len(groups) - len(held),
        "groups_test": len(held),
        "held_out_groups": held,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/independent_mas_v2/traces_public.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("results/independent_mas_v2/labels.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/independent_fair_baselines_v1"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    traces = read_jsonl(args.input)
    label_rows = read_jsonl(args.labels)
    labels = {r["episode_id"]: {"label": int(r["label"]), "scenario_family": r.get("scenario_family", "unknown")} for r in label_rows}
    eps = [public_episode(trace, labels) for trace in traces]
    y = np.asarray([e["label"] for e in eps], dtype=int)
    train, test, split = group_split(eps, args.holdout_fraction)
    if not len(test) or not len(train):
        raise ValueError("group split produced an empty partition")

    feature_sets = {
        "aggregate_logistic": np.asarray([aggregate_features(e) for e in eps]),
        "local_only": np.asarray([local_features(e) for e in eps]),
        "graph_features_learned": np.asarray([graph_features(e) for e in eps]),
    }
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for offset, (name, features) in enumerate(feature_sets.items()):
        model = fit_logistic(features[train], y[train], args.seed + offset)
        p_train = predict_logistic(model, features[train])
        p_test = predict_logistic(model, features[test])
        threshold = best_threshold(y[train], p_train)
        results[name] = {
            "metrics": metrics(y[test], p_test, threshold),
            "feature_count": int(features.shape[1]),
            "train_loss": float(np.mean(-(y[train] * np.log(np.clip(p_train, 1e-8, 1 - 1e-8)) + (1 - y[train]) * np.log(np.clip(1 - p_train, 1e-8, 1 - 1e-8))))),
        }
        predictions[name] = p_test

    unified = load_unified()
    theta0 = np.zeros(14)
    theta0[:7] = np.random.default_rng(args.seed).normal(0, 0.1, 7)
    theta0[7:11] = np.array([-1.0, 2.0, 0.5, 0.5])
    theta0[11:] = [-2.0, 4.0, 0.5]
    fit = minimize(
        unified.loss,
        theta0,
        args=([eps[i] for i in train], y[train]),
        method="L-BFGS-B",
        bounds=[(-8, 8)] * 14,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    p_train = np.asarray([unified.forward(fit.x, eps[i]) for i in train])
    p_test = np.asarray([unified.forward(fit.x, eps[i]) for i in test])
    threshold = best_threshold(y[train], p_train)
    results["unified_v2"] = {
        "metrics": metrics(y[test], p_test, threshold),
        "feature_count": 14,
        "train_loss": float(fit.fun),
        "optimizer_success": bool(fit.success),
        "iterations": int(fit.nit),
    }
    predictions["unified_v2"] = p_test

    # Full Layer-1 CPT + Layer-2 uncertainty-aware monitor, evaluated on the
    # same public traces, group split and train-only threshold.
    two = load_two_layer(); train_eps = [eps[i] for i in train]; test_eps = [eps[i] for i in test]
    bn = two.fit_bn(train_eps, y[train]); theta, opt = two.fit_model(train_eps, y[train], bn, args.seed + 100)
    p_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps]); p_test = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in test_eps]); threshold = best_threshold(y[train], p_train)
    results["two_layer_bn_graph_v2"] = {"metrics": metrics(y[test], p_test, threshold), "feature_count": 13, "train_loss": float(opt["train_loss"]), "optimizer_success": bool(opt["success"]), "iterations": int(opt["iterations"])}
    predictions["two_layer_bn_graph_v2"] = p_test

    args.out.mkdir(parents=True, exist_ok=True)
    split_payload = {
        **split,
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_episode_ids": [eps[i]["episode_id"] for i in train],
        "test_episode_ids": [eps[i]["episode_id"] for i in test],
    }
    (args.out / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, i in enumerate(test):
            handle.write(
                json.dumps(
                    {
                        "episode_id": eps[i]["episode_id"],
                        "scenario_family": eps[i].get("scenario_family"),
                        "label": int(y[i]),
                        **{name: float(scores[row]) for name, scores in predictions.items()},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "dataset": {
            "name": "independent_mas_v2",
            "input": str(args.input),
            "labels": str(args.labels),
            "n": len(eps),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "public_fields_only": True,
            "forbidden_model_fields": sorted(FORBIDDEN),
        },
        "split": split_payload,
        "methods": results,
        "unified_parameters": [float(value) for value in fit.x],
        "feature_contract": {
            "shared_trace_fields": ["episode_id", "agents", "workflow_dag", "events", "scenario_family"],
            "model_excludes": sorted(FORBIDDEN),
            "note": "scenario_family is used only to construct the group split, never as a model feature",
        },
        "limitations": [
            "The benchmark labels are hidden-effect labels generated by a synthetic process.",
            "Group holdout is deterministic by sorted scenario_family; held-out families are recorded in split.json.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# independent_mas_v2 Fair Baselines v1",
        "",
        "All methods use the public traces and the same scenario-family group holdout. Labels are joined only for fitting and evaluation.",
        "",
        f"- n={len(eps)}, positive={int(y.sum())}, train={len(train)}, test={len(test)}",
        f"- groups: train={split['groups_train']}, test={split['groups_test']}",
        "",
        "| method | F1 | precision | recall | AUROC | AUPRC | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        m = result["metrics"]
        auc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        lines.append(
            f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{auc} | {m['auprc']:.3f} | {m['brier']:.3f} | {m['ece_10bin']:.3f} |"
        )
    lines += [
        "",
        "## Fairness contract",
        "- `scenario_family` is split metadata only; it is not a model input.",
        "- No method reads `label`, `external_effect`, `hidden_fields`, or `label_source` while constructing features.",
        "- Thresholds are selected on the common training partition only.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
