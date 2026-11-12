"""Auditable policy/deviation template heads for the MAS risk monitor.

This experiment adds two small, named risk-template heads to the existing
runtime monitor:

* ``policy_template``: authorization/capability mismatch evidence around
  privileged tool calls;
* ``deviation_template``: normative-vs-runtime workflow deviation evidence,
  with explicit unexpected-edge bridge features.

The heads receive exactly the public trace contract (events and normative
``workflow_dag``).  Labels, mechanism names, hidden authorization, hidden
edges and terminal effects are evaluation-only.  All standardization,
parameters and thresholds are fit on the training topology families only.
This is an evaluation extension, not a modification of the core monitor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

PRIV = {"payments.write", "filesystem.write", "shell.exec"}
FORBIDDEN_TOP = {"label", "external_effect", "hidden_fields", "label_source", "scenario_family"}
FORBIDDEN_EVENT = {"label", "external_effect", "tool_side_effect", "hidden_fields", "label_source"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_public(trace: dict) -> None:
    leaked = FORBIDDEN_TOP.intersection(trace)
    if leaked:
        raise ValueError(f"forbidden public fields: {sorted(leaked)}")
    for event in trace.get("events", []):
        leaked_event = FORBIDDEN_EVENT.intersection(event)
        if leaked_event:
            raise ValueError(f"forbidden event fields: {sorted(leaked_event)}")


def edges(ep: dict) -> list[dict]:
    return [e for e in ep.get("events", []) if e.get("event_type") in {"message_sent", "delegation"} and e.get("actor") and e.get("target")]


def workflow_edges(ep: dict) -> set[tuple[str, str]]:
    dag = ep.get("workflow_dag") or {}
    return {(str(u), str(v)) for u, v in dag.get("edges", []) if u and v and u != v}


def sources_sinks(ep: dict) -> tuple[set[str], set[str]]:
    src = {e.get("actor") for e in ep.get("events", []) if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted"}
    sink = {e.get("actor") for e in ep.get("events", []) if e.get("event_type") == "tool_call" and e.get("tool") in PRIV}
    return src, sink


def reachable(start: set[str], rows: list[tuple[str, str]]) -> set[str]:
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v in rows:
        adj[u].add(v)
    seen = set(start)
    todo = list(start)
    while todo:
        u = todo.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return seen


def can_reach(target: set[str], rows: list[tuple[str, str]]) -> set[str]:
    rev = [(v, u) for u, v in rows]
    return reachable(target, rev)


def has_path(src: set[str], target: set[str], rows: list[tuple[str, str]]) -> bool:
    return bool(reachable(src, rows).intersection(target))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


POLICY_NAMES = (
    "privileged_call_count",
    "privileged_mismatch_count",
    "mismatch_rate",
    "any_privileged_mismatch",
    "mismatch_on_untainted_sink",
    "isolated_privileged_sink",
    "permission_denied_count",
    "privileged_unknown_provenance",
)

DEVIATION_NAMES = (
    "unexpected_edge_count",
    "unexpected_edge_rate",
    "unexpected_to_privileged_sink",
    "unexpected_from_untrusted_source",
    "unexpected_untrusted_count",
    "unexpected_bridge_source_to_sink",
    "unexpected_mean_confidence",
    "expected_edge_missing_rate",
    "expected_runtime_overlap",
    "normative_edge_count",
    "observed_all_path_to_sink",
    "observed_untrusted_path_to_sink",
)


def template_features(ep: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return policy/deviation features and named raw evidence.

    Features are computed entirely from the public trace.  The explicit
    ``permission_mismatch`` field is treated as a noisy pre-effect policy
    telemetry signal, not as a terminal label.
    """
    run = edges(ep)
    expected = workflow_edges(ep)
    runtime_pairs = [(str(e["actor"]), str(e["target"])) for e in run]
    runtime_set = set(runtime_pairs)
    unexpected = [e for e in run if (str(e["actor"]), str(e["target"])) not in expected or e.get("edge_kind") == "unexpected"]
    expected_runtime = [e for e in run if (str(e["actor"]), str(e["target"])) in expected and e.get("edge_kind") != "unexpected"]
    missing = expected - runtime_set
    src, sink = sources_sinks(ep)
    all_seen = reachable(src, runtime_pairs)
    taint_pairs = [(str(e["actor"]), str(e["target"])) for e in run if e.get("data_class") == "untrusted"]
    taint_seen = reachable(src, taint_pairs)
    reverse_sink = can_reach(sink, runtime_pairs)
    reverse_taint_sink = can_reach(sink, taint_pairs)

    priv_calls = [e for e in ep.get("events", []) if e.get("event_type") == "tool_call" and e.get("tool") in PRIV]
    mismatches = [e for e in priv_calls if bool(e.get("permission_mismatch")) or bool(e.get("permission_denied"))]
    mismatch_actors = {e.get("actor") for e in mismatches}
    tainted_sink = {a for a in sink if a in taint_seen}
    untainted_mismatch = mismatch_actors.difference(tainted_sink)
    isolated = {a for a in sink if a not in all_seen and a not in src}
    unknown_sink = {e.get("actor") for e in priv_calls if e.get("data_class") in {None, "unknown", "redacted"}}

    unexpected_sink = [e for e in unexpected if e.get("target") in sink]
    unexpected_source = [e for e in unexpected if e.get("actor") in src or e.get("actor") in taint_seen]
    unexpected_untrusted = [e for e in unexpected if e.get("data_class") == "untrusted"]
    # A runtime-only edge is a bridge when its source is reachable from an
    # untrusted source and its target can reach a privileged sink.  This is a
    # graph evidence template, not a causal claim.
    bridge = [e for e in unexpected if e.get("actor") in taint_seen and e.get("target") in reverse_sink]
    conf = []
    for e in unexpected:
        try:
            conf.append(float(e.get("edge_confidence", 0.0)))
        except (TypeError, ValueError):
            pass

    n_priv = max(1, len(priv_calls)); n_run = max(1, len(run)); n_exp = max(1, len(expected))
    policy = np.asarray([
        min(len(priv_calls) / 3.0, 1.0),
        min(len(mismatches) / 3.0, 1.0),
        len(mismatches) / n_priv,
        float(bool(mismatches)),
        len(untainted_mismatch) / n_priv,
        len(isolated) / n_priv,
        min(sum(bool(e.get("permission_denied")) for e in priv_calls) / 3.0, 1.0),
        len(unknown_sink) / n_priv,
    ], dtype=float)
    deviation = np.asarray([
        min(len(unexpected) / 5.0, 1.0),
        len(unexpected) / n_run,
        len(unexpected_sink) / n_priv,
        len(unexpected_source) / n_run,
        min(len(unexpected_untrusted) / 3.0, 1.0),
        len(bridge) / n_run,
        _mean(conf),
        len(missing) / n_exp,
        len(expected_runtime) / n_exp,
        min(len(expected) / 20.0, 1.0),
        float(bool(sink.intersection(all_seen))),
        float(bool(sink.intersection(taint_seen))),
    ], dtype=float)
    raw = {
        "policy": {"mismatches": len(mismatches), "privileged_calls": len(priv_calls), "untainted_mismatch_actors": sorted(untainted_mismatch), "isolated_privileged_sinks": sorted(isolated)},
        "deviation": {"unexpected": len(unexpected), "missing_expected": len(missing), "bridge": len(bridge), "unexpected_to_sink": len(unexpected_sink)},
    }
    return policy, deviation, raw


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "precision": float(precision_score(y, p >= threshold, zero_division=0)),
        "recall": float(recall_score(y, p >= threshold, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    vals = np.unique(np.r_[0.0, p, 1.0])
    return float(max(vals, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/traces_public.jsonl'))
    ap.add_argument("--labels", type=Path, default=Path('experiments/synthetic_mas/runs/independent_mas_v3/labels.jsonl'))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "experiments/model_mechanisms/runs/policy_deviation_templates_v1")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--holdout-fraction", type=float, default=.20)
    args = ap.parse_args()
    fair = load_module("fair_policy_v1", Path(__file__).with_name("evaluate_independent_fair_baselines_v1.py"))
    two = load_module("two_policy_v1", Path(__file__).with_name("evaluate_two_layer_v2.py"))
    traces = read_jsonl(args.input)
    label_rows = read_jsonl(args.labels)
    labels = {r["episode_id"]: {"label": int(r["label"]), "scenario_family": r.get("scenario_family", "unknown"), "mechanism": r.get("mechanism", "unknown")} for r in label_rows}
    for t in traces:
        validate_public(t)
    eps = [fair.public_episode(t, labels) for t in traces]
    y = np.asarray([e["label"] for e in eps], dtype=int)
    train, test, split = fair.group_split(eps, args.holdout_fraction)
    tr_eps = [eps[i] for i in train]; te_eps = [eps[i] for i in test]
    ytr, yte = y[train], y[test]

    p_tr = np.asarray([0.0] * len(tr_eps)); p_te = np.asarray([0.0] * len(te_eps))
    bn = two.fit_bn(tr_eps, ytr)
    theta, optimizer = two.fit_model(tr_eps, ytr, bn, args.seed + 7)
    for i, e in enumerate(tr_eps):
        p_tr[i] = two._propagate(theta, two.event_features(e), bn)
    for i, e in enumerate(te_eps):
        p_te[i] = two._propagate(theta, two.event_features(e), bn)

    xptr = np.asarray([template_features(e)[0] for e in tr_eps]); xpte = np.asarray([template_features(e)[0] for e in te_eps])
    xdtr = np.asarray([template_features(e)[1] for e in tr_eps]); xdte = np.asarray([template_features(e)[1] for e in te_eps])
    policy_model = fair.fit_logistic(xptr, ytr, args.seed + 8)
    deviation_model = fair.fit_logistic(xdtr, ytr, args.seed + 9)
    p_policy_tr = fair.predict_logistic(policy_model, xptr); p_policy_te = fair.predict_logistic(policy_model, xpte)
    p_dev_tr = fair.predict_logistic(deviation_model, xdtr); p_dev_te = fair.predict_logistic(deviation_model, xdte)
    # Fusion uses named evidence scores, not the full unrestricted feature
    # vector.  The head therefore remains inspectable and train-only.
    coverage_tr = np.asarray([1.0 - xdtr[i, 1] for i in range(len(xdtr))])
    coverage_te = np.asarray([1.0 - xdte[i, 1] for i in range(len(xdte))])
    fusion_xtr = np.c_[logit(p_tr), logit(p_policy_tr), logit(p_dev_tr), coverage_tr, xptr[:, 2], xdtr[:, 5]]
    fusion_xte = np.c_[logit(p_te), logit(p_policy_te), logit(p_dev_te), coverage_te, xpte[:, 2], xdte[:, 5]]
    fusion_model = fair.fit_logistic(fusion_xtr, ytr, args.seed + 10)
    p_fusion_tr = fair.predict_logistic(fusion_model, fusion_xtr); p_fusion_te = fair.predict_logistic(fusion_model, fusion_xte)
    thresholds = {"two_layer": best_threshold(ytr, p_tr), "policy_template": best_threshold(ytr, p_policy_tr), "deviation_template": best_threshold(ytr, p_dev_tr), "policy_deviation_fusion": best_threshold(ytr, p_fusion_tr)}
    ps = {"two_layer": p_te, "policy_template": p_policy_te, "deviation_template": p_dev_te, "policy_deviation_fusion": p_fusion_te}
    results = {name: metric(yte, pp, thresholds[name]) for name, pp in ps.items()}

    # Mechanism metadata is joined only after inference, for diagnosis.
    mechanism = np.asarray([labels[e["episode_id"]]["mechanism"] for e in te_eps])
    by_mechanism = {}
    for m in sorted(set(mechanism)):
        mask = mechanism == m
        by_mechanism[m] = {name: metric(yte[mask], pp[mask], thresholds[name]) for name, pp in ps.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "model": {"name": "PolicyDeviationTemplateMonitor-v1", "policy_head": "privileged authorization/capability mismatch templates", "deviation_head": "normative workflow deviation + unexpected-edge bridge templates", "fusion": "train-only logistic over named base/template scores", "causal_claim": False, "public_field_contract": "events + workflow_dag + agents only; labels and hidden fields excluded"},
        "dataset": {"input": str(args.input), "labels": str(args.labels), "n": len(eps), "positive": int(y.sum()), "negative": int(len(y) - y.sum())},
        "split": {**split, "train_count": int(len(train)), "test_count": int(len(test))},
        "feature_names": {"policy": list(POLICY_NAMES), "deviation": list(DEVIATION_NAMES), "fusion": ["logit(two_layer)", "logit(policy_template)", "logit(deviation_template)", "runtime_coverage", "mismatch_rate", "unexpected_bridge_rate"]},
        "metrics": results,
        "mechanism_metrics": by_mechanism,
        "parameters": {"optimizer": optimizer, "policy": [float(v) for v in policy_model[0]], "deviation": [float(v) for v in deviation_model[0]], "fusion": [float(v) for v in fusion_model[0]]},
        "limitations": ["The benchmark is synthetic and uses noisy public permission telemetry.", "workflow_dag is normative context, not a discovered causal graph.", "Template evidence is a risk score; permission_mismatch is not treated as a terminal outcome.", "Strict topology-family holdout is used, but external validity still requires real API-agent traces."],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Policy/deviation risk templates v1", "", "在相同公开字段契约和 topology-family group holdout 下，新增授权异常与规范偏离模板。", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in results.items():
        lines.append(f"| {name} | {value['f1']:.3f} | {value['precision']:.3f} | {value['recall']:.3f} | {value['auroc'] if value['auroc'] is not None else 'n/a'} | {value['auprc']:.3f} | {value['brier']:.3f} |")
    lines += ["", "## 按机制测试集结果", "", "| mechanism | method | F1 | AUROC | AUPRC |", "|---|---|---:|---:|---:|"]
    for m, vals in by_mechanism.items():
        for name, value in vals.items():
            lines.append(f"| {m} | {name} | {value['f1']:.3f} | {value['auroc'] if value['auroc'] is not None else 'n/a'} | {value['auprc']:.3f} |")
    lines += ["", "## 解释", "- policy_template 只看公开的权限不匹配、特权调用及其运行时可达性证据。", "- deviation_template 比较规范 workflow_dag 与实际 message/delegation 图，并显式记录 unexpected-edge 是否连接不可信源和特权汇。", "- 两个模板不读取机制标签；机制分解仅用于错误分析。"]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as h:
        for i, e in enumerate(te_eps):
            pol, dev, raw = template_features(e)
            h.write(json.dumps({"episode_id": e["episode_id"], "label": int(yte[i]), "mechanism": str(mechanism[i]), "two_layer": float(p_te[i]), "policy_template": float(p_policy_te[i]), "deviation_template": float(p_dev_te[i]), "policy_deviation_fusion": float(p_fusion_te[i]), "policy_features": pol.tolist(), "deviation_features": dev.tolist(), "evidence": raw}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
