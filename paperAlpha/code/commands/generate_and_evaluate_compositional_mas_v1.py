"""Controlled MAS composition benchmark and transparent risk baselines.

The episode label is generated from a structural rule (reachability from an
untrusted ingress agent to a privileged sink across observed communication
edges).  No terminal ``risk``/``violation`` event is emitted.  This is a
synthetic sanity benchmark for testing whether a monitor uses composition,
not evidence of performance on a real deployment.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

AGENTS = ["ingress", "planner", "reviewer", "executor"]
WORKFLOW_EDGES = [("ingress", "planner"), ("planner", "reviewer"), ("reviewer", "executor")]
PRIV_TOOLS = {"payments.write", "filesystem.write", "shell.exec"}
PUBLIC_TOOLS = {"search.read", "calculator.read", "filesystem.read"}


def _event(seq, actor, typ, target=None, tool=None, data_class="clean", edge_kind=None):
    return {
        "event_id": f"e{seq:03d}", "sequence": seq, "actor": actor,
        "event_type": typ, "target": target, "tool": tool,
        "data_class": data_class, "edge_kind": edge_kind,
    }


def _has_cross_path(events):
    sources = {e["actor"] for e in events if e["event_type"] == "input_received" and e["data_class"] == "untrusted"}
    sinks = {e["actor"] for e in events if e["event_type"] == "tool_call" and e["tool"] in PRIV_TOOLS}
    graph = defaultdict(set)
    for e in events:
        # Only an edge carrying the untrusted data can propagate the hazard;
        # a clean post-sanitization message is intentionally not a path edge.
        if e["event_type"] in {"message_sent", "delegation"} and e.get("target") and e.get("data_class") == "untrusted":
            graph[e["actor"]].add(e["target"])
    seen = set(sources)
    todo = list(sources)
    while todo:
        cur = todo.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt); todo.append(nxt)
    return bool(sinks & seen), sources, sinks, graph


def generate_episode(i, rng):
    # Negatives deliberately contain both ingredients, but disconnect them.
    family = rng.choices(
        ["composition_attack", "direct_attack", "disconnected", "sanitized", "benign_workflow"],
        weights=[25, 15, 25, 20, 15], k=1)[0]
    events = []
    seq = 0
    # Every family has the same two ingredients (untrusted input + a privileged
    # call); only their graph connectivity determines the independent label.
    events.append(_event(seq, "ingress", "input_received", data_class="untrusted")); seq += 1

    if family == "composition_attack":
        for a, b in WORKFLOW_EDGES:
            events.append(_event(seq, a, "message_sent", target=b, data_class="untrusted", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "executor", "tool_call", tool=rng.choice(sorted(PRIV_TOOLS)), data_class="untrusted")); seq += 1
    elif family == "direct_attack":
        events.append(_event(seq, "ingress", "message_sent", target="executor", data_class="untrusted", edge_kind="unexpected")); seq += 1
        events.append(_event(seq, "executor", "tool_call", tool=rng.choice(sorted(PRIV_TOOLS)), data_class="untrusted")); seq += 1
    elif family == "disconnected":
        # Untrusted input is dropped; executor independently performs a valid write.
        events.append(_event(seq, "ingress", "sanitization", data_class="untrusted")); seq += 1
        events.append(_event(seq, "reviewer", "message_sent", target="executor", data_class="clean", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "executor", "tool_call", tool=rng.choice(sorted(PRIV_TOOLS)), data_class="clean")); seq += 1
    elif family == "sanitized":
        events.append(_event(seq, "ingress", "sanitization", data_class="clean")); seq += 1
        events.append(_event(seq, "ingress", "message_sent", target="planner", data_class="clean", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "planner", "message_sent", target="reviewer", data_class="clean", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "reviewer", "message_sent", target="executor", data_class="clean", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "executor", "tool_call", tool=rng.choice(sorted(PRIV_TOOLS)), data_class="clean")); seq += 1
    else:  # benign_workflow
        for a, b in WORKFLOW_EDGES:
            events.append(_event(seq, a, "message_sent", target=b, data_class="clean", edge_kind="workflow")); seq += 1
        events.append(_event(seq, "executor", "tool_call", tool=rng.choice(sorted(PRIV_TOOLS)), data_class="clean")); seq += 1

    # Add random decoy activity to equalize aggregate counts.  Decoys can be
    # cross-agent edges, but are sampled so edge count alone does not reveal
    # the label; only reachability through the explicit event graph does.
    decoy_n = rng.randint(0, 4)
    for _ in range(decoy_n):
        actor = rng.choice(AGENTS)
        if rng.random() < .55:
            target = rng.choice([a for a in AGENTS if a != actor])
            events.append(_event(seq, actor, "message_sent", target=target, data_class="clean", edge_kind="decoy"))
        else:
            events.append(_event(seq, actor, "tool_call", tool=rng.choice(sorted(PUBLIC_TOOLS)), data_class="clean"))
        seq += 1

    label, sources, sinks, graph = _has_cross_path(events)
    dyn_edges = [{"source": e["actor"], "target": e["target"],
                  "sequence": e["sequence"], "edge_kind": e.get("edge_kind"),
                  "data_class": e.get("data_class")}
                 for e in events if e["event_type"] in {"message_sent", "delegation"} and e.get("target")]
    return {
        "episode_id": f"syn-mas-{i:05d}", "scenario_family": family,
        "workflow_dag": {"nodes": AGENTS, "edges": [list(x) for x in WORKFLOW_EDGES]},
        "dynamic_graph": {"nodes": sorted({a for e in events for a in ([e["actor"]] + ([e["target"]] if e.get("target") else []))}), "edges": dyn_edges,
                           "edge_policy": "explicit message/delegation only; sequence is metadata, not causality"},
        "events": events, "label": int(label),
        "label_rule": "reachable(untrusted_ingress, privileged_sink) via explicit cross-agent message/delegation edges",
        "label_evidence": {"source_agents": sorted(sources), "sink_agents": sorted(sinks)},
    }


def features(ep):
    ev = ep["events"]
    agents = set()
    for e in ev:
        agents.add(e["actor"])
        if e.get("target"): agents.add(e["target"])
    input_u = {e["actor"] for e in ev if e["event_type"] == "input_received" and e["data_class"] == "untrusted"}
    priv = {e["actor"] for e in ev if e["event_type"] == "tool_call" and e.get("tool") in PRIV_TOOLS}
    graph = defaultdict(set)
    cross = 0
    for e in ev:
        if e["event_type"] in {"message_sent", "delegation"} and e.get("target") and e.get("data_class") == "untrusted":
            graph[e["actor"]].add(e["target"]); cross += e["actor"] != e["target"]
    seen = set(input_u); q = list(input_u)
    while q:
        for nxt in graph[q.pop()]:
            if nxt not in seen: seen.add(nxt); q.append(nxt)
    path = float(bool(priv & seen))
    # local features intentionally omit graph reachability for single-agent baselines.
    return {
        "event_count": len(ev), "agent_count": len(agents), "input_untrusted": len(input_u),
        "privileged_calls": len(priv), "message_count": sum(e["event_type"] == "message_sent" for e in ev),
        "cross_agent_edges": cross, "path_reachability": path,
        "local_overlap": float(bool(input_u & priv)),
    }


def evaluate(y, scores, threshold=0.5):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)), "threshold": threshold}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/compositional_mas_v1")); args = ap.parse_args()
    rng = random.Random(args.seed)
    eps = [generate_episode(i, rng) for i in range(args.n)]
    idx = np.arange(len(eps)); y = np.array([e["label"] for e in eps]); train_idx, test_idx = train_test_split(idx, test_size=.3, random_state=args.seed, stratify=y)
    rows = [features(e) for e in eps]
    # Fit flat logistic only on non-graph observables. Threshold is selected on train.
    # Flat baseline gets aggregate counts but not graph connectivity or local
    # source/sink overlap; this tests the value of explicit composition.
    flat_names = ["event_count", "agent_count", "input_untrusted", "privileged_calls", "message_count"]
    X = np.array([[r[k] for k in flat_names] for r in rows]); clf = LogisticRegression(max_iter=1000, random_state=args.seed).fit(X[train_idx], y[train_idx])
    flat_test = clf.predict_proba(X[test_idx])[:, 1]
    # local/single-agent BN: max calibrated local co-occurrence (cannot see path).
    local_train = np.array([rows[i]["local_overlap"] for i in train_idx]); local_test = np.array([rows[i]["local_overlap"] for i in test_idx])
    # Dynamic graph BN uses only explicit reachability and a smoothed prior.
    graph_train = np.array([rows[i]["path_reachability"] for i in train_idx]); graph_test = np.array([rows[i]["path_reachability"] for i in test_idx])
    prior = (y[train_idx].sum() + 1.0) / (len(train_idx) + 2.0)
    # hierarchical fusion: graph composition gates local node evidence.
    fusion_test = graph_test * (0.75 + 0.25 * local_test) + (1 - graph_test) * prior * 0.05
    event_test = np.array([rows[i]["event_count"] for i in test_idx], dtype=float)
    # pick train-only thresholds maximizing F1, then apply unchanged to test.
    def best_thr(scores):
        vals = np.unique(np.r_[0.0, scores, 1.0]); return max(vals, key=lambda t: f1_score(y[train_idx], scores >= t, zero_division=0))
    event_train = np.array([rows[i]["event_count"] for i in train_idx], dtype=float); event_train = event_train / max(event_train.max(), 1)
    event_test_n = event_test / max(event_test.max(), 1)
    local_thr = best_thr(local_train); graph_thr = best_thr(graph_train); event_thr = best_thr(event_train)
    # Logistic threshold chosen on train scores.
    flat_train = clf.predict_proba(X[train_idx])[:, 1]; flat_thr = best_thr(flat_train)
    metrics = {
        "event_count": evaluate(y[test_idx], event_test_n, event_thr),
        "single_agent_bn": evaluate(y[test_idx], local_test, local_thr),
        "flat_logistic": evaluate(y[test_idx], flat_test, flat_thr),
        "dynamic_graph_bn": evaluate(y[test_idx], graph_test, graph_thr),
        "hierarchical_fusion": evaluate(y[test_idx], fusion_test, 0.5),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in eps: f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.out / "features.jsonl").open("w", encoding="utf-8") as f:
        for e, r in zip(eps, rows): f.write(json.dumps({"episode_id": e["episode_id"], "label": e["label"], **r}, sort_keys=True) + "\n")
    report = {"dataset": {"n": len(eps), "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "seed": args.seed, "label_source": "independent structural reachability rule; no terminal risk event in input"}, "split": {"train": len(train_idx), "test": len(test_idx)}, "metrics": metrics, "feature_names": flat_names}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Compositional MAS v1", "", "这是一个可控的合成基准：标签由‘不可信入口到特权工具的跨 Agent 可达性’独立生成；事件中没有终态 risk/violation 字段。结果只用于验证组合建模管线，不代表真实系统效果。", "", f"- episodes: {len(eps)} (positive={int(y.sum())}, negative={int(len(y)-y.sum())})", f"- split: train={len(train_idx)}, test={len(test_idx)}", "", "| method | F1 | precision | recall | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items(): md.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} |")
    md += ["", "## Interpretation", "- `dynamic_graph_bn` 显式检查跨 Agent 图上的路径；`hierarchical_fusion` 用路径证据门控节点证据。", "- `single_agent_bn` 只看同一 Agent 是否同时出现不可信输入与特权调用，无法识别跨 Agent 组合。", "- 所有阈值和 logistic 参数只在训练集选择；测试集仅用于一次报告。", "- 该基准的标签规则是合成的，后续须在独立人工标注或真实运行轨迹上复现。"]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
