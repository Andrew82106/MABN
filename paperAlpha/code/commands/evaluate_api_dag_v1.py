"""Small API-assisted workflow-DAG diagnostic.

This evaluates whether a low-cost language model can recover the declared
workflow scaffold from task specifications.  It is deliberately separate
from the risk-monitor score: the reference DAG is normative task metadata,
not a causal ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def _call(endpoint: str, key: str, model: str, prompt: str, max_tokens: int, timeout: int) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a workflow analyst. Return only valid JSON; do not use markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            return text, payload
        except Exception as exc:  # network/provider failures are retained in the report
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last))


def _task_prompt(task: dict[str, Any]) -> str:
    roles = [{"role": a.get("role"), "description": a.get("description"), "useful": a.get("tool_necessity", {}).get("useful", [])} for a in task.get("agents", [])]
    return json.dumps({
        "task_id": task.get("task_id"),
        "goal": task.get("goal"),
        "roles": roles,
        "allowed_tools": sorted(task.get("tools", [])),
    }, ensure_ascii=False)


def _direct_prompt(task: dict[str, Any]) -> str:
    return (
        "Recover the task's normative workflow DAG from the specification below. "
        "Create one node for each distinct role/tool operation that is part of a required path. "
        "Do not invent tools. Each node must use an allowed role and an allowed tool exactly. "
        "Add an edge only when the specification states or strongly implies that the source operation "
        "precedes the target operation for the same role. The result must be acyclic. "
        "Return exactly {\"nodes\":[{\"role\":...,\"tool\":...}],\"edges\":[{\"source\":\"role:tool\",\"target\":\"role:tool\"}]}\n\n"
        + _task_prompt(task)
    )


def _nodes_prompt(task: dict[str, Any]) -> str:
    return (
        "Extract the required workflow operations from this task specification. "
        "Return exactly {\"nodes\":[{\"role\":...,\"tool\":...}]} . "
        "Use only exact role and tool strings from the input; omit isolated optional tools. "
        "Do not output edges.\n\n" + _task_prompt(task)
    )


def _edges_prompt(task: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    return (
        "Given the task specification and the extracted nodes below, infer only explicit or strongly implied "
        "within-role precedence relations. Return exactly {\"edges\":[{\"source\":\"role:tool\",\"target\":\"role:tool\"}]} . "
        "Use only listed node ids, never add nodes, and keep the graph acyclic.\n\n"
        + _task_prompt(task)
        + "\nEXTRACTED_NODES=" + json.dumps(nodes, ensure_ascii=False)
    )


def _node_set(graph: dict[str, Any]) -> set[str]:
    out = set()
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and node.get("role") and node.get("tool"):
            out.add(f"{node['role']}:{node['tool']}")
    return out


def _edge_set(graph: dict[str, Any]) -> set[tuple[str, str]]:
    out = set()
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("source") and edge.get("target"):
            source, target = str(edge["source"]), str(edge["target"])
            # Some models repeat the field label in the endpoint, e.g.
            # role:agent:tool.  The canonical endpoint is agent:tool.
            if source.startswith("role:"):
                source = source[len("role:") :]
            if target.startswith("role:"):
                target = target[len("role:") :]
            out.add((source, target))
    return out


def _reference(task: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    graph = task.get("workflow_dag", {})
    return _node_set(graph), _edge_set(graph)


def _score(pred: set[Any], truth: set[Any]) -> dict[str, float | int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not truth else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _is_dag(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        if src not in nodes or dst not in nodes:
            return False
        if dst not in adjacency[src]:
            adjacency[src].add(dst)
            indegree[dst] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def _mean(rows: list[dict[str, Any]], key: str, section: str) -> float:
    vals = [float(row[section][key]) for row in rows if row.get(section, {}).get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("PAPERALPHA_API_ENDPOINT", "http://localhost:58661/v1/chat/completions"))
    ap.add_argument("--key", default=os.environ.get("PAPERALPHA_LOCAL_API_KEY", ""))
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--n-per-domain", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=str(ROOT / 'experiments/model_mechanisms/runs/submission__dag_api_v1'))
    ap.add_argument("--recompute", action="store_true", help="recompute metrics from an existing progress.json")
    args = ap.parse_args()
    out = Path(args.out)
    if args.recompute:
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        rows = progress["rows"]
        selected = rows
        task_map = {json.loads(line).get("task_id"): json.loads(line) for line in (ROOT / 'data/processed/mas_adapter/normalized_tasks.jsonl').read_text(encoding="utf-8").splitlines() if line.strip()}
        for row in rows:
            if row.get("status") != "ok":
                continue
            task = task_map[row["task_id"]]
            truth_nodes, truth_edges = _reference(task)
            direct = json.loads((out / f"{row['task_id']}_direct.json").read_text(encoding="utf-8"))["parsed"]
            staged = json.loads((out / f"{row['task_id']}_staged.json").read_text(encoding="utf-8"))["parsed"]
            direct_nodes, direct_edges = _node_set(direct), _edge_set(direct)
            staged_nodes = set(staged.get("nodes", []))
            staged_graph = {"nodes": [{"role": node.split(":", 1)[0], "tool": node.split(":", 1)[1]} for node in staged_nodes], "edges": [{"source": edge[0], "target": edge[1]} for edge in staged.get("edges", [])]}
            staged_edges = _edge_set(staged_graph)
            row["direct"] = {"nodes": sorted(direct_nodes), "edges": sorted([list(e) for e in direct_edges]), "node_score": _score(direct_nodes, truth_nodes), "edge_score": _score(direct_edges, truth_edges), "is_dag": _is_dag(direct_nodes, direct_edges)}
            row["staged"] = {"nodes": sorted(staged_nodes), "edges": sorted([list(e) for e in staged_edges]), "node_score": _score(staged_nodes, truth_nodes), "edge_score": _score(staged_edges, truth_edges), "is_dag": _is_dag(staged_nodes, staged_edges)}
    elif not args.key:
        raise SystemExit("missing --key or PAPERALPHA_LOCAL_API_KEY")
    else:
        tasks = [json.loads(line) for line in (ROOT / 'data/processed/mas_adapter/normalized_tasks.jsonl').read_text(encoding="utf-8").splitlines() if line.strip()]
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in tasks:
            by_domain[str(task.get("domain", "unknown"))].append(task)
        selected = []
        for domain in sorted(by_domain):
            selected.extend(sorted(by_domain[domain], key=lambda t: t.get("task_id", ""))[: args.n_per_domain])
        out.mkdir(parents=True, exist_ok=True)
        rows = []
    if not args.recompute:
      for index, task in enumerate(selected, 1):
        row: dict[str, Any] = {"task_id": task.get("task_id"), "domain": task.get("domain"), "status": "ok"}
        truth_nodes, truth_edges = _reference(task)
        try:
            direct_text, direct_raw = _call(args.endpoint, args.key, args.model, _direct_prompt(task), 1200, args.timeout)
            direct = _json_from_text(direct_text)
            row["direct"] = {"nodes": sorted(_node_set(direct)), "edges": sorted([list(e) for e in _edge_set(direct)]), "node_score": _score(_node_set(direct), truth_nodes), "edge_score": _score(_edge_set(direct), truth_edges), "is_dag": _is_dag(_node_set(direct), _edge_set(direct))}
            (out / f"{task.get('task_id')}_direct.json").write_text(json.dumps({"response": direct_raw, "parsed": direct}, ensure_ascii=False, indent=2), encoding="utf-8")

            node_text, node_raw = _call(args.endpoint, args.key, args.model, _nodes_prompt(task), 800, args.timeout)
            node_graph = _json_from_text(node_text)
            nodes = sorted(_node_set(node_graph))
            edge_text, edge_raw = _call(args.endpoint, args.key, args.model, _edges_prompt(task, [{"role": n.split(":", 1)[0], "tool": n.split(":", 1)[1]} for n in nodes]), 800, args.timeout)
            edge_graph = _json_from_text(edge_text)
            staged = {"nodes": nodes, "edges": sorted([list(e) for e in _edge_set(edge_graph)])}
            row["staged"] = {"nodes": nodes, "edges": staged["edges"], "node_score": _score(set(nodes), truth_nodes), "edge_score": _score(_edge_set(edge_graph), truth_edges), "is_dag": _is_dag(set(nodes), _edge_set(edge_graph))}
            (out / f"{task.get('task_id')}_staged.json").write_text(json.dumps({"node_response": node_raw, "edge_response": edge_raw, "parsed": staged}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)
        (out / "progress.json").write_text(json.dumps({"completed": index, "total": len(selected), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    methods = {"direct": [r for r in rows if r.get("status") == "ok"], "staged": [r for r in rows if r.get("status") == "ok"]}
    summary = {"model": args.model, "endpoint": args.endpoint, "selected": len(selected), "completed": sum(r.get("status") == "ok" for r in rows), "errors": [r for r in rows if r.get("status") != "ok"], "methods": {}}
    for method in methods:
        valid = methods[method]
        summary["methods"][method] = {
            "n": len(valid),
            "node_f1": sum(float(r[method]["node_score"]["f1"]) for r in valid) / len(valid) if valid else 0.0,
            "edge_f1": sum(float(r[method]["edge_score"]["f1"]) for r in valid) / len(valid) if valid else 0.0,
            "node_precision": sum(float(r[method]["node_score"]["precision"]) for r in valid) / len(valid) if valid else 0.0,
            "node_recall": sum(float(r[method]["node_score"]["recall"]) for r in valid) / len(valid) if valid else 0.0,
            "edge_precision": sum(float(r[method]["edge_score"]["precision"]) for r in valid) / len(valid) if valid else 0.0,
            "edge_recall": sum(float(r[method]["edge_score"]["recall"]) for r in valid) / len(valid) if valid else 0.0,
            "dag_rate": sum(bool(r[method]["is_dag"]) for r in valid) / len(valid) if valid else 0.0,
        }
    summary["limitations"] = [
        "Reference workflow_dag is normative task metadata, not a causal graph or causal discovery ground truth.",
        "The sample is a low-cost API diagnostic on a small, deterministic task subset; it is not the final risk-monitor evaluation.",
        "Errors and malformed outputs are retained; no outcome-based case replacement is performed.",
    ]
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# API-assisted DAG diagnostic v1", "", f"Model: `{args.model}`; task specifications: {len(selected)}; completed: {summary['completed']}", "", "| method | node F1 | edge F1 | node precision | node recall | edge precision | edge recall | DAG rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in summary["methods"].items():
        report.append(f"| {method} | {values['node_f1']:.3f} | {values['edge_f1']:.3f} | {values['node_precision']:.3f} | {values['node_recall']:.3f} | {values['edge_precision']:.3f} | {values['edge_recall']:.3f} | {values['dag_rate']:.3f} |")
    report += ["", "该结果只说明模型能否从任务规范恢复一个可用的工作流脚手架；它不证明发现了真实因果关系，也不直接替代风险模型输入。"]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
