"""Normalize HarnessAudit multi-agent task specifications for graph experiments."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value is None else [value])


def _norm_access_rule(rule: dict[str, Any]) -> dict[str, Any]:
    match = rule.get("match") or {}
    return {
        "role": rule.get("role"),
        "access_type": rule.get("access_type"),
        "match": match,
        "severity": rule.get("severity", "unspecified"),
        "description": rule.get("description", ""),
        "is_forbidden": bool(match.get("forbidden_to")) or rule.get("access_type") == "information.data_leak",
    }


def normalize_task(path: Path, root: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    agents = []
    role_tools: dict[str, list[str]] = {}
    for a in _as_list(raw.get("agents")):
        if not isinstance(a, dict):
            continue
        role = a.get("role") or "unknown_role"
        tn = a.get("tool_necessity") or {}
        useful = [str(x) for x in _as_list(tn.get("useful"))]
        forbidden = [str(x) for x in _as_list(tn.get("forbidden"))]
        unnecessary = [str(x) for x in _as_list(tn.get("unnecessary"))]
        role_tools[role] = useful
        agents.append({"role": role, "description": a.get("description", ""), "tool_necessity": {"useful": useful, "forbidden": forbidden, "unnecessary": unnecessary}})

    gt_paths = raw.get("ground_truth_tool_paths") or {}
    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str]] = set()
    for role, paths in gt_paths.items():
        for path_seq in _as_list(paths):
            seq = [str(x) for x in _as_list(path_seq)]
            prev = None
            for idx, tool in enumerate(seq):
                node_id = f"{role}:{tool}"
                nodes.setdefault(node_id, {"id": node_id, "role": role, "tool": tool, "position": idx})
                if prev is not None and prev != node_id:
                    edges.add((prev, node_id))
                prev = node_id
    for role, tools in role_tools.items():
        for idx, tool in enumerate(tools):
            node_id = f"{role}:{tool}"
            nodes.setdefault(node_id, {"id": node_id, "role": role, "tool": tool, "position": idx})

    checkpoints = _as_list(raw.get("completion_checkpoints"))
    checkpoint_tools = [cp.get("tool_name") for cp in checkpoints if isinstance(cp, dict)]
    workflow_stages = [{"name": cp.get("name"), "tool": cp.get("tool_name"), "type": cp.get("type"), "weight": cp.get("weight")} for cp in checkpoints if isinstance(cp, dict)]
    all_tools = sorted({n["tool"] for n in nodes.values()} | {str(t) for t in checkpoint_tools if t})
    access_rules = [_norm_access_rule(r) for r in _as_list(raw.get("access_rules")) if isinstance(r, dict)]
    return {
        "source": {"path": str(path.relative_to(root)).replace("\\", "/"), "format": "harnessaudit_task_yaml"},
        "task_id": raw.get("task_id") or path.stem,
        "domain": raw.get("domain"), "category": raw.get("category"), "modality": raw.get("modality"),
        "goal": raw.get("goal", ""), "metadata": raw.get("metadata") or {}, "agents": agents, "tools": all_tools,
        "access_rules": access_rules, "completion_checkpoints": checkpoints, "workflow_stages": workflow_stages,
        "workflow_dag": {"nodes": list(nodes.values()), "edges": [{"source": a, "target": b, "relation": "declared_role_sequence"} for a, b in sorted(edges)], "construction": "ground_truth_tool_paths plus useful-tool nodes; no LLM inference"},
        "ground_truth_tool_paths": gt_paths,
    }


def build_report(records: list[dict[str, Any]], input_dir: Path, output_dir: Path) -> str:
    domains = Counter(r.get("domain") for r in records)
    categories = Counter(r.get("category") for r in records)
    roles = Counter(a["role"] for r in records for a in r["agents"])
    tools = Counter(t for r in records for t in r["tools"])
    access = Counter(rule.get("access_type") for r in records for rule in r["access_rules"])
    severities = Counter(rule.get("severity") for r in records for rule in r["access_rules"])
    coverage = {"goal": sum(bool(r.get("goal")) for r in records), "agents": sum(bool(r.get("agents")) for r in records), "access_rules": sum(bool(r.get("access_rules")) for r in records), "completion_checkpoints": sum(bool(r.get("completion_checkpoints")) for r in records), "workflow_stages": sum(bool(r.get("workflow_stages")) for r in records), "ground_truth_tool_paths": sum(bool(r.get("ground_truth_tool_paths")) for r in records), "workflow_dag_nodes": sum(bool(r["workflow_dag"]["nodes"]) for r in records)}
    lines = ["# HarnessAudit multi-agent task adapter report", "", f"- Input: `{input_dir}`", f"- Task YAML files normalized: **{len(records)}**", f"- Output: `{output_dir}`", "", "## Coverage", "", "| Field | Tasks containing field | Coverage |", "|---|---:|---:|"]
    for k, n in coverage.items():
        lines.append(f"| `{k}` | {n} | {n / len(records):.1%} |" if records else f"| `{k}` | 0 | 0.0% |")
    lines += ["", "## Cardinalities", "", f"- Domains: {dict(domains)}", f"- Categories (top 15): {dict(categories.most_common(15))}", f"- Distinct roles: {len(roles)}; occurrences: {dict(roles)}", f"- Distinct tools: {len(tools)}; top 20: {dict(tools.most_common(20))}", f"- Access-rule types: {dict(access)}", f"- Access-rule severities: {dict(severities)}", "", "## Normalization contract", "", "Each JSONL record keeps the source-relative path, original goal/metadata, agent tool-necessity declarations, access rules, completion checkpoints, and a deterministic workflow DAG. DAG nodes are `role:tool`; edges come only from declared `ground_truth_tool_paths`, so this adapter does not claim to discover causal structure. Useful tools absent from a ground-truth path are retained as isolated template candidates.", "", "## Limitations", "", "- This is a task-specification adapter, not an execution-trace parser; it does not run HarnessAudit or call an LLM.", "- The resulting DAG is normative scaffolding for Layer 1. Layer 2 should be built from runtime traces when available.", "- Access rules are preserved as policy evidence, not outcome labels."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=(Path(__file__).resolve().parents[2] / "code/external/HarnessAudit/multi_agent/tasks"))
    ap.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[2] / "data/processed/mas_adapter"))
    args = ap.parse_args()
    files = sorted(args.input.rglob("*.yaml"))
    root = args.input.parents[3]
    records = [normalize_task(p, root) for p in files]
    args.output.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output / "normalized_tasks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Normalized HarnessAudit MAS task", "type": "object", "required": ["source", "task_id", "agents", "access_rules", "workflow_dag"], "properties": {"task_id": {"type": "string"}, "domain": {"type": ["string", "null"]}, "category": {"type": ["string", "null"]}, "agents": {"type": "array"}, "access_rules": {"type": "array"}, "completion_checkpoints": {"type": "array"}, "workflow_stages": {"type": "array"}, "workflow_dag": {"type": "object", "required": ["nodes", "edges"]}}}
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "ADAPTER_REPORT.md").write_text(build_report(records, args.input, args.output), encoding="utf-8")
    print(json.dumps({"tasks": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
