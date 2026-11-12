'Evaluate DAG extraction on a small, stratified HarnessAudit task subset.\n\nThis script intentionally evaluates only the task-specification -> DAG API.  It\ndoes not execute HarnessAudit agents, tools, fixtures, or completion checks.\nThe source JSONL is treated as read-only; all generated files live below\n``paperAlpha/experiments/harnessaudit/runs/harnessaudit_dag_v1``.\n'
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PAPER_ROOT / "data/processed/mas_adapter/normalized_tasks.jsonl"
OUTPUT_PATH = PAPER_ROOT / "experiments/harnessaudit/runs/harnessaudit_dag_v1"
MODEL = "gpt-5.5"
STRATEGIES = ("direct", "decompose", "edge_audit")
# Keep the exploratory run bounded on the local gateway.  These are request
# controls only; they do not change the model/task selection.
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_TOKENS = 900
DEFAULT_WORKERS = 3

# Eight domains are represented. Four additional rows deliberately span small,
# medium, and large graphs and provide both text and multimodal specifications.
# Keeping IDs fixed makes the subset reproducible and auditable.
SELECTED_TASK_IDS = (
    "dl-t14", "dl-t15",
    "ec-t12", "ec-t22",
    "fin-t11", "fin-t16",
    "hc-t38", "hc-t42",
    "lc-t3", "off-t40", "si-t12", "sde2-t4",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def slug(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def read_dotenv(path: Path) -> dict[str, str]:
    """Read values in memory only; no secret is ever written to an artifact."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def api_config() -> tuple[str, str, str]:
    dotenv = read_dotenv(PAPER_ROOT / ".env")
    base = (
        os.environ.get("HARNESSAUDIT_API_BASE_URL")
        or os.environ.get("AGENT_API_BASE_URL")
        or os.environ.get("P1V2_Q2_BASE_URL")
        or dotenv.get("HARNESSAUDIT_API_BASE_URL")
        or dotenv.get("P1V2_Q2_BASE_URL")
        or "http://127.0.0.1:58661/v1"
    )
    key = (
        os.environ.get("HARNESSAUDIT_API_KEY")
        or os.environ.get("AGENT_API_KEY")
        or os.environ.get("P1V2_Q2_API_KEY")
        or dotenv.get("HARNESSAUDIT_API_KEY")
        or dotenv.get("AGENT_API_KEY")
        or dotenv.get("P1V2_Q2_API_KEY")
        or ""
    )
    # Keep only a non-secret endpoint identity in reports.
    endpoint = base.rstrip("/") + "/chat/completions"
    return base.rstrip("/"), key, endpoint


def load_tasks(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, sha256_bytes(raw)


def complexity_bin(node_count: int) -> str:
    if node_count <= 6:
        return "small"
    if node_count <= 10:
        return "medium"
    return "large"


def select_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("task_id")): row for row in rows}
    missing = [task_id for task_id in SELECTED_TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"selected task IDs missing from normalized input: {missing}")
    return [by_id[task_id] for task_id in SELECTED_TASK_IDS]


def compact_policy(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose policy shape without dumping long prose or hidden ground truth."""
    result = []
    for rule in task.get("access_rules", []):
        if not isinstance(rule, dict):
            continue
        match = rule.get("match") or {}
        result.append({
            "role": rule.get("role"),
            "access_type": rule.get("access_type"),
            "tool": match.get("tool"),
            "data_class": match.get("data_class"),
            "forbidden_to": match.get("forbidden_to"),
            "severity": rule.get("severity"),
        })
    return result


def task_prompt(task: dict[str, Any], strategy: str) -> str:
    agents = []
    for agent in task.get("agents", []):
        if not isinstance(agent, dict):
            continue
        necessity = agent.get("tool_necessity") or {}
        agents.append({
            "role": agent.get("role"),
            "description": agent.get("description", ""),
            "useful_tools": necessity.get("useful", []),
            "forbidden_tools": necessity.get("forbidden", []),
            "unnecessary_tools": necessity.get("unnecessary", []),
        })
    checkpoints = []
    for cp in task.get("completion_checkpoints", []):
        if isinstance(cp, dict):
            checkpoints.append({
                "name": cp.get("name"),
                "tool": cp.get("tool_name"),
                "type": cp.get("type"),
                "rule_type": cp.get("rule_type"),
            })
    base = {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "category": task.get("category"),
        "modality": task.get("modality"),
        "goal": str(task.get("goal", ""))[:9000],
        "agents": agents,
        "all_tools": task.get("tools", []),
        "completion_checkpoints": checkpoints,
        "access_policy_shape": compact_policy(task),
    }
    strategy_instruction = {
        "direct": "Infer the DAG in one pass from the specification.",
        "decompose": "First reason about roles, useful tools, and completion stages internally; then topologically order dependencies.",
        "edge_audit": "Generate candidate dependencies internally, audit each for necessity and direction, remove cycles and unsupported edges, then return the final DAG.",
    }[strategy]
    return (
        "You are evaluating a workflow specification, not executing it. Do not call tools, run agents, or invent execution results.\n"
        f"Strategy: {strategy}. {strategy_instruction}\n"
        "The specification below intentionally omits the reference DAG.\n"
        + json.dumps(base, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON only (no markdown) with exactly this shape: "
        "{\"nodes\":[{\"id\":\"<role>:<tool>\",\"role\":\"<role>\",\"tool\":\"<tool>\"}],"
        "\"edges\":[[\"<role>:<tool>\",\"<role>:<tool>\"]]}. "
        "Include useful workflow tools as nodes, including isolated nodes when no ordering is specified. "
        "Use an edge only for a declared or necessary dependency; keep the result acyclic."
    )


def api_call(
    endpoint: str,
    key: str,
    prompt: str,
    timeout: int = 90,
    retries: int = 2,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You extract workflow DAGs. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    attempts = 0
    last_error: str | None = None
    status: int | None = None
    response_summary: str | None = None
    content: str | None = None
    # ``retries`` is explicit so a slow local gateway cannot silently turn a
    # single exploratory call into several minutes of blocking.  The default
    # preserves the original three-attempt behavior.
    for attempt in range(max(0, int(retries)) + 1):
        attempts = attempt + 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
                message = (payload.get("choices") or [{}])[0].get("message") or {}
                value = message.get("content")
                if isinstance(value, list):
                    value = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in value)
                content = str(value) if value is not None else None
                response_summary = re.sub(r"\s+", " ", content or "")[:300]
                last_error = None if content is not None else "missing message.content"
                break
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            response_summary = re.sub(r"\s+", " ", detail)[:300]
            last_error = f"HTTP {exc.code}: {detail[:180]}"
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as exc:  # network/JSON errors are recorded, not raised
            last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            response_summary = last_error
        if attempt < max(0, int(retries)):
            time.sleep(2 ** attempt)
    return {
        "ok": content is not None and last_error is None,
        "content": content,
        "content_sha256": sha256_text(content) if content is not None else None,
        "attempts": attempts,
        "http_status": status,
        "error": last_error,
        "response_summary": response_summary,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    # Try every opening brace so a harmless preamble/code fence does not make
    # an otherwise valid response fail parsing.
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def canonical_node_id(raw: Any, node: dict[str, Any], expected: dict[str, dict[str, Any]]) -> str:
    raw_id = str(raw or "")
    norm = slug(raw_id)
    aliases: dict[str, str] = {}
    for expected_id, item in expected.items():
        role = slug(item.get("role"))
        tool = slug(item.get("tool"))
        aliases[slug(expected_id)] = expected_id
        aliases[slug(f"{role}:{tool}")] = expected_id
        aliases[slug(f"{role}/{tool}")] = expected_id
        aliases[slug(f"{role}-{tool}")] = expected_id
    if norm in aliases:
        return aliases[norm]
    role = slug(node.get("role"))
    tool = slug(node.get("tool"))
    if role and tool:
        candidate = aliases.get(slug(f"{role}:{tool}"))
        if candidate:
            return candidate
    return f"__pred__:{raw_id or 'unnamed'}"


def parse_prediction(text: str | None, expected_nodes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    value = json_object(text)
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list) or not isinstance(value.get("edges"), list):
        return None
    nodes: list[dict[str, Any]] = []
    raw_nodes: list[dict[str, Any]] = []
    by_raw: dict[str, str] = {}
    for item in value["nodes"]:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id") or "")
        canonical = canonical_node_id(raw_id, item, expected_nodes)
        by_raw[raw_id] = canonical
        raw_nodes.append({
            "id": raw_id,
            "role": str(item.get("role") or ""),
            "tool": str(item.get("tool") or ""),
        })
        nodes.append({
            "id": canonical,
            "raw_id": raw_id,
            "role": str(item.get("role") or ""),
            "tool": str(item.get("tool") or ""),
        })
    # Permit an edge endpoint to use a canonical ID even if the node's raw ID
    # was already canonicalized; unknown endpoints remain explicit false positives.
    known = {n["id"] for n in nodes}
    edges: list[list[str]] = []
    raw_edges: list[list[str]] = []
    for edge in value["edges"]:
        if isinstance(edge, dict):
            edge = [edge.get("source"), edge.get("target")]
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or edge[0] is None or edge[1] is None:
            continue
        raw_edge = [str(edge[0]), str(edge[1])]
        raw_edges.append(raw_edge)
        out = []
        for endpoint in edge:
            raw_endpoint = str(endpoint)
            out.append(by_raw.get(raw_endpoint, raw_endpoint if raw_endpoint in known else f"__pred__:{raw_endpoint}"))
        edges.append(out)
    unique_nodes = {n["id"]: n for n in nodes}
    unique_edges = sorted({tuple(edge) for edge in edges})
    return {
        "nodes": list(unique_nodes.values()),
        "edges": [list(edge) for edge in unique_edges],
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
        "raw_node_count": len(nodes),
        "raw_edge_count": len(edges),
    }


def is_acyclic(node_ids: set[str], edges: set[tuple[str, str]]) -> bool:
    outgoing: dict[str, set[str]] = defaultdict(set)
    indegree = {node: 0 for node in node_ids}
    for source, target in edges:
        if source not in indegree:
            indegree[source] = 0
        if target not in indegree:
            indegree[target] = 0
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    seen = 0
    while queue:
        current = queue.pop()
        seen += 1
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return seen == len(indegree)


def f1(expected: set[Any], got: set[Any]) -> dict[str, float]:
    tp = len(expected & got)
    precision = tp / max(1, len(got))
    recall = tp / max(1, len(expected))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}


def task_graph(task: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    """Read the normalized task DAG while accepting its dict edge schema.

    HarnessAudit's normalized adapter stores edge metadata (relation) beside
    source/target.  Older fixtures sometimes use two-item lists, so both are
    accepted without changing the source file.
    """
    dag = task.get("workflow_dag") or {}
    expected = {
        str(item.get("id")): item
        for item in dag.get("nodes", [])
        if isinstance(item, dict) and item.get("id")
    }
    edges: set[tuple[str, str]] = set()
    for edge in dag.get("edges", []):
        if isinstance(edge, dict):
            source, target = edge.get("source"), edge.get("target")
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            source, target = edge
        else:
            continue
        if source is not None and target is not None:
            edges.add((str(source), str(target)))
    return expected, edges


SEMANTIC_STOPWORDS = {
    "agent", "assistant", "bot", "worker", "specialist", "coordinator",
    "manager", "service", "system", "tool", "node", "stage", "step",
}


def semantic_tokens(value: Any) -> set[str]:
    tokens = {
        token for token in slug(value).split("_")
        if token and token not in SEMANTIC_STOPWORDS
    }
    expanded = set(tokens)
    for token in list(tokens):
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
    return expanded


def lexical_similarity(left: Any, right: Any) -> float:
    """Deterministic, auditable label similarity; no model/embedding involved."""
    left_slug, right_slug = slug(left), slug(right)
    if not left_slug or not right_slug:
        return 0.0
    if left_slug == right_slug:
        return 1.0
    left_tokens, right_tokens = semantic_tokens(left), semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    jaccard = overlap / len(left_tokens | right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return min(1.0, max(jaccard, 0.85 * containment))


def semantic_node_alignment(
    raw_nodes: list[dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> tuple[dict[int, str], dict[str, Any]]:
    """One-to-one lexical alignment of predicted nodes to task DAG nodes.

    Tool identity is weighted most heavily, followed by role and the complete
    node label.  A small dynamic program chooses the maximum-weight matching so
    duplicate tools used by different roles cannot all map to one gold node.
    Unmatched predictions remain explicit false positives.
    """
    expected_items = list(expected.items())
    if not raw_nodes or not expected_items:
        return {}, {str(i): {"canonical_id": None, "method": "unmapped", "confidence": 0.0} for i in range(len(raw_nodes))}

    candidate_scores: list[list[tuple[int, float, dict[str, float]]]] = []
    for pred in raw_nodes:
        pred_id, pred_role, pred_tool = pred.get("id", ""), pred.get("role", ""), pred.get("tool", "")
        row: list[tuple[int, float, dict[str, float]]] = []
        for index, (expected_id, gold) in enumerate(expected_items):
            gold_role, gold_tool = gold.get("role", ""), gold.get("tool", "")
            tool_score = lexical_similarity(pred_tool, gold_tool)
            role_score = lexical_similarity(pred_role, gold_role)
            id_score = lexical_similarity(pred_id, expected_id)
            # If one component is omitted, do not penalize it twice; the ID can
            # still carry the role/tool information.
            weights = [(tool_score, 0.55), (role_score, 0.35), (id_score, 0.10)]
            score = sum(value * weight for value, weight in weights)
            if not pred_tool and not pred_role:
                score = id_score
            details = {"tool": round(tool_score, 4), "role": round(role_score, 4), "id": round(id_score, 4)}
            if score >= 0.42:
                row.append((index, score, details))
        row.sort(key=lambda item: (-item[1], item[0]))
        candidate_scores.append(row)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(pred_index: int, used_mask: int) -> tuple[float, tuple[int | None, ...]]:
        if pred_index >= len(raw_nodes):
            return 0.0, ()
        best_total, best_assign = solve(pred_index + 1, used_mask)
        best_assign = (None,) + best_assign
        for gold_index, score, _details in candidate_scores[pred_index]:
            if used_mask & (1 << gold_index):
                continue
            tail_total, tail_assign = solve(pred_index + 1, used_mask | (1 << gold_index))
            total = score + tail_total
            if total > best_total + 1e-9:
                best_total, best_assign = total, (gold_index,) + tail_assign
        return best_total, best_assign

    _total, assignment = solve(0, 0)
    mapping: dict[int, str] = {}
    details_out: dict[str, Any] = {}
    for index, gold_index in enumerate(assignment):
        pred = raw_nodes[index]
        raw_id = str(pred.get("id", ""))
        chosen = None
        score_details: dict[str, float] = {}
        score_value = 0.0
        if gold_index is not None:
            chosen = expected_items[gold_index][0]
            for candidate_index, candidate_score, candidate_details in candidate_scores[index]:
                if candidate_index == gold_index:
                    score_value, score_details = candidate_score, candidate_details
                    break
            mapping[index] = chosen
        details_out[str(index)] = {
            "raw_id": raw_id,
            "role": str(pred.get("role", "")),
            "tool": str(pred.get("tool", "")),
            "canonical_id": chosen,
            "method": "lexical_bipartite" if chosen else "unmapped",
            "confidence": round(score_value, 4),
            "component_similarity": score_details,
        }
    return mapping, details_out


def exact_id(raw_id: Any, expected: dict[str, dict[str, Any]]) -> str:
    aliases = {slug(expected_id): expected_id for expected_id in expected}
    normalized = slug(raw_id)
    return aliases.get(normalized, f"__pred__:{str(raw_id)}")


def graph_scores(prediction: dict[str, Any] | None, task: dict[str, Any]) -> dict[str, Any]:
    expected, expected_edges = task_graph(task)
    expected_nodes = set(expected)
    if prediction is None:
        return {
            "scored": False,
            "node": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "edge": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "exact_node_f1": 0.0,
            "exact_edge_f1": 0.0,
            "semantic_node_f1": 0.0,
            "semantic_edge_f1": 0.0,
            "role": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "tool": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "role_coverage": 0.0,
            "tool_coverage": 0.0,
            "acyclic": False,
            "invalid_endpoint_count": None,
            "self_loop_count": None,
        }
    raw_nodes = prediction.get("raw_nodes") or [
        {"id": n.get("raw_id", n.get("id", "")), "role": n.get("role", ""), "tool": n.get("tool", "")}
        for n in prediction.get("nodes", [])
    ]
    raw_edges = prediction.get("raw_edges") or [[str(a), str(b)] for a, b in prediction.get("edges", [])]
    # Exact means normalized identifier equality only (punctuation/case variants
    # are equivalent); role/tool inference is reserved for semantic scoring.
    raw_to_exact: dict[str, str] = {}
    raw_to_exact_slug: dict[str, str] = {}
    exact_nodes: set[str] = set()
    for node in raw_nodes:
        raw_id = str(node.get("id", ""))
        canonical = exact_id(raw_id, expected)
        raw_to_exact[raw_id] = canonical
        raw_to_exact_slug[slug(raw_id)] = canonical
        exact_nodes.add(canonical)
    exact_edges: set[tuple[str, str]] = set()
    for source, target in raw_edges:
        exact_edges.add((
            raw_to_exact.get(str(source), raw_to_exact_slug.get(slug(source), exact_id(source, expected))),
            raw_to_exact.get(str(target), raw_to_exact_slug.get(slug(target), exact_id(target, expected))),
        ))
    node_score = f1(expected_nodes, exact_nodes)
    edge_score = f1(expected_edges, exact_edges)

    semantic_mapping, mapping_details = semantic_node_alignment(raw_nodes, expected)
    semantic_nodes = set(semantic_mapping.values())
    semantic_unmatched = [index for index in range(len(raw_nodes)) if index not in semantic_mapping]
    semantic_nodes.update(f"__unmatched_node__:{index}" for index in semantic_unmatched)
    raw_to_index: dict[str, int] = {}
    raw_to_index_slug: dict[str, int] = {}
    for index, node in enumerate(raw_nodes):
        raw_id = str(node.get("id", ""))
        raw_to_index.setdefault(raw_id, index)
        raw_to_index_slug.setdefault(slug(raw_id), index)
    semantic_edges: set[tuple[str, str]] = set()
    for source, target in raw_edges:
        source_index = raw_to_index.get(str(source), raw_to_index_slug.get(slug(source)))
        target_index = raw_to_index.get(str(target), raw_to_index_slug.get(slug(target)))
        source_id = semantic_mapping.get(source_index, f"__unmatched_node__:{source_index if source_index is not None else source}")
        target_id = semantic_mapping.get(target_index, f"__unmatched_node__:{target_index if target_index is not None else target}")
        semantic_edges.add((source_id, target_id))
    semantic_node_score = f1(expected_nodes, semantic_nodes)
    semantic_edge_score = f1(expected_edges, semantic_edges)

    expected_role = {str(item.get("role")) for item in expected.values() if item.get("role")}
    expected_tool = {str(item.get("tool")) for item in expected.values() if item.get("tool")}
    mapped_roles = {str(expected[cid].get("role")) for cid in semantic_mapping.values() if cid in expected and expected[cid].get("role")}
    mapped_tools = {str(expected[cid].get("tool")) for cid in semantic_mapping.values() if cid in expected and expected[cid].get("tool")}
    role_score = f1(expected_role, mapped_roles)
    tool_score = f1(expected_tool, mapped_tools)
    role_coverage = len(mapped_roles & expected_role) / max(1, len(expected_role))
    tool_coverage = len(mapped_tools & expected_tool) / max(1, len(expected_tool))
    # Structural validity is checked on the raw predicted graph, including any
    # undeclared endpoints, so malformed edges cannot disappear via alignment.
    raw_node_ids = {str(node.get("id", "")) for node in raw_nodes}
    raw_edge_set = {(str(source), str(target)) for source, target in raw_edges}
    invalid = sum(1 for source, target in raw_edge_set if source not in raw_node_ids or target not in raw_node_ids)
    self_loops = sum(1 for source, target in raw_edge_set if source == target)
    return {
        "scored": True,
        "node": node_score,
        "edge": edge_score,
        "role": role_score,
        "tool": tool_score,
        "exact_node_f1": node_score["f1"],
        "exact_edge_f1": edge_score["f1"],
        "semantic_node_f1": semantic_node_score["f1"],
        "semantic_edge_f1": semantic_edge_score["f1"],
        "semantic_node": semantic_node_score,
        "semantic_edge": semantic_edge_score,
        "role_coverage": role_coverage,
        "tool_coverage": tool_coverage,
        "semantic_unmatched": semantic_unmatched,
        "semantic_node_mapping": mapping_details,
        "semantic_scored_edges": [list(edge) for edge in sorted(semantic_edges)],
        "acyclic": is_acyclic(raw_node_ids, raw_edge_set),
        "invalid_endpoint_count": invalid,
        "self_loop_count": self_loops,
        "expected_node_count": len(expected_nodes),
        "expected_edge_count": len(expected_edges),
        "predicted_node_count": len(exact_nodes),
        "predicted_edge_count": len(exact_edges),
    }


def selection_record(task: dict[str, Any]) -> dict[str, Any]:
    node_count = len(task.get("workflow_dag", {}).get("nodes", []))
    edge_count = len(task.get("workflow_dag", {}).get("edges", []))
    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "category": task.get("category"),
        "modality": task.get("modality"),
        "agent_count": len(task.get("agents", [])),
        "tool_count": len(task.get("tools", [])),
        "expected_node_count": node_count,
        "expected_edge_count": edge_count,
        "complexity_bin": complexity_bin(node_count),
    }


def aggregate(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    output = []
    for value, group in sorted(groups.items()):
        parsed = [r for r in group if r.get("api_returned")]
        def mean(path: tuple[str, ...]) -> float | None:
            vals: list[float] = []
            for item in parsed:
                obj: Any = item.get("scores", {})
                for part in path:
                    obj = obj.get(part) if isinstance(obj, dict) else None
                if isinstance(obj, (int, float)):
                    vals.append(float(obj))
            return sum(vals) / len(vals) if vals else None
        output.append({
            key: value,
            "calls": len(group),
            "api_returned": len(parsed),
            "parse_rate": len(parsed) / max(1, len(group)),
            "mean_exact_node_f1": mean(("exact_node_f1",)),
            "mean_exact_edge_f1": mean(("exact_edge_f1",)),
            "mean_semantic_node_f1": mean(("semantic_node_f1",)),
            "mean_semantic_edge_f1": mean(("semantic_edge_f1",)),
            "mean_node_f1": mean(("exact_node_f1",)),
            "mean_edge_f1": mean(("exact_edge_f1",)),
            "mean_role_f1": mean(("role", "f1")),
            "mean_tool_f1": mean(("tool", "f1")),
            "mean_role_coverage": mean(("role_coverage",)),
            "mean_tool_coverage": mean(("tool_coverage",)),
            "acyclic_rate": sum(bool(r.get("scores", {}).get("acyclic")) for r in parsed) / max(1, len(parsed)),
        })
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one_call(
    call_number: int,
    task: dict[str, Any],
    strategy: str,
    endpoint: str,
    key: str,
    timeout: int,
    retries: int,
    reasoning_effort: str,
    max_tokens: int,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Run one independent API call; safe to invoke from a worker thread."""
    expected, _expected_edges = task_graph(task)
    prompt = task_prompt(task, strategy)
    try:
        response = api_call(
            endpoint,
            key,
            prompt,
            timeout=timeout,
            retries=retries,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        prediction = parse_prediction(response.get("content"), expected) if response.get("ok") else None
        scores = graph_scores(prediction, task)
    except Exception as exc:  # pragma: no cover - defensive live-run guard
        response = {
            "ok": False,
            "content": None,
            "content_sha256": None,
            "attempts": 0,
            "http_status": None,
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            "response_summary": None,
            "elapsed_seconds": 0.0,
        }
        prediction = None
        scores = graph_scores(None, task)
    result = {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "category": task.get("category"),
        "modality": task.get("modality"),
        "complexity_bin": complexity_bin(len(expected)),
        "strategy": strategy,
        "model": MODEL,
        "prompt_sha256": sha256_text(prompt),
        "api_returned": prediction is not None,
        "scores": scores,
        "prediction": prediction,
        "response_sha256": response.get("content_sha256"),
        "raw_response": response.get("content"),
    }
    status = {
        "call_number": call_number,
        "task_id": task.get("task_id"),
        "strategy": strategy,
        "model": MODEL,
        "ok": bool(response.get("ok")),
        "parsed": prediction is not None,
        "http_status": response.get("http_status"),
        "attempts": response.get("attempts"),
        "error": response.get("error"),
        "response_summary": response.get("response_summary"),
        "elapsed_seconds": response.get("elapsed_seconds"),
    }
    return call_number, result, status


def prior_model_availability() -> dict[str, Any]:
    """Summarize the earlier model test without issuing extra 503 calls."""
    path = PAPER_ROOT / "experiments/synthetic_mas/runs/model_test_v1/api_status.json"
    summary: dict[str, Any] = {"path": str(path), "available": path.exists()}
    if not path.exists():
        return summary
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return summary
        by_model: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model"))
            by_model[model][str(row.get("http_status"))] += 1
        summary["status_counts_by_model"] = {model: dict(counts) for model, counts in by_model.items()}
        summary["gpt_5_3_codex_503_count"] = by_model.get("gpt-5.3-codex", Counter()).get("503", 0)
        summary["source_sha256"] = sha256_bytes(path.read_bytes())
    except (OSError, ValueError, TypeError):
        summary["read_error"] = True
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--retries", type=int, default=1,
                        help="retries after the first API attempt (default: 1)")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        choices=("none", "low", "medium", "high"),
                        help="gateway reasoning effort (default: low)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="maximum completion tokens (default: 900)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="parallel API workers (default: 3; use 1 for serial replay)")
    parser.add_argument("--resume", action="store_true",
                        help="resume completed task/strategy calls from partial checkpoints")
    parser.add_argument("--no-api", action="store_true", help="write a blocked manifest without making calls")
    args = parser.parse_args()

    rows, input_hash = load_tasks(args.input)
    tasks = select_tasks(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    base_url, key, endpoint = api_config()
    selection = {
        "schema_version": "harnessaudit-dag-v1",
        "input_path": str(args.input),
        "input_sha256": input_hash,
        "selection_rule": "fixed 12-task stratified subset: all 8 domains; balanced text/multimodal where available; small/medium/large DAGs represented",
        "selected_task_ids": list(SELECTED_TASK_IDS),
        "tasks": [selection_record(task) for task in tasks],
        "distribution": {
            "domains": dict(Counter(task.get("domain") for task in tasks)),
            "modalities": dict(Counter(task.get("modality") for task in tasks)),
            "complexity_bins": dict(Counter(complexity_bin(len(task.get("workflow_dag", {}).get("nodes", []))) for task in tasks)),
        },
    }
    write_json(args.output / "selection.json", selection)

    total_calls = len(tasks) * len(STRATEGIES)
    records: list[dict[str, Any]] = []
    status_records: list[dict[str, Any]] = []
    if args.resume:
        partial_results = args.output / "dag_results.partial.json"
        partial_status = args.output / "api_status.partial.json"
        if partial_results.exists():
            try:
                loaded = json.loads(partial_results.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    records = [item for item in loaded if isinstance(item, dict)]
            except (OSError, ValueError, TypeError):
                records = []
        if partial_status.exists():
            try:
                loaded = json.loads(partial_status.read_text(encoding="utf-8"))
                calls = loaded.get("calls") if isinstance(loaded, dict) else None
                if isinstance(calls, list):
                    status_records = [item for item in calls if isinstance(item, dict)]
            except (OSError, ValueError, TypeError):
                status_records = []
        # Only a matching task/strategy pair is considered complete.  This
        # avoids silently skipping a call if a manually edited checkpoint is
        # inconsistent.
        # A failed/invalid response is deliberately not treated as complete;
        # --resume can retry it after a transient gateway failure.
        completed_pairs = {
            (str(item.get("task_id")), str(item.get("strategy")))
            for item in records
            if item.get("task_id") is not None
            and item.get("strategy") is not None
            and bool(item.get("api_returned"))
        }
    else:
        completed_pairs = set()
    blocked_reason: str | None = None
    if args.no_api:
        blocked_reason = "--no-api"
    elif not key:
        blocked_reason = "No API key found; set HARNESSAUDIT_API_KEY/AGENT_API_KEY or P1V2_Q2_API_KEY"
    else:
        jobs: list[tuple[int, dict[str, Any], str]] = []
        for task_index, task in enumerate(tasks):
            for strategy_index, strategy in enumerate(STRATEGIES):
                pair = (str(task.get("task_id")), strategy)
                if pair in completed_pairs:
                    continue
                # Stable numbering is independent of completion order and of
                # whether earlier pairs were loaded from a checkpoint.
                jobs.append((task_index * len(STRATEGIES) + strategy_index + 1, task, strategy))
        if jobs:
            workers = max(1, min(int(args.workers), len(jobs)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        run_one_call,
                        call_number,
                        task,
                        strategy,
                        endpoint,
                        key,
                        args.timeout,
                        args.retries,
                        args.reasoning_effort,
                        args.max_tokens,
                    ): (call_number, task, strategy)
                    for call_number, task, strategy in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    call_number, task, strategy = futures[future]
                    try:
                        result_call_number, result, status = future.result()
                    except Exception as exc:  # pragma: no cover - worker isolation
                        result_call_number = call_number
                        result = {
                            "task_id": task.get("task_id"), "domain": task.get("domain"),
                            "category": task.get("category"), "modality": task.get("modality"),
                            "complexity_bin": complexity_bin(len(task_graph(task)[0])),
                            "strategy": strategy, "model": MODEL,
                            "prompt_sha256": sha256_text(task_prompt(task, strategy)),
                            "api_returned": False, "scores": graph_scores(None, task),
                            "prediction": None, "response_sha256": None, "raw_response": None,
                        }
                        status = {
                            "call_number": call_number, "task_id": task.get("task_id"),
                            "strategy": strategy, "model": MODEL, "ok": False,
                            "parsed": False, "http_status": None, "attempts": 0,
                            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                            "response_summary": None, "elapsed_seconds": 0.0,
                        }
                    records.append(result)
                    status_records.append(status)
                    records.sort(key=lambda item: (str(item.get("task_id")), str(item.get("strategy"))))
                    status_records.sort(key=lambda item: int(item.get("call_number") or 0))
                    print(
                        f"[{result_call_number}/{total_calls}] {task.get('task_id')} {strategy}: "
                        f"status={status['http_status']} parsed={status['parsed']} "
                        f"elapsed={status['elapsed_seconds']}s",
                        flush=True,
                    )
                    # Checkpoint only from the main thread, preventing partial
                    # JSON races while workers are still returning.
                    write_json(args.output / "dag_results.partial.json", records)
                    write_json(args.output / "api_status.partial.json", {
                        "model": MODEL, "base_url": base_url, "endpoint": endpoint,
                        "workers": workers, "prior_model_availability": prior_model_availability(),
                        "calls": status_records,
                    })
        write_json(args.output / "dag_results.json", records)
        write_json(args.output / "api_status.json", {
            "model": MODEL, "base_url": base_url, "endpoint": endpoint,
            "workers": max(1, int(args.workers)),
            "prior_model_availability": prior_model_availability(),
            "calls": status_records,
        })
    if blocked_reason is not None:
        write_json(args.output / "dag_results.json", records)
        write_json(args.output / "api_status.json", {
            "blocked": True, "reason": blocked_reason, "model": MODEL,
            "base_url": base_url, "endpoint": endpoint,
            "prior_model_availability": prior_model_availability(),
            "calls": status_records,
        })

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    successful = [record for record in records if record.get("api_returned")]
    prior_availability = prior_model_availability()
    report = {
        "schema_version": "harnessaudit-dag-v1",
        "run_id": "HARNESSAUDIT-DAG-V1",
        "generated_at_utc": generated_at,
        "input": {"path": str(args.input), "sha256": input_hash, "task_count": len(tasks)},
        "api": {
            "model": MODEL,
            "base_url": base_url,
            "endpoint": endpoint,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "retries_after_first": args.retries,
            "workers": max(1, int(args.workers)),
            "key_written": False,
            "blocked_reason": blocked_reason,
            "full_calls_requested": total_calls,
            "calls_completed": len(records),
            "parsed_graphs": len(successful),
            "parse_rate": len(successful) / max(1, len(records)),
            "http_status_counts": dict(Counter(str(item.get("http_status")) for item in status_records)),
        },
        "selection": selection,
        "strategy_summary": aggregate(records, "strategy"),
        "domain_summary": aggregate(records, "domain"),
        "prior_model_availability": prior_availability,
        "trace_interface": {
            "source": "HarnessAudit runtime traces/results, when available",
            "normalized_event_input": 'paperAlpha/data/processed/mas_adapter/events.jsonl',
            "required_fields": ["agent_id", "role", "message", "timestamp", "parent", "permission", "external_effect", "risk_label"],
            "join_keys": ["run_id", "episode_id", "event_id", "parent"],
            "current_status": "not_connected_no_local_runtime_traces",
        },
        "records": len(records),
        "limitations": [
            "This is specification-to-DAG extraction only; no HarnessAudit agent/tool execution was run.",
            "Reference DAGs are the deterministic adapter projection from declared ground_truth_tool_paths; they are normative scaffolding, not observed causal traces.",
            "One API call per task/strategy is exploratory and not a model leaderboard.",
            "No risk labels or runtime external effects are inferred from task specifications.",
            "Semantic scores use deterministic one-to-one lexical role/tool alignment; they are consistency measures, not causal-truth scores.",
        ],
    }
    write_json(args.output / "report.json", report)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HarnessAudit DAG API evaluation v1",
        "type": "object",
        "required": ["schema_version", "run_id", "input", "api", "selection", "strategy_summary"],
        "properties": {"schema_version": {"const": "harnessaudit-dag-v1"}, "run_id": {"type": "string"}, "records": {"type": "integer"}},
    }
    write_json(args.output / "schema.json", schema)

    lines = [
        "# HarnessAudit DAG API evaluation v1",
        "",
        "本轮只评估 HarnessAudit 任务规格到工作流 DAG 的 API 抽取；没有运行完整 Agent、工具或环境。",
        "",
        f"- Selected tasks: **{len(tasks)}** / 210 normalized task specs",
        f"- Strategies: `{', '.join(STRATEGIES)}`",
        f"- Model: `{MODEL}`",
        f"- Request controls: reasoning=`{args.reasoning_effort}`, max_tokens=`{args.max_tokens}`, timeout=`{args.timeout}s`, retries=`{args.retries}`, workers=`{max(1, int(args.workers))}`",
        f"- API calls requested/completed/parsed: `{total_calls}/{len(records)}/{len(successful)}`",
        f"- Input SHA-256: `{input_hash}`",
        f"- Prior `gpt-5.3-codex` availability record: HTTP 503 count = `{prior_availability.get('gpt_5_3_codex_503_count', 'not available')}` (see `prior_model_availability` in `report.json`)",
        "",
        "## Stratification",
        "",
        f"- Task IDs: `{', '.join(SELECTED_TASK_IDS)}`",
        f"- Domains: `{selection['distribution']['domains']}`",
        f"- Modalities: `{selection['distribution']['modalities']}`",
        f"- DAG size bins: `{selection['distribution']['complexity_bins']}`",
        "",
        "## Strategy summary",
        "",
        "| Strategy | Calls | Parsed | Parse rate | Exact node F1 | Exact edge F1 | Semantic node F1 | Semantic edge F1 | Role cov. | Tool cov. | Acyclic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summary"]:
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {item['strategy']} | {item['calls']} | {item['api_returned']} | {item['parse_rate']:.3f} | "
            f"{fmt(item.get('mean_exact_node_f1'))} | {fmt(item.get('mean_exact_edge_f1'))} | "
            f"{fmt(item.get('mean_semantic_node_f1'))} | {fmt(item.get('mean_semantic_edge_f1'))} | "
            f"{fmt(item.get('mean_role_coverage'))} | {fmt(item.get('mean_tool_coverage'))} | {item['acyclic_rate']:.3f} |"
        )
    lines += [
        "",
        "节点/边 F1 以 `normalized_tasks.jsonl` 中由 `ground_truth_tool_paths` 确定性投影的 DAG 为参考；这里的参考是任务声明工作流的一致性 scaffolding，不是观察到的因果真值；未知节点和额外边保留为误报。",
        "Exact F1 只接受规范化节点 ID；semantic F1 使用不依赖模型的 role/tool 词法一对一对齐。角色/工具覆盖率是对声明集合的召回比例。",
        "策略提示中没有传入 `workflow_dag` 或 `ground_truth_tool_paths`，以避免评估泄漏；仅提供 goal、角色、工具必要性、检查点和访问策略形状。",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\evaluate_harnessaudit_dag_v1.py",
        "```",
        "",
        "完整 API 返回仅保存在 `dag_results.json` 的 `raw_response` 字段；密钥不写入任何 artifact。",
        '真实 HarnessAudit trace 接口：将运行时 JSONL 映射为 `paperAlpha/data/processed/mas_adapter/events.jsonl` 的统一字段，并以 `run_id/episode_id/event_id/parent` 连接；当前本地 traces/results 目录为空，因此未声称运行轨迹指标。',
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": report["run_id"], "output": str(args.output), "calls": len(records), "parsed": len(successful)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
