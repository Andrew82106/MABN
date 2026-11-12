from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/synthetic_mas/runs/model_test_v1'
OUT.mkdir(parents=True, exist_ok=True)

CANONICAL = {
    "nodes": [
        {"id": "intake", "role": "request intake", "risk_template": "goal_input"},
        {"id": "extract", "role": "evidence extraction", "risk_template": "input_pollution"},
        {"id": "verify", "role": "independent verification", "risk_template": "provenance"},
        {"id": "analyze", "role": "risk analysis", "risk_template": "composition"},
        {"id": "policy", "role": "policy review", "risk_template": "authorization"},
        {"id": "report", "role": "report publication", "risk_template": "external_effect"},
    ],
    "edges": [
        ["intake", "extract"],
        ["extract", "verify"],
        ["verify", "analyze"],
        ["analyze", "policy"],
        ["policy", "report"],
    ],
}

WORKFLOWS = [
    ("vendor_review", "Review a vendor using intake, dossier extraction, verification, risk analysis, policy review, and public reporting."),
    ("access_request", "Process an access request through intake, identity verification, authorization analysis, policy review, and audit reporting."),
    ("research_digest", "Collect research, extract claims, verify sources, assess risks, and publish a digest."),
    ("incident_triage", "Triage an incident, gather evidence, independently verify it, assess impact, approve actions, and report."),
    ("invoice_payment", "Review an invoice, extract fields, verify supplier and amount, assess fraud risk, approve policy, and submit payment."),
    ("data_export", "Handle a data export request, identify records, verify recipient and authorization, review policy, and deliver the export."),
]

FEATURES = [
    "untrusted_input",
    "permission_mismatch",
    "memory_poison",
    "tool_side_effect",
    "cross_agent_propagation",
    "anomaly_path",
    "risk_accumulation",
]

KEYWORDS = {
    "untrusted_input": ["external", "injected", "untrusted", "poisoned", "malicious"],
    "permission_mismatch": ["unauthorized", "wrong recipient", "permission", "approval", "credential"],
    "memory_poison": ["memory", "stale", "persisted", "shared state"],
    "tool_side_effect": ["send", "delete", "payment", "execute", "publish", "export"],
    "cross_agent_propagation": ["agent", "delegate", "handoff", "shared", "propagate"],
    "anomaly_path": ["retry", "loop", "unexpected", "bypass", "deviate"],
    "risk_accumulation": ["multiple", "combined", "cumulative", "chain", "fragment"],
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def canonical_prompt(workflow: str, description: str) -> str:
    return (
        f"Workflow name: {workflow}\nDescription: {description}\n"
        "Return JSON only with keys nodes (array of objects with id, role) and edges (array of [source,target]). "
        "The graph must be a DAG and include the main functional dependencies."
    )


def api_chat(base_url: str, key: str, model: str, prompt: str, timeout: int = 25) -> tuple[str | None, str | None, dict[str, Any]]:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You extract workflow DAGs. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    meta: dict[str, Any] = {"attempts": 0, "status": None, "response_summary": None}
    last_error = None
    for attempt in range(3):
        meta["attempts"] = attempt + 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
                content = payload["choices"][0]["message"]["content"]
                meta["status"] = resp.status
                meta["response_summary"] = str(content).replace("\n", " ")[:240]
                return content, None, meta
        except urllib.error.HTTPError as exc:
            meta["status"] = exc.code
            try:
                body = exc.read().decode(errors="replace")
            except Exception:
                body = ""
            meta["response_summary"] = body.replace("\n", " ")[:240]
            last_error = f"HTTP {exc.code}: {body[:160]}"
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_error = type(exc).__name__ + ": " + str(exc)[:180]
            meta["response_summary"] = last_error
        if attempt < 2:
            time.sleep(2**attempt)
    return None, last_error, meta


def parse_dag(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        ids = {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id")}
        clean_edges = [[str(a), str(b)] for e in edges if isinstance(e, list) and len(e) == 2 for a, b in [e] if str(a) in ids and str(b) in ids and a != b]
        if len(ids) < 3:
            return None
        return {"nodes": [{"id": str(n["id"]), "role": str(n.get("role", n["id"]))} for n in nodes if n.get("id")], "edges": clean_edges}
    except (ValueError, TypeError, KeyError):
        return None


def has_cycle(nodes: list[dict[str, str]], edges: list[list[str]]) -> bool:
    graph = defaultdict(list)
    indegree = {n["id"]: 0 for n in nodes}
    for a, b in edges:
        if a in indegree and b in indegree:
            graph[a].append(b)
            indegree[b] += 1
    queue = [n for n, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return seen != len(indegree)


def normalize_dag(dag: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    if not dag:
        return CANONICAL, True
    nodes = dag["nodes"]
    edges = dag["edges"]
    if has_cycle(nodes, edges):
        # deterministic cycle repair: retain the earliest edge in node order
        kept: list[list[str]] = []
        for edge in edges:
            trial = kept + [edge]
            if not has_cycle(nodes, trial):
                kept.append(edge)
        edges = kept
    return {"nodes": nodes, "edges": edges}, False


def strategy_prompt(strategy: str, workflow: str, description: str) -> str:
    if strategy == "direct":
        return canonical_prompt(workflow, description)
    if strategy == "decompose":
        return (
            f"Workflow: {workflow}\n{description}\n"
            "First list roles, tools, and task stages internally. Then infer dependency edges and topologically sort them. "
            "Return JSON only with nodes [{id,role}] and edges [[source,target]]."
        )
    return (
        f"Workflow: {workflow}\n{description}\n"
        "Generate candidate dependency edges, audit every edge for necessity and direction, remove any edge that creates a cycle, "
        "then return JSON only with nodes [{id,role}] and edges [[source,target]]."
    )


## Semantic DAG alignment
#
# The reference graph uses stable functional concepts, while an LLM often
# chooses task-specific names (for example ``dossier_extraction`` or
# ``public_reporting``).  Keep this map explicit and deterministic so the
# semantic score is auditable and does not silently depend on another model.
# A later iteration can replace the lexical fallback with embeddings plus
# human adjudication; this first version intentionally has no external
# dependency.
SEMANTIC_NODE_ID_MAP = {
    # Intake / request registration.
    "request_intake": "intake",
    "intake_specialist": "intake",
    "request_registration": "intake",
    "request_submission": "intake",
    "collect_research": "intake",
    "triage_incident": "intake",
    "review_invoice": "intake",
    # Evidence or dossier extraction.
    "dossier_extraction": "extract",
    "dossier_extractor": "extract",
    "evidence_extraction": "extract",
    "evidence_extractor": "extract",
    "claim_extraction": "extract",
    "field_extraction": "extract",
    "record_identification": "extract",
    "record_extraction": "extract",
    # Verification / validation.
    "verification": "verify",
    "verification_analyst": "verify",
    "identity_verification": "verify",
    "identity_verification_analyst": "verify",
    "source_verification": "verify",
    "source_validation": "verify",
    "independent_verification": "verify",
    # Analysis / assessment.
    "risk_analysis": "analyze",
    "risk_analyst": "analyze",
    "authorization_analysis": "analyze",
    "authorization_analyst": "analyze",
    "impact_assessment": "analyze",
    "fraud_analysis": "analyze",
    # Policy / authorization review.
    "policy_review": "policy",
    "policy_reviewer": "policy",
    "authorization_review": "policy",
    "approval_review": "policy",
    "compliance_review": "policy",
    # Reporting / final publication or delivery.
    "public_reporting": "report",
    "public_reporting_specialist": "report",
    "audit_reporting": "report",
    "audit_reporting_specialist": "report",
    "report_publication": "report",
    "reporting": "report",
    "publication": "report",
    "payment_submission": "report",
    "submit_payment": "report",
    "prepare_export": "report",
    "data_delivery": "report",
}

# Role/label keywords are weighted.  Generic words such as ``review`` are
# weak evidence, while a functional head such as ``verification`` is strong.
SEMANTIC_NODE_KEYWORDS = {
    "intake": {
        "intake": 1.00, "request": 0.70, "register": 0.85,
        "registration": 0.85, "receive": 0.70, "submission": 0.75,
        "collect": 0.55, "normalize": 0.55, "triage": 0.45,
    },
    "extract": {
        "extract": 1.00, "extraction": 1.00, "dossier": 0.95,
        "evidence": 0.72, "gather": 0.72, "claims": 0.65,
        "fields": 0.65, "records": 0.65, "identify": 0.50,
    },
    "verify": {
        "verify": 1.00, "verification": 1.00, "validate": 0.92,
        "validation": 0.92, "identity": 0.82, "authenticate": 0.90,
        "source": 0.62, "provenance": 0.82, "independent": 0.52,
        "check": 0.50,
    },
    "analyze": {
        "analyze": 1.00, "analysis": 1.00, "assess": 0.92,
        "assessment": 0.92, "risk": 0.82, "impact": 0.76,
        "fraud": 0.72, "evaluate": 0.72, "evaluation": 0.72,
    },
    "policy": {
        "policy": 1.00, "authorization": 0.90, "authorize": 0.90,
        "approval": 0.86, "approve": 0.86, "compliance": 0.80,
        "criteria": 0.60, "permission": 0.58, "review": 0.35,
    },
    "report": {
        "report": 1.00, "reporting": 1.00, "publication": 0.90,
        "publish": 0.90, "audit": 0.58, "deliver": 0.62,
        "delivery": 0.62, "submit": 0.50, "export": 0.50,
    },
}

SEMANTIC_STOPWORDS = {
    "node", "stage", "step", "task", "agent", "specialist", "analyst",
    "reviewer", "worker", "process", "main", "functional", "approved",
}


def normalize_label(value: Any) -> str:
    """Normalize an ID/role for deterministic alias matching."""
    text = str(value or "")
    # Keep snake_case and make camelCase labels comparable without a dependency.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def semantic_tokens(value: Any) -> set[str]:
    normalized = normalize_label(value)
    tokens = {token for token in normalized.split("_") if token and token not in SEMANTIC_STOPWORDS}
    # A small, transparent morphology bridge covers common role inflections.
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
    return expanded


def infer_semantic_node(node: dict[str, Any]) -> tuple[str | None, str, float]:
    """Map one predicted node to the canonical role taxonomy.

    Explicit ID/role aliases take precedence over lexical scoring.  A weak or
    ambiguous label remains unmapped rather than being forced into a class.
    """
    raw_id = str(node.get("id", ""))
    raw_role = str(node.get("role", ""))
    norm_id = normalize_label(raw_id)
    norm_role = normalize_label(raw_role)

    if norm_id in {n["id"] for n in CANONICAL["nodes"]}:
        return norm_id, "canonical_id", 1.0
    if norm_id in SEMANTIC_NODE_ID_MAP:
        return SEMANTIC_NODE_ID_MAP[norm_id], "explicit_id_alias", 1.0
    if norm_role in SEMANTIC_NODE_ID_MAP:
        return SEMANTIC_NODE_ID_MAP[norm_role], "explicit_role_alias", 0.98

    id_tokens = semantic_tokens(raw_id)
    role_tokens = semantic_tokens(raw_role)
    if not (id_tokens or role_tokens):
        return None, "unmapped", 0.0
    scores: dict[str, float] = {}
    for canonical_id, keywords in SEMANTIC_NODE_KEYWORDS.items():
        score = 0.0
        for token, weight in keywords.items():
            if token in id_tokens:
                score += weight * 1.25
            elif token in role_tokens:
                score += weight
        scores[canonical_id] = score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # Require meaningful evidence and a margin.  This prevents a generic
    # ``review`` label from becoming policy merely by default.
    if best_score < 0.72 or (best_score - second_score < 0.12 and best_score < 1.0):
        return None, "unmapped", round(best_score, 4)
    confidence = min(0.95, best_score / (best_score + second_score + 1e-9))
    return best_id, "lexical_role_id", round(confidence, 4)


def semantic_node_alignment(dag: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    mapping: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}
    for node in dag.get("nodes", []):
        raw_id = str(node.get("id", ""))
        canonical_id, method, confidence = infer_semantic_node(node)
        if canonical_id is not None:
            mapping[raw_id] = canonical_id
        details[raw_id] = {
            "role": str(node.get("role", "")),
            "canonical_id": canonical_id,
            "method": method,
            "confidence": confidence,
        }
    return mapping, details


def f1_from_sets(expected: set[Any], got: set[Any]) -> float:
    return 2 * len(expected & got) / max(1, len(expected) + len(got))


def dag_quality(dag: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    exp_nodes = {n["id"] for n in expected["nodes"]}
    got_nodes = {n["id"] for n in dag["nodes"]}
    exp_edges = {tuple(e) for e in expected["edges"]}
    got_edges = {tuple(e) for e in dag["edges"]}
    exact_node_f1 = f1_from_sets(exp_nodes, got_nodes)
    exact_edge_f1 = f1_from_sets(exp_edges, got_edges)

    # Semantic scores compare canonical role concepts and directed edges after
    # endpoint alignment. Extra edges remain extra; no transitive-closure
    # credit is granted, so a genuinely different dependency stays visible.
    # Unmatched nodes/endpoints are retained as unique sentinel labels: they do
    # not receive a hit, but they still count as false positives in the score.
    mapping, mapping_details = semantic_node_alignment(dag)
    semantic_exp_nodes = {n["id"] for n in expected["nodes"]}
    semantic_unmatched = sorted(
        raw_id for raw_id, detail in mapping_details.items()
        if detail["canonical_id"] is None
    )
    semantic_got_nodes = set(mapping.values()) | {f"__unmatched_node__:{raw_id}" for raw_id in semantic_unmatched}
    semantic_exp_edges = {tuple(e) for e in expected["edges"]}
    semantic_got_edges = set()
    for a, b in got_edges:
        source = mapping.get(a, f"__unmatched_node__:{a}")
        target = mapping.get(b, f"__unmatched_node__:{b}")
        if source != target:
            semantic_got_edges.add((source, target))
    semantic_node_f1 = f1_from_sets(semantic_exp_nodes, semantic_got_nodes)
    semantic_edge_f1 = f1_from_sets(semantic_exp_edges, semantic_got_edges)
    semantic_mapped_edges = {
        (mapping[a], mapping[b])
        for a, b in got_edges
        if a in mapping and b in mapping and mapping[a] != mapping[b]
    }
    node_mapping = {
        raw_id: detail["canonical_id"]
        for raw_id, detail in mapping_details.items()
    }
    return {
        # Backward-compatible names remain exact-match metrics.
        "node_f1": exact_node_f1,
        "edge_f1": exact_edge_f1,
        "exact_node_f1": exact_node_f1,
        "exact_edge_f1": exact_edge_f1,
        "semantic_node_f1": semantic_node_f1,
        "semantic_edge_f1": semantic_edge_f1,
        # Compact per-result fields used by downstream review tooling.
        "node_mapping": node_mapping,
        "semantic_unmatched": semantic_unmatched,
        "semantic_node_mapping": mapping_details,
        "semantic_mapped_edges": [list(e) for e in sorted(semantic_mapped_edges)],
        "semantic_scored_edges": [list(e) for e in sorted(semantic_got_edges)],
        "semantic_unmapped_nodes": semantic_unmatched,
        "acyclic": float(not has_cycle(dag["nodes"], dag["edges"])),
    }


def make_scenarios(seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    patterns = [
        (0, [], "safe"),
        (0, ["tool_side_effect"], "safe_tool"),
        (1, ["untrusted_input", "cross_agent_propagation"], "prompt_infection"),
        (1, ["permission_mismatch", "tool_side_effect"], "authorization"),
        (1, ["memory_poison", "risk_accumulation"], "memory"),
        (1, ["anomaly_path", "risk_accumulation"], "cascade"),
    ]
    for family, (label, active, mechanism) in enumerate(patterns):
        for rep in range(8):
            vals = {f: 0 for f in FEATURES}
            for f in active:
                vals[f] = 1
            if label and rng.random() < 0.35:
                vals[rng.choice(FEATURES)] = 1
            if not label and rng.random() < 0.2:
                vals["cross_agent_propagation"] = 1
            text_bits = []
            for f, value in vals.items():
                if value:
                    text_bits.append(KEYWORDS[f][rep % len(KEYWORDS[f])])
            rows.append({
                "id": f"f{family}_r{rep}",
                "family": family,
                "label": label,
                "mechanism": mechanism,
                "features": vals,
                "text": f"{mechanism} workflow " + " ".join(text_bits) + " normal review evidence",
                "risk_node": ["report", "policy", "extract", "policy", "verify", "analyze"][family],
                "events": rep + 3,
            })
    return rows


def local_probs(row: dict[str, Any]) -> dict[str, float]:
    f = row["features"]
    return {
        "input": 0.15 + 0.65 * f["untrusted_input"],
        "auth": 0.12 + 0.72 * f["permission_mismatch"],
        "memory": 0.10 + 0.68 * f["memory_poison"],
        "tool": 0.10 + 0.74 * f["tool_side_effect"] * (0.4 + 0.6 * f["permission_mismatch"]),
        "composition": 0.12 + 0.62 * (f["cross_agent_propagation"] or f["risk_accumulation"]),
        "runtime": 0.10 + 0.58 * (f["anomaly_path"] or f["risk_accumulation"]),
    }


def fit_learned_bn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prior = (sum(r["label"] for r in rows) + 1) / (len(rows) + 2)
    conditional: dict[str, dict[int, float]] = {}
    for feature in FEATURES:
        conditional[feature] = {}
        for label in (0, 1):
            subset = [r for r in rows if r["label"] == label]
            conditional[feature][label] = (sum(r["features"][feature] for r in subset) + 1) / (len(subset) + 2)
    return {"prior": prior, "conditional": conditional}


def predict(row: dict[str, Any], method: str, learned: dict[str, Any] | None = None) -> tuple[float, str, list[str]]:
    f = row["features"]
    lp = local_probs(row)
    if method == "rules":
        p = 0.98 if f["permission_mismatch"] or (f["tool_side_effect"] and f["untrusted_input"]) else 0.04
    elif method == "text":
        hits = sum(any(k in row["text"].lower() for k in ks) for ks in KEYWORDS.values())
        p = min(0.95, 0.04 + 0.16 * hits)
    elif method == "flat_bn":
        p = sum(lp.values()) / len(lp)
    elif method == "dynamic_only":
        p = sigmoid(-2.2 + 1.1 * f["cross_agent_propagation"] + 1.0 * f["anomaly_path"] + 1.0 * f["risk_accumulation"] + 0.5 * f["tool_side_effect"])
    elif method == "learned_bn":
        if learned is None:
            raise ValueError("learned_bn requires fitted parameters")
        log_odds = math.log(learned["prior"] / (1 - learned["prior"]))
        for feature in FEATURES:
            p1 = learned["conditional"][feature][1]
            p0 = learned["conditional"][feature][0]
            log_odds += math.log((p1 if f[feature] else 1 - p1) / (p0 if f[feature] else 1 - p0))
        p = sigmoid(log_odds)
    elif method == "naive_noisy_or":
        p = 1.0
        for value in lp.values():
            p *= 1 - value
        p = 1 - p
    else:
        # Hierarchical BN: local templates + dynamic cross-agent evidence.
        interaction = f["cross_agent_propagation"] * (f["untrusted_input"] + f["memory_poison"]) + f["risk_accumulation"] * f["tool_side_effect"]
        p = sigmoid(-2.8 + 2.2 * lp["input"] + 2.0 * lp["auth"] + 1.8 * lp["memory"] + 1.7 * lp["tool"] + 1.9 * lp["composition"] + 1.4 * lp["runtime"] + 0.8 * interaction)
    scores = {
        "extract": lp["input"],
        "policy": lp["auth"],
        "verify": lp["memory"],
        "report": lp["tool"],
        "analyze": max(lp["composition"], lp["runtime"]),
        "intake": max(lp["input"], lp["auth"]),
    }
    node = max(scores, key=scores.get)
    return p, node, [k for k, v in f.items() if v]


def auc_pr(y: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(y)
    if positives == 0:
        return 0.0
    tp = fp = 0
    prev_recall = prev_prec = area = 0.0
    for i in order:
        if y[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        prec = tp / (tp + fp)
        area += (recall - prev_recall) * (prec + prev_prec) / 2
        prev_recall, prev_prec = recall, prec
    return area


def ece(y: list[int], scores: list[float], bins: int = 10) -> float:
    total = len(y)
    result = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if lo <= s < hi or (b == bins - 1 and s == hi)]
        if idx:
            result += len(idx) / total * abs(sum(y[i] for i in idx) / len(idx) - sum(scores[i] for i in idx) / len(idx))
    return result


def evaluate(rows: list[dict[str, Any]], method: str, learned: dict[str, Any] | None = None) -> dict[str, Any]:
    y, scores, nodes = [], [], []
    for row in rows:
        p, node, _ = predict(row, method, learned)
        y.append(row["label"])
        scores.append(p)
        nodes.append(node)
    negatives = [s for yy, s in zip(y, scores) if not yy]
    threshold = max(negatives, default=0.0)
    eligible = [i for i, s in enumerate(scores) if s > threshold + 1e-9]
    recall_low_fpr = sum(y[i] for i in eligible) / max(1, sum(y))
    loc = sum(1 for r, n in zip(rows, nodes) if r["label"] and n == r["risk_node"]) / max(1, sum(y))
    return {
        "n": len(rows),
        "positive_rate": sum(y) / len(y),
        "auprc": auc_pr(y, scores),
        "low_fpr_recall": recall_low_fpr,
        "ece": ece(y, scores),
        "risk_node_localization": loc,
        "mean_score_safe": sum(s for yy, s in zip(y, scores) if not yy) / max(1, sum(1 for yy in y if not yy)),
        "mean_score_risk": sum(s for yy, s in zip(y, scores) if yy) / max(1, sum(y)),
        "predictions": [{"id": r["id"], "label": r["label"], "score": round(s, 6), "node": n, "risk_node": r["risk_node"]} for r, s, n in zip(rows, scores, nodes)],
    }


def bootstrap_auprc(rows: list[dict[str, Any]], method: str, learned: dict[str, Any], seed: int, n_boot: int = 100) -> dict[str, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(evaluate(sample, method, learned)["auprc"])
    values.sort()
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1))
    return {"seed": seed, "n_boot": n_boot, "mean_auprc": mean, "sd_auprc": sd, "q025": values[2], "q975": values[97]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:58661/v1")
    parser.add_argument("--models", nargs="+", default=["gpt-5.5", "gpt-5.3-codex"])
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--api-sample", type=int, default=1, help="真实 API 调用的工作流数量；其余工作流用规范图离线记录")
    parser.add_argument("--reuse-api-results", action="store_true", help="复用已有 DAG/API 记录，仅重算指标与报告，不发起网络请求")
    args = parser.parse_args()
    key = os.environ.get("AGENT_API_KEY", "")

    plan = {
        "scope": "first runnable offline pilot for API-observable acyclic MAS risk monitoring",
        "data": "self-built fixed DAG scenarios; public HarnessAudit/A2ASecBench/MAST data not locally available",
        "methods": ["rules", "text", "flat_bn", "dynamic_only", "naive_noisy_or", "hierarchical_bn"],
        "api_models_requested": args.models,
        "api_key_written": False,
        "reuse_api_results": args.reuse_api_results,
        "api_sample_requested": args.api_sample,
        "workflow_count": len(WORKFLOWS),
        "coverage_note": (
            "每个模型/策略均覆盖 6 个工作流；每个工作流仅运行 1 次，结论仍属探索性"
            if args.api_sample >= len(WORKFLOWS)
            else f"每个模型/策略仅覆盖 {min(args.api_sample, len(WORKFLOWS))}/{len(WORKFLOWS)} 个工作流，只能作探索性结论"
        ),
        "dag_semantic_alignment": {
            "method": "deterministic explicit ID aliases plus normalized role/label keyword matching",
            "embedding_used": False,
            "human_adjudication_used": False,
            "unknown_nodes": "left unmapped; no semantic hit and retained as false positives",
            "edge_scoring": "directed endpoint pairs after node mapping; no transitive-closure credit",
        },
    }
    (OUT / "execution_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    dag_results = []
    api_status = []
    if args.reuse_api_results:
        result_path = OUT / "dag_extraction_results.json"
        status_path = OUT / "api_status.json"
        if not result_path.exists() or not status_path.exists():
            raise FileNotFoundError("--reuse-api-results requires existing dag_extraction_results.json and api_status.json")
        stored_results = json.loads(result_path.read_text(encoding="utf-8"))
        api_status = json.loads(status_path.read_text(encoding="utf-8"))
        quality_keys = {
            "node_f1", "edge_f1", "exact_node_f1", "exact_edge_f1", "semantic_node_f1",
            "semantic_edge_f1", "node_mapping", "semantic_unmatched", "semantic_node_mapping",
            "semantic_mapped_edges", "semantic_scored_edges", "semantic_unmapped_nodes", "acyclic",
        }
        for stored in stored_results:
            dag = {"nodes": stored.get("nodes", []), "edges": stored.get("edges", [])}
            base = {k: v for k, v in stored.items() if k not in quality_keys}
            dag_results.append({**base, **dag_quality(dag, CANONICAL), "nodes": dag["nodes"], "edges": dag["edges"]})

    if not dag_results:
        for model in args.models:
            for strategy in ("direct", "decompose", "edge_audit"):
                for wi, (workflow, desc) in enumerate(WORKFLOWS):
                    parsed = None
                    error = "skipped"
                    meta = {"attempts": 0, "status": None, "response_summary": None}
                    if not args.skip_api and key and wi < args.api_sample:
                        text, error, meta = api_chat(args.base_url, key, model, strategy_prompt(strategy, workflow, desc))
                        parsed = parse_dag(text)
                    elif wi >= args.api_sample:
                        error = "not_sampled"
                    dag, fallback = normalize_dag(parsed)
                    quality = dag_quality(dag, CANONICAL)
                    api_returned = error is None and parsed is not None
                    dag_results.append({"model": model, "strategy": strategy, "workflow": workflow, "fallback": fallback, "api_returned": api_returned, **quality, "nodes": dag["nodes"], "edges": dag["edges"]})
                    api_status.append({"model": model, "strategy": strategy, "workflow": workflow, "ok": api_returned, "error": error, "http_status": meta.get("status"), "attempts": meta.get("attempts"), "response_summary": meta.get("response_summary")})

    rows = make_scenarios()
    train = [r for r in rows if r["family"] % 3 != 0]
    test = [r for r in rows if r["family"] % 3 == 0]
    learned = fit_learned_bn(train)
    plan["methods"].append("learned_bn")
    metrics = {m: evaluate(test, m, learned) for m in plan["methods"]}
    learned_bootstrap = [bootstrap_auprc(test, "learned_bn", learned, seed) for seed in (11, 23, 37)]
    sampled_status = [x for x in api_status if x["error"] not in ("not_sampled", "skipped")]
    strategy_summary = []
    for model in args.models:
        for strategy in ("direct", "decompose", "edge_audit"):
            subset = [x for x in dag_results if x["model"] == model and x["strategy"] == strategy and not x["fallback"]]
            attempted = [x for x in api_status if x["model"] == model and x["strategy"] == strategy and x["error"] not in ("not_sampled", "skipped")]
            strategy_summary.append({
                "model": model,
                "strategy": strategy,
                "attempted": len(attempted),
                "success": len(subset),
                "success_rate": len(subset) / max(1, len(attempted)),
                # ``mean_node_f1``/``mean_edge_f1`` are retained as exact-match
                # aliases for compatibility with the first report schema.
                "mean_node_f1": sum(x["node_f1"] for x in subset) / max(1, len(subset)),
                "mean_edge_f1": sum(x["edge_f1"] for x in subset) / max(1, len(subset)),
                "mean_exact_node_f1": sum(x["exact_node_f1"] for x in subset) / max(1, len(subset)),
                "mean_exact_edge_f1": sum(x["exact_edge_f1"] for x in subset) / max(1, len(subset)),
                "mean_semantic_node_f1": sum(x["semantic_node_f1"] for x in subset) / max(1, len(subset)),
                "mean_semantic_edge_f1": sum(x["semantic_edge_f1"] for x in subset) / max(1, len(subset)),
            })
    availability_by_model = []
    for model in args.models:
        attempted = [x for x in api_status if x["model"] == model and x["error"] not in ("not_sampled", "skipped")]
        status_counts: dict[str, int] = defaultdict(int)
        for item in attempted:
            status_key = str(item["http_status"]) if item.get("http_status") is not None else ("ok" if item.get("ok") else "error")
            status_counts[status_key] += 1
        availability_by_model.append({
            "model": model,
            "attempted": len(attempted),
            "success": sum(1 for x in attempted if x.get("ok")),
            "success_rate": sum(1 for x in attempted if x.get("ok")) / max(1, len(attempted)),
            "http_status_counts": dict(sorted(status_counts.items())),
        })
    report = {
        "run_id": "MODEL-TEST-V1",
        "plan": plan,
        "dataset": {
            "source": "self-built",
            "n_total": len(rows),
            "n_train_like": len(train),
            "n_test": len(test),
            "split": "workflow family holdout: families 0 and 3 are test",
            "risk_mechanisms": sorted({r["mechanism"] for r in rows}),
        },
        "network": {
            "system_dag": CANONICAL,
            "local_templates": ["goal/input pollution", "authorization", "provenance/memory", "tool/external effect", "composition", "runtime accumulation"],
            "dynamic_features": ["cross_agent_propagation", "delegation/shared-state proxy", "permission/tool flow proxy", "anomaly_path", "risk_accumulation"],
            "parameters": {
                "intercept": -2.8,
                "local_weights": {"input": 2.2, "auth": 2.0, "memory": 1.8, "tool": 1.7, "composition": 1.9, "runtime": 1.4},
                "interaction_weight": 0.8,
                "parameter_status": "hand-calibrated pilot parameters; not learned CPTs",
            },
            "output": ["system risk probability", "risk node", "evidence path"],
        },
        "dag_extraction": {
            "summary": {
                "n_calls": len(dag_results),
                "api_sampled_calls": len(sampled_status),
                "api_sampled_success_rate": sum(x["ok"] for x in sampled_status) / max(1, len(sampled_status)),
                "api_success_rate_all_workflows": sum(not x["fallback"] for x in dag_results) / max(1, len(dag_results)),
                "real_api_mean_node_f1": sum(x["node_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "real_api_mean_edge_f1": sum(x["edge_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "real_api_mean_exact_node_f1": sum(x["exact_node_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "real_api_mean_exact_edge_f1": sum(x["exact_edge_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "real_api_mean_semantic_node_f1": sum(x["semantic_node_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "real_api_mean_semantic_edge_f1": sum(x["semantic_edge_f1"] for x in dag_results if x["api_returned"]) / max(1, sum(x["api_returned"] for x in dag_results)),
                "fallback_count": sum(x["fallback"] for x in dag_results),
            },
            "semantic_alignment": plan["dag_semantic_alignment"],
            "availability_by_model": availability_by_model,
            "strategy_summary": strategy_summary,
            "by_model_strategy": dag_results,
            "api_status": api_status,
        },
        "metrics": metrics,
        "learned_bn": {"fit_scope": "train workflow families (family % 3 != 0)", "test_scope": "held-out workflow families (family % 3 == 0)", "bootstrap": learned_bootstrap},
        "error_analysis": [
            "Pipeline smoke test only: 48 self-built rows with hand-designed mechanism templates, not a public benchmark result.",
            "Rules are strong on explicit authorization violations but miss compositional and memory risks.",
            "Text baseline sees lexical hints and is intentionally weak on provenance and cross-agent structure.",
            "Hierarchical BN parameters are hand-calibrated; learned_bn is the only train-fit variant and still has a tiny held-out family test.",
            "Lead time is represented by event count only; real timestamped execution traces are still required.",
            "LLM-as-Judge and published MAS baselines are registered but not claimed as reproduced without their exact releases.",
            "DAG semantic-match uses deterministic ID aliases plus normalized role/label keywords; it is not embedding-based and has no human adjudication in this run. Unknown predicted nodes remain unmatched, receive no hit, and remain false positives in the denominator. Semantic edges are scored as directed endpoint pairs after mapping; extra/redundant edges are not silently removed.",
            f"DAG API coverage: {plan['coverage_note']}.",
        ],
        "next_iteration": [
            "Replace synthetic rows with a verified HarnessAudit/A2ASecBench/MAST subset.",
            "Learn/calibrate CPTs on train families and add confidence intervals via bootstrap.",
            "Add true tool-return/external-effect timestamps for lead-time evaluation.",
            "Implement one native MAS baseline under the same visible-field contract.",
        ],
    }
    (OUT / "dag_extraction_results.json").write_text(json.dumps(dag_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "api_status.json").write_text(json.dumps(api_status, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    availability_line = "；".join(
        f"{item['model']}：{item['success']}/{item['attempted']} 成功（"
        + ", ".join(f"HTTP {status}×{count}" for status, count in item["http_status_counts"].items())
        + "）"
        for item in availability_by_model
    ) or "无真实 API 调用"
    lines = [
        "# Model Test V1",
        "",
        "本轮是最小可运行纵切片：自建固定 DAG 场景，API 仅用于 DAG 提取；风险监测为外部可观测、只预警不阻断。",
        "",
        "## DAG 提取",
        f"- 组合记录数：{len(dag_results)}；真实 API 抽样调用：{report['dag_extraction']['summary']['api_sampled_calls']}",
        f"- 抽样 API 成功率：{report['dag_extraction']['summary']['api_sampled_success_rate']:.3f}",
        f"- 全部工作流组合的非 fallback 比例：{report['dag_extraction']['summary']['api_success_rate_all_workflows']:.3f}",
        f"- 真实 API 返回的平均节点 F1（exact）：{report['dag_extraction']['summary']['real_api_mean_exact_node_f1']:.3f}",
        f"- 真实 API 返回的平均边 F1（exact）：{report['dag_extraction']['summary']['real_api_mean_exact_edge_f1']:.3f}",
        f"- 语义对齐后的平均节点 F1：{report['dag_extraction']['summary']['real_api_mean_semantic_node_f1']:.3f}",
        f"- 语义对齐后的平均边 F1：{report['dag_extraction']['summary']['real_api_mean_semantic_edge_f1']:.3f}",
        f"- fallback 记录数：{report['dag_extraction']['summary']['fallback_count']}（不计入上述均值）",
        f"- 模型可用性：{availability_line}",
        "",
        "节点语义对齐采用确定性的 ID 别名 + 规范化角色词匹配（未使用嵌入或人工复核）；未知节点不映射、不计入命中，但作为误报保留在分母。语义边按映射后的有向端点对直接计算，不用传递闭包替代额外边。",
        f"覆盖范围：{plan['coverage_note']}。若只抽取 2 个工作流，该结果只能作探索性结论；本轮即使覆盖 6 个工作流，也仍是每组合一次的探索性测试。",
        "每种模型/策略的抽样结果见 `dag_extraction_results.json` 和 `api_status.json`；未抽样工作流（如有）使用规范 DAG，不计作 API 成功。",
        "",
        "## 第一版网络",
        "",
        "节点：`intake → extract → verify → analyze → policy → report`。",
        "边：`intake→extract, extract→verify, verify→analyze, analyze→policy, policy→report`。",
        "局部模板：目标/输入污染、授权、来源/记忆、工具/外部效果、组合风险、运行时累积。",
        "参数：截距 -2.8；局部权重 input/auth/memory/tool/composition/runtime = 2.2/2.0/1.8/1.7/1.9/1.4；跨节点交互权重 0.8。均为本轮手工校准。",
        "",
        "## 测试集指标",
        "",
        "| 方法 | AUPRC | 低误报率召回 | ECE | 风险节点定位 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, value in metrics.items():
        lines.append(f"| {method} | {value['auprc']:.3f} | {value['low_fpr_recall']:.3f} | {value['ece']:.3f} | {value['risk_node_localization']:.3f} |")
    lines += [
        "",
        "`learned_bn` 仅用训练工作流族拟合条件概率，在留出工作流族测试；三组 bootstrap 的 AUPRC 均值/标准差见 `report.json`。",
        "",
    ]
    lines += [
        "",
        "## 可信边界",
        "",
        "- 可信：代码可运行、固定场景定义、DAG 提取成功/失败记录、同输入下的离线基线相对比较。",
        "- 假设：风险模板、先验和权重；自建标签；事件数作为预警时间代理。",
        "- 未完成：公开 HarnessAudit/A2ASecBench/MAST 的版本锁定与复现，真实外部效果，正式 LLM-as-Judge/原生 MAS baseline。",
        "",
        "详细结果见 `report.json`、`metrics.json`、`dag_extraction_results.json`。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": "MODEL-TEST-V1", "output": str(OUT), "api_sampled_success_rate": report["dag_extraction"]["summary"]["api_sampled_success_rate"], "metrics": {k: {m: round(v[m], 4) for m in ("auprc", "low_fpr_recall", "ece", "risk_node_localization")} for k, v in metrics.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
