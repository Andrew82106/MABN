'Project canonical A2ASecBench runs onto the shared Layer-2 event schema.\n\nOnly the five fixed run files under ``experiments/a2asecbench/runs/a2asecbench_v1`` are read.  The\nbenchmark rows are preserved as episode summaries and each explicit\n``details.events`` item becomes one observable runtime event.  This adapter is\nread-only with respect to the benchmark outputs: it never rewrites a source\nJSONL file and it does not turn an attack/control scenario into a security\nlabel.\n'

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/a2asecbench/runs/a2asecbench_v1'
DEFAULT_OUTPUT = ROOT / 'data/processed/mas_adapter/a2asecbench'
SCHEMA_VERSION = "mas-adapter-v1"
FEATURES = (
    "untrusted_input",
    "permission_mismatch",
    "memory_poison",
    "tool_side_effect",
    "cross_agent_propagation",
    "anomaly_path",
    "risk_accumulation",
)
RUN_ORDER = (
    ("as", "AS"),
    ("asrf_eval", "ASRF"),
    ("atsi_eval", "ATSI"),
    ("co_eval", "CO"),
    ("hotf_eval", "HOTF"),
)
SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|authorization|secret|password|passwd|credential|"
    r"private[_-]?key|bearer|hidden[_-]?prompt|system[_-]?prompt|internal[_-]?notes?)",
    re.IGNORECASE,
)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _relative(path: Path, base: Path = ROOT.parent) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "allowed", "allow", "success", "succeeded"}:
            return True
        if normalized in {"false", "no", "n", "0", "denied", "deny", "failed", "failure"}:
            return False
    return None


def _text(value: Any, limit: int = 6000) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        result = value
    else:
        result = _canonical(_redact(value))
    return result[:limit] if result else None


def _event_type(event: dict[str, Any]) -> str:
    value = event.get("event_type") or event.get("type")
    return str(value) if value is not None else "unknown"


def _metrics(details: dict[str, Any]) -> dict[str, Any]:
    value = details.get("metrics")
    return value if isinstance(value, dict) else {}


def _events(details: dict[str, Any]) -> list[dict[str, Any]]:
    value = details.get("events")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def observed_violation(details: dict[str, Any]) -> bool | None:
    """Return an observed runtime indicator, not a benchmark/risk label."""
    metrics = _metrics(details)
    events = _events(details)
    if not metrics and not events:
        # AS/AgentCard rows have an outcome but no runtime event stream.
        return None
    positive_counters = (
        "forbidden_uri_dereferences",
        "artifact_execution_error_count",
        "cycle_detected_count",
        "max_depth_exceeded_count",
    )
    if any(float(metrics.get(key, 0) or 0) > 0 for key in positive_counters):
        return True
    positive_types = {
        "artifact_executed",
        "task_store_pressure",
        "cycle_detected",
        "max_depth_exceeded",
    }
    return any(_event_type(event) in positive_types for event in events)


def _run_id(row: dict[str, Any], attack_id: str, source_name: str) -> str:
    value = row.get("run_id")
    if value:
        return str(value)
    config_hash = row.get("config_hash")
    if config_hash:
        return f"a2a-v1:{attack_id}:{config_hash}"
    # This fallback is explicitly marked as derived in source.context.
    return f"a2a-v1:{attack_id}:{source_name}"


def _episode_id(row: dict[str, Any], run_id: str) -> str:
    case_id = str(row.get("case_id", "unknown-case"))
    trial_index = row.get("trial_index", 0)
    return f"{run_id}:{case_id}:trial-{trial_index}"


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _endpoint(event: dict[str, Any], evidence: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = _first(event.get(name), evidence.get(name))
        if value is not None:
            return str(value)
    return None


def _actor_role(event: dict[str, Any], evidence: dict[str, Any]) -> tuple[str | None, str | None]:
    actor = _first(event.get("actor_id"), event.get("agent_id"))
    actor = str(actor) if actor is not None else None
    explicit_role = _first(event.get("actor_role"), event.get("role"))
    evidence_agent = _first(evidence.get("agent_id"), evidence.get("actor_id"))
    evidence_role = _first(evidence.get("agent_role"), evidence.get("role"))
    # evidence.agent_role normally describes the peer named by evidence.agent_id,
    # not the event actor (often ``sut``).  Join it only when the identities
    # explicitly agree; otherwise retain it as target_role in context.
    role = explicit_role
    if role is None and actor is not None and evidence_agent is not None and str(evidence_agent) == actor:
        role = evidence_role
    return actor, (str(role) if role is not None else None)


def _message_endpoints(
    event_type: str,
    event: dict[str, Any],
    evidence: dict[str, Any],
    actor: str | None,
) -> tuple[str | None, str | None]:
    source = _endpoint(event, evidence, "source_agent_id", "message_source", "sender_id")
    target = _endpoint(event, evidence, "target_agent_id", "message_target", "recipient_id")
    if event_type == "delegation_edge":
        source = _endpoint(event, evidence, "source_agent_id")
        target = _endpoint(event, evidence, "target_agent_id")
    elif event_type in {"a2a_message_sent", "a2a_agent_card_fetched"}:
        source = source or actor
        target = target or _endpoint(event, evidence, "agent_id", "next_agent_id")
    elif event_type == "peer_response_returned":
        source = source or _endpoint(event, evidence, "agent_id") or actor
        target = target or _endpoint(event, evidence, "next_agent_id")
    return source, target


def _parent(event: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    value = _first(event.get("parent"), event.get("parent_task_id"), evidence.get("parent_task_id"))
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if item is not None and str(item)]


def _permission(event: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any] | None:
    value = _first(event.get("permission"), evidence.get("permission"))
    if isinstance(value, dict):
        return _redact(value)
    if value is not None:
        return {"permission": _redact(value)}
    return None


def _external_effect(event: dict[str, Any], evidence: dict[str, Any]) -> tuple[bool | None, str | None, str | None]:
    for source, key in ((event, "external_effect"), (event, "external_effect_observed"), (evidence, "external_effect")):
        if key in source:
            parsed = _as_bool(source.get(key))
            if parsed is not None:
                return parsed, f"explicit:{key}", str(_first(source.get("external_effect_scope"), "unknown"))
    event_type = _event_type(event)
    component = str(event.get("component") or "")
    # The offline harness records local trap/preview activity.  It is an
    # observed local sink, not an Internet side effect; make that distinction
    # explicit instead of treating a tool/component name as proof of effect.
    if component == "local_http_trap" and event_type in {"uri_dereference", "artifact_executed"}:
        return False, "offline_reference_local_harness_sink", "local_sandbox"
    return None, None, None


def _feature_flags(
    row: dict[str, Any],
    details: dict[str, Any],
    event: dict[str, Any],
    event_type: str,
    source: str | None,
    target: str | None,
    effect: bool | None,
) -> tuple[dict[str, int], list[str]]:
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    private = event.get("private_tags") if isinstance(event.get("private_tags"), dict) else {}
    metrics = _metrics(details)
    # Feature flags are event-local observations.  Row-level scenario names
    # and aggregate metrics stay in outcome_flags/context so they cannot make
    # every event in an episode look like the same violation.
    text_blob = " ".join(
        str(value or "")
        for value in (event_type, event.get("public_surface"), evidence, private)
    ).lower()
    reasons: list[str] = []
    untrusted = any(term in text_blob for term in ("spoof", "forgery", "injection", "untrusted", "malicious", "poison", "cloaking"))
    permission = (
        str(evidence.get("uri_class", "")).lower() == "forbidden_internal"
        or any(term in text_blob for term in ("unauthorized", "forbidden", "permission mismatch", "bypass"))
    )
    memory = any(term in text_blob for term in ("memory poison", "memory_poison", "shared memory", "persisted state"))
    side_effect_types = {"artifact_executed", "uri_dereference", "task_created", "task_store_pressure"}
    side_effect = event_type in side_effect_types or effect is True
    cross_agent = bool(source and target and source != target) or event_type in {
        "a2a_message_sent", "a2a_agent_card_fetched", "peer_response_returned", "delegation_edge"
    }
    anomaly = event_type in {"artifact_executed", "cycle_detected", "max_depth_exceeded", "task_store_pressure"}
    anomaly = anomaly or permission or any(term in text_blob for term in ("error", "failed", "timeout", "rejected"))
    accumulation = event_type in {"cycle_detected", "task_store_pressure", "task_store_snapshot"}
    # Event-local counters may indicate repetition without importing the row's
    # aggregate totals into every event.
    accumulation = accumulation or (
        event_type == "delegation_edge" and float(evidence.get("depth", 0) or 0) > 1
    )
    flags = {
        "untrusted_input": int(untrusted),
        "permission_mismatch": int(permission),
        "memory_poison": int(memory),
        "tool_side_effect": int(side_effect),
        "cross_agent_propagation": int(cross_agent),
        "anomaly_path": int(anomaly),
        "risk_accumulation": int(accumulation),
    }
    if untrusted:
        reasons.append("visible scenario/evidence indicates spoofing, forgery, injection, or untrusted input")
    if permission:
        reasons.append("visible forbidden/authorization evidence")
    if memory:
        reasons.append("visible memory/shared-state evidence")
    if side_effect:
        reasons.append(f"explicit runtime event type {event_type!r} or explicit effect")
    if cross_agent:
        reasons.append("explicit A2A/delegation endpoint or protocol event")
    if anomaly:
        reasons.append("explicit runtime anomaly/violation evidence")
    if accumulation:
        reasons.append("explicit repeated/cumulative runtime count")
    return flags, reasons


def _outcome_flags(
    row: dict[str, Any],
    details: dict[str, Any],
    violation: bool | None,
    *,
    include_row_evidence: bool = True,
) -> dict[str, Any]:
    """Keep episode-level metrics once; avoid duplicating large payloads per event."""
    result = {
        "attack_id": row.get("attack_id"),
        "case_id": row.get("case_id"),
        "trial_index": row.get("trial_index"),
        "scenario": details.get("scenario"),
        "domain": details.get("domain"),
        "difficulty": details.get("difficulty"),
        "vector": details.get("vector"),
        "scenario_role": "control" if details.get("is_control") is True else "attack",
        "observed_violation": violation,
        "success": row.get("success"),
        "control_pass": details.get("control_pass"),
        "overblocked": _first(row.get("overblocked"), details.get("overblocked")),
    }
    if include_row_evidence:
        result["metrics"] = _metrics(details)
        result["success_evidence"] = details.get("success_evidence")
    return _redact(result)


def normalize_event(
    row: dict[str, Any],
    details: dict[str, Any],
    event: dict[str, Any],
    event_index: int,
    source_path: Path,
    source_line: int,
    run_id: str,
    episode_id: str,
    violation: bool | None,
) -> dict[str, Any]:
    event_type = _event_type(event)
    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    private = event.get("private_tags") if isinstance(event.get("private_tags"), dict) else {}
    actor, role = _actor_role(event, evidence)
    source, target = _message_endpoints(event_type, event, evidence, actor)
    effect, effect_basis, effect_scope = _external_effect(event, evidence)
    parent = _parent(event, evidence)
    flags, reasons = _feature_flags(row, details, event, event_type, source, target, effect)
    timestamp_value = event.get("timestamp_monotonic")
    timestamp = str(timestamp_value) if timestamp_value is not None else None
    event_id = f"{episode_id}:event-{event_index:04d}"
    # Full row metrics/success evidence are retained once in episodes.jsonl;
    # event rows keep only compact outcome flags plus their own evidence.
    outcome = _outcome_flags(row, details, violation, include_row_evidence=False)
    outcome["event_type"] = event_type
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "record_kind": "runtime_event",
        "event_type": event_type,
        "stage": None,
        "agent_id": actor,
        "role": role,
        "message": None,
        "message_text": None,
        "message_id": None,
        "message_source": source,
        "message_target": target,
        "tool": None,
        "tool_arguments": None,
        "tool_result": None,
        "timestamp": timestamp,
        "parent": parent,
        "chain_parent": None,
        "permission": _permission(event, evidence),
        "external_effect": effect,
        "external_effect_basis": effect_basis,
        "external_effect_scope": effect_scope,
        "risk_label": None,
        "risk_label_source": None,
        "outcome_flags": outcome,
        "features": flags,
        "feature_reasons": reasons,
        "source": {
            "path": _relative(source_path),
            "line": source_line,
            "event_index": event_index,
            "artifact_type": "a2asecbench_offline_runtime",
            "authenticity": "observed_offline_reference_sut",
            "benchmark": "A2ASecBench",
            "attack_id": row.get("attack_id"),
            "config_hash": row.get("config_hash"),
            "timestamp_kind": "timestamp_monotonic_relative",
        },
        "context": {
            "component": event.get("component"),
            "public_surface": event.get("public_surface"),
            "evidence": _redact(evidence),
            "private_tags": _redact(private),
            "target_role": _first(evidence.get("agent_role"), evidence.get("target_role")),
            "scenario_role": "control" if details.get("is_control") is True else "attack",
            "scenario": details.get("scenario"),
            "domain": details.get("domain"),
            "vector": details.get("vector"),
            "sequence": event_index,
            "trial_id": event.get("trial_id"),
            "timestamp_monotonic": timestamp_value,
            "causal_edge_policy": "only explicit endpoints/parent retained; sequence is not causal",
        },
        "eligibility": {
            "structural_graph": bool(actor or role or source or target or event.get("component")),
            "temporal_graph": timestamp is not None,
            "bn_feature_row": True,
            "risk_evaluation": False,
            "replay_evidence": True,
        },
        "source_record_hash": _sha256_value(event),
    }


def normalize_episode(
    row: dict[str, Any],
    details: dict[str, Any],
    source_path: Path,
    source_line: int,
    run_id: str,
    episode_id: str,
    events: list[dict[str, Any]],
    violation: bool | None,
) -> dict[str, Any]:
    aggregate = {name: int(any(event.get("features", {}).get(name, 0) for event in events)) for name in FEATURES}
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "benchmark_episode",
        "run_id": run_id,
        "episode_id": episode_id,
        "attack_id": row.get("attack_id"),
        "case_id": row.get("case_id"),
        "trial_index": row.get("trial_index"),
        "scenario_role": "control" if details.get("is_control") is True else "attack",
        "observed_violation": violation,
        "risk_label": None,
        "success": row.get("success"),
        "control_pass": details.get("control_pass"),
        "overblocked": _first(row.get("overblocked"), details.get("overblocked")),
        "event_count": len(events),
        # replay_bn_adapter accepts either an event list or this compact count
        # when it builds feature evidence for an episode.
        "events": len(events),
        "event_ids": [event["event_id"] for event in events],
        "features": aggregate,
        "outcome_flags": _outcome_flags(row, details, violation),
        "source": {
            "path": _relative(source_path),
            "line": source_line,
            "artifact_type": "a2asecbench_offline_row",
            "authenticity": "observed_offline_reference_sut",
            "benchmark": "A2ASecBench",
            "config_hash": row.get("config_hash"),
        },
        "eligibility": {
            "structural_graph": bool(events),
            "temporal_graph": any(event.get("timestamp") is not None for event in events),
            "bn_feature_row": True,
            "risk_evaluation": False,
            "replay_evidence": True,
        },
        "source_record_hash": _sha256_value(row),
    }


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "mas-adapter-a2asecbench-runtime-event.schema.json",
        "title": "A2ASecBench runtime event projected onto mas-adapter-v1",
        "type": "object",
        "required": [
            "schema_version", "event_id", "record_kind", "event_type", "agent_id", "role",
            "message", "tool", "timestamp", "parent", "permission", "external_effect",
            "risk_label", "features", "source",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "event_id": {"type": "string"},
            "run_id": {"type": ["string", "null"]},
            "episode_id": {"type": ["string", "null"]},
            "record_kind": {"type": "string"},
            "event_type": {"type": "string"},
            "agent_id": {"type": ["string", "null"]},
            "role": {"type": ["string", "null"]},
            "message": {},
            "message_text": {"type": ["string", "null"]},
            "message_source": {"type": ["string", "null"]},
            "message_target": {"type": ["string", "null"]},
            "tool": {"type": ["string", "null"]},
            "tool_arguments": {},
            "tool_result": {},
            "timestamp": {"type": ["string", "null"]},
            "parent": {"type": "array", "items": {"type": "string"}},
            "permission": {"type": ["object", "null"]},
            "external_effect": {"type": ["boolean", "null"]},
            "risk_label": {"type": ["integer", "null"], "enum": [0, 1, None]},
            "features": {"type": "object", "required": list(FEATURES)},
            "source": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    invalid: list[dict[str, Any]] = []
    blank_lines = 0
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return rows, [{"file": _relative(path), "line": 0, "error": f"{type(exc).__name__}: {exc}"}], blank_lines
    for line_no, line in enumerate(raw_lines, 1):
        if not line.strip():
            blank_lines += 1
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid.append({"file": _relative(path), "line": line_no, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(value, dict):
            invalid.append({"file": _relative(path), "line": line_no, "error": f"expected object, got {type(value).__name__}"})
            continue
        rows.append((line_no, value))
    return rows, invalid, blank_lines


def _discover(input_path: Path) -> list[tuple[str, Path, str]]:
    """Return (directory name, run path, attack id) in deterministic order."""
    if input_path.is_file():
        name = input_path.parent.name
        attack = dict(RUN_ORDER).get(name, name.upper())
        return [(name, input_path, attack)]
    result: list[tuple[str, Path, str]] = []
    for name, attack in RUN_ORDER:
        path = input_path / name / "run.jsonl"
        if path.is_file():
            result.append((name, path, attack))
    return result


def _coverage(
    files: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    blank_lines: int,
    manifest_match: dict[str, Any] | None,
) -> dict[str, Any]:
    total = len(events)
    field_names = (
        "agent_id", "role", "message", "tool", "timestamp", "parent", "permission",
        "external_effect", "risk_label", "message_source", "message_target",
    )
    field_counts = {
        field: sum(bool(event.get(field)) if field == "parent" else event.get(field) is not None for event in events)
        for field in field_names
    }
    effects = Counter("true" if event.get("external_effect") is True else "false" if event.get("external_effect") is False else "null" for event in events)
    violations = Counter("true" if item.get("observed_violation") is True else "false" if item.get("observed_violation") is False else "null" for item in episodes)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "A2ASecBench",
        "files": files,
        "episodes": len(episodes),
        "events": total,
        "invalid_json_lines": invalid,
        "blank_lines": blank_lines,
        "scenario_roles": dict(Counter(str(item.get("scenario_role")) for item in episodes)),
        "observed_violation_episodes": dict(violations),
        "explicit_risk_label_events": sum(event.get("risk_label") is not None for event in events),
        "field_coverage": {
            field: {"count": count, "fraction": count / total if total else 0.0}
            for field, count in field_counts.items()
        },
        "external_effect_states": dict(effects),
        "event_types": dict(Counter(str(event.get("event_type")) for event in events)),
        "status": "coverage_only_no_risk_labels" if episodes else "no_canonical_rows",
        "risk_label_policy": "A2ASecBench scenario_role and observed_violation are retained separately; no attack/control inference is mapped to risk_label.",
        "timestamp_policy": "timestamp_monotonic is retained as a relative monotonic value in context and stringified in timestamp.",
        "dynamic_graph_contract": {
            "nodes": "explicit actor_id, evidence agent IDs, and component values",
            "edges": "explicit message/delegation endpoints and parent_task_id only",
            "sequence": "retained for ordering; never promoted to causal edge",
            "evidence": "public_surface, redacted evidence/private_tags, metrics and outcome_flags",
        },
        "canonical_manifest_match": manifest_match,
    }


def _manifest_match(input_path: Path, files: list[dict[str, Any]]) -> dict[str, Any] | None:
    manifest_path = input_path / "A2ASECBENCH_MANIFEST.json" if input_path.is_dir() else input_path.parent.parent / "A2ASECBENCH_MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"manifest": _relative(manifest_path), "available": False}
    expected = {str(run.get("run_path")): run.get("run_sha256") for run in manifest.get("runs", []) if isinstance(run, dict)}
    checked = []
    for file in files:
        path = Path(file["absolute_path"])
        key = _relative(path)
        checked.append({"path": key, "expected_sha256": expected.get(key), "observed_sha256": file.get("sha256"), "match": expected.get(key) == file.get("sha256") if key in expected else None})
    return {"manifest": _relative(manifest_path), "available": True, "files": checked}


def _report(coverage: dict[str, Any], output_dir: Path) -> str:
    roles = coverage.get("scenario_roles", {})
    violations = coverage.get("observed_violation_episodes", {})
    lines = [
        "# A2ASecBench Layer-2 adapter",
        "",
        "这是对 canonical A2ASecBench 离线 reference SUT 输出的只读投影；源 JSONL 未被改写。它不是我们的模型结果，也不把 benchmark 的攻击/控制身份当作 `risk_label`。",
        "",
        f"- Canonical files: **{len(coverage.get('files', []))}**",
        f"- Episodes: **{coverage.get('episodes', 0)}** (`{roles}`)",
        f"- Runtime events: **{coverage.get('events', 0)}**",
        f"- Invalid JSON lines: **{len(coverage.get('invalid_json_lines', []))}**",
        f"- Observed violation states: `{violations}`",
        f"- Explicit risk-label events: **{coverage.get('explicit_risk_label_events', 0)}**",
        f"- Status: `{coverage.get('status')}`",
        "",
        "## 字段映射",
        "",
        "- 每个 benchmark row 保留为一个 episode；`details.events[]` 的每项成为一个 runtime event。",
        "- `actor_id` → `agent_id`；只有显式一致的角色才映射到 `role`。peer 的 `agent_role` 保留在 `context.target_role`，避免把 `sut` 错标为 peer 角色。",
        "- A2A message/delegation 的显式 source/target → `message_source`/`message_target`；`parent_task_id` → `parent`。只有时间顺序时保留 `context.sequence`，不制造因果边。",
        "- `timestamp_monotonic` 原样保留在 `context.timestamp_monotonic`，并以字符串写入共享 `timestamp`；它是离线 harness 的相对单调时间。",
        "- `scenario_role` 与 `observed_violation` 放在 episode/outcome flags；没有显式 per-event 安全标签时 `risk_label=null`。",
        "- 本地 HTTP trap/preview 只在有明确 local harness sink 时标为 `external_effect=false, scope=local_sandbox`；普通组件或工具名不会自动推断外部副作用。",
        "",
        "## 运行与重放",
        "",
        "```powershell",
        "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\adapt_a2asecbench.py",
        "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\replay_bn_adapter.py --input paperAlpha\\data\\mas_adapter\\a2asecbench\\episodes.jsonl --output paperAlpha\\data\\mas_adapter\\a2asecbench\\bn_replay.json",
        "```",
        "",
        "BN replay 在没有完整显式风险标签时只能输出 exploratory score；不得据此报告 AUPRC、ECE 或召回率。",
        "",
        f"Output directory: `{output_dir}`。机器可读覆盖见 `coverage.json`，事件约束见 `schema.json`；当前 replay 快照为 `bn_replay.json`。",
        "",
    ]
    return "\n".join(lines)


def adapt(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if input_path.is_dir() and (output_dir == input_path or output_dir.is_relative_to(input_path)):
        raise ValueError("output directory must be outside the canonical input root")
    discovered = _discover(input_path)
    if not discovered:
        raise FileNotFoundError(f"no canonical A2ASecBench run.jsonl found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_events: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    file_infos: list[dict[str, Any]] = []
    blank_lines = 0

    for run_name, path, attack_id in discovered:
        rows, file_invalid, file_blanks = _read_jsonl(path)
        invalid.extend(file_invalid)
        blank_lines += file_blanks
        raw_bytes = path.read_bytes() if path.is_file() else b""
        file_info = {
            "name": run_name,
            "attack_id": attack_id,
            "absolute_path": str(path),
            "path": _relative(path),
            "sha256": _sha256_bytes(raw_bytes),
            "line_count": len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0,
            "valid_rows": len(rows),
            "invalid_json_lines": len(file_invalid),
        }
        file_infos.append(file_info)
        for line_no, row in rows:
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            actual_attack = str(row.get("attack_id", attack_id))
            run_id = _run_id(row, actual_attack, run_name)
            episode_id = _episode_id(row, run_id)
            violation = observed_violation(details)
            row_events = [
                normalize_event(row, details, event, event_index, path, line_no, run_id, episode_id, violation)
                for event_index, event in enumerate(_events(details))
            ]
            all_events.extend(row_events)
            all_episodes.append(normalize_episode(row, details, path, line_no, run_id, episode_id, row_events, violation))

    manifest_match = _manifest_match(input_path if input_path.is_dir() else input_path.parent.parent, file_infos)
    # absolute_path is useful during matching but should not leak into the
    # portable coverage artifact.
    public_files = [{key: value for key, value in info.items() if key != "absolute_path"} for info in file_infos]
    coverage = _coverage(public_files, all_episodes, all_events, invalid, blank_lines, manifest_match)
    (output_dir / "events.jsonl").write_text("".join(_canonical(event) + "\n" for event in all_events), encoding="utf-8")
    (output_dir / "episodes.jsonl").write_text("".join(_canonical(episode) + "\n" for episode in all_episodes), encoding="utf-8")
    (output_dir / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "schema.json").write_text(json.dumps(_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_json = {
        **coverage,
        "output_dir": _relative(output_dir),
        "source_read_only": True,
        "event_file": _relative(output_dir / "events.jsonl"),
        "episode_file": _relative(output_dir / "episodes.jsonl"),
    }
    (output_dir / "ADAPTER_REPORT.json").write_text(json.dumps(report_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ADAPTER_REPORT.md").write_text(_report(coverage, output_dir), encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "A2ASecBench",
        "files": len(file_infos),
        "episodes": len(all_episodes),
        "events": len(all_events),
        "invalid_json_lines": len(invalid),
        "status": coverage["status"],
        "risk_label_events": coverage["explicit_risk_label_events"],
        "observed_violation_episodes": coverage["observed_violation_episodes"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = adapt(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if summary["invalid_json_lines"] == 0 else 1)


if __name__ == "__main__":
    main()
