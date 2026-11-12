"""Read-only adapter for HarnessAudit runtime JSONL traces.

HISTORICAL COVERAGE PROJECTION, NOT A LEAKAGE-SAFE MODEL INPUT CONTRACT.
Native access_decision is post-run; this legacy adapter exposes its text in
permission/features. Do not feed those rows/features to an online monitor.
Use project_harnessaudit_observations.py for source-aware observation/evaluation
separation. Historical outputs and their schemas are intentionally unchanged.

HarnessAudit's ``TraceWriter`` emits one JSON object per line.  This module
projects those objects (and a small set of compatible JSON containers) onto
the ``mas-adapter-v1`` Layer-2 event contract used by ``adapt_pre_exp1.py``.
It deliberately keeps *observations* separate from labels: an access
decision or a trace-end violation count is retained in ``outcome_flags`` but
is not silently converted to ``risk_label``.

The adapter never opens a network connection and never writes below the input
root.  It only reads source files and writes a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'code/external/HarnessAudit/multi_agent/traces'
DEFAULT_OUTPUT = ROOT / 'data/processed/mas_adapter/harnessaudit_traces'

SCHEMA_VERSION = "mas-adapter-v1"
RECORD_KIND = "runtime_trace"
FEATURES = (
    "untrusted_input",
    "permission_mismatch",
    "memory_poison",
    "tool_side_effect",
    "cross_agent_propagation",
    "anomaly_path",
    "risk_accumulation",
)

_MISSING = object()
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|password|passwd|credential|"
    r"private[_-]?key|bearer|canary|hidden[_-]?prompt|system[_-]?prompt|internal[_-]?notes?)",
    re.IGNORECASE,
)
_LABEL_KEYS = (
    "risk_label",
    "security_label",
    "attack_label",
    "is_risky",
    "unsafe",
    "malicious",
)
_EVENT_CONTAINER_KEYS = ("events", "records", "entries", "trace_events", "trace", "items")


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value is not _MISSING and value != "":
            return value
    return None


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("payload")
    return value if isinstance(value, dict) else {}


def _raw_event(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("raw_event")
    return value if isinstance(value, dict) else {}


def _field(record: dict[str, Any], *names: str) -> Any:
    """Find a field without treating unrelated fields as evidence."""
    for source in (record, _payload(record), _raw_event(record)):
        for name in names:
            if name in source and source[name] is not None and source[name] != "":
                return source[name]
    return None


def _redact(value: Any) -> Any:
    """Recursively redact secret-like keys while retaining observable shape."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[str(key)] = "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _text(value: Any, limit: int = 6000) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(_redact(value), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
    return text[:limit] if text else None


def _hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        encoded = repr(value).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _as_ids(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            item = _first(item.get("id"), item.get("event_id"), item.get("task_id"))
        if item is not None and str(item) not in result:
            result.append(str(item))
    return result


def _trace_id(record: dict[str, Any], fallback: str | None = None) -> str | None:
    value = _field(record, "trace_id", "run_id", "audit_id")
    return str(value) if value is not None else fallback


def _event_type(record: dict[str, Any]) -> str:
    value = _field(record, "event_type", "type", "record_type", "kind")
    return str(value) if value is not None else "unknown"


def _timestamp(record: dict[str, Any]) -> str | None:
    value = _field(record, "timestamp", "recorded_at", "created_at", "occurred_at", "time")
    return str(value) if value is not None else None


def _is_access_event(event_type: str) -> bool:
    return event_type.lower() in {"access_decision", "access_check", "permission_decision"}


def _actor_role(
    record: dict[str, Any],
    role_by_agent: dict[str, str],
    action_lookup: dict[tuple[str | None, str], dict[str, Any]],
    trace_id: str | None,
    event_type: str,
) -> tuple[str | None, str | None, str | None]:
    actor = _field(record, "agent_id", "actor_id", "sender_id", "source_agent", "caller_id")
    role = _field(record, "agent_role", "role", "actor_role", "sender_role", "source_role")
    basis = None
    if actor is not None:
        actor = str(actor)
    if role is not None:
        role = str(role)
    if actor and role is None:
        role = role_by_agent.get(actor)
        if role:
            basis = "trace_start.agents"

    # HarnessAudit AccessDecisionEvent refers to an action by sequence number,
    # rather than repeating the actor.  Joining that explicit reference is
    # deterministic evidence; it is not a guessed causal edge.
    if actor is None and _is_access_event(event_type):
        sequence = _field(record, "action_sequence_num", "sequence_num", "sequence")
        if sequence is not None:
            action = action_lookup.get((trace_id, str(sequence)))
            if action:
                actor = action.get("agent_id")
                role = action.get("role")
                basis = "action_sequence_num"
    return actor, role, basis


def _message_parts(record: dict[str, Any], event_type: str, actor: str | None) -> tuple[Any, str | None, str | None, str | None]:
    content = _field(record, "content", "message", "text", "body", "prompt", "response")
    message_id = _field(record, "message_id", "communication_id", "msg_id", "message_uuid")
    source = _field(record, "message_source", "source_agent", "sender_id", "agent_id")
    target = _field(record, "message_target", "target_agent", "recipient_id", "receiver_id")
    if event_type.lower() == "communication" and content is None:
        content = _field(record, "payload")
    if source is None and event_type.lower() == "communication":
        source = actor
    return (
        content,
        str(message_id) if message_id is not None else None,
        str(source) if source is not None else None,
        str(target) if target is not None else None,
    )


def _tool_parts(record: dict[str, Any], event_type: str) -> tuple[str | None, Any, Any]:
    tool = _field(record, "tool_name", "tool", "name", "function_name")
    args = _field(record, "tool_args", "tool_arguments", "arguments", "args", "input")
    result = _field(record, "tool_result", "result", "output", "response")
    # Do not classify an arbitrary message as a tool call.  A supplied tool
    # name remains authoritative even if the event discriminator is generic.
    if tool is None and "tool" not in event_type.lower() and "function" not in event_type.lower():
        args = None
        result = None
    return (str(tool) if tool is not None else None, args, result)


def _parent_ids(record: dict[str, Any]) -> list[str]:
    value = _field(
        record,
        "parent",
        "parent_id",
        "parent_event_id",
        "parent_task_id",
        "in_reply_to",
        "caused_by",
    )
    return _as_ids(value)


def _permission(record: dict[str, Any], event_type: str) -> dict[str, Any] | None:
    value = _field(record, "permission", "permission_context", "authorization", "access_policy")
    if isinstance(value, dict):
        evidence = _redact(value)
        # Preserve a common explicit alias in the shared permission shape.
        if "permission_allowed" not in evidence and "allowed" in evidence:
            parsed = _as_bool(evidence.get("allowed"))
            evidence["permission_allowed"] = parsed if parsed is not None else evidence.get("allowed")
        return evidence
    if value is not None:
        allowed = _as_bool(value)
        return {"permission_allowed": allowed if allowed is not None else _redact(value)}
    allowed = _field(record, "permission_allowed", "authorization_allowed")
    if allowed is not None:
        parsed = _as_bool(allowed)
        return {"permission_allowed": parsed if parsed is not None else _redact(allowed)}
    if _is_access_event(event_type):
        # These are explicit fields on AccessDecisionEvent.  Keep them as
        # permission evidence, while leaving risk_label independent.
        evidence: dict[str, Any] = {}
        for key in ("violated", "violation_code", "severity", "rule_description"):
            if key in record and record[key] is not None:
                evidence[key] = _redact(record[key])
        if evidence:
            return evidence
    return None


def _external_effect(record: dict[str, Any]) -> tuple[bool | None, str | None, str | None]:
    for source_name, source in (("record", record), ("payload", _payload(record)), ("raw_event", _raw_event(record))):
        for key in ("external_effect", "external_effect_observed", "side_effect_observed", "effect_observed"):
            if key in source:
                parsed = _as_bool(source[key])
                if parsed is not None:
                    scope = _field(record, "external_effect_scope", "effect_scope", "side_effect_scope")
                    return parsed, f"{source_name}.{key} (explicit)", str(scope) if scope is not None else None

    # Counters are accepted only when explicitly named as network/external
    # calls.  A tool name by itself does not prove that an effect occurred.
    for source_name, source in (("record", record), ("payload", _payload(record)), ("raw_event", _raw_event(record))):
        for key in ("remote_network_calls", "network_calls", "external_network_calls"):
            value = source.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return True, f"{source_name}.{key}>0 (explicit counter)", "remote_service"

    local = _field(record, "local_only", "local_mock", "sandbox_only")
    if _as_bool(local) is True:
        return False, "local_only/local_mock (explicit)", "local_sandbox"
    return None, None, None


def _label(record: dict[str, Any]) -> tuple[int | None, str | None]:
    sources: list[tuple[str, dict[str, Any]]] = [("record", record), ("payload", _payload(record)), ("raw_event", _raw_event(record))]
    for source_name, source in sources:
        for key in _LABEL_KEYS:
            if key not in source:
                continue
            value = source[key]
            parsed = _as_bool(value)
            if parsed is not None:
                return int(parsed), f"{source_name}.{key} (explicit)"
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"safe", "benign", "normal", "negative", "0"}:
                    return 0, f"{source_name}.{key} (explicit:{normalized})"
                if normalized in {"unsafe", "risky", "malicious", "attack", "positive", "risk", "1"}:
                    return 1, f"{source_name}.{key} (explicit:{normalized})"
    return None, None


def _outcome_flags(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "violated", "violation_code", "severity", "rule_description", "observed_violation",
        "n_actions", "n_tool_calls", "n_communications", "n_violations", "violations_by_code",
        "sar_tool", "sar_resource", "sar_flow", "sar_avg", "avs", "tcr",
        "protocol_safety_score", "operational_safety_score", "task_completion_score",
        "error", "success", "status", "decision", "task_success", "terminal_state",
        "failure_code", "perturbation",
    )
    result: dict[str, Any] = {}
    sources = (record, _payload(record), _raw_event(record))
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                result[key] = _redact(source[key])
    return result


def _features(
    record: dict[str, Any],
    event_type: str,
    message_text: str | None,
    tool: str | None,
    source: str | None,
    target: str | None,
    effect: bool | None,
    permission: dict[str, Any] | None,
) -> tuple[dict[str, int], list[str]]:
    blob = " ".join(
        str(value or "")
        for value in (
            event_type,
            message_text,
            tool,
            _field(record, "violation_code", "severity", "rule_description", "error", "status"),
            _field(record, "scenario", "condition", "perturbation"),
        )
    ).lower()
    reasons: list[str] = []
    untrusted = any(word in blob for word in ("untrusted", "injected", "malicious", "poisoned", "prompt injection", "dangerous"))
    if untrusted:
        reasons.append("lexical.untrusted_input")
    denied = permission is not None and _as_bool(permission.get("permission_allowed")) is False
    mismatch = denied or any(word in blob for word in ("unauthorized", "forbidden", "permission mismatch", "denied", "bypass"))
    if mismatch:
        reasons.append("permission/authorization")
    memory = any(word in blob for word in ("memory poison", "memory_poison", "stale memory", "persisted state", "shared memory"))
    if memory:
        reasons.append("lexical.memory_poison")
    tool_intent = any(word in (tool or "").lower() for word in ("publish", "send", "delete", "payment", "export", "execute", "write", "sink"))
    side_effect = effect is True or tool_intent
    if side_effect:
        reasons.append("tool/external_effect" if effect is True else "tool/side_effect_intent")
    cross_agent = bool(source and target and source != target)
    if cross_agent or any(word in blob for word in ("delegate", "delegation", "handoff", "propagate", "mailbox")):
        cross_agent = True
        reasons.append("message/agent_boundary")
    anomaly = any(word in blob for word in ("retry", "invalid", "rejected", "failed", "exception", "timeout", "loop", "deviate"))
    if anomaly:
        reasons.append("failure/retry")
    accumulation = any(word in blob for word in ("multiple", "combined", "cumulative", "chain", "fragment", "accumulation"))
    if accumulation:
        reasons.append("lexical.accumulation")
    return (
        {
            "untrusted_input": int(untrusted),
            "permission_mismatch": int(mismatch),
            "memory_poison": int(memory),
            "tool_side_effect": int(side_effect),
            "cross_agent_propagation": int(cross_agent),
            "anomaly_path": int(anomaly),
            "risk_accumulation": int(accumulation),
        },
        reasons,
    )


def _source_root(path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent
    return input_path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _trace_format(records: list[dict[str, Any]]) -> str:
    types = {_event_type(record) for record in records}
    expected = {"trace_start", "tool_call", "communication", "access_decision", "trace_end"}
    return "HarnessAudit TraceWriter JSONL" if types & expected else "runtime JSONL (compatible projection)"


def _container_records(value: Any, context: str = "root") -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yield event objects from direct records or a JSON event container."""
    if isinstance(value, dict):
        if any(key in value for key in ("type", "event_type", "record_type", "kind")):
            yield value, None
            return
        for key in _EVENT_CONTAINER_KEYS:
            items = value.get(key)
            if isinstance(items, list):
                base = {k: v for k, v in value.items() if k not in _EVENT_CONTAINER_KEYS}
                for index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    merged = dict(base)
                    merged.update(item)
                    yield merged, f"{context}.{key}[{index}]"
                return
        yield value, None
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                yield from _container_records(item, f"{context}[{index}]")


def _read_file(path: Path) -> tuple[list[tuple[dict[str, Any], int, str | None]], int]:
    """Read one JSON/JSONL file; return records and malformed-item count."""
    rows: list[tuple[dict[str, Any], int, str | None]] = []
    errors = 0
    try:
        if path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    errors += 1
                    continue
                found = list(_container_records(value, f"line[{line_no}]"))
                if not found:
                    errors += 1
                rows.extend((record, line_no, embedded) for record, embedded in found)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.extend((record, 1, embedded) for record, embedded in _container_records(value))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        errors += 1
    return rows, errors


def discover_trace_files(input_path: Path, pattern: str = "*.jsonl") -> list[Path]:
    """Resolve a file or directory without scanning generated output."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []
    candidates = sorted({*input_path.rglob(pattern)})
    # A caller may use ``*.json`` for exported containers.  The default is
    # intentionally JSONL to avoid treating task specs as runtime traces.
    return [path for path in candidates if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}]


def _meta_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _redact(record[key])
        for key in ("task_id", "domain", "framework", "model", "run_kind", "execution_mode", "perturbation")
        if key in record and record[key] is not None
    }


def normalize_record(
    record: dict[str, Any],
    path: Path,
    line_no: int,
    input_root: Path,
    *,
    trace_meta: dict[str, Any] | None = None,
    role_by_agent: dict[str, str] | None = None,
    action_lookup: dict[tuple[str | None, str], dict[str, Any]] | None = None,
    embedded_path: str | None = None,
) -> dict[str, Any]:
    """Normalize one raw HarnessAudit event onto the shared Layer-2 schema."""
    trace_meta = trace_meta or {}
    role_by_agent = role_by_agent or {}
    action_lookup = action_lookup or {}
    event_type = _event_type(record)
    trace_id = _trace_id(record, trace_meta.get("trace_id"))
    # HarnessAudit's canonical trace events expose ``trace_id`` (the run
    # identity), but generally do not expose an episode identity.  Do not
    # silently relabel a run as an episode: preserve only an explicit
    # ``episode_id``/``episode`` field and leave the shared field null.
    episode_value = _field(record, "episode_id", "episode")
    episode_id = str(episode_value) if episode_value is not None else None
    actor, role, actor_basis = _actor_role(record, role_by_agent, action_lookup, trace_id, event_type)
    message, message_id, source, target = _message_parts(record, event_type, actor)
    tool, tool_args, tool_result = _tool_parts(record, event_type)
    permission = _permission(record, event_type)
    effect, effect_basis, effect_scope = _external_effect(record)
    label, label_source = _label(record)
    message_text = _text(message)
    features, feature_reasons = _features(record, event_type, message_text, tool, source, target, effect, permission)
    event_id_value = _field(record, "event_id", "id", "event_uuid")
    source_hash = _hash(record)
    event_id = str(event_id_value) if event_id_value is not None else f"adapt-{source_hash[:20]}-{line_no}"
    source_rel = _relative(path, input_root)
    context = {
        "task_id": trace_meta.get("task_id", _field(record, "task_id")),
        "domain": trace_meta.get("domain", _field(record, "domain")),
        "framework": trace_meta.get("framework", _field(record, "framework")),
        "model": trace_meta.get("model", _field(record, "model", "model_id")),
        "sequence_num": _field(record, "sequence_num", "action_sequence_num", "sequence"),
        "actor_basis": actor_basis,
    }
    # Keep absent values as null; this makes missing evidence visible to
    # downstream coverage checks instead of silently manufacturing defaults.
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "run_id": trace_id,
        "episode_id": episode_id,
        "record_kind": RECORD_KIND,
        "event_type": event_type,
        "stage": _field(record, "stage", "phase"),
        "agent_id": actor,
        "role": role,
        "message": _redact(message) if message is not None else None,
        "message_text": message_text,
        "message_id": message_id,
        "message_source": source,
        "message_target": target,
        "tool": tool,
        "tool_arguments": _redact(tool_args) if tool_args is not None else None,
        "tool_result": _redact(tool_result) if tool_result is not None else None,
        "timestamp": _timestamp(record),
        "parent": _parent_ids(record),
        "chain_parent": _first(_field(record, "previous_event_sha256", "previous_event_id", "prev_event_id")),
        "permission": permission,
        "external_effect": effect,
        "external_effect_basis": effect_basis,
        "external_effect_scope": effect_scope,
        "risk_label": label,
        "risk_label_source": label_source,
        "outcome_flags": _outcome_flags(record),
        "features": features,
        "feature_reasons": feature_reasons,
        "source": {
            "path": source_rel,
            "line": line_no,
            "embedded_path": embedded_path,
            "artifact_type": RECORD_KIND,
            "authenticity": "observed_harnessaudit_runtime_trace",
            "trace_format": trace_meta.get("trace_format"),
            "framework": trace_meta.get("framework", _field(record, "framework")),
            "model": trace_meta.get("model", _field(record, "model", "model_id")),
        },
        "context": context,
        "eligibility": {
            "structural_graph": bool(actor or role or source or target or tool),
            "temporal_graph": bool(_timestamp(record)),
            "bn_feature_row": True,
            "risk_evaluation": label is not None,
            "replay_evidence": True,
        },
        "source_record_hash": source_hash,
    }


def _schema() -> dict[str, Any]:
    fields: dict[str, Any] = {
        "schema_version": {"const": SCHEMA_VERSION},
        "event_id": {"type": "string"},
        "run_id": {"type": ["string", "null"]},
        "episode_id": {"type": ["string", "null"]},
        "record_kind": {"const": RECORD_KIND},
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
        "chain_parent": {"type": ["string", "null"]},
        "permission": {"type": ["object", "null"]},
        "external_effect": {"type": ["boolean", "null"]},
        "external_effect_basis": {"type": ["string", "null"]},
        "external_effect_scope": {"type": ["string", "null"]},
        "risk_label": {"type": ["integer", "null"], "enum": [0, 1, None]},
        "risk_label_source": {"type": ["string", "null"]},
        "source": {"type": "object"},
        "context": {"type": "object"},
        "eligibility": {"type": "object"},
        "features": {"type": "object", "required": list(FEATURES)},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "mas-adapter-harnessaudit-runtime-event.schema.json",
        "title": "HarnessAudit runtime event projected onto mas-adapter-v1",
        "type": "object",
        "required": ["schema_version", "event_id", "record_kind", "event_type", "agent_id", "role", "message", "tool", "timestamp", "parent", "permission", "external_effect", "risk_label", "features", "source"],
        "properties": fields,
        "additionalProperties": True,
    }


def _coverage(events: list[dict[str, Any]], files: list[Path], invalid: int, input_root: Path) -> dict[str, Any]:
    total = len(events)
    fields = ("agent_id", "role", "message", "tool", "timestamp", "parent", "permission", "external_effect", "risk_label")
    field_counts = {
        field: sum(bool(event.get(field)) if field == "parent" else event.get(field) is not None for event in events)
        for field in fields
    }
    types = Counter(str(event.get("event_type")) for event in events)
    traces = sorted({str(event["run_id"]) for event in events if event.get("run_id") is not None})
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(input_root),
        "files_read": len(files),
        "files": [_relative(path, input_root) for path in files],
        "invalid_json_items": invalid,
        "event_count": total,
        "trace_count": len(traces),
        "trace_ids": traces,
        "event_types": dict(types),
        "field_coverage": {
            field: {"count": count, "fraction": count / total if total else 0.0}
            for field, count in field_counts.items()
        },
        "explicit_risk_label_events": sum(event.get("risk_label") is not None for event in events),
        "timestamp_events": sum(event.get("timestamp") is not None for event in events),
        "structural_events": sum(bool(event.get("agent_id") or event.get("role") or event.get("tool") or event.get("message_source") or event.get("message_target")) for event in events),
        "status": (
            "no_runtime_traces"
            if total == 0
            else (
                "coverage_only_no_risk_labels"
                if not any(event.get("risk_label") is not None for event in events)
                else "labels_present_metrics_not_computed"
            )
        ),
        "label_policy": "Only explicit risk/security/attack label aliases are mapped; violated and trace-end metrics stay in outcome_flags.",
    }


def _report(coverage: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# HarnessAudit runtime trace adapter",
        "",
        "This is a read-only Layer-2 projection of HarnessAudit runtime JSONL. Source files are not rewritten.",
        "",
        f"- Input root: `{coverage['input_root']}`",
        f"- Files read: **{coverage['files_read']}**",
        f"- Runtime events: **{coverage['event_count']}**",
        f"- Trace IDs: **{coverage['trace_count']}**",
        f"- Invalid JSON items: **{coverage['invalid_json_items']}**",
        f"- Status: `{coverage['status']}`",
        f"- Output: `{output_dir}`",
        "",
        "## Shared field contract",
        "",
        "The event fields mirror `adapt_pre_exp1.py`: `agent_id`, `role`, `message`/`message_text`, `tool`/`tool_arguments`/`tool_result`, `timestamp`, `parent`, `permission`, `external_effect`, `risk_label`, and the seven BN feature names. HarnessAudit `trace_id` is kept as `run_id`; because its runtime schema normally has no episode field, `episode_id` remains `null` unless explicitly supplied. Other missing source values remain `null` or an empty parent list.",
        "",
        "## Label and effect policy",
        "",
        "Only explicit binary risk/security/attack aliases are mapped to `risk_label`. HarnessAudit `violated`, violation codes, SAR/TCR/AVS, and trace-end counters are retained in `outcome_flags`; they are not silently promoted to a risk label. `external_effect` is tri-state and requires an explicit effect flag or explicitly named network counter. Tool names alone do not prove that an effect occurred.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\adapt_harnessaudit_traces.py --input <trace.jsonl> --output <out-dir>",
        "```",
        "",
        "Outputs: `events.jsonl`, `coverage.json`, a machine-readable schema, and `ADAPTER_REPORT.md`.",
    ]
    return "\n".join(lines) + "\n"


def adapt(input_path: Path, output_dir: Path, pattern: str = "*.jsonl", limit_files: int | None = None, limit_events: int | None = None) -> dict[str, Any]:
    """Adapt all selected files and write a new output directory."""
    input_resolved = input_path.resolve()
    output_resolved = output_dir.resolve()
    if input_path.is_dir() and (output_resolved == input_resolved or output_resolved.is_relative_to(input_resolved)):
        raise ValueError("output directory must be outside the input trace directory (source is read-only)")
    if input_path.is_file():
        if output_resolved == input_resolved:
            raise ValueError("output directory cannot be the input trace file")
        # A single input file may itself be named ``events.jsonl`` (or one of
        # the other adapter output names).  Writing into its parent could then
        # overwrite the source even though the output argument is a directory.
        if output_resolved == input_resolved.parent and input_resolved.name.lower() in {
            "events.jsonl", "coverage.json", "schema.json", "harnessaudit_trace_schema.json",
            "summary.json", "adapter_report.md",
        }:
            raise ValueError("output directory would overwrite the input trace file")
    files = discover_trace_files(input_path, pattern)
    if limit_files is not None:
        files = files[: max(0, limit_files)]
    input_root = _source_root(input_path, input_path)
    all_events: list[dict[str, Any]] = []
    invalid = 0

    for path in files:
        rows, parse_errors = _read_file(path)
        invalid += parse_errors
        if not rows:
            continue

        # Collect explicit trace metadata and role declarations before
        # normalizing action rows.  A file may contain more than one trace;
        # records with their own trace_id always win over the file fallback.
        starts = [record for record, _, _ in rows if _event_type(record).lower() == "trace_start"]
        default_trace = _trace_id(starts[0]) if len({_trace_id(r) for r in starts if _trace_id(r)} or set()) == 1 else None
        meta_by_trace: dict[str | None, dict[str, Any]] = defaultdict(dict)
        roles_by_trace: dict[str | None, dict[str, str]] = defaultdict(dict)
        for record in starts:
            rid = _trace_id(record, default_trace)
            meta = meta_by_trace[rid]
            meta.update(_meta_from_record(record))
            meta["trace_id"] = rid
            meta["trace_format"] = _trace_format([r for r, _, _ in rows])
            agents = record.get("agents")
            if isinstance(agents, list):
                for agent in agents:
                    if isinstance(agent, dict) and agent.get("agent_id") is not None:
                        aid = str(agent["agent_id"])
                        role = _first(agent.get("role"), agent.get("agent_role"))
                        if role is not None:
                            roles_by_trace[rid][aid] = str(role)
        if default_trace is not None and default_trace not in meta_by_trace:
            meta_by_trace[default_trace] = {"trace_id": default_trace, "trace_format": _trace_format([r for r, _, _ in rows])}

        # Build explicit sequence joins for AccessDecisionEvent.
        action_lookup: dict[tuple[str | None, str], dict[str, Any]] = {}
        provisional: list[tuple[dict[str, Any], int, str | None, str | None, dict[str, Any], dict[str, str]]] = []
        for record, line_no, embedded in rows:
            rid = _trace_id(record, default_trace)
            meta = meta_by_trace.get(rid, {"trace_id": rid, "trace_format": _trace_format([r for r, _, _ in rows])})
            roles = roles_by_trace.get(rid, {})
            provisional.append((record, line_no, embedded, rid, meta, roles))
            etype = _event_type(record).lower()
            if etype in {"tool_call", "tool", "function_call"}:
                sequence = _field(record, "sequence_num", "sequence", "action_sequence_num")
                if sequence is not None:
                    actor = _field(record, "agent_id", "actor_id", "sender_id")
                    role = _field(record, "agent_role", "role", "actor_role")
                    if actor is not None:
                        actor = str(actor)
                        role = str(role) if role is not None else roles.get(actor)
                    action_lookup[(rid, str(sequence))] = {"agent_id": actor, "role": role}

        for record, line_no, embedded, rid, meta, roles in provisional:
            if limit_events is not None and len(all_events) >= max(0, limit_events):
                break
            event = normalize_record(
                record,
                path,
                line_no,
                input_root,
                trace_meta=meta,
                role_by_agent=roles,
                action_lookup=action_lookup,
                embedded_path=embedded,
            )
            # Preserve the explicit action-sequence relationship on access
            # decisions as a parent edge when a matching action is present.
            if _is_access_event(event["event_type"]) and not event["parent"]:
                sequence = _field(record, "action_sequence_num", "sequence_num", "sequence")
                matched = action_lookup.get((rid, str(sequence))) if sequence is not None else None
                if matched:
                    # Resolve the event ID by scanning already emitted events
                    # in this trace/sequence, without inventing an ID.
                    for prior in reversed(all_events):
                        if prior.get("run_id") == rid and prior.get("event_type", "").lower() in {"tool_call", "tool", "function_call"} and prior.get("source", {}).get("line") <= line_no:
                            prior_seq = prior.get("context", {}).get("sequence_num")
                            if prior_seq is not None and str(prior_seq) == str(sequence):
                                event["parent"] = [prior["event_id"]]
                                break
                    # If the source did not expose a sequence on the prior
                    # normalized row, leave parent empty rather than guessing.
            all_events.append(event)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    coverage = _coverage(all_events, files, invalid, input_root)
    (output_dir / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path = output_dir / "schema.json"
    if schema_path.exists():
        # Re-running this adapter should refresh its own schema.  If callers
        # intentionally share the parent directory with another adapter,
        # preserve that adapter's schema under its original name.
        ours = False
        try:
            existing = json.loads(schema_path.read_text(encoding="utf-8"))
            ours = existing.get("$id") == "mas-adapter-harnessaudit-runtime-event.schema.json"
        except (OSError, UnicodeError, ValueError, TypeError):
            pass
        if not ours:
            schema_path = output_dir / "harnessaudit_trace_schema.json"
    schema_path.write_text(json.dumps(_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ADAPTER_REPORT.md").write_text(_report(coverage, output_dir), encoding="utf-8")
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "files": len(files),
        "events": len(all_events),
        "invalid_json_items": invalid,
        "trace_count": coverage["trace_count"],
        "risk_label_events": coverage["explicit_risk_label_events"],
        "schema": str(schema_path),
        "status": coverage["status"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


# Descriptive aliases make the small adapter convenient to import from an
# offline analysis notebook while keeping the CLI-oriented names above.
normalize_trace_event = normalize_record
adapt_trace = adapt


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only HarnessAudit runtime trace adapter")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="trace JSONL file or directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new output directory")
    parser.add_argument("--pattern", default="*.jsonl", help="directory glob (default: *.jsonl)")
    parser.add_argument("--limit-files", type=int, default=None, help="read at most N files")
    parser.add_argument("--limit-events", type=int, default=None, help="write at most N normalized events")
    args = parser.parse_args()
    print(json.dumps(adapt(args.input, args.output, args.pattern, args.limit_files, args.limit_events), ensure_ascii=False))


if __name__ == "__main__":
    main()
