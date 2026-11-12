"""Read-only adapter for the pre_exp1 evidence tree.

The pre_exp1 tree intentionally contains several generations of schemas.  This
script does not edit or re-hash any source artifact.  It projects event-like
records into a small, append-only JSONL contract and keeps an independent
artifact inventory for manifests, outcomes, replays and reports.

The adapter is deliberately conservative about labels: only an explicit
``risk_label`` (or a small allow-list of equivalent binary labels) becomes a
label.  ``risk_level``, ``risk_score``, ``condition`` and ``leak_detected`` are
kept as outcome fields and never silently converted into a security target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'archive/early_pre_exp1/data/pre_exp1'
DEFAULT_OUTPUT = ROOT / 'data/processed/mas_adapter'

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

SENSITIVE_KEY_RE = re.compile(
    r"(?:secret|canary|credential|api[_-]?key|authorization|access[_-]?token|"
    r"private[_-]?key|internal[_-]?notes?)",
    re.I,
)
# Run ids occur both in filenames and in parent directories (Q1/Q2 runs).
# Keep this deliberately narrow so a generic directory such as ``ledger`` is
# never promoted to a run id.
RUN_ID_RE = re.compile(
    r"(?:P[0-9][A-Z0-9_-]+|Q2D-[A-Z0-9_-]+|REPAIR-[A-Z0-9_-]+)",
    re.I,
)

MESSAGE_EVENT_TYPES = {
    "task_received",
    "message_sent",
    "message_delivered",
    "message_gate",
    "model_call_requested",
    "model_output",
    "model_call_failed",
}
TOOL_EVENT_TYPES = {
    "tool_call",
    "tool_read_internal_record",
    "tool_publish_local_sink",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def redact(value: Any) -> Any:
    """Remove obvious secrets while retaining enough structure for replay."""
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if SENSITIVE_KEY_RE.search(str(k)) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return [redact(v) for v in value]
    return value


def as_text(value: Any, limit: int = 6000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = canonical_json(redact(value))
    return text[:limit]


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def nested(record: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur: Any = record
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return None


def path_run_id(path: Path) -> str | None:
    """Best-effort run id from a filename when a record has none."""
    stem = path.name
    for prefix in (
        "events_",
        "outcomes_",
        "manifest_",
        "replay_",
        "transcripts_",
        "metadata_",
        "validation_",
        "public_sink_",
        "unstarted_",
    ):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    stem = re.sub(r"\.(jsonl?|md)$", "", stem, flags=re.I)
    # Directory names are often the authoritative ID for q1/q2 artifacts.
    for part in reversed(path.parts):
        if RUN_ID_RE.fullmatch(part):
            return part
    # A filename-derived id is useful only if it looks like a real run id.
    return stem if stem and RUN_ID_RE.fullmatch(stem) else None


def run_id_of(record: dict[str, Any], path: Path) -> str | None:
    value = first_nonempty(
        record.get("run_id"),
        nested(record, ("payload", "run_id")),
        nested(record, ("metadata", "run_id")),
    )
    return str(value) if value is not None else path_run_id(path)


def episode_id_of(record: dict[str, Any], path: Path) -> str | None:
    value = first_nonempty(
        record.get("episode_id"),
        nested(record, ("payload", "episode_id")),
        nested(record, ("task_card", "episode_id")),
        nested(record, ("response", "episode_id")),
        nested(record, ("payload", "replay_action", "message", "episode_id")),
    )
    if value is not None:
        return str(value)
    # Qualification rows often expose a task/request id instead of an
    # episode.  Prefer those stable units over a shared case name; only then
    # fall back to the expected fixture episode/case.
    value = first_nonempty(
        record.get("task_id"),
        record.get("request_id"),
        record.get("expected_episode_id"),
        record.get("case_id"),
    )
    return str(value) if value is not None else None


def timestamp_of(record: dict[str, Any]) -> str | None:
    value = first_nonempty(
        record.get("timestamp"),
        record.get("recorded_at"),
        record.get("created_at"),
        record.get("time"),
        nested(record, ("payload", "timestamp")),
        nested(record, ("payload", "recorded_at")),
        nested(record, ("metadata", "timestamp")),
        nested(record, ("metadata", "recorded_at")),
    )
    return str(value) if value is not None else None


def actor_and_role(record: dict[str, Any]) -> tuple[str | None, str | None]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    action = details.get("replay_action") if isinstance(details.get("replay_action"), dict) else {}
    action_message = action.get("message") if isinstance(action.get("message"), dict) else {}
    task_card = record.get("task_card") if isinstance(record.get("task_card"), dict) else {}
    actor = first_nonempty(
        record.get("agent_id"),
        record.get("role_id"),
        record.get("source_agent"),
        payload.get("source_agent"),
        payload.get("agent_id"),
        action.get("agent_id"),
        action_message.get("recipient_id"),
        record.get("role"),
        record.get("position"),
        task_card.get("agent_role"),
        task_card.get("role"),
    )
    role = first_nonempty(
        record.get("role_id"),
        record.get("role"),
        task_card.get("agent_role"),
        task_card.get("role"),
        payload.get("role"),
        record.get("position"),
        actor,
    )
    return (str(actor) if actor is not None else None, str(role) if role is not None else None)


def parent_ids_of(record: dict[str, Any]) -> list[str]:
    candidates = [
        record.get("parent"),
        record.get("parent_id"),
        record.get("parent_message_ids"),
        nested(record, ("payload", "parent_message_ids")),
        nested(record, ("payload", "message", "parent_message_ids")),
        nested(record, ("message", "parent_message_ids")),
        nested(record, ("details", "replay_action", "message", "parent_message_ids")),
        nested(record, ("details", "message", "parent_message_ids")),
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        items = candidate if isinstance(candidate, list) else [candidate]
        for item in items:
            if item is not None and str(item) not in result:
                result.append(str(item))
    return result


def message_parts(record: dict[str, Any], event_type: str | None) -> tuple[Any, str | None, str | None, str | None]:
    """Return content, message id, source and target without exposing hidden reasoning."""
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    action = details.get("replay_action") if isinstance(details.get("replay_action"), dict) else {}

    msg = payload if event_type in {"task_received", "message_sent"} else None
    if event_type == "message_delivered":
        msg = first_nonempty(action.get("message"), details.get("canonical_payload"))
    elif event_type == "message_gate":
        msg = first_nonempty(details.get("output_content"), details.get("input_content"), details.get("canonical_payload"))
    elif event_type == "model_call_requested":
        msg = first_nonempty(payload.get("visible_messages"), payload.get("request_payload"))
    elif event_type in {"model_output", "model_call_failed"}:
        msg = first_nonempty(payload.get("parsed_response"), payload.get("raw_output"), payload.get("error"))
    if msg is None:
        # q1 transcripts and readiness fixtures use a flat schema.
        msg = first_nonempty(
            record.get("raw_output"),
            record.get("raw_response"),
            record.get("response"),
            record.get("request_payload"),
            record.get("task_card"),
            record.get("content"),
        )
    if msg is None and event_type not in {"episode_started", "episode_finished", "episode_completed"}:
        # Preserve a compact structural payload for ledger/qualification rows.
        msg = first_nonempty(record.get("payload"), record)

    message_id = first_nonempty(
        record.get("message_id"),
        payload.get("message_id"),
        nested(record, ("payload", "replay_action", "message", "message_id")),
        nested(record, ("details", "replay_action", "message", "message_id")),
        record.get("request_id"),
    )
    source = first_nonempty(
        record.get("source_agent"),
        payload.get("source_agent"),
        nested(record, ("payload", "replay_action", "message", "sender_id")),
        nested(record, ("details", "replay_action", "message", "sender_id")),
    )
    target = first_nonempty(
        record.get("target_agent"),
        payload.get("target_agent"),
        nested(record, ("payload", "replay_action", "message", "recipient_id")),
        nested(record, ("details", "replay_action", "message", "recipient_id")),
    )
    return msg, (str(message_id) if message_id is not None else None), (str(source) if source is not None else None), (str(target) if target is not None else None)


def tool_parts(record: dict[str, Any], event_type: str | None) -> tuple[str | None, Any, Any]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    action = details.get("replay_action") if isinstance(details.get("replay_action"), dict) else {}
    name = first_nonempty(
        payload.get("tool_name"),
        record.get("tool_name"),
        details.get("tool_name"),
        action.get("tool_name"),
    )
    if name is None and event_type == "tool_read_internal_record":
        name = "get_internal_vendor_record"
    elif name is None and event_type == "tool_publish_local_sink":
        name = "publish_public_report"
    arguments = first_nonempty(payload.get("arguments"), details.get("request"), action.get("request"))
    result = first_nonempty(payload.get("result"), details.get("result"), action.get("result"))
    return (str(name) if name is not None else None, arguments, result)


def permission_of(record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    values: dict[str, Any] = {}
    for source in (record, payload, details):
        for key in (
            "permission_allowed",
            "allowed_tools",
            "permission",
            "permission_scope",
            "credential_scope",
            "scope",
            "gated",
            "allowed",
        ):
            if key in source and source[key] is not None:
                values[key] = redact(source[key])
    if not values:
        return None
    return values


def external_effect_of(record: dict[str, Any], tool: str | None) -> tuple[bool | None, str | None]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    event_type = str(record.get("event_type") or record.get("record_type") or record.get("kind") or "").lower()
    if "external_effect" in record:
        return bool(record["external_effect"]), "explicit.external_effect"
    if "external_effect" in payload:
        return bool(payload["external_effect"]), "payload.external_effect"
    if payload.get("permission_allowed") is False:
        return False, "permission_denied"
    if event_type in {"tool_publish_local_sink", "tool_publish", "public_sink_publish"}:
        return True, "event_type"
    name = (tool or "").lower()
    if any(word in name for word in ("publish", "send", "delete", "payment", "export", "execute", "write", "sink")):
        return True, "tool_name"
    if isinstance(payload.get("result"), dict):
        status = str(payload["result"].get("status", "")).lower()
        if status in {"denied", "rejected", "blocked", "failed"}:
            return False, "tool_result_status"
    if event_type in {"episode_started", "episode_finished", "episode_completed", "task_received", "model_output"}:
        return None, None
    return None, None


def external_effect_scope(record: dict[str, Any], effect: bool | None) -> str | None:
    """Describe the boundary of an observed effect without upgrading it.

    ``tool_publish_local_sink`` is an observable state mutation, but it is not
    an internet/provider side effect.  Keeping the scope beside the tri-state
    value prevents downstream users from treating every publication event as
    a real external action.
    """
    if effect is None:
        return None
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    event_type = str(record.get("event_type") or record.get("record_type") or record.get("kind") or "").lower()
    if (
        record.get("local_only") is True
        or payload.get("local_only") is True
        or payload.get("local_mock") is True
        or details.get("local_only") is True
        or details.get("local_mock") is True
    ):
        return "local_sandbox"
    if "local_sink" in event_type or "loopback" in event_type:
        return "local_sandbox"
    # Explicit counters are stronger than lexical guesses.
    for source in (record, payload, details):
        for key in ("remote_network_calls", "network_calls", "external_network_calls"):
            value = source.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return "remote_service"
    if event_type in {"tool_publish_local_sink", "public_sink_publish"}:
        return "local_sandbox"
    return "external_or_unspecified"


def explicit_label(record: dict[str, Any]) -> tuple[int | None, str | None]:
    """Accept only explicit binary labels; never infer from outcome fields."""
    allowed_keys = (
        "risk_label",
        "security_label",
        "attack_label",
        "is_risky",
        "unsafe",
        "malicious",
    )
    sources = [record]
    for key in ("payload", "outcome", "response", "result"):
        if isinstance(record.get(key), dict):
            sources.append(record[key])
    for source in sources:
        for key in allowed_keys:
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, bool):
                return int(value), f"{key} (explicit)"
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
                return int(value), f"{key} (explicit)"
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"safe", "benign", "normal", "negative", "0"}:
                    return 0, f"{key} (explicit:{normalized})"
                if normalized in {"unsafe", "risky", "malicious", "attack", "positive", "risk", "1"}:
                    return 1, f"{key} (explicit:{normalized})"
    return None, None


def outcome_flags(record: dict[str, Any]) -> dict[str, Any]:
    """Keep outcome indicators separate from ``risk_label``."""
    result: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = [record]
    for key in ("payload", "outcome", "response", "result"):
        if isinstance(record.get(key), dict):
            candidates.append(record[key])
    keys = (
        "condition",
        "treatment",
        "applied_treatment",
        "requested_treatment",
        "intervention_applied",
        "leak_detected",
        "canary_leak_detected",
        "unauthorized_tool_attempt",
        "forbidden_internal_field_present",
        "risk_level",
        "risk_score",
        "public_risk_score",
        "failure_code",
        "terminal_state",
        "task_success",
        "report_published",
    )
    for source in candidates:
        for key in keys:
            if key in source and source[key] is not None:
                result[key] = redact(source[key])
    return result


def infer_features(
    record: dict[str, Any],
    message_text: str,
    tool: str | None,
    source: str | None,
    target: str | None,
    external_effect: bool | None,
    permission: dict[str, Any] | None,
) -> tuple[dict[str, int], list[str]]:
    event_name = str(first_nonempty(record.get("event_type"), record.get("record_type"), record.get("kind"), "")).lower()
    blob = " ".join(
        str(x or "")
        for x in (
            event_name,
            message_text,
            tool,
            record.get("treatment"),
            record.get("scenario"),
            record.get("condition"),
            record.get("failure_code"),
            record.get("terminal_state"),
        )
    ).lower()
    reasons: list[str] = []

    untrusted = any(word in blob for word in ("untrusted", "injected", "malicious", "poisoned", "dangerous_original", "prompt injection"))
    if untrusted:
        reasons.append("lexical.untrusted_input")

    permission_denied = bool(permission and permission.get("permission_allowed") is False)
    permission_mismatch = permission_denied or any(word in blob for word in ("unauthorized", "permission mismatch", "wrong recipient", "bypass authorization"))
    if permission_mismatch:
        reasons.append("permission/authorization")

    memory_poison = any(word in blob for word in ("memory poison", "memory_poison", "stale memory", "persisted state", "shared memory"))
    if memory_poison:
        reasons.append("lexical.memory_poison")

    side_effect = external_effect is True or any(word in (tool or "").lower() for word in ("publish", "send", "delete", "payment", "export", "execute", "write", "sink"))
    if side_effect:
        reasons.append("tool/external_effect")

    cross_agent = bool(source and target and source != target)
    if cross_agent or any(word in blob for word in ("delegate", "delegation", "handoff", "shared state", "propagate")):
        cross_agent = True
        reasons.append("message/agent_boundary")

    anomaly = any(word in blob for word in ("retry", "invalid", "rejected", "failed", "exception", "timeout", "loop", "bypass", "deviate"))
    if anomaly:
        reasons.append("failure/retry")

    accumulation = any(word in blob for word in ("multiple", "combined", "cumulative", "chain", "fragment", "accumulation"))
    if accumulation:
        reasons.append("lexical.accumulation")

    return (
        {
            "untrusted_input": int(untrusted),
            "permission_mismatch": int(permission_mismatch),
            "memory_poison": int(memory_poison),
            "tool_side_effect": int(side_effect),
            "cross_agent_propagation": int(cross_agent),
            "anomaly_path": int(anomaly),
            "risk_accumulation": int(accumulation),
        },
        reasons,
    )


def classify_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if path.suffix.lower() == ".md":
        return "report"
    if "manifest" in name or "manifests" in parts:
        return "manifest"
    if "outcome" in name or "outcomes" in parts or "public_sink" in parts:
        return "outcome"
    if "validation" in name or "validation" in parts:
        return "validation"
    if "replay" in name or "replay" in parts or "interim" in parts:
        return "replay"
    if "transcript" in name or "transcripts" in parts:
        return "transcript"
    if "ledger" in name:
        return "ledger"
    if "raw" in parts or "events" in parts or name.startswith("events_"):
        return "raw_event"
    if "metadata" in parts:
        return "metadata"
    return "other"


def source_authenticity(
    record: dict[str, Any],
    path: Path,
    artifact_type: str,
    run_meta: dict[str, Any] | None = None,
) -> str:
    if artifact_type in {"manifest", "outcome", "validation", "replay", "report"}:
        return "derived_or_summary_artifact"
    run_meta = run_meta or {}
    blob = " ".join(
        str(first_nonempty(record.get(k), nested(record, ("payload", k)), ""))
        for k in ("execution_mode", "provider_kind", "provider_id", "model_id", "run_kind", "kind")
    ).lower()
    meta_blob = " ".join(str(run_meta.get(k, "")) for k in (
        "execution_mode", "provider_kind", "provider_id", "model_id", "run_kind", "source_kind"
    )).lower()
    path_blob = path.as_posix().lower()
    combined = f"{blob} {meta_blob} {path_blob}"
    if bool(run_meta.get("is_derived_audit")) or "derived_repair_audit" in combined or "/repair" in path_blob:
        return "derived_audit"
    if any(word in combined for word in ("deterministic", "fake", "offline", "test_double", "readiness_dry", "readiness-dry", "q2a")):
        return "deterministic_fixture_or_test_double"
    if any(word in combined for word in ("remote", "api", "live_probe")):
        return "remote_or_api_attempt"
    if any(word in combined for word in ("local_model", "ollama", "qwen", "ministral", "-local-", "_local")):
        return "observed_local_model_trace"
    if "p0-smoke" in path_blob:
        return "deterministic_fixture_or_test_double"
    return "unclassified_trace"


def source_record_type(record: dict[str, Any]) -> str:
    return str(first_nonempty(record.get("event_type"), record.get("record_type"), record.get("kind"), "artifact"))


def is_event_like(record: dict[str, Any], artifact_type: str) -> bool:
    if artifact_type in {"raw_event", "transcript", "ledger", "metadata"}:
        return True
    # Some JSON transcript/outcome containers embed records; flattening is done
    # separately by ``flatten_json_container``.
    return False


def flatten_json_container(value: Any, context: str = "root") -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield embedded event/record objects from a JSON container."""
    if not isinstance(value, dict):
        return
    for key in ("records", "events", "entries"):
        items = value.get(key)
        if isinstance(items, list):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    yield f"{context}.{key}[{index}]", item


def normalize_event(
    record: dict[str, Any],
    path: Path,
    line_no: int,
    artifact_type: str,
    run_meta: dict[str, dict[str, Any]],
    embedded_path: str | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    event_type = source_record_type(record)
    run_id = run_id_of(record, path)
    episode_id = episode_id_of(record, path)
    actor, role = actor_and_role(record)
    msg, message_id, source, target = message_parts(record, event_type)
    source = source or (actor if event_type in MESSAGE_EVENT_TYPES else None)
    tool, arguments, result = tool_parts(record, event_type)
    permission = permission_of(record)
    external_effect, effect_basis = external_effect_of(record, tool)
    effect_scope = external_effect_scope(record, external_effect)
    label, label_source = explicit_label(record)
    message_text = as_text(msg)
    features, feature_reasons = infer_features(record, message_text, tool, source, target, external_effect, permission)
    timestamp = timestamp_of(record)
    base = input_root or DEFAULT_INPUT
    source_rel = path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix()
    source_hash = sha256_value(record)
    event_id = first_nonempty(record.get("event_id"), record.get("id"), message_id)
    if event_id is None:
        event_id = f"adapt-{source_hash[:20]}-{line_no}"
    meta = run_meta.get(str(run_id), {}) if run_id is not None else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(event_id),
        "run_id": run_id,
        "episode_id": episode_id,
        "record_kind": artifact_type,
        "event_type": event_type,
        "stage": first_nonempty(record.get("stage"), record.get("phase"), nested(record, ("payload", "phase"))),
        "agent_id": actor,
        "role": role,
        "message": redact(msg) if msg is not None else None,
        "message_text": message_text or None,
        "message_id": message_id,
        "message_source": source,
        "message_target": target,
        "tool": tool,
        "tool_arguments": redact(arguments) if arguments is not None else None,
        "tool_result": redact(result) if result is not None else None,
        "timestamp": timestamp,
        "parent": parent_ids_of(record),
        "chain_parent": first_nonempty(record.get("previous_event_sha256"), record.get("previous_event_id")),
        "permission": permission,
        "external_effect": external_effect,
        "external_effect_basis": effect_basis,
        "external_effect_scope": effect_scope,
        "risk_label": label,
        "risk_label_source": label_source,
        "outcome_flags": outcome_flags(record),
        "features": features,
        "feature_reasons": feature_reasons,
        "source": {
            "path": source_rel,
            "line": line_no,
            "embedded_path": embedded_path,
            "artifact_type": artifact_type,
            "authenticity": source_authenticity(record, path, artifact_type, meta),
            "execution_mode": first_nonempty(record.get("execution_mode"), meta.get("execution_mode")),
            "run_kind": first_nonempty(record.get("run_kind"), nested(record, ("payload", "run_kind")), meta.get("run_kind")),
            "provider_kind": first_nonempty(record.get("provider_kind"), nested(record, ("payload", "provider_kind")), meta.get("provider_kind")),
            "provider": first_nonempty(record.get("provider_id"), nested(record, ("payload", "provider_id")), meta.get("provider_id")),
            "model": first_nonempty(record.get("model_id"), nested(record, ("payload", "model_id")), meta.get("model_id")),
        },
        "context": {
            # These are copied only when explicitly present.  In particular,
            # eligibility flags describe the source study stage, not a newly
            # asserted security outcome.
            "phase": first_nonempty(record.get("phase"), meta.get("phase")),
            "data_role": first_nonempty(record.get("data_role"), meta.get("data_role")),
            "condition": first_nonempty(record.get("condition"), meta.get("condition")),
            "scientific_gate": first_nonempty(record.get("scientific_gate"), meta.get("scientific_gate")),
            "eligible_for_scientific_analysis": first_nonempty(record.get("eligible_for_scientific_analysis"), meta.get("eligible_for_scientific_analysis")),
            "eligible_for_causal_effect_analysis": record.get("eligible_for_causal_effect_analysis"),
            "eligible_for_confirmatory_analysis": record.get("eligible_for_confirmatory_analysis"),
            "risk_seed_present": record.get("risk_seed_present"),
        },
        "eligibility": {
            "structural_graph": bool(actor or role or source or target or tool),
            "temporal_graph": bool(timestamp),
            "bn_feature_row": True,
            "risk_evaluation": label is not None,
            "replay_evidence": bool(meta.get("has_replay")),
        },
        "source_record_hash": source_hash,
    }


def load_json_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read one file without modifying it; return records and parse errors."""
    errors = 0
    try:
        if path.suffix.lower() == ".jsonl":
            records: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
                except (ValueError, TypeError):
                    errors += 1
            return records, errors
        value = json.loads(path.read_text(encoding="utf-8"))
        return ([value] if isinstance(value, dict) else []), errors
    except (OSError, UnicodeError, ValueError, TypeError):
        return [], 1


def build_run_meta(files: list[Path], input_root: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = defaultdict(dict)
    for path in files:
        if classify_path(path) != "manifest" or path.suffix.lower() != ".json":
            continue
        records, _ = load_json_records(path)
        for record in records:
            run_id = run_id_of(record, path)
            if run_id is None:
                continue
            provider = record.get("provider") if isinstance(record.get("provider"), dict) else record.get("model")
            item = meta[str(run_id)]
            item.update(
                {
                    "manifest_path": path.relative_to(input_root).as_posix(),
                    "status": first_nonempty(record.get("status"), item.get("status")),
                    "decision": first_nonempty(record.get("decision"), item.get("decision")),
                    "condition": first_nonempty(record.get("condition"), item.get("condition")),
                    "phase": first_nonempty(record.get("phase"), item.get("phase")),
                    "data_role": first_nonempty(record.get("data_role"), item.get("data_role")),
                    "scientific_gate": first_nonempty(record.get("scientific_gate"), item.get("scientific_gate")),
                    "execution_mode": first_nonempty(record.get("execution_mode"), record.get("source_execution_mode"), item.get("execution_mode")),
                    "provider_id": provider.get("provider_id") if isinstance(provider, dict) else item.get("provider_id"),
                    "model_id": provider.get("model_id") if isinstance(provider, dict) else item.get("model_id"),
                    "eligible_for_scientific_analysis": record.get("eligible_for_scientific_analysis", item.get("eligible_for_scientific_analysis")),
                    "real_model_calls": first_nonempty(record.get("real_model_calls"), item.get("real_model_calls")),
                    "remote_network_calls": first_nonempty(record.get("remote_network_calls"), item.get("remote_network_calls")),
                    "p1_go": first_nonempty(record.get("p1_go"), item.get("p1_go")),
                    "p2_allowed": first_nonempty(record.get("p2_allowed"), item.get("p2_allowed")),
                    "is_derived_audit": first_nonempty(record.get("is_derived_audit"), item.get("is_derived_audit")),
                }
            )
    # Manifests are not present for every Q1/Q2 run.  Recover only provenance
    # metadata that is explicitly recorded in event/ledger rows so those runs
    # are still classified correctly; this does not invent a provider or call.
    for path in files:
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        artifact_type = classify_path(path)
        records, _ = load_json_records(path)
        for record in records:
            run_id = run_id_of(record, path)
            if run_id is None:
                continue
            item = meta[str(run_id)]
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
            item["execution_mode"] = first_nonempty(
                item.get("execution_mode"), record.get("execution_mode"), record.get("source_execution_mode"),
                payload.get("execution_mode"),
            )
            item["provider_kind"] = first_nonempty(item.get("provider_kind"), record.get("provider_kind"), payload.get("provider_kind"))
            item["provider_id"] = first_nonempty(item.get("provider_id"), record.get("provider_id"), payload.get("provider_id"), provider.get("provider_id"))
            item["model_id"] = first_nonempty(item.get("model_id"), record.get("model_id"), payload.get("model_id"), provider.get("model_id"), record.get("model"))
            item["run_kind"] = first_nonempty(item.get("run_kind"), record.get("run_kind"), payload.get("run_kind"))
            item["phase"] = first_nonempty(item.get("phase"), record.get("phase"), payload.get("phase"))
            item["data_role"] = first_nonempty(item.get("data_role"), record.get("data_role"), payload.get("data_role"))
            item["scientific_gate"] = first_nonempty(item.get("scientific_gate"), record.get("scientific_gate"), payload.get("scientific_gate"))
            item["real_model_calls"] = first_nonempty(item.get("real_model_calls"), record.get("real_model_calls"))
            item["remote_network_calls"] = first_nonempty(item.get("remote_network_calls"), record.get("remote_network_calls"))
            if artifact_type == "replay":
                item["has_replay"] = True
            if artifact_type == "validation":
                item["has_validation"] = True
    # Link validation/replay presence by run id from filenames/directories.
    for path in files:
        kind = classify_path(path)
        if kind not in {"validation", "replay"}:
            continue
        records, _ = load_json_records(path)
        ids = {run_id_of(r, path) for r in records} or {path_run_id(path)}
        for run_id in ids:
            if run_id is None:
                continue
            item = meta[str(run_id)]
            if kind == "validation":
                item["has_validation"] = True
            else:
                item["has_replay"] = True
    return meta


def artifact_inventory(path: Path, input_root: Path, run_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact_type = classify_path(path)
    rel = path.relative_to(input_root).as_posix()
    raw = path.read_bytes()
    records, parse_errors = load_json_records(path) if path.suffix.lower() in {".json", ".jsonl"} else ([], 0)
    key_counts: Counter[str] = Counter()
    record_types: Counter[str] = Counter()
    run_ids: set[str] = set()
    for record in records:
        key_counts.update(str(k) for k in record)
        record_types[source_record_type(record)] += 1
        rid = run_id_of(record, path)
        if rid:
            run_ids.add(str(rid))
    if not run_ids:
        rid = path_run_id(path)
        if rid:
            run_ids.add(rid)
    linked = [run_meta.get(rid, {}) for rid in sorted(run_ids)]
    statuses = sorted({str(x.get("status")) for x in linked if x.get("status") is not None})
    decisions = sorted({str(x.get("decision")) for x in linked if x.get("decision") is not None})
    linked_meta = linked[0] if linked else {}
    authenticity = "derived_or_summary_artifact" if artifact_type in {"manifest", "outcome", "validation", "replay", "report"} else source_authenticity(records[0] if records else {}, path, artifact_type, linked_meta)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": sha256_bytes((rel + ":" + sha256_bytes(raw)).encode("utf-8"))[:24],
        "path": rel,
        "artifact_type": artifact_type,
        "suffix": path.suffix.lower(),
        "bytes": len(raw),
        "records": len(records),
        "parse_errors": parse_errors,
        "top_level_keys": sorted(key_counts),
        "record_types": dict(record_types),
        "run_ids": sorted(run_ids),
        "authenticity": authenticity,
        "linked_statuses": statuses,
        "linked_decisions": decisions,
        "read_only_projection": True,
    }


def infer_risk_node(row: dict[str, Any]) -> str | None:
    text = " ".join(
        str(x or "")
        for x in (row.get("text"), row.get("roles"), row.get("tools"), row.get("event_types"))
    ).lower()
    scores = {
        "extract": sum(text.count(x) for x in ("extract", "dossier", "evidence", "claim")),
        "verify": sum(text.count(x) for x in ("verify", "validation", "provenance", "identity")),
        "analyze": sum(text.count(x) for x in ("risk", "analysis", "assess", "impact", "fraud")),
        "policy": sum(text.count(x) for x in ("policy", "authorization", "approval", "permission")),
        "report": sum(text.count(x) for x in ("report", "publish", "delivery", "export")),
        "intake": sum(text.count(x) for x in ("intake", "request", "task_received")),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else None


def aggregate_bn_rows(events: list[dict[str, Any]], run_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        run_id = str(event.get("run_id") or "__unknown_run__")
        episode_id = str(event.get("episode_id") or "__run__")
        groups[(run_id, episode_id)].append(event)
    rows: list[dict[str, Any]] = []
    for (run_id, episode_id), group in sorted(groups.items()):
        features = {name: int(any(int(e.get("features", {}).get(name, 0)) for e in group)) for name in FEATURES}
        texts: list[str] = []
        for event in group:
            text = event.get("message_text")
            if text and text not in texts:
                texts.append(str(text))
        labels = [e.get("risk_label") for e in group if e.get("risk_label") is not None]
        label = labels[0] if labels and all(x == labels[0] for x in labels) else None
        timestamps = [e.get("timestamp") for e in group if e.get("timestamp")]
        roles = sorted({str(e.get("role")) for e in group if e.get("role")})
        tools = sorted({str(e.get("tool")) for e in group if e.get("tool")})
        event_types = sorted({str(e.get("event_type")) for e in group if e.get("event_type")})
        meta = run_meta.get(run_id, {})
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": f"{run_id}:{episode_id}",
            "run_id": run_id,
            "episode_id": episode_id,
            "label": label,
            "label_source": sorted({str(e.get("risk_label_source")) for e in group if e.get("risk_label_source")}),
            "label_status": "explicit_and_consistent" if label is not None else "unavailable_or_inconsistent",
            "features": features,
            "text": " ".join(texts)[:12000],
            "events": len(group),
            "roles": roles,
            "tools": tools,
            "event_types": event_types,
            "risk_node": infer_risk_node({"text": " ".join(texts), "roles": roles, "tools": tools, "event_types": event_types}),
            "timestamp_first": min(timestamps) if timestamps else None,
            "timestamp_last": max(timestamps) if timestamps else None,
            "timestamp_coverage": len(timestamps) / len(group) if group else 0.0,
            "agent_coverage": sum(bool(e.get("agent_id") or e.get("role")) for e in group) / len(group) if group else 0.0,
            "source_event_ids": [str(e.get("event_id")) for e in group],
            "source_authenticities": sorted({str(e.get("source", {}).get("authenticity")) for e in group}),
            "replay_evidence": bool(meta.get("has_replay")),
            "validation_evidence": bool(meta.get("has_validation")),
            "eligible_for_metrics": label is not None,
            "evaluation_note": "仅结构/字段覆盖；无明确 risk_label 时不得计算安全指标。" if label is None else None,
        }
        # Feed exactly the row shape consumed by run_model_test_v1.predict.
        row["bn_input"] = {"features": features, "text": row["text"], "label": label}
        rows.append(row)
    return rows


def score_with_existing_bn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the existing hierarchical-BN predictor without inventing labels."""
    module_path = ROOT / 'code/commands/run_model_test_v1.py'
    try:
        spec = importlib.util.spec_from_file_location("model_test_v1_for_adapter", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("module spec unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict = module.predict
        outputs = []
        for row in rows:
            score, node, evidence = predict(row["bn_input"], "hierarchical_bn")
            outputs.append(
                {
                    "id": row["id"],
                    "score": round(float(score), 8),
                    "risk_node": node,
                    "evidence": evidence,
                }
            )
        labels = [row.get("label") for row in rows]
        status = "scored_labels_present_metrics_not_run" if labels and all(x in (0, 1) for x in labels) else "coverage_only_no_risk_labels"
        return {
            "status": status,
            "metric_status": "not_evaluated_no_complete_risk_labels" if status == "coverage_only_no_risk_labels" else "labels_present_metrics_not_run",
            "predictor": 'paperAlpha/code/commands/run_model_test_v1.py:predict(method=hierarchical_bn)',
            "n_rows": len(outputs),
            "metrics": None,
            "predictions": outputs,
        }
    except Exception as exc:  # pragma: no cover - defensive environment path
        return {
            "status": "bn_import_failed",
            "predictor": str(module_path),
            "n_rows": 0,
            "metrics": None,
            "predictions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def coverage_summary(
    events: list[dict[str, Any]],
    bn_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    run_meta: dict[str, dict[str, Any]],
    input_root: Path,
) -> dict[str, Any]:
    fields = (
        "agent_id",
        "role",
        "message",
        "message_source",
        "message_target",
        "tool",
        "timestamp",
        "parent",
        "chain_parent",
        "permission",
        "external_effect",
        "external_effect_scope",
        "risk_label",
    )
    field_counts = {
        field: sum(1 for e in events if e.get(field) not in (None, "", [], {}))
        for field in fields
    }
    by_auth: dict[str, dict[str, Any]] = {}
    auth_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        auth_groups[str(e.get("source", {}).get("authenticity", "unclassified"))].append(e)
    for auth, group in sorted(auth_groups.items()):
        by_auth[auth] = {
            "events": len(group),
            "runs": sorted({str(e.get("run_id")) for e in group if e.get("run_id")}),
            "field_coverage": {field: sum(1 for e in group if e.get(field) not in (None, "", [], {})) / len(group) for field in fields},
            "timestamped_events": sum(bool(e.get("timestamp")) for e in group),
            "explicit_labels": sum(e.get("risk_label") is not None for e in group),
        }
    artifact_types = Counter(str(a.get("artifact_type")) for a in artifacts)
    stage_files = Counter(str(a.get("path", "").split("/", 1)[0]) for a in artifacts)
    stage_records = Counter()
    for artifact in artifacts:
        stage = str(artifact.get("path", "").split("/", 1)[0])
        stage_records[stage] += int(artifact.get("records", 0) or 0)
    event_types = Counter(str(e.get("event_type")) for e in events)
    execution_modes = Counter(str(e.get("source", {}).get("execution_mode")) for e in events)
    phases = Counter(str(e.get("stage")) for e in events if e.get("stage") is not None)
    data_roles = Counter()
    for e in events:
        # data_role is retained in the source record only when it is useful for
        # classification; infer it from the source artifact metadata where it
        # is not promoted to a core event field.
        source_path = str(e.get("source", {}).get("path", ""))
        if "p1_gate" in source_path:
            data_roles["p1_gate_pilot"] += 1
        elif "engineering_dry" in source_path or "DRY" in source_path:
            data_roles["engineering_dry_or_fixture"] += 1
        elif "qualification" in source_path:
            data_roles["qualification"] += 1
    outcome_flag_counts = Counter(
        key
        for event in events
        for key in (event.get("outcome_flags") or {})
    )
    parse_error_files = sum(1 for a in artifacts if int(a.get("parse_errors", 0) or 0) > 0)
    parse_error_records = sum(int(a.get("parse_errors", 0) or 0) for a in artifacts)
    source_hashes = [str(e.get("source_record_hash")) for e in events if e.get("source_record_hash")]
    unique_runs = {str(e.get("run_id")) for e in events if e.get("run_id")}
    unique_episodes = {str(e.get("episode_id")) for e in events if e.get("episode_id")}
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(input_root),
        "artifact_count": len(artifacts),
        "source_file_count_including_gitkeep": sum(1 for p in input_root.rglob("*") if p.is_file()),
        "event_count": len(events),
        "bn_episode_count": len(bn_rows),
        "unique_source_record_hashes": len(set(source_hashes)),
        "duplicate_projection_records": len(source_hashes) - len(set(source_hashes)),
        "unique_run_ids_in_events": len(unique_runs),
        "unique_episode_ids_in_events": len(unique_episodes),
        "run_count_from_manifests": len(run_meta),
        "source_record_count": sum(int(a.get("records", 0) or 0) for a in artifacts),
        "parse_error_files": parse_error_files,
        "parse_error_records": parse_error_records,
        "artifact_type_counts": dict(sorted(artifact_types.items())),
        "stage_file_counts": dict(sorted(stage_files.items())),
        "stage_record_counts": dict(sorted(stage_records.items())),
        "event_type_counts": dict(event_types.most_common()),
        "execution_mode_counts": dict(execution_modes.most_common()),
        "phase_counts": dict(phases.most_common()),
        "data_role_path_counts": dict(data_roles.most_common()),
        "outcome_flag_counts_in_events": dict(outcome_flag_counts.most_common()),
        "field_coverage_count": field_counts,
        "field_coverage_rate": {field: field_counts[field] / len(events) if events else 0.0 for field in fields},
        "explicit_risk_label_events": sum(e.get("risk_label") is not None for e in events),
        "explicit_risk_label_episodes": sum(row.get("label") is not None for row in bn_rows),
        "outcome_fields_are_not_labels": True,
        "by_authenticity": by_auth,
        "eligible_for_temporal_bn_events": sum(bool(e.get("eligibility", {}).get("temporal_graph")) for e in events),
        "eligible_for_structural_graph_events": sum(bool(e.get("eligibility", {}).get("structural_graph")) for e in events),
        "reproducibility_notes": {
            "raw_sources_read_only": True,
            "normalized_records_keep_source_hash": True,
            "timestamps_are_never_synthesized": True,
            "labels_are_never_inferred_from_outcomes": True,
        },
    }


def schema_markdown() -> str:
    return """# MAS adapter schema (mas-adapter-v1)

`events.jsonl` 是只读投影，每行对应一个原始事件、transcript 记录或 ledger 记录。原文件不被改写。

核心字段：

- `agent_id`, `role`: 事件执行者；消息事件另有 `message_source`/`message_target`。
- `record_kind`: `raw_event`、`transcript`、`ledger` 或 `metadata`，用于筛选不同证据视图。
- `message`, `message_text`: 经过敏感键脱敏的结构化消息及文本摘要；不记录隐藏思维链。
- `tool`, `tool_arguments`, `tool_result`: 工具名称及脱敏参数/结果。
- `timestamp`: 原始 `timestamp`/`recorded_at`；缺失时保持 `null`，不用序号伪造时间。
- `parent`: 消息父 ID；`chain_parent` 单独保存 hash-chain 上一条记录，不能当作因果父边。
- `permission`: 原始权限证据；没有证据时为 `null`。
- `external_effect`: `true/false/null` 三值；`external_effect_scope` 区分
  `local_sandbox` 与 `remote_service`，不能把本地 sink 当成互联网副作用。
- `risk_label`: 只接受显式二元风险字段；`risk_level`、`risk_score`、`condition`、`leak_detected` 只进入 `outcome_flags`。
- `features`: 与 `run_model_test_v1.py` 相同的七个输入特征，供离线回放；这些是可观测线索，不是标签。

`bn_replay.json` 按 `(run_id, episode_id)` 聚合上述特征，并调用既有 `predict(..., "hierarchical_bn")`。无显式标签时状态为 `coverage_only_no_risk_labels`，`metrics` 必须为 `null`。

`artifacts.jsonl` 包含 manifests、outcomes、validation、replay、reports 等文件的 schema/哈希覆盖摘要；它们不混入事件流。

机器可读约束见 `event_schema.json` 与 `artifact_schema.json`；源字段之外的扩展字段允许保留，以避免破坏不同阶段的证据。
"""


def machine_schema() -> dict[str, Any]:
    """Return a compact JSON Schema for the normalized event contract.

    The schema intentionally leaves source-specific payload details open: the
    adapter's job is to preserve heterogeneous evidence while making the
    shared fields machine-checkable.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "mas-adapter-event.schema.json",
        "title": "pre_exp1 normalized observable event",
        "type": "object",
        "required": ["schema_version", "event_id", "record_kind", "event_type", "agent_id", "role", "message", "tool", "timestamp", "parent", "permission", "external_effect", "risk_label", "features", "source"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "event_id": {"type": "string"},
            "record_kind": {"type": "string"},
            "run_id": {"type": ["string", "null"]},
            "episode_id": {"type": ["string", "null"]},
            "event_type": {"type": "string"},
            "agent_id": {"type": ["string", "null"]},
            "role": {"type": ["string", "null"]},
            "message": {},
            "message_text": {"type": ["string", "null"]},
            "message_source": {"type": ["string", "null"]},
            "message_target": {"type": ["string", "null"]},
            "tool": {"type": ["string", "null"]},
            "timestamp": {"type": ["string", "null"]},
            "parent": {"type": "array", "items": {"type": "string"}},
            "chain_parent": {"type": ["string", "null"]},
            "permission": {"type": ["object", "null"]},
            "external_effect": {"type": ["boolean", "null"]},
            "external_effect_scope": {"type": ["string", "null"]},
            "risk_label": {"type": ["integer", "null"], "enum": [0, 1, None]},
            "features": {"type": "object", "required": list(FEATURES)},
            "source": {"type": "object"},
        },
        "additionalProperties": True,
    }


def artifact_machine_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "mas-adapter-artifact.schema.json",
        "title": "pre_exp1 source artifact inventory row",
        "type": "object",
        "required": ["schema_version", "artifact_id", "path", "artifact_type", "bytes", "records", "parse_errors", "read_only_projection"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "artifact_id": {"type": "string"},
            "path": {"type": "string"},
            "artifact_type": {"type": "string"},
            "bytes": {"type": "integer", "minimum": 0},
            "records": {"type": "integer", "minimum": 0},
            "parse_errors": {"type": "integer", "minimum": 0},
            "read_only_projection": {"const": True},
        },
        "additionalProperties": True,
    }


def report_markdown(summary: dict[str, Any], bn_replay: dict[str, Any], artifacts: list[dict[str, Any]], run_meta: dict[str, dict[str, Any]]) -> str:
    auth_lines = []
    for key, value in summary.get("by_authenticity", {}).items():
        auth_lines.append(f"- `{key}`：{value['events']} events，{len(value['runs'])} runs，时间戳 {value['timestamped_events']}/{value['events']}。")
    label_events = summary.get("explicit_risk_label_events", 0)
    stage_lines = [
        f"- `{key}`：{value} files / {summary.get('stage_record_counts', {}).get(key, 0)} source records。"
        for key, value in summary.get("stage_file_counts", {}).items()
    ]
    field_rates = summary.get("field_coverage_rate", {})
    field_lines = [
        f"| `{key}` | {summary.get('field_coverage_count', {}).get(key, 0)} | {value:.1%} |"
        for key, value in field_rates.items()
    ]
    event_lines = [
        f"- `{key}`: {value}"
        for key, value in list(summary.get("event_type_counts", {}).items())[:18]
    ]
    mode_lines = [f"- `{key}`: {value}" for key, value in summary.get("execution_mode_counts", {}).items()]
    return "\n".join(
        [
            "# Pre-exp1 → MAS adapter report",
            "",
            "本报告由只读适配器生成；原始 `paperAlpha/data/pre_exp1` 未被修改。",
            "",
            "## 实际覆盖",
            "",
            f"- 输入文件：{summary.get('source_file_count_including_gitkeep', summary['artifact_count'])}（含占位 `.gitkeep`）；纳入适配：{summary['artifact_count']}；源记录：{summary.get('source_record_count', 0)}。",
            f"- 投影事件/记录：{summary['event_count']}（唯一源记录 hash {summary.get('unique_source_record_hashes', 0)}，重复投影 {summary.get('duplicate_projection_records', 0)}）；聚合 episode/run 单元：{summary['bn_episode_count']}；解析错误文件/记录：{summary.get('parse_error_files', 0)}/{summary.get('parse_error_records', 0)}。",
            f"- 明确 `risk_label` 事件：{label_events}；明确标签 episode：{summary['explicit_risk_label_episodes']}。",
            f"- 时间戳事件：{summary['eligible_for_temporal_bn_events']}；有 agent/role/消息/工具结构证据的事件：{summary['eligible_for_structural_graph_events']}。",
            f"- 原始事件中保留的 outcome 标记计数：{summary.get('outcome_flag_counts_in_events', {})}；`risk_level`、`risk_score`、`condition`、`leak_detected` 未转成安全标签。",
            "",
            "### 文件/阶段统计",
            "",
            *stage_lines,
            "",
            "### 统一字段覆盖",
            "",
            "| 字段 | 有值记录 | 覆盖率 |",
            "|---|---:|---:|",
            *field_lines,
            "",
            "### 事件与执行模式（投影后）",
            "",
            *event_lines,
            "",
            *mode_lines,
            "",
            "### 轨迹性质",
            "",
            *auth_lines,
            "",
            "- 可作正常结构/回放对照：P0 smoke、P1 benign/engineering dry（确定性 test-double，含多角色消息/工具链）；不代表真实攻击分布。",
            "- 可作真实本地模型结构诊断：P1 Qwen3/Mistral qualification、P1 scientific Qwen3 pilot、P1 benign local shakedown；其中 pilot 有 20 episodes/180 次模型请求，但 gate/validation 未通过。",
            "- 仅作 API/协议资格证据：P1v2 Q1 local、Q2A/Q2B/Q2C；没有完整多 Agent 交互轨迹或成功远程安全结果，不能充当 MAS 风险样本。",
            "- `derived_or_summary_artifact`（manifest/outcome/replay/report）单独列在 `artifacts.jsonl`，不混入原始事件流。",
            "",
            "## BN 离线回放",
            "",
            f"- 状态：`{bn_replay.get('status')}`；预测行数：{bn_replay.get('n_rows', 0)}。",
            "- 调用接口：`run_model_test_v1.py:predict(method=hierarchical_bn)`。",
            f"- metric_status：`{bn_replay.get('metric_status', 'not_evaluated_no_complete_risk_labels')}`；`metrics` 为 null。",
            "- 本轮只做特征/结构覆盖回放；没有完整且经确认的风险标签时不计算 AUPRC、ECE、召回等安全指标。",
            "",
            "## Benchmark 可得性",
            "",
            '- HarnessAudit：本地 `code/external/HarnessAudit` 已有官方源码/任务规范（210 个 multi-agent task YAML、fixtures/perturbation specs），但 `multi_agent/traces` 与 `results` 为空；因此环境/任务可复现，论文运行轨迹不可直接复用。',
            "- HarnessAudit HF：README 同时出现公开 `LCZZZZ/HarnessAudit` 与匿名 401 的 `eric-ai-lab/HarnessAudit` 入口，版本/匿名复现存在 blocker；本适配器不自动下载。",
            "- A2ASecBench：官方代码和静态 cases/offline configs 可获得，但当前工作区没有本地副本，官方仓库 results 不是完整运行集；需另行锁版本并运行。",
            "- 结论：本轮不声称复现任何公开基准安全指标。",
            "",
            "## 已知 blocker",
            "",
            "- 多数事件缺少真实外部效果和统一权限语义；`external_effect`/`permission` 缺证据时保持 `null`；`local_sandbox` 不等于远程副作用。",
            "- q1/q2 ledger 有 hash-chain 顺序但不一定有绝对时间；`chain_parent` 不当作因果父边。",
            "- 适配后的 transcript/ledger 与 raw event 可能是同一运行的不同视图；下游应按 `source.path`、`source.artifact_type` 和 `source_record_hash` 去重，不能把投影条数当独立样本数。",
            "",
            "## 最小命令",
            "",
            "```powershell",
            "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\adapt_pre_exp1.py",
            "conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\\scripts\\replay_bn_adapter.py --input paperAlpha\\data\\mas_adapter\\bn_episodes.jsonl --output paperAlpha\\data\\mas_adapter\\bn_replay_standalone.json",
            "```",
            "",
            '输出：`data/processed/mas_adapter/events.jsonl`、`artifacts.jsonl`、`bn_episodes.jsonl`、`bn_replay.json`、`coverage.json`、`event_schema.json`、`artifact_schema.json`、`SCHEMA.md`。',
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-files", type=int, default=0, help="仅调试时限制输入文件数；0=全部")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in sorted(input_root.rglob("*"))
        if p.is_file() and p.name != ".gitkeep" and p.suffix.lower() in {".json", ".jsonl", ".md"}
    ]
    if args.limit_files > 0:
        files = files[: args.limit_files]

    run_meta = build_run_meta(files, input_root)
    artifacts = [artifact_inventory(path, input_root, run_meta) for path in files]
    events: list[dict[str, Any]] = []
    for path in files:
        artifact_type = classify_path(path)
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        records, _ = load_json_records(path)
        for line_no, record in enumerate(records, start=1):
            if is_event_like(record, artifact_type):
                events.append(normalize_event(record, path, line_no, artifact_type, run_meta, input_root=input_root))
            # Flatten records embedded in q2 transcripts/ledger JSON containers.
            if path.suffix.lower() == ".json" and artifact_type in {"transcript", "ledger", "replay", "outcome"}:
                try:
                    container = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, ValueError, TypeError):
                    container = None
                for embedded_path, embedded in flatten_json_container(container):
                    events.append(normalize_event(embedded, path, line_no, artifact_type, run_meta, embedded_path, input_root))

    # Stable event ordering makes diffs/replays auditable.
    events.sort(key=lambda e: (str(e.get("run_id")), str(e.get("episode_id")), str(e.get("timestamp") or ""), str(e.get("event_id"))))
    bn_rows = aggregate_bn_rows(events, run_meta)
    bn_replay = score_with_existing_bn(bn_rows)
    summary = coverage_summary(events, bn_rows, artifacts, run_meta, input_root)
    summary["input_file_limit"] = args.limit_files
    summary["source_files"] = [a["path"] for a in artifacts]

    def write_jsonl(name: str, rows: Iterable[dict[str, Any]]) -> None:
        (output_dir / name).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")

    write_jsonl("events.jsonl", events)
    write_jsonl("artifacts.jsonl", artifacts)
    write_jsonl("bn_episodes.jsonl", bn_rows)
    (output_dir / "bn_replay.json").write_text(json.dumps(bn_replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "coverage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "event_schema.json").write_text(json.dumps(machine_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "artifact_schema.json").write_text(json.dumps(artifact_machine_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SCHEMA.md").write_text(schema_markdown(), encoding="utf-8")
    (output_dir / "ADAPTER_REPORT.md").write_text(report_markdown(summary, bn_replay, artifacts, run_meta), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "artifacts": len(artifacts),
                "events": len(events),
                "bn_episodes": len(bn_rows),
                "explicit_risk_label_events": summary["explicit_risk_label_events"],
                "bn_status": bn_replay.get("status"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
