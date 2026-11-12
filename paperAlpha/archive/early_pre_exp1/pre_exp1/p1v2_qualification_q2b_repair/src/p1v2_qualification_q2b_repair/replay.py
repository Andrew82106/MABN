"""Independent no-I/O semantic replay of repair-native audit evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, HISTORICAL_PRE_REWORK_AUDIT_ID, ORIGINAL_RUN_ID, PACKAGE_ROOT, canonical_json, load_json, original_tree_hashes, sha256_file, sha256_text, validate_audit_id
from .protocol import frozen_metadata_profile, load_assets, request_sha256


def _safe_identity_ok(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"availability", "provider_declared_model", "request_id", "system_fingerprint", "provider_version"} and value.get("availability") == "available" and value.get("provider_declared_model") == "gpt-5.3-codex-spark" and all(isinstance(value.get(key), str) and value.get(key) for key in ("request_id", "system_fingerprint", "provider_version"))


def _expected_metadata_identity(assets: Any) -> dict[str, Any]:
    """Independent replay oracle derived only from the frozen provider profile."""
    target_model_id, model_ids = frozen_metadata_profile(assets)
    return {
        "normalized_models_sha256": sha256_text(canonical_json({"model_ids": list(model_ids)})),
        "target_model_present": target_model_id in model_ids,
        "model_count": len(model_ids),
    }


def _replay_deadline_execution(execution: Any, errors: list[str]) -> None:
    expected_stages = ("metadata_start", "completion", "metadata_end")
    if not isinstance(execution, dict) or set(execution) != {"calls", "flow_status", "runner_kind", "stages", "termination_reason"}:
        errors.append("replay_deadline_shape")
        return
    if execution.get("runner_kind") != "offline_fake_staged_deadline_guard" or execution.get("flow_status") != "completed" or execution.get("termination_reason") is not None:
        errors.append("replay_deadline_status")
        return
    calls = execution.get("calls")
    stages = execution.get("stages")
    if not isinstance(calls, list) or not isinstance(stages, list) or len(calls) != 3 or len(stages) != 3:
        errors.append("replay_deadline_count")
        return
    deadline_at: float | None = None
    for expected_stage, call, stage in zip(expected_stages, calls, stages):
        if not isinstance(call, dict) or set(call) != {"deadline_at_seconds", "remaining_before_seconds", "stage", "timeout_seconds"} or not isinstance(stage, dict) or set(stage) != {"io_attempted", "reason", "stage", "status", "timeout_seconds"}:
            errors.append("replay_deadline_item_shape")
            return
        remaining = call.get("remaining_before_seconds")
        timeout = call.get("timeout_seconds")
        current_deadline = call.get("deadline_at_seconds")
        if call.get("stage") != expected_stage or stage.get("stage") != expected_stage or stage.get("status") != "completed" or stage.get("reason") is not None or stage.get("io_attempted") is not True:
            errors.append("replay_deadline_order")
            return
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not isinstance(current_deadline, (int, float)) or isinstance(current_deadline, bool):
            errors.append("replay_deadline_timing")
            return
        if remaining <= 0 or timeout != min(60.0, float(remaining)) or stage.get("timeout_seconds") != timeout:
            errors.append("replay_deadline_timeout")
            return
        if deadline_at is None:
            deadline_at = float(current_deadline)
        elif float(current_deadline) != deadline_at:
            errors.append("replay_deadline_deadline")
            return


def _replay_ledger(path: Path, audit_id: str, transcript: dict[str, Any], expected_metadata: dict[str, Any] | None, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        errors.append("replay_ledger_unreadable")
        return
    if len(lines) != 5:
        errors.append("replay_ledger_count")
        return
    events: list[dict[str, Any]] = []
    previous = "GENESIS"
    for index, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append("replay_ledger_json")
            return
        if not isinstance(event, dict) or set(event) != {"kind", "payload", "previous_event_sha256", "sequence", "event_sha256"}:
            errors.append("replay_ledger_shape")
            return
        core = {key: event[key] for key in ("kind", "payload", "previous_event_sha256", "sequence")}
        if event["sequence"] != index or event["previous_event_sha256"] != previous or event["event_sha256"] != sha256_text(canonical_json(core)):
            errors.append("replay_ledger_chain")
            return
        previous = event["event_sha256"]
        events.append(event)
    if expected_metadata is None:
        errors.append("replay_metadata_oracle")
        return
    record = transcript["records"][0]
    expected = [
        ("audit_started", {"audit_id": audit_id, "artifact_kind": "derived_audit"}),
        ("metadata_start", expected_metadata),
        ("completion_attempted", {"episode_id": record["task_card"]["episode_id"], "request_sha256": record["request_sha256"]}),
        ("completion_failed", {"episode_id": record["task_card"]["episode_id"], "request_sha256": record["request_sha256"], "failure_code": record["failure_code"], "safe_identity": record["safe_identity"]}),
        ("metadata_end", expected_metadata | {"consistent": True}),
    ]
    for event, (kind, payload) in zip(events, expected):
        if event["kind"] != kind or event["payload"] != payload:
            errors.append("replay_ledger_semantics")
            return


def replay_audit(audit_id: str, *, asset_root: Path = PACKAGE_ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_audit_id(audit_id)
        assets = load_assets(asset_root)
        audit_dir = data_root / "audits" / audit_id
        audit = load_json(audit_dir / "audit.json")
        transcript = load_json(audit_dir / "transcript.json")
        outcome = load_json(audit_dir / "outcome.json")
        manifest = load_json(audit_dir / "manifest.json")
    except Exception:
        return {"passed": False, "audit_id": audit_id, "errors": ["replay_inputs_unreadable"]}
    if not isinstance(audit, dict) or not isinstance(transcript, dict) or not isinstance(outcome, dict) or not isinstance(manifest, dict):
        return {"passed": False, "audit_id": audit_id, "errors": ["replay_shape"]}
    required = {"artifact_kind": "derived_audit", "is_derived_audit": True, "is_new_run": False, "derived_from_run_id": ORIGINAL_RUN_ID, "source_execution_mode": "remote_staged_qualification", "audit_id": audit_id, "historical_pre_rework_audit_id": HISTORICAL_PRE_REWORK_AUDIT_ID}
    if any(audit.get(key) != value for key, value in required.items()):
        errors.append("replay_provenance")
    current_original = original_tree_hashes()
    if audit.get("original_tree_sha256_pre") != current_original or audit.get("original_tree_sha256_post") != current_original or audit.get("original_tree_unchanged") is not True:
        errors.append("replay_original_tree")
    try:
        expected_metadata = _expected_metadata_identity(assets)
    except (TypeError, ValueError):
        expected_metadata = None
        errors.append("replay_metadata_oracle")
    local_delivery_path = audit_dir / "reports" / "DELIVERY.md"
    declared_artifacts = manifest.get("artifact_sha256")
    try:
        if not isinstance(declared_artifacts, dict) or declared_artifacts.get("reports/DELIVERY.md") != sha256_file(local_delivery_path):
            errors.append("replay_local_delivery_hash")
    except OSError:
        errors.append("replay_local_delivery_hash")
    records = transcript.get("records")
    metadata = transcript.get("metadata")
    if transcript.get("transcript_version") != "repair-1.1.0":
        errors.append("replay_transcript_version")
    _replay_deadline_execution(transcript.get("deadline_execution"), errors)
    if not isinstance(records, list) or len(records) != 1 or not isinstance(metadata, dict):
        errors.append("replay_transcript")
    else:
        record = records[0]
        expected_record = {"status", "failure_code", "failure_class", "request_sha256", "safe_identity", "task_card"}
        if not isinstance(record, dict) or set(record) != expected_record:
            errors.append("replay_record_shape")
        else:
            if record.get("status") != "failed" or record.get("failure_code") != "tool_calls" or record.get("failure_class") != "model_output" or record.get("task_card") != assets.task:
                errors.append("replay_record_semantics")
            if record.get("request_sha256") != request_sha256(assets.task, assets):
                errors.append("replay_request_hash")
            if not _safe_identity_ok(record.get("safe_identity")):
                errors.append("replay_safe_identity")
        if set(metadata) != {"start", "end", "consistent"} or metadata.get("consistent") is not True or metadata.get("start") != metadata.get("end"):
            errors.append("replay_metadata")
        elif expected_metadata is None or metadata.get("start") != expected_metadata or metadata.get("end") != expected_metadata:
            errors.append("replay_metadata_oracle")
    expected_outcome = {"repair_decision": "passed", "metadata_calls": 2, "completion_calls": 1, "retry_count": 0, "failure_record_count": 1, "identity_consistent": True}
    if outcome != expected_outcome:
        errors.append("replay_outcome")
    if isinstance(transcript, dict) and isinstance(transcript.get("records"), list) and transcript["records"] and isinstance(transcript.get("metadata"), dict):
        _replay_ledger(audit_dir / "ledger.jsonl", audit_id, transcript, expected_metadata, errors)
    else:
        errors.append("replay_ledger_precondition")
    return {"passed": not errors, "audit_id": audit_id, "errors": sorted(set(errors)), "derived_repair_decision": "passed" if not errors else "rework"}
