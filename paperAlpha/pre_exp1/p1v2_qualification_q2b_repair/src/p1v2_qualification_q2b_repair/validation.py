"""Fail-closed validator for repair-native derived audit artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, HISTORICAL_PRE_REWORK_AUDIT_ID, ORIGINAL_RUN_ID, PACKAGE_ROOT, canonical_json, load_json, original_tree_hashes, sha256_file, sha256_text, source_file_hashes, validate_audit_id
from .protocol import FROZEN_ASSETS, build_request, frozen_metadata_profile, load_assets, request_sha256


def _add(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def _load(path: Path, errors: list[str], code: str) -> Any | None:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _add(errors, code)
        return None


def _expected_metadata_identity(assets: Any) -> dict[str, Any]:
    """Independent validator oracle derived only from the frozen provider profile."""
    target_model_id, model_ids = frozen_metadata_profile(assets)
    return {
        "normalized_models_sha256": sha256_text(canonical_json({"model_ids": list(model_ids)})),
        "target_model_present": target_model_id in model_ids,
        "model_count": len(model_ids),
    }


def _validate_ledger(path: Path, audit_id: str, transcript: Any, expected_metadata: dict[str, Any] | None, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _add(errors, "ledger_unreadable")
        return
    if len(lines) != 5 or not isinstance(transcript, dict):
        _add(errors, "ledger_event_count")
        return
    previous = "GENESIS"
    events: list[dict[str, Any]] = []
    for sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            _add(errors, "ledger_json")
            return
        if not isinstance(event, dict) or set(event) != {"kind", "payload", "previous_event_sha256", "sequence", "event_sha256"}:
            _add(errors, "ledger_shape")
            return
        core = {key: event[key] for key in ("kind", "payload", "previous_event_sha256", "sequence")}
        if event["sequence"] != sequence or event["previous_event_sha256"] != previous or event["event_sha256"] != sha256_text(canonical_json(core)):
            _add(errors, "ledger_chain")
            return
        previous = event["event_sha256"]
        events.append(event)
    records = transcript.get("records")
    metadata = transcript.get("metadata")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(metadata, dict) or expected_metadata is None:
        _add(errors, "ledger_transcript_shape")
        return
    record = records[0]
    expected = [
        ("audit_started", {"audit_id": audit_id, "artifact_kind": "derived_audit"}),
        ("metadata_start", expected_metadata),
        ("completion_attempted", {"episode_id": record.get("task_card", {}).get("episode_id"), "request_sha256": record.get("request_sha256")}),
        ("completion_failed", {"episode_id": record.get("task_card", {}).get("episode_id"), "request_sha256": record.get("request_sha256"), "failure_code": record.get("failure_code"), "safe_identity": record.get("safe_identity")}),
        ("metadata_end", expected_metadata | {"consistent": True}),
    ]
    for event, (kind, payload) in zip(events, expected):
        if event.get("kind") != kind or event.get("payload") != payload:
            _add(errors, "ledger_semantic_identity")
            return


def _validate_safe_identity(identity: Any) -> bool:
    return isinstance(identity, dict) and set(identity) == {"availability", "provider_declared_model", "request_id", "system_fingerprint", "provider_version"} and identity.get("availability") == "available" and identity.get("provider_declared_model") == "gpt-5.3-codex-spark" and all(isinstance(identity.get(key), str) and identity.get(key) for key in ("request_id", "system_fingerprint", "provider_version"))


def _validate_deadline_execution(execution: Any, errors: list[str]) -> None:
    """Check that all three fake I/O phases actually passed through one guard."""
    expected_stages = ("metadata_start", "completion", "metadata_end")
    if not isinstance(execution, dict) or set(execution) != {"calls", "flow_status", "runner_kind", "stages", "termination_reason"}:
        _add(errors, "deadline_execution_shape")
        return
    if execution.get("runner_kind") != "offline_fake_staged_deadline_guard" or execution.get("flow_status") != "completed" or execution.get("termination_reason") is not None:
        _add(errors, "deadline_execution_status")
        return
    calls = execution.get("calls")
    stages = execution.get("stages")
    if not isinstance(calls, list) or not isinstance(stages, list) or len(calls) != 3 or len(stages) != 3:
        _add(errors, "deadline_execution_count")
        return
    deadline_at: float | None = None
    for expected_stage, call, stage in zip(expected_stages, calls, stages):
        if not isinstance(call, dict) or set(call) != {"deadline_at_seconds", "remaining_before_seconds", "stage", "timeout_seconds"}:
            _add(errors, "deadline_execution_call_shape")
            return
        if not isinstance(stage, dict) or set(stage) != {"io_attempted", "reason", "stage", "status", "timeout_seconds"}:
            _add(errors, "deadline_execution_stage_shape")
            return
        remaining = call.get("remaining_before_seconds")
        timeout = call.get("timeout_seconds")
        current_deadline = call.get("deadline_at_seconds")
        if call.get("stage") != expected_stage or stage.get("stage") != expected_stage or stage.get("status") != "completed" or stage.get("reason") is not None or stage.get("io_attempted") is not True:
            _add(errors, "deadline_execution_order")
            return
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not isinstance(current_deadline, (int, float)) or isinstance(current_deadline, bool):
            _add(errors, "deadline_execution_timing")
            return
        if remaining <= 0 or timeout != min(60.0, float(remaining)) or stage.get("timeout_seconds") != timeout:
            _add(errors, "deadline_execution_timeout")
            return
        if deadline_at is None:
            deadline_at = float(current_deadline)
        elif float(current_deadline) != deadline_at:
            _add(errors, "deadline_execution_deadline")
            return


def _validate_transcript(transcript: Any, assets: Any, expected_metadata: dict[str, Any] | None, errors: list[str]) -> bool:
    if not isinstance(transcript, dict) or set(transcript) != {"transcript_version", "deadline_execution", "records", "metadata", "stage_summary"} or transcript.get("transcript_version") != "repair-1.1.0":
        _add(errors, "transcript_shape")
        return False
    _validate_deadline_execution(transcript.get("deadline_execution"), errors)
    records = transcript.get("records")
    metadata = transcript.get("metadata")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(metadata, dict):
        _add(errors, "transcript_records")
        return False
    record = records[0]
    expected_record_keys = {"status", "failure_code", "failure_class", "request_sha256", "safe_identity", "task_card"}
    if not isinstance(record, dict) or set(record) != expected_record_keys:
        _add(errors, "failed_record_shape")
        return False
    if record.get("status") != "failed" or record.get("failure_code") != "tool_calls" or record.get("failure_class") != "model_output" or record.get("task_card") != assets.task:
        _add(errors, "failed_record_semantics")
    if record.get("request_sha256") != request_sha256(assets.task, assets):
        _add(errors, "failed_request_hash")
    if not _validate_safe_identity(record.get("safe_identity")):
        _add(errors, "failed_safe_identity")
    expected_metadata_keys = {"start", "end", "consistent"}
    if set(metadata) != expected_metadata_keys or metadata.get("consistent") is not True or metadata.get("start") != metadata.get("end"):
        _add(errors, "metadata_identity")
    elif expected_metadata is None:
        _add(errors, "metadata_identity_oracle")
    elif metadata.get("start") != expected_metadata or metadata.get("end") != expected_metadata:
        _add(errors, "metadata_identity_oracle")
    else:
        pass
    expected_summary = {"screen": {"started": 1, "completed": 0, "failed": 1, "unstarted": 0}, "confirmation": {"started": 0, "completed": 0, "failed": 0, "unstarted": 0}}
    if transcript.get("stage_summary") != expected_summary:
        _add(errors, "stage_summary")
    return not errors


def _validate_source_contracts(assets: Any, errors: list[str]) -> None:
    contracts = _load(assets.root / "provenance" / "source_contracts.json", errors, "source_contracts_unreadable")
    expected = {path for path in FROZEN_ASSETS if path != "provenance/source_contracts.json"}
    if not isinstance(contracts, dict) or contracts.get("contract_version") != "1.0.0" or contracts.get("source_kind") != "q2b_repair_native_fixtures":
        _add(errors, "source_contracts")
        return
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != expected or any(declared.get(path) != assets.hashes[path] for path in expected):
        _add(errors, "source_contracts")


def validate_audit(audit_id: str, *, asset_root: Path = PACKAGE_ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_audit_id(audit_id)
    except ValueError:
        return {"passed": False, "audit_id": audit_id, "errors": ["invalid_audit_id"]}
    audit_dir = data_root / "audits" / audit_id
    manifest = _load(audit_dir / "manifest.json", errors, "manifest_unreadable")
    audit = _load(audit_dir / "audit.json", errors, "audit_unreadable")
    transcript = _load(audit_dir / "transcript.json", errors, "transcript_unreadable")
    outcome = _load(audit_dir / "outcome.json", errors, "outcome_unreadable")
    repair_report = _load(audit_dir / "repair_report.json", errors, "repair_report_unreadable")
    try:
        assets = load_assets(asset_root)
    except Exception:
        assets = None
        _add(errors, "assets_unreadable")
    if assets is not None:
        _validate_source_contracts(assets, errors)
    expected_metadata: dict[str, Any] | None = None
    if assets is not None:
        try:
            expected_metadata = _expected_metadata_identity(assets)
        except (TypeError, ValueError):
            _add(errors, "metadata_identity_oracle")
    if not isinstance(manifest, dict) or not isinstance(audit, dict):
        return {"passed": False, "audit_id": audit_id, "errors": errors}
    required_provenance = {
        "artifact_kind": "derived_audit",
        "is_derived_audit": True,
        "is_new_run": False,
        "derived_from_run_id": ORIGINAL_RUN_ID,
        "source_execution_mode": "remote_staged_qualification",
        "audit_id": audit_id,
        "historical_pre_rework_audit_id": HISTORICAL_PRE_REWORK_AUDIT_ID,
    }
    for key, value in required_provenance.items():
        if manifest.get(key) != value or audit.get(key) != value:
            _add(errors, "derived_provenance")
    if assets is not None and manifest.get("frozen_asset_sha256") != assets.hashes:
        _add(errors, "frozen_asset_hash")
    if manifest.get("repair_code_sha256") != source_file_hashes():
        _add(errors, "repair_code_hash")
    current_original = original_tree_hashes()
    if manifest.get("original_tree_sha256_pre") != current_original or manifest.get("original_tree_sha256_post") != current_original or audit.get("original_tree_sha256_pre") != current_original or audit.get("original_tree_sha256_post") != current_original or audit.get("original_tree_unchanged") is not True:
        _add(errors, "original_tree_changed")
    artifacts = {
        "ledger.jsonl": audit_dir / "ledger.jsonl",
        "transcript.json": audit_dir / "transcript.json",
        "outcome.json": audit_dir / "outcome.json",
        "audit.json": audit_dir / "audit.json",
        "repair_report.json": audit_dir / "repair_report.json",
        "reports/DELIVERY.md": audit_dir / "reports" / "DELIVERY.md",
    }
    declared = manifest.get("artifact_sha256")
    if not isinstance(declared, dict) or not set(artifacts).issubset(declared):
        _add(errors, "artifact_manifest")
    else:
        for name, path in artifacts.items():
            try:
                if declared[name] != sha256_file(path):
                    _add(errors, "artifact_hash_" + name)
            except OSError:
                _add(errors, "artifact_missing_" + name)
    if assets is not None:
        _validate_transcript(transcript, assets, expected_metadata, errors)
    _validate_ledger(audit_dir / "ledger.jsonl", audit_id, transcript, expected_metadata, errors)
    if not isinstance(outcome, dict) or outcome != {"repair_decision": "passed", "metadata_calls": 2, "completion_calls": 1, "retry_count": 0, "failure_record_count": 1, "identity_consistent": True}:
        _add(errors, "outcome_semantics")
    if not isinstance(repair_report, dict) or repair_report.get("repair_decision") != "passed" or repair_report.get("historical_pre_rework_audit_id") != HISTORICAL_PRE_REWORK_AUDIT_ID or repair_report.get("deadline_execution_verified") is not True or repair_report.get("network_calls") != 0 or repair_report.get("env_reads") != 0 or repair_report.get("live_completion_calls") != 0 or repair_report.get("p1_p2_started") is not False:
        _add(errors, "repair_report_semantics")
    return {"passed": not errors, "audit_id": audit_id, "errors": errors}
