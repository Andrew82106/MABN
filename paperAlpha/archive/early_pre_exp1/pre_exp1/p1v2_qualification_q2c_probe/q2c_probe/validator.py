"""Fail-closed offline validator for a single Q2-C run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    CARDS,
    TARGET_MODEL,
    profile_commitment,
    public_profile_snapshot,
    schema_version_for_run_kind,
)
from .evidence import (
    REQUIRED_ARTIFACTS,
    bounded_identity_value,
    derive_outcome,
    derive_report,
    expected_run_kind_for_id,
    read_json,
    required_artifacts_for_kind,
    run_directory,
    scan_artifacts_for_forbidden_text,
    transcript_semantic_view,
    validate_local_delivery_text,
    verify_ledger,
    verify_manifest,
)
from .historical_integrity import historical_tree_snapshot
from .live_runner import (
    LIVE_GUARD_NAME,
    live_config_snapshot,
    live_execution_source_hashes,
    live_policy_commitment,
    live_runner_source_sha256,
    live_source_bundle_sha256,
    live_transport_source_sha256,
)
from .runner import FATAL_OUTCOMES, expected_fake_semantic_view


ALLOWED_OUTCOMES = {
    "content_match",
    "response_contract_mismatch",
    "non_json",
    "field_mismatch",
    "tool_calls",
    "function_call",
    "reasoning_detected",
    "refusal",
    "empty_content",
    "content_too_long",
    "malformed_envelope",
    "model_mismatch",
    "identity_missing",
    "unsafe_provider_identity",
    "transport_error",
    "timeout",
    "http_error",
    "connection_error",
    "redirect",
    "deadline_exhausted",
    "unstarted",
}


def _safe_identity(identity: Any, expected_target: bool) -> bool:
    if not isinstance(identity, dict):
        return False
    expected_keys = {
        "declared_model",
        "request_id",
        "system_fingerprint",
        "provider_version",
    }
    if set(identity) != expected_keys:
        return False
    for value in identity.values():
        if value is not None and bounded_identity_value(value) != value:
            return False
    if expected_target and identity.get("declared_model") != TARGET_MODEL:
        return False
    return True


def _validate_transcript(transcript: Any, run_id: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(transcript, dict):
        return ["transcript_not_object"]
    if transcript.get("run_id") != run_id:
        errors.append("transcript_run_id")
    run_kind = transcript.get("run_kind")
    if run_kind not in {"offline_fake", "live_probe"}:
        errors.append("transcript_run_kind")
    else:
        if transcript.get("schema_version") != schema_version_for_run_kind(run_kind):
            errors.append("transcript_schema")
        if transcript.get("profile_commitment") != profile_commitment(run_kind):
            errors.append("transcript_profile_commitment")
    metadata = transcript.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata_shape")
    else:
        required_metadata = {
            "status",
            "target_model_present",
            "model_count",
            "model_list_sha256",
            "stop_reason",
            "timeout_seconds",
            "elapsed_seconds",
        }
        if set(metadata) != required_metadata:
            errors.append("metadata_fields")
        if metadata.get("status") == "accepted":
            if metadata.get("target_model_present") is not True:
                errors.append("metadata_target_presence")
            if not isinstance(metadata.get("model_count"), int) or metadata["model_count"] < 1:
                errors.append("metadata_count")
            digest = metadata.get("model_list_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append("metadata_hash")
        elif metadata.get("target_model_present") is True:
            errors.append("metadata_status_target_conflict")
    records = transcript.get("records")
    if not isinstance(records, list) or len(records) != len(CARDS):
        return errors + ["record_count"]
    seen_unstarted = False
    seen_fatal = False
    started_count = 0
    for sequence, (card, record) in enumerate(zip(CARDS, records), start=1):
        if not isinstance(record, dict):
            errors.append("record_shape")
            continue
        if (
            record.get("sequence") != sequence
            or record.get("card_id") != card.card_id
            or record.get("profile") != card.profile
            or record.get("request_sha256") != card.request_sha256
        ):
            errors.append("record_frozen_profile")
        started = record.get("started")
        category = record.get("outcome_category")
        if not isinstance(started, bool) or category not in ALLOWED_OUTCOMES:
            errors.append("record_status")
            continue
        if not started:
            seen_unstarted = True
            if category != "unstarted" or record.get("provider_identity") is not None:
                errors.append("unstarted_record")
        else:
            started_count += 1
            if seen_unstarted or seen_fatal or category == "unstarted":
                errors.append("record_state_machine")
            if category in {"transport_error", "timeout", "http_error", "connection_error", "redirect", "deadline_exhausted"}:
                if record.get("provider_identity") is not None:
                    errors.append("transport_identity")
            else:
                expected_target = category != "model_mismatch"
                if not _safe_identity(record.get("provider_identity"), expected_target):
                    errors.append("record_identity")
            if category in FATAL_OUTCOMES:
                seen_fatal = True
        content_length = record.get("content_length")
        content_hash = record.get("content_sha256")
        if content_length is None:
            if content_hash is not None:
                errors.append("content_hash_without_content")
        elif not isinstance(content_length, int) or content_length < 0:
            errors.append("content_length")
        elif not isinstance(content_hash, str) or len(content_hash) != 64:
            errors.append("content_hash")
        if not isinstance(record.get("expected_fields_satisfied"), bool):
            errors.append("record_expected_flag")
    metadata_status = metadata.get("status") if isinstance(metadata, dict) else None
    if metadata_status != "accepted" and started_count != 0:
        errors.append("metadata_gate")
    expected_calls = (1 if metadata_status != "unstarted" else 0) + started_count
    if transcript.get("transport_call_count") != expected_calls:
        errors.append("transport_call_count")
    if not isinstance(transcript.get("total_deadline_seconds"), (int, float)):
        errors.append("deadline_budget")
    if run_kind == "live_probe":
        required_live_fields = {
            "live_policy_commitment",
            "live_runner_source_sha256",
            "live_transport_source_sha256",
            "metadata_call_count",
            "completion_call_count",
        }
        if not required_live_fields.issubset(transcript):
            errors.append("live_transcript_fields")
        if transcript.get("live_policy_commitment") != live_policy_commitment():
            errors.append("live_policy_commitment")
        if transcript.get("live_runner_source_sha256") != live_runner_source_sha256():
            errors.append("live_runner_source")
        if transcript.get("live_transport_source_sha256") != live_transport_source_sha256():
            errors.append("live_transport_source")
        if type(transcript.get("metadata_call_count")) is not int or transcript.get("metadata_call_count") not in {0, 1}:
            errors.append("live_metadata_count")
        if type(transcript.get("completion_call_count")) is not int or not 0 <= transcript["completion_call_count"] <= 4:
            errors.append("live_completion_count")
        if transcript.get("metadata_call_count", -1) + transcript.get("completion_call_count", -1) != transcript.get("transport_call_count"):
            errors.append("live_transport_counts")
    return errors


def _validate_config(config: Any, run_kind: str) -> list[str]:
    if run_kind == "live_probe":
        return [] if config == live_config_snapshot() else ["live_config_snapshot"]
    expected = {
        "schema_version": schema_version_for_run_kind("offline_fake"),
        "offline_only": True,
        "profile_commitment": profile_commitment("offline_fake"),
        "public_profile": public_profile_snapshot("offline_fake"),
    }
    return [] if config == expected else ["config_snapshot"]


def _validate_live_guard(data_root: Path, run_dir: Path, run_id: str) -> list[str]:
    try:
        local_guard = read_json(run_dir / LIVE_GUARD_NAME)
        root_guard = read_json(data_root / LIVE_GUARD_NAME)
    except (OSError, ValueError):
        return ["live_guard_read"]
    expected = {
        "schema_version": schema_version_for_run_kind("live_probe"),
        "run_id": run_id,
        "state": "reserved",
        "live_policy_commitment": live_policy_commitment(),
    }
    if local_guard != expected or root_guard != expected:
        return ["live_guard_semantics"]
    return []


def _validate_history(history: Any) -> list[str]:
    if not isinstance(history, dict) or history.get("unchanged") is not True:
        return ["historical_integrity_shape"]
    before = history.get("before")
    after = history.get("after")
    if before != after:
        return ["historical_integrity_record"]
    try:
        current = historical_tree_snapshot()
    except (OSError, ValueError):
        return ["historical_integrity_read"]
    return [] if current == before else ["historical_tree_changed"]


def validate_run(data_root: Path, run_id: str) -> list[str]:
    """Validate artifacts without network, credentials, or historical imports."""
    try:
        run_dir = run_directory(data_root, run_id)
    except (OSError, ValueError):
        return ["run_path"]
    if not run_dir.is_dir() or run_dir.is_symlink():
        return ["run_directory"]
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return ["manifest_missing"]
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
        config = read_json(run_dir / "config_snapshot.json")
        transcript = read_json(run_dir / "transcript.json")
        ledger = read_json(run_dir / "ledger.json")
        outcome = read_json(run_dir / "outcome.json")
        report = read_json(run_dir / "report.json")
        history = read_json(run_dir / "historical_integrity.json")
    except (OSError, ValueError):
        return ["artifact_parse"]
    run_kind = transcript.get("run_kind") if isinstance(transcript, dict) else None
    if run_kind not in {"offline_fake", "live_probe"}:
        return ["transcript_run_kind"]
    try:
        if expected_run_kind_for_id(run_id) != run_kind:
            return ["run_id_kind_mismatch"]
    except ValueError:
        return ["run_id_kind_mismatch"]
    errors.extend(verify_manifest(run_dir, manifest))
    if manifest.get("run_id") != run_id or manifest.get("run_kind") != run_kind:
        errors.append("manifest_identity")
    if manifest.get("schema_version") != schema_version_for_run_kind(run_kind):
        errors.append("manifest_schema")
    if manifest.get("profile_commitment") != profile_commitment(run_kind):
        errors.append("manifest_profile_commitment")
    artifact_paths = [run_dir / relative for relative in required_artifacts_for_kind(run_kind)]
    errors.extend(scan_artifacts_for_forbidden_text(artifact_paths))
    errors.extend(_validate_config(config, run_kind))
    errors.extend(_validate_transcript(transcript, run_id))
    errors.extend(verify_ledger(transcript, ledger))
    try:
        expected_outcome = derive_outcome(transcript)
        expected_report = derive_report(transcript, expected_outcome)
    except (KeyError, TypeError, ValueError):
        errors.append("run_kind_artifact_semantics")
    else:
        if outcome != expected_outcome:
            errors.append("outcome_semantics")
        if report != expected_report:
            errors.append("report_semantics")
    expected_report_kind = "offline_fake_transport_report" if run_kind == "offline_fake" else "live_probe_safe_summary"
    expected_offline_only = run_kind == "offline_fake"
    if (
        not isinstance(outcome, dict)
        or outcome.get("run_kind") != run_kind
        or not isinstance(report, dict)
        or report.get("report_kind") != expected_report_kind
        or report.get("offline_only") is not expected_offline_only
    ):
        errors.append("run_kind_artifact_identity")
    errors.extend(_validate_history(history))
    if run_kind == "offline_fake" and transcript_semantic_view(transcript) != expected_fake_semantic_view():
        errors.append("fake_oracle_semantics")
    if run_kind == "live_probe":
        expected_manifest_fields = {
            "live_policy_commitment": live_policy_commitment(),
            "live_runner_source_sha256": live_runner_source_sha256(),
            "live_transport_source_sha256": live_transport_source_sha256(),
            "live_source_bundle_sha256": live_source_bundle_sha256(),
            "live_execution_source_hashes": live_execution_source_hashes(),
            "metadata_call_count": transcript.get("metadata_call_count"),
            "completion_call_count": transcript.get("completion_call_count"),
        }
        for key, value in expected_manifest_fields.items():
            if manifest.get(key) != value:
                errors.append("live_manifest_binding")
        errors.extend(_validate_live_guard(data_root, run_dir, run_id))
    try:
        delivery_text = (run_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
    except OSError:
        errors.append("local_delivery_read")
    else:
        for delivery_error in validate_local_delivery_text(delivery_text, run_id, outcome):
            errors.append(f"local_delivery_{delivery_error}")
    return sorted(set(errors))
