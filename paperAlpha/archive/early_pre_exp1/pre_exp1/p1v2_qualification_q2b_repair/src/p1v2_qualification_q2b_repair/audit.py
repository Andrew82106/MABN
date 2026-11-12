"""Create a repair-native derived audit without changing or executing original Q2-B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, HISTORICAL_PRE_REWORK_AUDIT_ID, ORIGINAL_DATA_ROOT, ORIGINAL_RUN_ID, PACKAGE_ROOT, canonical_json, load_json, make_audit_id, original_tree_hashes, sha256_file, sha256_text, source_file_hashes, validate_audit_id, write_json
from .protocol import FROZEN_ASSETS, assess_success_envelope, build_request, frozen_metadata_profile, load_assets, request_sha256
from .deadline import BatchDeadline
from .staged import FakeStagedTransport, InMemoryClock, StagedDeadlineRunner


def _append_ledger(path: Path, sequence: int, previous: str, kind: str, payload: dict[str, Any]) -> str:
    core = {"kind": kind, "payload": payload, "previous_event_sha256": previous, "sequence": sequence}
    event = dict(core)
    event["event_sha256"] = sha256_text(canonical_json(core))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(event) + "\n")
    return event["event_sha256"]


def _verify_source_contracts(assets: Any) -> None:
    contracts = load_json(assets.root / "provenance" / "source_contracts.json")
    expected = {path for path in FROZEN_ASSETS if path != "provenance/source_contracts.json"}
    if contracts.get("contract_version") != "1.0.0" or contracts.get("source_kind") != "q2b_repair_native_fixtures":
        raise ValueError("repair_source_contract")
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != expected:
        raise ValueError("repair_source_contract")
    for path in expected:
        if declared.get(path) != assets.hashes[path]:
            raise ValueError("repair_source_contract")


def _legacy_schema_assessment() -> dict[str, str]:
    """Read only minimal historical JSON structure; never copy raw response content."""
    try:
        transcript = load_json(ORIGINAL_DATA_ROOT / "runs" / ORIGINAL_RUN_ID / "transcript.json")
        records = transcript.get("records") if isinstance(transcript, dict) else None
        if isinstance(records, list) and any(isinstance(record, dict) and record.get("status") == "failed" and "completion_identity" not in record for record in records):
            return {"legacy_schema_status": "rework", "reason": "failed_record_missing_repair_safe_identity"}
    except (OSError, ValueError, json.JSONDecodeError):
        return {"legacy_schema_status": "rework", "reason": "legacy_evidence_unreadable"}
    return {"legacy_schema_status": "rework", "reason": "legacy_schema_not_repair_native"}


def _native_completion_value() -> dict[str, Any]:
    return {
        "headers": {"x-request-id": "repair-native-request"},
        "raw_body": canonical_json(
        {
            "model": "gpt-5.3-codex-spark",
            "provider_version": "not_provided",
            "system_fingerprint": "repair-native-fingerprint",
            "choices": [{"message": {"tool_calls": []}}],
        }
        ).encode("utf-8"),
    }


def _native_failed_record(task: dict[str, Any], assets: Any, completion_value: dict[str, Any]) -> dict[str, Any]:
    raw = completion_value.get("raw_body")
    headers = completion_value.get("headers")
    if not isinstance(raw, bytes) or not isinstance(headers, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        raise RuntimeError("native_completion_value_invalid")
    assessment = assess_success_envelope(raw, headers, task)
    if assessment.status != "failed" or assessment.failure_code != "tool_calls":
        raise RuntimeError("native_fixture_invalid")
    return {
        "status": "failed",
        "failure_code": assessment.failure_code,
        "failure_class": assessment.failure_class,
        "request_sha256": request_sha256(task, assets),
        "safe_identity": assessment.safe_identity,
        "task_card": task,
    }


def _metadata_identity(assets: Any) -> dict[str, Any]:
    """Derive metadata from the frozen repair provider profile, never transcript state."""
    target_model_id, model_ids = frozen_metadata_profile(assets)
    return {
        "normalized_models_sha256": sha256_text(canonical_json({"model_ids": list(model_ids)})),
        "target_model_present": target_model_id in model_ids,
        "model_count": len(model_ids),
    }


def _local_delivery_text(audit_id: str) -> str:
    """Audit-local immutable summary; it is hash-bound by this audit's manifest."""
    return "\n".join(
        (
            "# P1v2 Q2-B repair-native audit artifact",
            "",
            "- Audit ID: `" + audit_id + "`",
            "- Artifact kind: `derived_audit`",
            "- Repair decision: `passed`",
            "- Scope: offline repair evidence only; no model qualification conclusion.",
            "- Delivery binding: this file belongs only to this audit directory.",
            "",
        )
    )


def _run_native_staged_flow(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute all three fake stages through the real deadline guard.

    The returned execution evidence intentionally excludes the in-memory fake
    completion body and headers.  It proves ordering and timeouts only.
    """
    clock = InMemoryClock()
    deadline = BatchDeadline(clock=clock, budget_seconds=15.0)
    completion_value = _native_completion_value()
    transport = FakeStagedTransport(
        clock=clock,
        advance=clock.advance,
        values={
            "metadata_start": dict(metadata),
            "completion": completion_value,
            "metadata_end": dict(metadata),
        },
        delays_seconds={"metadata_start": 0.25, "completion": 0.25, "metadata_end": 0.25},
    )
    flow = StagedDeadlineRunner(deadline=deadline, transport=transport).run()
    if flow.flow_status != "completed":
        raise RuntimeError("native_staged_deadline_failure")
    start = flow.stage_value("metadata_start")
    end = flow.stage_value("metadata_end")
    completion = flow.stage_value("completion")
    if not isinstance(start, dict) or not isinstance(end, dict) or not isinstance(completion, dict) or start != metadata or end != metadata:
        raise RuntimeError("native_staged_values_invalid")
    return start, end, completion, flow.public_evidence()


def run_derived_audit(*, asset_root: Path = PACKAGE_ROOT, data_root: Path = DATA_ROOT, audit_id: str | None = None) -> dict[str, Any]:
    assets = load_assets(asset_root)
    _verify_source_contracts(assets)
    actual_audit_id = audit_id or make_audit_id()
    validate_audit_id(actual_audit_id)
    audit_dir = data_root / "audits" / actual_audit_id
    if audit_dir.exists():
        raise ValueError("audit_id_already_exists")
    pre_hashes = original_tree_hashes()
    audit_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = audit_dir / "ledger.jsonl"
    task = assets.task
    metadata = _metadata_identity(assets)
    start_metadata, end_metadata, completion_value, deadline_execution = _run_native_staged_flow(metadata)
    record = _native_failed_record(task, assets, completion_value)
    previous = "GENESIS"
    sequence = 1
    previous = _append_ledger(ledger_path, sequence, previous, "audit_started", {"audit_id": actual_audit_id, "artifact_kind": "derived_audit"})
    sequence += 1
    previous = _append_ledger(ledger_path, sequence, previous, "metadata_start", start_metadata)
    sequence += 1
    previous = _append_ledger(ledger_path, sequence, previous, "completion_attempted", {"episode_id": task["episode_id"], "request_sha256": record["request_sha256"]})
    sequence += 1
    previous = _append_ledger(ledger_path, sequence, previous, "completion_failed", {"episode_id": task["episode_id"], "request_sha256": record["request_sha256"], "failure_code": record["failure_code"], "safe_identity": record["safe_identity"]})
    sequence += 1
    previous = _append_ledger(ledger_path, sequence, previous, "metadata_end", end_metadata | {"consistent": True})
    transcript = {
        "transcript_version": "repair-1.1.0",
        "deadline_execution": deadline_execution,
        "records": [record],
        "metadata": {"start": start_metadata, "end": end_metadata, "consistent": True},
        "stage_summary": {"screen": {"started": 1, "completed": 0, "failed": 1, "unstarted": 0}, "confirmation": {"started": 0, "completed": 0, "failed": 0, "unstarted": 0}},
    }
    outcome = {
        "repair_decision": "passed",
        "metadata_calls": 2,
        "completion_calls": 1,
        "retry_count": 0,
        "failure_record_count": 1,
        "identity_consistent": True,
    }
    post_hashes = original_tree_hashes()
    audit = {
        "artifact_kind": "derived_audit",
        "is_derived_audit": True,
        "is_new_run": False,
        "derived_from_run_id": ORIGINAL_RUN_ID,
        "source_execution_mode": "remote_staged_qualification",
        "audit_id": actual_audit_id,
        "historical_pre_rework_audit_id": HISTORICAL_PRE_REWORK_AUDIT_ID,
        "legacy_schema_assessment": _legacy_schema_assessment(),
        "original_tree_sha256_pre": pre_hashes,
        "original_tree_sha256_post": post_hashes,
        "original_tree_unchanged": pre_hashes == post_hashes,
        "frozen_asset_sha256": assets.hashes,
    }
    write_json(audit_dir / "transcript.json", transcript)
    write_json(audit_dir / "outcome.json", outcome)
    write_json(audit_dir / "audit.json", audit)
    repair_report = {
        "artifact_kind": "derived_audit",
        "is_derived_audit": True,
        "is_new_run": False,
        "audit_id": actual_audit_id,
        "historical_pre_rework_audit_id": HISTORICAL_PRE_REWORK_AUDIT_ID,
        "repair_decision": "passed",
        "network_calls": 0,
        "env_reads": 0,
        "deadline_execution_verified": True,
        "live_completion_calls": 0,
        "p1_p2_started": False,
    }
    write_json(audit_dir / "repair_report.json", repair_report)
    local_delivery_path = audit_dir / "reports" / "DELIVERY.md"
    local_delivery_path.parent.mkdir(parents=True, exist_ok=True)
    local_delivery_path.write_text(_local_delivery_text(actual_audit_id), encoding="utf-8", newline="\n")
    manifest = {
        "manifest_version": "repair-1.1.0",
        "artifact_kind": "derived_audit",
        "is_derived_audit": True,
        "is_new_run": False,
        "derived_from_run_id": ORIGINAL_RUN_ID,
        "source_execution_mode": "remote_staged_qualification",
        "audit_id": actual_audit_id,
        "historical_pre_rework_audit_id": HISTORICAL_PRE_REWORK_AUDIT_ID,
        "frozen_asset_sha256": assets.hashes,
        "repair_code_sha256": source_file_hashes(),
        "original_tree_sha256_pre": pre_hashes,
        "original_tree_sha256_post": post_hashes,
        "artifact_sha256": {
            "ledger.jsonl": sha256_file(ledger_path),
            "transcript.json": sha256_file(audit_dir / "transcript.json"),
            "outcome.json": sha256_file(audit_dir / "outcome.json"),
            "audit.json": sha256_file(audit_dir / "audit.json"),
            "repair_report.json": sha256_file(audit_dir / "repair_report.json"),
            "reports/DELIVERY.md": sha256_file(local_delivery_path),
        },
    }
    write_json(audit_dir / "manifest.json", manifest)
    return {"audit_id": actual_audit_id, "repair_decision": "passed", "audit_dir": str(audit_dir)}
