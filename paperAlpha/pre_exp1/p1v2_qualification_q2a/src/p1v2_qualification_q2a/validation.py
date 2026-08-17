"""Independent, read-only validation of a Q2-A evidence package.

This module has no execution-client imports and performs no network or environment
access.  It validates only persisted files and frozen local assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, PACKAGE_ROOT, canonical_json, load_json, relative_file_hashes, sha256_file, sha256_text, validate_run_id
from .protocol import FROZEN_ASSET_PATHS, RESPONSE_FIELDS, load_frozen_assets, validate_task_card, verify_identity_profile


def _error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def _current_code_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "src", (".py",)).items():
        result["src/" + relative] = digest
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "scripts", (".py",)).items():
        result["scripts/" + relative] = digest
    return result


def _load(path: Path, errors: list[str], code: str) -> Any | None:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _error(errors, code)
        return None


def _validate_source_contracts(asset_root: Path, assets: Any, errors: list[str]) -> None:
    contracts = _load(asset_root / "provenance" / "source_contracts.json", errors, "source_contracts_unreadable")
    if not isinstance(contracts, dict):
        return
    expected_paths = {path for path in FROZEN_ASSET_PATHS if path != "provenance/source_contracts.json"}
    if contracts.get("contract_version") != "1.0.0":
        _error(errors, "source_contracts_version")
    if contracts.get("source_kind") != "q2a_native_frozen_inputs":
        _error(errors, "source_contracts_kind")
    if contracts.get("historical_runtime_imports") != []:
        _error(errors, "source_contracts_history")
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != expected_paths:
        _error(errors, "source_contracts_paths")
        return
    for relative in expected_paths:
        if declared.get(relative) != assets.asset_hashes.get(relative):
            _error(errors, "source_contracts_hash")


def _validate_ledger(path: Path, run_id: str, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _error(errors, "ledger_unreadable")
        return
    if not lines:
        _error(errors, "ledger_empty")
        return
    previous_hash = "GENESIS"
    for expected_sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            _error(errors, "ledger_json")
            return
        if not isinstance(event, dict) or set(event) != {"kind", "payload", "previous_event_sha256", "sequence", "event_sha256"}:
            _error(errors, "ledger_shape")
            return
        core = {key: event[key] for key in ("kind", "payload", "previous_event_sha256", "sequence")}
        if event["sequence"] != expected_sequence or event["previous_event_sha256"] != previous_hash:
            _error(errors, "ledger_chain")
            return
        if event["event_sha256"] != sha256_text(canonical_json(core)):
            _error(errors, "ledger_chain")
            return
        if expected_sequence == 1:
            payload = event["payload"]
            if event["kind"] != "run_started" or not isinstance(payload, dict) or payload.get("run_id") != run_id:
                _error(errors, "ledger_run_id")
        previous_hash = event["event_sha256"]


def _response_matches_task(response: Any, task: dict[str, Any]) -> bool:
    if not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS):
        return False
    return response == {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }


def _validate_transcript(transcript: Any, assets: Any, errors: list[str]) -> dict[str, Any]:
    facts = {"screen_ok": False, "all_records_completed": False, "identities_ok": False, "attempted": 0}
    if not isinstance(transcript, dict):
        _error(errors, "transcript_shape")
        return facts
    records = transcript.get("records")
    if not isinstance(records, list) or len(records) != 128:
        _error(errors, "transcript_record_count")
        return facts
    expected_tasks = assets.screen_tasks + assets.confirmation_tasks
    expected_by_id = {task["episode_id"]: task for task in expected_tasks}
    seen_ids: set[str] = set()
    screen_records: list[dict[str, Any]] = []
    confirmation_records: list[dict[str, Any]] = []
    sink_records: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("task_card"), dict):
            _error(errors, "transcript_record_shape")
            continue
        task = record["task_card"]
        try:
            validate_task_card(task)
        except Exception:
            _error(errors, "transcript_task_card")
            continue
        episode_id = task["episode_id"]
        if episode_id in seen_ids or expected_by_id.get(episode_id) != task:
            _error(errors, "transcript_task_overlap")
            continue
        seen_ids.add(episode_id)
        status = record.get("status")
        if status not in {"completed", "failed", "unstarted"}:
            _error(errors, "transcript_status")
            continue
        if status == "completed":
            facts["attempted"] += 1
            if not _response_matches_task(record.get("response"), task):
                _error(errors, "transcript_response")
            if not isinstance(record.get("request_sha256"), str) or not isinstance(record.get("response_content_sha256"), str):
                _error(errors, "transcript_hash")
            if task["agent_role"] == "publisher":
                if record.get("public_sink_record") != record.get("response"):
                    _error(errors, "public_sink_publisher")
                else:
                    sink_records.append(record["public_sink_record"])
            elif "public_sink_record" in record:
                _error(errors, "public_sink_coordinator")
        elif status == "failed":
            facts["attempted"] += 1
            if not isinstance(record.get("failure_code"), str) or "response" in record or "public_sink_record" in record:
                _error(errors, "failed_record_safety")
        else:
            if "public_sink_record" in record or "response" in record:
                _error(errors, "unstarted_record_safety")
            if not isinstance(record.get("reason"), str):
                _error(errors, "unstarted_record_reason")
        (screen_records if task["phase"] == "screen" else confirmation_records).append(record)
    if seen_ids != set(expected_by_id):
        _error(errors, "transcript_episodes")
    if len(screen_records) != 32 or len(confirmation_records) != 96:
        _error(errors, "transcript_phase_count")
    screen_ok = all(record.get("status") == "completed" for record in screen_records) and len(screen_records) == 32
    if transcript.get("screen_ok") is not screen_ok:
        _error(errors, "screen_state")
    if screen_ok:
        if transcript.get("confirmation_started") is not True:
            _error(errors, "confirmation_state")
    else:
        if transcript.get("confirmation_started") is not False or any(record.get("status") != "unstarted" for record in confirmation_records):
            _error(errors, "confirmation_blocking")
    if transcript.get("public_sink_records") != sink_records:
        _error(errors, "public_sink_ledger")
    identities = transcript.get("identity_evidence")
    identities_ok = False
    if isinstance(identities, dict):
        preflight = identities.get("preflight")
        postflight = identities.get("postflight")
        if isinstance(preflight, dict) and isinstance(postflight, dict) and preflight.get("status") == "accepted" and postflight.get("status") == "accepted":
            try:
                verify_identity_profile(preflight["identity"], assets)
                verify_identity_profile(postflight["identity"], assets)
                identities_ok = True
            except Exception:
                _error(errors, "identity_evidence")
        elif not isinstance(preflight, dict) or not isinstance(postflight, dict):
            _error(errors, "identity_evidence")
    else:
        _error(errors, "identity_evidence")
    facts["screen_ok"] = screen_ok
    facts["all_records_completed"] = all(record.get("status") == "completed" for record in records)
    facts["identities_ok"] = identities_ok
    return facts


def validate_run(run_id: str, *, asset_root: Path = PACKAGE_ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_run_id(run_id)
    except ValueError:
        return {"passed": False, "run_id": run_id, "errors": ["invalid_run_id"]}
    run_dir = data_root / "runs" / run_id
    manifest = _load(run_dir / "manifest.json", errors, "manifest_unreadable")
    outcome = _load(run_dir / "outcome.json", errors, "outcome_unreadable")
    transcript = _load(run_dir / "transcript.json", errors, "transcript_unreadable")
    report = _load(run_dir / "readiness_report.json", errors, "report_unreadable")
    try:
        assets = load_frozen_assets(asset_root)
    except Exception:
        assets = None
        _error(errors, "frozen_assets_unreadable")
    if assets is not None:
        _validate_source_contracts(asset_root, assets, errors)
    if not isinstance(manifest, dict):
        return {"passed": False, "run_id": run_id, "errors": errors}
    if manifest.get("run_id") != run_id:
        _error(errors, "manifest_run_id")
    if assets is not None and manifest.get("frozen_asset_sha256") != assets.asset_hashes:
        _error(errors, "frozen_asset_hash")
    if manifest.get("code_provenance_sha256") != _current_code_provenance():
        _error(errors, "code_provenance_hash")
    artifact_paths = {
        "ledger.jsonl": run_dir / "ledger.jsonl",
        "transcript.json": run_dir / "transcript.json",
        "outcome.json": run_dir / "outcome.json",
        "readiness_report.json": run_dir / "readiness_report.json",
        "reports/DELIVERY.md": data_root / "reports" / "DELIVERY.md",
    }
    declared_hashes = manifest.get("artifact_sha256")
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != set(artifact_paths):
        _error(errors, "manifest_artifact_paths")
    else:
        for relative, path in artifact_paths.items():
            try:
                if declared_hashes[relative] != sha256_file(path):
                    _error(errors, "artifact_hash_" + relative.replace("/", "_"))
            except OSError:
                _error(errors, "artifact_missing_" + relative.replace("/", "_"))
    _validate_ledger(run_dir / "ledger.jsonl", run_id, errors)
    facts = _validate_transcript(transcript, assets, errors) if assets is not None else {"screen_ok": False, "all_records_completed": False, "identities_ok": False, "attempted": 0}
    if not isinstance(outcome, dict):
        _error(errors, "outcome_shape")
    else:
        if outcome.get("run_id") != run_id or manifest.get("decision") != outcome.get("decision"):
            _error(errors, "outcome_run_id")
        decision = outcome.get("decision")
        if decision not in {"ready_for_q2b", "rework"}:
            _error(errors, "outcome_decision")
        expected_ready = facts["screen_ok"] and facts["all_records_completed"] and facts["identities_ok"]
        if (decision == "ready_for_q2b") != expected_ready:
            _error(errors, "outcome_decision_logic")
        counts = outcome.get("call_counts")
        if not isinstance(counts, dict):
            _error(errors, "outcome_counts")
        else:
            for key in ("real_network_calls", "real_model_calls", "real_credential_reads", "real_environment_reads"):
                if counts.get(key) != 0:
                    _error(errors, "real_io_count_" + key)
            if counts.get("mock_inference_calls") != facts["attempted"]:
                _error(errors, "mock_inference_count")
            if counts.get("mock_transport_calls") != counts.get("mock_inference_calls", -1) + counts.get("mock_identity_checks", -1):
                _error(errors, "mock_transport_count")
            if counts.get("mock_total_events") != counts.get("mock_transport_calls"):
                _error(errors, "mock_total_count")
    if not isinstance(report, dict):
        _error(errors, "report_shape")
    else:
        if report.get("run_id") != run_id or report.get("decision") != manifest.get("decision"):
            _error(errors, "report_run_id")
        if report.get("real_credential_read") != "no" or report.get("real_network_call_count") != 0 or report.get("real_model_call_count") != 0 or report.get("real_environment_read_count") != 0:
            _error(errors, "report_real_io")
        if report.get("new_dependencies") != [] or report.get("history_runtime_imports") != [] or report.get("live_execution_started") is not False:
            _error(errors, "report_boundary")
    try:
        delivery = (data_root / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
        required_delivery_literals = (
            "Run ID: " + run_id,
            "Engineering decision: " + str(manifest.get("decision")),
            "Real network calls: 0",
            "Real model calls: 0",
            "Real credential reads: no",
            "New dependencies: none",
            "No real Q2-B, P1, or P2 execution was started.",
        )
        if any(literal not in delivery for literal in required_delivery_literals):
            _error(errors, "delivery_report_content")
    except OSError:
        _error(errors, "delivery_report_unreadable")
    return {"passed": not errors, "run_id": run_id, "errors": errors}
