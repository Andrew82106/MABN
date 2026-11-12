"""Independent, read-only Q2-B evidence validator with no live-client dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DATA_ROOT, PACKAGE_ROOT, canonical_json, load_json, relative_file_hashes, sha256_file, sha256_text, validate_run_id
from .protocol import FROZEN_ASSET_PATHS, RESPONSE_FIELDS, build_request, load_frozen_assets, validate_task_card


def _add(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def _load(path: Path, errors: list[str], code: str) -> Any | None:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        _add(errors, code)
        return None


def _current_code_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "src", (".py",)).items():
        result["src/" + relative] = digest
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "scripts", (".py",)).items():
        result["scripts/" + relative] = digest
    return result


def _validate_source_contracts(asset_root: Path, assets: Any, errors: list[str]) -> None:
    contracts = _load(asset_root / "provenance" / "source_contracts.json", errors, "source_contracts_unreadable")
    if not isinstance(contracts, dict):
        return
    expected_paths = {path for path in FROZEN_ASSET_PATHS if path != "provenance/source_contracts.json"}
    if contracts.get("contract_version") != "1.0.0" or contracts.get("source_kind") != "q2b_native_frozen_inputs" or contracts.get("historical_runtime_imports") != []:
        _add(errors, "source_contracts_contract")
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != expected_paths:
        _add(errors, "source_contracts_paths")
        return
    for relative in expected_paths:
        if declared.get(relative) != assets.asset_hashes.get(relative):
            _add(errors, "source_contracts_hash")


def _validate_ledger(path: Path, run_id: str, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _add(errors, "ledger_unreadable")
        return
    if not lines:
        _add(errors, "ledger_empty")
        return
    previous = "GENESIS"
    for sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            _add(errors, "ledger_json")
            return
        expected_keys = {"kind", "payload", "previous_event_sha256", "sequence", "event_sha256"}
        if not isinstance(event, dict) or set(event) != expected_keys:
            _add(errors, "ledger_shape")
            return
        core = {key: event[key] for key in ("kind", "payload", "previous_event_sha256", "sequence")}
        if event["sequence"] != sequence or event["previous_event_sha256"] != previous or event["event_sha256"] != sha256_text(canonical_json(core)):
            _add(errors, "ledger_chain")
            return
        if sequence == 1 and (event["kind"] != "run_started" or not isinstance(event["payload"], dict) or event["payload"].get("run_id") != run_id):
            _add(errors, "ledger_run_id")
        previous = event["event_sha256"]


def _expected_response(task: dict[str, Any]) -> dict[str, str]:
    return {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }


def _expected_stage(task: dict[str, Any]) -> str:
    if task["phase"] == "confirmation":
        return "confirmation"
    return "screen_gate" if int(task["episode_id"].rsplit("-", 1)[1]) <= 8 else "screen_remainder"


def _validate_transcript(transcript: Any, assets: Any, errors: list[str]) -> dict[str, Any]:
    facts = {"all_completed": False, "model_failure_codes": set(), "infrastructure_record_failure": False, "attempted": 0, "stage_summary": {}}
    if not isinstance(transcript, dict) or not isinstance(transcript.get("records"), list):
        _add(errors, "transcript_shape")
        return facts
    records = transcript["records"]
    expected_tasks = assets.screen_tasks + assets.confirmation_tasks
    expected_by_id = {task["episode_id"]: task for task in expected_tasks}
    if len(records) != len(expected_tasks):
        _add(errors, "transcript_record_count")
        return facts
    seen: set[str] = set()
    public_sink_records: list[dict[str, str]] = []
    by_stage: dict[str, list[dict[str, Any]]] = {"screen_gate": [], "screen_remainder": [], "confirmation": []}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("task_card"), dict):
            _add(errors, "transcript_record_shape")
            continue
        task = record["task_card"]
        try:
            validate_task_card(task)
        except Exception:
            _add(errors, "transcript_task_card")
            continue
        task_id = task["episode_id"]
        if task_id in seen or expected_by_id.get(task_id) != task:
            _add(errors, "transcript_task_overlap")
            continue
        seen.add(task_id)
        stage = _expected_stage(task)
        if record.get("stage") != stage:
            _add(errors, "transcript_stage")
        by_stage[stage].append(record)
        status = record.get("status")
        if status == "completed":
            facts["attempted"] += 1
            if record.get("response") != _expected_response(task) or not isinstance(record.get("response_content_sha256"), str):
                _add(errors, "transcript_response")
            expected_hash = sha256_text(canonical_json(build_request(task, assets)))
            if record.get("request_sha256") != expected_hash:
                _add(errors, "transcript_request_hash")
            identity = record.get("completion_identity")
            if not isinstance(identity, dict) or set(identity) != {"provider_declared_model", "request_id", "system_fingerprint", "provider_version"} or identity.get("provider_declared_model") != assets.protocol["model"]:
                _add(errors, "completion_identity")
            if task["agent_role"] == "publisher":
                if record.get("public_sink_record") != record.get("response"):
                    _add(errors, "public_sink_publisher")
                else:
                    public_sink_records.append(record["public_sink_record"])
            elif "public_sink_record" in record:
                _add(errors, "public_sink_coordinator")
        elif status == "failed":
            facts["attempted"] += 1
            failure_class = record.get("failure_class")
            if failure_class not in {"model_output", "infrastructure"} or not isinstance(record.get("failure_code"), str):
                _add(errors, "failed_record_shape")
            if any(field in record for field in ("response", "completion_identity", "public_sink_record")):
                _add(errors, "failed_record_safety")
            if failure_class == "model_output":
                facts["model_failure_codes"].add(record["failure_code"])
            if failure_class == "infrastructure":
                facts["infrastructure_record_failure"] = True
        elif status == "unstarted":
            if not isinstance(record.get("reason"), str) or any(field in record for field in ("response", "completion_identity", "public_sink_record")):
                _add(errors, "unstarted_record_safety")
        else:
            _add(errors, "transcript_status")
    if seen != set(expected_by_id):
        _add(errors, "transcript_episodes")
    if transcript.get("public_sink_records") != public_sink_records:
        _add(errors, "public_sink_ledger")
    for stage, expected_count in (("screen_gate", 8), ("screen_remainder", 24), ("confirmation", 96)):
        if len(by_stage[stage]) != expected_count:
            _add(errors, "stage_card_count")
    gate = by_stage["screen_gate"]
    remainder = by_stage["screen_remainder"]
    confirmation = by_stage["confirmation"]
    gate_all_completed = len(gate) == 8 and all(record.get("status") == "completed" for record in gate)
    remainder_all_completed = len(remainder) == 24 and all(record.get("status") == "completed" for record in remainder)
    if not gate_all_completed and any(record.get("status") != "unstarted" for record in remainder + confirmation):
        _add(errors, "screen_gate_blocking")
    if gate_all_completed and not remainder_all_completed and any(record.get("status") != "unstarted" for record in confirmation):
        _add(errors, "screen_remainder_blocking")
    facts["all_completed"] = len(records) == 128 and all(record.get("status") == "completed" for record in records)
    facts["stage_summary"] = {
        stage: {
            "started": sum(record.get("status") in {"completed", "failed"} for record in current),
            "completed": sum(record.get("status") == "completed" for record in current),
            "failed": sum(record.get("status") == "failed" for record in current),
            "unstarted": sum(record.get("status") == "unstarted" for record in current),
        }
        for stage, current in by_stage.items()
    }
    return facts


def _identity_is_complete(identity: Any) -> bool:
    if not isinstance(identity, dict) or identity.get("consistent") is not True:
        return False
    start, end = identity.get("start"), identity.get("end")
    required = {"status", "normalized_models_sha256", "target_model_present", "model_count"}
    return (
        isinstance(start, dict)
        and isinstance(end, dict)
        and set(start) == required
        and set(end) == required
        and start.get("status") == "accepted"
        and end.get("status") == "accepted"
        and start.get("target_model_present") is True
        and end.get("target_model_present") is True
        and isinstance(start.get("normalized_models_sha256"), str)
        and start.get("normalized_models_sha256") == end.get("normalized_models_sha256")
    )


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
        _add(errors, "frozen_assets_unreadable")
    if assets is not None:
        _validate_source_contracts(asset_root, assets, errors)
    if not isinstance(manifest, dict):
        return {"passed": False, "run_id": run_id, "errors": errors}
    if manifest.get("run_id") != run_id:
        _add(errors, "manifest_run_id")
    if assets is not None and manifest.get("frozen_asset_sha256") != assets.asset_hashes:
        _add(errors, "frozen_asset_hash")
    if manifest.get("code_provenance_sha256") != _current_code_provenance():
        _add(errors, "code_provenance_hash")
    artifacts = {
        "ledger.jsonl": run_dir / "ledger.jsonl",
        "transcript.json": run_dir / "transcript.json",
        "outcome.json": run_dir / "outcome.json",
        "readiness_report.json": run_dir / "readiness_report.json",
        "reports/DELIVERY.md": data_root / "reports" / "DELIVERY.md",
    }
    declared = manifest.get("artifact_sha256")
    if not isinstance(declared, dict) or set(declared) != set(artifacts):
        _add(errors, "manifest_artifact_paths")
    else:
        for relative, path in artifacts.items():
            try:
                if declared[relative] != sha256_file(path):
                    _add(errors, "artifact_hash_" + relative.replace("/", "_"))
            except OSError:
                _add(errors, "artifact_missing_" + relative.replace("/", "_"))
    _validate_ledger(run_dir / "ledger.jsonl", run_id, errors)
    facts = _validate_transcript(transcript, assets, errors) if assets is not None else {"all_completed": False, "model_failure_codes": set(), "infrastructure_record_failure": True, "attempted": 0, "stage_summary": {}}
    if not isinstance(outcome, dict):
        _add(errors, "outcome_shape")
    else:
        if outcome.get("run_id") != run_id or outcome.get("decision") != manifest.get("decision"):
            _add(errors, "outcome_run_id")
        counts = outcome.get("transport_counts")
        if not isinstance(counts, dict):
            _add(errors, "outcome_counts")
        else:
            if counts.get("retry_count") != 0 or counts.get("metadata_calls", -1) > 2 or counts.get("completion_calls", -1) > 128:
                _add(errors, "transport_budget_or_retry")
            if counts.get("completion_calls") != facts["attempted"]:
                _add(errors, "completion_count")
        if outcome.get("stage_summary") != facts["stage_summary"]:
            _add(errors, "stage_summary")
        model_failure_codes = sorted(facts["model_failure_codes"])
        if outcome.get("model_failure_codes") != model_failure_codes:
            _add(errors, "model_failure_codes")
        identity_complete = _identity_is_complete(outcome.get("identity")) and outcome.get("identity") == (transcript.get("identity") if isinstance(transcript, dict) else None)
        declared_infra = outcome.get("infrastructure_failure")
        has_infra = facts["infrastructure_record_failure"] or declared_infra != "none" or not identity_complete
        if has_infra:
            expected_decision = "rework"
        elif model_failure_codes:
            expected_decision = "not_qualified"
        elif facts["all_completed"]:
            expected_decision = "qualified"
        else:
            expected_decision = "rework"
        if outcome.get("decision") != expected_decision:
            _add(errors, "outcome_decision_logic")
    if not isinstance(report, dict):
        _add(errors, "report_shape")
    else:
        if report.get("run_id") != run_id or report.get("decision") != manifest.get("decision"):
            _add(errors, "report_run_id")
        if report.get("retry_count") != 0 or report.get("history_runtime_imports") != [] or report.get("p1_p2_started") is not False or report.get("server_error_bodies_persisted") is not False or report.get("rejected_reasoning_persisted") is not False:
            _add(errors, "report_boundary")
    try:
        delivery = (data_root / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
        required = ("Run ID: " + run_id, "Decision: " + str(manifest.get("decision")), "Retry count: 0", "P1/P2 lock status: no P1 or P2 execution was started.")
        if any(item not in delivery for item in required):
            _add(errors, "delivery_report_content")
    except OSError:
        _add(errors, "delivery_report_unreadable")
    return {"passed": not errors, "run_id": run_id, "errors": errors}
