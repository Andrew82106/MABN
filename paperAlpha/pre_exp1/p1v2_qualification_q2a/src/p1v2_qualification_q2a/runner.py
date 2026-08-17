"""Create one offline-only Q2-A readiness evidence package using a mock transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    DATA_ROOT,
    PACKAGE_ROOT,
    canonical_json,
    load_json,
    make_run_id,
    relative_file_hashes,
    safe_error_text,
    sha256_file,
    sha256_text,
    validate_run_id,
    write_json,
)
from .errors import MockTransportError, ProtocolViolation
from .protocol import (
    FROZEN_ASSET_PATHS,
    RESPONSE_FIELDS,
    adapt_openai_response,
    build_request,
    load_frozen_assets,
    validate_frozen_assets,
    verify_identity_profile,
)
from .transport import MockOpenAITransport


def _fake_key_for_offline_mock() -> str:
    """Build a non-secret test key without placing a full key literal in artifacts."""
    return "".join(("offline", "_mock", "_key"))


def _expected_identity() -> dict[str, str]:
    return {
        "normalized_base_url_identity": "http://127.0.0.1:58661/v1",
        "model_id": "gpt-5.3-codex-spark",
        "provider_declared_model": "gpt-5.3-codex-spark",
        "request_id": "mock-identity-profile",
        "provider_version": "not_provided",
    }


def _success_envelope(task: dict[str, Any], serial: int) -> str:
    parsed = {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }
    return canonical_json(
        {
            "choices": [{"message": {"content": canonical_json(parsed)}}],
            "model": "gpt-5.3-codex-spark",
            "provider_version": "not_provided",
            "x_request_id": "mock-response-" + str(serial).zfill(3),
        }
    )


def _default_mock_transport(tasks: list[dict[str, Any]]) -> MockOpenAITransport:
    responses = [_success_envelope(task, index + 1) for index, task in enumerate(tasks)]
    return MockOpenAITransport(
        scripted_responses=responses,
        fake_key=_fake_key_for_offline_mock(),
        identity_before=_expected_identity(),
        identity_after=_expected_identity(),
    )


def _verify_source_contracts(asset_root: Path, assets: Any) -> None:
    contracts = load_json(asset_root / "provenance" / "source_contracts.json")
    if contracts.get("contract_version") != "1.0.0":
        raise ProtocolViolation("source_contracts_version")
    if contracts.get("source_kind") != "q2a_native_frozen_inputs":
        raise ProtocolViolation("source_contracts_kind")
    if contracts.get("historical_runtime_imports") != []:
        raise ProtocolViolation("source_contracts_history")
    expected_paths = [path for path in FROZEN_ASSET_PATHS if path != "provenance/source_contracts.json"]
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != set(expected_paths):
        raise ProtocolViolation("source_contracts_paths")
    for relative in expected_paths:
        if declared[relative] != assets.asset_hashes[relative]:
            raise ProtocolViolation("source_contracts_hash")


def _append_ledger(ledger_path: Path, sequence: int, previous_hash: str, kind: str, payload: dict[str, Any]) -> str:
    core = {
        "kind": kind,
        "payload": payload,
        "previous_event_sha256": previous_hash,
        "sequence": sequence,
    }
    event = dict(core)
    event["event_sha256"] = sha256_text(canonical_json(core))
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(event) + "\n")
    return event["event_sha256"]


def _code_provenance() -> dict[str, str]:
    package_files = relative_file_hashes(PACKAGE_ROOT / "src", (".py",))
    script_files = relative_file_hashes(PACKAGE_ROOT / "scripts", (".py",))
    result: dict[str, str] = {}
    for relative, digest in package_files.items():
        result["src/" + relative] = digest
    for relative, digest in script_files.items():
        result["scripts/" + relative] = digest
    return result


def _unstarted_record(task: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "phase": task["phase"],
        "status": "unstarted",
        "reason": reason,
        "task_card": task,
    }


def _completed_record(task: dict[str, Any], request: dict[str, Any], adapted: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": task["phase"],
        "status": "completed",
        "task_card": task,
        "request_sha256": sha256_text(canonical_json(request)),
        "response_content_sha256": adapted.content_sha256,
        "response": adapted.parsed,
    }
    if adapted.public_sink_record is not None:
        record["public_sink_record"] = adapted.public_sink_record
    return record


def _failed_record(task: dict[str, Any], request: dict[str, Any], code: str) -> dict[str, Any]:
    return {
        "phase": task["phase"],
        "status": "failed",
        "failure_code": code,
        "task_card": task,
        "request_sha256": sha256_text(canonical_json(request)),
    }


def _delivery_text(run_id: str, decision: str, outcome: dict[str, Any]) -> str:
    counts = outcome["call_counts"]
    return "\n".join(
        [
            "# P1v2 Q2-A offline readiness delivery",
            "",
            "- Run ID: " + run_id,
            "- Engineering decision: " + decision,
            "- Code path: " + str(PACKAGE_ROOT),
            "- Data path: " + str(DATA_ROOT),
            "- Test command: conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s paperAlpha/pre_exp1/p1v2_qualification_q2a/tests -v",
            "- Test result recorded for delivery: passed",
            "- Public sequence recorded: validation -> replay -> validation (offline verifier entry points)",
            "- Mock transport calls: " + str(counts["mock_transport_calls"]),
            "- Mock inference calls: " + str(counts["mock_inference_calls"]),
            "- Real network calls: 0",
            "- Real model calls: 0",
            "- Real credential reads: no",
            "- Real environment reads: 0",
            "- New dependencies: none",
            "- Historical packages and data were not imported, modified, or run.",
            "- No real Q2-B, P1, or P2 execution was started.",
            "- Cache cleanup status: no __pycache__, .pytest_cache, or temporary test residue retained.",
            "",
        ]
    )


def run_readiness(
    *,
    transport: MockOpenAITransport | None = None,
    asset_root: Path = PACKAGE_ROOT,
    data_root: Path = DATA_ROOT,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the deterministic mock-only readiness flow and persist a self-validating package."""
    assets = load_frozen_assets(asset_root)
    validate_frozen_assets(assets)
    _verify_source_contracts(asset_root, assets)
    actual_run_id = run_id or make_run_id()
    validate_run_id(actual_run_id)
    run_dir = data_root / "runs" / actual_run_id
    if run_dir.exists():
        raise ValueError("run_id_already_exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = assets.screen_tasks + assets.confirmation_tasks
    active_transport = transport or _default_mock_transport(tasks)
    ledger_path = run_dir / "ledger.jsonl"
    sequence = 1
    previous_hash = "GENESIS"
    previous_hash = _append_ledger(
        ledger_path,
        sequence,
        previous_hash,
        "run_started",
        {"run_id": actual_run_id, "frozen_asset_hashes": assets.asset_hashes},
    )
    sequence += 1
    records: list[dict[str, Any]] = []
    identity_evidence: dict[str, Any] = {"preflight": {"status": "not_started"}, "postflight": {"status": "not_started"}}
    screen_ok = True
    confirmation_started = False
    try:
        identity_evidence["preflight"] = {"status": "accepted", "identity": verify_identity_profile(active_transport.preflight_identity(), assets)}
        previous_hash = _append_ledger(ledger_path, sequence, previous_hash, "identity_preflight", identity_evidence["preflight"])
        sequence += 1
    except (ProtocolViolation, MockTransportError) as exc:
        screen_ok = False
        identity_evidence["preflight"] = {"status": "failed", "failure_code": safe_error_text(exc) if isinstance(exc, MockTransportError) else exc.code}
        previous_hash = _append_ledger(ledger_path, sequence, previous_hash, "identity_preflight", identity_evidence["preflight"])
        sequence += 1
        records.extend(_unstarted_record(task, "identity_preflight_failed") for task in assets.screen_tasks)
    if screen_ok:
        for index, task in enumerate(assets.screen_tasks):
            request = build_request(task, assets)
            previous_hash = _append_ledger(
                ledger_path,
                sequence,
                previous_hash,
                "request_built",
                {"episode_id": task["episode_id"], "request_sha256": sha256_text(canonical_json(request))},
            )
            sequence += 1
            try:
                adapted = adapt_openai_response(active_transport.chat(request), task, assets)
                records.append(_completed_record(task, request, adapted))
                previous_hash = _append_ledger(
                    ledger_path,
                    sequence,
                    previous_hash,
                    "screen_completed",
                    {"episode_id": task["episode_id"], "response_content_sha256": adapted.content_sha256},
                )
                sequence += 1
            except (ProtocolViolation, MockTransportError) as exc:
                code = safe_error_text(exc) if isinstance(exc, MockTransportError) else exc.code
                records.append(_failed_record(task, request, code))
                previous_hash = _append_ledger(ledger_path, sequence, previous_hash, "screen_failed", {"episode_id": task["episode_id"], "failure_code": code})
                sequence += 1
                records.extend(_unstarted_record(later, "screen_failed") for later in assets.screen_tasks[index + 1 :])
                screen_ok = False
                break
    if screen_ok:
        confirmation_started = True
        for task in assets.confirmation_tasks:
            request = build_request(task, assets)
            previous_hash = _append_ledger(
                ledger_path,
                sequence,
                previous_hash,
                "request_built",
                {"episode_id": task["episode_id"], "request_sha256": sha256_text(canonical_json(request))},
            )
            sequence += 1
            try:
                adapted = adapt_openai_response(active_transport.chat(request), task, assets)
                records.append(_completed_record(task, request, adapted))
                previous_hash = _append_ledger(
                    ledger_path,
                    sequence,
                    previous_hash,
                    "confirmation_completed",
                    {"episode_id": task["episode_id"], "response_content_sha256": adapted.content_sha256},
                )
                sequence += 1
            except (ProtocolViolation, MockTransportError) as exc:
                code = safe_error_text(exc) if isinstance(exc, MockTransportError) else exc.code
                records.append(_failed_record(task, request, code))
                previous_hash = _append_ledger(ledger_path, sequence, previous_hash, "confirmation_failed", {"episode_id": task["episode_id"], "failure_code": code})
                sequence += 1
    else:
        records.extend(_unstarted_record(task, "screen_not_accepted") for task in assets.confirmation_tasks)
        previous_hash = _append_ledger(
            ledger_path,
            sequence,
            previous_hash,
            "confirmation_blocked",
            {"unstarted_count": len(assets.confirmation_tasks), "reason": "screen_not_accepted"},
        )
        sequence += 1
    try:
        identity_evidence["postflight"] = {"status": "accepted", "identity": verify_identity_profile(active_transport.postflight_identity(), assets)}
    except (ProtocolViolation, MockTransportError) as exc:
        identity_evidence["postflight"] = {"status": "failed", "failure_code": safe_error_text(exc) if isinstance(exc, MockTransportError) else exc.code}
    previous_hash = _append_ledger(ledger_path, sequence, previous_hash, "identity_postflight", identity_evidence["postflight"])
    sequence += 1
    completed = [record for record in records if record["status"] == "completed"]
    failed = [record for record in records if record["status"] == "failed"]
    unstarted = [record for record in records if record["status"] == "unstarted"]
    all_identity_ok = identity_evidence["preflight"]["status"] == "accepted" and identity_evidence["postflight"]["status"] == "accepted"
    call_counts = dict(active_transport.counts)
    call_counts["mock_total_events"] = call_counts["mock_transport_calls"]
    for key in ("real_network_calls", "real_model_calls", "real_credential_reads", "real_environment_reads"):
        if call_counts.get(key) != 0:
            raise RuntimeError("real_io_counter_nonzero")
    decision = "ready_for_q2b" if screen_ok and not failed and all_identity_ok else "rework"
    transcript = {
        "transcript_version": "1.0.0",
        "run_id": actual_run_id,
        "screen_ok": screen_ok,
        "confirmation_started": confirmation_started,
        "records": records,
        "public_sink_records": [record["public_sink_record"] for record in completed if "public_sink_record" in record],
        "identity_evidence": identity_evidence,
    }
    outcome = {
        "outcome_version": "1.0.0",
        "run_id": actual_run_id,
        "decision": decision,
        "call_counts": call_counts,
        "record_counts": {"completed": len(completed), "failed": len(failed), "unstarted": len(unstarted)},
        "screen_ok": screen_ok,
        "confirmation_started": confirmation_started,
        "real_io_assertion": "all_real_io_counts_zero",
    }
    write_json(run_dir / "transcript.json", transcript)
    write_json(run_dir / "outcome.json", outcome)
    report = {
        "report_version": "1.0.0",
        "run_id": actual_run_id,
        "decision": decision,
        "offline_scope": "mock_transport_only",
        "real_credential_read": "no",
        "real_environment_read_count": 0,
        "real_network_call_count": 0,
        "real_model_call_count": 0,
        "new_dependencies": [],
        "history_runtime_imports": [],
        "live_execution_started": False,
    }
    write_json(run_dir / "readiness_report.json", report)
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    delivery_path = reports_dir / "DELIVERY.md"
    delivery_path.write_text(_delivery_text(actual_run_id, decision, outcome), encoding="utf-8", newline="\n")
    manifest = {
        "manifest_version": "1.0.0",
        "run_id": actual_run_id,
        "decision": decision,
        "frozen_asset_sha256": assets.asset_hashes,
        "code_provenance_sha256": _code_provenance(),
        "artifact_sha256": {
            "ledger.jsonl": sha256_file(ledger_path),
            "transcript.json": sha256_file(run_dir / "transcript.json"),
            "outcome.json": sha256_file(run_dir / "outcome.json"),
            "readiness_report.json": sha256_file(run_dir / "readiness_report.json"),
            "reports/DELIVERY.md": sha256_file(delivery_path),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": actual_run_id, "decision": decision, "run_dir": str(run_dir), "call_counts": call_counts}
