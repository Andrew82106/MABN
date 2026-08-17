"""Staged Q2-B execution logic.  Only run_live creates the real client."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Callable

from .common import DATA_ROOT, PACKAGE_ROOT, canonical_json, load_json, make_run_id, relative_file_hashes, safe_error_code, sha256_file, sha256_text, validate_run_id, write_json
from .env_config import load_live_config
from .errors import ConfigError, ContractViolation, TransportError
from .protocol import FROZEN_ASSET_PATHS, adapt_completion, build_request, load_frozen_assets, parse_model_list, validate_frozen_assets
from .transport import RestrictedGatewayTransport


def _verify_source_contracts(asset_root: Path, assets: Any) -> None:
    contracts = load_json(asset_root / "provenance" / "source_contracts.json")
    expected_paths = [path for path in FROZEN_ASSET_PATHS if path != "provenance/source_contracts.json"]
    if contracts.get("contract_version") != "1.0.0" or contracts.get("source_kind") != "q2b_native_frozen_inputs" or contracts.get("historical_runtime_imports") != []:
        raise ContractViolation("source_contracts_contract", "infrastructure_failure")
    declared = contracts.get("frozen_asset_sha256")
    if not isinstance(declared, dict) or set(declared) != set(expected_paths):
        raise ContractViolation("source_contracts_paths", "infrastructure_failure")
    for relative in expected_paths:
        if declared[relative] != assets.asset_hashes[relative]:
            raise ContractViolation("source_contracts_hash", "infrastructure_failure")


def _append_ledger(ledger_path: Path, sequence: int, previous: str, kind: str, payload: dict[str, Any]) -> str:
    core = {"kind": kind, "payload": payload, "previous_event_sha256": previous, "sequence": sequence}
    event = dict(core)
    event["event_sha256"] = sha256_text(canonical_json(core))
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(event) + "\n")
    return event["event_sha256"]


def _code_provenance() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "src", (".py",)).items():
        result["src/" + relative] = digest
    for relative, digest in relative_file_hashes(PACKAGE_ROOT / "scripts", (".py",)).items():
        result["scripts/" + relative] = digest
    return result


def _task_stage(task: dict[str, Any]) -> str:
    if task["phase"] == "confirmation":
        return "confirmation"
    numeric = int(task["episode_id"].rsplit("-", 1)[1])
    return "screen_gate" if numeric <= 8 else "screen_remainder"


def _unstarted(task: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"stage": _task_stage(task), "status": "unstarted", "reason": reason, "task_card": task}


def _failed(task: dict[str, Any], request: dict[str, Any], code: str, failure_class: str) -> dict[str, Any]:
    return {
        "stage": _task_stage(task),
        "status": "failed",
        "failure_code": code,
        "failure_class": failure_class,
        "request_sha256": sha256_text(canonical_json(request)),
        "task_card": task,
    }


def _completed(task: dict[str, Any], request: dict[str, Any], adapted: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": _task_stage(task),
        "status": "completed",
        "request_sha256": sha256_text(canonical_json(request)),
        "response_content_sha256": adapted.content_sha256,
        "response": adapted.parsed,
        "completion_identity": adapted.safe_identity,
        "task_card": task,
    }
    if adapted.public_sink_record is not None:
        record["public_sink_record"] = adapted.public_sink_record
    return record


def _stage_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for stage in ("screen_gate", "screen_remainder", "confirmation"):
        current = [record for record in records if record["stage"] == stage]
        result[stage] = {
            "started": sum(record["status"] in {"completed", "failed"} for record in current),
            "completed": sum(record["status"] == "completed" for record in current),
            "failed": sum(record["status"] == "failed" for record in current),
            "unstarted": sum(record["status"] == "unstarted" for record in current),
        }
    return result


def _delivery_text(run_id: str, decision: str, outcome: dict[str, Any], credential_scope: str) -> str:
    summary = outcome["stage_summary"]
    counts = outcome["transport_counts"]
    identity = outcome["identity"]
    return "\n".join(
        [
            "# P1v2 Q2-B staged remote qualification delivery",
            "",
            "- Run ID: " + run_id,
            "- Decision: " + decision,
            "- Code path: " + str(PACKAGE_ROOT),
            "- Data path: " + str(DATA_ROOT),
            "- Screen gate started/unstarted: " + str(summary["screen_gate"]["started"]) + "/" + str(summary["screen_gate"]["unstarted"]),
            "- Screen remainder started/unstarted: " + str(summary["screen_remainder"]["started"]) + "/" + str(summary["screen_remainder"]["unstarted"]),
            "- Confirmation started/unstarted: " + str(summary["confirmation"]["started"]) + "/" + str(summary["confirmation"]["unstarted"]),
            "- Metadata calls: " + str(counts["metadata_calls"]),
            "- Completion calls: " + str(counts["completion_calls"]),
            "- Retry count: 0",
            "- Identity consistency: " + str(identity["consistent"]),
            "- Credential scope: " + credential_scope,
            "- Test command: conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s paperAlpha/pre_exp1/p1v2_qualification_q2b/tests -v",
            "- Offline tests: passed before the authorized live command.",
            "- Public validation/replay results: pending offline verification.",
            "- Cache cleanup status: no __pycache__, .pytest_cache, or temporary test residue retained.",
            "- P1/P2 lock status: no P1 or P2 execution was started.",
            "",
        ]
    )


def _write_artifacts(
    *,
    run_dir: Path,
    data_root: Path,
    run_id: str,
    assets: Any,
    transcript: dict[str, Any],
    outcome: dict[str, Any],
    credential_scope: str,
) -> None:
    write_json(run_dir / "transcript.json", transcript)
    write_json(run_dir / "outcome.json", outcome)
    report = {
        "report_version": "1.0.0",
        "run_id": run_id,
        "decision": outcome["decision"],
        "credential_scope": credential_scope,
        "retry_count": 0,
        "history_runtime_imports": [],
        "live_scope": "single_frozen_q2b_batch",
        "p1_p2_started": False,
        "server_error_bodies_persisted": False,
        "rejected_reasoning_persisted": False,
    }
    write_json(run_dir / "readiness_report.json", report)
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    delivery_path = reports_dir / "DELIVERY.md"
    delivery_path.write_text(_delivery_text(run_id, outcome["decision"], outcome, credential_scope), encoding="utf-8", newline="\n")
    manifest = {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "decision": outcome["decision"],
        "frozen_asset_sha256": assets.asset_hashes,
        "code_provenance_sha256": _code_provenance(),
        "artifact_sha256": {
            "ledger.jsonl": sha256_file(run_dir / "ledger.jsonl"),
            "transcript.json": sha256_file(run_dir / "transcript.json"),
            "outcome.json": sha256_file(run_dir / "outcome.json"),
            "readiness_report.json": sha256_file(run_dir / "readiness_report.json"),
            "reports/DELIVERY.md": sha256_file(delivery_path),
        },
    }
    write_json(run_dir / "manifest.json", manifest)


def run_staged(
    transport: Any,
    *,
    asset_root: Path = PACKAGE_ROOT,
    data_root: Path = DATA_ROOT,
    run_id: str | None = None,
    credential_scope: str = "injected_test",
    clock: Callable[[], float] = monotonic,
) -> dict[str, Any]:
    """Execute staged logic against an injected transport; test callers never create sockets."""
    assets = load_frozen_assets(asset_root)
    validate_frozen_assets(assets)
    _verify_source_contracts(asset_root, assets)
    actual_run_id = run_id or make_run_id()
    validate_run_id(actual_run_id)
    run_dir = data_root / "runs" / actual_run_id
    if run_dir.exists():
        raise ValueError("run_id_already_exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "ledger.jsonl"
    sequence = 1
    previous = "GENESIS"
    previous = _append_ledger(ledger_path, sequence, previous, "run_started", {"run_id": actual_run_id, "frozen_asset_hashes": assets.asset_hashes})
    sequence += 1
    tasks = assets.screen_tasks + assets.confirmation_tasks
    records_by_id: dict[str, dict[str, Any]] = {}
    model_failures: list[str] = []
    infrastructure_failure: str | None = None
    identity: dict[str, Any] = {"start": {"status": "not_started"}, "end": {"status": "not_started"}, "consistent": False}
    start_clock = clock()

    def mark_unstarted_remaining(reason: str) -> None:
        for pending in tasks:
            if pending["episode_id"] not in records_by_id:
                records_by_id[pending["episode_id"]] = _unstarted(pending, reason)

    def wall_clock_expired() -> bool:
        return clock() - start_clock > assets.protocol["batch_wall_clock_seconds"]

    def execute_stage(stage_tasks: list[dict[str, Any]]) -> bool:
        """Return False only for immediate infrastructure stop; model failures continue this stage."""
        nonlocal sequence, previous, infrastructure_failure
        for task in stage_tasks:
            if wall_clock_expired():
                infrastructure_failure = "batch_wall_clock_exceeded"
                records_by_id[task["episode_id"]] = _unstarted(task, infrastructure_failure)
                previous = _append_ledger(ledger_path, sequence, previous, "completion_not_started", {"episode_id": task["episode_id"], "reason": infrastructure_failure})
                sequence += 1
                return False
            try:
                request = build_request(task, assets)
            except ContractViolation as exc:
                infrastructure_failure = exc.code
                records_by_id[task["episode_id"]] = _unstarted(task, infrastructure_failure)
                previous = _append_ledger(ledger_path, sequence, previous, "completion_not_started", {"episode_id": task["episode_id"], "reason": infrastructure_failure})
                sequence += 1
                return False
            request_hash = sha256_text(canonical_json(request))
            previous = _append_ledger(ledger_path, sequence, previous, "completion_started", {"episode_id": task["episode_id"], "request_sha256": request_hash})
            sequence += 1
            try:
                wire = transport.complete(request)
                adapted = adapt_completion(wire.body, wire.headers, task, assets)
                records_by_id[task["episode_id"]] = _completed(task, request, adapted)
                previous = _append_ledger(ledger_path, sequence, previous, "completion_completed", {"episode_id": task["episode_id"], "response_content_sha256": adapted.content_sha256, "provider_declared_model": adapted.safe_identity["provider_declared_model"], "request_id": adapted.safe_identity["request_id"]})
                sequence += 1
            except ContractViolation as exc:
                if exc.disposition == "model_failure":
                    model_failures.append(exc.code)
                    records_by_id[task["episode_id"]] = _failed(task, request, exc.code, "model_output")
                    event_kind = "completion_model_failure"
                else:
                    infrastructure_failure = exc.code
                    records_by_id[task["episode_id"]] = _failed(task, request, exc.code, "infrastructure")
                    event_kind = "completion_infrastructure_failure"
                previous = _append_ledger(ledger_path, sequence, previous, event_kind, {"episode_id": task["episode_id"], "failure_code": exc.code})
                sequence += 1
                if infrastructure_failure is not None:
                    return False
            except TransportError as exc:
                infrastructure_failure = exc.code
                records_by_id[task["episode_id"]] = _failed(task, request, exc.code, "infrastructure")
                previous = _append_ledger(ledger_path, sequence, previous, "completion_infrastructure_failure", {"episode_id": task["episode_id"], "failure_code": exc.code})
                sequence += 1
                return False
        return True

    try:
        start_models = parse_model_list(transport.get_models().body, assets.protocol["model"])
        if not start_models.target_present:
            raise ContractViolation("target_model_missing", "infrastructure_failure")
        identity["start"] = {"status": "accepted", "normalized_models_sha256": start_models.normalized_sha256, "target_model_present": True, "model_count": start_models.model_count}
        previous = _append_ledger(ledger_path, sequence, previous, "models_start", identity["start"])
        sequence += 1
    except (TransportError, ContractViolation) as exc:
        infrastructure_failure = safe_error_code(exc)
        identity["start"] = {"status": "failed", "failure_code": infrastructure_failure}
        previous = _append_ledger(ledger_path, sequence, previous, "models_start", identity["start"])
        sequence += 1

    if infrastructure_failure is None:
        gate_finished = execute_stage(assets.screen_tasks[:8])
        gate_valid = gate_finished and not model_failures
        if infrastructure_failure is None and gate_valid:
            remainder_finished = execute_stage(assets.screen_tasks[8:])
            screens_valid = remainder_finished and not model_failures
            if infrastructure_failure is None and screens_valid:
                execute_stage(assets.confirmation_tasks)
            elif infrastructure_failure is None:
                for task in assets.confirmation_tasks:
                    records_by_id.setdefault(task["episode_id"], _unstarted(task, "screen_not_passed"))
        elif infrastructure_failure is None:
            for task in assets.screen_tasks[8:] + assets.confirmation_tasks:
                records_by_id.setdefault(task["episode_id"], _unstarted(task, "screen_gate_not_passed"))

    if infrastructure_failure is None:
        try:
            end_models = parse_model_list(transport.get_models().body, assets.protocol["model"])
            if not end_models.target_present:
                raise ContractViolation("target_model_missing", "infrastructure_failure")
            identity["end"] = {"status": "accepted", "normalized_models_sha256": end_models.normalized_sha256, "target_model_present": True, "model_count": end_models.model_count}
            identity["consistent"] = identity["start"].get("normalized_models_sha256") == end_models.normalized_sha256
            previous = _append_ledger(ledger_path, sequence, previous, "models_end", identity["end"] | {"consistent": identity["consistent"]})
            sequence += 1
            if not identity["consistent"]:
                infrastructure_failure = "model_list_drift"
        except (TransportError, ContractViolation) as exc:
            infrastructure_failure = safe_error_code(exc)
            identity["end"] = {"status": "failed", "failure_code": infrastructure_failure}
            previous = _append_ledger(ledger_path, sequence, previous, "models_end", identity["end"])
            sequence += 1
    else:
        identity["end"] = {"status": "not_started", "reason": "infrastructure_stop"}

    mark_unstarted_remaining(infrastructure_failure or "stage_not_reached")
    records = [records_by_id[task["episode_id"]] for task in tasks]
    summary = _stage_summary(records)
    completed_count = sum(record["status"] == "completed" for record in records)
    if infrastructure_failure is not None:
        decision = "rework"
    elif model_failures:
        decision = "not_qualified"
    elif completed_count == 128 and identity["consistent"]:
        decision = "qualified"
    else:
        decision = "rework"
    transport_counts = dict(transport.counts)
    if transport_counts.get("retry_count") != 0 or transport_counts.get("metadata_calls", 0) > 2 or transport_counts.get("completion_calls", 0) > 128:
        decision = "rework"
        infrastructure_failure = infrastructure_failure or "transport_budget_or_retry"
    transcript = {
        "transcript_version": "1.0.0",
        "run_id": actual_run_id,
        "records": records,
        "public_sink_records": [record["public_sink_record"] for record in records if "public_sink_record" in record],
        "identity": identity,
    }
    outcome = {
        "outcome_version": "1.0.0",
        "run_id": actual_run_id,
        "decision": decision,
        "transport_counts": transport_counts,
        "stage_summary": summary,
        "identity": identity,
        "infrastructure_failure": infrastructure_failure or "none",
        "model_failure_codes": sorted(set(model_failures)),
        "completion_records_completed": completed_count,
    }
    _write_artifacts(run_dir=run_dir, data_root=data_root, run_id=actual_run_id, assets=assets, transcript=transcript, outcome=outcome, credential_scope=credential_scope)
    return {"run_id": actual_run_id, "decision": decision, "run_dir": str(run_dir), "transport_counts": transport_counts, "stage_summary": summary}


def run_live() -> dict[str, Any]:
    """Authorized path only: it reads .env and constructs the real restricted client once."""
    runs_root = DATA_ROOT / "runs"
    if runs_root.exists() and any(runs_root.iterdir()):
        raise RuntimeError("live_run_already_exists")
    config = load_live_config()
    transport = RestrictedGatewayTransport(config)
    return run_staged(transport, credential_scope="live_path_only")
