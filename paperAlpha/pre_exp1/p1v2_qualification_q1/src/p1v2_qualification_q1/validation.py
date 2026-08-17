"""Offline-only validation for a completed Q1 artifact."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import common
from .common import (
    IDENTITY_FALSE_FIELDS,
    append_jsonl,
    artifact_paths,
    canonical_json_bytes,
    code_provenance,
    fixed_identity,
    load_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    static_input_hashes,
    validate_run_id,
    verify_source_contracts,
)
from .errors import ValidationError
from .protocol import build_chat_payload, load_tasks, response_validation, sink_record


def _error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def _read_object(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        value = load_json(path)
    except Exception:
        _error(errors, f"invalid_{label}")
        return None
    if not isinstance(value, dict):
        _error(errors, f"invalid_{label}")
        return None
    return value


def _expected_decision(outcome: dict[str, Any], terminal_states: list[str]) -> str:
    env = outcome.get("environment_fault_codes")
    output = outcome.get("output_fault_codes")
    real_calls = outcome.get("real_model_calls")
    if isinstance(env, list) and env:
        return "rework"
    if not isinstance(real_calls, int) or real_calls == 0:
        return "rework"
    if isinstance(output, list) and output:
        return "not_qualified"
    if real_calls != 128 or len(terminal_states) != 128:
        return "rework"
    if any(item != "model_output" for item in terminal_states):
        return "not_qualified"
    if outcome.get("local_loopback_metadata_calls") != 2:
        return "rework"
    return "qualified"


def _safe_records(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path)
    except Exception:
        _error(errors, f"invalid_{label}")
        return []


def _compare_hashes(paths: dict[str, Path], manifest: dict[str, Any], errors: list[str]) -> None:
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        _error(errors, "manifest_artifact_hashes")
        return
    for key in ("events", "transcripts", "public_sink", "metadata", "unstarted", "outcome"):
        if hashes.get(key) != sha256_file(paths[key]):
            _error(errors, f"artifact_hash_mismatch:{key}")
    if manifest.get("outcome_sha256") != sha256_file(paths["outcome"]):
        _error(errors, "outcome_hash_mismatch")


def _validate_provenance(manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        source_contracts = verify_source_contracts()
        if manifest.get("source_contracts") != source_contracts:
            _error(errors, "manifest_source_contracts_mismatch")
    except Exception:
        _error(errors, "source_contract_drift")
    if manifest.get("static_input_hashes") != static_input_hashes():
        _error(errors, "q1_static_input_drift")
    if manifest.get("code_provenance") != code_provenance():
        _error(errors, "q1_code_provenance_drift")
    try:
        q0_core = common.q0_manifest_core()
        if manifest.get("q0_current_manifest_core") != q0_core:
            _error(errors, "q0_manifest_core_drift")
    except Exception:
        _error(errors, "q0_manifest_core_invalid")


def _validate_identity(outcome: dict[str, Any], errors: list[str]) -> None:
    for field in IDENTITY_FALSE_FIELDS:
        if outcome.get(field) is not False:
            _error(errors, f"identity_flag_not_false:{field}")
    if outcome.get("all_p1_p2_eligibility_flags_false") is not True:
        _error(errors, "identity_summary_missing")
    if outcome.get("remote_network_calls") != 0 or outcome.get("replay_model_calls") != 0 or outcome.get("replay_network_calls") != 0:
        _error(errors, "forbidden_network_or_replay_count")


def _validate_ledger(
    paths: dict[str, Path], outcome: dict[str, Any], errors: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = _safe_records(paths["events"], errors, "events")
    transcripts = _safe_records(paths["transcripts"], errors, "transcripts")
    sinks = _safe_records(paths["public_sink"], errors, "public_sink")
    unstarted = _safe_records(paths["unstarted"], errors, "unstarted")
    terminals = [item for item in events if item.get("record_type") == "model_terminal"]
    authorization_positions = [index for index, item in enumerate(events) if item.get("record_type") == "live_authorization_consumed"]
    if len(terminals) != len(transcripts):
        _error(errors, "terminal_transcript_count_mismatch")
    if outcome.get("real_model_calls") != len(terminals):
        _error(errors, "real_model_call_count_mismatch")
    if terminals:
        if len(authorization_positions) != 1 or authorization_positions[0] >= next(index for index, item in enumerate(events) if item.get("record_type") == "model_terminal"):
            _error(errors, "live_authorization_order")
    elif authorization_positions:
        _error(errors, "unexpected_live_authorization")
    request_ids: set[str] = set()
    transcript_by_id: dict[str, dict[str, Any]] = {}
    for transcript in transcripts:
        request_id = transcript.get("request_id")
        if not isinstance(request_id, str) or request_id in transcript_by_id:
            _error(errors, "duplicate_or_invalid_transcript_id")
            continue
        transcript_by_id[request_id] = transcript
    for event in terminals:
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or request_id in request_ids:
            _error(errors, "duplicate_or_invalid_terminal_id")
            continue
        request_ids.add(request_id)
        transcript = transcript_by_id.get(request_id)
        if transcript is None:
            _error(errors, "missing_transcript")
            continue
        for key in ("stage", "role", "task_id", "episode_id", "terminal_state", "failure_code", "payload_sha256", "prompt_sha256", "attempt", "retry_count"):
            if event.get(key) != transcript.get(key):
                _error(errors, f"terminal_transcript_mismatch:{key}")
        if event.get("attempt") != 1 or event.get("retry_count") != 0:
            _error(errors, "retry_or_attempt_violation")
    if set(transcript_by_id) != request_ids:
        _error(errors, "orphan_transcript")
    return events, transcripts, sinks, unstarted


def _validate_tasks_and_outputs(
    events: list[dict[str, Any]],
    transcripts: list[dict[str, Any]],
    sinks: list[dict[str, Any]],
    unstarted: list[dict[str, Any]],
    outcome: dict[str, Any],
    errors: list[str],
) -> None:
    try:
        screen = load_tasks("screen")
        confirmation = load_tasks("confirmation")
    except Exception:
        _error(errors, "frozen_task_fixtures_invalid")
        return
    task_by_id = {task["task_id"]: task for task in screen + confirmation}
    terminals = [item for item in events if item.get("record_type") == "model_terminal"]
    transcripts_by_id = {item.get("request_id"): item for item in transcripts if isinstance(item.get("request_id"), str)}
    terminal_task_ids: list[str] = []
    expected_sinks: list[dict[str, str]] = []
    terminal_states: list[str] = []
    for terminal in terminals:
        task_id = terminal.get("task_id")
        task = task_by_id.get(task_id)
        transcript = transcripts_by_id.get(terminal.get("request_id"))
        if task is None or transcript is None:
            _error(errors, "unknown_task_or_transcript")
            continue
        terminal_task_ids.append(task_id)
        terminal_states.append(str(terminal.get("terminal_state")))
        if terminal.get("stage") != task["stage"] or terminal.get("role") != task["role"] or terminal.get("episode_id") != task["episode_id"]:
            _error(errors, "terminal_task_identity_mismatch")
        payload = transcript.get("request_payload")
        expected_payload = build_chat_payload(task)
        if payload != expected_payload or transcript.get("payload_sha256") != sha256_text(canonical_json_bytes(expected_payload).decode("utf-8")):
            _error(errors, "frozen_payload_mismatch")
        if transcript.get("prompt_sha256") != sha256_text(expected_payload["messages"][0]["content"]):
            _error(errors, "prompt_hash_mismatch")
        terminal_state = terminal.get("terminal_state")
        failure_code = terminal.get("failure_code")
        raw_output = transcript.get("raw_output")
        has_reasoning = transcript.get("thinking_detected") is True or failure_code == "thinking_tag_detected"
        if has_reasoning and "raw_output" in transcript:
            _error(errors, "reasoning_content_leaked")
        if isinstance(raw_output, str):
            if transcript.get("raw_output_sha256") != sha256_text(raw_output) or transcript.get("raw_output_length") != len(raw_output):
                _error(errors, "raw_output_digest_mismatch")
            parsed, expected_fault = response_validation(raw_output, task)
            if terminal_state == "model_output" and expected_fault is not None:
                _error(errors, "accepted_invalid_model_output")
            if terminal_state == "model_output_rejected" and expected_fault != failure_code:
                _error(errors, "rejected_output_fault_mismatch")
            if terminal_state == "model_output" and task["role"] == "publisher" and parsed is not None:
                expected_sinks.append(sink_record(parsed))
        elif terminal_state == "model_output":
            _error(errors, "accepted_output_missing_raw_content")
        if terminal_state not in {"model_output", "model_output_rejected", "model_call_failed"}:
            _error(errors, "invalid_terminal_state")
    if len(set(terminal_task_ids)) != len(terminal_task_ids):
        _error(errors, "duplicate_terminal_task")
    unstarted_ids: list[str] = []
    for item in unstarted:
        task = task_by_id.get(item.get("task_id"))
        if task is None or item.get("record_type") != "unstarted_task":
            _error(errors, "invalid_unstarted_task")
            continue
        if any(item.get(field) != task[field] for field in ("stage", "role", "episode_id")):
            _error(errors, "unstarted_task_identity_mismatch")
        unstarted_ids.append(task["task_id"])
    if len(set(unstarted_ids)) != len(unstarted_ids) or set(unstarted_ids).intersection(terminal_task_ids):
        _error(errors, "unstarted_task_overlap")
    if outcome.get("real_model_calls", 0) > 0:
        all_ids = set(task_by_id)
        if set(terminal_task_ids).union(unstarted_ids) != all_ids:
            _error(errors, "task_coverage_mismatch")
    screen_events = [item for item in terminals if item.get("stage") == "screen"]
    confirmation_events = [item for item in terminals if item.get("stage") == "confirmation"]
    if confirmation_events and (len(screen_events) != 32 or any(item.get("terminal_state") != "model_output" for item in screen_events)):
        _error(errors, "confirmation_gate_violation")
    if len(sinks) > 80 or sinks != expected_sinks:
        _error(errors, "public_sink_violation")
    if outcome.get("screen_executed") != len(screen_events) or outcome.get("confirmation_executed") != len(confirmation_events):
        _error(errors, "stage_count_mismatch")
    if outcome.get("screen_unstarted") != sum(1 for item in unstarted if item.get("stage") == "screen"):
        _error(errors, "screen_unstarted_count_mismatch")
    if outcome.get("confirmation_unstarted") != sum(1 for item in unstarted if item.get("stage") == "confirmation"):
        _error(errors, "confirmation_unstarted_count_mismatch")
    actual_terminal_counts = dict(sorted(Counter(terminal_states).items()))
    if outcome.get("terminal_state_counts") != actual_terminal_counts:
        _error(errors, "terminal_state_count_mismatch")
    if outcome.get("qualification_batch_decision") != _expected_decision(outcome, terminal_states):
        _error(errors, "decision_priority_or_count_mismatch")
    if outcome.get("qualification_candidate_status") != outcome.get("qualification_batch_decision"):
        _error(errors, "candidate_decision_mismatch")


def _validate_metadata(paths: dict[str, Path], outcome: dict[str, Any], errors: list[str]) -> None:
    metadata = _safe_records(paths["metadata"], errors, "metadata")
    tag_records = [item for item in metadata if item.get("record_type") == "loopback_tags"]
    if outcome.get("local_loopback_metadata_calls") != len(tag_records):
        _error(errors, "metadata_call_count_mismatch")
    if outcome.get("local_loopback_http_calls") != outcome.get("real_model_calls", 0) + len(tag_records):
        _error(errors, "loopback_http_count_mismatch")
    if outcome.get("real_model_calls", 0) > 0:
        phases = [item.get("phase") for item in tag_records]
        if phases != ["start", "end"]:
            _error(errors, "metadata_phase_mismatch")
        if not tag_records or outcome.get("start_model_fingerprint") != tag_records[0].get("fingerprint"):
            _error(errors, "start_fingerprint_mismatch")
        if not tag_records or outcome.get("end_model_fingerprint") != tag_records[-1].get("fingerprint"):
            _error(errors, "end_fingerprint_mismatch")
        if outcome.get("start_model_fingerprint") != outcome.get("end_model_fingerprint"):
            env = outcome.get("environment_fault_codes")
            if outcome.get("qualification_batch_decision") != "rework" or not isinstance(env, list) or "model_fingerprint_drift" not in env:
                _error(errors, "fingerprint_drift_not_reworked")
    version_records = [item for item in metadata if item.get("record_type") == "ollama_version"]
    if outcome.get("real_model_calls", 0) > 0 and [item.get("phase") for item in version_records] != ["start", "end"]:
        _error(errors, "version_phase_mismatch")


def render_delivery(run_id: str, outcome: dict[str, Any], manifest: dict[str, Any]) -> str:
    """A deterministic, non-sensitive report; validator can detect later tampering."""
    return (
        "# P1v2 Q1 local qualification delivery\n\n"
        f"- Run ID: `{run_id}`\n"
        "- Protocol: `P1V2_Q1_LIVE_LOCAL`; local model: `qwen3:8b`.\n"
        "- Q0 current run: `P1V2Q-READINESS-DRY-20260801T010106000000Z`.\n"
        f"- Single decision: `{outcome.get('qualification_batch_decision')}`.\n"
        f"- Screen/Confirmation executed: {outcome.get('screen_executed')}/{outcome.get('confirmation_executed')}; unstarted: {outcome.get('screen_unstarted')}/{outcome.get('confirmation_unstarted')}.\n"
        f"- Calls: inference={outcome.get('real_model_calls')}, metadata={outcome.get('local_loopback_metadata_calls')}, loopback={outcome.get('local_loopback_http_calls')}, remote=0, replay-model=0, replay-network=0.\n"
        f"- Start/end fingerprint equal: `{outcome.get('start_model_fingerprint') == outcome.get('end_model_fingerprint')}`.\n"
        f"- Interpreter: `{manifest.get('runtime', {}).get('python_executable')}` / Python `{manifest.get('runtime', {}).get('python_version')}`; dependency changes: none.\n"
        "- Live command: `conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q1/scripts/run_q1_live.py --allow-live-q1`.\n"
        "- Q1 validation/replay are offline ledger checks. Q0 public pre/post validation and replay are recorded by the external acceptance command log.\n"
        f"- Manifest source-tree hash: `{manifest.get('code_provenance', {}).get('source_tree_hash')}`.\n"
        "- No `.env`, remote API, retry, model switch, P1 run, or P2 run is authorized by this artifact.\n"
    )


def validate_run(run_id: str, write_audit: bool = True) -> dict[str, Any]:
    """Read only the artifact and frozen inputs; never creates transports or network calls."""
    errors: list[str] = []
    try:
        validate_run_id(run_id)
        paths = artifact_paths(run_id)
    except Exception:
        return {"passed": False, "run_id": run_id, "errors": ["invalid_run_id"], "replay_model_calls": 0, "replay_network_calls": 0}
    required = ("events", "transcripts", "public_sink", "metadata", "unstarted", "outcome", "manifest")
    if not paths["run_dir"].is_dir() or any(not paths[key].is_file() for key in required):
        result = {"passed": False, "run_id": run_id, "errors": ["missing_run_artifact"], "replay_model_calls": 0, "replay_network_calls": 0}
        return result
    manifest = _read_object(paths["manifest"], errors, "manifest")
    outcome = _read_object(paths["outcome"], errors, "outcome")
    if manifest is None or outcome is None:
        result = {"passed": False, "run_id": run_id, "errors": errors, "replay_model_calls": 0, "replay_network_calls": 0}
        if write_audit:
            append_jsonl(paths["validation"], {"record_type": "validation", **result})
        return result
    if manifest.get("run_id") != run_id or outcome.get("run_id") != run_id:
        _error(errors, "run_id_mismatch")
    _compare_hashes(paths, manifest, errors)
    _validate_provenance(manifest, errors)
    _validate_identity(outcome, errors)
    events, transcripts, sinks, unstarted = _validate_ledger(paths, outcome, errors)
    expected_request_digests = [
        {
            "request_id": record["request_id"],
            "payload_sha256": record["payload_sha256"],
            "raw_output_sha256": record.get("raw_output_sha256"),
            "suppressed_output_sha256": record.get("suppressed_output_sha256"),
            "transport_response_sha256": record.get("transport_response_sha256"),
            "terminal_state": record["terminal_state"],
            "failure_code": record.get("failure_code"),
        }
        for record in transcripts
    ]
    if manifest.get("request_digests") != expected_request_digests:
        _error(errors, "manifest_request_digests_mismatch")
    _validate_tasks_and_outputs(events, transcripts, sinks, unstarted, outcome, errors)
    _validate_metadata(paths, outcome, errors)
    if paths["delivery"].exists() and paths["delivery"].read_text(encoding="utf-8") != render_delivery(run_id, outcome, manifest):
        _error(errors, "delivery_report_tampered")
    result = {
        "passed": not errors,
        "run_id": run_id,
        "errors": errors,
        "real_model_calls": outcome.get("real_model_calls"),
        "local_loopback_metadata_calls": outcome.get("local_loopback_metadata_calls"),
        "local_loopback_http_calls": outcome.get("local_loopback_http_calls"),
        "remote_network_calls": 0,
        "replay_model_calls": 0,
        "replay_network_calls": 0,
    }
    if write_audit:
        append_jsonl(paths["validation"], {"record_type": "validation", **result})
    return result
