"""One-shot Q1 execution state machine.

The public CLI exposes no model, endpoint, path, task, prompt, or retry
arguments.  Tests use an in-process mock transport; the public runner creates
the only permitted loopback transport.
"""

from __future__ import annotations

import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from . import common
from .common import (
    append_jsonl,
    artifact_paths,
    canonical_json_bytes,
    code_provenance,
    fixed_identity,
    initialize_run,
    load_protocol,
    new_run_id,
    q0_manifest_core,
    read_jsonl,
    sha256_file,
    sha256_text,
    static_input_hashes,
    verify_source_contracts,
    write_json_once,
)
from .errors import PreflightError, Q1Error, TransportError, ValidationError
from .protocol import build_chat_payload, content_digest, load_tasks, response_validation, sink_record, verify_frozen_contract_assets
from .transport import ChatResult, ModelFingerprint, OllamaLoopbackTransport, VersionEvidence


class Transport(Protocol):
    def get_tags(self, timeout: float = 10.0) -> ModelFingerprint: ...

    def chat(self, payload: dict[str, Any], timeout: float) -> ChatResult: ...

    def ollama_version(self) -> VersionEvidence: ...


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_code(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    if not text:
        text = type(exc).__name__
    return text[:160]


@dataclass
class AttemptState:
    run_id: str
    paths: dict[str, Path]
    clock: Callable[[], float]
    protocol: dict[str, Any] | None = None
    source_contracts: list[dict[str, str]] = field(default_factory=list)
    q0_core: dict[str, Any] | None = None
    start_fingerprint: ModelFingerprint | None = None
    end_fingerprint: ModelFingerprint | None = None
    start_version: VersionEvidence | None = None
    end_version: VersionEvidence | None = None
    real_model_calls: int = 0
    metadata_calls: int = 0
    output_faults: list[str] = field(default_factory=list)
    environment_faults: list[str] = field(default_factory=list)
    terminal_states: list[str] = field(default_factory=list)
    stage_counts: Counter[str] = field(default_factory=Counter)
    unstarted_counts: Counter[str] = field(default_factory=Counter)
    authorization_written: bool = False

    def event(self, record: dict[str, Any]) -> None:
        append_jsonl(self.paths["events"], record)

    def metadata(self, record: dict[str, Any]) -> None:
        append_jsonl(self.paths["metadata"], record)

    def mark_unstarted(self, task: dict[str, Any], reason: str) -> None:
        append_jsonl(
            self.paths["unstarted"],
            {
                "record_type": "unstarted_task",
                "run_id": self.run_id,
                "stage": task["stage"],
                "role": task["role"],
                "task_id": task["task_id"],
                "episode_id": task["episode_id"],
                "reason": reason,
            },
        )
        self.unstarted_counts[task["stage"]] += 1


def _base_event(state: AttemptState, record_type: str) -> dict[str, Any]:
    return {"record_type": record_type, "run_id": state.run_id, "recorded_at": _utc_timestamp()}


def _record_preflight(state: AttemptState, status: str, detail: str) -> None:
    state.event({**_base_event(state, "preflight"), "status": status, "detail": detail})


def _record_version(state: AttemptState, phase: str, evidence: VersionEvidence | None, error: str | None = None) -> None:
    record: dict[str, Any] = {**_base_event(state, "ollama_version"), "phase": phase}
    if evidence is not None:
        record["evidence"] = evidence.as_dict()
    if error is not None:
        record["error_code"] = error
    state.metadata(record)


def _record_tags(state: AttemptState, phase: str, fingerprint: ModelFingerprint | None, error: str | None = None) -> None:
    record: dict[str, Any] = {**_base_event(state, "loopback_tags"), "phase": phase}
    if fingerprint is not None:
        record["fingerprint"] = fingerprint.as_dict()
        record["fingerprint_sha256"] = fingerprint.sha256
    if error is not None:
        record["error_code"] = error
    state.metadata(record)


def _make_request_id(state: AttemptState, task: dict[str, Any]) -> str:
    return f"{state.run_id}:{task['stage']}:{task['task_id']}"


def _transcript_base(state: AttemptState, task: dict[str, Any], request_id: str, payload: dict[str, Any], elapsed: float) -> dict[str, Any]:
    payload_hash = sha256_file_from_bytes(canonical_json_bytes(payload))
    return {
        "record_type": "model_transcript",
        "run_id": state.run_id,
        "request_id": request_id,
        "stage": task["stage"],
        "role": task["role"],
        "task_id": task["task_id"],
        "episode_id": task["episode_id"],
        "prompt_sha256": sha256_text(payload["messages"][0]["content"]),
        "request_payload": payload,
        "payload_sha256": payload_hash,
        "elapsed_seconds": round(elapsed, 6),
        "attempt": 1,
        "retry_count": 0,
    }


def sha256_file_from_bytes(value: bytes) -> str:
    """Named to make ledger readers distinguish body hashes from file hashes."""
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _write_terminal(state: AttemptState, task: dict[str, Any], request_id: str, transcript: dict[str, Any], terminal_state: str, failure_code: str | None) -> None:
    transcript["terminal_state"] = terminal_state
    transcript["failure_code"] = failure_code
    append_jsonl(state.paths["transcripts"], transcript)
    state.event(
        {
            **_base_event(state, "model_terminal"),
            "request_id": request_id,
            "stage": task["stage"],
            "role": task["role"],
            "task_id": task["task_id"],
            "episode_id": task["episode_id"],
            "terminal_state": terminal_state,
            "failure_code": failure_code,
            "prompt_sha256": transcript["prompt_sha256"],
            "payload_sha256": transcript["payload_sha256"],
            "raw_output_sha256": transcript.get("raw_output_sha256"),
            "raw_output_length": transcript.get("raw_output_length", 0),
            "elapsed_seconds": transcript["elapsed_seconds"],
            "attempt": 1,
            "retry_count": 0,
        }
    )
    state.terminal_states.append(terminal_state)
    state.stage_counts[task["stage"]] += 1


def _invoke_task(state: AttemptState, transport: Transport, task: dict[str, Any], timeout: float) -> str:
    payload = build_chat_payload(task)
    request_id = _make_request_id(state, task)
    started = state.clock()
    state.real_model_calls += 1
    try:
        result = transport.chat(payload, timeout=timeout)
    except Exception as exc:
        elapsed = max(0.0, state.clock() - started)
        transcript = _transcript_base(state, task, request_id, payload, elapsed)
        transcript["transport_error_code"] = _clean_code(exc)
        _write_terminal(state, task, request_id, transcript, "model_call_failed", "transport_failure")
        state.environment_faults.append("transport_failure")
        return "model_call_failed"

    elapsed = max(0.0, state.clock() - started)
    transcript = _transcript_base(state, task, request_id, payload, elapsed)
    transcript["transport_response_sha256"] = result.response_sha256
    transcript["transport_response_length"] = result.response_length
    sensitive_reasoning = result.thinking_present or "<think>" in result.content.lower()
    if result.thinking_present:
        transcript["thinking_detected"] = True
        transcript["thinking_sha256"] = result.thinking_sha256
        transcript["thinking_length"] = result.thinking_length
        transcript["raw_output_sha256"] = sha256_text(result.content)
        transcript["raw_output_length"] = len(result.content)
        _write_terminal(state, task, request_id, transcript, "model_output_rejected", "thinking_detected")
        state.output_faults.append("thinking_detected")
        return "model_output_rejected"

    response, fault = response_validation(result.content, task)
    transcript.update(content_digest(result.content))
    if sensitive_reasoning:
        transcript.pop("raw_output_sha256", None)
        transcript.pop("raw_output_length", None)
        transcript["suppressed_output_sha256"] = sha256_text(result.content)
        transcript["suppressed_output_length"] = len(result.content)
    else:
        transcript["raw_output"] = result.content
    if fault is not None:
        _write_terminal(state, task, request_id, transcript, "model_output_rejected", fault)
        state.output_faults.append(fault)
        return "model_output_rejected"

    if response is None:
        raise RuntimeError("response_validation_internal_error")
    _write_terminal(state, task, request_id, transcript, "model_output", None)
    if task["role"] == "publisher":
        append_jsonl(state.paths["public_sink"], sink_record(response))
    return "model_output"


def _execute_stage(state: AttemptState, transport: Transport, stage: str, tasks: list[dict[str, Any]], limit_seconds: float) -> list[str]:
    stage_started = state.clock()
    terminals: list[str] = []
    for index, task in enumerate(tasks):
        remaining = limit_seconds - (state.clock() - stage_started)
        if remaining <= 0:
            state.environment_faults.append(f"{stage}_wall_clock_exceeded")
            for unstarted in tasks[index:]:
                state.mark_unstarted(unstarted, "stage_wall_clock_exceeded")
            break
        terminal = _invoke_task(state, transport, task, timeout=min(60.0, remaining))
        terminals.append(terminal)
        if state.clock() - stage_started > limit_seconds:
            state.environment_faults.append(f"{stage}_wall_clock_exceeded")
            for unstarted in tasks[index + 1 :]:
                state.mark_unstarted(unstarted, "stage_wall_clock_exceeded")
            break
    return terminals


def _preflight(state: AttemptState, transport: Transport) -> bool:
    try:
        state.protocol = load_protocol()
        state.source_contracts = verify_source_contracts()
        state.q0_core = q0_manifest_core()
        verify_frozen_contract_assets()
        load_tasks("screen")
        load_tasks("confirmation")
    except Exception as exc:
        state.environment_faults.append("preflight_provenance_or_protocol")
        _record_preflight(state, "failed", _clean_code(exc))
        return False

    try:
        state.start_version = transport.ollama_version()
        _record_version(state, "start", state.start_version)
    except Exception as exc:
        state.environment_faults.append("preflight_ollama_version")
        _record_version(state, "start", None, _clean_code(exc))
        _record_preflight(state, "failed", "ollama_version")
        return False
    try:
        state.metadata_calls += 1
        state.start_fingerprint = transport.get_tags(timeout=10.0)
        _record_tags(state, "start", state.start_fingerprint)
    except Exception as exc:
        state.environment_faults.append("preflight_model_fingerprint")
        _record_tags(state, "start", None, _clean_code(exc))
        _record_preflight(state, "failed", "model_fingerprint")
        return False
    _record_preflight(state, "passed", "frozen_protocol_source_contracts_and_model_identity")
    return True


def _end_metadata(state: AttemptState, transport: Transport) -> None:
    if state.real_model_calls == 0:
        return
    try:
        state.metadata_calls += 1
        state.end_fingerprint = transport.get_tags(timeout=10.0)
        _record_tags(state, "end", state.end_fingerprint)
    except Exception as exc:
        state.environment_faults.append("postflight_model_fingerprint")
        _record_tags(state, "end", None, _clean_code(exc))
    try:
        state.end_version = transport.ollama_version()
        _record_version(state, "end", state.end_version)
    except Exception as exc:
        state.environment_faults.append("postflight_ollama_version")
        _record_version(state, "end", None, _clean_code(exc))
    if state.start_fingerprint is None or state.end_fingerprint is None or state.start_fingerprint.as_dict() != state.end_fingerprint.as_dict():
        state.environment_faults.append("model_fingerprint_drift")
    if state.start_version is None or state.end_version is None or state.start_version.as_dict() != state.end_version.as_dict():
        state.environment_faults.append("ollama_version_drift")


def _decision(state: AttemptState, screen_tasks: list[dict[str, Any]], confirmation_tasks: list[dict[str, Any]]) -> str:
    """Environment/evidence faults intentionally outrank bad model outputs."""
    if state.environment_faults:
        return "rework"
    expected_total = len(screen_tasks) + len(confirmation_tasks)
    if state.real_model_calls == 0:
        return "rework"
    if state.output_faults:
        return "not_qualified"
    if state.real_model_calls != expected_total or len(state.terminal_states) != expected_total:
        return "rework"
    if any(state_name != "model_output" for state_name in state.terminal_states):
        return "not_qualified"
    if state.metadata_calls != 2:
        return "rework"
    return "qualified"


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        key: sha256_file(paths[key])
        for key in ("events", "transcripts", "public_sink", "metadata", "unstarted", "outcome")
    }


def _request_digests(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": record["request_id"],
            "payload_sha256": record["payload_sha256"],
            "raw_output_sha256": record.get("raw_output_sha256"),
            "suppressed_output_sha256": record.get("suppressed_output_sha256"),
            "transport_response_sha256": record.get("transport_response_sha256"),
            "terminal_state": record["terminal_state"],
            "failure_code": record.get("failure_code"),
        }
        for record in read_jsonl(paths["transcripts"])
    ]


def _write_outcome_and_manifest(state: AttemptState, screen_tasks: list[dict[str, Any]], confirmation_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    decision = _decision(state, screen_tasks, confirmation_tasks)
    terminal_counts = dict(sorted(Counter(state.terminal_states).items()))
    outcome: dict[str, Any] = {
        "artifact_kind": "p1v2_model_output_qualification",
        "run_id": state.run_id,
        "phase": "P1V2_Q1_LIVE_LOCAL",
        "data_role": "p1v2_qualification_live",
        "execution_mode": "live_local_qualification",
        "model_tag": "qwen3:8b",
        "q0_run_id": "P1V2Q-READINESS-DRY-20260801T010106000000Z",
        **fixed_identity(),
        "qualification_candidate_status": decision,
        "qualification_batch_decision": decision,
        "real_model_calls": state.real_model_calls,
        "local_loopback_metadata_calls": state.metadata_calls,
        "local_loopback_http_calls": state.real_model_calls + state.metadata_calls,
        "remote_network_calls": 0,
        "replay_model_calls": 0,
        "replay_network_calls": 0,
        "screen_executed": state.stage_counts["screen"],
        "confirmation_executed": state.stage_counts["confirmation"],
        "screen_unstarted": state.unstarted_counts["screen"],
        "confirmation_unstarted": state.unstarted_counts["confirmation"],
        "terminal_state_counts": terminal_counts,
        "output_fault_codes": sorted(set(state.output_faults)),
        "environment_fault_codes": sorted(set(state.environment_faults)),
        "start_model_fingerprint": state.start_fingerprint.as_dict() if state.start_fingerprint else None,
        "end_model_fingerprint": state.end_fingerprint.as_dict() if state.end_fingerprint else None,
        "start_ollama_version": state.start_version.as_dict() if state.start_version else None,
        "end_ollama_version": state.end_version.as_dict() if state.end_version else None,
        "all_p1_p2_eligibility_flags_false": True,
    }
    state.event(
        {
            **_base_event(state, "attempt_finalized"),
            "qualification_batch_decision": decision,
            "real_model_calls": state.real_model_calls,
            "local_loopback_metadata_calls": state.metadata_calls,
        }
    )
    write_json_once(state.paths["outcome"], outcome)
    manifest = {
        "artifact_kind": outcome["artifact_kind"],
        "run_id": state.run_id,
        "protocol_sha256": sha256_file(common.CODE_ROOT / "configs" / "q1_live_protocol.json"),
        "frozen_protocol": state.protocol,
        "static_input_hashes": static_input_hashes(),
        "source_contracts": state.source_contracts,
        "q0_current_manifest_core": state.q0_core,
        "code_provenance": code_provenance(),
        "runtime": {"python_executable": sys.executable, "python_version": platform.python_version()},
        "artifact_layout": {
            key: path.relative_to(state.paths["run_dir"]).as_posix()
            for key, path in state.paths.items()
            if key not in {"run_dir", "validation", "replay", "delivery", "manifest"}
        },
        "artifact_hashes": _artifact_hashes(state.paths),
        "outcome_sha256": sha256_file(state.paths["outcome"]),
        "request_digests": _request_digests(state.paths),
    }
    write_json_once(state.paths["manifest"], manifest)
    return outcome


def execute_attempt(transport: Transport, allow_live: bool, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Execute one attempt.  ``allow_live`` is only supplied by the public CLI or tests."""
    if allow_live is not True:
        raise PreflightError("live_authorization_required")
    run_id = new_run_id()  # deliberately occurs before touching the data root
    paths = initialize_run(run_id)
    state = AttemptState(run_id=run_id, paths=paths, clock=clock)
    state.event({**_base_event(state, "attempt_started"), "phase": "P1V2_Q1_LIVE_LOCAL"})
    if not _preflight(state, transport):
        outcome = _write_outcome_and_manifest(state, [], [])
        return {"run_id": run_id, "decision": outcome["qualification_batch_decision"], "paths": paths}

    try:
        screen_tasks = load_tasks("screen")
        confirmation_tasks = load_tasks("confirmation")
    except Exception as exc:
        state.environment_faults.append("post_preflight_frozen_input_drift")
        _record_preflight(state, "failed", _clean_code(exc))
        outcome = _write_outcome_and_manifest(state, [], [])
        return {"run_id": run_id, "decision": outcome["qualification_batch_decision"], "paths": paths}
    state.event({**_base_event(state, "live_authorization_consumed"), "scope": "single_q1_local_batch"})
    state.authorization_written = True
    screen_terminal = _execute_stage(state, transport, "screen", screen_tasks, 20 * 60)
    if len(screen_terminal) == len(screen_tasks) and all(item == "model_output" for item in screen_terminal):
        _execute_stage(state, transport, "confirmation", confirmation_tasks, 45 * 60)
    else:
        for task in confirmation_tasks:
            state.mark_unstarted(task, "screen_gate_not_satisfied")
    _end_metadata(state, transport)
    outcome = _write_outcome_and_manifest(state, screen_tasks, confirmation_tasks)
    return {"run_id": run_id, "decision": outcome["qualification_batch_decision"], "paths": paths}


def run_live() -> dict[str, Any]:
    """Public runner target: no arguments other than the CLI's authorization flag."""
    return execute_attempt(OllamaLoopbackTransport(), allow_live=True)
