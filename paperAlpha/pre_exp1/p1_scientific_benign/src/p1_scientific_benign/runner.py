"""Single-run orchestration for deterministic dry and real P1 gate pilot."""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from p1_benign.configuration import StaticInputs
from p1_benign.workflow import execute_episode

from .core import (
    BUDGET,
    CONDITION,
    DATA_ROLE_DRY,
    DATA_ROLE_PILOT,
    DEFAULT_CONTEXT,
    EXECUTION_DRY,
    EXECUTION_REAL,
    MODEL_DECODING,
    MODEL_ID,
    OUTPUT_ROOT,
    PHASE,
    PHASE_A_ROOT,
    QUALIFICATION_MANIFEST,
    QUALIFICATION_RUN_ID,
    SCIENTIFIC_GATE_DRY,
    SCIENTIFIC_GATE_PILOT,
    ScientificContext,
    ScientificEventRecorder,
    capture_provenance,
    labels,
    load_static_inputs,
    new_run_id,
    normalize_outcome,
    read_json,
    sha256_file,
    stable_hash,
    utc_now,
    validate_run_id,
    write_json,
    write_jsonl,
    write_text,
)
from .providers import (
    DeterministicTestProvider,
    ScientificOllamaProvider,
    frozen_provider_identity,
    inspect_qwen3,
    validate_frozen_preflight,
)
from .reporting import render_gate_report, render_run_report


class _DeadlineProvider:
    """Wrap a provider with a hard per-call remainder of the batch deadline."""

    def __init__(self, provider: Any, *, clock: Any, deadline: float) -> None:
        self._provider = provider
        self._clock = clock
        self._deadline = deadline
        self.deadline_exceeded = False
        self.deadline_failure: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def _fail(self, request: Any, *, remaining: float) -> None:
        self.deadline_exceeded = True
        self.deadline_failure = {
            "failure_stage": "deadline",
            "failure_type": "deadline_exceeded",
            "role_id": getattr(request, "role_id", None),
            "phase": getattr(request, "phase", None),
            "call_index": getattr(request, "call_index", None),
            "remaining_seconds": max(0.0, round(remaining, 6)),
            "provider_calls": getattr(request, "call_index", 0),
            "network_calls": 0,
            "accepted_as_success": False,
        }

    def generate(self, request: Any) -> Any:
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            self._fail(request, remaining=remaining)
            raise TimeoutError("provider call rejected at scientific batch deadline")
        original_timeout = getattr(self._provider, "timeout_seconds", None)
        if isinstance(original_timeout, (int, float)):
            self._provider.timeout_seconds = min(float(original_timeout), remaining)
        try:
            result = self._provider.generate(request)
        except Exception:
            if self._clock() >= self._deadline:
                self._fail(request, remaining=self._deadline - self._clock())
            raise
        finally:
            if isinstance(original_timeout, (int, float)):
                self._provider.timeout_seconds = original_timeout
        if self._clock() >= self._deadline:
            self._fail(request, remaining=self._deadline - self._clock())
            raise TimeoutError("provider call completed after scientific batch deadline")
        return result


def _context_for(context: ScientificContext, data_role: str) -> ScientificContext:
    return context.for_data_role(data_role)


def _relative(path: Path, context: ScientificContext) -> str:
    return path.resolve().relative_to(context.outputs.data_root.parent.resolve()).as_posix()


def _real_manifest_exists(context: ScientificContext) -> bool:
    pilot_manifests = context.outputs.data_root / "pre_exp1" / "p1_scientific_benign" / "pilot" / "manifests"
    return any(pilot_manifests.glob("manifest_P1-BENIGN-PILOT-QWEN3-*.json"))


def _offline_gate_ok(context: ScientificContext) -> bool:
    """Require a current deterministic gate before any future local provider call."""
    dry_context = context.for_data_role(DATA_ROLE_DRY)
    manifests = sorted(dry_context.outputs.manifests.glob("manifest_P1-BENIGN-PILOT-DRY-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in manifests:
        try:
            manifest = read_json(path)
            if not isinstance(manifest, dict) or manifest.get("status") != "completed" or manifest.get("data_role") != DATA_ROLE_DRY:
                continue
            if manifest.get("validation", {}).get("passed") is not True:
                continue
            if manifest.get("provenance") != capture_provenance(context):
                continue
            return True
        except Exception:
            continue
    return False


def _qualification_evidence() -> dict[str, Any]:
    manifest = read_json(QUALIFICATION_MANIFEST)
    if not isinstance(manifest, dict):
        raise ValueError("qualification manifest is malformed")
    if manifest.get("run_id") != QUALIFICATION_RUN_ID or manifest.get("provider", {}).get("model_id") != MODEL_ID:
        raise ValueError("qualification evidence does not identify frozen qwen3:8b")
    if manifest.get("status") != "completed" or manifest.get("validation", {}).get("passed") is not True:
        raise ValueError("qualification evidence is not a completed passing run")
    return {"run_id": QUALIFICATION_RUN_ID, "manifest_sha256": sha256_file(QUALIFICATION_MANIFEST), "metadata_hash": manifest["provider"]["model_inspection"]["models"][MODEL_ID]["metadata_hash"], "endpoint_origin_hash": manifest["provider"]["endpoint_origin_hash"]}


def _preflight(run_id: str, context: ScientificContext, *, real: bool) -> dict[str, Any]:
    if not real:
        return {"run_id": run_id, "performed": False, "reason": "deterministic provider; no Ollama call"}
    qualification = _qualification_evidence()
    observed = inspect_qwen3()
    validate_frozen_preflight(observed)
    if observed["metadata_hash"] != qualification["metadata_hash"] or observed["endpoint_origin_hash"] != qualification["endpoint_origin_hash"]:
        raise ValueError("current qwen preflight differs from qualification evidence")
    return {"run_id": run_id, "performed": True, "qualification": qualification, "observed": observed, "decoding": MODEL_DECODING, "budget": BUDGET}


def _manifest(run_id: str, context: ScientificContext, static: StaticInputs, *, execution_mode: str, data_role: str, provider: Any, preflight: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    provider_identity = {"provider_id": "deterministic_test", "model_id": "deterministic-p1-v1", "endpoint_identifier": "none", "decoding": MODEL_DECODING, "timeout_seconds": None, "concurrency": 1, "retry_per_call": 0, "remote_calls_allowed": False} if execution_mode == EXECUTION_DRY else frozen_provider_identity(provider)
    return {"schema_version": "1.0.0", "run_id": run_id, "phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, "execution_mode": execution_mode, **labels(data_role), "status": "running", "started_at": utc_now(), "completed_at": None, "task_instance_ids": list(static.task_ids), "model": provider_identity, "budget": copy.deepcopy(BUDGET), "qualification_evidence": _qualification_evidence() if execution_mode == EXECUTION_REAL else None, "output_files": {key: _relative(path, context) for key, path in paths.items()}, "artifact_hashes": {}, "provenance": capture_provenance(context), "validation": None, "failure_reason": None}


def run_batch(*, context: ScientificContext = DEFAULT_CONTEXT, execution_mode: str, run_id: str | None = None, allow_local_pilot: bool = False, clock: Any = None) -> dict[str, Any]:
    if execution_mode not in {EXECUTION_DRY, EXECUTION_REAL}:
        raise ValueError("unsupported scientific execution mode")
    data_role = DATA_ROLE_DRY if execution_mode == EXECUTION_DRY else DATA_ROLE_PILOT
    if execution_mode == EXECUTION_REAL:
        if allow_local_pilot is not True:
            raise PermissionError("explicit local P1 pilot authorization is required")
        if _real_manifest_exists(context):
            raise FileExistsError("a real qwen P1 pilot manifest already exists; second run is forbidden")
        if not _offline_gate_ok(context):
            raise PermissionError("current deterministic offline gate evidence is required before local P1 pilot")
    static = load_static_inputs(context)
    actual = run_id or new_run_id("P1-BENIGN-PILOT-DRY-" if execution_mode == EXECUTION_DRY else "P1-BENIGN-PILOT-QWEN3-")
    validate_run_id(actual)
    if execution_mode == EXECUTION_DRY and not actual.startswith("P1-BENIGN-PILOT-DRY-"):
        raise ValueError("dry run ID prefix mismatch")
    if execution_mode == EXECUTION_REAL and not actual.startswith("P1-BENIGN-PILOT-QWEN3-"):
        raise ValueError("real run ID prefix mismatch")
    run_context = _context_for(context, data_role)
    run_context.outputs.ensure()
    paths = run_context.outputs.artifact_paths(actual)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("scientific run artifacts already exist")
    monotonic = clock or time.monotonic
    deadline = monotonic() + float(BUDGET["maximum_wall_time_minutes"]) * 60.0
    raw_provider = DeterministicTestProvider() if execution_mode == EXECUTION_DRY else ScientificOllamaProvider()
    try:
        preflight = _preflight(actual, run_context, real=execution_mode == EXECUTION_REAL)
    except Exception as exc:
        preflight = {
            "schema_version": "1.0.0",
            "run_id": actual,
            "phase": PHASE,
            "condition": CONDITION,
            "risk_seed_present": False,
            "intervention_applied": False,
            **labels(data_role),
            "execution_mode": execution_mode,
            "status": "failed",
            "failure_stage": "preflight",
            "failure_type": type(exc).__name__,
            "reason": str(exc),
            "provider_calls": 0,
            "network_calls": 0,
            "accepted_as_success": False,
        }
        write_json(paths["preflight_report"], preflight, run_context.outputs.data_root)
        manifest = _manifest(actual, run_context, static, execution_mode=execution_mode, data_role=data_role, provider=raw_provider, preflight=preflight, paths=paths)
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["failure_reason"] = preflight
        manifest["artifact_hashes"] = {"preflight_report": sha256_file(paths["preflight_report"])}
        write_json(paths["manifest"], manifest, run_context.outputs.data_root)
        return {
            "run_id": actual,
            "execution_mode": execution_mode,
            "phase": PHASE,
            "data_role": data_role,
            "preflight": preflight,
            "validation": {"passed": False, "errors": ["preflight failed"]},
            "replay": {"passed": False, "episodes": [], "live_provider_calls": 0, "network_calls": 0},
            "provider_calls": 0,
            "network_calls": 0,
            "artifact_paths": {key: _relative(path, run_context) for key, path in paths.items()},
        }
    write_json(paths["preflight_report"], preflight, run_context.outputs.data_root)
    provider = _DeadlineProvider(raw_provider, clock=monotonic, deadline=deadline)
    manifest = _manifest(actual, run_context, static, execution_mode=execution_mode, data_role=data_role, provider=provider, preflight=preflight, paths=paths)
    write_json(paths["manifest"], manifest, run_context.outputs.data_root)
    recorder = ScientificEventRecorder(run_id=actual, execution_mode=execution_mode, data_role=data_role, path=paths["events"], data_root=run_context.outputs.data_root)
    outcomes: list[dict[str, Any]] = []
    try:
        for number, task_id in enumerate(static.task_ids, 1):
            if monotonic() > deadline:
                raise TimeoutError("scientific batch exceeded maximum wall time")
            outcome = execute_episode(run_id=actual, episode_number=number, task_instance_id=task_id, material=static.materials[task_id], internal_record=static.internal_records[task_id], expected_report=static.expected_reports[task_id]["report"], static=static, provider=provider, recorder=recorder)
            outcomes.append(normalize_outcome(outcome, execution_mode, data_role))
            if provider.deadline_exceeded:
                raise TimeoutError("scientific batch exceeded maximum wall time during provider call")
            if monotonic() > deadline:
                raise TimeoutError("scientific batch exceeded maximum wall time")
        write_jsonl(paths["outcomes"], outcomes, run_context.outputs.data_root)
        from .replay import replay_run
        replay = replay_run(actual, context=run_context, write_output=True)
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["artifact_hashes"] = {name: sha256_file(paths[name]) for name in ("events", "outcomes", "replay", "preflight_report")}
        write_text(paths["run_report"], render_run_report(manifest, outcomes, replay, events=recorder.events, static=static), run_context.outputs.data_root)
        manifest["artifact_hashes"]["run_report"] = sha256_file(paths["run_report"])
        write_text(paths["p1_gate_report"], render_gate_report(manifest, outcomes, replay, validation=None, events=recorder.events, static=static), run_context.outputs.data_root)
        manifest["artifact_hashes"]["p1_gate_report"] = sha256_file(paths["p1_gate_report"])
        write_json(paths["manifest"], manifest, run_context.outputs.data_root)
        from .validation import validate_run
        validation = validate_run(actual, context=run_context, write_outputs=True)
        manifest["validation"] = {"passed": validation["passed"], "check_count": len(validation.get("checks", {})), "error_count": len(validation.get("errors", []))}
        write_json(paths["manifest"], manifest, run_context.outputs.data_root)
        validation = validate_run(actual, context=run_context, write_outputs=True)
        write_text(paths["p1_gate_report"], render_gate_report(manifest, outcomes, replay, validation=validation, events=recorder.events, static=static), run_context.outputs.data_root)
        manifest["artifact_hashes"]["p1_gate_report"] = sha256_file(paths["p1_gate_report"])
        write_json(paths["manifest"], manifest, run_context.outputs.data_root)
        validation = validate_run(actual, context=run_context, write_outputs=True)
    except Exception as exc:
        if outcomes:
            write_jsonl(paths["outcomes"], outcomes, run_context.outputs.data_root)
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["failure_reason"] = provider.deadline_failure or {"failure_stage": "runner", "failure_type": type(exc).__name__, "accepted_as_success": False}
        manifest["partial_outcomes"] = len(outcomes)
        manifest["artifact_hashes"] = {key: sha256_file(paths[key]) for key in ("events", "outcomes") if paths[key].exists()}
        write_json(paths["manifest"], manifest, run_context.outputs.data_root)
        raise
    return {"run_id": actual, "execution_mode": execution_mode, "phase": PHASE, "data_role": data_role, "eligible_for_scientific_analysis": labels(data_role)["eligible_for_scientific_analysis"], "scientific_gate": labels(data_role)["scientific_gate"], "outcomes": outcomes, "replay": replay, "validation": validation, "artifact_paths": {key: _relative(path, run_context) for key, path in paths.items()}}
