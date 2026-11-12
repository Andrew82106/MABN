"""Transcript-only replay for the scientific P1 pilot."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from p1_benign.providers import TranscriptProvider
from p1_benign.workflow import execute_episode

from .core import (
    CONDITION,
    DATA_ROLE_DRY,
    DATA_ROLE_PILOT,
    DEFAULT_CONTEXT,
    EXECUTION_DRY,
    PHASE,
    ScientificContext,
    ScientificEventRecorder,
    load_static_inputs,
    labels,
    normalize_outcome,
    read_json,
    read_jsonl,
    safe_error,
    stable_hash,
    validate_run_id,
    write_json,
)


def _failed(run_id: str, errors: list[str], data_role: str | None = None) -> dict[str, Any]:
    role_labels = labels(data_role) if data_role in {DATA_ROLE_DRY, DATA_ROLE_PILOT} else {"data_role": data_role, "eligible_for_scientific_analysis": False, "scientific_analysis_scope": None, "eligible_for_p1_gate_analysis": False, "eligible_for_confirmatory_analysis": False, "eligible_for_causal_effect_analysis": False, "scientific_gate": None}
    return {"schema_version": "1.0.0", "run_id": run_id, "phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, **role_labels, "replay_method": "recorded_observable_outputs", "network_calls": 0, "live_provider_calls": 0, "episodes": [], "errors": errors, "passed": False}


def _actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: row["sequence"]):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("model event payload is malformed")
        if event.get("event_type") == "model_output":
            actions.append({"role_id": event.get("role_id"), "phase": payload.get("phase"), "response": copy.deepcopy(payload.get("parsed_response")), "latency_ms": payload.get("latency_ms"), "token_usage": copy.deepcopy(payload.get("token_usage")), "finish_reason": payload.get("finish_reason"), "reasoning_discarded": payload.get("reasoning_discarded"), "failed": False})
        elif event.get("event_type") == "model_call_failed":
            actions.append({"role_id": event.get("role_id"), "phase": payload.get("phase"), "failed": True, "error_type": payload.get("error_type")})
    return actions


def _trace(events: list[dict[str, Any]], *, recorded_endpoint: str | None = None) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda row: row.get("sequence", -1)), 1):
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            raise ValueError("trace event is malformed")
        value = copy.deepcopy(event)
        for key in ("event_id", "recorded_at", "sequence", "payload_hash"):
            value.pop(key, None)
        if value.get("event_type") == "model_call_requested" and recorded_endpoint is not None:
            value["payload"]["endpoint_identifier"] = recorded_endpoint
        if value.get("event_type") == "episode_finished" and isinstance(value["payload"].get("outcome"), dict):
            value["payload"]["outcome"].pop("episode_wall_duration_ms", None)
        value["episode_sequence"] = index
        trace.append(value)
    return trace


def _outcome(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    result.pop("episode_wall_duration_ms", None)
    return result


def replay_run(run_id: str, *, context: ScientificContext = DEFAULT_CONTEXT, write_output: bool = True) -> dict[str, Any]:
    try:
        validate_run_id(run_id)
        paths = context.outputs.artifact_paths(run_id)
        manifest, events, outcomes = read_json(paths["manifest"]), read_jsonl(paths["events"]), read_jsonl(paths["outcomes"])
        if not isinstance(manifest, dict) or manifest.get("phase") != PHASE or manifest.get("condition") != CONDITION:
            raise ValueError("scientific manifest labels are malformed")
        data_role = manifest.get("data_role")
        if data_role not in {DATA_ROLE_DRY, DATA_ROLE_PILOT}:
            raise ValueError("scientific data role is malformed")
        static = load_static_inputs(context)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[event["episode_id"]].append(event)
        outcome_by_id = {row["episode_id"]: row for row in outcomes}
        episodes: list[dict[str, Any]] = []
        for number, task_id in enumerate(static.task_ids, 1):
            episode_id = f"{run_id}-E{number:03d}"
            episode_events = grouped[episode_id]
            recorded = outcome_by_id[episode_id]
            actions = _actions(episode_events)
            recorder = ScientificEventRecorder(run_id=run_id, execution_mode=manifest["execution_mode"], data_role=data_role)
            replayed = None
            error = None
            try:
                provider = TranscriptProvider(provider_id=manifest["model"]["provider_id"], model_id=manifest["model"]["model_id"], decoding=manifest["model"]["decoding"], actions=actions)
                replayed = normalize_outcome(execute_episode(run_id=run_id, episode_number=number, task_instance_id=task_id, material=static.materials[task_id], internal_record=static.internal_records[task_id], expected_report=static.expected_reports[task_id]["report"], static=static, provider=provider, recorder=recorder), manifest["execution_mode"], data_role)
                exhausted = provider.exhausted
            except Exception as exc:
                error, exhausted = type(exc).__name__, False
            recorded_endpoint = next((event.get("payload", {}).get("endpoint_identifier") for event in episode_events if event.get("event_type") == "model_call_requested"), None)
            recorded_trace, replayed_trace = _trace(episode_events), _trace(recorder.events, recorded_endpoint=recorded_endpoint)
            matched = bool(error is None and exhausted and recorded_trace == replayed_trace and _outcome(recorded) == _outcome(replayed))
            episodes.append({"episode_id": episode_id, "task_instance_id": task_id, "matched": matched, "recorded_outcome_hash": stable_hash(_outcome(recorded)), "replayed_outcome_hash": stable_hash(_outcome(replayed) if replayed else {"error": error}), "recorded_trace_hash": stable_hash(recorded_trace), "replayed_trace_hash": stable_hash(replayed_trace), "transcript_model_actions": len(actions), "pure_action_count": sum(row.get("event_type") in {"model_output", "model_call_failed", "message_sent", "tool_call"} for row in recorder.events), "live_provider_calls": 0, "error": error})
        result = {"schema_version": "1.0.0", "run_id": run_id, "phase": PHASE, "condition": CONDITION, "risk_seed_present": False, "intervention_applied": False, **labels(data_role), "replay_method": "recorded_observable_outputs", "network_calls": 0, "live_provider_calls": 0, "transcript_model_actions": sum(row["transcript_model_actions"] for row in episodes), "pure_action_count": sum(row["pure_action_count"] for row in episodes), "episodes": episodes, "errors": [row["error"] for row in episodes if row["error"]], "passed": len(episodes) == 20 and all(row["matched"] for row in episodes)}
    except Exception as exc:
        result = _failed(run_id, [safe_error(exc)], locals().get("data_role"))
    if write_output:
        try:
            write_json(context.outputs.artifact_paths(run_id)["replay"], result, context.outputs.data_root)
        except Exception:
            pass
    return result
