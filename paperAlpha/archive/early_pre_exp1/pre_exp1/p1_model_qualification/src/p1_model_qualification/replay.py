"""Transcript-only replay for qualification artifacts."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from p1_benign.providers import TranscriptProvider
from p1_benign.workflow import execute_episode

from .core import (
    CONDITION,
    DEFAULT_CONTEXT,
    EXECUTION_MODE,
    PHASE,
    QualificationContext,
    QualificationEventRecorder,
    SCHEMA_VERSION,
    load_static_inputs,
    normalize_outcome,
    read_json,
    read_jsonl,
    safe_error,
    stable_hash,
    validate_run_id,
    write_json,
)


def _failed(run_id: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE,
        "replay_method": "recorded_observable_outputs",
        "network_calls": 0,
        "live_provider_calls": 0,
        "transcript_model_actions": 0,
        "pure_action_count": 0,
        "episodes": [],
        "errors": errors,
        "passed": False,
    }


def _actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: row["sequence"]):
        payload = event["payload"]
        if event["event_type"] == "model_output":
            result.append(
                {
                    "role_id": event["role_id"],
                    "phase": payload["phase"],
                    "response": copy.deepcopy(payload["parsed_response"]),
                    "latency_ms": payload["latency_ms"],
                    "token_usage": copy.deepcopy(payload["token_usage"]),
                    "finish_reason": payload["finish_reason"],
                    "reasoning_discarded": payload["reasoning_discarded"],
                    "failed": False,
                }
            )
        elif event["event_type"] == "model_call_failed":
            result.append(
                {
                    "role_id": event["role_id"],
                    "phase": payload["phase"],
                    "failed": True,
                    "error_type": payload["error_type"],
                }
            )
    return result


def _normal_trace(
    events: list[dict[str, Any]],
    *,
    replay_endpoint: str | None = None,
    recorded_endpoint: str | None = None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda row: row["sequence"]), start=1):
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            raise ValueError("trace event is malformed")
        value = copy.deepcopy(event)
        value.pop("event_id", None)
        value.pop("recorded_at", None)
        value.pop("sequence", None)
        value.pop("payload_hash", None)
        payload = value["payload"]
        if (
            value.get("event_type") == "model_call_requested"
            and replay_endpoint is not None
            and payload.get("endpoint_identifier") == replay_endpoint
        ):
            payload["endpoint_identifier"] = recorded_endpoint
        if value.get("event_type") == "episode_finished":
            outcome = payload.get("outcome")
            if not isinstance(outcome, dict):
                raise ValueError("episode_finished has no outcome")
            outcome.pop("episode_wall_duration_ms", None)
        value["episode_sequence"] = index
        trace.append(value)
    return trace


def _outcome(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result = copy.deepcopy(value)
    result.pop("episode_wall_duration_ms", None)
    return result


def replay_run(
    run_id: str,
    *,
    context: QualificationContext = DEFAULT_CONTEXT,
    write_output: bool = True,
) -> dict[str, Any]:
    try:
        validate_run_id(run_id)
        paths = context.outputs.artifact_paths(run_id)
        manifest = read_json(paths["manifest"])
        events = read_jsonl(paths["events"])
        outcomes = read_jsonl(paths["outcomes"])
        if not isinstance(manifest, dict) or not isinstance(manifest.get("provider"), dict):
            raise ValueError("manifest is malformed")
        if manifest.get("phase") != PHASE or manifest.get("condition") != CONDITION:
            raise ValueError("manifest labels do not describe qualification")
        if manifest.get("execution_mode") not in {EXECUTION_MODE, "test_double"}:
            raise ValueError("manifest execution mode is invalid")
        static = load_static_inputs(context)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[event["episode_id"]].append(event)
        outcome_by_id = {outcome["episode_id"]: outcome for outcome in outcomes}
        episodes: list[dict[str, Any]] = []
        for number, task_id in enumerate(static.task_ids, start=1):
            episode_id = f"{run_id}-E{number:03d}"
            episode_events = grouped[episode_id]
            recorded = outcome_by_id[episode_id]
            actions = _actions(episode_events)
            recorder = QualificationEventRecorder(
                run_id=run_id,
                execution_mode=manifest["execution_mode"],
            )
            replayed: dict[str, Any] | None = None
            error: str | None = None
            try:
                provider = TranscriptProvider(
                    provider_id=manifest["provider"]["provider_id"],
                    model_id=manifest["provider"]["model_id"],
                    decoding=manifest["provider"]["decoding"],
                    actions=actions,
                )
                replayed = normalize_outcome(
                    execute_episode(
                        run_id=run_id,
                        episode_number=number,
                        task_instance_id=task_id,
                        material=static.materials[task_id],
                        internal_record=static.internal_records[task_id],
                        expected_report=static.expected_reports[task_id]["report"],
                        static=static.workflow,
                        provider=provider,
                        recorder=recorder,
                    ),
                    manifest["execution_mode"],
                )
                exhausted = provider.exhausted
            except Exception as exc:
                error = type(exc).__name__
                exhausted = False
            recorded_trace = _normal_trace(episode_events)
            replayed_trace = _normal_trace(
                recorder.events,
                replay_endpoint="transcript_replay",
                recorded_endpoint=manifest["provider"]["endpoint_identifier"],
            )
            matched = bool(
                error is None
                and exhausted
                and recorded_trace == replayed_trace
                and _outcome(recorded) == _outcome(replayed)
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "task_instance_id": task_id,
                    "matched": matched,
                    "recorded_outcome_hash": stable_hash(_outcome(recorded)),
                    "replayed_outcome_hash": stable_hash(_outcome(replayed) if replayed else {"error": error}),
                    "recorded_trace_hash": stable_hash(recorded_trace),
                    "replayed_trace_hash": stable_hash(replayed_trace),
                    "transcript_model_actions": len(actions),
                    "pure_action_count": sum(
                        row["event_type"] in {"model_output", "model_call_failed", "message_sent", "tool_call"}
                        for row in recorder.events
                    ),
                    "live_provider_calls": 0,
                    "error": error,
                }
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": PHASE,
            "condition": CONDITION,
            "risk_seed_present": False,
            "intervention_applied": False,
            "execution_mode": manifest["execution_mode"],
            "eligible_for_scientific_analysis": False,
            "replayed_at": manifest.get("started_at"),
            "replay_method": "recorded_observable_outputs",
            "network_calls": 0,
            "live_provider_calls": 0,
            "transcript_model_actions": sum(row["transcript_model_actions"] for row in episodes),
            "pure_action_count": sum(row["pure_action_count"] for row in episodes),
            "episodes": episodes,
            "errors": [row["error"] for row in episodes if row["error"]],
            "passed": len(episodes) == 3 and all(row["matched"] for row in episodes),
        }
    except Exception as exc:
        result = _failed(run_id, [safe_error(exc)])
    if write_output:
        try:
            write_json(context.outputs.artifact_paths(run_id)["replay"], result, context.outputs.data_root)
        except Exception:
            pass
    return result
