"""Transcript replay using recorded observable model actions only."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from .configuration import load_static_inputs
from .core import (
    CONDITION,
    INTERVENTION_APPLIED,
    PHASE,
    RISK_SEED_PRESENT,
    SCHEMA_VERSION,
    EventRecorder,
    ProjectContext,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now,
    validate_run_id,
    write_json,
)
from .providers import TranscriptProvider
from .workflow import execute_episode


TRACE_NORMALIZATION_VERSION = "p1-observable-trace-v2"
TRACE_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "run_id",
        "episode_id",
        "task_instance_id",
        "sequence",
        "recorded_at",
        "phase",
        "condition",
        "risk_seed_present",
        "intervention_applied",
        "execution_mode",
        "eligible_for_scientific_analysis",
        "event_type",
        "role_id",
        "source_agent",
        "target_agent",
        "payload",
        "payload_hash",
    }
)
TRACE_EXCLUDED_FIELDS = frozenset(
    {
        # Independently validated derived identifiers/hashes.
        "event_id",
        "payload_hash",
        # Independently validated global order and nondeterministic clock.
        "sequence",
        "recorded_at",
    }
)
TRACE_PROJECTED_FIELDS = tuple(
    sorted(TRACE_EVENT_FIELDS - TRACE_EXCLUDED_FIELDS - {"payload"})
)


def _replay_structure_errors(
    manifest: Any,
    events: Any,
    outcomes: Any,
) -> list[str]:
    """Return type prerequisites for deterministic transcript traversal."""

    errors: list[str] = []

    def reject(path: str) -> None:
        errors.append(f"untrusted_artifact_structure:{path}")

    def mapping(value: Any, path: str) -> bool:
        if not isinstance(value, dict):
            reject(path)
            return False
        return True

    def sequence(value: Any, path: str) -> bool:
        if not isinstance(value, list):
            reject(path)
            return False
        return True

    def require_message(value: Any, path: str) -> None:
        if not mapping(value, path):
            return
        if not isinstance(value.get("message_id"), str):
            reject(f"{path}.message_id")
        mapping(value.get("content"), f"{path}.content")
        parents = value.get("parent_message_ids")
        if sequence(parents, f"{path}.parent_message_ids"):
            if not all(isinstance(parent, str) for parent in parents):
                reject(f"{path}.parent_message_ids[]")

    if not mapping(manifest, "manifest"):
        return errors
    if not sequence(events, "events"):
        return errors
    if not sequence(outcomes, "outcomes"):
        return errors

    provider = manifest.get("provider")
    if mapping(provider, "manifest.provider"):
        mapping(provider.get("decoding"), "manifest.provider.decoding")
        for field in ("provider_id", "model_id", "endpoint_identifier"):
            if not isinstance(provider.get(field), str):
                reject(f"manifest.provider.{field}")
    task_ids = manifest.get("task_instance_ids")
    if sequence(task_ids, "manifest.task_instance_ids") and not all(
        isinstance(task_id, str) for task_id in task_ids
    ):
        reject("manifest.task_instance_ids[]")
    if not isinstance(manifest.get("execution_mode"), str):
        reject("manifest.execution_mode")
    if not isinstance(
        manifest.get("eligible_for_scientific_analysis"),
        bool,
    ):
        reject("manifest.eligible_for_scientific_analysis")

    for index, event in enumerate(events):
        base = f"events[{index}]"
        if not mapping(event, base):
            continue
        if not isinstance(event.get("sequence"), int) or isinstance(
            event.get("sequence"),
            bool,
        ):
            reject(f"{base}.sequence")
        if not isinstance(event.get("episode_id"), str):
            reject(f"{base}.episode_id")
        event_type = event.get("event_type")
        if not isinstance(event_type, str):
            reject(f"{base}.event_type")
            continue
        payload = event.get("payload")
        if not mapping(payload, f"{base}.payload"):
            continue
        if event_type in {"model_output", "model_call_failed"}:
            if not isinstance(payload.get("phase"), str):
                reject(f"{base}.payload.phase")
            if event.get("role_id") is not None and not isinstance(
                event.get("role_id"),
                str,
            ):
                reject(f"{base}.role_id")
        if event_type in {"task_received", "message_sent"}:
            require_message(payload, f"{base}.payload")
        elif event_type == "model_output":
            response = payload.get("parsed_response")
            if mapping(
                payload.get("parsed_response"),
                f"{base}.payload.parsed_response",
            ):
                for field in ("content", "arguments"):
                    if field in response:
                        mapping(
                            response.get(field),
                            f"{base}.payload.parsed_response.{field}",
                        )
            mapping(
                payload.get("token_usage"),
                f"{base}.payload.token_usage",
            )
        elif event_type == "episode_finished":
            embedded = payload.get("outcome")
            if not mapping(embedded, f"{base}.payload.outcome"):
                continue
            mapping(
                embedded.get("role_durations_ms"),
                f"{base}.payload.outcome.role_durations_ms",
            )
        elif event_type == "model_call_requested":
            mapping(payload.get("decoding"), f"{base}.payload.decoding")
            visible = payload.get("visible_messages")
            if sequence(visible, f"{base}.payload.visible_messages"):
                for message_index, visible_message in enumerate(visible):
                    require_message(
                        visible_message,
                        (
                            f"{base}.payload.visible_messages"
                            f"[{message_index}]"
                        ),
                    )
        elif event_type == "tool_call":
            mapping(payload.get("arguments"), f"{base}.payload.arguments")
            mapping(payload.get("result"), f"{base}.payload.result")

    for index, outcome in enumerate(outcomes):
        base = f"outcomes[{index}]"
        if not mapping(outcome, base):
            continue
        if not isinstance(outcome.get("episode_id"), str):
            reject(f"{base}.episode_id")
        if not isinstance(outcome.get("final_state_hash"), str):
            reject(f"{base}.final_state_hash")
        mapping(
            outcome.get("role_durations_ms"),
            f"{base}.role_durations_ms",
        )
        for field in (
            "required_field_accuracy",
            "provider_call_count",
            "structured_output_count",
            "publication_count",
            "total_provider_duration_ms",
            "episode_wall_duration_ms",
        ):
            if not isinstance(outcome.get(field), (int, float)):
                reject(f"{base}.{field}")
    return errors


def _failed_replay_result(
    run_id: str,
    *,
    manifest: Any = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, zero-call replay failure document."""

    manifest_mapping = manifest if isinstance(manifest, dict) else {}
    started_at = manifest_mapping.get("started_at")
    execution_mode = manifest_mapping.get("execution_mode")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE,
        "condition": CONDITION,
        "risk_seed_present": RISK_SEED_PRESENT,
        "intervention_applied": INTERVENTION_APPLIED,
        "execution_mode": (
            execution_mode
            if isinstance(execution_mode, str)
            else "invalid_artifact"
        ),
        "eligible_for_scientific_analysis": False,
        "replayed_at": (
            started_at if isinstance(started_at, str) else utc_now()
        ),
        "replay_method": "recorded_observable_outputs",
        "trace_normalization_version": TRACE_NORMALIZATION_VERSION,
        "network_calls": 0,
        "live_provider_calls": 0,
        "transcript_model_actions": 0,
        "pure_action_count": 0,
        "episodes": [],
        "errors": errors or ["untrusted_artifact_processing"],
        "passed": False,
    }


def _transcript_actions(
    episode_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for event in sorted(episode_events, key=lambda item: item["sequence"]):
        if event["event_type"] == "model_output":
            payload = event["payload"]
            actions.append(
                {
                    "role_id": event["role_id"],
                    "phase": payload["phase"],
                    "response": copy.deepcopy(payload["parsed_response"]),
                    "latency_ms": payload["latency_ms"],
                    "token_usage": copy.deepcopy(payload["token_usage"]),
                    "finish_reason": payload.get("finish_reason"),
                    "reasoning_discarded": payload.get(
                        "reasoning_discarded",
                        False,
                    ),
                    "failed": False,
                }
            )
        elif event["event_type"] == "model_call_failed":
            payload = event["payload"]
            actions.append(
                {
                    "role_id": event["role_id"],
                    "phase": payload["phase"],
                    "failed": True,
                    "error_type": payload["error_type"],
                }
            )
    return actions


def _comparable_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(outcome)
    value.pop("episode_wall_duration_ms", None)
    return value


def _normalized_trace(
    episode_events: list[dict[str, Any]],
    *,
    replay_endpoint_mapping: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize an explicit allowlist of nondeterministic metadata.

    ``recorded_at`` and global ``sequence`` are validated independently;
    derived event/payload hashes are independently recomputed. Every other
    frozen envelope field is projected explicitly. Exact input key checking
    makes a future field fail closed until this normalization is revised.
    """

    trace: list[dict[str, Any]] = []
    for episode_sequence, event in enumerate(
        sorted(episode_events, key=lambda item: item["sequence"]),
        start=1,
    ):
        if set(event) != TRACE_EVENT_FIELDS:
            raise ValueError(
                "Event envelope differs from trace projection contract"
            )
        payload = copy.deepcopy(event["payload"])
        if (
            event["event_type"] == "model_call_requested"
            and replay_endpoint_mapping is not None
            and payload.get("endpoint_identifier")
            == replay_endpoint_mapping[0]
        ):
            # Transcript replay uses one known in-memory transport. Map only
            # that exact identifier back to the recorded provider identifier,
            # preserving every recorded endpoint value as tamper evidence.
            payload["endpoint_identifier"] = replay_endpoint_mapping[1]
        if event["event_type"] == "episode_finished":
            embedded_outcome = payload.get("outcome")
            if not isinstance(embedded_outcome, dict):
                raise ValueError(
                    "episode_finished outcome must be an object"
                )
            embedded_outcome.pop("episode_wall_duration_ms", None)
        projected = {
            field: copy.deepcopy(event[field])
            for field in TRACE_PROJECTED_FIELDS
        }
        projected["episode_sequence"] = episode_sequence
        projected["payload"] = payload
        trace.append(projected)
    return trace


def _replay_run_inner(
    run_id: str,
    *,
    context: ProjectContext,
    write_output: bool = True,
) -> dict[str, Any]:
    """Replay a P1 transcript with zero test/local/remote model calls."""

    validate_run_id(run_id)
    paths = context.outputs.artifact_paths(run_id)
    manifest = read_json(paths["manifest"])
    events = read_jsonl(paths["events"])
    outcomes = read_jsonl(paths["outcomes"])
    structure_errors = _replay_structure_errors(
        manifest,
        events,
        outcomes,
    )
    if structure_errors:
        result = _failed_replay_result(
            run_id,
            manifest=manifest,
            errors=structure_errors,
        )
        if write_output:
            write_json(paths["replay"], result)
        return result
    static = load_static_inputs(context)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_episode[event["episode_id"]].append(event)
    outcome_by_episode = {
        outcome["episode_id"]: outcome for outcome in outcomes
    }

    episode_results: list[dict[str, Any]] = []
    total_pure_actions = 0
    total_transcript_actions = 0
    for episode_number, task_id in enumerate(
        manifest["task_instance_ids"],
        start=1,
    ):
        episode_id = f"{run_id}-E{episode_number:03d}"
        recorded = outcome_by_episode[episode_id]
        episode_events = by_episode[episode_id]
        actions: list[dict[str, Any]] = []
        memory_recorder = EventRecorder(
            run_id=run_id,
            execution_mode=manifest["execution_mode"],
            eligible_for_scientific_analysis=manifest[
                "eligible_for_scientific_analysis"
            ],
        )
        replayed: dict[str, Any] | None = None
        replay_error_type: str | None = None
        try:
            actions = _transcript_actions(episode_events)
            provider = TranscriptProvider(
                provider_id=manifest["provider"]["provider_id"],
                model_id=manifest["provider"]["model_id"],
                decoding=manifest["provider"]["decoding"],
                actions=actions,
            )
            replayed = execute_episode(
                run_id=run_id,
                episode_number=episode_number,
                task_instance_id=task_id,
                material=static.materials[task_id],
                internal_record=static.internal_records[task_id],
                expected_report=static.expected_reports[task_id]["report"],
                static=static,
                provider=provider,
                recorder=memory_recorder,
            )
            provider_exhausted = provider.exhausted
        except Exception as exc:
            # Recorded artifacts are untrusted validator input. A malformed
            # transcript must yield a fresh, deterministic failed replay
            # artifact rather than aborting validation or preserving a stale
            # successful replay.
            replay_error_type = type(exc).__name__
            provider_exhausted = False
        try:
            recorded_trace = _normalized_trace(episode_events)
        except Exception as exc:
            replay_error_type = replay_error_type or type(exc).__name__
            recorded_trace = []
        try:
            replayed_trace = _normalized_trace(
                memory_recorder.events,
                replay_endpoint_mapping=(
                    "transcript_replay",
                    manifest["provider"]["endpoint_identifier"],
                ),
            )
        except Exception as exc:
            replay_error_type = replay_error_type or type(exc).__name__
            replayed_trace = []
        trace_matched = (
            replay_error_type is None
            and recorded_trace == replayed_trace
        )
        pure_actions = sum(
            event["event_type"]
            in {
                "model_output",
                "model_call_failed",
                "message_sent",
                "tool_call",
            }
            for event in memory_recorder.events
        )
        matched = (
            replay_error_type is None
            and provider_exhausted
            and trace_matched
            and replayed is not None
            and _comparable_outcome(recorded)
            == _comparable_outcome(replayed)
        )
        replay_failure_fingerprint = {
            "episode_id": episode_id,
            "error_type": replay_error_type or "ReplayMismatch",
            "partial_trace_hash": stable_hash(replayed_trace),
        }
        total_pure_actions += pure_actions
        total_transcript_actions += len(actions)
        episode_results.append(
            {
                "episode_id": episode_id,
                "task_instance_id": task_id,
                "matched": matched,
                "recorded_outcome_hash": stable_hash(
                    _comparable_outcome(recorded)
                ),
                "replayed_outcome_hash": stable_hash(
                    _comparable_outcome(replayed)
                    if replayed is not None
                    else replay_failure_fingerprint
                ),
                "recorded_final_state_hash": recorded[
                    "final_state_hash"
                ],
                "replayed_final_state_hash": (
                    replayed["final_state_hash"]
                    if replayed is not None
                    else stable_hash(replay_failure_fingerprint)
                ),
                "recorded_trace_hash": stable_hash(recorded_trace),
                "replayed_trace_hash": stable_hash(replayed_trace),
                "trace_matched": trace_matched,
                "transcript_model_actions": len(actions),
                "pure_action_count": pure_actions,
                "live_provider_calls": 0,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "phase": manifest["phase"],
        "condition": manifest["condition"],
        "risk_seed_present": manifest["risk_seed_present"],
        "intervention_applied": manifest["intervention_applied"],
        "execution_mode": manifest["execution_mode"],
        "eligible_for_scientific_analysis": manifest[
            "eligible_for_scientific_analysis"
        ],
        # Stable across standalone replay invocations so replay does not
        # invalidate the artifact hash already recorded in the manifest.
        "replayed_at": manifest["started_at"],
        "replay_method": "recorded_observable_outputs",
        "trace_normalization_version": TRACE_NORMALIZATION_VERSION,
        "network_calls": 0,
        "live_provider_calls": 0,
        "transcript_model_actions": total_transcript_actions,
        "pure_action_count": total_pure_actions,
        "episodes": episode_results,
        "passed": bool(episode_results)
        and all(item["matched"] for item in episode_results),
    }
    if write_output:
        write_json(paths["replay"], result)
    return result


def replay_run(
    run_id: str,
    *,
    context: ProjectContext,
    write_output: bool = True,
) -> dict[str, Any]:
    """Replay untrusted artifacts without propagating parser failures."""

    try:
        return _replay_run_inner(
            run_id,
            context=context,
            write_output=write_output,
        )
    except Exception as exc:
        manifest: Any = None
        paths = None
        try:
            validate_run_id(run_id)
            paths = context.outputs.artifact_paths(run_id)
        except Exception:
            pass
        if paths is not None:
            try:
                candidate = read_json(paths["manifest"])
                if isinstance(candidate, dict):
                    manifest = candidate
            except Exception:
                pass
        result = _failed_replay_result(
            run_id,
            manifest=manifest,
            errors=[
                "untrusted_artifact_processing:"
                + type(exc).__name__
            ],
        )
        if write_output and paths is not None:
            try:
                write_json(paths["replay"], result)
            except Exception:
                pass
        return result
