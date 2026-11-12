"""Fail-closed validation for untrusted qualification artifacts."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from p1_benign.event_contracts import validate_event_payload_contract
from p1_benign.workflow import PRIMARY_OUTCOME_FIELDS, recompute_outcome_from_events

from .core import (
    CANDIDATE_MODELS,
    CONDITION,
    DEFAULT_CONTEXT,
    EXECUTION_MODE,
    PHASE,
    QualificationContext,
    SCHEMA_VERSION,
    TASK_ORDER,
    capture_provenance,
    load_static_inputs,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    validate_run_id,
    write_json,
    write_text,
)
from .replay import replay_run


EVENT_FIELDS = {
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
CORE_ARTIFACTS = ("events", "outcomes", "replay", "run_report")


def _failure(run_id: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE,
        "passed": False,
        "checks": {"untrusted_artifact_structure_safe": False},
        "errors": errors or ["untrusted_artifact_processing"],
    }


def _output_path_valid(path: Any, context: QualificationContext) -> bool:
    if not isinstance(path, str) or not path:
        return False
    candidate = (context.outputs.data_root.parent / path).resolve()
    try:
        candidate.relative_to(context.outputs.root.resolve())
    except ValueError:
        return False
    return True


def _event_checks(events: list[dict[str, Any]], run_id: str) -> list[str]:
    errors: list[str] = []
    previous = 0
    for event in events:
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            errors.append("event envelope is malformed")
            continue
        if event.get("run_id") != run_id or event.get("phase") != PHASE:
            errors.append("event run or phase label is inconsistent")
        if event.get("condition") != CONDITION or event.get("risk_seed_present") is not False or event.get("intervention_applied") is not False or event.get("eligible_for_scientific_analysis") is not False:
            errors.append("event condition or analysis labels are invalid")
        if event.get("execution_mode") not in {EXECUTION_MODE, "test_double"}:
            errors.append("event execution mode is invalid")
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence != previous + 1:
            errors.append("event sequence is non-contiguous")
        previous = sequence if isinstance(sequence, int) else previous
        payload = event.get("payload")
        if not isinstance(payload, dict):
            errors.append("event payload is not an object")
            continue
        if event.get("payload_hash") != stable_hash(payload):
            errors.append("event payload hash does not match")
        identity = {
            "run_id": run_id,
            "episode_id": event.get("episode_id"),
            "sequence": event.get("sequence"),
            "event_type": event.get("event_type"),
            "payload_hash": event.get("payload_hash"),
        }
        if event.get("event_id") != f"P1Q-EVT-{stable_hash(identity)[:24]}":
            errors.append("event ID does not match the canonical event identity")
        contract_payload = copy.deepcopy(payload)
        if event.get("event_type") == "episode_finished" and isinstance(contract_payload.get("outcome"), dict):
            contract_payload["outcome"]["phase"] = "P1_PHASE_A"
        contract_errors = validate_event_payload_contract(
            event.get("event_type"),
            contract_payload,
            role_id=event.get("role_id"),
            source_agent=event.get("source_agent"),
            target_agent=event.get("target_agent"),
        )
        if contract_errors:
            errors.append("event payload contract violation")
    return errors


def _report(result: dict[str, Any]) -> str:
    lines = [f"# Qualification validation: {result['run_id']}", ""]
    lines.append(f"- passed: `{str(result['passed']).lower()}`")
    for name, passed in sorted(result.get("checks", {}).items()):
        lines.append(f"- {name}: `{str(passed).lower()}`")
    if result.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    return "\n".join(lines) + "\n"


def validate_run(
    run_id: str,
    *,
    context: QualificationContext = DEFAULT_CONTEXT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    try:
        validate_run_id(run_id)
        paths = context.outputs.artifact_paths(run_id)
        manifest = read_json(paths["manifest"])
        events = read_jsonl(paths["events"])
        outcomes = read_jsonl(paths["outcomes"])
        stored_replay = read_json(paths["replay"])
        if not isinstance(manifest, dict) or not isinstance(stored_replay, dict):
            raise ValueError("manifest or replay root is not an object")
        static = load_static_inputs(context)
        errors: list[str] = []
        labels = {
            "run_id": run_id,
            "phase": PHASE,
            "condition": CONDITION,
            "risk_seed_present": False,
            "intervention_applied": False,
            "execution_mode": manifest.get("execution_mode"),
            "eligible_for_scientific_analysis": False,
            "scientific_gate": "NOT_STARTED",
        }
        for key, expected in labels.items():
            if manifest.get(key) != expected:
                errors.append(f"manifest label mismatch: {key}")
        budget = manifest.get("budget")
        if budget != static.qualification["budget"]:
            errors.append("manifest budget differs from the frozen budget")
        provider = manifest.get("provider")
        if not isinstance(provider, dict) or provider.get("model_id") not in CANDIDATE_MODELS + ("deterministic-p1-v1",):
            errors.append("manifest provider is not an approved candidate or test double")
        if manifest.get("execution_mode") == EXECUTION_MODE:
            if not isinstance(provider, dict) or provider.get("endpoint_identifier") != "loopback_ollama":
                errors.append("local qualification provider is not loopback Ollama")
        if manifest.get("task_instance_ids") != list(TASK_ORDER):
            errors.append("manifest task order is not the frozen three-task order")
        output_files = manifest.get("output_files")
        if not isinstance(output_files, dict) or set(output_files) != {
            "events", "outcomes", "manifest", "replay", "validation", "run_report", "validation_report"
        } or not all(_output_path_valid(value, context) for value in output_files.values()):
            errors.append("manifest output paths leave the qualification data boundary")
        artifact_hashes = manifest.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(CORE_ARTIFACTS):
            errors.append("manifest core artifact hash map is malformed")
        else:
            for name in CORE_ARTIFACTS:
                if artifact_hashes.get(name) != sha256_file(paths[name]):
                    errors.append(f"artifact hash mismatch: {name}")
        if manifest.get("provenance") != capture_provenance(context):
            errors.append("source, static input, reused Phase A, or environment provenance drift")
        errors.extend(_event_checks(events, run_id))
        if len(outcomes) != 3 or any(not isinstance(row, dict) for row in outcomes):
            errors.append("outcome collection must contain exactly three objects")
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("episode_id"), str):
                by_episode[event["episode_id"]].append(event)
        outcome_by_id = {
            row.get("episode_id"): row for row in outcomes if isinstance(row, dict) and isinstance(row.get("episode_id"), str)
        }
        sandbox_ids: set[str] = set()
        for number, task_id in enumerate(TASK_ORDER, start=1):
            episode_id = f"{run_id}-E{number:03d}"
            recorded = outcome_by_id.get(episode_id)
            episode_events = by_episode.get(episode_id, [])
            if recorded is None:
                errors.append("missing outcome for frozen task order")
                continue
            if recorded.get("task_instance_id") != task_id or recorded.get("phase") != PHASE:
                errors.append("outcome task or phase label is inconsistent")
            started = next((row for row in episode_events if row.get("event_type") == "episode_started"), None)
            if not isinstance(started, dict) or not isinstance(started.get("payload"), dict):
                errors.append("episode has no valid initial state evidence")
            else:
                sandbox = started["payload"].get("sandbox_instance_id")
                if not isinstance(sandbox, str) or sandbox in sandbox_ids:
                    errors.append("episode sandbox isolation evidence is invalid")
                sandbox_ids.add(sandbox)
            try:
                recomputed = recompute_outcome_from_events(
                    recorded_outcome=recorded,
                    episode_events=episode_events,
                    expected_report=static.expected_reports[task_id]["report"],
                    internal_record=static.internal_records[task_id],
                )
                if any(recorded.get(key) != recomputed[key] for key in PRIMARY_OUTCOME_FIELDS):
                    errors.append("authoritative tool-state endpoints contradict stored outcome")
            except Exception:
                errors.append("authoritative endpoint recomputation failed")
        fresh_replay = replay_run(run_id, context=context, write_output=False)
        if stored_replay != fresh_replay:
            errors.append("stored replay differs from a fresh transcript replay")
        if fresh_replay.get("passed") is not True or len(fresh_replay.get("episodes", [])) != 3:
            errors.append("fresh transcript replay did not match all three episodes")
        checks = {
            "manifest_labels_frozen": not any(error.startswith("manifest label") for error in errors),
            "budget_and_provider_frozen": not any("budget" in error or "provider" in error for error in errors),
            "output_boundary_safe": not any("output paths" in error for error in errors),
            "artifact_hashes_match": not any(error.startswith("artifact hash") for error in errors),
            "provenance_matches": not any("provenance drift" in error for error in errors),
            "event_contracts_and_hashes_match": not any(error.startswith("event ") for error in errors),
            "endpoints_recomputed_from_tool_state": not any("authoritative" in error for error in errors),
            "state_isolation_evidenced": not any("sandbox" in error for error in errors),
            "fresh_replay_matches": not any("replay" in error for error in errors),
            "untrusted_artifact_structure_safe": True,
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": PHASE,
            "passed": not errors,
            "checks": checks,
            "errors": errors,
        }
    except Exception as exc:
        result = _failure(run_id, [f"untrusted_artifact_processing:{type(exc).__name__}"])
    if write_outputs:
        try:
            paths = context.outputs.artifact_paths(run_id)
            write_json(paths["validation"], result, context.outputs.data_root)
            write_text(paths["validation_report"], _report(result), context.outputs.data_root)
        except Exception:
            pass
    return result
