"""Read-only semantic replay of persisted Q2-B evidence, without a live client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import DATA_ROOT, PACKAGE_ROOT, load_json, validate_run_id
from .protocol import RESPONSE_FIELDS, load_frozen_assets, validate_task_card


def _expected(task: dict[str, Any]) -> dict[str, str]:
    return {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }


def replay_run(run_id: str, *, asset_root: Path = PACKAGE_ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_run_id(run_id)
        assets = load_frozen_assets(asset_root)
        transcript = load_json(data_root / "runs" / run_id / "transcript.json")
        outcome = load_json(data_root / "runs" / run_id / "outcome.json")
    except Exception:
        return {"passed": False, "run_id": run_id, "errors": ["replay_inputs_unreadable"]}
    if not isinstance(transcript, dict) or not isinstance(outcome, dict) or not isinstance(transcript.get("records"), list):
        return {"passed": False, "run_id": run_id, "errors": ["replay_shape"]}
    records = transcript["records"]
    expected_tasks = assets.screen_tasks + assets.confirmation_tasks
    expected_by_id = {task["episode_id"]: task for task in expected_tasks}
    if len(records) != len(expected_tasks):
        return {"passed": False, "run_id": run_id, "errors": ["replay_record_count"]}
    seen: set[str] = set()
    model_failures = False
    infrastructure_failure = False
    attempted = 0
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("task_card"), dict):
            errors.append("replay_record_shape")
            continue
        task = record["task_card"]
        try:
            validate_task_card(task)
        except Exception:
            errors.append("replay_task_card")
            continue
        task_id = task["episode_id"]
        if task_id in seen or expected_by_id.get(task_id) != task:
            errors.append("replay_episode")
            continue
        seen.add(task_id)
        status = record.get("status")
        if status == "completed":
            attempted += 1
            response = record.get("response")
            if not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS) or response != _expected(task):
                errors.append("replay_response")
        elif status == "failed":
            attempted += 1
            if record.get("failure_class") == "model_output":
                model_failures = True
            elif record.get("failure_class") == "infrastructure":
                infrastructure_failure = True
            else:
                errors.append("replay_failure_class")
        elif status != "unstarted":
            errors.append("replay_status")
    if seen != set(expected_by_id):
        errors.append("replay_episodes")
    gate = records[:8]
    remainder = records[8:32]
    confirmation = records[32:]
    gate_ok = all(record.get("status") == "completed" for record in gate)
    remainder_ok = all(record.get("status") == "completed" for record in remainder)
    if not gate_ok and any(record.get("status") != "unstarted" for record in remainder + confirmation):
        errors.append("replay_gate_blocking")
    if gate_ok and not remainder_ok and any(record.get("status") != "unstarted" for record in confirmation):
        errors.append("replay_screen_blocking")
    identity = transcript.get("identity")
    identity_ok = (
        isinstance(identity, dict)
        and identity.get("consistent") is True
        and isinstance(identity.get("start"), dict)
        and isinstance(identity.get("end"), dict)
        and identity["start"].get("status") == "accepted"
        and identity["end"].get("status") == "accepted"
        and identity["start"].get("target_model_present") is True
        and identity["end"].get("target_model_present") is True
        and identity["start"].get("normalized_models_sha256") == identity["end"].get("normalized_models_sha256")
    )
    if not identity_ok:
        infrastructure_failure = True
    if outcome.get("infrastructure_failure") != "none":
        infrastructure_failure = True
    if infrastructure_failure:
        derived = "rework"
    elif model_failures:
        derived = "not_qualified"
    elif all(record.get("status") == "completed" for record in records):
        derived = "qualified"
    else:
        derived = "rework"
    counts = outcome.get("transport_counts")
    if not isinstance(counts, dict) or counts.get("completion_calls") != attempted or counts.get("retry_count") != 0:
        errors.append("replay_counts")
    if outcome.get("decision") != derived:
        errors.append("replay_decision")
    return {"passed": not errors, "run_id": run_id, "errors": sorted(set(errors)), "derived_decision": derived, "replayed_attempted_records": attempted}
