"""Read-only semantic replay of a persisted Q2-A mock evidence package.

Replay reconstructs state-machine consequences solely from persisted task cards and
accepted response records; it neither contacts nor imports an execution client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import DATA_ROOT, PACKAGE_ROOT, load_json, validate_run_id
from .protocol import RESPONSE_FIELDS, load_frozen_assets, validate_task_card, verify_identity_profile


def _expected_response(task: dict[str, Any]) -> dict[str, str]:
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
    if not isinstance(transcript, dict) or not isinstance(outcome, dict):
        return {"passed": False, "run_id": run_id, "errors": ["replay_shape"]}
    records = transcript.get("records")
    expected_tasks = assets.screen_tasks + assets.confirmation_tasks
    expected_by_id = {task["episode_id"]: task for task in expected_tasks}
    if not isinstance(records, list) or len(records) != len(expected_tasks):
        return {"passed": False, "run_id": run_id, "errors": ["replay_record_count"]}
    seen: set[str] = set()
    screen_complete = True
    all_complete = True
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
            if not isinstance(response, dict) or set(response) != set(RESPONSE_FIELDS) or response != _expected_response(task):
                errors.append("replay_response")
        elif status == "failed":
            attempted += 1
            all_complete = False
        elif status == "unstarted":
            all_complete = False
        else:
            errors.append("replay_status")
        if task["phase"] == "screen" and status != "completed":
            screen_complete = False
    if seen != set(expected_by_id):
        errors.append("replay_episodes")
    confirmation_records = [record for record in records if isinstance(record, dict) and isinstance(record.get("task_card"), dict) and record["task_card"].get("phase") == "confirmation"]
    if not screen_complete and any(record.get("status") != "unstarted" for record in confirmation_records):
        errors.append("replay_confirmation_blocking")
    if transcript.get("screen_ok") is not screen_complete:
        errors.append("replay_screen_state")
    if transcript.get("confirmation_started") is not screen_complete:
        errors.append("replay_confirmation_state")
    identities_ok = False
    identities = transcript.get("identity_evidence")
    if isinstance(identities, dict):
        try:
            if identities["preflight"]["status"] == "accepted" and identities["postflight"]["status"] == "accepted":
                verify_identity_profile(identities["preflight"]["identity"], assets)
                verify_identity_profile(identities["postflight"]["identity"], assets)
                identities_ok = True
        except Exception:
            errors.append("replay_identity")
    else:
        errors.append("replay_identity")
    derived_decision = "ready_for_q2b" if screen_complete and all_complete and identities_ok else "rework"
    if outcome.get("decision") != derived_decision:
        errors.append("replay_decision")
    counts = outcome.get("call_counts")
    if not isinstance(counts, dict) or counts.get("mock_inference_calls") != attempted:
        errors.append("replay_count")
    if isinstance(counts, dict) and any(counts.get(key) != 0 for key in ("real_network_calls", "real_model_calls", "real_credential_reads", "real_environment_reads")):
        errors.append("replay_real_io")
    return {
        "passed": not errors,
        "run_id": run_id,
        "errors": sorted(set(errors)),
        "derived_decision": derived_decision,
        "replayed_attempted_records": attempted,
    }
