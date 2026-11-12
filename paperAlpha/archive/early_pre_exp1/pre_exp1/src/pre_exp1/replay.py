"""Deterministic transcript replay using pure state actions only."""

from __future__ import annotations

import copy
import json
from typing import Any

from .event_log import read_events
from .paths import DEFAULT_PATHS, ProjectPaths, require_data_path
from .state import apply_replay_action, stable_hash


def replay_episode(
    events: list[dict[str, Any]],
    episode_id: str,
) -> dict[str, Any]:
    episode_events = [
        event for event in events if event["episode_id"] == episode_id
    ]
    if not episode_events:
        raise ValueError(f"No transcript events for episode {episode_id}")
    starts = [
        event for event in episode_events if event["event_type"] == "episode_started"
    ]
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one episode_started for {episode_id}")
    completions = [
        event
        for event in episode_events
        if event["event_type"] == "episode_completed"
    ]
    if len(completions) != 1:
        raise ValueError(f"Expected exactly one episode_completed for {episode_id}")
    state = copy.deepcopy(starts[0]["details"]["initial_state"])
    actions_applied = 0
    for event in episode_events:
        action = event["details"].get("replay_action")
        if action is not None:
            apply_replay_action(state, action)
            actions_applied += 1
    completion = completions[0]
    expected_hash = completion["details"]["final_state_hash"]
    actual_hash = stable_hash(state)
    return {
        "episode_id": episode_id,
        "scenario": starts[0]["details"]["scenario"],
        "treatment": starts[0]["treatment"],
        "task_status": state["task_status"],
        "matched": actual_hash == expected_hash,
        "expected_final_state_hash": expected_hash,
        "actual_final_state_hash": actual_hash,
        "pure_actions_applied": actions_applied,
        "replayed_state": state,
    }


def replay_run(
    run_id: str,
    *,
    write_output: bool = True,
    paths: ProjectPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    event_path = paths.raw_dir / f"events_{run_id}.jsonl"
    events = read_events(event_path, data_root=paths.data_root)
    episode_ids = list(dict.fromkeys(event["episode_id"] for event in events))
    results = [replay_episode(events, episode_id) for episode_id in episode_ids]
    summary = {
        "run_id": run_id,
        "passed": bool(results) and all(result["matched"] for result in results),
        "execution_mode": "pure_state_actions",
        "pure_actions_applied": sum(
            result["pure_actions_applied"] for result in results
        ),
        "live_tool_calls": 0,
        "episodes": results,
    }
    if write_output:
        output_path = require_data_path(
            paths.interim_dir / f"replay_{run_id}.json",
            paths.data_root,
        )
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    return summary

