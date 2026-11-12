"""Episode state creation, cloning, hashing, reset, and pure replay actions."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_initial_state(
    episode_id: str,
    fixture_record: dict[str, Any],
) -> dict[str, Any]:
    """Create an isolated state and instantiate an explicitly fake secret."""

    record = copy.deepcopy(fixture_record)
    template = record.pop("test_secret_template")
    record["test_secret"] = template.format(episode_id=episode_id)
    return {
        "episode_id": episode_id,
        "vendor_records": {record["vendor_id"]: record},
        "messages": [],
        "tool_state": {
            "database_reads": {},
            "publications": {},
        },
        "task_status": "not_started",
        "outcome": None,
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def reset_tool_state(state: dict[str, Any]) -> None:
    state["tool_state"] = {
        "database_reads": {},
        "publications": {},
    }


def _append_unique_message(state: dict[str, Any], message: dict[str, Any]) -> None:
    if not any(
        existing["message_id"] == message["message_id"]
        for existing in state["messages"]
    ):
        state["messages"].append(copy.deepcopy(message))


def apply_replay_action(state: dict[str, Any], action: dict[str, Any]) -> None:
    """Apply a finite, deterministic state transition.

    Actions are idempotent where an idempotency key or object ID is present.
    This same pure function is used by the live stub and transcript replay.
    """

    action_type = action["type"]
    if action_type == "record_message":
        _append_unique_message(state, action["message"])
        return
    if action_type == "record_database_read":
        key = action["idempotency_key"]
        state["tool_state"]["database_reads"].setdefault(
            key,
            {
                "agent_id": action["agent_id"],
                "vendor_id": action["vendor_id"],
                "result": copy.deepcopy(action["result"]),
                "request_fingerprint": action.get("request_fingerprint"),
            },
        )
        return
    if action_type == "record_publication":
        key = action["idempotency_key"]
        state["tool_state"]["publications"].setdefault(
            key,
            {
                "agent_id": action["agent_id"],
                "report": copy.deepcopy(action["report"]),
                "request_fingerprint": action.get("request_fingerprint"),
            },
        )
        return
    if action_type == "set_task_status":
        state["task_status"] = action["status"]
        return
    if action_type == "set_outcome":
        state["outcome"] = copy.deepcopy(action["outcome"])
        return
    if action_type == "reset_tools":
        reset_tool_state(state)
        return
    raise ValueError(f"Unknown replay action: {action_type}")


def states_are_isolated(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Probe nested mutable objects without leaving either state modified."""

    if first is second:
        return False
    marker = {"probe": True}
    first["messages"].append(marker)
    isolated = marker not in second["messages"]
    first["messages"].pop()
    return isolated
