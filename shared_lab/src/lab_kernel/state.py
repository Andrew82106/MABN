"""Pure deterministic state transitions used by execution and replay."""

from __future__ import annotations

import copy
from typing import Any, MutableMapping, Sequence

from .canonical import content_hash
from .models import Action


class StateTransitionError(RuntimeError):
    pass


def state_hash(state: dict[str, Any]) -> str:
    return content_hash(state)


def _parent(state: MutableMapping[str, Any], path: Sequence[str]) -> tuple[MutableMapping[str, Any], str]:
    if not path:
        raise StateTransitionError("State path must not be empty")
    cursor: MutableMapping[str, Any] = state
    for part in path[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise StateTransitionError(f"Non-mapping state path component: {part}")
        cursor = child
    return cursor, path[-1]


def apply_action(state: dict[str, Any], action: Action) -> dict[str, Any]:
    """Return a new state; never mutate the caller's object."""

    result = copy.deepcopy(state)
    params = dict(action.params)
    if action.kind == "set_value":
        parent, key = _parent(result, list(params["path"]))
        parent[key] = copy.deepcopy(params["value"])
    elif action.kind == "append_value":
        parent, key = _parent(result, list(params["path"]))
        target = parent.setdefault(key, [])
        if not isinstance(target, list):
            raise StateTransitionError("append_value target is not a list")
        target.append(copy.deepcopy(params["value"]))
    elif action.kind == "send_message":
        result.setdefault("messages", []).append(copy.deepcopy(params["message"]))
    elif action.kind == "isolate_agent":
        target = str(params["agent_id"])
        result["active_agents"] = [item for item in result.get("active_agents", []) if item != target]
        result["edges"] = [
            edge for edge in result.get("edges", [])
            if edge.get("source") != target and edge.get("target") != target
        ]
    elif action.kind == "cut_edge":
        edge_id = str(params["edge_id"])
        result["edges"] = [edge for edge in result.get("edges", []) if edge.get("id") != edge_id]
    elif action.kind == "revoke_permission":
        target = str(params["agent_id"])
        capability = str(params["capability"])
        current = result.setdefault("permissions", {}).get(target, [])
        result["permissions"][target] = [item for item in current if item != capability]
    elif action.kind == "rollback_state":
        snapshot = copy.deepcopy(params["snapshot"])
        if content_hash(snapshot) != params["snapshot_hash"]:
            raise StateTransitionError("Rollback snapshot hash mismatch")
        result = snapshot
    else:
        raise StateTransitionError(f"Unknown action kind: {action.kind}")
    return result
