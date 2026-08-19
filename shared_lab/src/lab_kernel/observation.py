"""Default-deny, scenario-declared role observation projection."""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Sequence


class ObservationError(RuntimeError):
    pass


_BASE_KEYS = {
    "agent_id", "role", "role_prompt", "task_prompt", "public", "inbox", "tool_results"
}


def _read_path(state: Mapping[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = state
    for part in path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ObservationError(f"Configured observation path is unavailable: {list(path)!r}")
        cursor = cursor[part]
    return copy.deepcopy(cursor)


def _write_path(target: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = target
    for part in path[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ObservationError(f"Observation paths collide at {part!r}")
        cursor = child
    cursor[path[-1]] = copy.deepcopy(value)


def build_observation(scenario: Any, state: Mapping[str, Any], agent_id: str) -> dict[str, Any]:
    policy = scenario.observation_policy(agent_id)
    public: dict[str, Any] = {}
    for path in policy["state_paths"]:
        _write_path(public, path, _read_path(state, path))
    inbox = [
        copy.deepcopy(item) for item in state.get("messages", [])
        if isinstance(item, Mapping) and item.get("recipient") == agent_id
    ]
    own_results = copy.deepcopy(state.get("tool_results", {}).get(agent_id, []))
    agent = scenario.agent_config(agent_id)
    return {
        "agent_id": agent_id,
        "role": agent["role"],
        "role_prompt": scenario.role_prompt(agent_id),
        "task_prompt": scenario.task_prompt(),
        "public": public,
        "inbox": inbox,
        "tool_results": own_results,
    }


def finalize_hook_observation(
    base: Mapping[str, Any], transformed: Mapping[str, Any], allowed_hook_fields: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(transformed, Mapping):
        raise ObservationError("Observation hook returned a non-mapping")
    allowed_extra = set(allowed_hook_fields)
    unexpected = set(transformed) - _BASE_KEYS - allowed_extra
    if unexpected:
        raise ObservationError(f"Observation hook returned undeclared fields: {sorted(unexpected)}")
    result = {
        key: copy.deepcopy(value) for key, value in base.items()
        if key in transformed
    }
    for key in allowed_extra:
        if key in transformed:
            result[key] = copy.deepcopy(transformed[key])
    return result


__all__ = ["ObservationError", "build_observation", "finalize_hook_observation"]
