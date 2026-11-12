"""Code-enforced Agent-to-tool permissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .event_log import AppendOnlyEventLog


class PermissionDenied(RuntimeError):
    pass


class PermissionEnforcer:
    def __init__(self, allowed_tools: dict[str, set[str]]) -> None:
        self._allowed_tools = {
            agent_id: frozenset(tools)
            for agent_id, tools in allowed_tools.items()
        }

    @classmethod
    def from_config(cls, path: Path) -> "PermissionEnforcer":
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(
            {
                agent["agent_id"]: set(agent.get("allowed_tools", []))
                for agent in config["agents"]
            }
        )

    def is_allowed(self, agent_id: str, tool_name: str) -> bool:
        return tool_name in self._allowed_tools.get(agent_id, frozenset())

    def require(
        self,
        agent_id: str,
        tool_name: str,
        *,
        event_log: AppendOnlyEventLog,
        episode_id: str,
        treatment: str,
        payload: Any,
        state_hash: str,
    ) -> None:
        if self.is_allowed(agent_id, tool_name):
            return
        reason = f"Agent {agent_id!r} is not allowed to call {tool_name!r}"
        event_log.append(
            episode_id=episode_id,
            event_type="permission_denied",
            agent_id=agent_id,
            edge_id=None,
            treatment=treatment,
            payload=payload,
            state_before_hash=state_hash,
            state_after_hash=state_hash,
            details={"tool_name": tool_name, "reason": reason},
        )
        raise PermissionDenied(reason)

