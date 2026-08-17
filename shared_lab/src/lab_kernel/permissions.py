"""Fail-closed, state-backed action authorization."""

from __future__ import annotations

from typing import Any, Mapping


class PermissionDenied(RuntimeError):
    pass


class PermissionEnforcer:
    def is_allowed(self, state: Mapping[str, Any], actor: str, capability: str) -> bool:
        if actor not in state.get("active_agents", []):
            return False
        allowed = state.get("permissions", {}).get(actor, [])
        return capability in allowed or "*" in allowed

    def require(self, state: Mapping[str, Any], actor: str, capability: str) -> None:
        if not self.is_allowed(state, actor, capability):
            raise PermissionDenied(f"{actor!r} lacks capability {capability!r}")
