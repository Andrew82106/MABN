"""Fail-closed, state-backed action authorization."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class PermissionDenied(RuntimeError):
    pass


_REQUIRED_CAPABILITY_BY_KIND: Mapping[str, str] = MappingProxyType({
    "set_value": "state.write",
    "append_value": "state.write",
    "send_message": "message.send",
    "isolate_agent": "defense.isolate",
    "cut_edge": "defense.cut_edge",
    "revoke_permission": "defense.revoke",
    "rollback_state": "defense.rollback",
})


class PermissionEnforcer:
    def is_allowed(self, state: Mapping[str, Any], actor: str, capability: str) -> bool:
        if actor not in state.get("active_agents", []):
            return False
        allowed = state.get("permissions", {}).get(actor, [])
        return capability in allowed or "*" in allowed

    def require(self, state: Mapping[str, Any], actor: str, capability: str) -> None:
        if not self.is_allowed(state, actor, capability):
            raise PermissionDenied(f"{actor!r} lacks capability {capability!r}")

    def require_action(
        self,
        state: Mapping[str, Any],
        actor: str,
        kind: str,
        claimed_capability: str,
    ) -> None:
        """Bind an action kind to its kernel-required capability before execution."""

        required = _REQUIRED_CAPABILITY_BY_KIND.get(kind)
        if required is None:
            raise PermissionDenied(f"Unknown action kind {kind!r}")
        if claimed_capability != required:
            raise PermissionDenied(
                f"Action kind {kind!r} requires capability {required!r}; "
                f"claimed {claimed_capability!r}"
            )
        self.require(state, actor, required)
