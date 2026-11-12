"""Stable public contracts shared by kernels, agents, and hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Action:
    actor: str
    capability: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "capability": self.capability,
            "kind": self.kind,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Action":
        return cls(
            actor=str(value["actor"]),
            capability=str(value["capability"]),
            kind=str(value["kind"]),
            params=dict(value.get("params", {})),
        )


@dataclass(frozen=True)
class AgentRequest:
    """Untrusted agent intent; authority is deliberately absent."""

    actor: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"actor": self.actor, "kind": self.kind, "params": dict(self.params)}


class Agent(Protocol):
    def act(self, observation: Mapping[str, Any]) -> Sequence[Action | AgentRequest]: ...


class Hook(Protocol):
    def before_action(self, action: Action, context: Mapping[str, Any]) -> Action | None: ...

    def transform_observation(
        self, agent_id: str, observation: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
