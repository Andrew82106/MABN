"""Deterministic offline test agent."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model_agent import ModelTransportError
from .models import Action


@dataclass
class StubAgent:
    scripted_actions: Sequence[Sequence[Action]]
    calls: int = field(default=0, init=False)

    def act(self, observation: Mapping[str, Any]) -> Sequence[Action]:
        del observation
        index = self.calls
        self.calls += 1
        if index >= len(self.scripted_actions):
            return ()
        return tuple(self.scripted_actions[index])


@dataclass
class FakeModelTransport:
    scripted_outcomes: Sequence[Mapping[str, Any] | BaseException]
    calls: int = field(default=0, init=False)
    requests: list[Mapping[str, Any]] = field(default_factory=list, init=False)
    timeout_values: list[float] = field(default_factory=list, init=False)

    def complete(
        self, request: Mapping[str, Any], *, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.requests.append(copy.deepcopy(dict(request)))
        self.timeout_values.append(timeout_seconds)
        index = self.calls
        self.calls += 1
        if index >= len(self.scripted_outcomes):
            raise ModelTransportError("No scripted fake response")
        outcome = self.scripted_outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return copy.deepcopy(dict(outcome))
