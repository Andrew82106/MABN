"""Deterministic offline test agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
