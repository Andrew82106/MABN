"""Deterministic configurable round scheduler."""

from __future__ import annotations

from typing import Mapping

from .kernel import Kernel
from .models import Agent


class Scheduler:
    def __init__(self, kernel: Kernel, agents: Mapping[str, Agent]) -> None:
        self.kernel = kernel
        self.agents = dict(agents)

    def run(self) -> None:
        schedule = self.kernel.scenario.config["schedule"]
        for _round in range(int(schedule.get("max_rounds", 1))):
            for agent_id in schedule["order"]:
                if agent_id not in self.kernel.state.get("active_agents", []):
                    continue
                agent = self.agents.get(agent_id)
                if agent is None:
                    continue
                for action in agent.act(self.kernel.observation_for(agent_id)):
                    if action.actor != agent_id:
                        raise ValueError("Agent returned an action for a different actor")
                    self.kernel.dispatch(action)
        self.kernel.finish()
