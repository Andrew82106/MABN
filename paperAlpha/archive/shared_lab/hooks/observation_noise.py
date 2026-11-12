"""Deterministic field masking, event omission, and label flipping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lab_kernel.models import Action


@dataclass(frozen=True)
class ObservationNoise:
    rules: Sequence[Mapping[str, Any]]

    def before_action(self, action: Action, context: Mapping[str, Any]) -> Action | None:
        del context
        return action

    def transform_observation(self, agent_id: str, observation: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        del context
        result = copy.deepcopy(dict(observation))
        for rule in self.rules:
            if rule.get("agent_id") not in (None, agent_id):
                continue
            mode = rule["mode"]
            if mode == "drop_field":
                result.pop(rule["field"], None)
            elif mode == "drop_inbox_matching":
                key, value = rule["key"], rule["value"]
                result["inbox"] = [item for item in result.get("inbox", []) if item.get(key) != value]
            elif mode == "replace_value":
                result[rule["field"]] = copy.deepcopy(rule["value"])
            else:
                raise ValueError(f"Unknown observation noise mode: {mode}")
        return result
