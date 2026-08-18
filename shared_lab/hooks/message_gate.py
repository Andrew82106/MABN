"""Edge/time-selective message replacement, deletion, and safe rewrite."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lab_kernel.models import Action


@dataclass(frozen=True)
class MessageGate:
    rules: Sequence[Mapping[str, Any]]

    def before_action(self, action: Action, context: Mapping[str, Any]) -> Action | None:
        del context
        if action.kind != "send_message":
            return action
        message = dict(action.params["message"])
        for rule in self.rules:
            if rule.get("edge_id") != message.get("edge_id"):
                continue
            if "round" in rule and rule["round"] != message.get("round"):
                continue
            mode = rule["mode"]
            if mode == "drop":
                return None
            changed = copy.deepcopy(message)
            if mode == "replace":
                changed["content"] = copy.deepcopy(rule["content"])
            elif mode == "safe_rewrite":
                replacements = rule.get("replacements", {})
                text = str(changed.get("content", ""))
                for source, target in replacements.items():
                    text = text.replace(source, target)
                changed["content"] = text
            else:
                raise ValueError(f"Unknown message gate mode: {mode}")
            return Action(action.actor, action.capability, action.kind, {"message": changed})
        return action


    def transform_observation(self, agent_id: str, observation: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        del agent_id, context
        return observation
