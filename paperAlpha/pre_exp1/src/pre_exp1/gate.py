"""Frozen Original/Safe/Drop message gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .event_log import AppendOnlyEventLog
from .graph import GraphConfig
from .models import Message, TreatmentName
from .state import stable_hash
from .templates import TemplateRegistry


@dataclass(frozen=True)
class GateResult:
    message: Message
    requested_treatment: TreatmentName
    applied_treatment: TreatmentName
    input_hash: str
    output_hash: str


class MessageGate:
    def __init__(
        self,
        *,
        graph: GraphConfig,
        safe_template: dict[str, Any],
        drop_template: dict[str, Any],
        version: str,
    ) -> None:
        self.graph = graph
        self.gated_edge = graph.gated_edge_id
        self.safe_template = copy.deepcopy(safe_template)
        self.drop_template = copy.deepcopy(drop_template)
        self.version = version

    @classmethod
    def from_runtime(
        cls,
        graph: GraphConfig,
        treatments: dict[str, Any],
        templates: TemplateRegistry,
    ) -> "MessageGate":
        return cls(
            graph=graph,
            safe_template=templates.get(
                treatments["treatments"]["safe"]["template_id"]
            ).content,
            drop_template=templates.get(
                treatments["treatments"]["drop"]["template_id"]
            ).content,
            version=str(treatments["version"]),
        )

    def apply(
        self,
        message: Message,
        treatment: TreatmentName,
        *,
        event_log: AppendOnlyEventLog,
        episode_id: str,
        state_hash: str,
    ) -> GateResult:
        self.graph.validate_message(message)
        input_hash = stable_hash(message.content)
        applied: TreatmentName = treatment
        content = copy.deepcopy(message.content)
        if message.edge_id != self.gated_edge:
            applied = "original"
        elif treatment == "safe":
            content = copy.deepcopy(self.safe_template)
        elif treatment == "drop":
            content = copy.deepcopy(self.drop_template)
        elif treatment != "original":
            raise ValueError(f"Unknown treatment: {treatment}")
        output = Message(
            message_id=message.message_id,
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            edge_id=message.edge_id,
            content=content,
        )
        output_hash = stable_hash(output.content)
        event_log.append(
            episode_id=episode_id,
            event_type="message_gate",
            agent_id=message.sender_id,
            edge_id=message.edge_id,
            treatment=treatment,
            payload=output.content,
            state_before_hash=state_hash,
            state_after_hash=state_hash,
            details={
                "gate_version": self.version,
                "requested_treatment": treatment,
                "applied_treatment": applied,
                "input_content_hash": input_hash,
                "output_content_hash": output_hash,
                "input_content": copy.deepcopy(message.content),
                "output_content": copy.deepcopy(output.content),
                "targeted_edge": message.edge_id == self.gated_edge,
            },
        )
        return GateResult(
            message=output,
            requested_treatment=treatment,
            applied_treatment=applied,
            input_hash=input_hash,
            output_hash=output_hash,
        )
