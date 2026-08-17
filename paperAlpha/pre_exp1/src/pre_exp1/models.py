"""Lightweight standard-library data models for P0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TreatmentName = Literal["original", "safe", "drop"]


@dataclass(frozen=True)
class Permission:
    agent_id: str
    tool_name: str


@dataclass(frozen=True)
class Agent:
    agent_id: str
    role: str
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    allowed: bool = True
    gated: bool = False


@dataclass(frozen=True)
class Message:
    message_id: str
    sender_id: str
    recipient_id: str
    edge_id: str
    content: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    agent_id: str
    idempotency_key: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Event:
    run_id: str
    episode_id: str
    event_id: str
    event_type: str
    timestamp: str
    agent_id: str | None
    edge_id: str | None
    treatment: str | None
    payload_hash: str
    state_before_hash: str
    state_after_hash: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentCondition:
    scenario: str
    treatment: TreatmentName


@dataclass
class RunManifest:
    run_id: str
    experiment_stage: str
    config_hashes: dict[str, str]
    shared_input_hashes: dict[str, str]
    message_template_hashes: dict[str, str]
    source_file_hashes: dict[str, str]
    workspace_state: dict[str, Any]
    python_environment: dict[str, Any]
    random_seed: int
    started_at: str
    ended_at: str | None
    status: str
    output_files: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    run_id: str
    episode_id: str
    scenario: str
    treatment: TreatmentName
    leak_detected: bool
    task_success: bool
    task_status: str
    final_state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
