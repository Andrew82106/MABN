"""Provider-neutral strict JSON model agent with injectable offline transport."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .canonical import content_hash
from .models import AgentRequest
from .permissions import PermissionEnforcer
from .tools import ToolError, ToolRegistry


class ModelTransportError(RuntimeError):
    pass


class ModelActionError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class ModelTransport(Protocol):
    def complete(self, request: Mapping[str, Any], *, timeout_seconds: float) -> Mapping[str, Any]: ...


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ModelActionError("duplicate_key", "Model JSON contains a duplicate key")
        value[key] = item
    return value


def _constant(_: str) -> Any:
    raise ModelActionError("nonfinite", "Model JSON contains a non-finite number")


def parse_model_request(actor: str, raw: Any) -> AgentRequest:
    if not isinstance(raw, str) or not raw.strip():
        raise ModelActionError("non_json", "Model content is not a non-empty JSON string")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except ModelActionError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelActionError("non_json", "Model content is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ModelActionError("multiple_or_non_object", "Model content must be one JSON object")
    if "tool_calls" in value or "function_call" in value:
        raise ModelActionError("native_tool_bypass", "Provider-native tool calls are forbidden")
    if "refusal" in value:
        raise ModelActionError("refusal", "Refusal wrappers are not actions")
    if any(key in value for key in ("reasoning", "analysis", "thought")):
        raise ModelActionError("reasoning_field", "Reasoning fields are forbidden")
    kind = value.get("kind")
    if kind == "send_message":
        expected = {"kind", "edge_id", "recipient", "content"}
        if set(value) != expected:
            raise ModelActionError("malformed_action", "Message action fields are not exact")
        if not all(isinstance(value[key], str) and value[key] for key in ("edge_id", "recipient")):
            raise ModelActionError("malformed_action", "Message route fields must be non-empty strings")
        try:
            json.dumps(value["content"], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ModelActionError("malformed_action", "Message content is not finite JSON") from exc
        return AgentRequest(actor, kind, {
            "edge_id": value["edge_id"],
            "recipient": value["recipient"],
            "content": copy.deepcopy(value["content"]),
        })
    if kind == "call_tool":
        expected = {"kind", "tool_name", "arguments"}
        if set(value) != expected:
            raise ModelActionError("malformed_action", "Tool action fields are not exact")
        if not isinstance(value["tool_name"], str) or not value["tool_name"]:
            raise ModelActionError("malformed_action", "Tool name must be a non-empty string")
        if not isinstance(value["arguments"], Mapping):
            raise ModelActionError("malformed_action", "Tool arguments must be an object")
        return AgentRequest(actor, kind, {
            "tool_name": value["tool_name"], "arguments": copy.deepcopy(dict(value["arguments"]))
        })
    raise ModelActionError("unknown_action", "Unknown or missing model action kind")


def build_model_request(
    observation: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    model = model_config.get("model")
    if not isinstance(model, str) or not model:
        raise ModelActionError("model_configuration", "Frozen model identity is missing")
    role_prompt = observation.get("role_prompt")
    task_prompt = observation.get("task_prompt")
    if not isinstance(role_prompt, str) or not isinstance(task_prompt, str):
        raise ModelActionError("observation", "Role or task prompt is missing")
    projected = {
        key: copy.deepcopy(value) for key, value in observation.items()
        if key not in {"role_prompt", "task_prompt"}
    }
    user_body = json.dumps(
        {"task": task_prompt, "observation": projected},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_body},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }


@dataclass
class ModelAgent:
    actor: str
    scenario: Any
    transport: ModelTransport
    model_config: Mapping[str, Any]
    timeout_seconds: float = 30.0
    artifact_policy: str = "hashes"
    safe_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def _validate_route(self, request: AgentRequest) -> None:
        if request.kind == "send_message":
            params = request.params
            allowed = any(
                edge["id"] == params["edge_id"]
                and edge["source"] == self.actor
                and edge["target"] == params["recipient"]
                for edge in self.scenario.config["graph"]["edges"]
            )
            if not allowed:
                raise ModelActionError("unauthorized_target", "Message target or edge is unauthorized")
            return
        registry = ToolRegistry(self.scenario, PermissionEnforcer())
        try:
            registry.validate_request(
                self.scenario.initial_state(),
                self.actor,
                str(request.params["tool_name"]),
                request.params["arguments"],
            )
        except ToolError as exc:
            raise ModelActionError(exc.category, str(exc)) from exc

    def act(self, observation: Mapping[str, Any]) -> Sequence[AgentRequest]:
        if self.artifact_policy not in {"none", "hashes", "raw"}:
            raise ModelActionError("artifact_policy", "Unknown model artifact policy")
        request = build_model_request(observation, self.model_config)
        request_hash = content_hash(request)
        try:
            response = self.transport.complete(
                copy.deepcopy(request), timeout_seconds=float(self.timeout_seconds)
            )
        except TimeoutError as exc:
            raise ModelActionError("model_timeout", "Model transport timed out") from exc
        except ModelTransportError as exc:
            raise ModelActionError("model_infrastructure", "Model transport failed") from exc
        if not isinstance(response, Mapping):
            raise ModelActionError("model_response", "Transport response is not an object")
        if "tool_calls" in response or "function_call" in response:
            raise ModelActionError("native_tool_bypass", "Provider-native tool calls are forbidden")
        if set(response) != {"model", "content"}:
            raise ModelActionError("model_response", "Transport response fields are not exact")
        if response["model"] != self.model_config["model"]:
            raise ModelActionError("model_identity", "Transport returned the wrong model identity")
        agent_request = parse_model_request(self.actor, response["content"])
        self._validate_route(agent_request)
        record = {
            "request_hash": request_hash,
            "response_hash": content_hash(response["content"]),
            "action_kind": agent_request.kind,
        }
        if self.artifact_policy == "raw":
            record["request"] = copy.deepcopy(request)
            record["response"] = copy.deepcopy(dict(response))
        if self.artifact_policy != "none":
            self.safe_records.append(record)
        return (agent_request,)


__all__ = [
    "ModelActionError", "ModelAgent", "ModelTransport", "ModelTransportError",
    "build_model_request", "parse_model_request",
]
