"""Scenario-registered, capability-derived controlled tool execution."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import content_hash
from .permissions import PermissionDenied, PermissionEnforcer
from .state import StateTransitionError, apply_effects


class ToolError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ToolExecution:
    tool_name: str
    required_capability: str
    result: Mapping[str, Any]
    result_summary: Mapping[str, Any]
    effects: Sequence[Mapping[str, Any]]
    next_state: Mapping[str, Any]


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "null":
        return value is None
    return False


def validate_parameters(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    expected_type = schema.get("type")
    if not isinstance(expected_type, str) or not _matches_type(value, expected_type):
        raise ToolError("invalid_parameters", f"{path} has the wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError("invalid_parameters", f"{path} is outside the allowed values")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ToolError("invalid_parameters", f"{path} uses an invalid object contract")
        missing = set(required) - set(value)
        if missing:
            raise ToolError("invalid_parameters", f"{path} is missing required fields")
        extra = set(value) - set(properties)
        if extra and schema.get("additionalProperties") is False:
            raise ToolError("invalid_parameters", f"{path} has extra fields")
        for key, child in value.items():
            if key in properties:
                validate_parameters(child, properties[key], f"{path}.{key}")
    if isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            raise ToolError("invalid_parameters", f"{path} uses an invalid array contract")
        for index, child in enumerate(value):
            validate_parameters(child, item_schema, f"{path}[{index}]")


class ToolRegistry:
    def __init__(self, scenario: Any, permissions: PermissionEnforcer) -> None:
        self.scenario = scenario
        self.permissions = permissions
        self._specs = {item["name"]: copy.deepcopy(item) for item in scenario.config["tools"]}
        self.execution_count = 0

    def spec(self, tool_name: str) -> Mapping[str, Any]:
        spec = self._specs.get(tool_name)
        if spec is None:
            raise ToolError("unknown_tool", f"Unknown tool {tool_name!r}")
        return copy.deepcopy(spec)

    def validate_request(
        self, state: Mapping[str, Any], actor: str, tool_name: str, arguments: Any
    ) -> Mapping[str, Any]:
        spec = self.spec(tool_name)
        try:
            self.permissions.require(state, actor, spec["capability"])
        except PermissionDenied as exc:
            raise ToolError("permission_denied", str(exc)) from exc
        validate_parameters(arguments, spec["parameters"])
        return spec

    def execute(
        self, state: Mapping[str, Any], actor: str, tool_name: str, arguments: Any
    ) -> ToolExecution:
        spec = self.validate_request(state, actor, tool_name, arguments)
        handler = self.scenario.load_callable(spec["handler"])
        self.execution_count += 1
        try:
            raw = handler(
                copy.deepcopy(dict(state)),
                copy.deepcopy(dict(arguments)),
                {
                    "actor": actor,
                    "tool_name": tool_name,
                    "scenario": copy.deepcopy(self.scenario.config),
                },
            )
        except Exception as exc:
            raise ToolError("handler_error", f"Tool handler failed: {type(exc).__name__}") from exc
        if not isinstance(raw, Mapping) or set(raw) != {"result", "effects"}:
            raise ToolError("handler_error", "Tool handler returned an invalid outcome")
        result = raw["result"]
        business_effects = raw["effects"]
        if (
            not isinstance(result, Mapping)
            or not all(isinstance(key, str) and key for key in result)
            or not isinstance(business_effects, list)
        ):
            raise ToolError("handler_error", "Tool outcome result or effects are invalid")
        try:
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            json.dumps(business_effects, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolError("handler_error", "Tool result or effects are not finite JSON") from exc
        allowed_effects = set(spec.get("effect_kinds", []))
        for effect in business_effects:
            if (
                not isinstance(effect, Mapping)
                or set(effect) != {"kind", "params"}
                or effect["kind"] not in allowed_effects
                or not isinstance(effect["params"], Mapping)
            ):
                raise ToolError("handler_error", "Tool handler returned an undeclared effect")
        if not spec["mutates_state"] and business_effects:
            raise ToolError("handler_error", "Read-only tool returned business state effects")
        result_entry = {"tool_name": tool_name, "result": copy.deepcopy(dict(result))}
        effects = [copy.deepcopy(dict(item)) for item in business_effects]
        effects.append({
            "kind": "append_value",
            "params": {"path": ["tool_results", actor], "value": result_entry},
        })
        try:
            next_state = apply_effects(state, effects)
        except (KeyError, TypeError, StateTransitionError) as exc:
            raise ToolError("handler_error", "Tool effects are invalid") from exc
        summary = {
            "result_hash": content_hash(result),
            "result_keys": sorted(result),
            "visibility": spec["result_visibility"],
        }
        return ToolExecution(
            tool_name=tool_name,
            required_capability=spec["capability"],
            result=copy.deepcopy(dict(result)),
            result_summary=summary,
            effects=tuple(effects),
            next_state=next_state,
        )


__all__ = ["ToolError", "ToolExecution", "ToolRegistry", "validate_parameters"]
