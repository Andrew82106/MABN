"""Declarative scenario package loading and validation."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import tree_hash


class ScenarioError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scenario:
    root: Path
    config: Mapping[str, Any]
    tree_sha256: str
    file_hashes: Mapping[str, str]

    def initial_state(self) -> dict[str, Any]:
        state = copy.deepcopy(self.config.get("initial_state", {}))
        agents = self.config["agents"]
        state["active_agents"] = [agent["id"] for agent in agents]
        state["permissions"] = {
            agent["id"]: list(agent.get("capabilities", [])) for agent in agents
        }
        state["edges"] = copy.deepcopy(self.config["graph"]["edges"])
        state.setdefault("messages", [])
        state.setdefault("tool_results", {agent["id"]: [] for agent in agents})
        return state

    def agent_config(self, agent_id: str) -> Mapping[str, Any]:
        for agent in self.config["agents"]:
            if agent["id"] == agent_id:
                return copy.deepcopy(agent)
        raise ScenarioError(f"Unknown agent: {agent_id}")

    def role_prompt(self, agent_id: str) -> str:
        relative = self.agent_config(agent_id)["prompt_file"]
        return (self.root / relative).read_text(encoding="utf-8")

    def task_prompt(self) -> str:
        return (self.root / self.config["task"]["prompt_file"]).read_text(encoding="utf-8")

    def observation_policy(self, agent_id: str) -> Mapping[str, Any]:
        if agent_id not in self.config["observations"]:
            raise ScenarioError(f"No observation policy for agent: {agent_id}")
        return copy.deepcopy(self.config["observations"][agent_id])

    def load_callable(self, reference: Mapping[str, Any]) -> Callable[..., Any]:
        relative = reference["module"]
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ScenarioError("Scenario callable escapes scenario root")
        spec = importlib.util.spec_from_file_location(
            f"scenario_callable_{self.tree_sha256}_{reference['function']}", path
        )
        if spec is None or spec.loader is None:
            raise ScenarioError(f"Cannot load scenario callable: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, reference["function"], None)
        if not callable(function):
            raise ScenarioError("Scenario callable is not callable")
        return function

    def evaluator(self) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
        return self.load_callable(self.config["evaluator"])


def _at_path(value: Mapping[str, Any], path: list[str]) -> Any:
    cursor: Any = value
    for part in path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ScenarioError(f"Observation path is outside initial state: {path!r}")
        cursor = cursor[part]
    return cursor


def _validate_parameter_schema(schema: Any, path: str) -> None:
    if not isinstance(schema, Mapping) or schema.get("type") not in {
        "object", "array", "string", "boolean", "integer", "number", "null"
    }:
        raise ScenarioError(f"Invalid tool parameter schema at {path}")
    if schema["type"] == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if (
            not isinstance(properties, Mapping)
            or not isinstance(required, list)
            or len(required) != len(set(required))
            or set(required) - set(properties)
            or schema.get("additionalProperties") is not False
        ):
            raise ScenarioError(f"Invalid closed object schema at {path}")
        for key, child in properties.items():
            if not isinstance(key, str) or not key:
                raise ScenarioError(f"Invalid parameter name at {path}")
            _validate_parameter_schema(child, f"{path}.{key}")
    if schema["type"] == "array":
        _validate_parameter_schema(schema.get("items"), f"{path}[]")


def _validate(config: Mapping[str, Any], root: Path) -> None:
    required = {
        "schema_version", "scenario_id", "agents", "graph", "schedule", "task",
        "evaluator", "model_bindings", "observations", "tools",
    }
    missing = required - set(config)
    if missing:
        raise ScenarioError(f"Missing scenario fields: {sorted(missing)}")
    agents = config["agents"]
    identifiers = [agent["id"] for agent in agents]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ScenarioError("Agent identifiers must be non-empty and unique")
    known = set(identifiers)
    observations = config["observations"]
    if not isinstance(observations, Mapping) or set(observations) != known:
        raise ScenarioError("Observation policies must name every known agent exactly once")
    initial = config.get("initial_state", {})
    for agent_id, policy in observations.items():
        if not isinstance(policy, Mapping) or set(policy) != {"state_paths", "hook_fields"}:
            raise ScenarioError(f"Invalid observation policy for {agent_id}")
        paths = policy["state_paths"]
        hook_fields = policy["hook_fields"]
        if not isinstance(paths, list) or not isinstance(hook_fields, list):
            raise ScenarioError(f"Invalid observation rule collection for {agent_id}")
        normalized: list[tuple[str, ...]] = []
        for path in paths:
            if (
                not isinstance(path, list)
                or not path
                or not all(isinstance(part, str) and part for part in path)
            ):
                raise ScenarioError(f"Invalid observation path for {agent_id}")
            normalized.append(tuple(path))
            _at_path(initial, path)
        if len(normalized) != len(set(normalized)):
            raise ScenarioError(f"Duplicate observation path for {agent_id}")
        if (
            len(hook_fields) != len(set(hook_fields))
            or not all(isinstance(item, str) and item and item != "state" for item in hook_fields)
        ):
            raise ScenarioError(f"Invalid or duplicate hook field for {agent_id}")
    edge_ids: set[str] = set()
    for edge in config["graph"]["edges"]:
        if edge["id"] in edge_ids or edge["source"] not in known or edge["target"] not in known:
            raise ScenarioError("Graph contains duplicate or invalid edge")
        edge_ids.add(edge["id"])
    if set(config["schedule"]["order"]) - known:
        raise ScenarioError("Schedule contains unknown agents")
    tools = config["tools"]
    if not isinstance(tools, list):
        raise ScenarioError("Tools must be a list")
    names: set[str] = set()
    declared_capabilities = {
        capability for agent in agents for capability in agent.get("capabilities", [])
    }
    for tool in tools:
        expected = {
            "name", "capability", "parameters", "handler", "mutates_state",
            "effect_kinds", "result_visibility",
        }
        if not isinstance(tool, Mapping) or set(tool) != expected:
            raise ScenarioError("Tool declaration has unexpected fields")
        if not isinstance(tool["name"], str) or not tool["name"] or tool["name"] in names:
            raise ScenarioError("Tool names must be non-empty and unique")
        names.add(tool["name"])
        if tool["capability"] not in declared_capabilities:
            raise ScenarioError(f"Tool capability is not assigned: {tool['name']}")
        _validate_parameter_schema(tool["parameters"], f"tool:{tool['name']}")
        if (
            not isinstance(tool["mutates_state"], bool)
            or tool["result_visibility"] != "caller"
            or not isinstance(tool["effect_kinds"], list)
            or len(tool["effect_kinds"]) != len(set(tool["effect_kinds"]))
            or set(tool["effect_kinds"]) - {"set_value", "append_value"}
        ):
            raise ScenarioError(f"Invalid tool effects or visibility: {tool['name']}")
        handler = tool["handler"]
        if (
            not isinstance(handler, Mapping)
            or set(handler) != {"module", "function"}
            or not isinstance(handler["function"], str)
            or not handler["function"]
        ):
            raise ScenarioError(f"Invalid tool handler: {tool['name']}")
        handler_path = (root / handler["module"]).resolve()
        if root.resolve() not in handler_path.parents or not handler_path.is_file():
            raise ScenarioError(f"Invalid tool handler file: {tool['name']}")
    for agent in agents:
        prompt = (root / agent["prompt_file"]).resolve()
        if root.resolve() not in prompt.parents or not prompt.is_file():
            raise ScenarioError(f"Invalid prompt file for {agent['id']}")
    referenced_files = [config["task"]["prompt_file"], config["evaluator"]["module"]]
    for relative in referenced_files:
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ScenarioError(f"Invalid scenario file: {relative}")


def load_scenario(root: Path) -> Scenario:
    root = root.resolve()
    with (root / "scenario.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate(config, root)
    digest, hashes = tree_hash(root)
    return Scenario(root=root, config=config, tree_sha256=digest, file_hashes=hashes)
