"""Declarative scenario package loading and validation."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
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
        return state

    def evaluator(self) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
        relative = self.config["evaluator"]["module"]
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ScenarioError("Evaluator escapes scenario root")
        spec = importlib.util.spec_from_file_location(f"scenario_evaluator_{self.tree_sha256}", path)
        if spec is None or spec.loader is None:
            raise ScenarioError(f"Cannot load evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, self.config["evaluator"].get("function", "evaluate"), None)
        if not callable(function):
            raise ScenarioError("Scenario evaluator is not callable")
        return function


def _validate(config: Mapping[str, Any], root: Path) -> None:
    required = {"schema_version", "scenario_id", "agents", "graph", "schedule", "evaluator", "model_bindings"}
    missing = required - set(config)
    if missing:
        raise ScenarioError(f"Missing scenario fields: {sorted(missing)}")
    agents = config["agents"]
    identifiers = [agent["id"] for agent in agents]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ScenarioError("Agent identifiers must be non-empty and unique")
    known = set(identifiers)
    edge_ids: set[str] = set()
    for edge in config["graph"]["edges"]:
        if edge["id"] in edge_ids or edge["source"] not in known or edge["target"] not in known:
            raise ScenarioError("Graph contains duplicate or invalid edge")
        edge_ids.add(edge["id"])
    if set(config["schedule"]["order"]) - known:
        raise ScenarioError("Schedule contains unknown agents")
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
