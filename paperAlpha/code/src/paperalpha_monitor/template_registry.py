"""Validated knowledge-backed CPT registry and an explicit conditional BN graph.

Only a supplied Snapshot is compiled: no trace, model output, workflow edges or
future events are loaded. The graph is parameterized, not a causal discovery.
It shares CPTs across instances and marginalizes Bernoulli risks through OR.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .streaming_bn import Snapshot


from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = static_resource('knowledge/templates/risk_templates_v1.json')
DEFAULT_CATALOG = static_resource('knowledge/risks/risk_catalog.json')


def _integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _unique(rows: list[dict], key: str, label: str) -> dict:
    if not isinstance(rows, list) or any(not isinstance(row, dict) or key not in row for row in rows):
        raise ValueError(f"invalid {label}")
    values = [row[key] for row in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")
    return dict(zip(values, rows))


def _assert_dag(nodes: tuple[dict, ...], edges: tuple[tuple[str, str], ...]) -> None:
    identities = {node["id"] for node in nodes}
    if len(identities) != len(nodes):
        raise ValueError("duplicate compiled node")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate compiled edge")
    adjacency = defaultdict(list)
    incoming = Counter()
    for source, target in edges:
        if source not in identities or target not in identities:
            raise ValueError("undeclared compiled edge endpoint")
        adjacency[source].append(target)
        incoming[target] += 1
    todo = [node for node in identities if not incoming[node]]
    visited = 0
    while todo:
        node = todo.pop()
        visited += 1
        for target in adjacency[node]:
            incoming[target] -= 1
            if not incoming[target]:
                todo.append(target)
    if visited != len(nodes):
        raise ValueError("compiled probability graph must be acyclic")


@dataclass(frozen=True)
class CompiledSnapshotGraph:
    sequence: int
    nodes: tuple[dict, ...]
    edges: tuple[tuple[str, str], ...]
    shared_cpts: dict
    n_cells: int
    interpretation: str = "Conditional independence and episode OR are modeling assumptions, not causal claims."

    def __post_init__(self) -> None:
        _assert_dag(self.nodes, self.edges)

    def risk_probability(self, theta: np.ndarray) -> float:
        """Exact marginal P(episode_OR=1 | observed parents, shared logits)."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_cells,) or not np.isfinite(theta).all():
            raise ValueError(f"expected {self.n_cells} finite CPT logits")
        cells = [node["cpt"]["selected_cell"] for node in self.nodes if node["kind"] == "risk"]
        hazard = float(np.logaddexp(0.0, theta[cells]).sum())
        return float(-np.expm1(-hazard))

    def to_dict(self) -> dict:
        return copy.deepcopy({
            "sequence": self.sequence, "nodes": list(self.nodes), "edges": [list(edge) for edge in self.edges],
            "shared_cpts": self.shared_cpts, "n_cells": self.n_cells,
            "output": "episode_OR", "interpretation": self.interpretation,
        })


class TemplateRegistry:
    """Validated registry; specs can replace hard-coded Python template tuples."""

    def __init__(self, specification: dict, catalog: dict):
        specification, catalog = copy.deepcopy(specification), copy.deepcopy(catalog)
        if specification.get("schema_version") != "1.0":
            raise ValueError("unsupported registry schema_version")
        _text(specification.get("registry_id"), "registry_id")
        self.registry_id = specification["registry_id"]
        templates = _unique(specification.get("templates"), "id", "template id")
        entries = _unique(catalog.get("entries"), "id", "catalog risk id")
        sources = _unique(catalog.get("sources"), "id", "catalog source id")
        if not templates:
            raise ValueError("at least one template required")
        self._templates = templates
        self._catalog_entries, self._sources = entries, sources
        self._specification = specification
        self.offsets: dict[str, int] = {}
        self.cell_names: list[str] = []
        specs = []
        shared_parent_semantics = {}
        for name, template in templates.items():
            _text(name, "template id")
            if template.get("scope") not in {"action", "global"}:
                raise ValueError("template scope must be action or global")
            dims = template.get("dims")
            if not isinstance(dims, list) or not dims or any(not _integer(d) or d < 2 for d in dims):
                raise ValueError(f"invalid dims for {name}")
            parents = template.get("parents")
            parent_map = _unique(parents, "name", f"parent name in {name}")
            if len(parents) != len(dims):
                raise ValueError(f"dims/parents mismatch for {name}")
            for parent, size in zip(parents, dims):
                _text(parent["name"], "parent name")
                states = parent.get("states")
                state_map = _unique(states, "value", f"parent state in {name}")
                if len(states) != size or list(state_map) != list(range(size)) or any(not _integer(state["value"]) for state in states):
                    raise ValueError(f"dims/state cardinality or order mismatch for {name}")
                for state in states:
                    _text(state.get("name"), "state name")
                    _text(state.get("meaning"), "state meaning")
                unknown = parent.get("unknown_values")
                if not isinstance(unknown, list) or any(not _integer(value) or value not in state_map for value in unknown):
                    raise ValueError("invalid unknown_values")
                if parent.get("availability") not in {"observed_prefix", "workflow_available"}:
                    raise ValueError("invalid parent availability")
                previous = shared_parent_semantics.setdefault(parent["name"], parent)
                if previous != parent:
                    raise ValueError(f"inconsistent shared parent semantics: {parent['name']}")
            if template.get("cpt") != {"sharing": "all_instances_of_template", "index_order": "C", "link": "sigmoid"}:
                raise ValueError(f"unsupported CPT specification for {name}")
            for field in ("trigger", "unknown_handling"):
                _text(template.get(field), field)
            for field in ("required_observations", "benign_controls", "evidence_gaps"):
                values = template.get(field)
                if not isinstance(values, list) or not values:
                    raise ValueError(f"{field} must be nonempty")
                for value in values:
                    _text(value, field)
            links = template.get("catalog_links")
            _unique(links, "risk_id", f"risk link in {name}")
            for link in links:
                risk_id = link["risk_id"]
                if risk_id not in entries:
                    raise ValueError(f"unknown catalog risk: {risk_id}")
                source_ids = link.get("source_ids")
                if not isinstance(source_ids, list) or not source_ids or any(not _integer(value) for value in source_ids) or len(set(source_ids)) != len(source_ids):
                    raise ValueError("source_ids must contain unique integer IDs")
                for source in source_ids:
                    if source not in sources or source not in entries[risk_id].get("source_ids", []):
                        raise ValueError(f"source {source} does not belong to catalog risk {risk_id}")
                _text(link.get("coverage"), "catalog coverage limitation")
            self.offsets[name] = len(self.cell_names)
            names = tuple(parent_map)
            specs.append((name, tuple(dims), names))
            for values in itertools.product(*(range(dim) for dim in dims)):
                self.cell_names.append(name + ":" + ",".join(f"{key}={value}" for key, value in zip(names, values)))
        self.specs = tuple(specs)
        self.n_cells = len(self.cell_names)
        expected = specification.get("expected_total_cells")
        if not _integer(expected) or expected != self.n_cells:
            raise ValueError("expected_total_cells mismatch")

    def semantic_metadata(self, template_id: str | None = None) -> dict:
        """Return independent metadata, including exact existing catalog records."""
        templates = [self._templates[template_id]] if template_id is not None else list(self._templates.values())
        result = {"registry_id": self._specification["registry_id"], "scope_limit": self._specification.get("scope_limit"), "templates": []}
        for template in templates:
            row = copy.deepcopy(template)
            for link in row["catalog_links"]:
                link["catalog_entry"] = copy.deepcopy(self._catalog_entries[link["risk_id"]])
                link["sources"] = [copy.deepcopy(self._sources[source]) for source in link["source_ids"]]
            result["templates"].append(row)
        return result

    def cell_index(self, template_id: str, parent_values: list[int] | tuple[int, ...]) -> int:
        if template_id not in self._templates:
            raise ValueError(f"unknown template {template_id}")
        dims = self._templates[template_id]["dims"]
        if len(parent_values) != len(dims) or any(not _integer(value) or not 0 <= value < size for value, size in zip(parent_values, dims)):
            raise ValueError(f"invalid parent states for {template_id}")
        index = 0
        for value, size in zip(parent_values, dims):
            index = index * size + int(value)
        return self.offsets[template_id] + index

    def compile_snapshot(self, snapshot: "Snapshot") -> CompiledSnapshotGraph:
        """Compile observed parents -> shared-CPT Bernoulli risks -> episode OR.

        Arbitrary named_evidence fields are not read. Only workflow_available
        is consulted for epistemic status, never to add paths or risk factors.
        """
        if not _integer(snapshot.sequence) or snapshot.sequence < -1:
            raise ValueError("invalid snapshot sequence")
        workflow_available = snapshot.named_evidence.get("workflow_available")
        if workflow_available is not None and not isinstance(workflow_available, bool):
            raise ValueError("workflow_available must be boolean when supplied")
        nodes, edges, tables = {}, [], {}
        action_groups: dict[tuple[str, str], str] = {}
        group_evidence = {}
        risk_ids = []
        for factor in snapshot.factors:
            name = factor["template"]
            if name not in self._templates:
                raise ValueError(f"unknown template {name}")
            template = self._templates[name]
            values = factor["parents"]
            cell = self.cell_index(name, values)
            if not _integer(factor.get("cell")) or factor["cell"] != cell or factor.get("cell_name") != self.cell_names[cell]:
                raise ValueError("snapshot cell or name does not match named parent states")
            sequence = factor.get("sequence")
            if not _integer(sequence) or not -1 <= sequence <= snapshot.sequence:
                raise ValueError("factor sequence is outside observed prefix")
            paths = factor.get("observed_paths", {})
            if not isinstance(paths, dict):
                raise ValueError("observed_paths must be a dictionary")
            for origin, witness in paths.items():
                _text(origin, "origin key")
                if not isinstance(witness, (list, tuple)) or not witness or any(not _integer(time) or not 0 <= time <= sequence for time in witness) or any(left >= right for left, right in zip(witness, witness[1:])):
                    raise ValueError("observed path is not a chronological prefix witness")
            if template["scope"] == "action":
                actor, action_id = factor.get("actor"), factor.get("action_id")
                _text(actor, "actor")
                _text(action_id, "action_id")
                group = action_groups.setdefault((actor, action_id), f"action:{len(action_groups)}")
            else:
                actor, action_id, group = None, None, "global"
                if sequence != snapshot.sequence:
                    raise ValueError("global context must describe current snapshot")
            evidence = {"actor": actor, "action_id": action_id, "sequence": int(sequence), "observed_paths": copy.deepcopy(paths)}
            for key in ("phase", "tool", "origin_identity"):
                if key in factor:
                    evidence[key] = copy.deepcopy(factor[key])
            if group in group_evidence and group_evidence[group] != evidence:
                raise ValueError("inconsistent evidence for the same action")
            group_evidence[group] = evidence
            risk_id = f"risk:{group}:{name}"
            if risk_id in nodes:
                raise ValueError("duplicate risk instance")
            parent_ids = []
            for parent, value in zip(template["parents"], values):
                parent_id = f"observed:{group}:{parent['name']}"
                status = "unknown" if value in parent["unknown_values"] else "observed_state"
                if parent["availability"] == "workflow_available":
                    status = "observed_state" if workflow_available else "unavailable" if workflow_available is False else "availability_unknown"
                    if workflow_available is not True and value != 0:
                        raise ValueError("workflow deviation asserted without available workflow")
                node = {"id": parent_id, "kind": "observed", "variable": parent["name"], "scope": template["scope"], "states": copy.deepcopy(parent["states"]), "observed_state": int(value), "state_name": parent["states"][value]["name"], "knowledge_status": status, "distribution": {"type": "conditioned", "value": int(value)}, **evidence}
                if parent_id in nodes and nodes[parent_id] != node:
                    raise ValueError("inconsistent shared observed parent")
                nodes[parent_id] = node
                parent_ids.append(parent_id)
                edges.append((parent_id, risk_id))
            rows = math.prod(template["dims"])
            tables[name] = {"dims": copy.deepcopy(template["dims"]), "parent_names": [parent["name"] for parent in template["parents"]], "index_order": "C", "parameter_indices": list(range(self.offsets[name], self.offsets[name] + rows)), "distribution": "Bernoulli", "q1": "sigmoid(theta[cell])", "q0": "1-q1", "sharing": "all_instances_of_template"}
            nodes[risk_id] = {"id": risk_id, "kind": "risk", "template": name, "scope": template["scope"], "states": [0, 1], "parents": parent_ids, "distribution": "Bernoulli", "cpt": {"table": name, "parent_states": [int(value) for value in values], "selected_cell": cell, "selected_row": cell - self.offsets[name], "cell_name": self.cell_names[cell]}, "catalog_links": copy.deepcopy(template["catalog_links"]), **evidence}
            risk_ids.append(risk_id)
            edges.append((risk_id, "episode_OR"))
        nodes["episode_OR"] = {"id": "episode_OR", "kind": "outcome", "states": [0, 1], "parents": risk_ids, "distribution": {"type": "deterministic_OR", "empty_value": 0, "definition": "P(Y=1|risks)=1 iff at least one risk is 1"}}
        return CompiledSnapshotGraph(int(snapshot.sequence), tuple(nodes.values()), tuple(edges), tables, self.n_cells)


def load_registry(config_path: str | Path = DEFAULT_CONFIG, catalog_path: str | Path = DEFAULT_CATALOG) -> TemplateRegistry:
    return TemplateRegistry(json.loads(Path(config_path).read_text(encoding="utf-8")), json.loads(Path(catalog_path).read_text(encoding="utf-8")))


def compile_snapshot(snapshot: "Snapshot", registry: TemplateRegistry | None = None) -> CompiledSnapshotGraph:
    return (registry if registry is not None else load_registry()).compile_snapshot(snapshot)
