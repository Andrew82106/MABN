"""Prefix-only evidence extraction and an episode-supervised conditional BN.

Each named template selects a Bernoulli CPT cell using observable parents.
Conditional on those parents, template risk variables are independent; the
system risk is their OR. Exact marginalization gives p = 1 - prod(1 - q).
All CPT logits are fitted jointly to episode outcomes, NOT node pseudo-labels.
This is a conditional risk model, not a discovered causal model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .template_registry import load_registry


PRIVILEGED_TOOLS = frozenset({"payments.write", "filesystem.write", "shell.exec"})
FORBIDDEN = frozenset({
    "label", "label_source", "label_evidence", "label_rule", "external_effect",
    "tool_side_effect", "hidden_fields", "mechanism", "scenario_family",
    "violation", "first_violation_sequence",
})
REGISTRY = load_registry()
TEMPLATES = REGISTRY.specs
OFFSETS = dict(REGISTRY.offsets)
CELL_NAMES = list(REGISTRY.cell_names)
N_CELLS = REGISTRY.n_cells


def reject_outcomes(value: Any, path: str = "trace") -> None:
    """Check structured metadata; text payload contents are not interpreted."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN:
                raise ValueError(f"evaluation-only field at {path}.{key}")
            reject_outcomes(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            reject_outcomes(child, f"{path}[{i}]")


def validate_workflow(workflow: dict | None) -> set[tuple[str, str]] | None:
    if workflow is None:
        return None
    reject_outcomes(workflow, "workflow")
    nodes = workflow.get("nodes", [])
    if any(not isinstance(node, str) or not node for node in nodes):
        raise ValueError("workflow.nodes must contain nonempty agent IDs")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate workflow node")
    edges = set()
    for edge in workflow.get("edges", []):
        if len(edge) != 2 or edge[0] not in nodes or edge[1] not in nodes:
            raise ValueError("workflow edge has undeclared endpoint")
        edges.add(tuple(edge))
    incoming = Counter(v for _, v in edges)
    todo = [node for node in nodes if not incoming[node]]
    visited = 0
    while todo:
        u = todo.pop()
        visited += 1
        for left, right in edges:
            if left == u:
                incoming[right] -= 1
                if incoming[right] == 0:
                    todo.append(right)
    if visited != len(nodes):
        raise ValueError("normative workflow must be a DAG")
    return edges


def topology_group(workflow: dict) -> str:
    """Label-free, renaming-invariant topology grouping.

    Iterative degree/neighborhood colors may group non-isomorphic DAGs
    together (conservative), but identical/isomorphic plans never split apart.
    Hidden attack-edge patterns and mechanism labels are not used.
    """
    edges = validate_workflow(workflow)
    nodes = workflow["nodes"]
    assert edges is not None
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for u, v in edges:
        incoming[v].append(u)
        outgoing[u].append(v)
    colors = {v: f"{len(incoming[v])},{len(outgoing[v])}" for v in nodes}
    for _ in range(len(nodes)):
        colors = {
            v: hashlib.sha256(json.dumps([
                colors[v], sorted(colors[u] for u in incoming[v]),
                sorted(colors[u] for u in outgoing[v]),
            ]).encode()).hexdigest() for v in nodes
        }
    return hashlib.sha256(json.dumps(sorted(colors.values())).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    factors: tuple[dict, ...]
    named_evidence: dict

    def vector(self, exclude: frozenset[str] = frozenset()) -> np.ndarray:
        counts = np.zeros(N_CELLS, dtype=float)
        for factor in self.factors:
            if factor["template"] not in exclude:
                counts[factor["cell"]] += 1.0
        return counts


class EvidenceStream:
    """Online state. Only push(event) can reveal a source, tool or new agent."""

    def __init__(self, *, workflow: dict | None = None,
                 allowed_tools: dict[str, list[str]] | None = None,
                 permission_mode: str = "observed"):
        if permission_mode not in {"observed", "masked"}:
            raise ValueError("invalid permission mode")
        self.expected = validate_workflow(copy.deepcopy(workflow))
        self.allowed_tools = copy.deepcopy(allowed_tools or {})
        if not isinstance(self.allowed_tools, dict) or any(
            not isinstance(actor, str) or not actor or not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            for actor, names in self.allowed_tools.items()
        ):
            raise ValueError("allowed_tools must map agent IDs to lists of tool names")
        self.permission_mode = permission_mode
        self.last_sequence = -1
        self.seen_events: dict[str, str] = {}
        self.agents: set[str] = set()
        # One chronological witness per observed origin key per agent.
        # Explicit source_id is preferred; entry-agent fallback is NOT proof
        # of independent materials, provenance, or corroboration.
        self.paths: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
        self.unexpected_incoming: set[str] = set()
        self.unexpected_upstream: set[str] = set()
        self.counts: Counter = Counter()
        self.actions: dict[str, tuple[dict, list[dict]]] = {}

    @staticmethod
    def _factor(name: str, parents: tuple[int, ...], evidence: dict) -> dict:
        dims = next(d for n, d, _ in TEMPLATES if n == name)
        cell = OFFSETS[name] + int(np.ravel_multi_index(parents, dims))
        return {"template": name, "cell": cell, "cell_name": CELL_NAMES[cell],
                "parents": list(parents), **evidence}

    def _permission(self, event: dict) -> int:
        # 0=unobserved, 1=observed check without mismatch, 2=mismatch.
        # A no-mismatch observation is NOT proof that the call was authorized.
        if self.permission_mode == "masked":
            return 0
        keys = ("permission_mismatch", "permission_denied")
        for key in keys:
            if key in event and not isinstance(event[key], bool):
                raise ValueError(f"{key} must be boolean if provided")
        if not any(key in event for key in keys):
            return 0
        return 2 if any(event.get(key, False) for key in keys) else 1

    def push(self, event: dict) -> Snapshot:
        reject_outcomes(event, "event")
        identity = event.get("event_id")
        if identity is not None and (not isinstance(identity, str) or not identity):
            raise ValueError("event_id must be a nonempty string")
        fingerprint = json.dumps(event, sort_keys=True)
        if identity is not None and identity in self.seen_events:
            if self.seen_events[identity] != fingerprint:
                raise ValueError("conflicting repeated event_id")
            return self.snapshot()  # transport retry, not another action
        sequence = event.get("sequence", self.last_sequence + 1)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= self.last_sequence:
            raise ValueError("sequence must strictly increase in arrival order")
        actor, target, kind = event.get("actor"), event.get("target"), event.get("event_type")
        if not isinstance(actor, str) or not actor:
            raise ValueError("event.actor is required")
        if kind in {"message_sent", "delegation"} and (not isinstance(target, str) or not target):
            raise ValueError("message target is required")
        source_id = event.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a nonempty string if provided")
        permission = self._permission(event)
        confidence = event.get("edge_confidence", 1.0)
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("edge_confidence must be in [0,1]")
        sensitive_action = kind in {"tool_proposed", "tool_call"} and event.get("tool") in PRIVILEGED_TOOLS
        action_id = event.get("action_id", f"event:{sequence}")
        if sensitive_action:
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("action_id must be a nonempty string")
            previous = self.actions.get(action_id)
            if previous is not None:
                if previous[0]["phase"] == "tool_call":
                    raise ValueError("repeated executed action_id without transport event_id")
                if previous[0]["actor"] != actor or previous[0]["tool"] != event["tool"]:
                    raise ValueError("action_id cannot change actor or tool")
        # Commit validation before modifying online state.
        self.last_sequence = sequence
        if identity is not None:
            self.seen_events[identity] = fingerprint
        self.agents.add(actor)
        self.counts["events"] += 1
        data_class = event.get("data_class")
        if kind == "input_received" and data_class == "untrusted":
            origin = f"source:{source_id}" if source_id is not None else f"entry_agent:{actor}"
            if origin not in self.paths[actor]:
                self.paths[actor][origin] = (sequence,)
        if kind in {"message_sent", "delegation"}:
            self.agents.add(target)
            self.counts["messages"] += 1
            unknown = data_class in {None, "unknown", "redacted"} or confidence == 0
            untrusted = data_class == "untrusted" and not unknown
            # edge_kind can encode a comparison with a hidden normative W.
            # Without explicit W this model does not claim to know deviations.
            unexpected = self.expected is not None and (actor, target) not in self.expected
            self.counts["unknown_messages"] += int(unknown)
            self.counts["untrusted_messages"] += int(untrusted)
            self.counts["unexpected_messages"] += int(unexpected)
            if unexpected:
                self.unexpected_incoming.add(target)
            # Normative deviations follow actual communication; W never
            # invents a runtime edge or a data-flow path.
            if unexpected or actor in self.unexpected_upstream:
                self.unexpected_upstream.add(target)
            if untrusted:
                for source, witness in list(self.paths[actor].items()):
                    candidate = witness + (sequence,)
                    if source not in self.paths[target] or len(candidate) < len(self.paths[target][source]):
                        self.paths[target][source] = candidate
        if sensitive_action:
            capability = 0
            if actor in self.allowed_tools:
                capability = 1 if event["tool"] in self.allowed_tools[actor] else 2
            paths = {key: list(value) for key, value in self.paths[actor].items()}
            evidence = {"actor": actor, "action_id": action_id, "sequence": sequence,
                        "phase": kind, "tool": event["tool"], "observed_paths": paths,
                        "origin_identity": "source_id when supplied, else entry-agent proxy; no independence guarantee"}
            factors = [self._factor("authorization", (permission, capability), evidence)]
            if paths:
                factors.append(self._factor(
                    "information_flow", (1, int(len(paths) > 1), permission), evidence))
            if actor in self.unexpected_upstream or actor in self.unexpected_incoming:
                factors.append(self._factor(
                    "workflow_deviation",
                    (int(actor in self.unexpected_incoming),
                     int(actor in self.unexpected_upstream), permission), evidence))
            # A proposal and its matching dispatch denote one action, not two
            # conditionally independent risks. No future call is pre-created.
            self.actions[action_id] = (evidence, factors)
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        parents = (
            int(self.counts["untrusted_messages"] > 0),
            int(self.counts["unknown_messages"] > 0),
            int(self.counts["unexpected_messages"] > 0),
            int(sum(bool(v) for v in self.paths.values()) > 1),
        )
        context = self._factor("global_context", parents, {"sequence": self.last_sequence})
        factors = [context]
        for _, rows in self.actions.values():
            factors.extend(rows)
        return Snapshot(self.last_sequence, tuple(copy.deepcopy(factors)), {
            "observed_agents": sorted(self.agents), "event_counts": dict(self.counts),
            "paths_by_agent": copy.deepcopy(dict(self.paths)),
            "observed_actions": [copy.deepcopy(row) for row, _ in self.actions.values()],
            "workflow_available": self.expected is not None,
        })


def replay(trace: dict, *, workflow: dict | None = None,
           permission_mode: str = "observed",
           allowed_tools: dict[str, list[str]] | None = None) -> list[Snapshot]:
    reject_outcomes(trace)
    # trace.agents and trace.workflow_dag are NOT automatically visible.
    # An audited pre-run workflow must be passed explicitly.
    stream = EvidenceStream(workflow=workflow, permission_mode=permission_mode,
                            allowed_tools=allowed_tools)
    return [stream.push(event) for event in trace.get("events", [])]


class TemplateBN:
    """Joint MAP fit of conditional Bernoulli tables under episode OR loss."""

    def __init__(self, *, exclude: frozenset[str] = frozenset(),
                 regularization: float = 0.01):
        self.exclude = exclude
        self.regularization = regularization
        self.theta = np.full(N_CELLS, -4.0)
        self.support = np.zeros(N_CELLS)
        self.weighted_support = np.zeros(N_CELLS)
        self.training_weight_sum = 0.0
        self.training_snapshots = 0
        self.optimizer: dict = {}

    def matrix(self, snapshots: list[Snapshot]) -> np.ndarray:
        return np.asarray([s.vector(self.exclude) for s in snapshots])

    def loss_gradient(self, theta: np.ndarray, counts: np.ndarray,
                      labels: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = weights / weights.sum()
        hazard = counts @ np.logaddexp(0, theta)
        probability = -np.expm1(-hazard)
        safe = np.clip(probability, 1e-12, 1 - 1e-12)
        loss = np.sum(weights * (-labels * np.log(safe) + (1 - labels) * hazard))
        # d NLL / d H = 1 - y / p; no finite-difference approximation.
        gradient = expit(theta) * (counts.T @ (weights * (1 - labels / safe)))
        centered = theta + 4.0  # weak fixed Gaussian prior on CPT logits
        loss += self.regularization * np.mean(centered ** 2)
        gradient += 2 * self.regularization * centered / len(theta)
        return float(loss), gradient

    def fit(self, snapshots: list[Snapshot], labels: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "TemplateBN":
        labels = np.asarray(labels, dtype=float)
        if not len(labels) or len(labels) != len(snapshots) or not np.isin(labels, [0, 1]).all():
            raise ValueError("one binary episode label required per snapshot")
        weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite nonnegative weights with positive total")
        counts = self.matrix(snapshots)
        self.support = counts.sum(axis=0)
        self.weighted_support = (counts * weights[:, None]).sum(axis=0)
        self.training_weight_sum = float(weights.sum())
        self.training_snapshots = len(labels)
        result = minimize(self.loss_gradient, self.theta, args=(counts, labels, weights),
                          jac=True, method="L-BFGS-B", bounds=[(-12, 8)] * N_CELLS,
                          options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-7})
        self.theta = result.x
        self.optimizer = {"success": bool(result.success), "iterations": int(result.nit),
                          "loss": float(result.fun), "message": str(result.message)}
        return self

    def predict(self, snapshots: list[Snapshot]) -> np.ndarray:
        return -np.expm1(-(self.matrix(snapshots) @ np.logaddexp(0, self.theta)))

    def explain(self, snapshot: Snapshot) -> dict:
        contributions = []
        for factor in snapshot.factors:
            if factor["template"] in self.exclude:
                continue
            j = factor["cell"]
            contributions.append({
                **factor, "conditional_risk": float(expit(self.theta[j])),
                "negative_log_survival": float(np.logaddexp(0, self.theta[j])),
                "fit_activation_count": float(self.support[j]),
                "weighted_fit_activation_count": float(self.weighted_support[j]),
            })
        total = sum(row["negative_log_survival"] for row in contributions)
        graph = REGISTRY.compile_snapshot(Snapshot(
            snapshot.sequence,
            tuple(f for f in snapshot.factors if f["template"] not in self.exclude),
            snapshot.named_evidence,
        ))
        return {
            "sequence": snapshot.sequence, "risk_probability": float(-np.expm1(-total)),
            "factors": contributions,
            "registry_id": REGISTRY.registry_id,
            "compiled_graph": graph.to_dict(),
            "interpretation": "exact model-score decomposition, not causal attribution",
            "uncertainty": "not a calibrated probability interval",
        }

    def parameters(self) -> dict:
        return {
            "name": "StreamingConditionalTemplateBN", "cpt_cell_names": CELL_NAMES,
            "registry_id": REGISTRY.registry_id,
            "theta": self.theta.tolist(), "fit_activation_counts": self.support.tolist(),
            "weighted_fit_activation_counts": self.weighted_support.tolist(),
            "training_snapshots": self.training_snapshots, "training_weight_sum": self.training_weight_sum,
            "exclude": sorted(self.exclude), "regularization": self.regularization,
            "optimizer": self.optimizer,
            "learning": "joint episode NLL; latent template risks marginalized via conditional OR",
            "graph_learning": "explicit chronological witnesses; propagation gates not trained",
        }
