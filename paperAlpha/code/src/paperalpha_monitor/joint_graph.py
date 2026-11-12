"""Compile the joint model into an executable, time-unrolled probability DAG.

Bernoulli gates are eliminated in chronological order into a joint state
vector. This is an exact interpreter for the declared OR-state model, not a
generic BN solver, causal discovery engine, or arbitrary graph optimizer.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.special import expit

from .dynamic_joint_bn import KNOWLEDGE, HEAD, N_PARAMETERS, OBSERVABLES, PARAMETER_NAMES, CONFIDENCE
from .streaming_bn import reject_outcomes
from .template_registry import _assert_dag


class CompiledJointGraph:
    parameter_names = tuple(PARAMETER_NAMES)
    schema = "executable-joint-bn-v1"

    def __init__(self, variables, nodes, edges, context):
        edges = [tuple(edge) for edge in edges]
        _assert_dag(tuple(nodes), tuple(edges))
        declared = [(parent, node["id"]) for node in nodes for parent in node["parents"]]
        if sorted(edges) != sorted(declared):
            raise ValueError("exported edges must match executable parent bindings")
        self._variables = copy.deepcopy(variables)
        self._nodes = copy.deepcopy(nodes)
        self._edges = copy.deepcopy(edges)
        self._context = copy.deepcopy(context)

    def to_dict(self):
        return copy.deepcopy({"schema": self.schema, "registry": KNOWLEDGE.metadata(),
            "parameter_names": list(self.parameter_names), "variables": self._variables, "nodes": self._nodes,
            "edges": self._edges, "context": self._context,
            "interpretation": "Knowledge-constrained time-unrolled conditional probability DAG; not identified causal truth."})

    @staticmethod
    def head_fields(step):
        return {"observed_nodes": list(step.observed_nodes), "capable_nodes": list(step.capable_nodes),
                "terminal_nodes": list(step.terminal_nodes), "latent_feature_names": list(KNOWLEDGE.latent_summaries)}

    @staticmethod
    def head_variables(node):
        return sorted(set(node["observed_nodes"] + node["capable_nodes"] + node["terminal_nodes"]))

    def head_features(self, bits, node):
        observed, capable, terminal = (tuple(node[k]) for k in ("observed_nodes", "capable_nodes", "terminal_nodes"))
        count = bits[:, observed].sum(axis=1)
        count_n = len(observed)
        return np.column_stack((count / max(1, count_n), count * (count - 1) / max(1, count_n * (count_n - 1)),
            bits[:, capable].sum(axis=1) / max(1, len(capable)), bits[:, terminal].sum(axis=1) / max(1, len(terminal))))

    def predict(self, theta, *, use_messages=True, use_latent_head=True):
        """Evaluate the exported node rules without calling DynamicJointBN."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameter_names),) or not np.isfinite(theta).all():
            raise ValueError("invalid graph parameter vector")
        n = len(self._variables)
        states = np.arange(1 << n)
        bits = ((states[:, None] >> np.arange(n)) & 1).astype(float)
        distribution = np.zeros(len(states))
        distribution[0] = 1.
        by_id, current, gate_values, scores = {}, {}, {}, []
        for node in self._nodes:
            if any(parent not in by_id for parent in node["parents"]):
                raise ValueError("graph order must precede every dependent node")
            by_id[node["id"]] = node
            kind = node["kind"]
            if kind == "gate":
                gate_values[node["id"]] = float(expit(theta[node["cell"]] + theta[CONFIDENCE] * node["confidence_feature"]))
            elif kind == "state":
                if node["rule"] not in {"constant_zero", "seed_or", "carrier_or"}:
                    raise ValueError("unsupported executable state rule")
                receiver = node["variable"]
                if node["rule"] == "constant_zero":
                    if receiver in current:
                        raise ValueError("state variable initialized twice")
                else:
                    prior, gate_id = node["parents"][:2]
                    if current.get(receiver) != prior:
                        raise ValueError("state update references a stale prior")
                    gate = by_id[gate_id]
                    if use_messages or not gate["is_message"]:
                        mask = (states & (1 << receiver)) == 0
                        if node["rule"] == "carrier_or":
                            carrier = by_id[node["parents"][2]]
                            if current.get(carrier["variable"]) != carrier["id"]:
                                raise ValueError("carrier references a stale state")
                            mask &= (states & (1 << carrier["variable"])) != 0
                        source = states[mask]
                        target = source | (1 << receiver)
                        moved = distribution[source] * gate_values[gate_id]
                        distribution = distribution.copy()
                        distribution[source] -= moved
                        distribution[target] += moved
                current[receiver] = node["id"]
            elif kind == "system_risk":
                state_parents = [current[i] for i in self.head_variables(node)]
                if node["parents"][:-1] != state_parents:
                    raise ValueError("system head references stale/missing state versions")
                visible = np.asarray(by_id[node["parents"][-1]]["values"])
                latent = self.head_features(bits, node)
                if not use_latent_head:
                    latent = np.zeros_like(latent)
                conditional = expit(visible @ theta[HEAD:HEAD + len(OBSERVABLES)] + latent @ theta[HEAD + len(OBSERVABLES):])
                scores.append(float(distribution @ conditional))
            elif kind != "conditioned":
                raise ValueError("unsupported executable node kind")
        return np.asarray(scores)


def compile_joint_graph(trace):
    """Use only this compiled prefix; call with a prefix to omit future IDs."""
    if any(v.get("kind") in {"information_state", "authorization_state"} for v in trace.variables):
        raise ValueError("typed trace requires the typed executable graph compiler")
    return _compile_graph(trace, CompiledJointGraph)


def _compile_graph(trace, graph_type):
    """Shared chronological gate/state compiler; readout semantics are explicit."""
    nodes, edges, current, versions = [], [], {}, {}
    variables = list(trace.variables)
    reject_outcomes(variables)

    def add(node):
        edges.extend((parent, node["id"]) for parent in node["parents"])
        nodes.append(node)

    def initialize(index, sequence):
        if not isinstance(index, int) or not 0 <= index < len(variables):
            raise ValueError("invalid graph variable index")
        if index not in current:
            current[index], versions[index] = f"state:{index}:0", 0
            add({"id": current[index], "kind": "state", "variable": index, "version": 0,
                 "rule": "constant_zero", "parents": [], "sequence": sequence, "scope": copy.deepcopy(variables[index])})
        return current[index]

    for step_index, step in enumerate(trace.steps):
        head_fields = graph_type.head_fields(step)
        for index in graph_type.head_variables(head_fields):
            initialize(index, step.sequence)
        for operation_index, operation in enumerate(step.operations):
            description = KNOWLEDGE.describe_operation(operation)
            reject_outcomes(operation.evidence)
            receiver_prior = initialize(operation.receiver, step.sequence)
            carrier = None if operation.sender is None else initialize(operation.sender, step.sequence)
            token = f"{step_index}:{operation_index}"
            observed_id = "parents:" + token
            add({"id": observed_id, "kind": "conditioned", "parents": [], "sequence": step.sequence,
                 "parent_values": description["parents"], "confidence_feature": operation.confidence_feature,
                 "evidence": copy.deepcopy(operation.evidence)})
            gate_id = "gate:" + token
            gate_metadata = {key: value for key, value in description.items() if key != "parents"}
            add({"id": gate_id, "kind": "gate", "parents": [observed_id], "sequence": step.sequence,
                 "distribution": "Bernoulli", "cell": operation.parameter,
                 "confidence_feature": operation.confidence_feature if operation.is_message else 0.,
                 "is_message": operation.is_message, "selected_parent_states": description["parents"], **gate_metadata})
            versions[operation.receiver] += 1
            state_id = f"state:{operation.receiver}:{versions[operation.receiver]}"
            add({"id": state_id, "kind": "state", "variable": operation.receiver,
                 "version": versions[operation.receiver], "rule": description["update"],
                 "parents": [receiver_prior, gate_id] + ([] if carrier is None else [carrier]),
                 "sequence": step.sequence, "scope": copy.deepcopy(variables[operation.receiver])})
            current[operation.receiver] = state_id
        observed_id = f"head_observed:{step_index}"
        add({"id": observed_id, "kind": "conditioned", "parents": [], "sequence": step.sequence,
             "feature_names": list(OBSERVABLES), "values": step.observables.tolist()})
        state_parents = [current[i] for i in graph_type.head_variables(head_fields)]
        add({"id": f"system_risk:{step_index}", "kind": "system_risk", "sequence": step.sequence,
             "parents": state_parents + [observed_id], "distribution": "conditional Bernoulli logistic",
             "parameter_indices": list(range(HEAD, len(graph_type.parameter_names))), **head_fields})
    if not trace.steps:
        raise ValueError("a nonempty observed prefix is required")
    return graph_type(variables, nodes, edges, {"workflow": copy.deepcopy(trace.workflow),
        "last_sequence": trace.steps[-1].sequence, "readouts": len(trace.steps),
        "head_interpretation": "Named engineered summaries, not a literature-proven risk mechanism"})
