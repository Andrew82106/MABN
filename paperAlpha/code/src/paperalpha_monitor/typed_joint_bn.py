"""Typed information/authorization state extension of the exact joint BN.

Information follows observed messages. An authorization anomaly affects a
separate state and cannot masquerade as message-carried contamination. The
joint distribution retains correlations and overlap between both types.
This is a conditional risk model, not a discovered causal graph.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .dynamic_joint_bn import (DynamicJointBN, JointTrace, Step, KNOWLEDGE, HEAD,
    OBSERVABLES, PARAMETER_NAMES as BASE_PARAMETER_NAMES, initial_parameters, compile_trace)

from ._resources import static_resource

CONFIG_PATH = static_resource('knowledge/templates/typed_joint_bn_v2.json')
SPEC = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
EXPECTED_BINDINGS = {"source": "origin_hypothesis", "adoption": "information_state",
                     "authorization": "authorization_state", "transmission": "information_state"}
EXTRA_FEATURES = ("authorization_any", "information_authorization_overlap_any")
if SPEC.get("base_registry_id") != KNOWLEDGE.registry_id or SPEC.get("state_bindings") != EXPECTED_BINDINGS:
    raise ValueError("unsupported typed-state binding specification")
if SPEC.get("extra_head_features") != list(EXTRA_FEATURES) or SPEC.get("extra_head_initial_coefficients") != [0., 0.]:
    raise ValueError("unsupported typed-head specification")
SPEC_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
PARAMETER_NAMES = list(BASE_PARAMETER_NAMES) + ["head:" + name for name in EXTRA_FEATURES]


@dataclass(frozen=True)
class TypedStep(Step):
    node_pairs: tuple[tuple[int, int | None], ...]
    capable_pairs: tuple[tuple[int, int | None], ...]
    terminal_pairs: tuple[tuple[int, int | None], ...]


def type_trace(base: JointTrace, max_variables=12):
    """Relabel local states and allocate authorization bits when first observed.

    Allocation in a compiled full trace does not expose future variables to
    earlier readout features: pairs are captured separately at each prefix.
    """
    if not isinstance(max_variables, int) or isinstance(max_variables, bool) or max_variables < 1:
        raise ValueError("positive integer variable limit required")
    variables = copy.deepcopy(list(base.variables))
    for variable in variables:
        if variable["kind"] == "workflow_node":
            variable["kind"] = "information_state"
            variable["risk_channel"] = "information"
    authorizations, steps = {}, []
    if len(variables) > max_variables:
        raise ValueError("typed exact state exceeds variable limit")
    for step in base.steps:
        operations = []
        for operation in step.operations:
            template = operation.evidence["template"]
            if template not in SPEC["state_bindings"]:
                raise ValueError("event has no typed-state binding")
            receiver = operation.receiver
            if template == "authorization":
                if receiver not in authorizations:
                    if len(variables) >= max_variables:
                        raise ValueError("typed exact state exceeds variable limit")
                    authorizations[receiver] = len(variables)
                    variables.append({"kind": "authorization_state", "id": variables[receiver]["id"],
                                      "role": variables[receiver]["role"], "risk_channel": "authorization"})
                receiver = authorizations[receiver]
            evidence = {**copy.deepcopy(operation.evidence), "typed_state_binding": SPEC["state_bindings"][template]}
            operations.append(replace(operation, receiver=receiver, evidence=evidence))
        def pairs(indices):
            return tuple((i, authorizations.get(i)) for i in indices)
        steps.append(TypedStep(step.sequence, tuple(operations), step.observables.copy(), step.observed_nodes,
                               step.capable_nodes, step.terminal_nodes, pairs(step.observed_nodes),
                               pairs(step.capable_nodes), pairs(step.terminal_nodes)))
    return JointTrace(tuple(variables), tuple(steps), copy.deepcopy(base.workflow))


def compile_typed_trace(trace, *, max_variables=12, **context):
    return type_trace(compile_trace(trace, max_variables=max_variables, **context), max_variables)


def _channel_bits(bits, pairs):
    information = np.column_stack([bits[:, i] for i, _ in pairs]) if pairs else np.zeros((len(bits), 0))
    authorization = np.column_stack([np.zeros(len(bits)) if a is None else bits[:, a] for _, a in pairs]) if pairs else np.zeros((len(bits), 0))
    return information, authorization


def typed_features(bits, step, include_typed_head=True):
    information, authorization = _channel_bits(bits, step.node_pairs)
    union = np.maximum(information, authorization)
    count, n = union.sum(axis=1), len(step.node_pairs)
    def union_fraction(pairs):
        a, b = _channel_bits(bits, pairs)
        return np.maximum(a, b).sum(axis=1) / max(1, len(pairs))
    features = np.column_stack((count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
        union_fraction(step.capable_pairs), union_fraction(step.terminal_pairs),
        np.any(authorization > 0, axis=1), np.any(information * authorization > 0, axis=1))).astype(float)
    if not include_typed_head:
        features[:, -2:] = 0.
    return features


class TypedJointBN(DynamicJointBN):
    def __init__(self, *, typed_head=True, **kwargs):
        self.typed_head = typed_head
        super().__init__(**kwargs)

    def initial_parameters(self):
        return np.r_[initial_parameters(), SPEC["extra_head_initial_coefficients"]]

    def _state_features(self, bits, step):
        if not isinstance(step, TypedStep):
            raise ValueError("TypedJointBN requires the typed compiler")
        return typed_features(bits, step, self.typed_head)

    def _regularization_count(self):
        # Do not weaken the old parameter penalty merely by adding two zeros.
        return len(BASE_PARAMETER_NAMES)

    def parameters(self):
        result = super().parameters()
        result.update(name="TypedJointBN-v2", parameter_names=PARAMETER_NAMES, typed_head=self.typed_head,
                      typed_registry={"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256},
                      regularization_denominator=len(BASE_PARAMETER_NAMES),
                      inference="exact joint filtering over origin, information and authorization states",
                      limitations=SPEC["boundaries"])
        return result

    def explain(self, trace, *, include_graph=False):
        result = super().explain(trace)
        result["typed_registry"] = {"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256}
        result["latent_feature_names"] = ["union_node_fraction", "union_pair_fraction", "union_capable_fraction",
                                          "union_terminal_fraction", *EXTRA_FEATURES]
        result["typed_head"] = self.typed_head
        result["interpretation"] = "Typed model-state hypotheses and exact probability mixture, not verified compromise, permission grants or causal attribution"
        # The frozen structural specification keeps its historical release
        # note for saved-model compatibility; report current API capabilities
        # here rather than rewriting that schema/fingerprint retroactively.
        result["limitations"] = [item for item in SPEC["boundaries"] if not item.startswith("Batch/prefix compiler only")]
        result["implementation_capabilities"] = {"incremental_adapter": "TypedRiskStream",
                                                  "executable_graph": "compile_typed_graph",
                                                  "raw_text_extraction": False, "real_framework_hooks": False}
        if include_graph:
            from .typed_graph import compile_typed_graph
            graph = compile_typed_graph(trace, typed_head=self.typed_head)
            result["knowledge_graph"] = graph.to_dict()
            result["graph_risk_probability"] = float(graph.predict(self.theta, use_messages=self.use_messages,
                                                                    use_latent_head=self.use_latent_head)[-1])
        return result
