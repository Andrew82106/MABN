"""Incremental typed-state compilation and exact risk monitoring.

No batch recompilation or event-history replay. Typed variables are allocated
in arrival order, so their bit positions may differ from the old batch compiler.
Stable mappings preserve identity and joint probability mass as variables grow.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from .joint_stream import JointEventCompiler, JointRiskStream
from .streaming_bn import PRIVILEGED_TOOLS
from .typed_joint_bn import (TypedJointBN, TypedStep, PARAMETER_NAMES, SPEC, SPEC_SHA256,
                             EXTRA_FEATURES, typed_features)


class TypedEventCompiler:
    """Wrap the common evidence compiler, not a second event interpretation."""

    def __init__(self, *, max_variables=12, **context):
        self._base = JointEventCompiler(max_variables=max_variables, **context)
        self.max_variables = max_variables
        self.variables, self._base_indices, self._authorization = [], {}, {}
        self.last_step = None
        self._sync_base_variables()

    @property
    def event_count(self):
        return self._base.event_count

    @property
    def last_was_duplicate(self):
        return self._base.last_was_duplicate

    @property
    def workflow(self):
        return copy.deepcopy(self._base.workflow)

    def _sync_base_variables(self):
        for index in range(len(self._base_indices), len(self._base.variables)):
            variable = copy.deepcopy(self._base.variables[index])
            if variable["kind"] == "workflow_node":
                variable.update(kind="information_state", risk_channel="information")
            self._base_indices[index] = len(self.variables)
            self.variables.append(variable)

    def _preflight(self, event):
        keys = self._base._preflight(event)
        new_base_variables = len(keys - self._base.indices.keys())
        actor = self._base.aliases.get(event["actor"], event["actor"])
        new_authorization = (event["event_type"] in {"tool_proposed", "tool_call"}
                             and event.get("tool") in PRIVILEGED_TOOLS and actor not in self._authorization)
        if len(self.variables) + new_base_variables + int(new_authorization) > self.max_variables:
            raise ValueError(f"typed exact state exceeds {self.max_variables} variables; event was not accepted")

    def push(self, event):
        # Capacity is checked before the common validator commits. Other invalid
        # fields/action changes are rejected atomically by the common compiler.
        self._preflight(event)
        step = self._base.push(event)
        self._sync_base_variables()
        operations = []
        for operation in step.operations:
            template = operation.evidence["template"]
            receiver = self._base_indices[operation.receiver]
            sender = None if operation.sender is None else self._base_indices[operation.sender]
            if template == "authorization":
                information = self.variables[receiver]
                identity = information["id"]
                if identity not in self._authorization:
                    self._authorization[identity] = len(self.variables)
                    self.variables.append({"kind": "authorization_state", "id": identity,
                        "role": information["role"], "risk_channel": "authorization"})
                receiver = self._authorization[identity]
            evidence = {**copy.deepcopy(operation.evidence), "typed_state_binding": SPEC["state_bindings"][template]}
            operations.append(replace(operation, receiver=receiver, sender=sender, evidence=evidence))

        def indices(values):
            return tuple(self._base_indices[i] for i in values)

        def pairs(values):
            return tuple((i, self._authorization.get(self.variables[i]["id"])) for i in indices(values))

        self.last_step = TypedStep(step.sequence, tuple(operations), step.observables.copy(),
            indices(step.observed_nodes), indices(step.capable_nodes), indices(step.terminal_nodes),
            pairs(step.observed_nodes), pairs(step.capable_nodes), pairs(step.terminal_nodes))
        return copy.deepcopy(self.last_step)


class TypedRiskStream(JointRiskStream):
    """Opt-in typed monitor. Does not replace JointRiskStream or block actions."""

    parameter_names = tuple(PARAMETER_NAMES)
    compiler_type = TypedEventCompiler

    def __init__(self, model, *, threshold=None, **context):
        if not isinstance(model, TypedJointBN):
            raise ValueError("TypedRiskStream requires a TypedJointBN model")
        self._typed_head = model.typed_head
        super().__init__(model, threshold=threshold, **context)

    def _state_features(self, bits, step):
        return typed_features(bits, step, self._typed_head)

    def _fingerprint_extension(self):
        return {"typed_head": self._typed_head, "typed_specification_sha256": SPEC_SHA256,
                "model_name": "TypedJointBN-v2"}

    def snapshot(self):
        return {**super().snapshot(), "model_name": "TypedJointBN-v2",
                "state_semantics": "information transport and local authorization hazard are distinct hypotheses"}

    def explain(self):
        result = super().explain()
        result.update(typed_registry={"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256},
            typed_head=self._typed_head,
            latent_feature_names=["union_node_fraction", "union_pair_fraction", "union_capable_fraction",
                                  "union_terminal_fraction", *EXTRA_FEATURES],
            limitations=[item for item in SPEC["boundaries"] if not item.startswith("Batch/prefix compiler only")]
                        + ["Raw-text evidence extraction and real agent-framework hooks are not implemented.",
                           "Runtime identity/deduplication evidence grows with the episode; constant memory or latency is not claimed."],
            interpretation="Exact typed model-state marginals and risk mixture, not validated compromise, valid grants or causal attribution")
        return result
