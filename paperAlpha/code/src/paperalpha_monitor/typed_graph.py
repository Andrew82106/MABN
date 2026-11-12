"""Executable DAG for the typed model; no call to its forward/feature functions.

The generic graph interpreter eliminates Bernoulli gates chronologically.
This module declares the separate state channels and independently evaluates
the named typed readout. It is not a general-purpose or causal BN solver.
"""
from __future__ import annotations

import copy

import numpy as np

from .joint_graph import CompiledJointGraph, _compile_graph
from .typed_joint_bn import PARAMETER_NAMES, SPEC, SPEC_SHA256, TypedStep


FEATURE_NAMES = ("union_node_fraction", "union_pair_fraction", "union_capable_fraction",
                 "union_terminal_fraction", "authorization_any", "information_authorization_overlap_any")


class CompiledTypedGraph(CompiledJointGraph):
    parameter_names = tuple(PARAMETER_NAMES)
    schema = "executable-typed-joint-bn-v2"

    @staticmethod
    def head_fields(step):
        if not isinstance(step, TypedStep):
            raise ValueError("typed graph requires TypedStep records")
        return {"node_pairs": [list(p) for p in step.node_pairs],
                "capable_pairs": [list(p) for p in step.capable_pairs],
                "terminal_pairs": [list(p) for p in step.terminal_pairs],
                "latent_feature_names": list(FEATURE_NAMES)}

    @staticmethod
    def head_variables(node):
        return sorted({i for key in ("node_pairs", "capable_pairs", "terminal_pairs")
                       for pair in node[key] for i in pair if i is not None})

    def head_features(self, bits, node):
        # Deliberately independent of typed_joint_bn.typed_features. Construct
        # logical state indicators from the EXPORTED pairs, not a saved trace.
        def union_columns(pairs):
            columns = [np.logical_or(bits[:, information], False if authorization is None else bits[:, authorization])
                       for information, authorization in pairs]
            return np.column_stack(columns) if columns else np.zeros((len(bits), 0), dtype=bool)

        pairs = node["node_pairs"]
        unions = union_columns(pairs)
        count, n = unions.sum(axis=1), len(pairs)
        auth = np.zeros(len(bits), dtype=bool)
        overlap = np.zeros(len(bits), dtype=bool)
        for information, authorization in pairs:
            if authorization is not None:
                auth |= bits[:, authorization].astype(bool)
                overlap |= np.logical_and(bits[:, information], bits[:, authorization])
        values = np.column_stack((count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
            union_columns(node["capable_pairs"]).sum(axis=1) / max(1, len(node["capable_pairs"])),
            union_columns(node["terminal_pairs"]).sum(axis=1) / max(1, len(node["terminal_pairs"])), auth, overlap)).astype(float)
        if not self._context["typed_head"]:
            values[:, -2:] = 0.
        return values

    def to_dict(self):
        return {**super().to_dict(), "typed_registry": {"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256},
                "state_bindings": copy.deepcopy(SPEC["state_bindings"]),
                "interpretation": "Executable typed model-state hypotheses, not verified compromise, grants or causal attribution."}


def compile_typed_graph(trace, *, typed_head=True):
    """Export only supplied observations; use a prefix trace to omit future IDs."""
    if not isinstance(typed_head, bool):
        raise ValueError("typed_head must be boolean")
    variables = trace.variables
    if any(v.get("kind") not in {"origin_hypothesis", "information_state", "authorization_state"} for v in variables):
        raise ValueError("typed graph requires typed variable identities")
    for step in trace.steps:
        if not isinstance(step, TypedStep):
            raise ValueError("typed graph requires TypedStep records")
        for key in ("node_pairs", "capable_pairs", "terminal_pairs"):
            for information, authorization in getattr(step, key):
                if variables[information]["kind"] != "information_state":
                    raise ValueError("readout information binding differs")
                if authorization is not None and (variables[authorization]["kind"] != "authorization_state" or
                                                 variables[information]["id"] != variables[authorization]["id"]):
                    raise ValueError("readout authorization binding differs")
        for operation in step.operations:
            template = operation.evidence["template"]
            expected = SPEC["state_bindings"].get(template)
            if operation.evidence.get("typed_state_binding") != expected or variables[operation.receiver]["kind"] != expected:
                raise ValueError("typed operation receiver binding differs")
            carrier = {"source": None, "adoption": "origin_hypothesis", "authorization": None,
                       "transmission": "information_state"}.get(template)
            actual = None if operation.sender is None else variables[operation.sender]["kind"]
            if carrier != actual:
                raise ValueError("typed operation carrier binding differs")
    graph = _compile_graph(trace, CompiledTypedGraph)
    graph._context["typed_head"] = typed_head
    graph._context["state_interpretation"] = "Information can transfer in a message; authorization hazard does not. Both persist."
    return graph
