"""Capability-conditioned action BN for partially observed API tool calls.

This extends the action mechanism, not its historical performance evidence.
Public descriptions select learned CPT cells; they never grant permission.
Partial action scope is retained explicitly and cannot accept a bound check.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib

import numpy as np

from .action_evidence import ActionEventCompiler, ActionStep, BINDING_FIELDS, canonical
from .action_joint_bn import (ActionScopedJointBN, ActionRiskStream, ACTION_OBSERVABLES,
                              PARAMETER_NAMES as V3_NAMES, SPEC)
from .dynamic_joint_bn import AUTHORIZE, TRANSMIT, CONFIDENCE, HEAD, KNOWLEDGE, JointTrace
from .streaming_bn import reject_outcomes
from .tool_capabilities import CAPABILITIES, ToolCapabilityRegistry, fingerprint


AUTH_WIDTH = TRANSMIT - AUTHORIZE
EXTRA_AUTH = AUTH_WIDTH * (len(CAPABILITIES) - 1)
CAP_HEAD, CAP_CONFIDENCE = HEAD + EXTRA_AUTH, CONFIDENCE + EXTRA_AUTH
CAP_OBSERVABLES = (*ACTION_OBSERVABLES[:11], "tool_manifest_provided", "tool_using_node_fraction",
                   "unknown_action_capability_fraction", "incomplete_action_scope_fraction")
CAP_STATE_FEATURES = ("information_node_fraction", "information_pair_fraction", "information_tool_node_fraction",
                      "information_terminal_fraction", *SPEC["head_features"])
PARAMETER_NAMES = (*V3_NAMES[:AUTHORIZE],
    *(f"authorization:{capability}:{role}:{permission}" for capability in CAPABILITIES
      for role in KNOWLEDGE.states("role") for permission in KNOWLEDGE.states("permission")),
    *V3_NAMES[TRANSMIT:HEAD], *("head:" + name for name in (*CAP_OBSERVABLES, *CAP_STATE_FEATURES)))
MODEL_SPEC = {"registry_id": "capability-action-v4", "capabilities": list(CAPABILITIES),
    "parameter_names": list(PARAMETER_NAMES), "primary_capability_only": True,
    "scope_contract": "principal/resource may be null; accepted checks require complete immutable binding",
    "unknown_contract": "explicit learned unknown cell and coverage feature, never a safe default",
    "head_contract": "15 observed and 8 latent summaries; tool capability is not permission"}
MODEL_SHA256 = fingerprint(MODEL_SPEC)
ACTION_KINDS = frozenset({"tool_proposed", "tool_call", "authorization_checked", "action_cancelled"})


@dataclass(frozen=True)
class CapabilityActionStep(ActionStep):
    capability_registry_sha256: str


class CapabilityActionCompiler(ActionEventCompiler):
    step_type = CapabilityActionStep

    def __init__(self, *, tool_registry, available_tools=None, **context):
        if not isinstance(tool_registry, ToolCapabilityRegistry):
            raise ValueError("a reviewed public tool registry is required")
        if "allowed_tools" in context:
            raise ValueError("use available_tools: tool availability is not action authorization")
        if available_tools is not None and (not isinstance(available_tools, dict) or any(
            not isinstance(actor, str) or not actor or not isinstance(names, (list, tuple))
            or any(not isinstance(name, str) or not name for name in names)
            for actor, names in available_tools.items())):
            raise ValueError("available_tools must map actors to public tool names")
        self._registry = ToolCapabilityRegistry(tool_registry.specification())
        self._available_tools = copy.deepcopy(available_tools)
        super().__init__(**context)

    def _is_action(self, event):
        return event.get("event_type") in ACTION_KINDS

    def _validate_event(self, event):
        # Argument payload is opaque data, not a feature namespace. A legitimate
        # argument named 'label' remains in the binding but cannot become a target.
        reject_outcomes({k: v for k, v in event.items() if k != "arguments"}, "event")

    def _sanitized(self, event):
        if self._is_action(event):
            # The legacy compiler must not learn anything from tool aliases,
            # bare permission flags or opaque action arguments.
            clean = {k: event[k] for k in ("event_id", "sequence", "actor") if k in event}
            return {**clean, "event_type": "action_observation"}
        return super()._sanitized(event)

    def _binding(self, event):
        bound = {}
        for field in BINDING_FIELDS:
            value = event.get(field)
            if field == "arguments":
                if not isinstance(value, dict):
                    raise ValueError("action requires structured arguments")
            elif field in {"principal", "resource"} and value is None:
                pass
            elif not isinstance(value, str) or not value:
                raise ValueError(f"action requires explicit {field}; empty scope is not unknown")
            bound[field] = copy.deepcopy(value)
        return bound, hashlib.sha256(canonical(bound).encode()).hexdigest()

    def _action_metadata(self, event, bound):
        profile = self._registry.resolve(bound["tool"])
        missing = [field for field in ("principal", "resource") if bound[field] is None]
        return {"capability_profile": profile, "scope_complete": not missing, "missing_scope_fields": missing}

    def _validate_check_scope(self, bound):
        if any(bound[field] is None for field in ("principal", "resource")):
            raise ValueError("a permission check requires complete action scope; partial scope remains unknown")

    def _operation_parameter(self, operation):
        return operation.parameter + (EXTRA_AUTH if operation.parameter >= TRANSMIT else 0)

    def _authorization_parameter(self, action, status):
        capability = action["capability_profile"]["capability"]
        return super()._authorization_parameter(action, status) + CAPABILITIES.index(capability) * AUTH_WIDTH

    def push(self, event):
        step = super().push(event)
        if self.last_was_duplicate:
            return step
        # Every observed tool operation can require scope/authorization,
        # including reading private information. No 'harmless-name' bypass.
        actors = {a["node"] for a in step.actions}
        for actor, names in (self._available_tools or {}).items():
            if names:
                actors.add(self._base.aliases.get(actor, actor))
        capable = tuple(sorted(self._base_indices[index] for (kind, identity), index in self._base.indices.items()
                               if kind == "node" and identity in actors))
        active = [a for a in step.actions if a["phase"] != "cancelled"]
        n = max(1, len(active))
        observed = step.observables.copy()
        observed[11:13] = [self._available_tools is not None,
                          len(set(capable).intersection(step.observed_nodes)) / max(1, len(step.observed_nodes))]
        observed = np.r_[observed, sum(a["capability_profile"]["capability"] == "unknown" for a in active) / n,
                         sum(not a["scope_complete"] for a in active) / n]
        actions = {a["action_id"]: a for a in step.actions}
        operations = []
        for op in step.operations:
            if op.evidence.get("template") == "action_authorization":
                a = actions[op.evidence["action_id"]]
                op = replace(op, evidence={**op.evidence, "registry_id": MODEL_SPEC["registry_id"],
                    "capability": a["capability_profile"]["capability"],
                    "capability_registry_sha256": self._registry.sha256, "scope_complete": a["scope_complete"]})
            operations.append(op)
        self.last_step = CapabilityActionStep(step.sequence, tuple(operations), observed, step.observed_nodes,
            capable, step.terminal_nodes, step.pending_pairs, step.dispatched_pairs, step.actions,
            step.ignored_evidence, self._registry.sha256)
        return copy.deepcopy(self.last_step)


def compile_capability_trace(trace, **context):
    reject_outcomes({k: v for k, v in trace.items() if k != "events"})
    compiler = CapabilityActionCompiler(**context)
    steps = tuple(compiler.push(event) for event in trace.get("events", []))
    if not steps:
        raise ValueError("a nonempty episode is required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), steps, compiler.workflow)


def coverage(step):
    active = [a for a in step.actions if a["phase"] != "cancelled"]
    return {"registry_sha256": step.capability_registry_sha256,
        "active_or_dispatched_action_count": len(active),
        "unknown_capability_actions": [a["action_id"] for a in active if a["capability_profile"]["capability"] == "unknown"],
        "incomplete_scope_actions": [a["action_id"] for a in active if not a["scope_complete"]],
        "meaning": "coverage diagnostics, not permission or safety certificates"}


class CapabilityActionBN(ActionScopedJointBN):
    head_index, confidence_index = CAP_HEAD, CAP_CONFIDENCE
    parameter_names, observable_names = PARAMETER_NAMES, CAP_OBSERVABLES
    model_name, step_type = "CapabilityActionBN-v4", CapabilityActionStep

    def initial_parameters(self):
        old = super().initial_parameters()
        return np.r_[old[:AUTHORIZE], np.tile(old[AUTHORIZE:TRANSMIT], len(CAPABILITIES)),
                     old[TRANSMIT:HEAD], old[HEAD:HEAD + 13], [0., 0.], old[HEAD + 13:]]

    def _forward(self, theta, trace, retain=False):
        for step in trace.steps:
            self._validate_step(step)
        if len({s.capability_registry_sha256 for s in trace.steps}) != 1:
            raise ValueError("one fixed capability registry is required within an episode")
        return super()._forward(theta, trace, retain)

    def parameters(self):
        result = super().parameters()
        result["capability_schema"] = {**copy.deepcopy(MODEL_SPEC), "sha256": MODEL_SHA256}
        result["limitations"] += ["Manual primary capability annotations are not verified runtime effects.",
            "Unknown capability and partial scope are modeled, not inferred safe.",
            "No learned performance, automatic text/profile extraction or scalable inference claim."]
        return result

    def explain(self, trace, **kwargs):
        return {**super().explain(trace, **kwargs), "coverage": coverage(trace.steps[-1])}


class CapabilityRiskStream(ActionRiskStream):
    compiler_type, model_type = CapabilityActionCompiler, CapabilityActionBN

    def snapshot(self):
        return {**super().snapshot(), "coverage": coverage(self.compiler.last_step)}
