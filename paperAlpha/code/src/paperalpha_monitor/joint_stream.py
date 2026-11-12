"""Incremental compilation and exact online filtering for DynamicJointBN.

Only push(event) reveals runtime facts. Past events are not replayed when an
event arrives. Newly allocated binary variables start at zero; existing joint
mass, including shared-source dependencies, is retained unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from scipy.special import expit

from .dynamic_joint_bn import (
    ADOPT, AUTHORIZE, CLASSES, CONFIDENCE, EXPECTED, HEAD, N_PARAMETERS,
    OBSERVABLES, PARAMETER_NAMES, ROLES, TRANSMIT, KNOWLEDGE, Operation, Step,
    latent_features, state_bits, transition, transition_pairs,
)
from .streaming_bn import EvidenceStream, PRIVILEGED_TOOLS, reject_outcomes, validate_workflow


class JointEventCompiler:
    """One episode's observed evidence, with stable append-only variable IDs."""

    def __init__(self, *, workflow=None, roles=None, allowed_tools=None, permission_mode="observed", max_variables=10):
        if not isinstance(max_variables, int) or isinstance(max_variables, bool) or max_variables < 1:
            raise ValueError("max_variables must be a positive integer")
        self.max_variables = max_variables
        self.workflow = copy.deepcopy(workflow)
        self.expected = validate_workflow(self.workflow)
        self.aliases = copy.deepcopy((self.workflow or {}).get("agent_to_node", {}))
        if not isinstance(self.aliases, dict) or any(not isinstance(k, str) or not k or v not in self.workflow["nodes"] for k, v in self.aliases.items()):
            raise ValueError("agent_to_node must map agent IDs to declared workflow nodes")
        if any(k in self.workflow["nodes"] and k != v for k, v in self.aliases.items()):
            raise ValueError("a declared workflow node cannot alias a different node")
        self.roles = copy.deepcopy(roles if roles is not None else (self.workflow or {}).get("roles", {}))
        if not isinstance(self.roles, dict) or any(not isinstance(k, str) or not k or v not in ROLES for k, v in self.roles.items()):
            raise ValueError("roles must use the declared role categories")
        self.allowed_tools = copy.deepcopy(allowed_tools if allowed_tools is not None else {})
        self.validator = EvidenceStream(allowed_tools=self.allowed_tools, permission_mode=permission_mode)
        self.variables, self.indices = [], {}
        self.seen_nodes, self.capable, self.terminal, self.origins = set(), set(), set(), set()
        self.seen_ingestions, self.seen_auth_actions, self.auth_status = set(), set(), {}
        self.counters = {"messages": 0, "untrusted": 0, "unknown": 0, "unexpected": 0}
        self.known_capability_nodes = {self.aliases.get(actor, actor) for actor, names in self.allowed_tools.items()
                                       if set(names).intersection(PRIVILEGED_TOOLS)}
        self.last_step, self.event_count, self.last_was_duplicate = None, 0, False
        if self.workflow is not None:
            for identity in self.workflow["nodes"]:
                self._node(identity)
            nonterminal = {left for left, _ in self.expected}
            self.terminal = {self.indices[("node", identity)] for identity in self.workflow["nodes"] if identity not in nonterminal}
            self.capable.update(self.indices[("node", identity)] for identity in self.workflow["nodes"] if identity in self.known_capability_nodes)

    def _variable(self, key, metadata):
        if key not in self.indices:
            if len(self.variables) >= self.max_variables:
                raise ValueError(f"exact joint state exceeds {self.max_variables} variables; explicitly group the workflow or choose a separately validated approximation")
            self.indices[key] = len(self.variables)
            self.variables.append(metadata)
        return self.indices[key]

    def _node(self, actor):
        identity = self.aliases.get(actor, actor)
        return self._variable(("node", identity), {"kind": "workflow_node", "id": identity,
                                                   "role": self.roles.get(identity, "unknown")})

    def _preflight(self, event):
        """Check allocation/field errors before the event validator commits."""
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        reject_outcomes(event, "event")
        actor, kind = event.get("actor"), event.get("event_type")
        if not isinstance(actor, str) or not actor or not isinstance(kind, str) or not kind:
            raise ValueError("event.actor and event.event_type must be nonempty strings")
        if event.get("data_class") is not None and not isinstance(event["data_class"], str):
            raise ValueError("data_class must be a string if provided")
        keys = {("node", self.aliases.get(actor, actor))}
        if kind in {"message_sent", "delegation"}:
            target = event.get("target")
            if not isinstance(target, str) or not target:
                raise ValueError("message target is required")
            keys.add(("node", self.aliases.get(target, target)))
        source_id = event.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a nonempty string if provided")
        if kind == "input_received":
            origin = "source:" + source_id if source_id is not None else "entry_agent:" + actor
            keys.add(("origin", origin))
        if len(set(self.indices).union(keys)) > self.max_variables:
            raise ValueError(f"exact joint state exceeds {self.max_variables} variables; event was not accepted")
        return keys

    def push(self, event):
        self._preflight(event)
        previous_sequence = self.validator.last_sequence
        sequence = self.validator.push(event).sequence
        duplicate = sequence == previous_sequence
        operations = []
        actor = self.aliases.get(event["actor"], event["actor"])
        receiver = self._node(event["actor"])
        role = ROLES.index(self.roles.get(actor, "unknown"))
        self.seen_nodes.add(receiver)
        if actor in self.known_capability_nodes:
            self.capable.add(receiver)
        kind = event["event_type"]
        value = event.get("data_class")
        trust = CLASSES.index(value) if value in CLASSES else 2
        evidence = {"sequence": sequence, "actor": event["actor"], "node": actor, "event_type": kind}
        if not duplicate and kind == "input_received":
            origin = "source:" + event["source_id"] if event.get("source_id") else "entry_agent:" + event["actor"]
            source = self._variable(("origin", origin), {"kind": "origin_hypothesis", "id": origin})
            if source not in self.origins:
                cell = KNOWLEDGE.cell_index("source", {"trust": CLASSES[trust]})
                operations.append(Operation(source, None, cell, 0., False,
                                            {**evidence, "template": "source", "origin": origin, "trust": CLASSES[trust]}))
                self.origins.add(source)
            if (source, receiver) not in self.seen_ingestions:
                cell = KNOWLEDGE.cell_index("adoption", {"role": ROLES[role], "trust": CLASSES[trust]})
                operations.append(Operation(receiver, source, cell, 0., False,
                                            {**evidence, "template": "adoption", "origin": origin, "role": ROLES[role], "trust": CLASSES[trust]}))
                self.seen_ingestions.add((source, receiver))
        if not duplicate and kind in {"message_sent", "delegation"}:
            sender, receiver = receiver, self._node(event["target"])
            target = self.aliases.get(event["target"], event["target"])
            self.seen_nodes.add(receiver)
            if target in self.known_capability_nodes:
                self.capable.add(receiver)
            expected_state = 0 if self.expected is None else (1 if actor == target or (actor, target) in self.expected else 2)
            confidence = event.get("edge_confidence", 1.)
            if confidence == 0:
                trust = 2
            self.counters["messages"] += 1
            self.counters["untrusted"] += trust == 1
            self.counters["unknown"] += trust == 2
            self.counters["unexpected"] += expected_state == 2
            if sender != receiver:
                cell = KNOWLEDGE.cell_index("transmission", {"trust": CLASSES[trust], "expected": EXPECTED[expected_state], "delegation": str(int(kind == "delegation"))})
                operations.append(Operation(receiver, sender, cell, 0. if trust == 2 else confidence - .5, True,
                                            {**evidence, "template": "transmission", "target": event["target"], "target_node": target,
                                             "trust": CLASSES[trust], "expected": EXPECTED[expected_state], "confidence": confidence}))
        if not duplicate and kind in {"tool_proposed", "tool_call"} and event.get("tool") in PRIVILEGED_TOOLS:
            receiver = self._node(event["actor"])
            self.capable.add(receiver)
            action = event.get("action_id", f"event:{sequence}")
            status = self.validator._permission(event)
            self.auth_status[action] = status
            if action not in self.seen_auth_actions:
                cell = KNOWLEDGE.cell_index("authorization", {"role": ROLES[role], "permission": KNOWLEDGE.states("permission")[status]})
                operations.append(Operation(receiver, None, cell, 0., False,
                                            {**evidence, "template": "authorization", "action_id": action,
                                             "tool": event["tool"], "permission_status": status, "role": ROLES[role]}))
                self.seen_auth_actions.add(action)
        messages = max(1, self.counters["messages"])
        observed_capable = self.capable.intersection(self.seen_nodes)
        observables = np.asarray([
            1., math.log1p(len(self.seen_nodes)) / math.log(17), math.log1p(self.counters["messages"]) / math.log(33),
            self.counters["untrusted"] / messages, self.counters["unknown"] / messages, float(self.expected is not None),
            self.counters["unexpected"] / messages, math.log1p(len(self.origins)) / math.log(9),
            float(bool(self.auth_status)), float(any(v > 0 for v in self.auth_status.values())),
            float(any(v == 2 for v in self.auth_status.values())), float(bool(self.allowed_tools)),
            len(observed_capable) / max(1, len(self.seen_nodes)),
        ])
        for operation in operations:
            KNOWLEDGE.describe_operation(operation)
        self.last_step = Step(sequence, tuple(operations), observables, tuple(sorted(self.seen_nodes)),
                              tuple(sorted(self.capable)), tuple(sorted(self.terminal)))
        self.event_count += int(not duplicate)
        self.last_was_duplicate = duplicate
        return copy.deepcopy(self.last_step)


class JointRiskStream:
    """An in-process, passive monitor; no model API calls or automatic blocking.

    Model parameters and policy context are copied at episode start. Threshold
    is optional and must come from a separate validation/calibration protocol.
    A missing or rejected event is never reported as a new safe observation.
    """

    parameter_names = tuple(PARAMETER_NAMES)
    compiler_type = JointEventCompiler

    def _state_features(self, bits, step):
        return latent_features(bits, step)

    def _fingerprint_extension(self):
        return {}

    def __init__(self, model, *, threshold=None, **context):
        self._theta = np.asarray(model.theta, dtype=float).copy()
        if self._theta.shape != (len(self.parameter_names),) or not np.isfinite(self._theta).all():
            raise ValueError("finite model parameter vector with the expected layout required")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                      or not math.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf)):
            raise ValueError("threshold must be a finite probability or the validated all-negative threshold")
        self.threshold = threshold
        self._use_messages, self._use_latent_head = model.use_messages, model.use_latent_head
        self._compiler = self.compiler_type(**context)
        self._distribution = np.zeros(1 << len(self._compiler.variables))
        self._distribution[0] = 1.
        self._probability, self._conditional, self._first_alarm = None, None, None
        self._last_transitions = []
        self._model_fingerprint = hashlib.sha256(json.dumps({"theta": self._theta.tolist(),
            "use_messages": self._use_messages, "use_latent_head": self._use_latent_head,
            "knowledge_specification_sha256": KNOWLEDGE.fingerprint, **self._fingerprint_extension()}, sort_keys=True).encode()).hexdigest()

    def push(self, event):
        # Compilation validates before mutation, including state-capacity errors.
        step = self._compiler.push(event)
        size = 1 << len(self._compiler.variables)
        if size != len(self._distribution):
            expanded = np.zeros(size)
            expanded[:len(self._distribution)] = self._distribution
            self._distribution = expanded
        transitions = []
        for operation in step.operations:
            if operation.is_message and not self._use_messages:
                continue
            logit = self._theta[operation.parameter] + (self._theta[CONFIDENCE] * operation.confidence_feature if operation.is_message else 0.)
            q = float(expit(logit))
            sources, targets = transition_pairs(len(self._compiler.variables), operation.receiver, operation.sender)
            self._distribution = transition(self._distribution, q, sources, targets)
            transitions.append({**operation.evidence, "q": q, "parameter": self.parameter_names[operation.parameter],
                                "knowledge": KNOWLEDGE.describe_operation(operation)})
        bits = state_bits(len(self._compiler.variables))
        features = self._state_features(bits, step)
        if not self._use_latent_head:
            features = np.zeros_like(features)
        self._conditional = expit(step.observables @ self._theta[HEAD:HEAD + len(OBSERVABLES)]
                                 + features @ self._theta[HEAD + len(OBSERVABLES):])
        self._probability = float(self._distribution @ self._conditional)
        self._last_transitions = transitions
        if self.threshold is not None and self._probability >= self.threshold and self._first_alarm is None:
            self._first_alarm = step.sequence
        return self.snapshot()

    def snapshot(self):
        if self._probability is None:
            raise ValueError("no event observed; no risk prediction is available")
        return {"sequence": self._compiler.last_step.sequence, "accepted_event_count": self._compiler.event_count,
                "duplicate_event": self._compiler.last_was_duplicate, "risk_probability": self._probability,
                "alarm": None if self.threshold is None else self._probability >= self.threshold,
                "threshold": self.threshold, "first_alarm_sequence": self._first_alarm,
                "model_fingerprint": self._model_fingerprint,
                "target": "terminal task risk given events received so far; not current confirmed harm"}

    def explain(self):
        result = self.snapshot()
        bits = state_bits(len(self._compiler.variables))
        variables = self._compiler.variables
        marginals = self._distribution @ bits
        important = np.argsort(self._distribution)[-8:][::-1]
        selected = set(important.tolist())
        remainder = [i for i in range(len(self._distribution)) if i not in selected]
        return {**result, "state_count": len(self._distribution), "mass_sum": float(self._distribution.sum()),
                "knowledge_registry": {"id": KNOWLEDGE.registry_id, "specification_sha256": KNOWLEDGE.fingerprint},
                "variables": [{**copy.deepcopy(value), "marginal": float(marginals[i])} for i, value in enumerate(variables)],
                "latest_event_transitions": copy.deepcopy(self._last_transitions),
                "observed_features": dict(zip(OBSERVABLES, self._compiler.last_step.observables.tolist())),
                "system_mixture_terms": [{"active_variables": [copy.deepcopy(variables[j]) for j in np.where(bits[i] > 0)[0]],
                    "state_mass": float(self._distribution[i]), "conditional_system_risk": float(self._conditional[i]),
                    "risk_contribution": float(self._distribution[i] * self._conditional[i])} for i in important],
                "remaining_risk_contribution": float(self._distribution[remainder] @ self._conditional[remainder]),
                "interpretation": "exact current model mixture and latent marginals; not causal attribution or identified compromise"}
