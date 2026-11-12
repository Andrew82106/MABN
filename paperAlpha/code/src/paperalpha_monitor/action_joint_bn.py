"""Differentiable joint BN with replaceable pending action risk factors.

Authorization estimates are action-scoped; dispatch-time information is
snapshotted so later messages cannot rewrite the earlier action's context.
This conditional state model is not a Bayesian sensor-likelihood update.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.special import expit

from .action_evidence import ActionStep, ActionEventCompiler, SPEC, SPEC_SHA256
from .dynamic_joint_bn import (DynamicJointBN, HEAD, CONFIDENCE, OBSERVABLES, KNOWLEDGE,
    PARAMETER_NAMES as BASE_NAMES, initial_parameters, state_bits, transition_pairs, transition, latent_features)

ACTION_OBSERVABLES = tuple(SPEC["observable_overrides"].get(str(i), name) for i, name in enumerate(OBSERVABLES))
PARAMETER_NAMES = list(BASE_NAMES[:HEAD]) + ["head:" + name for name in [*ACTION_OBSERVABLES, *SPEC["information_head_features"], *SPEC["head_features"]]]


def action_features(bits, step):
    if not isinstance(step, ActionStep):
        raise ValueError("ActionScopedJointBN requires the action compiler")
    def any_flags(pairs, overlap):
        result = np.zeros(len(bits), dtype=bool)
        for authorization, information in pairs:
            value = bits[:, authorization].astype(bool)
            result |= value & bits[:, information].astype(bool) if overlap else value
        return result
    return np.column_stack((latent_features(bits, step), any_flags(step.pending_pairs, False),
                            any_flags(step.dispatched_pairs, False), any_flags(step.pending_pairs, True),
                            any_flags(step.dispatched_pairs, True))).astype(float)


def update(distribution, op, theta, n, confidence_index=CONFIDENCE):
    q = 1. if op.rule == "copy" else float(expit(theta[op.parameter] + (theta[confidence_index] * op.confidence_feature if op.is_message else 0.)))
    sources, targets = transition_pairs(n, op.receiver, op.sender)
    if op.rule == "replace":
        mass = distribution[sources] + distribution[targets]
        result = distribution.copy()
        result[sources], result[targets] = mass * (1 - q), mass * q
    elif op.rule in {"copy", "seed_or", "carrier_or"}:
        mass = distribution[sources].copy()
        result = transition(distribution, q, sources, targets)
    else:
        raise ValueError("unsupported action-state update")
    return result, (op, q, sources, targets, mass)


class ActionScopedJointBN(DynamicJointBN):
    head_index = HEAD
    confidence_index = CONFIDENCE
    parameter_names = tuple(PARAMETER_NAMES)
    observable_names = ACTION_OBSERVABLES
    model_name = SPEC["model_name"]
    step_type = ActionStep

    def _validate_step(self, step):
        if type(step) is not self.step_type:
            raise ValueError(f"{self.model_name} requires its matching action compiler")

    def initial_parameters(self):
        return np.r_[initial_parameters(), SPEC["head_initial_coefficients"]]

    def _state_features(self, bits, step):
        return action_features(bits, step)

    def _forward(self, theta, trace, retain=False):
        bits = state_bits(trace.n_variables)
        distribution = np.zeros(len(bits))
        distribution[0] = 1.
        probabilities, tape = [], []
        for step in trace.steps:
            self._validate_step(step)
            operations = []
            for op in step.operations:
                if op.is_message and not self.use_messages:
                    continue
                distribution, cache = update(distribution, op, theta, trace.n_variables, self.confidence_index)
                if retain:
                    operations.append(cache)
            features = action_features(bits, step)
            if not self.use_latent_head:
                features = np.zeros_like(features)
            boundary = self.head_index + len(self.observable_names)
            conditional = expit(step.observables @ theta[self.head_index:boundary] + features @ theta[boundary:])
            probability = float(distribution @ conditional)
            probabilities.append(probability)
            if retain:
                tape.append((operations, distribution.copy(), features, conditional, probability))
        return np.asarray(probabilities), tape, distribution

    def episode_loss_gradient(self, theta, trace, label, prefix_training=True):
        _, tape, _ = self._forward(theta, trace, retain=True)
        gradient, adjoint, loss = np.zeros_like(theta), np.zeros(1 << trace.n_variables), 0.
        for t in range(len(trace.steps) - 1, -1, -1):
            operations, distribution, features, conditional, probability = tape[t]
            weight = 1 / len(trace.steps) if prefix_training else float(t == len(trace.steps) - 1)
            safe = np.clip(probability, 1e-12, 1 - 1e-12)
            loss += weight * (-label * np.log(safe) - (1 - label) * np.log1p(-safe))
            derivative = weight * (safe - label) / (safe * (1 - safe))
            head_derivative = derivative * distribution * conditional * (1 - conditional)
            boundary = self.head_index + len(self.observable_names)
            gradient[self.head_index:boundary] += head_derivative.sum() * trace.steps[t].observables
            gradient[boundary:] += features.T @ head_derivative
            adjoint += derivative * conditional
            for op, q, sources, targets, mass in reversed(operations):
                if op.rule != "copy":
                    gate_gradient = float(mass @ (adjoint[targets] - adjoint[sources])) * q * (1 - q)
                    gradient[op.parameter] += gate_gradient
                    if op.is_message:
                        gradient[self.confidence_index] += gate_gradient * op.confidence_feature
                prior = (1 - q) * adjoint[sources] + q * adjoint[targets]
                adjoint[sources] = prior
                if op.rule == "replace":
                    adjoint[targets] = prior
        return float(loss), gradient

    def parameters(self):
        return {"name": self.model_name, "parameter_names": list(self.parameter_names), "theta": self.theta.tolist(),
                "knowledge_registry": {"id": KNOWLEDGE.registry_id, "specification_sha256": KNOWLEDGE.fingerprint},
                "action_registry": {"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256},
                "optimizer": copy.deepcopy(self.optimizer), "regularization": self.regularization, "max_iter": self.max_iter,
                "use_messages": self.use_messages, "use_latent_head": self.use_latent_head,
                "training": "Joint gradients through replace/copy/OR updates and episode-balanced prefix loss.",
                "limitations": copy.deepcopy(SPEC["boundaries"])}

    def explain(self, trace, *, include_graph=False):
        if include_graph:
            raise ValueError("action-scoped executable graph export is not implemented; do not substitute the typed v2 graph")
        probabilities, tape, distribution = self._forward(self.theta, trace, retain=True)
        return explain_state(self, trace.variables, trace.steps[-1], distribution, tape[-1][3], probabilities[-1],
                             prefix_probabilities=probabilities.tolist())


def explain_state(model, variables, step, distribution, conditional, probability, **extra):
    bits = state_bits(len(variables))
    marginal = distribution @ bits
    ranked = np.argsort(distribution)[-8:][::-1]
    remainder = np.ones(len(distribution), dtype=bool)
    remainder[ranked] = False
    actions = []
    for action in step.actions:
        item = copy.deepcopy(action)
        item["authorization_risk_hypothesis"] = float(marginal[action["authorization_variable"]])
        info = action["information_at_dispatch_variable"]
        item["information_at_dispatch_hypothesis"] = None if info is None else float(marginal[info])
        actions.append(item)
    return {"model_name": model.model_name, "risk_probability": float(probability), "sequence": step.sequence,
        "variables": [{**copy.deepcopy(value), "marginal": float(marginal[i])} for i, value in enumerate(variables)],
        "actions": actions, "mass_sum": float(distribution.sum()), "joint_state_count": len(distribution),
        "observed_features": dict(zip(model.observable_names, step.observables.tolist())),
        "latest_event_updates": [{**copy.deepcopy(op.evidence), "rule": op.rule,
            "parameter": None if op.rule == "copy" else model.parameter_names[op.parameter],
            "q": 1. if op.rule == "copy" else float(expit(model.theta[op.parameter] + (model.theta[model.confidence_index] * op.confidence_feature if op.is_message else 0.)))}
            for op in step.operations if model.use_messages or not op.is_message],
        "ignored_evidence": copy.deepcopy(step.ignored_evidence),
        "system_mixture_terms": [{"active_variables": [copy.deepcopy(variables[j]) for j in np.where(bits[i] > 0)[0]],
                                  "state_mass": float(distribution[i]), "conditional_system_risk": float(conditional[i]),
                                  "risk_contribution": float(distribution[i] * conditional[i])} for i in ranked],
        "remaining_risk_contribution": float(distribution[remainder] @ conditional[remainder]),
        "interpretation": "Conditional action-risk hypotheses and dispatch-time snapshots, not proven grants, harm or causal truth.", **extra}


class ActionRiskStream:
    compiler_type = ActionEventCompiler
    model_type = ActionScopedJointBN

    def __init__(self, model, *, threshold=None, **context):
        if type(model) is not self.model_type:
            raise ValueError(f"{type(self).__name__} requires {self.model_type.__name__}")
        self.model = copy.deepcopy(model)
        if self.model.theta.shape != (len(self.model.parameter_names),) or not np.isfinite(self.model.theta).all():
            raise ValueError("finite action model parameters required")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                      or not math.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf)):
            raise ValueError("threshold must be a finite validated probability threshold")
        self.threshold, self.first_alarm = threshold, None
        self.compiler = self.compiler_type(**context)
        self.distribution = np.zeros(1 << len(self.compiler.variables))
        self.distribution[0] = 1.
        self.probability, self.conditional = None, None

    def push(self, event):
        step = self.compiler.push(event)
        count = len(self.compiler.variables)
        if len(self.distribution) != 1 << count:
            expanded = np.zeros(1 << count)
            expanded[:len(self.distribution)] = self.distribution
            self.distribution = expanded
        for op in step.operations:
            if op.is_message and not self.model.use_messages:
                continue
            self.distribution, _ = update(self.distribution, op, self.model.theta, count, self.model.confidence_index)
        features = action_features(state_bits(count), step)
        if not self.model.use_latent_head:
            features = np.zeros_like(features)
        boundary = self.model.head_index + len(self.model.observable_names)
        self.conditional = expit(step.observables @ self.model.theta[self.model.head_index:boundary] + features @ self.model.theta[boundary:])
        self.probability = float(self.distribution @ self.conditional)
        if self.threshold is not None and self.probability >= self.threshold and self.first_alarm is None:
            self.first_alarm = step.sequence
        return self.snapshot()

    def snapshot(self):
        if self.probability is None:
            raise ValueError("no event has been accepted")
        return {"model_name": self.model.model_name, "risk_probability": self.probability, "sequence": self.compiler.last_step.sequence,
                "accepted_event_count": self.compiler.event_count, "duplicate_event": self.compiler.last_was_duplicate,
                "alarm": None if self.threshold is None else self.probability >= self.threshold,
                "threshold": self.threshold, "first_alarm_sequence": self.first_alarm}

    def explain(self):
        if self.probability is None:
            raise ValueError("no event has been accepted")
        return {**explain_state(self.model, self.compiler.variables, self.compiler.last_step, self.distribution, self.conditional, self.probability),
                **self.snapshot()}
