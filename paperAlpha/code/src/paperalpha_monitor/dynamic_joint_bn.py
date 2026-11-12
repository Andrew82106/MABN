"""Exactly filtered, differentiable conditional DBN for small workflow graphs.

Local source/adoption/authorization CPTs create node states. Actual messages
use shared learned transition CPTs. A conditional Bernoulli system head is
marginalized over the JOINT state, preserving shared-source/diamond dependence.
All parameters receive gradients from episode-weighted prefix outcome loss.

Latents are modeling hypotheses, not identifiable true agent compromise. Exact
enumeration is exponential: larger graphs fail explicitly, never silently use
an independence approximation. No roles/capabilities are inferred from future
events. The target is final episode outcome given a prefix, not current harm.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .streaming_bn import EvidenceStream, PRIVILEGED_TOOLS, reject_outcomes, validate_workflow

from .joint_knowledge import load_joint_registry

KNOWLEDGE = load_joint_registry()
ROLES = KNOWLEDGE.states("role")
CLASSES = KNOWLEDGE.states("trust")
EXPECTED = KNOWLEDGE.states("expected")
OBSERVABLES = KNOWLEDGE.observables
LATENT_SUMMARIES = KNOWLEDGE.latent_summaries
PARAMETER_NAMES = list(KNOWLEDGE.parameter_names)
ADOPT = KNOWLEDGE.offsets["adoption"]
AUTHORIZE = KNOWLEDGE.offsets["authorization"]
TRANSMIT = KNOWLEDGE.offsets["transmission"]
CONFIDENCE = KNOWLEDGE.confidence_offset
HEAD = KNOWLEDGE.head_offset
N_PARAMETERS = len(PARAMETER_NAMES)


def initial_parameters():
    return np.asarray(KNOWLEDGE.initial_values, dtype=float).copy()

@lru_cache(maxsize=20)
def state_bits(n):
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
    bits.setflags(write=False)
    return bits


@lru_cache(maxsize=2048)
def transition_pairs(n, receiver, sender):
    states = np.arange(1 << n)
    mask = (states & (1 << receiver)) == 0
    if sender is not None:
        mask &= (states & (1 << sender)) != 0
    sources = states[mask]
    targets = sources | (1 << receiver)
    sources.setflags(write=False)
    targets.setflags(write=False)
    return sources, targets


def transition(distribution, probability, sources, targets):
    """Bernoulli OR update; one-to-one zero-bit to one-bit transfers."""
    result = distribution.copy()
    transferred = distribution[sources] * probability
    result[sources] -= transferred
    result[targets] += transferred
    return result


@dataclass(frozen=True)
class Operation:
    receiver: int
    sender: int | None
    parameter: int
    confidence_feature: float
    is_message: bool
    evidence: dict


@dataclass(frozen=True)
class Step:
    sequence: int
    operations: tuple[Operation, ...]
    observables: np.ndarray
    observed_nodes: tuple[int, ...]
    capable_nodes: tuple[int, ...]
    terminal_nodes: tuple[int, ...]


@dataclass(frozen=True)
class JointTrace:
    variables: tuple[dict, ...]
    steps: tuple[Step, ...]
    workflow: dict | None

    @property
    def n_variables(self):
        return len(self.variables)


def _class(event):
    value = event.get("data_class")
    return CLASSES.index(value) if value in CLASSES else 2


def compile_trace(trace: dict, *, workflow: dict | None = None, roles: dict | None = None,
                  allowed_tools: dict | None = None, permission_mode="observed", max_variables=10) -> JointTrace:
    """Compile a batch with the same incremental evidence builder used online.

    Only declared pre-run context is available before the first push. Later
    allocation does not enter earlier readouts, and no message is replayed.
    """
    from .joint_stream import JointEventCompiler

    reject_outcomes(trace)
    compiler = JointEventCompiler(workflow=workflow, roles=roles, allowed_tools=allowed_tools,
                                  permission_mode=permission_mode, max_variables=max_variables)
    steps = tuple(compiler.push(event) for event in trace.get("events", []))
    if not steps:
        raise ValueError("empty episode is outside the prediction protocol")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), steps, copy.deepcopy(compiler.workflow))

def latent_features(bits, step):
    count = bits[:, step.observed_nodes].sum(axis=1)
    n = len(step.observed_nodes)
    return np.column_stack((
        count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
        bits[:, step.capable_nodes].sum(axis=1) / max(1, len(step.capable_nodes)),
        bits[:, step.terminal_nodes].sum(axis=1) / max(1, len(step.terminal_nodes)),
    ))


class DynamicJointBN:
    def __init__(self, regularization=.01, max_iter=100, use_messages=True, use_latent_head=True):
        self.regularization, self.max_iter = regularization, max_iter
        self.use_messages, self.use_latent_head = use_messages, use_latent_head
        self.theta = self.initial_parameters()
        self.optimizer = {}

    def initial_parameters(self):
        return initial_parameters()

    def _state_features(self, bits, step):
        return latent_features(bits, step)

    def _regularization_count(self):
        return len(self.theta)

    def _forward(self, theta, trace, retain=False):
        bits = state_bits(trace.n_variables)
        distribution = np.zeros(len(bits))
        distribution[0] = 1.
        probabilities, tape = [], []
        for step in trace.steps:
            operations = []
            for operation in step.operations:
                if operation.is_message and not self.use_messages:
                    continue
                logit = theta[operation.parameter] + (theta[CONFIDENCE] * operation.confidence_feature if operation.is_message else 0.)
                q = float(expit(logit))
                sources, targets = transition_pairs(trace.n_variables, operation.receiver, operation.sender)
                if retain:
                    operations.append((operation, q, sources, targets, distribution[sources].copy()))
                distribution = transition(distribution, q, sources, targets)
            features = self._state_features(bits, step)
            if not self.use_latent_head:
                features = np.zeros_like(features)
            logits = step.observables @ theta[HEAD:HEAD + len(OBSERVABLES)] + features @ theta[HEAD + len(OBSERVABLES):]
            conditional = expit(logits)
            probability = float(distribution @ conditional)
            probabilities.append(probability)
            if retain:
                tape.append((operations, distribution.copy(), features, conditional, probability))
        return np.asarray(probabilities), tape, distribution

    def episode_loss_gradient(self, theta, trace, label, prefix_training=True):
        probabilities, tape, _ = self._forward(theta, trace, retain=True)
        gradient = np.zeros_like(theta)
        adjoint = np.zeros(1 << trace.n_variables)
        loss = 0.
        weights = np.ones(len(trace.steps)) / len(trace.steps) if prefix_training else np.eye(1, len(trace.steps), len(trace.steps) - 1)[0]
        for t in range(len(trace.steps) - 1, -1, -1):
            operations, distribution, features, conditional, probability = tape[t]
            safe = np.clip(probability, 1e-12, 1 - 1e-12)
            loss += weights[t] * (-label * np.log(safe) - (1 - label) * np.log1p(-safe))
            derivative = weights[t] * (safe - label) / (safe * (1 - safe))
            head_derivative = derivative * distribution * conditional * (1 - conditional)
            gradient[HEAD:HEAD + len(OBSERVABLES)] += head_derivative.sum() * trace.steps[t].observables
            gradient[HEAD + len(OBSERVABLES):] += features.T @ head_derivative
            adjoint += derivative * conditional
            for operation, q, sources, targets, source_mass in reversed(operations):
                gate_gradient = float(source_mass @ (adjoint[targets] - adjoint[sources])) * q * (1 - q)
                gradient[operation.parameter] += gate_gradient
                if operation.is_message:
                    gradient[CONFIDENCE] += gate_gradient * operation.confidence_feature
                adjoint[sources] = (1 - q) * adjoint[sources] + q * adjoint[targets]
        return float(loss), gradient

    def loss_gradient(self, theta, traces, labels, prefix_training=True):
        loss, gradient = 0., np.zeros_like(theta)
        for trace, label in zip(traces, labels):
            value, derivative = self.episode_loss_gradient(theta, trace, label, prefix_training)
            loss += value / len(traces)
            gradient += derivative / len(traces)
        centered = theta - self.initial_parameters()
        loss += self.regularization * np.sum(centered ** 2) / self._regularization_count()
        gradient += 2 * self.regularization * centered / self._regularization_count()
        return float(loss), gradient

    def fit(self, traces, labels, *, prefix_training=True):
        labels = np.asarray(labels, dtype=float)
        if not len(traces) or len(traces) != len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError("one binary terminal outcome per compiled episode required")
        iterations = 0

        def progress(theta):
            nonlocal iterations
            iterations += 1
            if iterations % 10 == 0:
                print(f"joint BN iteration={iterations}", flush=True)

        fit = minimize(self.loss_gradient, self.theta, args=(traces, labels, prefix_training), jac=True,
                       method="L-BFGS-B", bounds=[(-8., 8.)] * len(self.theta), callback=progress,
                       options={"maxiter": self.max_iter, "ftol": 1e-9, "gtol": 1e-6})
        self.theta = fit.x
        self.optimizer = {"success": bool(fit.success), "iterations": int(fit.nit), "loss": float(fit.fun), "message": str(fit.message)}
        return self

    def predict(self, traces):
        return [self._forward(self.theta, trace)[0] for trace in traces]

    def explain(self, trace, *, include_graph=False):
        probabilities, tape, distribution = self._forward(self.theta, trace, retain=True)
        bits = state_bits(trace.n_variables)
        marginals = distribution @ bits
        conditional = tape[-1][3]
        important = np.argsort(distribution)[-8:][::-1]
        selected = set(important.tolist())
        remainder = [i for i in range(len(distribution)) if i not in selected]
        result = {
            "risk_probability": float(probabilities[-1]), "prefix_probabilities": probabilities.tolist(),
            "state_count": len(distribution), "mass_sum": float(distribution.sum()),
            "variables": [{**value, "marginal": float(marginals[i])} for i, value in enumerate(trace.variables)],
            "transitions": [{**operation.evidence, "q": q, "parameter": PARAMETER_NAMES[operation.parameter]}
                            for operations, *_ in tape for operation, q, *_ in operations],
            "system_mixture_terms": [{"active_variables": [trace.variables[j] for j in np.where(bits[i] > 0)[0]],
                                      "state_mass": float(distribution[i]), "conditional_system_risk": float(conditional[i]),
                                      "risk_contribution": float(distribution[i] * conditional[i])} for i in important],
            "remaining_risk_contribution": float(distribution[remainder] @ conditional[remainder]),
            "interpretation": "exact model mixture and latent marginals, not causal attribution or identifiable compromise truth",
        }
        if include_graph:
            from .joint_graph import compile_joint_graph
            graph = compile_joint_graph(trace)
            result["knowledge_graph"] = graph.to_dict()
            result["graph_risk_probability"] = float(graph.predict(self.theta, use_messages=self.use_messages,
                                                                    use_latent_head=self.use_latent_head)[-1])
        return result

    def parameters(self):
        return {"name": "DynamicJointBN-v1", "parameter_names": PARAMETER_NAMES, "theta": self.theta.tolist(),
                "knowledge_registry": {"id": KNOWLEDGE.registry_id, "specification_sha256": KNOWLEDGE.fingerprint},
                "optimizer": self.optimizer, "regularization": self.regularization, "max_iter": self.max_iter,
                "use_messages": self.use_messages, "use_latent_head": self.use_latent_head,
                "inference": "exact conditional joint-state filtering; no independent-marginal closure",
                "training": "joint gradients for local CPTs, message transitions and system head; episode-balanced prefix loss"}
