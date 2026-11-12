"""Exact action-variable elimination for the existing capability BN.

Conditional on visible evidence, each authorization bit has an independent
Bernoulli factor. Pending bits can therefore be summed out analytically.
Dispatched A/D bits affect all future readouts only through (any A, any A&D).
Their three reachable states remain JOINT with the information/source bits.

This is not independent-node approximation, truncated history or a new target.
It is specific to v4's factors/readout; new cross-action factors invalidate it.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import expit

from .capability_action_bn import (CapabilityActionBN, CapabilityActionCompiler, CAP_HEAD,
    CAP_CONFIDENCE, PARAMETER_NAMES, coverage)
from .dynamic_joint_bn import JointTrace, Step, state_bits, transition_pairs, latent_features
from .streaming_bn import reject_outcomes


HISTORY_STATES = ("no_authorization_risk", "authorization_without_information_overlap", "information_authorization_overlap")


class _LogicalCompiler(CapabilityActionCompiler):
    def _validate_capacity(self, keys, new_action, dispatch):
        # Only C/Z bits are enumerated by the compact filter. A/D identities
        # remain in audit metadata, never used to allocate a 2**actions vector.
        if len(set(self._base.indices).union(keys)) > self.max_variables:
            raise ValueError("compact information/source variable limit exceeded")


@dataclass(frozen=True)
class Dispatch:
    action_id: str
    node: int
    parameter: int


@dataclass(frozen=True)
class CompactActionStep(Step):
    pending: tuple[tuple[int, int], ...]  # (information node, authorization cell)
    dispatches: tuple[Dispatch, ...]
    actions: tuple[dict, ...]
    ignored_evidence: tuple[dict, ...]
    capability_registry_sha256: str


class CompactActionCompiler:
    def __init__(self, *, max_information_variables=12, **context):
        if "max_variables" in context:
            raise ValueError("use max_information_variables; actions are analytically eliminated")
        self._logical = _LogicalCompiler(max_variables=max_information_variables, **context)
        self.variables, self._indices, self.last_step = [], {}, None
        self._sync()

    def _sync(self):
        for old, variable in enumerate(self._logical.variables):
            if variable["kind"] in {"information_state", "origin_hypothesis"} and old not in self._indices:
                self._indices[old] = len(self.variables)
                self.variables.append(copy.deepcopy(variable))

    @property
    def workflow(self):
        return self._logical.workflow

    @property
    def event_count(self):
        return self._logical.event_count

    @property
    def last_was_duplicate(self):
        return self._logical.last_was_duplicate

    def push(self, event):
        step = self._logical.push(event)
        self._sync()
        mapping = self._indices
        node_ids = {v["id"]: i for i, v in enumerate(self.variables) if v["kind"] == "information_state"}
        actions = []
        for action in step.actions:
            item = copy.deepcopy(action)
            item.pop("authorization_variable")
            item.pop("information_at_dispatch_variable")
            item["authorization_parameter"] = self._logical._authorization_parameter(action, action["permission_status"])
            item["information_variable"] = node_ids[action["node"]]
            actions.append(item)
        by_id = {a["action_id"]: a for a in actions}
        operations, dispatches = [], []
        for op in step.operations:
            if op.rule in {"seed_or", "carrier_or"}:
                operations.append(replace(op, receiver=mapping[op.receiver],
                    sender=None if op.sender is None else mapping[op.sender]))
            elif op.rule == "copy":
                a = by_id[op.evidence["action_id"]]
                dispatches.append(Dispatch(a["action_id"], a["information_variable"], a["authorization_parameter"]))
            elif op.rule != "replace":
                raise ValueError("compact filter does not support this factor")
        pending = tuple((a["information_variable"], a["authorization_parameter"]) for a in actions if a["phase"] == "pending")
        remap = lambda ids: tuple(mapping[i] for i in ids)
        self.last_step = CompactActionStep(step.sequence, tuple(operations), step.observables.copy(),
            remap(step.observed_nodes), remap(step.capable_nodes), remap(step.terminal_nodes), pending,
            tuple(dispatches), tuple(actions), step.ignored_evidence, step.capability_registry_sha256)
        return copy.deepcopy(self.last_step)


def compile_compact_trace(trace, **context):
    reject_outcomes({k: v for k, v in trace.items() if k != "events"})
    compiler = CompactActionCompiler(**context)
    steps = tuple(compiler.push(event) for event in trace.get("events", []))
    if not steps:
        raise ValueError("a nonempty episode is required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), steps, compiler.workflow)


def pending_mixture(theta, bits, pending, derivatives=False):
    """P(no A), P(any A but no A*C), P(any A*C), conditional on C.

    Separate products over affected/unaffected nodes use log1p/softplus and
    expm1, avoiding division by vanishing products and cancellation near 1.
    Group counts preserve repeated actions sharing a cell; they are not dedup.
    """
    n, p = len(bits), len(theta)
    on, off = np.zeros(n), np.zeros(n)
    don = np.zeros((n, p)) if derivatives else None
    doff = np.zeros((n, p)) if derivatives else None
    for (node, parameter), count in Counter(pending).items():
        mask = bits[:, node].astype(bool)
        log_r = -np.logaddexp(0., theta[parameter]) * count
        on[mask] += log_r
        off[~mask] += log_r
        if derivatives:
            value = -count * expit(theta[parameter])
            don[mask, parameter] += value
            doff[~mask, parameter] += value
    pon, poff = np.exp(on), np.exp(off)
    none, only, overlap = np.exp(on + off), pon * (-np.expm1(off)), -np.expm1(on)
    probabilities = np.column_stack((none, only, overlap))
    if not derivatives:
        return probabilities, None
    jac = np.stack((none[:, None] * (don + doff),
                    only[:, None] * don - (pon * poff)[:, None] * doff,
                    -pon[:, None] * don), axis=1)
    return probabilities, jac


class CompactFilter:
    """Streaming exact filter and optional forward-mode parameter sensitivities."""
    def __init__(self, theta, *, use_messages=True, use_latent_head=True, derivatives=False):
        self.theta = np.asarray(theta, dtype=float).copy()
        if self.theta.shape != (len(PARAMETER_NAMES),) or not np.isfinite(self.theta).all():
            raise ValueError("finite capability-v4 parameter vector required")
        self.use_messages, self.use_latent_head, self.derivatives = use_messages, use_latent_head, derivatives
        self.distribution = np.array([[1., 0., 0.]])
        self.jacobian = np.zeros((1, 3, len(theta))) if derivatives else None
        self.dispatch_information = {}
        self.registry_sha256 = None
        self.probability, self.gradient, self.mixture, self.conditional = None, None, None, None

    def _expand(self, n):
        size = 1 << n
        if size < len(self.distribution):
            raise ValueError("information variable identities must be append-only")
        if size > len(self.distribution):
            fresh = np.zeros((size, 3))
            fresh[:len(self.distribution)] = self.distribution
            self.distribution = fresh
            if self.derivatives:
                jac = np.zeros((size, 3, len(self.theta)))
                jac[:len(self.jacobian)] = self.jacobian
                self.jacobian = jac

    def _move(self, sources, targets, q, dq):
        # Several history states can merge into one target. add.at is essential
        # here: advanced-index += would silently lose duplicate-target mass.
        flat = self.distribution.reshape(-1)
        mass = flat[sources].copy()
        moved = mass * q
        if self.derivatives:
            jac = self.jacobian.reshape(-1, len(self.theta))
            gradient = jac[sources].copy() * q + mass[:, None] * dq
            jac[sources] -= gradient
            np.add.at(jac, targets, gradient)
        flat[sources] -= moved
        np.add.at(flat, targets, moved)

    def _gate(self, parameter, confidence=0.):
        q = float(expit(self.theta[parameter] + self.theta[CAP_CONFIDENCE] * confidence))
        dq = None
        if self.derivatives:
            dq = np.zeros(len(self.theta))
            dq[parameter] += q * (1 - q)
            dq[CAP_CONFIDENCE] += confidence * q * (1 - q)
        return q, dq

    def _dispatch_gate(self, dispatch):
        return self._gate(dispatch.parameter)

    def apply(self, step, variables):
        if type(step) is not CompactActionStep:
            raise ValueError("compact filter requires the compact compiler")
        if self.registry_sha256 is not None and self.registry_sha256 != step.capability_registry_sha256:
            raise ValueError("one fixed capability registry is required within an episode")
        self.registry_sha256 = step.capability_registry_sha256
        self._expand(len(variables))
        bits = state_bits(len(variables))
        for op in step.operations:
            if op.is_message and not self.use_messages:
                continue
            q, dq = self._gate(op.parameter, op.confidence_feature if op.is_message else 0.)
            sources, targets = transition_pairs(len(variables), op.receiver, op.sender)
            self._move((sources[:, None] * 3 + np.arange(3)).ravel(),
                       (targets[:, None] * 3 + np.arange(3)).ravel(), q, dq)
        for dispatch in step.dispatches:
            q, dq = self._dispatch_gate(dispatch)
            marginal = self.distribution.sum(axis=1) @ bits[:, dispatch.node]
            self.dispatch_information[dispatch.action_id] = float(marginal)
            clean = np.flatnonzero(bits[:, dispatch.node] == 0)
            affected = np.flatnonzero(bits[:, dispatch.node] == 1)
            sources = np.r_[clean * 3, affected * 3, affected * 3 + 1]
            targets = np.r_[clean * 3 + 1, affected * 3 + 2, affected * 3 + 2]
            self._move(sources, targets, q, dq)
        pending, pending_jac = pending_mixture(self.theta, bits, step.pending, self.derivatives)
        info = latent_features(bits, step)
        any_risk, overlap = np.array([0., 1., 1.]), np.array([0., 0., 1.])
        if not self.use_latent_head:
            info, any_risk, overlap = np.zeros_like(info), np.zeros(3), np.zeros(3)
        b = self.theta[CAP_HEAD:CAP_HEAD + 23]
        logits = (step.observables @ b[:15] + info @ b[15:19])[:, None, None]
        logits = logits + (b[19] * any_risk + b[21] * overlap)[None, None, :]
        logits = logits + (b[20] * any_risk + b[22] * overlap)[None, :, None]
        conditional = expit(logits)
        weights = self.distribution[:, :, None] * pending[:, None, :]
        probability = float(np.sum(weights * conditional))
        self.probability, self.mixture, self.conditional = probability, weights, conditional
        if self.derivatives:
            grad = np.einsum("shk,sp,shp->k", self.jacobian, pending, conditional)
            grad += np.einsum("sh,spk,shp->k", self.distribution, pending_jac, conditional)
            dh = weights * conditional * (1 - conditional)
            head = np.r_[dh.sum() * step.observables, np.einsum("shp,si->i", dh, info),
                np.sum(dh * any_risk[None, None, :]), np.sum(dh * any_risk[None, :, None]),
                np.sum(dh * overlap[None, None, :]), np.sum(dh * overlap[None, :, None])]
            grad[CAP_HEAD:CAP_HEAD + 23] += head
            self.gradient = grad
        return probability

    def explain(self, step, variables):
        bits = state_bits(len(variables))
        marginals = self.distribution.sum(axis=1) @ bits
        actions = [{**copy.deepcopy(a), "authorization_risk_hypothesis": float(expit(self.theta[a["authorization_parameter"]])),
                    "information_at_dispatch_hypothesis": self.dispatch_information.get(a["action_id"])} for a in step.actions]
        # Nine named system contributions, exactly marginalizing information
        # bits. This is not the old full per-action joint-state explanation.
        contributions = np.sum(self.mixture * self.conditional, axis=0)
        return {"risk_probability": self.probability, "sequence": step.sequence,
            "mass_sum": float(self.distribution.sum()), "information_binary_variables": len(variables),
            "retained_joint_states": int(self.distribution.size), "history_states": list(HISTORY_STATES),
            "history_state_probabilities": self.distribution.sum(axis=0).tolist(),
            "variables": [{**copy.deepcopy(v), "marginal": float(marginals[i])} for i, v in enumerate(variables)],
            "actions": actions, "coverage": coverage(step), "ignored_evidence": copy.deepcopy(step.ignored_evidence),
            "system_mixture_terms": [{"history": HISTORY_STATES[h], "pending": HISTORY_STATES[p],
                                      "risk_contribution": float(contributions[h, p])} for h in range(3) for p in range(3)],
            "interpretation": "Exact v4 score-preserving action elimination; no arbitrary joint queries over discarded action bits, no causal or calibrated safety claim."}


class CompactCapabilityBN(CapabilityActionBN):
    model_name, step_type = "CapabilityActionBN-v4-exact-compact", CompactActionStep

    def _engine(self, theta, *, derivatives=False):
        return CompactFilter(theta, use_messages=self.use_messages, use_latent_head=self.use_latent_head, derivatives=derivatives)

    def _forward(self, theta, trace, retain=False):
        if retain:
            raise ValueError("compact filter uses forward sensitivities, not the full-action reverse tape")
        engine = self._engine(theta)
        probabilities = [engine.apply(step, trace.variables) for step in trace.steps]
        return np.asarray(probabilities), [], engine.distribution

    def episode_loss_gradient(self, theta, trace, label, prefix_training=True):
        engine = self._engine(theta, derivatives=True)
        loss, gradient = 0., np.zeros_like(theta)
        for i, step in enumerate(trace.steps):
            probability = engine.apply(step, trace.variables)
            weight = 1 / len(trace.steps) if prefix_training else float(i == len(trace.steps) - 1)
            safe = np.clip(probability, 1e-12, 1 - 1e-12)
            loss += weight * (-label * np.log(safe) - (1 - label) * np.log1p(-safe))
            gradient += weight * (safe - label) / (safe * (1 - safe)) * engine.gradient
        return float(loss), gradient

    def parameters(self):
        result = super().parameters()
        result["training"] = "Analytic forward sensitivities through exact action elimination; same v4 loss and parameters."
        result["inference"] = {"method": "exact_v4_action_elimination", "history_states": list(HISTORY_STATES),
            "preconditions": ["authorization factors depend only on visible capability/role/latest check",
                "authorization bits do not propagate as information", "dispatched factors cannot be revised",
                "head uses only pending/dispatched any-authorization and any-overlap summaries"],
            "complexity": "3 * 2**information_source_bits state masses, not exponential in number of actions"}
        result["limitations"] = [s for s in result["limitations"] if not s.startswith("Exact state grows") and "scalable inference claim" not in s]
        result["limitations"] += ["Information/source-state inference remains exponential; action metadata still grows with history.",
            "Compression is exact only for v4 factors/readout. New cross-action or revised historical factors need a new derivation.",
            "No new accuracy, calibration, automatic semantic extraction or arbitrary joint action explanation."]
        return result

    def explain(self, trace, *, include_graph=False):
        if include_graph:
            raise ValueError("compact executable graph export is not implemented")
        engine = self._engine(self.theta)
        probabilities = [engine.apply(step, trace.variables) for step in trace.steps]
        return {**engine.explain(trace.steps[-1], trace.variables), "model_name": self.model_name,
                "prefix_probabilities": probabilities}


class CompactCapabilityStream:
    def __init__(self, model, *, threshold=None, **context):
        if type(model) is not CompactCapabilityBN:
            raise ValueError("CompactCapabilityStream requires CompactCapabilityBN")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf)):
            raise ValueError("threshold must be a finite validated probability threshold")
        self.model = copy.deepcopy(model)
        self.compiler = CompactActionCompiler(**context)
        self.engine = self.model._engine(self.model.theta)
        self.threshold, self.first_alarm = threshold, None

    def push(self, event):
        step = self.compiler.push(event)
        probability = self.engine.apply(step, self.compiler.variables)
        if self.threshold is not None and probability >= self.threshold and self.first_alarm is None:
            self.first_alarm = step.sequence
        return self.snapshot()

    def snapshot(self):
        if self.engine.probability is None:
            raise ValueError("no event has been accepted")
        return {"model_name": self.model.model_name, "risk_probability": self.engine.probability,
            "sequence": self.compiler.last_step.sequence, "accepted_event_count": self.compiler.event_count,
            "duplicate_event": self.compiler.last_was_duplicate, "threshold": self.threshold,
            "alarm": None if self.threshold is None else self.engine.probability >= self.threshold,
            "first_alarm_sequence": self.first_alarm, "coverage": coverage(self.compiler.last_step),
            "retained_joint_states": int(self.engine.distribution.size)}

    def explain(self):
        self.snapshot()
        return {**self.engine.explain(self.compiler.last_step, self.compiler.variables), **self.snapshot()}
