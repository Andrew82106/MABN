"""Compose learned disclosure states with evidence-bound shared authority.

The target is literal appearance AND failure of a declared referenced-authority
policy, for a fixed proposed recipient/purpose in a static authority window.
Not a general privacy law evaluator, causal model, or calibrated deployment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp

from .privacy_model_independent import ModelIndependentPrivacyBN, compile_packets, validate
from .privacy_flow_data import KINDS
from .privacy_supervision_controls import _solve
from .shared_authority_graph import AuthorityProofGraph, PRIOR_NAMES


@dataclass(frozen=True)
class ReleaseContext:
    task_id: str
    actor: str
    principal: str
    recipient: str
    purpose: str
    observed_through_sequence: int

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.task_id, self.actor, self.principal, self.recipient, self.purpose)):
            raise ValueError("explicit task, actor, principal, recipient and purpose required")
        if type(self.observed_through_sequence) is not int or self.observed_through_sequence < 0:
            raise ValueError("observed prefix cursor required")


@dataclass(frozen=True)
class ReleaseBinding:
    field_names: tuple[str, ...]
    action_id: str
    resource: str


class ScopedRelease:
    """One final message, one bound disclosure proposal per distinct literal.

    A packet's objects are the tracked literal universe, NOT a new allowlist.
    Legacy packets tracked only prohibited values; authorized-value training
    therefore needs explicitly collected new data. No legacy labels are reused
    as authority evidence or silently generalized to those new objects.
    """
    def __init__(self, packet, graph, context, bindings):
        validate(packet)
        if not isinstance(graph, AuthorityProofGraph) or not isinstance(context, ReleaseContext):
            raise ValueError("typed authority graph and release context required")
        graph.validate()
        if not graph.window.first_sequence <= context.observed_through_sequence <= graph.window.last_sequence:
            raise ValueError("prefix outside the declared fixed-authority window")
        fields = [tuple(sorted(g["field_names"])) for g in packet["groups"]]
        bindings = tuple(bindings)
        mapping = {}
        for binding in bindings:
            if not isinstance(binding, ReleaseBinding) or not isinstance(binding.field_names, tuple):
                raise ValueError("explicit typed literal-to-proposal bindings required")
            names = tuple(sorted(binding.field_names))
            if names in mapping or len(names) != len(set(names)):
                raise ValueError("one binding per distinct literal, including aliases")
            mapping[names] = binding
        if set(mapping) != set(fields) or len(mapping) != len(fields):
            raise ValueError("bindings must cover exactly the tracked literal groups")
        action_ids = [mapping[names].action_id for names in fields]
        if len(set(action_ids)) != len(action_ids) or set(action_ids) != set(graph.proofs):
            raise ValueError("each graph proof must bind exactly one tracked literal")
        for names, action_id in zip(fields, action_ids):
            proof = graph.proofs[action_id]
            bound, evidence = proof.evidence["action_binding"], proof.evidence
            expected = {"action_id": action_id, "actor": context.actor, "principal": context.principal,
                "tool": "release_literal", "resource": mapping[names].resource,
                "arguments": {"recipient": context.recipient, "purpose": context.purpose, "field_names": list(names)}}
            if bound != expected or evidence["task_id"] != context.task_id:
                raise ValueError("proposal differs from the task/recipient/purpose/literal binding")
            if proof.sequence > context.observed_through_sequence:
                raise ValueError("future proposal or evidence is unavailable at this prefix")
        self.packet, self.graph = copy.deepcopy(packet), copy.deepcopy(graph)
        self.context, self.bindings = context, tuple(copy.deepcopy(bindings))
        self.action_ids = tuple(action_ids)

    def validate(self):
        ScopedRelease(self.packet, self.graph, self.context, self.bindings)

    def specification(self):
        return {"knowledge_ids": ["RB009", "RB010", "RB011", "RB012", "RB013", "RB014"],
            "context": copy.deepcopy(self.context.__dict__), "bindings": [copy.deepcopy(b.__dict__) for b in self.bindings],
            "authority": self.graph.specification(), "presence_schema": self.packet["schema"],
            "target": "F_j AND referenced_authority_failure_j; count DISTINCT tracked literal values",
            "not_modeled": ["Whether sharing is necessary beyond an explicitly bound purpose",
                "Paraphrased disclosure", "Unobserved policy changes after the prefix", "Unbound recipients or other system hazards"]}


def count_distribution(probabilities):
    """Exact conditional count distribution; last axis is the literal axis."""
    q = np.asarray(probabilities, float)
    if q.ndim < 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("conditional Bernoulli probabilities required")
    dist = np.ones((*q.shape[:-1], 1))
    for i in range(q.shape[-1]):
        p = q[..., i, None]
        updated = np.zeros((*q.shape[:-1], dist.shape[-1] + 1))
        updated[..., :-1] += dist * (1 - p)
        updated[..., 1:] += dist * p
        dist = updated
    return dist


class PolicyGatedPrivacyBN:
    """39 presence + 3 shared-fact logits + 1 authorized-emission coefficient.

    Initial authority and role-state factors are conditionally independent.
    Final emissions may depend on authority through a learned coefficient.
    Dependencies within each component and shared literal identities survive.
    """
    MODEL_NAME = "PolicyGatedPrivacyBN-v1"
    EMISSION_NAMES = ("authorized_final_emission",)

    def __init__(self, presence=None, *, prior_logits=(0., 0., 0.), regularization=.001):
        self.presence = copy.deepcopy(presence or ModelIndependentPrivacyBN())
        if not isinstance(self.presence, ModelIndependentPrivacyBN):
            raise ValueError("identity-free stage presence model required")
        priors = np.asarray(prior_logits, float)
        if priors.shape != (3,) or not np.isfinite(priors).all() or not np.isfinite(regularization) or regularization < 0:
            raise ValueError("finite priors and nonnegative regularization required")
        self.n = len(self.presence.theta)
        self.names = [*("presence:" + n for n in self.presence.names), *("authority:" + n for n in PRIOR_NAMES), *self.EMISSION_NAMES]
        self.theta = np.r_[self.presence.theta, priors, np.zeros(len(self.EMISSION_NAMES))]
        self.regularization, self.optimizer = regularization, None

    def _emission_features(self, sample, failed):
        return (1 - failed)[..., None]

    def _emission_graph(self, sample, index, nodes):
        return [f"A:{index}"], [float(self.theta[-1]), 0.]

    def _components(self, sample, theta):
        if not isinstance(sample, ScopedRelease):
            raise ValueError("bound scoped release required")
        sample.validate()
        theta = np.asarray(theta, float)
        if theta.shape != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError("compatible finite parameter vector required")
        data = compile_packets([sample.packet])
        x, u, v, lu, lv, log_prior, qw, rf = self.presence._parts(data, theta[:self.n])
        _, _, mass, scores, failures = sample.graph.worlds(theta[self.n:self.n + 3])
        failed = np.array([failures[a] for a in sample.action_ids], float)  # literal, authority-world
        return data, x[0], u, v, lu[0], lv[0], log_prior[0], qw[0], rf[0], mass, scores, failed

    def predict(self, sample, *, at_least=1, theta=None):
        if type(at_least) is not int or at_least < 1:
            raise ValueError("positive integer violation count required")
        t = self.theta if theta is None else np.asarray(theta, float)
        data, x, u, v, lu, lv, prior, qw, rf, mass, _, failed = self._components(sample, t)
        shift = np.einsum("jbe,e->jb", self._emission_features(sample, failed), t[self.n + 3:])
        logits = rf[:, :, None] + shift[:, None, :]
        if data["stage"] == 1:
            q = expit(qw)[:, :, None]
            r = (1 - q) * expit(logits) + q * expit(logits + t[self.presence.d + len(KINDS)])
            states = np.exp(prior)
        else:
            log_post = prior + self.presence._evidence(qw[None], data["w"], data["mask"])[0]
            states = np.exp(log_post - logsumexp(log_post))
            r = expit(logits)
        joint_weights = states[:, None] * mass[None, :]
        local_violation = r * failed[:, None, :]
        distributions = count_distribution(local_violation.transpose(1, 2, 0))
        count_probabilities = np.sum(joint_weights[:, :, None] * distributions, axis=(0, 1))
        risk = float(count_probabilities[at_least:].sum())
        conditional_risk = distributions[..., at_least:].sum(axis=-1)
        per_world = states @ conditional_risk
        return {"model_probability": risk, "violation_count_probabilities": count_probabilities.tolist(),
            "appearance_marginals": np.einsum("sb,jsb->j", joint_weights, r).tolist(),
            "violation_marginals": np.einsum("sb,jsb->j", joint_weights, local_violation).tolist(),
            "reference_failure_marginals": (failed @ mass).tolist(), "stage_state_weights": states.tolist(),
            "authority_world_count": len(mass), "authority_state_conditional_range": [float(per_world.min()), float(per_world.max())],
            "range_meaning": "Range of learned presence-model predictions over compatible authority states; NOT confidence or logical safety bounds.",
            "at_least": at_least, "phase": "before proposed final release", "calibrated": False,
            "joint_likelihood_fitted": self.optimizer is not None,
            "jointly_trained": self.optimizer is not None and set(self.optimizer["trainable_names"]) == set(self.names),
            "optimized_parameter_count": 0 if self.optimizer is None else len(self.optimizer["trainable_names"]),
            "proves_system_safety": False,
            "fixed_window_assumption": True, "scope": sample.specification()}

    def probability_graph(self, sample, *, at_least=1):
        """Export executable binary factors, not just a picture of two scores."""
        if type(at_least) is not int or at_least < 1:
            raise ValueError("positive integer count threshold required")
        sample.validate()
        data = compile_packets([sample.packet])
        t, base, nodes = self.theta, self.presence, []
        x = np.r_[1., data["context"][0]]
        for fact in sample.graph.facts.values():
            nodes.append({"id": "fact:" + fact.identity, "parents": [], "kind": "binary_cpt",
                "probability_one": [float(expit(t[self.n + fact.kind]))], "meaning": fact.provenance})
        pu = float(expit(x @ t[2 * base.d:2 * base.d + base.k]))
        pv = x @ t[2 * base.d + base.k:2 * base.d + 2 * base.k]
        nodes.append({"id": "U", "parents": [], "kind": "binary_cpt", "probability_one": [pu]})
        nodes.append({"id": "V", "parents": ["U"] if base.coupled else [], "kind": "binary_cpt",
            "probability_one": [float(expit(pv + i * t[self.n - 1])) for i in (0, 1)] if base.coupled else [float(expit(pv))]})
        targets = []
        for j, group in enumerate(sample.packet["groups"]):
            cid, wid, aid, fid, hid = [f"{name}:{j}" for name in ("C", "W", "A", "F", "H")]
            proof = sample.graph.proofs[sample.action_ids[j]]
            paths = [["fact:" + f for f in path] for path in proof.paths]
            nodes.append({"id": aid, "parents": sorted({f for path in paths for f in path}),
                "kind": "all_paths_failed", "support_paths": paths, "binding_sha256": proof.binding_sha256})
            nodes.append({"id": cid, "parents": [], "kind": "binary_cpt",
                "probability_one": [float(group["coordinator_present"])], "observed": int(group["coordinator_present"])})
            w_probs = []
            for u in (0, 1):
                for c in (0, 1):
                    design = data["dc"][0, j].copy()
                    design[len(KINDS)] = c
                    w_probs.append(float(expit(design @ t[:base.d] + u * t[base.boost_start])))
            w_node = {"id": wid, "parents": ["U", cid], "kind": "binary_cpt", "probability_one": w_probs}
            if sample.packet["stage"] == 2:
                w_node["observed"] = int(group["worker_present"])
            nodes.append(w_node)
            emission_parents, emission_offsets = self._emission_graph(sample, j, nodes)
            f_probs = []
            for v in (0, 1):
                for w in (0, 1):
                    for offset in emission_offsets:
                        design = data["dw"][0, j].copy()
                        design[len(KINDS)] = w
                        f_probs.append(float(expit(design @ t[base.d:2 * base.d] + v * t[base.boost_start + 1] + offset)))
            nodes.append({"id": fid, "parents": ["V", wid, *emission_parents], "kind": "binary_cpt", "probability_one": f_probs,
                "field_names": list(group["field_names"])})
            nodes.append({"id": hid, "parents": [fid, aid], "kind": "and"})
            targets.append(hid)
        nodes.append({"id": "system_violation", "parents": targets, "kind": "count_at_least", "threshold": at_least})
        return {"schema": "executable-scoped-privacy-graph-v1", "nodes": nodes, "output_id": "system_violation",
            "conditional_context": data["context"][0].tolist(), "scope": sample.specification(),
            "semantics": "Binary CP tables in listed-parent big-endian order; observed-node likelihoods retained. Context is conditioned on, not an intervention graph."}

    def _likelihood(self, sample, f, authority_labels, theta, violation_label=None, at_least=1):
        data, x, u, v, lu, lv, prior, qw, rf, mass, score, failed = self._components(sample, theta)
        f = np.asarray(f)
        if data["stage"] != 2 or f.shape != (len(sample.packet["groups"]),) or not np.isin(f, [0, 1]).all():
            raise ValueError("observed worker prefix and aligned final-presence labels required")
        if len(authority_labels) != len(f) or any(a is not None and (type(a) is not int or a not in (0, 1)) for a in authority_labels):
            raise ValueError("aligned authority-failure labels, or explicit None, required")
        compatible = np.ones(len(mass), bool)
        for j, label in enumerate(authority_labels):
            if label is not None:
                compatible &= failed[j] == label
        if violation_label is not None:
            if type(violation_label) is not int or violation_label not in (0, 1):
                raise ValueError("optional binary system-violation label required")
            compatible &= (np.sum(f[:, None] * failed, axis=0) >= at_least) == violation_label
        if not compatible.any():
            raise ValueError("labels contradict the declared shared proof graph or system target")
        emission_features = self._emission_features(sample, failed)
        shift = np.einsum("jbe,e->jb", emission_features, theta[self.n + 3:])
        logits_f = rf[:, :, None] + shift[:, None, :]
        log_w = self.presence._evidence(qw[None], data["w"], data["mask"])[0]
        log_f = np.sum(f[:, None, None] * logits_f - np.logaddexp(0., logits_f), axis=0)
        joint = prior[:, None] + np.log(mass)[None, :] + log_w[:, None] + log_f
        joint[:, ~compatible] = -np.inf
        log_likelihood = logsumexp(joint)
        post = np.exp(joint - log_likelihood)
        ew = (data["w"][0, :, None] - expit(qw)) * post.sum(axis=1)[None, :]
        ef = (f[:, None, None] - expit(logits_f)) * post[None, :, :]
        d, k, boost = self.presence.d, self.presence.k, self.presence.boost_start
        grad = np.zeros_like(theta)
        grad[:d] = -np.einsum("js,jd->d", ew, data["dc"][0])
        grad[d:2 * d] = -np.einsum("jsb,jd->d", ef, data["dw"][0])
        state_posterior = post.sum(axis=1)
        grad[2 * d:2 * d + k] = -x * (state_posterior @ u - expit(lu))
        rv = state_posterior * (v - expit(lv))
        grad[2 * d + k:2 * d + 2 * k] = -x * rv.sum()
        grad[boost] = -np.sum(ew * u)
        grad[boost + 1] = -np.sum(ef * v[None, :, None])
        if self.presence.coupled:
            grad[self.n - 1] = -np.sum(rv * u)
        grad[self.n:self.n + 3] = -(post.sum(axis=0) @ score)
        grad[self.n + 3:] = -np.einsum("jsb,jbe->e", ef, emission_features)
        return float(-log_likelihood), grad

    def loss_gradient(self, theta, samples, final_labels, authority_labels, *, weights=None, violation_labels=None, at_least=1):
        if not samples or len(samples) != len(final_labels) or len(samples) != len(authority_labels):
            raise ValueError("aligned nonempty task supervision required")
        if type(at_least) is not int or at_least < 1:
            raise ValueError("positive integer system count threshold required")
        violations = [None] * len(samples) if violation_labels is None else list(violation_labels)
        if len(violations) != len(samples):
            raise ValueError("one optional system label per task required")
        w = np.ones(len(samples)) if weights is None else np.asarray(weights, float)
        if w.shape != (len(samples),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("positive finite task/frequency weights required")
        w = w / w.sum()
        theta = np.asarray(theta, float)
        loss, grad = 0., np.zeros_like(theta)
        for weight, sample, f, a, y in zip(w, samples, final_labels, authority_labels, violations):
            value, derivative = self._likelihood(sample, f, a, theta, y, at_least)
            loss += weight * value
            grad += weight * derivative
        return float(loss + self.regularization * (theta @ theta)), grad + 2 * self.regularization * theta

    def fit(self, samples, final_labels, authority_labels, *, weights=None, max_iter=500, trainable_names=None,
            violation_labels=None, at_least=1):
        selected = self.names if trainable_names is None else list(trainable_names)
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(self.names):
            raise ValueError("explicit unique trainable parameter names required")
        positions = np.array([self.names.index(name) for name in selected])
        initial = self.theta.copy()
        bounds = [(-12., 12.)] * len(initial)
        bounds[self.presence.boost_start:self.presence.boost_start + 2] = [(0., 12.)] * 2

        def objective(values):
            theta = initial.copy()
            theta[positions] = values
            loss, gradient = self.loss_gradient(theta, samples, final_labels, authority_labels, weights=weights,
                violation_labels=violation_labels, at_least=at_least)
            return loss, gradient[positions]

        fitted, info = _solve(objective, initial[positions], max_iter, bounds=[bounds[i] for i in positions])
        self.theta[positions] = fitted
        self.optimizer = {**info, "trainable_names": selected,
            "system_label_target": {"at_least": at_least} if violation_labels is not None else None,
            "scope": "Joint W/F/available-reference-label likelihood; fitting does not establish calibration or identifiability."}
        return self

    def parameters(self):
        return {"model": self.MODEL_NAME, "presence_definition": self.presence.parameters(),
            "names": self.names, "theta": self.theta.tolist(), "regularization": self.regularization,
            "optimizer": self.optimizer, "scope": "Fixed final-message literal universe, bound reference policy and static window only"}

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL_NAME:
            raise ValueError("explicit policy-gated model required")
        model = cls(ModelIndependentPrivacyBN.restore(saved["presence_definition"]), regularization=saved["regularization"])
        if model.names != saved["names"]:
            raise ValueError("saved parameter layout differs")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite compatible saved coefficients required")
        model.optimizer = saved["optimizer"]
        return model
