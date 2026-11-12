"""Shared-authority fragments inside the existing system-state BN.

Enumerate shared authority facts; condition each existing process/history filter
on the SAME assignment for all actions; marginalize the worlds afterward. This
preserves root/action dependence rather than substituting action marginals.
Only the task head is used here; old tool-tier heads are deliberately excluded.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize
from .capability_action_bn import CAP_HEAD

from .delegation_conditioned_bn import DelegationConditionedBN
from .knowledge_scoped_bn import KnowledgeFilter
from .shared_authority_graph import PRIOR_NAMES, SharedAuthorityTrace
from .typed_action_supervision import bce


class _ConditionalWorldFilter(KnowledgeFilter):
    def __init__(self, theta, *, action_failures, shared_head, **options):
        self.action_failures, self.shared_head = action_failures, np.asarray(shared_head)
        self.seen_dispatches = []
        super().__init__(theta, **options)

    def apply(self, step, variables):
        self.seen_dispatches.extend(d.action_id for d in step.base.dispatches)
        n = len(self.seen_dispatches)
        failed = sum(self.action_failures[a] for a in self.seen_dispatches)
        self.joint_features = np.array([failed / max(1, n), failed * (failed - 1) / max(1, n * (n - 1))])
        # Conditional on a shared-fact world these are fixed named covariates,
        # not marginal failure probabilities multiplied as if independent.
        previous_bias = self.theta[CAP_HEAD]
        self.theta[CAP_HEAD] += self.shared_head @ self.joint_features
        try:
            value = super().apply(step, variables)
            self.shared_head_gradient = self.gradient[CAP_HEAD] * self.joint_features if self.derivatives else None
            return value
        finally:
            self.theta[CAP_HEAD] = previous_bias

    def _factor(self, factor, bits):
        if factor.template_id != "authorization":
            return super()._factor(factor, bits)
        if factor.action_id not in self.action_failures:
            raise ValueError("every authorization factor needs its scoped proof")
        q = float(self.action_failures[factor.action_id])
        histories = self.history_tuples[:, factor.channel]
        old = np.arange(self.history_width)
        sources, targets = [], []
        for process in (0, 1):
            rows = np.flatnonzero(bits[:, factor.node] == process)
            active = histories < (2 if process else 1)
            target = old[active] + ((2 if process else 1) - histories[active]) * self.history_powers[factor.channel]
            sources.append((rows[:, None] * self.history_width + old[active]).ravel())
            targets.append((rows[:, None] * self.history_width + target).ravel())
        self._move(np.concatenate(sources), np.concatenate(targets), q,
            np.zeros(len(self.theta)) if self.derivatives else None)
        self.observed_channels.add(factor.channel)
        self.factor_records.setdefault(factor.action_id, []).append({"template": factor.template_id,
            "world_conditioned_reference_failure": q, "meaning": "Boolean proof value in one shared world, not a calibrated prediction."})


class SharedAuthorityJointBN:
    def __init__(self, *, regularization=.002, max_iter=150, max_joint_states=65536):
        parent = DelegationConditionedBN()
        self.registry, self.base_count = parent.registry, parent.base_count
        self.prior_slice = slice(self.base_count, self.base_count + len(PRIOR_NAMES))
        self.parameter_names = (*parent.parameter_names[:self.base_count], *("shared_prior:" + n for n in PRIOR_NAMES),
            "shared_head:failed_action_fraction", "shared_head:failed_action_pair_fraction")
        self.initial = np.r_[parent.theta[:self.base_count], [-2., -3., 0.], [0., 0.]]
        self.theta = self.initial.copy()
        self.regularization, self.max_iter, self.max_joint_states = regularization, max_iter, max_joint_states
        self.optimizer, self.fitted_targets = {}, []

    def import_system_parameters(self, parent):
        if not isinstance(parent, DelegationConditionedBN) or parent.base_count != self.base_count:
            raise ValueError("explicit compatible parent model required")
        self.theta[:self.base_count] = parent.theta[:self.base_count]
        # A changed factorization is not a previously trained/calibrated model.
        self.initial = self.theta.copy()
        self.optimizer, self.fitted_targets = {}, []

    def forward(self, bundle, theta=None, *, derivatives=False):
        if not isinstance(bundle, SharedAuthorityTrace) or not bundle.graph.proofs:
            raise ValueError("shared compiled trace with at least one observed action required")
        theta = self.theta if theta is None else np.asarray(theta, dtype=float)
        if theta.shape != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError("finite shared-system parameter vector required")
        trace, graph = bundle.trace, bundle.graph
        dispatches = {d.action_id for s in trace.steps for d in s.base.dispatches}
        if dispatches != set(graph.proofs):
            raise ValueError("proofs must cover exactly the dispatched actions")
        for step in trace.steps:
            for factor in step.mechanism_factors:
                if factor.template_id == "authorization":
                    proof = graph.proofs[factor.action_id]
                    if proof.sequence != step.sequence or proof.evidence != factor.evidence.get("delegation_relations"):
                        raise ValueError("shared proof differs from this factor's prefix evidence")
            for action in step.base.actions:
                if action["action_id"] in graph.proofs:
                    from paperalpha_runtime.semantic_evidence import digest
                    proof = graph.proofs[action["action_id"]]
                    if proof.binding_sha256 != digest(action["binding"]):
                        raise ValueError("proof and model action binding differ")
        ids, states, mass, score, failures = graph.worlds(theta[self.prior_slice])
        joint_size = len(states) * (1 << len(trace.variables)) * 3 ** len(self.registry.channels)
        if joint_size > self.max_joint_states:
            raise ValueError("shared joint-state capacity exceeded; no silent marginal approximation")
        values = np.zeros((len(states), len(trace.steps)))
        gradients = np.zeros((len(trace.steps), self.base_count)) if derivatives else None
        head_gradients = np.zeros((len(trace.steps), 2)) if derivatives else None
        for world in range(len(states)):
            engine = _ConditionalWorldFilter(theta[:self.base_count], registry=self.registry,
                action_failures={k: bool(v[world]) for k, v in failures.items()}, shared_head=theta[-2:], derivatives=derivatives)
            for t, step in enumerate(trace.steps):
                values[world, t] = engine.apply(step, trace.variables)
                if derivatives:
                    gradients[t] += mass[world] * engine.gradient
                    head_gradients[t] += mass[world] * engine.shared_head_gradient
        probabilities = mass @ values
        full_gradient = None
        if derivatives:
            prior_gradient = ((values - probabilities[None, :]) * mass[:, None]).T @ score
            full_gradient = np.column_stack((gradients, prior_gradient, head_gradients))
        return {"probabilities": probabilities, "gradient": full_gradient,
            "world_conditioned_probabilities": values, "world_masses": mass,
            "shared_fact_count": len(ids), "joint_state_bound": joint_size}

    def loss_gradient(self, theta, traces, targets, *, target="system_outcome", at_least=1, weights=None):
        if not traces or len(traces) != len(targets) or any(type(y) is not int or y not in (0, 1) for y in targets):
            raise ValueError("explicit binary endpoint targets required")
        if target not in {"system_outcome", "reference_failure_count"}:
            raise ValueError("target meaning must be explicit")
        weights = np.ones(len(traces)) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != (len(traces),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("positive finite observation weights required")
        weights = weights / weights.sum()
        loss, gradient = 0., np.zeros(len(theta))
        for trace, y, weight in zip(traces, targets, weights):
            if target == "system_outcome":
                output = self.forward(trace, theta, derivatives=True)
                p, dp = output["probabilities"][-1], output["gradient"][-1]
            else:
                p, short = trace.graph.query(theta[self.prior_slice], at_least=at_least)
                if (p == 0. and y == 1) or (p == 1. and y == 0):
                    raise ValueError("reference label contradicts deterministic evidence under the declared observation contract")
                dp = np.r_[np.zeros(self.base_count), short, [0., 0.]]
            value, derivative = bce(p, y)
            loss += value * weight
            gradient += derivative * dp * weight
        centered = theta - self.initial
        return float(loss + self.regularization * (centered @ centered)), gradient + 2 * self.regularization * centered

    def fit(self, traces, targets, *, target="system_outcome", at_least=1, weights=None):
        fn = lambda theta: self.loss_gradient(theta, traces, targets, target=target, at_least=at_least, weights=weights)
        initial = fn(self.theta)[0]
        bounds = [(-8., 8.)] * (len(self.theta) - 2) + [(0., 8.)] * 2
        fitted = minimize(fn, self.theta, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6})
        self.theta = fitted.x
        projected = fitted.jac.copy()
        for i, (lo, hi) in enumerate(bounds):
            if (fitted.x[i] <= lo + 1e-9 and projected[i] > 0) or (fitted.x[i] >= hi - 1e-9 and projected[i] < 0):
                projected[i] = 0.
        self.optimizer = {"success": bool(fitted.success), "iterations": int(fitted.nit),
            "initial_loss": initial, "loss": float(fitted.fun), "message": str(fitted.message),
            "projected_gradient_max": float(np.max(np.abs(projected)))}
        self.fitted_targets = [target + (":" + str(at_least) if target == "reference_failure_count" else "")]
        return self

    def parameters(self):
        return {"model": "SharedAuthorityJointBN-research", "parameter_names": list(self.parameter_names),
            "theta": self.theta.tolist(), "registry_sha256": self.registry.sha256,
            "optimizer": copy.deepcopy(self.optimizer), "fitted_targets": list(self.fitted_targets),
            "scope": "Shared authority fragments conditional on an explicit static uncertainty window; not complete dynamic authorization or verified safety.",
            "factor_change": "Resolved authorization factors replaced by world-conditioned AND/OR proofs; old local authorization/duty coefficients no longer affect that channel.",
            "joint_head": "Named failure fraction and failed-pair fraction are computed in each shared world, not from products of marginal probabilities; new head coefficients are nonnegative during fitting.",
            "calibrated": False}


class SharedAuthorityStream:
    """Prefix-only adapter; currently replays each prefix, not cached inference.

No old scores are edited. Untrained diagnostics require an explicit opt-in;
fitting only the reference-count query does not qualify the system outcome head.
"""
    def __init__(self, model, compiler, *, allow_untrained_diagnostic=False):
        if "system_outcome" not in model.fitted_targets and not allow_untrained_diagnostic:
            raise ValueError("system output head requires its own supervision")
        self.model, self.compiler = model, compiler

    def push(self, record):
        step = self.compiler.push(record)
        result = {"sequence": record["sequence"], "dispatches": [], "calibrated": False,
            "proves_safety": False, "inference": "Exact prefix replay under a static-uncertainty-window contract"}
        if step is None or not step.base.dispatches:
            return result
        bundle = self.compiler.trace()
        output = self.model.forward(bundle)
        priors = self.model.theta[self.model.prior_slice]
        from .authority_qualification import qualify
        for d in step.base.dispatches:
            evidence = bundle.graph.proofs[d.action_id].evidence
            result["dispatches"].append({"action_id": d.action_id,
                **qualify(evidence, output["probabilities"][-1]),
                "reference_policy_any_failure": bundle.graph.query(priors, at_least=1)[0],
                "reference_policy_two_or_more_failures": bundle.graph.query(priors, at_least=2)[0],
                "reference_query_scope": "All actions observed in this window so far, not this action alone.",
                "shared_fact_count": output["shared_fact_count"], "joint_state_bound": output["joint_state_bound"],
                "fitted_targets": list(self.model.fitted_targets)})
        return result
