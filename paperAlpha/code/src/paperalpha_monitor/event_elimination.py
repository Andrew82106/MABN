"""Exact log-domain variable elimination with conditional-score derivatives.

This replaces exhaustive intermediate-event enumeration, NOT the probability
model. Complexity is bounded by intermediate factor width and retained output
width. No threshold pruning, sampling or independent-marginal fallback.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.special import expit, logsumexp

from paperalpha_runtime.canonical import digest
from .observable_event_dag import ObservableEventLaw, DAGWorkflowRiskBN, DAGWorkflowInput, restore_event_law
from .workflow_risk_bn import WorkflowFragmentRegistry


@dataclass
class LogFactor:
    scope: tuple
    log_mass: np.ndarray
    score: np.ndarray


class EliminatingEventLaw(ObservableEventLaw):
    def __init__(self, specification=None, *, max_factor_variables=16):
        super().__init__(specification)
        if type(max_factor_variables) is not int or not 1 <= max_factor_variables <= 18:
            raise ValueError("one to 18 variables per materialized factor required")
        self.max_factor_variables = max_factor_variables

    def _check(self, dag, theta):
        dag.validate()
        if digest(self.specification) != self.sha256:
            raise ValueError("event law changed")
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.names),) or not np.isfinite(theta).all():
            raise ValueError("finite compatible event coefficients required")
        return theta

    def _capacity(self, scope):
        if len(scope) > self.max_factor_variables:
            raise ValueError("exact intermediate factor width exceeded; no independent fallback")

    def _node_factor(self, dag, identity, theta, observed):
        full = set(dag.parents[identity]) | {identity}
        scope = tuple(n for n in dag.order if n in full and n not in observed)
        self._capacity(scope)
        states = np.asarray(list(product((0., 1.), repeat=len(scope)))).reshape(-1, len(scope)) if scope else np.empty((1, 0))
        values = {n: np.full(len(states), observed[n], float) if n in observed else states[:, scope.index(n)] for n in full}
        parents = np.asarray([values[p] for p in dag.parents[identity]]).T if dag.parents[identity] else np.empty((len(states), 0))
        design = np.asarray([self.node_design(dag, identity, row) for row in parents])
        logits, y = design @ theta, values[identity]
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite conditional logits")
        shape = (2,) * len(scope)
        return LogFactor(scope, (y * logits - np.logaddexp(0., logits)).reshape(shape),
            ((y - expit(logits))[:, None] * design).reshape((*shape, len(theta))))

    def _multiply(self, factors, order):
        scope = tuple(n for n in order if any(n in f.scope for f in factors))
        self._capacity(scope)
        log_mass = np.zeros((2,) * len(scope))
        score = np.zeros((*log_mass.shape, len(self.names)))
        for factor in factors:
            # Every scope is in the same DAG order, so broadcasting cannot
            # silently permute a factor's variable axes.
            assert tuple(n for n in scope if n in factor.scope) == factor.scope
            shape = tuple(2 if n in factor.scope else 1 for n in scope)
            log_mass += factor.log_mass.reshape(shape)
            score += factor.score.reshape((*shape, len(self.names)))
        return LogFactor(scope, log_mass, score)

    @staticmethod
    def _sum_out(factor, variable):
        axis = factor.scope.index(variable)
        z = logsumexp(factor.log_mass, axis=axis)
        weights = np.exp(factor.log_mass - np.expand_dims(z, axis))
        score = np.sum(weights[..., None] * factor.score, axis=axis)
        return LogFactor(tuple(n for n in factor.scope if n != variable), z, score)

    def _initial_factors(self, dag, theta, observed, retained):
        # Unobserved leaf descendants of the evidence/query integrate to one;
        # ancestral pruning is exact, even with missing historical variables.
        active = set(observed) | set(retained)
        pending = list(active)
        while pending:
            for parent in dag.parents[pending.pop()]:
                if parent not in active:
                    active.add(parent); pending.append(parent)
        factors = [self._node_factor(dag, n, theta, observed) for n in dag.order if n in active]
        return active, factors

    def _partition(self, dag, theta, observed, retained=()):
        active, factors = self._initial_factors(dag, theta, observed, retained)
        peak = max((len(f.scope) for f in factors), default=0)
        remaining = active - set(observed) - set(retained)
        eliminated = []
        while remaining:
            adjacency = {tuple(sorted(pair)) for f in factors for pair in combinations(f.scope, 2)}
            def cost(variable):
                neighbors = set().union(*(set(f.scope) for f in factors if variable in f.scope)) - {variable}
                fill = sum(tuple(sorted(pair)) not in adjacency for pair in combinations(sorted(neighbors), 2))
                return fill, len(neighbors), dag.order.index(variable)
            variable = min(remaining, key=cost)
            bucket = [f for f in factors if variable in f.scope]
            factors = [f for f in factors if variable not in f.scope]
            joined = self._multiply(bucket, dag.order)
            peak = max(peak, len(joined.scope))
            factors.append(self._sum_out(joined, variable))
            remaining.remove(variable); eliminated.append(variable)
        final = self._multiply(factors, dag.order)
        peak = max(peak, len(final.scope))
        return final, {"algorithm": "exact_log_score_variable_elimination", "elimination_order": eliminated,
            "active_events": len(active), "pruned_events": len(dag.ids) - len(active),
            "peak_factor_variables": peak, "peak_factor_entries": 2 ** peak,
            "retained_joint_variables": len(final.scope), "total_unobserved_events": len(dag.ids) - len(dag.observations)}

    def conditional(self, dag, targets, theta, *, max_free=16):
        theta = self._check(dag, theta)
        future = set(dag.ids) - set(dag.observations) - set(dag.missing_past)
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= future:
            raise ValueError("distinct unobserved future targets required")
        if type(max_free) is not int or not 1 <= max_free <= 18 or len(targets) > max_free:
            raise ValueError("retained joint-output capacity exceeded")
        observed = {n: v["value"] for n, v in dag.observations.items()}
        factor, stats = self._partition(dag, theta, observed, targets)
        axes = [factor.scope.index(n) for n in targets]
        logs = np.transpose(factor.log_mass, axes).reshape(-1)
        scores = np.transpose(factor.score, [*axes, len(axes)]).reshape(-1, len(theta))
        log_evidence = float(logsumexp(logs))
        mass = np.exp(logs - log_evidence)
        return {"states": np.asarray(list(product((0., 1.), repeat=len(targets)))),
            "log_probabilities": logs - log_evidence, "scores": scores - mass @ scores,
            "log_evidence": log_evidence, "free_events": stats["total_unobserved_events"],
            "inference": stats}

    def label_log_likelihood(self, dag, labels, theta):
        """A scalar likelihood without enumerating all labelled joint states."""
        theta = self._check(dag, theta)
        future = set(dag.ids) - set(dag.observations) - set(dag.missing_past)
        if not isinstance(labels, dict) or not labels or not set(labels) <= future or any(type(v) is not int or v not in (0, 1) for v in labels.values()):
            raise ValueError("named future binary labels required")
        observed = {n: v["value"] for n, v in dag.observations.items()}
        denominator, before = self._partition(dag, theta, observed)
        numerator, after = self._partition(dag, theta, {**observed, **labels})
        assert not denominator.scope and not numerator.scope
        return {"log_probability": float(numerator.log_mass-denominator.log_mass),
            "score": numerator.score-denominator.score, "inference": {"prefix": before, "labelled": after}}


class EliminatingDAGWorkflowRiskBN(DAGWorkflowRiskBN):
    MODEL = "ObservableDAGWorkflowRiskBN-VE-v1"

    def __init__(self, *, event_law=None, max_factor_variables=16, **options):
        super().__init__(event_law=event_law or EliminatingEventLaw(max_factor_variables=max_factor_variables), **options)
        if not isinstance(self.event_law, EliminatingEventLaw):
            raise ValueError("eliminating event law required")

    def _occurrence_worlds(self, sample, theta):
        if not isinstance(sample, DAGWorkflowInput):
            raise ValueError("linked event-policy input required")
        if len(sample.action_ids) + len(sample.authority.facts) > self.max_binary_variables:
            raise ValueError("joint policy-output capacity exceeded")
        law = self.event_law.conditional(sample.event_dag, [sample.queries[a]["event_id"] for a in sample.action_ids],
            theta[:self.event_count], max_free=self.max_binary_variables)
        scores = np.zeros((len(law["states"]), len(self.names)))
        scores[:, :self.event_count] = law["scores"]
        return law["states"], law["log_probabilities"], scores

    def predict(self, sample, **kwargs):
        result = super().predict(sample, **kwargs)
        result.pop("enumerated_future_event_worlds", None)
        result.pop("enumerated_unobserved_event_worlds", None)
        result["inference"] = {"algorithm": "exact_variable_elimination", "max_factor_variables": self.event_law.max_factor_variables,
            "retained_policy_event_worlds": 2 ** len(sample.action_ids),
            "total_unobserved_events": len(sample.event_dag.ids) - len(sample.event_dag.observations),
            "approximation": False}
        result["limitations"][1] = "Up to 12 functional nodes/256 events; exact factor and retained-policy widths remain bounded. Dense graphs and many joint policy targets can still be intractable."
        return result

    def predict_known_authorization_union(self, sample, *, derivatives=False):
        """Exact selected-policy query without materializing all output states.

        This path is only valid for fully known authorization-only risk. Other
        policies/facts require the general joint composer, never a quiet fallback.
        """
        if not isinstance(sample, DAGWorkflowInput):
            raise ValueError("linked event-policy input required")
        sample.validate(); self.registry.validate()
        if sample.authority.facts or sample.contracts or sample.combinations or any(p["quantity"] is not None for p in sample.proposals):
            raise ValueError("authorization-union query requires known authority and no other risk policy")
        if "authorization" not in self.registry.templates:
            raise ValueError("authorization fragment is not enabled")
        _, _, mass, _, failed = sample.authority.worlds(self.theta[self._authority_indices()])
        if len(mass) != 1 or mass[0] != 1:
            raise ValueError("exact known authority required")
        targets = [sample.queries[a]["event_id"] for a in sample.action_ids if bool(failed[a][0])]
        if targets:
            result = self.event_law.label_log_likelihood(sample.event_dag, dict.fromkeys(targets, 0), self.theta[:self.event_count])
            if result["log_probability"] > 1e-10:
                raise ValueError("conditional event probability exceeds one")
            log_absent = min(0., result["log_probability"])
            all_absent = float(np.exp(log_absent))
            probability = float(-np.expm1(log_absent))
            grad = np.r_[-all_absent*result["score"], np.zeros(len(self.names)-self.event_count)]
            inference = result["inference"]
        else:
            probability, grad, inference = 0., np.zeros(len(self.names)), {"known_no_invalid_authorization": True}
        output = {"model": self.MODEL, "input_sha256": sample.sha256, "probabilities": {"system:any": probability},
            "query_scope": "fully-known authorization-only union", "invalid_authorization_events": targets,
            "inference": inference, "approximation": False}
        if derivatives:
            output["gradients"] = {"system:any": grad}
        return output

    def parameters(self):
        return {**super().parameters(), "max_factor_variables": self.event_law.max_factor_variables,
            "inference_algorithm": "exact_log_score_variable_elimination"}

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL or saved.get("inference_algorithm") != "exact_log_score_variable_elimination":
            raise ValueError("wrong eliminating model")
        registry = WorkflowFragmentRegistry(saved["registry"])
        law = restore_event_law(saved["event_law"], max_factor_variables=saved["max_factor_variables"])
        model = cls(event_law=law, registry=registry, regularization=saved["regularization"], max_binary_variables=saved["max_binary_variables"])
        if registry.sha256 != saved["registry_sha256"] or law.sha256 != saved["event_law_sha256"] or list(model.names) != saved["names"]:
            raise ValueError("model/template identity mismatch")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("compatible finite model required")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model
