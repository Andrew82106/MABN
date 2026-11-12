"""Observed-content learning from ancestor-complete partial grouped traces.

Successful content labels are retained without inventing outcomes for failed or
unattempted calls. This is an ignorable-missingness working likelihood, not the
likelihood of the complete API log. Retaining partial traces cannot, by itself,
remove selection bias or supply a model for attempt/completion probabilities.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize

from paperalpha_runtime.canonical import digest
from .grouped_event_learning import _records, _validate_record_bindings
from .observable_event_dag import ObservableEventDAG, ObservableEventLaw, DAGWorkflowRiskBN
from .observable_event_learning import validated_labels, complete_trace_objective


class PartialGroupedEventTrace:
    """A declared shared ancestor block and indexed continuation slots.

    ``shared_event_ids`` describes topology, not which replies succeeded.
    Labels may be absent, but every observed child's parents must be observed
    in the prefix, shared block, or that same continuation. At least one future
    continuation slot is declared; ``[{}, {}]`` preserves two slots with no
    observed content, unlike compressing them away. Failure/attempt statuses
    remain in the collector and never become binary content labels here.
    """

    def __init__(self, dag, shared_event_ids, shared_labels, continuations):
        if not isinstance(dag, ObservableEventDAG):
            raise ValueError("validated early event DAG required")
        if not isinstance(shared_event_ids, (list, tuple)):
            raise ValueError("explicit ordered shared event IDs required")
        self.dag = copy.deepcopy(dag)
        self.shared_event_ids = tuple(shared_event_ids)
        self.shared_labels = copy.deepcopy(shared_labels)
        self.continuations = copy.deepcopy(continuations)
        self._check()
        self.sha256 = digest(self.specification())

    def specification(self):
        return {"schema": "ancestor-complete-partial-grouped-event-trace-v1",
                "dag": self.dag.specification(), "shared_event_ids": list(self.shared_event_ids),
                "shared_labels": copy.deepcopy(self.shared_labels),
                "continuations": copy.deepcopy(self.continuations)}

    def _check(self):
        dag = self.dag
        dag.validate()
        observed = set(dag.observations)
        if dag.missing_past or any(not set(dag.parents[n]) <= observed for n in observed):
            raise ValueError("ancestor-complete early prefix without missing past required")
        ids = self.shared_event_ids
        if not ids or any(not isinstance(n, str) for n in ids) or len(set(ids)) != len(ids):
            raise ValueError("nonempty distinct structural shared event IDs required")
        shared = set(ids)
        future = set(dag.ids)-observed
        if not shared <= future:
            raise ValueError("shared block must contain only named future events")
        suffix = future-shared
        if not suffix or any(not set(dag.parents[n]) <= observed | shared for n in shared):
            raise ValueError("nonempty shared ancestor block and remaining suffix required")
        if not isinstance(self.shared_labels, dict) or not set(self.shared_labels) <= shared:
            raise ValueError("shared observations must belong to the declared shared block")
        if not isinstance(self.continuations, list) or not self.continuations:
            raise ValueError("nonempty list of predeclared continuation slots required; empty slots use {}")
        available_shared = observed | set(self.shared_labels)
        if any(not set(dag.parents[n]) <= available_shared for n in self.shared_labels):
            raise ValueError("observed shared events require all actual ancestors")
        if self.shared_labels:
            validated_labels(dag, self.shared_labels)
        seen = _records(self.shared_labels)
        end = max((v["sequence"] for v in self.shared_labels.values()), default=dag.prefix_sequence)
        for continuation in self.continuations:
            if not isinstance(continuation, dict) or not set(continuation) <= suffix:
                raise ValueError("each indexed continuation may label only its remaining future events")
            available = available_shared | set(continuation)
            if any(not set(dag.parents[n]) <= available for n in continuation):
                raise ValueError("observed continuation requires its own actual parents; no cross-continuation borrowing")
            if not continuation:
                continue
            validated_labels(dag, {**self.shared_labels, **continuation})
            if min(v["sequence"] for v in continuation.values()) <= end:
                raise ValueError("continuation records must follow earlier slots on one journal clock")
            end = max(v["sequence"] for v in continuation.values())
            records = _records(continuation)
            if records & seen:
                raise ValueError("reused response record is not another observed continuation")
            seen.update(records)
        _validate_record_bindings(dag, [dag.observations, self.shared_labels, *self.continuations])

    def validate(self):
        if digest(self.specification()) != self.sha256:
            raise ValueError("partial grouped observation evidence changed")
        self._check()


def _check_law(law):
    if not isinstance(law, ObservableEventLaw) or not getattr(law, "factorizes_into_independent_events", True):
        raise ValueError("supported independent logistic event law required; joint message factors need another likelihood")
    if digest(law.specification) != law.sha256:
        raise ValueError("event probability templates changed")


def _normalizer(groups, normalization_groups):
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("nonempty partial grouped traces required")
    value = len(groups) if normalization_groups is None else normalization_groups
    if type(value) is not int or value < len(groups):
        raise ValueError("normalization_groups must be an integer at least the number of supplied upstream groups")
    return value


def _check_penalty(regularization):
    if not np.isscalar(regularization) or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("finite nonnegative regularization required")


def partial_grouped_design(law, groups):
    """One factor per observed event, with no implicit absent-event labels."""
    _check_law(law)
    _normalizer(groups, None)
    x, y, rows, seen = [], [], [], set()
    shared_count = informative = nonempty_slots = fully_observed_slots = 0
    shared_slots = continuation_slots = 0
    for group_index, group in enumerate(groups):
        if not isinstance(group, PartialGroupedEventTrace):
            raise ValueError("typed partial grouped trace required")
        group.validate()
        dag = group.dag
        records = set().union(*(_records(block) for block in [group.shared_labels, *group.continuations]))
        if seen & records:
            raise ValueError("response records reused across training groups; use scoped IDs")
        seen.update(records)
        informative += bool(group.shared_labels or any(group.continuations))
        shared_slots += len(group.shared_event_ids)
        suffix = set(dag.ids)-set(dag.observations)-set(group.shared_event_ids)
        continuation_slots += len(suffix)*len(group.continuations)
        observed = {n: v["value"] for n, v in dag.observations.items()}
        shared_values = {n: v["value"] for n, v in group.shared_labels.items()}

        def append(identity, labels, values, continuation_index):
            design = np.asarray(law.node_design(dag, identity, [values[n] for n in dag.parents[identity]]), float)
            if design.shape != (len(law.names),) or not np.isfinite(design).all():
                raise ValueError("finite compatible local event features required")
            x.append(design)
            y.append(labels[identity]["value"])
            rows.append({"group": group_index, "continuation": continuation_index, "event": identity,
                         "record_id": labels[identity]["record_id"], "source": labels[identity]["source"]})

        for identity in dag.order:
            if identity in group.shared_labels:
                append(identity, group.shared_labels, {**observed, **shared_values}, None)
                shared_count += 1
        for index, continuation in enumerate(group.continuations):
            nonempty_slots += bool(continuation)
            fully_observed_slots += set(continuation) == suffix and set(group.shared_labels) == set(group.shared_event_ids)
            values = {**observed, **shared_values, **{n: v["value"] for n, v in continuation.items()}}
            for identity in dag.order:
                if identity in continuation:
                    append(identity, continuation, values, index)
    return np.asarray(x, float).reshape((-1, len(law.names))), np.asarray(y, float), {
        "groups": len(groups), "informative_groups": informative, "empty_groups": len(groups)-informative,
        "shared_event_labels": shared_count, "continuation_event_labels": len(y)-shared_count,
        "continuations": sum(len(g.continuations) for g in groups),
        "nonempty_continuations": nonempty_slots, "fully_observed_continuations": fully_observed_slots,
        "planned_shared_event_slots": shared_slots, "unobserved_shared_event_slots": shared_slots-shared_count,
        "planned_continuation_event_slots": continuation_slots,
        "unobserved_continuation_event_slots": continuation_slots-(len(y)-shared_count),
        "unique_response_records": len(seen), "rows": rows,
        "target": "ancestor-complete observed-content likelihood under ignorable missingness, not complete API-log likelihood",
        "normalization": "sum observed shared factors once and each actual continuation factor once; divide by explicit upstream-group normalizer, not labels or continuations",
        "unobserved_slot_meaning": "No binary label. Failure, refusal, incomplete and not-attempted statuses remain separate collector evidence."}


def partial_grouped_loss_gradient(law, theta, groups, regularization=0., normalization_groups=None):
    _check_law(law)
    normalizer = _normalizer(groups, normalization_groups)
    _check_penalty(regularization)
    theta = np.asarray(theta, float)
    if theta.shape != (len(law.names),) or not np.isfinite(theta).all():
        raise ValueError("finite compatible event coefficients required")
    x, y, _ = partial_grouped_design(law, groups)
    return complete_trace_objective(theta, x, y, normalizer, regularization)


def fit_partial_grouped_event_traces(groups, regularization=.001, max_iter=500,
                                   model_class=DAGWorkflowRiskBN, normalization_groups=None):
    normalizer = _normalizer(groups, normalization_groups)
    _check_penalty(regularization)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("positive integer max_iter required")
    model = model_class(regularization=regularization)
    x, y, audit = partial_grouped_design(model.event_law, groups)
    if len(y) == 0:
        raise ValueError("not informative: no observed content labels; an empty group is not a training example")
    result = minimize(lambda theta: complete_trace_objective(theta, x, y, normalizer, regularization),
                      np.zeros(model.event_count), method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("nonfinite partial grouped fit")
    model.theta[:model.event_count] = result.x
    model.optimizer = {"success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "loss": float(result.fun), "max_gradient": float(np.max(np.abs(result.jac))),
        "supervision": "observed shared content factors once plus every actually observed continuation content factor",
        "fitted_parameter_names": list(model.event_law.names), "authority_priors_fitted": False,
        "group_sha256": [g.sha256 for g in groups], "normalization_groups": normalizer,
        "training_audit": {**audit, "normalization_groups": normalizer},
        "assumptions": ["The missingness/selection mechanism is ignorable for the working content model; this is an assumption, not established by an API protocol.",
            "Given modeled parent states and prefix cues, event factors and repeated continuations obey the specified conditional independences.",
            "Literal states may omit wording that affects both response completion and content; retaining partial observations does not remove that selection bias."],
        "target": "Observed-content likelihood, not the full API-log likelihood or unconditional workflow risk.",
        "unimplemented": ["observed children with unobserved ancestors", "missing past content", "joint message-hurdle factors",
                          "attempt/completion/failure likelihood", "nonignorable missingness correction"],
        "calibrated": False}
    if hasattr(model.event_law, "feature_coverage"):
        model.optimizer["feature_coverage"] = model.event_law.feature_coverage(x)
    for group in groups:
        group.validate()
    return model
