"""Learning from later-observed intermediate events without prefix leakage.

This is event-likelihood supervision. It does not pretend that the separately
stored later text was available at prediction time, or fit authority priors.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .observable_event_dag import ObservableEventDAG, DAGWorkflowRiskBN


def validated_labels(dag, labels):
    dag.validate()
    if not isinstance(labels, dict) or not labels or not set(labels) <= set(dag.ids) - set(dag.observations) - set(dag.missing_past):
        raise ValueError("separate labels for named future events required")
    spec = dag.specification()
    last = dag.prefix_sequence
    for identity, observation in labels.items():
        if not isinstance(observation, dict) or type(observation.get("sequence")) is not int or observation["sequence"] <= dag.prefix_sequence:
            raise ValueError("future supervision must have a later evidence sequence")
        last = max(last, observation["sequence"])
        spec["observations"][identity] = copy.deepcopy(observation)
    spec["prefix_sequence"] = last
    # An unlabeled predecessor need not be observed. For label validation only,
    # mark such predecessors as past-missing rather than filling with zeros.
    past = set(spec["observations"]) | set(spec.get("missing_past", []))
    changed = True
    while changed:
        changed = False
        for edge in spec["edges"]:
            if edge["target"] in past and edge["source"] not in past:
                past.add(edge["source"]); changed = True
    missing = [n for n in dag.order if n in past and n not in spec["observations"]]
    if missing:
        spec["missing_past"] = missing
    ObservableEventDAG(**spec)  # validates literal semantics and known chronology
    return tuple(n for n in dag.order if n in labels)


def trace_loss_gradient(law, theta, dags, labels, regularization):
    if not dags or len(dags) != len(labels) or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("aligned nonempty traces and nonnegative finite penalty required")
    theta = np.asarray(theta, float)
    total, gradient = 0., np.zeros_like(theta)
    for dag, target in zip(dags, labels):
        identities = validated_labels(dag, target)
        if hasattr(law, "label_log_likelihood"):
            value = law.label_log_likelihood(dag, {n: target[n]["value"] for n in identities}, theta)
            total -= value["log_probability"]
            gradient -= value["score"]
            continue
        joint = law.conditional(dag, identities, theta)
        bits = [target[n]["value"] for n in identities]
        index = sum(bit * 2 ** (len(bits) - i - 1) for i, bit in enumerate(bits))
        total -= joint["log_probabilities"][index]
        gradient -= joint["scores"][index]
    return float(total / len(dags) + regularization * (theta @ theta)), gradient / len(dags) + 2 * regularization * theta


def complete_trace_design(law, dags, labels):
    """Exact local factorization when EVERY unobserved event is labelled.

    Requires an ancestor-complete prefix. Partial labels or missing historical
    content require marginal inference and are not silently imputed here.
    """
    if not getattr(law, "factorizes_into_independent_events", True):
        raise ValueError("joint message factors require their message likelihood, not independent event loss")
    if not dags or len(dags) != len(labels):
        raise ValueError("aligned complete traces required")
    x, y = [], []
    for dag, outcomes in zip(dags, labels):
        identities = validated_labels(dag, outcomes)
        if dag.missing_past or set(identities) != set(dag.ids)-set(dag.observations):
            raise ValueError("all unobserved events need actual labels for the factorized objective")
        if any(not set(dag.parents[n]) <= set(dag.observations) for n in dag.observations):
            raise ValueError("an ancestor-complete prefix is required")
        values = {n: obs["value"] for n, obs in {**dag.observations, **outcomes}.items()}
        for n in identities:
            x.append(law.node_design(dag, n, [values[p] for p in dag.parents[n]])); y.append(values[n])
    return np.asarray(x), np.asarray(y)


def complete_trace_objective(theta, x, y, trace_count, regularization):
    logits = x @ theta
    return (float(np.sum(np.logaddexp(0., logits)-y*logits) / trace_count + regularization*(theta@theta)),
            x.T @ (expit(logits)-y) / trace_count + 2*regularization*theta)


def fit_event_traces(dags, labels, *, regularization=.001, max_iter=500, model_class=DAGWorkflowRiskBN):
    """One joint likelihood per workflow; do not count its prefixes twice."""
    model = model_class(regularization=regularization)
    if not getattr(model.event_law, "factorizes_into_independent_events", True):
        raise ValueError("use the joint message trainer for message-factor models")
    before = [d.sha256 for d in dags]
    complete = bool(dags) and len(dags) == len(labels) and all(not d.missing_past and set(label) == set(d.ids)-set(d.observations)
        and all(set(d.parents[n]) <= set(d.observations) for n in d.observations) for d, label in zip(dags, labels))
    if complete:
        x, y = complete_trace_design(model.event_law, dags, labels)
        objective = lambda theta: complete_trace_objective(theta, x, y, len(dags), regularization)
    else:
        objective = lambda theta: trace_loss_gradient(model.event_law, theta, dags, labels, regularization)
    result = minimize(objective,
        np.zeros(model.event_count), method="L-BFGS-B", jac=True,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("nonfinite trace fit")
    model.theta[:model.event_count] = result.x
    model.optimizer = {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
        "loss": float(result.fun), "max_gradient": float(np.max(np.abs(result.jac))),
        "supervision": "joint future observable-event likelihood per workflow; intermediate labels remain separate from prefixes",
        "fitted_parameter_names": list(model.event_law.names), "authority_priors_fitted": False,
        "trace_count": len(dags), "label_count": sum(len(row) for row in labels),
        "objective_evaluation": "exact complete-observation local factorization" if complete else "marginalized conditional likelihood"}
    if complete and hasattr(model.event_law, "feature_coverage"):
        model.optimizer["feature_coverage"] = model.event_law.feature_coverage(x)
    if before != [d.sha256 for d in dags]:
        raise ValueError("training changed a prefix")
    for dag in dags:
        dag.validate()
    return model
