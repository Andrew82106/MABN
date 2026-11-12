"""Learn from one upstream run followed by repeated downstream continuations.

The shared message is one observation, not one new observation per continuation.
This is ordinary grouped joint likelihood, not a new latent-state model. Actual
message-content labels remain separate from the early forecast input.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize

from paperalpha_runtime.canonical import digest
from .observable_event_dag import ObservableEventDAG, DAGWorkflowRiskBN
from .observable_event_learning import validated_labels, complete_trace_objective


def _records(labels):
    # ``source`` describes evidence provenance; changing that string must not
    # manufacture another observation of the same collection-scoped record.
    return {v["record_id"] for v in labels.values()}


def _validate_record_bindings(dag, blocks):
    """Bind every observed message to one payload, journal tick and producer.

    One message may label several *different* literals for the same actor and
    workflow node. Those labels are not separate API responses. Reusing its ID
    for another producer or another copy of the same event is not permitted.
    Prefix observations participate so a later label cannot relabel past data.
    """
    records, ticks = {}, {}
    for block in blocks:
        for identity, observation in block.items():
            event = dag.lookup[identity]
            record_id, sequence = observation["record_id"], observation["sequence"]
            if sequence in ticks and ticks[sequence] != record_id:
                raise ValueError("different response records cannot share one journal sequence")
            ticks[sequence] = record_id
            payload = tuple(observation[k] for k in ("source", "text", "sequence", "kind"))
            producer = (event["node"], event["actor"])
            subject = (event["semantics"], event["subject"].casefold()
                       if event["semantics"] == "literal_appearance" else event["subject"])
            if record_id not in records:
                records[record_id] = (payload, producer, {subject})
                continue
            old_payload, old_producer, subjects = records[record_id]
            if payload != old_payload:
                raise ValueError("one response record has conflicting payload, source, kind or sequence")
            if producer != old_producer:
                raise ValueError("one response record cannot cross agents or workflow nodes")
            if subject in subjects:
                raise ValueError("the same response event is labelled more than once")
            if observation["kind"] != "complete_message" or subject[0] != "literal_appearance" or any(s[0] != "literal_appearance" for s in subjects):
                raise ValueError("only distinct literals in one actual message may share a response record")
            subjects.add(subject)


class GroupedEventTrace:
    """Complete shared future ancestors, followed by >=1 complete continuations.

    Sequence values are from one group-wide journal. Record IDs must include
    their collection/group scope, independent of the evidence source string.
    Distinct IDs are necessary, not proof that
    API samples are statistically independent; the collector must verify calls.
    """

    def __init__(self, dag, shared_labels, continuations):
        if not isinstance(dag, ObservableEventDAG):
            raise ValueError("validated early event DAG required")
        self.dag = copy.deepcopy(dag)
        self.shared_labels = copy.deepcopy(shared_labels)
        self.continuations = copy.deepcopy(continuations)
        self._check()
        self.sha256 = digest(self.specification())

    def specification(self):
        return {"schema": "grouped-event-trace-v1", "dag": self.dag.specification(),
                "shared_labels": copy.deepcopy(self.shared_labels),
                "continuations": copy.deepcopy(self.continuations)}

    def _check(self):
        dag = self.dag; dag.validate()
        if dag.missing_past or any(not set(dag.parents[n]) <= set(dag.observations) for n in dag.observations):
            raise ValueError("ancestor-complete early prefix required")
        shared = set(validated_labels(dag, self.shared_labels))
        future = set(dag.ids) - set(dag.observations)
        suffix = future - shared
        if not shared or not suffix or any(not set(dag.parents[n]) <= shared | set(dag.observations) for n in shared):
            raise ValueError("nonempty shared ancestor block and downstream suffix required")
        if not isinstance(self.continuations, list) or not self.continuations:
            raise ValueError("one or more completed downstream continuations required")
        seen = _records(self.shared_labels)
        end = max(v["sequence"] for v in self.shared_labels.values())
        for continuation in self.continuations:
            if not isinstance(continuation, dict) or set(continuation) != suffix:
                raise ValueError("each continuation needs every remaining future event")
            validated_labels(dag, {**self.shared_labels, **continuation})
            if min(v["sequence"] for v in continuation.values()) <= end:
                raise ValueError("continuations must follow shared records on one journal clock")
            end = max(v["sequence"] for v in continuation.values())
            tokens = _records(continuation)
            if tokens & seen:
                raise ValueError("reused response record is not a new continuation")
            seen.update(tokens)
        _validate_record_bindings(dag, [dag.observations, self.shared_labels, *self.continuations])

    def validate(self):
        if digest(self.specification()) != self.sha256:
            raise ValueError("grouped observation evidence changed")
        self._check()


def grouped_complete_design(law, groups):
    """Sum shared factors once and all distinct continuation factors once."""
    if not getattr(law, "factorizes_into_independent_events", True):
        raise ValueError("grouped message-factor likelihood is not implemented")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("nonempty grouped actual traces required")
    x, y, provenance, seen = [], [], [], set()
    shared_count = 0
    for group_index, group in enumerate(groups):
        if not isinstance(group, GroupedEventTrace):
            raise ValueError("typed grouped trace required")
        group.validate(); dag = group.dag
        records = _records(group.shared_labels)
        for c in group.continuations: records |= _records(c)
        if seen & records:
            raise ValueError("response records reused across training groups; use scoped IDs")
        seen |= records
        observed = {n: v["value"] for n, v in dag.observations.items()}
        shared_values = {n: v["value"] for n, v in group.shared_labels.items()}
        def append(identity, labels, values, continuation):
            x.append(law.node_design(dag, identity, [values[n] for n in dag.parents[identity]]))
            y.append(labels[identity]["value"])
            provenance.append({"group": group_index, "continuation": continuation, "event": identity,
                               "record_id": labels[identity]["record_id"], "source": labels[identity]["source"]})
        for identity in dag.order:
            if identity in group.shared_labels:
                append(identity, group.shared_labels, {**observed, **shared_values}, None)
                shared_count += 1
        for i, continuation in enumerate(group.continuations):
            values = {**observed, **shared_values, **{n: v["value"] for n, v in continuation.items()}}
            for identity in dag.order:
                if identity in continuation: append(identity, continuation, values, i)
    return np.asarray(x), np.asarray(y), {
        "groups": len(groups), "shared_event_labels": shared_count,
        "continuation_event_labels": len(y)-shared_count,
        "continuations": sum(len(g.continuations) for g in groups),
        "unique_response_records": len(seen), "rows": provenance,
        "normalization": "negative joint log likelihood averaged over upstream-run groups, not over flattened continuations"}


def grouped_loss_gradient(law, theta, groups, regularization=0.):
    theta = np.asarray(theta, float)
    if theta.shape != (len(law.names),) or not np.isfinite(theta).all():
        raise ValueError("finite compatible event coefficients required")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("finite nonnegative regularization required")
    x, y, _ = grouped_complete_design(law, groups)
    return complete_trace_objective(theta, x, y, len(groups), regularization)


def fit_grouped_event_traces(groups, *, regularization=.001, max_iter=500, model_class=DAGWorkflowRiskBN):
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("finite nonnegative regularization required")
    model = model_class(regularization=regularization)
    x, y, audit = grouped_complete_design(model.event_law, groups)
    result = minimize(lambda theta: complete_trace_objective(theta, x, y, len(groups), regularization),
                      np.zeros(model.event_count), method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("nonfinite grouped fit")
    model.theta[:model.event_count] = result.x
    model.optimizer = {"success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "loss": float(result.fun),
        "max_gradient": float(np.max(np.abs(result.jac))),
        "supervision": "one shared-message likelihood plus every actual continuation likelihood per group",
        "authority_priors_fitted": False, "group_sha256": [g.sha256 for g in groups],
        "training_audit": audit,
        "assumptions": ["Different continuations are conditionally independent given the modeled parent states and prefix.",
                        "A literal-presence bit may omit relevant shared wording; that sufficiency assumption requires data checks.",
                        "Grouped likelihood prevents duplicate upstream counting, not uncertainty under arbitrary group dependence."],
        "unimplemented": ["partial grouped labels", "missing past messages", "joint message-hurdle factors"],
        "calibrated": False}
    if hasattr(model.event_law, "feature_coverage"):
        model.optimizer["feature_coverage"] = model.event_law.feature_coverage(x)
    for group in groups: group.validate()
    return model
