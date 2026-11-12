"""Descriptive support checks for grouped real-message development data.

No fitting, significance tests or safety claims. Repeated continuations are
averaged inside upstream runs; tasks/layouts are never pooled for covariance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from paperalpha_runtime.canonical import digest
from .grouped_event_learning import GroupedEventTrace, grouped_complete_design
from .observable_event_dag import ObservableEventLaw

PUBLISHERS = ("publish-a", "publish-b")


def pair_moments(pairs_by_group):
    """Equal-upstream-run moments; null means the quantity is not estimable.

    The cross-repeat contrast uses distinct continuations within a run and
    distinct runs between messages. It removes the same-reply product, but is
    not a causal effect or an independence test. Its expectation is a covariance
    of message-conditional response probabilities only under the stated sampling
    assumptions. Selection on completed pairs can violate those assumptions.
    """
    arrays = [np.asarray(p, dtype=float) for p in pairs_by_group]
    if any(p.ndim != 2 or p.shape[1] != 2 or not len(p)
           or not np.isin(p, (0., 1.)).all() for p in arrays):
        raise ValueError("nonempty two-binary-label continuation arrays required")
    if not arrays:
        return {"groups": 0, "continuation_pairs": 0, "joint_frequencies": None,
                "descriptive_covariance": None, "within_group_covariance": None,
                "between_group_covariance": None, "cross_repeat_contrast": None,
                "cross_repeat_eligible_groups": 0}
    means = np.asarray([p.mean(axis=0) for p in arrays])
    products = np.asarray([np.mean(p[:, 0] * p[:, 1]) for p in arrays])
    mean_a, mean_b = means.mean(axis=0)
    total = float(products.mean() - mean_a * mean_b)
    within = float(np.mean(products - means[:, 0] * means[:, 1]))
    between = float(np.mean(means[:, 0] * means[:, 1]) - mean_a * mean_b)
    frequencies = [float(np.mean([np.mean((p[:, 0] == a) & (p[:, 1] == b))
                                  for p in arrays])) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
    repeated = [p for p in arrays if len(p) >= 2]
    contrast = None
    if len(repeated) >= 2:
        repeated_means = np.asarray([p.mean(axis=0) for p in repeated])
        same_message_cross_repeat = float(np.mean([
            (p[:, 0].sum() * p[:, 1].sum() - (p[:, 0] * p[:, 1]).sum()) / (len(p) * (len(p) - 1))
            for p in repeated]))
        a, b = repeated_means.T
        n = len(repeated)
        different_message = float((a.sum() * b.sum() - (a * b).sum()) / (n * (n - 1)))
        contrast = {"same_message_different_continuations": same_message_cross_repeat,
                    "different_upstream_groups": different_message,
                    "difference": same_message_cross_repeat - different_message,
                    "same_continuation_minus_cross_repeat": float(np.mean([
                        np.mean(p[:, 0] * p[:, 1]) for p in repeated])) - same_message_cross_repeat}
    return {"groups": len(arrays), "continuation_pairs": sum(map(len, arrays)),
            "joint_state_order": ["00", "01", "10", "11"], "joint_frequencies": frequencies,
            "marginal_a": float(mean_a), "marginal_b": float(mean_b),
            "any_appearance": float(1 - frequencies[0]), "both_appearance": frequencies[3],
            "descriptive_covariance": total, "within_group_covariance": within,
            "between_group_covariance": between, "decomposition_error": float(total - within - between),
            "groups_with_within_message_variation": {
                node: sum(len(np.unique(p[:, i])) > 1 for p in arrays) for i, node in enumerate(PUBLISHERS)},
            "cross_repeat_contrast": contrast, "cross_repeat_eligible_groups": len(repeated)}


def _pairs(group):
    return [[continuation[n]["value"] for n in PUBLISHERS] for continuation in group["continuations"]]


def diagnose_verified_collection(verified):
    """Input must come from read-only raw-journal verification, not a score file."""
    if verified.get("schema") != "verified-repeated-message-development-v1":
        raise ValueError("verified repeated-message development data required")
    groups = verified["grouped_traces"]
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("duplicate upstream group")
    traces, strata = [], defaultdict(list)
    for group in groups:
        eligible = group["upstream_complete"] and bool(group["continuations"]) and group["response_identity_qualified"]
        if group["eligible_for_grouped_fit"] != eligible or group["eligible_for_statistics"] != eligible:
            raise ValueError("eligibility differs from actual completed calls")
        if group["eligible_for_grouped_fit"]:
            traces.append(GroupedEventTrace(group["dag"], group["shared_labels"], group["continuations"]))
        strata[(group["base_task"], group["layout"])].append(group)
    support = None
    if traces:
        law = ObservableEventLaw()
        x, y, audit = grouped_complete_design(law, traces)
        families = defaultdict(list)
        for row, value in zip(audit["rows"], y):
            families[traces[row["group"]].dag.lookup[row["event"]]["family"]].append(int(value))
        support = {"design_columns": len(law.names), "design_rank": int(np.linalg.matrix_rank(x)),
                   "training_event_rows": len(y), "unique_training_responses": audit["unique_response_records"],
                   "eligible_upstream_groups": len(traces),
                   "future_labels_by_family": {family: {"total": len(values), "positive": sum(values),
                                                        "negative": len(values) - sum(values)}
                                               for family, values in sorted(families.items())},
                   "nonzero_design_columns": [name for i, name in enumerate(law.names) if np.any(x[:, i] != 0)]}
    summaries = []
    for (task, layout), rows in sorted(strata.items()):
        usable = [g for g in rows if g["eligible_for_grouped_fit"]]
        by_state = defaultdict(list)
        for group in usable:
            state = tuple((n, label["value"]) for n, label in sorted(group["shared_labels"].items()))
            by_state[state].append(group)
        states = []
        for state, subset in sorted(by_state.items()):
            states.append({"reviewer_state": dict(state),
                           "distinct_review_message_vectors": len({digest({n: label["text"] for n, label in sorted(g["shared_labels"].items())}) for g in subset}),
                           **pair_moments([_pairs(g) for g in subset])})
        reviewer_counts = Counter((n, label["value"]) for g in rows for n, label in g["shared_labels"].items())
        summaries.append({"base_task": task, "layout": layout, "planned_groups": len(rows),
                          "complete_groups": sum(g["complete"] for g in rows),
                          "upstream_complete_groups": sum(g["upstream_complete"] for g in rows),
                          "eligible_groups": len(usable), "failure_records": sum(len(g["failures"]) for g in rows),
                          "response_identity_excluded_groups": [g["id"] for g in rows if not g["response_identity_qualified"]],
                          "partial_continuations": sum(len(g["incomplete_continuations"]) for g in rows),
                          "raw_recorded_reviewer_label_counts": [{"node": n, "value": v, "count": count}
                                                    for (n, v), count in sorted(reviewer_counts.items())],
                          "all_reviewer_states": pair_moments([_pairs(g) for g in usable]),
                          "conditioned_on_reviewer_state": states})
    return {"schema": "repeated-message-support-diagnostic-v1", "groups": len(groups),
            "base_tasks": len({g["base_task"] for g in groups}), "strata": summaries,
            "event_learning_support": support, "fits": 0, "model_performance_evaluated": False,
            "response_identity_exclusions": [{"id": g["id"], "issues": g["response_identity_issues"]}
                                             for g in groups if not g["response_identity_qualified"]],
            "limits": [
                "Development data only. There is no independent test or prospective model score here.",
                "Each upstream group has equal descriptive weight; continuations are not independent upstream runs.",
                "Within/between covariance is an algebraic sample decomposition, not identified latent variance.",
                "The cross-repeat contrast requires at least two completed continuations in at least two groups.",
                "A probability-covariance interpretation assumes identically sampled upstream groups and conditionally independent continuations given exact messages and task; neither is established by record IDs.",
                "Residual association within a reviewer-bit stratum may reflect omitted wording, runtime drift, finite-sample noise or completion selection; it is not wording causality.",
                "No significance, calibration or universal minimum-sample claim is made; zero observed variation is not proof of independence.",
                "Tasks, reviewer instructions and relevance differ together; do not pool tasks to infer the effect of a reviewer policy.",
                "Only complete continuation pairs enter moments and training support. Retained partial/failure counts disclose, but do not repair, completion-selection bias."]}
