"""Evidence for intermediate event meanings, separate from endpoint accuracy.

Node names are not empirical validation. Score each labelled intermediate
event against its later actual observation, using only the earlier prefix.
"""
from __future__ import annotations

import math
import numpy as np

from .observable_event_learning import validated_labels


def intermediate_semantics(model, dags, labels, terminal_ids):
    if not dags or len(dags) != len(labels) or len(dags) != len(terminal_ids):
        raise ValueError("aligned traces, future labels and terminal names required")
    rows = []
    for index, (dag, observed, terminals) in enumerate(zip(dags, labels, terminal_ids)):
        validated_labels(dag, observed)
        future = set(dag.ids) - set(dag.observations) - set(dag.missing_past)
        if not set(terminals) <= future:
            raise ValueError("terminal targets must be actual future events")
        for identity in dag.order:
            if identity not in future or identity in terminals:
                continue
            probability = float(np.exp(model.event_law.conditional(dag, [identity], model.theta[:model.event_count])["log_probabilities"][1]))
            rows.append({"trace_index": index, "event_id": identity, "family": dag.lookup[identity]["family"],
                "probability_one": probability, "label": observed[identity]["value"] if identity in observed else None})
    by_family = {}
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        known = [r for r in group if r["label"] is not None]
        n = len(known)
        by_family[family] = {"future_events": len(group), "labelled_events": n,
            "positives": sum(r["label"] for r in known),
            "brier": sum((r["probability_one"]-r["label"])**2 for r in known) / n if n else None,
            "accuracy_at_half": sum((r["probability_one"] >= .5) == bool(r["label"]) for r in known) / n if n else None,
            "log_loss_clipped_1e_12": -sum(math.log(max(r["probability_one"] if r["label"] else 1-r["probability_one"], 1e-12)) for r in known) / n if n else None}
    return {"per_event": rows, "by_family": by_family,
        "meaning": "Observed-label alignment diagnostic, not causal identification, calibration certification, or a held-out claim by itself."}


def binary_target_coverage(values):
    if not isinstance(values, (list, tuple)) or not values or any(type(y) is not int or y not in (0, 1) for y in values):
        raise ValueError("nonempty actual binary target labels required")
    positives, negatives = sum(values), len(values) - sum(values)
    return {"n": len(values), "positives": positives, "negatives": negatives,
        "positive_class_observed": positives > 0, "negative_class_observed": negatives > 0,
        "recall_estimable": positives > 0, "false_positive_rate_estimable": negatives > 0,
        "class_coverage_for_discrimination": positives > 0 and negatives > 0,
        "scope": "Class presence only. Small nonzero counts do not establish precision, statistical power or deployment representativeness."}
