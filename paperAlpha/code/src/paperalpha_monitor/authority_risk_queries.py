"""Target-scoped reference queries, separate from endpoint-trained risk heads.

Logical bounds describe compatible Boolean outcomes, not confidence intervals.
No point estimate is promoted to a future-harm or calibrated safety forecast.
"""
from __future__ import annotations

import math

import numpy as np

from .shared_authority_graph import AuthorityProofGraph


def query_reference_risk(graph, prior_logits, *, at_least=1):
    if not isinstance(graph, AuthorityProofGraph) or type(at_least) is not int or at_least < 1:
        raise ValueError("proof graph and positive integer threshold required")
    _, _, mass, _, failures = graph.worlds(prior_logits)
    counts = np.sum(list(failures.values()), axis=0) if failures else np.zeros(len(mass))
    event = (counts >= at_least).astype(float)
    lower, upper = int(event.min()), int(event.max())
    return {"target": {"event": "reference_policy_failure_count", "at_least": at_least,
            "scope": "action entries already observed in this authority window", "action_count": len(graph.proofs)},
        "model_probability": float(mass @ event),
        "logical_lower": lower, "logical_upper": upper,
        "bounds_meaning": "Minimum/maximum Boolean outcome over compatible fact states; NOT confidence or credible intervals.",
        "evidence_decides_target": lower == upper,
        "prior_dependent": lower != upper, "uncertain_fact_count": len(graph.facts),
        "future_forecast": False, "calibrated": False, "proves_system_safety": False,
        "scope_note": "Zero for fewer observed actions than threshold does not certify other risks or future actions."}


def qualify_two_action_endpoint(action_count, system_score, logistic_score, *, run_completed):
    """Both previous fitted heads share the same two-action endpoint domain.

This is an applicability gate, not a prediction correction or a fitted model.
Raw diagnostics remain available; no original acquisition score is rewritten.
"""
    if type(action_count) is not int or action_count < 0 or type(run_completed) is not bool:
        raise ValueError("explicit action count and completion status required")
    scores = {"system_bn": float(system_score), "same_information_logistic": float(logistic_score)}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        raise ValueError("finite probability diagnostics required")
    applicable = run_completed and action_count == 2
    return {"target": "At least two reference-policy failures at a completed two-action endpoint.",
        "applicable": applicable, "scores": scores if applicable else None,
        "inapplicable_reason": None if applicable else "Endpoint-trained heads are not qualified for this prefix/count or incomplete run.",
        "raw_diagnostic_scores": scores, "future_forecast": False, "calibrated": False,
        "selection_note": "Record every excluded run/prefix; do not relabel refusal as successful monitoring."}
