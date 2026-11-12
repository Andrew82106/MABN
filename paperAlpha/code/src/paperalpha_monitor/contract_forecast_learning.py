"""Trace-grouped conditional composite scoring of separate next-call windows.

A trace contributes the SUM of its window conditional negative log scores, and
traces receive equal weight. A is re-estimated within each window; supervision
may refer repeatedly to the same underlying state. This is a conditional
prediction/composite objective, NOT a generative joint sequence likelihood.
No state transitions or cross-window latent-state joint are implemented.
Duplicate forecasts of the same next invocation must not be treated as new
observations. The caller owns the window schedule and family-held-out splitting.
None labels are marginalized, never converted to observed zero. No policy/risk
head is fitted, and no optimizer summary retains training cases or identities.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize

from .contract_forecast import ContractForecastBN, DEFAULT_FEATURE_NAMES


def _aligned(traces, targets):
    if not isinstance(traces, (list, tuple)) or not traces or len(traces) != len(targets):
        raise ValueError("aligned nonempty independent trace groups required")
    for trace, labels in zip(traces, targets):
        if not isinstance(trace, (list, tuple)) or not trace or len(trace) != len(labels):
            raise ValueError("each trace needs aligned next-call inputs and separate targets")


def contract_trace_loss_gradient(model, theta, traces, targets, regularization=0.):
    """Mean trace-grouped composite negative log score plus lambda*||theta||².

    The A/C/B/S marginal likelihood is exact WITHIN a window. Summing such
    scores does not establish a generative joint distribution across windows.
    """
    _aligned(traces, targets)
    if not np.isscalar(regularization) or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("finite nonnegative regularization required")
    theta = model._theta(theta)
    total, gradient = 0., np.zeros_like(theta)
    for trace, labels in zip(traces, targets):
        for sample, target in zip(trace, labels):
            likelihood = model.label_log_likelihood(sample, target, theta=theta)
            total -= likelihood["log_probability"]
            gradient -= likelihood["score"]
    return (float(total / len(traces) + regularization * (theta @ theta)),
            gradient / len(traces) + 2 * regularization * theta)


def fit_contract_traces(traces, targets, *, feature_names=DEFAULT_FEATURE_NAMES,
                        regularization=.001, max_iter=500, initial=None):
    """Fit conditional event factors, not risk outcomes or API model weights."""
    _aligned(traces, targets)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("positive iteration bound required")
    model = ContractForecastBN(feature_names, theta=initial)
    observed, informative = 0, 0
    reference = ContractForecastBN(feature_names)
    for trace, labels in zip(traces, targets):
        for sample, target in zip(trace, labels):
            check = model.label_log_likelihood(sample, target)
            observed += len(check["observed_variables"])
            # Hard evidence repeated as a label is not new supervision. Test
            # support under an unsaturated reference, not the starting logits.
            informative += reference.label_log_likelihood(sample, target)["log_probability"] < -1e-12
    if informative == 0:
        raise ValueError("training needs at least one informative actually observed event label")
    original = copy.deepcopy([(sample.features, sample.current_a) for trace in traces for sample in trace])
    result = minimize(lambda theta: contract_trace_loss_gradient(model, theta, traces, targets, regularization),
                      model.theta.copy(), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all():
        raise ValueError("nonfinite next-call fit")
    model.theta = result.x.copy()
    model.optimizer = {"success": bool(result.success), "message": str(result.message),
                       "iterations": int(result.nit), "loss": float(result.fun),
                       "max_gradient": float(np.max(np.abs(result.jac))),
                       "trace_count": len(traces), "next_call_factors": sum(map(len, traces)),
                       "observed_label_fields": observed, "regularization": float(regularization),
                       "informative_next_call_factors": int(informative),
                       "objective": "mean over traces of summed per-window conditional composite negative log scores; not a generative sequence joint likelihood",
                       "partial_supervision": "exact within-window marginal likelihood over unobserved A/C/B/S",
                       "A_reestimated_each_window": True, "cross_window_state_model": False,
                       "risk_head_fitted": False, "calibration_evaluated": False}
    if original != [(sample.features, sample.current_a) for trace in traces for sample in trace]:
        raise ValueError("training changed supplied prefix features")
    return model
