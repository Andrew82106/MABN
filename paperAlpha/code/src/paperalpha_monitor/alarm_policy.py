"""Causal-in-time two-phase alarm decisions for arbitrary prefix risk scores.

The phase is observable, not a causal interpretation of risk. This module does
not change probabilities, fit models or consume outcome labels. It consumes
already accepted runtime events. Threshold calibration is empirical only.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .streaming_bn import PRIVILEGED_TOOLS

PHASES = ("before_dispatch", "after_dispatch")
ALL_NEGATIVE = float(np.nextafter(1., np.inf))


def dispatch_phases(events):
    after = False
    phases = []
    for event in events:
        after = after or (event.get("event_type") == "tool_call" and event.get("tool") in PRIVILEGED_TOOLS)
        phases.append(PHASES[int(after)])
    return phases


def _scores(values, n):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not n or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("one finite probability per nonempty observed event prefix required")
    return values


def phase_maxima(events, scores):
    scores = _scores(scores, len(events))
    phases = dispatch_phases(events)
    return {phase: max((float(score) for score, p in zip(scores, phases) if p == phase), default=None)
            for phase in PHASES}


def calibrate_phase(maxima, alpha):
    """None means no event in this phase, NOT a score of zero."""
    if not maxima or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("nonempty calibration episodes and budget in [0,1] required")
    observed = np.asarray([value for value in maxima if value is not None], dtype=float)
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
        raise ValueError("invalid phase maximum")
    allowed = math.floor(alpha * len(maxima))
    # With no calibration coverage, explicitly disable this phase.
    if not len(observed):
        threshold = ALL_NEGATIVE
    else:
        candidates = np.unique(np.r_[0., np.nextafter(observed, np.inf), ALL_NEGATIVE])
        threshold = next(float(t) for t in candidates if np.count_nonzero(observed >= t) <= allowed)
    alarms = int(np.count_nonzero(observed >= threshold))
    return {"threshold": threshold, "negative_episode_n": len(maxima),
            "phase_observed_episode_n": len(observed), "allowed_alarm_n": allowed,
            "actual_alarm_n": alarms, "empirical_fpr_all_negative_episodes": alarms / len(maxima),
            "budget": alpha, "phase_disabled_no_calibration_coverage": not len(observed)}


def calibrate_policy(negative_episodes, phase_budgets, total_budget):
    """Input contains only (events, score-vector) validation-negative pairs."""
    if set(phase_budgets) != set(PHASES) or not math.isfinite(total_budget) or not 0 <= total_budget <= 1:
        raise ValueError("invalid policy budget")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in phase_budgets.values()):
        raise ValueError("invalid phase budget")
    if sum(phase_budgets.values()) > total_budget + 1e-15:
        raise ValueError("phase budgets exceed total budget")
    maxima = [phase_maxima(events, scores) for events, scores in negative_episodes]
    phases = {phase: calibrate_phase([row[phase] for row in maxima], phase_budgets[phase]) for phase in PHASES}
    policy = {"name": "fixed-two-phase-v1", "switch": "first_observed_privileged_dispatch_inclusive",
              "thresholds": {phase: phases[phase]["threshold"] for phase in PHASES}}
    alarms = [any(row[p] is not None and row[p] >= policy["thresholds"][p] for p in PHASES) for row in maxima]
    actual = int(sum(alarms))
    union_bound = sum(phase["actual_alarm_n"] for phase in phases.values())
    if actual > union_bound or actual / len(maxima) > total_budget + 1e-15:
        raise AssertionError("empirical full-trajectory budget violated")
    return {"policy": policy, "phase_calibration": phases, "negative_episode_n": len(maxima),
            "any_alarm_n": actual, "empirical_any_alarm_fpr": actual / len(maxima),
            "union_bound_alarm_n": union_bound, "total_budget": total_budget,
            "guarantee_scope": "empirical validation negatives only, not a population or deployment guarantee"}


class PhaseAlarmStream:
    """Small decision layer; pass only events accepted by the risk monitor."""

    def __init__(self, policy):
        if not isinstance(policy, dict) or policy.get("name") != "fixed-two-phase-v1" or policy.get("switch") != "first_observed_privileged_dispatch_inclusive":
            raise ValueError("unsupported alarm policy")
        thresholds = policy.get("thresholds", {})
        if set(thresholds) != set(PHASES) or any(isinstance(t, bool) or not isinstance(t, (int, float))
                or not math.isfinite(t) or not 0 <= t <= ALL_NEGATIVE for t in thresholds.values()):
            raise ValueError("invalid phase thresholds")
        self._policy = copy.deepcopy(policy)
        self._after = False

    def push(self, event, probability):
        if not isinstance(event, dict) or not isinstance(event.get("event_type"), str):
            raise ValueError("accepted event dictionary required")
        probability = float(_scores([probability], 1)[0])
        self._after = self._after or (event.get("event_type") == "tool_call" and event.get("tool") in PRIVILEGED_TOOLS)
        phase = PHASES[int(self._after)]
        threshold = self._policy["thresholds"][phase]
        return {"phase": phase, "threshold": threshold, "alarm": probability >= threshold}


def decide_policy(events, scores, policy):
    scores = _scores(scores, len(events))
    stream = PhaseAlarmStream(policy)
    return [stream.push(event, float(score)) for event, score in zip(events, scores)]
