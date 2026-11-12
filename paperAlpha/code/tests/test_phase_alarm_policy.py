from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from evaluate_phase_alarm_policy import alarm_metrics
import evaluate_prefix_forecast_v1 as protocol
from paperalpha_monitor.alarm_policy import (ALL_NEGATIVE, PHASES, PhaseAlarmStream,
    calibrate_phase, calibrate_policy, decide_policy, dispatch_phases)


def events(tool="shell.exec"):
    return [{"sequence": 0, "event_type": "input_received"},
            {"sequence": 1, "event_type": "tool_proposed", "tool": tool},
            {"sequence": 2, "event_type": "tool_call", "tool": tool},
            {"sequence": 3, "event_type": "message_sent"}]


def policy(pre=.4, post=.9):
    return {"name": "fixed-two-phase-v1", "switch": "first_observed_privileged_dispatch_inclusive",
            "thresholds": dict(zip(PHASES, [pre, post]))}


def test_phase_uses_only_current_and_past_dispatch_and_includes_dispatch_event():
    es = events()
    assert dispatch_phases(es) == [PHASES[0]] * 2 + [PHASES[1]] * 2
    assert dispatch_phases(events("search.read")) == [PHASES[0]] * 4
    decisions = decide_policy(es, [.5] * 4, policy())
    assert [d["alarm"] for d in decisions] == [True, True, False, False]
    for n in range(1, len(es) + 1):
        assert decide_policy(es[:n], [.5] * n, policy()) == decisions[:n]


def test_missing_phase_does_not_drop_episodes_from_denominator_or_fake_zero_score():
    result = calibrate_phase([None] * 9 + [.9], .1)
    assert result["threshold"] == 0
    assert result["negative_episode_n"] == 10 and result["phase_observed_episode_n"] == 1
    assert result["empirical_fpr_all_negative_episodes"] == .1
    none = calibrate_phase([None] * 10, .1)
    assert none["threshold"] == ALL_NEGATIVE and none["actual_alarm_n"] == 0
    assert none["phase_disabled_no_calibration_coverage"]


def test_tied_score_one_supports_all_negative_threshold():
    result = calibrate_phase([1., 1., None], 0.)
    assert result["threshold"] == ALL_NEGATIVE
    assert result["actual_alarm_n"] == 0
    assert not decide_policy(events(), [1.] * 4, policy(ALL_NEGATIVE, ALL_NEGATIVE))[0]["alarm"]


def test_empirical_union_budget_counts_both_phases_without_independence():
    rng = np.random.default_rng(123)
    pairs = [(events("search.read" if i % 5 == 0 else "shell.exec")[i % 3:], rng.random(4 - i % 3)) for i in range(101)]
    fitted = calibrate_policy(pairs, dict(zip(PHASES, [.08, .02])), .1)
    alarms = [any(d["alarm"] for d in decide_policy(es, scores, fitted["policy"])) for es, scores in pairs]
    assert sum(alarms) == fitted["any_alarm_n"]
    assert sum(alarms) <= fitted["union_bound_alarm_n"] <= 10
    assert all(p["negative_episode_n"] == 101 for p in fitted["phase_calibration"].values())
    with pytest.raises(ValueError, match="exceed"):
        calibrate_policy(pairs, dict(zip(PHASES, [.1, .1])), .1)


def test_no_pre_dispatch_opportunity_is_not_removed_from_early_recall_denominator():
    es = [events()[2:], events()]
    decisions = [[{"alarm": True}] * len(e) for e in es]
    result = alarm_metrics([1, 1], es, decisions)["warning_proxies"]["first_privileged_dispatch"]
    assert result == {"eligible_positive_n": 2, "strictly_before_alarm_n": 1, "recall": .5}


def test_global_alarm_metrics_match_existing_protocol():
    es = [events(), events("search.read"), events()]
    scores = [[.1, .2, .9, .3], [.8, .2, .1, .1], [.6, .6, .6, .6]]
    labels = [1, 0, 1]
    actual = alarm_metrics(labels, es, [[{"alarm": p >= .5} for p in ps] for ps in scores])
    old = protocol.evaluate_scores(labels, scores, [{"events": e} for e in es], .5)
    assert actual == {k: old[k] for k in actual}


def test_invalid_score_does_not_advance_phase_and_policy_is_copied():
    p = policy()
    stream = PhaseAlarmStream(p)
    p["thresholds"][PHASES[0]] = 1
    with pytest.raises(ValueError):
        stream.push(events()[2], float("nan"))
    assert stream.push(events()[0], .5) == {"phase": PHASES[0], "threshold": .4, "alarm": True}


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan")])
def test_invalid_scores_are_rejected(value):
    with pytest.raises(ValueError):
        calibrate_phase([value], .1)
