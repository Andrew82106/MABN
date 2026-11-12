import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from diagnose_joint_alarm_timing import summarize_negative_scores


def trace(privileged=True):
    return {"events": [{"event_type": "input_received"}, {"event_type": "message_sent"},
                       {"event_type": "tool_call", "tool": "shell.exec" if privileged else "search.read"}]}


def test_late_peaks_do_not_turn_ties_into_late_evidence():
    result = summarize_negative_scores([(trace(), [.2, .3, .9]), (trace(), [.4, .4, .4])], .8, .1)
    assert result["first_peak_at_or_after_dispatch_n"] == 1
    assert result["pre_dispatch_negative_alarm_fraction_at_actual_threshold"] == 0
    assert result["counterfactual_pre_dispatch_only_threshold"]["threshold"] == np.nextafter(.4, np.inf)


def test_no_dispatch_uses_whole_observed_episode_but_no_false_late_peak():
    result = summarize_negative_scores([(trace(False), [.2, .3, .9])], .8, .1)
    assert result["with_privileged_dispatch_n"] == 0
    assert result["late_first_peak_fraction_given_dispatch"] is None
    assert result["pre_dispatch_negative_alarm_fraction_at_actual_threshold"] == 1


def test_mismatched_scores_fail():
    with pytest.raises(ValueError):
        summarize_negative_scores([(trace(), [.1])], .8, .1)
