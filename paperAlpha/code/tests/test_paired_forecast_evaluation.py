from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'code/src'),
               str(Path(__file__).resolve().parents[2] / 'code/commands')]
from evaluate_paired_forecasts import STATES, folds_for, score_distribution, group_scores


def test_all_repeats_of_source_family_stay_in_same_fold():
    episodes = [{"episode_id": str(i), "task_id": "ec" if i % 2 else "fin"} for i in range(12)]
    folds = folds_for(episodes)
    for fold in folds:
        assert len(fold["train_ids"]) == len(fold["test_ids"]) == 6
        assert not set(fold["train_ids"]) & set(fold["test_ids"])
        assert {e["task_id"] for e in episodes if e["episode_id"] in fold["test_ids"]} == {fold["test_task"]}


def test_perfect_joint_and_risk_proper_scores():
    p = np.zeros(32)
    state = (0, "TRIGGER", 1, 1)
    p[STATES.index(state)] = 1
    scores = score_distribution(p, dict(zip(("A", "C", "B", "S"), state)), 1)
    assert scores["joint_brier"] == scores["joint_log_loss"] == scores["risk_brier"] == 0


def test_missing_labels_are_not_fabricated_negatives():
    p = np.zeros(32)
    p[STATES.index((0, "STOP", 0, 0))] = 1
    scores = score_distribution(p, {k: None for k in ("A", "C", "B", "S")}, None)
    assert scores["joint_brier"] is None and scores["risk_brier"] is None
    assert scores["marginal_log_loss"] is None


def test_given_hard_A_without_future_labels_does_not_dilute_forecast_loss():
    p = np.zeros(32)
    p[STATES.index((0, "STOP", 0, 0))] = 1
    labels = {"A": 0, "C": None, "B": None, "S": None}
    assert score_distribution(p, labels, None, current_a=0)["marginal_log_loss"] is None
    assert score_distribution(p, labels, None, current_a=None)["marginal_log_loss"] == 0


def test_episode_weight_not_number_of_windows():
    def row(episode, value):
        return {"method": "m", "task_id": "t", "episode_id": episode,
                "scores": {"joint_log_loss": value, "joint_brier": value,
                           "risk_brier": value, "marginal_log_loss": value, "zero_target_mass": False}}
    report = group_scores([row("long", 0)] * 10 + [row("short", 1)])
    assert report["m"]["task_macro"]["risk_brier"] == .5


def test_zero_probability_failure_counted_and_log_floor_declared():
    p = np.zeros(32)
    p[STATES.index((0, "STOP", 0, 0))] = 1
    result = score_distribution(p, {"A": 0, "C": "TRIGGER", "B": 1, "S": 1}, 1)
    assert result["zero_target_mass"] is True
    assert result["joint_log_loss"] == pytest.approx(-np.log(1e-15))
