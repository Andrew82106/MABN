import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from branch_publisher_fixtures import fixture
from paperalpha_monitor.branch_benchmark import field_features, rule_distribution, metrics
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput


def test_classifier_sees_nonpooled_branch_arrangement_and_ownership():
    separated = fixture(w=((1, 0), (0, 1)))
    concentrated = fixture(w=((1, 0), (1, 0)))
    assert separated.release.packet == concentrated.release.packet
    assert not np.array_equal(field_features(separated), field_features(concentrated))
    assert not np.array_equal(field_features(separated), field_features(fixture(owners=(1, 0))))


def test_arbitrary_nonlabel_task_text_does_not_become_identifier_feature():
    sample = fixture()
    raw = copy.deepcopy(sample.observation)
    raw["publisher_payload"]["user_request"] = "A different public request identifier is not a feature."
    other = BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)
    assert np.array_equal(field_features(sample), field_features(other))


def test_rule_shared_gate_and_metrics():
    sample = fixture(current="root_unknown")
    dist = rule_distribution(sample, "visible_compliance_shared_gate")
    assert dist.tolist() == [.5, 0., .5]
    out = metrics([[1, 0, 0], [0, 0, 1]], [{"targets": {"1": 0, "2": 0}}, {"targets": {"1": 1, "2": 1}}])
    assert out["1"]["brier"] == out["2"]["brier"] == 0
    with pytest.raises(ValueError):
        rule_distribution(sample, "unknown")
