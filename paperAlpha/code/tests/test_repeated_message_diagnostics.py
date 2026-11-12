"""Constructed mathematics and protocol fixtures, not empirical safety results."""
import asyncio
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.repeated_message_diagnostics import pair_moments, diagnose_verified_collection
from paperalpha_runtime import repeated_message_study as study
import collect_repeated_message_study as runner


def test_equal_group_weight_not_flattened_continuation_weight():
    result = pair_moments([[[0, 0], [0, 0]], [[1, 1]]])
    assert result["marginal_a"] == result["marginal_b"] == .5
    assert result["joint_frequencies"] == [.5, 0, 0, .5]
    assert result["descriptive_covariance"] == .25
    assert result["within_group_covariance"] == 0
    assert result["between_group_covariance"] == .25
    assert result["cross_repeat_contrast"] is None


def test_cross_repeat_contrast_excludes_same_reply_products():
    result = pair_moments([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])
    assert result["descriptive_covariance"] == .25
    assert result["within_group_covariance"] == .25
    assert result["between_group_covariance"] == 0
    # With only two repeats even a positive paired sample covariance does not
    # imply a positive message-propensity covariance estimate.
    assert result["cross_repeat_contrast"] == {
        "same_message_different_continuations": 0., "different_upstream_groups": .25,
        "difference": -.25, "same_continuation_minus_cross_repeat": .5}


def test_cross_repeat_contrast_matches_direct_double_sums():
    arrays = [np.array([[0, 1], [1, 1]]), np.array([[1, 0], [1, 1], [0, 0]]), np.array([[1, 0], [0, 1]])]
    within = np.mean([np.mean([a[k, 0] * a[l, 1] for k in range(len(a)) for l in range(len(a)) if k != l]) for a in arrays])
    across = np.mean([a[:, 0].mean() * b[:, 1].mean() for i, a in enumerate(arrays) for j, b in enumerate(arrays) if i != j])
    result = pair_moments(arrays)
    assert result["cross_repeat_contrast"]["difference"] == pytest.approx(within - across)
    assert abs(result["decomposition_error"]) < 1e-15
    assert sum(result["joint_frequencies"]) == pytest.approx(1)


@pytest.mark.parametrize("bad", [[[]], [[[1, 2]]], [[[np.nan, 0]]], [[[1]]], [[[0, 1, 1]]]])
def test_invalid_binary_pair_data_rejected(bad):
    with pytest.raises(ValueError): pair_moments(bad)


def test_empty_and_single_group_are_not_claimed_independence():
    assert pair_moments([])["descriptive_covariance"] is None
    assert pair_moments([[[0, 0], [0, 0]]])["cross_repeat_contrast"] is None


def test_fake_collection_integration_and_unchanged_source(tmp_path):
    from test_repeated_message_study import FakeAPI
    out = tmp_path / "diagnostic_fixture"
    runner.prepare(out)
    selected = [c for c in study.cases() if c["repeat"] == 0]
    groups = []
    api = FakeAPI()
    for case in selected:
        asyncio.run(runner.collect_group(out, case, api))
        groups.append(runner.verify_group(out, case))
    before = {str(p): runner.sha(p) for p in out.rglob("*.json")}
    report = diagnose_verified_collection({"schema": "verified-repeated-message-development-v1", "grouped_traces": groups})
    assert report["groups"] == 8 and report["base_tasks"] == 4
    assert len(report["strata"]) == 8
    assert report["event_learning_support"]["training_event_rows"] == 44
    assert report["event_learning_support"]["unique_training_responses"] == 44
    assert not report["model_performance_evaluated"] and report["fits"] == 0
    assert before == {str(p): runner.sha(p) for p in out.rglob("*.json")}
    excluded = copy.deepcopy(groups)
    for group in excluded:
        group.update(response_identity_qualified=False, response_identity_issues=[{"reason": "fixture_duplicate_id"}],
                     eligible_for_grouped_fit=False, eligible_for_statistics=False)
    empty = diagnose_verified_collection({"schema": "verified-repeated-message-development-v1", "grouped_traces": excluded})
    assert empty["event_learning_support"] is None
    assert len(empty["response_identity_exclusions"]) == 8
    assert sum(s["complete_groups"] for s in empty["strata"]) == 8
    assert all(s["all_reviewer_states"]["descriptive_covariance"] is None for s in empty["strata"])
    duplicate = {"schema": "verified-repeated-message-development-v1", "grouped_traces": groups + [groups[0]]}
    with pytest.raises(ValueError, match="duplicate"): diagnose_verified_collection(duplicate)
    mutated = copy.deepcopy(groups)
    mutated[0]["eligible_for_grouped_fit"] = False
    with pytest.raises(ValueError, match="eligibility"):
        diagnose_verified_collection({"schema": "verified-repeated-message-development-v1", "grouped_traces": mutated})
