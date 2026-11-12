"""Constructed evidence only: no API calls or existing-data scoring."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]

from test_repeated_message_evaluation import cohort, with_failures
from paperalpha_monitor.partial_repeated_message_evaluation import (
    SUPERVISION_MODES, NORMALIZATION_GROUPS, continuation_slots, partial_trace,
    independent_partial_table, fit_partial_matched_glm, evaluate_partial_collection)
from paperalpha_monitor.partial_grouped_event_learning import (
    PartialGroupedEventTrace, partial_grouped_design, partial_grouped_loss_gradient,
    fit_partial_grouped_event_traces)
from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
from paperalpha_monitor.repeated_message_evaluation import (
    PENALTIES, PARENT_COLUMNS, checked_groups, bn_joint, after_review_dag)
from paperalpha_monitor.structured_event_baseline import glm_objective
from paperalpha_monitor import partial_repeated_message_evaluation as partial_evaluation


def partial_cohort():
    value = with_failures()
    partial, shared_only = value["grouped_traces"][3:5]
    original = partial["continuations"]
    partial.update(continuations=[], complete=False, eligible_for_grouped_fit=False,
        eligible_for_statistics=False, failures=["fixture-two-incomplete-pairs"],
        incomplete_continuations=[{"index": 0, "completed": False, "labels": {"publish-a": original[0]["publish-a"]}},
                                  {"index": 1, "completed": False, "labels": {"publish-b": original[1]["publish-b"]}}])
    shared_only.update(continuations=[], complete=False, eligible_for_grouped_fit=False,
        eligible_for_statistics=False, failures=["fixture-all-publishers-failed"],
        incomplete_continuations=[{"index": k, "completed": False, "labels": {}} for k in range(2)])
    return value


def test_reconstruct_original_slots_not_concatenated_lists():
    value = with_failures()
    first, second = value["grouped_traces"][:2]
    slots = continuation_slots(second)
    assert list(slots[0]) == ["publish-b"]
    assert slots[0] == second["incomplete_continuations"][0]["labels"]
    assert slots[1] == second["continuations"][0]
    assert continuation_slots(first) == [{}, {}]
    full = partial_trace(second)
    complete = partial_trace(second, mode="complete_pairs")
    assert len(full.continuations) == len(complete.continuations) == 2
    assert complete.continuations[0] == {} and complete.continuations[1] == slots[1]
    assert full.shared_labels == complete.shared_labels
    assert partial_trace(first).shared_labels == {}


@pytest.mark.parametrize("mutation", ["bad_index", "duplicate_index", "wrong_complete", "full_incomplete", "missing_slot", "after_upstream_failure"])
def test_invalid_continuation_index_contract_rejected(mutation):
    group = copy.deepcopy(partial_cohort()["grouped_traces"][3])
    if mutation == "bad_index": group["incomplete_continuations"][0]["index"] = True
    elif mutation == "duplicate_index": group["incomplete_continuations"][1]["index"] = 0
    elif mutation == "wrong_complete": group["incomplete_continuations"][0]["completed"] = True
    elif mutation == "full_incomplete": group["incomplete_continuations"][0]["labels"]["publish-b"] = {}
    elif mutation == "missing_slot": group["incomplete_continuations"].pop()
    else: group["upstream_complete"] = False
    with pytest.raises(ValueError): continuation_slots(group)


def test_partial_shared_upstream_is_informative_without_inventing_suffix():
    group = next(g for g in cohort()["grouped_traces"] if g["layout"] == "separate")
    first_review = next(iter(group["shared_labels"]))
    group.update(upstream_complete=False, complete=False,
        shared_labels={first_review: group["shared_labels"][first_review]}, continuations=[],
        incomplete_continuations=[], eligible_for_grouped_fit=False, eligible_for_statistics=False)
    trace = partial_trace(group)
    assert len(trace.shared_event_ids) == 2
    assert len(trace.shared_labels) == 1 and trace.continuations == [{}, {}]
    x, y, _ = independent_partial_table([trace])
    assert x.shape == (1, 25) and y.shape == (1,)
    assert partial_trace(group, mode="complete_pairs").shared_labels == {}


@pytest.mark.parametrize("penalty", PENALTIES)
def test_independent_glm_matches_partial_bn_with_fixed_planned_denominator(penalty):
    traces = [partial_trace(g) for g in partial_cohort()["grouped_traces"][:7]
              if g["response_identity_qualified"]]
    law = DAGWorkflowRiskBN().event_law
    x, y, families = independent_partial_table(traces)
    a, b, audit = partial_grouped_design(law, traces)
    assert np.array_equal(x, a) and np.array_equal(y, b)
    assert len(families) == len(y) and audit["empty_groups"] == 1
    theta = np.random.default_rng(182).normal(0, .2, 25)
    actual = glm_objective(theta, x, y, NORMALIZATION_GROUPS, penalty)
    expected = partial_grouped_loss_gradient(law, theta, traces, penalty, normalization_groups=NORMALIZATION_GROUPS)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert np.allclose(actual[1], expected[1], atol=1e-12)
    assert glm_objective(np.zeros(25), x, y, NORMALIZATION_GROUPS, penalty)[0] == pytest.approx(len(y)*np.log(2)/36)
    bn = fit_partial_grouped_event_traces(traces, regularization=penalty, normalization_groups=36)
    glm = fit_partial_matched_glm(traces, penalty=penalty)
    assert np.max(abs(bn.theta[:25]-glm.coef)) < 1e-7
    assert bn.optimizer["normalization_groups"] == glm.fit_report["normalization_groups"] == 36
    assert not bn.optimizer["authority_priors_fitted"]


def test_glm_is_independently_constructed_and_rejects_reused_records():
    group = cohort()["grouped_traces"][0]
    trace = partial_trace(group)
    with pytest.raises(ValueError, match="across training groups"):
        independent_partial_table([trace, trace])
    empty = PartialGroupedEventTrace(trace.dag, trace.shared_event_ids, {}, [{}, {}])
    x, y, families = independent_partial_table([empty])
    assert x.shape == (0, 25) and not len(y) and not families
    with pytest.raises(ValueError, match="not evaluable"):
        fit_partial_matched_glm([empty], penalty=.001)
    with pytest.raises(ValueError, match="normalization"):
        fit_partial_matched_glm([trace], penalty=.001, normalization_groups=True)


def test_missing_ancestor_is_rejected_not_filled_with_zero():
    group = cohort()["grouped_traces"][0]
    with pytest.raises(ValueError, match="parents"):
        PartialGroupedEventTrace(group["dag"], list(group["shared_labels"]), {},
                                [{"publish-a": group["continuations"][0]["publish-a"]}, {}])


@pytest.fixture(scope="module")
def evaluated():
    return evaluate_partial_collection(partial_cohort())


def test_partial_coverage_retains_original_eligibility_and_effective_labels(evaluated):
    report = evaluated
    assert report["development_only"] and not report["calibrated"] and not report["old_test_read"]
    assert not report["model_expressiveness_changed"] and report["prospective_predictions"] == 0
    assert report["selected_penalty"] is report["selected_supervision_mode"] is None
    assert report["coverage"]["eligible_training_groups"] == report["coverage"]["eligible_scoring_groups"] == 44
    assert report["coverage"]["identity_excluded_groups"] == 1
    complete, partial = (report["supervision_coverage"][m] for m in SUPERVISION_MODES)
    assert complete["identity_qualified_input_groups"] == partial["identity_qualified_input_groups"] == 47
    assert complete["informative_training_groups"] == 44 and complete["empty_training_groups"] == 3
    assert partial["informative_training_groups"] == 46 and partial["empty_training_groups"] == 1
    assert partial["informative_not_originally_eligible_groups"] == 2
    assert complete["effective_labels"]["publisher_incomplete_pairs"] == 0
    assert partial["effective_labels"]["publisher_incomplete_pairs"] == 3
    assert partial["effective_labels"]["publisher_complete_pairs"] == complete["effective_labels"]["publisher_complete_pairs"] == 174
    for coverage in (complete, partial):
        labels = coverage["effective_labels"]
        assert labels["positive"]+labels["negative"] == labels["total"]
        assert labels["shared"]+labels["publisher_total"] == labels["total"]
        assert coverage["original_eligibility_unchanged"]
    assert len(report["models"]) == 16 and len(report["summaries"]) == 32
    assert len(report["predictions"]) == 44*2*2*4*2
    assert len(report["paired_group_contrasts"]) == 44*2*2*3*2
    assert len(report["paired_supervision_contrasts"]) == 44*2*2*4
    assert any(g["partial_training_informative"] and not g["eligible_for_grouped_fit"] for g in report["group_eligibility"])
    json.dumps(report, allow_nan=False)


def test_same_score_records_both_modes_and_no_task_leakage(evaluated):
    for fold in evaluated["folds"]:
        assert len(fold["planned_train_ids"]) == fold["normalization_groups"] == 36
        assert fold["held_out_task"] not in fold["training_tasks"]
        assert not set(fold["planned_train_ids"]) & set(fold["score_ids"])
        for mode in SUPERVISION_MODES:
            report = fold["supervision_modes"][mode]
            assert report["normalization_groups"] == 36 and report["status"] == "evaluated"
            assert set(report["train_ids"]) <= set(fold["planned_train_ids"])
    indexed = {(r["group_id"], r["penalty"], r["phase"], r["method"], r["supervision_mode"]): r
               for r in evaluated["predictions"]}
    for key, row in indexed.items():
        opposite = indexed[*key[:-1], "all_observed" if key[-1] == "complete_pairs" else "complete_pairs"]
        for field in ("input_sha256", "actual_response_records", "actual_outcomes", "continuations"):
            assert row[field] == opposite[field]
        assert len(row["actual_outcomes"]) == row["continuations"]
        assert all(len(bits) == 2 for bits in row["actual_outcomes"])
    assert all(s["groups"] == 44 and s["continuations"] == 87 for s in evaluated["summaries"])
    assert all(c["same_scored_group_ids"] and c["direct_summary_comparison_valid"]
               for c in evaluated["supervision_comparison_coverage"])
    assert all(s["groups"] == 44 for s in evaluated["paired_supervision_summaries"])


def test_equivalence_independence_and_restore_audits(evaluated):
    audit = evaluated["numerical_audit"]
    assert audit["status"] == "passed"
    assert audit["comparison_counts"]["bn_glm_joint"] == 44*2*2*2
    assert all(v < 1e-8 for v in audit["maximum_difference"].values())
    for contrast in evaluated["paired_group_contrasts"]:
        if contrast["control"] == "matched_glm" or (contrast["phase"] == "after_review" and contrast["control"] == "same_marginals_independent"):
            assert all(abs(v) < 1e-8 for v in contrast["delta_bn_minus_control"].values() if v is not None)
    _, traces = checked_groups(partial_cohort())
    for model in evaluated["models"]:
        assert model["bn_glm_max_coefficient_difference"] < 1e-7
        restored = DAGWorkflowRiskBN.restore(model["no_parent_retrained"])
        assert not np.any(restored.theta[list(PARENT_COLUMNS)])
        row = next(r for r in evaluated["predictions"] if r["method"] == "no_parent_retrained"
                   and (r["held_out_task"], r["penalty"], r["supervision_mode"]) ==
                   (model["held_out_task"], model["penalty"], model["supervision_mode"]))
        trace = traces[row["group_id"]]
        dag = trace.dag if row["phase"] == "early" else after_review_dag(trace)
        assert np.allclose(bn_joint(restored, dag), row["joint"], atol=1e-12)


def test_empty_collection_not_evaluable_without_fabricated_fit_or_score():
    value = cohort()
    for group in value["grouped_traces"]:
        group.update(upstream_complete=False, complete=False, shared_labels={}, continuations=[],
            eligible_for_grouped_fit=False, eligible_for_statistics=False, api_calls=1, failures=["fixture"])
    report = evaluate_partial_collection(value)
    assert not report["models"] and not report["predictions"] and not report["paired_supervision_contrasts"]
    assert len(report["folds"]) == 4 and len(report["summaries"]) == 32
    assert all(m["status"] == "not_evaluable_no_information" for f in report["folds"] for m in f["supervision_modes"].values())
    assert all(s["overall"] is None for s in report["summaries"])
    assert report["supervision_coverage"]["all_observed"]["empty_training_groups"] == 48


def test_one_mode_has_no_information_marks_unpaired_summaries(monkeypatch):
    value = cohort()
    scoring_task = value["grouped_traces"][0]["base_task"]
    for group in value["grouped_traces"]:
        if group["base_task"] != scoring_task:
            group.update(continuations=[], complete=False, eligible_for_grouped_fit=False,
                eligible_for_statistics=False, failures=["fixture-all-publishers-failed"],
                incomplete_continuations=[{"index": k, "completed": False, "labels": {}} for k in range(2)])

    def fixed_bn(traces, *, model_class=DAGWorkflowRiskBN, **kwargs):
        model = model_class()
        model.optimizer = {"success": True, "fixture_only": True}
        return model

    def fixed_glm(traces, *, penalty, **kwargs):
        model = partial_evaluation.StructuredEventBaseline(kind="glm", penalty=penalty)
        model.coef = np.zeros(25)
        model.fit_report = {"fixture_only": True}
        return model

    monkeypatch.setattr(partial_evaluation, "fit_partial_grouped_event_traces", fixed_bn)
    monkeypatch.setattr(partial_evaluation, "fit_partial_matched_glm", fixed_glm)
    report = evaluate_partial_collection(value)
    fold = next(f for f in report["folds"] if f["held_out_task"] == scoring_task)
    assert fold["supervision_modes"]["complete_pairs"]["status"] == "not_evaluable_no_information"
    assert fold["supervision_modes"]["all_observed"]["status"] == "evaluated"
    assert report["predictions"] and not report["paired_supervision_contrasts"]
    for coverage in report["supervision_comparison_coverage"]:
        assert not coverage["same_scored_group_ids"] and not coverage["direct_summary_comparison_valid"]
        assert coverage["scored_groups_by_mode"] == {"complete_pairs": 0, "all_observed": 12}
        assert coverage["common_score_ids"] == []
    assert all(s["groups"] == 0 and s["overall"] is None for s in report["paired_supervision_summaries"])


def test_input_objects_and_original_flags_not_mutated():
    value = partial_cohort()
    before = [(g["dag"].specification(), copy.deepcopy({k: v for k, v in g.items() if k != "dag"}))
              for g in value["grouped_traces"]]
    for group in value["grouped_traces"]:
        if group["response_identity_qualified"]:
            partial_trace(group)
            partial_trace(group, mode="complete_pairs")
    after = [(g["dag"].specification(), {k: v for k, v in g.items() if k != "dag"}) for g in value["grouped_traces"]]
    assert before == after


@pytest.mark.parametrize("mutation", ["missing_group", "duplicate_group", "unverified", "invalid_eligibility"])
def test_original_cohort_contract_still_required(mutation):
    value = cohort()
    if mutation == "missing_group": value["grouped_traces"].pop()
    elif mutation == "duplicate_group": value["grouped_traces"][0] = copy.deepcopy(value["grouped_traces"][1])
    elif mutation == "unverified": value["schema"] = "not-verified"
    else: value["grouped_traces"][0]["response_identity_qualified"] = False
    with pytest.raises(ValueError): evaluate_partial_collection(value)
