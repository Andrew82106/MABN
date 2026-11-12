"""Constructed missing-content checks, not evidence of real API performance."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from test_grouped_event_learning import fixture, multiple_literals
from paperalpha_monitor.grouped_event_learning import grouped_complete_design, grouped_loss_gradient, fit_grouped_event_traces
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, DAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import trace_loss_gradient
from paperalpha_monitor.partial_grouped_event_learning import (PartialGroupedEventTrace,
    partial_grouped_design, partial_grouped_loss_gradient, fit_partial_grouped_event_traces)
from paperalpha_monitor.repeated_message_evaluation import _NoParentBN, PARENT_COLUMNS


def partial(full=None, *, shared=None, continuations=None):
    full = fixture() if full is None else full
    return PartialGroupedEventTrace(full.dag, tuple(full.shared_labels),
        full.shared_labels if shared is None else shared,
        full.continuations if continuations is None else continuations)


@pytest.mark.parametrize("layout", ["shared", "separate"])
def test_complete_observations_exactly_reduce_to_original_grouped_trainer(layout):
    full = fixture(layout)
    group, law = partial(full), DAGWorkflowRiskBN().event_law
    x, y, audit = partial_grouped_design(law, [group])
    old_x, old_y, old_audit = grouped_complete_design(law, [full])
    assert np.array_equal(x, old_x) and np.array_equal(y, old_y)
    for field in ("groups", "shared_event_labels", "continuation_event_labels", "continuations", "unique_response_records", "rows"):
        assert audit[field] == old_audit[field]
    theta = np.random.default_rng(563).normal(0, .2, 25)
    actual = partial_grouped_loss_gradient(law, theta, [group], .03)
    expected = grouped_loss_gradient(law, theta, [full], .03)
    assert actual[0] == expected[0] and np.array_equal(actual[1], expected[1])
    fitted = fit_partial_grouped_event_traces([group], regularization=.03)
    original = fit_grouped_event_traces([full], regularization=.03)
    assert np.array_equal(fitted.theta, original.theta)


def test_partial_siblings_keep_original_slots_and_shared_factor_once():
    full = fixture(values=((1, 0), (0, 1)))
    group = partial(full, continuations=[{}, {"publish-b": full.continuations[1]["publish-b"]}])
    law = DAGWorkflowRiskBN().event_law
    x, y, audit = partial_grouped_design(law, [group])
    assert x.shape == (2, 25) and y.tolist() == [1., 1.]
    assert audit["shared_event_labels"] == audit["continuation_event_labels"] == 1
    assert audit["continuations"] == 2 and audit["nonempty_continuations"] == 1
    assert audit["unobserved_continuation_event_slots"] == 3
    assert [v["continuation"] for v in audit["rows"]] == [None, 1]
    # Ancestor-closed observed likelihood equals exact marginal likelihood.
    theta = np.random.default_rng(524).normal(0, .4, 25)
    expected = trace_loss_gradient(law, theta, [full.dag],
                                  [{**full.shared_labels, **group.continuations[1]}], .01)
    actual = partial_grouped_loss_gradient(law, theta, [group], .01)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert np.allclose(actual[1], expected[1], atol=1e-12)


def test_incomplete_shared_block_retains_actual_success_and_no_fake_negatives():
    full = fixture("separate")
    group = partial(full, shared={"review-a": full.shared_labels["review-a"]}, continuations=[{}, {}])
    x, y, audit = partial_grouped_design(DAGWorkflowRiskBN().event_law, [group])
    assert x.shape == (1, 25) and y.tolist() == [1.]
    assert audit["planned_shared_event_slots"] == 2 and audit["unobserved_shared_event_slots"] == 1
    assert audit["continuation_event_labels"] == 0
    assert audit["informative_groups"] == 1
    assert group.shared_event_ids == ("review-a", "review-b")


def test_all_empty_groups_keep_zero_information_and_reject_fit():
    group = partial(shared={}, continuations=[{}, {}])
    law = DAGWorkflowRiskBN().event_law
    x, y, audit = partial_grouped_design(law, [group])
    assert x.shape == (0, 25) and y.shape == (0,)
    assert audit["empty_groups"] == 1 and audit["informative_groups"] == 0
    assert audit["unique_response_records"] == 0 and audit["rows"] == []
    theta = np.ones(25)
    value, gradient = partial_grouped_loss_gradient(law, theta, [group], .03, normalization_groups=36)
    assert value == pytest.approx(.75) and np.allclose(gradient, .06)
    with pytest.raises(ValueError, match="not informative"):
        fit_partial_grouped_event_traces([group], normalization_groups=36)


def test_explicit_normalizer_preserves_penalty_scale_and_empty_groups_do_not_change_fit():
    informative = partial(fixture(identity="observed"))
    empty = partial(fixture(identity="empty"), shared={}, continuations=[{}, {}])
    law = DAGWorkflowRiskBN().event_law
    theta = np.random.default_rng(548).normal(0, .3, 25)
    one = partial_grouped_loss_gradient(law, theta, [informative], .03, normalization_groups=36)
    two = partial_grouped_loss_gradient(law, theta, [informative, empty], .03, normalization_groups=36)
    assert one[0] == two[0] and np.array_equal(one[1], two[1])
    rescaled = partial_grouped_loss_gradient(law, theta, [informative, empty], .03*36/2)
    assert rescaled[0] == pytest.approx(two[0]*36/2)
    assert np.allclose(rescaled[1], two[1]*36/2)
    first = fit_partial_grouped_event_traces([informative], normalization_groups=36)
    second = fit_partial_grouped_event_traces([informative, empty], normalization_groups=36)
    assert np.array_equal(first.theta, second.theta)
    assert second.optimizer["normalization_groups"] == 36
    assert second.optimizer["training_audit"]["empty_groups"] == 1


def test_partial_gradient_restore_and_no_parent_ablation():
    a, b = fixture(identity="one"), fixture("separate", identity="two")
    groups = [partial(a, continuations=[a.continuations[0], {}]),
              partial(b, shared={"review-a": b.shared_labels["review-a"]}, continuations=[{}, {}])]
    before = [g.specification() for g in groups]
    law = DAGWorkflowRiskBN().event_law
    theta = np.random.default_rng(888).normal(0, .4, 25)
    value, gradient = partial_grouped_loss_gradient(law, theta, groups, .03, normalization_groups=36)
    numeric = approx_derivative(lambda t: partial_grouped_loss_gradient(law, t, groups, .03, normalization_groups=36)[0], theta).ravel()
    assert np.isfinite(value) and np.allclose(gradient, numeric, atol=1e-8)
    model = fit_partial_grouped_event_traces(groups, regularization=.03, normalization_groups=36)
    restored = DAGWorkflowRiskBN.restore(model.parameters())
    assert np.array_equal(restored.theta, model.theta)
    assert np.array_equal(model.theta[25:], np.zeros(3))
    assert model.optimizer["authority_priors_fitted"] is False and not model.optimizer["calibrated"]
    assert "not the full API-log" in model.optimizer["target"]
    no_parent = fit_partial_grouped_event_traces(groups, model_class=_NoParentBN, normalization_groups=36)
    assert not np.any(no_parent.theta[list(PARENT_COLUMNS)])
    assert before == [g.specification() for g in groups]


def test_a_missing_shared_parent_cannot_be_filled_by_a_bit_or_mean():
    full = fixture()
    with pytest.raises(ValueError, match="actual parents"):
        partial(full, shared={}, continuations=[full.continuations[0], {}])


def test_missing_parent_cannot_be_borrowed_from_another_continuation():
    full = fixture()
    spec = full.dag.specification()
    spec["workflow"]["edges"].append(["publish-a", "publish-b"])
    edge = copy.deepcopy(spec["edges"][-1])
    edge.update(source="publish-a", target="publish-b")
    spec["edges"].append(edge)
    dag = ObservableEventDAG(**spec)
    with pytest.raises(ValueError, match="cross-continuation"):
        PartialGroupedEventTrace(dag, list(full.shared_labels), full.shared_labels,
            [{"publish-a": full.continuations[0]["publish-a"]}, {"publish-b": full.continuations[1]["publish-b"]}])


@pytest.mark.parametrize("mutation", ["no_ids", "duplicate_ids", "observed_id", "nonancestor_block", "all_future_shared",
    "wrong_shared_label", "shared_parent_missing", "no_slots", "wrong_slot_type", "wrong_suffix_label", "missing_past"])
def test_invalid_structures_and_nonancestor_successes_rejected(mutation):
    full = fixture()
    dag, ids, shared, suffix = full.dag, list(full.shared_labels), copy.deepcopy(full.shared_labels), copy.deepcopy(full.continuations)
    if mutation == "no_ids": ids = []
    elif mutation == "duplicate_ids": ids *= 2
    elif mutation == "observed_id": ids = ["source"]
    elif mutation == "nonancestor_block": ids = ["publish-a"]
    elif mutation == "all_future_shared": ids = [n for n in dag.ids if n not in dag.observations]
    elif mutation == "wrong_shared_label": shared["publish-a"] = suffix[0]["publish-a"]
    elif mutation == "shared_parent_missing":
        ids = ["review-a", "publish-a"]
        shared = {"publish-a": suffix[0]["publish-a"]}
        suffix = [{}, {}]
    elif mutation == "no_slots": suffix = []
    elif mutation == "wrong_slot_type": suffix = [None, {}]
    elif mutation == "wrong_suffix_label": suffix[0]["review-a"] = shared["review-a"]
    else:
        spec = dag.specification()
        spec["observations"] = {}
        spec["missing_past"] = ["source"]
        dag = ObservableEventDAG(**spec)
    with pytest.raises(ValueError): PartialGroupedEventTrace(dag, ids, shared, suffix)


@pytest.mark.parametrize("mutation", ["backward_slot", "changed_source_id", "prefix_alias", "cross_actor",
                                     "same_tick", "wrong_literal", "tampered"])
def test_record_identity_chronology_and_evidence_guards_remain(mutation):
    full = fixture()
    shared, suffix = copy.deepcopy(full.shared_labels), copy.deepcopy(full.continuations)
    if mutation == "backward_slot": suffix.reverse()
    elif mutation == "changed_source_id":
        suffix[1]["publish-a"]["record_id"] = suffix[0]["publish-a"]["record_id"]
        suffix[1]["publish-a"]["source"] = "renamed-source"
    elif mutation == "prefix_alias": shared["review-a"]["record_id"] = full.dag.observations["source"]["record_id"]
    elif mutation == "cross_actor": suffix[0]["publish-b"] = copy.deepcopy(suffix[0]["publish-a"])
    elif mutation == "same_tick": suffix[0]["publish-b"]["sequence"] = suffix[0]["publish-a"]["sequence"]
    elif mutation == "wrong_literal": suffix[0]["publish-a"]["value"] ^= 1
    else:
        group = partial(full)
        group.continuations[0].pop("publish-a")
        with pytest.raises(ValueError, match="evidence changed"): group.validate()
        return
    with pytest.raises(ValueError): partial(full, shared=shared, continuations=suffix)


def test_distinct_literals_same_message_allowed_and_cross_group_alias_rejected():
    spec, labels, suffix = multiple_literals()
    group = PartialGroupedEventTrace(ObservableEventDAG(**spec), list(labels), labels, [suffix[0], {}])
    law = DAGWorkflowRiskBN().event_law
    x, y, audit = partial_grouped_design(law, [group])
    assert len(y) == 4 and audit["unique_response_records"] == 3
    with pytest.raises(ValueError, match="reused across training groups"):
        partial_grouped_design(law, [group, group])


@pytest.mark.parametrize("normalizer", [True, 0, -1, .5, 1., float("nan"), 1])
def test_normalizer_is_explicit_integer_not_smaller_than_input_groups(normalizer):
    groups = [partial(fixture(identity="a")), partial(fixture(identity="b"))]
    with pytest.raises(ValueError, match="normalization_groups"):
        partial_grouped_loss_gradient(DAGWorkflowRiskBN().event_law, np.zeros(25), groups, normalization_groups=normalizer)
    with pytest.raises(ValueError, match="normalization_groups"):
        fit_partial_grouped_event_traces(groups, normalization_groups=normalizer)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_bad_penalty_rejected(bad):
    with pytest.raises(ValueError): fit_partial_grouped_event_traces([partial()], regularization=bad)


@pytest.mark.parametrize("kind", ["unknown", "nonfactorized", "changed_spec"])
def test_generic_or_joint_laws_are_not_silently_fit_with_logistic_loss(kind):
    law = DAGWorkflowRiskBN().event_law
    if kind == "unknown": law = SimpleNamespace(names=law.names, factorizes_into_independent_events=True, design=law.design)
    elif kind == "nonfactorized": law.factorizes_into_independent_events = False
    else: law.specification["changed"] = True
    with pytest.raises(ValueError): partial_grouped_design(law, [partial()])
    with pytest.raises(ValueError): partial_grouped_loss_gradient(law, np.zeros(25), [partial()])
