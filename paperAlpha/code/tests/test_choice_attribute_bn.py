"""Attribute semantics, policy composition and equal-information controls."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]

from paperalpha_runtime.exclusive_purchase import cases, prefix
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input
from paperalpha_monitor.choice_attribute_bn import (AttributedDispatchInput, AttributeChoiceWorkflowRiskBN,
                                                      attribute_rows, warranty_claim)
from paperalpha_monitor.exclusive_dispatch_bn import ExclusiveDispatchWorkflowRiskBN
from scalar_binary_graph import evaluate


def packet(priority="balanced", *, revoked=False):
    case = cases()[0]; case["revoke_parent_after_proposals"] = revoked
    return prefix(case, json.dumps({"priority": priority, "work_order": "Constructed visible instruction."}))


def sample(priority="balanced", *, complete=True, revoked=False):
    return compile_attribute_input(packet(priority, revoked=revoked), authority_complete=complete)


def test_actual_catalogue_attributes_break_balanced_alias_without_sku_features():
    s = sample(); x, evidence = attribute_rows(s)
    assert x.shape == (4, 10)
    assert x[0, 3] == pytest.approx(35 / 90) and x[1, 3] == pytest.approx(65 / 90)
    assert x[0, 4] == pytest.approx(1 / 3) and x[1, 4] == 1
    assert not np.array_equal(x[0], x[1])
    assert evidence[0]["warranty_claim"]["years"] == 1
    m = AttributeChoiceWorkflowRiskBN(); m.theta[4] = 2
    p = m.predict(s)["dispatch_probabilities"]
    assert p[s.action_ids[1]] > p[s.action_ids[0]]


@pytest.mark.parametrize("text,years", [("one-year warranty.", 1), ("3 year warranty", 3),
                                       ("1.5-year warranty", 1.5), ("No warranty specified", None),
                                       ("one-year warranty or three-year warranty", None)])
def test_warranty_is_literal_and_unknown_is_not_zero(text, years):
    assert warranty_claim(text)["years"] == years


def test_missing_warranty_has_a_separate_indicator():
    value = packet(); value["branches"][0]["catalogue"][0]["description"] = "Details unavailable"
    x, _ = attribute_rows(compile_attribute_input(value))
    assert x[0, 4] == 0 and x[0, 5] == 1 and x[1, 5] == 0


@pytest.mark.parametrize("complete,revoked", [(True, False), (False, False), (True, True)])
def test_risk_graph_and_all_gradient_directions(complete, revoked):
    s = sample("quality", complete=complete, revoked=revoked); m = AttributeChoiceWorkflowRiskBN()
    m.theta = np.random.default_rng(45).normal(0, .2, len(m.theta))
    for conditions in (None, {s.action_ids[1]: 1}, {s.action_ids[0]: 0, s.action_ids[2]: 0}):
        result = m.predict(s, derivatives=True, event_conditions=conditions)
        for name, probability in result["probabilities"].items():
            assert evaluate(m.probability_graph(s, output=name, event_conditions=conditions)) == pytest.approx(probability, abs=1e-12)
        numeric = approx_derivative(lambda t: m.predict(s, theta=t, event_conditions=conditions)["probabilities"]["system:any"], m.theta).ravel()
        assert np.allclose(numeric, result["gradients"]["system:any"], atol=1e-8)


def test_policy_snapshot_changes_risk_not_choice_scores():
    low = packet(); high = copy.deepcopy(low); high["contracts"][0]["limit"] = "170"
    a, b = compile_attribute_input(low), compile_attribute_input(high)
    m = AttributeChoiceWorkflowRiskBN(); m.theta[4] = 2
    assert np.array_equal(attribute_rows(a)[0], attribute_rows(b)[0])
    assert m.predict(a)["dispatch_probabilities"] == m.predict(b)["dispatch_probabilities"]
    assert m.predict(a)["probabilities"]["system:any"] > 0
    assert m.predict(b)["probabilities"]["system:any"] == 0
    revoked = sample(revoked=True)
    assert np.array_equal(attribute_rows(a)[0], attribute_rows(revoked)[0])
    assert m.predict(revoked)["probabilities"]["system:any"] > m.predict(a)["probabilities"]["system:any"]


def test_catalogue_order_and_opaque_sku_names_do_not_change_choices():
    original = packet(); other = copy.deepcopy(original)
    mapping = {}
    for b in other["branches"]:
        for i, item in enumerate(b["catalogue"]):
            mapping[item["sku"]] = b["actor"] + "-opaque-" + str(9 - i)
            item["sku"] = mapping[item["sku"]]
        b["catalogue"].reverse()
    for action in other["candidate_actions"]:
        action["arguments"]["sku"] = mapping[action["arguments"]["sku"]]
    a, b = compile_attribute_input(original), compile_attribute_input(other)
    # Old support/opposition uses suffixes; the new score never consumes it.
    m = AttributeChoiceWorkflowRiskBN(); m.theta = np.random.default_rng(901).normal(0, .4, len(m.theta))
    assert np.array_equal(attribute_rows(a)[0], attribute_rows(b)[0])
    assert m.predict(a)["probabilities"] == pytest.approx(m.predict(b)["probabilities"], abs=1e-12)


def test_group_order_and_one_hot_support_are_preserved():
    a = sample(); evidence = copy.deepcopy(a.attribute_evidence)
    groups = copy.deepcopy(a.choice_groups)
    for g in groups: g["action_ids"].reverse()
    b = AttributedDispatchInput(workflow=a.workflow, authority=a.authority, proposals=a.proposals,
                               contracts=a.contracts, cutoff=a.cutoff, horizon=a.horizon,
                               choice_groups=groups, attribute_evidence=evidence)
    m = AttributeChoiceWorkflowRiskBN(); m.theta[4] = .7
    states, logs, _ = m._occurrence_worlds(a, m.theta)
    assert np.isfinite(logs).sum() == 9
    assert m.predict(a)["probabilities"] == pytest.approx(m.predict(b)["probabilities"], abs=1e-12)
    with pytest.raises(ValueError, match="zero model probability"):
        m.predict(a, event_conditions={a.action_ids[0]: 1, a.action_ids[1]: 1})


@pytest.mark.parametrize("mutation", ["future", "wrong_actor", "wrong_price", "wrong_limit", "missing_candidate",
                                      "planner", "outcome", "post_binding"])
def test_attribute_bindings_reject_wrong_evidence(mutation):
    s = sample(); ev = copy.deepcopy(s.attribute_evidence); r = ev["assignments"]["buyer-a"]
    if mutation == "future": r["sequence"] = s.cutoff + 1
    elif mutation == "wrong_actor": r["intended_recipient"] = "buyer-b"
    elif mutation == "wrong_price": r["assignment"]["catalogue"][0]["price"] = "1"
    elif mutation == "wrong_limit": r["assignment"]["single_limit"] = "100"
    elif mutation == "missing_candidate": r["assignment"]["catalogue"].pop()
    elif mutation == "planner": ev["planner_text"] = json.dumps({"priority": "quality", "work_order": "Changed"})
    elif mutation == "outcome": ev["chosen_sku"] = "a-enhanced"
    elif mutation == "post_binding":
        s.attribute_evidence["assignments"]["buyer-a"]["assignment"]["catalogue"][0]["description"] = "three-year warranty"
        with pytest.raises(ValueError): s.validate()
        return
    with pytest.raises(ValueError): AttributedDispatchInput.from_input(s, ev)


def test_joint_and_partial_training_gradient_and_restore():
    samples = [sample(p, complete=False) for p in ("economy", "balanced", "quality")]
    labels = [{"dispatch": {s.action_ids[0 if i == 0 else 1]: 1}} for i, s in enumerate(samples)]
    m = AttributeChoiceWorkflowRiskBN(regularization=.02)
    theta = np.random.default_rng(101).normal(0, .2, len(m.theta))
    loss, grad = m.loss_gradient(theta, samples, labels)
    numeric = approx_derivative(lambda t: m.loss_gradient(t, samples, labels)[0], theta).ravel()
    assert np.isfinite(loss) and np.allclose(numeric, grad, atol=1e-8)
    m.fit(samples, labels, max_iter=200)
    assert m.optimizer["success"]
    restored = AttributeChoiceWorkflowRiskBN.restore(m.parameters())
    assert restored.predict(samples[1]) == m.predict(samples[1])
    bad = copy.deepcopy(m.parameters()); bad["attribute_specification"]["warranty_scale_years"] = 99
    with pytest.raises(ValueError): AttributeChoiceWorkflowRiskBN.restore(bad)


def test_old_choice_law_does_not_gain_attribute_features_implicitly():
    s = sample(); old = ExclusiveDispatchWorkflowRiskBN()
    old.theta[:3] = [1, 2, -1]
    result = old.predict(s)["dispatch_probabilities"]
    assert result[s.action_ids[0]] == result[s.action_ids[1]]
    # Historical model is an explicitly named old-feature ablation, not silently upgraded.


@pytest.mark.parametrize("text", ["No three-year warranty is included.", "A three-year warranty may apply.",
    "Three-year warranty if registered.", "Optional protection; three-year warranty.",
    "twenty-one-year warranty", "-1-year warranty", "A one-year warranty can be purchased separately.",
    "Coverage availability varies; three-year warranty.", "Eligibility determines coverage; three-year warranty."])
def test_qualified_or_unsupported_warranty_text_stays_unknown(text):
    assert warranty_claim(text)["years"] is None


def test_actor_catalogue_cannot_be_split_into_independent_slots():
    s = sample(); groups = copy.deepcopy(s.choice_groups)
    original = groups.pop(0)
    groups.extend({**original, "id": "split:" + a, "action_ids": [a]} for a in original["action_ids"])
    with pytest.raises(ValueError, match="exactly one exclusive slot"):
        AttributedDispatchInput(workflow=s.workflow, authority=s.authority, proposals=s.proposals,
                                contracts=s.contracts, cutoff=s.cutoff, horizon=s.horizon,
                                choice_groups=groups, attribute_evidence=s.attribute_evidence)


def test_forecast_delivery_condition_and_standalone_graph_identity_are_explicit():
    s = sample(); m = AttributeChoiceWorkflowRiskBN()
    result = m.predict(s); graph = m.probability_graph(s)
    assert "not already-delivered" in result["forecast_conditions"]["input_delivery"]
    assert graph["model"] == m.MODEL and graph["attribute_program_sha256"] == m.attribute_sha256
    assert graph["forecast_input_timing"] == "prepared_for_next_call"
    assert "not time order" in graph["categorical_edge_meaning"]
    ev = copy.deepcopy(s.attribute_evidence)
    ev["assignments"]["buyer-a"]["delivery_status"] = "observed_delivery"
    with pytest.raises(ValueError): AttributedDispatchInput.from_input(s, ev)


def test_catalogue_preamble_must_match_the_bound_assignment_not_arbitrary_prose():
    text = "Basic a lamp; meets essential requirements; one-year warranty."
    assert warranty_claim(text)["years"] is None
    assert warranty_claim(text, assignment_need="a lamp")["years"] == 1
    assert warranty_claim(text, assignment_need="a microphone")["years"] is None
    assert warranty_claim("Coverage availability varies; three-year warranty.", assignment_need="a lamp")["years"] is None


def test_graph_snapshots_change_with_model_parameters_or_input_evidence():
    m = AttributeChoiceWorkflowRiskBN(); first = sample(); a = m.probability_graph(first)
    m.theta[4] += .3; b = m.probability_graph(first)
    assert a["parameter_snapshot_sha256"] != b["parameter_snapshot_sha256"]
    assert a["attribute_evidence_sha256"] == b["attribute_evidence_sha256"]
    other = packet(); other["branches"][0]["catalogue"][0]["description"] = "one-year warranty."
    c = m.probability_graph(compile_attribute_input(other))
    assert b["attribute_evidence_sha256"] != c["attribute_evidence_sha256"]
    assert b["input_sha256"] != c["input_sha256"]
