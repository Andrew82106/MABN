import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from workflow_risk_fixtures import fixture, rebuild
from scalar_binary_graph import evaluate
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN, WorkflowFragmentRegistry


def model_with_rates():
    model = WorkflowRiskBN(regularization=0.)
    model.theta[1:3] = logit([.6, .7])
    model.theta[-3:] = logit([.2, .3, .4])
    return model


def test_individually_safe_nodes_can_have_system_budget_risk():
    sample, model = fixture(authority="known"), model_with_rates()
    result = model.predict(sample)
    assert result["probabilities"]["node:step-0"] == 0.
    assert result["probabilities"]["node:step-1"] == 0.
    assert result["probabilities"]["channel:cumulative_budget"] == pytest.approx(.6 * .7)
    assert result["probabilities"]["system:any"] == pytest.approx(.42)


def test_same_dispatch_is_shared_across_mechanisms_not_independent_scores():
    model, sample = model_with_rates(), fixture(combination=True)
    p = model.predict(sample)["probabilities"]
    assert p["channel:authorization"] == pytest.approx(.2 * (1 - .4 * .3))
    assert p["channel:cumulative_budget"] == pytest.approx(.42)
    assert p["channel:prohibited_combination"] == pytest.approx(.42)
    assert p["system:any"] == pytest.approx(.42 + .2 * (.6 * .3 + .4 * .7))
    assert p["system:any"] != pytest.approx(sum(v for k, v in p.items() if k.startswith("channel:")))
    assert p["system:any"] == pytest.approx(model.predict(fixture())["probabilities"]["system:any"])


@pytest.mark.parametrize("shared,expected", [(True, .176), (False, .2432)])
def test_same_action_marginals_different_shared_authority_system_risk(shared, expected):
    model = model_with_rates()
    p = model.predict(fixture(shared=shared, budget="1000"))["probabilities"]
    assert p["authorization:a-0"] == pytest.approx(.12)
    assert p["authorization:a-1"] == pytest.approx(.14)
    assert p["system:any"] == pytest.approx(expected)


@pytest.mark.parametrize("amounts,limit,prior,expected", [(("0.1", "0.2"), "0.3", "0", 0.),
    (("0.1", "0.2"), "0.299999999999999999999999999999", "0", .42),
    (("1", "1"), "10", "11", 1.)])
def test_exact_decimal_budget_and_existing_overrun(amounts, limit, prior, expected):
    p = model_with_rates().predict(fixture(authority="known", amounts=amounts, budget=limit, prior=prior))["probabilities"]
    assert p["channel:cumulative_budget"] == pytest.approx(expected)


def test_absent_limits_not_safety_and_single_bound_separate_from_authority():
    model = model_with_rates()
    missing = model.predict(fixture(authority="known", budget=None, single_limit=None))
    assert missing["coverage"] == {"authorization": True, "single_bound": False, "cumulative_budget": False, "prohibited_combination": False}
    known = model.predict(fixture(authority="known", budget=None, single_limit="50"))
    assert known["probabilities"]["channel:authorization"] == 0
    assert known["probabilities"]["channel:single_bound"] == pytest.approx(.88)


def test_config_removal_changes_executable_topology_not_just_explanation():
    registry = WorkflowFragmentRegistry()
    spec = copy.deepcopy(registry.specification)
    spec["fragments"] = [f for f in spec["fragments"] if f["id"] != "cumulative_budget"]
    model = WorkflowRiskBN(registry=WorkflowFragmentRegistry(spec))
    p = model.predict(fixture(authority="known"))
    assert p["probabilities"]["system:any"] == 0
    assert p["disabled_templates"] == ["cumulative_budget"]
    assert all(n["id"] != "cumulative_budget:budget" for n in model.probability_graph(fixture())["nodes"])


@pytest.mark.parametrize("layout,rename", [("fork", ""), ("chain", ""), ("fork", "renamed-")])
def test_normative_edges_and_actor_names_do_not_invent_dependence(layout, rename):
    model = model_with_rates()
    p = model.predict(fixture(layout=layout, rename=rename))["probabilities"]["system:any"]
    assert p == pytest.approx(.512)


@pytest.mark.parametrize("authority", ["root", "delegation", "opaque", "known", "invalid"])
@pytest.mark.parametrize("combination", [False, True])
def test_every_query_equals_independent_existing_graph_interpreter(authority, combination):
    model = model_with_rates()
    sample = fixture(authority=authority, combination=combination)
    result = model.predict(sample, derivatives=True)
    assert result["mass_sum"] == pytest.approx(1.)
    for key, p in result["probabilities"].items():
        assert evaluate(model.probability_graph(sample, output=key)) == pytest.approx(p, abs=2e-14)
        numeric = approx_derivative(lambda t: model.predict(sample, theta=t)["probabilities"][key], model.theta).ravel()
        np.testing.assert_allclose(result["gradients"][key], numeric, atol=2e-8, rtol=1e-5)


def gradient_samples():
    samples, labels = [], []
    for mode in ("root", "delegation", "opaque"):
        for cue in ("support", "opposition", "unknown"):
            s = fixture(families=("payment", "release", "other"), visible=(cue,) * 3, amounts=("60", "0", "0"),
                        authority=mode, shared=False, budget=None, combination=True)
            samples.append(s)
            labels.append({"dispatch": {"a-0": 1, "a-1": 0, "a-2": 1}, "facts": {next(iter(s.authority.facts)): 1}, "risks": {"system:any": 1}})
    return samples, labels


def test_joint_partial_supervision_all_parameter_gradients():
    samples, labels = gradient_samples()
    model = WorkflowRiskBN(regularization=0.)
    model.theta = np.random.default_rng(712).normal(0, .2, len(model.theta))
    loss, gradient = model.loss_gradient(model.theta, samples, labels)
    numeric = approx_derivative(lambda t: model.loss_gradient(t, samples, labels)[0], model.theta).ravel()
    assert np.isfinite(loss) and len(model.names) == 12
    assert np.all(np.abs(gradient) > 1e-5)
    np.testing.assert_allclose(gradient, numeric, atol=3e-8, rtol=2e-5)


def test_unobserved_irrelevant_dispatch_is_not_falsely_identified():
    samples, labels = gradient_samples()
    for label in labels:
        del label["dispatch"]["a-2"]
    model = WorkflowRiskBN(regularization=0.)
    _, gradient = model.loss_gradient(model.theta, samples, labels)
    # First action already violates a labeled failed reference. A system=1
    # label alone says nothing about dispatch of the third independent action.
    np.testing.assert_allclose(gradient[6:9], 0., atol=1e-14)


def test_registry_mutation_is_not_silently_accepted():
    model = WorkflowRiskBN()
    model.registry.templates["authorization"]["meaning"] = "altered"
    with pytest.raises(ValueError, match="configuration changed"):
        model.predict(fixture())


def test_fit_restore_and_shared_parameter_learning_across_workflows():
    # Exact constructed frequencies, not independent sampled/API outcomes.
    samples, labels, weights = [], [], []
    for layout in ("fork", "chain"):
        s = fixture(authority="known", layout=layout, budget=None)
        for a, b, count in ((0, 0, 12), (0, 1, 28), (1, 0, 18), (1, 1, 42)):
            samples.append(s)
            labels.append({"dispatch": {"a-0": a, "a-1": b}})
            weights.append(count)
    model = WorkflowRiskBN(regularization=0.).fit(samples, labels, weights=weights)
    assert model.optimizer["success"]
    assert model.optimizer["loss"] < model.optimizer["initial_loss"]
    result = model.predict(samples[0])
    assert result["dispatch_probabilities"]["a-0"] == pytest.approx(.6, abs=2e-6)
    assert result["dispatch_probabilities"]["a-1"] == pytest.approx(.7, abs=2e-6)
    restored = WorkflowRiskBN.restore(model.parameters())
    assert restored.predict(samples[0])["probabilities"] == result["probabilities"]


@pytest.mark.parametrize("mutation", ["future_cue", "wrong_actor", "changed_args", "outcome_input", "cycle", "future_limit", "missing_history", "wrong_currency", "wrong_scope", "wrong_combination", "expiry"])
def test_bad_prefix_or_policy_inputs_rejected(mutation):
    sample = fixture(combination=True)
    p, c, w, combos = map(copy.deepcopy, (sample.proposals, sample.contracts, sample.workflow, sample.combinations))
    changes = dict(proposals=p, contracts=c, workflow=w, combinations=combos)
    if mutation == "future_cue": p[0]["evidence"]["sequence"] = 21
    if mutation == "wrong_actor": w["actors"]["step-0"] = ["someone-else"]
    if mutation == "changed_args": p[0]["binding"]["arguments"]["amount"] = "61"
    if mutation == "outcome_input": p[0]["outcome"] = 1
    if mutation == "cycle": w["edges"].append(["end", "plan"])
    if mutation == "future_limit": p[0]["quantity"]["limit_sequence"] = 21
    if mutation == "missing_history": c[0]["history_complete"] = False
    if mutation == "wrong_currency": c[0]["currency"] = "unknown"
    if mutation == "wrong_scope": c[0]["task_id"] = "other-task"
    if mutation == "wrong_combination": combos[0]["bindings"]["a-0"] = "wrong-digest"
    if mutation == "expiry": changes["horizon"] = 101
    with pytest.raises(ValueError): rebuild(sample, **changes)


@pytest.mark.parametrize("kind", ["proposals", "order", "capacity", "contradiction", "future_label", "source"])
def test_mutation_capacity_and_label_boundaries(kind):
    sample, model = fixture(authority="known"), WorkflowRiskBN()
    with pytest.raises(ValueError):
        if kind == "proposals":
            sample.proposals[0]["visible"] = "unknown"
            model.predict(sample)
        elif kind == "order":
            sample.action_ids = tuple(reversed(sample.action_ids))
            model.predict(sample)
        elif kind == "capacity": WorkflowRiskBN(max_binary_variables=1).predict(sample)
        elif kind == "contradiction": model.loss_gradient(model.theta, [sample], [{"dispatch": {"a-0": 0, "a-1": 0}, "risks": {"system:any": 1}}])
        elif kind == "future_label": model.loss_gradient(model.theta, [sample], [{"dispatch": {"not-proposed": 1}}])
        elif kind == "source":
            spec = copy.deepcopy(model.registry.specification)
            spec["fragments"][0]["source_ids"] = [999]
            WorkflowFragmentRegistry(spec)
