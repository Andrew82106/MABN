"""Conditional scenario queries are not predictions or causal interventions."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from workflow_risk_fixtures import fixture as action_fixture
from branch_workflow_fixtures import fixture as branch_fixture
from scalar_binary_graph import evaluate
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN
from paperalpha_monitor.branch_workflow_risk import BranchWorkflowRiskBN


def branch_model():
    model = BranchWorkflowRiskBN(regularization=0.)
    model.theta += np.random.default_rng(919).normal(0., .2, len(model.theta))
    start = model.branch_model.presence.boost_start
    model.theta[start:start + 2] = [2., 3.]
    model.theta[model.branch_model.n - 1] = 1.3
    return model


@pytest.mark.parametrize("authority", ["known", "invalid", "root", "delegation", "opaque"])
@pytest.mark.parametrize("conditions", [{"a-0": 1}, {"a-0": 0, "a-1": 1}, {"a-0": 1, "a-1": 1}])
def test_action_conditional_queries_match_exported_graph(authority, conditions):
    sample = action_fixture(authority=authority, combination=True)
    model = WorkflowRiskBN()
    model.theta = np.random.default_rng(18).normal(0., .5, len(model.theta))
    result = model.predict(sample, event_conditions=conditions)
    assert result["query_mode"] == "conditional_event_scenario"
    for action, value in conditions.items():
        assert result["dispatch_probabilities"][action] == pytest.approx(value, abs=1e-14)
    for query, probability in result["probabilities"].items():
        graph = model.probability_graph(sample, output=query, event_conditions=conditions)
        assert evaluate(graph) == pytest.approx(probability, abs=2e-13)


def test_forecast_and_current_action_assessment_are_not_the_same_target():
    sample = action_fixture(authority="invalid", budget=None)
    model = WorkflowRiskBN()
    before = model.predict(sample)
    after = model.predict(sample, event_conditions={"a-0": 1, "a-1": 0}, derivatives=True)
    assert before["probabilities"]["system:any"] == pytest.approx(.75)
    assert after["probabilities"]["system:any"] == pytest.approx(1.)
    np.testing.assert_allclose(after["gradients"]["system:any"], 0., atol=2e-14)
    # Neither the model nor the observed prefix is changed by a scenario query.
    assert before == model.predict(sample)


@pytest.mark.parametrize("conditions", [{"release-0": 1}, {"release-1": 0, "payment": 1}, {"release-0": 1, "release-1": 1, "payment": 0}])
def test_conditional_branch_graph_and_full_gradient(conditions):
    sample = branch_fixture(current="root_unknown", payment=True)
    model = branch_model()
    result = model.predict(sample, event_conditions=conditions, derivatives=True)
    for query, probability in result["probabilities"].items():
        graph = model.probability_graph(sample, output=query, event_conditions=conditions)
        assert evaluate(graph) == pytest.approx(probability, abs=3e-13)
    numeric = approx_derivative(lambda theta: model.predict(sample, theta=theta, event_conditions=conditions)["probabilities"]["system:any"], model.theta).ravel()
    np.testing.assert_allclose(result["gradients"]["system:any"], numeric, atol=5e-8, rtol=3e-5)


def test_conditioning_updates_other_field_but_is_not_do_intervention():
    sample = branch_fixture(current="valid", payment=False)
    model = branch_model()
    before = model.predict(sample)
    after = model.predict(sample, event_conditions={"release-0": 1})
    assert after["appearance_probabilities"]["release-1"] > before["appearance_probabilities"]["release-1"] + .005
    graph = model.probability_graph(sample, event_conditions={"release-0": 1})
    old_graph = model.probability_graph(sample)
    node = next(n for n in graph["nodes"] if n["id"] == "event:release-0")
    old_node = next(n for n in old_graph["nodes"] if n["id"] == "event:release-0")
    assert node["parents"] == old_node["parents"]
    assert node["probability_one"] == old_node["probability_one"]
    assert node["observed"] == 1


def test_empty_conditions_preserve_old_output_and_graph():
    model, sample = branch_model(), branch_fixture(payment=True)
    saved = copy.deepcopy(sample.specification())
    assert model.predict(sample) == model.predict(sample, event_conditions={})
    assert model.probability_graph(sample) == model.probability_graph(sample, event_conditions={})
    assert "query_mode" not in model.predict(sample)
    assert sample.specification() == saved


@pytest.mark.parametrize("conditions", [{"system:any": 1}, {"a-0": True}, {"a-0": .5}, {"a-0": 1.}, {"a-0": 2}, [1, 0]])
def test_outcomes_truthy_values_and_non_events_cannot_be_conditioned(conditions):
    model, sample = WorkflowRiskBN(), action_fixture()
    with pytest.raises(ValueError):
        model.predict(sample, event_conditions=conditions)
    with pytest.raises(ValueError):
        model.probability_graph(sample, event_conditions=conditions)
