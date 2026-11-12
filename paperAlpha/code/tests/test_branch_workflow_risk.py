import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from branch_workflow_fixtures import fixture
from branch_graph_evidence import evidence_probability
from scalar_binary_graph import evaluate
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.branch_workflow_risk import BranchWorkflowRiskBN, BranchWorkflowInput
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN


def model():
    branch = BranchPublisherPrivacyBN(regularization=0.)
    branch.theta += np.random.default_rng(409).normal(0, .15, len(branch.theta))
    branch.theta[branch.presence.boost_start:branch.presence.boost_start + 2] = [2., 3.]
    branch.theta[branch.n - 1] = 1.2
    result = BranchWorkflowRiskBN(branch_model=branch, regularization=0.)
    result.theta[result.branch_count + 1] = logit(.4)
    return result


@pytest.mark.parametrize("current", ["valid", "revoked", "root_unknown", "delegation_unknown", "opaque"])
@pytest.mark.parametrize("visible", ["supported", "contradicted", "unresolved"])
def test_join_preserves_original_authorization_prediction(current, visible):
    m, sample = model(), fixture(current=current, visible=visible, prohibit=False)
    p = m.predict(sample)
    original = m.branch_model.predict(sample.branch, theta=m.theta[:m.branch_count])
    assert p["probabilities"]["system:any"] == pytest.approx(original["model_probability"], abs=3e-13)
    assert list(p["appearance_probabilities"].values()) == pytest.approx(original["appearance_marginals"], abs=3e-13)
    assert p["dispatch_probabilities"] == {}
    assert p["branch_reference"]["prefix_journal_sequence"] == 20 > sample.cutoff == 12


def test_joint_interface_is_not_product_of_marginals():
    m, sample = model(), fixture(current="valid")
    law = m.branch_model.joint_appearance_law(sample.branch, theta=m.theta[:m.branch_count])
    result = m.predict(sample)
    both = float(law["probabilities"][-1])
    marginals = list(result["appearance_probabilities"].values())
    assert abs(both - np.prod(marginals)) > .005
    assert result["probabilities"]["prohibited_combination:literal-pair"] == pytest.approx(both)
    assert result["probabilities"]["system:any"] == pytest.approx(both)


@pytest.mark.parametrize("payment", [False, True])
@pytest.mark.parametrize("current", ["valid", "root_unknown", "opaque"])
def test_composed_graph_all_queries_and_conditional_likelihood(payment, current):
    m, sample = model(), fixture(current=current, payment=payment)
    result = m.predict(sample)
    for key, expected in result["probabilities"].items():
        assert evaluate(m.probability_graph(sample, output=key)) == pytest.approx(expected, abs=2e-13)
    graph = m.probability_graph(sample)
    label = {"events": {sample.branch.release.action_ids[0]: 1}, "risks": {"channel:prohibited_combination": 0}}
    if payment:
        label["events"]["payment"] = 1
    evidence = {"event:" + k: v for k, v in label["events"].items()}
    evidence.update(label["risks"])
    chance = evidence_probability(graph, evidence) / evidence_probability(graph, {})
    loss, _ = m.loss_gradient(m.theta, [sample], [label])
    assert loss == pytest.approx(-np.log(chance), abs=3e-13)


def test_number_appearance_never_charged_as_payment():
    m, sample = model(), fixture(current="valid", payment=True)
    p = m.predict(sample)
    assert p["dispatch_probabilities"] == pytest.approx({"payment": .4})
    assert p["probabilities"]["channel:cumulative_budget"] == pytest.approx(.4)
    both = p["probabilities"]["channel:prohibited_combination"]
    assert p["probabilities"]["system:any"] == pytest.approx(.4 + .6 * both)
    budget = next(r for r in p["compiled_fragments"] if r["template"] == "cumulative_budget")
    assert budget["actions"] == ["payment"]
    assert sample.branch.release.packet["groups"][1]["kind"] == "number"


def test_all_composed_prediction_and_likelihood_derivatives():
    m = model()
    samples = [fixture(current=kind, payment=True, visible=visible) for kind, visible in
               (("root_unknown", "supported"), ("delegation_unknown", "contradicted"), ("opaque", "unresolved"))]
    labels = [{"events": {"release-0": 1, "release-1": 0, "payment": 0},
               "risks": {"authorization:release-0": 1}} for _ in samples]
    value, grad = m.loss_gradient(m.theta, samples, labels)
    numeric = approx_derivative(lambda t: m.loss_gradient(t, samples, labels)[0], m.theta).ravel()
    assert np.isfinite(value) and len(grad) == 54
    np.testing.assert_allclose(grad, numeric, atol=3e-7, rtol=2e-5)
    assert np.all(np.abs(grad[m._authority_indices()]) > 1e-5)
    assert abs(grad[m.branch_count + 1]) > .01
    # Optional release/other tool dispatch predictors have no examples here.
    np.testing.assert_allclose(grad[m.branch_count + 3:], 0., atol=1e-14)
    sample = samples[0]
    result = m.predict(sample, derivatives=True)
    numeric = approx_derivative(lambda t: m.predict(sample, theta=t)["probabilities"]["system:any"], m.theta).ravel()
    np.testing.assert_allclose(result["gradients"]["system:any"], numeric, atol=3e-8, rtol=2e-5)


def test_appearance_law_normalizer_gradients_and_authority_separation():
    m, sample = model(), fixture()
    branch = m.branch_model
    law = branch.joint_appearance_law(sample.branch)
    assert law["probabilities"].sum() == pytest.approx(1.)
    np.testing.assert_allclose(law["probabilities"] @ law["log_probability_gradients"], 0., atol=1e-14)
    np.testing.assert_allclose(law["log_probability_gradients"][:, branch.n:branch.n + 3], 0., atol=1e-14)
    numeric = approx_derivative(lambda t: branch.joint_appearance_law(sample.branch, theta=t)["log_probabilities"], branch.theta)
    np.testing.assert_allclose(law["log_probability_gradients"], numeric, atol=2e-8, rtol=3e-5)
    graph = branch.probability_graph(sample.branch)
    assert np.exp(law["log_worker_evidence"]) == pytest.approx(evidence_probability(graph, {}), abs=2e-14)
    for bits, p in zip(law["states"], law["probabilities"]):
        joint = evidence_probability(graph, {f"F:{i}": int(v) for i, v in enumerate(bits)})
        assert p == pytest.approx(joint / evidence_probability(graph, {}), abs=2e-13)


def test_checkpoints_do_not_claim_inherited_optimizer_or_duplicate_authority():
    m, sample = model(), fixture(payment=True)
    saved = m.parameters()
    assert saved["optimizer"] is None and saved["branch_definition"]["optimizer"] is None
    other = BranchWorkflowRiskBN.restore(saved)
    assert other.predict(sample)["probabilities"] == m.predict(sample)["probabilities"]
    graph = m.probability_graph(sample)
    assert sum(n["id"].startswith("fact:") for n in graph["nodes"]) == len(sample.authority.facts)
    assert not any(n["id"].startswith("appearance:fact:") for n in graph["nodes"])
    assert not any(any(p.startswith("failed:") for p in n["parents"]) for n in graph["nodes"] if n["id"].startswith("event:"))


def test_joint_fit_recovers_constructed_event_frequencies_not_hidden_parameters():
    teacher, sample = model(), fixture(payment=True)
    worlds = teacher._worlds(sample, teacher.theta)
    labels, weights = [], []
    for i, bits in enumerate(worlds["states"]):
        for j, facts in enumerate(worlds["fact_states"]):
            labels.append({"events": dict(zip(sample.action_ids, map(int, bits))),
                           "facts": dict(zip(worlds["fact_ids"], map(int, facts)))})
            weights.append(float(np.exp(worlds["log_mass"][i, j])))
    learner = model()
    learner.theta += np.random.default_rng(180).normal(0., .25, len(learner.theta))
    before = learner.predict(sample)["probabilities"]["system:any"]
    learner.fit([sample] * len(labels), labels, weights=weights, max_iter=300)
    assert learner.optimizer["success"]
    assert learner.optimizer["loss"] < learner.optimizer["initial_loss"] - .001
    expected = teacher.predict(sample)["probabilities"]["system:any"]
    assert abs(before - expected) > .005
    assert learner.predict(sample)["probabilities"]["system:any"] == pytest.approx(expected, abs=1e-5)
    # The full future-event distribution is the recovery target, not any one
    # nonidentifiable latent coefficient and not empirical deployment accuracy.
    np.testing.assert_allclose(np.exp(learner._worlds(sample, learner.theta)["log_mass"]), np.exp(worlds["log_mass"]), atol=1e-5)


@pytest.mark.parametrize("kind", ["wrong_predictor", "wrong_label", "worker_names", "pool_flag", "changed_input", "changed_clock", "changed_type", "capacity"])
def test_adapter_rejects_semantic_or_binding_errors(kind):
    m, sample = model(), fixture()
    with pytest.raises(ValueError):
        if kind == "wrong_predictor": WorkflowRiskBN().predict(sample)
        elif kind == "wrong_label": m.loss_gradient(m.theta, [sample], [{"dispatch": {"release-0": 1}}])
        elif kind == "worker_names":
            sample.branch.workers = ("forged-a", "forged-b")
            m.predict(sample)
        elif kind == "pool_flag":
            sample.branch.legacy_pool_agrees = not sample.branch.legacy_pool_agrees
            m.predict(sample)
        elif kind == "changed_input":
            sample.branch.observation["publisher_payload"]["user_request"] += " altered"
            m.predict(sample)
        elif kind == "changed_clock":
            sample.proposals[0]["evidence"]["prefix_journal_sequence"] += 1
            m.predict(sample)
        elif kind == "changed_type":
            sample.event_semantics["release-0"] = "tool_dispatch"
            m.predict(sample)
        elif kind == "capacity": BranchWorkflowRiskBN(max_binary_variables=3).predict(sample)
