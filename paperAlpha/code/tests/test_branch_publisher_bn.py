import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from branch_publisher_fixtures import fixture
from branch_graph_evidence import evidence_probability
from scalar_binary_graph import evaluate
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput
from paperalpha_monitor.privacy_model_independent import ModelIndependentPrivacyBN
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN
from paperalpha_runtime.branched_publisher_data import joined_workers


def model(coupled=True):
    bn = BranchPublisherPrivacyBN(ModelIndependentPrivacyBN(coupled=coupled), regularization=.003)
    bn.theta += np.random.default_rng(23).normal(0., .15, len(bn.theta))
    bn.theta[bn.n + 3:] = [1.3, -1.7, -1.1]
    if coupled:
        bn.theta[bn.n - 1] = 2.1
    return bn


@pytest.mark.parametrize("coupled", [True, False])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("current", ["valid", "revoked", "root_unknown", "delegation_unknown", "opaque"])
@pytest.mark.parametrize("visible", ["supported", "contradicted", "unresolved"])
def test_executable_branch_graph(coupled, shared, current, visible):
    bn, sample = model(coupled), fixture(current=current, visible=visible, shared=shared)
    for k in (1, 2, 3):
        prediction = bn.predict(sample, at_least=k)
        graph = bn.probability_graph(sample, at_least=k)
        assert prediction["model_probability"] == pytest.approx(evaluate(graph), abs=2e-13)
        assert sum(prediction["violation_count_probabilities"]) == pytest.approx(1., abs=2e-14)
        assert len(prediction["stage_state_weights"]) == 8
        assert sum(node["id"].startswith("U:") for node in graph["nodes"]) == 2
        assert sum(node["id"].startswith("W:") for node in graph["nodes"]) == 4
        assert all(not any(p.startswith("A:") for p in node["parents"]) for node in graph["nodes"] if node["id"].startswith("F:"))


@pytest.mark.parametrize("coupled", [True, False])
def test_all_parameters_gradients_with_partial_authority_and_system_labels(coupled):
    bn = model(coupled)
    assert len(bn.theta) == (45 if coupled else 44)
    samples = [fixture(current=c, visible=v, shared=s, w=w) for c, v, s, w in (
        ("root_unknown", "supported", True, ((1, 0), (1, 0))),
        ("delegation_unknown", "contradicted", False, ((0, 1), (0, 1))),
        ("opaque", "unresolved", False, ((1, 1), (0, 0))),
        ("revoked", "supported", True, ((0, 0), (1, 1))))]
    f, a, y = [[1, 0], [0, 1], [1, 1], [0, 1]], [[1, 1], [None, 0], [None, None], [1, 1]], [1, 0, 1, 1]
    fun = lambda t: bn.loss_gradient(t, samples, f, a, weights=[1, 2, 3, 4], violation_labels=y)
    loss, grad = fun(bn.theta)
    numeric = approx_derivative(lambda t: fun(t)[0], bn.theta, method="3-point").ravel()
    assert np.isfinite(loss) and max(abs(grad - numeric)) < 2e-7
    assert all(abs(grad[i]) > 1e-8 for i in range(bn.n, bn.n + 6))


def test_matrix_arrangement_affects_model_even_when_pooled_values_identical():
    bn = model()
    bn.theta[bn.presence.boost_start] = 3.
    bn.theta[bn.presence.boost_start + 1] = 3.
    # Remove ownership and kind differences to isolate the shared-within-worker state.
    bn.theta[-1] = 0.
    separated = fixture(w=((1, 0), (0, 1)), kinds=("text", "text"), c=(0, 0))
    concentrated = fixture(w=((1, 0), (1, 0)), kinds=("text", "text"), c=(0, 0))
    assert separated.release.packet == concentrated.release.packet
    assert abs(bn.predict(separated)["model_probability"] - bn.predict(concentrated)["model_probability"]) > 1e-5
    # If the publisher is disconnected from worker states, branch arrangement
    # should not create a spurious difference in final appearance predictions.
    bn.theta[bn.n - 1] = 0.
    assert bn.predict(separated)["model_probability"] == pytest.approx(bn.predict(concentrated)["model_probability"], abs=1e-14)


def test_rename_workers_equivariance_and_no_identity_feature():
    bn, sample = model(), fixture(w=((1, 1), (0, 1)))
    raw = copy.deepcopy(sample.observation)
    raw["worker_messages"] = {"worker-a": sample.observation["worker_messages"]["worker-b"],
                              "worker-b": sample.observation["worker_messages"]["worker-a"]}
    raw["field_owners"] = {name: "worker-b" if value == "worker-a" else "worker-a" for name, value in raw["field_owners"].items()}
    raw["publisher_payload"]["earlier_agent_message"] = raw["pooled_scope"]["worker_text"] = joined_workers(raw["worker_messages"])
    other = BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)
    a, b = bn.predict(sample), bn.predict(other)
    assert a["violation_count_probabilities"] == pytest.approx(b["violation_count_probabilities"], abs=1e-14)
    assert a["worker_state_marginals"] == pytest.approx(b["worker_state_marginals"][::-1], abs=1e-14)


def test_true_authority_does_not_change_behavior_prediction():
    bn = model()
    results = [bn.predict(fixture(current=c)) for c in ("valid", "revoked", "root_unknown")]
    assert results[0]["model_probability"] == 0 < results[2]["model_probability"] < results[1]["model_probability"]
    assert all(r["appearance_marginals"] == pytest.approx(results[0]["appearance_marginals"], abs=1e-14) for r in results)
    bn.regularization = 0
    sample = fixture(current="root_unknown")
    _, grad = bn.loss_gradient(bn.theta, [sample], [[1, 0]], [[None, None]])
    assert max(abs(grad[bn.n:bn.n + 3])) < 1e-13


def test_shared_authority_not_replaced_by_independent_failure_marginals():
    bn = model()
    # Context/vertical differs between runtime fixtures: equalize it explicitly.
    shared, separate = fixture(current="root_unknown", shared=True), fixture(current="root_unknown", shared=False)
    raw = copy.deepcopy(separate.observation)
    raw["pooled_scope"]["vertical"] = shared.observation["pooled_scope"]["vertical"]
    separate = BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)
    a, b = bn.predict(shared), bn.predict(separate)
    assert a["reference_failure_marginals"] == pytest.approx(b["reference_failure_marginals"], abs=1e-14)
    assert a["appearance_marginals"] == pytest.approx(b["appearance_marginals"], abs=1e-14)
    assert a["model_probability"] < b["model_probability"]
    assert a["violation_count_probabilities"][2] > b["violation_count_probabilities"][2]


@pytest.mark.parametrize("current", ["revoked", "root_unknown", "delegation_unknown", "opaque"])
def test_likelihood_equals_scalar_joint_evidence_not_only_prediction(current):
    bn, sample = model(), fixture(current=current, shared=False, w=((1, 1), (0, 1)))
    f, a, y = [1, 0], [None, None], 1
    loss = bn._likelihood(sample, f, a, bn.theta, y, 1)[0]
    graph = bn.probability_graph(sample)
    evidence = {"F:0": 1, "F:1": 0, "system_violation": y}
    assert np.exp(-loss) == pytest.approx(evidence_probability(graph, evidence), rel=1e-12, abs=1e-15)


def test_fit_restore_and_trainable_subset():
    bn = model()
    samples = [fixture(visible=v, w=w) for v, w in (("supported", ((1, 0), (0, 1))),
        ("contradicted", ((0, 1), (1, 0))), ("unresolved", ((1, 1), (0, 0))), ("supported", ((0, 0), (1, 1))))]
    f, a = [[1, 1], [0, 0], [1, 0], [0, 1]], [[1, 1]] * 4
    before = bn.loss_gradient(bn.theta, samples, f, a)[0]
    bn.fit(samples, f, a, max_iter=400)
    assert bn.optimizer["success"] and bn.optimizer["loss"] < before
    assert bn.predict(samples[0])["jointly_trained"]
    saved = json.loads(json.dumps(bn.parameters()))
    assert BranchPublisherPrivacyBN.restore(saved).predict(samples[0]) == bn.predict(samples[0])
    with pytest.raises(ValueError):
        PublisherViewPrivacyBN.restore(saved)
    partial = model()
    original = partial.theta.copy()
    partial.fit(samples, f, a, trainable_names=list(partial.EMISSION_NAMES), max_iter=400)
    assert np.array_equal(original[:-3], partial.theta[:-3])
    assert partial.predict(samples[0])["optimized_parameter_count"] == 3
    assert not partial.predict(samples[0])["jointly_trained"]


@pytest.mark.parametrize("bad", ["pooled", "mutated", "future", "labels", "theta", "threshold"])
def test_invalid_inputs_rejected(bad):
    bn, sample = model(), fixture()
    with pytest.raises(ValueError):
        if bad == "pooled":
            bn.predict(sample.release)
        elif bad == "mutated":
            sample.worker_presence[0, 0] = False
            bn.predict(sample)
        elif bad == "future":
            sample.available_journal_sequence = 21
            bn.predict(sample)
        elif bad == "labels":
            bn.loss_gradient(bn.theta, [sample], [[1, 1]], [[0, 0]])
        elif bad == "theta":
            bn.predict(sample, theta=np.zeros(44))
        else:
            bn.predict(sample, at_least=0)
