import copy
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from policy_privacy_fixtures import fixture
from paperalpha_monitor.policy_gated_privacy import PolicyGatedPrivacyBN, ScopedRelease, ReleaseBinding, count_distribution
from paperalpha_monitor.privacy_model_independent import ModelIndependentPrivacyBN, compile_packets
from scalar_binary_graph import evaluate


def scalar(model, sample, at_least=1):
    """Independently enumerate authority, U/V, W and final vectors."""
    packet, p = sample.packet, model.presence
    x = np.r_[1., compile_packets([packet])["context"][0]]
    t = model.theta
    _, _, authority_mass, _, failed = sample.graph.worlds(t[model.n:model.n + 3])
    total, positive = 0., 0.
    for a, mass_a in enumerate(authority_mass):
        for u, v in product((0, 1), repeat=2):
            pu = expit(x @ t[2 * p.d:2 * p.d + p.k])
            pv = expit(x @ t[2 * p.d + p.k:2 * p.d + 2 * p.k] + (u * t[model.n - 1] if p.coupled else 0.))
            mass_s = (pu if u else 1 - pu) * (pv if v else 1 - pv)
            for w in product((0, 1), repeat=len(packet["groups"])):
                if packet["stage"] == 2 and w != tuple(int(g["worker_present"]) for g in packet["groups"]):
                    continue
                mass_w, f_probs = 1., []
                for j, g in enumerate(packet["groups"]):
                    kind = np.eye(4)[("canary", "number", "structured", "text").index(g["kind"])]
                    q = expit(np.r_[kind, float(g["coordinator_present"]), x[1:]] @ t[:p.d] + u * t[p.boost_start])
                    mass_w *= q if w[j] else 1 - q
                    f_probs.append(expit(np.r_[kind, w[j], x[1:]] @ t[p.d:2 * p.d] + v * t[p.boost_start + 1]
                        + t[-1] * (not failed[sample.action_ids[j]][a])))
                for final in product((0, 1), repeat=len(f_probs)):
                    mass = mass_a * mass_s * mass_w * np.prod([q if f else 1 - q for q, f in zip(f_probs, final)])
                    total += mass
                    violations = sum(f and failed[sample.action_ids[j]][a] for j, f in enumerate(final))
                    positive += mass * (violations >= at_least)
    return positive / total


def constant_presence(p=.5, authority_invalidity=.2):
    model = PolicyGatedPrivacyBN(prior_logits=[logit(authority_invalidity), 0., 0.])
    model.theta[:model.n] = 0.
    model.theta[model.presence.d:model.presence.d + 4] = logit(p)
    return model


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("at_least", [1, 2, 3])
def test_full_joint_inference_matches_independent_enumeration(stage, shared, at_least):
    sample = fixture(shared=shared, stage=stage, w=(0, 1))
    model = PolicyGatedPrivacyBN(ModelIndependentPrivacyBN(seed=78), prior_logits=[-.7, -.2, .4])
    model.theta[-1] = 1.3
    result = model.predict(sample, at_least=at_least)
    assert result["model_probability"] == pytest.approx(scalar(model, sample, at_least), abs=1e-12)
    assert evaluate(model.probability_graph(sample, at_least=at_least)) == pytest.approx(result["model_probability"], abs=1e-12)
    assert sum(result["violation_count_probabilities"]) == pytest.approx(1.)


@pytest.mark.parametrize("stage", [1, 2])
def test_all_denied_reduces_to_original_presence_model(stage):
    sample = fixture(status="revoked", stage=stage)
    presence = ModelIndependentPrivacyBN(seed=19)
    model = PolicyGatedPrivacyBN(presence)
    assert model.predict(sample)["model_probability"] == pytest.approx(presence.predict_details([sample.packet])["probabilities"][0])


def test_authorized_sensitive_presence_is_not_policy_violation():
    model, sample = constant_presence(.8), fixture(status="valid")
    result = model.predict(sample)
    assert result["appearance_marginals"] == pytest.approx([.8, .8])
    assert result["model_probability"] == 0. and result["violation_marginals"] == [0., 0.]
    assert not result["proves_system_safety"]
    assert model.predict(fixture(status="recipient_mismatch"))["model_probability"] == pytest.approx(.96)


def test_shared_authority_preserved_not_product_of_field_marginals():
    model = constant_presence(.5, .2)
    a, b = model.predict(fixture(shared=True)), model.predict(fixture(shared=False))
    assert a["violation_marginals"] == pytest.approx([.1, .1])
    assert b["violation_marginals"] == pytest.approx(a["violation_marginals"])
    assert a["model_probability"] == pytest.approx(.15)
    assert b["model_probability"] == pytest.approx(.19)
    assert model.predict(fixture(shared=True), at_least=2)["model_probability"] == pytest.approx(.05)
    assert model.predict(fixture(shared=False), at_least=2)["model_probability"] == pytest.approx(.01)


def test_shared_emission_state_is_not_flattened_before_authority_composition():
    model = constant_presence(.05, .2)
    model.theta[model.presence.boost_start + 1] = 5.
    sample = fixture(shared=True)
    result = model.predict(sample)
    marginals = np.array(result["appearance_marginals"])
    wrong = .2 * (1 - np.prod(1 - marginals))
    assert abs(result["model_probability"] - wrong) > .02
    assert result["model_probability"] == pytest.approx(scalar(model, sample), abs=1e-12)


@pytest.mark.parametrize("partial", [False, True])
def test_all_joint_gradients_match_finite_difference(partial):
    model = PolicyGatedPrivacyBN(ModelIndependentPrivacyBN(seed=4), prior_logits=[-.9, -.4, .2], regularization=.002)
    model.theta[-1] = .8
    samples = [fixture(shared=False), fixture(status="valid"), fixture(status="mixed"), fixture(status="missing_root"), fixture(unknown_components="delegation")]
    f = [[1, 0], [1, 1], [0, 1], [1, 0], [0, 1]]
    a = [[1, None] if partial else [1, 0], [0, 0], [1, 0], [None, 1] if partial else [0, 1], [1, 1]]
    fn = lambda t: model.loss_gradient(t, samples, f, a)[0]
    _, analytic = model.loss_gradient(model.theta, samples, f, a)
    numerical = approx_derivative(fn, model.theta).ravel()
    assert len(model.theta) == 43
    np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-5)
    assert abs(analytic[-1]) > .01 and abs(analytic[model.n]) > .01
    assert np.all(abs(analytic[model.n:model.n + 3]) > .01)


def test_all_denied_joint_training_reduces_to_legacy_object_likelihood():
    samples = [fixture(status="revoked", w=(0, 1)), fixture(status="revoked", w=(1, 0))]
    presence = ModelIndependentPrivacyBN(seed=2, regularization=0.)
    model = PolicyGatedPrivacyBN(presence, regularization=0.)
    labels = np.array([[1, 0], [0, 1]])
    a = [[1, 1], [1, 1]]
    old_loss, old_gradient = presence.loss_gradient(presence.theta, compile_packets([s.packet for s in samples]), labels)
    loss, gradient = model.loss_gradient(model.theta, samples, labels, a)
    assert loss == pytest.approx(old_loss, abs=1e-12)
    np.testing.assert_allclose(gradient[:model.n], old_gradient, atol=1e-12)
    np.testing.assert_array_equal(gradient[model.n:], 0.)


def test_system_outcome_supervises_joint_structure_without_fake_authority_labels():
    model = PolicyGatedPrivacyBN(ModelIndependentPrivacyBN(seed=24), prior_logits=[-1., -.5, .4], regularization=0.)
    samples = [fixture(shared=True), fixture(shared=False)]
    f, a, y = [[1, 1], [0, 1]], [[None, None], [None, None]], [1, 0]
    fn = lambda t: model.loss_gradient(t, samples, f, a, violation_labels=y)[0]
    analytic = model.loss_gradient(model.theta, samples, f, a, violation_labels=y)[1]
    np.testing.assert_allclose(analytic, approx_derivative(fn, model.theta).ravel(), atol=1e-7, rtol=1e-5)
    assert abs(analytic[model.n]) > .01
    with pytest.raises(ValueError, match="contradict"):
        model.loss_gradient(model.theta, [fixture(status="valid")], [[1, 1]], [[None, None]], violation_labels=[1])


@pytest.mark.parametrize("field,value", [("recipient", "changed"), ("purpose", "changed"), ("task_id", "changed"), ("actor", "changed")])
def test_binding_cannot_be_reused_for_another_release(field, value):
    s = fixture()
    with pytest.raises(ValueError, match="binding"):
        ScopedRelease(s.packet, s.graph, replace(s.context, **{field: value}), s.bindings)


def test_future_proofs_and_partial_alias_bindings_are_rejected():
    s = fixture(aliases=True)
    with pytest.raises(ValueError, match="future"):
        ScopedRelease(s.packet, s.graph, replace(s.context, observed_through_sequence=10), s.bindings)
    with pytest.raises(ValueError, match="exactly"):
        ScopedRelease(s.packet, s.graph, s.context, [replace(s.bindings[0], field_names=("record_ref",)), s.bindings[1]])
    assert len(s.packet["groups"]) == 2 and sum(len(g["field_names"]) for g in s.packet["groups"]) == 3


def test_reference_label_contradictions_and_mutations_fail():
    model, sample = PolicyGatedPrivacyBN(), fixture(shared=True)
    with pytest.raises(ValueError, match="contradict"):
        model.loss_gradient(model.theta, [sample], [[1, 0]], [[1, 0]])
    sample.context = replace(sample.context, recipient="changed")
    with pytest.raises(ValueError, match="binding"):
        model.predict(sample)


def test_observation_input_contains_no_final_labels():
    s = fixture()
    changed = copy.deepcopy(s.packet)
    changed["final_labels"] = [0, 0]
    with pytest.raises(ValueError, match="whitelist"):
        ScopedRelease(changed, s.graph, s.context, s.bindings)


def test_count_distribution_exact_boundaries():
    np.testing.assert_array_equal(count_distribution([0, 1, 1]), [0, 0, 1, 0])
    with pytest.raises(ValueError):
        count_distribution([1.1])


def test_fit_and_restore_joint_model_without_touching_original():
    presence = ModelIndependentPrivacyBN(seed=1)
    before = presence.theta.copy()
    model = PolicyGatedPrivacyBN(presence, regularization=.01)
    samples = [fixture(status=s, w=w) for s, w in (("valid", (0, 1)), ("revoked", (1, 0)), ("uncertain", (1, 1)), ("mixed", (0, 0)))]
    f, a = [[1, 0], [0, 1], [1, 1], [1, 0]], [[0, 0], [1, 1], [1, 1], [1, 0]]
    model.fit(samples, f, a, max_iter=500)
    assert model.optimizer["success"] and model.optimizer["loss"] < model.optimizer["initial_loss"]
    assert len(model.optimizer["trainable_names"]) == 43
    assert np.count_nonzero(abs(model.theta[:model.n] - before) > 1e-5) > 10
    restored = PolicyGatedPrivacyBN.restore(model.parameters())
    assert restored.predict(samples[0]) == model.predict(samples[0])
    np.testing.assert_array_equal(before, presence.theta)


def test_dependency_free_canonical_preserves_old_binding_hashes():
    from paperalpha_runtime.canonical import canonical, digest
    from paperalpha_runtime.semantic_evidence import digest as old_digest
    from paperalpha_monitor.action_evidence import canonical as old_canonical
    values = [{"收件人": "内部审阅者", "arguments": {"a": 1, "z": None}, "data": [False, .125]}, {}, ["x", 1]]
    for value in values:
        assert canonical(value) == old_canonical(value)
        assert digest(value) == old_digest(value)
    with pytest.raises(ValueError):
        canonical(float("nan"))


def test_partial_fit_does_not_claim_all_components_were_jointly_optimized():
    model = constant_presence()
    samples = [fixture(status="valid"), fixture(status="revoked"), fixture()]
    model.fit(samples, [[1, 0], [0, 1], [1, 0]], [[0, 0], [1, 1], [1, 1]], trainable_names=model.names[-4:])
    result = model.predict(samples[0])
    assert result["joint_likelihood_fitted"] and not result["jointly_trained"]
    assert result["optimized_parameter_count"] == 4
