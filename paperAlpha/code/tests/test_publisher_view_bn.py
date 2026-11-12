import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from publisher_view_fixtures import fixture
from scalar_binary_graph import evaluate
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN, PublisherInput, ViewSeparatedRelease
from paperalpha_monitor.policy_gated_privacy import PolicyGatedPrivacyBN, ScopedRelease


def model():
    result = PublisherViewPrivacyBN(regularization=.003)
    result.theta += np.random.default_rng(13).normal(0, .15, len(result.theta))
    result.theta[-2:] = [1.7, -1.3]
    return result


@pytest.mark.parametrize("stage,visible", [(1, "missing"), (2, "supported"), (2, "contradicted"), (2, "unresolved"), (2, "missing")])
@pytest.mark.parametrize("current", ["valid", "revoked", "root_unknown", "delegation_unknown", "opaque"])
def test_executable_graph(stage, current, visible):
    sample = fixture(stage=stage, current=current, visible=visible)
    bn = model()
    for k in (1, 2):
        graph = bn.probability_graph(sample, at_least=k)
        assert bn.predict(sample, at_least=k)["model_probability"] == pytest.approx(evaluate(graph), abs=1e-12)
        for node in graph["nodes"]:
            if node["id"].startswith("F:"):
                assert not any(p.startswith("A:") for p in node["parents"])


def test_monitor_truth_does_not_change_emission_when_visible_input_fixed():
    bn = model()
    samples = [fixture(current=c, visible="supported") for c in ("valid", "revoked", "root_unknown")]
    results = [bn.predict(s) for s in samples]
    for r in results[1:]:
        assert r["appearance_marginals"] == pytest.approx(results[0]["appearance_marginals"], abs=1e-14)
    assert results[0]["model_probability"] == 0 < results[2]["model_probability"] < results[1]["model_probability"]
    assert not results[0]["publisher_input_is_internal_belief"]


def test_cache_changes_emission_but_not_reference_truth():
    bn = model()
    supported, denied = [bn.predict(fixture(current="revoked", visible=v)) for v in ("supported", "contradicted")]
    assert supported["reference_failure_marginals"] == denied["reference_failure_marginals"] == [1., 1.]
    assert supported["model_probability"] > denied["model_probability"]
    assert all(a > b for a, b in zip(supported["appearance_marginals"], denied["appearance_marginals"]))


def test_missing_is_not_contradicted_and_no_implicit_fallback():
    missing = fixture(visible="missing")
    assert np.array_equal(missing.visible_features, np.zeros((2, 2)))
    assert not model().predict(missing)["publisher_input_available"]
    old = ScopedRelease(missing.packet, missing.graph, missing.context, missing.bindings)
    with pytest.raises(ValueError):
        model().predict(old)
    with pytest.raises(ValueError):
        fixture(stage=1, visible="supported")


def test_reference_scope_and_prefix_isolation():
    sample = fixture()
    source = sample.publisher_input
    with pytest.raises(ValueError):
        ViewSeparatedRelease(sample, replace(source, captured_journal_sequence=21), trusted_transport_observers=["constructed-input"])
    with pytest.raises(ValueError):
        ViewSeparatedRelease(sample, source, trusted_transport_observers=["another-observer"])
    for field in ("task_id", "release_context", "release_bindings"):
        payload = copy.deepcopy(source.payload)
        payload[field] = "changed" if field == "task_id" else {} if field == "release_context" else []
        changed = PublisherInput.capture(payload, transport_observer="constructed-input", captured_journal_sequence=20, prefix_journal_sequence=20)
        with pytest.raises(ValueError):
            ViewSeparatedRelease(sample, changed, trusted_transport_observers=["constructed-input"])


def test_visible_input_mutation_future_record_and_outcome_rejected():
    for mutation in ("hash", "future", "label"):
        sample = fixture()
        payload = copy.deepcopy(sample.publisher_input.payload)
        if mutation == "hash":
            payload["user_request"] = "changed"
            source = replace(sample.publisher_input, payload=payload)
        else:
            if mutation == "future":
                payload["cached_authority_records"][-1]["sequence"] = 15
            else:
                payload["final_outcome"] = 1
            source = PublisherInput.capture(payload, transport_observer="constructed-input", captured_journal_sequence=20, prefix_journal_sequence=20)
        with pytest.raises(ValueError):
            ViewSeparatedRelease(sample, source, trusted_transport_observers=["constructed-input"])
    sample = fixture()
    sample.visible_features[0] = [0, 1]
    with pytest.raises(ValueError):
        model().predict(sample)


def test_zero_visibility_coefficients_recover_old_zero_beta_model():
    bn = model()
    bn.theta[-2:] = 0
    old = PolicyGatedPrivacyBN(bn.presence, regularization=bn.regularization)
    old.theta[:-1] = bn.theta[:-2]
    old.theta[-1] = 0
    for current in ("valid", "revoked", "root_unknown", "delegation_unknown", "opaque"):
        sample = fixture(current=current)
        for k in (1, 2):
            assert bn.predict(sample, at_least=k)["model_probability"] == pytest.approx(old.predict(sample, at_least=k)["model_probability"], abs=1e-15)


def test_all_44_parameter_gradients_and_partial_labels():
    bn = model()
    assert len(bn.theta) == 44
    samples = [fixture(current=c, visible=v, w=w) for c, v, w in (
        ("root_unknown", "supported", (1, 0)), ("delegation_unknown", "contradicted", (0, 1)),
        ("opaque", "missing", (1, 1)), ("revoked", "unresolved", (0, 0)))]
    f, a, y = [[1, 0], [0, 1], [1, 1], [0, 1]], [[1, 1], [None, 0], [None, None], [1, 1]], [1, 0, 1, 1]
    fun = lambda t: bn.loss_gradient(t, samples, f, a, violation_labels=y, weights=[1, 2, 3, 4])
    loss, grad = fun(bn.theta)
    numerical = approx_derivative(lambda t: fun(t)[0], bn.theta, method="3-point").ravel()
    assert np.isfinite(loss) and np.max(abs(numerical - grad)) < 2e-7
    assert all(abs(grad[i]) > 1e-8 for i in range(bn.n, bn.n + 5))


def test_no_authority_labels_no_cue_truth_leak_into_emission_likelihood():
    bn = model()
    bn.regularization = 0
    sample = fixture(current="root_unknown", visible="supported")
    grad = bn.loss_gradient(bn.theta, [sample], [[1, 0]], [[None, None]])[1]
    assert np.max(abs(grad[bn.n:bn.n + 3])) < 1e-12


def test_joint_optimization_restore_and_partial_fit_metadata():
    bn = model()
    samples = [fixture(current="revoked", visible=v, w=w) for v, w in (("supported", (1, 1)), ("contradicted", (1, 0)), ("unresolved", (0, 1)), ("missing", (0, 0)))]
    f, a = [[1, 1], [0, 0], [1, 0], [0, 1]], [[1, 1]] * 4
    initial = bn.loss_gradient(bn.theta, samples, f, a)[0]
    bn.fit(samples, f, a, max_iter=400)
    assert bn.optimizer["success"] and bn.optimizer["loss"] < initial
    assert bn.predict(samples[0])["jointly_trained"]
    restored = PublisherViewPrivacyBN.restore(json.loads(json.dumps(bn.parameters())))
    assert restored.predict(samples[0]) == bn.predict(samples[0])
    with pytest.raises(ValueError):
        PolicyGatedPrivacyBN.restore(bn.parameters())
    partial = model().fit(samples, f, a, trainable_names=list(PublisherViewPrivacyBN.EMISSION_NAMES))
    assert partial.predict(samples[0])["optimized_parameter_count"] == 2
    assert not partial.predict(samples[0])["jointly_trained"]
