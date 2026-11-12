"""Synthetic math checks only; no projection, API, real traces or DB oracle."""
from itertools import product
import copy
import json

import numpy as np
import pytest
from scipy.special import expit, softmax

from paperalpha_monitor.contract_forecast import ContractForecastBN, ContractForecastInput, CALL_TYPES


def fixture_model():
    model = ContractForecastBN(("bias", "native_support", "cross_role_evidence"))
    model.theta = np.random.default_rng(27).normal(0, .7, len(model.parameter_names))
    sample = ContractForecastInput({"bias": 1., "native_support": .6, "cross_role_evidence": 1.})
    return model, sample


def finite_difference(function, theta, epsilon=1e-6):
    result = np.empty_like(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta); step[i] = epsilon
        result[i] = (function(theta + step) - function(theta - step)) / (2 * epsilon)
    return result


def independent_same_factor_glm(model, sample):
    """Independently evaluate the public named GLM factors, not joint/_local."""
    coefficients = {row["name"]: row["coefficient"] for row in model.parameter_table()}
    probabilities = []
    def dot(prefix, features):
        return sum(coefficients[prefix + ":" + name] * value for name, value in features.items())
    for a, c, b, s in product((0, 1), CALL_TYPES, (0, 1), (0, 1)):
        if sample.current_a is not None:
            qa = float(a == sample.current_a)
        else:
            pa = expit(dot("A", sample.features)); qa = pa if a else 1-pa
        phi_c = {**sample.features, "parent:A": a}
        pc = softmax([*(dot("C=" + kind, phi_c) for kind in CALL_TYPES[:-1]), 0.])
        qc = pc[CALL_TYPES.index(c)]
        if c in {"OTHER", "STOP"}:
            qb, qs = float(b == 0), float(s == 0)
        else:
            phi_b = {**sample.features, "parent:A": a, "parent:C=TRIGGER": int(c == "TRIGGER")}
            pb = expit(dot("B", phi_b)); qb = pb if b else 1-pb
            phi_s = {**phi_b, "parent:B": b}
            ps = expit(dot("S", phi_s)); qs = ps if s else 1-ps
        probabilities.append(qa * qc * qb * qs)
    return np.asarray(probabilities)


@pytest.mark.parametrize("known", [None, 0, 1])
def test_normalization_support_clamping_and_wrong_object_retention(known):
    model, sample = fixture_model()
    sample = ContractForecastInput(sample.features, known)
    joint = model.joint(sample)
    assert len(joint["states"]) == 32
    assert joint["probabilities"].sum() == pytest.approx(1., abs=1e-14)
    assert np.isfinite(joint["scores"]).all()
    for (a, c, b, s), probability in zip(joint["states"], joint["probabilities"]):
        invalid = (known is not None and a != known) or (c in {"STOP", "OTHER"} and (b or s))
        assert (probability == 0) if invalid else (probability > 0)
    # Correct current A never erases future wrong-object successful triggers.
    assert any(p > 0 for (a, c, b, s), p in zip(joint["states"], joint["probabilities"])
               if c == "TRIGGER" and b == 0 and s == 1)
    assert np.max(np.abs(joint["probabilities"] @ joint["scores"])) < 1e-13
    if known is not None:
        a_columns = [i for i, n in enumerate(model.parameter_names) if n.startswith("A:")]
        assert np.max(np.abs(joint["scores"][:, a_columns])) < 1e-14


@pytest.mark.parametrize("known", [None, 0, 1])
def test_same_factor_glm_is_numerically_the_same_distribution(known):
    model, sample = fixture_model()
    sample = ContractForecastInput(sample.features, known)
    expected = independent_same_factor_glm(model, sample)
    np.testing.assert_allclose(model.joint(sample)["probabilities"], expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("labels", [
    {"A": 1, "C": "TRIGGER", "B": 0, "S": 1},
    {"C": "TRIGGER", "B": None, "S": 1},
    {"A": None, "S": 0}, {"C": "STOP"}, {},
])
def test_complete_and_partial_label_gradients_match_finite_differences(labels):
    model, sample = fixture_model()
    result = model.label_log_likelihood(sample, labels)
    numeric = finite_difference(lambda theta: model.label_log_likelihood(sample, labels, theta=theta)["log_probability"], model.theta)
    np.testing.assert_allclose(result["score"], numeric, atol=4e-9, rtol=3e-6)


def test_none_is_exact_marginalization_not_false_label():
    model, sample = fixture_model()
    joint = model.joint(sample)
    truth = sum(p for state, p in zip(joint["states"], joint["probabilities"]) if state[1] == "TRIGGER" and state[3] == 1)
    missing = model.label_log_likelihood(sample, {"C": "TRIGGER", "B": None, "S": 1})
    forced = model.label_log_likelihood(sample, {"C": "TRIGGER", "B": 0, "S": 1})
    assert np.exp(missing["log_probability"]) == pytest.approx(truth)
    assert missing["log_probability"] > forced["log_probability"]
    absent = model.label_log_likelihood(sample, {name: None for name in ("A", "B", "C", "S")})
    assert absent["log_probability"] == pytest.approx(0., abs=1e-14)
    np.testing.assert_allclose(absent["score"], 0., atol=1e-14)


def test_fixed_gate_risk_gradient_and_joint_explanation_are_exact():
    model, sample = fixture_model()
    result = model.predict(sample, explain=True, derivatives=True)
    worlds = result["world_contributions"]
    risk = sum(w["probability"] for w in worlds if w["C"] == "TRIGGER" and not(w["A"] and w["B"]) and w["S"])
    assert result["violation_probability"] == pytest.approx(risk)
    assert sum(w["violation_contribution"] for w in worlds) == pytest.approx(risk)
    for world in worlds:
        assert np.prod(list(world["cpd_probabilities"].values())) == pytest.approx(world["probability"], abs=1e-14)
    assert sum(result["joint_T_P_E"].values()) == pytest.approx(1.)
    def risk_at(theta):
        joint = model.joint(sample, theta=theta)
        return sum(p for (a, c, b, s), p in zip(joint["states"], joint["probabilities"])
                   if c == "TRIGGER" and not(a and b) and s)
    numeric = finite_difference(risk_at, model.theta)
    np.testing.assert_allclose(list(result["risk_gradient"].values()), numeric, atol=3e-9)
    assert not any("risk" in name.lower() for name in model.parameter_names)
    json.dumps(result, allow_nan=False)


def test_independence_control_preserves_marginals_and_structural_zeros():
    model, sample = fixture_model()
    original, control = model.predict(sample), model.independent_control(sample)
    assert sum(control["probabilities"]) == pytest.approx(1.)
    for variable in original["marginals"]:
        assert control["marginals"][variable] == pytest.approx(original["marginals"][variable])
    for (a, c, b, s), probability in zip(model.states, control["probabilities"]):
        if c in {"OTHER", "STOP"} and (b or s):
            assert probability == 0
    assert abs(control["violation_probability"] - original["violation_probability"]) > 1e-5
    assert "given_C" in control["control"] and control["trained_baseline"] is False


def test_metadata_never_changes_predictions_or_checkpoint():
    model, sample = fixture_model()
    other = ContractForecastInput(sample.features, metadata={"contract_id": "synthetic-contract", "task_id": "never-a-feature"})
    before = model.parameters()
    np.testing.assert_array_equal(model.joint(sample)["probabilities"], model.joint(other)["probabilities"])
    assert model.predict(other)["metadata"] == other.metadata
    assert model.parameters() == before and "never-a-feature" not in json.dumps(before)


def test_checkpoint_and_disk_roundtrip(tmp_path):
    model, sample = fixture_model()
    model.optimizer = {"success": True, "risk_head_fitted": False}
    saved = model.parameters()
    restored = ContractForecastBN.restore(saved)
    assert restored.parameters() == saved
    np.testing.assert_array_equal(model.joint(sample)["probabilities"], restored.joint(sample)["probabilities"])
    path = tmp_path / "forecast.json"; model.save(path)
    loaded = ContractForecastBN.load(path)
    assert loaded.predict(sample, explain=True) == model.predict(sample, explain=True)


@pytest.mark.parametrize("key", ["risk_gate", "support", "forecast_scope", "parameter_names", "state_semantics"])
def test_checkpoint_cannot_rewrite_fixed_policy_or_parameter_meaning(key):
    model, _ = fixture_model(); saved = model.parameters(); saved[key] = "changed"
    with pytest.raises(ValueError):
        ContractForecastBN.restore(saved)


@pytest.mark.parametrize("labels", [{"C": "STOP", "S": 1}, {"C": "OTHER", "B": 1},
                                   {"V": 1}, {"C": "TRADE"}, {"S": False}, {"A": 2}])
def test_invalid_or_impossible_supervision_rejected(labels):
    model, sample = fixture_model()
    with pytest.raises(ValueError):
        model.label_log_likelihood(sample, labels)


def test_labels_cannot_override_current_hard_evidence():
    model, sample = fixture_model()
    with pytest.raises(ValueError, match="hard evidence"):
        model.label_log_likelihood(ContractForecastInput(sample.features, 1), {"A": 0})


@pytest.mark.parametrize("names", [("task_id",), ("actor_id",), ("record_hash",), ("bias", "bias"), ()])
def test_identifier_and_unstable_feature_schemas_rejected(names):
    with pytest.raises(ValueError): ContractForecastBN(names)


@pytest.mark.parametrize("features", [{"bias": 0}, {"bias": float("nan")}, {"bias": "1"},
                                     {"bias": True}, {"bias": 1, "extra": 0}])
def test_bad_prefix_features_rejected(features):
    with pytest.raises(ValueError): ContractForecastBN().predict(ContractForecastInput(features))


def test_prediction_and_input_copy_do_not_mutate_features():
    values = {"bias": 1}; sample = ContractForecastInput(values)
    values["bias"] = 0
    original = copy.deepcopy(sample.features)
    ContractForecastBN().predict(sample, explain=True)
    assert sample.features == original == {"bias": 1}


def test_stop_scope_excludes_failed_empty_and_unsupported_api_replies():
    model = ContractForecastBN()
    scope = model.predict(ContractForecastInput({"bias": 1}))["scope"]
    assert "successful terminal supported nonempty text reply" in scope
    assert "API failure, empty or unsupported replies are unknown" in scope
    assert scope == model.parameters()["forecast_scope"]
