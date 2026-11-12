"""Handcrafted same-information nonlinear-factor checks; no real outcomes."""
import copy
import json

import numpy as np
import pytest

from paperalpha_monitor.contract_forecast import ContractForecastBN, ContractForecastInput, CALL_TYPES
from paperalpha_monitor.contract_forecast_baselines import ContractFactorHistGB


CONFIG = {"max_iter": 15, "max_leaf_nodes": 3, "min_samples_leaf": 1,
          "learning_rate": .2, "random_state": 23}


def example():
    def x(v, a=None): return ContractForecastInput({"bias": 1., "native_support": v}, a)
    traces = [[x(0), x(.1)], [x(1), x(.9)], [x(.2, 0)], [x(.8, 1)]]
    labels = [[{"A": 0, "C": "TRIGGER", "B": 0, "S": 1}, {"C": "STOP"}],
              [{"A": 1, "C": "UPDATE", "B": 1, "S": 1}, {"C": "TRIGGER", "B": 1, "S": 0}],
              [{"C": "TRIGGER", "B": None, "S": 1}], [{"C": "OTHER"}]]
    return traces, labels


@pytest.mark.parametrize("a", [None, 0, 1])
def test_initial_prior_normalized_same_support_clamp_and_fixed_gate(a):
    model = ContractFactorHistGB(tree_config=CONFIG)
    sample = ContractForecastInput({"bias": 1}, a)
    result = model.joint(sample)
    np.testing.assert_allclose(result["probabilities"], ContractForecastBN().joint(sample)["probabilities"], atol=1e-15)
    assert result["probabilities"].sum() == pytest.approx(1)
    assert model.predict(sample)["violation_probability"] == pytest.approx(sum(
        p for (aa, c, b, s), p in zip(model.states, result["probabilities"])
        if c == "TRIGGER" and not(aa and b) and s))


def test_partial_labels_form_exact_posterior_not_zero_imputation():
    model = ContractFactorHistGB(tree_config=CONFIG)
    sample = ContractForecastInput({"bias": 1})
    posterior = model.label_log_likelihood(sample, {"C": "TRIGGER", "B": None, "S": 1})["posterior"]
    assert posterior.sum() == pytest.approx(1)
    assert sum(p for state, p in zip(model.states, posterior) if state[2] == 0) == pytest.approx(.5)
    assert sum(p for state, p in zip(model.states, posterior) if state[2] == 1) == pytest.approx(.5)
    assert sum(p for state, p in zip(model.states, posterior) if state[1] != "TRIGGER" or state[3] != 1) == 0


def test_training_uses_same_inputs_labels_and_records_actual_composite_rounds():
    traces, labels = example(); original = copy.deepcopy((traces, labels))
    model = ContractFactorHistGB(("bias", "native_support"), tree_config=CONFIG, max_rounds=3).fit(traces, labels)
    report = model.fit_report
    assert (traces, labels) == original
    assert report["final_composite_loss"] <= report["initial_composite_loss"] + 1e-12
    assert report["rounds"] and len(report["rounds"]) <= 3
    assert report["trace_count"] == 4 and report["next_call_windows"] == 6
    assert report["global_em_optimum_claimed"] is False
    assert report["boosting_m_step_guaranteed_monotone"] is False
    assert report["risk_head_fitted"] is report["test_labels_used"] is False
    assert report["posterior_training"]["window_posterior_mass"] == pytest.approx([1]*6)
    for trace in traces:
        for sample in trace:
            result = model.predict(sample, explain=True)
            assert sum(result["joint_T_P_E"].values()) == pytest.approx(1)
            assert sum(w["violation_contribution"] for w in result["world_contributions"]) == pytest.approx(result["violation_probability"])
    json.dumps(model.metadata(), allow_nan=False)


def test_known_A_never_trains_the_A_prior_or_breaks_prediction_clamp():
    sample = ContractForecastInput({"bias": 1}, 1)
    model = ContractFactorHistGB(tree_config=CONFIG, max_rounds=1).fit([[sample]], [[{"C": "TRIGGER", "B": 0, "S": 1}]])
    assert model.fit_report["head_summary"]["A"]["reason"] == "no_training_rows"
    assert model.predict(sample)["marginals"]["A"] == pytest.approx({"0": 0., "1": 1.})
    # Clamped A=1 still permits a wrong-object successful trigger.
    assert model.predict(sample)["violation_probability"] > 0


def test_structural_STOP_B_S_are_not_fake_training_rows():
    sample = ContractForecastInput({"bias": 1})
    model = ContractFactorHistGB(tree_config=CONFIG, max_rounds=1).fit([[sample]], [[{"A": 1, "C": "STOP", "B": 0, "S": 0}]])
    summary = model.fit_report["head_summary"]
    assert summary["B"]["reason"] == summary["S"]["reason"] == "no_training_rows"
    assert summary["C"]["reason"] == "single_posterior_class"
    assert all(v > 0 for v in summary["C"]["prior"])


def test_soft_rows_are_trace_normalized_not_world_counted():
    samples = [[ContractForecastInput({"bias": 1})], [ContractForecastInput({"bias": 1})]]
    labels = [[{"C": "TRIGGER", "S": 1}], [{"C": "UPDATE", "S": 0}]]
    model = ContractFactorHistGB(tree_config=CONFIG)
    rows, report = model._posterior_rows(samples, labels)
    for factor in ("A", "C", "B", "S"):
        assert sum(rows[factor].values()) == pytest.approx(1.)
    assert report["window_posterior_mass"] == pytest.approx([1., 1.])


def test_worse_m_step_is_recorded_and_previous_training_candidate_retained(monkeypatch):
    sample = ContractForecastInput({"bias": 1})
    model = ContractFactorHistGB(tree_config=CONFIG, max_rounds=2)
    before = model.joint(sample)["probabilities"].copy()
    bad = copy.deepcopy(model.heads)
    bad["C"]["prior"] = np.asarray([.001, .001, .001, .997])
    monkeypatch.setattr(model, "_fit_heads", lambda rows: bad)
    model.fit([[sample]], [[{"C": "TRIGGER", "S": 1}]])
    row = model.fit_report["rounds"][0]
    assert row["accepted"] is False
    assert row["candidate_composite_loss"] > row["loss_before"]
    assert model.fit_report["retained_initial_prior"] is True
    assert model.fit_report["accepted_rounds"] == 0
    np.testing.assert_array_equal(model.joint(sample)["probabilities"], before)


def test_checkpoint_parity_and_no_case_metadata_retained(tmp_path):
    traces, labels = example()
    traces[0][0].metadata["task_id"] = "synthetic-only-no-checkpoint"
    model = ContractFactorHistGB(("bias", "native_support"), tree_config=CONFIG, max_rounds=2).fit(traces, labels)
    path = tmp_path / "factor.joblib"; model.save(path)
    restored = ContractFactorHistGB.load(path)
    assert restored.metadata() == model.metadata()
    for sample in traces[0]:
        assert restored.predict(sample, explain=True) == model.predict(sample, explain=True)
    assert "synthetic-only-no-checkpoint" not in json.dumps(model.metadata())


def test_future_labels_remain_a_separate_training_argument():
    sample = ContractForecastInput({"bias": 1})
    model = ContractFactorHistGB(tree_config=CONFIG)
    before = model.predict(sample)
    labels = {"C": "TRIGGER", "S": 1}
    model.label_log_likelihood(sample, labels)
    labels["C"] = "STOP"
    assert model.predict(sample) == before
    with pytest.raises(TypeError): model.predict(sample, labels=labels)


def test_public_21_feature_projection_schema_is_accepted_without_ID_features():
    from paperalpha_monitor.contract_forecast_projection import FEATURE_NAMES
    assert len(FEATURE_NAMES) == 21
    values = dict.fromkeys(FEATURE_NAMES, 0.)
    values["bias"], values["state_unknown"] = 1., 1.
    sample = ContractForecastInput(values, metadata={"contract_id": "synthetic"})
    model = ContractFactorHistGB(FEATURE_NAMES, tree_config=CONFIG)
    assert model.predict(sample)["feature_names"] == list(FEATURE_NAMES)
    assert sum(model.joint(sample)["probabilities"]) == pytest.approx(1.)


@pytest.mark.parametrize("labels", [{}, {"A": None, "C": None, "B": None, "S": None}])
def test_no_informative_supervision_is_not_reported_as_training(labels):
    model = ContractFactorHistGB(tree_config=CONFIG)
    with pytest.raises(ValueError, match="informative"):
        model.fit([[ContractForecastInput({"bias": 1})]], [[labels]])


@pytest.mark.parametrize("labels", [{"C": "STOP", "S": 1}, {"V": 1}, {"B": False}])
def test_impossible_and_unsupported_labels_rejected(labels):
    with pytest.raises(ValueError):
        ContractFactorHistGB(tree_config=CONFIG).label_log_likelihood(ContractForecastInput({"bias": 1}), labels)


@pytest.mark.parametrize("options", [{"tree_config": {"early_stopping": True}},
    {"tree_config": {"random_state": None}}, {"max_rounds": 0},
    {"prior_pseudocount": 0}, {"prior_mix": 0}, {"prior_mix": 1}])
def test_explicit_prior_and_training_configuration(options):
    with pytest.raises(ValueError): ContractFactorHistGB(**options)
