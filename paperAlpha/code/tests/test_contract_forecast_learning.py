"""Fabricated within-window joint scores, grouped composite fit and partial labels."""
import numpy as np
import pytest

from paperalpha_monitor.contract_forecast import ContractForecastBN, ContractForecastInput
from paperalpha_monitor.contract_forecast_learning import contract_trace_loss_gradient, fit_contract_traces


def examples():
    def x(value, known=None):
        return ContractForecastInput({"bias": 1., "native_support": value}, known)
    traces = [[x(0.), x(.2)], [x(1.), x(.8)], [x(.3, 0)], [x(.7, 1)]]
    targets = [[{"A": 0, "C": "TRIGGER", "B": 0, "S": 1}, {"C": "STOP", "S": None}],
               [{"A": 1, "C": "UPDATE", "B": 1, "S": 1}, {"C": "TRIGGER", "B": 1, "S": 0}],
               [{"C": "TRIGGER", "B": None, "S": 1}], [{"C": "OTHER", "B": None, "S": None}]]
    return traces, targets


def test_trace_objective_gradient_and_trace_normalization():
    traces, targets = examples(); model = ContractForecastBN(("bias", "native_support"))
    theta = np.random.default_rng(11).normal(0, .3, len(model.parameter_names))
    value, gradient = contract_trace_loss_gradient(model, theta, traces, targets, .02)
    expected = -sum(model.label_log_likelihood(x, y, theta=theta)["log_probability"]
                    for trace, labels in zip(traces, targets) for x, y in zip(trace, labels)) / len(traces) + .02*(theta@theta)
    assert value == pytest.approx(expected)
    numeric = np.empty_like(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta); step[i] = 1e-6
        plus = contract_trace_loss_gradient(model, theta+step, traces, targets, .02)[0]
        minus = contract_trace_loss_gradient(model, theta-step, traces, targets, .02)[0]
        numeric[i] = (plus-minus)/2e-6
    np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-5)


def test_fit_improves_objective_handles_partial_supervision_and_restores():
    traces, targets = examples(); names = ("bias", "native_support")
    original = ContractForecastBN(names)
    before = contract_trace_loss_gradient(original, original.theta, traces, targets, .03)[0]
    fitted = fit_contract_traces(traces, targets, feature_names=names, regularization=.03)
    assert fitted.optimizer["success"] is True
    assert fitted.optimizer["loss"] < before - .1
    assert fitted.optimizer["risk_head_fitted"] is False
    assert "composite negative log scores" in fitted.optimizer["objective"]
    assert "not a generative sequence joint likelihood" in fitted.optimizer["objective"]
    assert fitted.optimizer["A_reestimated_each_window"] is True
    assert fitted.optimizer["cross_window_state_model"] is False
    assert fitted.optimizer["trace_count"] == 4 and fitted.optimizer["next_call_factors"] == 6
    assert np.isfinite(fitted.theta).all()
    restored = ContractForecastBN.restore(fitted.parameters())
    assert fitted.predict(traces[0][0], explain=True) == restored.predict(traces[0][0], explain=True)


def test_fully_unlabelled_training_is_not_claimed_as_learning():
    traces, targets = examples()
    targets = [[{"A": None, "C": None, "B": None, "S": None} for _ in trace] for trace in traces]
    with pytest.raises(ValueError, match="observed event label"):
        fit_contract_traces(traces, targets, feature_names=("bias", "native_support"))


def test_repeating_established_A_as_a_label_is_not_new_training_information():
    with pytest.raises(ValueError, match="informative"):
        fit_contract_traces([[ContractForecastInput({"bias": 1.}, current_a=1)]], [[{"A": 1}]])


@pytest.mark.parametrize("penalty", [-1., float("inf"), float("nan")])
def test_invalid_penalty_rejected(penalty):
    traces, targets = examples(); model = ContractForecastBN(("bias", "native_support"))
    with pytest.raises(ValueError): contract_trace_loss_gradient(model, model.theta, traces, targets, penalty)


def test_misaligned_or_empty_traces_rejected():
    with pytest.raises(ValueError): fit_contract_traces([], [])
    with pytest.raises(ValueError): fit_contract_traces([[]], [[]])
    with pytest.raises(ValueError): fit_contract_traces([[ContractForecastInput({"bias": 1})]], [[]])
