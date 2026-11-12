from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.privacy_flow_data import batch, context, KINDS
from paperalpha_monitor.privacy_stage_bn import PrivacyStageBN
from paperalpha_monitor.privacy_supervision_controls import outcome_matrix
from test_privacy_supervision_controls import cases


@pytest.mark.parametrize("coupled", [False, True])
def test_stage_parameters_full_gradient(coupled):
    packets, labels = cases(2)
    data = batch(packets)
    f = outcome_matrix(data, labels)
    model = PrivacyStageBN(coupled=coupled, seed=14)
    fn = lambda t: model.loss_gradient(t, data, f)[0]
    np.testing.assert_allclose(model.loss_gradient(model.theta, data, f)[1],
        approx_derivative(fn, model.theta).ravel(), rtol=1e-5, atol=1e-7)


def scalar(model, packet):
    x = np.r_[1., context(packet)]
    t, d, k = model.theta, model.d, model.k
    pu = expit(x @ t[2 * d:2 * d + k])
    boosts = t[model.boost_start:model.boost_start + 2]
    mass_all, mass_positive = 0., 0.
    for u, v in product((0, 1), repeat=2):
        pv = expit(x @ t[2 * d + k:2 * d + 2 * k] + (u * t[-1] if model.coupled else 0.))
        mass = (pu if u else 1 - pu) * (pv if v else 1 - pv)
        # Full W/F vector enumeration, distinct from production marginalization.
        for ws in product((0, 1), repeat=len(packet["groups"])):
            if packet["stage"] == 2 and ws != tuple(int(g["worker_present"]) for g in packet["groups"]):
                continue
            wm, fs = mass, []
            for j, field in enumerate(packet["groups"]):
                kind = np.eye(len(KINDS))[KINDS.index(field["kind"])]
                q = expit(np.r_[kind, float(field["coordinator_present"]), x[1:]] @ t[:d] + u * boosts[0])
                wm *= q if ws[j] else 1 - q
                fs.append(expit(np.r_[kind, ws[j], x[1:]] @ t[d:2 * d] + v * boosts[1]))
            for final in product((0, 1), repeat=len(fs)):
                probability = wm * np.prod([p if y else 1 - p for y, p in zip(final, fs)])
                mass_all += probability
                mass_positive += probability * any(final)
    return mass_positive / mass_all


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("coupled", [False, True])
def test_exact_stage_inference(stage, coupled):
    packets, _ = cases(stage)
    model = PrivacyStageBN(coupled=coupled, seed=19)
    actual = model.predict_details(packets)
    for i, packet in enumerate(packets):
        assert actual["probabilities"][i] == pytest.approx(scalar(model, packet), abs=1e-12)
    np.testing.assert_allclose(actual["state_weights"].sum(axis=1), 1., atol=1e-12)


def test_zero_coupling_reduces_to_independent_stage_states():
    a, b = PrivacyStageBN(coupled=True, seed=34), PrivacyStageBN(coupled=False)
    a.theta[-1] = 0.
    b.theta[:] = a.theta[:-1]
    for stage in (1, 2):
        packets, _ = cases(stage)
        np.testing.assert_allclose(a.predict_details(packets)["probabilities"], b.predict_details(packets)["probabilities"], atol=1e-13)


def test_negative_coupling_is_supported_without_negative_local_emissions():
    model = PrivacyStageBN(coupled=True)
    model.theta[-1] = -3.
    params = model.parameters()
    assert params["state_link_log_odds"] < 0.
    assert all(v > 0 for v in params["local_boosts"])
    packets, _ = cases(2)
    np.testing.assert_array_equal(model.predict_details(packets)["probabilities"],
        PrivacyStageBN.restore(params).predict_details(packets)["probabilities"])


def test_fitting_lowers_joint_objective():
    packets, labels = cases(2)
    model = PrivacyStageBN().fit(packets, labels)
    assert model.optimizer["success"]
    assert model.optimizer["loss"] < model.optimizer["initial_loss"]
