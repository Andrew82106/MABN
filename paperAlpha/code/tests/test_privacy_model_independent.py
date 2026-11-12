import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.privacy_flow_data import MODELS, batch
from paperalpha_monitor.privacy_model_independent import (from_legacy, make_packet, compile_packets,
    graph_features, ModelIndependentPrivacyBN)
from paperalpha_monitor.privacy_stage_bn import PrivacyStageBN
from paperalpha_monitor.privacy_supervision_controls import outcome_matrix
from test_privacy_flow_bn import example
from test_privacy_supervision_controls import cases


def test_identity_is_removed_not_renamed():
    initial, old = example()
    projected = from_legacy(old)
    assert "model" not in projected
    for identity in MODELS:
        changed = copy.deepcopy(old)
        changed["model"] = identity
        assert projected == from_legacy(changed)
    raw = make_packet(initial, old["vertical"], "CANARY_SSN_AAAA Alice Example", "123456 Alice Example")
    assert raw == projected


@pytest.mark.parametrize("coupled", [False, True])
def test_pruned_parameter_gradient(coupled):
    old, labels = cases(2)
    packets = list(map(from_legacy, old))
    data = compile_packets(packets)
    f = outcome_matrix(data, labels)
    model = ModelIndependentPrivacyBN(coupled=coupled, seed=76)
    assert len(model.theta) == (39 if coupled else 38)
    assert not any(":model:" in name for name in model.names)
    fn = lambda t: model.loss_gradient(t, data, f)[0]
    np.testing.assert_allclose(model.loss_gradient(model.theta, data, f)[1], approx_derivative(fn, model.theta).ravel(), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("stage", [1, 2])
def test_equivalent_to_zero_identity_coefficients_in_full_model(stage):
    old, _ = cases(stage)
    full = PrivacyStageBN(seed=87)
    neutral = ModelIndependentPrivacyBN(seed=87)
    for i, name in enumerate(full.names):
        if ":model:" in name:
            full.theta[i] = 0.
    a = full.predict_details(old)
    b = neutral.predict_details(list(map(from_legacy, old)))
    np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-13)
    np.testing.assert_allclose(a["state_weights"], b["state_weights"], atol=1e-13)


@pytest.mark.parametrize("field", ["model", "label", "final_text", "trace_id"])
def test_identity_free_input_whitelist(field):
    packet = from_legacy(example()[1])
    packet[field] = "not an input"
    with pytest.raises(ValueError, match="whitelist"):
        ModelIndependentPrivacyBN().predict_details([packet])


def test_fit_restore_and_same_input_baselines():
    old, labels = cases(2)
    packets = list(map(from_legacy, old))
    model = ModelIndependentPrivacyBN().fit(packets, labels)
    assert model.optimizer["success"]
    assert graph_features(packets).shape[1] == 22
    np.testing.assert_array_equal(model.predict_details(packets)["probabilities"],
        ModelIndependentPrivacyBN.restore(model.parameters()).predict_details(packets)["probabilities"])
