import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_same_information_strong_baselines_v1.py"
SPEC = importlib.util.spec_from_file_location("same_information_strong_baselines_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trace():
    return {
        "episode_id": "unit-1",
        "agents": ["a0", "a1"],
        "workflow_dag": {"nodes": ["a0", "a1"], "edges": [["a0", "a1"]]},
        "events": [
            {"actor": "a0", "event_type": "input_received", "data_class": "untrusted", "sequence": 0},
            {"actor": "a0", "target": "a1", "event_type": "message_sent", "data_class": "untrusted", "edge_kind": "workflow", "edge_confidence": 0.9, "sequence": 1},
            {"actor": "a1", "event_type": "tool_call", "tool": "payments.write", "data_class": "clean", "permission_mismatch": True, "sequence": 2},
        ],
    }


def test_public_union_contract_is_78_dimensions_and_label_blind():
    x, contract = MODULE.feature_matrix([_trace()])
    assert x.shape == (1, 78)
    assert sum(block["dim"] for block in contract["blocks"]) == 78
    assert not any(value in " ".join(contract["blocks"][0].keys()) for value in ("label", "hidden"))


def test_public_label_fields_are_rejected_before_feature_construction():
    bad = _trace()
    bad["label"] = 1
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.feature_matrix([bad])
    bad = _trace()
    bad["events"][0]["mechanism"] = "hidden"
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.feature_matrix([bad])


def test_fixed_model_hyperparameters_and_calibration_are_train_only():
    logistic = MODULE._make_model("logistic", 7)
    histgb = MODULE._make_model("histgb", 7)
    assert logistic[-1].get_params()["C"] == 1.0
    params = histgb.get_params()
    assert params["max_iter"] == 150
    assert params["max_leaf_nodes"] == 15
    assert params["max_depth"] == 3
    assert params["min_samples_leaf"] == 20
    # The calibration closure must not need test labels or test features beyond
    # the raw score vector it receives.
    apply, _ = MODULE._calibrate(np.asarray([.1, .9, .2, .8]), np.asarray([0, 1, 0, 1]))
    assert np.all((apply(np.asarray([.3, .7])) > 0) & (apply(np.asarray([.3, .7])) < 1))
