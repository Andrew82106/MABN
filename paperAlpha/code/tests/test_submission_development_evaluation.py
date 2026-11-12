import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    scripts = Path(__file__).resolve().parents[2] / 'code/commands'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("submission_development_evaluation_test", scripts / "evaluate_submission_development.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prediction():
    return {"targets": ["a", "b"], "union_probability": .76,
        "target_marginals": {"a": .2, "b": .7},
        "components": [{"targets": ["a"], "log_probabilities": np.log([.8, .2]).tolist()},
                       {"targets": ["b"], "log_probabilities": np.log([.3, .7]).tolist()}]}


def test_joint_brier_matches_explicit_four_world_distribution():
    m = module()
    value = m.score(prediction(), {"a": 0, "b": 1})
    probabilities = np.array([.24, .56, .06, .14])
    expected = np.square(probabilities-np.array([0., 1., 0., 0.])).sum()
    assert value["joint_brier"] == pytest.approx(expected)
    assert value["joint_log_loss_per_target"] == pytest.approx(-np.log(.56)/2)
    assert value["event_brier"] == pytest.approx((.2**2+.3**2)/2)
    assert not value["zero_probability_target_vector"]


def test_no_false_precision_when_target_vector_has_zero_mass():
    m = module(); p = prediction()
    p["components"][0]["log_probabilities"] = [0., -float("inf")]
    p["target_marginals"]["a"] = 0.
    p["union_probability"] = .7
    value = m.score(p, {"a": 1, "b": 1})
    assert value["zero_probability_target_vector"]
    assert value["joint_log_loss_per_target"] == pytest.approx(-np.log(1e-300)/2)
    assert all(np.isfinite(v) for v in value.values())


def test_no_api_or_native_guard_claims_in_methods():
    m = module()
    assert len(set(m.METHODS)) == 6
    assert {"matched_glm", "matched_histgb", "relational_bn"} <= set(m.METHODS)
    assert "blindguard" not in m.METHODS
