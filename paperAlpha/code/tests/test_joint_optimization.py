import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from continue_joint_optimization import continue_fit, projected_gradient


class Quadratic:
    def __init__(self, theta, success=True):
        self.theta = np.asarray(theta, float)
        self.optimizer = {"success": success}
        self.max_iter = 1

    def loss_gradient(self, theta, traces, labels, prefix):
        assert traces == ["fit-only"] and labels == [1] and prefix
        return float(np.sum(theta ** 2)), 2 * theta


CONFIG = {"bounds": [-8., 8.], "stationarity_tolerance": 1e-5,
          "max_additional_iterations": 30, "ftol": 1e-12, "gtol": 1e-6}


def test_projected_gradient_preserves_feasible_descent_at_bounds():
    np.testing.assert_equal(projected_gradient([-8, -8, 8, 8, 0], [1, -1, -1, 1, 2]), [0, -1, 0, 1, 2])
    with pytest.raises(ValueError):
        projected_gradient([9], [1])


def test_success_flag_alone_does_not_prove_stationarity():
    model = Quadratic([3., -2.])
    result = continue_fit(model, ["fit-only"], [1], CONFIG)
    assert result["selected"] and result["accepted"] and result["after"]["stationarity_pass"]
    assert result["after"]["fit_objective"] < result["before"]["fit_objective"]


def test_stationary_success_is_kept_unchanged():
    model = Quadratic([0., 0.])
    before = copy.deepcopy(model.__dict__)
    result = continue_fit(model, ["fit-only"], [1], CONFIG)
    assert not result["selected"] and not result["accepted"]
    np.testing.assert_equal(model.theta, before["theta"])
    assert model.optimizer == before["optimizer"] and model.max_iter == before["max_iter"]
