import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_runtime.duty_graph import public_packet, DUTY_FEATURE_NAMES
from paperalpha_monitor.duty_conditioned_bn import DutyConditionedBN, DutyConditioner
from paperalpha_monitor.knowledge_scoped_bn import compile_knowledge_journal
from paperalpha_monitor.typed_action_supervision import TypedActionSupervisionBN, TraceTargets, ActionTarget
from test_scoped_budget_bn import scenario, context
from test_duty_graph import configuration, annotation


def fixture():
    config = configuration()
    config["agents"][0]["role"], config["agents"][1]["role"] = "a", "b"
    packet, aliases = public_packet(config)
    conditioner = DutyConditioner(packet, annotation(packet), aliases)
    trace = compile_knowledge_journal(scenario(), **context())
    return trace, conditioner


@pytest.mark.parametrize("episode", [None, 1])
def test_all_438_joint_training_gradients_include_local_duty_parents(episode):
    original, conditioner = fixture()
    model, trace = DutyConditionedBN(), conditioner.trace(original)
    assert len(model.theta) == 438
    theta = model.theta + np.random.default_rng(22).normal(0, .3, len(model.theta))
    labels = TraceTargets({"first": ActionTarget(1, 0), "second": ActionTarget(0, 1)}, episode)
    _, analytic = model.loss_gradient(theta, [trace], [labels])
    numeric = approx_derivative(lambda t: model.loss_gradient(t, [trace], [labels])[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, rtol=3e-5, atol=5e-8)
    assert np.any(np.abs(analytic[list(model.registry.duty_indices)]) > 1e-5)


def test_zero_duty_weights_recover_v8_exactly():
    original, conditioner = fixture()
    old, new = TypedActionSupervisionBN(), DutyConditionedBN()
    old.theta += np.random.default_rng(67).normal(0, .5, len(old.theta))
    new.import_v8(old)
    a, b = old.predict_actions(original), new.predict_actions(conditioner.trace(original))
    for aid in a:
        for k in ("forbidden", "unnecessary", "current_authorization_hypothesis"):
            assert a[aid][k] == pytest.approx(b[aid][k], abs=2e-15)


def test_duty_changes_local_factor_without_inventing_permission_or_changing_process():
    original, conditioner = fixture()
    trace, model = conditioner.trace(original), DutyConditionedBN()
    old = model.engine()
    model.theta[model.registry.duty_indices[0]] = 2.
    new = model.engine()
    for step in trace.steps:
        old.apply(step, trace.variables)
        new.apply(step, trace.variables)
        np.testing.assert_allclose(old.distribution.sum(axis=1), new.distribution.sum(axis=1), atol=3e-15, rtol=0)
    assert new.action_predictions["first"]["current_authorization_hypothesis"] > old.action_predictions["first"]["current_authorization_hypothesis"]
    assert all(a["permission_status"] == "unknown" for a in trace.steps[-1].base.actions)
    assert not np.allclose(old.distribution, new.distribution)


def test_necessity_conditional_gradient_does_not_train_duty_authorization_parents():
    original, conditioner = fixture()
    model, trace = DutyConditionedBN(), conditioner.trace(original)
    model.theta += np.random.default_rng(923).normal(0, .5, len(model.theta))
    engine = model.engine(derivatives=True)
    for step in trace.steps:
        engine.apply(step, trace.variables)
    for _, gradient in engine.action_gradients.values():
        np.testing.assert_allclose(gradient[list(model.registry.duty_indices)], 0., atol=3e-15)


def test_base_factors_are_not_mutated_and_mismatched_registries_rejected():
    original, conditioner = fixture()
    n = len(original.steps[1].mechanism_factors[0].parameter_indices)
    new = conditioner.trace(original)
    assert len(original.steps[1].mechanism_factors[0].parameter_indices) == n
    assert len(new.steps[1].mechanism_factors[0].parameter_indices) == n + len(DUTY_FEATURE_NAMES)
    with pytest.raises(ValueError, match="unchanged base"):
        conditioner.trace(new)


def test_training_can_learn_declared_duties_and_preserves_untrained_global_head():
    original, conditioner = fixture()
    model, trace = DutyConditionedBN(max_iter=70), conditioner.trace(original)
    model.fit([trace], [TraceTargets({"first": ActionTarget(1, 0), "second": ActionTarget(0, 0)})])
    assert model.optimizer["loss"] < model.optimizer["initial_loss"]
    assert np.any(np.abs(model.theta[list(model.registry.duty_indices)]) > 1e-6)
    with pytest.raises(ValueError, match="untrained episode"):
        model.predict_episode(trace)
