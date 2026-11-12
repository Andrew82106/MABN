import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, compile_knowledge_journal
from paperalpha_monitor.typed_action_supervision import (TypedActionSupervisionBN, ActionTarget, TraceTargets,
    HIGH_NAMES, LOW_NAMES, LOW_CELLS, local_features, bce)
from test_scoped_budget_bn import scenario, context


def example():
    return compile_knowledge_journal(scenario(), **context())


@pytest.mark.parametrize("episode,graph", [(None, True), (1, True), (0, False)])
def test_full_typed_loss_gradient_matches_finite_difference(episode, graph):
    model = TypedActionSupervisionBN(use_graph=graph)
    theta = model.theta + np.random.default_rng(811).normal(0, .4, len(model.theta))
    labels = TraceTargets({"first": ActionTarget(0, 1), "second": ActionTarget(1, 0)}, episode)
    loss, analytic = model.loss_gradient(theta, [example()], [labels])
    numeric = approx_derivative(lambda t: model.loss_gradient(t, [example()], [labels])[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=5e-8, rtol=3e-5)
    assert math.isfinite(loss)


def test_observation_heads_do_not_change_v7_filter_or_system_head():
    trace = example()
    model, old = TypedActionSupervisionBN(), KnowledgeScopedBN()
    model.theta += np.random.default_rng(119).normal(0, .6, len(model.theta))
    old.theta = model.theta[:model.base_count].copy()
    a, b = model.engine(derivatives=True), old._engine(old.theta, derivatives=True)
    for step in trace.steps:
        pa, pb = a.apply(step, trace.variables), b.apply(step, trace.variables)
        assert pa == pytest.approx(pb, abs=3e-15)
        np.testing.assert_array_equal(a.distribution, b.distribution)
        np.testing.assert_allclose(a.gradient[:model.base_count], b.gradient, atol=3e-15)
        np.testing.assert_array_equal(a.gradient[model.base_count:], 0.)


def test_independent_scalar_emission_enumeration_at_each_dispatch():
    trace, model = example(), TypedActionSupervisionBN()
    model.theta += np.random.default_rng(270).normal(0, .5, len(model.theta))
    engine = model.engine()
    # Capture the pre-factor state. Independently assemble scalar conditional
    # logits and integrate, without using production head matrices or latent_features.
    original, expected = engine._factor, {}
    def capture(factor, bits):
        if factor.template_id == "authorization":
            step, theta, n = engine.current_step, model.theta, model.base_count
            action = next(a for a in step.base.actions if a["action_id"] == factor.action_id)
            cell = LOW_CELLS.index((action["capability_profile"]["capability"], action["role"]))
            q = 1 / (1 + math.exp(-sum(theta[i] * v for i, v in zip(factor.parameter_indices, factor.values))))
            high, low = 0., 0.
            for state in range(len(engine.distribution)):
                own = (state >> factor.node) & 1
                nodes = step.base.observed_nodes
                count = sum((state >> i) & 1 for i in nodes)
                process = [own, count / max(1, len(nodes)), count * (count - 1) / max(1, len(nodes) * (len(nodes) - 1))]
                for ns in (step.base.capable_nodes, step.base.terminal_nodes):
                    process.append(sum((state >> i) & 1 for i in ns) / max(1, len(ns)))
                for h in range(engine.history_width):
                    prior = (h // (3 ** (len(model.registry.channels) - 1))) % 3
                    mass = engine.distribution[state, h]
                    for r, weight in ((0, 1 - q), (1, q)):
                        x = [*step.base.observables, *process, r, own * r, int(prior > 0), int(prior == 2)]
                        z = sum(v * w for v, w in zip(x, theta[n:n + len(HIGH_NAMES)]))
                        high += mass * weight / (1 + math.exp(-z))
                    z = theta[n + len(HIGH_NAMES) + cell]
                    z += sum(v * w for v, w in zip([*step.base.observables[1:], *process], theta[n + len(HIGH_NAMES) + len(LOW_CELLS):]))
                    low += mass / (1 + math.exp(-z))
            expected[factor.action_id] = (high, low)
        original(factor, bits)
    engine._factor = capture
    for step in trace.steps:
        engine.apply(step, trace.variables)
    for key, (high, low) in expected.items():
        row = engine.action_predictions[key]
        assert row["forbidden"] == pytest.approx(high, abs=3e-15)
        assert row["unnecessary_given_not_forbidden"] == pytest.approx(low, abs=3e-15)
        assert sum(row[k] for k in ("forbidden", "unnecessary", "no_tool_tier_violation")) == pytest.approx(1.)


def test_conditional_necessity_loss_excludes_authorization_parameters():
    model, trace = TypedActionSupervisionBN(), example()
    model.theta += np.random.default_rng(911).normal(0, .5, len(model.theta))
    engine = model.engine(derivatives=True)
    for step in trace.steps:
        engine.apply(step, trace.variables)
    authorization = [i for i, name in enumerate(model.parameter_names) if name.startswith("authorization:")]
    for _, low_gradient in engine.action_gradients.values():
        np.testing.assert_allclose(low_gradient[authorization], 0., atol=5e-16)
    assert np.any(np.abs(engine.action_gradients["second"][0][authorization]) > 1e-6)


def test_action_only_supervision_cannot_train_or_expose_global_outcome_head():
    model = TypedActionSupervisionBN(regularization=0)
    theta = model.theta + np.random.default_rng(651).normal(0, .5, len(model.theta))
    _, grad = model.trace_loss_gradient(theta, example(), TraceTargets({"first": ActionTarget(0, 1)}))
    heads = [i for i, k in enumerate(model.parameter_names[:model.base_count]) if k.startswith("head:")]
    np.testing.assert_array_equal(grad[heads], 0.)
    with pytest.raises(ValueError, match="untrained episode"):
        model.predict_episode(example())


def test_future_outcomes_and_results_do_not_change_earlier_action_score():
    model = TypedActionSupervisionBN()
    model.theta += np.random.default_rng(6).normal(0, .4, len(model.theta))
    rows = scenario()
    prefix = compile_knowledge_journal(rows[:3], **context())
    assert model.predict_actions(prefix)["first"] == model.predict_actions(example())["first"]
    rows[3]["payload"]["result"] = "Everything failed; this text arrives AFTER first dispatch."
    changed = compile_knowledge_journal(rows, **context())
    assert model.predict_actions(changed)["first"] == model.predict_actions(prefix)["first"]


def test_training_reduces_typed_objective_without_changing_untrained_head():
    model = TypedActionSupervisionBN(max_iter=80)
    start = model.theta.copy()
    labels = TraceTargets({"first": ActionTarget(0, 1), "second": ActionTarget(1, 0)})
    model.fit([example()], [labels])
    assert model.optimizer["loss"] < model.optimizer["initial_loss"] - .01
    assert "episode_outcome" not in model.fitted_targets
    heads = [i for i, k in enumerate(model.parameter_names[:model.base_count]) if k.startswith("head:")]
    np.testing.assert_array_equal(model.theta[heads], start[heads])


@pytest.mark.parametrize("forbidden,unnecessary", [(1, 1), (None, 1), (False, 0), (0, -1), (0, .5)])
def test_invalid_or_conflated_targets_rejected(forbidden, unnecessary):
    with pytest.raises(ValueError):
        ActionTarget(forbidden, unnecessary)


def test_unknown_or_future_action_id_cannot_be_supervised():
    model = TypedActionSupervisionBN()
    with pytest.raises(ValueError, match="observed dispatch"):
        model.trace_loss_gradient(model.theta, example(), TraceTargets({"future": ActionTarget(0, 0)}))


def test_missing_labels_not_manufactured_into_negative_supervision():
    model = TypedActionSupervisionBN()
    a = TraceTargets({"first": ActionTarget(1, 0)})
    b = TraceTargets({"first": ActionTarget(1, None), "second": ActionTarget(None, None)})
    va, ga = model.trace_loss_gradient(model.theta, example(), a)
    vb, gb = model.trace_loss_gradient(model.theta, example(), b)
    assert va == vb
    np.testing.assert_array_equal(ga, gb)


def test_permission_vocabulary_preserves_unknown_as_distinct_observation():
    action = example().steps[-1].base.actions[0]
    assert len(local_features(action)) == len(LOW_CELLS) + 5
