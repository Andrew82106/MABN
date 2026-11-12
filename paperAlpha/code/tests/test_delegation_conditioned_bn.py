import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.delegation_conditioned_bn import DelegationConditionedBN, DelegationEvidenceCompiler
from paperalpha_monitor.delegation_evidence import FEATURE_NAMES
from paperalpha_monitor.duty_conditioned_bn import DutyConditionedBN, DutyConditioner
from paperalpha_monitor.typed_action_supervision import TraceTargets, ActionTarget
from paperalpha_monitor.tool_capabilities import load_tool_registry
from paperalpha_runtime.duty_graph import public_packet
from test_duty_graph import configuration, annotation
from test_delegation_evidence import action, chain


def compiler():
    config = configuration()
    config["agents"][0]["role"], config["agents"][1]["role"] = "planner", "worker"
    packet, aliases = public_packet(config)
    return DelegationEvidenceCompiler(duties=DutyConditioner(packet, annotation(packet), aliases), task_id="task-a",
        root_principals=["user"], trusted_observers=["adapter"], coverage_complete=True,
        tool_registry=load_tool_registry(), roles={"planner": "reasoner", "worker": "executor"},
        workflow={"nodes": ["planner", "worker"], "edges": [["planner", "worker"]]})


def fixture(invalid=False):
    obj = compiler()
    events = chain()
    if invalid:
        events[0]["payload"]["can_delegate"] = False
    for event in events:
        obj.push(event)
    obj.push({"event_id": "msg", "sequence": 3, "actor": "planner", "event_type": "message_sent",
        "target": "worker", "content": "Proceed with the payment.", "data_class": "unknown"})
    obj.push({"event_id": "dispatch", "sequence": 10, "event_type": "tool_call", **action(), "grant_ids": ["child"]})
    return obj


@pytest.mark.parametrize("action_label,episode", [(True, None), (False, 1), (True, 0)])
def test_all_451_gradients_for_typed_and_endpoint_training(action_label, episode):
    trace, model = fixture(True).trace(), DelegationConditionedBN()
    assert len(model.theta) == 451
    theta = model.theta + np.random.default_rng(682).normal(0, .2, len(model.theta))
    targets = TraceTargets({"pay": ActionTarget(1, 0)} if action_label else {}, episode)
    _, analytic = model.loss_gradient(theta, [trace], [targets])
    numeric = approx_derivative(lambda t: model.loss_gradient(t, [trace], [targets])[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=5e-8, rtol=3e-5)
    assert np.any(np.abs(analytic[list(model.registry.relation_indices)]) > 1e-7)


def test_binding_metadata_is_unchanged_and_zero_relation_weights_collapse_pairs():
    normal, invalid = fixture(), fixture(True)
    a, b = normal.trace(), invalid.trace()
    assert a.steps[-1].base.actions == b.steps[-1].base.actions
    assert a.steps[-1].base.actions[0]["permission_status"] == "unknown"
    model = DelegationConditionedBN()
    assert model.predict_actions(a)["pay"]["forbidden"] == model.predict_actions(b)["pay"]["forbidden"]
    model.theta[model.registry.relation_indices[FEATURE_NAMES.index("delegation_not_allowed")]] = 2.
    assert model.predict_actions(b)["pay"]["forbidden"] > model.predict_actions(a)["pay"]["forbidden"]
    old, new = model.engine(), model.engine()
    for x, y in zip(a.steps, b.steps):
        old.apply(x, a.variables)
        new.apply(y, b.variables)
        np.testing.assert_allclose(old.distribution.sum(axis=1), new.distribution.sum(axis=1), atol=2e-15)


def test_live_push_equals_replay_and_future_records_do_not_rewrite_past():
    obj, model = fixture(), DelegationConditionedBN()
    engine = model.engine()
    values = [engine.apply(step, obj.trace().variables) for step in obj.steps]
    old = copy.deepcopy(obj.snapshots["pay"])
    obj.push({"event_id": "r", "sequence": 11, "kind": "revoke", "observer": "adapter",
        "authenticated_actor": "user", "payload": {"grant_id": "root"}})
    assert obj.snapshots["pay"] == old
    replay = model.engine()
    assert values == [replay.apply(step, obj.trace().variables) for step in obj.steps]
    with pytest.raises(ValueError, match="order"):
        obj.push({"sequence": 9})


def test_supervision_is_final_prefix_only_and_no_early_outcome_label():
    model, trace = DelegationConditionedBN(regularization=0), fixture(True).trace()
    target = TraceTargets({}, 1)
    value, gradient = model.trace_loss_gradient(model.theta, trace, target)
    engine = model.engine(derivatives=True)
    for step in trace.steps:
        engine.apply(step, trace.variables)
    assert value == pytest.approx(-np.log(engine.probability), abs=2e-15)
    np.testing.assert_allclose(gradient, -engine.gradient / engine.probability, atol=2e-15)


def test_endpoint_fit_learns_relations_and_keeps_tool_targets_untrained():
    model = DelegationConditionedBN(regularization=.002, max_iter=100)
    model.fit([fixture(False).trace(), fixture(True).trace()], [TraceTargets({}, 0), TraceTargets({}, 1)])
    assert model.optimizer["success"] and model.optimizer["loss"] < model.optimizer["initial_loss"]
    assert model.fitted_targets == ["episode_outcome"]
    assert model.predict_episode(fixture(True).trace())[-1] > model.predict_episode(fixture(False).trace())[-1]


def test_missing_records_get_explicit_qualification_not_a_safe_label():
    obj = compiler()
    obj.push({"event_id": "dispatch", "sequence": 10, "event_type": "tool_call", **action(), "grant_ids": ["missing"]})
    model = DelegationConditionedBN()
    with pytest.raises(ValueError, match="supervision"):
        model.delegation_reports(obj.trace())
    model.fitted_targets = ["episode_outcome"]  # Test reporting contract, not fitted accuracy.
    report = model.delegation_reports(obj.trace())[0]
    assert report["evidence_insufficient"] and report["conclusion"] == "insufficient_evidence"
    assert not report["proves_safety"] and not report["calibrated"]


def test_unrelated_invalid_alternative_cannot_raise_scoped_fragment_score():
    normal = fixture()
    altered = compiler()
    from test_delegation_evidence import grant
    for e in chain() + [grant(3, "irrelevant", "user", "another-actor")]:
        altered.push(e)
    altered.push({"event_id": "msg", "sequence": 4, "actor": "planner", "event_type": "message_sent",
        "target": "worker", "content": "Proceed with the payment.", "data_class": "unknown"})
    altered.push({"event_id": "dispatch", "sequence": 10, "event_type": "tool_call", **action(), "grant_ids": ["child", "irrelevant"]})
    model = DelegationConditionedBN()
    model.theta += np.random.default_rng(913).normal(0, .3, len(model.theta))
    assert model.predict_actions(normal.trace())["pay"]["forbidden"] == pytest.approx(
        model.predict_actions(altered.trace())["pay"]["forbidden"], abs=3e-15)
