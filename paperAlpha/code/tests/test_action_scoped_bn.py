import copy
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.action_evidence import ActionEventCompiler, compile_action_trace
from paperalpha_monitor.action_joint_bn import ActionScopedJointBN, ActionRiskStream, PARAMETER_NAMES
from paperalpha_monitor.dynamic_joint_bn import HEAD, CONFIDENCE, OBSERVABLES, KNOWLEDGE


def action(sequence, kind="tool_proposed", identity="x", **changes):
    return {"sequence": sequence, "event_id": f"event-{sequence}", "event_type": kind, "action_id": identity,
            "actor": "executor", "principal": "user-1", "tool": "shell.exec", "resource": "workspace-1",
            "arguments": {"command": "inspect"}, **changes}


def check(sequence, status="mismatch", revision=0, **changes):
    values = {"checker_id": "policy-service", "permission_status": status, "check_revision": revision,
              "valid_until_sequence": 20, **changes}
    return action(sequence, "authorization_checked", **values)


def events():
    return [
        {"sequence": 0, "event_type": "input_received", "actor": "reader", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 1, "event_type": "message_sent", "actor": "reader", "target": "executor", "data_class": "untrusted", "edge_confidence": .8},
        action(2), check(3, "mismatch"), check(4, "no_mismatch", 1, valid_until_sequence=5),
        {"sequence": 6, "event_type": "message_sent", "actor": "reader", "target": "executor", "data_class": "clean"},
        check(7, "mismatch", 2), action(8, "tool_call"),
        {"sequence": 9, "event_type": "input_received", "actor": "executor", "source_id": "late", "data_class": "untrusted"},
        action(10, identity="y", arguments={"command": "inspect-other"}),
        action(11, "action_cancelled", identity="y", arguments={"command": "inspect-other"}),
    ]


def context():
    return {"trusted_checkers": ["policy-service"], "roles": {"reader": "reader", "executor": "executor"}}


def auth(stream, identity="x"):
    return next(a for a in stream.explain()["actions"] if a["action_id"] == identity)


def test_later_check_revises_same_action_instead_of_adding_an_independent_risk():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    initial = auth(stream)["authorization_risk_hypothesis"]
    stream.push(check(1))
    bad = auth(stream)["authorization_risk_hypothesis"]
    stream.push(check(2, "no_mismatch", 1))
    good = auth(stream)["authorization_risk_hypothesis"]
    assert good < initial < bad
    assert len(stream.compiler.variables) == 2  # one node, one action
    stream.push(check(3, "no_mismatch", 2))
    assert auth(stream)["authorization_risk_hypothesis"] == pytest.approx(good)
    assert auth(stream)["phase"] == "pending"


def test_auth_check_for_one_action_cannot_change_other_actions():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    stream.push(action(1, identity="y"))
    previous = auth(stream, "y")["authorization_risk_hypothesis"]
    stream.push(check(2))
    assert auth(stream, "y")["authorization_risk_hypothesis"] == pytest.approx(previous)
    assert auth(stream)["authorization_risk_hypothesis"] > previous


@pytest.mark.parametrize("change", [{"principal": "user-2"}, {"actor": "other"}, {"tool": "payments.write"},
                                    {"resource": "elsewhere"}, {"arguments": {"command": "different"}}])
def test_scope_changes_are_rejected_atomically(change):
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    previous = stream.explain()
    with pytest.raises(ValueError, match="binding changed"):
        stream.push(check(1, **change))
    assert stream.explain() == previous
    assert stream.push(check(1))["accepted_event_count"] == 2


def test_untrusted_check_self_report_and_raw_flags_do_not_create_authorization():
    model = ActionScopedJointBN()
    for mode in ("observed", "masked"):
        stream = ActionRiskStream(model, **context(), permission_mode=mode)
        stream.push(action(0, permission_mismatch=True))
        initial = auth(stream)["authorization_risk_hypothesis"]
        stream.push(check(1, checker_id="the-agent"))
        assert auth(stream)["authorization_risk_hypothesis"] == initial
        assert stream.explain()["ignored_evidence"][0]["reason"] == "unconfigured_checker"
        stream.push(check(2))
        assert (auth(stream)["authorization_risk_hypothesis"] == initial) == (mode == "masked")


def test_expiry_updates_pending_risk_on_unrelated_event_and_not_executed_history():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    unknown = auth(stream)["authorization_risk_hypothesis"]
    stream.push(check(1, "no_mismatch", valid_until_sequence=2))
    stream.push({"sequence": 3, "actor": "executor", "event_type": "heartbeat"})
    assert auth(stream)["authorization_risk_hypothesis"] == pytest.approx(unknown)
    assert stream.explain()["latest_event_updates"][0]["reason"] == "check_expired"
    stream.push(check(4, "no_mismatch", 1, valid_until_sequence=5))
    stream.push(action(5, "tool_call"))
    historical = auth(stream)["authorization_risk_hypothesis"]
    stream.push({"sequence": 6, "actor": "executor", "event_type": "heartbeat"})
    assert auth(stream)["authorization_risk_hypothesis"] == pytest.approx(historical)
    assert auth(stream)["permission_status"] == "no_mismatch"
    with pytest.raises(ValueError, match="post-hoc"):
        stream.push(check(7, "mismatch", 2))


def test_dispatch_information_does_not_change_after_later_infection():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0, "tool_call"))
    assert auth(stream)["information_at_dispatch_hypothesis"] == 0
    stream.push({"sequence": 1, "actor": "executor", "event_type": "input_received", "source_id": "late", "data_class": "untrusted"})
    assert auth(stream)["information_at_dispatch_hypothesis"] == 0
    info = next(v for v in stream.explain()["variables"] if v["kind"] == "information_state")
    assert info["marginal"] > 0


def test_cancellation_removes_pending_only_and_does_not_erase_other_dispatched_action():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0, "tool_call", identity="done"))
    stream.push(action(1))
    stream.push(check(2))
    done = auth(stream, "done")
    stream.push(action(3, "action_cancelled"))
    assert not stream.compiler.last_step.pending_pairs
    assert len(stream.compiler.last_step.dispatched_pairs) == 1
    assert auth(stream, "done") == done
    assert auth(stream)["phase"] == "cancelled"
    with pytest.raises(ValueError, match="cancelled"):
        stream.push(action(4, "tool_call"))


def test_duplicates_do_not_reapply_a_check_or_copy_or_expire():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    proposal, checked, dispatch = action(0), check(1), action(2, "tool_call")
    for event in (proposal, checked, dispatch):
        stream.push(event)
        before = stream.distribution.copy()
        result = stream.push(copy.deepcopy(event))
        assert result["duplicate_event"]
        np.testing.assert_equal(stream.distribution, before)
    before = stream.explain()
    with pytest.raises(ValueError, match="conflicting"):
        stream.push({**proposal, "permission_mismatch": True})
    assert stream.explain() == before
    with pytest.raises(ValueError, match="completed"):
        stream.push(action(3, "tool_call"))


@pytest.mark.parametrize("bad", [check(1, revision=-1), check(1, revision=True), check(1, valid_until_sequence=0),
                                check(1, permission_status="safe"), action(1, arguments={"v": float("nan")}),
                                {"sequence": 1, "actor": "executor", "event_type": "tool_call", "tool": "shell.exec"}])
def test_invalid_events_do_not_consume_sequence_or_mutate_state(bad):
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    before = stream.explain()
    with pytest.raises((ValueError, TypeError)):
        stream.push(bad)
    assert stream.explain() == before
    stream.push(check(1))


def test_capacity_failure_before_dispatch_is_atomic():
    stream = ActionRiskStream(ActionScopedJointBN(), **context(), max_variables=2)
    stream.push(action(0))
    before = stream.explain()
    with pytest.raises(ValueError, match="variable limit"):
        stream.push(action(1, "tool_call"))
    assert stream.explain() == before
    stream.push(check(1))


@pytest.mark.parametrize("use_messages,use_latent_head", [(True, True), (False, True), (True, False)])
def test_stream_and_every_batch_prefix_match_including_complete_joint_mass(use_messages, use_latent_head):
    model = ActionScopedJointBN(use_messages=use_messages, use_latent_head=use_latent_head)
    model.theta += np.random.default_rng(914).normal(0, .2, len(model.theta))
    stream = ActionRiskStream(model, **context())
    for index, event in enumerate(events()):
        actual = stream.push(event)
        trace = compile_action_trace({"events": events()[:index + 1]}, **context())
        scores, _, mass = model._forward(model.theta, trace)
        assert actual["risk_probability"] == pytest.approx(scores[-1], abs=1e-13)
        assert list(trace.variables) == stream.compiler.variables
        np.testing.assert_allclose(mass, stream.distribution, atol=1e-14)
    all_scores = model.predict([compile_action_trace({"events": events()}, **context())])[0]
    prefix_scores = [model.predict([compile_action_trace({"events": events()[:i]}, **context())])[0][-1] for i in range(1, len(events()) + 1)]
    np.testing.assert_allclose(all_scores, prefix_scores, atol=1e-14)


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_73_gradients_through_replacement_expiry_and_dispatch_match_finite_differences(label, prefix):
    model = ActionScopedJointBN(regularization=.02)
    compiled = compile_action_trace({"events": events()}, **context())
    theta = model.theta + np.random.default_rng(441).normal(0, .2, len(model.theta))
    assert len(theta) == len(PARAMETER_NAMES) == 73
    value, gradient = model.loss_gradient(theta, [compiled], [label], prefix)
    numeric = approx_derivative(lambda x: model.loss_gradient(x, [compiled], [label], prefix)[0], theta).ravel()
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-5)


def independent_enumeration(model, trace):
    """Scalar dictionary enumeration, not production transition/feature code."""
    mass, scores = {tuple([0] * trace.n_variables): 1.}, []
    for step in trace.steps:
        for operation in step.operations:
            if operation.is_message and not model.use_messages:
                continue
            q = 1. if operation.rule == "copy" else float(expit(model.theta[operation.parameter] + (model.theta[CONFIDENCE] * operation.confidence_feature if operation.is_message else 0)))
            result = defaultdict(float)
            for state, probability in mass.items():
                if operation.rule == "copy":
                    target = list(state)
                    target[operation.receiver] = state[operation.sender]
                    result[tuple(target)] += probability
                elif operation.rule == "replace":
                    for value, weight in ((0, 1 - q), (1, q)):
                        target = list(state)
                        target[operation.receiver] = value
                        result[tuple(target)] += probability * weight
                elif state[operation.receiver] or (operation.sender is not None and not state[operation.sender]):
                    result[state] += probability
                else:
                    target = list(state)
                    target[operation.receiver] = 1
                    result[state] += probability * (1 - q)
                    result[tuple(target)] += probability * q
            mass = dict(result)
        risk = 0.
        for state, probability in mass.items():
            count, n = sum(state[i] for i in step.observed_nodes), len(step.observed_nodes)
            features = [count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
                sum(state[i] for i in step.capable_nodes) / max(1, len(step.capable_nodes)),
                sum(state[i] for i in step.terminal_nodes) / max(1, len(step.terminal_nodes)),
                any(state[a] for a, _ in step.pending_pairs), any(state[a] for a, _ in step.dispatched_pairs),
                any(state[a] and state[c] for a, c in step.pending_pairs), any(state[a] and state[d] for a, d in step.dispatched_pairs)]
            if not model.use_latent_head:
                features = [0.] * len(features)
            logit = sum(float(x) * y for x, y in zip([*step.observables, *features], model.theta[HEAD:]))
            risk += probability * float(expit(logit))
        scores.append(risk)
    vector = np.zeros(1 << trace.n_variables)
    for state, probability in mass.items():
        vector[sum(value * 2 ** i for i, value in enumerate(state))] = probability
    return scores, vector


def test_independent_enumeration_matches_all_states_and_explanation():
    model = ActionScopedJointBN()
    model.theta += np.random.default_rng(53).normal(0, .8, len(model.theta))
    trace = compile_action_trace({"events": events()}, **context())
    expected, expected_mass = independent_enumeration(model, trace)
    scores, _, mass = model._forward(model.theta, trace)
    np.testing.assert_allclose(scores, expected, atol=1e-13)
    np.testing.assert_allclose(mass, expected_mass, atol=1e-13)
    explanation = model.explain(trace)
    assert sum(x["risk_contribution"] for x in explanation["system_mixture_terms"]) + explanation["remaining_risk_contribution"] == pytest.approx(scores[-1], abs=1e-13)
    with pytest.raises(ValueError, match="not implemented"):
        model.explain(trace, include_graph=True)


def test_fit_exercises_joint_training_but_does_not_claim_fixture_accuracy():
    model = ActionScopedJointBN(max_iter=20)
    a = compile_action_trace({"events": [action(0), check(1), action(2, "tool_call")]}, **context())
    b = compile_action_trace({"events": [action(0), check(1, "no_mismatch"), action(2, "tool_call")]}, **context())
    before, _ = model.loss_gradient(model.theta, [a, b], [1, 0])
    model.fit([a, b], [1, 0])
    after, _ = model.loss_gradient(model.theta, [a, b], [1, 0])
    assert after < before
    assert np.isfinite(model.theta).all()


def test_authorization_alone_is_not_transferred_in_a_message():
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    stream.push(action(0))
    stream.push(check(1))
    stream.push({"sequence": 2, "actor": "executor", "target": "reader", "event_type": "message_sent", "data_class": "untrusted"})
    assert auth(stream)["authorization_risk_hypothesis"] > 0
    assert all(v["marginal"] == 0 for v in stream.explain()["variables"] if v["kind"] == "information_state")


def test_different_agents_grouped_into_same_workflow_node_cannot_share_action_binding():
    scope = {"workflow": {"nodes": ["execution"], "edges": [], "agent_to_node": {"executor": "execution", "other": "execution"}},
             "trusted_checkers": ["policy-service"]}
    stream = ActionRiskStream(ActionScopedJointBN(), **scope)
    stream.push(action(0))
    with pytest.raises(ValueError, match="binding changed"):
        stream.push(check(1, actor="other"))
    stream.push(check(1))
    assert auth(stream)["node"] == "execution"


def test_check_stream_revisions_and_identity_cannot_be_silently_mixed():
    stream = ActionRiskStream(ActionScopedJointBN(), trusted_checkers=["policy-service", "other-service"])
    stream.push(action(0))
    stream.push(check(1))
    before = stream.explain()
    with pytest.raises(ValueError, match="revisions"):
        stream.push(check(2))
    assert stream.explain() == before
    with pytest.raises(ValueError, match="checker streams"):
        stream.push(check(2, revision=1, checker_id="other-service"))
    assert stream.explain() == before
    stream.push(check(2, revision=1))


def test_online_action_model_does_not_recompile_or_replay(monkeypatch):
    import paperalpha_monitor.action_evidence as module
    def forbidden(*a, **k):
        raise AssertionError("no batch replay in incremental monitor")
    monkeypatch.setattr(module, "compile_action_trace", forbidden)
    monkeypatch.setattr(ActionScopedJointBN, "_forward", forbidden)
    stream = ActionRiskStream(ActionScopedJointBN(), **context())
    for event in events():
        assert 0 <= stream.push(event)["risk_probability"] <= 1


def test_superseded_prior_has_no_terminal_gradient_but_can_train_from_earlier_prefixes():
    model = ActionScopedJointBN(regularization=0)
    trace = compile_action_trace({"events": [action(0), check(1), action(2, "tool_call")]}, **context())
    unknown = KNOWLEDGE.cell_index("authorization", {"role": "executor", "permission": "unknown"})
    _, terminal_gradient = model.episode_loss_gradient(model.theta, trace, 0, False)
    _, prefix_gradient = model.episode_loss_gradient(model.theta, trace, 0, True)
    assert abs(terminal_gradient[unknown]) < 1e-14
    assert abs(prefix_gradient[unknown]) > 1e-6
