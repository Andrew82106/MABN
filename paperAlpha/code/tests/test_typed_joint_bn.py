import copy
from pathlib import Path
import sys
import math
from collections import defaultdict

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))

from paperalpha_monitor.dynamic_joint_bn import DynamicJointBN, compile_trace, HEAD, CONFIDENCE, OBSERVABLES
from paperalpha_monitor.typed_joint_bn import TypedJointBN, compile_typed_trace, type_trace, PARAMETER_NAMES


def events(after=True):
    before = [{"sequence": 0, "actor": "a", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
              {"sequence": 1, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted"},
              {"sequence": 2, "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": True}]
    if after:
        before.append({"sequence": 3, "actor": "b", "target": "c", "event_type": "message_sent", "data_class": "clean"})
    return before


def marginal(model, trace, kind, identity):
    values = model.explain(trace)["variables"]
    return next(v["marginal"] for v in values if v["kind"] == kind and v["id"] == identity)


def test_authorization_anomaly_cannot_become_message_carried_information():
    es = [{"sequence": 0, "actor": "a", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": True},
          {"sequence": 1, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted"}]
    old, new = DynamicJointBN(), TypedJointBN()
    old_trace, new_trace = compile_trace({"events": es}), compile_typed_trace({"events": es})
    assert marginal(old, old_trace, "workflow_node", "b") > 0
    assert marginal(new, new_trace, "information_state", "b") == 0
    assert marginal(new, new_trace, "information_state", "a") == 0
    assert marginal(new, new_trace, "authorization_state", "a") > 0
    assert not any(v["kind"] == "authorization_state" and v["id"] == "b" for v in new_trace.variables)


def test_information_still_flows_after_an_authorization_event():
    trace = compile_typed_trace({"events": events()})
    model = TypedJointBN()
    assert marginal(model, trace, "information_state", "c") > 0
    assert marginal(model, trace, "authorization_state", "b") > 0


@pytest.mark.parametrize("role", ["unknown", "reader", "reasoner", "reviewer", "executor"])
@pytest.mark.parametrize("trust", ["clean", "untrusted", "unknown"])
def test_information_channel_is_invariant_to_authorization_flag_for_every_role_and_message_class(role, trust):
    model = TypedJointBN()
    model.theta = np.random.default_rng(42).uniform(-4, 4, len(model.theta))
    values = []
    for mismatch in (False, True):
        es = [{"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"},
              {"sequence": 1, "actor": "a", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": mismatch},
              {"sequence": 2, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": trust}]
        trace = compile_typed_trace({"events": es}, roles={"a": role})
        values.append(marginal(model, trace, "information_state", "b"))
    assert values[0] == pytest.approx(values[1], abs=1e-14)


def test_collapsed_head_matches_old_model_without_later_messages_including_loss_gradient():
    old_trace = compile_trace({"events": events(False)})
    typed = type_trace(old_trace)
    old, new = DynamicJointBN(), TypedJointBN(typed_head=False)
    old.theta += np.random.default_rng(57).normal(0, .4, size=len(old.theta))
    new.theta = np.r_[old.theta, 0., 0.]
    np.testing.assert_allclose(old.predict([old_trace])[0], new.predict([typed])[0], atol=1e-14)
    old_loss, old_gradient = old.loss_gradient(old.theta, [old_trace], [1])
    new_loss, new_gradient = new.loss_gradient(new.theta, [typed], [1])
    assert old_loss == pytest.approx(new_loss, abs=1e-12)
    np.testing.assert_allclose(old_gradient, new_gradient[:-2], atol=1e-12)
    np.testing.assert_array_equal(new_gradient[-2:], [0, 0])


def test_typed_gradients_match_finite_differences_for_both_targets_and_head_modes():
    trace = compile_typed_trace({"events": events()})
    for head in (True, False):
        model = TypedJointBN(typed_head=head)
        theta = model.theta + np.random.default_rng(11).normal(0, .1, len(model.theta))
        for label in (0, 1):
            _, gradient = model.loss_gradient(theta, [trace], [label])
            numeric = np.zeros_like(theta)
            for i in range(len(theta)):
                a, b = theta.copy(), theta.copy()
                a[i] += 1e-5
                b[i] -= 1e-5
                numeric[i] = (model.loss_gradient(a, [trace], [label])[0] - model.loss_gradient(b, [trace], [label])[0]) / 2e-5
            np.testing.assert_allclose(gradient, numeric, atol=2e-9, rtol=2e-5)


def test_future_authorization_and_nodes_do_not_change_earlier_prefix_scores():
    es, model = events(), TypedJointBN()
    model.theta[-2:] = [1.3, -.7]
    full = model.predict([compile_typed_trace({"events": es})])[0]
    for n in range(1, len(es) + 1):
        prefix = compile_typed_trace({"events": es[:n]})
        np.testing.assert_allclose(model.predict([prefix])[0], full[:n], atol=1e-14)
        if n < 3:
            assert not any(v["kind"] == "authorization_state" for v in prefix.variables)


def test_unknown_permissions_and_workflow_do_not_gain_hidden_information():
    a, b = events(), events()
    b[2]["permission_mismatch"] = False
    x, y = [compile_typed_trace({"events": es}, permission_mode="masked") for es in (a, b)]
    np.testing.assert_allclose(*TypedJointBN().predict([x, y]), atol=1e-14)


def test_parameter_layout_exact_mixture_and_capacity_are_explicit():
    model, trace = TypedJointBN(), compile_typed_trace({"events": events()})
    explanation = model.explain(trace)
    assert len(model.theta) == len(PARAMETER_NAMES) == 71
    assert explanation["mass_sum"] == pytest.approx(1.)
    assert sum(r["risk_contribution"] for r in explanation["system_mixture_terms"]) + explanation["remaining_risk_contribution"] == pytest.approx(explanation["risk_probability"])
    with pytest.raises(ValueError, match="variable limit"):
        compile_typed_trace({"events": events()}, max_variables=4)
    graph_explanation = model.explain(trace, include_graph=True)
    assert graph_explanation["graph_risk_probability"] == pytest.approx(explanation["risk_probability"], abs=1e-12)
    with pytest.raises(ValueError, match="typed compiler"):
        model.predict([compile_trace({"events": events()})])


def test_shared_origin_survives_separate_authorization_without_duplication():
    es = events()
    es.insert(1, {"sequence": 1, "actor": "d", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"})
    for i, e in enumerate(es):
        e["sequence"] = i
    trace = compile_typed_trace({"events": es})
    assert sum(v["kind"] == "origin_hypothesis" for v in trace.variables) == 1
    before = copy.deepcopy(trace.variables)
    TypedJointBN().predict([trace])
    assert trace.variables == before


def test_explicit_workflow_group_shares_information_but_has_separate_authorization_state():
    workflow = {"nodes": ["reader", "writer"], "edges": [["reader", "writer"]],
                "agent_to_node": {"a": "reader", "b": "reader", "c": "writer"}}
    trace = compile_typed_trace({"events": events()}, workflow=workflow)
    assert sum(v["kind"] == "information_state" for v in trace.variables) == 2
    assert [(v["id"], v["risk_channel"]) for v in trace.variables if v["kind"] == "authorization_state"] == [("reader", "authorization")]
    assert marginal(TypedJointBN(), trace, "information_state", "writer") > 0


def enumerated_probabilities(trace, theta):
    """Small-state reference: explicit dictionaries and scalar feature logic.

    Does not use the dense transition, typed_features, or model forward code.
    """
    sigmoid = lambda x: 1 / (1 + math.exp(-float(x)))
    distribution = {(0,) * trace.n_variables: 1.}
    probabilities = []
    for step in trace.steps:
        for operation in step.operations:
            q = sigmoid(theta[operation.parameter] + (theta[CONFIDENCE] * operation.confidence_feature if operation.is_message else 0))
            updated = defaultdict(float)
            for state, probability in distribution.items():
                if state[operation.receiver] or (operation.sender is not None and not state[operation.sender]):
                    updated[state] += probability
                else:
                    changed = list(state)
                    changed[operation.receiver] = 1
                    updated[state] += probability * (1 - q)
                    updated[tuple(changed)] += probability * q
            distribution = updated
        risk = 0.
        for state, probability in distribution.items():
            def union(pairs):
                return [int(bool(state[i] or (a is not None and state[a]))) for i, a in pairs]
            count, n = sum(union(step.node_pairs)), len(step.node_pairs)
            features = [count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
                sum(union(step.capable_pairs)) / max(1, len(step.capable_pairs)),
                sum(union(step.terminal_pairs)) / max(1, len(step.terminal_pairs)),
                int(any(a is not None and state[a] for _, a in step.node_pairs)),
                int(any(a is not None and state[a] and state[i] for i, a in step.node_pairs))]
            risk += probability * sigmoid(sum(v * w for v, w in zip(step.observables, theta[HEAD:HEAD + len(OBSERVABLES)]))
                                           + sum(v * w for v, w in zip(features, theta[HEAD + len(OBSERVABLES):])))
        probabilities.append(risk)
    return np.asarray(probabilities)


def test_independent_scalar_state_enumeration_agrees_with_typed_joint_filter():
    trace, model = compile_typed_trace({"events": events()}), TypedJointBN()
    model.theta += np.random.default_rng(81).normal(0, .4, len(model.theta))
    np.testing.assert_allclose(model.predict([trace])[0], enumerated_probabilities(trace, model.theta), atol=2e-14)
    _, gradient = model.loss_gradient(model.theta, [trace], [1])
    def loss(theta):
        return -np.log(enumerated_probabilities(trace, theta)).mean() + model.regularization * np.sum((theta - model.initial_parameters()) ** 2) / 69
    numeric = []
    for i in range(len(model.theta)):
        a, b = model.theta.copy(), model.theta.copy()
        a[i] += 1e-5
        b[i] -= 1e-5
        numeric.append((loss(a) - loss(b)) / 2e-5)
    np.testing.assert_allclose(gradient, numeric, atol=2e-9, rtol=2e-5)
