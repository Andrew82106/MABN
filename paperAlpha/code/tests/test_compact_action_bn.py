import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.compact_action_bn import (CompactCapabilityBN, CompactCapabilityStream,
    compile_compact_trace, pending_mixture)
from paperalpha_monitor.capability_action_bn import CapabilityActionBN, CapabilityRiskStream, compile_capability_trace, PARAMETER_NAMES
from paperalpha_monitor.dynamic_joint_bn import state_bits
from test_capability_action_bn import context, scenario, action, full_action, checked, registry


def marginalize_full(full_trace, full_mass, compact_trace):
    """Independent projection of full A/D states onto compact history states."""
    mapping = [(next(i for i, v in enumerate(full_trace.variables) if v["kind"] == c["kind"] and v["id"] == c["id"]), j)
               for j, c in enumerate(compact_trace.variables)]
    result = np.zeros((1 << compact_trace.n_variables, 3))
    step = full_trace.steps[-1]
    for state, mass in enumerate(full_mass):
        index = sum(((state >> old) & 1) << new for old, new in mapping)
        any_a = any((state >> a) & 1 for a, _ in step.dispatched_pairs)
        overlap = any(((state >> a) & 1) and ((state >> d) & 1) for a, d in step.dispatched_pairs)
        result[index, 2 if overlap else 1 if any_a else 0] += mass
    return result


@pytest.mark.parametrize("messages,latent", [(True, True), (False, True), (True, False)])
def test_every_prefix_scores_and_joint_marginals_match_full_enumeration(messages, latent):
    full = CapabilityActionBN(use_messages=messages, use_latent_head=latent)
    compact = CompactCapabilityBN(use_messages=messages, use_latent_head=latent)
    full.theta += np.random.default_rng(27).normal(0, .8, len(full.theta))
    compact.theta = full.theta.copy()
    stream = CompactCapabilityStream(compact, **context())
    for i, event in enumerate(scenario()):
        stream.push(event)
        prefix = {"events": scenario()[:i + 1]}
        a, b = compile_capability_trace(prefix, **context()), compile_compact_trace(prefix, **context())
        pa, _, ma = full._forward(full.theta, a)
        pb, _, mb = compact._forward(compact.theta, b)
        np.testing.assert_allclose(pa, pb, rtol=0, atol=4e-15)
        np.testing.assert_allclose(marginalize_full(a, ma, b), mb, rtol=0, atol=4e-15)
        np.testing.assert_allclose(stream.engine.distribution, mb, rtol=0, atol=4e-15)
        assert stream.snapshot()["risk_probability"] == pytest.approx(pb[-1], abs=2e-15)
        old_actions = {a["action_id"]: a for a in full.explain(a)["actions"]}
        for new in stream.explain()["actions"]:
            old = old_actions[new["action_id"]]
            for key in ("authorization_risk_hypothesis", "information_at_dispatch_hypothesis"):
                assert new[key] == pytest.approx(old[key], abs=3e-15) if old[key] is not None else new[key] is None
        assert sum(t["risk_contribution"] for t in stream.explain()["system_mixture_terms"]) == pytest.approx(pb[-1])


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_150_gradients_equal_full_action_reverse_gradients(label, prefix):
    full, compact = CapabilityActionBN(), CompactCapabilityBN()
    theta = full.theta + np.random.default_rng(74).normal(0, .6, 150)
    # Adds a second pending action so pending products and shared parameter
    # counts are exercised while retaining a tractable full-action reference.
    events = scenario()[:7] + [action(8, action_id="pending-a"), action(9, action_id="pending-b")]
    a = compile_capability_trace({"events": events}, **context())
    b = compile_compact_trace({"events": events}, **context())
    old_loss, old_grad = full.episode_loss_gradient(theta, a, label, prefix)
    new_loss, new_grad = compact.episode_loss_gradient(theta, b, label, prefix)
    assert new_loss == pytest.approx(old_loss, abs=2e-14)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-10, atol=1e-13)
    numeric = approx_derivative(lambda t: compact.episode_loss_gradient(t, b, label, prefix)[0], theta, method="3-point").ravel()
    np.testing.assert_allclose(new_grad, numeric, rtol=2e-5, atol=3e-8)


def test_later_information_cannot_rewrite_past_dispatched_overlap():
    model = CompactCapabilityBN()
    stream = CompactCapabilityStream(model, **context())
    stream.push(action(0, "tool_call"))
    before = stream.engine.distribution.sum(axis=0)
    stream.push({"sequence": 1, "actor": "executor", "event_type": "input_received", "source_id": "late", "data_class": "untrusted"})
    np.testing.assert_allclose(stream.engine.distribution.sum(axis=0), before, atol=1e-15)
    assert stream.explain()["actions"][0]["information_at_dispatch_hypothesis"] == 0
    stream.push(action(2, "tool_call", action_id="after"))
    assert stream.engine.distribution[:, 2].sum() > 0


def test_pending_products_do_not_assume_independent_information_nodes():
    # A shared source fans out to two agents, with a join and two dispatches.
    events = [
        {"sequence": 0, "actor": "reader", "event_type": "input_received", "source_id": "shared", "data_class": "untrusted"},
        {"sequence": 1, "actor": "executor", "event_type": "input_received", "source_id": "shared", "data_class": "untrusted"},
        {"sequence": 2, "actor": "reader", "target": "executor", "event_type": "message_sent", "data_class": "untrusted"},
        action(3, actor="reader", action_id="a"), action(4, action_id="b"),
        action(5, "tool_call", actor="reader", action_id="a"), action(6, "tool_call", action_id="b")]
    a, b = compile_capability_trace({"events": events}, **context()), compile_compact_trace({"events": events}, **context())
    full, compact = CapabilityActionBN(), CompactCapabilityBN()
    for seed in range(5):
        theta = full.initial_parameters() + np.random.default_rng(seed).normal(0, 1, 150)
        full.theta, compact.theta = theta.copy(), theta.copy()
        np.testing.assert_allclose(full.predict([a]), compact.predict([b]), atol=1e-14, rtol=0)


def test_compact_duplicate_and_rejection_do_not_change_joint_state():
    stream = CompactCapabilityStream(CompactCapabilityBN(), **context())
    stream.push(full_action(0))
    stream.push(checked(1))
    before = stream.engine.distribution.copy()
    assert stream.push(checked(1))["duplicate_event"]
    np.testing.assert_array_equal(stream.engine.distribution, before)
    previous = stream.explain()
    with pytest.raises(ValueError, match="binding changed"):
        stream.push({**checked(2), "resource": "other"})
    assert stream.explain() == previous
    stream.push(full_action(2, "tool_call"))
    before = stream.engine.distribution.copy()
    stream.push(full_action(2, "tool_call"))
    np.testing.assert_array_equal(stream.engine.distribution, before)


def test_only_information_capacity_remains_and_failure_is_atomic():
    stream = CompactCapabilityStream(CompactCapabilityBN(), **context(), max_information_variables=1)
    for i in range(20):
        stream.push(action(i, "tool_call", action_id=f"a{i}"))
    assert stream.engine.distribution.shape == (2, 3)
    before = stream.explain()
    with pytest.raises(ValueError, match="variable"):
        stream.push({"sequence": 20, "actor": "executor", "event_type": "input_received", "source_id": "new"})
    assert stream.explain() == before


def test_extreme_and_many_pending_factors_stay_normalized_with_finite_gradients():
    theta = CompactCapabilityBN().theta
    cell = PARAMETER_NAMES.index("authorization:payment_effect:executor:unknown")
    for value in (-1000., -8., 0., 8., 1000.):
        theta[cell] = value
        pending = [(0, cell)] * 1000 + [(1, cell)] * 300
        probabilities, jacobian = pending_mixture(theta, state_bits(2), pending, True)
        assert np.isfinite(probabilities).all() and np.isfinite(jacobian).all()
        assert probabilities.min() >= 0
        np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=2e-15)
        np.testing.assert_allclose(jacobian.sum(axis=1), 0., atol=2e-15)


def test_future_actions_and_inputs_do_not_change_past_prefixes():
    model = CompactCapabilityBN()
    events = scenario() + [{"sequence": 10, "actor": "executor", "event_type": "input_received", "source_id": "future"}]
    full = compile_compact_trace({"events": events}, **context())
    for i in range(1, len(events)):
        prefix = compile_compact_trace({"events": events[:i]}, **context())
        np.testing.assert_allclose(model.predict([full])[0][:i], model.predict([prefix])[0], atol=2e-15, rtol=0)


def test_full_and_compact_fits_share_loss_regularization_and_parameter_layout():
    full, compact = CapabilityActionBN(max_iter=4), CompactCapabilityBN(max_iter=4)
    events = [scenario(), scenario()[:3]]
    a = [compile_capability_trace({"events": e}, **context()) for e in events]
    b = [compile_compact_trace({"events": e}, **context()) for e in events]
    full.fit(a, [0, 1])
    compact.fit(b, [0, 1])
    np.testing.assert_allclose(full.theta, compact.theta, rtol=1e-10, atol=1e-11)
    assert full.parameters()["parameter_names"] == compact.parameters()["parameter_names"]
    assert compact.parameters()["inference"]["method"] == "exact_v4_action_elimination"


def test_compact_filter_rejects_full_trace_instead_of_misreading_variable_ids():
    with pytest.raises(ValueError, match="compact compiler"):
        CompactCapabilityBN().predict([compile_capability_trace({"events": [action(0)]}, **context())])


def test_all_capability_role_check_cells_match_full_regularized_objective():
    from paperalpha_monitor.tool_capabilities import CAPABILITIES
    from paperalpha_monitor.dynamic_joint_bn import KNOWLEDGE
    full, compact, left, right = CapabilityActionBN(), CompactCapabilityBN(), [], []
    for k in range(30):
        cap, role = CAPABILITIES[k // 5], KNOWLEDGE.states("role")[k % 5]
        events = [
            {"sequence": 0, "actor": "reader", "event_type": "input_received", "source_id": "d", "data_class": KNOWLEDGE.states("trust")[k % 3]},
            {"sequence": 1, "actor": "reader", "target": "executor", "event_type": "delegation" if k % 2 else "message_sent", "data_class": "untrusted", "edge_confidence": .9},
            full_action(2, tool=cap), checked(3, tool=cap), checked(4, "no_mismatch", tool=cap),
            full_action(5, "tool_call", tool=cap), action(6, action_id="pending", tool="unknown")]
        ctx = context(roles={"reader": role, "executor": role})
        left.append(compile_capability_trace({"events": events}, **ctx))
        right.append(compile_compact_trace({"events": events}, **ctx))
    theta = full.theta + np.random.default_rng(765).normal(0, .3, 150)
    labels = np.arange(30) % 2
    a, da = full.loss_gradient(theta, left, labels)
    b, db = compact.loss_gradient(theta, right, labels)
    assert b == pytest.approx(a, abs=2e-14)
    np.testing.assert_allclose(db, da, atol=2e-13, rtol=1e-9)


def test_workflow_grouping_unknown_tools_and_masked_checks_keep_same_readout():
    workflow = {"nodes": ["read-group", "act-group"], "edges": [["read-group", "act-group"]],
                "agent_to_node": {"reader": "read-group", "executor": "act-group"}}
    ctx = context(workflow=workflow, roles={"read-group": "reader", "act-group": "executor"},
                  available_tools={"reader": ["information_read"], "executor": ["payment_effect"]}, permission_mode="masked")
    a, b = compile_capability_trace({"events": scenario()}, **ctx), compile_compact_trace({"events": scenario()}, **ctx)
    full, compact = CapabilityActionBN(), CompactCapabilityBN()
    np.testing.assert_allclose(full.predict([a]), compact.predict([b]), atol=1e-14, rtol=0)


def test_different_registries_cannot_be_spliced_inside_compact_episode():
    from dataclasses import replace
    trace = compile_compact_trace({"events": [action(0), action(1, action_id="b")]}, **context())
    changed = replace(trace.steps[1], capability_registry_sha256="different")
    with pytest.raises(ValueError, match="fixed capability registry"):
        CompactCapabilityBN().predict([replace(trace, steps=(trace.steps[0], changed))])
