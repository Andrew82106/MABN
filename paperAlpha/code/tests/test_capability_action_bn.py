import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.action_evidence import compile_action_trace
from paperalpha_monitor.action_joint_bn import ActionScopedJointBN, ActionRiskStream
from paperalpha_monitor.capability_action_bn import (CapabilityActionBN, CapabilityRiskStream,
    compile_capability_trace, CAPABILITIES, PARAMETER_NAMES, CAP_HEAD, CAP_CONFIDENCE)
from paperalpha_monitor.tool_capabilities import ToolCapabilityRegistry, load_tool_registry
from paperalpha_monitor.dynamic_joint_bn import KNOWLEDGE


def registry():
    return ToolCapabilityRegistry({"schema_version": "1", "registry_id": "test-public",
        "source": {"url": "fixture://public-tools", "revision": "1", "selection": "test descriptions only"},
        "tools": [{"name": capability, "description": "Declared capability: " + capability,
            "capability": capability, "supporting_quote": capability, "annotation_method": "manual_review"}
            for capability in CAPABILITIES]})


def action(t, kind="tool_proposed", **extra):
    return {"sequence": t, "event_id": f"e-{t}", "event_type": kind, "actor": "executor", "action_id": "x",
            "tool": "payment_effect", "arguments": {"amount": 5}, **extra}


def full_action(t, kind="tool_proposed", **extra):
    return action(t, kind, principal="user", resource="account", **extra)


def checked(t, status="mismatch", **extra):
    return full_action(t, "authorization_checked", checker_id="policy", permission_status=status,
                       check_revision=t, valid_until_sequence=t + 1, **extra)


def context(**extra):
    return {"tool_registry": registry(), "trusted_checkers": ["policy"],
            "roles": {"executor": "executor", "reader": "reader"}, **extra}


def scenario():
    return [
        {"sequence": 0, "actor": "reader", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 1, "actor": "reader", "target": "executor", "event_type": "message_sent", "data_class": "untrusted", "edge_confidence": .8},
        full_action(2), checked(3),
        {"sequence": 5, "actor": "executor", "event_type": "heartbeat"},
        checked(6, "no_mismatch"), full_action(7, "tool_call"),
        action(8, tool="unregistered", action_id="y"),
        action(9, "action_cancelled", tool="unregistered", action_id="y")]


def test_partial_and_unknown_actions_are_not_discarded_or_invented_as_authorized():
    stream = CapabilityRiskStream(CapabilityActionBN(), **context())
    stream.push(action(0, "tool_call", tool="new_tool"))
    explanation = stream.explain()
    a = explanation["actions"][0]
    assert a["binding"]["principal"] is None and a["binding"]["resource"] is None
    assert a["permission_status"] == "unknown" and a["accepted_check"] is None
    assert a["capability_profile"]["capability"] == "unknown"
    assert a["authorization_risk_hypothesis"] > 0
    assert explanation["coverage"]["unknown_capability_actions"] == ["x"]
    assert explanation["coverage"]["incomplete_scope_actions"] == ["x"]
    assert explanation["observed_features"]["unknown_action_capability_fraction"] == 1
    assert explanation["observed_features"]["incomplete_action_scope_fraction"] == 1


@pytest.mark.parametrize("capability", CAPABILITIES)
def test_capability_selects_the_corresponding_trainable_parameter(capability):
    model = CapabilityActionBN()
    index = PARAMETER_NAMES.index(f"authorization:{capability}:executor:unknown")
    model.theta[index] = 1.75
    stream = CapabilityRiskStream(model, **context())
    stream.push(action(0, tool=capability))
    report = stream.explain()
    assert report["latest_event_updates"][0]["parameter"] == PARAMETER_NAMES[index]
    assert report["actions"][0]["authorization_risk_hypothesis"] == pytest.approx(expit(1.75))
    model.theta[index] = -3.
    other = CapabilityRiskStream(model, **context())
    other.push(action(0, tool=capability))
    assert other.snapshot()["risk_probability"] < stream.snapshot()["risk_probability"]


def test_tool_rename_including_legacy_privileged_alias_preserves_scores_and_joint_mass():
    original = registry().specification()
    renamed = copy.deepcopy(original)
    next(t for t in renamed["tools"] if t["name"] == "payment_effect")["name"] = "shell.exec"
    model = CapabilityActionBN()
    model.theta += np.random.default_rng(41).normal(0, .3, len(model.theta))
    before = CapabilityRiskStream(model, **context(available_tools={"executor": ["payment_effect"]}))
    after = CapabilityRiskStream(model, **context(tool_registry=ToolCapabilityRegistry(renamed), available_tools={"executor": ["shell.exec"]}))
    for event in scenario():
        changed = copy.deepcopy(event)
        if changed.get("tool") == "payment_effect":
            changed["tool"] = "shell.exec"
        assert before.push(event)["risk_probability"] == after.push(changed)["risk_probability"]
        np.testing.assert_array_equal(before.distribution, after.distribution)


def test_descriptor_capability_not_name_changes_risk_and_availability_never_grants_permission():
    spec = registry().specification()
    next(t for t in spec["tools"] if t["name"] == "payment_effect").update(
        capability="information_read", description="Read account information", supporting_quote="Read account")
    model = CapabilityActionBN()
    model.theta[PARAMETER_NAMES.index("authorization:payment_effect:executor:unknown")] = 2
    model.theta[PARAMETER_NAMES.index("authorization:information_read:executor:unknown")] = -4
    streams = [CapabilityRiskStream(model, **context(tool_registry=r, available_tools={"executor": ["payment_effect"]}))
               for r in (registry(), ToolCapabilityRegistry(spec))]
    scores = [s.push(action(0))["risk_probability"] for s in streams]
    assert scores[0] > scores[1]
    assert all(s.explain()["actions"][0]["permission_status"] == "unknown" for s in streams)


def test_partial_scope_cannot_accept_permission_checks_and_rejection_is_atomic():
    stream = CapabilityRiskStream(CapabilityActionBN(), **context())
    stream.push(action(0))
    before = stream.explain()
    partial = {**checked(1), "principal": None, "resource": None}
    with pytest.raises(ValueError, match="complete action scope"):
        stream.push(partial)
    assert before == stream.explain()
    with pytest.raises(ValueError, match="binding changed"):
        stream.push(checked(1))
    assert before == stream.explain()
    stream.push(action(1, "tool_call"))


@pytest.mark.parametrize("field", ["action_id", "actor", "tool", "arguments"])
def test_required_observation_identity_is_not_inferred(field):
    e = action(0)
    del e[field]
    with pytest.raises(ValueError):
        CapabilityRiskStream(CapabilityActionBN(), **context()).push(e)


def test_authorize_payment_name_and_tool_result_do_not_create_a_trusted_check():
    stream = CapabilityRiskStream(CapabilityActionBN(), tool_registry=load_tool_registry())
    stream.push(action(0, "tool_call", tool="authorize_payment", permission_denied=False))
    a = stream.explain()["actions"][0]
    assert a["capability_profile"]["capability"] == "payment_effect"
    assert a["permission_status"] == "unknown"
    assert a["accepted_check"] is None


def test_payload_is_opaque_but_top_level_target_fields_are_rejected():
    model = CapabilityActionBN()
    traces = [compile_capability_trace({"events": [action(0, arguments={"label": label, "capability": "safe"})]}, **context())
              for label in (0, 1)]
    np.testing.assert_array_equal(model.predict(traces)[0], model.predict(traces)[1])
    for field in ("label", "external_effect", "hidden_fields", "violation"):
        with pytest.raises(ValueError):
            compile_capability_trace({"events": [action(0, **{field: True})]}, **context())
        with pytest.raises(ValueError):
            compile_capability_trace({"events": [action(0)], field: True}, **context())


@pytest.mark.parametrize("use_messages,use_latent_head", [(True, True), (False, True), (True, False)])
def test_every_prefix_batch_online_and_duplicate_are_identical(use_messages, use_latent_head):
    model = CapabilityActionBN(use_messages=use_messages, use_latent_head=use_latent_head)
    model.theta += np.random.default_rng(52).normal(0, .4, len(model.theta))
    stream = CapabilityRiskStream(model, **context())
    for i, event in enumerate(scenario()):
        score = stream.push(event)["risk_probability"]
        trace = compile_capability_trace({"events": scenario()[:i + 1]}, **context())
        scores, _, mass = model._forward(model.theta, trace)
        assert score == pytest.approx(scores[-1], abs=1e-15)
        np.testing.assert_array_equal(stream.distribution, mass)
        assert model.explain(trace)["coverage"] == stream.explain()["coverage"]
        if "event_id" in event:
            assert stream.push(copy.deepcopy(event))["duplicate_event"]
            np.testing.assert_array_equal(stream.distribution, mass)
    full = compile_capability_trace({"events": scenario()}, **context())
    for i in range(len(scenario())):
        prefix = compile_capability_trace({"events": scenario()[:i + 1]}, **context())
        assert model.predict([prefix])[0][-1] == pytest.approx(model.predict([full])[0][i], abs=1e-15)


def test_capacity_failure_preserves_all_state():
    stream = CapabilityRiskStream(CapabilityActionBN(), **context(), max_variables=2)
    stream.push(full_action(0))
    before = stream.explain()
    with pytest.raises(ValueError, match="variable limit"):
        stream.push(full_action(1, "tool_call"))
    assert before == stream.explain()
    stream.push(checked(1))


def test_wrong_compiler_model_and_mixed_episode_registries_are_rejected():
    v3 = compile_action_trace({"events": [full_action(0, tool="shell.exec")]})
    v4 = compile_capability_trace({"events": [full_action(0)]}, **context())
    for model, trace in ((ActionScopedJointBN(), v4), (CapabilityActionBN(), v3)):
        with pytest.raises(ValueError, match="matching action compiler"):
            model.predict([trace])
    with pytest.raises(ValueError):
        ActionRiskStream(CapabilityActionBN())
    with pytest.raises(ValueError):
        CapabilityRiskStream(ActionScopedJointBN(), **context())
    from dataclasses import replace
    changed = replace(v4.steps[0], capability_registry_sha256="different")
    with pytest.raises(ValueError, match="fixed capability registry"):
        CapabilityActionBN().predict([replace(v4, steps=(*v4.steps, changed))])


@pytest.mark.parametrize("mutation", [
    lambda s: s["tools"][0].update(label=1),
    lambda s: s["tools"][0].update(supporting_quote="not in description"),
    lambda s: s["tools"][0].update(capability="safe"),
    lambda s: s["tools"][0].update(annotation_method="automatic_llm"),
    lambda s: s["tools"].append(s["tools"][0]),
    lambda s: s.update(access_rules=[]),
])
def test_registry_rejects_unqualified_annotations_and_non_public_fields(mutation):
    spec = registry().specification()
    mutation(spec)
    with pytest.raises(ValueError):
        ToolCapabilityRegistry(spec)


def test_registry_is_copied_at_construction_resolution_and_episode_start():
    spec = registry().specification()
    reg = ToolCapabilityRegistry(spec)
    spec["tools"][0]["capability"] = "unknown"
    reg.resolve("information_read")["capability"] = "unknown"
    assert reg.resolve("information_read")["capability"] == "information_read"
    stream = CapabilityRiskStream(CapabilityActionBN(), **context(tool_registry=reg))
    reg._entries["information_read"]["capability"] = "unknown"
    stream.push(action(0, tool="information_read"))
    assert stream.explain()["actions"][0]["capability_profile"]["capability"] == "information_read"


def gradient_traces():
    result = []
    for k in range(30):
        cap, role = CAPABILITIES[k // 5], KNOWLEDGE.states("role")[k % 5]
        expected = (k // 2) % 3
        workflow = None if expected == 0 else {"nodes": ["reader", "executor"], "edges": [["reader", "executor"]] if expected == 1 else []}
        events = [
            {"sequence": 0, "actor": "reader", "event_type": "input_received", "source_id": "d", "data_class": KNOWLEDGE.states("trust")[k % 3]},
            {"sequence": 1, "actor": "reader", "target": "executor", "event_type": "delegation" if k % 2 else "message_sent", "data_class": KNOWLEDGE.states("trust")[(k // 6) % 3], "edge_confidence": .7},
            full_action(2, tool=cap), checked(3, tool=cap), checked(4, "no_mismatch", tool=cap),
            full_action(5, "tool_call", tool=cap), action(6, action_id="partial", tool="unregistered")]
        result.append(compile_capability_trace({"events": events}, **context(workflow=workflow,
            roles={"reader": role, "executor": role}, available_tools={"executor": [cap]} if k % 2 else None)))
    return result


@pytest.mark.parametrize("label,prefix", [(0, True), (1, True), (0, False), (1, False)])
def test_all_150_parameter_gradients_match_finite_differences(label, prefix):
    model = CapabilityActionBN()
    assert len(model.theta) == len(PARAMETER_NAMES) == 150
    assert len(set(PARAMETER_NAMES)) == 150
    assert CAP_HEAD == 127 and CAP_CONFIDENCE == 126
    traces = gradient_traces()
    theta = model.theta + np.random.default_rng(24).normal(0, .2, 150)
    labels = np.full(len(traces), label)
    loss, analytic = model.loss_gradient(theta, traces, labels, prefix)
    numeric = approx_derivative(lambda t: model.loss_gradient(t, traces, labels, prefix)[0], theta, method="3-point").ravel()
    assert np.isfinite(loss)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-8)


def test_joint_fitting_improves_fixture_loss_without_claiming_predictive_performance():
    traces = gradient_traces()[:6]
    labels = np.array([0, 0, 0, 1, 1, 1])
    model = CapabilityActionBN(max_iter=20)
    before = model.loss_gradient(model.theta, traces, labels)[0]
    model.fit(traces, labels)
    assert model.loss_gradient(model.theta, traces, labels)[0] < before
    assert model.parameters()["name"] == "CapabilityActionBN-v4"


def test_independent_dictionary_enumeration_matches_joint_model():
    """No use of production transitions, state_bits or action_features."""
    from collections import defaultdict
    model = CapabilityActionBN()
    model.theta += np.random.default_rng(9).normal(0, .5, 150)
    trace = compile_capability_trace({"events": scenario()}, **context())
    distribution, expected = {tuple([0] * trace.n_variables): 1.}, []
    for step in trace.steps:
        for op in step.operations:
            q = 1. if op.rule == "copy" else expit(model.theta[op.parameter] +
                (model.theta[CAP_CONFIDENCE] * op.confidence_feature if op.is_message else 0))
            changed = defaultdict(float)
            for state, mass in distribution.items():
                if op.rule == "replace":
                    for value, weight in ((0, 1 - q), (1, q)):
                        updated = list(state)
                        updated[op.receiver] = value
                        changed[tuple(updated)] += mass * weight
                elif not state[op.receiver] and (op.sender is None or state[op.sender]):
                    updated = list(state)
                    updated[op.receiver] = 1
                    changed[state] += mass * (1 - q)
                    changed[tuple(updated)] += mass * q
                else:
                    changed[state] += mass
            distribution = dict(changed)
        risk = 0.
        for state, mass in distribution.items():
            n = len(step.observed_nodes)
            count = sum(state[i] for i in step.observed_nodes)
            fraction = lambda ids: sum(state[i] for i in ids) / max(1, len(ids))
            features = [count / max(1, n), count * (count - 1) / max(1, n * (n - 1)),
                fraction(step.capable_nodes), fraction(step.terminal_nodes),
                any(state[a] for a, _ in step.pending_pairs), any(state[a] for a, _ in step.dispatched_pairs),
                any(state[a] and state[c] for a, c in step.pending_pairs),
                any(state[a] and state[c] for a, c in step.dispatched_pairs)]
            values = [*step.observables, *features]
            risk += mass * expit(sum(v * w for v, w in zip(values, model.theta[CAP_HEAD:])))
        expected.append(risk)
    scores, _, mass = model._forward(model.theta, trace)
    np.testing.assert_allclose(scores, expected, rtol=1e-13, atol=1e-14)
    for state, probability in distribution.items():
        index = sum(bit * (2 ** i) for i, bit in enumerate(state))
        assert mass[index] == pytest.approx(probability, abs=1e-14)
