import copy
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.dynamic_joint_bn import DynamicJointBN, compile_trace
from paperalpha_monitor.typed_joint_bn import TypedJointBN, compile_typed_trace, PARAMETER_NAMES, SPEC_SHA256
import paperalpha_monitor.typed_joint_bn as core
from paperalpha_monitor.joint_graph import compile_joint_graph
from paperalpha_monitor.typed_graph import CompiledTypedGraph, compile_typed_graph


def events():
    return [
        {"sequence": 0, "actor": "b", "event_type": "tool_proposed", "tool": "shell.exec", "action_id": "x", "permission_mismatch": True},
        {"sequence": 1, "actor": "a", "event_type": "input_received", "source_id": "shared", "data_class": "untrusted"},
        {"sequence": 2, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted", "edge_confidence": .8},
        {"sequence": 3, "actor": "a", "target": "c", "event_type": "delegation", "data_class": "unknown"},
        {"sequence": 4, "actor": "b", "target": "d", "event_type": "message_sent", "data_class": "clean"},
        {"sequence": 5, "actor": "c", "target": "d", "event_type": "message_sent", "data_class": "untrusted"},
        {"sequence": 6, "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "action_id": "x", "permission_mismatch": True},
        {"sequence": 7, "actor": "d", "event_type": "input_received", "source_id": "shared", "data_class": "untrusted"},
        {"sequence": 8, "actor": "d", "event_type": "tool_call", "tool": "payments.write", "permission_mismatch": False},
    ]


def context():
    return {"workflow": {"nodes": ["a", "b", "c", "d"], "edges": [["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]]},
            "roles": {"a": "reader", "b": "reasoner", "c": "reviewer", "d": "executor"},
            "allowed_tools": {"b": ["shell.exec"], "d": ["payments.write"]}}


@pytest.mark.parametrize("typed_head", [True, False])
@pytest.mark.parametrize("use_messages", [True, False])
@pytest.mark.parametrize("use_latent_head", [True, False])
def test_graph_executes_exported_rules_without_using_model_forward_or_features(typed_head, use_messages, use_latent_head, monkeypatch):
    compiled = compile_typed_trace({"events": events()}, **context())
    model = TypedJointBN(typed_head=typed_head, use_messages=use_messages, use_latent_head=use_latent_head)
    model.theta = np.random.default_rng(851).normal(size=len(PARAMETER_NAMES))
    expected = model.predict([compiled])[0]
    graph = compile_typed_graph(compiled, typed_head=typed_head)
    record = json.loads(json.dumps(graph.to_dict()))
    # Round trip JSON-shaped fields; the interpreter keeps no original trace.
    reconstructed = CompiledTypedGraph(record["variables"], record["nodes"], record["edges"], record["context"])
    def forbidden(*a, **kw):
        raise AssertionError("graph must execute independently of the model/typed feature function")
    monkeypatch.setattr(DynamicJointBN, "_forward", forbidden)
    monkeypatch.setattr(core, "typed_features", forbidden)
    np.testing.assert_allclose(reconstructed.predict(model.theta, use_messages=use_messages, use_latent_head=use_latent_head), expected, atol=1e-13)
    assert record["typed_registry"]["specification_sha256"] == SPEC_SHA256
    assert record["parameter_names"] == PARAMETER_NAMES
    gates = [n for n in record["nodes"] if n["kind"] == "gate"]
    assert all(n["knowledge_links"] for n in gates)


@pytest.mark.parametrize("label", [0, 1])
def test_exported_graph_finite_difference_matches_joint_training_gradient(label):
    compiled = compile_typed_trace({"events": events()}, **context())
    model = TypedJointBN(regularization=0)
    theta = np.random.default_rng(874).normal(0, .3, len(model.theta))
    graph = compile_typed_graph(compiled)
    def loss(values):
        p = graph.predict(values)
        return (-label * np.log(p) - (1 - label) * np.log1p(-p)).mean()
    numeric = approx_derivative(loss, theta).ravel()
    _, actual = model.episode_loss_gradient(theta, compiled, label)
    np.testing.assert_allclose(numeric, actual, atol=2e-8, rtol=2e-5)


def test_export_contains_typed_auth_parents_but_messages_carry_only_information():
    compiled = compile_typed_trace({"events": events()}, **context())
    record = compile_typed_graph(compiled).to_dict()
    nodes = {n["id"]: n for n in record["nodes"]}
    for node in nodes.values():
        if node["kind"] == "state" and node["rule"] == "carrier_or":
            assert node["scope"]["kind"] == "information_state"
            assert nodes[node["parents"][2]]["scope"]["kind"] in {"information_state", "origin_hypothesis"}
        if node["kind"] == "system_risk":
            assert any(nodes[p].get("scope", {}).get("kind") == "authorization_state" for p in node["parents"])
    assert len([v for v in record["variables"] if v["kind"] == "origin_hypothesis"]) == 1
    # One auth gate for proposal+dispatch of x, another for d's separate action.
    assert len([n for n in nodes.values() if n.get("template") == "authorization"]) == 2


def test_prefix_graph_omits_future_identities_and_is_detached():
    compiled = compile_typed_trace({"events": events()[:1]})
    graph = compile_typed_graph(compiled)
    record = graph.to_dict()
    assert {v["id"] for v in record["variables"]} == {"b"}
    assert len(record["variables"]) == 2
    record["nodes"].clear()
    record["state_bindings"].clear()
    assert graph.to_dict()["nodes"] and graph.to_dict()["state_bindings"]


def test_wrong_compiler_or_typed_binding_is_rejected():
    raw = {"events": events()}
    base, typed = compile_trace(raw), compile_typed_trace(raw)
    with pytest.raises(ValueError, match="typed variable"):
        compile_typed_graph(base)
    with pytest.raises(ValueError, match="typed executable"):
        compile_joint_graph(typed)
    bad = copy.deepcopy(typed)
    bad.steps[0].operations[0].evidence["typed_state_binding"] = "information_state"
    with pytest.raises(ValueError, match="receiver binding"):
        compile_typed_graph(bad)
    wrong_pair = replace(typed.steps[2], node_pairs=((0, 1),))
    with pytest.raises(ValueError, match="authorization binding"):
        compile_typed_graph(replace(typed, steps=(wrong_pair,)))
    with pytest.raises(ValueError, match="boolean"):
        compile_typed_graph(typed, typed_head="false")


def test_model_explanation_and_graph_share_the_actual_readout_mode():
    compiled = compile_typed_trace({"events": events()})
    model = TypedJointBN(typed_head=False, use_messages=False)
    model.theta[-2:] = 7.
    result = model.explain(compiled, include_graph=True)
    assert result["risk_probability"] == pytest.approx(result["graph_risk_probability"], abs=1e-13)
    assert result["knowledge_graph"]["context"]["typed_head"] is False


def test_graph_cannot_draw_edges_different_from_its_calculation():
    record = compile_typed_graph(compile_typed_trace({"events": events()})).to_dict()
    with pytest.raises(ValueError, match="edges must match"):
        CompiledTypedGraph(record["variables"], record["nodes"], record["edges"][:-1], record["context"])


def test_unknown_rule_is_not_silently_treated_as_a_seed():
    record = compile_typed_graph(compile_typed_trace({"events": events()})).to_dict()
    next(n for n in record["nodes"] if n["kind"] == "state" and n["rule"] != "constant_zero")["rule"] = "clear"
    graph = CompiledTypedGraph(record["variables"], record["nodes"], record["edges"], record["context"])
    with pytest.raises(ValueError, match="state rule"):
        graph.predict(TypedJointBN().theta)
