import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.dynamic_joint_bn import DynamicJointBN, KNOWLEDGE, N_PARAMETERS, compile_trace
from paperalpha_monitor.joint_graph import compile_joint_graph
from paperalpha_monitor.joint_knowledge import JointKnowledgeRegistry, load_joint_registry


def specification():
    return json.loads((ROOT / 'knowledge/templates/joint_risk_templates_v1.json').read_text(encoding="utf-8"))


def catalog():
    return json.loads((ROOT / 'knowledge/risks/risk_catalog.json').read_text(encoding="utf-8"))


def trace():
    events = [
        {"sequence": 0, "actor": "a", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 1, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted", "edge_confidence": .7},
        {"sequence": 2, "actor": "a", "target": "c", "event_type": "delegation", "data_class": "clean"},
        {"sequence": 3, "actor": "b", "target": "d", "event_type": "message_sent", "data_class": "unknown"},
        {"sequence": 4, "actor": "c", "target": "d", "event_type": "message_sent", "data_class": "untrusted"},
        {"sequence": 5, "actor": "d", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": True},
    ]
    return compile_trace({"events": events}, workflow={"nodes": ["a", "b", "c", "d"], "edges": [["a", "b"], ["b", "d"]]},
                         roles={"a": "reader", "b": "reasoner", "c": "reviewer", "d": "executor"})


def test_layout_and_initialization_reproduce_v1_parameterization():
    registry = load_joint_registry()
    assert len(registry.parameter_names) == N_PARAMETERS == 69
    assert registry.offsets == {"source": 0, "adoption": 3, "authorization": 18, "transmission": 33}
    assert registry.confidence_offset == 51 and registry.head_offset == 52
    expected = np.r_[[-4, -1, -2], np.tile([-2, .5, -1], 5), np.tile([-3, -4, -1], 5),
                     np.repeat([-2, 1, -.5], 6), 0, -.7, np.zeros(12), np.full(4, .5)]
    np.testing.assert_equal(registry.initial_values, expected)
    assert registry.cell_index("authorization", {"role": "executor", "permission": "mismatch"}) == 32


@pytest.mark.parametrize("change, message", [
    (lambda s: s["templates"][0]["knowledge_links"][0].update(source_ids=[6]), "source ID"),
    (lambda s: s["templates"][1].update(update="seed_or"), "binding/update"),
    (lambda s: s["templates"][0].update(initial_logits=[0]), "initial logit"),
    (lambda s: s["head"]["observables"].reverse(), "head features/order"),
    (lambda s: s["axes"]["permission"]["states"].reverse(), "axis states/order"),
])
def test_misbound_knowledge_or_semantics_fail_instead_of_relabeling(change, message):
    spec = specification()
    change(spec)
    with pytest.raises(ValueError, match=message):
        JointKnowledgeRegistry(spec, catalog())


def test_operation_evidence_must_match_selected_template_row():
    compiled = trace()
    op = compiled.steps[1].operations[0]
    assert KNOWLEDGE.describe_operation(op)["parents"]["trust"] == "untrusted"
    altered = copy.deepcopy(op)
    altered.evidence["trust"] = "clean"
    with pytest.raises(ValueError, match="evidence disagrees"):
        KNOWLEDGE.describe_operation(altered)


@pytest.mark.parametrize("kwargs", [{}, {"use_messages": False}, {"use_latent_head": False}])
def test_executable_graph_matches_joint_model_without_calling_its_forward(kwargs, monkeypatch):
    compiled = trace()
    model = DynamicJointBN(**kwargs)
    model.theta = np.random.default_rng(17).normal(size=N_PARAMETERS)
    expected = model.predict([compiled])[0]
    graph = compile_joint_graph(compiled)
    def forbidden(*args, **kw):
        raise AssertionError("graph interpreter must not call model forward")
    monkeypatch.setattr(DynamicJointBN, "_forward", forbidden)
    np.testing.assert_allclose(graph.predict(model.theta, **kwargs), expected, atol=1e-12)
    exported = graph.to_dict()
    gates = [n for n in exported["nodes"] if n["kind"] == "gate"]
    assert all(n["knowledge_links"] for n in gates)
    assert {n["template"] for n in gates} == {"source", "adoption", "authorization", "transmission"}
    # Shared origin has one initializer and one seed update, not one per branch.
    origin_states = [n for n in exported["nodes"] if n["kind"] == "state" and n["scope"]["kind"] == "origin_hypothesis"]
    assert len(origin_states) == 2


def test_graph_loss_derivative_agrees_with_trainable_model_gradient():
    compiled = trace()
    graph = compile_joint_graph(compiled)
    theta = np.linspace(-.7, .8, N_PARAMETERS)
    model = DynamicJointBN(regularization=0)
    def graph_loss(values):
        p = graph.predict(values)
        return -np.log(p).mean()
    numerical = approx_derivative(graph_loss, theta).ravel()
    _, gradient = model.episode_loss_gradient(theta, compiled, 1)
    np.testing.assert_allclose(numerical, gradient, atol=1e-8, rtol=1e-5)


def test_model_explanation_contains_an_executable_not_decorative_graph():
    compiled, model = trace(), DynamicJointBN()
    explanation = model.explain(compiled, include_graph=True)
    assert explanation["risk_probability"] == pytest.approx(explanation["graph_risk_probability"], abs=1e-12)
    assert explanation["knowledge_graph"]["registry"]["registry_id"] == KNOWLEDGE.registry_id


def test_graph_export_is_detached_and_contains_no_future_of_supplied_prefix():
    compiled = compile_trace({"events": [{"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"}]})
    graph = compile_joint_graph(compiled)
    first = graph.to_dict()
    assert {v["id"] for v in first["variables"]} == {"a", "entry_agent:a"}
    first["nodes"].clear()
    first["registry"]["templates"].clear()
    assert graph.to_dict()["nodes"] and graph.to_dict()["registry"]["templates"]
