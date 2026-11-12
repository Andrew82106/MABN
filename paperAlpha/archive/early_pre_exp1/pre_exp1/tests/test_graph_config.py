from __future__ import annotations

import copy
import json

import pytest

from pre_exp1.gate import MessageGate
from pre_exp1.graph import GraphConfig, GraphConfigError, MessageRouteError
from pre_exp1.models import Message


def _graph_dict(runtime):
    return json.loads(runtime.graph_path.read_text(encoding="utf-8"))


def test_disabling_focal_edge_prevents_startup(runtime):
    config = _graph_dict(runtime)
    focal = next(
        edge
        for edge in config["edges"]
        if edge["edge_id"] == config["gated_edge"]
    )
    focal["allowed"] = False
    with pytest.raises(GraphConfigError, match="gated edge must be allowed"):
        GraphConfig.from_dict(config)


def test_nonexistent_gated_edge_prevents_startup(runtime):
    config = _graph_dict(runtime)
    config["gated_edge"] = "missing_edge"
    for edge in config["edges"]:
        edge["gated"] = False
    with pytest.raises(GraphConfigError, match="does not exist"):
        GraphConfig.from_dict(config)


def test_message_endpoints_must_match_edge(runtime):
    edge = runtime.graph.gated_edge
    message = Message(
        message_id="BAD-ROUTE",
        sender_id=edge.target,
        recipient_id=edge.source,
        edge_id=edge.edge_id,
        content=runtime.templates.get("dangerous_original").content,
    )
    with pytest.raises(MessageRouteError, match="do not match edge"):
        runtime.graph.validate_message(message)


@pytest.mark.parametrize("mutation", ["bad_order", "cycle"])
def test_invalid_execution_order_or_cycle_cannot_run(runtime, mutation):
    config = _graph_dict(runtime)
    if mutation == "bad_order":
        config["execution_order"][1], config["execution_order"][2] = (
            config["execution_order"][2],
            config["execution_order"][1],
        )
    else:
        config["edges"].append(
            {
                "edge_id": "cycle_edge",
                "source": config["execution_order"][-1],
                "target": config["execution_order"][0],
                "allowed": True,
                "gated": False,
            }
        )
    with pytest.raises(GraphConfigError, match="DAG topology"):
        GraphConfig.from_dict(config)


def test_changing_legal_gated_edge_changes_gate_target(
    runtime, event_log_factory
):
    config = _graph_dict(runtime)
    old_gate = config["gated_edge"]
    new_gate = config["edges"][0]["edge_id"]
    assert new_gate != old_gate
    config["gated_edge"] = new_gate
    for edge in config["edges"]:
        edge["gated"] = edge["edge_id"] == new_gate
    graph = GraphConfig.from_dict(config)
    treatments = copy.deepcopy(runtime.treatments)
    treatments["gated_edge"] = new_gate
    gate = MessageGate.from_runtime(graph, treatments, runtime.templates)
    content = runtime.templates.get("dangerous_original").content
    old_edge = graph.edges[old_gate]
    old_message = Message(
        message_id="OLD",
        sender_id=old_edge.source,
        recipient_id=old_edge.target,
        edge_id=old_edge.edge_id,
        content=content,
    )
    old_result = gate.apply(
        old_message,
        "safe",
        event_log=event_log_factory("GRAPH-OLD"),
        episode_id="EP",
        state_hash="0" * 64,
    )
    new_edge = graph.edges[new_gate]
    new_message = Message(
        message_id="NEW",
        sender_id=new_edge.source,
        recipient_id=new_edge.target,
        edge_id=new_edge.edge_id,
        content=content,
    )
    new_result = gate.apply(
        new_message,
        "safe",
        event_log=event_log_factory("GRAPH-NEW"),
        episode_id="EP",
        state_hash="0" * 64,
    )
    assert old_result.message.content == content
    assert old_result.applied_treatment == "original"
    assert new_result.message.content == runtime.templates.get("safe").content
    assert new_result.applied_treatment == "safe"
