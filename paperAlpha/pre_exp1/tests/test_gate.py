from __future__ import annotations

from pre_exp1.gate import MessageGate
from pre_exp1.models import Message
from pre_exp1.state import stable_hash


def _message(runtime, edge_id=None, content=None) -> Message:
    edge = runtime.graph.edges[edge_id or runtime.graph.gated_edge_id]
    return Message(
        message_id="MSG-1",
        sender_id=edge.source,
        recipient_id=edge.target,
        edge_id=edge.edge_id,
        content=content
        or runtime.templates.get("dangerous_original").content,
    )


def _apply(treatment, event_log_factory, runtime, edge_id=None):
    gate = MessageGate.from_runtime(
        runtime.graph,
        runtime.treatments,
        runtime.templates,
    )
    log = event_log_factory()
    message = _message(runtime, edge_id=edge_id)
    result = gate.apply(
        message,
        treatment,
        event_log=log,
        episode_id="TEST-EP",
        state_hash="0" * 64,
    )
    return gate, message, result, list(log.read())


def test_original_does_not_change_frozen_input(
    event_log_factory, runtime
):
    _, message, result, _ = _apply("original", event_log_factory, runtime)
    assert message.content == runtime.templates.get("dangerous_original").content
    assert result.message.content == message.content
    assert result.input_hash == result.output_hash


def test_safe_uses_frozen_template(event_log_factory, runtime):
    gate, _, result, _ = _apply("safe", event_log_factory, runtime)
    assert result.message.content == runtime.templates.get("safe").content
    assert result.message.content == gate.safe_template
    assert "test_secret" not in result.message.content["requested_fields"]
    result.message.content["requested_fields"].append("mutation")
    _, _, second, _ = _apply("safe", event_log_factory, runtime)
    assert "mutation" not in second.message.content["requested_fields"]


def test_drop_uses_frozen_noop_template(event_log_factory, runtime):
    gate, original, result, _ = _apply("drop", event_log_factory, runtime)
    assert result.message.content == runtime.templates.get("drop").content
    assert result.message.content == gate.drop_template
    assert result.message.content != original.content
    assert result.message.content["noop"] is True
    assert "test_secret" not in str(result.message.content)


def test_gate_only_affects_graph_configured_edge(
    event_log_factory, runtime
):
    non_gated = next(
        edge_id
        for edge_id in runtime.graph.edges
        if edge_id != runtime.graph.gated_edge_id
    )
    _, message, result, events = _apply(
        "safe",
        event_log_factory,
        runtime,
        edge_id=non_gated,
    )
    assert result.message.content == message.content
    assert result.applied_treatment == "original"
    assert events[0]["details"]["targeted_edge"] is False
    assert events[0]["treatment"] == "safe"


def test_gate_records_correct_before_and_after_hashes(
    event_log_factory, runtime
):
    _, message, result, events = _apply("safe", event_log_factory, runtime)
    event = events[0]
    assert event["details"]["input_content_hash"] == stable_hash(message.content)
    assert event["details"]["output_content_hash"] == stable_hash(
        result.message.content
    )
    assert event["payload_hash"] == result.output_hash
    assert event["treatment"] == "safe"

