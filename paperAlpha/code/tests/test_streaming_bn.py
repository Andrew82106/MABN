import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
from paperalpha_monitor.streaming_bn import (  # noqa: E402
    EvidenceStream, N_CELLS, TemplateBN, replay, topology_group,
)


def event(sequence, actor, event_type, **kwargs):
    return {"sequence": sequence, "actor": actor, "event_type": event_type, **kwargs}


def chain():
    return [
        event(0, "a", "input_received", data_class="untrusted"),
        event(1, "a", "message_sent", target="b", data_class="untrusted"),
        event(2, "b", "tool_call", tool="shell.exec", permission_mismatch=True),
    ]


def test_appending_future_does_not_change_any_prefix():
    original = {"events": chain()}
    changed = {"agents": ["future_sink"], "events": chain() + [
        event(3, "new_source", "input_received", data_class="untrusted"),
        event(4, "new_source", "tool_call", tool="payments.write"),
    ]}
    for k in range(1, 4):
        expected = replay({"events": original["events"][:k]})[-1]
        actual = replay(changed)[k - 1]
        assert expected.named_evidence == actual.named_evidence
        np.testing.assert_equal(expected.vector(), actual.vector())
        assert "new_source" not in actual.named_evidence["observed_agents"]


def test_future_sink_is_not_visible_at_message_time():
    snapshots = replay({"events": chain()})
    assert snapshots[1].named_evidence["observed_actions"] == []
    assert all(f["template"] == "global_context" for f in snapshots[1].factors)
    assert len(snapshots[-1].named_evidence["observed_actions"]) == 1


def test_source_received_later_cannot_travel_backward():
    events = [
        event(0, "a", "message_sent", target="b", data_class="untrusted"),
        event(1, "a", "input_received", data_class="untrusted"),
        event(2, "b", "tool_call", tool="shell.exec"),
    ]
    assert not any(f["template"] == "information_flow" for f in replay({"events": events})[-1].factors)


def test_reverse_message_order_is_not_a_path():
    events = [
        event(0, "a", "input_received", data_class="untrusted"),
        event(1, "b", "message_sent", target="c", data_class="untrusted"),
        event(2, "a", "message_sent", target="b", data_class="untrusted"),
        event(3, "c", "tool_call", tool="shell.exec"),
    ]
    assert not any(f["template"] == "information_flow" for f in replay({"events": events})[-1].factors)


def test_runtime_cycle_has_chronological_witness_and_no_future_loop():
    events = chain()[:2] + [
        event(2, "b", "message_sent", target="a", data_class="untrusted"),
        event(3, "b", "tool_call", tool="shell.exec"),
    ]
    snapshot = replay({"events": events})[-1]
    path = next(f for f in snapshot.factors if f["template"] == "information_flow")["observed_paths"]["entry_agent:a"]
    assert path == [0, 1]


def test_workflow_cannot_supply_a_missing_message():
    workflow = {"nodes": ["a", "b"], "edges": [["a", "b"]]}
    events = [chain()[0], event(1, "b", "tool_call", tool="shell.exec")]
    snapshot = replay({"events": events}, workflow=workflow)[-1]
    assert not any(f["template"] == "information_flow" for f in snapshot.factors)


def test_absent_workflow_is_not_all_unexpected():
    events = [event(0, "a", "message_sent", target="b", data_class="clean")]
    unknown = replay({"events": events})[-1]
    known = replay({"events": events}, workflow={"nodes": ["a", "b"], "edges": []})[-1]
    assert unknown.named_evidence["event_counts"]["unexpected_messages"] == 0
    assert known.named_evidence["event_counts"]["unexpected_messages"] == 1


def test_runtime_only_ignores_workflow_derived_edge_annotation():
    events = [event(0, "a", "message_sent", target="b", data_class="clean")]
    changed = copy.deepcopy(events)
    changed[0]["edge_kind"] = "unexpected"
    np.testing.assert_equal(replay({"events": events})[-1].vector(),
                            replay({"events": changed})[-1].vector())


@pytest.mark.parametrize("second_source, multiple", [("doc1", 0), ("doc2", 1)])
def test_explicit_origin_identity_across_different_agents(second_source, multiple):
    events = [
        event(0, "a", "input_received", data_class="untrusted", source_id="doc1"),
        event(1, "b", "input_received", data_class="untrusted", source_id=second_source),
        event(2, "a", "message_sent", target="c", data_class="untrusted"),
        event(3, "b", "message_sent", target="c", data_class="untrusted"),
        event(4, "c", "tool_call", tool="shell.exec"),
    ]
    flow = next(f for f in replay({"events": events})[-1].factors if f["template"] == "information_flow")
    assert flow["parents"][1] == multiple
    assert len(flow["observed_paths"]) == multiple + 1


def test_retries_and_proposal_dispatch_do_not_double_count_actions():
    stream = EvidenceStream()
    proposed = event(0, "a", "tool_proposed", tool="shell.exec", action_id="x", event_id="e0")
    first = stream.push(proposed)
    np.testing.assert_equal(first.vector(), stream.push(proposed).vector())
    final = stream.push(event(1, "a", "tool_call", tool="shell.exec", action_id="x"))
    assert len(final.named_evidence["observed_actions"]) == 1
    assert final.named_evidence["observed_actions"][0]["phase"] == "tool_call"


@pytest.mark.parametrize("phase, actor, tool", [
    ("tool_call", "a", "shell.exec"),
    ("tool_proposed", "b", "shell.exec"),
    ("tool_proposed", "a", "payments.write"),
])
def test_rejected_action_update_does_not_mutate_state(phase, actor, tool):
    stream = EvidenceStream()
    before = stream.push(event(0, "a", phase, tool="shell.exec", action_id="x"))
    with pytest.raises(ValueError, match="action_id"):
        stream.push(event(1, actor, "tool_call", tool=tool, action_id="x", event_id="bad"))
    after = stream.snapshot()
    assert before.sequence == after.sequence
    assert before.named_evidence == after.named_evidence
    np.testing.assert_equal(before.vector(), after.vector())
    assert "bad" not in stream.seen_events


def test_later_message_cannot_rewrite_an_earlier_action():
    events = [chain()[0], event(1, "b", "tool_call", tool="shell.exec"),
              event(2, "a", "message_sent", target="b", data_class="untrusted")]
    final = replay({"events": events})[-1]
    assert final.named_evidence["paths_by_agent"]["b"]
    assert not any(f["template"] == "information_flow" for f in final.factors)


@pytest.mark.parametrize("capabilities", [{"a": "shell.exec"}, {"a": [3]}])
def test_malformed_capability_manifest_rejected(capabilities):
    with pytest.raises(ValueError, match="allowed_tools"):
        EvidenceStream(allowed_tools=capabilities)


def test_permission_unknown_is_not_matched_and_masking_is_invariant():
    events = [event(0, "a", "tool_call", tool="shell.exec")]
    unknown = replay({"events": events})[-1]
    matched_events = copy.deepcopy(events)
    matched_events[0]["permission_mismatch"] = False
    matched = replay({"events": matched_events})[-1]
    assert unknown.factors[1]["parents"][0] == 0
    assert matched.factors[1]["parents"][0] == 1
    np.testing.assert_equal(unknown.vector(), replay({"events": matched_events}, permission_mode="masked")[-1].vector())


@pytest.mark.parametrize("field", ["external_effect", "tool_side_effect", "label", "mechanism"])
def test_nested_label_channels_rejected(field):
    with pytest.raises(ValueError, match="evaluation-only"):
        replay({"events": [event(0, "a", "tool_call", metadata={field: 1})]})


def test_capability_manifest_is_explicit_and_validated_separately():
    events = [event(0, "a", "tool_call", tool="shell.exec")]
    snapshot = replay({"events": events}, allowed_tools={"a": ["search.read"]})[-1]
    assert snapshot.factors[1]["parents"][1] == 2


def test_late_or_conflicting_event_rejected():
    stream = EvidenceStream()
    first = event(2, "a", "input_received", event_id="x")
    stream.push(first)
    with pytest.raises(ValueError, match="strictly increase"):
        stream.push(event(1, "a", "input_received"))
    with pytest.raises(ValueError, match="conflicting"):
        stream.push({**first, "data_class": "untrusted"})


def test_same_source_repeated_is_not_multiple_sources():
    events = chain()[:2] + [
        event(2, "a", "message_sent", target="b", data_class="untrusted"),
        event(3, "b", "tool_call", tool="shell.exec"),
    ]
    snapshot = replay({"events": events})[-1]
    flow = next(f for f in snapshot.factors if f["template"] == "information_flow")
    assert flow["parents"][1] == 0


def test_topology_group_ignores_names_but_not_known_cycle():
    w1 = {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}
    w2 = {"nodes": ["z", "x", "y"], "edges": [["x", "y"], ["z", "x"]]}
    assert topology_group(w1) == topology_group(w2)
    with pytest.raises(ValueError, match="DAG"):
        EvidenceStream(workflow={"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]})


def test_episode_objective_gradient_and_exact_decomposition():
    snapshots = [replay({"events": chain()})[-1], replay({"events": [
        event(0, "a", "tool_call", tool="shell.exec", permission_mismatch=False)]})[-1]]
    model = TemplateBN()
    x = model.matrix(snapshots)
    y = np.asarray([1, 0])
    theta = np.linspace(-4, -1, N_CELLS)
    _, analytic = model.loss_gradient(theta, x, y)
    numeric = approx_derivative(lambda t: model.loss_gradient(t, x, y)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
    model.fit(snapshots, y)
    explanation = model.explain(snapshots[0])
    h = sum(row["negative_log_survival"] for row in explanation["factors"])
    assert explanation["risk_probability"] == pytest.approx(-np.expm1(-h))
    assert model.predict(snapshots)[0] == pytest.approx(explanation["risk_probability"])
    assert explanation["registry_id"] == "streaming-risk-templates-v1"
    assert any(node["kind"] == "risk" for node in explanation["compiled_graph"]["nodes"])
    assert model.optimizer["success"]


def test_explanation_graph_respects_template_ablation():
    snapshot = replay({"events": chain()})[-1]
    model = TemplateBN(exclude=frozenset({"authorization"}))
    explanation = model.explain(snapshot)
    risks = [node for node in explanation["compiled_graph"]["nodes"] if node["kind"] == "risk"]
    assert risks
    assert not any(node["template"] == "authorization" for node in risks)
    total = sum(np.logaddexp(0, model.theta[node["cpt"]["selected_cell"]]) for node in risks)
    assert -np.expm1(-total) == pytest.approx(model.predict([snapshot])[0])


def test_weighted_objective_gradient_and_duplicate_invariance():
    snapshots = [replay({"events": chain()})[-1], replay({"events": [
        event(0, "a", "tool_call", tool="shell.exec", permission_mismatch=False)]})[-1]]
    model = TemplateBN()
    counts = model.matrix(snapshots)
    labels = np.asarray([1, 0])
    weights = np.asarray([0.4, 0.6])
    theta = np.linspace(-5, -1, N_CELLS)
    loss, analytic = model.loss_gradient(theta, counts, labels, weights)
    numeric = approx_derivative(lambda t: model.loss_gradient(t, counts, labels, weights)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
    duplicated_loss, duplicated_grad = model.loss_gradient(
        theta, counts[[0, 0, 1]], labels[[0, 0, 1]], np.asarray([0.2, 0.2, 0.6]))
    assert duplicated_loss == pytest.approx(loss)
    np.testing.assert_allclose(duplicated_grad, analytic)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan"), 1], [1]])
def test_invalid_training_weights_rejected(weights):
    snapshots = [replay({"events": chain()})[-1]] * 2
    with pytest.raises(ValueError, match="sample_weight"):
        TemplateBN().fit(snapshots, np.asarray([1, 0]), sample_weight=np.asarray(weights))
