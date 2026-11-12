import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from paperalpha_monitor import DynamicJointBN, TypedJointBN, TypedRiskStream
from paperalpha_monitor.typed_joint_bn import compile_typed_trace
import paperalpha_monitor.typed_joint_bn as typed_core
from paperalpha_monitor.dynamic_joint_bn import state_bits
from monitor_joint_events import load_model, monitor_lines


def events():
    # Deliberately introduce new nodes/sources AFTER an authorization variable.
    # Online bit order must not confuse the new node with that existing state.
    return [
        {"sequence": 0, "actor": "a", "event_type": "tool_proposed", "tool": "shell.exec", "action_id": "a-1", "permission_mismatch": True},
        {"sequence": 1, "actor": "b", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 2, "actor": "b", "target": "a", "event_type": "message_sent", "data_class": "untrusted"},
        {"sequence": 3, "actor": "a", "target": "c", "event_type": "delegation", "data_class": "unknown"},
        {"sequence": 4, "actor": "a", "event_type": "tool_call", "tool": "shell.exec", "action_id": "a-1", "permission_mismatch": True},
        {"sequence": 5, "actor": "c", "event_type": "input_received", "source_id": "doc", "data_class": "untrusted"},
        {"sequence": 6, "actor": "c", "event_type": "tool_call", "tool": "payments.write", "permission_mismatch": False},
        {"sequence": 7, "actor": "c", "target": "d", "event_type": "message_sent", "data_class": "clean"},
    ]


def identity(variable):
    return variable["kind"], variable["id"]


def projected_distribution(distribution, variables, target_variables):
    bits = state_bits(len(variables)).astype(np.int64)
    indices = {identity(v): i for i, v in enumerate(variables)}
    positions = [indices[identity(v)] for v in target_variables]
    bucket = bits[:, positions] @ (1 << np.arange(len(positions)))
    return np.bincount(bucket, weights=distribution, minlength=1 << len(positions))


@pytest.mark.parametrize("permissions", ["observed", "masked"])
@pytest.mark.parametrize("with_workflow", [False, True])
@pytest.mark.parametrize("typed_head", [False, True])
def test_every_prefix_matches_batch_probabilities_and_full_joint_mass(permissions, with_workflow, typed_head):
    context = {"permission_mode": permissions}
    if with_workflow:
        context.update(workflow={"nodes": ["b", "a", "c"], "edges": [["b", "a"], ["a", "c"]]},
                       roles={"a": "executor", "b": "reader", "c": "reviewer"}, allowed_tools={"a": ["shell.exec"], "c": ["payments.write"]})
    model = TypedJointBN(typed_head=typed_head)
    model.theta += np.random.default_rng(312).normal(0, .3, len(model.theta))
    stream = TypedRiskStream(model, **context)
    for i, event in enumerate(events()):
        result = stream.push(event)
        compiled = compile_typed_trace({"events": events()[:i + 1]}, **context)
        probabilities, _, distribution = model._forward(model.theta, compiled)
        assert result["risk_probability"] == pytest.approx(probabilities[-1], abs=1e-13)
        projected = projected_distribution(distribution, compiled.variables, stream._compiler.variables)
        np.testing.assert_allclose(stream._distribution, projected, atol=1e-14, rtol=1e-14)
        explanation = stream.explain()
        assert sum(x["risk_contribution"] for x in explanation["system_mixture_terms"]) + explanation["remaining_risk_contribution"] == pytest.approx(result["risk_probability"], abs=1e-13)
        assert not any("Batch/prefix compiler only" in x for x in explanation["limitations"])


def test_online_never_recompiles_or_replays_batch_history(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("incremental interface cannot replay the batch compiler or filter")
    monkeypatch.setattr(typed_core, "compile_typed_trace", forbidden)
    monkeypatch.setattr(typed_core, "type_trace", forbidden)
    monkeypatch.setattr(DynamicJointBN, "_forward", forbidden)
    stream = TypedRiskStream(TypedJointBN())
    for event in events():
        assert 0 <= stream.push(event)["risk_probability"] <= 1


def test_future_nodes_are_not_preallocated_and_identity_order_is_stable():
    stream = TypedRiskStream(TypedJointBN())
    stream.push(events()[0])
    first = stream.explain()["variables"]
    assert [identity(v) for v in first] == [("information_state", "a"), ("authorization_state", "a")]
    stream.push(events()[1])
    later = stream.explain()["variables"]
    assert [identity(v) for v in later[:2]] == [identity(v) for v in first]
    assert identity(later[2]) == ("information_state", "b")


def test_typed_capacity_rejection_does_not_consume_event_or_action():
    stream = TypedRiskStream(TypedJointBN(), max_variables=2)
    source = {"sequence": 0, "actor": "a", "event_type": "input_received", "source_id": "doc"}
    stream.push(source)
    before = stream.explain()
    with pytest.raises(ValueError, match="event was not accepted"):
        stream.push({**events()[0], "sequence": 1})
    assert stream.explain() == before
    assert not stream._compiler._base.validator.actions
    assert stream.push({"sequence": 1, "actor": "a", "event_type": "tool_call", "tool": "search.read"})["accepted_event_count"] == 2


@pytest.mark.parametrize("bad", [
    {**events()[1], "source_id": []},
    {**events()[1], "data_class": []},
    {**events()[2], "sequence": 1, "edge_confidence": -1},
    {**events()[0], "sequence": 1, "permission_mismatch": "yes"},
    {**events()[0], "sequence": 1, "actor": "b"},
    {**events()[0], "sequence": 0},
])
def test_invalid_event_is_atomic(bad):
    stream = TypedRiskStream(TypedJointBN())
    stream.push(events()[0])
    before = stream.explain()
    with pytest.raises((ValueError, TypeError)):
        stream.push(bad)
    assert stream.explain() == before
    assert stream.push(events()[1])["accepted_event_count"] == 2


def test_duplicate_proposal_and_dispatch_do_not_allocate_or_inject_twice():
    stream = TypedRiskStream(TypedJointBN(), threshold=0.)
    proposal = {**events()[0], "event_id": "proposal-0"}
    one = stream.push(proposal)
    before = stream._distribution.copy()
    duplicate = stream.push(copy.deepcopy(proposal))
    assert duplicate["duplicate_event"] and duplicate["accepted_event_count"] == 1
    np.testing.assert_equal(before, stream._distribution)
    dispatch = {**events()[4], "event_id": "call-0"}
    result = stream.push(dispatch)
    assert result["first_alarm_sequence"] == one["first_alarm_sequence"] == 0
    assert len(stream._compiler.variables) == 2
    assert stream.explain()["latest_event_transitions"] == []
    np.testing.assert_equal(before, stream._distribution)


def test_model_and_context_changes_do_not_reconfigure_live_stream():
    model = TypedJointBN()
    context = {"roles": {"a": "executor"}}
    stream = TypedRiskStream(model, **context)
    reference = TypedRiskStream(copy.deepcopy(model), **copy.deepcopy(context))
    model.theta[:] = 8
    model.typed_head = False
    context["roles"]["a"] = "reader"
    for event in events():
        assert stream.push(event) == reference.push(event)


def test_no_event_and_wrong_model_never_report_safety():
    with pytest.raises(ValueError, match="no event"):
        TypedRiskStream(TypedJointBN()).snapshot()
    with pytest.raises(ValueError, match="requires"):
        TypedRiskStream(DynamicJointBN())
    assert TypedRiskStream(TypedJointBN()).push(events()[0])["alarm"] is None


def test_typed_json_loading_cli_and_rejection_recovery(tmp_path):
    path = tmp_path / "typed.json"
    path.write_text(json.dumps({"typed_joint_bn": {"bn": TypedJointBN().parameters()}}), encoding="utf-8")
    loaded = load_model(path, "typed_joint_bn")
    assert isinstance(loaded, TypedJointBN)
    command = [sys.executable, "-B", str(ROOT / 'code/commands/monitor_joint_events.py'), "--model-parameters", str(path),
               "--method", "typed_joint_bn", "--threshold", "0", "--explain"]
    result = subprocess.run(command, input=json.dumps(events()[0]) + "\n{invalid\n" + json.dumps(events()[1]) + "\n",
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 1, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["status"] for r in rows] == ["accepted", "rejected", "accepted"]
    assert rows[0]["model_name"] == "TypedJointBN-v2" and rows[0]["alarm"]
    assert rows[1]["risk_probability"] is None and rows[1]["alarm"] is None
    assert rows[2]["accepted_event_count"] == 2
    assert rows[2]["first_alarm_sequence"] == 0
    value = json.loads(path.read_text())
    value["typed_joint_bn"]["bn"]["typed_registry"]["specification_sha256"] = "wrong"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="specification"):
        load_model(path, "typed_joint_bn")
