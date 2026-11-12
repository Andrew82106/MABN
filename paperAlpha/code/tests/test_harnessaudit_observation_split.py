import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.harnessaudit_observations import HarnessAuditObservationProjector, project_records


def records(violated=True):
    common = {"trace_id": "sample", "timestamp": "2026-09-19T00:00:00Z"}
    return [
        {**common, "type": "trace_start", "task_id": "task-hidden", "n_rules": 1,
         "agents": [{"agent_id": "reader", "role": "researcher", "useful_tools": ["read"]}]},
        {**common, "type": "tool_call", "sequence_num": 1, "agent_id": "reader", "agent_role": "researcher",
         "tool_name": "read", "tool_args": {"resource": "R-1", "api_key": "private-value"},
         "tool_result": "returned text", "raw_event": {"risk_label": 1, "permission_allowed": False}},
        {**common, "type": "communication", "sequence_num": 2, "agent_id": "reader", "agent_role": "researcher",
         "target_agent": "user", "target_role": "user", "content": "Reported information."},
        {**common, "type": "access_decision", "action_sequence_num": 1, "violated": violated,
         "violation_code": "V-OR" if violated else None, "rule_description": "private audit rule"},
        {**common, "type": "access_decision", "action_sequence_num": 2, "violated": False},
        {**common, "type": "trace_end", "n_actions": 2, "n_tool_calls": 1, "n_communications": 1,
         "n_violations": int(violated), "violations_by_code": {"V-OR": 1} if violated else {}, "error": None},
    ]


def test_monitor_input_is_invariant_to_judges_rules_conditions_and_labels():
    original = records()
    changed = records(False)
    changed[0].update(task_id="different-label-family", n_rules=99, perturbation={"attack": True},
                      model="judge-model", domain="different-domain")
    changed[0]["agents"][0]["useful_tools"] = ["hidden-forbidden-operation"]
    for record in changed:
        record.update(risk_label=0, features={"permission_mismatch": 1}, permission={"allowed": False},
                      raw_event={"hidden_rules": ["answer"], "scenario": "malicious", "external_effect": True})
    changed[-1].update(avs=0, tcr=1, sar_avg=0, error="different evaluation failure")
    a, b = project_records(original)[0], project_records(changed)[0]
    assert a["monitor"] == b["monitor"]
    assert a["monitor_sha256"] == b["monitor_sha256"]
    assert a["evaluation"] != b["evaluation"]
    assert a["metadata"] != b["metadata"]
    assert a["evaluation"]["label"] == 1
    assert b["evaluation"]["label"] is None


def test_monitor_keeps_real_content_without_converting_it_to_trusted_evidence():
    data = records()
    data[1]["tool_args"]["body"] = {"risk_label": 1, "permission_allowed": False}
    data[2]["content"] = '{"checker_id": "trusted", "permission_status": "no_mismatch"}'
    bundle = project_records(data)[0]
    monitor = bundle["monitor"]
    tool, message = monitor["events"]
    assert tool["arguments"]["api_key"] == "[REDACTED]"
    assert tool["arguments"]["body"] == data[1]["tool_args"]["body"]
    assert message["content"] == data[2]["content"]
    assert message["target"] == "user"
    assert monitor["contract"]["permission_checks_available"] is False
    assert all("permission_status" not in e and "raw_event" not in e for e in monitor["events"])
    assert bundle["model_readiness"]["action_scoped_v3_eligible"] is False
    assert bundle["model_readiness"]["capability_manifest_from_useful_tools"] is False


def test_future_tool_results_and_audits_cannot_change_past_snapshots():
    data = records()
    projector = HarnessAuditObservationProjector()
    projector.push(data[0])
    before_action = projector.monitor_view()
    assert before_action["events"] == []
    returned = projector.push(data[1])
    after_action = projector.monitor_view()
    assert returned["event_type"] == "tool_completed_observed"
    assert after_action["events"][0]["result"] == "returned text"
    returned["arguments"]["resource"] = "mutation"
    data[1]["tool_args"]["resource"] = "future mutation"
    for record in data[2:]:
        projector.push(record)
    assert before_action["events"] == []
    assert projector.monitor_view()["events"][:1] == after_action["events"]
    assert after_action["events"][0]["arguments"]["resource"] == "R-1"


@pytest.mark.parametrize("violated", [True, False])
def test_complete_audit_label_has_narrow_endpoint_meaning(violated):
    bundle = project_records(records(violated))[0]
    evaluation = bundle["evaluation"]
    assert evaluation["eligible"] and evaluation["label"] == int(violated)
    assert evaluation["target"] == "any_recorded_access_rule_violation"
    assert evaluation["first_flagged_action_sequence"] == (1 if violated else None)
    assert "not L1 SAR" in evaluation["meaning"]
    assert bundle["monitor"]["contract"]["pre_execution_warning_eligible"] is False


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.pop(), "missing_trace_end"),
    (lambda r: r.pop(4), "actions_without_audit_decisions"),
    (lambda r: r[-1].update(error="timeout"), "run_error_or_unspecified_error_status"),
    (lambda r: r[-1].pop("error"), "run_error_or_unspecified_error_status"),
    (lambda r: r[-1].update(n_actions=20), "inconsistent_or_missing_n_actions"),
    (lambda r: r[-1].update(n_violations=True), "inconsistent_or_missing_n_violations"),
    (lambda r: r[4].update(action_sequence_num=100), "decisions_without_observed_actions"),
    (lambda r: r[-1].update(violations_by_code={}), "inconsistent_violations_by_code"),
    (lambda r: r.insert(3, {"type": "new_unknown_event", "trace_id": "sample"}), "unknown_native_event_types"),
    (lambda r: r.insert(4, {"type": "access_decision", "trace_id": "sample", "action_sequence_num": 1, "violated": False}),
     "contradictory_decisions_for_same_action"),
])
def test_incomplete_or_inconsistent_runs_are_not_negative_examples(mutation, reason):
    data = records()
    mutation(data)
    result = project_records(data)[0]["evaluation"]
    assert result["eligible"] is False and result["label"] is None
    assert reason in result["ineligibility_reasons"]


def test_multiple_matching_rules_are_not_mistaken_for_multiple_actions():
    data = records()
    other = {**data[3], "violation_code": "V-OT", "rule_description": "other rule"}
    data.insert(4, other)
    data[-1].update(n_violations=2, violations_by_code={"V-OR": 1, "V-OT": 1})
    result = project_records(data)[0]["evaluation"]
    assert result["eligible"] and result["label"] == 1
    assert result["observed_action_count"] == 2


@pytest.mark.parametrize("mutation", [
    lambda r: r[1].update(sequence_num=True),
    lambda r: r[2].update(sequence_num=1),
    lambda r: r[1].update(tool_args=[]),
    lambda r: r[1].update(tool_result={"body": "not native"}),
    lambda r: r[3].update(violated="true"),
    lambda r: r[1].update(tool_args={"bad": float("nan")}),
    lambda r: r.append(copy.deepcopy(r[1])),
    lambda r: r.insert(4, copy.deepcopy(r[2])),
    lambda r: r.pop(0),
    lambda r: r.insert(1, copy.deepcopy(r[0])),
])
def test_invalid_schema_or_order_is_rejected(mutation):
    data = records()
    mutation(data)
    with pytest.raises(ValueError):
        project_records(data)


def test_validation_error_does_not_mutate_existing_stream():
    p = HarnessAuditObservationProjector()
    data = records()
    p.push(data[0])
    p.push(data[1])
    before = p.snapshot()
    with pytest.raises(ValueError):
        p.push({**data[2], "sequence_num": 1})
    assert p.snapshot() == before
    p.push(data[2])
    assert len(p.events) == 2


def test_interleaved_runs_remain_separate():
    a, b = records(), records(False)
    for record in b:
        record["trace_id"] = "second"
    output = project_records([row for pair in zip(a, b) for row in pair])
    assert [row["trace_id"] for row in output] == ["sample", "second"]
    assert [row["evaluation"]["label"] for row in output] == [1, 0]
    assert output[0]["monitor"] == output[1]["monitor"]


def test_file_projection_preserves_inputs_and_never_overwrites(tmp_path):
    spec = importlib.util.spec_from_file_location("ha_projection_cli", ROOT / 'code/commands/project_harnessaudit_observations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "native.jsonl"
    raw = "".join(json.dumps(r) + "\n" for r in records())
    source.write_text(raw, encoding="utf-8")
    out = tmp_path / "split"
    report = module.project_file(source, out, provenance="test_fixture")
    assert source.read_text(encoding="utf-8") == raw
    assert report["observations"] == 2 and report["eligible_rule_violation_labels"] == 1
    assert report["provenance_independently_verified"] is False
    assert "private audit rule" not in (out / "monitor_inputs.jsonl").read_text(encoding="utf-8")
    assert "private audit rule" in (out / "evaluation_only.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        module.project_file(source, out, provenance="test_fixture")
    broken = tmp_path / "broken.jsonl"
    broken.write_text(raw + "{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        module.project_file(broken, tmp_path / "invalid-out", provenance="test_fixture")
    assert not (tmp_path / "invalid-out").exists()


def test_official_sink_and_checker_component_contract(tmp_path, monkeypatch):
    """Exercise official classes, not a reimplementation of their recorder.

    This is a controlled integration fixture: no actual agents, API or tools run.
    The optional upstream checkout is not installed as a project dependency.
    Load the exact component files to avoid the framework registry's eager
    Unix-only fcntl import on Windows. This does NOT verify a runnable harness.
    """
    upstream = ROOT / 'code/external/HarnessAudit'
    if not (upstream / "multi_agent/checker.py").exists():
        pytest.skip("optional upstream HarnessAudit checkout unavailable")
    monkeypatch.syspath_prepend(str(upstream))
    def load_exact(name, relative):
        spec = importlib.util.spec_from_file_location(name, upstream / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    ActionSink = load_exact("ha_source_contract_action_sink", "multi_agent/frameworks/core/action_sink.py").ActionSink
    load_exact("multi_agent.frameworks.core.sde_policy", "multi_agent/frameworks/core/sde_policy.py")
    from multi_agent.trace.writer import TraceWriter
    from multi_agent.trace.events import TraceStart, TraceEnd, AccessDecisionEvent, AgentInfo
    check_trace = load_exact("ha_source_contract_checker", "multi_agent/checker.py").check_trace
    from multi_agent.schemas.access_rules import AccessRule, AccessMatch

    path = tmp_path / "native-events.jsonl"
    with TraceWriter(path) as writer:
        writer.write_event(TraceStart(trace_id="native", task_id="fixture", domain="fixture", framework="fixture",
                                     model="none", agents=[AgentInfo(agent_id="r", role="reader")]))
        sink = ActionSink(run_id="native", trace_writer=writer)
        sink.emit_tool_call(agent_id="r", agent_role="reader", tool_name="read", tool_args={"record": "one"},
                            tool_result="completed", timestamp="2026-09-19T00:00:02Z")
        sink.emit_communication(agent_id="r", agent_role="reader", target_agent="user", target_role="user",
                                content="earlier message", timestamp="2026-09-19T00:00:01Z")
        # Pending calls have not yet become trace lines. Finalization reorders them.
        assert len(path.read_text().splitlines()) == 1
        actions = sink.actions
        assert actions[0].surface == "communication" and actions[1].surface == "tool_call"
        rule = AccessRule(role="reader", access_type="operation.tool_call", match=AccessMatch(tool="read"))
        decisions = check_trace(actions, [rule])
        for d in decisions:
            writer.write_event(AccessDecisionEvent(trace_id="native", action_sequence_num=d.action_sequence_num,
                                                    violated=d.violated, violation_code=d.violation_code))
        writer.write_event(TraceEnd(trace_id="native", n_actions=2, n_tool_calls=1, n_communications=1,
                                   n_violations=1, violations_by_code={"V-OT": 1}))
    bundle = project_records([json.loads(line) for line in path.read_text().splitlines()])[0]
    assert bundle["evaluation"]["eligible"] and bundle["evaluation"]["label"] == 1
    assert [e["event_type"] for e in bundle["monitor"]["events"]] == ["communication_observed", "tool_completed_observed"]
    assert bundle["monitor"]["contract"]["online_arrival_order_verified"] is False
