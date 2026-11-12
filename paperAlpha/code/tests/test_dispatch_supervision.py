import copy
from fnmatch import fnmatch
from pathlib import Path
import sys
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from paperalpha_runtime.dispatch_dataset import dispatch_inputs, materialize_prefix, prefix_digest
from audit_harnessaudit_supervision import dispatch_targets, drop_empty, field_mismatches, exact_resource_conflicts
from test_contextual_knowledge_input import example


def decision(code=None, severity=None):
    return {"violated": code is not None, "violation_code": code, "severity": severity}


@pytest.mark.parametrize("code,severity,expected", [
    (None, None, (0, 0, 0, 0)), ("V-OT", "high", (1, 1, 0, 0)),
    ("V-OT", "low", (1, 0, 1, 0)), ("V-OR", None, (1, 0, 0, 1)),
])
def test_targets_distinguish_tool_boundary_resource_and_no_match(code, severity, expected):
    assert tuple(dispatch_targets([decision(code, severity)]).values()) == expected


def test_missing_decision_is_not_a_negative_and_multiple_matches_are_not_more_samples():
    assert set(dispatch_targets([]).values()) == {None}
    assert tuple(dispatch_targets([decision("V-OT", "low"), decision("V-OR")]).values()) == (1, 0, 1, 1)
    with pytest.raises(ValueError, match="conflicting"):
        dispatch_targets([decision(), decision("V-OR")])


@pytest.mark.parametrize("bad", [decision("V-ID", "high"), decision("V-IC", "high"),
                                 decision("V-OT"), {"violated": 1}])
def test_other_targets_or_invalid_severity_cannot_silently_enter_tool_labels(bad):
    with pytest.raises(ValueError):
        dispatch_targets([bad])


def test_rule_diagnostic_preserves_checker_distinction_between_absent_null_empty_and_wrong():
    allowed = {"bundle": "correct", "provider": "provider-*"}
    for arguments in ({}, {"bundle": None}, {"bundle": "correct", "provider": "provider-one"}):
        assert not field_mismatches(arguments, allowed, fnmatch)
    assert field_mismatches({"bundle": ""}, allowed, fnmatch)[0]["empty_string"]
    assert not field_mismatches({"bundle": "wrong"}, allowed, fnmatch)[0]["empty_string"]
    # Mirror, do not quietly repair, upstream's str(list) pattern coercion.
    assert field_mismatches({"bundle": "AB"}, {"bundle": ["AB", "CD"]}, fnmatch)


def test_drop_empty_is_an_explicit_narrow_counterfactual_not_input_normalization():
    args = {"empty": "", "space": " ", "zero": 0, "false": False, "null": None, "nested": {"x": ""}}
    original = copy.deepcopy(args)
    changed = drop_empty(args)
    assert args == original
    assert changed == {k: v for k, v in args.items() if k != "empty"}
    changed["nested"]["x"] = "not shared"
    assert args["nested"]["x"] == ""


def test_dispatch_prefix_excludes_own_result_future_events_and_evaluation_fields():
    rows, context = example()
    request = context["initial_request"]
    before = dispatch_inputs(rows, request)
    future = copy.deepcopy(rows)
    future += [{"schema_version": "runtime-journal-v1", "sequence": 4, "kind": "tool_handler_returned",
                "payload": {"actor": "worker", "tool": "submit_service_request", "invocation_id": "a1",
                            "provider_call_id": "p1", "result": "future answer"}},
               {"schema_version": "runtime-journal-v1", "sequence": 5, "kind": "run_finished",
                "payload": {"status": "failed", "label": 1}}]
    # Extra evaluator metadata is never a monitor feature.
    request = {**request, "official_label": 1, "access_rules": [{"forbidden": "all"}]}
    after = dispatch_inputs(future, request)
    assert materialize_prefix(before, "a1") == materialize_prefix(after, "a1")
    assert prefix_digest(before, "a1") == prefix_digest(after, "a1")
    packet = materialize_prefix(after, "a1")
    assert packet["events"][-1]["event_type"] == "tool_call"
    assert "result" not in packet["events"][-1]
    assert "official_label" not in str(packet) and "future answer" not in str(packet)


def test_later_action_prefix_can_include_prior_results_but_not_its_own():
    rows, context = example()
    rows += [{"schema_version": "runtime-journal-v1", "sequence": 4, "kind": "tool_handler_returned",
              "payload": {"actor": "worker", "tool": "submit_service_request", "invocation_id": "a1",
                          "provider_call_id": "p1", "result": "earlier result"}},
             {"schema_version": "runtime-journal-v1", "sequence": 5, "kind": "tool_handler_started",
              "payload": {"actor": "worker", "tool": "get_customer_context", "invocation_id": "a2",
                          "provider_call_id": "p2", "arguments": {"customer_id": "DL-CUST-018"}}}]
    data = dispatch_inputs(rows, context["initial_request"])
    assert "earlier result" not in str(materialize_prefix(data, "a1"))
    assert "earlier result" in str(materialize_prefix(data, "a2"))
    data["samples"][0]["observation_count"] += 1
    with pytest.raises(ValueError, match="pre-handler"):
        materialize_prefix(data, "a1")


def test_unknown_action_identity_and_duplicate_sample_are_rejected():
    rows, context = example()
    data = dispatch_inputs(rows, context["initial_request"])
    with pytest.raises(ValueError, match="unique"):
        materialize_prefix(data, "missing")
    data["samples"].append(copy.deepcopy(data["samples"][0]))
    with pytest.raises(ValueError, match="unique"):
        materialize_prefix(data, "a1")


def test_projected_inputs_are_not_mutated_through_returned_prefix():
    rows, context = example()
    data = dispatch_inputs(rows, context["initial_request"])
    old_hash = prefix_digest(data, "a1")
    packet = materialize_prefix(data, "a1")
    packet["events"][-1]["arguments"]["request_id"] = "changed"
    packet["context"]["goal"]["content"] = "changed"
    assert prefix_digest(data, "a1") == old_hash


def test_official_worker_does_not_require_monitor_numpy_dependency():
    sdk = ROOT / '.local/environments/collection/harnessaudit_oai_20260919/Scripts/python.exe'
    if not sdk.exists():
        pytest.skip("optional existing SDK audit environment is unavailable")
    result = subprocess.run([str(sdk), "-X", "utf8", "-B", str(ROOT / 'code/commands/audit_harnessaudit_supervision.py'),
                             "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_contradictory_literal_rules_are_unavoidable_only_for_required_nonnull_fields():
    rules = [{"role": "worker", "access_type": "operation.resource", "match": {"tool": "lookup", "allowed_args": {"id": value}}}
             for value in ("ONE", "TWO")]
    schema = {"lookup": {"required": ["id"], "properties": {"id": {"type": "string"}}}}
    conflict = exact_resource_conflicts(rules, schema, [("worker", "lookup")], fnmatch)[0]
    assert conflict["unavoidable_for_schema_valid_call"]
    assert conflict["distinct_literals"] == ["ONE", "TWO"]
    for altered in ({}, {"lookup": {"required": ["id"], "properties": {"id": {"type": ["string", "null"]}}}}):
        conflict = exact_resource_conflicts(rules, altered, [("worker", "lookup")], fnmatch)[0]
        assert not conflict["unavoidable_for_schema_valid_call"]


def test_policy_conflict_check_respects_role_tool_and_does_not_claim_pattern_solver():
    rules = [{"role": "worker", "access_type": "operation.resource", "match": {"tool": "lookup", "allowed_args": {"id": value}}}
             for value in ("ONE", "ONE")]
    assert not exact_resource_conflicts(rules, {}, [("worker", "lookup")], fnmatch)
    rules[1]["match"]["allowed_args"]["id"] = "OTHER*"
    assert not exact_resource_conflicts(rules, {}, [("worker", "lookup")], fnmatch)
    rules[1]["match"]["allowed_args"]["id"] = "TWO"
    assert not exact_resource_conflicts(rules, {}, [("other", "lookup"), ("worker", "different")], fnmatch)


def test_literal_conflicts_use_the_actual_matcher_case_semantics():
    rules = [{"role": "worker", "access_type": "operation.resource", "match": {"tool": "lookup", "allowed_args": {"id": value}}}
             for value in ("ONE", "one")]
    insensitive = lambda value, pattern: fnmatch(value.lower(), pattern.lower())
    assert not exact_resource_conflicts(rules, {}, [("worker", "lookup")], insensitive)
