import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
import verify_capability_action_bn as verify


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_actual_handler_projection_never_promotes_results_labels_or_invented_scope(tmp_path):
    path = tmp_path / "events.jsonl"
    start = {"event_type": "tool_handler_started", "invocation_id": "call-1", "actor": "actor",
             "tool": "authorize_payment", "arguments": {"amount": 5}, "principal": "not-trusted",
             "resource": "not-trusted", "label": 1, "result": "approved", "permission_mismatch": False}
    end = {**start, "event_type": "tool_handler_returned", "result": "failed"}
    write_rows(path, [start, end])
    events = verify.lifecycle_events(path)
    assert events == [{"sequence": 0, "event_id": "call-1", "event_type": "tool_call", "action_id": "call-1",
                       "actor": "actor", "tool": "authorize_payment", "arguments": {"amount": 5},
                       "principal": None, "resource": None}]
    changed = copy.deepcopy(start)
    changed.update(label=0, result="rejected", permission_mismatch=True)
    write_rows(path, [changed, end])
    assert verify.lifecycle_events(path) == events


def test_duplicate_actual_invocations_are_rejected_not_merged(tmp_path):
    path = tmp_path / "events.jsonl"
    e = {"event_type": "tool_handler_started", "invocation_id": "x", "actor": "a", "tool": "t", "arguments": {}}
    write_rows(path, [e, e])
    with pytest.raises(ValueError, match="duplicate"):
        verify.lifecycle_events(path)


def test_bounded_replay_does_not_reset_or_report_full_coverage():
    events = [{"sequence": i, "event_type": "tool_call", "event_id": f"e{i}", "action_id": f"a{i}",
               "actor": "actor", "tool": "authorize_payment", "arguments": {}} for i in range(9)]
    result = verify.bounded_replay(verify.CapabilityActionBN(), verify.load_tool_registry(), events)
    assert result["accepted_prefix_events"] == 5
    assert result["unprocessed_events"] == 4
    assert result["stop"]["first_unprocessed_sequence"] == 5
    assert not result["whole_episode_covered"]
    assert not result["parameters_trained"]
    assert not result["risk_scores_are_performance_evidence"]


def test_coverage_is_from_public_names_not_tool_results_or_benchmark_labels(tmp_path):
    path = tmp_path / "arrival_events.jsonl"
    e = {"surface": "tool_call", "arrival_index": 3,
         "data": {"tool_name": "authorize_payment", "tool_result": "success", "risk": 1, "access_rules": []}}
    write_rows(path, [e])
    result = verify.tool_coverage(verify.load_tool_registry(), [tmp_path])
    e["data"].update(tool_result="failure", risk=0, access_rules=["different"])
    write_rows(path, [e])
    assert result == verify.tool_coverage(verify.load_tool_registry(), [tmp_path])
    assert result["resolved_primary_capabilities"] == {"payment_effect": 1}


def test_public_profile_comparison_does_not_use_private_catalog_annotations(tmp_path):
    registry = verify.load_tool_registry()
    catalog = tmp_path / "tools.yaml"
    public = [{"name": p["name"], "description": p["description"], "is_resource": False, "mock_response": "bad"}
              for p in registry.specification()["tools"]]
    catalog.write_text(json.dumps({"tools": public}), encoding="utf-8")
    result = verify.public_profile_check(registry, catalog)
    for p in public:
        p.update(is_resource=True, mock_response="good", hidden_label=1)
    catalog.write_text(json.dumps({"tools": public}), encoding="utf-8")
    assert result == verify.public_profile_check(registry, catalog)
    public[0]["description"] = "changed public definition"
    catalog.write_text(json.dumps({"tools": public}), encoding="utf-8")
    with pytest.raises(AssertionError, match="description differs"):
        verify.public_profile_check(registry, catalog)
