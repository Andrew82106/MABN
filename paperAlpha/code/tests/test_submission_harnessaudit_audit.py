"""Synthetic SDK transport and real in-memory Bank; never a network call."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
import audit_submission_harnessaudit as audit
from paperalpha_runtime.harnessaudit_node_executor import HarnessAuditNodeExecutor
from test_harnessaudit_node_executor import FakeProvider, prohibit_network, response
from test_submission_harnessaudit_collection import synthetic_case


def collect(tmp_path, failure=False):
    case = synthetic_case()
    plan = {"settings": copy.deepcopy(audit.collection.SETTINGS), "upstream": str(audit.collection.DEFAULT_UPSTREAM), "sha256": "synthetic-plan"}
    def factory(folder):
        script = ([TimeoutError("synthetic timeout")] if failure else [
            response(1, tool="get_market_data", arguments={"tickers": "AAPL"}),
            response(2, text="Observed synthetic fixture market data."),
            response(3, text="Final summary from actual predecessor.")])
        return HarnessAuditNodeExecutor(receipt_dir=folder, model_provider=FakeProvider(script, folder),
            max_calls=plan["settings"]["max_calls"], max_turns=plan["settings"]["max_turns"],
            turn_timeout_seconds=plan["settings"]["turn_timeout_seconds"], max_output_tokens=plan["settings"]["max_output_tokens"])
    asyncio.run(audit.collection.collect_case(tmp_path, plan, case, factory))
    return plan, case


def test_complete_fake_sdk_collection_recomputes(tmp_path):
    plan, case = collect(tmp_path)
    report = audit.audit_case(tmp_path, plan, case)
    assert report["status"] == "completed" and report["checks"] == "passed"
    assert report["api_attempts"] == report["api_responses"] == 3
    assert report["actual_tool_invocations"] == report["model_tool_requests"] == report["bank_dispatched"] == 1
    assert report["user_deliveries"] == 1 and report["api_errors"] == 0


def test_failed_real_attempt_is_retained_not_safe(tmp_path):
    plan, case = collect(tmp_path, failure=True)
    report = audit.audit_case(tmp_path, plan, case)
    assert report["status"] == "failed" and report["api_attempts"] == report["api_errors"] == 1
    assert report["api_responses"] == report["actual_tool_invocations"] == report["user_deliveries"] == 0


@pytest.mark.parametrize("corruption", ["monitor_private", "future_prefix", "model_input", "tool_arguments", "state_chain", "native_label", "node_role", "report_text", "request_hash"])
def test_tampering_is_detected_without_model_calls(tmp_path, corruption):
    plan, case = collect(tmp_path)
    folder = tmp_path / "cases/fake"
    targets = {"monitor_private": "MONITOR.json", "future_prefix": "stages/01.json", "model_input": "api/calls/000001.claim.json",
               "tool_arguments": "api/calls/000001.response.json", "state_chain": "ORACLE.json", "native_label": "LABELS.json",
               "node_role": "api/nodes/000001.claim.json", "report_text": "api/nodes/000002.response.json",
               "request_hash": "api/calls/000001.claim.json"}
    path = folder / targets[corruption]
    data = audit.collection.read(path)
    if corruption == "monitor_private": data["initial_state"] = {"hidden": "forbidden"}
    elif corruption == "future_prefix": data["prefix_before"]["future"] = "future-label"
    elif corruption == "model_input":
        data["request"]["input"][0]["content"] = "forged additional oracle input"
        data["request_sha256"] = audit.collection.digest(data["request"])
    elif corruption == "tool_arguments": data["response"]["output"][0]["arguments"] = '{"tickers":"FORGED"}'
    elif corruption == "state_chain": data["tool_transitions"][0]["state_before"] = {}
    elif corruption == "native_label": data["complete_episode_violation"] = True
    elif corruption == "node_role": data["packet"]["bank_role"] = "forged-role"
    elif corruption == "report_text": data["report"] = "unobserved report"
    else: data["request_sha256"] = "wrong"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((audit.AuditError, ValueError)):
        audit.audit_case(tmp_path, plan, case)
