import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/audit_harnessaudit_api_pilot.py'
spec = importlib.util.spec_from_file_location("ha_acquisition_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def data():
    binding = {"invocation_id": "i1", "provider_call_id": "c1", "actor": "a", "tool": "read"}
    stages = [
        {**binding, "event_type": "tool_handler_started", "monotonic_ns": 10, "arguments_raw": '{"x":1}', "arguments": {"x": 1}},
        {**binding, "event_type": "tool_handler_returned", "monotonic_ns": 11, "result": "ok"},
    ]
    native = [{"agent_id": "a", "tool_name": "read", "tool_args": {"x": 1}, "tool_result": "ok",
               "raw_event": {"paperalpha_invocation_id": "i1", "provider_call_id": "c1"}}]
    replies = [{"response": {"output": [{"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"x":1}'}]}}]
    return stages, native, replies


def test_lifecycle_audit_matches_real_id_chain():
    result = audit.verify_lifecycle(*data())
    assert result["started"] == result["returned"] == 1 and result["failed"] == 0


@pytest.mark.parametrize("mutation", [
    lambda s, n, r: s[0].update(result="future result"),
    lambda s, n, r: s[0].update(violated=True),
    lambda s, n, r: s[1].update(monotonic_ns=9),
    lambda s, n, r: s[1].update(actor="wrong"),
    lambda s, n, r: s.pop(),
    lambda s, n, r: s.insert(1, copy.deepcopy(s[0])),
    lambda s, n, r: n[0].update(tool_args={"x": 2}),
    lambda s, n, r: n[0].update(tool_result="different"),
    lambda s, n, r: n[0]["raw_event"].update(paperalpha_invocation_id="missing"),
    lambda s, n, r: r[0]["response"]["output"][0].update(arguments='{"x":2}'),
    lambda s, n, r: n.clear(),
])
def test_inconsistent_or_leaky_lifecycle_fails(mutation):
    stages, native, replies = data()
    mutation(stages, native, replies)
    with pytest.raises(AssertionError):
        audit.verify_lifecycle(stages, native, replies)


def test_thrown_handler_is_not_invented_as_completed_action():
    stages, native, replies = data()
    stages[1].update(event_type="tool_handler_failed", error_type="RuntimeError")
    stages[1].pop("result")
    result = audit.verify_lifecycle(stages, [], replies)
    assert result["failed"] == 1 and result["returned"] == 0
