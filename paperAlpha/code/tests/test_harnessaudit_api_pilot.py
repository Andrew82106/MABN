import asyncio
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/collect_harnessaudit_api_pilot.py'
spec = importlib.util.spec_from_file_location("ha_api_pilot_test", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


@pytest.mark.parametrize("url", ["http://localhost:58661/v1", "http://127.0.0.1:58661/v1/"])
def test_authorized_endpoint(url):
    assert pilot.validate_endpoint(url) == url.rstrip("/")


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "http://localhost.evil:58661/v1", "http://127.0.0.1:1234/v1",
    "http://user:key@localhost:58661/v1", "http://localhost:58661/v1?redirect=other",
    "http://localhost:58661/v1#secret", "http://localhost:58661/v1/responses",
])
def test_pilot_cannot_silently_switch_endpoints(url):
    with pytest.raises(ValueError):
        pilot.validate_endpoint(url)


def test_call_budget_is_shared_and_recorded_without_extra_model_calls(tmp_path):
    ledger = pilot.CallLedger(tmp_path, 2)
    a, sa = ledger.begin({"input": "first"})
    b, sb = ledger.begin({"input": "second"})
    # Two outstanding concurrent calls still consume the whole shared budget.
    with pytest.raises(pilot.RequestBudgetExceeded):
        ledger.begin({"input": "should not execute"})
    ledger.finish(b, sb, response={"usage": {"input_tokens": 10, "output_tokens": 3}})
    ledger.finish(a, sa, error=RuntimeError("private error details"))
    assert (ledger.started, ledger.finished, ledger.failed, ledger.blocked) == (2, 1, 1, 1)
    assert (ledger.input_tokens, ledger.output_tokens) == (10, 3)
    assert len((tmp_path / "api_requests.jsonl").read_text().splitlines()) == 2
    replies = [json.loads(x) for x in (tmp_path / "api_responses.jsonl").read_text().splitlines()]
    assert [r["call"] for r in replies] == [2, 1]
    assert replies[1]["error_type"] == "RuntimeError"
    assert "private error details" not in (tmp_path / "api_responses.jsonl").read_text()


@pytest.mark.parametrize("budget", [0, 33, True, 1.5])
def test_invalid_budget(budget, tmp_path):
    with pytest.raises(ValueError):
        pilot.CallLedger(tmp_path, budget)


def test_jsonable_handles_nested_dataclasses_without_serializing_arbitrary_objects():
    @dataclass
    class Result:
        usage: dict
        output: list
    value = Result({"input_tokens": 1}, [{"type": "message", "text": "hello"}])
    assert pilot.jsonable(value)["usage"] == {"input_tokens": 1}
    with pytest.raises(TypeError):
        pilot.jsonable(object())


def test_missing_key_or_existing_output_stops_before_imports_or_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = SimpleNamespace(out=tmp_path, upstream=tmp_path)
    monkeypatch.delenv("PAPERALPHA_API_KEY", raising=False)
    monkeypatch.setenv("PAPERALPHA_API_BASE", "http://localhost:58661/v1")
    with pytest.raises(ValueError, match="PAPERALPHA_API_KEY"):
        asyncio.run(pilot.collect(args))
    monkeypatch.setenv("PAPERALPHA_API_KEY", "fixture-only")
    with pytest.raises(ValueError, match="never overwritten"):
        asyncio.run(pilot.collect(args))


def test_tool_hook_is_before_original_handler_and_keeps_result_unchanged(tmp_path):
    from types import SimpleNamespace
    recorder = pilot.ToolLifecycleRecorder(tmp_path)
    ctx = SimpleNamespace(context=SimpleNamespace(agent_role="agent-a"), tool_call_id="sdk-call-1")
    async def original(context, args):
        rows = [json.loads(line) for line in (tmp_path / "tool_lifecycle.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["event_type"] == "tool_handler_started"
        assert "result" not in rows[0] and rows[0]["arguments"] == {"x": 2}
        assert recorder.active.get()["provider_call_id"] == "sdk-call-1"
        assert context is ctx and args == '{"x":2}'
        return "actual original result"
    tool = recorder.wrap(SimpleNamespace(name="read", on_invoke_tool=original))
    assert asyncio.run(tool.on_invoke_tool(ctx, '{"x":2}')) == "actual original result"
    rows = [json.loads(line) for line in (tmp_path / "tool_lifecycle.jsonl").read_text().splitlines()]
    assert rows[0]["invocation_id"] == rows[1]["invocation_id"]
    assert rows[0]["monotonic_ns"] <= rows[1]["monotonic_ns"]
    assert (recorder.started, recorder.returned, recorder.failed) == (1, 1, 0)
    assert recorder.active.get() is None


def test_concurrent_identical_calls_get_distinct_bindings_and_failures_do_not_vanish(tmp_path):
    from types import SimpleNamespace
    recorder = pilot.ToolLifecycleRecorder(tmp_path)
    observed = {}
    async def original(context, args):
        before = dict(recorder.active.get())
        await asyncio.sleep(0)
        assert recorder.active.get() == before
        observed[context.context.agent_role] = before
        if context.context.agent_role == "a":
            raise RuntimeError("failure")
        return "ok"
    tool = recorder.wrap(SimpleNamespace(name="read", on_invoke_tool=original))
    async def run():
        ctxs = [SimpleNamespace(context=SimpleNamespace(agent_role=role), tool_call_id="repeated-provider-id") for role in ("a", "b")]
        return await asyncio.gather(*(tool.on_invoke_tool(c, '{"x":2}') for c in ctxs), return_exceptions=True)
    result = asyncio.run(run())
    assert isinstance(result[0], RuntimeError) and result[1] == "ok"
    assert observed["a"]["invocation_id"] != observed["b"]["invocation_id"]
    assert (recorder.started, recorder.returned, recorder.failed) == (2, 1, 1)
    rows = [json.loads(line) for line in (tmp_path / "tool_lifecycle.jsonl").read_text().splitlines()]
    assert sorted(r["event_type"] for r in rows) == ["tool_handler_failed", "tool_handler_returned", "tool_handler_started", "tool_handler_started"]


def test_invalid_tool_arguments_are_preserved_for_the_original_handler(tmp_path):
    from types import SimpleNamespace
    recorder = pilot.ToolLifecycleRecorder(tmp_path)
    async def original(_, args):
        assert args == "broken-json"
        return "[ERROR] Invalid JSON arguments"
    tool = recorder.wrap(SimpleNamespace(name="read", on_invoke_tool=original))
    ctx = SimpleNamespace(context=SimpleNamespace(agent_role="a"), tool_call_id="c")
    result = asyncio.run(tool.on_invoke_tool(ctx, "broken-json"))
    assert result.startswith("[ERROR]")
    start = json.loads((tmp_path / "tool_lifecycle.jsonl").read_text().splitlines()[0])
    assert start["arguments"] is None and start["arguments_raw"] == "broken-json"
