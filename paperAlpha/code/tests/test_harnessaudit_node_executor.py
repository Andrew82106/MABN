"""Real official-SDK loops with fake models/tools only; network prohibited."""
import asyncio
import copy
import inspect
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.harnessaudit_node_executor import (
    HarnessAuditNodeExecutor, RequestBudgetExceeded, MODEL, parameter_schema)
import paperalpha_runtime.harnessaudit_node_executor as implementation


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    original = socket.socket.connect
    socketpair_code = getattr(getattr(socket, "_fallback_socketpair", None), "__code__", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in executor fake tests")
    def guarded(sock, address):
        if socketpair_code is not None and inspect.currentframe().f_back.f_code is socketpair_code:
            return original(sock, address)
        return forbidden(sock, address)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", guarded)


def response(number, *, text=None, tool=None, arguments=None):
    if tool is not None:
        output = [ResponseFunctionToolCall(id=f"fc-{number}", call_id=f"call-{number}", type="function_call",
                    name=tool, arguments=json.dumps(arguments or {}), status="completed")]
    else:
        output = [ResponseOutputMessage(id=f"message-{number}", type="message", role="assistant", status="completed",
                    content=[ResponseOutputText(type="output_text", text=text or "", annotations=[])])]
    return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=2, output_tokens=2, total_tokens=4),
                         response_id=f"response-{number}", request_id=f"request-{number}")


class FakeModel(Model):
    def __init__(self, script, folder):
        self.script, self.folder, self.requests = list(script), folder, []

    async def get_response(self, *args, **kwargs):
        values = inspect.signature(Model.get_response).bind(self, *args, **kwargs).arguments
        self.requests.append(values)
        claim = self.folder / "calls" / f"{len(self.requests):06d}.claim.json"
        assert claim.is_file()  # Model code cannot run before durable claim.
        assert json.loads(claim.read_text(encoding="utf-8"))["request"]["model_requested"] == MODEL
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item

    async def stream_response(self, *args, **kwargs):
        raise AssertionError("streaming not expected")
        yield


class FakeProvider(ModelProvider):
    def __init__(self, script, folder):
        self.model, self.requested_names = FakeModel(script, folder), []

    def get_model(self, name):
        self.requested_names.append(name)
        return self.model


def packet(node="work", predecessors=None):
    return {"node_id": node, "bank_role": "synthetic_operator", "role_binding_source": "synthetic public binding",
        "instructions": "Use synthetic tools, then give the actual result.",
        "stage_purpose": {"purpose": "Inspect a synthetic value.", "source": "declared public test purpose"},
        "input": {"original_task": "Inspect the synthetic fixture.", "predecessor_reports": predecessors or []},
        "delivery_target": "harness_report_store",
        "tools": [{"name": "read_value", "description": "Read synthetic value.", "params": {"key": "(str) A key"}}]}


def executor(tmp_path, script, **kwargs):
    folder = tmp_path / "receipts"
    provider = FakeProvider(script, folder)
    return HarnessAuditNodeExecutor(receipt_dir=folder, model_provider=provider, **kwargs), provider


async def unused(*args):
    raise AssertionError("no tool expected")


def test_official_sdk_tool_loop_has_durable_claims_and_only_sandbox_actions(tmp_path, monkeypatch):
    ex, provider = executor(tmp_path, [response(1, tool="read_value", arguments={"key": "A"}),
                                       response(2, text="Observed SYNTHETIC_VALUE.")])
    called, fsyncs = [], []
    original = implementation.os.fsync
    def fsync(fd):
        fsyncs.append(fd)
        return original(fd)
    monkeypatch.setattr(implementation.os, "fsync", fsync)
    async def invoke(name, arguments, call_id):
        called.append((name, arguments, call_id))
        return "SYNTHETIC_VALUE"
    assert asyncio.run(ex(packet(), invoke)) == "Observed SYNTHETIC_VALUE."
    assert called == [("read_value", {"key": "A"}, "call-1")]
    assert ex.calls == 2 and len(fsyncs) == 6
    assert "SYNTHETIC_VALUE" in json.dumps(provider.model.requests[1]["input"])
    assert provider.requested_names == [MODEL]
    assert len(list((ex.receipt_dir / "calls").glob("*.claim.json"))) == 2
    assert len(list((ex.receipt_dir / "calls").glob("*.response.json"))) == 2
    for req in provider.model.requests:
        assert req["previous_response_id"] is None and req["conversation_id"] is None
        settings = req["model_settings"]
        assert settings.reasoning.effort == "low" and settings.store is False
        assert settings.parallel_tool_calls is False and settings.retry.max_retries == 0
    assert not list(ex.receipt_dir.rglob("*.error.json"))


def test_nodes_have_fresh_context_and_only_declared_predecessor_content(tmp_path):
    ex, provider = executor(tmp_path, [response(1, text="FIRST_NODE_REPORT"), response(2, text="SECOND_NODE_REPORT")])
    asyncio.run(ex(packet("left"), unused))
    right = packet("right", [{"sender": "root", "content": "ROOT_REPORT_ONLY",
        "sender_role": "synthetic_root", "sender_role_source": "explicit binding"}])
    asyncio.run(ex(right, unused))
    seen = provider.model.requests[1]["input"]
    assert "FIRST_NODE_REPORT" not in json.dumps(seen)
    assert "ROOT_REPORT_ONLY" in json.dumps(seen)
    assert provider.model.requests[1]["system_instructions"] == right["instructions"]
    with pytest.raises(RuntimeError, match="repeated"):
        asyncio.run(ex(right, unused))


def test_total_budget_is_shared_across_nodes_and_reopening_is_forbidden(tmp_path):
    ex, provider = executor(tmp_path, [response(1, text="one"), response(2, text="never")], max_calls=1)
    asyncio.run(ex(packet("one"), unused))
    with pytest.raises(RequestBudgetExceeded):
        asyncio.run(ex(packet("two"), unused))
    assert len(provider.model.requests) == ex.calls == 1
    assert not (ex.receipt_dir / "calls/000002.claim.json").exists()
    assert json.loads((ex.receipt_dir / "nodes/000002.error.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        HarnessAuditNodeExecutor(receipt_dir=ex.receipt_dir, model_provider=provider)


def test_failed_claim_fsync_prevents_model_call_and_cannot_be_resumed(tmp_path, monkeypatch):
    ex, provider = executor(tmp_path, [response(1, text="never called")])
    original, calls = implementation.os.fsync, []
    def failing_fsync(fd):
        calls.append(fd)
        if len(calls) == 2:  # Node claim first, model-call claim second.
            raise OSError("synthetic claim persistence failure")
        return original(fd)
    monkeypatch.setattr(implementation.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        asyncio.run(ex(packet(), unused))
    assert not provider.model.requests and ex.calls == 0
    assert (ex.receipt_dir / "calls/000001.claim.json").exists()
    assert not (ex.receipt_dir / "calls/000001.response.json").exists()
    with pytest.raises(FileExistsError):
        HarnessAuditNodeExecutor(receipt_dir=ex.receipt_dir, model_provider=provider)


@pytest.mark.parametrize("failure", ["provider", "timeout", "cancel", "blank", "duplicate", "tool"])
def test_failures_remain_failures_without_retries_or_secret_exception_text(tmp_path, failure):
    async def slow():
        await asyncio.sleep(1)
    script = {"provider": [RuntimeError("PRIVATE-SECRET-ERROR")], "timeout": [slow],
        "cancel": [asyncio.CancelledError("PRIVATE-SECRET-ERROR")], "blank": [response(1, text="")],
        "duplicate": [response(1, tool="read_value"), response(1, text="duplicate")],
        "tool": [response(1, tool="read_value")]}[failure]
    ex, provider = executor(tmp_path, script, turn_timeout_seconds=.03 if failure == "timeout" else 2)
    async def invoke(*args):
        if failure == "tool":
            raise RuntimeError("PRIVATE-SECRET-ERROR")
        return "value"
    with pytest.raises(BaseException):
        asyncio.run(ex(packet(), invoke))
    assert len(provider.model.requests) == (2 if failure == "duplicate" else 1)
    assert json.loads((ex.receipt_dir / "nodes/000001.error.json").read_text())["safety"] == "not_evaluated"
    assert not (ex.receipt_dir / "nodes/000001.response.json").exists()
    assert "PRIVATE-SECRET-ERROR" not in "".join(p.read_text() for p in ex.receipt_dir.rglob("*.json"))
    with pytest.raises(RuntimeError, match="post-failure"):
        asyncio.run(ex(packet("next"), unused))


def test_max_turns_is_not_synthetic_completion(tmp_path):
    ex, provider = executor(tmp_path, [response(1, tool="read_value"), response(2, text="unreached")], max_turns=1)
    async def invoke(*args):
        return "value"
    with pytest.raises(Exception, match="turns"):
        asyncio.run(ex(packet(), invoke))
    assert len(provider.model.requests) == 1
    assert not (ex.receipt_dir / "nodes/000001.response.json").exists()


@pytest.mark.parametrize("change", ["oracle", "label", "missing_role", "duplicate_tools"])
def test_hidden_or_malformed_packet_rejected_before_any_model_claim(tmp_path, change):
    ex, provider = executor(tmp_path, [])
    value = packet()
    if change == "oracle": value["evaluation_config"] = {"label": 1}
    elif change == "label": value["input"]["future_label"] = 1
    elif change == "missing_role": value["input"]["predecessor_reports"] = [{"sender": "root", "content": "x"}]
    else: value["tools"] *= 2
    with pytest.raises(ValueError):
        asyncio.run(ex(value, unused))
    assert not provider.model.requests and not list((ex.receipt_dir / "calls").iterdir())


def test_schema_matches_author_helper_for_every_reviewed_tool(tmp_path):
    from test_harnessaudit_sandbox_adapter import config
    from paperalpha_runtime.harnessaudit_sandbox_adapter import HarnessAuditSandboxAdapter
    import importlib.util
    path = ROOT / 'code/external/HarnessAudit/multi_agent/frameworks/oai/tools.py'
    spec = importlib.util.spec_from_file_location("author_schema_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for domain in ("office", "ecommerce", "finance"):
        adapter = HarnessAuditSandboxAdapter(**config(tmp_path / domain, domain))
        for tool in adapter.monitor_prefix()["tools"]:
            assert parameter_schema(tool) == module._build_params_schema(SimpleNamespace(params=tool.get("params", {})))


def test_real_client_constructor_rejects_hidden_transport_retries(tmp_path):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="synthetic-not-used", base_url="http://127.0.0.1:1/v1", max_retries=1)
    with pytest.raises(ValueError, match="max_retries=0"):
        HarnessAuditNodeExecutor.from_openai_client(client=client, receipt_dir=tmp_path / "never")
    assert not (tmp_path / "never").exists()
    asyncio.run(client.close())


def test_sdk_executor_integrates_with_sandbox_without_oracle_access(tmp_path):
    from test_harnessaudit_sandbox_adapter import config, MUTATIONS
    from paperalpha_runtime.harnessaudit_sandbox_adapter import HarnessAuditSandboxAdapter
    tool, args, table = MUTATIONS["office"]
    ex, provider = executor(tmp_path, [response(1, tool=tool, arguments=args), response(2, text="Created synthetic user."),
                                       response(3, text="Final synthetic summary.")])
    adapter = HarnessAuditSandboxAdapter(**config(tmp_path, output_nodes=["review"]))
    result = asyncio.run(adapter.run(ex, journal_path=tmp_path / "journal.jsonl"))
    assert result["status"] == "completed" and result["tool_invocations"] == 1
    assert adapter.oracle_view()["initial_state"][table] != adapter.oracle_view()["final_state"][table]
    assert len([x for x in adapter.oracle_actions("fake")['actions'] if x.get("target_role") == "user"]) == 1
    receipts = "".join(p.read_text() for p in ex.receipt_dir.rglob("*.json"))
    assert "PRIVATE-ORACLE-SENTINEL" not in receipts and "PRIVATE-NECESSITY" not in receipts
    assert "Created synthetic user." in json.dumps(provider.model.requests[-1]["input"])
