"""Bounded official-SDK node loop; sandbox tools are its only action surface.

The controller supplies the provider/client. This module reads no credentials,
environment variables, benchmark labels or database. Receipts contain SDK
requests/responses, not raw HTTP response envelopes. A claimed directory is
never resumed or overwritten: an interrupted attempt stays an interrupted one.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import time

from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner
from agents.model_settings import ModelRetrySettings
from agents.models.interface import Model, ModelProvider
from openai.types.shared import Reasoning


MODEL = "gpt-5.5"
PACKET_FIELDS = {"node_id", "bank_role", "role_binding_source", "instructions", "input",
                 "delivery_target", "stage_purpose", "tools"}
_TYPE_PREFIX = re.compile(r"^\((int|float|str|string|bool)\)\s*")


class RequestBudgetExceeded(RuntimeError):
    pass


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("receipt value is not an explicit JSON value")


def _canonical(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _durable_new(path, value):
    """Exclusive creation and file fsync BEFORE any dependent external action."""
    payload = _canonical(value) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parameter_schema(tool):
    """Identical to pinned upstream oai/tools.py::_build_params_schema.

    All declared params are required; optionality in author descriptions is
    retained verbatim, not reinterpreted by this adapter.
    """
    params = tool.get("params", {})
    if not isinstance(params, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           for k, v in params.items()):
        raise ValueError("author tool params must map names to descriptions")
    properties = {}
    for name, description in params.items():
        match = _TYPE_PREFIX.match(description)
        kind = match.group(1) if match else "string"
        properties[name] = {"type": {"int": "integer", "float": "number", "bool": "boolean"}.get(kind, "string"),
                            "description": description}
    return {"type": "object", "properties": properties, "required": list(params),
            "additionalProperties": False}


def _packet(value):
    if not isinstance(value, dict) or set(value) != PACKET_FIELDS:
        raise ValueError("exact public sandbox node packet required")
    value = json.loads(_canonical(value))
    for key in ("node_id", "bank_role", "role_binding_source", "instructions"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("explicit nonempty node and role instructions required")
    if value["delivery_target"] not in {"user", "harness_report_store"}:
        raise ValueError("declared sandbox output endpoint required")
    purpose = value["stage_purpose"]
    if purpose is not None and (not isinstance(purpose, dict) or set(purpose) != {"purpose", "source"}
                               or any(not isinstance(x, str) or not x.strip() for x in purpose.values())):
        raise ValueError("public stage purpose and provenance required")
    data = value["input"]
    if (not isinstance(data, dict) or set(data) != {"original_task", "predecessor_reports"}
            or not isinstance(data["original_task"], str) or not data["original_task"].strip()
            or not isinstance(data["predecessor_reports"], list)):
        raise ValueError("only original task and actual predecessor reports are model inputs")
    for report in data["predecessor_reports"]:
        if (not isinstance(report, dict) or set(report) != {"sender", "content", "sender_role", "sender_role_source"}
                or any(not isinstance(x, str) or not x.strip() for x in report.values())):
            raise ValueError("predecessor report requires observed content and declared role binding")
    if not isinstance(value["tools"], list):
        raise ValueError("explicit author tool catalogue required")
    names = []
    for tool in value["tools"]:
        if (not isinstance(tool, dict) or not {"name", "description", "params"} <= set(tool)
                or not set(tool) <= {"name", "description", "params", "is_resource"}
                or not isinstance(tool["name"], str) or not tool["name"]
                or not isinstance(tool["description"], str)):
            raise ValueError("public author tool declaration required")
        parameter_schema(tool)
        names.append(tool["name"])
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique")
    return value


class HarnessAuditNodeExecutor:
    """One episode, fresh SDK context per node, shared hard request budget.

    Injected providers are trusted code and must not retry internally. For a
    real OpenAI-compatible client use from_openai_client(), which checks the
    client's retry setting. The executor never creates a client from secrets.
    """

    def __init__(self, *, receipt_dir, model_provider, max_calls=32, max_turns=8,
                 turn_timeout_seconds=45, max_output_tokens=1200):
        for value, cap in ((max_calls, 256), (max_turns, 32), (max_output_tokens, 32768)):
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError("finite positive request/turn/token budget required")
        if (isinstance(turn_timeout_seconds, bool) or not isinstance(turn_timeout_seconds, (int, float))
                or not math.isfinite(turn_timeout_seconds) or not 0 < turn_timeout_seconds <= 300):
            raise ValueError("finite positive model-turn timeout up to 300 seconds required")
        if not isinstance(model_provider, ModelProvider):
            raise TypeError("explicit official-SDK ModelProvider required")
        self.receipt_dir = Path(receipt_dir)
        self.provider = model_provider
        self.max_calls, self.max_turns = max_calls, max_turns
        self.turn_timeout_seconds = float(turn_timeout_seconds)
        self.settings = ModelSettings(reasoning=Reasoning(effort="low"), store=False,
            parallel_tool_calls=False, max_tokens=max_output_tokens, retry=ModelRetrySettings(max_retries=0))
        self.calls = 0
        self._active, self._failed, self._nodes, self._response_ids = False, False, set(), set()
        self.receipt_dir.mkdir(parents=True, exist_ok=False)
        (self.receipt_dir / "calls").mkdir()
        (self.receipt_dir / "nodes").mkdir()
        _durable_new(self.receipt_dir / "CONFIG.json", {"schema": "harnessaudit-node-executor-v1",
            "model_requested": MODEL, "model_settings": self.settings.to_json_dict(),
            "max_calls": max_calls, "max_turns": max_turns, "turn_timeout_seconds": self.turn_timeout_seconds,
            "context": "fresh per node, full in-node history, no sessions or previous response IDs",
            "receipt_surface": "official SDK Model request/response; not raw HTTP response envelope",
            "retry_contract": "runner retries zero; injected provider must also have transport retries zero",
            "versions": {name: importlib.metadata.version(name) for name in ("openai", "openai-agents")}})

    @classmethod
    def from_openai_client(cls, *, client, **kwargs):
        from openai import AsyncOpenAI
        from agents.models.openai_provider import OpenAIProvider
        if not isinstance(client, AsyncOpenAI) or client.max_retries != 0:
            raise ValueError("explicit AsyncOpenAI client with max_retries=0 required")
        return cls(model_provider=OpenAIProvider(openai_client=client, use_responses=True,
                                                use_responses_websocket=False), **kwargs)

    def _recorded_model(self, node_id):
        owner = self
        underlying = self.provider.get_model(MODEL)
        # Also reject a caller injecting an ordinary OpenAI model with retries.
        client = getattr(underlying, "_client", None)
        if client is not None and getattr(client, "max_retries", None) != 0:
            raise ValueError("provider transport must disable retries")

        class RecordedModel(Model):
            async def get_response(self, *args, **kwargs):
                values = inspect.signature(Model.get_response).bind(self, *args, **kwargs).arguments
                if (values.get("previous_response_id") is not None or values.get("conversation_id") is not None
                        or values.get("prompt") is not None or values.get("handoffs") or values.get("output_schema") is not None):
                    raise ValueError("node model call cannot reuse remote state, prompts, handoffs or output schemas")
                settings = values["model_settings"].to_json_dict()
                if (settings != owner.settings.to_json_dict()):
                    raise ValueError("fixed node model settings changed")
                if owner.calls >= owner.max_calls:
                    raise RequestBudgetExceeded("episode request budget reached")
                number = owner.calls + 1
                stem = owner.receipt_dir / "calls" / f"{number:06d}"
                request = {"model_requested": MODEL, "system_instructions": values["system_instructions"],
                    "input": values["input"], "model_settings": settings,
                    "tools": [{"name": t.name, "description": t.description, "parameters": t.params_json_schema}
                              for t in values["tools"]],
                    "previous_response_id": None, "conversation_id": None}
                claim = {"schema": "sdk-model-call-claim", "call": number, "node_id": node_id,
                    "claimed_at": datetime.now(timezone.utc).isoformat(), "request": request,
                    "request_sha256": hashlib.sha256(_canonical(request).encode()).hexdigest()}
                _durable_new(str(stem) + ".claim.json", claim)
                owner.calls = number
                start = time.perf_counter()
                try:
                    response = await asyncio.wait_for(underlying.get_response(*args, **kwargs),
                                                       timeout=owner.turn_timeout_seconds)
                    _durable_new(str(stem) + ".response.json", {"schema": "sdk-model-call-response",
                        "call": number, "node_id": node_id, "request_sha256": claim["request_sha256"],
                        "seconds": time.perf_counter()-start, "response": response})
                    if not response.response_id or response.response_id in owner._response_ids:
                        raise ValueError("missing or duplicate model response identity")
                    owner._response_ids.add(response.response_id)
                    return response
                except BaseException as exc:
                    _durable_new(str(stem) + ".error.json", {"schema": "sdk-model-call-error",
                        "call": number, "node_id": node_id, "request_sha256": claim["request_sha256"],
                        "seconds": time.perf_counter()-start, "error_type": type(exc).__name__,
                        "response_receipt_exists": Path(str(stem) + ".response.json").exists()})
                    raise

            async def stream_response(self, *args, **kwargs):
                raise RuntimeError("streaming is outside this node executor")
                yield  # pragma: no cover

        return RecordedModel()

    async def __call__(self, packet, invoke_tool):
        packet = _packet(packet)
        node = packet["node_id"]
        if not callable(invoke_tool):
            raise TypeError("node-bound sandbox tool callback required")
        if self._active or self._failed or node in self._nodes:
            raise RuntimeError("no overlapping, repeated or post-failure node executions")
        self._active = True
        self._nodes.add(node)
        stem = self.receipt_dir / "nodes" / f"{len(self._nodes):06d}"
        try:
            _durable_new(str(stem) + ".claim.json", {"node_id": node, "packet": packet,
                "packet_sha256": hashlib.sha256(_canonical(packet).encode()).hexdigest(), "calls_before": self.calls})
            tools = []
            for declaration in packet["tools"]:
                def handler(name):
                    async def invoke(context, arguments):
                        data = json.loads(arguments)
                        if not isinstance(data, dict):
                            raise ValueError("tool arguments must be an object")
                        return await invoke_tool(name, data, getattr(context, "tool_call_id", None))
                    return invoke
                tools.append(FunctionTool(name=declaration["name"], description=declaration["description"],
                    params_json_schema=parameter_schema(declaration), strict_json_schema=True,
                    on_invoke_tool=handler(declaration["name"]), _use_default_failure_error_function=False))
            agent = Agent(name=node, instructions=packet["instructions"], tools=tools,
                          model=self._recorded_model(node), model_settings=copy.deepcopy(self.settings))
            result = await Runner.run(agent, _canonical(packet["input"]), context=None, max_turns=self.max_turns,
                run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False,
                                     model_settings=copy.deepcopy(self.settings)),
                session=None, previous_response_id=None, auto_previous_response_id=False, conversation_id=None)
            report = result.final_output
            if not isinstance(report, str) or not report.strip():
                raise ValueError("node returned no actual text report")
            _durable_new(str(stem) + ".response.json", {"node_id": node, "report": report,
                "calls_after": self.calls, "status": "completed", "safety": "not_evaluated"})
            return report
        except BaseException as exc:
            self._failed = True
            _durable_new(str(stem) + ".error.json", {"node_id": node, "calls_after": self.calls,
                "status": "failed", "error_type": type(exc).__name__, "safety": "not_evaluated"})
            raise
        finally:
            self._active = False
