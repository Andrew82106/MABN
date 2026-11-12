"""Bounded API pilot using HarnessAudit's official OAI adapter and SQLite bank.

Not a paper-score reproduction: uses a documented Windows import workaround,
bounded model settings, no LLM judges, and an extra arrival-time recorder.
Credentials are environment-only. No native shell/file/payment tools are exposed.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
from importlib.machinery import ModuleSpec
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.journal import RuntimeJournal
DEFAULT_UPSTREAM = ROOT / 'code/external/HarnessAudit'
EXPECTED_REVISION = "6317162590aeeb1c8dde32b880ac199933343e4a"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported recorded value: {type(value).__name__}")


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")


def validate_endpoint(base_url):
    parsed = urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.port != 58661 or parsed.path.rstrip("/") != "/v1"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("pilot is restricted to the user-authorized localhost:58661/v1 endpoint")
    return base_url.rstrip("/")


def import_official_oai_components(upstream):
    """Keep official components unchanged; bypass only the eager registry import.

    The registry imports unrelated ClawTeam/fcntl modules on Windows. This
    process-local namespace shim exposes the original package directory so all
    core/OAI relative imports execute their original files. No fcntl emulation.
    """
    upstream = upstream.resolve()
    sys.path.insert(0, str(upstream))
    importlib.import_module("multi_agent")
    name = "multi_agent.frameworks"
    if name in sys.modules:
        raise RuntimeError("run this pilot in a fresh process; framework registry is already loaded")
    package = types.ModuleType(name)
    package.__path__ = [str(upstream / "multi_agent/frameworks")]
    package.__package__ = name
    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package
    return importlib.import_module("multi_agent.frameworks.oai.adapter").OAIAdapter


class RequestBudgetExceeded(RuntimeError):
    pass


class CallLedger:
    def __init__(self, output, budget):
        if type(budget) is not int or not 1 <= budget <= 32:
            raise ValueError("pilot request budget must be between 1 and 32")
        self.output, self.budget = output, budget
        self.started = self.finished = self.failed = self.blocked = 0
        self.input_tokens = self.output_tokens = 0

    def begin(self, request):
        if self.started >= self.budget:
            self.blocked += 1
            raise RequestBudgetExceeded("pilot request budget reached")
        self.started += 1
        number = self.started
        append_json(self.output / "api_requests.jsonl", {"call": number, "recorded_at": utc_now(), **request})
        print(f"model call {number}/{self.budget}", flush=True)
        return number, time.monotonic()

    def finish(self, number, started, response=None, error=None):
        if error is not None:
            self.failed += 1
            payload = {"error_type": type(error).__name__}
        else:
            self.finished += 1
            payload = {"response": jsonable(response)}
            usage = payload["response"].get("usage", {})
            # OpenAI-compatible Chat Completions commonly reports
            # prompt_tokens/completion_tokens, while Responses-style clients
            # use input_tokens/output_tokens.  Preserve both forms so API
            # cost reports do not silently show zero tokens.
            self.input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            self.output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        append_json(self.output / "api_responses.jsonl", {"call": number, "recorded_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - started, **payload})


class ToolLifecycleRecorder:
    """Observe the SDK boundary before its original handler; never authorize it.

    Local invocation IDs distinguish repeat executions even if provider IDs
    repeat. Context-local bindings link concurrent calls without text matching.
    """
    def __init__(self, output, journal=None):
        self.output, self.started, self.returned, self.failed = output, 0, 0, 0
        self.journal = journal
        self.active = ContextVar("harnessaudit_pilot_invocation", default=None)

    def record(self, payload):
        if self.journal is not None:
            row = self.journal.emit(payload["event_type"], jsonable(payload))
            payload = {**payload, "journal_sequence": row["sequence"]}
        append_json(self.output / "tool_lifecycle.jsonl", payload)

    def wrap(self, tool):
        original = tool.on_invoke_tool

        async def invoke(ctx, arguments_json):
            self.started += 1
            binding = {"invocation_id": f"invocation-{self.started:06d}",
                       "provider_call_id": getattr(ctx, "tool_call_id", None),
                       "actor": ctx.context.agent_role, "tool": tool.name}
            try:
                arguments = json.loads(arguments_json)
            except (TypeError, ValueError):
                arguments = None
            self.record({
                **binding, "event_type": "tool_handler_started", "observed_at": utc_now(),
                "monotonic_ns": time.monotonic_ns(), "arguments": arguments, "arguments_raw": arguments_json,
                "visibility": "before original domain handler; no result or audit decision yet"})
            token = self.active.set(binding)
            try:
                result = await original(ctx, arguments_json)
            except BaseException as exc:
                self.failed += 1
                self.record({
                    **binding, "event_type": "tool_handler_failed", "observed_at": utc_now(),
                    "monotonic_ns": time.monotonic_ns(), "error_type": type(exc).__name__})
                raise
            else:
                self.returned += 1
                self.record({
                    **binding, "event_type": "tool_handler_returned", "observed_at": utc_now(),
                    "monotonic_ns": time.monotonic_ns(), "result": result,
                    "meaning": "handler returned; not proof of successful operation or external effect"})
                return result
            finally:
                self.active.reset(token)

        tool.on_invoke_tool = invoke
        return tool


def make_arrival_sink(base_class, output, lifecycle, journal=None):
    class ArrivalSink(base_class):
        def _record(self, kind, data):
            link = {}
            if journal is not None:
                if kind == "communication":
                    row = journal.emit("communication", {"actor": data["agent_id"], "target": data["target_agent"],
                        "content": data["content"], "native_arrival_index": self._ordinal + 1})
                else:
                    binding = lifecycle.active.get()
                    if binding is None:
                        raise ValueError("native tool result has no instrumented handler entry")
                    row = journal.emit("tool_native_result", {**binding, "arguments": data["tool_args"],
                        "result": data["tool_result"], "native_arrival_index": self._ordinal + 1})
                link["journal_sequence"] = row["sequence"]
            append_json(output / "arrival_events.jsonl", {"observed_at": utc_now(),
                "arrival_index": self._ordinal + 1, "surface": kind, "data": data, **link,
                "visibility": "after_tool_return" if kind == "tool_call" else "communication_emission"})

        def emit_tool_call(self, **kwargs):
            binding = lifecycle.active.get()
            if binding is not None:
                kwargs = {**kwargs, "raw_event": {**(kwargs.get("raw_event") or {}),
                    "paperalpha_invocation_id": binding["invocation_id"], "provider_call_id": binding["provider_call_id"]}}
            self._record("tool_call", kwargs)
            return super().emit_tool_call(**kwargs)

        def emit_communication(self, **kwargs):
            self._record("communication", kwargs)
            return super().emit_communication(**kwargs)
    return ArrivalSink


def make_recording_provider(provider, ledger):
    from agents.models.interface import Model, ModelProvider

    class RecordedModel(Model):
        def __init__(self, name, underlying):
            self.name, self.underlying = name, underlying

        async def get_response(self, *args, **kwargs):
            bound = inspect.signature(self.underlying.get_response).bind(*args, **kwargs).arguments
            fields = {key: jsonable(bound.get(key)) for key in
                      ("system_instructions", "input", "model_settings", "previous_response_id", "conversation_id", "prompt")}
            fields["tools"] = [{"name": t.name, "description": getattr(t, "description", None),
                                "params_json_schema": getattr(t, "params_json_schema", None)} for t in bound.get("tools", [])]
            number, started = ledger.begin({"model_requested": self.name, **fields})
            try:
                response = await self.underlying.get_response(*args, **kwargs)
            except BaseException as exc:
                ledger.finish(number, started, error=exc)
                raise
            ledger.finish(number, started, response=response)
            return response

        async def stream_response(self, *args, **kwargs):
            raise RuntimeError("streaming is outside this bounded pilot")
            yield  # pragma: no cover -- preserve async-generator protocol

    class RecordedProvider(ModelProvider):
        def get_model(self, model_name):
            return RecordedModel(model_name, provider.get_model(model_name))

    return RecordedProvider()


async def collect(args):
    base_url = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://127.0.0.1:58661/v1"))
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("PAPERALPHA_API_KEY is required; no credential file is read")
    output = args.out.resolve()
    if output.exists():
        raise ValueError("output must be new; prior runs are never overwritten")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != EXPECTED_REVISION or dirty:
        raise ValueError("pilot requires the checked, unchanged upstream revision")
    OAIAdapter = import_official_oai_components(args.upstream)
    from multi_agent.loader import load_task_with_tools, resolve_task_path
    from multi_agent.banks import get_bank
    from multi_agent.frameworks.core.base import RunContext
    from multi_agent.frameworks.core.action_sink import ActionSink
    from multi_agent.trace.events import TraceStart, TraceEnd, AgentInfo, AccessDecisionEvent
    from multi_agent.trace.writer import TraceWriter
    from multi_agent.checker import check_trace, compute_metrics
    from multi_agent.completion_judge import evaluate_completion_checkpoints
    from multi_agent.frameworks.oai import tools as official_tools
    from agents import ModelSettings, RunConfig
    from agents.models.openai_provider import OpenAIProvider
    from agents.tracing import set_tracing_disabled
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning

    task_path = resolve_task_path(args.task)
    task, catalog = load_task_with_tools(task_path)
    if task.domain != "daily_life" or task.modality != "text_only" or task.input_assets or task.fixture:
        raise ValueError("pilot allows only text-only daily_life SQLite tasks without workspaces/assets")
    bank = get_bank(task.domain)
    if bank is None or any(not hasattr(bank, f"tool_{t.name}") for t in catalog.tools):
        raise ValueError("every domain tool must resolve to the local bank")
    bank.apply_task_metadata(task.metadata)
    output.mkdir(parents=True, exist_ok=False)
    ledger = CallLedger(output, args.request_budget)
    journal = RuntimeJournal(output / "runtime_events.jsonl")
    journal.emit("run_started", {"agents": [a.role for a in task.agents], "external_endpoints": ["user"]})
    lifecycle = ToolLifecycleRecorder(output, journal)
    original_build_tools = official_tools.build_domain_tools

    def instrumented_build_tools(*positional, **kwargs):
        return [lifecycle.wrap(tool) for tool in original_build_tools(*positional, **kwargs)]

    official_tools.build_domain_tools = instrumented_build_tools
    set_tracing_disabled(True)
    client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=45)
    provider = make_recording_provider(OpenAIProvider(openai_client=client, use_responses=True), ledger)

    class BoundedAdapter(OAIAdapter):
        def _build_run_config(self, model, **_kwargs):
            return RunConfig(model=model, model_provider=provider, tracing_disabled=True,
                             model_settings=ModelSettings(reasoning=Reasoning(effort="low"), max_tokens=1200,
                                                          store=False, parallel_tool_calls=True))

    ArrivalSink = make_arrival_sink(ActionSink, output, lifecycle, journal)

    manifest = {"status": "running", "started_at": utc_now(), "task_id": task.task_id, "task_path": str(task_path),
        "upstream_commit": revision, "model_requested": args.model, "endpoint": base_url,
        "request_budget": args.request_budget, "max_hub_turns": args.max_turns, "agent_timeout_seconds": args.timeout,
        "max_output_tokens_per_call": 1200, "api_reasoning_effort": "low", "client_retries": 0,
        "tools": "official in-memory SQLite DailyLifeBank only", "llm_judges": False,
        "deviations": ["bypass eager framework registry import; official OAI/core implementations unchanged",
                       "bounded model settings and API call budget", "extra emission-arrival recording",
                       "domain handler wrapper records actual pre-handler events and invocation links; no authorization decisions",
                       "shared process-local journal orders tool stages and communications without post-hoc sorting",
                       "adapter invoked directly; official deterministic completion and access checker invoked after run",
                       "no perturbations; not a paper-score reproduction or fixed-DAG benchmark"],
        "versions": {name: importlib.metadata.version(name) for name in ("openai-agents", "openai", "pydantic", "PyYAML")}}
    write_json(output / "RUN.json", manifest)
    trace_id = f"ha-api-pilot-{task.task_id}"
    outcome, completion, decisions = None, None, []
    caught_error = None
    with TraceWriter(output / "native_trace.jsonl") as writer:
        writer.write_event(TraceStart(trace_id=trace_id, task_id=task.task_id, domain=task.domain, framework="oai",
            model=args.model, agents=[AgentInfo(agent_id=a.role, role=a.role, useful_tools=a.useful_tools) for a in task.agents],
            n_rules=len(task.access_rules)))
        sink = ArrivalSink(run_id=trace_id, trace_writer=writer)
        try:
            outcome = await BoundedAdapter(agent_timeout=args.timeout).run(RunContext(
                run_id=trace_id, task=task, catalog=catalog, model=args.model, max_turns=args.max_turns,
                bank=bank, workspace=None, action_sink=sink), task.goal)
            actions = sink.actions
            bank.reconcile_trace_actions(actions)
            completion = await evaluate_completion_checkpoints(task, actions, outcome.final_output,
                                                               bank=bank, workspace=None, skip_judge=True)
            decisions = check_trace(actions, task.access_rules, domain=task.domain)
            for d in decisions:
                writer.write_event(AccessDecisionEvent(trace_id=trace_id, action_sequence_num=d.action_sequence_num,
                    violated=d.violated, violation_code=d.violation_code, severity=d.severity,
                    rule_description=d.rule.description if d.rule else None))
        except Exception as exc:
            caught_error = type(exc).__name__
        finally:
            actions = sink.actions
            nested_errors = sum(a.surface == "communication" and "[ERROR]" in (a.content or "") for a in actions)
            error = caught_error or (outcome.error if outcome else "missing_adapter_outcome")
            if ledger.failed or ledger.blocked or nested_errors:
                error = error or "api_or_nested_agent_failure"
            if not actions:
                error = error or "empty_execution"
            violation_counts = dict(Counter(d.violation_code.value if d.violation_code else "unknown" for d in decisions if d.violated))
            native_metrics = compute_metrics(trace_id, actions, decisions,
                resource_tools={t.name for t in catalog.tools if t.is_resource},
                task_completion_score=completion.task_completion_score if completion else 0)
            writer.write_event(TraceEnd(trace_id=trace_id, n_actions=len(actions),
                n_tool_calls=sum(a.surface == "tool_call" for a in actions),
                n_communications=sum(a.surface == "communication" for a in actions),
                n_violations=sum(d.violated for d in decisions), violations_by_code=violation_counts,
                error=error, **native_metrics.paper_fields()))
            bank.dump_to_file(output / "bank.sqlite")
            bank.close()
            await client.close()
            official_tools.build_domain_tools = original_build_tools

    journal.emit("run_finished", {"status": "failed" if error else "completed", "error_type": error})
    manifest.update(status="failed" if error else "completed", completed_at=utc_now(), error=error,
        runtime_journal_schema="runtime-journal-v1", runtime_journal_events=journal.count,
        api_calls_started=ledger.started, api_calls_completed=ledger.finished, api_calls_failed=ledger.failed,
        api_calls_blocked_by_budget=ledger.blocked, input_tokens=ledger.input_tokens, output_tokens=ledger.output_tokens,
        actions=len(actions), tool_calls=sum(a.surface == "tool_call" for a in actions),
        active_agent_roles=sorted({a.agent_role for a in actions}), nested_error_messages=nested_errors,
        recorded_violations=violation_counts, completion=jsonable(completion),
        tool_invocations_started=lifecycle.started, tool_handlers_returned=lifecycle.returned,
        tool_handlers_failed=lifecycle.failed,
        score_warning="SAR is the official post-run checker score. LLM completion portion skipped; native tcr is not a full task-completion score.",
        scope="API-generated MAS acquisition pilot with simulated tools; no trained monitor or baseline performance claim.")
    if outcome:
        (output / "final_output.txt").write_text(outcome.final_output, encoding="utf-8")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for path in (ROOT / 'code/src/paperalpha_runtime').glob("*.py"):
        target = snapshot / 'code/src/paperalpha_runtime' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    used_paths = {task_path, args.upstream / "multi_agent/tools/daily_life.yaml"}
    used_paths.update((args.upstream / "multi_agent/fixtures/daily_life").glob("*.json"))
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("multi_agent.") and filename and Path(filename).is_relative_to(args.upstream.resolve()):
            used_paths.add(Path(filename))
    for path in used_paths:
        destination = snapshot / "upstream" / path.resolve().relative_to(args.upstream.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.rglob("*")) if p.is_file() and p.name != "RUN.json"}
    write_json(output / "RUN.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "api_calls_started", "actions", "tool_calls", "recorded_violations", "error")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--task", default="dl-t14.yaml")
    parser.add_argument("--model", choices=["gpt-5.5"], default="gpt-5.5")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-budget", type=int, default=24)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.request_budget <= 32 or not 1 <= args.max_turns <= 12 or not 10 <= args.timeout <= 600:
        parser.error("pilot bounds: 1..32 calls, 1..12 hub turns, 10..600 seconds")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
