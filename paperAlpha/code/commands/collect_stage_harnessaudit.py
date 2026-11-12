"""Actual API execution of a prospectively frozen stage-DAG adaptation.

Reuses original HarnessAudit tools/checker and SDK runner. Two already-used
development tasks; not held-out data, an original-benchmark reproduction, or
monitor training. No retries/replacement; every failed attempt is preserved.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from harnessaudit_cohort import ROOT, REVISION, sha, digest, now, write_new, check_upstream
from collect_fixed_dag_harnessaudit import load_plan as load_original_plan, public_task
from collect_harnessaudit_api_pilot import (CallLedger, ToolLifecycleRecorder, make_recording_provider,
    make_arrival_sink, import_official_oai_components, validate_endpoint, write_json, append_json, jsonable)

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.stage_workflow import daily_life_relay_stages, execute_stages
from paperalpha_runtime.stage_capture import StageRuntimeJournal, stage_instructions

SETTINGS = {"model": "gpt-5.5", "reasoning": "low", "max_output_tokens": 1800,
            "request_budget": 32, "stage_max_turns": 8, "stage_timeout_seconds": 180, "attempts_per_task": 1}
LOCAL_SOURCES = ['code/commands/collect_stage_harnessaudit.py', 'code/commands/collect_fixed_dag_harnessaudit.py',
    'code/commands/collect_harnessaudit_api_pilot.py', 'code/commands/harnessaudit_cohort.py',
    'code/commands/audit_harnessaudit_supervision.py', 'code/commands/summarize_harnessaudit_cohort.py',
    'code/commands/audit_harnessaudit_api_pilot.py']


class StageLedger(CallLedger):
    def __init__(self, output, budget, journal):
        super().__init__(output, budget)
        self.journal = journal

    def begin(self, request):
        stage, actor, _ = self.journal.active()
        return super().begin({**request, "stage_id": stage, "actor": actor})


class StageLifecycleRecorder(ToolLifecycleRecorder):
    def record(self, payload):
        row = self.journal.emit(payload["event_type"], jsonable(payload))
        append_json(self.output / "tool_lifecycle.jsonl", {**row["payload"], "journal_sequence": row["sequence"]})


def prepare(original_plan, output):
    original = load_original_plan(original_plan.resolve())
    if output.exists():
        raise ValueError("new stage plan directory required")
    upstream = Path(original["upstream"])
    import_official_oai_components(upstream)
    from multi_agent.loader import load_task_with_tools
    from multi_agent.checker import check_action
    from multi_agent.schemas.actions import ObservableAction
    tasks = []
    for previous in original["tasks"]:
        task, _ = load_task_with_tools(upstream / previous["task_path"])
        public = public_task(task)
        workflow = daily_life_relay_stages(public["agents"])
        rules = [r for r in task.access_rules if r.access_type == "information.communication"]
        conflicts = []
        for index, (source, target) in enumerate(workflow["edges"], 1):
            actor, recipient = workflow["owners"][source], workflow["owners"][target]
            action = ObservableAction(run_id="prospective-stage-check", sequence_num=index, surface="communication",
                agent_id=actor, agent_role=actor, target_agent=recipient, target_role=recipient, content="")
            conflicts += [d.rule.description for d in check_action(action, rules, domain=task.domain) if d.violated]
        if conflicts:
            raise ValueError("new template still has communication conflicts; do not silently replace tasks")
        tasks.append({k: previous[k] for k in ("task_id", "task_path", "task_sha256", "category")})
        tasks[-1].update(public_configuration=public, stage_workflow=workflow, static_communication_conflicts=conflicts)
    plan = {"schema": "stage-harnessaudit-development-v1", "created_at": now(), "upstream": str(upstream),
        "upstream_commit": REVISION, "original_plan_sha256": original["plan_sha256"],
        "settings": SETTINGS, "max_total_requests": 64, "tasks": tasks,
        "protocol": {"selection": "Both previously used fixed-DAG development tasks; no replacement by outcomes.",
            "contexts": "Fresh Agent/Runner.run per functional stage; no sessions, prior response IDs, or inherited transcripts.",
            "observations": "Explicit stage scopes on every request, tool lifecycle and message; physical actor unchanged.",
            "tools": "All original tools for all roles; original simulator only; passive monitor does not block tools.",
            "policies": "Communication preflight is evaluation-side. Hidden policies/useful_tools never enter agent prompts or monitor.",
            "labels": "Unchanged original rules; partial run label null; no verified harm/normal labels inferred.",
            "comparison": "New workflow and stage prompts, stochastic new executions; not a controlled safety-gain comparison.",
            "scope": "Development stage-aware capture and replay, not training, independent test or MAS baseline performance."}}
    output.mkdir(parents=True)
    local = [ROOT / p for p in LOCAL_SOURCES] + list((ROOT / 'code/src/paperalpha_runtime').glob("*.py"))
    upstream_files = list((upstream / "multi_agent").rglob("*.py"))
    upstream_files += [upstream / "multi_agent/tools/daily_life.yaml", *(upstream / "multi_agent/fixtures/daily_life").glob("*.json")]
    upstream_files += [upstream / t["task_path"] for t in tasks]
    hashes = {}
    for source, relative in [(p, "local/" + p.relative_to(ROOT).as_posix()) for p in local] + [(p, "upstream/" + p.relative_to(upstream).as_posix()) for p in upstream_files]:
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(source.read_bytes())
        hashes[relative] = sha(dest)
    plan["source_sha256"] = hashes
    plan["plan_sha256"] = digest(plan)
    write_new(output / "PLAN.json", plan)
    return {"tasks": [t["task_id"] for t in tasks], "plan_sha256": plan["plan_sha256"], "maximum_requests": 64}


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("schema") != "stage-harnessaudit-development-v1" or
            digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256")):
        raise ValueError("unsupported or changed stage acquisition plan")
    upstream = Path(plan["upstream"])
    check_upstream(upstream)
    for relative, expected in plan["source_sha256"].items():
        base, name = relative.split("/", 1)
        if sha((ROOT if base == "local" else upstream) / name) != expected or sha(path.parent / "source_snapshot" / relative) != expected:
            raise ValueError("frozen source changed: " + relative)
    return plan


async def collect(plan_path, task_id):
    plan = load_plan(plan_path)
    choices = [t for t in plan["tasks"] if t["task_id"] == task_id]
    if len(choices) != 1:
        raise ValueError("task not in frozen plan")
    selected, settings = choices[0], plan["settings"]
    output = plan_path.parent / "runs" / task_id
    if output.exists():
        raise ValueError("one attempt only; do not restart or overwrite")
    endpoint = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("environment-only API key required")
    upstream = Path(plan["upstream"])
    import_official_oai_components(upstream)
    from multi_agent.loader import load_task_with_tools
    from multi_agent.banks import get_bank
    from multi_agent.frameworks.core.action_sink import ActionSink
    from multi_agent.frameworks.oai.adapter import MultiAgentContext
    from multi_agent.frameworks.oai.tools import build_domain_tools
    from multi_agent.trace.events import TraceStart, TraceEnd, AgentInfo, AccessDecisionEvent
    from multi_agent.trace.writer import TraceWriter
    from multi_agent.checker import check_trace, compute_metrics
    from multi_agent.completion_judge import evaluate_completion_checkpoints
    from agents import Agent, ModelSettings, RunConfig, Runner
    from agents.models.openai_provider import OpenAIProvider
    from agents.tracing import set_tracing_disabled
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning
    task, catalog = load_task_with_tools(upstream / selected["task_path"])
    if public_task(task) != selected["public_configuration"] or sha(upstream / selected["task_path"]) != selected["task_sha256"]:
        raise ValueError("changed task")
    bank = get_bank(task.domain)
    if task.domain != "daily_life" or task.modality != "text_only" or task.input_assets or task.fixture or bank is None:
        raise ValueError("only reviewed local text-only simulator supported")
    if any(not hasattr(bank, f"tool_{t.name}") for t in catalog.tools):
        raise ValueError("all tools must use original local simulator handlers")
    bank.apply_task_metadata(task.metadata)
    output.mkdir(parents=True)
    for relative in plan["source_sha256"]:
        if relative.startswith("upstream/multi_agent/tasks/") and relative != "upstream/" + selected["task_path"]:
            continue
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write((plan_path.parent / "source_snapshot" / relative).read_bytes())
    workflow = selected["stage_workflow"]
    journal = StageRuntimeJournal(output / "runtime_events.jsonl", workflow)
    journal.emit("run_started", {"agents": [a.role for a in task.agents], "external_endpoints": ["user"],
        "public_configuration": selected["public_configuration"], "declared_stage_workflow": workflow})
    life, ledger = StageLifecycleRecorder(output, journal), StageLedger(output, settings["request_budget"], journal)
    set_tracing_disabled(True)
    client = AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=45)
    provider = make_recording_provider(OpenAIProvider(openai_client=client, use_responses=True), ledger)
    config = RunConfig(model=settings["model"], model_provider=provider, tracing_disabled=True,
        model_settings=ModelSettings(reasoning=Reasoning(effort=settings["reasoning"]), store=False,
            max_tokens=settings["max_output_tokens"], parallel_tool_calls=False))
    tools = [life.wrap(t) for t in build_domain_tools(catalog)]
    Sink = make_arrival_sink(ActionSink, output, life, journal)
    manifest = {"schema": "stage-dag-api-run-v1", "status": "running", "started_at": now(), "task_id": task_id,
        "upstream_commit": REVISION, "plan_sha256": plan["plan_sha256"], "model_requested": settings["model"],
        "request_budget": settings["request_budget"], "settings": settings, "endpoint": endpoint,
        "stage_workflow": workflow, "protocol": plan["protocol"],
        "versions": {n: importlib.metadata.version(n) for n in ("openai", "openai-agents", "pydantic", "PyYAML")}}
    write_json(output / "RUN.json", manifest)
    outputs, error = {}, None
    actors = {a["role"]: a for a in selected["public_configuration"]["agents"]}
    trace_id = "ha-stage-dag-" + task_id
    with TraceWriter(output / "native_trace.jsonl") as writer:
        writer.write_event(TraceStart(trace_id=trace_id, task_id=task_id, domain=task.domain, framework="stage_dag_oai",
            model=settings["model"], agents=[AgentInfo(agent_id=a.role, role=a.role, useful_tools=a.useful_tools) for a in task.agents], n_rules=len(task.access_rules)))
        sink = Sink(run_id=trace_id, trace_writer=writer)
        def emit(delivery):
            with journal.scope(delivery["stage_id"], delivery.get("target_stage_id")):
                sink.emit_communication(agent_id=delivery["actor"], agent_role=delivery["actor"],
                    target_agent=delivery["target"], target_role=delivery["target"], content=delivery["content"],
                    raw_event={"framework": "stage_dag_oai", "stage_id": delivery["stage_id"],
                               "target_stage_id": delivery.get("target_stage_id")})
        async def execute(stage, actor, content):
            begin, first_sequence = ledger.started, journal.count
            prompt = stage_instructions(actors[actor], stage, workflow["purposes"][stage])
            append_json(output / "stage_runs.jsonl", {"stage_id": stage, "actor": actor, "status": "started", "at": now(),
                "first_possible_call": begin + 1, "journal_before": first_sequence, "input": content, "instructions": prompt})
            bank.set_current_agent(actor)
            agent = Agent(name=actor, instructions=prompt, tools=list(tools), model=settings["model"])
            context = MultiAgentContext(action_sink=sink, bank=bank, workspace=None, agent_role=actor, run_config=config)
            try:
                with journal.scope(stage):
                    result = await asyncio.wait_for(Runner.run(agent, content, context=context, run_config=config,
                        max_turns=settings["stage_max_turns"]), timeout=settings["stage_timeout_seconds"])
                text = result.final_output
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty/nontext stage report")
                append_json(output / "stage_runs.jsonl", {"stage_id": stage, "actor": actor, "status": "completed", "at": now(),
                    "calls_started": ledger.started - begin, "last_call": ledger.started, "journal_after": journal.count, "output": text})
                outputs[stage] = text
                return text
            except BaseException as exc:
                append_json(output / "stage_runs.jsonl", {"stage_id": stage, "actor": actor, "status": "failed", "at": now(),
                    "calls_started": ledger.started - begin, "last_call": ledger.started, "journal_after": journal.count,
                    "error_type": type(exc).__name__})
                raise
        try:
            outputs = await execute_stages(workflow, task.goal, execute, emit)
            emit({"actor": workflow["owners"]["stage:final"], "target": "user", "stage_id": "stage:final", "content": outputs["stage:final"]})
        except Exception as exc:
            error = type(exc).__name__
        finally:
            actions = sink.actions
            if ledger.failed or ledger.blocked or not actions:
                error = error or "api_failure_budget_or_empty_execution"
            decisions = check_trace(actions, task.access_rules, domain=task.domain)
            for d in decisions:
                writer.write_event(AccessDecisionEvent(trace_id=trace_id, action_sequence_num=d.action_sequence_num,
                    violated=d.violated, violation_code=d.violation_code, severity=d.severity,
                    rule_description=d.rule.description if d.rule else None))
            bank.reconcile_trace_actions(actions)
            completion = await evaluate_completion_checkpoints(task, actions, outputs.get("stage:final", ""),
                                                               bank=bank, workspace=None, skip_judge=True)
            counts = dict(Counter(d.violation_code.value for d in decisions if d.violated))
            metrics = compute_metrics(trace_id, actions, decisions, resource_tools={t.name for t in catalog.tools if t.is_resource},
                                      task_completion_score=completion.task_completion_score)
            writer.write_event(TraceEnd(trace_id=trace_id, n_actions=len(actions), n_tool_calls=sum(a.surface == "tool_call" for a in actions),
                n_communications=sum(a.surface == "communication" for a in actions), n_violations=sum(d.violated for d in decisions),
                violations_by_code=counts, error=error, **metrics.paper_fields()))
            bank.dump_to_file(output / "bank.sqlite")
            bank.close()
            await client.close()
    journal.emit("run_finished", {"status": "failed" if error else "completed", "error_type": error})
    write_new(output / "stage_outputs.json", outputs)
    manifest.update(status="failed" if error else "completed", completed_at=now(), error=error,
        api_calls_started=ledger.started, api_calls_completed=ledger.finished, api_calls_failed=ledger.failed,
        api_calls_blocked_by_budget=ledger.blocked, input_tokens=ledger.input_tokens, output_tokens=ledger.output_tokens,
        actions=len(actions), tool_calls=sum(a.surface == "tool_call" for a in actions),
        tool_invocations_started=life.started, tool_handlers_returned=life.returned, tool_handlers_failed=life.failed,
        runtime_journal_schema="runtime-journal-v1", runtime_journal_events=journal.count,
        completed_stages=list(outputs), recorded_violations=counts, completion=jsonable(completion),
        episode_rule_label=None if error else int(any(d.violated for d in decisions)),
        label_meaning="Unchanged official rules, not confirmed harm; previously used development tasks.",
        no_monitor_training=True, no_monitor_performance_claim=True)
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "RUN.json"}
    write_json(output / "RUN.json", manifest)
    return {k: manifest[k] for k in ("task_id", "status", "api_calls_started", "tool_calls", "completed_stages", "recorded_violations", "episode_rule_label")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--original-plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    result = prepare(args.original_plan.resolve(), args.out.resolve()) if args.mode == "prepare" else asyncio.run(collect(args.plan.resolve(), args.task))
    print(json.dumps(result, ensure_ascii=False))
