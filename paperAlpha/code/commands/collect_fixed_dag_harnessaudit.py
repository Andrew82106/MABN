"""Prospectively frozen fixed-DAG adaptation backed by HarnessAudit's simulator.

Official tasks/tools/checker stay unchanged. Replace ONLY hub/spoke orchestration
and its role prompts with a declared forward-only workflow. No hidden policies
are sent to agents/monitor. This is development acquisition, not original-paper
reproduction, safe/unsafe balancing, training, or a monitor performance claim.
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

from harnessaudit_cohort import ROOT, UPSTREAM, REVISION, check_upstream, inventory, sha, digest, write_new, now
from audit_harnessaudit_supervision import exact_resource_conflicts
from collect_harnessaudit_api_pilot import (CallLedger, ToolLifecycleRecorder, make_arrival_sink,
    make_recording_provider, import_official_oai_components, validate_endpoint, write_json, append_json, jsonable)

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.fixed_workflow import public_daily_life_workflow, node_instructions, execute_workflow
from paperalpha_runtime.journal import RuntimeJournal

USED_TASKS = {"dl-t11", "dl-t14", "dl-t18", "dl-t23", "dl-t16", "dl-t7"}
SETTINGS = {"model": "gpt-5.5", "reasoning": "low", "max_output_tokens": 1800,
            "request_budget": 32, "node_max_turns": 8, "node_timeout_seconds": 180, "attempts_per_task": 1}
SEED = "paperalpha-fixed-dag-pilot-20260919-v1"
LOCAL_SOURCES = ['code/commands/collect_fixed_dag_harnessaudit.py', 'code/commands/collect_harnessaudit_api_pilot.py',
                 'code/commands/harnessaudit_cohort.py', 'code/commands/audit_harnessaudit_supervision.py',
                 'code/commands/summarize_harnessaudit_cohort.py', 'code/commands/audit_harnessaudit_api_pilot.py']


def public_task(task):
    return {"goal": task.goal, "agents": [{k: getattr(a, k) for k in ("role", "description", "system_prompt")} for a in task.agents]}


def prepare(output, upstream=UPSTREAM):
    output, upstream = output.resolve(), upstream.resolve()
    if output.exists():
        raise ValueError("new plan directory required")
    check_upstream(upstream)
    import_official_oai_components(upstream)
    from multi_agent.loader import load_task_with_tools
    from multi_agent.frameworks.oai.tools import _build_params_schema
    from multi_agent.checker import _arg_matches_allowed
    candidates = inventory(upstream)
    inspected, eligible = [], []
    for candidate in candidates:
        if "outside_reviewed_text_only_sqlite_scope" in candidate["exclusion_reasons"]:
            inspected.append({**candidate, "dag_pilot_eligible": False})
            continue
        task, catalog = load_task_with_tools(upstream / candidate["task_path"])
        public = public_task(task)
        try:
            workflow = public_daily_life_workflow(public["agents"])
            unsupported = False
        except ValueError:
            workflow, unsupported = None, True
        schemas = {t.name: _build_params_schema(t) for t in catalog.tools}
        conflicts = exact_resource_conflicts([r.model_dump(mode="json") for r in task.access_rules], schemas,
            [(a.role, t.name) for a in task.agents for t in catalog.tools], _arg_matches_allowed)
        reasons = []
        if task.task_id in USED_TASKS:
            reasons.append("previous_development_task")
        if unsupported:
            reasons.append("unsupported_public_functional_template")
        if any(c["unavoidable_for_schema_valid_call"] for c in conflicts):
            reasons.append("contradictory_required_literal_resource_constraints")
        row = {**candidate, "dag_pilot_eligible": not reasons, "dag_exclusion_reasons": reasons,
               "literal_conflicts": conflicts, "public_configuration": public, "workflow": workflow}
        inspected.append(row)
        if not reasons:
            eligible.append(row)
    # Two different categories, selected without new outcome input.
    chosen, categories = [], set()
    for row in sorted(eligible, key=lambda r: (digest([SEED, r["task_id"]]), r["task_id"])):
        if row["category"] not in categories:
            chosen.append(row)
            categories.add(row["category"])
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("fewer than two eligible categories; do not silently relax policy checks")
    plan = {"schema": "fixed-dag-harnessaudit-development-v1", "created_at": now(),
            "upstream": str(upstream), "upstream_commit": REVISION, "selection_seed": SEED,
            "settings": SETTINGS, "inventory": inspected, "tasks": chosen, "max_total_requests": 64,
            "selection": "Two distinct categories by fixed SHA order after metadata/satisfiability screening; no new outcome selection.",
            "protocol": {"workflow": "Declared role DAG; deterministic serial topological scheduling; no backward messaging or re-entry.",
                         "sources": "Original goal and role descriptions only; hidden rules/useful-tool sets stay evaluation-side.",
                         "tools": "All original daily-life tools available to every role; no scope blocking by this monitor.",
                         "labels": "Retain unchanged official rule labels, stratified high/low/resource; normal/harm not presumed.",
                         "screening_limits": "Literal conflicts only; not complete policy correctness, absence of empty-field artifacts, or verified harm labels.",
                         "negative_control": "Unperturbed attempts, not assumed safe or complete; retain every failure and positive outcome.",
                         "target": "Development readiness only. No fitting, final test, online latency or SOTA claims.",
                         "deviations": "New code orchestration and explicit role prompts; not original HarnessAudit hub/spoke results.",
                         "dag_scope": "Inter-agent control/message DAG; local tool loops and shared database dependencies are not a discovered causal graph."}}
    output.mkdir(parents=True)
    local = [ROOT / p for p in LOCAL_SOURCES] + list((ROOT / 'code/src/paperalpha_runtime').glob("*.py"))
    upstream_files = list((upstream / "multi_agent").rglob("*.py"))
    upstream_files += [upstream / "multi_agent/tools/daily_life.yaml", *(upstream / "multi_agent/fixtures/daily_life").glob("*.json")]
    upstream_files += [upstream / r["task_path"] for r in inspected]
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
    return {"tasks": [t["task_id"] for t in chosen], "eligible_tasks": len(eligible),
            "inspected_text_tasks": sum("literal_conflicts" in r for r in inspected), "plan_sha256": plan["plan_sha256"]}


def load_plan(path):
    path = path.resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "fixed-dag-harnessaudit-development-v1" or digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
        raise ValueError("unsupported or changed plan")
    upstream = Path(plan["upstream"])
    check_upstream(upstream)
    for relative, expected in plan["source_sha256"].items():
        base, name = relative.split("/", 1)
        current = (ROOT if base == "local" else upstream) / name
        if sha(current) != expected or sha(path.parent / "source_snapshot" / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    return plan


async def collect(plan_path, task_id):
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    matches = [t for t in plan["tasks"] if t["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError("task must be in the frozen two-task plan")
    selected, settings = matches[0], plan["settings"]
    output = plan_path.parent / "runs" / task_id
    if output.exists():
        raise ValueError("attempt already exists; no restart, overwrite or replacement")
    base_url = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
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
        raise ValueError("task differs from frozen public configuration")
    bank = get_bank(task.domain)
    if task.domain != "daily_life" or task.modality != "text_only" or task.input_assets or task.fixture or bank is None:
        raise ValueError("only reviewed text-only simulated daily-life tasks are supported")
    if any(not hasattr(bank, f"tool_{t.name}") for t in catalog.tools):
        raise ValueError("all tools must use local simulator handlers")
    bank.apply_task_metadata(task.metadata)
    output.mkdir(parents=True, exist_ok=False)
    # Populate per-run snapshots before any model calls. Existing verifier uses
    # this exact directory; only this task YAML is included here.
    for relative in plan["source_sha256"]:
        if relative.startswith("upstream/multi_agent/tasks/") and relative != "upstream/" + selected["task_path"]:
            continue
        src = plan_path.parent / "source_snapshot" / relative
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(src.read_bytes())
    workflow = selected["workflow"]
    journal = RuntimeJournal(output / "runtime_events.jsonl")
    initial = {"agents": workflow["nodes"], "external_endpoints": ["user"],
               "public_configuration": selected["public_configuration"], "declared_workflow": workflow,
               "workflow_source": "prospectively frozen code orchestration, not inference or hidden task policy"}
    journal.emit("run_started", initial)
    lifecycle = ToolLifecycleRecorder(output, journal)
    ledger = CallLedger(output, settings["request_budget"])
    set_tracing_disabled(True)
    client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=45)
    provider = make_recording_provider(OpenAIProvider(openai_client=client, use_responses=True), ledger)
    run_config = RunConfig(model=settings["model"], model_provider=provider, tracing_disabled=True,
        model_settings=ModelSettings(reasoning=Reasoning(effort=settings["reasoning"]), store=False,
                                     max_tokens=settings["max_output_tokens"], parallel_tool_calls=False))
    tools = [lifecycle.wrap(t) for t in build_domain_tools(catalog)]
    ArrivalSink = make_arrival_sink(ActionSink, output, lifecycle, journal)
    trace_id = "ha-fixed-dag-" + task_id
    manifest = {"schema": "fixed-dag-api-run-v1", "status": "running", "started_at": now(),
                "task_id": task_id, "upstream_commit": REVISION, "plan_sha256": plan["plan_sha256"],
                "model_requested": settings["model"], "request_budget": settings["request_budget"],
                "settings": settings, "endpoint": base_url, "workflow": workflow, "protocol": plan["protocol"],
                "versions": {n: importlib.metadata.version(n) for n in ("openai", "openai-agents", "pydantic", "PyYAML")}}
    write_json(output / "RUN.json", manifest)
    outputs, completion, decisions, error = {}, None, [], None
    actors = {a["role"]: a for a in selected["public_configuration"]["agents"]}
    with TraceWriter(output / "native_trace.jsonl") as writer:
        writer.write_event(TraceStart(trace_id=trace_id, task_id=task_id, domain=task.domain, framework="fixed_dag_oai",
            model=settings["model"], agents=[AgentInfo(agent_id=a.role, role=a.role, useful_tools=a.useful_tools) for a in task.agents], n_rules=len(task.access_rules)))
        sink = ArrivalSink(run_id=trace_id, trace_writer=writer)

        def emit(sender, target, content):
            sink.emit_communication(agent_id=sender, agent_role=sender, target_agent=target, target_role=target,
                                    content=content, raw_event={"framework": "fixed_dag_oai", "delivery": "declared_predecessor_report"})

        async def execute(node, content):
            begin = ledger.started
            append_json(output / "node_runs.jsonl", {"node": node, "stage": "started", "at": now(),
                "first_possible_call": begin + 1, "input": content, "instructions": node_instructions(actors[node])})
            bank.set_current_agent(node)
            agent = Agent(name=node, instructions=node_instructions(actors[node]), tools=list(tools), model=settings["model"])
            context = MultiAgentContext(action_sink=sink, bank=bank, workspace=None, agent_role=node, run_config=run_config)
            try:
                result = await asyncio.wait_for(Runner.run(agent, content, context=context, run_config=run_config,
                    max_turns=settings["node_max_turns"]), timeout=settings["node_timeout_seconds"])
                text = str(result.final_output or "")
                append_json(output / "node_runs.jsonl", {"node": node, "stage": "completed", "at": now(),
                    "calls_started": ledger.started - begin, "last_call": ledger.started, "output": text})
                outputs[node] = text
                return text
            except BaseException as exc:
                append_json(output / "node_runs.jsonl", {"node": node, "stage": "failed", "at": now(),
                    "calls_started": ledger.started - begin, "last_call": ledger.started, "error_type": type(exc).__name__})
                raise

        try:
            outputs = await execute_workflow(workflow, task.goal, execute, emit)
            emit("concierge_coordinator", "user", outputs["concierge_coordinator"])
        except Exception as exc:
            error = type(exc).__name__
        finally:
            actions = sink.actions
            if ledger.failed or ledger.blocked or not actions:
                error = error or "api_failure_budget_or_empty_execution"
            # Labels are computed even for partial runs, but not promoted to
            # full-episode outcomes when a node failed.
            decisions = check_trace(actions, task.access_rules, domain=task.domain)
            for d in decisions:
                writer.write_event(AccessDecisionEvent(trace_id=trace_id, action_sequence_num=d.action_sequence_num,
                    violated=d.violated, violation_code=d.violation_code, severity=d.severity,
                    rule_description=d.rule.description if d.rule else None))
            bank.reconcile_trace_actions(actions)
            completion = await evaluate_completion_checkpoints(task, actions, outputs.get("concierge_coordinator", ""),
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
    write_new(output / "node_outputs.json", outputs)
    manifest.update(status="failed" if error else "completed", completed_at=now(), error=error,
        api_calls_started=ledger.started, api_calls_completed=ledger.finished, api_calls_failed=ledger.failed,
        api_calls_blocked_by_budget=ledger.blocked, input_tokens=ledger.input_tokens, output_tokens=ledger.output_tokens,
        actions=len(actions), tool_calls=sum(a.surface == "tool_call" for a in actions),
        tool_invocations_started=lifecycle.started, tool_handlers_returned=lifecycle.returned, tool_handlers_failed=lifecycle.failed,
        runtime_journal_schema="runtime-journal-v1", runtime_journal_events=journal.count,
        completed_nodes=list(outputs), recorded_violations=counts, completion=jsonable(completion),
        episode_rule_label=None if error else int(any(d.violated for d in decisions)),
        label_meaning="Unchanged official rule matches, not actual harm; policy preflight is only partial.",
        no_monitor_training=True, no_monitor_performance_claim=True)
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "RUN.json"}
    write_json(output / "RUN.json", manifest)
    return {k: manifest[k] for k in ("task_id", "status", "api_calls_started", "tool_calls", "completed_nodes", "recorded_violations", "episode_rule_label")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    result = prepare(args.out) if args.mode == "prepare" else asyncio.run(collect(args.plan, args.task))
    print(json.dumps(result, ensure_ascii=False))
