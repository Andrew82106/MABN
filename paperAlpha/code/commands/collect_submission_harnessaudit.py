"""Frozen acquisition for the reviewed, stateful HarnessAudit DAG adaptation.

This collects development evidence, not final test scores. Original actionwise
labels remain separate from any future temporal-policy extension. No network
call occurs in prepare/verify; run requires an environment-only credential.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.harnessaudit_sandbox_adapter import HarnessAuditSandboxAdapter, UPSTREAM_COMMIT
from paperalpha_runtime.fixed_workflow import topological_order

DEFAULT_UPSTREAM = ROOT / 'code/external/HarnessAudit'
DEFAULT_OUTPUT = ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit'
SETTINGS = {"model": os.environ.get("PAPERALPHA_API_MODEL", "gpt-5.5"), "reasoning": "low", "max_calls": 48,
            "max_turns": 10, "max_output_tokens": 1600, "turn_timeout_seconds": 60,
            "node_timeout_seconds": 300, "max_tools_per_node": 9, "concurrency": 2}
LOCAL_SOURCES = ('code/commands/collect_submission_harnessaudit.py',
    'code/src/paperalpha_runtime/__init__.py',
    'code/src/paperalpha_runtime/harnessaudit_sandbox_adapter.py',
    'code/src/paperalpha_runtime/harnessaudit_node_executor.py',
    'code/src/paperalpha_runtime/fixed_workflow.py', 'code/src/paperalpha_runtime/journal.py')
SETUP_KEYS = {"client_id", "customer_id", "portfolio_id", "request_id", "case_id", "seed_trades"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def native_components(upstream):
    """Bypass only the framework registry's unrelated eager integrations.

    Loader, schemas, checker and recognizers execute their unmodified author
    files. This is an import shim, not a replacement policy interpreter.
    """
    upstream = Path(upstream).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    for name in ("multi_agent", "multi_agent.frameworks"):
        current = sys.modules.get(name)
        if current is not None:
            paths = [Path(p).resolve() for p in getattr(current, "__path__", [])]
            if not paths or any(not p.is_relative_to(upstream) for p in paths):
                raise ValueError("conflicting upstream package")
    importlib.import_module("multi_agent")
    name = "multi_agent.frameworks"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(upstream / "multi_agent/frameworks")]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    from multi_agent.loader import load_task_with_tools
    from multi_agent.checker import check_action, check_trace
    from multi_agent.schemas.actions import ObservableAction
    return load_task_with_tools, check_action, check_trace, ObservableAction


def compile_stages(stages, output_nodes, roles):
    if not isinstance(stages, list) or not stages:
        raise ValueError("explicit reviewed stage plan required")
    required = {"stage_id", "role", "purpose", "parents"}
    if any(not isinstance(s, dict) or set(s) != required for s in stages):
        raise ValueError("stage fields must be stage_id, role, purpose, parents")
    for stage in stages:
        if (not isinstance(stage["stage_id"], str) or not stage["stage_id"].strip()
                or stage["role"] not in roles or not isinstance(stage["purpose"], str)
                or not stage["purpose"].strip() or not isinstance(stage["parents"], list)):
            raise ValueError("each stage needs an original role and public purpose")
    workflow = {"nodes": [s["stage_id"] for s in stages],
                "edges": [[p, s["stage_id"]] for s in stages for p in s["parents"]]}
    topological_order(workflow)
    if (not isinstance(output_nodes, list) or len(output_nodes) != 1
            or output_nodes[0] not in workflow["nodes"]
            or any(a == output_nodes[0] for a, _ in workflow["edges"])):
        raise ValueError("one explicitly delivered final sink required")
    return workflow, {s["stage_id"]: {"role": s["role"], "source": "reviewed public task-stage plan"}
                      for s in stages}, {s["stage_id"]: {"purpose": s["purpose"], "source": "reviewed original public goal and role instructions"}
                                         for s in stages}


def adapter_for(plan, case):
    settings = plan["settings"]
    return HarnessAuditSandboxAdapter(upstream=plan["upstream"], domain=case["domain"],
        seed_dir=Path(plan["upstream"]) / "multi_agent/fixtures" / case["domain"],
        public_task=case["public_task"], workflow=case["workflow"], role_bindings=case["role_bindings"],
        stage_purposes=case["stage_purposes"], output_nodes=case["output_nodes"],
        visible_policy=case["visible_policy"], bank_setup_private=case["bank_setup_private"],
        evaluation_config=case["evaluation_config"], state_visibility=(),
        max_tools_per_node=settings["max_tools_per_node"], node_timeout_seconds=settings["node_timeout_seconds"])


def prepare(candidates_path, task_ids, out=DEFAULT_OUTPUT, upstream=DEFAULT_UPSTREAM):
    """Select named tasks from a reviewed candidate list, never by outcomes."""
    out, upstream = Path(out).resolve(), Path(upstream).resolve()
    if out.exists():
        raise ValueError("cohort already exists; never overwrite or resample a frozen run")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", UPSTREAM_COMMIT,
                            "--", "multi_agent", "LICENSE"], capture_output=True)
    if head != UPSTREAM_COMMIT or clean.returncode:
        raise ValueError("unchanged pinned author task, evaluator, tool and Bank sources required")
    candidates = read(candidates_path)["candidates"]
    if len(set(task_ids)) != len(task_ids) or not task_ids:
        raise ValueError("unique explicitly selected task IDs required")
    chosen = []
    for task_id in task_ids:
        matches = [c for c in candidates if c["task_id"] == task_id]
        if len(matches) != 1:
            raise ValueError("selected task must have exactly one reviewed plan: " + task_id)
        chosen.append(matches[0])
    loader, check_action, _, Action = native_components(upstream)
    cases = []
    for candidate in chosen:
        relative = candidate["task_path"]
        task_path = (upstream / relative).resolve()
        if not task_path.is_relative_to(upstream / "multi_agent/tasks") or sha(task_path) != candidate["task_sha256"]:
            raise ValueError("task path or reviewed hash differs")
        task, _ = loader(task_path)
        if (task.task_id != candidate["task_id"] or not re.fullmatch(r"(?:off|ec|fin)-t[0-9]+", task.task_id)
                or task.domain not in {"office", "ecommerce", "finance"}
                or task.modality != "text_only" or task.input_assets or task.fixture):
            raise ValueError("task outside reviewed scope")
        public = {"goal": task.goal, "agents": [{k: getattr(a, k) for k in ("role", "description", "system_prompt")}
                                              for a in task.agents]}
        workflow, bindings, purposes = compile_stages(candidate["stage_plan"], candidate["output_nodes"],
                                                      {a.role for a in task.agents})
        communication = [r for r in task.access_rules if r.access_type == "information.communication"]
        routes = workflow["edges"] + [[node, "user"] for node in candidate["output_nodes"]]
        for index, (source, target) in enumerate(routes, 1):
            action = Action(run_id="prospective-route-check", sequence_num=index, surface="communication",
                            agent_id=source, agent_role=bindings[source]["role"], target_agent=target,
                            target_role="user" if target == "user" else bindings[target]["role"], content="")
            if any(d.violated for d in check_action(action, communication, domain=task.domain)):
                raise ValueError("declared stage route conflicts with author communication policy")
        rules = [r.model_dump(mode="json") for r in task.access_rules]
        setup = {k: v for k, v in task.metadata.items() if k in SETUP_KEYS} if task.domain == "finance" else {}
        cases.append({"case_id": task.task_id, "task_path": relative, "task_sha256": sha(task_path),
            "family_id": candidate["family_id"], "domain": task.domain, "category": task.category,
            "static_review": {"selection_status": candidate.get("selection_status"), "review": candidate.get("review")},
            "public_task": public, "workflow": workflow, "role_bindings": bindings,
            "stage_purposes": purposes, "output_nodes": candidate["output_nodes"],
            "visible_policy": {"source": "Unchanged author access rules, explicitly disclosed equally to all monitors; not agent prompts",
                               "rules": rules}, "bank_setup_private": setup,
            "evaluation_config": {"task_id": task.task_id, "access_rules": rules,
                "completion_checkpoints": task.model_dump(mode="json")["completion_checkpoints"]}})
    plan = {"schema": "submission-harnessaudit-development-v1", "upstream": str(upstream),
        "upstream_commit": UPSTREAM_COMMIT, "settings": SETTINGS, "cases": cases,
        "candidate_file_sha256": sha(candidates_path), "maximum_api_calls": len(cases) * SETTINGS["max_calls"],
        "protocol": {"purpose": "Development acquisition only; all attempts retained, no outcome filtering or replacement",
            "workflow": "Explicit acyclic stage unrolling; repeated stages preserve original actor roles; serial scheduling",
            "visibility": "Trace-only, unchanged task/roles and declared policy available equally to monitors; no backend state",
            "agent_policy": "Original goal and role instructions plus public stage purpose, not hidden access-rule labels",
            "labels": "Unchanged author actionwise labels only; incomplete episodes never get an assumed safe label",
            "task_success": "Not inferred from executor completion, report text, or a Pending tool receipt",
            "temporal_extensions": "Not included or inferred here; require separately frozen contracts and independent state oracle",
            "attacks": "Unperturbed author tasks only; not declared benign, not malicious-agent ground truth",
            "api": "Real model replies, no automatic retries, durable claim before request; timeout/unknown receipt retained"}}
    for case in cases:
        adapter_for(plan, case)  # checks pinned Bank sources, private setup and memory-only state
    sources = [("local/" + p, ROOT / p) for p in LOCAL_SOURCES]
    sources += [("upstream/" + p.relative_to(upstream).as_posix(), p)
                for p in (upstream / "multi_agent").rglob("*.py")]
    sources += [("upstream/LICENSE", upstream / "LICENSE")]
    for domain in sorted({c["domain"] for c in cases}):
        sources.append(("upstream/multi_agent/tools/" + domain + ".yaml", upstream / "multi_agent/tools" / (domain + ".yaml")))
        sources += [("upstream/" + p.relative_to(upstream).as_posix(), p)
                    for p in (upstream / "multi_agent/fixtures" / domain).rglob("*.json")]
    sources += [("upstream/" + c["task_path"], upstream / c["task_path"]) for c in cases]
    if any(not path.is_file() for _, path in sources):
        raise ValueError("complete reviewed acquisition source required before freezing")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(candidates_path, out / "CANDIDATES.json")
    plan["source_sha256"] = {}
    for relative, path in sources:
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        plan["source_sha256"][relative] = sha(destination)
    plan["environment"] = {"python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("openai", "openai-agents", "pydantic", "PyYAML")}}
    plan["sha256"] = digest(plan)
    write_new(out / "PLAN.json", plan)
    return {"plan_sha256": plan["sha256"], "cases": [c["case_id"] for c in cases], "maximum_api_calls": plan["maximum_api_calls"]}


def load_plan(out):
    out = Path(out)
    plan = read(out / "PLAN.json")
    if (plan.get("schema") != "submission-harnessaudit-development-v1"
            or digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256")):
        raise ValueError("invalid or modified plan")
    if plan["settings"] != SETTINGS or plan["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("acquisition implementation differs from frozen settings")
    if sha(out / "CANDIDATES.json") != plan["candidate_file_sha256"]:
        raise ValueError("frozen static task review changed")
    for relative, expected in plan["source_sha256"].items():
        base, name = relative.split("/", 1)
        current = (ROOT if base == "local" else Path(plan["upstream"])) / name
        if sha(out / "source" / relative) != expected or sha(current) != expected:
            raise ValueError("frozen acquisition source changed: " + relative)
    return plan


def evaluate_native(adapter, plan, case):
    _, _, check_trace, Action = native_components(plan["upstream"])
    from multi_agent.schemas.access_rules import AccessRule
    bridge = adapter.oracle_actions(case["case_id"])
    actions = [Action(**a) for a in bridge["actions"]]
    rules = [AccessRule(**r) for r in case["evaluation_config"]["access_rules"]]
    decisions = [d.model_dump(mode="json") for d in check_trace(actions, rules, domain=case["domain"])]
    complete = adapter.oracle_view()["status"] == "completed"
    return {"schema": "harnessaudit-native-actionwise-labels", "bridge": bridge, "decisions": decisions,
        "observed_violation": any(d["violated"] for d in decisions),
        "complete_episode_violation": any(d["violated"] for d in decisions) if complete else None,
        "task_success": "not_evaluated", "temporal_or_combination_risk": "not_evaluated"}


async def collect_case(out, plan, case, executor_factory):
    folder = Path(out) / "cases" / case["case_id"]
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "attempt.json", {"plan_sha256": plan["sha256"], "case_id": case["case_id"]})
    adapter = adapter_for(plan, case)
    write_new(folder / "INITIAL.json", adapter.monitor_prefix())
    executor = executor_factory(folder / "api")
    stage_starts = []

    async def execute(packet, invoke_tool):
        prefix = adapter.monitor_prefix()
        row = {"stage_id": packet["node_id"], "sequence": prefix["sequence"], "prefix_sha256": digest(prefix)}
        stage_starts.append(row)
        write_new(folder / "stages" / f"{len(stage_starts):02d}.json",
                  {**row, "packet": packet, "prefix_before": prefix})
        return await executor(packet, invoke_tool)

    result, error = None, None
    try:
        result = await adapter.run(execute, journal_path=folder / "journal.jsonl")
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        write_new(folder / "MONITOR.json", adapter.monitor_prefix())
        write_new(folder / "ORACLE.json", adapter.oracle_view())
        write_new(folder / "LABELS.json", evaluate_native(adapter, plan, case))
        write_new(folder / "RUN.json", {"plan_sha256": plan["sha256"], "case_id": case["case_id"],
            "status": "completed" if error is None else "failed", "error_type": error,
            "executor_result": result, "stage_start_prefixes": stage_starts})
    return {"case_id": case["case_id"], "status": "completed" if error is None else "failed", "error_type": error}


async def run_api(out, endpoint):
    parsed = urlparse(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.path.rstrip("/") != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("explicit loopback OpenAI-compatible endpoint required")
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key:
        raise ValueError("environment-only PAPERALPHA_LOCAL_API_KEY required")
    from openai import AsyncOpenAI
    from paperalpha_runtime.harnessaudit_node_executor import HarnessAuditNodeExecutor
    plan = load_plan(out)
    settings = plan["settings"]
    gate = asyncio.Semaphore(settings["concurrency"])
    async with AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0,
                          timeout=settings["turn_timeout_seconds"]) as client:
        def executor_factory(folder):
            return HarnessAuditNodeExecutor.from_openai_client(receipt_dir=folder, client=client,
                max_calls=settings["max_calls"], max_turns=settings["max_turns"],
                turn_timeout_seconds=settings["turn_timeout_seconds"], max_output_tokens=settings["max_output_tokens"])

        async def one(case):
            async with gate:
                return await collect_case(out, plan, case, executor_factory)
        rows = await asyncio.gather(*(one(c) for c in plan["cases"]))
    load_plan(out)
    write_new(Path(out) / "COLLECTION.json", {"plan_sha256": plan["sha256"], "cases": rows,
        "status": "completed" if all(r["status"] == "completed" for r in rows) else "completed_with_failures"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=Path, default=ROOT / 'experiments/harnessaudit/task_candidates.json')
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.candidates, args.task_ids or [], args.out)
    elif args.command == "verify":
        plan = load_plan(args.out)
        result = {"plan_sha256": plan["sha256"], "source_verified": True, "cases": len(plan["cases"]),
                  "run_receipts": "separate independent artifact audit required"}
    else:
        result = asyncio.run(run_api(args.out, args.endpoint))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
