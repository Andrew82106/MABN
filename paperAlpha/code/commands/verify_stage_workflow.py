"""Stage protocol: old-real-run equivalence and NEW-plan static rule checks.

No new API calls, model fitting, replacement labels, or safety-performance
measurement. New relay plans are inspected, not claimed as executed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from harnessaudit_cohort import ROOT, sha, write_new
from summarize_harnessaudit_cohort import read_rows
from collect_fixed_dag_harnessaudit import load_plan, import_official_oai_components
from verify_fixed_dag_acquisition import check_execution

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.stage_workflow import daily_life_relay_stages, validate_stage_workflow


def named_stages(workflow):
    names = {a: "stage:" + a for a in workflow["nodes"]}
    return {"schema": "stage-workflow-v1", "context_mode": "fresh_per_stage",
        "nodes": [names[a] for a in workflow["nodes"]],
        "edges": [[names[a], names[b]] for a, b in workflow["edges"]],
        "owners": {names[a]: a for a in workflow["nodes"]},
        "roles": {names[a]: workflow["roles"][a] for a in workflow["nodes"]},
        "purposes": {names[a]: "Original one-pass node; rename-only compatibility audit." for a in workflow["nodes"]}}


def derive_stage_rows(rows, workflow):
    result = copy.deepcopy(rows)
    by_actor = {a: s for s, a in workflow["owners"].items()}
    if len(by_actor) != len(workflow["nodes"]):
        raise ValueError("old run rename-only derivation requires unique owners; do not guess repeated stages")
    result[0]["payload"]["declared_stage_workflow"] = copy.deepcopy(workflow)
    for row in result[1:]:
        data = row["payload"]
        if "actor" in data:
            data["stage_id"] = by_actor[data["actor"]]
            if row["kind"] == "communication" and data["target"] in by_actor:
                data["target_stage_id"] = by_actor[data["target"]]
    return result


def inspect_policy(plan_path, task_id, output):
    plan = load_plan(plan_path)
    selected = next(t for t in plan["tasks"] if t["task_id"] == task_id)
    saved = plan_path.parent / "source_snapshot/upstream"
    import_official_oai_components(saved)
    from multi_agent.loader import load_task_with_tools
    from multi_agent.checker import check_action
    from multi_agent.schemas.actions import ObservableAction
    task, _ = load_task_with_tools(saved / selected["task_path"])
    rules = [r for r in task.access_rules if r.access_type == "information.communication"]
    proposed = daily_life_relay_stages(selected["public_configuration"]["agents"])
    graphs = {"original_plan": {"edges": selected["workflow"]["edges"],
        "owners": {a: a for a in selected["workflow"]["nodes"]}}, "proposed_stage_plan": proposed}
    result = {"task_id": task_id, "proposed_workflow": proposed, "proposed_plan_executed": False}
    for name, graph in graphs.items():
        conflicts = []
        for index, (source, target) in enumerate(graph["edges"], 1):
            actor, recipient = graph["owners"][source], graph["owners"][target]
            action = ObservableAction(run_id="static-stage-plan-audit", sequence_num=index,
                surface="communication", agent_id=actor, agent_role=actor, target_agent=recipient,
                target_role=recipient, content="")
            matches = [d for d in check_action(action, rules, domain=task.domain) if d.violated]
            if matches:
                conflicts.append({"source_stage": source, "target_stage": target, "actor": actor, "target": recipient,
                                  "rules": [d.rule.description for d in matches]})
        result[name] = {"planned_edges": len(graph["edges"]), "communication_policy_conflicts": conflicts}
    result["scope"] = "Original checker, communication rules only, empty message body. No tool/content authorization, harm or task-success claim."
    write_new(output, result)
    return result


def verify(plan_path, output, audit_python):
    import numpy as np
    from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN, compile_knowledge_journal
    from paperalpha_monitor.staged_knowledge_bn import StageKnowledgeCompiler, compile_stage_journal
    from paperalpha_monitor.tool_capabilities import load_tool_registry
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new output directory required")
    output.mkdir(parents=True)
    results = []
    for selected in plan["tasks"]:
        task_id = selected["task_id"]
        run = plan_path.parent / "runs" / task_id
        rows = read_rows(run / "runtime_events.jsonl")
        manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
        execution = check_execution(selected, manifest, rows, read_rows(run / "node_runs.jsonl"),
            json.loads((run / "node_outputs.json").read_text(encoding="utf-8")), read_rows(run / "api_requests.jsonl"))
        stage_workflow = named_stages(selected["workflow"])
        validate_stage_workflow(stage_workflow)
        stage_rows = derive_stage_rows(rows, stage_workflow)
        context = dict(stage_workflow=stage_workflow, tool_registry=load_tool_registry())
        old = compile_knowledge_journal(rows, workflow=selected["workflow"], tool_registry=load_tool_registry())
        new = compile_stage_journal(stage_rows, **context)
        differences, gradient_differences = [], []
        model = KnowledgeScopedBN()
        probes = [model.theta.copy()] + [model.theta + np.random.default_rng(seed).normal(0, .4, len(model.theta)) for seed in range(5)]
        for theta in probes:
            model.theta = theta
            a, b = model.predict([old])[0], model.predict([new])[0]
            differences.append(float(np.max(np.abs(a - b))))
            _, ga = model.episode_loss_gradient(theta, old, 1)
            _, gb = model.episode_loss_gradient(theta, new, 1)
            gradient_differences.append(float(np.max(np.abs(ga - gb))))
            np.testing.assert_allclose(a, b, atol=3e-14, rtol=0)
            np.testing.assert_allclose(ga, gb, atol=3e-13, rtol=0)
        # Separate streaming pass at default parameters.
        model = KnowledgeScopedBN()
        compiler, engine, online = StageKnowledgeCompiler(**context), model._engine(model.theta), []
        for row in stage_rows:
            step = compiler.push(row)
            if step is not None:
                online.append(engine.apply(step, compiler.variables))
        np.testing.assert_array_equal(online, model.predict([new])[0])
        old_actions, new_actions = old.steps[-1].base.actions, new.steps[-1].base.actions
        assert [a["binding"] for a in old_actions] == [a["binding"] for a in new_actions]
        assert [a["permission_status"] for a in old_actions] == [a["permission_status"] for a in new_actions]
        write_new(output / task_id / "rename_only_stage_rows.json", stage_rows)
        write_new(output / task_id / "stage_explanation.json", model.explain(new))
        static_path = output / task_id / "static_policy.json"
        worker = subprocess.run([str(audit_python), "-X", "utf8", "-B", str(Path(__file__).resolve()),
            "--policy-worker", "--plan", str(plan_path), "--task", task_id, "--out", str(static_path)],
            capture_output=True, text=True, encoding="utf-8")
        if worker.returncode:
            write_new(output / task_id / "failure.json", {"stdout": worker.stdout, "stderr": worker.stderr})
            raise RuntimeError("saved original checker worker failed")
        policy = json.loads(static_path.read_text(encoding="utf-8"))
        results.append({"task_id": task_id, "original_journal_sha256": sha(run / "runtime_events.jsonl"),
            "execution": execution, "model_observations": len(online), "actions": len(new_actions), "probes": len(probes),
            "rename_score_max_error": max(differences), "rename_gradient_max_error": max(gradient_differences),
            "stream_batch_max_error": 0., "physical_bindings_preserved": True, "permission_statuses_preserved": True,
            "old_conflicting_edges": len(policy["original_plan"]["communication_policy_conflicts"]),
            "proposed_conflicting_edges": len(policy["proposed_stage_plan"]["communication_policy_conflicts"]),
            "proposed_stage_plan_executed": False})
    sources = [Path(__file__), ROOT / 'code/commands/verify_fixed_dag_acquisition.py',
        *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
        *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']
    for source in sources:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
    report = {"input_plan_sha256": plan["plan_sha256"], "runs": results, "parameters": 345,
        "trained": False, "new_api_calls": 0,
        "scope": "Stage identity and fresh-context access protocol, old-real-run equivalence, prospective communication-policy check only.",
        "remaining": ["New relay plan has not been run with API agents.",
            "Shared memory and automatic stage discovery are unsupported; original runtime actors are not stage IDs.",
            "No new risk templates, calibrated risk, permissions, reliable safety dataset or MAS baseline advantage.",
            "Exact process state still exponential in stages; six stages need 1728 states with three mechanisms."],
        "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-python", type=Path)
    parser.add_argument("--policy-worker", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.policy_worker:
        report = inspect_policy(args.plan.resolve(), args.task, args.out.resolve())
        print(json.dumps({"task": args.task, "proposed_plan_executed": False}))
    else:
        report = verify(args.plan.resolve(), args.out.resolve(), args.audit_python)
        print(json.dumps({"runs": report["runs"], "trained": False, "new_api_calls": 0}, indent=2))
