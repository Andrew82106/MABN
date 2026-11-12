"""Verify actual DAG execution, API inputs, original rule audit and v7 replay.

Not model fitting, new labels, or an independent safety-performance test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import subprocess
import sys
import types

from collect_fixed_dag_harnessaudit import load_plan
from harnessaudit_cohort import ROOT, digest, sha, write_new
from summarize_harnessaudit_cohort import read_rows

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.fixed_workflow import topological_order, node_input, node_instructions


def audit_full_snapshot(run, output):
    """Reuse the unchanged audit, bypass only eager unrelated frameworks.

    The original small snapshots omitted the registry __init__.py; our full
    snapshot includes it. The registry imports Unix-only fcntl on Windows.
    Bind its package path when the audit loads the checker, just as acquisition
    already does. No checker function, task rule or artifact is rewritten.
    """
    import audit_harnessaudit_api_pilot as original_audit
    original_import = importlib.import_module
    saved = run.resolve() / "source_snapshot/upstream"
    def isolated_import(name, package=None):
        if name == "multi_agent.checker" and "multi_agent.frameworks" not in sys.modules:
            namespace = types.ModuleType("multi_agent.frameworks")
            namespace.__path__ = [str(saved / "multi_agent/frameworks")]
            namespace.__spec__ = ModuleSpec("multi_agent.frameworks", loader=None, is_package=True)
            sys.modules["multi_agent.frameworks"] = namespace
        return original_import(name, package)
    importlib.import_module = isolated_import
    try:
        report = original_audit.audit(run, output)
    finally:
        importlib.import_module = original_import
    # Evaluation-side check: the scheduler itself can violate a task's peer
    # communication policy, even though the control graph is acyclic.
    loader = original_import("multi_agent.loader")
    checker = original_import("multi_agent.checker")
    Action = original_import("multi_agent.schemas.actions").ObservableAction
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    task_path = next((saved / "multi_agent/tasks").rglob(manifest["task_id"] + ".yaml"))
    task, _ = loader.load_task_with_tools(task_path)
    conflicts = []
    communication_rules = [r for r in task.access_rules if r.access_type == "information.communication"]
    for index, (sender, target) in enumerate(manifest["workflow"]["edges"], 1):
        action = Action(run_id="static-plan-check", sequence_num=index, surface="communication", agent_id=sender,
                        agent_role=sender, target_agent=target, target_role=target, content="")
        matches = [d for d in checker.check_action(action, communication_rules, domain=task.domain) if d.violated]
        if matches:
            conflicts.append({"sender": sender, "target": target,
                              "rules": [{"description": d.rule.description, "severity": d.severity} for d in matches]})
    report.update(framework_registry_import_bypassed=True, planned_communication_policy_conflicts=conflicts,
                  eligible_as_normal_control=not conflicts and manifest["episode_rule_label"] == 0,
                  limitation="Static scheduler creates these deliveries; do not attribute them to autonomous malicious agent choices.")
    original_audit_output = output.parent / (output.stem + "_base.json")
    # Keep the original verifier's report, then publish the richer new report.
    with original_audit_output.open("xb") as handle:
        handle.write(output.read_bytes())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def check_execution(selected, manifest, journal, node_rows, outputs, requests):
    """Check recorded behavior, not just whether the advertised DAG is valid."""
    workflow = selected["workflow"]
    order = topological_order(workflow)
    initial = journal[0]
    if (initial["kind"] != "run_started" or initial["payload"].get("declared_workflow") != workflow
            or initial["payload"].get("public_configuration") != selected["public_configuration"]):
        raise ValueError("initial public task/workflow differs from frozen plan")
    starts = [r for r in node_rows if r["stage"] == "started"]
    finishes = [r for r in node_rows if r["stage"] in {"completed", "failed"}]
    if len(node_rows) != 2 * len(starts) or any(
            node_rows[i] != start or node_rows[i + 1] != end
            for i, (start, end) in zip(range(0, len(node_rows), 2), zip(starts, finishes))):
        raise ValueError("node lifecycle records must alternate start and terminal event")
    executed = [r["node"] for r in starts]
    if executed != order[:len(executed)] or len(starts) != len(finishes):
        raise ValueError("nodes did not execute once in the declared topological order")
    actors = {a["role"]: a for a in selected["public_configuration"]["agents"]}
    request_map = {r["call"]: r for r in requests}
    if [r["call"] for r in requests] != list(range(1, len(requests) + 1)):
        raise ValueError("actual request IDs must be contiguous and unique")
    expected_calls, finished_outputs, expected_messages = 1, {}, []
    for start, end in zip(starts, finishes):
        node = start["node"]
        if end["node"] != node or start["first_possible_call"] != expected_calls:
            raise ValueError("node call intervals overlap or skip requests")
        parents = [u for u, v in workflow["edges"] if v == node]
        reports = [{"sender": u, "content": finished_outputs[u]} for u in parents]
        expected = node_input(selected["public_configuration"]["goal"], reports)
        if start["input"] != expected or start["instructions"] != node_instructions(actors[node]):
            raise ValueError("node received undeclared/future reports or changed instructions")
        expected_messages.extend((u, node, finished_outputs[u]) for u in parents)
        calls = end["calls_started"]
        if type(calls) is not int or calls < 0 or (end["stage"] == "completed" and calls == 0):
            raise ValueError("completed node requires an actual model request")
        if calls:
            request = request_map[expected_calls]
            first_input = request["input"]
            if first_input != expected and first_input != [{"role": "user", "content": expected}]:
                raise ValueError("actual first model input differs from declared node input")
        for number in range(expected_calls, expected_calls + calls):
            request = request_map[number]
            if request["system_instructions"] != start["instructions"] or request["model_requested"] != manifest["model_requested"]:
                raise ValueError("actual model request changed role instructions/model")
        expected_calls += calls
        if end["last_call"] != expected_calls - 1:
            raise ValueError("invalid node call interval")
        if end["stage"] == "completed":
            if outputs.get(node) != end["output"] or not end["output"].strip():
                raise ValueError("completed node report mismatch")
            finished_outputs[node] = end["output"]
        elif end is not finishes[-1]:
            raise ValueError("execution continued after a failed node")
    if expected_calls - 1 != len(requests) or outputs != finished_outputs:
        raise ValueError("unassigned model requests or extra node reports")
    messages = [(r["payload"]["actor"], r["payload"]["target"], r["payload"]["content"])
                for r in journal if r["kind"] == "communication"]
    internal = [m for m in messages if m[1] != "user"]
    if internal != expected_messages:
        raise ValueError("actual delivered messages differ from declared predecessor reports")
    if manifest["status"] == "completed":
        if (executed != order or sum(m[1] == "user" for m in messages) != 1
                or messages[-1:] != [("concierge_coordinator", "user", outputs["concierge_coordinator"])]):
            raise ValueError("completed run is missing nodes or its final user output")
    elif any(m[1] == "user" for m in messages):
        raise ValueError("failed workflow fabricated a final user response")
    return {"planned_nodes": len(order), "started_nodes": len(starts), "completed_nodes": len(outputs),
            "delivered_internal_edges": len(internal), "actual_inputs_match_predecessors": True,
            "all_inter_agent_edges_forward": all(order.index(u) < order.index(v) for u, v, _ in internal),
            "scope": "Fixed control/message DAG, not discovery, causal dependencies, or absence of local tool loops."}


def replay_model(journal, workflow):
    from paperalpha_monitor.knowledge_scoped_bn import JournalKnowledgeCompiler, KnowledgeScopedBN, compile_knowledge_journal
    from paperalpha_monitor.tool_capabilities import load_tool_registry

    model = KnowledgeScopedBN()
    registry = load_tool_registry()
    compiler = JournalKnowledgeCompiler(workflow=workflow, tool_registry=registry)
    engine = model._engine(model.theta)
    scores, factors = [], Counter()
    for row in journal:
        step = compiler.push(row)
        if step is not None:
            scores.append(engine.apply(step, compiler.variables))
            factors.update(f.template_id for f in step.mechanism_factors)
    trace = compile_knowledge_journal(journal, workflow=workflow, tool_registry=registry)
    batch = model.predict([trace])[0]
    if len(batch) != len(scores) or any(float(a) != float(b) for a, b in zip(batch, scores)):
        raise ValueError("v7 streaming and batch replay disagree")
    explanation = engine.explain(compiler.last_step, compiler.variables)
    return {"model": model.model_name, "parameters": len(model.theta), "trained": False,
            "model_events": len(scores), "actions": len(explanation["actions"]),
            "factors": dict(factors), "retained_joint_states": explanation["retained_joint_states"],
            "unknown_permissions": sum(a["permission_status"] == "unknown" for a in explanation["actions"]),
            "unknown_roles": sum(a["role"] == "unknown" for a in explanation["actions"]),
            "incremental_batch_error": 0., "text_semantic_annotations": 0,
            "meaning": "Declared workflow and roles consumed; untrained coverage, not safety accuracy or online warning."}


def verify(plan_path, output, audit_python):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new verification directory required")
    output.mkdir(parents=True)
    rows = []
    for selected in plan["tasks"]:
        identity = selected["task_id"]
        run = plan_path.parent / "runs" / identity
        manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
        if manifest["status"] not in {"completed", "failed"} or manifest["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("run not finalized under the frozen plan")
        audit_output = output / "audits" / f"{identity}.json"
        result = subprocess.run([str(audit_python), "-X", "utf8", "-B", str(Path(__file__).resolve()),
                                 "--audit-worker", "--run", str(run), "--out", str(audit_output)], capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            write_new(output / f"failure-{identity}.json", {"stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError("original official-label/acquisition audit failed")
        journal, node_rows, requests = (read_rows(run / name) for name in ("runtime_events.jsonl", "node_runs.jsonl", "api_requests.jsonl"))
        outputs = json.loads((run / "node_outputs.json").read_text(encoding="utf-8"))
        execution = check_execution(selected, manifest, journal, node_rows, outputs, requests)
        audit = json.loads(audit_output.read_text(encoding="utf-8"))
        replay = replay_model(journal, selected["workflow"])
        decisions = [r for r in read_rows(run / "native_trace.jsonl") if r["type"] == "access_decision" and r["violated"]]
        row = {"task_id": identity, "run_status": manifest["status"], "run_sha256": sha(run / "RUN.json"),
               "api_calls": manifest["api_calls_started"], "input_tokens": manifest["input_tokens"], "output_tokens": manifest["output_tokens"],
               "official_episode_label": manifest["episode_rule_label"], "completion": manifest["completion"],
               "planned_communication_policy_conflicts": audit["planned_communication_policy_conflicts"],
               "eligible_as_normal_control": audit["eligible_as_normal_control"],
               "rule_match_profile": dict(Counter(f"{d['violation_code']}:{d.get('severity') or 'ungraded'}" for d in decisions)),
               "execution": execution, "model_replay": replay}
        rows.append(row)
    sources = [Path(__file__), *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
               *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']
    for source in sources:
        dest = output / "source_snapshot" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(source.read_bytes())
    report = {"plan_sha256": plan["plan_sha256"], "runs": rows,
              "totals": {k: sum(r[k] for r in rows) for k in ("api_calls", "input_tokens", "output_tokens")},
              "no_model_fitting": True, "no_new_api_calls_in_verification": True,
              "scope": "First declared fixed-DAG execution and v7 coverage. Unperturbed does not mean safe; tiny development set, not original HarnessAudit scores.",
              "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-python", type=Path)
    parser.add_argument("--audit-worker", action="store_true")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.audit_worker:
        result = audit_full_snapshot(args.run, args.out)
        print(json.dumps({"status": result["status"]}))
    else:
        result = verify(args.plan, args.out, args.audit_python)
        print(json.dumps({"totals": result["totals"], "runs": result["runs"]}, ensure_ascii=False, indent=2))
