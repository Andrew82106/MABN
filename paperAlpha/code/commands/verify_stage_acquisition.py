"""Verify stage-aware API acquisition, fresh inputs, original labels and BN replay."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import subprocess
import sys
import types

from collect_stage_harnessaudit import load_plan
from harnessaudit_cohort import ROOT, sha, write_new
from summarize_harnessaudit_cohort import read_rows

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.stage_workflow import validate_stage_workflow, StageBindings
from paperalpha_runtime.stage_capture import stage_instructions


def official_audit(run, output):
    import audit_harnessaudit_api_pilot as original
    original_import = importlib.import_module
    saved = run / "source_snapshot/upstream"
    def isolated_import(name, package=None):
        if name == "multi_agent.checker" and "multi_agent.frameworks" not in sys.modules:
            module = types.ModuleType("multi_agent.frameworks")
            module.__path__ = [str(saved / "multi_agent/frameworks")]
            module.__spec__ = ModuleSpec("multi_agent.frameworks", loader=None, is_package=True)
            sys.modules["multi_agent.frameworks"] = module
        return original_import(name, package)
    importlib.import_module = isolated_import
    try:
        report = original.audit(run, output)
    finally:
        importlib.import_module = original_import
    # Retain the original auditor's result. Additional checks have a separate file.
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    loader = original_import("multi_agent.loader")
    path = next((saved / "multi_agent/tasks").rglob(manifest["task_id"] + ".yaml"))
    task, catalog = loader.load_task_with_tools(path)
    factory = original_import("multi_agent.frameworks.oai.tools").build_domain_tools
    expected = [{"name": t.name, "description": t.description, "params_json_schema": t.params_json_schema} for t in factory(catalog)]
    requests = read_rows(run / "api_requests.jsonl")
    if any(r["tools"] != expected for r in requests):
        raise ValueError("actual tools/descriptions/schemas differ from original full catalog")
    write_new(output.with_name(output.stem + "_tool_catalog.json"), {"tool_count": len(expected),
        "requests_checked": len(requests), "all_original_tools_available_to_all_stages": True,
        "meaning": "Tool availability only, not permission to use every tool."})
    return report


def check_execution(selected, manifest, journal, stages, outputs, requests, responses):
    workflow = selected["stage_workflow"]
    order = validate_stage_workflow(workflow)
    initial = journal[0]
    if (initial["kind"] != "run_started" or initial["payload"].get("declared_stage_workflow") != workflow
            or initial["payload"].get("public_configuration") != selected["public_configuration"]
            or manifest["stage_workflow"] != workflow):
        raise ValueError("stage/task configuration differs from frozen plan")
    bindings = StageBindings(workflow)
    for row in journal:
        bindings.observe(row)
    if len(stages) % 2:
        raise ValueError("incomplete stage lifecycle")
    starts, ends = stages[::2], stages[1::2]
    executed = [s["stage_id"] for s in starts]
    if executed != order[:len(executed)] or any(s["status"] != "started" for s in starts):
        raise ValueError("stage execution did not follow declared order")
    if any(e["status"] not in {"completed", "failed"} for e in ends):
        raise ValueError("stage requires completed/failed terminal status")
    if [r["call"] for r in requests] != list(range(1, len(requests) + 1)):
        raise ValueError("request numbering must be unique/contiguous")
    request_map, response_map = {r["call"]: r for r in requests}, {r["call"]: r for r in responses}
    actors = {a["role"]: a for a in selected["public_configuration"]["agents"]}
    completed, expected_messages, covered_tools, previous_after, next_call = {}, [], set(), 1, 1
    for start, end in zip(starts, ends):
        stage, actor = start["stage_id"], workflow["owners"][start["stage_id"]]
        if (end["stage_id"] != stage or start["actor"] != actor or end["actor"] != actor
                or start["first_possible_call"] != next_call):
            raise ValueError("stage, physical actor or call interval mismatch")
        parents = [u for u, v in workflow["edges"] if v == stage]
        deliveries = [{"actor": workflow["owners"][p], "target": actor, "stage_id": p,
                      "target_stage_id": stage, "content": completed[p]} for p in parents]
        expected_input = {"original_task": selected["public_configuration"]["goal"], "stage_id": stage,
                          "purpose": workflow["purposes"][stage], "predecessor_reports": deliveries}
        prompt = stage_instructions(actors[actor], stage, workflow["purposes"][stage])
        if json.loads(start["input"]) != expected_input or start["instructions"] != prompt:
            raise ValueError("changed stage input, future context or instructions")
        before, after = start["journal_before"], end["journal_after"]
        if type(before) is not int or type(after) is not int or not previous_after <= before <= after < len(journal):
            raise ValueError("invalid recorded stage journal interval")
        preceding = [r for r in journal if previous_after < r["sequence"] <= before]
        actual_deliveries = [{k: r["payload"].get(k) for k in ("actor", "target", "stage_id", "target_stage_id", "content")} for r in preceding]
        if any(r["kind"] != "communication" for r in preceding) or actual_deliveries != deliveries:
            raise ValueError("actual predecessor deliveries do not match the fresh context")
        expected_messages.extend(deliveries)
        interval = [r for r in journal if before < r["sequence"] <= after]
        for row in interval:
            if (not row["kind"].startswith("tool_") or row["payload"]["stage_id"] != stage
                    or row["payload"]["actor"] != actor):
                raise ValueError("stage interval contains an unbound tool/communication")
            covered_tools.add(row["sequence"])
        calls = end["calls_started"]
        if type(calls) is not int or calls < 0 or (end["status"] == "completed" and calls == 0):
            raise ValueError("invalid API call count for stage")
        for number in range(next_call, next_call + calls):
            request = request_map[number]
            if (request["stage_id"] != stage or request["actor"] != actor or request["system_instructions"] != prompt
                    or request["model_requested"] != manifest["model_requested"]):
                raise ValueError("actual API request has different stage/actor/instructions/model")
            config = request["model_settings"]
            if (config.get("reasoning", {}).get("effort") != manifest["settings"]["reasoning"]
                    or config.get("max_tokens") != manifest["settings"]["max_output_tokens"]
                    or config.get("store") is not False or config.get("parallel_tool_calls") is not False):
                raise ValueError("actual request settings differ from declared settings")
            if request.get("conversation_id") is not None or request.get("previous_response_id") is not None or request.get("prompt") is not None:
                raise ValueError("unexpected server conversation/response/prompt context")
            if number == next_call:
                actual_input = request["input"]
                if actual_input != start["input"] and actual_input != [{"role": "user", "content": start["input"]}]:
                    raise ValueError("first API input is not the fresh stage input")
        next_call += calls
        if end["last_call"] != next_call - 1:
            raise ValueError("noncontiguous stage request interval")
        if end["status"] == "completed":
            response = response_map[end["last_call"]].get("response", {})
            messages = [r for r in response.get("output", []) if r.get("type") == "message"]
            actual_text = "" if not messages else "".join(p.get("text", "") for p in messages[-1]["content"] if p.get("type") == "output_text")
            if not actual_text.strip() or actual_text != end["output"] or outputs.get(stage) != actual_text:
                raise ValueError("stage report differs from actual final API text")
            completed[stage] = actual_text
        elif end is not ends[-1]:
            raise ValueError("execution continued after failure")
        previous_after = after
    actual_messages = [{k: r["payload"].get(k) for k in ("actor", "target", "stage_id", "target_stage_id", "content")}
                       for r in journal if r["kind"] == "communication"]
    if manifest["status"] == "completed":
        if executed != order or len(completed) != len(order):
            raise ValueError("completed run is missing completed stages")
        final = order[-1]
        expected_messages.append({"actor": workflow["owners"][final], "target": "user", "stage_id": final,
                                  "target_stage_id": None, "content": completed[final]})
    if (actual_messages != expected_messages or outputs != completed or next_call - 1 != len(requests)
            or covered_tools != {r["sequence"] for r in journal if r["kind"].startswith("tool_")}):
        raise ValueError("unassigned requests/tools, unexpected delivery, or fabricated stage output")
    return {"planned_stages": len(order), "started_stages": len(starts), "completed_stages": len(completed),
        "physical_agents": len(set(workflow["owners"].values())), "internal_deliveries": sum(m["target_stage_id"] is not None for m in actual_messages),
        "actual_inputs_and_outputs_verified": True, "fresh_first_inputs_verified": True,
        "physical_tool_actors_and_stages_verified": True, "scope": "Runtime context/call bindings, not semantic correctness or safety."}


def replay(journal, workflow):
    import numpy as np
    from paperalpha_monitor.staged_knowledge_bn import StageKnowledgeCompiler, compile_stage_journal
    from paperalpha_monitor.knowledge_scoped_bn import KnowledgeScopedBN
    from paperalpha_monitor.tool_capabilities import load_tool_registry
    context = dict(stage_workflow=workflow, tool_registry=load_tool_registry())
    model, compiler = KnowledgeScopedBN(), StageKnowledgeCompiler(**context)
    engine, scores, factors = model._engine(model.theta), [], Counter()
    for row in journal:
        step = compiler.push(row)
        if step is not None:
            scores.append(engine.apply(step, compiler.variables))
            factors.update(f.template_id for f in step.mechanism_factors)
    trace = compile_stage_journal(journal, **context)
    np.testing.assert_array_equal(scores, model.predict([trace])[0])
    explanation = engine.explain(compiler.last_step, compiler.variables)
    return {"model": model.model_name, "parameters": len(model.theta), "trained": False, "observations": len(scores),
            "actions": len(explanation["actions"]), "factors": dict(factors), "joint_states": explanation["retained_joint_states"],
            "unknown_permissions": sum(a["permission_status"] == "unknown" for a in explanation["actions"]),
            "unknown_roles": sum(a["role"] == "unknown" for a in explanation["actions"]),
            "incremental_batch_error": 0., "semantic_annotations": 0}, explanation


def verify(plan_path, output, audit_python):
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new verification output required")
    output.mkdir(parents=True)
    results = []
    for selected in plan["tasks"]:
        task_id = selected["task_id"]
        run = plan_path.parent / "runs" / task_id
        manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
        if (manifest["status"] not in {"completed", "failed"} or manifest["plan_sha256"] != plan["plan_sha256"]
                or manifest["settings"] != plan["settings"]):
            raise ValueError("unfinalized or differently configured run")
        result = subprocess.run([str(audit_python), "-X", "utf8", "-B", str(Path(__file__).resolve()), "--audit-worker",
            "--run", str(run), "--out", str(output / task_id / "AUDIT.json")], capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            write_new(output / task_id / "audit_failure.json", {"stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError("original audit failed; see retained failure")
        journal, stages, requests, responses = (read_rows(run / p) for p in
            ("runtime_events.jsonl", "stage_runs.jsonl", "api_requests.jsonl", "api_responses.jsonl"))
        outputs = json.loads((run / "stage_outputs.json").read_text(encoding="utf-8"))
        execution = check_execution(selected, manifest, journal, stages, outputs, requests, responses)
        model, explanation = replay(journal, selected["stage_workflow"])
        write_new(output / task_id / "model_explanation.json", explanation)
        decisions = [r for r in read_rows(run / "native_trace.jsonl") if r["type"] == "access_decision" and r["violated"]]
        results.append({"task_id": task_id, "run_sha256": sha(run / "RUN.json"), "status": manifest["status"],
            "error": manifest["error"], "api_calls": manifest["api_calls_started"], "tool_calls": manifest["tool_calls"],
            "input_tokens": manifest["input_tokens"], "output_tokens": manifest["output_tokens"],
            "episode_rule_label": manifest["episode_rule_label"], "completion": manifest["completion"],
            "rule_profile": dict(Counter(f"{d['violation_code']}:{d.get('severity') or 'ungraded'}" for d in decisions)),
            "execution": execution, "model_replay": model})
    sources = [Path(__file__), ROOT / 'code/commands/collect_stage_harnessaudit.py', *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"),
        *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json"), ROOT / 'knowledge/risks/risk_catalog.json']
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
    report = {"plan_sha256": plan["plan_sha256"], "runs": results,
        "totals": {k: sum(r[k] for r in results) for k in ("api_calls", "tool_calls", "input_tokens", "output_tokens")},
        "no_model_fitting": True, "no_new_api_calls_in_verifier": True,
        "scope": "Actual stage-aware development acquisition and untrained replay. Failed attempts retained, no accuracy/normality inference.",
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
        report = official_audit(args.run.resolve(), args.out.resolve())
        print(json.dumps({"status": report["status"]}))
    else:
        report = verify(args.plan.resolve(), args.out.resolve(), args.audit_python)
        print(json.dumps({"totals": report["totals"], "runs": report["runs"]}, ensure_ascii=False, indent=2))
