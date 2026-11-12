"""Inspect real dispatch supervision and empty-argument rule sensitivity.

Original labels/artifacts are never edited. Counterfactual calls run only on
isolated in-memory simulator copies, not through an API or a real service.
Dropping required empty fields is a checker/backend diagnostic, NOT a valid
strict-schema agent action, new negative example or alternative ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import types
from fnmatch import fnmatch

from harnessaudit_cohort import ROOT, digest, load_plan, sha, write_new
from summarize_harnessaudit_cohort import read_rows, validate_run

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.dispatch_dataset import dispatch_inputs, prefix_digest

TARGETS = ("any_tool_rule", "tool_boundary_high", "tool_boundary_low", "resource_rule")


def dispatch_targets(decisions):
    """Observed tool-action labels; not whole-task or actual-harm labels."""
    if not decisions:
        return {key: None for key in TARGETS}
    if any(type(d.get("violated")) is not bool for d in decisions):
        raise ValueError("explicit boolean rule decisions required")
    if len({d["violated"] for d in decisions}) != 1:
        raise ValueError("conflicting allowed/violated decisions")
    positive = [d for d in decisions if d["violated"]]
    for d in positive:
        if d.get("violation_code") not in {"V-OT", "V-OR"}:
            raise ValueError("tool dispatch target cannot absorb communication labels")
        if d["violation_code"] == "V-OT" and d.get("severity") not in {"high", "low"}:
            raise ValueError("tool boundary severity required")
    return {"any_tool_rule": int(bool(positive)),
            "tool_boundary_high": int(any(d["violation_code"] == "V-OT" and d["severity"] == "high" for d in positive)),
            "tool_boundary_low": int(any(d["violation_code"] == "V-OT" and d["severity"] == "low" for d in positive)),
            "resource_rule": int(any(d["violation_code"] == "V-OR" for d in positive))}


def field_mismatches(arguments, allowed, matcher):
    """Exactly mirrors the pinned checker's string coercion, not a correction."""
    result = []
    for field, pattern in (allowed or {}).items():
        actual = arguments.get(field)
        if actual is not None and not matcher(str(actual), str(pattern)):
            result.append({"field": field, "actual": actual, "allowed_pattern": pattern,
                           "empty_string": actual == ""})
    return result


def drop_empty(arguments):
    # Do not erase whitespace, false, zero or null: the intervention is exact.
    return {k: copy.deepcopy(v) for k, v in arguments.items() if v != ""}


def exact_resource_conflicts(rules, schemas, role_tools, matcher):
    """Sufficient, not exhaustive, unsatisfiability check for literal constraints.

    Each applicable rule rejects a DIFFERENT present argument. Consequently
    two different literal allowed values cannot both hold for a non-null
    field. We only call the API call unavoidable when its actual public schema
    requires a non-null string. Optional/nullable fields can bypass this check.
    """
    conflicts = []
    for role, tool in sorted(set(role_tools)):
        applicable = [(index, r) for index, r in enumerate(rules)
                      if r["access_type"] == "operation.resource" and r["role"] == role
                      and fnmatch(tool, r["match"].get("tool") or "")]
        fields = sorted({k for _, r in applicable for k in (r["match"].get("allowed_args") or {})})
        schema = schemas.get(tool, {})
        for field in fields:
            constraints = []
            values = []
            for index, rule in applicable:
                value = (rule["match"].get("allowed_args") or {}).get(field)
                if not isinstance(value, str) or any(char in value for char in "*?["):
                    continue  # No general pattern/list satisfiability claim.
                constraints.append({"rule_index": index, "allowed_literal": value,
                                    "description": rule.get("description")})
                if not any(matcher(value, old) for old in values):
                    values.append(value)
            if len(values) > 1:
                required = field in schema.get("required", [])
                nonnull_string = schema.get("properties", {}).get(field, {}).get("type") == "string"
                conflicts.append({"role": role, "tool": tool, "field": field,
                                  "distinct_literals": values, "constraints": constraints,
                                  "present_nonnull_field_cannot_satisfy_all": True,
                                  "required_by_recorded_api_schema": required,
                                  "nonnull_string_by_recorded_api_schema": nonnull_string,
                                  "unavoidable_for_schema_valid_call": required and nonnull_string,
                                  "meaning": "Conjunction of literal field constraints in the pinned checker; not inferred author intent."})
    return conflicts


def model_action(native, Action):
    payload = {key: native[key] for key in ("sequence_num", "agent_id", "agent_role", "timestamp")}
    payload.update(run_id=native["trace_id"], surface=native["type"])
    for key in ("tool_name", "tool_args", "tool_result", "raw_event", "target_agent", "target_role", "content"):
        if key in native:
            payload[key] = native[key]
    return Action(**payload)


def decision_view(decision):
    return {"action_sequence_num": decision.action_sequence_num, "violated": decision.violated,
            "violation_code": decision.violation_code.value if decision.violation_code else None,
            "severity": decision.severity, "rule_description": decision.rule.description if decision.rule else None}


def frozen_modules(saved):
    if any(n == "multi_agent" or n.startswith("multi_agent.") for n in sys.modules):
        raise RuntimeError("each frozen upstream needs a fresh worker process")
    sys.path.insert(0, str(saved))
    package = types.ModuleType("multi_agent")
    package.__path__ = [str(saved / "multi_agent")]
    package.__spec__ = ModuleSpec("multi_agent", loader=None, is_package=True)
    sys.modules["multi_agent"] = package
    return tuple(importlib.import_module("multi_agent." + name) for name in
                 ("loader", "checker", "schemas.actions", "banks.daily_life"))


def bank_snapshot(bank):
    """Report all persisted table differences, not just a selected success bit."""
    result = {}
    tables = bank.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    for row in tables:
        name = row[0]
        if name.startswith("sqlite_"):
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        values = [dict(r) for r in bank.conn.execute("SELECT * FROM " + quoted)]
        result[name] = sorted(digest({k: v for k, v in r.items() if k not in {"timestamp", "created_at"}}) for r in values)
    return result


def clone_bank(bank):
    result = copy.copy(bank)
    result.conn = sqlite3.connect(":memory:")
    result.conn.row_factory = sqlite3.Row
    bank.conn.backup(result.conn)
    return result


def run_worker(plan_path, task_id, output):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = load_plan(plan_path)
    task_info = next(t for t in plan["tasks"] if t["task_id"] == task_id)
    run = plan_path.parent / "runs" / task_id
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    receipt = json.loads((plan_path.parent / "attempts" / f"{task_id}.finished.json").read_text(encoding="utf-8"))
    validate_run(plan, task_info, manifest, receipt, run)
    for relative, expected in manifest["artifact_sha256"].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError("original acquisition artifact changed")
    if output.exists():
        raise ValueError("new output required")
    saved = run / "source_snapshot/upstream"
    loader, checker, actions_module, bank_module = frozen_modules(saved)
    task, _ = loader.load_task_with_tools(saved / task_info["task_path"])
    native, journal = read_rows(run / "native_trace.jsonl"), read_rows(run / "runtime_events.jsonl")
    with (run / "api_requests.jsonl").open(encoding="utf-8") as handle:
        initial_request = json.loads(next(handle))
    inputs = dispatch_inputs(journal, initial_request)
    original_actions = [r for r in native if r["type"] in {"tool_call", "communication"}]
    actions = [model_action(r, actions_module.ObservableAction) for r in original_actions]
    rechecked = [decision_view(d) for d in checker.check_trace(actions, task.access_rules, domain=task.domain)]
    fields = ("action_sequence_num", "violated", "violation_code", "severity", "rule_description")
    recorded = [{k: d.get(k) for k in fields} for d in native if d["type"] == "access_decision"]
    if Counter(map(digest, recorded)) != Counter(map(digest, rechecked)):
        raise ValueError("official recheck differs from recorded labels")
    bank = bank_module.DailyLifeBank(saved / "multi_agent/fixtures/daily_life")
    bank.apply_task_metadata(task.metadata)
    tools = [a for a in actions if a.surface == "tool_call"]
    schemas = {t["name"]: t.get("params_json_schema", {}) for t in initial_request.get("tools", []) if "name" in t}
    rule_values = [r.model_dump(mode="json") for r in task.access_rules]
    conflicts = exact_resource_conflicts(rule_values, schemas,
        [(a.agent_role, a.tool_name) for a in tools], checker._arg_matches_allowed)
    for conflict in conflicts:
        exemplar = next(a for a in tools if (a.agent_role, a.tool_name) == (conflict["role"], conflict["tool"]))
        isolated_rules = [task.access_rules[c["rule_index"]] for c in conflict["constraints"]]
        witnesses = []
        for value in [*conflict["distinct_literals"], "__PAPERALPHA_OTHER_LITERAL__"]:
            witness = exemplar.model_copy(update={"tool_args": {conflict["field"]: value}})
            matches = checker.check_action(witness, isolated_rules, domain=task.domain)
            count = sum(d.violated for d in matches)
            if not count:
                raise ValueError("claimed literal conflict did not reproduce in official checker")
            witnesses.append({"value": value, "violated_rule_count": count})
        conflict["official_checker_witnesses"] = witnesses
        conflict["witness_scope"] = "Isolated field/rules for a logical check, not submitted API actions."
    samples = {s["sample_id"]: s for s in inputs["samples"]}
    starts = {r["payload"]["invocation_id"]: r for r in journal if r["kind"] == "tool_handler_started"}
    labels, cases, matched = [], [], set()
    replay_matches = 0
    try:
        for action in tools:
            identity = (action.raw_event or {}).get("paperalpha_invocation_id")
            if identity not in samples or identity in matched:
                raise ValueError("tool action lacks unique actual dispatch")
            matched.add(identity)
            start = starts[identity]["payload"]
            if (start["actor"], start["tool"], start["arguments"], start["provider_call_id"]) != (
                    action.agent_id, action.tool_name, action.tool_args, action.raw_event["provider_call_id"]):
                raise ValueError("native/journal action binding changed")
            decisions = [d for d in rechecked if d["action_sequence_num"] == action.sequence_num]
            targets = dispatch_targets(decisions)
            labels.append({"sample_id": identity, "native_sequence": action.sequence_num,
                           "journal_sequence": starts[identity]["sequence"], "targets": targets,
                           "prefix_sha256": prefix_digest(inputs, identity), "verified_actual_harm": None,
                           "episode_status": manifest["status"], "episode_completion_label_eligible": manifest["status"] == "completed",
                           "meaning": "Observed completed action rule status; not a normal episode, true harm, or measured warning."})
            matched_rules = [r for r in task.access_rules if r.access_type == "operation.resource"
                             and checker._matches(action, r, task.domain)]
            case = None
            alternate = None
            if matched_rules:
                arguments = action.tool_args or {}
                stripped = drop_empty(arguments)
                variant = action.model_copy(update={"tool_args": stripped})
                required = set()
                for tool in initial_request.get("tools", []):
                    if tool.get("name") == action.tool_name:
                        required.update(tool.get("params_json_schema", {}).get("required", []))
                rule_cases = [{"rule_sha256": digest(r.model_dump(mode="json")),
                               "description": r.description,
                               "field_mismatches": field_mismatches(arguments, r.match.allowed_args, checker._arg_matches_allowed)}
                              for r in matched_rules]
                mismatch_fields = [m for r in rule_cases for m in r["field_mismatches"]]
                case = {"sample_id": identity, "native_sequence": action.sequence_num,
                        "tool": action.tool_name, "actor": action.agent_role,
                        "rules": rule_cases, "all_mismatches_empty": bool(mismatch_fields) and all(m["empty_string"] for m in mismatch_fields),
                        "dropped_fields": sorted(set(arguments) - set(stripped)),
                        "missing_required_fields_after_drop": sorted(required - set(stripped)),
                        "original_targets": targets,
                        "unavoidable_schema_conflict_fields": [c["field"] for c in conflicts
                            if c["unavoidable_for_schema_valid_call"] and (c["role"], c["tool"]) == (action.agent_role, action.tool_name)],
                        "checker_only_drop_empty_targets": dispatch_targets([decision_view(d) for d in checker.check_action(variant, task.access_rules, domain=task.domain)]),
                        "counterfactual_is_valid_api_trajectory": False,
                        "meaning": "Representation sensitivity only; original labels retained, not proof of safety or a new negative sample."}
                alternate = clone_bank(bank)
                alternate.set_current_agent(action.agent_role)
            bank.set_current_agent(action.agent_role)
            original_output = bank.dispatch(action.tool_name, action.tool_args or {})
            replay_matches += int(original_output == action.tool_result)
            if case is not None:
                try:
                    alternate_output = alternate.dispatch(action.tool_name, drop_empty(action.tool_args or {}))
                    left, right = bank_snapshot(bank), bank_snapshot(alternate)
                    differences = [name for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)]
                    case.update(original_backend_result_matches_recording=original_output == action.tool_result,
                                backend_result_equal=original_output == alternate_output,
                                original_backend_output=original_output, alternate_backend_output=alternate_output,
                                changed_tables_ignoring_timestamps=differences,
                                non_audit_tables_equal=not any(name != "audit_log" for name in differences))
                    cases.append(case)
                finally:
                    alternate.close()
    finally:
        bank.close()
    # Unfinished/failed calls do not get a fictitious negative label.
    for identity in sorted(set(samples) - matched):
        labels.append({"sample_id": identity, "targets": dispatch_targets([]),
                       "prefix_sha256": prefix_digest(inputs, identity), "verified_actual_harm": None,
                       "meaning": "No completed native action with independently rechecked decision."})
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("multi_agent.") and path and not Path(path).resolve().is_relative_to(saved):
            raise ValueError("non-frozen official module loaded")
    output.mkdir(parents=True)
    write_new(output / "inputs.json", inputs)
    write_new(output / "evaluation.json", {"labels": labels, "resource_cases": cases, "literal_policy_conflicts": conflicts,
        "targets_are_retrospective": True, "fixed_dag": False, "independent_test": False})
    report = {"task_id": task_id, "status": manifest["status"], "samples": len(samples),
              "official_decisions_reproduced": len(rechecked), "original_tools": len(tools),
              "backend_result_replay_matches": replay_matches,
              "target_counts": {key: dict(Counter(str(r["targets"][key]) for r in labels)) for key in TARGETS},
              "resource_flagged_actions": len(cases), "resource_rule_matches": sum(len(c["rules"]) for c in cases),
              "literal_conflict_fields": len(conflicts),
              "unavoidable_conflict_actions": sum(bool(c["unavoidable_schema_conflict_fields"]) for c in cases),
              "empty_only_resource_actions": sum(c["all_mismatches_empty"] for c in cases),
              "drop_empty_removes_resource_flag": sum(c["checker_only_drop_empty_targets"]["resource_rule"] == 0 for c in cases),
              "drop_empty_same_backend_result": sum(c["backend_result_equal"] for c in cases),
              "drop_empty_same_non_audit_tables": sum(c["non_audit_tables_equal"] for c in cases),
              "original_run_sha256": sha(run / "RUN.json"), "plan_sha256": plan["plan_sha256"],
              "original_artifacts_verified": len(manifest["artifact_sha256"]),
              "artifact_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
              "no_model_fitting": True, "no_new_api_calls": True,
              "scope": "Post-hoc development diagnostics; no original-label changes, class manufacture or held-out performance."}
    write_new(output / "REPORT.json", report)
    return report


def aggregate(plan_path, output, worker_python):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new report directory required")
    output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / 'code/commands/harnessaudit_cohort.py', ROOT / 'code/commands/summarize_harnessaudit_cohort.py',
               ROOT / 'code/src/paperalpha_runtime/dispatch_dataset.py', ROOT / 'code/src/paperalpha_runtime/journal.py',
               ROOT / 'code/src/paperalpha_runtime/initial_task_context.py', ROOT / 'code/src/paperalpha_runtime/semantic_evidence.py']
    for source in sources:
        dest = output / "source_snapshot" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(source.read_bytes())
    reports = []
    for task in plan["tasks"]:
        dest = output / "runs" / task["task_id"]
        process = subprocess.run([str(worker_python), "-X", "utf8", "-B", str(Path(__file__).resolve()), "worker",
                                  "--plan", str(plan_path), "--task", task["task_id"], "--out", str(dest)],
                                 capture_output=True, text=True, encoding="utf-8")
        if process.returncode:
            write_new(output / f"failure-{task['task_id']}.json", {"stderr": process.stderr, "stdout": process.stdout,
                                                                "returncode": process.returncode})
            raise RuntimeError("supervision worker failed; retain diagnostics without automatic retry")
        reports.append(json.loads((dest / "REPORT.json").read_text(encoding="utf-8")))
    sums = ("samples", "official_decisions_reproduced", "original_tools", "backend_result_replay_matches",
            "resource_flagged_actions", "resource_rule_matches", "empty_only_resource_actions",
            "drop_empty_removes_resource_flag", "drop_empty_same_backend_result", "drop_empty_same_non_audit_tables",
            "literal_conflict_fields", "unavoidable_conflict_actions")
    result = {"scope": "Action-level supervision and label semantics; not model effectiveness.",
              "plan_sha256": plan["plan_sha256"], "runs": reports,
              "totals": {k: sum(r[k] for r in reports) for k in sums},
              "target_counts": {k: dict(sum((Counter(r["target_counts"][k]) for r in reports), Counter())) for k in TARGETS},
              "original_labels_unchanged": True, "independent_samples": False,
              "ready_for_primary_safety_training": False,
              "training_blockers": ["Tiny correlated development cohort; not a train/test split.",
                                    "Some resource labels are unavoidable under schema-valid inputs or sensitive to empty placeholders.",
                                    "No high-severity tool-boundary positives and no verified-harm labels.",
                                    "Current v7 outcome forecast is not an action-specific supervision head."],
              "no_new_api_calls": True, "no_model_fitting": True,
              "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_new(output / "REPORT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["worker", "aggregate"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    result = run_worker(args.plan, args.task, args.out) if args.mode == "worker" else aggregate(args.plan, args.out, args.worker_python)
    print(json.dumps(result.get("totals", {"samples": result.get("samples")}), ensure_ascii=False))
