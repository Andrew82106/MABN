"""Recompute official labels and report suitability, without fitting a monitor.

Every planned attempt appears in the report. Monitor observations and post-run
labels are separate artifacts; neither rule matches nor task policies enter the
observation projection. A complete execution is not necessarily task success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from harnessaudit_cohort import ROOT, digest, load_plan, sha, write_new

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.journal import RuntimeEvidenceProjector


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run(plan, task, manifest, receipt, run):
    settings = plan["settings"]
    pairs = {"task_id": task["task_id"], "upstream_commit": plan["upstream_commit"],
             "model_requested": settings["model"], "request_budget": settings["request_budget"],
             "max_hub_turns": settings["max_turns"], "agent_timeout_seconds": settings["timeout"]}
    if any(manifest.get(k) != v for k, v in pairs.items()):
        raise ValueError("run settings do not match frozen plan")
    if receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("task_id") != task["task_id"]:
        raise ValueError("receipt is not bound to this task/plan")
    if receipt.get("run_manifest_sha256") != sha(run / "RUN.json"):
        raise ValueError("run manifest changed after process completion")
    if manifest.get("status") not in {"completed", "failed"} or manifest["started_at"] < plan["created_at"]:
        raise ValueError("run is not finalized or predates the cohort")
    if sha(run / "source_snapshot/upstream" / task["task_path"]) != task["task_sha256"]:
        raise ValueError("executed task does not match selected task")
    if sha(run / "source_snapshot/collect_harnessaudit_api_pilot.py") != plan["source_sha256"]['code/commands/collect_harnessaudit_api_pilot.py']:
        raise ValueError("executed pilot differs from frozen pilot")


def qualified_label(evaluation, run_status, audit_ok):
    reasons = list(evaluation["ineligibility_reasons"])
    if not audit_ok:
        reasons.append("acquisition_audit_failed")
    if run_status != "completed":
        reasons.append("acquisition_not_completed")
    label = evaluation["label"] if not reasons and evaluation["eligible"] else None
    return {"target": evaluation["target"], "label": label,
            "eligible": label in (0, 1), "ineligibility_reasons": sorted(set(reasons)),
            "meaning": evaluation["meaning"]}


def class_readiness(rows):
    counts = Counter("null" if row["label"] is None else str(row["label"]) for row in rows)
    both = counts["0"] > 0 and counts["1"] > 0
    return {"label_counts": {key: counts[key] for key in ("0", "1", "null")},
            "both_outcome_classes_present": both, "final_test_eligible": False,
            "independent_performance_claim_eligible": False,
            "next_data_requirement": "Freeze grouped fitting/evaluation and sample-size protocol; this tiny development collection is not a final test."
                if both else "Acquire independently checked negative and positive outcomes; do not relabel failures or delete positives to manufacture balance."}


def rule_match_profile(evaluation):
    """Descriptive breakdown only; do not change the frozen binary target.

    Official V-OT includes low-severity unnecessary/unmentioned tools, whereas
    V-OR deliberately has no severity. Neither is a verified-harm annotation.
    Count rule matches and distinct actions separately.
    """
    groups = {}
    for decision in evaluation["decisions"]:
        if not decision["violated"]:
            continue
        key = f"{decision.get('violation_code') or 'unknown'}:{decision.get('severity') or 'ungraded'}"
        item = groups.setdefault(key, {"rule_matches": 0, "action_sequences": set()})
        item["rule_matches"] += 1
        item["action_sequences"].add(decision["action_sequence_num"])
    return {key: {"rule_matches": item["rule_matches"], "distinct_actions": len(item["action_sequences"])}
            for key, item in sorted(groups.items())}


def inspect_task(plan, task, cohort, output, audit_python):
    # Keep manifest/label helpers usable in the lightweight SDK audit runtime.
    from paperalpha_monitor.harnessaudit_observations import project_records

    identity = task["task_id"]
    run = cohort / "runs" / identity
    receipt_path = cohort / "attempts" / f"{identity}.finished.json"
    row = {"task_id": identity, "category": task["category"], "label": None,
           "status": "unavailable", "audit_passed": False}
    if not receipt_path.exists() or not (run / "RUN.json").exists():
        row["reason"] = "missing_process_completion_or_run_manifest; inspect live handle before any retry"
        return row
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    validate_run(plan, task, manifest, receipt, run)
    row.update(status=manifest["status"], api_calls=manifest["api_calls_started"],
               input_tokens=manifest["input_tokens"], output_tokens=manifest["output_tokens"],
               api_calls_failed=manifest["api_calls_failed"], actions=manifest["actions"],
               tool_calls=manifest["tool_calls"], active_agents=len(manifest["active_agent_roles"]),
               violation_rule_matches=manifest["recorded_violations"], run_manifest_sha256=sha(run / "RUN.json"))
    audit_path = output / "audits" / f"{identity}.json"
    # A fresh subprocess binds imports to this run's frozen official checker.
    check = subprocess.run([str(audit_python), "-X", "utf8", "-B", str(ROOT / 'code/commands/audit_harnessaudit_api_pilot.py'),
                            "--run", str(run), "--out", str(audit_path)], capture_output=True, text=True, encoding="utf-8")
    row["audit_passed"] = check.returncode == 0 and audit_path.exists()
    if not row["audit_passed"]:
        write_new(output / "audit_failures" / f"{identity}.json",
                  {"returncode": check.returncode, "stderr": check.stderr, "stdout": check.stdout})
    bundles = project_records(read_rows(run / "native_trace.jsonl"))
    if len(bundles) != 1:
        raise ValueError("one trace per run required")
    evaluation = bundles[0]["evaluation"]
    row["rule_match_profile"] = rule_match_profile(evaluation)
    label = qualified_label(evaluation, manifest["status"], row["audit_passed"])
    row.update(label)
    if row["audit_passed"]:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        row.update(violated_actions=audit["violated_actions"], unflagged_actions=audit["observed_actions"] - audit["violated_actions"])
    projector, events = RuntimeEvidenceProjector(), []
    for raw in read_rows(run / "runtime_events.jsonl"):
        event = projector.push(raw)
        if event is not None:
            events.append(event)
    # Task/category/policies/audits remain evaluation-side only.
    observations = {"schema": "cohort-runtime-observations-v1", "events": events,
                    "meaning": "Visible journal projection only; no post-run labels, checks, task IDs or category features."}
    write_new(output / "observations" / f"{identity}.json", observations)
    write_new(output / "evaluation" / f"{identity}.json", {"qualification": label, "official_evaluation": evaluation})
    row.update(monitor_events=len(events), observations_sha256=digest(observations),
               journal=projector.report())
    return row


def summarize(plan_path, output, audit_python=None):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new report directory required; keep earlier audits")
    output.mkdir(parents=True, exist_ok=False)
    audit_python = Path(audit_python or sys.executable).resolve()
    if not audit_python.is_file():
        raise ValueError("audit Python executable must exist")
    rows = [inspect_task(plan, task, plan_path.parent, output, audit_python) for task in plan["tasks"]]
    report = {"scope": "Development acquisition and independently recomputed rule labels, not trained model performance.",
              "plan_sha256": plan["plan_sha256"], "tasks": rows, "readiness": class_readiness(rows),
              "runtimes": {"projection_python": sys.executable, "frozen_upstream_audit_python": str(audit_python)},
              "totals": {key: sum(r.get(key, 0) for r in rows) for key in
                         ("api_calls", "api_calls_failed", "input_tokens", "output_tokens", "actions", "tool_calls", "monitor_events")},
              "all_planned_attempts_present": all(r["status"] in {"completed", "failed"} for r in rows),
              "all_acquisition_audits_passed": all(r["audit_passed"] for r in rows),
              "no_model_training": True, "no_baseline_performance_claim": True,
              "limitations": ["Shared task templates; new task IDs do not prove unseen-family generalization.",
                              "Native hub/spoke topology includes return edges; not the target fixed-DAG test.",
                              "Rule-match target includes resource-scope V-OR and does not measure verified external harm.",
                              "V-OT also includes low-severity unnecessary/unmentioned tools; report strata instead of calling every flag dangerous.",
                              "No perturbations, independent action-mechanism labels or synchronous text extraction.",
                              "More than one label class is necessary but not sufficient for meaningful evaluation."]}
    if report["totals"]["api_calls"] > plan["max_total_requests"]:
        raise ValueError("cohort exceeded frozen request budget")
    source = output / "source_snapshot" / Path(__file__).name
    source.parent.mkdir()
    with source.open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-python", type=Path, help="Python with the collector's official SDK/pydantic dependencies")
    args = parser.parse_args()
    result = summarize(args.plan, args.out, args.audit_python)
    print(json.dumps({"readiness": result["readiness"], "totals": result["totals"],
                      "all_acquisition_audits_passed": result["all_acquisition_audits_passed"]}, ensure_ascii=False))
