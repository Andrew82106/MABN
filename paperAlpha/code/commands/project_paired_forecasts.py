"""Build separate public request-time inputs and private future labels.

This offline development projection runs NO model/API and never trains. All
planned episodes and all actual SDK claims remain in the denominator. Future
targets come from a separate private state implementation after input timing
has been verified. An uncovered input is explicit, not a negative example.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_submission_harnessaudit import read, sha, write_new
from paperalpha_monitor.paired_forecast_observer import project_paired_forecast_input
from paperalpha_monitor.contract_forecast_projection import ForecastProjectionUncovered
from paperalpha_runtime.harnessaudit_forecast_targets import project_next_reply_target

SOURCES = ('code/commands/project_paired_forecasts.py', 'code/tests/test_paired_forecast_dataset.py',
           'code/src/paperalpha_monitor/paired_forecast_observer.py',
           'code/src/paperalpha_monitor/contract_forecast_projection.py',
           'code/src/paperalpha_monitor/harnessaudit_relational_evidence.py',
           'code/src/paperalpha_runtime/harnessaudit_forecast_targets.py',
           'code/src/paperalpha_runtime/harnessaudit_relational_oracle.py',
           'code/src/paperalpha_runtime/journal.py', 'knowledge/risks/relational_contracts.json')
SUPPORTED = {"routed_case_before_compensation", "bound_risk_update_before_trade"}


def project_claim(stage, claim, journal, contract, template, episode_id, oracle, response, error):
    """Only the input projection receives the visible channel; never the oracle."""
    identity = {"episode_id": episode_id, "claim_number": claim["call"], "stage_id": claim["node_id"]}
    try:
        projected = project_paired_forecast_input(stage, claim, journal, contract,
            legitimate_initial_prefix=template, episode_id=episode_id)
    except ForecastProjectionUncovered as exc:
        return ({**identity, "covered": False, "reason": exc.code},
                {**identity, "covered": False, "reason": "input_timing_not_certified",
                 "labels": {k: None for k in ("A", "C", "B", "S")}, "risk_label": None})
    target = project_next_reply_target(oracle, contract, request=claim, response=response,
                                      horizon=projected["prefix_horizon"], response_error=error)
    if (projected["current_a"] is not None and target["labels"]["A"] is not None
            and projected["current_a"] != target["labels"]["A"]):
        raise ValueError("visible hard current evidence contradicts the independent state oracle")
    return ({**identity, "covered": True, "projection": projected}, {**identity, **target})


def verify_manifest(root, entries, *, scope=".", excluded=()):
    root = Path(root).resolve()
    if not isinstance(entries, dict) or not entries:
        raise ValueError("completed acquisition file manifest required")
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("audited acquisition changed: " + relative)
    folder = (root / scope).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("manifest scope escapes cohort")
    actual = {p.relative_to(root).as_posix() for p in folder.rglob("*")
              if p.is_file() and p.relative_to(root).as_posix() not in set(excluded)}
    if actual != set(entries):
        raise ValueError("acquisition file inventory differs from completed audit")


def run(cohort, output):
    import collect_paired_harnessaudit as paired
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("derived dataset already exists; do not overwrite")
    plan = paired.load_plan(cohort)
    audit = read(cohort / "private/ARTIFACT_AUDIT.json")
    if audit.get("status") != "passed" or audit.get("plan_sha256") != plan["sha256"]:
        raise ValueError("passed matching completed acquisition audit required")
    # The paired audit manifest uses cohort-relative paths, including private/.
    verify_manifest(cohort, audit["artifact_sha256"], scope="private", excluded=("private/ARTIFACT_AUDIT.json",))
    verify_manifest(cohort, audit["public_file_sha256"], scope="observable")
    contracts = {c["task_id"]: c for c in read(ROOT / 'knowledge/risks/relational_contracts.json')["contracts"]
                 if c["id"] in SUPPORTED}
    if {c["id"] for c in contracts.values()} != SUPPORTED:
        raise ValueError("the two independently reviewed contract definitions required")
    snapshots = {name: sha(ROOT / name) for name in SOURCES}
    output.mkdir(parents=True)
    for name in SOURCES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    project_plan = {"schema": "paired-request-forecast-projection-plan-v1",
        "mode": "development_only_offline_request_time_replay",
        "cohort": str(cohort), "acquisition_plan_sha256": plan["sha256"],
        "acquisition_audit_sha256": sha(cohort / "private/ARTIFACT_AUDIT.json"),
        "source_sha256": snapshots, "episode_ids": [c["case_id"] for c in plan["cases"]],
        "contracts": contracts,
        "primary_scope": "The first actual invocation in the next API reply, not all calls in a batch or whole-episode safety.",
        "denominator": "All frozen episodes and actual SDK claims, including incomplete/failed replies and uncovered timing.",
        "public": "Only public/ files: safe role/policy/workflow/text prefixes and numeric features.",
        "private": "private/ targets, source/ implementations, and this acquisition-linked plan are evaluator-only.",
        "generalization_limit": "Two task/contract families with three repeated pairs each; not a sufficiently diverse confirmation test or independent sample of arbitrary MAS."}
    write_new(output / "PLAN.json", project_plan)
    summaries = []
    for case in plan["cases"]:
        episode, task = case["case_id"], case["source_task_id"]
        folder = cohort / "private/cases" / episode
        template = paired._legal_template(cohort, case)
        journal = read(folder / "MONITOR.json")["journal"]
        oracle = read(folder / "ORACLE.json")
        stages = [read(p) for p in sorted((folder / "stages").glob("*.json"))]
        by_stage = {s["stage_id"]: s for s in stages}
        if len(by_stage) != len(stages):
            raise ValueError("duplicate stage identity")
        windows, labels = [], []
        for claim_path in sorted((folder / "api/calls").glob("*.claim.json")):
            claim = read(claim_path)
            stem = claim_path.name.removesuffix(".claim.json")
            response_path = claim_path.with_name(stem + ".response.json")
            error_path = claim_path.with_name(stem + ".error.json")
            observed, target = project_claim(by_stage[claim["node_id"]], claim, journal,
                contracts[task], template, episode, oracle,
                read(response_path) if response_path.exists() else None,
                read(error_path) if error_path.exists() else None)
            windows.append(observed)
            labels.append(target)
        write_new(output / "public" / (episode + ".json"), {"episode_id": episode, "task_id": task,
            "contract_id": contracts[task]["id"], "windows": windows,
            "boundary": "Task/episode/stage identities are grouping metadata, never numeric model features."})
        write_new(output / "private" / (episode + ".json"), {"episode_id": episode, "task_id": task,
            "labels": labels, "private_evaluation_only": True})
        summaries.append({"episode_id": episode, "task_id": task, "actual_sdk_claims": len(windows),
            "covered_inputs": sum(w["covered"] for w in windows),
            "known_risk_targets": sum(t.get("risk_label") is not None for t in labels),
            "positive_risk_targets": sum(t.get("risk_label") == 1 for t in labels),
            "call_types": dict(Counter(t["labels"]["C"] or "unknown" for t in labels)),
            "ignored_later_batch_calls": sum(t.get("ignored_later_batch_calls", 0) for t in labels),
            "label_errors": [t["errors"] for t in labels if t.get("errors")]})
    if any(sha(ROOT / name) != value for name, value in snapshots.items()):
        raise ValueError("projection implementation changed during execution")
    verify_manifest(cohort, audit["artifact_sha256"], scope="private", excluded=("private/ARTIFACT_AUDIT.json",))
    verify_manifest(cohort, audit["public_file_sha256"], scope="observable")
    manifest = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    report = {"schema": "paired-request-forecast-projection-report-v1", "status": "projected",
        "plan_sha256": sha(output / "PLAN.json"), "episodes": summaries,
        "actual_sdk_claims": sum(s["actual_sdk_claims"] for s in summaries),
        "covered_inputs": sum(s["covered_inputs"] for s in summaries),
        "known_risk_targets": sum(s["known_risk_targets"] for s in summaries),
        "positive_risk_targets": sum(s["positive_risk_targets"] for s in summaries),
        "artifact_sha256": manifest, "api_calls": 0, "trained_models": 0,
        "performance_claimed": False, "publication_ready": False}
    write_new(output / "private/REPORT.json", report)
    return {k: report[k] for k in ("status", "actual_sdk_claims", "covered_inputs", "known_risk_targets", "positive_risk_targets", "api_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired')
    parser.add_argument("--output", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired/forecast')
    args = parser.parse_args()
    print(json.dumps(run(args.cohort, args.output), ensure_ascii=False))
