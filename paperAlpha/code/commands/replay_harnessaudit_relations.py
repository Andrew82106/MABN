"""Offline development check of API evidence against independent state labels.

Not a forecasting benchmark: two task-specific contracts, no learned model,
and no new API calls. Original acquisition files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.harnessaudit_relational_evidence import interpret_relational_evidence
from paperalpha_runtime.harnessaudit_relational_oracle import evaluate_contract

SOURCES = ('code/commands/replay_harnessaudit_relations.py',
           'code/src/paperalpha_monitor/harnessaudit_relational_evidence.py',
           'code/src/paperalpha_runtime/harnessaudit_relational_oracle.py',
           'code/tests/test_harnessaudit_relational_evidence.py',
           'code/tests/test_harnessaudit_relational_oracle.py',
           'code/tests/test_harnessaudit_relational_replay.py',
           'knowledge/risks/relational_contracts.json')
SUPPORTED = {"routed_case_before_compensation", "bound_risk_update_before_trade"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def compare_actions(evidence, labels):
    """Unknown is coverage, never disagreement or a silently imputed negative."""
    actual = {row["invocation_id"]: row for row in labels["events"]}
    predicted = {row["invocation_id"]: row for row in evidence["actions"]}
    if len(actual) != len(labels["events"]) or len(predicted) != len(evidence["actions"]):
        raise ValueError("duplicate trigger identity")
    if set(actual) != set(predicted):
        raise ValueError("monitor and oracle trigger inventories disagree")
    rows = []
    for identity, row in predicted.items():
        target = actual[identity]
        checks = {}
        for kind in ("attempt", "effect"):
            estimate, truth = row[kind]["violation"], target[kind + "_violation"]
            checks[kind] = {"monitor": estimate, "oracle": truth,
                            "known_both": estimate is not None and truth is not None,
                            "agree": None if estimate is None or truth is None else estimate == truth}
        rows.append({"invocation_id": identity, "checks": checks})
    return rows


def run(cohort, contracts_path, output):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("derived replay already exists; never overwrite evidence")
    audit = read(cohort / "ARTIFACT_AUDIT.json")
    if audit.get("status") != "passed" or not audit.get("artifact_sha256"):
        raise ValueError("passed acquisition artifact audit required")
    for relative, expected in audit["artifact_sha256"].items():
        path = (cohort / relative).resolve()
        if not path.is_relative_to(cohort) or sha(path) != expected:
            raise ValueError("audited acquisition changed: " + relative)
    contracts = [c for c in read(contracts_path)["contracts"] if c["id"] in SUPPORTED]
    if len(contracts) != 2 or {c["id"] for c in contracts} != SUPPORTED:
        raise ValueError("the two reviewed contracts must be explicitly present once")
    source_hashes = {s: sha(ROOT / s) for s in SOURCES}
    output.mkdir(parents=True)
    for relative in SOURCES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    plan = {"schema": "harnessaudit-relational-replay-plan-1", "mode": "development_only_offline",
            "cohort": str(cohort), "acquisition_audit_sha256": sha(cohort / "ARTIFACT_AUDIT.json"),
            "contracts_sha256": sha(contracts_path), "contracts": contracts, "source_sha256": source_hashes,
            "exposure": "Root already inspected aggregate tool inventories and acquisition records. Both implementations were developed without reading these real outputs. Not an untouched confirmation test.",
            "boundary": "Current-evidence correctness and coverage only; not future risk accuracy, native MAS baseline superiority, task success or global safety."}
    write_new(output / "PLAN.json", plan)
    cases = []
    for contract in contracts:
        task = contract["task_id"]
        folder = cohort / "cases" / task
        # Distinct input channels: monitor function never receives ORACLE.json.
        evidence = interpret_relational_evidence(read(folder / "MONITOR.json"), contract)
        labels = evaluate_contract(read(folder / "ORACLE.json"), contract, task_id=task)
        comparisons = compare_actions(evidence, labels)
        prefixes = []
        for path in sorted((folder / "stages").glob("*.json")):
            stage = read(path)
            prefix = stage["prefix_before"]
            result = interpret_relational_evidence(prefix, contract)
            prefixes.append({"stage_id": stage["stage_id"], "source": str(path.relative_to(cohort)),
                             "source_sha256": sha(path), "interpretation": result})
        cases.append({"task_id": task, "contract_id": contract["id"], "visible_evidence": evidence,
                      "private_labels": labels, "comparisons": comparisons, "stage_prefixes": prefixes})
    comparisons = [check for case in cases for row in case["comparisons"] for check in row["checks"].values()]
    disagreements = sum(c["agree"] is False for c in comparisons)
    report = {"schema": "harnessaudit-relational-replay-report-1", "status": "disagreement_found" if disagreements else "replayed",
              "plan_sha256": sha(output / "PLAN.json"), "cases": cases, "total_trigger_calls": sum(len(c["comparisons"]) for c in cases),
              "known_comparisons": sum(c["known_both"] for c in comparisons), "unknown_comparisons": sum(not c["known_both"] for c in comparisons),
              "disagreements": disagreements, "api_calls": 0,
              "limitations": ["Two already exposed development tasks, not statistical evidence of generalization.",
                              "No-trigger closed episodes are retained; no target action is not successful task completion.",
                              "Private labels are evaluation-only. Unknown visible evidence is reported, not imputed safe.",
                              "The original native labels and source acquisition remain unchanged."]}
    if any(sha(ROOT / s) != value for s, value in source_hashes.items()):
        raise ValueError("implementation changed during replay")
    write_new(output / "REPORT.json", report)
    return {k: report[k] for k in ("status", "total_trigger_calls", "known_comparisons", "unknown_comparisons", "disagreements", "api_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit')
    parser.add_argument("--contracts", type=Path, default=ROOT / 'knowledge/risks/relational_contracts.json')
    parser.add_argument("--output", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit/relational')
    args = parser.parse_args()
    print(json.dumps(run(args.cohort, args.contracts, args.output), sort_keys=True))
