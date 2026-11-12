"""Re-audit historical artifacts with repaired validators; no new performance run.

Reads old results, never calls their writing verification entry points, and
exclusively creates a separate receipt directory. No API calls or fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from diagnose_dependence_envelope import inspect_witness
from verify_crossed_purchase import verify_flow, metrics
from paperalpha_runtime.canonical import digest

SCRIPT = 'code/commands/verify_repeat_upgrade_history.py'
REPAIRS = ('code/commands/diagnose_dependence_envelope.py', 'code/commands/verify_crossed_purchase.py')
PRIMARY_ARTIFACTS = ("PLAN.json", "COLLECTION.json", "EVALUATION.json", "VERIFIED_OUTCOMES.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def tree_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(out, crossed, envelopes):
    out, crossed, envelopes = (Path(p).resolve() for p in (out, crossed, envelopes))
    for old in (crossed, envelopes):
        require(out != old and old not in out.parents and out not in old.parents,
                "receipt and historical directories must be disjoint")
    # Existing output is never reused, even if an earlier observation timed out.
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    plan, collection = read(crossed / "PLAN.json"), read(crossed / "COLLECTION.json")
    evaluation, outcomes = read(crossed / "EVALUATION.json"), read(crossed / "VERIFIED_OUTCOMES.json")
    envelope_plan, envelope_report = read(envelopes / "PLAN.json"), read(envelopes / "REPORT.json")
    names = sorted(set(plan["source_sha256"]) | set(envelope_plan["source_sha256"]) | set(REPAIRS) | {SCRIPT})
    current_hashes = {name: sha(ROOT / name) for name in names}
    old_hashes = {"crossed": tree_hashes(crossed), "envelopes": tree_hashes(envelopes)}
    environment = {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
                   "executable_sha256": sha(sys.executable), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pytest")}}
    header = {"schema": "repaired-history-audit-v1", "started_at_utc": started,
              "crossed_source": str(crossed), "envelope_source": str(envelopes),
              "script": SCRIPT, "script_sha256": current_hashes[SCRIPT],
              "environment": environment, "current_source_sha256": current_hashes,
              "new_api_calls": 0, "new_fits": 0, "new_model_predictions": 0,
              "scope": "Retrospective correctness re-audit of exposed artifacts using repaired validators. Not new data, a new independent test, or evidence of improved model performance."}
    write(out / "START.json", header)
    write(out / "HISTORICAL_FILES_BEFORE.json", old_hashes)
    try:
        changes = {}
        for tag, directory, frozen_plan in (("crossed", crossed, plan), ("envelopes", envelopes, envelope_plan)):
            for name, expected in frozen_plan["source_sha256"].items():
                require(sha(directory / "source" / name) == expected, "historical frozen source changed: " + tag + ":" + name)
            changes[tag] = {name: {"historical_sha256": expected, "current_sha256": current_hashes[name]}
                            for name, expected in frozen_plan["source_sha256"].items() if current_hashes[name] != expected}
        require(collection["plan_sha256"] == evaluation["plan_sha256"] == sha(crossed / "PLAN.json"), "crossed plan binding differs")
        require(evaluation["collection_sha256"] == sha(crossed / "COLLECTION.json"), "crossed collection binding differs")
        require(envelope_report["plan_sha256"] == sha(envelopes / "PLAN.json"), "envelope plan binding differs")
        for name, expected in envelope_plan["source_artifacts"].items():
            require(sha(crossed / name) == expected, "envelope source artifact binding differs: " + name)
        for name, expected in plan["model_sha256"].items():
            require(sha(crossed / "models" / name) == expected, "historical model changed: " + name)
        cases = plan["cases"]
        require(len(cases) == 24 and len({c["id"] for c in cases}) == 24, "expected all 24 historical flows")
        case_ids = {c["id"] for c in cases}
        require(set(collection["reports_sha256"]) == case_ids, "flow manifest coverage differs")
        original_labels = {r["id"]: r for r in outcomes["rows"]}
        require(len(outcomes["rows"]) == len(original_labels) == 24 and set(original_labels) == case_ids,
                "original outcome coverage differs")
        rebuilt, flow_receipts = {}, []
        for case in cases:
            identity = case["id"]
            result = verify_flow(case, crossed, collection["reports_sha256"][identity])
            require(result["complete"], "historical cohort unexpectedly incomplete: " + identity)
            original = original_labels[identity]
            require(result["label"] == original["label"], "historical label differs: " + identity)
            require(original["profile"] == case["profile"] and original["domain"] == case["domain"]
                    and original["priority"] == result["prefix"]["priority"], "historical outcome identity differs")
            rebuilt[identity] = result
            flow_receipts.append({"id": identity, "report_sha256": collection["reports_sha256"][identity],
                "forecast_sha256": sha(crossed / "runs" / identity / "FORECAST.json"),
                "input_sha256": result["sample"].sha256, "protocol_audit": result["protocol_audit"],
                "label": result["label"], "original_label_equal": True,
                "frozen_forecast_exactly_reproduced": True, "methods": sorted(result["forecast"]["methods"])})
        rows = list(rebuilt.values())
        summaries = {}
        for method in rows[0]["forecast"]["methods"]:
            require(all(method in r["forecast"]["methods"] for r in rows), "method coverage differs")
            summaries[method] = {"all_complete": metrics(rows, method),
                "by_profile": {p: metrics([r for r in rows if r["case"]["profile"] == p], method)
                               for p in sorted({c["profile"] for c in cases})},
                "by_priority": {p: metrics([r for r in rows if r["prefix"]["priority"] == p], method)
                                for p in ("economy", "balanced", "quality")}}
        require(summaries == evaluation["metrics"], "historical metrics do not reproduce exactly")
        expected_pairs = {(identity, method) for identity in case_ids for method in envelope_plan["methods"]}
        reported_pairs = [(r["id"], r["method"]) for r in envelope_report["rows"]]
        require(len(reported_pairs) == len(set(reported_pairs)) == 48 and set(reported_pairs) == expected_pairs,
                "expected all 48 unique case/method envelopes")
        envelope_receipts = []
        for row in envelope_report["rows"]:
            path = envelopes / row["envelope_file"]
            require(sha(path) == row["envelope_sha256"], "historical envelope changed: " + row["envelope_file"])
            report = read(path)
            require(set(report["queries"]) == set(envelope_plan["queries"]), "envelope query coverage differs")
            require(report["input_sha256"] == rebuilt[row["id"]]["sample"].sha256, "envelope input binding differs")
            method = row["method"]
            require(report["parameter_sha256"] == digest(read(crossed / "models" / (method + ".json"))),
                    "envelope parameter binding differs")
            error = inspect_witness(report)
            system = report["queries"]["system:any"]
            require({k: system[k] for k in ("lower", "reference", "upper", "width")} == row["system_risk"],
                    "historical envelope row differs")
            forecast = rebuilt[row["id"]]["forecast"]["methods"][method]
            require(abs(system["reference"] - forecast["risk_probability"]) <= 1e-12,
                    "envelope reference does not match original forecast")
            events = report["witness_problem"]["events"]
            envelope_receipts.append({"id": row["id"], "method": method,
                "envelope_file": row["envelope_file"], "envelope_sha256": row["envelope_sha256"],
                "queries": len(events), "endpoints": 2 * len(events), "max_certificate_error": error,
                "unknown_authority_block": "authority:joint_facts" in report["blocks"],
                "constant_zero_queries": [q for q, values in events.items() if all(v == 0. for v in values)],
                "system_equals_budget_truth_table": events["system:any"] == events["channel:cumulative_budget"],
                "system_risk": row["system_risk"], "original_point_forecast_unchanged": True})
        after = {"crossed": tree_hashes(crossed), "envelopes": tree_hashes(envelopes)}
        write(out / "HISTORICAL_FILES_AFTER.json", after)
        require(after == old_hashes, "historical artifacts changed during audit")
        require({name: sha(ROOT / name) for name in names} == current_hashes, "current audit source changed during execution")
        report = {**header, "finished_at_utc": now(), "status": "passed",
            "current_vs_historical_source_changes": changes,
            "historical_files_before_sha256": sha(out / "HISTORICAL_FILES_BEFORE.json"),
            "historical_files_after_sha256": sha(out / "HISTORICAL_FILES_AFTER.json"),
            "all_historical_file_hashes_unchanged": True,
            "original_four_artifacts": {n: {"before": old_hashes["crossed"][n], "after": after["crossed"][n]} for n in PRIMARY_ARTIFACTS},
            "summary": {"flows": len(flow_receipts),
                "requests_verified": sum(r["protocol_audit"]["requests"] for r in flow_receipts),
                "runtime_events_verified": sum(r["protocol_audit"]["runtime_events"] for r in flow_receipts),
                "prediction_vectors_reproduced": sum(len(r["methods"]) for r in flow_receipts),
                "original_metrics_exactly_reproduced": True,
                "envelopes": len(envelope_receipts), "queries": sum(r["queries"] for r in envelope_receipts),
                "endpoints": sum(r["endpoints"] for r in envelope_receipts),
                "max_certificate_error": max(r["max_certificate_error"] for r in envelope_receipts),
                "unknown_authority_blocks": sum(r["unknown_authority_block"] for r in envelope_receipts),
                "constant_zero_queries": sum(len(r["constant_zero_queries"]) for r in envelope_receipts),
                "system_budget_duplicate_tables": sum(r["system_equals_budget_truth_table"] for r in envelope_receipts)},
            "flow_audits": flow_receipts, "envelope_audits": envelope_receipts,
            "limits": ["Both historical cohorts were exposed before these validator repairs.",
                       "The 24 historical flows all completed; this receipt provides no empirical failure-case coverage.",
                       "Endpoint counts include duplicate and constant-zero truth tables; they are not independent mechanisms or samples.",
                       "Unknown-authorization behavior is not empirically covered by this cohort.",
                       "Strict certificate checking verifies the stated fixed-marginal optimization, not calibration or true risk coverage.",
                       "No old result, model parameter, threshold, label or default selection is changed."]}
        write(out / "REPORT.json", report)
        return {"report": str(out / "REPORT.json"), "report_sha256": sha(out / "REPORT.json"),
                "summary": report["summary"], "current_source_files": len(names),
                "source_changes": {tag: sorted(values) for tag, values in changes.items()}}
    except Exception as exc:
        write(out / "FAILURE.json", {"failed_at_utc": now(), "error_type": type(exc).__name__, "message": str(exc),
                                      "old_results_not_intentionally_modified": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crossed-source", type=Path, default=ROOT / 'experiments/purchase/runs/crossed_purchase_prospective_20260920')
    parser.add_argument("--envelope-source", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/dependence_envelope_diagnostic_20260930')
    args = parser.parse_args()
    print(json.dumps(audit(args.out, args.crossed_source, args.envelope_source), ensure_ascii=False, sort_keys=True))
