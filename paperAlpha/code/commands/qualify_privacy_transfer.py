"""Post-collection coverage diagnosis; never tune or replace frozen scores."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(plan, events, predictions):
    cases = {c["id"]: c for c in plan["cases"]}
    verified = {r["case_id"]: r for r in predictions}
    if len(verified) != len(predictions) or not set(verified) <= set(cases):
        raise ValueError("unique verified cases required")
    counts, types, rows = Counter(), Counter(), []
    for case_id, record in verified.items():
        prefixes = [e for e in events if e["kind"] == "prediction" and e["case_id"] == case_id]
        if len(prefixes) != 2 or [p["packet"]["stage"] for p in prefixes] != [1, 2]:
            raise ValueError("two verified cutoffs required")
        first, second = [p["packet"] for p in prefixes]
        if [(g["field_names"], g["coordinator_present"]) for g in first["groups"]] != [(g["field_names"], g["coordinator_present"]) for g in second["groups"]]:
            raise ValueError("earlier evidence changed")
        c, w = sum(g["coordinator_present"] for g in second["groups"]), sum(g["worker_present"] for g in second["groups"])
        y = record["outcome"]["y"]
        counts[f"C{int(bool(c))}W{int(bool(w))}F{y}"] += 1
        types.update(g["kind"] for g in second["groups"])
        rows.append({"case_id": case_id, "template_id": cases[case_id]["template_id"], "condition": cases[case_id]["condition"],
            "protected_objects": len(second["groups"]), "C_present_objects": c, "W_present_objects": w,
            "final_positive": y})
    positives = sum(r["final_positive"] for r in rows)
    c_runs = sum(r["C_present_objects"] > 0 for r in rows)
    w_runs = sum(r["W_present_objects"] > 0 for r in rows)
    both_labels = 0 < positives < len(rows)
    reasons = []
    if not positives:
        reasons.append("No positive final literal disclosures: sensitivity or ranking cannot be estimated.")
    if positives == len(rows):
        reasons.append("No negative completed tasks: false alarms cannot be estimated.")
    if not c_runs and not w_runs:
        reasons.append("No protected literal appeared in either internal stage; observed information-transfer contrasts are absent.")
    reasons.append("Self-authored tasks share one functional topology; not evidence of arbitrary-workflow generalization or original MAS method superiority.")
    return {"schema": "privacy-transfer-evidence-qualification-v1", "diagnostic_only": True,
        "counts": {"planned": len(cases), "completed": len(rows), "positive": positives, "negative": len(rows) - positives,
            "templates": len({r["template_id"] for r in rows}), "internal_C_positive_runs": c_runs, "internal_W_positive_runs": w_runs},
        "patterns": dict(counts), "protected_object_types": dict(types),
        "eligible_descriptive_checks": {"negative_case_false_alarms": len(rows) > positives, "binary_ranking": both_labels,
            "positive_case_sensitivity": bool(positives), "observed_internal_transfer_examples": bool(c_runs or w_runs)},
        "general_MAS_superiority_established": False,
        "interpretation": "These are necessary coverage checks, not proof that other validity requirements pass.",
        "reasons": reasons, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new diagnostic file required")
    plan = json.loads((args.run / "PLAN.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (args.run / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    verified_report = json.loads((args.verified / "REPORT.json").read_text(encoding="utf-8"))
    if verified_report["plan_sha256"] != plan["sha256"] or not verified_report["predictions_precede_future_calls"]:
        raise ValueError("matching completed audit required")
    predictions = json.loads((args.verified / "recomputed_predictions.json").read_text(encoding="utf-8"))
    report = diagnose(plan, events, predictions)
    report["source_sha256"] = sha(Path(__file__))
    report["input_sha256"] = {str(p.resolve()): sha(p) for p in (args.run / "PLAN.json", args.run / "journal.jsonl",
        args.verified / "REPORT.json", args.verified / "recomputed_predictions.json")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"counts": report["counts"], "eligible": report["eligible_descriptive_checks"]}), flush=True)


if __name__ == "__main__":
    main()
