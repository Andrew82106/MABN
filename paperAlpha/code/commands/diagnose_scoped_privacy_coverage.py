"""Post-hoc coverage/information audit; never trains or changes pilot scores."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


def information_witness(case, coordinator_text, worker_text, final_text, runtime):
    """Fixed actual texts, two policy states: a constructed witness, not new runs."""
    arms = []
    for treatment in ("valid", "stale_root"):
        arm = copy.deepcopy(case)
        arm["treatment"] = treatment
        records = runtime.issuance(arm) + runtime.authority_transition(arm)
        arms.append({"observation": runtime.observation(arm, coordinator_text, worker_text, records, "root_updates_missing"),
            "request": runtime.stage_request(arm, runtime.STAGES[2], worker_text),
            "label": runtime.outcome(arm, records, final_text)})
    return {"missing_view_identical": arms[0]["observation"] == arms[1]["observation"],
        "publisher_request_identical": arms[0]["request"] == arms[1]["request"],
        "targets_change": {k: arms[0]["label"]["targets"][k] != arms[1]["label"]["targets"][k] for k in ("1", "2")},
        "scope": "Controlled same-text policy witness; not a second API trajectory or proof of model causal behavior"}


def diagnose(run, audit, out):
    if out.exists():
        raise ValueError("new diagnosis output required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from collect_scoped_privacy_api import load, sha, write
    from paperalpha_runtime.privacy_transfer import delivered_text
    import paperalpha_runtime.scoped_privacy_api as runtime
    plan = load(run)
    report = json.loads((audit / "REPORT.json").read_text(encoding="utf-8"))
    assert report["plan_sha256"] == plan["sha256"]
    events = [json.loads(line) for line in (run / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = json.loads((audit / "recomputed_predictions.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in plan["cases"]}
    witnesses, mismatch_fields = [], 0
    for row in rows:
        case = cases[row["id"]]
        responses = [e for e in events if e["case_id"] == row["id"] and e["kind"] == "api_response"]
        assert [e["stage"] for e in responses] == list(runtime.STAGES)
        text = [delivered_text(e["response"])["text"] for e in responses]
        witness = information_witness(case, *text, runtime)
        witnesses.append({"id": row["id"], **witness})
        cache_truth = runtime.outcome(case, runtime.issuance(case), text[2])
        mismatch_fields += sum(a != b for a, b in zip(cache_truth["authority_failure"], row["outcome"]["authority_failure"]))
    profile = report["profiles_per_tracked_value"]
    result = {"audit_report_sha256": sha(audit / "REPORT.json"), "plan_sha256": plan["sha256"], "post_hoc": True,
        "real_completed_runs": len(rows), "distinct_templates": report["independent_templates"],
        "profile_counts": profile, "distinct_presence_profiles": len(profile),
        "has_worker_omission": any("W0" in p for p in profile), "has_final_redaction": any("F0" in p for p in profile),
        "both_system_outcomes": {k: 0 < report["positive_runs_by_target"][k] < len(rows) for k in ("1", "2")},
        "cached_vs_actual_authority_disagreements_per_field": mismatch_fields,
        "same_text_information_witnesses": witnesses,
        "full_record_persistence_brier": {k: report["metrics"]["full"]["0.5"][k]["current_presence_shared_gate"]["brier"] for k in ("1", "2")},
        "supported": ["Actual internal reference transmission occurred", "Both literal/reference-policy outcomes occurred",
            "Shared authority graph scores were delivered before final API calls"],
        "not_supported": ["General detection or cross-workflow performance", "Learned authority calibration",
            "Predicting content filtering: no F0 examples", "Outperforming all baselines or original MAS methods",
            "Inferring hidden revocation from unchanged observable evidence"],
        "next_model_requirement": "Separate the authority evidence available to the publisher from the monitor's current policy evidence; output generation and violation truth need not condition on the same record snapshot",
        "next_data_requirement": "New frozen workflows with legitimate internal use and actual final filtering/withholding contrasts, not forced final labels; retain these pilot outputs unchanged",
        "no_model_changes": True}
    write(out, result)
    print(json.dumps({k: result[k] for k in ("real_completed_runs", "distinct_presence_profiles", "cached_vs_actual_authority_disagreements_per_field", "has_final_redaction")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run.resolve(), args.audit.resolve(), args.out.resolve())
