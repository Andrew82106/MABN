"""Rebuild prospective forecasts, actual commands and labels without API access."""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read, write, sha, now
from collect_crossed_purchase import load
from evaluate_exclusive_purchase import validate_record_order
from evaluate_choice_attributes import support_diagnosis
from score_crossed_purchase import score
from paperalpha_runtime import crossed_purchase as study
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input


def audit_record_protocol(case, records, report):
    """Replay every recorded operation, even when a later operation failed.

    A missing next record on a failed run is an unobserved boundary, not an
    invented reply, dispatch, or outcome. Transport/storage errors cannot be
    reconstructed from an error class alone; recorded replies can be parsed.
    """
    complete = report.get("complete")
    if type(complete) is not bool or type(report.get("forecast_made")) is not bool:
        raise ValueError("explicit boolean completion and forecast flags required")
    failures = [r for r in records if r.get("kind") == "failure"]
    if complete:
        if failures or report.get("failure_type") is not None:
            raise ValueError("complete flow contains a failure")
        activity = records
    else:
        if (len(failures) != 1 or records[-1] is not failures[0]
                or not isinstance(report.get("failure_type"), str) or not report["failure_type"]
                or failures[0].get("error_type") != report["failure_type"]):
            raise ValueError("one matching terminal failure required")
        activity = records[:-1]
    cursor = 0
    events = []
    audit = {"requests": 0, "responses": 0, "parsed_proposals": 0, "runtime_events": 0}
    result = {"planner": None, "prefix_record": None, "marker": None,
              "events": events, "protocol_audit": audit}

    def take(kind, **binding):
        nonlocal cursor
        if cursor == len(activity):
            if complete: raise ValueError("complete flow missing " + kind)
            audit["unobserved_next_operation"] = {"kind": kind, **binding}
            return None
        record = activity[cursor]
        if record.get("kind") != kind or any(record.get(k) != v for k, v in binding.items()):
            raise ValueError("actual operation order/binding differs: expected " + kind)
        cursor += 1
        counter = {"request": "requests", "response": "responses", "parsed_proposal": "parsed_proposals",
                   "runtime_event": "runtime_events"}.get(kind)
        if counter: audit[counter] += 1
        return record

    invalid = object()
    def parse_response(response, parser):
        try:
            delivered = delivered_text(response["response"])
            if delivered["native_refusal"]: raise RuntimeError("native refusal is not a negative label")
            return delivered["text"], parser(delivered["text"])
        except Exception as exc:
            if complete or cursor != len(activity) or type(exc).__name__ != report["failure_type"]:
                raise ValueError("reply parsing failure does not match terminal failure") from exc
            audit["reproduced_response_failure"] = {"stage": response["stage"], "error_type": type(exc).__name__}
            return invalid

    request = take("request", stage="planner")
    if request is None: return result
    if request.get("request") != study.request(case, "planner"): raise ValueError("planner input differs")
    response = take("response", stage="planner")
    if response is None: return result
    parsed = parse_response(response, study.parse_planner)
    if parsed is invalid: return result
    result["planner"] = parsed[0]
    result["prefix_record"] = take("forecast_prefix")
    if result["prefix_record"] is None: return result
    if result["prefix_record"].get("value") != study.prefix(case, result["planner"]):
        raise ValueError("forecast prefix does not reconstruct")
    result["marker"] = take("frozen_forecast")
    if result["marker"] is None: return result
    if result["marker"].get("file") != "FORECAST.json": raise ValueError("forecast marker filename differs")
    engine = study.SimulatedDispatchSlots(case)
    for branch in case["branches"]:
        actor = branch["actor"]
        request = take("request", stage=actor)
        if request is None: return result
        if request.get("request") != study.request(case, actor, result["planner"]):
            raise ValueError("actual buyer request differs from prepared input")
        response = take("response", stage=actor)
        if response is None: return result
        parsed = parse_response(response, lambda text: study.parse_buyer(case, actor, text))
        if parsed is invalid: return result
        command = parsed[1]
        proposal = take("parsed_proposal", actor=actor)
        if proposal is None: return result
        if proposal.get("command") != command: raise ValueError("actual parsed command differs from reply")
        event = take("runtime_event")
        if event is None: return result
        actual = engine.execute(actor, command)
        if event.get("event") != actual: raise ValueError("local simulated execution differs")
        events.append(actual)
    if cursor != len(activity): raise ValueError("unexpected records after completed buyer slots")
    if not complete: audit["unobserved_next_operation"] = {"kind": "outcome_record"}
    return result


def verify_flow(case, out, expected_hash):
    directory = out / "runs" / case["id"]; report = read(directory / "REPORT.json")
    if sha(directory / "REPORT.json") != expected_hash: raise ValueError("case report changed")
    if report.get("case_id") != case["id"]: raise ValueError("case identity differs")
    actual_files = {p.name for p in directory.glob("*.json") if p.name != "REPORT.json"}
    if set(report["files_sha256"]) != actual_files: raise ValueError("raw file inventory differs")
    for name, h in report["files_sha256"].items():
        if sha(directory / name) != h: raise ValueError("raw record/forecast changed")
    records = [read(p) for p in sorted(directory.glob("record_*.json"))]
    validate_record_order(records)
    protocol = audit_record_protocol(case, records, report)
    calls = protocol["protocol_audit"]["requests"]
    if type(report.get("api_calls")) is not int or calls != report["api_calls"] or not 0 <= calls <= 3:
        raise ValueError("request count differs")
    prefix, marker = protocol["prefix_record"], protocol["marker"]
    if prefix is not None and read(directory / "PREFIX.json") != prefix:
        raise ValueError("forecast prefix copy differs")
    if report["forecast_made"] != (marker is not None): raise ValueError("forecast commitment flag differs")
    forecast = None
    if marker is not None:
        if marker.get("sha256") != sha(directory / "FORECAST.json"): raise ValueError("forecast commitment differs")
        forecast = read(directory / "FORECAST.json")
        if score(prefix["value"], out / "models") != forecast:
            raise ValueError("frozen forecast does not reproduce exactly")
    if not report["complete"]:
        return {"case": case, "complete": False, "failure": report["failure_type"], "forecast": forecast,
                "verified_runtime_events": protocol["events"], "protocol_audit": protocol["protocol_audit"]}
    if calls != 3 or forecast is None: raise ValueError("incomplete request/forecast set")
    value = prefix["value"]
    label = study.outcome(case, protocol["events"])
    if read(directory / "OUTCOME.json") != label: raise ValueError("outcome differs from actual execution")
    sample = compile_attribute_input(value)
    return {"case": case, "complete": True, "forecast": forecast, "prefix": value, "sample": sample,
            "x": np.asarray(study.independent_features(value, sample)), "label": label,
            "protocol_audit": protocol["protocol_audit"]}


def metrics(rows, name):
    if not rows: return None
    risk, y, brier, logloss, correct, abstentions = [], [], [], [], 0, 0
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in rows:
        prediction = row["forecast"]["methods"][name]; p = prediction["risk_probability"]; label = row["label"]["risk"]
        risk.append(p); y.append(label)
        confusion[("t" if int(p >= .5) == label else "f") + ("p" if p >= .5 else "n")] += 1
        for slot in prediction["slots"]:
            chosen = [i + 1 for i, a in enumerate(slot["action_ids"]) if row["label"]["dispatch"][a]]
            chosen = chosen[0] if chosen else 0; abstentions += int(chosen == 0)
            q = slot["probabilities"]; correct += int(np.argmax(q) == chosen)
            brier.append(sum((v - int(i == chosen)) ** 2 for i, v in enumerate(q)))
            logloss.append(-math.log(q[chosen]) if q[chosen] > 0 else None)
    return {"flows": len(rows), "positive_flows": sum(y), "risk_brier": float(np.mean((np.asarray(risk) - y) ** 2)),
            **confusion, "slots": len(brier), "actual_abstentions": abstentions, "slot_correct": correct,
            "slot_brier": float(np.mean(brier)), "slot_log_loss": None if None in logloss else float(np.mean(logloss)),
            "infinite_slot_log_loss": None in logloss}


def verify(out):
    plan = load(out); collection = read(out / "COLLECTION.json")
    if collection["plan_sha256"] != sha(out / "PLAN.json"): raise ValueError("collection plan changed")
    rows = [verify_flow(case, out, collection["reports_sha256"][case["id"]]) for case in plan["cases"]]
    complete = [r for r in rows if r["complete"]]
    if len(complete) != collection["complete_flows"]: raise ValueError("completion count differs")
    methods = list(next((r["forecast"]["methods"] for r in rows if r["forecast"]), {}))
    if any(set(r["forecast"]["methods"]) != set(methods) for r in rows if r["forecast"]):
        raise ValueError("methods changed across flows")
    summaries = {name: {"all_complete": metrics(complete, name),
        "by_profile": {v: metrics([r for r in complete if r["case"]["profile"] == v], name) for v in study.PROFILES},
        "by_priority": {v: metrics([r for r in complete if r["prefix"]["priority"] == v], name) for v in ("economy", "balanced", "quality")}}
        for name in methods}
    evidence = {"rows": [{"id": r["case"]["id"], "profile": r["case"]["profile"], "domain": r["case"]["domain"],
                         "priority": r["prefix"]["priority"], "label": r["label"]} for r in complete]}
    audit = {"verified_forecasts": sum(r["forecast"] is not None for r in rows),
             "verified_prediction_vectors": sum(len(r["forecast"]["methods"]) for r in rows if r["forecast"]),
             "all_forecasts_precede_buyer_requests": True, "exact_replay": True,
             "graphs": sum(r["forecast"]["numerical_audit"]["graphs"] for r in rows if r["forecast"]),
             "max_graph_error": max([r["forecast"]["numerical_audit"]["max_graph_error"] for r in rows if r["forecast"]] or [0.])}
    support = support_diagnosis(complete) if complete else None
    if support is not None:
        # Existing helper names this train_flows; these are NOT used to train.
        support["complete_probe_flows"] = support.pop("train_flows")
        support["interpretation"] = "Observed new-catalogue design support only. No fitting. Rank does not identify causal tastes or establish calibration."
        # Old helper includes a proposed old-catalogue null direction. Its
        # residual here is diagnostic: a nonzero value means confounding broken.
    report = {"schema": "crossed-purchase-verified-v1", "finished_at_utc": now(),
              "plan_sha256": sha(out / "PLAN.json"), "collection_sha256": sha(out / "COLLECTION.json"),
              "planned_flows": len(plan["cases"]), "complete_flows": len(complete), "api_calls": collection["api_calls"],
              "failed_flows": [{"id": r["case"]["id"], "failure": r["failure"], "forecast_made": r["forecast"] is not None,
                                "protocol_audit": r["protocol_audit"], "verified_runtime_events": r["verified_runtime_events"]}
                               for r in rows if not r["complete"]],
              "metrics": summaries, "audit": audit, "observed_feature_support": support,
              "new_fits": 0, "new_default_selected": False,
              "limits": ["Small authored prospective transfer cohort, not representative deployment or native MAS superiority.",
                         "Prepared-input forecasts condition on valid completed command protocol; incomplete calls are not risk-negative examples.",
                         "No abstention elicitation, missing-warranty variation, uncertain future planner or new risk family in this cohort.",
                         "Neutral IDs and minimal descriptions change representation as well as price/warranty correlations; profiles are not isolated causal interventions.",
                         "Frozen model/rule comparators have the same future forecasting target, not a guard's current allow/deny target."]}
    write(out / "VERIFIED_OUTCOMES.json", evidence); write(out / "EVALUATION.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("metrics", "observed_feature_support")}), flush=True)
    for name, item in summaries.items(): print(json.dumps({"method": name, **(item["all_complete"] or {})}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    verify(p.parse_args().out.resolve())
