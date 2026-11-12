"""Frozen, retrospective diagnostic of an already-exposed API cohort.

No collection, fitting, threshold selection, or replacement of old scores.
"""
from pathlib import Path
import argparse
import json
import math
from numbers import Real
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read, write, sha, now
from verify_crossed_purchase import verify_flow
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN
from paperalpha_monitor.dependence_envelope import dependence_envelope

EXTRA = ['code/src/paperalpha_monitor/dependence_envelope.py', 'code/tests/test_dependence_envelope.py',
         'code/commands/diagnose_dependence_envelope.py']


def prepare(source, out):
    old = read(source / "PLAN.json")
    # Changes to old scoring logic must be explicit, never silently recomputed.
    for name, h in old["source_sha256"].items():
        if sha(source / "source" / name) != h or sha(ROOT / name) != h:
            raise ValueError("original implementation differs: " + name)
    names = sorted(set(old["source_sha256"]) | set(EXTRA))
    out.mkdir(parents=True, exist_ok=False)
    for name in names:
        p = out / "source" / name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    plan = {"schema": "dependence-envelope-diagnostic-plan-v1", "created_at_utc": now(),
            "source_run": str(source), "source_sha256": {n: sha(out / "source" / n) for n in names},
            "source_artifacts": {n: sha(source / n) for n in ("PLAN.json", "COLLECTION.json",
                "EVALUATION.json", "VERIFIED_OUTCOMES.json", "TEST_JUNIT.xml")},
            "methods": ["attribute_bn_0.001", "attribute_bn_0.03"],
            "queries": ["system:any", "channel:authorization", "channel:single_bound",
                        "channel:cumulative_budget", "channel:prohibited_combination"],
            "status": "Post-outcome diagnostic of exposed cohort. Not a new prospective evaluation.",
            "protocol": ["No API calls, no new fit, no default or old score changes.",
                         "Preserve slot marginals and joint authority marginal; relax other dependence.",
                         "Report all flows and both frozen regularization settings.",
                         "Do not interpret probability envelopes as intervals covering binary outcomes.",
                         "Marginal prediction error is not repaired by dependence bounds."]}
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "plan_sha256": sha(out / "PLAN.json")}), flush=True)


def inspect_witness(report):
    """Separate scalar certificate check, no solver or sparse matrix helpers."""
    # Do not let NaN comparisons or zip's shortest-input behavior turn a
    # malformed witness into an apparently successful numerical certificate.
    tolerance = 1e-8

    def number(value):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("finite scalar witness values required")
        return float(value)

    def vector(value, length):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError("witness vector dimension mismatch")
        return [number(v) for v in value]

    if not isinstance(report, dict) or not isinstance(report.get("witness_problem"), dict):
        raise ValueError("explicit witness problem required")
    p = report["witness_problem"]
    states, blocks = p.get("states"), p.get("blocks")
    if (not isinstance(blocks, list) or not blocks
            or any(not isinstance(b, str) or not b for b in blocks)
            or len(set(blocks)) != len(blocks)):
        raise ValueError("unique named witness blocks required")
    if (not isinstance(states, list) or not states
            or any(not isinstance(s, list) or len(s) != len(blocks)
                   or any(type(v) is not int or v < 0 for v in s) for s in states)
            or len({tuple(s) for s in states}) != len(states)):
        raise ValueError("unique aligned nonnegative integer worlds required")
    reference = vector(p.get("reference_mass"), len(states))
    if min(reference) < 0. or abs(sum(reference) - 1.) > 1e-12:
        raise ValueError("normalized nonnegative reference mass required")
    events, queries = p.get("events"), report.get("queries")
    if (not isinstance(events, dict) or not events or not isinstance(queries, dict)
            or set(events) != set(queries)
            or any(not isinstance(q, str) or not q for q in events)):
        raise ValueError("matching named witness events and queries required")
    rows = [[1.] * len(states)]
    constraints = [{"block": "normalization", "category": None}]
    for j, block in enumerate(blocks):
        for cat in sorted(set(s[j] for s in states)):
            rows.append([float(s[j] == cat) for s in states])
            constraints.append({"block": block, "category": cat})
    marginal = [sum(a*b for a, b in zip(r, reference)) for r in rows]
    declared_marginal = vector(p.get("marginals"), len(rows))
    if p.get("constraint_rows") != constraints:
        raise ValueError("witness constraint rows do not match state categories")
    if max(abs(a-b) for a, b in zip(marginal, declared_marginal)) > tolerance:
        raise ValueError("declared witness marginals differ from reference")
    max_error = 0.
    for query, row in queries.items():
        truth = vector(events[query], len(states))
        if min(truth) < 0. or max(truth) > 1.:
            raise ValueError("bounded world-aligned risk values required")
        if not isinstance(row, dict):
            raise ValueError("query certificate object required")
        lower, upper, point, width = (number(row.get(k)) for k in ("lower", "upper", "reference", "width"))
        expected_point = sum(a*b for a, b in zip(truth, reference))
        if (lower < -tolerance or upper > 1. + tolerance or lower > upper + tolerance
                or not lower - tolerance <= point <= upper + tolerance
                or abs(point - expected_point) > tolerance or abs(width - (upper-lower)) > tolerance):
            raise ValueError("inconsistent witness query bounds or reference")
        witnesses = row.get("endpoint_witnesses")
        if not isinstance(witnesses, list) or len(witnesses) != 2:
            raise ValueError("exactly two endpoint witnesses required")
        for sign, end, witness in zip((1, -1), ("lower", "upper"), row["endpoint_witnesses"]):
            if not isinstance(witness, dict):
                raise ValueError("endpoint certificate object required")
            mass = vector(witness.get("mass"), len(states))
            dual = vector(witness.get("equality_dual"), len(rows))
            error = max(abs(sum(a*b for a, b in zip(r, mass)) - v) for r, v in zip(rows, marginal))
            objective = number(sum(a*b for a, b in zip(truth, mass)))
            dual_objective = number(sum(a*b for a, b in zip(marginal, dual)))
            dual_violation = max(number(sum(r[i]*d for r, d in zip(rows, dual))-sign*truth[i]) for i in range(len(states)))
            error = max(error, -min(mass), dual_violation, abs(objective-row[end]),
                        abs(sign*objective-dual_objective))
            if not math.isfinite(error) or error > tolerance:
                raise ValueError("independent primal/dual certificate failed")
            max_error = max(max_error, error)
    return max_error


def run(out):
    plan = read(out / "PLAN.json"); source = Path(plan["source_run"])
    if ROOT != (out / "source").resolve(): raise ValueError("run frozen source")
    for n, h in plan["source_sha256"].items():
        if sha(ROOT / n) != h: raise ValueError("diagnostic source changed")
    for n, h in plan["source_artifacts"].items():
        if sha(source / n) != h: raise ValueError("source artifact changed")
    original = read(source / "PLAN.json"); collection = read(source / "COLLECTION.json")
    for n, h in original["model_sha256"].items():
        if sha(source / "models" / n) != h: raise ValueError("frozen model changed")
    models = {n: AttributeChoiceWorkflowRiskBN.restore(read(source / "models" / (n+".json"))) for n in plan["methods"]}
    rows, max_error = [], 0.
    for case in original["cases"]:
        r = verify_flow(case, source, collection["reports_sha256"][case["id"]])
        if not r["complete"]: raise ValueError("diagnostic expects original complete cohort")
        for name, model in models.items():
            sample = r["sample"]
            report = dependence_envelope(model, sample, queries=plan["queries"], include_witness=True)
            max_error = max(max_error, inspect_witness(report))
            prediction = r["forecast"]["methods"][name]
            if abs(report["queries"]["system:any"]["reference"] - prediction["risk_probability"]) > 1e-12:
                raise ValueError("new component changed original point forecast")
            slots = []
            for g in prediction["slots"]:
                actual = next((i+1 for i, a in enumerate(g["action_ids"]) if r["label"]["dispatch"][a]), 0)
                predicted = int(np.argmax(g["probabilities"]))
                classes = report["queries"]["system:any"]["policy_equivalent_categories"]["dispatch:"+g["id"]]
                equivalent = any(actual in cls and predicted in cls for cls in classes)
                slots.append({"id": g["id"], "actual": actual, "predicted": predicted,
                              "correct": actual == predicted, "policy_equivalent": equivalent,
                              "categories": [None] + g["action_ids"]})
            path = Path("envelopes") / (case["id"]+"__"+name+".json")
            (out / path).parent.mkdir(parents=True, exist_ok=True); write(out / path, report)
            rows.append({"id": case["id"], "profile": case["profile"], "priority": r["prefix"]["priority"],
                         "method": name, "label": r["label"]["risk"], "slots": slots,
                         "system_risk": {k: report["queries"]["system:any"][k] for k in ("lower", "reference", "upper", "width")},
                         "envelope_file": path.as_posix(), "envelope_sha256": sha(out / path)})
    summaries = {}
    for name in models:
        selected = [r for r in rows if r["method"] == name]
        slots = [s for r in selected for s in r["slots"]]
        summaries[name] = {"flows": len(selected), "slots": len(slots), "slot_errors": sum(not s["correct"] for s in slots),
            "errors_between_policy_equivalent_choices": sum(not s["correct"] and s["policy_equivalent"] for s in slots),
            "mean_dependence_width": float(np.mean([r["system_risk"]["width"] for r in selected])),
            "max_dependence_width": max(r["system_risk"]["width"] for r in selected),
            "intervals_straddling_half": sum(r["system_risk"]["lower"] < .5 <= r["system_risk"]["upper"] for r in selected),
            "by_profile": {p: {"slot_errors": sum(not s["correct"] for r in selected if r["profile"] == p for s in r["slots"]),
                "mean_dependence_width": float(np.mean([r["system_risk"]["width"] for r in selected if r["profile"] == p]))}
                for p in sorted(set(r["profile"] for r in selected))}}
    junit = ET.parse(source / "TEST_JUNIT.xml").getroot()
    suites = list(junit.iter("testsuite"))
    receipt = {"source": str(source / "TEST_JUNIT.xml"), "sha256": sha(source / "TEST_JUNIT.xml"),
               "tests": sum(int(s.attrib["tests"]) for s in suites),
               "failures": sum(int(s.attrib["failures"]) for s in suites),
               "errors": sum(int(s.attrib["errors"]) for s in suites),
               "scope": "Historical suite recorded on September 20, not a new test execution."}
    report = {"schema": "dependence-envelope-diagnostic-v1", "finished_at_utc": now(),
              "plan_sha256": sha(out / "PLAN.json"), "summaries": summaries, "rows": rows,
              "audit": {"envelopes": len(rows), "queries": len(rows)*len(plan["queries"]),
                        "lp_endpoints": 2*len(rows)*len(plan["queries"]), "max_independent_certificate_error": max_error,
                        "original_point_forecasts_unchanged": True, "original_flows_reverified": len(original["cases"])},
              "historical_test_receipt": receipt, "new_api_calls": 0, "new_fits": 0,
              "limitations": plan["protocol"], "prospective_validation": False}
    write(out / "REPORT.json", report)
    print(json.dumps({"summaries": summaries, "audit": report["audit"], "historical_tests": receipt}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--source", type=Path, required=True); prep.add_argument("--out", type=Path, required=True)
    runner = sub.add_parser("run"); runner.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare": prepare(args.source.resolve(), args.out.resolve())
    else: run(args.out.resolve())
