"""One frozen, grouped DEVELOPMENT comparison; never an untouched test.

No API calls. The target is final C1 literal disclosure, not attack identity or
the first disclosure in the system. Future labels stay outside every predictor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.agentleak_submission import monitor_prefix
from paperalpha_monitor.submission_projection import project_prefix, future_labels
from paperalpha_monitor.relational_event_law import RelationalEventLaw
from paperalpha_monitor.event_elimination import EliminatingEventLaw, EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import fit_event_traces
from paperalpha_monitor.development_baselines import ConditionalFactorBaseline, target_vector_log_probability
from collect_submission_semantics import annotation_for_case, verify as verify_semantics, collection_lock as semantics_lock
from collect_submission_agentleak import verify as verify_collection, case_trace, collection_lock as behavior_lock

DEFAULT_COLLECTION = ROOT / 'experiments/privacy_flow/runs/submission__agentleak'
METHODS = ("relational_bn", "no_object_links_bn", "no_task_cues_bn", "mean_parent_bn", "matched_glm", "matched_histgb")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(collection, output):
    parent, receipts = read(collection / "PLAN.json"), read(collection / "COLLECTION.json")
    semantics = read(collection / "semantics/COLLECTION.json")
    if semantics["api_attempts"] != 16 or semantics["statuses"].get("not_started", 0):
        raise ValueError("all frozen semantic attempts must have finished before analysis freeze")
    if verify_collection(collection) != receipts or verify_semantics(collection / "semantics") != semantics:
        raise ValueError("collection summaries do not match immutable API receipts")
    for case in parent["cases"]:
        if case_trace(collection, case, parent) != read(collection / "cases" / case["case_id"] / "TRACE.json"):
            raise ValueError("stored trace differs from immutable stage receipts")
    if receipts["completed_cases"] != len(parent["cases"]) or receipts["response_identity_exclusions"]:
        raise ValueError("complete verified development collection required for this analysis")
    if output.exists():
        raise ValueError("one immutable development analysis; existing output will not be overwritten")
    sources = sorted({p.relative_to(ROOT).as_posix() for directory in ('code/src/paperalpha_monitor', 'code/src/paperalpha_runtime', "knowledge")
                      for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".json"}} |
                     {'code/commands/evaluate_submission_development.py', 'code/commands/collect_submission_semantics.py',
                      'code/commands/collect_submission_agentleak.py', 'code/tests/test_submission_development_evaluation.py',
                      'code/tests/test_submission_projection.py', 'code/tests/test_development_baselines.py', 'knowledge/risks/risk_catalog.json'})
    inputs = ["PLAN.json", "COLLECTION.json", "semantics/PLAN.json", "semantics/COLLECTION.json"]
    inputs += [p.relative_to(collection).as_posix() for directory in ("cases", "semantics/contexts", "semantics/classifications", "semantics/responses", "semantics/attempts", "semantics/failures")
               for p in (collection / directory).rglob("*.json") if p.is_file()]
    output.mkdir(parents=True)
    for name in sources:
        target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    plan = {"schema": "submission-grouped-development-evaluation", "development_only": True,
        "collection": str(collection.resolve()), "collection_plan_digest": parent["sha256"],
        "input_sha256": {name: sha(collection / name) for name in sorted(set(inputs))},
        "source_sha256": {name: sha(output / "source" / name) for name in sources},
        "methods": list(METHODS), "regularization": .001, "max_iter": 800,
        "environment": {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}, "numerical_threads": 1},
        "folds": [{"held_out": c["template_id"], "fit_groups": sorted(x["template_id"] for x in parent["cases"] if x["template_id"] != c["template_id"])} for c in parent["cases"]],
        "target": "final C1 literal-field-policy violation; finalizer output is never a predictor input",
        "phases": ["P0_setup", "P1_after_coordinator", "P2_after_worker"],
        "supervision": "One earliest-prefix joint future-event likelihood per fit task. Intermediate observed labels used in fitting only. No repeated-prefix training.",
        "annotation_timing": "Offline initial-input-only features with real extraction receipts; not a prospective online latency evaluation.",
        "metrics": ["system_brier", "system_log_loss", "event_brier", "joint_log_loss_per_target", "joint_brier"],
        "aggregation": "Compute per-task scores, then mean over held-out template groups at each phase. Alias fields are one literal event, not independent test samples.",
        "numeric_floors": {"system_probability": 1e-15, "joint_probability": 1e-300},
        "alarms": "Not evaluated: only two final-negative templates, insufficient independent fit/calibration/test support for a low-FPR claim.",
        "ablations_are_not_external_baselines": True,
        "matched_glm": "Expected to equal logistic BN factors; numerical attribution control, not an algorithm to beat by renaming.",
        "native_mas": "BlindGuard original scope needs all first-round node replies and attacker labels; neither early-warning comparison nor native attack benchmark is claimed on this A0 chain.",
        "semantic_collection_status": semantics.get("schema"), "old_holdout_read": False, "new_api_calls": 0}
    write_new(output / "PLAN.json", plan)
    return {"prepared": str(output), "plan_sha256": sha(output / "PLAN.json"), "folds": len(plan["folds"])}


def verify(output):
    plan = read(output / "PLAN.json"); collection = Path(plan["collection"])
    for name, expected in plan["input_sha256"].items():
        if sha(collection / name) != expected:
            raise ValueError("frozen input changed: " + name)
    for name, expected in plan["source_sha256"].items():
        if sha(output / "source" / name) != expected or sha(ROOT / name) != expected:
            raise ValueError("frozen analysis source changed: " + name)
    return plan


def load_rows(collection):
    parent = read(collection / "PLAN.json")
    rows = []
    for case in parent["cases"]:
        trace = read(collection / "cases" / case["case_id"] / "TRACE.json")
        if trace["status"] != "completed":
            raise ValueError("incomplete trace cannot enter complete supervision")
        annotation = annotation_for_case(collection / "semantics", case["case_id"])
        complete = monitor_prefix(case, trace["stages"])
        views = {}
        for name, refs, cues in (("full", None, annotation), ("no_objects", {}, annotation), ("no_cues", None, None)):
            samples = [project_prefix(monitor_prefix(case, trace["stages"][:k]), reference_fields=refs, annotation=cues) for k in range(3)]
            labels = future_labels(samples[0], complete)
            views[name] = {"samples": samples, "labels": labels}
        targets = [q["event_id"] for q in views["full"]["samples"][0].queries.values()]
        outcome = {n: views["full"]["labels"][n]["value"] for n in targets}
        if int(any(outcome.values())) != trace["labels"]["final_literal_policy_violation"]:
            raise ValueError("independent adapter and author policy oracle disagree")
        rows.append({"id": case["case_id"], "group": case["template_id"], "views": views,
            "targets": targets, "outcome": outcome, "y": int(any(outcome.values())),
            "annotation_available": annotation is not None})
    return rows


def score(prediction, outcomes):
    p, y = prediction["union_probability"], int(any(outcomes.values()))
    q = np.clip(p, 1e-15, 1-1e-15)
    truth_log = target_vector_log_probability(prediction, outcomes)
    squared_norm = float(np.exp(sum(float(logsumexp(2*np.asarray(t["log_probabilities"]))) for t in prediction["components"])))
    marginals = prediction["target_marginals"]
    return {"system_brier": float((p-y)**2), "system_log_loss": float(-y*np.log(q)-(1-y)*np.log1p(-q)),
        "event_brier": float(np.mean([(marginals[n]-bit)**2 for n, bit in outcomes.items()])),
        "joint_log_loss_per_target": float(-max(truth_log, np.log(1e-300))/len(outcomes)),
        "joint_brier": float(1+squared_norm-2*np.exp(truth_log)),
        "zero_probability_target_vector": not np.isfinite(truth_log)}


def run(output):
    plan = verify(output)
    write_new(output / "START.json", {"plan_sha256": sha(output / "PLAN.json"), "monotonic_start": time.monotonic()})
    rows = load_rows(Path(plan["collection"]))
    predictions, fits, audit = [], [], []
    for held in plan["folds"]:
        train = [r for r in rows if r["group"] in held["fit_groups"]]
        test = [r for r in rows if r["group"] == held["held_out"]]
        if len(test) != 1 or len(train) != len(rows)-1:
            raise ValueError("frozen template holdout mismatch")
        case = test[0]
        per_method = {}
        for name in METHODS:
            view = "no_objects" if name == "no_object_links_bn" else "no_cues" if name == "no_task_cues_bn" else "full"
            law = EliminatingEventLaw() if name == "mean_parent_bn" else RelationalEventLaw()
            dags = [r["views"][view]["samples"][0].event_dag for r in train]
            labels = [r["views"][view]["labels"] for r in train]
            start = time.perf_counter()
            native = None
            if name.endswith("_bn"):
                factory = lambda **options: EliminatingDAGWorkflowRiskBN(event_law=law, **options)
                native = fit_event_traces(dags, labels, regularization=plan["regularization"], max_iter=plan["max_iter"], model_class=factory)
                if not native.optimizer["success"]:
                    raise ValueError("BN optimizer failed; do not hide failure: " + name)
                estimator = ConditionalFactorBaseline(law.names, law.node_design).set_coefficients(native.theta[:native.event_count])
                metadata = native.parameters()
            else:
                estimator = ConditionalFactorBaseline(law.names, law.node_design,
                    kind="histgb" if name == "matched_histgb" else "glm", regularization=plan["regularization"])
                estimator.fit(dags, labels, task_ids=[r["id"] for r in train], group_ids=[r["group"] for r in train],
                    earliest_prefix_sequences={r["id"]: r["views"][view]["samples"][0].event_dag.prefix_sequence for r in train}, max_iter=plan["max_iter"])
                metadata = estimator.metadata()
            fit_seconds = time.perf_counter()-start
            destination = output / "models" / case["id"]
            write_new(destination / (name+".json"), metadata)
            if estimator.tree is not None:
                joblib.dump(estimator.tree, destination / (name+".joblib"))
            fits.append({"held_out": case["group"], "method": name, "seconds": fit_seconds,
                         "fit_groups": held["fit_groups"], "model_sha256": sha(destination/(name+".json"))})
            per_method[name] = []
            for phase, sample in enumerate(case["views"][view]["samples"]):
                start = time.perf_counter()
                predicted = estimator.predict(sample.event_dag, case["targets"])
                seconds = time.perf_counter()-start
                if native is not None:
                    native_start = time.perf_counter()
                    exact = native.predict_known_authorization_union(sample)["probabilities"]["system:any"]
                    native_seconds = time.perf_counter()-native_start
                    discrepancy = abs(exact-predicted["union_probability"])
                    if discrepancy > 1e-8:
                        raise ValueError("native VE and independent component enumeration disagree")
                    audit.append({"case": case["id"], "method": name, "phase": phase, "union_error": discrepancy})
                else:
                    native_seconds = None
                row = {"case": case["id"], "group": case["group"], "method": name, "phase": phase,
                    "input_sha256": sample.sha256, "y": case["y"], "probability": predicted["union_probability"],
                    "marginals": predicted["target_marginals"], "scores": score(predicted, case["outcome"]),
                    "component_inference_seconds": seconds, "native_inference_seconds": native_seconds}
                predictions.append(row); per_method[name].append(row)
        for phase in range(3):
            a, b = per_method["relational_bn"][phase], per_method["matched_glm"][phase]
            discrepancy = max([abs(a["probability"]-b["probability"]), *(abs(a["marginals"][n]-b["marginals"][n]) for n in a["marginals"])])
            if discrepancy > 1e-6:
                raise ValueError("same-factor GLM/BN equivalence failed")
            audit.append({"case": case["id"], "phase": phase, "matched_glm_max_error": discrepancy})
        print(json.dumps({"completed_development_fold": case["group"], "methods": len(METHODS)}), flush=True)
    summaries = []
    for name in METHODS:
        for phase in range(3):
            chosen = [r for r in predictions if r["method"] == name and r["phase"] == phase]
            summaries.append({"method": name, "phase": phase, "template_groups": len(chosen),
                "scores": {key: float(np.mean([r["scores"][key] for r in chosen])) for key in plan["metrics"]},
                "zero_probability_vectors": sum(r["scores"]["zero_probability_target_vector"] for r in chosen)})
    verify(output)
    write_new(output / "REPORT.json", {"schema": plan["schema"], "plan_sha256": sha(output/"PLAN.json"),
        "development_only": True, "new_api_calls": 0, "old_holdout_read": False, "model_performance_confirmed": False,
        "cases": len(rows), "annotation_available_cases": sum(r["annotation_available"] for r in rows),
        "positive_templates": sum(r["y"] for r in rows), "predictions": predictions, "fits": fits,
        "numerical_audit": audit, "summaries": summaries,
        "limits": ["Development structure selection only; no independent confirmation set.",
            "Cross-literal residual dependence is not modelled by these event factors.",
            "Matched GLM is mathematically the same factorization, not a method that a BN name should outperform.",
            "No native malicious-agent benchmark, prospective alarm, low-FPR guarantee or journal readiness is established."]})
    return {"completed": str(output), "report_sha256": sha(output/"REPORT.json"), "predictions": len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); output = args.output or args.collection / "evaluation"
    with threadpool_limits(limits=1):
        if args.command == "prepare":
            # Refuse a concurrent writer rather than freeze a mixed collection.
            with behavior_lock(args.collection), semantics_lock(args.collection / "semantics"):
                result = prepare(args.collection, output)
        else:
            result = run(output) if args.command == "run" else {"verified": bool(verify(output))}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
