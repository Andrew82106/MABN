"""Task semantics -> local events AND message gates, on fixed development splits.

No new behavior generation, no old test access, no validation-based selection.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import shutil
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, write, sha, load_data, frozen_plan, baseline_features, metric
from run_forecast_alarm_study import now, bind_partition, freeze_calibration
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.task_field_semantics import context_from_initial, bind_task_cues, validate_annotation, request_payload
from paperalpha_runtime.semantic_evidence import response_annotation
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN, compile_message_training
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN, SemanticArrivalWorkflowRiskBN
from paperalpha_monitor.forecast_alarm import forecast_alarm_metrics


def prepare(out, parent, extraction, *, arrival=False):
    old, previous, ext = read(parent / "PLAN.json"), read(parent / "REPORT.json"), read(extraction / "REPORT.json")
    if sha(parent / "PLAN.json") != previous["plan_sha256"]: raise ValueError("parent experiment changed")
    if sha(extraction / "PLAN.json") != ext["plan_sha256"]: raise ValueError("extraction changed")
    for name, expected in ext["artifact_sha256"].items():
        if sha(extraction / name) != expected: raise ValueError("extraction artifact changed")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(extraction, out / "extraction")
    for name, expected in old["inputs_sha256"].items():
        if sha(parent / "inputs" / name) != expected: raise ValueError("development input changed")
        dest = out / "inputs" / name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(parent / "inputs" / name, dest)
    extra = {'code/commands/run_task_cue_message_study.py', 'code/commands/verify_task_cue_message_study.py',
        'code/commands/collect_task_field_semantics.py', 'code/tests/test_task_cue_message.py', 'code/tests/test_task_cue_study.py'}
    if arrival:
        extra |= {'code/tests/test_semantic_arrival.py', 'code/tests/test_task_cue_relational_gap.py', 'code/tests/test_semantic_arrival_study.py'}
    names = sorted(set(old["source_sha256"]) | extra | {p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")})
    for name in names:
        dest = out / "source" / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    plan = {**old, "schema": "task-cue-message-development-v1", "parent": str(parent),
        "parent_plan_sha256": sha(parent / "PLAN.json"), "extraction_report_sha256": sha(out / "extraction/REPORT.json"),
        "source_sha256": {name: sha(out / "source" / name) for name in names}, "created_at_utc": now(),
        "config": {**copy.deepcopy(old["config"]), "bn_views": ["counts", "local", "task"], "classifier_views": ["counts", "task"]},
        "feature_contract": "counts: 8 prefix counts; task: same counts + 72-bin histogram of (vault,coordinator,worker,work-cue,publish-cue). No IDs/domain/template/model inputs.",
        "bn_controls": "counts: 55 event coefficients with zero cues; local: old45 event coefficients and local cues only; task:55 with local and aggregate message cues. All have 3 unfitted known-authority priors.",
        "training": "Same fit/calibration/validation assignment as parent. Same future worker+final joint likelihood for BN. Task classifiers use final labels only; this supervision difference is disclosed.",
        "new_api_calls": 0, "reused_classifier_calls": ext["api_calls"], "old_test_read": False,
        "scope": "Previously exposed fixed-workflow development. Offline initial-only field/task hypotheses, not permission facts, new target behavior, semantic gold, automatic DAG or MAS-wide superiority. Report every preset variant; no new default."}
    if arrival:
        plan["schema"] = "semantic-arrival-message-development-v1"
        plan["config"].update(bn_views=["counts", "task", "arrival"], probability_digits=8)
        plan["bn_controls"] = "Counts55 with zero cues; static task55; paired semantic-arrival65. Same local25 coefficients, losses and three unfitted known-authority priors. New gate pairs each named field cue with its own upstream appearance state."
        plan["score_transform"] = "All ten scoring methods use round(raw probability,8) before calibration, ranking and alarms; raw probabilities also retained. This operational resolution is not applied to training likelihood/gradients. Chosen to dominate the previously audited ~1e-15 numerical tie spread, not selected from this run's outcomes."
        plan["numerics"] = "Current exact late complete-message OR identity used; same-feature score ties must survive at every phase. All models refit and all thresholds recalibrated; no previous score/threshold transplanted."
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "plan_sha256": sha(out / "PLAN.json"), "partition_counts": plan["partition_counts"]}), flush=True)


def transform_probability(value, digits=None):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite probability required before score transform")
    if digits is None: return value
    if type(digits) is not int or not 1 <= digits <= 12:
        raise ValueError("explicit decimal probability precision in 1..12 required")
    return round(value, digits)


def score_artifact(artifact, config):
    if config.get("probability_digits") is not None:
        artifact["score_transform"] = {"mode": "round_nearest_even_decimal", "probability_digits": config["probability_digits"]}
    return artifact


def task_features(row, phase):
    """Complete joint histogram of available field inputs; early W is padding.

    Phase-specific estimators never interpret this padding as observed absence.
    """
    if phase not in ("early", "late"): raise ValueError("fixed phase required")
    sample = row[phase]; dag = sample.event_dag
    histogram = np.zeros((2, 2, 2, 3, 3))
    def category(cue): return 1 if cue == [1, 0] else 2 if cue == [0, 1] else 0
    for j in range(len(sample.action_ids)):
        stem = f"value-{j}:"
        v, c = (dag.observations[stem + p]["value"] for p in ("vault", "prepare"))
        w = dag.observations[stem + "work"]["value"] if phase == "late" else 0
        a, b = (category(dag.lookup[stem + p]["cues"]) for p in ("work", "publish"))
        histogram[v, c, w, a, b] += 1
    return np.r_[baseline_features(row, phase), histogram.ravel()]


def verify_request_replay(context, payload):
    # The recorded JSON string may have canonical key order from disk. Compare
    # its exact parsed content while keeping all API settings/instructions fixed.
    expected = request_payload(context)
    if {k: v for k, v in payload.items() if k != "input"} != {k: v for k, v in expected.items() if k != "input"}:
        raise ValueError("different API settings")
    entries = payload.get("input")
    if not isinstance(entries, list) or len(entries) != 1 or set(entries[0]) != {"role", "content"} or entries[0]["role"] != "user":
        raise ValueError("different API input envelope")
    if json.loads(entries[0]["content"]) != context:
        raise ValueError("different API input content")


def load_views(out, data):
    ext = out / "extraction"; report = read(ext / "REPORT.json"); plan = read(out / "PLAN.json")
    if sha(ext / "REPORT.json") != plan["extraction_report_sha256"]: raise ValueError("extraction report changed")
    for name, expected in report["artifact_sha256"].items():
        if sha(ext / name) != expected: raise ValueError("extraction material changed")
    bindings = {r["id"]: r for r in read(ext / "SOURCE_BINDINGS.json")}
    if set(bindings) != {r["id"] for r in data}: raise ValueError("complete initial source binding required")
    tasks, projected = {}, []
    for row in data:
        binding = bindings[row["id"]]; raw_path = Path(plan["upstream"]) / binding["source_path"]
        if sha(raw_path) != binding["source_sha256"]: raise ValueError("raw task source changed")
        initial = read(raw_path)["input"]
        context, rebuilt = context_from_initial(initial["request"], initial["vault"])
        if rebuilt != {k: binding[k] for k in rebuilt}: raise ValueError("initial-only normalization changed")
        name = digest(context) + ".json"
        if read(ext / "contexts" / name) != context: raise ValueError("context mismatch")
        classification = read(ext / "classifications" / name)
        if classification["context_sha256"] != digest(context): raise ValueError("hypotheses from wrong context")
        annotation = validate_annotation(context, classification["annotation"])
        verify_request_replay(context, read(ext / "requests" / name))
        if classification["status"] == "quote_anchored_hypothesis" and response_annotation(read(ext / "responses" / name)) != annotation:
            raise ValueError("stored categories differ from actual API output")
        task = {**row, **{p: bind_task_cues(row[p], context, annotation) for p in ("early", "late")}}
        tasks[row["id"]] = task
        projected.append({"id": row["id"], "context_sha256": digest(context), "hypothesis_status": classification["status"],
            "early_sha256": task["early"].sha256, "late_sha256": task["late"].sha256})
    return {"counts": {r["id"]: r for r in data}, "local": tasks, "task": tasks, "arrival": tasks}, projected


def gate_ranks(model, rows):
    batches, _ = compile_message_training(model.event_law, [r["early"].event_dag for r in rows], [r["labels"] for r in rows])
    ranks = {}
    for offset in sorted({b["offset"] for b in batches}):
        x = np.concatenate([b["g"] for b in batches if b["offset"] == offset])
        ranks[model.names[offset].split(":")[1]] = {"width": x.shape[1], "rank": int(np.linalg.matrix_rank(x)),
            "messages": len(x), "unique_inputs": len(np.unique(x, axis=0))}
    return ranks


def run(out):
    plan = frozen_plan(out); config = plan["config"]
    data, projection = load_data(out); write(out / "PROJECTION.json", projection)
    views, semantic = load_views(out, data); write(out / "TASK_PROJECTION.json", semantic)
    parts = bind_partition(data, plan); models, artifacts = {}, {}
    fit = {"fit_ids": [r["id"] for r in parts["fit"]], "fit_templates": sorted({r["template_id"] for r in parts["fit"]}), "models": {}}
    for view in config["bn_views"]:
        rows = [views[view][r["id"]] for r in parts["fit"]]
        for penalty in config["regularization"]:
            name = f"{view}_bn_{penalty}"; start = time.monotonic()
            cls = MessageHurdleWorkflowRiskBN if view == "local" else SemanticArrivalWorkflowRiskBN if view == "arrival" else TaskCueMessageWorkflowRiskBN
            model = cls(regularization=penalty).fit_messages([r["early"].event_dag for r in rows], [r["labels"] for r in rows])
            models[name] = (view, model); artifacts[name] = score_artifact(model.parameters(), config)
            write(out / "models" / (name + ".json"), artifacts[name])
            fit["models"][name] = {"view": view, "kind": "bn", "fingerprint": digest(artifacts[name]),
                "optimizer": model.optimizer, "gate_design_ranks": gate_ranks(model, rows)}
            print(json.dumps({"fitted": name, "success": model.optimizer["success"], "seconds": time.monotonic() - start}), flush=True)
    for view in config["classifier_views"]:
        rows = [views[view][r["id"]] for r in parts["fit"]]
        feature = baseline_features if view == "counts" else task_features
        settings = config["baseline"]
        for kind in ("logistic", "histgb"):
            name = view + "_" + kind; estimators, files = {}, {}
            for phase in config["phases"]:
                x = np.array([feature(r, phase) for r in rows]); y = np.array([r["y"] for r in rows])
                estimator = (LogisticRegression(C=settings["logistic_C"], max_iter=2000, tol=1e-10) if kind == "logistic" else
                    HistGradientBoostingClassifier(max_leaf_nodes=settings["histgb_leaves"], max_iter=settings["histgb_iterations"],
                        l2_regularization=settings["histgb_l2"], random_state=settings["random_state"]))
                estimator.fit(x, y); estimators[phase] = estimator
                filename = name + "_" + phase + ".joblib"; joblib.dump(estimator, out / "models" / filename); files[filename] = sha(out / "models" / filename)
            models[name] = (view, estimators)
            artifacts[name] = score_artifact({"model": kind, "view": view, "feature_width": x.shape[1], "files_sha256": files}, config)
            write(out / "models" / (name + ".json"), artifacts[name])
            fit["models"][name] = {"view": view, "kind": kind, "fingerprint": digest(artifacts[name])}
    fit["saved_at_utc"] = now(); fit["model_file_sha256"] = {p.name: sha(p) for p in (out / "models").iterdir()}
    write(out / "FIT.json", fit)

    def score(partition):
        scores = {}
        for name, (view, model) in models.items():
            rows = [views[view][r["id"]] for r in parts[partition]]; records, raw_records = [], []
            feature = baseline_features if view == "counts" else task_features
            p_arrays = {p: model[p].predict_proba(np.array([feature(r, p) for r in rows]))[:, 1] for p in config["phases"]} if isinstance(model, dict) else None
            for i, row in enumerate(rows):
                prefixes, raw_prefixes = [], []
                for phase in config["phases"]:
                    sample = row[phase]
                    raw_p = float(p_arrays[phase][i]) if p_arrays is not None else model.predict_known_authorization_union(sample)["probabilities"]["system:any"]
                    prefix = {"phase": phase, "sequence": sample.event_dag.prefix_sequence, "input_sha256": sample.sha256,
                        "probability": transform_probability(raw_p, config.get("probability_digits"))}
                    prefixes.append(prefix); raw_prefixes.append({**prefix, "probability": raw_p})
                episode = {"episode_id": row["id"], "model_fingerprint": digest(artifacts[name]), "target_id": config["target_id"], "prefixes": prefixes}
                records.append(episode); raw_records.append({**episode, "prefixes": raw_prefixes})
            write(out / "predictions" / partition / (name + ".json"), records); scores[name] = records
            if config.get("probability_digits") is not None:
                write(out / "raw_predictions" / partition / (name + ".json"), raw_records)
            print(json.dumps({"scored": name, "partition": partition, "flows": len(rows)}), flush=True)
        return scores

    cal = score("calibration")
    calibration = freeze_calibration(parts["calibration"], cal, artifacts, config, out)
    cal_sha = sha(out / "CALIBRATION.json")
    write(out / "VALIDATION_START.json", {"calibration_sha256": cal_sha, "started_at_utc": now(), "models_frozen": fit["model_file_sha256"]})
    val = score("validation")
    report = {"plan_sha256": sha(out / "PLAN.json"), "calibration_sha256": cal_sha, "metrics": {}, "phase_probability_metrics": {}, "template_metrics": {},
        "partition_counts": {p: {"flows": len(rs), "positive": sum(r["y"] for r in rs)} for p, rs in parts.items()},
        "new_api_calls": 0, "reused_classifier_calls": plan["reused_classifier_calls"], "new_target_agent_calls": 0,
        "old_test_read": False, "new_default_selected": False, "scope": plan["scope"]}
    for partition, scores in (("calibration", cal), ("validation", val)):
        labels = {r["id"]: r["y"] for r in parts[partition]}; templates = {r["id"]: r["template_id"] for r in parts[partition]}
        for name, episodes in scores.items():
            key = partition + "/" + name; report["metrics"][key] = {}; decisions = {}
            report["phase_probability_metrics"][key] = {p: metric([labels[e["episode_id"]] for e in episodes],
                [next(s["probability"] for s in e["prefixes"] if s["phase"] == p) for e in episodes]) for p in config["phases"]}
            for rule, candidate in calibration["models"][name].items():
                summary, actual = forecast_alarm_metrics(episodes, labels, candidate["policy"])
                report["metrics"][key][rule] = summary
                decisions[rule] = [{"episode_id": e["episode_id"], "decisions": d} for e, d in zip(episodes, actual)]
                report["template_metrics"][key + "/" + rule] = {t: forecast_alarm_metrics([e for e in episodes if templates[e["episode_id"]] == t],
                    {i: y for i, y in labels.items() if templates[i] == t}, candidate["policy"])[0] for t in sorted(set(templates.values()))}
            write(out / "decisions" / partition / (name + ".json"), decisions)
    if sha(out / "CALIBRATION.json") != cal_sha: raise ValueError("calibration changed")
    for name, expected in fit["model_file_sha256"].items():
        if sha(out / "models" / name) != expected: raise ValueError("model changed after fit")
    report["prediction_sha256"] = {str(p.relative_to(out)): sha(p) for p in (out / "predictions").rglob("*.json")}
    report["decision_sha256"] = {str(p.relative_to(out)): sha(p) for p in (out / "decisions").rglob("*.json")}
    if config.get("probability_digits") is not None:
        report["raw_prediction_sha256"] = {str(p.relative_to(out)): sha(p) for p in (out / "raw_predictions").rglob("*.json")}
        report["probability_digits"] = config["probability_digits"]
    report["completed_at_utc"] = now(); write(out / "REPORT.json", report)
    print(json.dumps({"completed": True, "scoring_methods": len(models)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--parent", type=Path); parser.add_argument("--extraction", type=Path); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrival", action="store_true", help="Prepare paired-arrival study with fixed 8-decimal operational scores")
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.out.resolve(), args.parent.resolve(), args.extraction.resolve(), arrival=args.arrival)
    else: run(args.out.resolve())
