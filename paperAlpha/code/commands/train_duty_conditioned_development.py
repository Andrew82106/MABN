"""Frozen v9 feasibility comparison on the same two known development tasks.

Opaque-ID public-duty extraction is primary. Original-ID extraction is only a
fixed-weight sensitivity test, never selected by held-out outcomes. Candidate
dependency edges/requirements are NOT runtime facts or scoring inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from train_typed_action_development import (ROOT, sha, write_new, now, load_data, verify_hashes,
    baseline_predictions, metrics, SETTINGS as PARENT_SETTINGS)
from paperalpha_runtime.duty_graph import DUTY_FEATURE_NAMES, duty_vectors
from paperalpha_monitor.duty_conditioned_bn import DutyConditioner, DutyConditionedBN, DutyStageCompiler
from paperalpha_monitor.typed_action_supervision import TypedActionSupervisionBN, VISIBLE_NAMES, bce

SETTINGS = {"regularization": .02, "max_iter": 200, "threshold": .5,
            "primary_duties": "opaque", "secondary_duties": "original", "episode_supervision": False}
METHODS = ("duty_bn_v9", "duty_bn_no_messages", "duty_visible_logistic")


def extraction(extract_run, task, version):
    identity = task + "__" + version
    item = json.loads((extract_run / "inputs" / (identity + ".json")).read_text(encoding="utf-8"))
    result = json.loads((extract_run / "extractions" / (identity + ".json")).read_text(encoding="utf-8"))
    if result["status"] != "quote_anchored":
        raise ValueError("predefined extraction unavailable; do not select a replacement by scores")
    vectors, graph = duty_vectors(item["packet"], result["annotation"], aliases=item["aliases"])
    if result["graph"] != graph or result["vectors"] != vectors:
        raise ValueError("extraction result changed")
    return item, result, vectors, graph


def sources():
    return [Path(__file__), ROOT / 'code/commands/train_typed_action_development.py', *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
        *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json"),
        ROOT / 'knowledge/risks/risk_catalog.json', ROOT / 'code/tests/test_duty_graph.py', ROOT / 'code/tests/test_duty_conditioned_bn.py']


def prepare(parent, extract_run, output):
    if output.exists():
        raise ValueError("new output directory required")
    plan, data = load_data(parent)
    report = json.loads((extract_run / "REPORT.json").read_text(encoding="utf-8"))
    verify_hashes(extract_run, report["artifacts_sha256"])
    output.mkdir(parents=True)
    hashes, data_hashes = {}, {}
    for source in sources():
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        hashes[source.relative_to(ROOT).as_posix()] = sha(source)
    for task in ("dl-t20", "dl-t22"):
        for version in ("opaque", "original"):
            item, result, _, _ = extraction(extract_run, task, version)
            path = output / "extractions" / (task + "__" + version + ".json")
            write_new(path, {"input": item, "output": result})
            data_hashes[path.relative_to(output).as_posix()] = sha(path)
    value = {"created_at": now(), "parent": str(parent), "parent_plan_sha256": sha(parent / "PLAN.json"),
        "parent_report_sha256": sha(parent / "REPORT.json"), "extract_run": str(extract_run),
        "extract_report_sha256": sha(extract_run / "REPORT.json"), "folds": plan["folds"], "settings": SETTINGS,
        "methods": METHODS, "source_sha256": hashes, "data_sha256": data_hashes,
        "interpretation": "Known development tasks, no new runtime data. Semantic role conditioning, not validation of MAS interaction risk.",
        "control_input_parity": "Enriched Logistic receives the same candidate-duty vector and original visible graph statistics.",
        "extraction_semantics": "No manual label correction. Quoted classifications are noisy. Dependency/condition proposals excluded from inference."}
    write_new(output / "PLAN.json", value)
    print(json.dumps({"prepared": str(output), "models_to_fit": 6, "new_API_calls": 0}), flush=True)


def load(output):
    plan = json.loads((output / "PLAN.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, plan["source_sha256"])
    verify_hashes(output, plan["data_sha256"])
    parent = Path(plan["parent"])
    if sha(parent / "PLAN.json") != plan["parent_plan_sha256"] or sha(parent / "REPORT.json") != plan["parent_report_sha256"]:
        raise ValueError("parent experiment changed")
    original, data = load_data(parent)
    conditions, vectors, alternate, alternative_conditions = {}, {}, {}, {}
    for entry in original["runs"]:
        key, task = entry["id"], entry["task_id"]
        for version in ("opaque", "original"):
            doc = json.loads((output / "extractions" / (task + "__" + version + ".json")).read_text(encoding="utf-8"))
            item, extracted = doc["input"], doc["output"]
            from paperalpha_runtime.semantic_evidence import digest
            if digest(data[key][2]["rows"][0]["payload"]["public_configuration"]) != item["configuration_sha256"]:
                raise ValueError("duty extraction is not this run's original public configuration")
            condition = DutyConditioner(item["packet"], extracted["annotation"], item["aliases"])
            if version == "opaque":
                conditions[key], vectors[key] = condition, condition.vectors
            else:
                alternate[key] = condition.trace(data[key][0])
                alternative_conditions[key] = condition
    return plan, data, conditions, vectors, alternate, alternative_conditions


def logistic_fit(rows, targets):
    width = len(VISIBLE_NAMES) + len(DUTY_FEATURE_NAMES)
    initial = np.zeros((2, width))
    initial[:, 0] = -2.
    def objective(theta):
        weights, loss, gradient = theta.reshape(2, width), 0., np.zeros((2, width))
        for records, labels in zip(rows, targets):
            for aid, row in records.items():
                x, target = np.asarray(row["visible_features"]), labels.actions[aid]
                p = expit(weights @ x)
                for j, y in ((0, target.forbidden), (1, target.unnecessary if target.forbidden == 0 else None)):
                    if y is None:
                        continue
                    loss += bce(p[j], y)[0] / (len(rows) * len(records))
                    gradient[j] += (p[j] - y) * x / (len(rows) * len(records))
        centered = weights - initial
        return loss + SETTINGS["regularization"] * np.sum(centered ** 2), (gradient + 2 * SETTINGS["regularization"] * centered).ravel()
    fit = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B", bounds=[(-8., 8.)] * initial.size,
        options={"maxiter": SETTINGS["max_iter"], "gtol": 1e-6, "ftol": 1e-10})
    return {"theta": fit.x.reshape(2, width).tolist(), "feature_names": [*VISIBLE_NAMES, *("duty:" + k for k in DUTY_FEATURE_NAMES)],
            "optimizer": {"success": bool(fit.success), "loss": float(fit.fun), "iterations": int(fit.nit),
                          "gradient_max": float(np.max(np.abs(fit.jac)))}}


def records_for(data, vectors):
    raw = {}
    for key, (trace, _, _, _) in data.items():
        raw[key] = TypedActionSupervisionBN().predict_actions(trace)
        for row in raw[key].values():
            row["visible_features"] += list(vectors[key][row["actor"]])
    return raw


def train(output):
    plan, data, conditions, vectors, alternate, alternative_conditions = load(output)
    if (output / "models").exists():
        raise ValueError("training already started; no overwrite")
    (output / "models").mkdir()
    traces = {key: conditions[key].trace(row[0]) for key, row in data.items()}
    raw = records_for(data, vectors)
    all_rows, fits, sensitivity = [], [], []
    for fold in plan["folds"]:
        train_ids, evaluation_ids = fold["train"], fold["evaluate"]
        train_traces, targets = [traces[k] for k in train_ids], [data[k][1] for k in train_ids]
        for method in METHODS:
            started = time.perf_counter()
            print("fit", fold["evaluation_task"], method, flush=True)
            if method.startswith("duty_bn"):
                model = DutyConditionedBN(regularization=SETTINGS["regularization"], max_iter=SETTINGS["max_iter"],
                    use_messages=method != "duty_bn_no_messages")
                model.fit(train_traces, targets)
                saved = model.parameters()
                predictions = {key: model.predict_actions(traces[key]) for key in evaluation_ids}
                stream_error = 0.
                for key in evaluation_ids:
                    condition = conditions[key]
                    compiler = DutyStageCompiler(packet=condition.packet, annotation=condition.annotation,
                        aliases=condition.aliases, **data[key][3])
                    engine = model.engine()
                    for row in data[key][2]["rows"]:
                        step = compiler.push(row)
                        if step is not None:
                            engine.apply(step, compiler.variables)
                    for aid, value in engine.action_predictions.items():
                        for target in ("forbidden", "unnecessary"):
                            stream_error = max(stream_error, abs(value[target] - predictions[key][aid][target]))
                if stream_error > 1e-12:
                    raise ValueError("fitted streaming mismatch")
                saved["stream_error"] = stream_error
                saved["learned_duty_weights"] = {name: float(model.theta[i]) for name, i in zip(DUTY_FEATURE_NAMES, model.registry.duty_indices)}
                if method == "duty_bn_v9":
                    errors, alternative_records = [], []
                    for key in evaluation_ids:
                        changed = model.predict_actions(alternate[key])
                        for aid, value in changed.items():
                            errors.extend(abs(value[t] - predictions[key][aid][t]) for t in ("forbidden", "unnecessary"))
                            label = data[key][1].actions[aid]
                            alternative_records.append({"targets": {"forbidden": label.forbidden, "unnecessary": label.unnecessary}, "prediction": value})
                    sensitivity.append({"fold": fold["evaluation_task"], "max_original_vs_opaque_score_difference": max(errors),
                                        "original_extraction_fixed_weights_metrics": metrics(alternative_records)})
            else:
                saved = logistic_fit([raw[k] for k in train_ids], targets)
                predictions = {key: baseline_predictions(saved, raw[key], "visible_logistic") for key in evaluation_ids}
            saved["seconds_including_prediction_checks"] = time.perf_counter() - started
            write_new(output / "models" / fold["evaluation_task"] / (method + ".json"), saved)
            fits.append({"fold": fold["evaluation_task"], "method": method, "optimizer": saved["optimizer"],
                         "learned_duty_weights": saved.get("learned_duty_weights")})
            records = []
            for key in evaluation_ids:
                for aid, prediction in predictions[key].items():
                    label = data[key][1].actions[aid]
                    records.append({"fold": fold["evaluation_task"], "run_id": key, "action_id": aid, "method": method,
                        "targets": {"forbidden": label.forbidden, "unnecessary": label.unnecessary}, "prediction": prediction})
            write_new(output / "predictions" / fold["evaluation_task"] / (method + ".json"), records)
            all_rows.extend(records)
            print(json.dumps({"fold": fold["evaluation_task"], "method": method, "metrics": metrics(records)}, ensure_ascii=False), flush=True)
    comparisons = {method: metrics([r for r in all_rows if r["method"] == method]) for method in METHODS}
    parent_report = json.loads((Path(plan["parent"]) / "REPORT.json").read_text(encoding="utf-8"))
    for method in ("typed_joint_v8", "visible_logistic"):
        comparisons[method + "_frozen_reference"] = parent_report["metrics_pooled_development_predictions"][method]
    report = {"plan_sha256": sha(output / "PLAN.json"), "model_parameters": 438, "fits": fits,
        "metrics": comparisons, "extraction_sensitivity": sensitivity, "tasks": 2, "actions": 53,
        "episode_head_trained": False, "dependencies_used_for_scoring": False, "new_API_calls_for_fitting": 0,
        "scope": "Known-task development comparison. Public duties affect local factors; no independent interaction-risk or actual-harm result.",
        "artifacts_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}}
    write_new(output / "REPORT.json", report)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent.resolve(), args.extract.resolve(), args.out.resolve())
    else:
        train(args.out.resolve())
