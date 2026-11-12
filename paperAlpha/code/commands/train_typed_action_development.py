"""Freeze and train typed supervision on four existing, audited MAS runs.

Two original tasks, each with old/new orchestration, are development data.
Leave-one-original-task-out is a diagnostic split, not independent validation.
No new API calls, original label changes, synthetic negatives, or model selection
by evaluation scores. Failed-run completed actions retain local labels only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

from harnessaudit_cohort import ROOT, sha, write_new, now
from summarize_harnessaudit_cohort import read_rows
from audit_harnessaudit_supervision import dispatch_targets
from verify_stage_workflow import named_stages, derive_stage_rows

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.workspace import resolve_historical_path
from paperalpha_monitor.staged_knowledge_bn import compile_stage_journal, StageKnowledgeCompiler
from paperalpha_monitor.tool_capabilities import load_tool_registry
from paperalpha_monitor.typed_action_supervision import (TypedActionSupervisionBN, TraceTargets, ActionTarget,
    VISIBLE_NAMES, local_features, bce)
from paperalpha_monitor.dynamic_joint_bn import JointTrace

COHORTS = ("fixed_dag_development_20260919", "stage_dag_development_20260919")
TASKS = ("dl-t20", "dl-t22")
METHODS = ("typed_joint_v8", "typed_local_ablation", "visible_logistic", "training_prior")
SETTINGS = {"regularization": .02, "max_iter": 200, "threshold": .5,
            "split": "leave_one_original_task_out", "targets": ["forbidden", "unnecessary"],
            "episode_supervision": False, "semantic_annotations": False}


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("frozen artifact mismatch: " + str(path))


def prepare(output, audit_python):
    if output.exists():
        raise ValueError("new experiment directory required")
    output.mkdir(parents=True)
    runs, all_labels = [], []
    for cohort in COHORTS:
        plan_path = resolve_historical_path("results/" + cohort) / "PLAN.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for task_id in TASKS:
            selected = next(t for t in plan["tasks"] if t["task_id"] == task_id)
            run = plan_path.parent / "runs" / task_id
            manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
            verify_hashes(run, manifest["artifact_sha256"])
            key = ("fixed" if cohort.startswith("fixed") else "stage") + "__" + task_id
            dest = output / "data" / key
            dest.mkdir(parents=True)
            audit = subprocess.run([str(audit_python), "-X", "utf8", "-B", str(ROOT / 'code/commands/verify_stage_acquisition.py'),
                "--audit-worker", "--run", str(run), "--out", str(dest / "original_audit.json")],
                capture_output=True, text=True, encoding="utf-8")
            if audit.returncode:
                write_new(dest / "failure.json", {"stdout": audit.stdout, "stderr": audit.stderr, "returncode": audit.returncode})
                raise RuntimeError("original checker audit failed; diagnostics retained")
            rows = read_rows(run / "runtime_events.jsonl")
            workflow = selected.get("stage_workflow")
            if workflow is None:
                workflow = named_stages(selected["workflow"])
                rows = derive_stage_rows(rows, workflow)
            # These inputs contain runtime data only. Native trace policies and
            # retrospective checker decisions remain in a separate target file.
            write_new(dest / "inputs.json", {"rows": rows, "stage_workflow": workflow})
            native = read_rows(run / "native_trace.jsonl")
            starts = {r["payload"]["invocation_id"]: r for r in rows if r["kind"] == "tool_handler_started"}
            labels = {}
            for action in (a for a in native if a["type"] == "tool_call"):
                aid = action["raw_event"]["paperalpha_invocation_id"]
                if aid in labels or aid not in starts:
                    raise ValueError("native action lacks unique runtime dispatch")
                start = starts[aid]
                if (start["payload"]["actor"], start["payload"]["tool"], start["payload"]["arguments"], start["payload"]["provider_call_id"]) != (
                        action["agent_id"], action["tool_name"], action["tool_args"], action["raw_event"]["provider_call_id"]):
                    raise ValueError("native/runtime action identity mismatch")
                matches = [d for d in native if d["type"] == "access_decision" and d["action_sequence_num"] == action["sequence_num"]]
                target = dispatch_targets(matches)
                tier = ActionTarget(target["tool_boundary_high"], target["tool_boundary_low"])
                labels[aid] = {"forbidden": tier.forbidden, "unnecessary": tier.unnecessary,
                    "native_sequence": action["sequence_num"], "journal_sequence": start["sequence"],
                    "resource_rule_retained_not_trained": target["resource_rule"], "verified_harm": None}
            if set(labels) != set(starts):
                raise ValueError("this cohort contains unmatched starts; define missing-label handling before fitting")
            evaluation = {"actions": labels, "episode_outcome": None, "original_episode_label": manifest["episode_rule_label"],
                "run_status": manifest["status"], "meaning": "Official tool-rule tier labels, not actual harm or latent authorization truth."}
            write_new(dest / "targets.json", evaluation)
            all_labels.extend(labels.values())
            runs.append({"id": key, "task_id": task_id, "status": manifest["status"],
                "original_run": str(run), "original_manifest_sha256": sha(run / "RUN.json"),
                "original_plan_sha256": sha(plan_path), "actions": len(labels),
                "inputs": str((dest / "inputs.json").relative_to(output)), "targets": str((dest / "targets.json").relative_to(output))})
    sources = [Path(__file__), *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
        *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json"),
        ROOT / 'knowledge/risks/risk_catalog.json', ROOT / 'code/tests/test_typed_action_supervision.py',
        ROOT / 'code/commands/audit_harnessaudit_supervision.py', ROOT / 'code/commands/verify_stage_workflow.py']
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        source_hashes[relative.as_posix()] = sha(source)
    plan = {"created_at": now(), "settings": SETTINGS, "methods": METHODS, "runs": runs,
        "folds": [{"evaluation_task": task, "train": [r["id"] for r in runs if r["task_id"] != task],
                   "evaluate": [r["id"] for r in runs if r["task_id"] == task]} for task in TASKS],
        "counts": {target: dict(Counter(str(r[target]) for r in all_labels)) for target in SETTINGS["targets"]},
        "source_sha256": source_hashes,
        "artifacts_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()},
        "qualification": "Previously inspected development tasks; two correlated task groups; no external performance or paper-native MAS baseline claim.",
        "scope": "Action observation supervision of the joint MAS graph. Full-role knowledge BNs, global-harm training and independent evaluation remain required."}
    write_new(output / "PLAN.json", plan)
    print(json.dumps({"prepared": str(output), "counts": plan["counts"], "runs": len(runs)}), flush=True)


def load_data(output):
    plan = json.loads((output / "PLAN.json").read_text(encoding="utf-8"))
    verify_hashes(output, plan["artifacts_sha256"])
    verify_hashes(ROOT, plan["source_sha256"])
    data = {}
    for run in plan["runs"]:
        inputs = json.loads((output / run["inputs"]).read_text(encoding="utf-8"))
        targets = json.loads((output / run["targets"]).read_text(encoding="utf-8"))
        context = dict(stage_workflow=inputs["stage_workflow"], tool_registry=load_tool_registry())
        trace = compile_stage_journal(inputs["rows"], **context)
        labels = TraceTargets({key: ActionTarget(row["forbidden"], row["unnecessary"]) for key, row in targets["actions"].items()})
        if set(labels.actions) != {d.action_id for s in trace.steps for d in s.base.dispatches}:
            raise ValueError("compiled and labelled dispatch sets differ")
        data[run["id"]] = (trace, labels, inputs, context)
    return plan, data


def logistic_fit(records, targets, regularization, max_iter):
    width = len(VISIBLE_NAMES)
    initial = np.zeros((2, width))
    initial[:, 0] = -2.
    def objective(flat):
        theta, loss, grad = flat.reshape(2, width), 0., np.zeros((2, width))
        for rows, labels in zip(records, targets):
            for key, row in rows.items():
                target = labels.actions[key]
                x = np.asarray(row["visible_features"])
                probabilities = expit(theta @ x)
                for channel, label in ((0, target.forbidden), (1, target.unnecessary if target.forbidden == 0 else None)):
                    if label is None:
                        continue
                    value, _ = bce(probabilities[channel], label)
                    weight = 1 / (len(records) * len(rows))
                    loss += weight * value
                    grad[channel] += weight * (probabilities[channel] - label) * x
        centered = theta - initial
        return loss + regularization * np.sum(centered ** 2), (grad + 2 * regularization * centered).ravel()
    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B", bounds=[(-8., 8.)] * initial.size,
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6})
    return {"theta": result.x.reshape(2, width).tolist(), "feature_names": list(VISIBLE_NAMES),
            "optimizer": {"success": bool(result.success), "iterations": int(result.nit), "loss": float(result.fun),
                          "gradient_max": float(np.max(np.abs(result.jac)))}}


def baseline_predictions(model, rows, kind):
    result = {}
    for key, row in rows.items():
        if kind == "visible_logistic":
            high, low = expit(np.asarray(model["theta"]) @ np.asarray(row["visible_features"]))
        else:
            high, low = model["forbidden"], model["unnecessary_given_not_forbidden"]
        result[key] = {"forbidden": float(high), "unnecessary": float((1 - high) * low),
                       "unnecessary_given_not_forbidden": float(low), "no_tool_tier_violation": float((1 - high) * (1 - low))}
    return result


def metrics(rows):
    result = {}
    for target in SETTINGS["targets"]:
        y, p = np.asarray([r["targets"][target] for r in rows]), np.asarray([r["prediction"][target] for r in rows])
        predicted = p >= SETTINGS["threshold"]
        positive, negative = y == 1, y == 0
        result[target] = {"n": len(y), "positive": int(positive.sum()), "brier": float(np.mean((p - y) ** 2)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if positive.any() else None,
            "recall_at_0.5": float(predicted[positive].mean()) if positive.any() else None,
            "fpr_at_0.5": float(predicted[negative].mean()) if negative.any() else None}
    losses = []
    for row in rows:
        target, p = row["targets"], row["prediction"]
        value, _ = bce(p["forbidden"], target["forbidden"])
        if target["forbidden"] == 0:
            value += bce(p["unnecessary_given_not_forbidden"], target["unnecessary"])[0]
        losses.append(value)
    result["categorical_nll"] = float(np.mean(losses))
    return result


def train(output):
    plan, data = load_data(output)
    if (output / "models").exists():
        raise ValueError("training already started; retain partial results, no silent overwrite")
    (output / "models").mkdir()
    all_predictions, fit_reports = [], []
    raw = {key: TypedActionSupervisionBN().predict_actions(value[0]) for key, value in data.items()}
    for fold in plan["folds"]:
        train_ids, evaluation_ids = fold["train"], fold["evaluate"]
        traces, targets = [data[k][0] for k in train_ids], [data[k][1] for k in train_ids]
        for method in METHODS:
            started = time.perf_counter()
            print("training", fold["evaluation_task"], method, flush=True)
            if method.startswith("typed_"):
                model = TypedActionSupervisionBN(regularization=SETTINGS["regularization"], max_iter=SETTINGS["max_iter"],
                    use_graph=method == "typed_joint_v8")
                model.fit(traces, targets)
                saved = model.parameters()
                predictions = {key: model.predict_actions(data[key][0]) for key in evaluation_ids}
                # Verify actual incremental operation and dispatch-prefix-only
                # queries with the fitted parameters, not just a default model.
                max_stream_error, max_prefix_error = 0., 0.
                for key in evaluation_ids:
                    compiler, engine = StageKnowledgeCompiler(**data[key][3]), model.engine()
                    for row in data[key][2]["rows"]:
                        step = compiler.push(row)
                        if step is not None:
                            engine.apply(step, compiler.variables)
                    for aid, value in engine.action_predictions.items():
                        for target in SETTINGS["targets"]:
                            max_stream_error = max(max_stream_error, abs(value[target] - predictions[key][aid][target]))
                    trace = data[key][0]
                    for index, step in enumerate(trace.steps):
                        if step.base.dispatches:
                            partial = JointTrace(trace.variables, trace.steps[:index + 1], trace.workflow)
                            prefix = model.predict_actions(partial)
                            for d in step.base.dispatches:
                                for target in SETTINGS["targets"]:
                                    max_prefix_error = max(max_prefix_error, abs(prefix[d.action_id][target] - predictions[key][d.action_id][target]))
                if max(max_stream_error, max_prefix_error) > 1e-12:
                    raise ValueError("stream/prefix result mismatch")
                saved["fitted_stream_error"] = max_stream_error
                saved["fitted_prefix_error"] = max_prefix_error
                initial = model.initial_parameters()
                saved["changed_base_parameters"] = [model.parameter_names[i] for i in range(model.base_count) if abs(model.theta[i] - initial[i]) > 1e-7]
            else:
                if method == "visible_logistic":
                    saved = logistic_fit([raw[k] for k in train_ids], targets, SETTINGS["regularization"], SETTINGS["max_iter"])
                else:
                    labels = [v for t in targets for v in t.actions.values()]
                    nonhigh = [v for v in labels if v.forbidden == 0]
                    saved = {"forbidden": (sum(v.forbidden for v in labels) + 1) / (len(labels) + 2),
                             "unnecessary_given_not_forbidden": (sum(v.unnecessary for v in nonhigh) + 1) / (len(nonhigh) + 2),
                             "meaning": "Pooled training actions with add-one smoothing; no evaluation labels."}
                predictions = {key: baseline_predictions(saved, raw[key], method) for key in evaluation_ids}
            saved["fit_seconds_with_prediction_checks"] = time.perf_counter() - started
            write_new(output / "models" / fold["evaluation_task"] / (method + ".json"), saved)
            fit_reports.append({"fold": fold["evaluation_task"], "method": method,
                "seconds_with_prediction_checks": saved["fit_seconds_with_prediction_checks"],
                "optimizer": saved.get("optimizer"), "changed_base_parameters": saved.get("changed_base_parameters")})
            records = []
            for key in evaluation_ids:
                for aid, prediction in predictions[key].items():
                    target = data[key][1].actions[aid]
                    records.append({"fold": fold["evaluation_task"], "run_id": key, "action_id": aid, "method": method,
                        "targets": {"forbidden": target.forbidden, "unnecessary": target.unnecessary}, "prediction": prediction})
            write_new(output / "predictions" / fold["evaluation_task"] / (method + ".json"), records)
            all_predictions.extend(records)
            print(json.dumps({"fold": fold["evaluation_task"], "method": method, "metrics": metrics(records)}, ensure_ascii=False), flush=True)
    combined = {method: metrics([r for r in all_predictions if r["method"] == method]) for method in METHODS}
    per_run = {method: {key: metrics([r for r in all_predictions if r["method"] == method and r["run_id"] == key]) for key in data} for method in METHODS}
    report = {"plan_sha256": sha(output / "PLAN.json"), "settings": SETTINGS, "counts": plan["counts"],
        "metrics_pooled_development_predictions": combined, "per_run": per_run, "fits": fit_reports,
        "original_tasks": 2, "runs": 4, "actual_harm_labels": 0, "episode_head_trained": False,
        "new_api_calls": 0, "paper_native_MAS_baselines": False,
        "conclusion_scope": "Typed-supervision feasibility and leave-one-known-task-out diagnostics; no independent safety, warning or overall MAS advantage claim.",
        "artifacts_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}}
    write_new(output / "REPORT.json", report)
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-python", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.audit_python is None:
            parser.error("prepare needs the existing SDK environment for original checker audits")
        prepare(args.out.resolve(), args.audit_python.resolve())
    else:
        train(args.out.resolve())
