"""Public train/validation integration of the complete event DAG, no API.

The old test split is not opened. Raw training/validation records are selected
by their prior frozen provenance and hash, then independently reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.agentleak_event_projection import project, later_labels, system_prompts
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.observable_event_learning import complete_trace_design, complete_trace_objective, fit_event_traces
from paperalpha_monitor.privacy_flow_data import prefix_packet, final_value_labels
from inspect_agentleak_release import inspect_trace, REVISION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(out, parent, upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("pinned AgentLeak checkout required")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("upstream tracked source changed")
    manifest = read(parent / "DATA.json")
    out.mkdir(parents=True, exist_ok=False)
    ids = set()
    for split in ("train", "validation"):
        path = parent / (split + ".json")
        if sha(path) != manifest["dataset_sha256"][split]:
            raise ValueError("old development split changed")
        ids.update(row["id"] for row in read(path))
        target = out / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    provenance = [row for row in read(parent / "provenance.json") if row["id"] in ids]
    if {row["id"] for row in provenance} != ids or any(row["split"] not in ("train", "validation") for row in provenance):
        raise ValueError("complete development-only provenance required")
    write(out / "inputs/provenance.json", provenance)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/run_agentleak_event_dag.py', 'code/commands/inspect_agentleak_release.py', 'code/commands/harnessaudit_cohort.py',
        'code/commands/scalar_binary_graph.py', 'code/commands/verify_agentleak_event_dag.py', 'code/tests/test_event_elimination.py',
        'code/tests/test_observable_event_dag.py', 'code/tests/test_relational_forecast_study.py', 'code/commands/branch_publisher_fixtures.py',
        'code/commands/workflow_risk_fixtures.py', 'code/commands/audit_relational_event_meaning.py', 'code/commands/run_relational_forecast_study.py', 'code/commands/verify_relational_forecast_study.py']
    for name in names:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    plan = {"schema": "agentleak-full-event-dag-development-v1", "upstream": str(upstream.resolve()), "revision": revision,
        "source_sha256": {n: sha(out / "source" / n) for n in sorted(names)},
        "inputs_sha256": {p.name: sha(p) for p in sorted((out / "inputs").glob("*.json"))},
        "parent": str(parent.resolve()), "regularization": [.001, .03], "max_iter": 500,
        "baseline": {"logistic_C": 1., "histgb_leaves": 7, "histgb_iterations": 100, "histgb_l2": .1, "random_state": 17},
        "training": "Complete worker+final per-value future labels, one early-prefix joint likelihood per original workflow; intermediate event meanings are supervised.",
        "inputs": "All protected literal groups; exact vault/prepare/work/publish delivery including the worker's direct vault access. No generator identity or domain feature.",
        "inference": "Exact variable elimination, no field deletion, no approximate independent-marginal fallback. Authorization-only union query is used only after verifying its exact scope.",
        "selection": "Report both BN penalties and all fixed baselines, no new default selected.",
        "old_test_read": False, "new_api_calls": 0,
        "scope": "Previously used public development splits, not new held-out evidence. Fixed two physical agents and three acyclic functional stages; literal exposure only.",
        "references": ["https://github.com/Privatris/AgentLeak/tree/" + revision,
            "https://pgmpy.org/api/generated/inference/pgmpy.inference.VariableElimination.html"]}
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "plan_sha256": sha(out / "PLAN.json"), "traces": len(ids), "sources": len(names)}))


def frozen_plan(out):
    plan = read(out / "PLAN.json")
    if ROOT != (out / "source").resolve():
        raise ValueError("run the frozen source")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source changed: " + name)
    for name, expected in plan["inputs_sha256"].items():
        if sha(out / "inputs" / name) != expected:
            raise ValueError("input changed")
    return plan


def load_data(out, *, verbose=True):
    plan = frozen_plan(out)
    provenance = {r["id"]: r for r in read(out / "inputs/provenance.json")}
    result, projection = [], []
    seen_templates = {}
    for split in ("train", "validation"):
        for old in read(out / "inputs" / (split + ".json")):
            source = provenance[old["id"]]
            if source["split"] != split or source["template_id"] != old["template_id"]:
                raise ValueError("source split identity changed")
            path = Path(plan["upstream"]) / source["source_path"]
            if sha(path) != source["source_sha256"]:
                raise ValueError("raw source changed")
            raw = read(path)
            if not inspect_trace(raw)["eligible"]:
                raise ValueError("original trace no longer eligible")
            initial, calls, messages = raw["input"], raw["llm_calls"], raw["channel_messages"]
            expected = system_prompts(initial)
            if [calls[i]["system_prompt"] for i in (1, 2, 3)] != [expected[n] for n in ("prepare", "work", "publish")]:
                raise ValueError("actual static stage prompts differ; do not invent vault delivery")
            early = project(initial, coordinator_text=messages[1]["content"], coordinator_record_id=calls[1]["call_id"])
            late = project(initial, coordinator_text=messages[1]["content"], coordinator_record_id=calls[1]["call_id"],
                worker_text=messages[2]["content"], worker_record_id=calls[2]["call_id"])
            labels = later_labels(initial, worker_text=messages[2]["content"], worker_record_id=calls[2]["call_id"],
                final_text=messages[4]["content"], final_record_id=calls[3]["call_id"])
            for stage in (1, 2):
                assert old["prefixes"][str(stage)] == prefix_packet(initial, model=raw["model"], vertical=raw["vertical"],
                    coordinator_text=messages[1]["content"], worker_text=messages[2]["content"] if stage == 2 else None)
            assert old["final_value_labels"] == final_value_labels(initial, messages[4]["content"])
            assert int(any(old["final_value_labels"])) == old["y"]
            if old["template_id"] in seen_templates and seen_templates[old["template_id"]] != split:
                raise ValueError("template leaked between development splits")
            seen_templates[old["template_id"]] = split
            row = {"id": old["id"], "split": split, "template_id": old["template_id"], "vertical": old["vertical"],
                "early": early, "late": late, "labels": labels, "y": old["y"]}
            result.append(row)
            projection.append({"id": old["id"], "split": split, "source_sha256": source["source_sha256"],
                "early_sha256": early.sha256, "late_sha256": late.sha256, "fields": len(early.action_ids),
                "events": len(early.event_dag.ids), "early_unobserved_events": len(early.event_dag.ids)-len(early.event_dag.observations)})
            if verbose and len(result) % 250 == 0:
                print(json.dumps({"projected": len(result), "split": split}), flush=True)
    return result, projection


def baseline_features(row, phase):
    dag = row[phase].event_dag
    n = len(row[phase].action_ids)
    v = np.asarray([dag.observations[f"value-{j}:vault"]["value"] for j in range(n)])
    c = np.asarray([dag.observations[f"value-{j}:prepare"]["value"] for j in range(n)])
    w = np.asarray([dag.observations[f"value-{j}:work"]["value"] for j in range(n)]) if phase == "late" else np.zeros(n)
    return [float(n), float(v.sum()), float(c.sum()), float((v*c).sum()), float(w.sum()), float((v*w).sum()), float((c*w).sum()), float((v*c*w).sum())]


def metric(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    y, p = np.asarray(y), np.asarray(p)
    return {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p-y)**2)), "mean_prediction": float(p.mean()),
        "nll_clipped_1e_12": float(np.mean(-y*np.log(np.clip(p, 1e-12, 1-1e-12))-(1-y)*np.log(np.clip(1-p, 1e-12, 1)))),
        "auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
        "f1_at_half": float(f1_score(y, p >= .5, zero_division=0))}


def run(out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    plan = frozen_plan(out)
    data, projection = load_data(out)
    write(out / "PROJECTION.json", projection)
    train = [r for r in data if r["split"] == "train"]
    dags, labels = [r["early"].event_dag for r in train], [r["labels"] for r in train]
    predictions, optimizers, equivalence = {}, {}, {}
    x_local, y_local = complete_trace_design(EliminatingDAGWorkflowRiskBN().event_law, dags, labels)
    for penalty in plan["regularization"]:
        started = time.monotonic()
        name = "event_dag_" + str(penalty)
        model = fit_event_traces(dags, labels, regularization=penalty, max_iter=plan["max_iter"], model_class=EliminatingDAGWorkflowRiskBN)
        write(out / "models" / (name + ".json"), model.parameters())
        optimizers[name] = model.optimizer
        print(json.dumps({"fitted": name, "loss": model.optimizer["loss"], "iterations": model.optimizer["iterations"]}), flush=True)
        control = LogisticRegression(C=1/(2*penalty*len(train)), fit_intercept=False, tol=1e-10, max_iter=1000).fit(x_local, y_local)
        control_loss = complete_trace_objective(control.coef_[0], x_local, y_local, len(train), penalty)[0]
        equivalence[name] = {"coefficient": control.coef_[0].tolist(), "training_loss": control_loss,
            "training_loss_difference": abs(control_loss-model.optimizer["loss"]), "max_local_logit_difference": float(np.max(np.abs(x_local@(model.theta[:25]-control.coef_[0]))))}
        records = []
        for i, row in enumerate(data):
            for phase in ("early", "late"):
                probability = model.predict_known_authorization_union(row[phase])["probabilities"]["system:any"]
                records.append({"id": row["id"], "split": row["split"], "phase": phase, "y": row["y"], "probability": probability,
                    "input_sha256": row[phase].sha256})
            if (i+1) % 500 == 0:
                print(json.dumps({"model": name, "scored_flows": i+1}), flush=True)
        predictions[name] = records
        write(out / "predictions" / (name + ".json"), records)
        print(json.dumps({"completed": name, "seconds": time.monotonic()-started}), flush=True)
    prevalence = float(np.mean([r["y"] for r in train]))
    for method in ("logistic", "histgb"):
        records = []
        for phase in ("early", "late"):
            x = np.asarray([baseline_features(r, phase) for r in train]); y = np.asarray([r["y"] for r in train])
            if method == "logistic":
                model = LogisticRegression(C=plan["baseline"]["logistic_C"], max_iter=1000, tol=1e-8).fit(x, y)
            else:
                model = HistGradientBoostingClassifier(max_leaf_nodes=plan["baseline"]["histgb_leaves"], max_iter=plan["baseline"]["histgb_iterations"],
                    l2_regularization=plan["baseline"]["histgb_l2"], random_state=plan["baseline"]["random_state"]).fit(x, y)
            dest = out / "models" / (method + "_" + phase + ".joblib")
            if dest.exists(): raise ValueError("refuse model overwrite")
            joblib.dump(model, dest)
            p = model.predict_proba(np.asarray([baseline_features(r, phase) for r in data]))[:, 1]
            records += [{"id": r["id"], "split": r["split"], "phase": phase, "y": r["y"], "probability": float(q)} for r, q in zip(data, p)]
        predictions[method] = records; write(out / "predictions" / (method + ".json"), records)
    for method in ("train_prevalence", "visible_persistence"):
        records = []
        for row in data:
            for phase in ("early", "late"):
                values = baseline_features(row, phase)
                probability = prevalence if method == "train_prevalence" else float((values[2] if phase == "early" else values[4]) > 0)
                records.append({"id": row["id"], "split": row["split"], "phase": phase, "y": row["y"], "probability": probability})
        predictions[method] = records; write(out / "predictions" / (method + ".json"), records)
    report = {"metrics": {name: {f"{split}/{phase}": metric([r["y"] for r in records if r["split"] == split and r["phase"] == phase],
        [r["probability"] for r in records if r["split"] == split and r["phase"] == phase]) for split in ("train", "validation") for phase in ("early", "late")} for name, records in predictions.items()},
        "optimizer": optimizers, "local_logistic_equivalence": equivalence, "local_training_design_shape": list(x_local.shape),
        "local_training_design_rank": int(np.linalg.matrix_rank(x_local)),
        "template_counts": {s: len({r["template_id"] for r in data if r["split"] == s}) for s in ("train", "validation")},
        "plan_sha256": sha(out / "PLAN.json"), "projection_sha256": sha(out / "PROJECTION.json"),
        "models_sha256": {p.name: sha(p) for p in sorted((out / "models").iterdir())},
        "predictions_sha256": {p.name: sha(p) for p in sorted((out / "predictions").iterdir())},
        "old_test_read": False, "new_api_calls": 0, "new_default_selected": False,
        "scope": "Previously used public development data. Entire fields retained; no generator identity. BN sees additional per-value/intermediate labels; classifiers see only terminal labels."}
    write(out / "REPORT.json", report)
    print(json.dumps({"completed": True, "flows": len(data), "predictions": sum(len(v) for v in predictions.values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    for flag in ("out", "parent", "upstream"):
        parser.add_argument("--"+flag, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.out.resolve(), args.parent.resolve(), args.upstream.resolve())
    else: run(args.out.resolve())
