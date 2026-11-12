"""Independent field-chain formula audit of the public event-DAG experiment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


def independent_fields(model, sample, phase):
    """Hand-written scalar chain marginalization, no inference-engine calls."""
    dag = sample.event_dag
    coefficient = dict(zip(model.names, model.theta))
    def sigmoid(x):
        return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
    pub0 = sigmoid(coefficient["event:publish:bias"])
    pub1 = sigmoid(coefficient["event:publish:bias"]+coefficient["event:publish:parent_mean"])
    values = []
    for j in range(len(sample.action_ids)):
        v = dag.observations[f"value-{j}:vault"]["value"]
        c = dag.observations[f"value-{j}:prepare"]["value"]
        assert dag.parents[f"value-{j}:work"] == (f"value-{j}:vault", f"value-{j}:prepare")
        assert dag.parents[f"value-{j}:publish"] == (f"value-{j}:work",)
        assert all(dag.lookup[f"value-{j}:"+stage]["cues"] == [0, 0] for stage in ("vault", "prepare", "work", "publish"))
        if phase == "late":
            w = dag.observations[f"value-{j}:work"]["value"]
        else:
            z = coefficient["event:work:bias"] + coefficient["event:work:parent_mean"]*(v+c)/2 + coefficient["event:work:multiple_parents_all"]*v*c
            w = sigmoid(z)
        values.append((1-w)*pub0+w*pub1)
    # Components share coefficients but no random nodes: this factorization is
    # a property of THIS declared graph, not an assumption made for every MAS.
    any_probability = 1-math.prod(1-p for p in values)
    return values, any_probability


def run(out, destination):
    sys.path[:0] = [str(out / "source/code/src"), str(out / "source/code/commands")]
    from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
    from paperalpha_monitor.observable_event_learning import complete_trace_design, complete_trace_objective
    import joblib
    spec = importlib.util.spec_from_file_location("frozen_public_study", out / "source/code/commands/run_agentleak_event_dag.py")
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    plan = frozen.frozen_plan(out)
    report = frozen.read(out / "REPORT.json")
    assert report["plan_sha256"] == frozen.sha(out / "PLAN.json")
    for filename, expected in report["models_sha256"].items():
        assert frozen.sha(out / "models" / filename) == expected
    for filename, expected in report["predictions_sha256"].items():
        assert frozen.sha(out / "predictions" / filename) == expected
    data, projection = frozen.load_data(out, verbose=False)
    assert projection == frozen.read(out / "PROJECTION.json")
    assert frozen.sha(out / "PROJECTION.json") == report["projection_sha256"]
    by_id = {r["id"]: r for r in data}
    train = [r for r in data if r["split"] == "train"]
    predictions, maximum_error, full_joint_checks, gradient_checks = 0, 0., 0, []
    stats, objective_checks = {}, []
    for filename in report["predictions_sha256"]:
        name = filename[:-5]
        records = frozen.read(out / "predictions" / filename)
        if name.startswith("event_dag_"):
            model = EliminatingDAGWorkflowRiskBN.restore(frozen.read(out / "models" / (name + ".json")))
            x, y = complete_trace_design(model.event_law, [r["early"].event_dag for r in train], [r["labels"] for r in train])
            loss = complete_trace_objective(model.theta[:25], x, y, len(train), model.regularization)[0]
            assert abs(loss-model.optimizer["loss"]) < 1e-10
            objective_checks.append(loss)
            independent_loss = 0.
            coeff = dict(zip(model.names, model.theta))
            for row in train:
                dag = row["early"].event_dag
                for j in range(len(row["early"].action_ids)):
                    v = dag.observations[f"value-{j}:vault"]["value"]
                    c = dag.observations[f"value-{j}:prepare"]["value"]
                    w, f = row["labels"][f"value-{j}:work"]["value"], row["labels"][f"value-{j}:publish"]["value"]
                    z_w = coeff["event:work:bias"]+coeff["event:work:parent_mean"]*(v+c)/2+coeff["event:work:multiple_parents_all"]*v*c
                    z_f = coeff["event:publish:bias"]+coeff["event:publish:parent_mean"]*w
                    independent_loss += float(np.logaddexp(0., z_w)-w*z_w + np.logaddexp(0., z_f)-f*z_f)
            independent_loss = independent_loss/len(train)+model.regularization*float(model.theta[:25]@model.theta[:25])
            assert abs(loss-independent_loss) < 1e-10
            for i, record in enumerate(records):
                row = by_id[record["id"]]; sample = row[record["phase"]]
                assert record["input_sha256"] == sample.sha256
                field_p, probability = independent_fields(model, sample, record["phase"])
                maximum_error = max(maximum_error, abs(probability-record["probability"]))
                if i % 400 == 0:
                    full = model.predict(sample)
                    assert abs(full["probabilities"]["system:any"]-probability) < 1e-12
                    assert np.max(np.abs(np.asarray(list(full["appearance_probabilities"].values()))-field_p)) < 1e-12
                    full_joint_checks += 1
                    if i % 1600 == 0:
                        fast = model.predict_known_authorization_union(sample, derivatives=True)
                        for j in range(25):
                            previous = model.theta[j]
                            model.theta[j] = previous+1e-5; plus = independent_fields(model, sample, record["phase"])[1]
                            model.theta[j] = previous-1e-5; minus = independent_fields(model, sample, record["phase"])[1]
                            model.theta[j] = previous
                            gradient_checks.append(abs((plus-minus)/2e-5-fast["gradients"]["system:any"][j]))
                predictions += 1
            largest = max(data, key=lambda row: len(row["early"].action_ids))["early"]
            joint = model.event_law.conditional(largest.event_dag, [q["event_id"] for q in largest.queries.values()], model.theta[:25])
            fast = model.predict_known_authorization_union(largest)
            stats[name] = {"full_target_joint": joint["inference"], "authorization_union": fast["inference"]}
        else:
            saved = {phase: joblib.load(out / "models" / (name+"_"+phase+".joblib")) for phase in ("early", "late")} if name in ("logistic", "histgb") else None
            for phase in ("early", "late"):
                subset = [r for r in records if r["phase"] == phase]
                rows = [by_id[r["id"]] for r in subset]
                if saved:
                    probabilities = saved[phase].predict_proba(np.asarray([frozen.baseline_features(r, phase) for r in rows]))[:, 1]
                elif name == "train_prevalence":
                    probabilities = np.full(len(rows), np.mean([r["y"] for r in train]))
                else:
                    probabilities = [float(frozen.baseline_features(r, phase)[2 if phase == "early" else 4] > 0) for r in rows]
                maximum_error = max(maximum_error, float(np.max(np.abs(probabilities-np.asarray([r["probability"] for r in subset])))))
                predictions += len(subset)
        for split in ("train", "validation"):
            for phase in ("early", "late"):
                subset = [r for r in records if r["split"] == split and r["phase"] == phase]
                assert frozen.metric([r["y"] for r in subset], [r["probability"] for r in subset]) == report["metrics"][name][split+"/"+phase]
                assert all(by_id[r["id"]]["y"] == r["y"] and by_id[r["id"]]["split"] == r["split"] for r in subset)
    assert maximum_error < 1e-10 and max(gradient_checks) < 2e-8
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), destination / "AUDITOR.py")
    result = {"source_workflows_reconstructed": len(data), "prefixes_reconstructed": 2*len(data), "predictions_recomputed": predictions,
        "maximum_prediction_error": maximum_error, "full_retained_joint_checks": full_joint_checks,
        "gradient_directions_against_independent_formula": len(gradient_checks), "maximum_gradient_error": max(gradient_checks),
        "complete_training_objectives_verified": len(objective_checks), "largest_graph_inference": stats,
        "new_api_calls": 0, "new_fit": False, "old_test_read": False,
        "plan_sha256": frozen.sha(out / "PLAN.json"), "report_sha256": frozen.sha(out / "REPORT.json"),
        "auditor_sha256": frozen.sha(destination / "AUDITOR.py")}
    frozen.write(destination / "REPORT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve(), args.destination.resolve())
