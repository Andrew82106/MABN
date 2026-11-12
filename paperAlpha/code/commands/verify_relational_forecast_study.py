"""Independent scalar graph/likelihood and saved-prediction audit, no API."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import logsumexp


def run(out, destination):
    sys.path[:0] = [str(out / "source/code/src"), str(out / "source/code/commands")]
    from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
    from paperalpha_monitor.observable_event_learning import trace_loss_gradient
    from scalar_binary_graph import evaluate
    spec = importlib.util.spec_from_file_location("frozen_relation_study", out / "source/code/commands/run_relational_forecast_study.py")
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    data, collection = frozen.verified_data(out)
    report = frozen.read(out / "fit/REPORT.json")
    assert report["plan_sha256"] == frozen.sha(out / "PLAN.json")
    assert report["predictions_sha256"] == frozen.sha(out / "fit/PREDICTIONS.json")
    for name, expected in report["models_sha256"].items():
        assert frozen.sha(out / "fit/models" / name) == expected
    original = frozen.read(out / "fit/PREDICTIONS.json")
    lookup = {r["case"]["id"]: r for r in data}
    graph_errors, likelihood_errors, gradient_errors = [], [], []
    predictions = 0
    def indicator_probability(nodes, targets, bits):
        # This output truth table selects a joint outcome; the scalar interpreter
        # independently conditions on all original prefix-observed nodes.
        indicator = {"id": "audit:outcome", "kind": "binary_cpt", "parents": list(targets),
            "probability_one": [float(tuple(state) == tuple(bits)) for state in product((0, 1), repeat=len(bits))]}
        return evaluate({"nodes": [*nodes, indicator], "output_id": indicator["id"]})
    for name in report["models_sha256"]:
        if not name.startswith("dag_"):
            continue
        model_name = name[:-5]
        model = DAGWorkflowRiskBN.restore(frozen.read(out / "fit/models" / name))
        for record in original[model_name]:
            sample = lookup[record["id"]]["samples"][record["phase"]]
            assert sample.sha256 == record["input_sha256"]
            replay = model.predict(sample)["probabilities"]
            assert replay == record["policy"]
            joint = model.event_law.conditional(sample.event_dag, ["publish-a", "publish-b"], model.theta[:25])
            assert np.max(np.abs(np.exp(joint["log_probabilities"]) - record["joint"])) < 1e-14
            for query, probability in replay.items():
                graph_errors.append(abs(evaluate(model.probability_graph(sample, output=query)) - probability))
            nodes = model.event_law.graph_nodes(sample.event_dag, model.theta[:25], {n: n for n in sample.event_dag.ids})
            for index, bits in enumerate(product((0, 1), repeat=2)):
                graph_errors.append(abs(indicator_probability(nodes, ["publish-a", "publish-b"], bits) - record["joint"][index]))
            predictions += 1
        train = [r for r in data if r["case"]["split"] == "train"]
        dags, labels, losses = [], [], []
        for row in train:
            dag = row["samples"]["early"].event_dag
            identities = [n for n in dag.order if n in row["labels"] and ("all_future" in model_name or n.startswith("publish-"))]
            nodes = model.event_law.graph_nodes(dag, model.theta[:25], {n: n for n in dag.ids})
            probability = indicator_probability(nodes, identities, [row["labels"][n]["value"] for n in identities])
            losses.append(-math.log(probability))
            dags.append(dag); labels.append({n: row["labels"][n] for n in identities})
        objective = sum(losses) / len(losses) + model.regularization * float(model.theta[:25] @ model.theta[:25])
        likelihood_errors.append(abs(objective - model.optimizer["loss"]))
        _, grad = trace_loss_gradient(model.event_law, model.theta[:25], dags, labels, model.regularization)
        for i in range(25):
            d = np.zeros(25); d[i] = 1e-5
            finite = (trace_loss_gradient(model.event_law, model.theta[:25] + d, dags, labels, model.regularization)[0] -
                      trace_loss_gradient(model.event_law, model.theta[:25] - d, dags, labels, model.regularization)[0]) / 2e-5
            gradient_errors.append(abs(finite - grad[i]))
    # Recompute the remaining saved methods, not merely trust their score table.
    for name, records in original.items():
        if name.startswith("dag_") and not name.endswith("_independent"):
            continue
        for record in records:
            row = lookup[record["id"]]; sample = row["samples"][record["phase"]]
            if name.endswith("_independent"):
                peer = next(r for r in original[name[:-12]] if r["id"] == record["id"] and r["phase"] == record["phase"])
                p = peer["joint"]; a, b = p[2] + p[3], p[1] + p[3]
                expected = [(1-a)*(1-b), (1-a)*b, a*(1-b), a*b]
            elif name.startswith("direct_joint_softmax"):
                saved = frozen.read(out / "fit/models" / (name + ".json"))[record["phase"]]
                z = np.asarray(frozen.features(row, record["phase"])) @ np.asarray(saved["coef"])
                expected = np.exp(z - logsumexp(z))
            else:
                dag = sample.event_dag
                bits = []
                for target in ("publish-a", "publish-b"):
                    parent = dag.parents[target][0]
                    value = dag.observations[parent]["value"] if parent in dag.observations else dag.observations["source"]["value"] * (1-dag.lookup[parent]["cues"][1])
                    bits.append(value if name == "preserve_unless_removed" else 0)
                expected = [float(i == 2*bits[0]+bits[1]) for i in range(4)]
            assert np.max(np.abs(np.asarray(expected) - record["joint"])) < 1e-12
            predictions += 1
    for name, records in original.items():
        for split in ("train", "validation"):
            for phase in ("early", "late"):
                subset = [r for r in records if r["split"] == split and r["phase"] == phase]
                assert frozen.metrics(subset) == report["metrics"][name][split + "/" + phase]
                # Separate plain scalar check of the two main proper scores.
                n = len(subset)
                joint = sum(sum((p - int(k == r["label"])) ** 2 for k, p in enumerate(r["joint"])) for r in subset) / n
                risk = sum((1 - r["joint"][0] - int(r["label"] != 0)) ** 2 for r in subset) / n
                assert abs(joint - report["metrics"][name][split + "/" + phase]["joint_brier"]) < 1e-12
                assert abs(risk - report["metrics"][name][split + "/" + phase]["any_brier"]) < 1e-12
    assert max(graph_errors) < 1e-12 and max(likelihood_errors) < 1e-10 and max(gradient_errors) < 2e-8
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), destination / "AUDITOR.py")
    result = {**collection, "predictions_replayed": predictions, "graph_queries": len(graph_errors),
        "maximum_graph_error": max(graph_errors), "training_objectives_scalar_verified": len(likelihood_errors),
        "maximum_objective_error": max(likelihood_errors), "gradient_directions": len(gradient_errors),
        "maximum_gradient_error": max(gradient_errors), "new_api_calls": 0, "new_fit": False,
        "plan_sha256": frozen.sha(out / "PLAN.json"), "report_sha256": frozen.sha(out / "fit/REPORT.json"),
        "auditor_sha256": frozen.sha(destination / "AUDITOR.py")}
    frozen.write(destination / "REPORT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve(), args.destination.resolve())
