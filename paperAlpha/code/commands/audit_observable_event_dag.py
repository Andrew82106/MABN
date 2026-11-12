"""Independent graph likelihoods and conditional gradients, without API calls."""
from __future__ import annotations

import argparse
from itertools import product
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands'), str(ROOT / 'code/tests')]
from fit_observable_event_dag import prepare as prepare_fit, read, write, sha, verify_plan, dataset, labels, bn_records
from paperalpha_monitor.observable_event_dag import DAGWorkflowRiskBN
from scalar_binary_graph import evaluate
from test_observable_event_dag import coupled_tool_fixture, missing_past_fixture


def prepare(reference, out):
    report = read(reference / "REPORT.json")
    prepare_fit(reference / "inputs", out)
    for name, expected in report["models_sha256"].items():
        assert sha(reference / "models" / name) == expected
        destination = out / "reference/models" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / "models" / name, destination)
    for name in ("REPORT.json", "PREDICTIONS.json", "PLAN.json"):
        shutil.copyfile(reference / name, out / "reference" / name)
    write(out / "REFERENCE.json", {"directory": str(reference), "files_sha256": {str(p.relative_to(out / "reference")): sha(p) for p in (out / "reference").rglob("*") if p.is_file()}})


def run(out):
    verify_plan(out)
    for name, expected in read(out / "REFERENCE.json")["files_sha256"].items():
        assert sha(out / "reference" / name) == expected
    rows, branches, samples = dataset(out)
    reference_report, reference_predictions = read(out / "reference/REPORT.json"), read(out / "reference/PREDICTIONS.json")
    graph_errors, likelihood_errors, gradient_errors, constructed = [], [], [], []
    for c in reference_report["candidates"]:
        if c["kind"] != "bn":
            continue
        model = DAGWorkflowRiskBN.restore(read(out / "reference/models" / (c["name"] + ".json")))
        variant = c["variant"]
        for split in rows:
            actual = bn_records(model, samples[split][variant], rows[split])
            assert actual == reference_predictions[c["name"]][split]
            for sample, record in zip(samples[split][variant], actual):
                for query, p in record["probabilities"].items():
                    graph_errors.append(abs(evaluate(model.probability_graph(sample, output=query)) - p))
        independent_losses = []
        for sample, row in zip(samples["train"][variant], rows["train"]):
            target = tuple(row["outcome"]["appearance"])
            graph = model.probability_graph(sample)
            graph["nodes"].append({"id": "observed_future_label_query", "parents": [model._event_node_id(a) for a in sample.action_ids],
                "kind": "binary_cpt", "probability_one": [float(bits == target) for bits in product((0, 1), repeat=len(target))]})
            graph["output_id"] = "observed_future_label_query"
            independent_losses.append(-math.log(evaluate(graph)))
        independent = float(np.mean(independent_losses) + model.regularization * (model.theta @ model.theta))
        production = model.loss_gradient(model.theta, samples["train"][variant], labels(rows["train"], samples["train"][variant]))[0]
        likelihood_errors.append(abs(independent - production))
    mechanisms = [coupled_tool_fixture(n=n, layout=layout) for layout, n in [("fork", 1), ("fork", 2), ("fork", 3), ("fork", 5), ("chain", 2), ("chain", 4)]]
    mechanisms += [coupled_tool_fixture(authority=a) for a in ("invalid", "root", "delegation", "opaque")]
    mechanisms.append(missing_past_fixture())
    for index, (sample, model) in enumerate(mechanisms):
        conditions = {sample.action_ids[0]: 1}
        result = model.predict(sample, derivatives=True, event_conditions=conditions)
        for query, p in result["probabilities"].items():
            graph_errors.append(abs(evaluate(model.probability_graph(sample, output=query, event_conditions=conditions)) - p))
        for j in range(len(model.theta)):
            delta = np.zeros_like(model.theta); delta[j] = 1e-5
            numeric = (model.predict(sample, theta=model.theta + delta, event_conditions=conditions)["probabilities"]["system:any"] -
                       model.predict(sample, theta=model.theta - delta, event_conditions=conditions)["probabilities"]["system:any"]) / 2e-5
            gradient_errors.append(abs(numeric - result["gradients"]["system:any"][j]))
        constructed.append({"index": index, "input": sample.specification(), "model": model.parameters(), "conditions": conditions,
                            "probabilities": result["probabilities"], "data_kind": "constructed-mechanism-not-api-performance"})
    assert max(graph_errors) < 1e-12 and max(likelihood_errors) < 1e-12 and max(gradient_errors) < 2e-8
    write(out / "CONSTRUCTED_QUERIES.json", constructed)
    result = {"real_development_prefixes": 16, "frozen_candidate_bn_models_replayed": 4,
        "independent_training_likelihoods": 32, "complete_regularized_training_objectives": len(likelihood_errors),
        "maximum_independent_objective_error": max(likelihood_errors), "constructed_inputs": len(constructed),
        "total_graph_queries": len(graph_errors), "maximum_graph_error": max(graph_errors),
        "conditional_gradient_directions": len(gradient_errors), "maximum_gradient_error": max(gradient_errors),
        "old_test_read": False, "new_api_calls": 0, "new_fit": False,
        "plan_sha256": sha(out / "PLAN.json"), "reference_manifest_sha256": sha(out / "REFERENCE.json"),
        "constructed_queries_sha256": sha(out / "CONSTRUCTED_QUERIES.json")}
    write(out / "REPORT.json", result)
    print(__import__("json").dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.reference.resolve(), args.out.resolve())
    else: run(args.out.resolve())
