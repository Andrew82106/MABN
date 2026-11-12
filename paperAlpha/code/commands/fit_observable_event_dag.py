"""Frozen development-only fit and executable-graph verification.

Only the existing train/validation prefixes are consumed. No API, no held-out
read, no inherited training advantage. Direct logistic equivalence is reported
instead of claiming a new algorithm merely because a classifier is a BN.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_monitor.branch_benchmark import sample_from_prefix, rule_distribution
from paperalpha_monitor.event_dag_projection import from_verified_branch
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, DAGWorkflowInput, DAGWorkflowRiskBN
from paperalpha_monitor.scoped_privacy_monitor import gated_distribution
from scalar_binary_graph import evaluate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        file.write("\n")


def prepare(inputs, out):
    out.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/fit_observable_event_dag.py', 'code/commands/audit_observable_event_dag.py', 'code/commands/scalar_binary_graph.py',
              'code/commands/branch_publisher_fixtures.py', 'code/commands/workflow_risk_fixtures.py', 'code/tests/test_observable_event_dag.py']
    for name in names:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    for split in ("train", "validation"):
        dest = out / "inputs" / (split + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / (split + ".json"), dest)
    write(out / "PLAN.json", {"schema": "observable-event-dag-development-fit-v1", "input_directory": str(inputs.resolve()),
        "sources_sha256": {n: sha(out / "source" / n) for n in sorted(names)},
        "inputs_sha256": {s: sha(out / "inputs" / (s + ".json")) for s in ("train", "validation")},
        "regularization": [.001, .03], "initialization": "zero for all 28 BN coefficients; no historical trained initialization",
        "max_iter": 500, "variants": ["delivery_dag", "without_delivery_edges"],
        "direct_control": "Identical observed-parent logistic design, fit_intercept=False, C=1/(2*lambda*training_flow_count), separate sklearn solver",
        "selection": "Report every prespecified candidate; no new default or held-out selection",
        "scope": "Existing 8 train / 8 validation API flows. Validation has been used previously. Not new independent performance or arbitrary-topology empirical validation.",
        "old_test_read": False, "new_api_calls": 0, "new_risk_labels": False})
    print(json.dumps({"prepared": str(out), "sources": len(names), "plan_sha256": sha(out / "PLAN.json")}))


def verify_plan(out):
    plan = read(out / "PLAN.json")
    if ROOT != (out / "source").resolve():
        raise ValueError("execute the frozen fit/verification script")
    for name, expected in plan["sources_sha256"].items():
        assert sha(out / "source" / name) == expected
    for split, expected in plan["inputs_sha256"].items():
        assert sha(out / "inputs" / (split + ".json")) == expected
    return plan


def no_edges(sample):
    spec = sample.event_dag.specification(); spec["edges"] = []
    dag = ObservableEventDAG(**spec)
    proposals = copy.deepcopy(sample.proposals)
    for p in proposals:
        if sample.event_semantics[p["action_id"]] == "literal_appearance":
            p["evidence"]["event_dag_sha256"] = dag.sha256
    return DAGWorkflowInput(event_dag=dag, queries=sample.queries, workflow=sample.workflow, authority=sample.authority,
        proposals=proposals, contracts=sample.contracts, combinations=sample.combinations,
        cutoff=sample.cutoff, horizon=sample.horizon, event_semantics=sample.event_semantics)


def dataset(out):
    rows, branches, samples = {}, {}, {}
    seen = set()
    for split in ("train", "validation"):
        rows[split] = read(out / "inputs" / (split + ".json"))
        assert len(rows[split]) == 8 and all(r["split"] == split for r in rows[split])
        for row in rows[split]:
            assert row["id"] not in seen; seen.add(row["id"])
        branches[split] = [sample_from_prefix(r["prefix"]) for r in rows[split]]
        samples[split] = {"delivery_dag": [from_verified_branch(b) for b in branches[split]]}
        samples[split]["without_delivery_edges"] = [no_edges(s) for s in samples[split]["delivery_dag"]]
    assert not ({r["template_id"] for r in rows["train"]} & {r["template_id"] for r in rows["validation"]})
    return rows, branches, samples


def labels(rows, samples):
    return [{"events": dict(zip(s.action_ids, r["outcome"]["appearance"]))} for r, s in zip(rows, samples)]


def direct_features(sample, model):
    result = []
    for aid in sample.action_ids:
        identity = sample.queries[aid]["event_id"]
        dag = sample.event_dag
        # This exact control applies only when all target parents are already
        # observed. It is NOT a substitute for marginalizing future parents.
        assert set(dag.parents[identity]) <= set(dag.observations)
        parents = [dag.observations[p]["value"] for p in dag.parents[identity]]
        result.append(model.event_law.design(dag.lookup[identity], parents))
    return np.asarray(result)


def bn_records(model, subset, rows):
    return [{"id": r["id"], "input_sha256": s.sha256,
             "probabilities": model.predict(s)["probabilities"],
             "appearance_probabilities": model.predict(s)["appearance_probabilities"]} for r, s in zip(rows, subset)]


def metric(predictions, rows):
    p = np.array([p["probabilities"]["system:any"] for p in predictions])
    y = np.array([r["outcome"]["targets"]["1"] for r in rows])
    positive = p >= .5
    tp, fp, fn = int(np.sum(positive & (y == 1))), int(np.sum(positive & (y == 0))), int(np.sum(~positive & (y == 1)))
    return {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p - y) ** 2)), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def direct_records(coef, branches, samples, rows, model):
    result = []
    for branch, s, row in zip(branches, samples, rows):
        q = expit(direct_features(s, model) @ coef)
        distribution = gated_distribution(branch.release, q, .5)
        result.append({"id": row["id"], "probabilities": {"system:any": float(distribution[1:].sum())},
                       "appearance_probabilities": dict(zip(s.action_ids, q.tolist()))})
    return result


def run(out):
    plan = verify_plan(out)
    rows, branches, samples = dataset(out)
    candidates, predictions = [], {}
    for variant in plan["variants"]:
        for penalty in plan["regularization"]:
            name = f"{variant}_lambda{penalty}"
            model = DAGWorkflowRiskBN(regularization=penalty)
            train = samples["train"][variant]
            model.fit(train, labels(rows["train"], train), max_iter=plan["max_iter"])
            write(out / "models" / (name + ".json"), model.parameters())
            pred = {s: bn_records(model, samples[s][variant], rows[s]) for s in rows}
            predictions[name] = pred
            candidates.append({"name": name, "kind": "bn", "variant": variant, "lambda": penalty, "optimizer": model.optimizer,
                               "metrics": {s: metric(pred[s], rows[s]) for s in rows}})
            print(json.dumps({"fitted": name, "optimizer": model.optimizer, "validation": candidates[-1]["metrics"]["validation"]}), flush=True)
    prototype = DAGWorkflowRiskBN()
    x = np.concatenate([direct_features(s, prototype) for s in samples["train"]["delivery_dag"]])
    y = np.concatenate([r["outcome"]["appearance"] for r in rows["train"]])
    for penalty in plan["regularization"]:
        name = f"direct_logistic_lambda{penalty}"
        clf = LogisticRegression(C=1 / (2 * penalty * len(rows["train"])), fit_intercept=False, tol=1e-10, max_iter=1000).fit(x, y)
        coef = clf.coef_[0]
        write(out / "models" / (name + ".json"), {"name": name, "kind": "direct_logistic", "coefficient_names": list(prototype.event_law.names),
                                                  "coefficient": coef.tolist(), "C": clf.C, "iterations": clf.n_iter_.tolist()})
        pred = {s: direct_records(coef, branches[s], samples[s]["delivery_dag"], rows[s], prototype) for s in rows}
        predictions[name] = pred
        candidates.append({"name": name, "kind": "direct_logistic", "lambda": penalty, "metrics": {s: metric(pred[s], rows[s]) for s in rows}})
    for rule in ("visible_compliance_shared_gate", "persistence_shared_gate"):
        pred = {s: [{"id": row["id"], "probabilities": {"system:any": float(rule_distribution(b, rule)[1:].sum())}} for b, row in zip(branches[s], rows[s])] for s in rows}
        predictions[rule] = pred
        candidates.append({"name": rule, "kind": "rule", "metrics": {s: metric(pred[s], rows[s]) for s in rows}})
    equivalence = {}
    for penalty in plan["regularization"]:
        differences = [abs(a["probabilities"]["system:any"] - b["probabilities"]["system:any"])
            for split in rows for a, b in zip(predictions[f"delivery_dag_lambda{penalty}"][split], predictions[f"direct_logistic_lambda{penalty}"][split])]
        equivalence[str(penalty)] = max(differences)
    write(out / "PREDICTIONS.json", predictions)
    active = np.flatnonzero(np.any(x != 0, axis=0))
    report = {"candidates": candidates, "direct_logistic_max_prediction_difference": equivalence,
        "training_field_design_shape": list(x.shape), "training_field_design_rank": int(np.linalg.matrix_rank(x)),
        "nonzero_design_columns": [prototype.event_law.names[i] for i in active],
        "constant_parent_means": sorted(set(x[:, prototype.event_law.names.index("event:publish:parent_mean")].tolist())),
        "all_target_parents_observed": True, "interpretation": "Observed-parent conditional BN reduces to logistic prediction plus shared policies. Nonzero parent coefficients with a constant parent mean do not identify a propagation effect.",
        "old_test_read": False, "new_api_calls": 0, "new_default_selected": False,
        "plan_sha256": sha(out / "PLAN.json"), "predictions_sha256": sha(out / "PREDICTIONS.json"),
        "models_sha256": {p.name: sha(p) for p in sorted((out / "models").glob("*.json"))}}
    write(out / "REPORT.json", report)
    print(json.dumps({"completed": True, "candidates": len(candidates), "logistic_equivalence": equivalence, "rank": report["training_field_design_rank"]}))


def verify(out, destination):
    verify_plan(out)
    report = read(out / "REPORT.json")
    assert report["plan_sha256"] == sha(out / "PLAN.json") and report["predictions_sha256"] == sha(out / "PREDICTIONS.json")
    for name, expected in report["models_sha256"].items():
        assert sha(out / "models" / name) == expected
    rows, branches, samples = dataset(out)
    original = read(out / "PREDICTIONS.json")
    graphs, differences, losses, restored_count = 0, [], [], 0
    for candidate in report["candidates"]:
        name = candidate["name"]
        if candidate["kind"] == "bn":
            model = DAGWorkflowRiskBN.restore(read(out / "models" / (name + ".json")))
            variant = candidate["variant"]
            pred = {s: bn_records(model, samples[s][variant], rows[s]) for s in rows}
            loss, grad = model.loss_gradient(model.theta, samples["train"][variant], labels(rows["train"], samples["train"][variant]))
            assert abs(loss - candidate["optimizer"]["loss"]) < 1e-12
            losses.append(loss)
            restored_count += 1
            for split in rows:
                for sample, record in zip(samples[split][variant], pred[split]):
                    for query, probability in record["probabilities"].items():
                        differences.append(abs(evaluate(model.probability_graph(sample, output=query)) - probability)); graphs += 1
        elif candidate["kind"] == "direct_logistic":
            coefficients = np.array(read(out / "models" / (name + ".json"))["coefficient"])
            pred = {s: direct_records(coefficients, branches[s], samples[s]["delivery_dag"], rows[s], DAGWorkflowRiskBN()) for s in rows}
        else:
            pred = {s: [{"id": row["id"], "probabilities": {"system:any": float(rule_distribution(b, name)[1:].sum())}} for b, row in zip(branches[s], rows[s])] for s in rows}
        assert pred == original[name]
        assert {s: metric(pred[s], rows[s]) for s in rows} == candidate["metrics"]
    assert max(differences) < 1e-12
    write(destination, {"candidates_recomputed": len(report["candidates"]), "predictions_recomputed": sum(len(v) for p in original.values() for v in p.values()),
        "graphs_recomputed": graphs, "maximum_graph_error": max(differences), "training_losses_recomputed": losses,
        "models_restored": restored_count, "new_api_calls": 0, "old_test_read": False,
        "report_sha256": sha(out / "REPORT.json"), "plan_sha256": sha(out / "PLAN.json")})
    print(json.dumps({"verified": True, "candidates": len(report["candidates"]), "graphs": graphs, "max_graph_error": max(differences)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "verify"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.inputs.resolve(), args.out.resolve())
    elif args.mode == "run": run(args.out.resolve())
    else: verify(args.out.resolve(), args.verification.resolve())
