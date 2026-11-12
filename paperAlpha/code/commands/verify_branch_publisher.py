"""Audit frozen development fits, selections, likelihood and executable graphs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np


def verify(run, out):
    if out.exists():
        raise ValueError("new audit directory required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from collect_privacy_transfer_api import sha, write
    from train_branch_publisher import load_data, restore_bn
    from paperalpha_monitor.branch_benchmark import classifier_distribution, rule_distribution
    from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
    from scalar_binary_graph import evaluate
    from branch_graph_evidence import evidence_probability
    import paperalpha_monitor.branch_publisher_bn as module
    assert Path(module.__file__).resolve() == run / "source_snapshot/code/src/paperalpha_monitor/branch_publisher_bn.py"
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        assert sha(run / name) == expected, name
    rows, samples = load_data(run)
    fs, auth = [r["outcome"]["appearance"] for r in rows["train"]], [r["outcome"]["authority_failure"] for r in rows["train"]]
    candidates, losses, gradients, evidence_errors, graph_errors = [], [], [], [], []
    source_objects, recomputed_scores = {}, 0
    for path in sorted((run / "candidates").glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        candidates.append(saved)
        family = saved["family"]
        obj = restore_bn(run / saved["file"]) if saved["file"] and saved["file"].endswith(".json") else (
            joblib.load(run / saved["file"]) if saved["file"] else None)
        source_objects[saved["id"]] = obj
        if family in plan["protocol"]["bn_families"]:
            xs = samples["train"] if isinstance(obj, BranchPublisherPrivacyBN) else [s.release for s in samples["train"]]
            loss, grad = obj.loss_gradient(obj.theta, xs, fs, auth)
            losses.append(abs(loss - saved["optimizer"]["loss"]))
            # Independent scalar summation of every factor, including observed
            # W likelihood, not merely differentiating the same vector loss.
            scalar_losses = []
            for sample, final, authority in zip(xs, fs, auth):
                evidence = {**{f"F:{i}": int(f) for i, f in enumerate(final)}, **{f"A:{i}": int(a) for i, a in enumerate(authority)}}
                probability = evidence_probability(obj.probability_graph(sample), evidence)
                scalar_losses.append(-np.log(probability))
            scalar_loss = np.mean(scalar_losses) + obj.regularization * (obj.theta @ obj.theta)
            evidence_errors.append(abs(scalar_loss - loss))
            for seed in (11, 29):
                direction = np.random.default_rng(seed).normal(size=len(obj.theta))
                direction /= np.linalg.norm(direction)
                eps = 1e-5
                diff = (obj.loss_gradient(obj.theta + eps * direction, xs, fs, auth)[0]
                        - obj.loss_gradient(obj.theta - eps * direction, xs, fs, auth)[0]) / (2 * eps)
                gradients.append(abs(diff - grad @ direction))
        for split, subset in samples.items():
            actual = []
            for sample in subset:
                if obj is None:
                    dist = rule_distribution(sample, family)
                elif saved["file"].endswith(".joblib"):
                    dist = classifier_distribution(sample, obj)
                else:
                    bound = sample if isinstance(obj, BranchPublisherPrivacyBN) else sample.release
                    dist = obj.predict(bound)["violation_count_probabilities"]
                actual.append(dist)
                recomputed_scores += 1
            actual = np.asarray(actual)
            assert np.max(abs(actual - np.asarray(saved["distributions"][split]))) < 1e-14
            for k in (1, 2):
                score = actual[:, k:].sum(axis=1)
                labels = np.array([r["outcome"]["targets"][str(k)] for r in rows[split]])
                brier = float(np.mean((score - labels) ** 2))
                assert abs(brier - saved["metrics"][split][str(k)]["brier"]) < 1e-14
                clipped = np.clip(score, 1e-12, 1 - 1e-12)
                nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
                assert abs(nll - saved["metrics"][split][str(k)]["nll_clipped_1e_12"]) < 1e-13
    selected = json.loads((run / "SELECTED.json").read_text(encoding="utf-8"))["selected"]
    independent_selection = {}
    for family in sorted({c["family"] for c in candidates}):
        eligible = [c for c in candidates if c["family"] == family]
        if family in plan["protocol"]["bn_families"]:
            starts = {}
            for c in eligible:
                if not c["optimizer"]["success"]:
                    continue
                key = c["lambda"]
                if key not in starts or (c["optimizer"]["loss"], c["id"]) < (starts[key]["optimizer"]["loss"], starts[key]["id"]):
                    starts[key] = c
            eligible = list(starts.values())
        independent_selection[family] = sorted(eligible, key=lambda c: (c["metrics"]["validation"]["1"]["brier"], c["id"]))[0]["id"]
    assert independent_selection == selected
    for identity in selected.values():
        obj = source_objects[identity]
        if obj is None or not hasattr(obj, "probability_graph"):
            continue
        for subset in samples.values():
            for sample in subset:
                bound = sample if isinstance(obj, BranchPublisherPrivacyBN) else sample.release
                for k in (1, 2):
                    graph_errors.append(abs(obj.predict(bound, at_least=k)["model_probability"] - evaluate(obj.probability_graph(bound, at_least=k))))
    assert max(losses) < 1e-10 and max(evidence_errors) < 1e-10 and max(gradients) < 2e-7 and max(graph_errors) < 1e-12
    result = {"report_sha256": sha(run / "REPORT.json"), "auditor_sha256": sha(Path(__file__)),
        "artifacts_checked": len(report["artifacts_sha256"]), "candidates": len(candidates), "predictions_recomputed": recomputed_scores,
        "objectives_recomputed": len(losses), "max_objective_error": max(losses),
        "scalar_joint_likelihood_checks": len(evidence_errors), "max_scalar_joint_likelihood_error": max(evidence_errors),
        "directional_gradients": len(gradients), "max_directional_gradient_error": max(gradients),
        "executable_graphs": len(graph_errors), "max_executable_graph_error": max(graph_errors),
        "selection_recomputed": True, "test_outputs_used": False,
        "scope": "Frozen development pilot, no refit/API call, no independent test or calibration claim"}
    write(out / "REPORT.json", result)
    shutil.copyfile(Path(__file__), out / "auditor_source.py")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
