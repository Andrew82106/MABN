"""Freeze/replay joint appearance-policy integration, without new test/API data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from branch_workflow_fixtures import fixture
from scalar_binary_graph import evaluate
from branch_graph_evidence import evidence_probability
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.branch_workflow_risk import BranchWorkflowRiskBN, BranchWorkflowInput
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def freeze(out, fit):
    out.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((ROOT / 'code/src/paperalpha_monitor').glob("*.py")), *sorted((ROOT / 'code/src/paperalpha_runtime').glob("*.py")),
        *sorted((ROOT / 'knowledge/templates').glob("*.json")), ROOT / 'knowledge/risks/risk_catalog.json',
        *[ROOT / 'code/commands' / name for name in ("audit_branch_workflow_risk.py", "branch_workflow_fixtures.py", "branch_publisher_fixtures.py", "scalar_binary_graph.py", "branch_graph_evidence.py")],
        ROOT / 'code/tests/test_branch_workflow_risk.py']
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = sha(target)
    freeze_plan = read(fit / "FROZEN.json")
    entry = freeze_plan["entries"]["branch"]
    fit_plan, fit_report = read(fit / "PLAN.json"), read(fit / "REPORT.json")
    if sha(fit / "PLAN.json") != freeze_plan["files_sha256"]["PLAN.json"]:
        raise ValueError("historical fit plan changed")
    candidate_path = "candidates/" + entry["id"] + ".json"
    if sha(fit / candidate_path) != fit_report["artifacts_sha256"][candidate_path]:
        raise ValueError("historical selected candidate changed")
    if sha(fit / "FROZEN.json") != fit_report["artifacts_sha256"]["FROZEN.json"]:
        raise ValueError("historical frozen selection changed")
    inputs = {entry["file"]: freeze_plan["files_sha256"][entry["file"]], candidate_path: sha(fit / candidate_path),
              "inputs/train.json": fit_plan["input_sha256"]["inputs/train.json"],
              "inputs/validation.json": fit_plan["input_sha256"]["inputs/validation.json"]}
    for name, expected in inputs.items():
        if sha(fit / name) != expected:
            raise ValueError("historical development input/model changed")
        target = out / "historical" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit / name, target)
    write(out / "PLAN.json", {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
        "historical_sha256": inputs, "selected_entry": entry, "candidate": candidate_path,
        "new_api_calls": 0, "heldout_test_read": False,
        "scope": "Component integration and old development-prefix replay; not new performance evaluation"})


def constructed_model():
    branch = BranchPublisherPrivacyBN(regularization=0.)
    branch.theta += np.random.default_rng(409).normal(0, .15, len(branch.theta))
    branch.theta[branch.presence.boost_start:branch.presence.boost_start + 2] = [2., 3.]
    branch.theta[branch.n - 1] = 1.2
    result = BranchWorkflowRiskBN(branch_model=branch, regularization=0.)
    result.theta[result.branch_count + 1] = logit(.4)
    return result


def run(out):
    if Path(__file__).resolve() != out / "source/code/commands/audit_branch_workflow_risk.py":
        raise ValueError("run the frozen audit entrypoint")
    plan = read(out / "PLAN.json")
    for name, expected in plan["source_sha256"].items():
        assert sha(out / "source" / name) == expected
    for name, expected in plan["historical_sha256"].items():
        assert sha(out / "historical" / name) == expected
    model = constructed_model()
    graph_count, mechanism_count, max_graph, max_likelihood = 0, 0, 0., 0.
    mechanism_rows, gaps = [], []
    for current, payment, shared in product(("valid", "revoked", "root_unknown", "delegation_unknown", "opaque"), (False, True), (False, True)):
        sample = fixture(current=current, payment=payment, shared=shared)
        detail = model.predict(sample)
        for key, probability in detail["probabilities"].items():
            scalar = evaluate(model.probability_graph(sample, output=key))
            max_graph = max(max_graph, abs(scalar - probability))
            graph_count += 1
        label = {"events": {"release-0": 1, "release-1": 0}}
        if payment:
            label["events"]["payment"] = 1
        graph = model.probability_graph(sample)
        conditional = evidence_probability(graph, {"event:" + k: v for k, v in label["events"].items()}) / evidence_probability(graph, {})
        loss, _ = model.loss_gradient(model.theta, [sample], [label])
        max_likelihood = max(max_likelihood, abs(loss + np.log(conditional)))
        law = model.branch_model.joint_appearance_law(sample.branch, theta=model.theta[:model.branch_count])
        marginal = law["probabilities"] @ law["states"]
        gap = float(law["probabilities"][-1] - np.prod(marginal))
        gaps.append(gap)
        mechanism_rows.append({"current": current, "payment": payment, "shared": shared, "input": sample.specification(),
            "probabilities": detail["probabilities"], "appearance_joint": law["probabilities"].tolist(), "cooccurrence_minus_product": gap})
        mechanism_count += 1
    samples = [fixture(current=kind, payment=True, visible=visible) for kind, visible in
               (("root_unknown", "supported"), ("delegation_unknown", "contradicted"), ("opaque", "unresolved"))]
    labels = [{"events": {"release-0": 1, "release-1": 0, "payment": 0}, "risks": {"authorization:release-0": 1}} for _ in samples]
    _, gradient = model.loss_gradient(model.theta, samples, labels)
    numeric = approx_derivative(lambda t: model.loss_gradient(t, samples, labels)[0], model.theta).ravel()
    max_gradient = float(np.max(np.abs(gradient - numeric)))
    prediction_gradient = model.predict(samples[0], derivatives=True)["gradients"]["system:any"]
    numeric = approx_derivative(lambda t: model.predict(samples[0], theta=t)["probabilities"]["system:any"], model.theta).ravel()
    max_prediction_gradient = float(np.max(np.abs(prediction_gradient - numeric)))
    # Joint objective optimization on constructed frequencies only.
    population = model._worlds(samples[0], model.theta)
    labels, weights = [], []
    for i, bits in enumerate(population["states"]):
        for j, facts in enumerate(population["fact_states"]):
            labels.append({"events": dict(zip(samples[0].action_ids, map(int, bits))), "facts": dict(zip(population["fact_ids"], map(int, facts)))})
            weights.append(float(np.exp(population["log_mass"][i, j])))
    learner = constructed_model()
    learner.theta += np.random.default_rng(180).normal(0, .25, len(learner.theta))
    learner.fit([samples[0]] * len(labels), labels, weights=weights, max_iter=300)
    recovery = float(np.max(np.abs(np.exp(learner._worlds(samples[0], learner.theta)["log_mass"]) - np.exp(population["log_mass"]))))
    assert learner.optimizer["success"] and recovery < 1e-5
    # Replay only ORIGINAL TRAIN/VALIDATION prefixes; no new policy labels are
    # invented for them and no parameter is refitted using their outcomes.
    branch = BranchPublisherPrivacyBN.restore(read(out / "historical" / plan["selected_entry"]["file"]))
    actual = BranchWorkflowRiskBN(branch_model=branch, regularization=0.)
    candidate = read(out / "historical" / plan["candidate"])
    replay_rows, max_replay, max_original, max_real_covariance = [], 0., 0., 0.
    for split in ("train", "validation"):
        rows = read(out / "historical/inputs" / (split + ".json"))
        assert len(rows) == 8
        for i, row in enumerate(rows):
            assert row["split"] == split
            prefix = row["prefix"]
            observation = BranchPrivacyInput(prefix["observation"], available_journal_sequence=prefix["seq"], prefix_journal_sequence=prefix["seq"])
            linked = BranchWorkflowInput.from_branch(observation)
            result = actual.predict(linked)
            p = result["probabilities"]["system:any"]
            old = float(sum(candidate["distributions"][split][i][1:]))
            current = branch.predict(observation)["model_probability"]
            max_replay, max_original = max(max_replay, abs(p - old)), max(max_original, abs(p - current))
            scalar = evaluate(actual.probability_graph(linked))
            max_graph = max(max_graph, abs(scalar - p))
            graph_count += 1
            law = branch.joint_appearance_law(observation)
            marginal = law["probabilities"] @ law["states"]
            covariance = float(law["probabilities"][-1] - np.prod(marginal))
            max_real_covariance = max(max_real_covariance, abs(covariance))
            replay_rows.append({"id": row["id"], "split": split, "prefix_sequence": prefix["seq"], "input_sha256": linked.sha256,
                "original_frozen_prediction": old, "composed_prediction": p, "scalar_prediction": scalar,
                "appearance_joint": law["probabilities"].tolist(), "cooccurrence_minus_product": covariance,
                "new_combination_policies": 0, "new_tool_events": 0})
    assert max(max_graph, max_likelihood, max_replay, max_original) < 2e-12
    assert max(max_gradient, max_prediction_gradient) < 5e-7
    report = {"constructed_inputs": mechanism_count, "graph_queries_including_real_replay": graph_count,
        "max_graph_error": max_graph, "max_conditional_likelihood_error": max_likelihood,
        "max_all_parameter_gradient_error": max_gradient, "max_prediction_gradient_error": max_prediction_gradient,
        "parameter_count": len(model.theta), "constructed_cooccurrence_minus_product_range": [min(gaps), max(gaps)],
        "frequency_recovery_max_error": recovery, "constructed_fit": learner.optimizer,
        "real_development_prefixes": len(replay_rows), "max_frozen_prediction_difference": max_replay,
        "max_current_original_prediction_difference": max_original,
        "real_development_max_cooccurrence_minus_product": max_real_covariance,
        "new_api_calls": 0, "heldout_test_read": False, "new_accuracy_evaluation": False,
        "interpretation": "Existing appearance law is connected without flattening dependence; selected real-data model still has inactive latent output path. No demonstrated performance improvement."}
    for name, value in (("REPORT.json", report), ("mechanisms.json", mechanism_rows), ("real_development_replay.json", replay_rows),
                        ("constructed_model.json", model.parameters()), ("frequency_fitted_model.json", learner.parameters()), ("historical_connected_model.json", actual.parameters())):
        write(out / name, value)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        freeze(args.out.resolve(), args.fit.resolve())
    else:
        run(args.out.resolve())
