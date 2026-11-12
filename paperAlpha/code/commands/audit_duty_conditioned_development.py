"""Replay frozen v9 estimators and verify actual fitting, not new claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_duty_conditioned_development import load, records_for, METHODS
from train_typed_action_development import verify_hashes, baseline_predictions, write_new, sha
from paperalpha_monitor.duty_conditioned_bn import DutyConditionedBN


def audit(run, output):
    if output.exists():
        raise ValueError("new output directory required")
    plan, data, conditions, vectors, _, alternatives = load(run)
    original = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    verify_hashes(run, original["artifacts_sha256"])
    traces = {k: conditions[k].trace(v[0]) for k, v in data.items()}
    raw, rows, score_error, gradient_error = records_for(data, vectors), [], 0., 0.
    diagnostics = []
    for fold in plan["folds"]:
        for method in METHODS:
            saved = json.loads((run / "models" / fold["evaluation_task"] / (method + ".json")).read_text(encoding="utf-8"))
            expected = json.loads((run / "predictions" / fold["evaluation_task"] / (method + ".json")).read_text(encoding="utf-8"))
            if method.startswith("duty_bn"):
                model = DutyConditionedBN(regularization=saved["regularization"], max_iter=saved["max_iter"],
                    use_graph=saved["use_graph"], use_messages=saved["use_messages"], use_text=saved["use_text"])
                assert list(model.parameter_names) == saved["parameter_names"]
                assert model.registry.sha256 == saved["registry_sha256"]
                model.theta = np.asarray(saved["theta"])
                targets = [data[k][1] for k in fold["train"]]
                training = [traces[k] for k in fold["train"]]
                value, gradient = model.loss_gradient(model.theta, training, targets)
                assert abs(value - saved["optimizer"]["loss"]) < 1e-12
                errors = []
                for seed in range(6):
                    direction = np.random.default_rng(seed).normal(size=len(model.theta))
                    direction /= np.linalg.norm(direction)
                    epsilon = 2e-5
                    numeric = (model.loss_gradient(model.theta + epsilon * direction, training, targets)[0]
                        - model.loss_gradient(model.theta - epsilon * direction, training, targets)[0]) / (2 * epsilon)
                    errors.append(abs(numeric - gradient @ direction))
                gradient_error = max(gradient_error, max(errors))
                predictions = {k: model.predict_actions(traces[k]) for k in fold["evaluate"]}
                diagnostics.append({"fold": fold["evaluation_task"], "method": method, "gradient_error": max(errors),
                    "nonzero_duty_weights": int(np.sum(np.abs(model.theta[list(model.registry.duty_indices)]) > 1e-7))})
            else:
                predictions = {k: baseline_predictions(saved, raw[k], "visible_logistic") for k in fold["evaluate"]}
            assert sum(map(len, predictions.values())) == len(expected)
            for row in expected:
                actual = predictions[row["run_id"]][row["action_id"]]
                label = data[row["run_id"]][1].actions[row["action_id"]]
                assert row["targets"] == {"forbidden": label.forbidden, "unnecessary": label.unnecessary}
                for target in ("forbidden", "unnecessary", "unnecessary_given_not_forbidden", "no_tool_tier_violation"):
                    score_error = max(score_error, abs(actual[target] - row["prediction"][target]))
                rows.append({**row, "prediction": actual})
    assert score_error < 1e-12 and gradient_error < 2e-7
    for method in METHODS:
        chosen = [r for r in rows if r["method"] == method]
        for target in ("forbidden", "unnecessary"):
            brier = np.mean([(r["prediction"][target] - r["targets"][target]) ** 2 for r in chosen])
            assert abs(brier - original["metrics"][method][target]["brier"]) < 1e-12
    # Deduplicate orchestration variants: public profiles belong to two tasks.
    matches = []
    for key in ("fixed__dl-t20", "fixed__dl-t22"):
        matches.extend({"task": key, "actor": a, "same": conditions[key].vectors[a] == alternatives[key].vectors[a]}
            for a in conditions[key].vectors)
    output.mkdir(parents=True)
    with (output / "audit_source.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    report = {"training_report_sha256": sha(run / "REPORT.json"), "fitted_estimators_replayed": 6,
        "action_predictions_replayed": len(rows), "max_score_error": score_error, "max_gradient_error": gradient_error,
        "duty_name_sensitivity": matches, "fitting_diagnostics": diagnostics,
        "scope": "Reproducibility audit; no new generalization or original-paper baseline evidence."}
    write_new(output / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.out.resolve())
