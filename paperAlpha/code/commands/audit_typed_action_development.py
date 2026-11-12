"""Recompute fitted v8 results and diagnose graph reliance, without refitting."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np

from train_typed_action_development import (ROOT, sha, write_new, load_data, verify_hashes,
    METHODS, SETTINGS, baseline_predictions, metrics)
from paperalpha_monitor.typed_action_supervision import TypedActionSupervisionBN, HIGH_NAMES


def restored(saved):
    model = TypedActionSupervisionBN(regularization=saved["regularization"], max_iter=saved["max_iter"],
        use_graph=saved["use_graph"], use_messages=saved["use_messages"], use_text=saved["use_text"])
    if saved["parameter_names"] != list(model.parameter_names) or saved["registry_sha256"] != model.registry.sha256:
        raise ValueError("saved model specification differs")
    model.theta = np.asarray(saved["theta"], dtype=float)
    model.fitted_targets = list(saved["fitted_targets"])
    return model


def audit(run, output):
    if output.exists():
        raise ValueError("new audit directory required")
    output.mkdir(parents=True)
    with (output / "audit_source.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    plan, data = load_data(run)
    original = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    verify_hashes(run, original["artifacts_sha256"])
    raw = {key: TypedActionSupervisionBN().predict_actions(value[0]) for key, value in data.items()}
    records, diagnostics, max_score_error, max_gradient_error = [], [], 0., 0.
    for fold in plan["folds"]:
        for method in METHODS:
            saved = json.loads((run / "models" / fold["evaluation_task"] / (method + ".json")).read_text(encoding="utf-8"))
            expected = json.loads((run / "predictions" / fold["evaluation_task"] / (method + ".json")).read_text(encoding="utf-8"))
            expected = {(r["run_id"], r["action_id"]): r for r in expected}
            if method.startswith("typed_"):
                model = restored(saved)
                predictions = {key: model.predict_actions(data[key][0]) for key in fold["evaluate"]}
                traces, targets = [data[k][0] for k in fold["train"]], [data[k][1] for k in fold["train"]]
                loss, gradient = model.loss_gradient(model.theta, traces, targets)
                if abs(loss - saved["optimizer"]["loss"]) > 1e-12:
                    raise ValueError("stored fit loss not reproduced")
                errors = []
                for seed in range(6):
                    direction = np.random.default_rng(seed).normal(size=len(model.theta))
                    direction /= np.linalg.norm(direction)
                    eps = 2e-5
                    numeric = (model.loss_gradient(model.theta + eps * direction, traces, targets)[0]
                             - model.loss_gradient(model.theta - eps * direction, traces, targets)[0]) / (2 * eps)
                    errors.append(abs(numeric - gradient @ direction))
                max_gradient_error = max(max_gradient_error, max(errors))
                if max(errors) > 2e-7:
                    raise ValueError("fitted directional gradient mismatch")
                initial = model.initial_parameters()
                untrained_head = [i for i, name in enumerate(model.parameter_names[:model.base_count]) if name.startswith("head:")]
                np.testing.assert_array_equal(model.theta[untrained_head], initial[untrained_head])
                if method == "typed_joint_v8":
                    variants = {"messages_disabled_fixed_weights": copy.deepcopy(model),
                                "joint_latent_readout_disabled_fixed_weights": copy.deepcopy(model)}
                    variants["messages_disabled_fixed_weights"].use_messages = False
                    latent = variants["joint_latent_readout_disabled_fixed_weights"]
                    for i in [*range(15, 20), 21, 22, 23]:
                        latent.theta[latent.base_count + i] = 0.
                    latent.theta[-5:] = 0.
                    changes = {}
                    for name, changed in variants.items():
                        modified, differences = [], []
                        for key in fold["evaluate"]:
                            rerun = changed.predict_actions(data[key][0])
                            for aid, prediction in rerun.items():
                                differences.extend(abs(prediction[t] - predictions[key][aid][t]) for t in SETTINGS["targets"])
                                target = data[key][1].actions[aid]
                                modified.append({"targets": {"forbidden": target.forbidden, "unnecessary": target.unnecessary}, "prediction": prediction})
                        changes[name] = {"max_probability_change": max(differences), "metrics": metrics(modified),
                                         "interpretation": "Fixed-weight computational sensitivity; not retrained ablation, causal intervention or selected upgrade."}
                    changed_names = [model.parameter_names[i] for i in range(model.base_count) if abs(model.theta[i] - initial[i]) > 1e-7]
                    diagnostics.append({"fold": fold["evaluation_task"], "directional_gradient_max_error": max(errors),
                        "changes": changes, "changed_base_parameters": changed_names,
                        "semantic_content_coefficients_changed": [n for n in changed_names if n.startswith("text:") and not any(
                            n.endswith(":" + f) for f in ("bias", "previous_state", "extraction_unavailable", "prior_context_omitted"))]})
            else:
                predictions = {key: baseline_predictions(saved, raw[key], method) for key in fold["evaluate"]}
            found = set()
            for key, predictions_for_run in predictions.items():
                for aid, prediction in predictions_for_run.items():
                    expected_row = expected[(key, aid)]
                    target = data[key][1].actions[aid]
                    if expected_row["targets"] != {"forbidden": target.forbidden, "unnecessary": target.unnecessary}:
                        raise ValueError("saved target mismatch")
                    found.add((key, aid))
                    for t in ("forbidden", "unnecessary", "unnecessary_given_not_forbidden", "no_tool_tier_violation"):
                        max_score_error = max(max_score_error, abs(prediction[t] - expected_row["prediction"][t]))
                    if abs(sum(prediction[t] for t in ("forbidden", "unnecessary", "no_tool_tier_violation")) - 1.) > 1e-12:
                        raise ValueError("incoherent categorical action distribution")
                    records.append({**expected_row, "prediction": prediction})
            if found != set(expected):
                raise ValueError("saved prediction coverage differs")
    if max_score_error > 1e-12:
        raise ValueError("saved predictions not reproducible")
    # Independently recompute key metrics, not just calling the fitting script.
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        for target in SETTINGS["targets"]:
            squared = sum((r["prediction"][target] - r["targets"][target]) ** 2 for r in rows) / len(rows)
            reported = original["metrics_pooled_development_predictions"][method][target]
            if abs(squared - reported["brier"]) > 1e-12:
                raise ValueError("Brier score not reproduced")
            if sum(r["prediction"][target] >= .5 for r in rows) != 0:
                raise ValueError("recorded zero-alarm finding changed")
    # Report true task-level replication and simple role/tool support; do not
    # imply 53 independent test subjects or two wholly unseen data sources.
    profile = []
    for key, (trace, targets, _, _) in data.items():
        for action in trace.steps[-1].base.actions:
            target = targets.actions[action["action_id"]]
            profile.append({"run": key, "action_id": action["action_id"], "actor": action["binding"]["actor"],
                "role": action["role"], "tool": action["binding"]["tool"], "stage": action["node"],
                "capability": action["capability_profile"]["capability"], "permission": action["permission_status"],
                "forbidden": target.forbidden, "unnecessary": target.unnecessary})
    write_new(output / "action_profile.json", profile)
    report = {"training_report_sha256": sha(run / "REPORT.json"), "fitted_models_recomputed": 8,
        "action_predictions_recomputed": len(records), "maximum_score_error": max_score_error,
        "maximum_fitted_directional_gradient_error": max_gradient_error, "diagnostics": diagnostics,
        "action_profile": {"permission": dict(Counter(r["permission"] for r in profile)),
                           "high_positive_actors": dict(Counter(r["actor"] for r in profile if r["forbidden"])),
                           "high_positive_runs": dict(Counter(r["run"] for r in profile if r["forbidden"]))},
        "conclusion_scope": "Fitted typed heads and gradients verified. No semantic text labels, verified harm, calibrated threshold, independent tasks or paper-native MAS baselines.",
        "artifacts_sha256": {name: sha(output / name) for name in ("action_profile.json", "audit_source.py")}}
    write_new(output / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.out.resolve())
