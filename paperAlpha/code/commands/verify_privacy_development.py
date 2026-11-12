"""Audit one frozen privacy development run without opening the old test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from scipy.special import expit


def close(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("dictionary schema mismatch")
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("list size mismatch")
        for a, b in zip(actual, expected):
            close(a, b)
    elif type(expected) is float:
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"numeric mismatch: {actual}, {expected}")
    elif actual != expected:
        raise ValueError("value mismatch")


def stage_reference(model, packet, context, kinds):
    """Scalar four-state inference, independently written from the model."""
    x = np.r_[1., context(packet)]
    t, d, k = model.theta, model.d, model.k
    pu = expit(x @ t[2 * d:2 * d + k])
    numerator, denominator = 0., 0.
    for u in (0, 1):
        pv = expit(x @ t[2 * d + k:2 * d + 2 * k] + (u * t[-1] if model.coupled else 0.))
        for v in (0, 1):
            weight = (pu if u else 1 - pu) * (pv if v else 1 - pv)
            safe = 1.
            for field in packet["groups"]:
                kind = np.eye(len(kinds))[kinds.index(field["kind"])]
                q = expit(np.r_[kind, float(field["coordinator_present"]), x[1:]] @ t[:d] + u * t[model.boost_start])
                r0 = expit(np.r_[kind, 0., x[1:]] @ t[d:2 * d] + v * t[model.boost_start + 1])
                r1 = expit(np.r_[kind, 1., x[1:]] @ t[d:2 * d] + v * t[model.boost_start + 1])
                if packet["stage"] == 1:
                    risk = (1 - q) * r0 + q * r1
                else:
                    weight *= q if field["worker_present"] else 1 - q
                    risk = r1 if field["worker_present"] else r0
                safe *= 1 - risk
            numerator += weight * (1 - safe)
            denominator += weight
    return numerator / denominator


def verify(run, kind, out):
    if out.exists():
        raise ValueError("new audit output required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from evaluate_privacy_flow import sha, write, metrics
    from evaluate_privacy_supervision_controls import guard_old_test, score_details, forest_details
    from paperalpha_monitor.privacy_flow_data import batch, baseline_features, context, KINDS
    from paperalpha_monitor.privacy_flow_bn import PrivacyFlowBN
    from paperalpha_monitor.privacy_supervision_controls import (DirectObjectRisk, SeparatedTransitionGLMs,
        direct_design, outcome_matrix, bernoulli_glm_loss)
    if kind == "stages":
        from paperalpha_monitor.privacy_stage_bn import PrivacyStageBN
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    attempts = guard_old_test(Path(plan["original_run"]).resolve())
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError("artifact hash mismatch: " + name)
    for name, expected in plan["source_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError("source hash mismatch")
    train, validation = [json.loads((run / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "validation")]
    if {r["template_id"] for r in train} & {r["template_id"] for r in validation}:
        raise ValueError("task-template overlap")
    candidates = json.loads((run / "CANDIDATES.json").read_text(encoding="utf-8"))
    selected = json.loads((run / "SELECTED.json").read_text(encoding="utf-8"))
    predictions = json.loads((run / "validation_predictions.json").read_text(encoding="utf-8"))
    max_loss, max_gradient, fitting_count = 0., 0., 0
    for c in candidates:
        if "optimizer" not in c:
            continue
        fitting_count += 1
        saved = json.loads((run / c["file"]).read_text(encoding="utf-8"))
        if kind == "controls":
            model = DirectObjectRisk.restore(saved)
            data = direct_design([r["prefixes"][str(model.stage)] for r in train])
            targets = np.asarray([r["y"] for r in train]) if model.supervision == "endpoint" else outcome_matrix(data, [r["final_value_labels"] for r in train])
        else:
            model = PrivacyStageBN.restore(saved)
            data = batch([r["prefixes"]["2"] for r in train])
            targets = outcome_matrix(data, [r["final_value_labels"] for r in train])
        value, grad = model.loss_gradient(model.theta, data, targets)
        max_loss = max(max_loss, abs(value - c["optimizer"]["loss"]))
        rng = np.random.default_rng(10477)
        for _ in range(2):
            direction = rng.normal(size=len(model.theta))
            direction /= np.linalg.norm(direction)
            eps = 1e-4
            numerical = (model.loss_gradient(model.theta + eps * direction, data, targets)[0] -
                model.loss_gradient(model.theta - eps * direction, data, targets)[0]) / (2 * eps)
            max_gradient = max(max_gradient, abs(numerical - grad @ direction))
    if max_loss > 1e-10 or max_gradient > 1e-6:
        raise ValueError("fitted objective or gradient differs")
    # Recompute validation for every candidate used in model selection.
    validated_candidates = 0
    for c in candidates:
        if "validation" not in c:
            continue
        validated_candidates += 1
        filename = c.get("training_best_file", c.get("file"))
        if "training_best_file" in c:
            same_group = [p for p in candidates if "optimizer" in p and p["method"] == c["method"]
                and (p.get("stage"), p.get("regularization", p.get("ridge"))) ==
                    (c.get("stage"), c.get("regularization", c.get("ridge"))) and p["optimizer"]["success"]]
            if min(same_group, key=lambda p: (p["optimizer"]["loss"], p["file"]))["file"] != filename:
                raise ValueError("initialization selection differs from training rule")
        if kind == "stages":
            model = PrivacyStageBN.restore(json.loads((run / filename).read_text(encoding="utf-8")))
            recomputed = {s: score_details(validation, model.predict_details([r["prefixes"][s] for r in validation])) for s in ("1", "2")}
        elif c["method"] == "direct_field_histgb":
            model = joblib.load(run / filename)
            recomputed = score_details(validation, forest_details(model, [r["prefixes"][c["stage"]] for r in validation]))
        else:
            model = DirectObjectRisk.restore(json.loads((run / filename).read_text(encoding="utf-8")))
            recomputed = score_details(validation, model.predict_details([r["prefixes"][c["stage"]] for r in validation]))
        close(recomputed, c["validation"])
    reconstructed = {"1": {}, "2": {}}
    max_reference = 0.
    if kind == "controls":
        for stage, methods in selected.items():
            packets = [r["prefixes"][stage] for r in validation]
            for method, filename in methods.items():
                options = [c for c in candidates if c["method"] == method and c.get("stage") == stage and "validation" in c]
                best = min(options, key=lambda c: (c["validation"]["endpoint"]["brier"], c.get("training_best_file", c.get("file"))))
                if best.get("training_best_file", best.get("file")) != filename:
                    raise ValueError("validation selection mismatch")
                if filename.endswith(".joblib"):
                    detail = forest_details(joblib.load(run / filename), packets)
                else:
                    detail = DirectObjectRisk.restore(json.loads((run / filename).read_text(encoding="utf-8"))).predict_details(packets)
                reconstructed[stage][method] = detail
            for name, filename in plan["old_selected"]["bn"].items():
                model = PrivacyFlowBN.restore(json.loads((run / "references" / filename).read_text(encoding="utf-8")))
                reconstructed[stage][name] = model.predict_details(packets)
            for name, filename in plan["old_selected"]["classifiers"][stage].items():
                model = joblib.load(run / "references" / filename)
                reconstructed[stage]["task_" + name] = {"probabilities": model.predict_proba(baseline_features(packets))[:, 1]}
            saved = json.loads((run / "separate_glms.json").read_text(encoding="utf-8"))
            model = SeparatedTransitionGLMs(regularization=saved["regularization"])
            model.worker, model.final = np.array(saved["worker"]), np.array(saved["final"])
            reconstructed[stage]["separately_fitted_glms"] = model.predict_details(packets)
    else:
        for method, filename in selected.items():
            options = [c for c in candidates if c["method"] == method and "validation" in c]
            best = min(options, key=lambda c: (np.mean([v["endpoint"]["brier"] for v in c["validation"].values()]), c["training_best_file"]))
            if best["training_best_file"] != filename:
                raise ValueError("validation selection mismatch")
            model = PrivacyStageBN.restore(json.loads((run / filename).read_text(encoding="utf-8")))
            for stage in ("1", "2"):
                packets = [r["prefixes"][stage] for r in validation]
                detail = model.predict_details(packets)
                reconstructed[stage][method] = detail
                independent = np.array([stage_reference(model, p, context, KINDS) for p in packets])
                max_reference = max(max_reference, float(np.max(np.abs(independent - detail["probabilities"]))))
    if len(predictions) != len(validation) or any(p["id"] != r["id"] or p["label"] != r["y"] for p, r in zip(predictions, validation)):
        raise ValueError("prediction identity mismatch")
    max_prediction, count, by_template = 0., 0, {}
    for stage, methods in reconstructed.items():
        by_template[stage] = {}
        for method, detail in methods.items():
            scored = score_details(validation, detail)
            expected = report["validation_results"][stage][method] if kind == "controls" else report["validation_results"][method][stage]
            close(scored, expected)
            old = [p["predictions"][stage][method] if kind == "controls" else p["predictions"][method][stage] for p in predictions]
            max_prediction = max(max_prediction, float(np.max(np.abs(detail["probabilities"] - [p["probability"] for p in old]))))
            for i, p in enumerate(old):
                if p["field_marginals"] is not None:
                    close(detail["field_marginals"][i], p["field_marginals"])
                if kind == "stages":
                    close(detail["state_weights"][i].tolist(), p["state_weights"])
                    close(detail["state_conditional_risk"][i].tolist(), p["state_conditional_risk"])
            count += len(old)
        for tid in sorted({r["template_id"] for r in validation}):
            indices = np.array([r["template_id"] == tid for r in validation])
            domain = next(r["vertical"] for r in validation if r["template_id"] == tid)
            by_template[stage][domain] = {m: metrics([r["y"] for r in validation if r["template_id"] == tid], d["probabilities"][indices])["brier"] for m, d in methods.items()}
    if max_prediction > 1e-12 or max_reference > 1e-12:
        raise ValueError("saved or independently computed predictions differ")
    if attempts or report["old_test_or_results_open_attempts"]:
        raise ValueError("old test boundary attempted")
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "verify_source.py")
    write(out / "REPORT.json", {"kind": kind, "run": str(run), "frozen_artifacts_checked": len(report["artifacts_sha256"]),
        "fitted_models_objective_checked": fitting_count, "selection_candidates_recomputed": validated_candidates,
        "objective_max_error": max_loss, "directional_gradient_max_error": max_gradient,
        "endpoint_predictions_recomputed": count, "saved_prediction_max_error": max_prediction,
        "independent_stage_scalar_max_error": max_reference if kind == "stages" else None,
        "by_validation_template_brier": by_template, "old_test_open_attempts": attempts,
        "development_only": True, "source_sha256": sha(out / "verify_source.py"), "run_report_sha256": sha(run / "REPORT.json")})
    print(json.dumps(json.loads((out / "REPORT.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", choices=("controls", "stages"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.kind, args.out.resolve())
