"""Training/validation-only attribution; never opens the old held-out test.

A fixed object-risk/OR architecture changes only supervision. Direct field
HistGB is a stronger fine-label control. Separate GLMs expose BN equivalence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from evaluate_privacy_flow import sha, write, metrics
from paperalpha_monitor.privacy_flow_data import baseline_features, digest, batch
from paperalpha_monitor.privacy_flow_bn import PrivacyFlowBN
from paperalpha_monitor.privacy_supervision_controls import (DirectObjectRisk, SeparatedTransitionGLMs,
    direct_design, outcome_matrix, object_predictions, bernoulli_glm_loss)

PROTOCOL = {"id": "privacy-supervision-attribution-validation-only-v1-20260919",
    "partitions": ["train", "validation"], "old_test_policy": "Never open, copy, evaluate, tune on or replace old test/results files",
    "direct_architecture": "Per-value sigmoid of kind, individual C, first-prefix context, plus W only at cutoff 2; OR readout",
    "controlled_difference": "Identical direct architecture/initialization/optimizer; only field-vs-endpoint supervision changes",
    "regularization": [.0001, .001, .01, .1], "seeds": [0, 17], "max_iter": 1000,
    "selection": "First lowest training objective per regularization; then lowest validation endpoint Brier, per cutoff/arm",
    "matched_lambda_contrasts": "Also report objects-minus-endpoint Brier at EACH identical regularization",
    "histgb_max_leaves": [3, 7, 15], "histgb_max_iter": 100, "histgb_l2": .1,
    "histgb_training": "Flatten observed per-value input and final per-value labels; OR over predictions",
    "normalization": "GLM loss sums object losses per trace, then averages traces; endpoint arm averages endpoint loss; same numeric ridge grid",
    "reference_models": "Old validation-selected BN and task classifiers, frozen and unchanged",
    "equivalence": "Independent chain BN likelihood separates into two independently trained Bernoulli GLMs with identical ridge",
    "scope": "Development attribution; validation reuse/hyperparameter selection prevents independent performance or novel-MAS-method claims",
    "loss_semantics": "Fine-grained labels are source exact-value rules, not human causal or semantic mechanism labels"}
SOURCES = ('code/commands/evaluate_privacy_supervision_controls.py', 'code/commands/evaluate_privacy_flow.py',
    'code/commands/inspect_agentleak_release.py', 'code/commands/harnessaudit_cohort.py',
    'code/src/paperalpha_monitor/privacy_flow_data.py', 'code/src/paperalpha_monitor/privacy_flow_bn.py',
    'code/src/paperalpha_monitor/privacy_supervision_controls.py', 'knowledge/templates/privacy_flow_bn_v1.json',
    'code/tests/test_privacy_supervision_controls.py', 'code/tests/test_privacy_flow_bn.py')
FORBIDDEN_OLD_FILES = {"test.json", "predictions.json", "REPORT.json", "explanations.json"}


def is_forbidden(original, path):
    if not isinstance(path, (str, bytes, Path)):
        return False
    if isinstance(path, bytes):
        path = path.decode()
    path = Path(path).resolve()
    return path.parent == original and path.name in FORBIDDEN_OLD_FILES


def guard_old_test(original):
    attempted = []
    def hook(event, args):
        if event == "open" and is_forbidden(original, args[0]):
            attempted.append(str(args[0]))
            raise PermissionError("held-out test and its old results are outside this development experiment")
    sys.addaudithook(hook)
    return attempted


def score_details(rows, detail):
    y = [r["y"] for r in rows]
    result = {"endpoint": metrics(y, detail["probabilities"])}
    if "field_marginals" in detail:
        losses, briers, micro_y, micro_p = [], [], [], []
        for row, p in zip(rows, detail["field_marginals"]):
            p, f = np.asarray(p, float), np.asarray(row["final_value_labels"], int)
            if p.shape != f.shape:
                raise ValueError("field prediction shape mismatch")
            briers.append(float(np.mean((p - f) ** 2)))
            clipped = np.clip(p, 1e-12, 1 - 1e-12)
            losses.append(float(-np.mean(f * np.log(clipped) + (1 - f) * np.log1p(-clipped))))
            micro_y.extend(f.tolist())
            micro_p.extend(p.tolist())
        result["objects"] = {"micro": metrics(micro_y, micro_p), "mean_trace_brier": float(np.mean(briers)),
            "mean_trace_nll": float(np.mean(losses))}
    return result


def forest_details(model, packets):
    data = direct_design(packets)
    q = np.zeros(data["mask"].shape)
    q[data["mask"]] = model.predict_proba(data["design"][data["mask"]])[:, 1]
    return {"probabilities": object_predictions(q, data["mask"]),
        "field_marginals": [q[i, data["mask"][i]].tolist() for i in range(len(packets))]}


def evaluate(original, out):
    if out.exists():
        raise ValueError("new output directory required")
    blocked = guard_old_test(original)
    data_manifest = json.loads((original / "DATA.json").read_text(encoding="utf-8"))
    old_selected = json.loads((original / "SELECTED.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    inputs = {}
    for split in ("train", "validation"):
        path = original / f"{split}.json"
        if sha(path) != data_manifest["dataset_sha256"][split]:
            raise ValueError("original partition hash changed")
        shutil.copyfile(path, out / path.name)
        inputs[path.name] = sha(path)
    old_files = {**old_selected["bn"], **{f"{s}_{m}": f for s, methods in old_selected["classifiers"].items() for m, f in methods.items()}}
    for name in old_files.values():
        target = out / "references" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
        inputs["references/" + name] = sha(target)
    for source in SOURCES:
        target = out / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    write(out / "PLAN.json", {"protocol": PROTOCOL, "old_selected": old_selected,
        "inputs_sha256": inputs, "original_run": str(original),
        "old_test_expected_hash_metadata_only": data_manifest["dataset_sha256"]["test"],
        "source_sha256": {p.relative_to(out).as_posix(): sha(p) for p in (out / "source_snapshot").rglob("*") if p.is_file()}})
    train, validation = [json.loads((out / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "validation")]
    results, candidates, predictions, selected, contrasts = {}, [], {}, {}, {}
    for stage in (1, 2):
        key = str(stage)
        pt, pv = [r["prefixes"][key] for r in train], [r["prefixes"][key] for r in validation]
        predictions[key], results[key], selected[key], contrast_options = {}, {}, {}, {}
        for supervision in ("endpoint", "objects"):
            method, candidates_ridge = "direct_glm_" + supervision, []
            target = [r["y"] for r in train] if supervision == "endpoint" else [r["final_value_labels"] for r in train]
            contrast_options[supervision] = {}
            for ridge in PROTOCOL["regularization"]:
                fits = []
                for seed in PROTOCOL["seeds"]:
                    started = time.perf_counter()
                    model = DirectObjectRisk(stage=stage, supervision=supervision, regularization=ridge, seed=seed).fit(pt, target)
                    filename = f"models/{method}_stage{stage}_lambda{ridge}_seed{seed}.json"
                    write(out / filename, model.parameters())
                    candidate = {"stage": key, "method": method, "regularization": ridge, "seed": seed,
                        "file": filename, "optimizer": model.optimizer, "seconds": time.perf_counter() - started}
                    candidates.append(candidate)
                    if model.optimizer["success"]:
                        fits.append((model.optimizer["loss"], filename, model))
                if not fits:
                    continue
                _, filename, model = min(fits, key=lambda a: (a[0], a[1]))
                detail = model.predict_details(pv)
                scored = score_details(validation, detail)
                candidates.append({"stage": key, "method": method, "regularization": ridge,
                    "training_best_file": filename, "validation": scored})
                candidates_ridge.append((scored["endpoint"]["brier"], filename, detail))
                contrast_options[supervision][str(ridge)] = scored
            if not candidates_ridge:
                raise ValueError("no converged direct control")
            _, filename, detail = min(candidates_ridge, key=lambda a: (a[0], a[1]))
            selected[key][method] = filename
            predictions[key][method], results[key][method] = detail, score_details(validation, detail)
            print(json.dumps({"stage": key, "selected": method, "file": filename, "validation": results[key][method]}), flush=True)
        contrasts[key] = {ridge: {"objects_minus_endpoint_brier": contrast_options["objects"][ridge]["endpoint"]["brier"] -
            contrast_options["endpoint"][ridge]["endpoint"]["brier"], "arms": {k: v[ridge] for k, v in contrast_options.items()}}
            for ridge in contrast_options["endpoint"] if ridge in contrast_options["objects"]}
        data = direct_design(pt)
        x = data["design"][data["mask"]]
        f = outcome_matrix(data, [r["final_value_labels"] for r in train])[data["mask"]]
        options = []
        for leaves in PROTOCOL["histgb_max_leaves"]:
            model = HistGradientBoostingClassifier(max_leaf_nodes=leaves, max_iter=100,
                l2_regularization=.1, early_stopping=False, random_state=0).fit(x, f)
            filename = f"models/direct_field_histgb_stage{stage}_leaves{leaves}.joblib"
            joblib.dump(model, out / filename)
            detail = forest_details(model, pv)
            scored = score_details(validation, detail)
            candidates.append({"stage": key, "method": "direct_field_histgb", "leaves": leaves, "file": filename, "validation": scored})
            options.append((scored["endpoint"]["brier"], filename, detail))
        _, filename, detail = min(options, key=lambda a: (a[0], a[1]))
        selected[key]["direct_field_histgb"] = filename
        predictions[key]["direct_field_histgb"], results[key]["direct_field_histgb"] = detail, score_details(validation, detail)
        for name, filename in old_selected["bn"].items():
            model = PrivacyFlowBN.restore(json.loads((out / "references" / filename).read_text(encoding="utf-8")))
            detail = model.predict_details(pv)
            predictions[key][name], results[key][name] = detail, score_details(validation, detail)
        for name, filename in old_selected["classifiers"][key].items():
            model = joblib.load(out / "references" / filename)
            detail = {"probabilities": model.predict_proba(baseline_features(pv))[:, 1]}
            predictions[key]["task_" + name], results[key]["task_" + name] = detail, score_details(validation, detail)
    # Independent likelihood equivalence and refit are train/validation only.
    previous = PrivacyFlowBN.restore(json.loads((out / "references" / old_selected["bn"]["independent_bn"]).read_text(encoding="utf-8")))
    train_packets = [r["prefixes"]["2"] for r in train]
    train_labels = [r["final_value_labels"] for r in train]
    data = batch(train_packets)
    f = outcome_matrix(data, train_labels)
    qloss, qgrad = bernoulli_glm_loss(previous.theta[:previous.d], data["dc"], data["w"], data["mask"], previous.regularization)
    rloss, rgrad = bernoulli_glm_loss(previous.theta[previous.d:], data["dw"], f, data["mask"], previous.regularization)
    jointloss, jointgrad = previous.loss_gradient(previous.theta, data, f)
    equivalent = SeparatedTransitionGLMs(regularization=previous.regularization)
    equivalent.worker, equivalent.final = previous.theta[:previous.d].copy(), previous.theta[previous.d:].copy()
    fixed_errors = {}
    for stage in ("1", "2"):
        pv = [r["prefixes"][stage] for r in validation]
        fixed_errors[stage] = float(np.max(np.abs(previous.predict(pv) - equivalent.predict_details(pv)["probabilities"])))
    fitted = SeparatedTransitionGLMs(regularization=previous.regularization).fit(train_packets, train_labels)
    write(out / "separate_glms.json", fitted.parameters())
    refit_errors = {}
    for stage in ("1", "2"):
        pv = [r["prefixes"][stage] for r in validation]
        detail = fitted.predict_details(pv)
        predictions[stage]["separately_fitted_glms"] = detail
        results[stage]["separately_fitted_glms"] = score_details(validation, detail)
        refit_errors[stage] = float(np.max(np.abs(previous.predict(pv) - detail["probabilities"])))
    equivalence = {"same_parameter_objective_difference": abs(jointloss - qloss - rloss),
        "same_parameter_gradient_max_difference": float(np.max(np.abs(jointgrad - np.r_[qgrad, rgrad]))),
        "same_parameter_validation_max_difference": fixed_errors,
        "separate_refit_validation_max_difference": refit_errors, "optimizers": fitted.optimizers,
        "interpretation": "This is the SAME independent probability factorization expressed without a BN engine; it is not an independent novel algorithm."}
    write(out / "EQUIVALENCE.json", equivalence)
    write(out / "CANDIDATES.json", candidates)
    write(out / "SELECTED.json", selected)
    write(out / "MATCHED_SUPERVISION.json", contrasts)
    rows = []
    for i, row in enumerate(validation):
        rows.append({"id": row["id"], "template_id": row["template_id"], "vertical": row["vertical"], "model": row["model"],
            "label": row["y"], "final_value_labels": row["final_value_labels"],
            "predictions": {s: {m: {"probability": float(d["probabilities"][i]),
                "field_marginals": d.get("field_marginals", [None] * len(validation))[i]} for m, d in methods.items()}
                for s, methods in predictions.items()}})
    write(out / "validation_predictions.json", rows)
    per_template = {}
    for stage, methods in predictions.items():
        per_template[stage] = {}
        for template in sorted({r["template_id"] for r in validation}):
            indices = [i for i, r in enumerate(validation) if r["template_id"] == template]
            per_template[stage][template] = {m: metrics([validation[i]["y"] for i in indices], d["probabilities"][indices]) for m, d in methods.items()}
    write(out / "REPORT.json", {"protocol": PROTOCOL, "counts": {"train": len(train), "validation": len(validation)},
        "old_test_or_results_open_attempts": blocked, "selected": selected, "validation_results": results,
        "by_validation_template": per_template, "equivalence": equivalence,
        "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    print(json.dumps({"validation_summary": {s: {m: {"endpoint_brier": v["endpoint"]["brier"],
        "object_brier": v.get("objects", {}).get("mean_trace_brier")} for m, v in methods.items()}
        for s, methods in results.items()}, "equivalence": equivalence}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.original.resolve(), args.out.resolve())
