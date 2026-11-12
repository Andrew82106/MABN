"""Fit and freeze identity-free estimators before new API collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from evaluate_privacy_flow import sha, write, metrics
from evaluate_privacy_supervision_controls import guard_old_test, score_details
from paperalpha_monitor.privacy_model_independent import (from_legacy, ModelIndependentPrivacyBN,
    graph_features, direct_features, BASELINE_NAMES)
from paperalpha_monitor.privacy_supervision_controls import outcome_matrix, object_predictions

PROTOCOL = {"id": "identity-free-privacy-training-v1-20260919", "regularization": [.0001, .001, .01],
    "seeds": [0, 17], "bn_selection": "Best training objective per lambda, then minimum mean validation Brier over both cutoffs",
    "classifier_selection": "Per-cutoff minimum validation Brier; fixed grids C=[.1,1,10], leaves=[3,7,15]",
    "observation": "No generator identity input/parameters, not merely renaming an unseen generator as a known one",
    "future_api": "Freeze models before new API outcomes; do not tune or recalibrate on the prospective cohort",
    "old_test": "Not read or used", "scope": "Still a fixed privacy pipeline with configured initial vault and allowlist"}
SOURCES = ('code/commands/train_privacy_model_independent.py', 'code/commands/evaluate_privacy_flow.py',
    'code/commands/evaluate_privacy_supervision_controls.py', 'code/commands/inspect_agentleak_release.py', 'code/commands/harnessaudit_cohort.py',
    'code/src/paperalpha_monitor/privacy_model_independent.py', 'code/src/paperalpha_monitor/privacy_stage_bn.py',
    'code/src/paperalpha_monitor/privacy_flow_data.py', 'code/src/paperalpha_monitor/privacy_supervision_controls.py',
    'code/src/paperalpha_monitor/privacy_flow_bn.py', 'knowledge/templates/privacy_flow_bn_v1.json', 'code/tests/test_privacy_model_independent.py')


def fit(parent, out):
    if out.exists():
        raise ValueError("new output directory required")
    old_plan = json.loads((parent / "PLAN.json").read_text(encoding="utf-8"))
    attempted = guard_old_test(Path(old_plan["original_run"]).resolve())
    old_report = json.loads((parent / "REPORT.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    dataset = {}
    for split in ("train", "validation"):
        path = parent / f"{split}.json"
        if sha(path) != old_report["artifacts_sha256"][path.name]:
            raise ValueError("development data changed")
        dataset[split] = json.loads(path.read_text(encoding="utf-8"))
        for row in dataset[split]:
            row["prefixes"] = {s: from_legacy(p) for s, p in row["prefixes"].items()}
            # Generator metadata is not needed even in training artifacts.
            row.pop("model", None)
        write(out / path.name, dataset[split])
    for source in SOURCES:
        path = out / "source_snapshot" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
    write(out / "PLAN.json", {"protocol": PROTOCOL, "parent": str(parent), "original_run": old_plan["original_run"],
        "parent_report_sha256": sha(parent / "REPORT.json"),
        "source_sha256": {p.relative_to(out).as_posix(): sha(p) for p in (out / "source_snapshot").rglob("*") if p.is_file()},
        "data_sha256": {s: sha(out / f"{s}.json") for s in dataset}})
    train, val = dataset["train"], dataset["validation"]
    labels = [r["final_value_labels"] for r in train]
    yt, yv = np.array([r["y"] for r in train]), np.array([r["y"] for r in val])
    candidates, selected, validation_scores = [], {"bn": {}, "classifiers": {"1": {}, "2": {}}}, {"1": {}, "2": {}}
    for coupled in (False, True):
        name, choices = ("bn_coupled" if coupled else "bn_independent"), []
        for ridge in PROTOCOL["regularization"]:
            fits = []
            for seed in PROTOCOL["seeds"]:
                model = ModelIndependentPrivacyBN(coupled=coupled, regularization=ridge, seed=seed).fit([r["prefixes"]["2"] for r in train], labels)
                filename = f"models/{name}_lambda{ridge}_seed{seed}.json"
                write(out / filename, model.parameters())
                candidates.append({"method": name, "ridge": ridge, "seed": seed, "file": filename, "optimizer": model.optimizer})
                if model.optimizer["success"]:
                    fits.append((model.optimizer["loss"], filename, model))
                print(json.dumps({"fit": filename, "converged": model.optimizer["success"], "loss": model.optimizer["loss"]}), flush=True)
            if not fits:
                continue
            _, filename, model = min(fits, key=lambda a: (a[0], a[1]))
            scores = {s: score_details(val, model.predict_details([r["prefixes"][s] for r in val])) for s in ("1", "2")}
            candidates.append({"method": name, "ridge": ridge, "training_best_file": filename, "validation": scores})
            choices.append((np.mean([v["endpoint"]["brier"] for v in scores.values()]), filename, scores))
        if not choices:
            raise ValueError("no converged BN; collection not authorized by a fitted model")
        _, selected["bn"][name], scores = min(choices, key=lambda a: (a[0], a[1]))
        for s in validation_scores:
            validation_scores[s][name] = scores[s]
    for stage in ("1", "2"):
        pt, pv = [r["prefixes"][stage] for r in train], [r["prefixes"][stage] for r in val]
        xt, xv = graph_features(pt), graph_features(pv)
        dt, dv = direct_features(pt), direct_features(pv)
        field_labels = outcome_matrix(dt, labels)[dt["mask"]]
        for method, grid in (("logistic", [.1, 1., 10.]), ("histgb", [3, 7, 15]), ("field_histgb", [3, 7, 15])):
            choices = []
            for value in grid:
                model = (make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=2000, random_state=0))
                    if method == "logistic" else HistGradientBoostingClassifier(max_leaf_nodes=value, max_iter=100,
                        l2_regularization=.1, early_stopping=False, random_state=0))
                model.fit(dt["design"][dt["mask"]] if method == "field_histgb" else xt,
                          field_labels if method == "field_histgb" else yt)
                filename = f"models/{method}_stage{stage}_grid{value}.joblib"
                joblib.dump(model, out / filename)
                if method == "field_histgb":
                    q = np.zeros(dv["mask"].shape)
                    q[dv["mask"]] = model.predict_proba(dv["design"][dv["mask"]])[:, 1]
                    detail = {"probabilities": object_predictions(q, dv["mask"]),
                        "field_marginals": [q[i, dv["mask"][i]].tolist() for i in range(len(val))]}
                else:
                    detail = {"probabilities": model.predict_proba(xv)[:, 1]}
                scores = score_details(val, detail)
                candidates.append({"method": method, "stage": stage, "grid": value, "file": filename, "validation": scores})
                choices.append((scores["endpoint"]["brier"], filename, scores))
            _, selected["classifiers"][stage][method], validation_scores[stage][method] = min(choices, key=lambda a: (a[0], a[1]))
    write(out / "CANDIDATES.json", candidates)
    write(out / "SELECTED.json", selected)
    frozen_files = {name for name in selected["bn"].values()} | {f for methods in selected["classifiers"].values() for f in methods.values()}
    write(out / "FROZEN.json", {"schema": "frozen-model-independent-privacy-v1", "selected": selected,
        "model_files_sha256": {f: sha(out / f) for f in sorted(frozen_files)}, "baseline_names": BASELINE_NAMES,
        "training_prevalence": float(yt.mean()), "training_traces": len(train), "model_identity_used": False})
    write(out / "REPORT.json", {"protocol": PROTOCOL, "validation_scores": validation_scores, "old_test_open_attempts": attempted,
        "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    print(json.dumps({"frozen": str(out / "FROZEN.json"), "validation_brier": {s: {m: v["endpoint"]["brier"] for m, v in scores.items()}
        for s, scores in validation_scores.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit(args.parent.resolve(), args.out.resolve())
