"""Fit stage-local vs coupled states on development partitions only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from evaluate_privacy_flow import sha, write
from evaluate_privacy_supervision_controls import guard_old_test, score_details, SOURCES as CONTROL_SOURCES
from paperalpha_monitor.privacy_stage_bn import PrivacyStageBN

PROTOCOL = {"id": "privacy-separate-stage-states-development-v1-20260919",
    "hypothesis": "Different MAS roles need separate latent disclosure states; their statistical association need not be positive",
    "controlled_comparison": "Identical local state/emission models, either no U-V association or one signed coupling coefficient",
    "regularization": [.0001, .001, .01], "seeds": [0, 17], "max_iter": 1000,
    "selection": "Lowest training joint NLL+ridge per lambda, then lowest mean validation endpoint Brier across both cutoffs",
    "data": "Existing train and validation only; no test file or test output opened",
    "local_boosts": "Nonnegative, can be exactly zero; no causal or internal-state interpretation",
    "supervision": "Identical worker/final per-object outcomes for both variants",
    "scope": "Development results, not unseen-task performance; candidate addition follows previous validation diagnostics",
    "no_replacement": "Old shared model, scores and held-out test remain unchanged"}
SOURCES = (*CONTROL_SOURCES, 'code/src/paperalpha_monitor/privacy_stage_bn.py', 'code/commands/evaluate_privacy_stage_states.py',
           'code/tests/test_privacy_stage_bn.py')


def evaluate(control_run, out):
    if out.exists():
        raise ValueError("new output directory required")
    previous_plan = json.loads((control_run / "PLAN.json").read_text(encoding="utf-8"))
    forbidden = guard_old_test(Path(previous_plan["original_run"]).resolve())
    previous = json.loads((control_run / "REPORT.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    for name in ("train.json", "validation.json"):
        if sha(control_run / name) != previous["artifacts_sha256"][name]:
            raise ValueError("development data hash mismatch")
        shutil.copyfile(control_run / name, out / name)
    for source in SOURCES:
        dest = out / "source_snapshot" / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, dest)
    write(out / "PLAN.json", {"protocol": PROTOCOL, "control_run": str(control_run),
        "control_report_sha256": sha(control_run / "REPORT.json"), "original_run": previous_plan["original_run"],
        "input_sha256": {n: sha(out / n) for n in ("train.json", "validation.json")},
        "source_sha256": {p.relative_to(out).as_posix(): sha(p) for p in (out / "source_snapshot").rglob("*") if p.is_file()}})
    train, validation = [json.loads((out / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "validation")]
    pt, labels = [r["prefixes"]["2"] for r in train], [r["final_value_labels"] for r in train]
    candidates, selected, prediction, results = [], {}, {}, {}
    for coupled in (False, True):
        name, options = ("coupled_stage_bn" if coupled else "independent_stage_bn"), []
        for ridge in PROTOCOL["regularization"]:
            fits = []
            for seed in PROTOCOL["seeds"]:
                model = PrivacyStageBN(coupled=coupled, regularization=ridge, seed=seed).fit(pt, labels)
                filename = f"models/{name}_lambda{ridge}_seed{seed}.json"
                write(out / filename, model.parameters())
                candidate = {"method": name, "ridge": ridge, "seed": seed, "file": filename, "optimizer": model.optimizer,
                    "coupling": model.parameters()["state_link_log_odds"], "local_boosts": model.parameters()["local_boosts"]}
                candidates.append(candidate)
                print(json.dumps(candidate), flush=True)
                if model.optimizer["success"]:
                    fits.append((model.optimizer["loss"], filename, model))
            if not fits:
                continue
            _, filename, model = min(fits, key=lambda a: (a[0], a[1]))
            detail = {s: model.predict_details([r["prefixes"][s] for r in validation]) for s in ("1", "2")}
            scores = {s: score_details(validation, d) for s, d in detail.items()}
            candidates.append({"method": name, "ridge": ridge, "training_best_file": filename, "validation": scores})
            options.append((np.mean([v["endpoint"]["brier"] for v in scores.values()]), filename, detail, scores))
        if not options:
            raise ValueError("no converged candidate; old models are not changed")
        _, filename, detail, scores = min(options, key=lambda a: (a[0], a[1]))
        selected[name], prediction[name], results[name] = filename, detail, scores
    rows = []
    for i, row in enumerate(validation):
        rows.append({"id": row["id"], "template_id": row["template_id"], "label": row["y"],
            "final_value_labels": row["final_value_labels"], "predictions": {method: {s: {
                "probability": float(d["probabilities"][i]), "field_marginals": d["field_marginals"][i],
                "state_weights": d["state_weights"][i].tolist(), "state_conditional_risk": d["state_conditional_risk"][i].tolist()}
                for s, d in states.items()} for method, states in prediction.items()}})
    write(out / "CANDIDATES.json", candidates)
    write(out / "SELECTED.json", selected)
    write(out / "validation_predictions.json", rows)
    write(out / "REPORT.json", {"protocol": PROTOCOL, "selected": selected, "validation_results": results,
        "reference_validation_results": previous["validation_results"], "old_test_or_results_open_attempts": forbidden,
        "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    print(json.dumps({"selected": selected, "validation_summary": {m: {s: {
        "endpoint_brier": v["endpoint"]["brier"], "object_brier": v["objects"]["mean_trace_brier"]}
        for s, v in stages.items()} for m, stages in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.control_run.resolve(), args.out.resolve())
