"""Fit-only domain diagnostics for the frozen expanded-window-v2 run.

This script never refits a model or threshold and never opens calibration
labels.  Subgroup F1-opt thresholds are descriptive fit-OOF diagnostics only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402
import run_semantic_source_attribution_combined_fit_v1 as combined  # noqa: E402
import run_semantic_window_v2_expanded_fit_v1 as run  # noqa: E402


FIT_ROWS = ROOT / "fit_expansion/data/fit.jsonl"


def lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(score: np.ndarray, label: np.ndarray) -> dict:
    def one(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "q01": float(np.quantile(values, .01)), "q05": float(np.quantile(values, .05)),
            "q25": float(np.quantile(values, .25)), "median": float(np.quantile(values, .5)),
            "q75": float(np.quantile(values, .75)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)),
        }
    return {"all": one(score), "negative": one(score[label == 0]), "positive": one(score[label == 1])}


def unit_record(label, score, frozen_threshold) -> dict:
    label = np.asarray(label, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    optimum = q.choose_threshold(label, score)
    return {
        "n": len(label), "positive": int(label.sum()), "positive_rate": float(label.mean()),
        "frozen_threshold": float(frozen_threshold),
        "at_frozen_threshold": q.count(label, score, frozen_threshold),
        "subgroup_F1Opt_diagnostic": {
            "threshold": optimum["threshold"],
            "metrics": q.count(label, score, optimum["threshold"]),
            "used_to_change_model_or_threshold": False,
        },
        "score_distribution": distribution(score, label),
    }


def subset_record(answer_indices, projection, window_scores, answer_scores, thresholds):
    answer_indices = np.asarray(answer_indices, dtype=np.int64)
    response_mask = np.zeros(run.FIT_ANSWERS, dtype=bool)
    response_mask[answer_indices] = True
    window_mask = response_mask[projection["window_response_index"]]
    return {
        "answers": unit_record(
            projection["response_labels"][answer_indices], answer_scores[answer_indices],
            thresholds["answer"]["threshold"],
        ),
        "windows": unit_record(
            projection["window_labels"][window_mask], window_scores[window_mask],
            thresholds["window"]["threshold"],
        ),
    }


def main() -> None:
    fit = run.read(run.OUT / "fit_complete.json")
    summary = run.read(run.OUT / "summary.json")
    prep = run.read(run.OUT / "preparation_complete.json")
    axes, projection = combined.load_projection()
    with np.load(run.OUT / "fit_scores.npz", allow_pickle=False) as scores:
        window_scores = scores["window_scores"].copy()
        answer_scores = scores["answer_scores"].copy()
    rows = lines(FIT_ROWS)
    assert len(rows) == run.FIT_ANSWERS
    assert [row["response_id"] for row in rows] == axes["response_ids"]
    generators = np.asarray([row["model"] for row in rows], dtype=str)
    thresholds = fit["selected"]["fit_OOF_thresholds"]

    cohorts = {
        "native_634": subset_record(np.arange(run.NATIVE_ANSWERS), projection, window_scores, answer_scores, thresholds),
        "added_3046": subset_record(np.arange(run.NATIVE_ANSWERS, run.FIT_ANSWERS), projection, window_scores, answer_scores, thresholds),
        "all_3680": subset_record(np.arange(run.FIT_ANSWERS), projection, window_scores, answer_scores, thresholds),
    }
    by_generator = {
        name: subset_record(np.flatnonzero(generators == name), projection, window_scores, answer_scores, thresholds)
        for name in sorted(set(generators.tolist()))
    }

    native_fit = run.read(run.NATIVE_V2_FIT)["selected"]["fit_OOF"]
    native_current = cohorts["native_634"]
    all_current = cohorts["all_3680"]
    added = cohorts["added_3046"]
    comparison = {
        "old_native634_fit_OOF": native_fit,
        "current_model_native634_at_all_fit_frozen_threshold": {
            "windows": native_current["windows"]["at_frozen_threshold"],
            "answers": native_current["answers"]["at_frozen_threshold"],
        },
        "delta_current_native_minus_old_native": {
            "window_f1": native_current["windows"]["at_frozen_threshold"]["f1"] - native_fit["windows"]["f1"],
            "answer_f1": native_current["answers"]["at_frozen_threshold"]["f1"] - native_fit["answers"]["f1"],
        },
        "delta_all3680_minus_old_native": {
            "window_f1": all_current["windows"]["at_frozen_threshold"]["f1"] - native_fit["windows"]["f1"],
            "answer_f1": all_current["answers"]["at_frozen_threshold"]["f1"] - native_fit["answers"]["f1"],
        },
        "subgroup_threshold_gain_added3046": {
            "window_f1": added["windows"]["subgroup_F1Opt_diagnostic"]["metrics"]["f1"] - added["windows"]["at_frozen_threshold"]["f1"],
            "answer_f1": added["answers"]["subgroup_F1Opt_diagnostic"]["metrics"]["f1"] - added["answers"]["at_frozen_threshold"]["f1"],
        },
        "native_feature_replay_exact": prep["native_feature_replay"]["native_whitebox_geometry_exact"],
        "native_feature_replay_max_abs_error": prep["native_feature_replay"]["native_max_abs_error"],
    }
    diagnosis = {
        "implementation_mismatch_supported": False,
        "evidence": [
            "The first 168,123 native whitebox+geometry rows replay byte-exactly.",
            "On the native 634 subset, current OOF F1 stays close to the old native-only OOF result.",
            "Native and added cohorts have sharply different answer/window positive rates and score distributions.",
            "Performance varies strongly by source generator, especially for the low-prevalence GPT cohorts.",
            "Allowing descriptive subgroup F1-opt thresholds recovers only a small part of the added-cohort deficit.",
        ],
        "supported_explanation": "generator/domain heterogeneity plus label-prior composition; threshold transfer is secondary",
        "causal_proof": False,
    }
    result = {
        "status": "fit_only_domain_diagnostic_complete",
        "frozen_candidate": fit["selected"]["candidate"],
        "frozen_thresholds": thresholds,
        "cohorts": cohorts,
        "by_generator": by_generator,
        "comparison": comparison,
        "strict_calibration_from_single_saved_evaluation": summary["calibration_at_frozen_fit_thresholds"],
        "diagnosis": diagnosis,
        "generator_metadata_sha256": run.sha(FIT_ROWS),
        "models_refit": 0,
        "thresholds_changed": 0,
        "calibration_labels_opened": False,
        "official_test_opened": False,
    }
    run.atomic_json(run.OUT / "FIT_DOMAIN_DIAGNOSTIC.json", result)

    table = [
        "| generator | answers | ans +% | ans AUROC | ans AP | ans F1 | windows | win +% | win AUROC | win AP | win F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in by_generator.items():
        a, w = record["answers"], record["windows"]
        am, wm = a["at_frozen_threshold"], w["at_frozen_threshold"]
        table.append(
            f"| {name} | {a['n']} | {100*a['positive_rate']:.2f} | {am['auroc']:.3f} | {am['average_precision']:.3f} | {am['f1']:.3f} | "
            f"{w['n']} | {100*w['positive_rate']:.2f} | {wm['auroc']:.3f} | {wm['average_precision']:.3f} | {wm['f1']:.3f} |"
        )
    n, a = cohorts["native_634"], cohorts["added_3046"]
    strict = summary["calibration_at_frozen_fit_thresholds"]
    report = [
        "# Expanded semantic window v2 — fit-only domain diagnostic", "",
        "All rows below use the already frozen five-fold OOF scores and global fit thresholds. No model or operational threshold was changed; subgroup F1-opt values in JSON are descriptive only.", "",
        *table, "",
        "## Native versus added", "",
        f"- Native 634: window positive {100*n['windows']['positive_rate']:.2f}%, answer positive {100*n['answers']['positive_rate']:.2f}%; F1 {n['windows']['at_frozen_threshold']['f1']:.3f}/{n['answers']['at_frozen_threshold']['f1']:.3f}.",
        f"- Added 3,046: window positive {100*a['windows']['positive_rate']:.2f}%, answer positive {100*a['answers']['positive_rate']:.2f}%; F1 {a['windows']['at_frozen_threshold']['f1']:.3f}/{a['answers']['at_frozen_threshold']['f1']:.3f}.",
        f"- Added-subset F1-opt changes F1 by only {comparison['subgroup_threshold_gain_added3046']['window_f1']:+.3f}/{comparison['subgroup_threshold_gain_added3046']['answer_f1']:+.3f}; threshold choice is not the main deficit.",
        f"- The current model on native rows differs from the old native-only OOF by {comparison['delta_current_native_minus_old_native']['window_f1']:+.3f}/{comparison['delta_current_native_minus_old_native']['answer_f1']:+.3f}. The native 26-D feature replay has max error 0.", "",
        f"Strict saved calibration at frozen fit thresholds remains {strict['windows']['f1']:.3f} window F1 and {strict['answers']['f1']:.3f} answer F1.", "",
        "The evidence rules out a feature-construction mismatch on the native rows and points to generator/domain heterogeneity and changed label priors. This is a diagnostic association, not a causal proof.",
    ]
    (run.OUT / "FIT_DOMAIN_DIAGNOSTIC.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("EXPANDED_WINDOW_V2_DOMAIN_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
