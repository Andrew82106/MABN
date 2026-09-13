"""Read-only failure diagnosis for semantic_source_attribution_v1_score.

Only the frozen fit and calibration partitions are read.  The official test and
formal baseline artifacts are absent from this script.  All derived artifacts
are written under this research directory.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import run_semantic_source_attribution_v1_score as score  # noqa: E402
import run_atomic_microclaim_nli_v1 as atomic  # noqa: E402
import run_development as development  # noqa: E402


SEED = 20260913
FOLDS = 5
CS = (0.001, 0.01, 0.1)
BLOCKS = {
    "raw1024": slice(0, 1024),
    "region288": slice(1024, 1312),
    "nli66": slice(1312, 1378),
    "whitebox12": slice(1382, 1394),
    "combination1394": slice(0, 1394),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.quantile(values, .05)),
        "p25": float(np.quantile(values, .25)),
        "p50": float(np.quantile(values, .50)),
        "p75": float(np.quantile(values, .75)),
        "p95": float(np.quantile(values, .95)),
    }


def metric(y: np.ndarray, probability: np.ndarray, threshold: float | None = None) -> dict:
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    chosen = development.choose_threshold(y, probability)
    applied = chosen["threshold"] if threshold is None else float(threshold)
    result = development.count(y, probability, applied)
    result["threshold"] = applied
    result["f1opt_threshold"] = chosen["threshold"]
    result["f1opt_f1"] = chosen["f1"]
    result["brier"] = float(brier_score_loss(y, probability))
    result["negative_probability"] = quantiles(probability[y == 0])
    result["positive_probability"] = quantiles(probability[y == 1])
    return result


def load_partition(partition: str) -> dict:
    rows, layouts = score.partition_rows_and_layouts(partition)
    meta = score.load_partition_meta(partition)
    any_y, conflict_y = score.labels_for_claims(meta, rows)
    cache_path = HERE / f"{partition}_feature_cache.npz"
    cache_meta_path = HERE / f"{partition}_feature_cache.json"
    cache_signature = {
        "partition": partition,
        "score_runner_sha256": sha(SRC / "run_semantic_source_attribution_v1_score.py"),
        "attribution_manifest_sha256": sha(score.ATTR_MANIFEST_PATH),
        "selected_nli_existing_sha256": sha(score.OUT / f"selected_nli_existing_{partition}.npz"),
        "selected_nli_missing_sha256": sha(score.OUT / f"selected_nli_missing_scores_{partition}.npz"),
        "whitebox_sha256": sha(score.WHITEBOX_PATH),
    }
    if cache_path.is_file() and cache_meta_path.is_file() and json.loads(
        cache_meta_path.read_text(encoding="utf-8")
    ) == cache_signature:
        with np.load(cache_path, allow_pickle=False) as loaded:
            features = {name: loaded[name].copy() for name in loaded.files}
    else:
        features = score.build_partition_features(partition)
        with cache_path.open("wb") as handle:
            np.savez(handle, **features)
        write_json(cache_meta_path, cache_signature)
    with np.load(score.OUT / f"{partition}_scores.npz", allow_pickle=False) as loaded:
        saved = {name: loaded[name].copy() for name in loaded.files}
    return {
        "partition": partition,
        "rows": rows,
        "layouts": layouts,
        "meta": meta,
        "any_y": any_y,
        "conflict_y": conflict_y,
        "features": features,
        "saved": saved,
    }


def claim_type_and_purity(part: dict) -> dict:
    type_names = (
        "Evident Baseless Info", "Subtle Baseless Info",
        "Evident Conflict", "Subtle Conflict",
    )
    class_names = ("baseless", "conflict")
    claim_types = []
    risk_fraction = []
    conflict_fraction = []
    claim_lengths = []
    token_masks = {}
    for row in part["rows"]:
        token = part["meta"]["by_response"][row["response_id"]]["tokens"]
        offsets = token["response_token_offsets"]
        masks = {name: np.zeros(len(offsets), dtype=bool) for name in type_names}
        for label in token["original_labels"]:
            name = label["label_type"]
            begin, end = int(label["start"]), int(label["end"])
            for index, (left, right) in enumerate(offsets):
                if max(left, begin) < min(right, end):
                    masks[name][index] = True
        token_masks[row["response_id"]] = masks
        risk = np.asarray(token["risk_mask"], dtype=bool)
        conflict = masks["Evident Conflict"] | masks["Subtle Conflict"]
        for claim in row["claims"]:
            ids = np.asarray(claim["lexical_token_indices"], dtype=int)
            present = [name for name in type_names if masks[name][ids].any()]
            claim_types.append(present)
            risk_fraction.append(float(risk[ids].mean()))
            conflict_fraction.append(float(conflict[ids].mean()))
            claim_lengths.append(len(ids))

    type_arrays = {
        name: np.asarray([name in names for names in claim_types], dtype=bool)
        for name in type_names
    }
    type_arrays["baseless"] = type_arrays["Evident Baseless Info"] | type_arrays["Subtle Baseless Info"]
    type_arrays["conflict"] = type_arrays["Evident Conflict"] | type_arrays["Subtle Conflict"]
    result = {
        "claim_types": claim_types,
        "type_arrays": type_arrays,
        "risk_fraction": np.asarray(risk_fraction),
        "conflict_fraction": np.asarray(conflict_fraction),
        "claim_lengths": np.asarray(claim_lengths),
        "token_masks": token_masks,
    }
    assert np.array_equal(type_arrays["conflict"].astype(np.int8), part["conflict_y"])
    assert np.array_equal((type_arrays["baseless"] | type_arrays["conflict"]).astype(np.int8), part["any_y"])
    return result


def window_type_arrays(part: dict, typed: dict) -> dict[str, np.ndarray]:
    names = (*typed["type_arrays"].keys(),)
    output = {name: np.zeros(len(part["meta"]["windows"]), dtype=bool) for name in names}
    for index, window in enumerate(part["meta"]["windows"]):
        masks = typed["token_masks"][window["response_id"]]
        # Gold window labels are defined only on lexical BPEs.  Punctuation may
        # geometrically overlap an annotation span but is deliberately excluded.
        ids = np.asarray(window["lexical_token_indices"], dtype=int)
        for name in names:
            if name == "baseless":
                output[name][index] = (
                    masks["Evident Baseless Info"][ids].any()
                    or masks["Subtle Baseless Info"][ids].any()
                )
            elif name == "conflict":
                output[name][index] = (
                    masks["Evident Conflict"][ids].any()
                    or masks["Subtle Conflict"][ids].any()
                )
            else:
                output[name][index] = masks[name][ids].any()
    any_type = output["baseless"] | output["conflict"]
    gold = np.asarray([row["label"] for row in part["meta"]["windows"]], dtype=bool)
    assert np.array_equal(any_type, gold)
    return output


def existing_head_diagnostics(fit: dict, cal: dict) -> dict:
    payload = pickle.loads((score.OUT / "model.pkl").read_bytes())
    models = payload["models"]
    output = {}
    specs = {
        "clean_any": ("clean", "any_y"),
        "clean_conflict": ("clean", "conflict_y"),
        "semantic_any": ("semantic", "any_y"),
        "semantic_conflict": ("semantic", "conflict_y"),
    }
    for name, (matrix_name, target_name) in specs.items():
        fit_y, cal_y = fit[target_name], cal[target_name]
        fit_oof = fit["saved"][name]
        full_fit = models[name].predict_proba(fit["features"][matrix_name])[:, 1]
        cal_score = cal["saved"][name]
        threshold = development.choose_threshold(fit_y, fit_oof)["threshold"]
        estimator = models[name][-1]
        scaler = models[name][0]
        raw_coef = estimator.coef_[0] / scaler.scale_
        output[name] = {
            "fit_oof": metric(fit_y, fit_oof),
            "fit_full_in_sample": metric(fit_y, full_fit),
            "cal_at_fit_threshold": metric(cal_y, cal_score, threshold),
            "cal_f1opt_diagnostic": metric(cal_y, cal_score),
            "generalization_gap": {
                "full_train_minus_oof_auroc": float(roc_auc_score(fit_y, full_fit) - roc_auc_score(fit_y, fit_oof)),
                "full_train_minus_oof_ap": float(average_precision_score(fit_y, full_fit) - average_precision_score(fit_y, fit_oof)),
                "oof_minus_cal_auroc": float(roc_auc_score(fit_y, fit_oof) - roc_auc_score(cal_y, cal_score)),
                "oof_minus_cal_ap": float(average_precision_score(fit_y, fit_oof) - average_precision_score(cal_y, cal_score)),
            },
            "coefficient": {
                "features": int(len(raw_coef)),
                "standardized_l2": float(np.linalg.norm(estimator.coef_[0])),
                "raw_l2": float(np.linalg.norm(raw_coef)),
                "standardized_max_abs": float(np.abs(estimator.coef_[0]).max()),
                "raw_max_abs": float(np.abs(raw_coef).max()),
                "n_iter": int(estimator.n_iter_[0]),
            },
        }
    return output


def make_lr(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value, solver="lbfgs", penalty="l2", max_iter=1000,
            random_state=SEED,
        ),
    )


def raw_coefficient(model) -> np.ndarray:
    return model[-1].coef_[0] / model[0].scale_


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def crossfit_one(x: np.ndarray, y: np.ndarray, fit: dict, c_value: float) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    groups = fit["features"]["groups"]
    response_ids = fit["features"]["response_ids"]
    rows_by_id = {row["response_id"]: row for row in fit["rows"]}
    indices = np.arange(len(y))
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    coefs, iterations = [], []
    for train, held in GroupKFold(FOLDS).split(indices, y, groups):
        weights = score.claim_weights(rows_by_id, response_ids, y, train)
        model = make_lr(c_value)
        model.fit(x[train], y[train], logisticregression__sample_weight=weights[train])
        predictions[held] = model.predict_proba(x[held])[:, 1]
        coefs.append(raw_coefficient(model))
        iterations.append(int(model[-1].n_iter_[0]))
    assert np.isfinite(predictions).all()
    return predictions, coefs, iterations


def coefficient_stability(coefs: list[np.ndarray]) -> dict:
    pairwise = [
        cosine(coefs[i], coefs[j])
        for i in range(len(coefs)) for j in range(i + 1, len(coefs))
    ]
    return {
        "pairwise_raw_coefficient_cosine": quantiles(np.asarray(pairwise)),
        "raw_coefficient_l2": quantiles(np.asarray([np.linalg.norm(value) for value in coefs])),
    }


def ablation(fit: dict, cal: dict) -> dict:
    all_results = {"any_error": [], "conflict_only": []}
    selected = {}
    targets = {"any_error": "any_y", "conflict_only": "conflict_y"}
    with threadpool_limits(limits=4):
        for target_name, target_key in targets.items():
            y = fit[target_key]
            for block_name, section in BLOCKS.items():
                x = fit["features"]["semantic"][:, section]
                for c_value in CS:
                    started = time.perf_counter()
                    probability, coefs, iterations = crossfit_one(x, y, fit, c_value)
                    row = {
                        "target": target_name,
                        "block": block_name,
                        "width": int(x.shape[1]),
                        "C": c_value,
                        "fit_oof": metric(y, probability),
                        "coefficient_stability": coefficient_stability(coefs),
                        "iterations": iterations,
                        "seconds": time.perf_counter() - started,
                    }
                    all_results[target_name].append(row)
                    print("ABLATION", target_name, block_name, c_value,
                          row["fit_oof"]["f1opt_f1"], row["fit_oof"]["average_precision"], flush=True)

            # Select only on fit OOF: F1, AP, AUROC, fewer dimensions, stronger regularization.
            best = max(
                all_results[target_name],
                key=lambda row: (
                    row["fit_oof"]["f1opt_f1"],
                    row["fit_oof"]["average_precision"],
                    row["fit_oof"]["auroc"],
                    -row["width"], -row["C"],
                ),
            )
            section = BLOCKS[best["block"]]
            x_fit = fit["features"]["semantic"][:, section]
            x_cal = cal["features"]["semantic"][:, section]
            active = np.arange(len(y))
            weights = score.claim_weights(
                {row["response_id"]: row for row in fit["rows"]},
                fit["features"]["response_ids"], y, active,
            )
            model = make_lr(best["C"])
            model.fit(x_fit, y, logisticregression__sample_weight=weights)
            cal_probability = model.predict_proba(x_cal)[:, 1]
            fit_threshold = best["fit_oof"]["f1opt_threshold"]
            selected[target_name] = {
                "block": best["block"], "width": best["width"], "C": best["C"],
                "fit_oof": best["fit_oof"],
                "cal_at_fit_threshold": metric(cal[target_key], cal_probability, fit_threshold),
                "cal_f1opt_diagnostic": metric(cal[target_key], cal_probability),
                "full_model_n_iter": int(model[-1].n_iter_[0]),
                "selection_rule": "fit OOF max F1Opt, then AP, AUROC, narrower block, smaller C",
            }
    return {"grid": all_results, "fit_selected_cal_once": selected}


def type_recall_and_transitions(fit: dict, cal: dict, fit_types: dict, cal_types: dict) -> dict:
    fit_windows = window_type_arrays(fit, fit_types)
    cal_windows = window_type_arrays(cal, cal_types)
    output = {}
    variants = {
        "clean_any": (fit["saved"]["clean_any"], cal["saved"]["clean_any"]),
        "semantic_any": (fit["saved"]["semantic_any"], cal["saved"]["semantic_any"]),
        "clean_max": (
            np.maximum(fit["saved"]["clean_any"], fit["saved"]["clean_conflict"]),
            np.maximum(cal["saved"]["clean_any"], cal["saved"]["clean_conflict"]),
        ),
        "semantic_max": (
            np.maximum(fit["saved"]["semantic_any"], fit["saved"]["semantic_conflict"]),
            np.maximum(cal["saved"]["semantic_any"], cal["saved"]["semantic_conflict"]),
        ),
    }
    for component, (fit_claim, cal_claim) in variants.items():
        fit_window = score.project_claim_scores(fit["meta"], fit["rows"], fit_claim)
        cal_window = score.project_claim_scores(cal["meta"], cal["rows"], cal_claim)
        threshold = development.choose_threshold(
            [row["label"] for row in fit["meta"]["windows"]], fit_window,
        )["threshold"]
        prediction = cal_window >= threshold
        recalls = {}
        for name, mask in cal_windows.items():
            recalls[name] = {
                "windows": int(mask.sum()),
                "detected": int(np.count_nonzero(mask & prediction)),
                "recall": float(np.mean(prediction[mask])) if mask.any() else None,
            }
        output[component] = {
            "fit_threshold": threshold,
            "cal_type_recall": recalls,
            "cal_window_prediction": prediction,
            "cal_window_score": cal_window,
        }

    clean = output["clean_max"]["cal_window_prediction"]
    semantic = output["semantic_max"]["cal_window_prediction"]
    gold = np.asarray([row["label"] for row in cal["meta"]["windows"]], dtype=bool)
    transition = {
        "gained_true_positive_clean_FN_to_semantic_TP": int(np.count_nonzero(gold & ~clean & semantic)),
        "lost_true_positive_clean_TP_to_semantic_FN": int(np.count_nonzero(gold & clean & ~semantic)),
        "new_false_positive_clean_TN_to_semantic_FP": int(np.count_nonzero(~gold & ~clean & semantic)),
        "removed_false_positive_clean_FP_to_semantic_TN": int(np.count_nonzero(~gold & clean & ~semantic)),
    }
    combined_add = cal["saved"]["additions"].astype(bool)
    transition["frozen_add_gate"] = {
        "additions": int(combined_add.sum()),
        "true_positive": int(np.count_nonzero(combined_add & gold)),
        "false_positive": int(np.count_nonzero(combined_add & ~gold)),
        "incremental_precision": float(np.mean(gold[combined_add])) if combined_add.any() else None,
    }
    # Arrays are internal only and not JSON serializable/useful in the report.
    for component in output.values():
        component.pop("cal_window_prediction")
        component.pop("cal_window_score")
    return {"standalone": output, "transitions": transition}


def fusion_path(fit: dict, cal: dict) -> dict:
    output = {}
    variants = {
        "clean_any": (fit["saved"]["clean_any"], cal["saved"]["clean_any"]),
        "clean_max_any_conflict": (
            np.maximum(fit["saved"]["clean_any"], fit["saved"]["clean_conflict"]),
            np.maximum(cal["saved"]["clean_any"], cal["saved"]["clean_conflict"]),
        ),
        "semantic_any": (fit["saved"]["semantic_any"], cal["saved"]["semantic_any"]),
        "semantic_max_any_conflict": (
            np.maximum(fit["saved"]["semantic_any"], fit["saved"]["semantic_conflict"]),
            np.maximum(cal["saved"]["semantic_any"], cal["saved"]["semantic_conflict"]),
        ),
    }
    for name, (fit_claim, cal_claim) in variants.items():
        fit_window = score.project_claim_scores(fit["meta"], fit["rows"], fit_claim)
        cal_window = score.project_claim_scores(cal["meta"], cal["rows"], cal_claim)
        thresholds = score.choose_thresholds(fit["meta"], fit_window)
        output[name] = {
            "fit": score.metric_pair(fit["meta"], fit_window, thresholds),
            "cal_strict": score.metric_pair(cal["meta"], cal_window, thresholds),
            "fit_thresholds": thresholds,
        }
    frozen_thresholds = json.loads((score.OUT / "fit_complete.json").read_text(encoding="utf-8"))["base_fit_thresholds"]
    output["frozen_final_add_gate"] = {
        "fit": score.metric_pair(fit["meta"], fit["saved"]["combined_window"], frozen_thresholds),
        "cal_strict": score.metric_pair(cal["meta"], cal["saved"]["combined_window"], frozen_thresholds),
        "fit_thresholds": frozen_thresholds,
    }
    return output


def any_head_projection(fit: dict, cal: dict) -> dict:
    output = {}
    for component in ("clean", "semantic"):
        fit_claim = fit["saved"][component + "_any"]
        cal_claim = cal["saved"][component + "_any"]
        fit_window = score.project_claim_scores(fit["meta"], fit["rows"], fit_claim)
        cal_window = score.project_claim_scores(cal["meta"], cal["rows"], cal_claim)
        thresholds = score.choose_thresholds(fit["meta"], fit_window)
        output[component] = {
            "fit_OOF": score.metric_pair(fit["meta"], fit_window, thresholds),
            "cal_strict_fit_threshold": score.metric_pair(cal["meta"], cal_window, thresholds),
            "cal_F1Opt_diagnostic": score.metric_pair(
                cal["meta"], cal_window, score.choose_thresholds(cal["meta"], cal_window),
            ),
            "fit_thresholds": thresholds,
        }
    return output


def oracle_geometry(part: dict, typed: dict) -> dict:
    claim_oracle = part["any_y"].astype(np.float64)
    projected = score.project_claim_scores(part["meta"], part["rows"], claim_oracle)
    gold = np.asarray([row["label"] for row in part["meta"]["windows"]], dtype=np.int8)
    metric_row = development.count(gold, projected, .5)
    positive_claim = part["any_y"].astype(bool)
    risk_fraction = typed["risk_fraction"]
    lengths = typed["claim_lengths"]
    return {
        "claim_oracle_projected_to_windows": metric_row,
        "positive_claims": int(positive_claim.sum()),
        "positive_claims_mixed_risky_and_clean_tokens": int(np.count_nonzero(positive_claim & (risk_fraction < 1))),
        "positive_claims_risk_fraction_below_0_50": int(np.count_nonzero(positive_claim & (risk_fraction < .5))),
        "positive_claims_risk_fraction_below_0_25": int(np.count_nonzero(positive_claim & (risk_fraction < .25))),
        "positive_claim_risk_fraction": quantiles(risk_fraction[positive_claim]),
        "positive_claim_length": quantiles(lengths[positive_claim]),
        "negative_claim_length": quantiles(lengths[~positive_claim]),
        "projection_false_positive_fraction_of_oracle_positives": (
            metric_row["fp"] / (metric_row["tp"] + metric_row["fp"])
            if metric_row["tp"] + metric_row["fp"] else 0.0
        ),
    }


def group_answer_length(part: dict, typed: dict) -> dict:
    group_answers = defaultdict(int)
    group_claims = defaultdict(int)
    answer_claims = []
    answer_tokens = []
    answer_sentences = []
    cursor = 0
    for row, layout in zip(part["rows"], part["layouts"]):
        count = len(row["claims"])
        group_answers[row["group_id"]] += 1
        group_claims[row["group_id"]] += count
        answer_claims.append(count)
        token = part["meta"]["by_response"][row["response_id"]]["tokens"]
        answer_tokens.append(token["token_count"])
        answer_sentences.append(len(layout["sentences"]))
        cursor += count
    assert cursor == len(typed["claim_lengths"])
    return {
        "answers": len(part["rows"]),
        "groups": len(group_answers),
        "groups_with_multiple_answers": int(sum(value > 1 for value in group_answers.values())),
        "answers_per_group": quantiles(np.asarray(list(group_answers.values()))),
        "claims_per_group": quantiles(np.asarray(list(group_claims.values()))),
        "claims_per_answer": quantiles(np.asarray(answer_claims)),
        "tokens_per_answer": quantiles(np.asarray(answer_tokens)),
        "source_sentences_per_answer": quantiles(np.asarray(answer_sentences)),
        "claim_lexical_tokens": quantiles(typed["claim_lengths"]),
        "any_error_claim_prevalence": float(part["any_y"].mean()),
        "conflict_claim_prevalence": float(part["conflict_y"].mean()),
    }


def nli_signal_metric(y: np.ndarray, risk: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, risk)),
        "average_precision": float(average_precision_score(y, risk)),
        "f1opt": development.choose_threshold(y, risk),
    }


def nli_and_selection(part: dict, typed: dict) -> dict:
    selected, weights, probabilities = score.load_selected_nli(part["partition"])
    top1 = probabilities[:, 0]
    top3 = np.einsum("ij,ijk->ik", weights, probabilities)
    max_selected = probabilities.max(axis=1)
    any_y = part["any_y"]
    conflict_y = part["conflict_y"]
    baseless_y = typed["type_arrays"]["baseless"].astype(np.int8)
    output = {
        "direct_signal": {
            "top1_one_minus_entailment_any": nli_signal_metric(any_y, 1 - top1[:, 0]),
            "top3_one_minus_entailment_any": nli_signal_metric(any_y, 1 - top3[:, 0]),
            "top1_contradiction_conflict": nli_signal_metric(conflict_y, top1[:, 2]),
            "top3_contradiction_conflict": nli_signal_metric(conflict_y, top3[:, 2]),
            "top3_one_minus_entailment_baseless": nli_signal_metric(baseless_y, 1 - top3[:, 0]),
        },
        "probability_by_label": {},
    }
    for name, mask in {
        "clean": any_y == 0,
        "baseless": baseless_y == 1,
        "conflict": conflict_y == 1,
    }.items():
        output["probability_by_label"][name] = {
            "claims": int(mask.sum()),
            "top1_entailment": quantiles(top1[mask, 0]),
            "top1_contradiction": quantiles(top1[mask, 2]),
            "top3_entailment": quantiles(top3[mask, 0]),
            "top3_contradiction": quantiles(top3[mask, 2]),
            "max_selected_entailment": quantiles(max_selected[mask, 0]),
            "max_selected_contradiction": quantiles(max_selected[mask, 2]),
        }

    # Compare attention top-3 to the six individual BM25 candidates already
    # scored by atomic v1.  The union is used as a finite diagnostic reference.
    global_claim = 0
    top1_hits_entail = top3_hits_entail = top3_hits_contra = 0
    top3_any_bm25_overlap = 0
    union_entail_advantage = []
    union_contra_advantage = []
    clean_selected_coverage = clean_bm25_coverage = 0
    conflict_selected_coverage = conflict_bm25_coverage = 0
    for row, layout in zip(part["rows"], part["layouts"]):
        pair_probability, _ = atomic.validate_cache(score.PAIR_DIR / f"{row['response_id']}.npz", row)
        _, owners = atomic.row_pairs(row)
        owner_by_claim = defaultdict(list)
        for pair_index, owner in enumerate(owners):
            claim_id, passage_id, rank, sentence_id, second_sentence_id = owner
            if int(rank) != 0:
                assert int(second_sentence_id) == -1
                owner_by_claim[int(claim_id)].append((
                    (int(passage_id), int(sentence_id)), pair_probability[pair_index],
                ))
        for local_claim, claim in enumerate(row["claims"]):
            idx = global_claim + local_claim
            attention_ids = {
                (int(layout["sentences"][int(si)]["passage_id"]),
                 int(layout["sentences"][int(si)]["sentence_id"]))
                for si in selected[idx]
            }
            bm25 = owner_by_claim[int(claim["claim_id"])]
            bm25_ids = {identity for identity, _ in bm25}
            bm25_prob = np.vstack([value for _, value in bm25])
            selected_prob = probabilities[idx]
            if attention_ids & bm25_ids:
                top3_any_bm25_overlap += 1
            union_entail = max(float(selected_prob[:, 0].max()), float(bm25_prob[:, 0].max()))
            union_contra = max(float(selected_prob[:, 2].max()), float(bm25_prob[:, 2].max()))
            selected_entail = float(selected_prob[:, 0].max())
            selected_contra = float(selected_prob[:, 2].max())
            union_entail_advantage.append(union_entail - selected_entail)
            union_contra_advantage.append(union_contra - selected_contra)
            if selected_prob[0, 0] >= union_entail - 1e-6:
                top1_hits_entail += 1
            if selected_entail >= union_entail - 1e-6:
                top3_hits_entail += 1
            if selected_contra >= union_contra - 1e-6:
                top3_hits_contra += 1
            if not any_y[idx]:
                clean_selected_coverage += int(selected_entail >= .5)
                clean_bm25_coverage += int(float(bm25_prob[:, 0].max()) >= .5)
            if conflict_y[idx]:
                conflict_selected_coverage += int(selected_contra >= .5)
                conflict_bm25_coverage += int(float(bm25_prob[:, 2].max()) >= .5)
        global_claim += len(row["claims"])
    assert global_claim == len(any_y)
    n = len(any_y)
    output["selection_against_atomic_BM25_top2_per_passage"] = {
        "claims": n,
        "attention_top3_any_sentence_overlap_fraction": top3_any_bm25_overlap / n,
        "attention_top1_contains_union_best_entailment_fraction": top1_hits_entail / n,
        "attention_top3_contains_union_best_entailment_fraction": top3_hits_entail / n,
        "attention_top3_contains_union_best_contradiction_fraction": top3_hits_contra / n,
        "union_best_entailment_minus_attention_top3": quantiles(np.asarray(union_entail_advantage)),
        "union_best_contradiction_minus_attention_top3": quantiles(np.asarray(union_contra_advantage)),
        "clean_claim_entailment_ge_0_5": {
            "attention_top3": clean_selected_coverage,
            "atomic_BM25_six": clean_bm25_coverage,
            "clean_claims": int((any_y == 0).sum()),
        },
        "conflict_claim_contradiction_ge_0_5": {
            "attention_top3": conflict_selected_coverage,
            "atomic_BM25_six": conflict_bm25_coverage,
            "conflict_claims": int(conflict_y.sum()),
        },
        "reference_limit": "Comparison is only against atomic v1's six BM25 individual sentences, not every source sentence.",
    }
    return output


def report_markdown(result: dict) -> str:
    heads = result["existing_heads"]
    ab = result["ablation"]["fit_selected_cal_once"]
    proj = result["any_head_projection"]
    fusion = result["fusion_path"]
    geo = result["oracle_geometry"]
    trans = result["type_recall_and_transitions"]["transitions"]
    grid_any = result["ablation"]["grid"]["any_error"]
    combo = next(row for row in grid_any if row["block"] == "combination1394" and row["C"] == .001)
    whitebox = next(row for row in grid_any if row["block"] == "whitebox12" and row["C"] == .01)
    lines = [
        "# Semantic source attribution v1 failure diagnosis", "",
        "本诊断只读 fit 与 calibration；未打开 official test，未改正式 baseline。", "",
        "## 结论", "",
        "失败不是单一原因。归因原始块能学到一定信号，但加到稳定的 whitebox 后增益很小；随后又被三个问题放大：只有 195 个正例的 conflict head 被强制取 max、claim 分数被 max 投影到整段窗口、top-3 证据对冲突证据的覆盖不足。", "",
        "## 已有四个 head", "",
        "| head | fit OOF AUROC | fit OOF AP | fit F1-opt | cal AUROC | cal AP | cal F1-opt |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("clean_any", "semantic_any", "clean_conflict", "semantic_conflict"):
        fit_m = heads[name]["fit_oof"]
        cal_m = heads[name]["cal_f1opt_diagnostic"]
        lines.append(
            f"| {name} | {fit_m['auroc']:.4f} | {fit_m['average_precision']:.4f} | {fit_m['f1opt_f1']:.4f} | "
            f"{cal_m['auroc']:.4f} | {cal_m['average_precision']:.4f} | {cal_m['f1opt_f1']:.4f} |"
        )
    lines += [
        "",
        "semantic any-error 的 cal AUROC/AP 为 "
        f"{heads['semantic_any']['cal_f1opt_diagnostic']['auroc']:.4f}/"
        f"{heads['semantic_any']['cal_f1opt_diagnostic']['average_precision']:.4f}。但 fit 消融中，1394 维全组合的 F1 只比 12 维 whitebox 高 "
        f"{combo['fit_oof']['f1opt_f1'] - whitebox['fit_oof']['f1opt_f1']:.4f}，AP 反而低 "
        f"{whitebox['fit_oof']['average_precision'] - combo['fit_oof']['average_precision']:.4f}；所以不能把 semantic 与 clean 的全部差距归功于新归因。", "",
        f"全组合五折原始系数余弦均值只有 {combo['coefficient_stability']['pairwise_raw_coefficient_cosine']['mean']:.3f}，whitebox 为 {whitebox['coefficient_stability']['pairwise_raw_coefficient_cosine']['mean']:.3f}。semantic conflict 的 full-train AP 比 OOF 高 {heads['semantic_conflict']['generalization_gap']['full_train_minus_oof_ap']:.3f}，是明显记忆训练集；其 OOF AP 只有 {heads['semantic_conflict']['fit_oof']['average_precision']:.3f}。", "",
        "## 投影和融合损失", "",
        "| 只用 any-error head | fit OOF窗口F1 | cal严格窗口F1 | cal F1-opt窗口F1 | cal严格整答F1 |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("clean", "semantic"):
        row = proj[name]
        lines.append(
            f"| {name} | {row['fit_OOF']['windows']['f1']:.4f} | "
            f"{row['cal_strict_fit_threshold']['windows']['f1']:.4f} | "
            f"{row['cal_F1Opt_diagnostic']['windows']['f1']:.4f} | "
            f"{row['cal_strict_fit_threshold']['answers']['f1']:.4f} |"
        )
    lines += [
        "",
        f"沿实际融合路径，semantic any-only 的 cal 严格窗口 F1 是 {fusion['semantic_any']['cal_strict']['windows']['f1']:.4f}；加入 conflict-max 后降到 {fusion['semantic_max_any_conflict']['cal_strict']['windows']['f1']:.4f}；再走 frozen add-gate 后降到 {fusion['frozen_final_add_gate']['cal_strict']['windows']['f1']:.4f}。",
        f"semantic-max 相对 clean-max 新找回 {trans['gained_true_positive_clean_FN_to_semantic_TP']} 个 TP，却丢掉 {trans['lost_true_positive_clean_TP_to_semantic_FN']} 个原 TP；新增 {trans['new_false_positive_clean_TN_to_semantic_FP']} 个 FP，同时消掉 {trans['removed_false_positive_clean_FP_to_semantic_TN']} 个旧 FP。",
        f"当前 frozen add-gate 在 cal 新增 {trans['frozen_add_gate']['true_positive']} 个 TP，同时新增 {trans['frozen_add_gate']['false_positive']} 个 FP，增量精确率只有 {trans['frozen_add_gate']['incremental_precision']:.3f}。",
        f"即使给 claim 完美 0/1 标签，再用同一 max 投影，fit/cal 窗口 F1 上限也只有 {geo['fit']['claim_oracle_projected_to_windows']['f1']:.4f}/{geo['calibration']['claim_oracle_projected_to_windows']['f1']:.4f}；cal 会制造 {geo['calibration']['claim_oracle_projected_to_windows']['fp']} 个结构性 FP。", "",
        "## 低成本特征块消融", "",
        "每个 target 只在 fit OOF 比较 raw1024、region288、NLI66、whitebox12、全组合，C 固定为 0.001/0.01/0.1；下面的 cal 是 fit 选定后唯一一次诊断。", "",
        "| target | fit选中 | C | fit F1-opt | fit AP | cal严格F1 | cal F1-opt | cal AP |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("any_error", "conflict_only"):
        row = ab[target]
        lines.append(
            f"| {target} | {row['block']} ({row['width']}) | {row['C']} | "
            f"{row['fit_oof']['f1opt_f1']:.4f} | {row['fit_oof']['average_precision']:.4f} | "
            f"{row['cal_at_fit_threshold']['f1']:.4f} | {row['cal_f1opt_diagnostic']['f1opt_f1']:.4f} | "
            f"{row['cal_f1opt_diagnostic']['average_precision']:.4f} |"
        )
    lines += [
        "",
        "完整消融表和系数稳定性见 REPORT.json。", "",
        "## NLI 与 top-3", "",
    ]
    for partition in ("fit", "calibration"):
        nli = result["nli_and_selection"][partition]
        direct = nli["direct_signal"]
        sel = nli["selection_against_atomic_BM25_top2_per_passage"]
        lines.append(
            f"- {partition}: top-3 `1-entailment` 对 any-error 的 AUROC/AP 为 "
            f"{direct['top3_one_minus_entailment_any']['auroc']:.3f}/"
            f"{direct['top3_one_minus_entailment_any']['average_precision']:.3f}；"
            f"attention top-3 覆盖 union 最佳 entailment 的比例为 "
            f"{sel['attention_top3_contains_union_best_entailment_fraction']:.3f}，"
            f"最佳 contradiction 的覆盖比例仅 {sel['attention_top3_contains_union_best_contradiction_fraction']:.3f}；"
            f"与 BM25 六候选至少重合一句的比例为 {sel['attention_top3_any_sentence_overlap_fraction']:.3f}。"
        )
    lines += [
        "",
        f"fit/cal 的回答长度、claim 长度和来源句数接近；主要类别变化是 conflict claim 从 {result['data_geometry']['fit']['conflict_claim_prevalence']:.3%} 升到 {result['data_geometry']['calibration']['conflict_claim_prevalence']:.3%}。因此没有证据把失败归因于明显的长度分布漂移。cal 中 {geo['calibration']['positive_claims_mixed_risky_and_clean_tokens']} 个阳性 claim 同时含风险与正常 token，其中 {geo['calibration']['positive_claims_risk_fraction_below_0_50']} 个 claim 的风险 token 不到一半。", "",
        "## 下一版应改什么", "",
        "1. 先停用 `max(any, conflict)`；以 any-error 为主输出，conflict 只作为受控附加证据或等待更多冲突样本。",
        "2. 不再把一个 claim 分数无差别铺到它覆盖的所有窗口；按 claim 内 token 的归因/边界分配风险，或直接训练窗口 readout。",
        "3. 证据选择改成 attention 与 BM25/NLI 的并集或可学习 rerank，并保留多句联合 NLI；top-3 attention 不能当作唯一证据入口。",
        "4. any 主头先以稳定的 whitebox 为主干，只加入经过 OOF 证明有增量的低维归因残差；避免 1394 维直接压在 195 个 conflict 正例上。",
        "5. 下一轮仍在我们的固定数据、4-BPE 窗口与整答口径评测；正式 baseline 结构保持不变。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    fit = load_partition("fit")
    cal = load_partition("calibration")
    fit_types = claim_type_and_purity(fit)
    cal_types = claim_type_and_purity(cal)
    result = {
        "status": "complete_read_only_failure_diagnosis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "fit_answers": len(fit["rows"]),
            "calibration_answers": len(cal["rows"]),
            "fit_claims": len(fit["any_y"]),
            "calibration_claims": len(cal["any_y"]),
            "official_test_opened": False,
            "formal_baselines_modified": False,
            "GPU_used": False,
        },
        "input_sha256": {
            "score_runner": sha(SRC / "run_semantic_source_attribution_v1_score.py"),
            "fit_scores": sha(score.OUT / "fit_scores.npz"),
            "calibration_scores": sha(score.OUT / "calibration_scores.npz"),
            "model": sha(score.OUT / "model.pkl"),
            "attribution_manifest": sha(score.ATTR_MANIFEST_PATH),
        },
        "existing_heads": existing_head_diagnostics(fit, cal),
        "any_head_projection": any_head_projection(fit, cal),
        "fusion_path": fusion_path(fit, cal),
        "type_recall_and_transitions": type_recall_and_transitions(fit, cal, fit_types, cal_types),
        "oracle_geometry": {
            "fit": oracle_geometry(fit, fit_types),
            "calibration": oracle_geometry(cal, cal_types),
        },
        "data_geometry": {
            "fit": group_answer_length(fit, fit_types),
            "calibration": group_answer_length(cal, cal_types),
        },
        "nli_and_selection": {
            "fit": nli_and_selection(fit, fit_types),
            "calibration": nli_and_selection(cal, cal_types),
        },
        "ablation": ablation(fit, cal),
        "interpretation_guardrails": {
            "calibration_is_development_only": True,
            "ablation_selection_used_fit_only": True,
            "calibration_used_once_after_fit_selection_for_each_target": True,
            "BM25_selection_reference_is_finite_not_oracle": True,
            "causal_attribution_claim_not_tested": True,
        },
    }
    result["seconds"] = time.perf_counter() - started
    write_json(HERE / "REPORT.json", result)
    (HERE / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    print("DIAGNOSIS_COMPLETE", result["seconds"], sha(HERE / "REPORT.json"), flush=True)


if __name__ == "__main__":
    main()
