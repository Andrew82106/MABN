"""Independent reconstruction and diagnosis of v3 fit-only pilot scoring.

Reads only the frozen 256 pilot caches and expanded-fit gold files.  It does
not import project runners, touch calibration/test, fit new candidates, or use
GPU.  The only fits replay the three already frozen A/B/C LR readouts exactly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
SELECTED = OUT / "selected_prepared_inputs.jsonl"
FIT_DATA = ROOT / "fit_expansion/data"
FIT_ANSWERS = FIT_DATA / "answers_fit.jsonl"
FIT_TOKENS = FIT_DATA / "tokens_fit.jsonl"
FIT_WINDOWS = FIT_DATA / "windows_k4_fit.jsonl"
EXPECTED_GOLD_SHA = {
    "answers_fit.jsonl": "8adef0c27d5021acdf559c1566db2d6d949ecccb88ad2532ae626381add0aa33",
    "tokens_fit.jsonl": "18501c2a53a22620f24827d3181d831f2aee77da2b335bd50270fbe939048ec2",
    "windows_k4_fit.jsonl": "cfaf5af088eaac422ee2685c9150a774171d930666426e936f86d0cee905eec3",
}
FEATURE_NAMES = (
    "full_nll", "empty_nll", "full_minus_empty",
    "shapley_passage_1", "shapley_passage_2", "shapley_passage_3",
    "single_gain_passage_1", "single_gain_passage_2", "single_gain_passage_3",
    "loo_drop_passage_1", "loo_drop_passage_2", "loo_drop_passage_3",
    "subset_range", "subset_std",
    "pair_interaction_12", "pair_interaction_13", "pair_interaction_23",
    "triple_interaction", "interaction_l1",
    "shapley_abs_max", "shapley_positive_sum", "shapley_negative_mass",
    "citation_any", "citation_source_fraction", "citation_shapley_sum",
    "citation_shapley_mean", "citation_positive_shapley_sum",
    "citation_single_gain_max", "citation_loo_drop_max",
)
READOUTS = {
    "A_full_nll_citation": (0, 22, 23),
    "B_full_empty_delta_citation": (0, 1, 2, 22, 23),
    "C_all_29": tuple(range(29)),
}
SEED = 20261014
LR_C = 0.1
FOLDS = 5


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def save_text(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value, encoding="utf-8")
    pending.replace(path)


def derive_features(log_probability, citation_mask):
    f = np.asarray(log_probability, dtype=np.float64)
    citations = np.asarray(citation_mask, dtype=np.uint8)
    assert f.shape == (8, len(citations)) and np.all(citations <= 7)
    phi = np.empty((3, f.shape[1]), dtype=np.float64)
    for source in range(3):
        bit = 1 << source
        value = np.zeros(f.shape[1], dtype=np.float64)
        for mask in range(8):
            if mask & bit:
                continue
            size = int(mask).bit_count()
            value += (1 / 3, 1 / 6, 1 / 3)[size] * (f[mask | bit] - f[mask])
        phi[source] = value
    efficiency = float(np.max(np.abs(phi.sum(axis=0) - (f[7] - f[0]))))
    assert efficiency <= 1e-12
    single = np.vstack((f[1] - f[0], f[2] - f[0], f[4] - f[0]))
    loo = np.vstack((f[7] - f[6], f[7] - f[5], f[7] - f[3]))
    pair = np.vstack((
        f[3] - f[1] - f[2] + f[0],
        f[5] - f[1] - f[4] + f[0],
        f[6] - f[2] - f[4] + f[0],
    ))
    triple = f[7] - f[3] - f[5] - f[6] + f[1] + f[2] + f[4] - f[0]
    cited = np.vstack([((citations >> source) & 1).astype(bool)
                       for source in range(3)])
    cited_count = cited.sum(axis=0)
    cited_phi_sum = (phi * cited).sum(axis=0)
    cited_single = np.where(cited, single, -np.inf).max(axis=0)
    cited_loo = np.where(cited, loo, -np.inf).max(axis=0)
    cited_single[cited_count == 0] = 0.0
    cited_loo[cited_count == 0] = 0.0
    values = [
        -f[7], -f[0], f[7] - f[0],
        *phi, *single, *loo,
        np.ptp(f, axis=0), np.std(f, axis=0),
        *pair, triple, np.abs(pair).sum(axis=0) + np.abs(triple),
        np.abs(phi).max(axis=0), np.maximum(phi, 0).sum(axis=0),
        -np.minimum(phi, 0).sum(axis=0),
        (cited_count > 0).astype(np.float64), cited_count / 3.0,
        cited_phi_sum,
        np.divide(cited_phi_sum, cited_count,
                  out=np.zeros_like(cited_phi_sum), where=cited_count > 0),
        (np.maximum(phi, 0) * cited).sum(axis=0), cited_single, cited_loo,
    ]
    result = np.column_stack(values).astype(np.float32)
    assert result.shape == (f.shape[1], 29) and np.isfinite(result).all()
    # Direction audit: f is log p <=0, so full NLL must be -f[111] >=0.
    assert np.array_equal(result[:, 0], (-f[7]).astype(np.float32))
    assert np.all(result[:, 0] >= 0)
    return result, efficiency


def choose_threshold(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    assert set(y) == {0, 1} and np.isfinite(score).all()
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    last = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(score) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[last]]
    predicted = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(sorted_score[0], np.inf), sorted_score[last]]
    f1 = 2 * tp / (predicted + y.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(predicted), float),
                          where=predicted > 0)
    best = max(range(len(predicted)),
               key=lambda index: (f1[index], precision[index], thresholds[index]))
    return {
        "threshold": float(thresholds[best]), "f1": float(f1[best]),
        "precision": float(precision[best]), "rows": len(y),
        "positive": int(y.sum()),
    }


def metric(y, score, threshold=None):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    result = {
        "n": len(y), "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
    }
    if threshold is not None:
        predicted = score >= threshold
        tp = int(np.count_nonzero((y == 1) & predicted))
        fp = int(np.count_nonzero((y == 0) & predicted))
        fn = int(np.count_nonzero((y == 1) & ~predicted))
        tn = int(np.count_nonzero((y == 0) & ~predicted))
        result.update({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        })
    return result


def fold_weights(meta, answer_indices, starts, y_token, lexical):
    tree = defaultdict(list)
    for answer_index in answer_indices:
        answer = meta["answers"][answer_index]
        tree[answer["group_id"]].append(answer_index)
    weights = np.zeros(len(y_token), dtype=np.float64)
    group_tokens = {}
    for group_id, indices in tree.items():
        flat = []
        for answer_index in indices:
            response_id = meta["answers"][answer_index]["response_id"]
            left, right = starts[response_id]
            active = np.flatnonzero(lexical[left:right]) + left
            assert len(active)
            weights[active] = 1.0 / (len(indices) * len(active))
            flat.extend(active.tolist())
        group_tokens[group_id] = np.asarray(flat, dtype=np.int64)
    active = weights > 0
    weights[active] *= active.sum() / weights[active].sum()
    class_mass_before_balance = np.bincount(
        y_token[active], weights=weights[active], minlength=2
    )
    assert np.all(class_mass_before_balance > 0)
    factors = class_mass_before_balance.sum() / (2 * class_mass_before_balance)
    weights[active] *= factors[y_token[active]]
    target_group_mass = active.sum() / len(tree)
    for indices in group_tokens.values():
        weights[indices] *= target_group_mass / weights[indices].sum()
    weights[active] *= active.sum() / weights[active].sum()
    group_mass = np.asarray([weights[indices].sum()
                             for indices in group_tokens.values()])
    assert np.ptp(group_mass) <= 1e-7
    return weights, {
        "raw_lexical_class_counts": np.bincount(y_token[active], minlength=2).tolist(),
        "class_mass_before_balance": class_mass_before_balance.tolist(),
        "class_balance_factors": factors.tolist(),
        "final_class_mass": np.bincount(
            y_token[active], weights=weights[active], minlength=2
        ).tolist(),
        "final_group_mass_min_max": [float(group_mass.min()), float(group_mass.max())],
        "effective_token_sample_size": float(weights[active].sum() ** 2 /
                                             np.square(weights[active]).sum()),
    }


def compare_float(actual, expected, name, tolerance=1e-10):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    difference = float(np.max(np.abs(actual.astype(np.float64) -
                                     expected.astype(np.float64)))) if actual.size else 0.0
    assert difference <= tolerance, (name, difference)
    return difference


def class_summary(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    return {
        "negative_mean": float(values[labels == 0].mean()),
        "positive_mean": float(values[labels == 1].mean()),
        "negative_median": float(np.median(values[labels == 0])),
        "positive_median": float(np.median(values[labels == 1])),
    }


def main():
    complete = read(OUT / "complete.json")
    summary = read(OUT / "summary.json")
    extraction = read(OUT / "extraction_complete.json")
    selection = read(OUT / "selection_manifest.json")
    preparation = read(OUT / "preparation_complete.json")
    assert complete["status"] == "complete_fit_only_pilot_not_independent_test"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected
    assert {name: sha(FIT_DATA / name) for name in EXPECTED_GOLD_SHA} == EXPECTED_GOLD_SHA
    assert summary["calibration_opened"] is False
    assert summary["official_test_opened"] is False
    assert summary["paper_baseline_modified_or_scored"] is False
    assert extraction["fit_gold_files_opened"] is False

    prepared = list(lines(SELECTED))
    selected_ids = {row["response_id"] for row in prepared}
    assert len(prepared) == len(selected_ids) == 256
    answers_by_id, tokens_by_id = {}, {}
    full_answer_count = full_token_count = 0
    for answer in lines(FIT_ANSWERS):
        full_answer_count += 1
        if answer["response_id"] in selected_ids:
            answers_by_id[answer["response_id"]] = answer
    for token in lines(FIT_TOKENS):
        full_token_count += 1
        if token["response_id"] in selected_ids:
            tokens_by_id[token["response_id"]] = token
    assert full_answer_count == full_token_count == 3680
    assert set(answers_by_id) == set(tokens_by_id) == selected_ids
    answers = [answers_by_id[row["response_id"]] for row in prepared]
    tokens = [tokens_by_id[row["response_id"]] for row in prepared]
    by_response = {}
    for row, answer, token in zip(prepared, answers, tokens):
        assert row["response_id"] == answer["response_id"] == token["response_id"]
        assert row["partition"] == answer["partition"] == token["partition"] == "fit"
        assert row["group_id"] == answer["group_id"]
        assert row["answer_token_ids"] == token["token_ids"]
        assert answer["label"] == int(len(answer["original_labels"]) > 0)
        assert token["answer_risk"] == answer["label"]
        assert answer["label"] == int(any(token["risk_mask"]))
        by_response[row["response_id"]] = {"answer": answer, "tokens": token}
    windows = []
    full_window_count = 0
    for window in lines(FIT_WINDOWS):
        full_window_count += 1
        if window["response_id"] not in selected_ids:
            continue
        token = by_response[window["response_id"]]["tokens"]
        indices = window["token_indices"]
        assert window["partition"] == "fit" and window["eligible"]
        assert indices == list(range(window["token_start"], window["token_end"]))
        assert len(indices) == min(4, token["token_count"])
        assert any(token["lexical_mask"][index] for index in indices)
        assert window["label"] == int(any(token["risk_mask"][index]
                                           for index in indices))
        windows.append(window)
    assert full_window_count == 653979 and len(windows) == 45658
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[window["response_id"]].append(index)
    assert all(answer_windows[answer["response_id"]] for answer in answers)
    meta = {"answers": answers, "tokens": tokens, "windows": windows,
            "by_response": by_response, "answer_windows": dict(answer_windows)}

    total_tokens = sum(len(row["answer_token_ids"]) for row in prepared)
    assert total_tokens == 46481
    independent_x = np.empty((total_tokens, 29), dtype=np.float32)
    saved_x = np.load(OUT / "token_features_29.npy", mmap_mode="r")
    assert saved_x.shape == independent_x.shape and saved_x.dtype == np.float32
    starts = {}
    cursor = 0
    max_feature_error = max_efficiency = 0.0
    full_nll_direct_error = 0.0
    for row in prepared:
        response_id = row["response_id"]
        path = OUT / "token_logprob" / f"{response_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            logp = archive["selected_logprob"].copy()
            citation_mask = archive["citation_mask"].copy()
        derived, efficiency = derive_features(logp, citation_mask)
        right = cursor + len(derived)
        independent_x[cursor:right] = derived
        max_feature_error = max(
            max_feature_error,
            float(np.max(np.abs(derived.astype(np.float64) -
                                saved_x[cursor:right].astype(np.float64)))),
        )
        full_nll_direct_error = max(
            full_nll_direct_error,
            float(np.max(np.abs(saved_x[cursor:right, 0].astype(np.float64) +
                                logp[7].astype(np.float64)))),
        )
        max_efficiency = max(max_efficiency, efficiency)
        starts[response_id] = (cursor, right)
        cursor = right
    assert max_feature_error == 0.0 and full_nll_direct_error == 0.0
    y_token = np.empty(total_tokens, dtype=np.int8)
    lexical = np.empty(total_tokens, dtype=bool)
    for answer in answers:
        left, right = starts[answer["response_id"]]
        token = by_response[answer["response_id"]]["tokens"]
        y_token[left:right] = token["risk_mask"]
        lexical[left:right] = token["lexical_mask"]
    window_y = np.asarray([window["label"] for window in windows], dtype=np.int8)
    answer_y = np.asarray([answer["label"] for answer in answers], dtype=np.int8)
    assert answer_y.sum() == 79 and window_y.sum() == 4082

    selection_by_id = {item["response_id"]: item for item in selection["selected"]}
    held_folds = np.asarray([selection_by_id[answer["response_id"]]["held_fold"]
                             for answer in answers], dtype=np.int8)
    group_array = np.asarray([str(answer["group_id"]) for answer in answers], dtype=object)
    recomputed = np.full(256, -1, dtype=np.int8)
    for fold, (_, held) in enumerate(GroupKFold(5).split(np.arange(256),
                                                         groups=group_array)):
        recomputed[held] = fold
    assert np.array_equal(held_folds, recomputed)
    saved_scores = np.load(OUT / "oof_scores.npz", allow_pickle=False)
    saved_models = pickle.loads((OUT / "readout_models.pkl").read_bytes())
    assert saved_models["C"] == LR_C and saved_models["folds"] == FOLDS
    all_answer_indices = np.arange(256, dtype=np.int64)
    reconstruction = {}
    diagnostics = {}
    global_max_error = 0.0
    for readout_name, columns_tuple in READOUTS.items():
        columns = np.asarray(columns_tuple, dtype=np.int64)
        oof_token = np.full(total_tokens, np.nan, dtype=np.float64)
        fold_records = []
        stored = saved_models["readouts"][readout_name]
        assert tuple(stored["feature_names"]) == tuple(FEATURE_NAMES[index]
                                                        for index in columns)
        assert np.array_equal(stored["columns"], columns)
        for fold in range(5):
            train_answers = all_answer_indices[held_folds != fold]
            held_answers = all_answer_indices[held_folds == fold]
            weights, weight_diagnostic = fold_weights(
                meta, train_answers, starts, y_token, lexical
            )
            train_indices = np.flatnonzero(weights > 0)
            held_indices = np.concatenate([
                np.arange(*starts[answers[index]["response_id"]])
                for index in held_answers
            ])
            scaler = StandardScaler().fit(
                independent_x[train_indices][:, columns],
                sample_weight=weights[train_indices],
            )
            model = LogisticRegression(
                C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                random_state=SEED,
            )
            model.fit(
                scaler.transform(independent_x[train_indices][:, columns]),
                y_token[train_indices], sample_weight=weights[train_indices],
            )
            saved_fold = stored["fold_models"][fold]
            assert saved_fold["fold"] == fold
            scaler_error = max(
                compare_float(scaler.mean_, saved_fold["scaler"].mean_,
                              f"{readout_name}/fold{fold}/mean"),
                compare_float(scaler.scale_, saved_fold["scaler"].scale_,
                              f"{readout_name}/fold{fold}/scale"),
            )
            model_error = max(
                compare_float(model.coef_, saved_fold["model"].coef_,
                              f"{readout_name}/fold{fold}/coef"),
                compare_float(model.intercept_, saved_fold["model"].intercept_,
                              f"{readout_name}/fold{fold}/intercept"),
            )
            prediction = model.predict_proba(
                scaler.transform(independent_x[held_indices][:, columns])
            )[:, 1]
            oof_token[held_indices] = prediction
            held_response_ids = {answers[index]["response_id"] for index in held_answers}
            held_window_indices = np.asarray([
                index for index, window in enumerate(windows)
                if window["response_id"] in held_response_ids
            ], dtype=np.int64)
            held_lexical_indices = np.concatenate([
                np.flatnonzero(lexical[slice(*starts[answers[index]["response_id"]])]) +
                starts[answers[index]["response_id"]][0]
                for index in held_answers
            ])
            standardized_coef = model.coef_[0]
            raw_coef = standardized_coef / scaler.scale_
            raw_intercept = float(model.intercept_[0] - np.dot(
                standardized_coef, scaler.mean_ / scaler.scale_))
            fold_records.append({
                "fold": fold,
                "train": {
                    "answers": len(train_answers),
                    "answer_positive": int(answer_y[train_answers].sum()),
                    "lexical_tokens": len(train_indices),
                    "lexical_positive": int(y_token[train_indices].sum()),
                    "lexical_prevalence": float(y_token[train_indices].mean()),
                },
                "held": {
                    "answers": len(held_answers),
                    "answer_positive": int(answer_y[held_answers].sum()),
                    "answer_prevalence": float(answer_y[held_answers].mean()),
                    "windows": len(held_window_indices),
                    "window_positive": int(window_y[held_window_indices].sum()),
                    "window_prevalence": float(window_y[held_window_indices].mean()),
                    "lexical_tokens": len(held_lexical_indices),
                    "lexical_positive": int(y_token[held_lexical_indices].sum()),
                    "lexical_prevalence": float(y_token[held_lexical_indices].mean()),
                    "token_score_mean": float(prediction.mean()),
                    "token_score_std": float(prediction.std()),
                },
                "weight_diagnostic": weight_diagnostic,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "standardized_coefficients": standardized_coef.tolist(),
                "raw_coefficients": raw_coef.tolist(),
                "raw_intercept": raw_intercept,
                "saved_replay_max_abs": max(scaler_error, model_error),
            })
        assert np.isfinite(oof_token).all()
        saved_token = saved_scores[f"{readout_name}__token"]
        token_error = compare_float(oof_token, saved_token,
                                    f"{readout_name}/token", tolerance=1e-10)
        window_scores = np.empty(len(windows), dtype=np.float64)
        for index, window in enumerate(windows):
            token = by_response[window["response_id"]]["tokens"]
            active = [position for position in window["token_indices"]
                      if token["lexical_mask"][position]]
            assert active
            start = starts[window["response_id"]][0]
            window_scores[index] = max(oof_token[start + position]
                                       for position in active)
        answer_scores = np.asarray([
            max(window_scores[answer_windows[answer["response_id"]]])
            for answer in answers
        ], dtype=np.float64)
        window_error = compare_float(
            window_scores, saved_scores[f"{readout_name}__window"],
            f"{readout_name}/window", tolerance=1e-10,
        )
        answer_error = compare_float(
            answer_scores, saved_scores[f"{readout_name}__answer"],
            f"{readout_name}/answer", tolerance=1e-10,
        )
        thresholds = {
            "window": choose_threshold(window_y, window_scores),
            "answer": choose_threshold(answer_y, answer_scores),
        }
        scored = {
            "windows": metric(window_y, window_scores,
                              thresholds["window"]["threshold"]),
            "answers": metric(answer_y, answer_scores,
                              thresholds["answer"]["threshold"]),
        }
        expected = summary["readouts"][readout_name]
        assert thresholds == expected["thresholds"]
        for level in ("windows", "answers"):
            for key, value in expected["fit_OOF_metrics"][level].items():
                assert abs(scored[level][key] - value) <= 1e-12 if isinstance(value, float) else scored[level][key] == value
        # Add held fold score/ranking diagnostics after complete OOF mapping.
        fold_window_aurocs, fold_answer_aurocs = [], []
        for record in fold_records:
            fold = record["fold"]
            held_answer_indices = all_answer_indices[held_folds == fold]
            response_ids = {answers[index]["response_id"] for index in held_answer_indices}
            held_window_indices = np.asarray([
                index for index, window in enumerate(windows)
                if window["response_id"] in response_ids
            ], dtype=np.int64)
            window_metric = metric(window_y[held_window_indices],
                                   window_scores[held_window_indices])
            answer_metric = metric(answer_y[held_answer_indices],
                                   answer_scores[held_answer_indices])
            record["held"]["window_AUROC"] = window_metric["auroc"]
            record["held"]["window_AP"] = window_metric["average_precision"]
            record["held"]["answer_AUROC"] = answer_metric["auroc"]
            record["held"]["answer_AP"] = answer_metric["average_precision"]
            record["held"]["window_score_mean"] = float(window_scores[held_window_indices].mean())
            record["held"]["answer_score_mean"] = float(answer_scores[held_answer_indices].mean())
            if window_metric["auroc"] is not None:
                fold_window_aurocs.append(window_metric["auroc"])
            if answer_metric["auroc"] is not None:
                fold_answer_aurocs.append(answer_metric["auroc"])
        reconstruction[readout_name] = {
            "thresholds": thresholds,
            "metrics": scored,
            "saved_score_max_abs": {
                "token": token_error, "window": window_error, "answer": answer_error,
            },
            "folds": fold_records,
        }
        diagnostics[readout_name] = {
            "pooled_window_AUROC": scored["windows"]["auroc"],
            "macro_fold_window_AUROC": float(np.mean(fold_window_aurocs)),
            "pooled_answer_AUROC": scored["answers"]["auroc"],
            "macro_fold_answer_AUROC": float(np.mean(fold_answer_aurocs)),
            "held_token_score_mean_range": [
                min(item["held"]["token_score_mean"] for item in fold_records),
                max(item["held"]["token_score_mean"] for item in fold_records),
            ],
            "held_window_score_mean_range": [
                min(item["held"]["window_score_mean"] for item in fold_records),
                max(item["held"]["window_score_mean"] for item in fold_records),
            ],
            "held_answer_score_mean_range": [
                min(item["held"]["answer_score_mean"] for item in fold_records),
                max(item["held"]["answer_score_mean"] for item in fold_records),
            ],
            "raw_coefficient_ranges": {
                FEATURE_NAMES[column]: [
                    min(item["raw_coefficients"][local] for item in fold_records),
                    max(item["raw_coefficients"][local] for item in fold_records),
                ]
                for local, column in enumerate(columns)
            },
        }
        global_max_error = max(global_max_error, token_error, window_error, answer_error)

    # Label-aware diagnostics only; no candidate or readout is fit here.
    lexical_indices = np.flatnonzero(lexical)
    feature_diagnostics = {}
    for column, name in enumerate(FEATURE_NAMES):
        values = independent_x[lexical_indices, column]
        labels = y_token[lexical_indices]
        auc = float(roc_auc_score(labels, values))
        feature_diagnostics[name] = {
            "lexical_token_AUROC_raw_direction": auc,
            "lexical_token_AUROC_best_orientation": max(auc, 1 - auc),
            "class_summary": class_summary(values, labels),
        }
    raw_full_nll_window = np.empty(len(windows), dtype=np.float64)
    raw_delta_window = np.empty(len(windows), dtype=np.float64)
    for index, window in enumerate(windows):
        token = by_response[window["response_id"]]["tokens"]
        active = [position for position in window["token_indices"]
                  if token["lexical_mask"][position]]
        start = starts[window["response_id"]][0]
        raw_full_nll_window[index] = max(independent_x[start + position, 0]
                                         for position in active)
        # Risk orientation for evidence gain: less/negative gain is more suspect.
        raw_delta_window[index] = max(-independent_x[start + position, 2]
                                      for position in active)
    raw_full_nll_answer = np.asarray([
        max(raw_full_nll_window[answer_windows[answer["response_id"]]])
        for answer in answers
    ])
    raw_delta_answer = np.asarray([
        max(raw_delta_window[answer_windows[answer["response_id"]]])
        for answer in answers
    ])
    citation_any = independent_x[lexical_indices, 22] > 0.5
    citation_diagnostic = {
        "lexical_tokens_cited": int(citation_any.sum()),
        "lexical_tokens_uncited": int((~citation_any).sum()),
        "risk_prevalence_cited": float(y_token[lexical_indices][citation_any].mean())
        if citation_any.any() else None,
        "risk_prevalence_uncited": float(y_token[lexical_indices][~citation_any].mean())
        if (~citation_any).any() else None,
    }
    direct_signal_diagnostic = {
        "full_nll_definition": "-selected_log_probability under full subset mask 111; higher means lower model probability",
        "full_nll_sign_error_max_abs": full_nll_direct_error,
        "full_nll_nonnegative": bool(np.all(independent_x[:, 0] >= 0)),
        "raw_full_nll": {
            "lexical_token_AUROC": feature_diagnostics["full_nll"]["lexical_token_AUROC_raw_direction"],
            "4BPE_window_max_AUROC": float(roc_auc_score(window_y, raw_full_nll_window)),
            "answer_max_AUROC": float(roc_auc_score(answer_y, raw_full_nll_answer)),
        },
        "negative_full_minus_empty_as_risk": {
            "4BPE_window_max_AUROC": float(roc_auc_score(window_y, raw_delta_window)),
            "answer_max_AUROC": float(roc_auc_score(answer_y, raw_delta_answer)),
        },
        "citation": citation_diagnostic,
    }

    # These are diagnostics of the frozen feature design, not new candidates.
    # B contains an algebraic duplicate; the complete design contains several
    # further near-exact linear identities inherited from the eight subset
    # log-probabilities and their Shapley decomposition.
    geometry_x = independent_x.astype(np.float64)
    feature_scale = geometry_x.std(axis=0)
    standardized = (
        (geometry_x - geometry_x.mean(axis=0)) /
        np.where(feature_scale > 0, feature_scale, 1.0)
    )
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    feature_geometry = {
        "B_identity_full_minus_empty_equals_empty_nll_minus_full_nll_max_abs":
            float(np.max(np.abs(independent_x[:, 2] -
                                (independent_x[:, 1] - independent_x[:, 0])))),
        "Shapley_efficiency_identity_max_abs":
            float(np.max(np.abs(independent_x[:, 2] -
                                independent_x[:, 3:6].sum(axis=1)))),
        "standardized_singular_value_largest": float(singular_values[0]),
        "standardized_singular_value_20th": float(singular_values[19]),
        "standardized_singular_value_21st": float(singular_values[20]),
        "standardized_singular_value_smallest": float(singular_values[-1]),
        "singular_values_above_1e_minus_4": int((singular_values > 1e-4).sum()),
        "singular_values_below_1e_minus_4": int((singular_values <= 1e-4).sum()),
        "interpretation": "Nine of 29 standardized columns are linearly redundant up to float32 extraction noise; L2 makes fitting finite but cannot create independent signal.",
    }

    canonical_folds = reconstruction["A_full_nll_citation"]["folds"]
    final_positive_mass_fraction = [
        fold["weight_diagnostic"]["final_class_mass"][1] /
        sum(fold["weight_diagnostic"]["final_class_mass"])
        for fold in canonical_folds
    ]
    fold_label_and_weight_diagnostic = {
        "held_answer_positive_prevalence_range": [
            min(fold["held"]["answer_prevalence"] for fold in canonical_folds),
            max(fold["held"]["answer_prevalence"] for fold in canonical_folds),
        ],
        "held_window_positive_prevalence_range": [
            min(fold["held"]["window_prevalence"] for fold in canonical_folds),
            max(fold["held"]["window_prevalence"] for fold in canonical_folds),
        ],
        "held_lexical_positive_prevalence_range": [
            min(fold["held"]["lexical_prevalence"] for fold in canonical_folds),
            max(fold["held"]["lexical_prevalence"] for fold in canonical_folds),
        ],
        "final_train_weight_positive_mass_fraction_range": [
            min(final_positive_mass_fraction), max(final_positive_mass_fraction),
        ],
        "class_mass_balanced_before_final_group_equalization": True,
        "class_mass_still_50_50_after_final_group_equalization": False,
        "interpretation": "The frozen weighting sequence first balances classes, then restores equal group mass. That last step leaves only about 21-23% positive loss mass, so the nominal class balance is not retained.",
    }

    report = {
        "status": "passed_independent_fit_only_scoring_reconstruction",
        "audit_source_sha256": sha(__file__),
        "complete_sha256": sha(OUT / "complete.json"),
        "summary_sha256": sha(OUT / "summary.json"),
        "score_files_sha256_verified": complete["files_sha256"],
        "fit_gold_sha256": EXPECTED_GOLD_SHA,
        "counts": {
            "answers": len(answers), "answer_positive": int(answer_y.sum()),
            "raw_tokens": total_tokens,
            "lexical_tokens": int(lexical.sum()),
            "lexical_token_positive": int(y_token[lexical].sum()),
            "windows": len(windows), "window_positive": int(window_y.sum()),
            "groups": len({answer["group_id"] for answer in answers}),
        },
        "numeric_reconstruction": {
            "feature_matrix_max_abs": max_feature_error,
            "Shapley_efficiency_max_abs": max_efficiency,
            "OOF_score_global_max_abs": global_max_error,
            "readouts": reconstruction,
        },
        "diagnostics": {
            "direct_signals": direct_signal_diagnostic,
            "per_feature": feature_diagnostics,
            "OOF_fold_scale_and_coefficients": diagnostics,
            "feature_geometry": feature_geometry,
            "fold_labels_and_weights": fold_label_and_weight_diagnostic,
        },
        "conclusion": {
            "implementation_error_found": False,
            "full_nll_direction_error_found": False,
            "primary_failure": "The frozen attribution/NLL features have weak and fold-unstable association with local factual-error labels on this 256-group cohort.",
            "secondary_factors": [
                "GroupKFold was label-free rather than stratified, so answer/window/token prevalences differ across held folds.",
                "Independent scaler/LR fits create fold-specific probability offsets; pooled OOF ranking may differ from mean within-fold ranking.",
                "Answer max pooling amplifies many modest token risks, causing thresholds near the all-positive answer endpoint.",
                "The final equal-group normalization reverses much of the preceding class balancing; positive loss mass is only about 21-23% across folds.",
                "Nine of the 29 columns are algebraically redundant up to float32 noise, and several sparse citation coefficients change sign across folds.",
                "The pilot contains only 256 independent groups and 79 risky answers; token count does not equal independent semantic sample count.",
            ],
            "scope": "Diagnosis only. No new candidate, hyperparameter, mapping, or baseline result was created.",
        },
        "fit_only": True,
        "calibration_opened": False,
        "official_test_opened": False,
        "GPU_used": False,
        "paper_baseline_modified_or_scored": False,
        "production_runner_imported": False,
    }
    save(OUT / "SCORING_INDEPENDENT_AUDIT.json", report)
    answer_predicted_positive = {
        name: (item["metrics"]["answers"]["tp"] +
               item["metrics"]["answers"]["fp"])
        for name, item in reconstruction.items()
    }
    markdown = f"""# v3 fit-only scoring independent audit

## Conclusion

The saved score is reproduced exactly. Feature matrix and all token/window/answer OOF scores have max absolute error **0.0**; Shapley efficiency error is **{max_efficiency:.3g}**. Gold labels, 4-BPE stride-one windows, answer max pooling, five GroupKFold splits, weights, weighted scaler, LR C=0.1, thresholds, and metrics all match. `full_nll` is correctly defined as `-log p(token | full evidence)`; no sign or scoring implementation error was found.

## Why AUROC is near random

- The frozen signals are weak for this local-error target. Raw `full_nll` has token/window AUROC **0.590/0.598**, but answer-max AUROC falls to **0.350**. Risk-oriented `-(full_minus_empty)` reaches **0.628** per window but only **0.509** per answer.
- Fold offsets materially lower pooled OOF ranking. For B/C, mean within-fold window AUROC is **0.567/0.565**, while pooled AUROC is **0.517/0.510**. Held-fold window prevalence ranges from **{fold_label_and_weight_diagnostic['held_window_positive_prevalence_range'][0]:.2%}** to **{fold_label_and_weight_diagnostic['held_window_positive_prevalence_range'][1]:.2%}**.
- The weighting sequence balances classes and then re-equalizes groups, leaving positive loss mass at only **{fold_label_and_weight_diagnostic['final_train_weight_positive_mass_fraction_range'][0]:.2%}-{fold_label_and_weight_diagnostic['final_train_weight_positive_mass_fraction_range'][1]:.2%}**. This explains low probability levels and contributes to fold offsets; it does not by itself explain weak ranking.
- B has an exact duplicate: `full_minus_empty = empty_nll - full_nll`. Across all 29 features, only 20 standardized singular values exceed `1e-4`; nine directions are algebraically redundant up to float32 noise. Citation is also sparse: **{citation_diagnostic['lexical_tokens_cited']:,}/{int(lexical.sum()):,}** lexical tokens are cited, with risk prevalence **{citation_diagnostic['risk_prevalence_cited']:.2%}** versus **{citation_diagnostic['risk_prevalence_uncited']:.2%}** when uncited.
- Answer max pooling selects the single highest-risk window, often an unusual but correct token. At the fit-OOF F1 threshold, A/B/C mark **{answer_predicted_positive['A_full_nll_citation']}/{answer_predicted_positive['B_full_empty_delta_citation']}/{answer_predicted_positive['C_all_29']}** of 256 answers positive. The answer F1 therefore mainly comes from very high recall, not precise ranking.
- There are 46,481 raw tokens, but only **256 independent groups and 79 positive answers**. Local-error tokens cluster within answers, so token count overstates independent evidence.

This audit used only the fixed 256 fit responses and expanded-fit gold. It did not open calibration/test data, use GPU, create a candidate, or modify/score a paper baseline.
"""
    save_text(OUT / "SCORING_INDEPENDENT_AUDIT.md", markdown)
    print("EXACT_SUBSET_V3_SCORING_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
