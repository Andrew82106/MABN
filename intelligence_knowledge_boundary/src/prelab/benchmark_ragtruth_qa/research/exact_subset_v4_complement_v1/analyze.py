"""Fit-only, CPU-only complementarity diagnosis for exact-subset and expanded-v4.

This script intentionally reads:
  * the frozen 256-answer exact-v3 pilot cache and fit gold files;
  * only the five expanded-v4 held-fold prediction files;
  * only the fit prefixes of expanded-v4 answers/examples.

It never opens the combined v4 score archive, calibration rows, or test artifacts.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXACT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
V4_DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_MODEL = ROOT / "results/microclaim_crossencoder_expanded_v4"
FIT_DATA = ROOT / "fit_expansion/data"

N_SELECTED = 256
N_SELECTED_TOKENS = 46_481
N_SELECTED_WINDOWS = 45_658
N_V4_FIT_ANSWERS = 3_680
N_V4_FIT_CLAIMS = 34_919
V4_FROZEN_WINDOW_THRESHOLD = 0.8225097060203552
FOLDS = 5

EXPECTED_SHA256 = {
    EXACT / "selection_manifest.json": "1c438ce9e70c64c0538000589a3e276c8354da1983df8fe5d3e01712ed90eb04",
    EXACT / "selected_prepared_inputs.jsonl": "f8c584e4c24565bf4e607d5673c9b54523e33a91b1365780d4f01e84ac06089b",
    EXACT / "token_features_29.npy": "238c0605b48727fee27d6894e99147e00363ac18436f3581b6800539eb08dd49",
    FIT_DATA / "answers_fit.jsonl": "8adef0c27d5021acdf559c1566db2d6d949ecccb88ad2532ae626381add0aa33",
    FIT_DATA / "tokens_fit.jsonl": "18501c2a53a22620f24827d3181d831f2aee77da2b335bd50270fbe939048ec2",
    FIT_DATA / "windows_k4_fit.jsonl": "cfaf5af088eaac422ee2685c9150a774171d930666426e936f86d0cee905eec3",
    V4_MODEL / "fold_0/predictions.npz": "93d3e0a2a33ad09f1ecc94c968bab4d09ab354a3e786c5e1adc9362bae70cc75",
    V4_MODEL / "fold_1/predictions.npz": "5498c06e206198d37c97a6d3a2b6bd23e87659824c66400f48741b0582e723b0",
    V4_MODEL / "fold_2/predictions.npz": "84d755be8e88771228363726bb7f1a1b4ac1c11c83f8d90a1f55003b9cdd5931",
    V4_MODEL / "fold_3/predictions.npz": "c2fde389b2cbb73e06adcc157f257a8b6c31bce372a5774adc3bf1d1118a693b",
    V4_MODEL / "fold_4/predictions.npz": "5c0429895e62028e46ccf371bd81af25780d13e690139d7ef70d176af33fed4a",
}
EXPECTED_V4_FIT_PREFIX_SHA256 = {
    V4_DATA / "answers.jsonl": "34eebfbefd451a5eabab4aba547b5bd8e918282e17fdafe56d83d525884a1865",
    V4_DATA / "examples.jsonl": "e688f297dc9a28925a99e3cbae36f98ee5e82fa37a619760808610b2513fab4a",
}
TYPE_NAMES = {
    "Evident Conflict": "conflict",
    "Subtle Conflict": "conflict",
    "Evident Baseless Info": "baseless",
    "Subtle Baseless Info": "baseless",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_lines(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fit_prefix(path: Path, count: int):
    """Read exactly count leading rows without reading the following cal row."""
    rows = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for _ in range(count):
            raw = stream.readline()
            assert raw, f"short fit prefix: {path}"
            digest.update(raw)
            rows.append(json.loads(raw.decode("utf-8")))
    return rows, digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def average_precision(labels, scores) -> float:
    """Non-interpolated AP with tied scores evaluated as one threshold."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    assert positives > 0 and len(y) == len(s) and np.isfinite(s).all()
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    ends = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1, len(s_sorted)]
    cumulative_tp = np.cumsum(y_sorted, dtype=np.int64)
    ap, previous_tp = 0.0, 0
    for end in ends:
        tp = int(cumulative_tp[end - 1])
        ap += ((tp - previous_tp) / positives) * (tp / int(end))
        previous_tp = tp
    return float(ap)


def binary_metrics(labels, scores, predicted) -> dict:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p)); tn = int(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y), "positive": int(y.sum()), "predicted_positive": int(p.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "average_precision": average_precision(y, scores),
    }


def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    output = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        output[order[left:right]] = 0.5 * (left + right - 1)
        left = right
    return output


def correlation(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1]),
    }


def empirical_cdf(training, query) -> np.ndarray:
    ordered = np.sort(np.asarray(training, dtype=np.float64), kind="mergesort")
    query = np.asarray(query, dtype=np.float64)
    return np.searchsorted(ordered, query, side="right") / len(ordered)


def load_v4_oof_claim_scores() -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(N_V4_FIT_CLAIMS, np.nan, dtype=np.float32)
    owning_fold = np.full(N_V4_FIT_CLAIMS, -1, dtype=np.int8)
    for fold in range(FOLDS):
        path = V4_MODEL / f"fold_{fold}/predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            indices = archive["held_indices"].astype(np.int64)
            held_scores = archive["held_scores"].astype(np.float32)
        assert len(indices) == len(held_scores) and len(np.unique(indices)) == len(indices)
        assert np.all((0 <= indices) & (indices < N_V4_FIT_CLAIMS))
        assert np.isnan(scores[indices]).all() and np.all(owning_fold[indices] == -1)
        scores[indices] = held_scores
        owning_fold[indices] = fold
    assert np.isfinite(scores).all() and np.all(owning_fold >= 0)
    return scores, owning_fold


def token_type_masks(token: dict) -> tuple[np.ndarray, np.ndarray]:
    n = int(token["token_count"])
    conflict = np.zeros(n, dtype=bool)
    baseless = np.zeros(n, dtype=bool)
    labels = token["original_labels"]
    mappings = token["span_token_mapping"]
    assert len(labels) == len(mappings)
    for span, mapping in zip(labels, mappings):
        category = TYPE_NAMES[span["label_type"]]
        indices = np.asarray(mapping["risk_token_indices"], dtype=np.int64)
        assert np.all((0 <= indices) & (indices < n))
        (conflict if category == "conflict" else baseless)[indices] = True
    risk = np.asarray(token["risk_mask"], dtype=bool)
    assert np.array_equal(conflict | baseless, risk)
    return conflict, baseless


def main() -> None:
    for name in ("SOURCE_LOCK.json", "SCORES.npz", "RESULTS.json", "REPORT.md", "complete.json"):
        assert not (HERE / name).exists(), f"refuse overwrite: {name}"

    actual_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED_SHA256}
    assert actual_hashes == {str(path.relative_to(ROOT)): expected
                             for path, expected in EXPECTED_SHA256.items()}

    selection = json.loads((EXACT / "selection_manifest.json").read_text(encoding="utf-8"))["selected"]
    prepared = list(json_lines(EXACT / "selected_prepared_inputs.jsonl"))
    assert len(selection) == len(prepared) == N_SELECTED
    selected_ids = [row["response_id"] for row in selection]
    assert selected_ids == [row["response_id"] for row in prepared]
    assert len(set(selected_ids)) == N_SELECTED
    pilot_fold_by_id = {row["response_id"]: int(row["held_fold"]) for row in selection}
    assert sorted(set(pilot_fold_by_id.values())) == list(range(FOLDS))

    fit_answers = {row["response_id"]: row for row in json_lines(FIT_DATA / "answers_fit.jsonl")
                   if row["response_id"] in pilot_fold_by_id}
    fit_tokens = {row["response_id"]: row for row in json_lines(FIT_DATA / "tokens_fit.jsonl")
                  if row["response_id"] in pilot_fold_by_id}
    fit_windows = [row for row in json_lines(FIT_DATA / "windows_k4_fit.jsonl")
                   if row["response_id"] in pilot_fold_by_id]
    assert set(fit_answers) == set(fit_tokens) == set(selected_ids)
    assert len(fit_windows) == N_SELECTED_WINDOWS

    features = np.load(EXACT / "token_features_29.npy", mmap_mode="r", allow_pickle=False)
    assert features.shape == (N_SELECTED_TOKENS, 29) and features.dtype == np.float32
    token_slice = {}
    cursor = 0
    for row in prepared:
        right = cursor + len(row["answer_token_ids"])
        token_slice[row["response_id"]] = (cursor, right)
        token = fit_tokens[row["response_id"]]
        answer = fit_answers[row["response_id"]]
        assert row["answer_token_ids"] == token["token_ids"]
        assert row["response_token_offsets"] == token["response_token_offsets"]
        assert row["answer_sha256"] == token["answer_sha256"] == answer["answer_sha256"]
        assert row["group_id"] == token["group_id"] == answer["group_id"]
        cursor = right
    assert cursor == N_SELECTED_TOKENS

    types = {rid: token_type_masks(token) for rid, token in fit_tokens.items()}
    exact_by_key = {}
    label_by_key = {}
    conflict_by_key = {}
    baseless_by_key = {}
    ordered_keys = []
    for window in fit_windows:
        rid = window["response_id"]
        indices = np.asarray(window["token_indices"], dtype=np.int64)
        assert len(indices) == 4 and np.array_equal(indices, np.arange(window["token_start"], window["token_end"]))
        left, right = token_slice[rid]
        assert np.all((0 <= indices) & (indices < right - left))
        key = (rid, int(window["token_start"]))
        assert key not in exact_by_key
        # Column 2 is the frozen full_minus_empty token feature.
        exact_by_key[key] = -float(np.mean(features[left + indices, 2], dtype=np.float64))
        label_by_key[key] = int(window["label"])
        conflict, baseless = types[rid]
        conflict_by_key[key] = bool(conflict[indices].any())
        baseless_by_key[key] = bool(baseless[indices].any())
        assert label_by_key[key] == int(conflict_by_key[key] or baseless_by_key[key])
        ordered_keys.append(key)
    assert len(set(ordered_keys)) == N_SELECTED_WINDOWS

    v4_answers, answers_prefix_hash = fit_prefix(V4_DATA / "answers.jsonl", N_V4_FIT_ANSWERS)
    v4_examples, examples_prefix_hash = fit_prefix(V4_DATA / "examples.jsonl", N_V4_FIT_CLAIMS)
    assert answers_prefix_hash == EXPECTED_V4_FIT_PREFIX_SHA256[V4_DATA / "answers.jsonl"]
    assert examples_prefix_hash == EXPECTED_V4_FIT_PREFIX_SHA256[V4_DATA / "examples.jsonl"]
    assert all(row["partition"] == "fit" for row in v4_answers)
    assert all(row["partition"] == "fit" for row in v4_examples)
    assert [row["response_index"] for row in v4_answers] == list(range(N_V4_FIT_ANSWERS))
    assert [row["example_index"] for row in v4_examples] == list(range(N_V4_FIT_CLAIMS))

    v4_claim_scores, v4_claim_fold = load_v4_oof_claim_scores()
    assert all(v4_claim_fold[row["example_index"]] == row["held_fold"] for row in v4_examples)
    selected_set = set(selected_ids)
    v4_answer_by_id = {row["response_id"]: row for row in v4_answers
                       if row["response_id"] in selected_set}
    v4_examples_by_id = defaultdict(list)
    for row in v4_examples:
        if row["response_id"] in selected_set:
            v4_examples_by_id[row["response_id"]].append(row)
    assert set(v4_answer_by_id) == set(v4_examples_by_id) == selected_set

    starts_by_id = defaultdict(set)
    for rid, start in ordered_keys:
        starts_by_id[rid].add(start)
    v4_by_key = {}
    selected_v4_fold_by_id = {}
    for rid in selected_ids:
        answer = v4_answer_by_id[rid]
        token = fit_tokens[rid]
        assert answer["answer_sha256"] == token["answer_sha256"]
        assert answer["token_ids"] == token["token_ids"]
        assert answer["response_token_offsets"] == token["response_token_offsets"]
        assert set(range(answer["window_array_start"], answer["window_array_end"]))
        eligible = {start for start in range(int(answer["token_count"]) - 3)
                    if any(answer["lexical_mask"][start:start + 4])}
        assert eligible == starts_by_id[rid]
        local = {start: -np.inf for start in eligible}
        fold_values = set()
        for claim in v4_examples_by_id[rid]:
            index = int(claim["example_index"])
            fold_values.add(int(claim["held_fold"]))
            for start in claim["mapped_eligible_window_starts"]:
                assert start in local
                local[start] = max(local[start], float(v4_claim_scores[index]))
        assert len(fold_values) == 1 and np.isfinite(list(local.values())).all()
        selected_v4_fold_by_id[rid] = fold_values.pop()
        v4_by_key.update({(rid, start): value for start, value in local.items()})
    assert set(v4_by_key) == set(ordered_keys)

    response_ids = np.asarray([rid for rid, _ in ordered_keys], dtype="U32")
    starts = np.asarray([start for _, start in ordered_keys], dtype=np.int32)
    labels = np.asarray([label_by_key[key] for key in ordered_keys], dtype=np.int8)
    conflict = np.asarray([conflict_by_key[key] for key in ordered_keys], dtype=bool)
    baseless = np.asarray([baseless_by_key[key] for key in ordered_keys], dtype=bool)
    exact = np.asarray([exact_by_key[key] for key in ordered_keys], dtype=np.float64)
    v4 = np.asarray([v4_by_key[key] for key in ordered_keys], dtype=np.float64)
    folds = np.asarray([pilot_fold_by_id[rid] for rid in response_ids], dtype=np.int8)
    assert labels.sum() == 4_082 and np.array_equal(labels.astype(bool), conflict | baseless)
    assert np.isfinite(exact).all() and np.isfinite(v4).all()

    v4_rank = np.full(len(labels), np.nan, dtype=np.float64)
    exact_rank = np.full(len(labels), np.nan, dtype=np.float64)
    fusion = np.full(len(labels), np.nan, dtype=np.float64)
    exact_pred = np.zeros(len(labels), dtype=bool)
    fusion_pred = np.zeros(len(labels), dtype=bool)
    fold_scale = []
    for fold in range(FOLDS):
        train = folds != fold
        held = folds == fold
        q = float(empirical_cdf(v4[train], [V4_FROZEN_WINDOW_THRESHOLD])[0])
        v4_rank[held] = empirical_cdf(v4[train], v4[held])
        exact_rank[held] = empirical_cdf(exact[train], exact[held])
        fusion[held] = 0.75 * v4_rank[held] + 0.25 * exact_rank[held]
        exact_pred[held] = exact_rank[held] >= q
        fusion_pred[held] = fusion[held] >= q
        fold_scale.append({
            "fold": fold, "train_windows": int(train.sum()), "held_windows": int(held.sum()),
            "v4_threshold_training_cdf_q": q,
            "v4_held_predicted_positive": int(np.sum(v4[held] >= V4_FROZEN_WINDOW_THRESHOLD)),
            "exact_held_predicted_positive": int(exact_pred[held].sum()),
            "fusion_held_predicted_positive": int(fusion_pred[held].sum()),
        })
    assert np.isfinite(v4_rank).all() and np.isfinite(exact_rank).all() and np.isfinite(fusion).all()
    v4_pred = v4 >= V4_FROZEN_WINDOW_THRESHOLD

    scores = {"v4": v4, "exact": exact, "fusion": fusion}
    predictions = {"v4": v4_pred, "exact": exact_pred, "fusion": fusion_pred}
    overall = {name: binary_metrics(labels, scores[name], predictions[name]) for name in scores}
    type_recall = {}
    for name in scores:
        type_recall[name] = {}
        for kind, mask in (("conflict", conflict), ("baseless", baseless)):
            denominator = int(mask.sum())
            detected = int(np.sum(predictions[name] & mask))
            type_recall[name][kind] = {
                "positive_windows": denominator, "detected": detected,
                "recall": float(detected / denominator),
            }

    v4_rejected = ~v4_pred
    miss_region = {
        "definition": "All selected windows with v4 raw score below its already-frozen fit-OOF threshold; positives inside are v4 false negatives.",
        "windows": int(v4_rejected.sum()),
        "positive_windows": int(labels[v4_rejected].sum()),
        "exact_average_precision": average_precision(labels[v4_rejected], exact[v4_rejected]),
        "positive_prevalence": float(labels[v4_rejected].mean()),
    }
    correlations = {
        "raw_v4_vs_exact": correlation(v4, exact),
        "crossfit_cdf_rank_v4_vs_exact": correlation(v4_rank, exact_rank),
        "v4_rejected_raw_v4_vs_exact": correlation(v4[v4_rejected], exact[v4_rejected]),
    }

    v4_model_fold = np.asarray([selected_v4_fold_by_id[rid] for rid in selected_ids], dtype=np.int8)
    pilot_answer_fold = np.asarray([pilot_fold_by_id[rid] for rid in selected_ids], dtype=np.int8)
    results = {
        "status": "complete_fit_only_exploratory_diagnosis",
        "identity": "Exploratory complementarity diagnosis only; not a new candidate and not selection evidence.",
        "cohort": {
            "answers": N_SELECTED, "distinct_groups": N_SELECTED,
            "raw_tokens": N_SELECTED_TOKENS, "eligible_windows": N_SELECTED_WINDOWS,
            "positive_windows": int(labels.sum()),
            "conflict_positive_windows": int(conflict.sum()),
            "baseless_positive_windows": int(baseless.sum()),
            "mixed_type_windows": int(np.sum(conflict & baseless)),
        },
        "fixed_scores": {
            "v4": "reconstructed max overlapping expanded-v4 held-fold OOF microclaim risk",
            "exact": "-mean of frozen full_minus_empty over the four original BPE tokens",
            "fusion": "0.75*v4_training_CDF_rank + 0.25*exact_training_CDF_rank",
        },
        "fixed_operating_rule": {
            "v4_threshold": V4_FROZEN_WINDOW_THRESHOLD,
            "source": "pre-existing expanded-v4 fit-OOF window threshold; copied before this diagnosis",
            "cdf": "right-continuous empirical count(training_score <= x)/n within each frozen exact-v3 pilot fold",
            "alternative_threshold": "q_f=CDF_train_v4(v4 frozen threshold); exact_rank>=q_f and fusion>=q_f",
            "searched_weights_features_or_thresholds": False,
        },
        "overall_window_metrics": overall,
        "type_recall_at_fixed_operating_rule": type_recall,
        "exact_inside_v4_rejected_region": miss_region,
        "correlations": correlations,
        "fold_scaling": fold_scale,
        "mapping_checks": {
            "selected_window_keys_unique": True,
            "exact_and_v4_key_sets_equal": True,
            "answer_hash_token_ids_offsets_equal": True,
            "window_labels_equal_type_union": True,
            "v4_claim_predictions_cover_every_fit_claim_once": True,
            "v4_claim_prediction_fold_matches_v4_example_fold": True,
            "pilot_fold_is_cdf_split_not_v4_model_fold": True,
            "selected_answers_with_same_pilot_and_v4_fold": int(np.sum(pilot_answer_fold == v4_model_fold)),
        },
        "limitations": [
            "The same frozen 256-answer fit cohort is used for diagnosis; this is not calibration or test evidence.",
            "The v4 threshold was previously chosen on expanded-v4 fit OOF labels; this diagnosis performs no new threshold search.",
            "Exact/fusion F1 depends on the predeclared label-free CDF transfer of that existing operating point.",
            "Type masks may overlap; mixed windows contribute to both conflict and baseless recall denominators.",
        ],
        "calibration_rows_or_scores_read": False,
        "official_test_read": False,
        "GPU_used": False,
    }

    np.savez_compressed(
        HERE / "SCORES.npz", response_id=response_ids, token_start=starts,
        pilot_fold=folds, label=labels, conflict=conflict, baseless=baseless,
        v4=v4, exact=exact, v4_rank=v4_rank, exact_rank=exact_rank, fusion=fusion,
        v4_pred=v4_pred, exact_pred=exact_pred, fusion_pred=fusion_pred,
    )
    write_json(HERE / "SOURCE_LOCK.json", {
        "full_fit_only_file_sha256": actual_hashes,
        "v4_fit_prefix_sha256": {
            str(path.relative_to(ROOT)): expected for path, expected in EXPECTED_V4_FIT_PREFIX_SHA256.items()
        },
        "v4_fit_prefix_rows": {"answers.jsonl": N_V4_FIT_ANSWERS, "examples.jsonl": N_V4_FIT_CLAIMS},
        "analyzer_sha256": sha256(Path(__file__)),
        "combined_v4_scores_npz_opened": False,
        "calibration_rows_or_scores_read": False,
        "official_test_read": False,
    })
    write_json(HERE / "RESULTS.json", results)

    report = [
        "# Exact-subset × expanded-v4 complement：fit-only exploratory diagnosis", "",
        "这是固定 256 答 fit cohort 上的互补性诊断，不是新候选、正式选择结果或独立验证。只计算预先指定的 v4、语义方向固定的 exact，以及 0.75/0.25 CDF-rank 融合；没有搜索权重、特征或阈值。", "",
        "## 总体 4-BPE 窗口", "",
        "| 固定分数 | F1 | AP | Precision | Recall | Predicted positive |", "|---|---:|---:|---:|---:|---:|",
    ]
    display = {"v4": "expanded-v4 OOF", "exact": "exact: -mean(full-empty)", "fusion": "0.75 v4-rank + 0.25 exact-rank"}
    for name in ("v4", "exact", "fusion"):
        item = overall[name]
        report.append(f"| {display[name]} | {item['f1']:.6f} | {item['average_precision']:.6f} | {item['precision']:.6f} | {item['recall']:.6f} | {item['predicted_positive']:,} |")
    report += ["", "F1 operating point固定如下：v4 使用既有 fit-OOF 阈值 `0.8225097060203552`。对 exact 与融合，在每个 frozen exact-v3 pilot held fold 内，仅用其余四折分数算右连续经验 CDF；把既有 v4 阈值在训练分布中的分位 `q_f` 原样迁移。标签不进入尺度化或阈值迁移。", "",
               "## 类型召回", "", "| 固定分数 | 冲突 recall | 无依据 recall |", "|---|---:|---:|",]
    for name in ("v4", "exact", "fusion"):
        report.append(f"| {display[name]} | {type_recall[name]['conflict']['recall']:.6f} | {type_recall[name]['baseless']['recall']:.6f} |")
    report += [
        "", "冲突与无依据按原始四类 span 在 raw BPE 上重建；mixed windows 同时计入两个分母。", "",
        "## 互补性", "",
        f"- v4 阈值以下共有 **{miss_region['windows']:,}** 个窗口，其中 **{miss_region['positive_windows']:,}** 个为 v4 false negatives；exact 在这个完整 v4-rejected 区域的 AP 为 **{miss_region['exact_average_precision']:.6f}**，区域正例率为 {miss_region['positive_prevalence']:.6f}。",
        f"- 全体 raw score：Pearson **{correlations['raw_v4_vs_exact']['pearson']:.6f}**，Spearman **{correlations['raw_v4_vs_exact']['spearman']:.6f}**。",
        f"- 交叉拟合 CDF ranks：Pearson **{correlations['crossfit_cdf_rank_v4_vs_exact']['pearson']:.6f}**，Spearman **{correlations['crossfit_cdf_rank_v4_vs_exact']['spearman']:.6f}**。", "",
        f"固定融合没有显示可用互补：相对 v4，AP 下降 **{overall['v4']['average_precision'] - overall['fusion']['average_precision']:.6f}**，固定 operating point 的 F1 下降 **{overall['v4']['f1'] - overall['fusion']['f1']:.6f}**；冲突召回也从 {type_recall['v4']['conflict']['recall']:.6f} 降至 {type_recall['fusion']['conflict']['recall']:.6f}。exact 在 v4-rejected 区域的 AP 仅比该区域 {miss_region['positive_prevalence']:.6f} 的正例率高 {miss_region['exact_average_precision'] - miss_region['positive_prevalence']:.6f}。低相关在这里没有转化为有效补漏，因此该融合不升格为新候选。", "",
        "## 映射核对与边界", "",
        f"从五个 `fold_*/predictions.npz` 独立组装 34,919 个 fit claim OOF，再按 expanded-v4 冻结 claim→window max 映射重建同一批 {N_SELECTED_WINDOWS:,} 个窗口。response id、answer hash、token ids、offsets、window `(response_id, token_start)` 和二元标签全部逐项相等。独立审计结果见 `AUDIT.json`。", "",
        "本诊断没有打开 combined v4 `scores.npz`，也没有读取 calibration 行/分数或 official test；没有加载模型或使用 GPU。结果只能回答这批 fit 样本上两个固定信号是否呈互补迹象，不能据此把融合登记为新候选。", "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(HERE / "complete.json", {
        "status": "complete_fit_only_exploratory_diagnosis_pending_independent_audit",
        "source_lock_sha256": sha256(HERE / "SOURCE_LOCK.json"),
        "scores_sha256": sha256(HERE / "SCORES.npz"),
        "results_sha256": sha256(HERE / "RESULTS.json"),
        "report_sha256": sha256(HERE / "REPORT.md"),
        "calibration_rows_or_scores_read": False, "official_test_read": False, "GPU_used": False,
    })
    print("EXACT_V4_COMPLEMENT_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
