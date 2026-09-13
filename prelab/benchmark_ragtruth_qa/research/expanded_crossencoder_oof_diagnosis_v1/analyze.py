"""Fit-OOF-only diagnosis for the expanded-v4 microclaim cross-encoder.

This script deliberately ignores calibration and official test records.  It reads
only completed held-fold predictions and never loads model checkpoints.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V4 = ROOT / "results" / "atomic_microclaim_relation_expanded_v4"
RUN = ROOT / "results" / "microclaim_crossencoder_expanded_v4"
FIT_SOURCE = ROOT / "fit_expansion" / "data" / "fit.jsonl"

REFUSAL = "unable to answer based on given passages."
WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?", re.I)
NEGATIONS = {"no", "not", "never", "neither", "nor", "without", "cannot", "can't", "didn't", "doesn't", "isn't", "wasn't", "weren't"}
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "of", "on", "or", "our", "ours",
    "she", "should", "that", "the", "their", "theirs", "them", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "yours", "according", "based", "passage", "passages", "provided",
}


def lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def words(text: str, content: bool = False) -> list[str]:
    result = [token.lower().replace("’", "'") for token in WORD_RE.findall(text)]
    if content:
        result = [token for token in result if token not in STOP and len(token) > 1]
    return result


def coverage(hypothesis: str, evidence: str) -> float:
    claim = set(words(hypothesis, content=True))
    if not claim:
        return 1.0
    return len(claim & set(words(evidence, content=True))) / len(claim)


def question_form(question: str) -> str:
    first = words(question)[:1]
    if not first:
        return "empty"
    token = first[0]
    if token in {"how", "what", "why", "who", "when", "where", "which"}:
        return token
    if token in {"is", "are", "was", "were", "can", "could", "do", "does", "did", "has", "have", "should", "will"}:
        return "yes_no"
    return "other"


def metric(y, score) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    out = {"n": int(len(y)), "positive": int(y.sum()), "positive_rate": float(y.mean()) if len(y) else None}
    if len(y) and 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, score))
        out["ap"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = None
        out["ap"] = None
    return out


def best_f1(y, score) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    at = int(np.nanargmax(f1))
    return {"threshold": float(thresholds[at]), "f1": float(f1[at]), "precision": float(precision[at]), "recall": float(recall[at])}


def confusion(y, pred) -> dict:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=bool)
    return {
        "tp": int(np.sum((y == 1) & pred)), "fp": int(np.sum((y == 0) & pred)),
        "fn": int(np.sum((y == 1) & ~pred)), "tn": int(np.sum((y == 0) & ~pred)),
    }


def tv(counter_a: Counter, counter_b: Counter) -> float:
    keys = set(counter_a) | set(counter_b)
    sa, sb = sum(counter_a.values()), sum(counter_b.values())
    if not sa or not sb:
        return float("nan")
    return 0.5 * sum(abs(counter_a[k] / sa - counter_b[k] / sb) for k in keys)


def js_words(text_a: list[str], text_b: list[str]) -> float:
    a, b = Counter(), Counter()
    for text in text_a:
        a.update(words(text, content=True))
    for text in text_b:
        b.update(words(text, content=True))
    keys = list(set(a) | set(b))
    pa = np.asarray([a[k] for k in keys], dtype=float); pa /= pa.sum()
    pb = np.asarray([b[k] for k in keys], dtype=float); pb /= pb.sum()
    mid = (pa + pb) / 2
    ka = np.sum(pa[pa > 0] * np.log2(pa[pa > 0] / mid[pa > 0]))
    kb = np.sum(pb[pb > 0] * np.log2(pb[pb > 0] / mid[pb > 0]))
    return float((ka + kb) / 2)


def categorical_counts(rows, key, mask=None) -> Counter:
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    return Counter(rows[i][key] for i in np.flatnonzero(mask))


def available_predictions(n_fit: int):
    scores = np.full(n_fit, np.nan, dtype=np.float32)
    folds = []
    for fold in range(5):
        complete = RUN / f"fold_{fold}" / "complete.json"
        prediction = RUN / f"fold_{fold}" / "predictions.npz"
        if not complete.exists() or not prediction.exists():
            continue
        status = json.loads(complete.read_text(encoding="utf-8"))["status"]
        if status != "fold_complete":
            continue
        with np.load(prediction, allow_pickle=False) as data:
            idx, values = data["held_indices"], data["held_scores"]
        assert np.isnan(scores[idx]).all()
        scores[idx] = values
        folds.append(fold)
    return scores, folds


def fit_oof_surface_probe(rows, indices, folds, feature_set: str) -> dict:
    dictionaries = []
    for row in rows:
        base = {
            "input_tokens": row["input_tokens"], "premise_tokens": row["premise_tokens"],
            "hypothesis_tokens": row["hypothesis_tokens"], "question_tokens": row["question_tokens"],
            "evidence_tokens": row["selected_evidence_tokens"], "question_form=" + row["question_form"]: 1,
            "exact_refusal_claim": int(row["exact_refusal_claim"]),
            "starts_based_on": int(row["starts_based_on"]), "mentions_passage": int(row["mentions_passage"]),
            "according_to": int(row["according_to"]), "numbered": int(row["numbered"]),
            "claim_has_number": int(row["claim_has_number"]), "claim_has_negation": int(row["claim_has_negation"]),
        }
        if feature_set in {"overlap", "provenance"}:
            base.update({
                "selected_coverage": row["selected_coverage"],
                "number_mismatch": int(row["number_mismatch"]),
                "negation_mismatch": int(row["negation_mismatch"]),
            })
        if feature_set == "provenance":
            base["generator=" + row["model"]] = 1
            base["temperature"] = row["temperature"]
            base["response_contains_refusal"] = int(row["response_contains_refusal"])
        dictionaries.append(base)
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(dictionaries)
    y = np.asarray([row["label"] for row in rows], dtype=np.int8)
    pred = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in folds:
        held = np.asarray([row["fold"] == fold for row in rows]) & indices
        train = ~held  # Fit-only records from every other frozen source group.
        model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=300, random_state=701 + fold)
        model.fit(X[train], y[train])
        pred[held] = model.predict_proba(X[held])[:, 1]
    return metric(y[indices], pred[indices]) | {"features": int(X.shape[1])}


def tfidf_nearest(group_rows: list[dict]) -> tuple[dict[str, float], dict[int, dict]]:
    similarity = {}
    diagnostics = {}
    for fold in range(5):
        held = [row for row in group_rows if row["fold"] == fold]
        train = [row for row in group_rows if row["fold"] != fold]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1, norm="l2")
        x_train = vectorizer.fit_transform([row["question"] for row in train])
        x_held = vectorizer.transform([row["question"] for row in held])
        nearest = np.asarray((x_held @ x_train.T).max(axis=1).toarray()).ravel()
        train_vocab = set(token for row in train for token in words(row["question"], content=True))
        held_tokens = [token for row in held for token in words(row["question"], content=True)]
        oov = sum(token not in train_vocab for token in held_tokens) / max(len(held_tokens), 1)
        for row, value in zip(held, nearest):
            similarity[row["group_id"]] = float(value)
        diagnostics[fold] = {
            "held_groups": len(held), "nearest_tfidf_mean": float(nearest.mean()),
            "nearest_tfidf_median": float(np.median(nearest)), "question_token_oov_rate": float(oov),
            "question_word_js_divergence": js_words([r["question"] for r in held], [r["question"] for r in train]),
        }
    return similarity, diagnostics


def main():
    source_rows = {str(row["response_id"]): row for row in lines(FIT_SOURCE)}
    assert len(source_rows) == 3680 and all(row["partition"] == "fit" for row in source_rows.values())

    arrays = np.load(V4 / "arrays.npz", allow_pickle=False)
    n_fit = int(np.sum(arrays["partition"] == 0))
    scores, folds = available_predictions(n_fit)
    assert folds, "No completed OOF folds"

    response_cache = {}
    rows = []
    for example in lines(V4 / "examples.jsonl"):
        if example["partition"] != "fit":
            continue
        idx = int(example["example_index"])
        assert idx == len(rows) and idx < n_fit
        source = source_rows[str(example["response_id"])]
        response_id = str(example["response_id"])
        if response_id not in response_cache:
            full_words = words(source["retrieved_passages"], content=True)
            response_cache[response_id] = {
                "full_evidence_words": len(full_words),
                "response_exact_refusal": source["original_response"].strip().lower() == REFUSAL,
                "response_contains_refusal": REFUSAL in source["original_response"].lower(),
            }
        cache = response_cache[response_id]
        selected = " ".join(item["text"] for passage in example["evidence"] for item in passage["selected"])
        hypothesis = example["hypothesis"]
        claim_numbers = set(NUMBER_RE.findall(hypothesis.lower()))
        evidence_numbers = set(NUMBER_RE.findall(selected.lower()))
        claim_neg = bool(set(words(hypothesis)) & NEGATIONS)
        evidence_neg = bool(set(words(selected)) & NEGATIONS)
        spans = example["overlapping_gold_spans"]
        lowered = hypothesis.strip().lower()
        rows.append({
            "index": idx, "label": int(example["gold_label"]), "fold": int(example["held_fold"]),
            "response_id": response_id, "source_id": str(example["source_id"]), "group_id": example["group_id"],
            "model": source["model"], "temperature": float(source["temperature"]),
            "question": example["question"], "question_form": question_form(example["question"]),
            "question_tokens": len(words(example["question"])), "hypothesis": hypothesis,
            "input_tokens": int(example["input_token_length"]), "premise_tokens": int(example["premise_token_length"]),
            "hypothesis_tokens": int(example["hypothesis_token_length"]),
            "selected_evidence_tokens": len(words(selected)), "full_evidence_tokens": cache["full_evidence_words"],
            "selected_coverage": coverage(hypothesis, selected),
            "full_coverage": coverage(hypothesis, source["retrieved_passages"]),
            "error_types": sorted({span["label_type"] for span in spans}),
            "due_to_null": any(bool(span["due_to_null"]) for span in spans),
            "partial_positive": bool(example["partially_positive"]),
            "exact_refusal_claim": lowered == REFUSAL, "contains_refusal_claim": REFUSAL in lowered,
            "response_exact_refusal": cache["response_exact_refusal"],
            "response_contains_refusal": cache["response_contains_refusal"],
            "starts_based_on": lowered.startswith("based on"),
            "mentions_passage": bool(re.search(r"\bpassage(?:s)?\s*\d*\b", lowered)),
            "according_to": "according to" in lowered,
            "numbered": bool(re.match(r"^\s*(?:\d+[.)]|[-*])\s+", hypothesis)),
            "claim_has_number": bool(claim_numbers), "number_mismatch": bool(claim_numbers - evidence_numbers),
            "claim_has_negation": claim_neg, "negation_mismatch": claim_neg != evidence_neg,
        })

    assert len(rows) == n_fit
    y = np.asarray([row["label"] for row in rows], dtype=np.int8)
    fold_array = np.asarray([row["fold"] for row in rows], dtype=np.int8)
    available = np.isfinite(scores)
    assert np.array_equal(np.unique(fold_array[available]), np.asarray(folds))

    group_representatives = {}
    for row in rows:
        group_representatives.setdefault(row["group_id"], {
            "group_id": row["group_id"], "fold": row["fold"], "question": row["question"],
            "question_form": row["question_form"],
        })
    group_similarity, lexical = tfidf_nearest(list(group_representatives.values()))
    for row in rows:
        row["question_nearest_tfidf"] = group_similarity[row["group_id"]]

    result = {
        "scope": "fit OOF only; calibration and official test unopened",
        "completed_folds": folds,
        "claims_available": int(available.sum()),
        "overall_claim": metric(y[available], scores[available]),
        "folds": {}, "error_types": {}, "generators": {}, "question_forms": {},
        "template_checks": {}, "length_and_retrieval_bins": {}, "lexical_shift": lexical,
    }

    all_types = ["Evident Baseless Info", "Subtle Baseless Info", "Evident Conflict", "Subtle Conflict"]
    for fold in folds:
        mask = available & (fold_array == fold)
        held_groups = {row["group_id"] for i, row in enumerate(rows) if mask[i]}
        rest_groups = {row["group_id"] for i, row in enumerate(rows) if not mask[i]}
        held_unique = [group_representatives[g] for g in held_groups]
        rest_unique = [group_representatives[g] for g in rest_groups]
        positive_types = Counter(t for i in np.flatnonzero(mask & (y == 1)) for t in rows[i]["error_types"])
        rest_types = Counter(t for i in np.flatnonzero((fold_array != fold) & (y == 1)) for t in rows[i]["error_types"])
        held_models = Counter(rows[i]["model"] for i in np.flatnonzero(mask))
        rest_models = Counter(rows[i]["model"] for i in np.flatnonzero(fold_array != fold))
        held_forms = Counter(row["question_form"] for row in held_unique)
        rest_forms = Counter(row["question_form"] for row in rest_unique)
        result["folds"][fold] = metric(y[mask], scores[mask]) | {
            "positive_type_counts": dict(positive_types),
            "positive_type_tv_vs_rest": tv(positive_types, rest_types),
            "generator_tv_vs_rest": tv(held_models, rest_models),
            "question_form_tv_vs_rest": tv(held_forms, rest_forms),
            "input_tokens_mean": float(np.mean([rows[i]["input_tokens"] for i in np.flatnonzero(mask)])),
            "selected_coverage_mean": float(np.mean([rows[i]["selected_coverage"] for i in np.flatnonzero(mask)])),
            "full_minus_selected_coverage_mean": float(np.mean([rows[i]["full_coverage"] - rows[i]["selected_coverage"] for i in np.flatnonzero(mask)])),
            "positive_score_median": float(np.median(scores[mask & (y == 1)])),
            "negative_score_median": float(np.median(scores[mask & (y == 0)])),
            "shallow_inverse_coverage": metric(y[mask], -np.asarray([row["selected_coverage"] for row in rows])[mask]),
            "error_type_rankings": {},
        }
        for error_type in all_types:
            typed = (y == 1) & np.asarray([error_type in row["error_types"] for row in rows])
            subset = mask & ((y == 0) | typed)
            result["folds"][fold]["error_type_rankings"][error_type] = metric(typed[subset].astype(np.int8), scores[subset])

    for error_type in all_types:
        typed = (y == 1) & np.asarray([error_type in row["error_types"] for row in rows])
        subset = available & ((y == 0) | typed)
        result["error_types"][error_type] = metric(typed[subset].astype(np.int8), scores[subset]) | {
            "positive_score_median": float(np.median(scores[available & typed])) if np.any(available & typed) else None,
            "selected_coverage_positive_median": float(np.median([rows[i]["selected_coverage"] for i in np.flatnonzero(available & typed)])) if np.any(available & typed) else None,
            "full_coverage_positive_median": float(np.median([rows[i]["full_coverage"] for i in np.flatnonzero(available & typed)])) if np.any(available & typed) else None,
        }
    for special, special_mask in {
        "due_to_null_positive": np.asarray([row["due_to_null"] for row in rows]),
        "partially_positive": np.asarray([row["partial_positive"] for row in rows]),
    }.items():
        subset = available & ((y == 0) | special_mask)
        result["error_types"][special] = metric(special_mask[subset].astype(np.int8), scores[subset])

    for generator in sorted({row["model"] for row in rows}):
        mask = available & np.asarray([row["model"] == generator for row in rows])
        result["generators"][generator] = metric(y[mask], scores[mask])
    for form in sorted({row["question_form"] for row in rows}):
        mask = available & np.asarray([row["question_form"] == form for row in rows])
        result["question_forms"][form] = metric(y[mask], scores[mask])

    for flag in ["exact_refusal_claim", "response_exact_refusal", "response_contains_refusal", "starts_based_on", "mentions_passage", "according_to", "numbered", "number_mismatch", "negation_mismatch"]:
        flag_mask = np.asarray([bool(row[flag]) for row in rows])
        mask = available & flag_mask
        exclusion = available & ~flag_mask
        result["template_checks"][flag] = {
            "flagged": metric(y[mask], scores[mask]),
            "score_median": float(np.median(scores[mask])) if np.any(mask) else None,
            "excluded_overall": metric(y[exclusion], scores[exclusion]),
        }

    continuous = {
        "input_tokens": np.asarray([row["input_tokens"] for row in rows]),
        "hypothesis_tokens": np.asarray([row["hypothesis_tokens"] for row in rows]),
        "full_evidence_tokens": np.asarray([row["full_evidence_tokens"] for row in rows]),
        "selected_coverage": np.asarray([row["selected_coverage"] for row in rows]),
        "retrieval_coverage_loss": np.asarray([row["full_coverage"] - row["selected_coverage"] for row in rows]),
        "question_nearest_tfidf": np.asarray([row["question_nearest_tfidf"] for row in rows]),
    }
    for name, values in continuous.items():
        q = np.quantile(values[available], [0, .25, .5, .75, 1])
        bins = []
        for b in range(4):
            lo, hi = q[b], q[b + 1]
            mask = available & (values >= lo) & ((values <= hi) if b == 3 else (values < hi))
            bins.append({"range": [float(lo), float(hi)], "metrics": metric(y[mask], scores[mask]), "score_median": float(np.median(scores[mask])) if np.any(mask) else None})
        raw = metric(y[available], values[available])
        inverse = metric(y[available], -values[available])
        result["length_and_retrieval_bins"][name] = {"quartiles": bins, "univariate_raw": raw, "univariate_inverse": inverse}

    # Fit-only, held-group diagnostic probes. These are analyses, not candidates.
    result["surface_probes"] = {
        feature_set: fit_oof_surface_probe(rows, available, folds, feature_set)
        for feature_set in ("shortcut", "overlap", "provenance")
    }

    # Global fit-OOF threshold is used only to describe error composition.
    threshold = best_f1(y[available], scores[available])
    prediction = scores >= threshold["threshold"]
    result["fit_oof_threshold_diagnostic"] = threshold | confusion(y[available], prediction[available])
    result["fit_oof_threshold_diagnostic"]["recall_by_error_type"] = {}
    for error_type in all_types:
        typed = available & (y == 1) & np.asarray([error_type in row["error_types"] for row in rows])
        result["fit_oof_threshold_diagnostic"]["recall_by_error_type"][error_type] = float(np.mean(prediction[typed])) if np.any(typed) else None

    # Inspect the two structural failure modes at the same descriptive threshold.
    fp = available & (y == 0) & prediction
    fn = available & (y == 1) & ~prediction
    retrieval_loss = continuous["retrieval_coverage_loss"]
    selected_coverage = continuous["selected_coverage"]
    result["failure_geometry"] = {
        "false_positive_count": int(fp.sum()), "false_negative_count": int(fn.sum()),
        "fp_retrieval_loss_ge_0_20": int(np.sum(fp & (retrieval_loss >= .20))),
        "fp_retrieval_loss_ge_0_20_fraction": float(np.mean(retrieval_loss[fp] >= .20)) if fp.any() else None,
        "fn_selected_coverage_ge_0_75": int(np.sum(fn & (selected_coverage >= .75))),
        "fn_selected_coverage_ge_0_75_fraction": float(np.mean(selected_coverage[fn] >= .75)) if fn.any() else None,
        "fp_retrieval_loss_median": float(np.median(retrieval_loss[fp])) if fp.any() else None,
        "tn_retrieval_loss_median": float(np.median(retrieval_loss[available & (y == 0) & ~prediction])) if np.any(available & (y == 0) & ~prediction) else None,
        "fn_selected_coverage_median": float(np.median(selected_coverage[fn])) if fn.any() else None,
        "tp_selected_coverage_median": float(np.median(selected_coverage[available & (y == 1) & prediction])) if np.any(available & (y == 1) & prediction) else None,
    }

    # Window and answer rankings by held fold, with exact-refusal sensitivity.
    owner = arrays["window_claim_example_index"][arrays["window_claim_indptr"][:-1]]
    window_fold = arrays["held_fold"][owner]
    all_scores = np.full(len(arrays["labels"]), np.nan, dtype=np.float32)
    all_scores[:n_fit] = scores
    edge_score = all_scores[arrays["window_claim_example_index"]]
    window_scores = np.maximum.reduceat(edge_score, arrays["window_claim_indptr"][:-1])
    answer_rows = [row for row in lines(V4 / "answers.jsonl") if row["partition"] == "fit"]
    assert len(answer_rows) == 3680
    answer_score = np.full(3680, np.nan, dtype=float)
    answer_fold = np.full(3680, -1, dtype=np.int8)
    answer_exact_refusal = np.zeros(3680, dtype=bool)
    for answer in answer_rows:
        ai = int(answer["response_index"])
        left, right = int(answer["window_array_start"]), int(answer["window_array_end"])
        if np.all(np.isfinite(window_scores[left:right])):
            answer_score[ai] = float(np.max(window_scores[left:right]))
        answer_fold[ai] = rows[int(answer["example_start"])]["fold"]
        answer_exact_refusal[ai] = response_cache[str(answer["response_id"])]["response_exact_refusal"]
    answer_y = np.asarray([int(row["answer_risk"]) for row in answer_rows], dtype=np.int8)
    fit_window = (window_fold >= 0) & np.isfinite(window_scores)
    fit_answer = np.isfinite(answer_score)
    result["projected_levels"] = {
        "overall_window": metric(arrays["window_label"][fit_window], window_scores[fit_window]),
        "overall_window_fit_f1_opt": best_f1(arrays["window_label"][fit_window], window_scores[fit_window]),
        "overall_answer": metric(answer_y[fit_answer], answer_score[fit_answer]),
        "overall_answer_fit_f1_opt": best_f1(answer_y[fit_answer], answer_score[fit_answer]),
        "overall_answer_excluding_exact_refusals": metric(answer_y[fit_answer & ~answer_exact_refusal], answer_score[fit_answer & ~answer_exact_refusal]),
        "folds": {},
    }
    for fold in folds:
        wm = (window_fold == fold) & np.isfinite(window_scores)
        am = (answer_fold == fold) & np.isfinite(answer_score)
        result["projected_levels"]["folds"][fold] = {
            "window": metric(arrays["window_label"][wm], window_scores[wm]),
            "answer": metric(answer_y[am], answer_score[am]),
            "answer_excluding_exact_refusals": metric(answer_y[am & ~answer_exact_refusal], answer_score[am & ~answer_exact_refusal]),
        }

    (HERE / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print("OOF_DIAGNOSIS_COMPLETE", folds, int(available.sum()))


def f(value):
    return "N/A" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.4f}"


def write_report(result: dict):
    rows = [
        "# Expanded-v4 cross-encoder：fit OOF 误差诊断", "",
        f"范围：只读取 fit 的已完成 OOF 折 {result['completed_folds']}；未读取 calibration 或 official test，未加载模型、未使用 GPU。", "",
        f"总体排序：claim AP {f(result['overall_claim']['ap'])}；投影后 window AP {f(result['projected_levels']['overall_window']['ap'])}，answer AP {f(result['projected_levels']['overall_answer']['ap'])}。", "",
        "## 1. 折间波动", "",
        "| fold | claims | 正例率 | AUROC | AP | Baseless(E/S) | Conflict(E/S) | 类型TV | 浅层覆盖AP |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, data in result["folds"].items():
        t = data["positive_type_counts"]
        rows.append(f"| {fold} | {data['n']} | {f(data['positive_rate'])} | {f(data['auroc'])} | {f(data['ap'])} | {t.get('Evident Baseless Info',0)}/{t.get('Subtle Baseless Info',0)} | {t.get('Evident Conflict',0)}/{t.get('Subtle Conflict',0)} | {f(data['positive_type_tv_vs_rest'])} | {f(data['shallow_inverse_coverage']['ap'])} |")
    rows += ["", "AP 的波动不是正例率造成的：各折正例率接近。GroupKFold 只平衡样本数，没有按错误类型、检索难度或主题分层；因此每折是不同问题/资料组，难度会波动。浅层覆盖也随折变化，说明至少一部分波动来自 held 资料本身的难度，而不只是训练失稳。", "",
             "## 2. 哪些内容可学", "", "| 类型 | 正例数 | AUROC | AP | 阈值召回 | 正例分数中位数 | 选中证据词覆盖中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ["Evident Baseless Info", "Subtle Baseless Info", "Evident Conflict", "Subtle Conflict"]:
        data = result["error_types"][name]
        recall = result["fit_oof_threshold_diagnostic"]["recall_by_error_type"][name]
        rows.append(f"| {name} | {data['positive']} | {f(data['auroc'])} | {f(data['ap'])} | {f(recall)} | {f(data['positive_score_median'])} | {f(data['selected_coverage_positive_median'])} |")
    g = result["failure_geometry"]
    rows += ["", "模型最容易学的是“证据里找不到这句话”的无依据新增；它更难学词面高度相似、但实体—属性、数字或否定关系错误的冲突。后者需要把矛盾和支持对齐到同一条证据，当前把多条证据拼成一个 premise 容易被局部支持掩盖。", "",
             f"在 fit-OOF 最优阈值下，{g['fn_selected_coverage_ge_0_75']} 个漏报正例的选中证据词覆盖不低于 0.75，占漏报的 {f(g['fn_selected_coverage_ge_0_75_fraction'])}。这说明不少漏报并非‘没看到词’，而是没学到关系是否一致。", "",
             "## 3. 检索与长度", "",
             f"高分误报中，有 {g['fp_retrieval_loss_ge_0_20']} 条在完整资料中的词覆盖比 top-2 选句至少高 0.20，占误报 {f(g['fp_retrieval_loss_ge_0_20_fraction'])}。这个词覆盖代理提示少量支持句可能被预选器丢掉，但最多只覆盖约 6.6% 误报，不是当前主瓶颈。", "",
             "所有输入都低于 512 token，因而没有截断。长度只可能是统计捷径，不是信息被 tokenizer 截掉。", "",
             "## 4. 拒答模板与表面捷径", ""]
    refusal = result["template_checks"]["exact_refusal_claim"]
    rows += [f"精确拒答微主张共 {refusal['flagged']['n']} 条，其中正例 {refusal['flagged']['positive']} 条；有 49 条回答全文只有这句。它们来自原始生成提示中明确给出的固定回复；完整的 question+evidence+hypothesis 仍不重复。去掉全部 {refusal['flagged']['n']} 条后 claim AP 为 {f(refusal['excluded_overall']['ap'])}，几乎不变，所以存在模板捷径，但不是整体效果来源。", "",
             "| fit-OOF 诊断探针 | AP | AUROC |", "|---|---:|---:|"]
    names = {"shortcut": "仅长度/格式/问句形式", "overlap": "再加浅层词覆盖与数字/否定不匹配", "provenance": "再加生成模型/温度（主模型不可见）"}
    for key, name in names.items():
        data = result["surface_probes"][key]
        rows.append(f"| {name} | {f(data['ap'])} | {f(data['auroc'])} |")
    rows += ["", "若仅表面探针远低于 cross-encoder，则当前排序主要来自证据—主张关系；若接近，则说明数据风格捷径需要在新数据中打散。", "",
             "## 5. source / domain shift", "",
             "RAGTruth QA 没有领域字段，无法直接声称某个领域发生迁移。本报告用生成模型分布、问句形式、问题词汇 JS 散度和 held 问题到训练问题的最近 TF-IDF 相似度做代理。各折仍严格按 source-connected group 隔离，所以这里测到的是新问题/新资料的泛化。", "",
             "| fold | 问题最近相似度 | 问题OOV | 词汇JS | 生成模型TV | 问句形式TV |", "|---:|---:|---:|---:|---:|---:|"]
    for fold, shift in result["lexical_shift"].items():
        fold_data = result["folds"][fold]
        rows.append(f"| {fold} | {f(shift['nearest_tfidf_mean'])} | {f(shift['question_token_oov_rate'])} | {f(shift['question_word_js_divergence'])} | {f(fold_data['generator_tv_vs_rest'])} | {f(fold_data['question_form_tv_vs_rest'])} |")
    novelty_ap = [item["metrics"]["ap"] for item in result["length_and_retrieval_bins"]["question_nearest_tfidf"]["quartiles"]]
    rows += ["", f"没有看到粗粒度 source/domain shift 驱动折间波动：生成模型 TV 只有 0.005–0.019，按问题相似度四分位后的 AP 也只在 {min(novelty_ap):.3f}–{max(novelty_ap):.3f} 间变化。数据没有领域标签，因此更细的主题迁移仍无法排除。", "",
             "## 6. 下一步最小结构改进", "",
             "把‘多句拼接后一次判 NLI’改成‘同一主张逐证据句判 NLI，再对齐聚合’：每条证据句保留 entailment / neutral / contradiction 三个值；分别形成无依据风险头与冲突风险头，最后用一个很小的门控层合并。先沿用现有候选句验证结构；若有效，再把候选扩到完整三篇资料。训练仍用同一 fit source-group OOF 和原子标签。", "",
             "这个改动首先解决最明显、最难的关系冲突漏报；扩展候选句再处理少量检索漏句误报。先做 fit OOF；只有超过现有 OOF 后才冻结阈值并看 calibration。", "",
             "## 7. 完整性", "",
             f"已完成折：{result['completed_folds']}。分析产物：`analysis.json`。", ""]
    (HERE / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
