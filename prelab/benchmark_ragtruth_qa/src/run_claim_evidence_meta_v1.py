"""Fit-only-selected claim detector over frozen white-box and evidence signals.

This is an in-house development candidate.  It leaves every formal baseline
unchanged.  Fit claims receive only group-held-out upstream white-box scores;
calibration is opened after the candidate family and fit-only winner are fixed.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_development as q
import run_retrieved_evidence_nli_v1 as nli


ROOT = q.ROOT
OUT = ROOT / "results/claim_evidence_meta_v1"
PREPARED = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
OOF_WHITEBOX = ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy"
GENERATION = ROOT / "results/development_v1/matrices/base.npy"
LOCAL_NLI = ROOT / "results/nli_local_signal_cuda_scoring_v1/window_nli_features.npy"
CITATION = ROOT / "results/citation_alignment_v1/window_features.npy"
RETRIEVED = ROOT / "results/retrieved_evidence_nli_citation_v2/claim_features.npy"
RETRIEVED_SCORE = ROOT / "results/retrieved_evidence_nli_citation_v2/citation_lr_scores.npz"

SEED = 20261016
FOLDS = 5
AGGREGATES = ("minimum", "q25", "mean", "median", "q75", "maximum", "std")


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    return np.asarray([
        values.min(), np.quantile(values, .25), values.mean(), np.median(values),
        np.quantile(values, .75), values.max(), values.std(),
    ], dtype=np.float64)


def text_features(row, claim):
    text = claim["text"]
    alnum = [ch for ch in text if ch.isalnum()]
    upper = sum(ch.isupper() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    claim_count = len(row["claims"])
    cid = claim["claim_id"]
    return np.asarray([
        np.log1p(len(text)), np.log1p(len(words)), np.log1p(len(claim["lexical_token_indices"])),
        cid / max(1, claim_count - 1), (cid + 1) / claim_count,
        digits / max(1, len(alnum)), upper / max(1, len(alnum)),
        float(any(ch.isdigit() for ch in text)), float("%" in text),
        float("$" in text or "€" in text or "£" in text),
        float(bool(claim["citation_parse"]["references"])),
        float(claim["citation_parse"]["status"] == "valid"),
        float(claim["citation_parse"]["status"] == "invalid"),
        float(" not " in f" {text.lower()} " or "n't" in text.lower()),
        float(any(mark in text for mark in (">", "<", "more", "less", "higher", "lower"))),
    ], dtype=np.float64)


TEXT_NAMES = (
    "log_chars", "log_words", "log_lexical_tokens", "claim_position_0", "claim_position_1",
    "digit_fraction", "uppercase_fraction", "has_digit", "has_percent", "has_money",
    "has_citation", "citation_valid", "citation_invalid", "has_negation", "has_comparison",
)


def make_model(name):
    if name == "lr_C0.01":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.01, solver="liblinear", max_iter=3000, random_state=SEED))
    if name == "lr_C0.1":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, solver="liblinear", max_iter=3000, random_state=SEED))
    if name == "hist_leaf7":
        return HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, max_leaf_nodes=7,
            min_samples_leaf=30, l2_regularization=1., early_stopping=False,
            random_state=SEED)
    if name == "hist_leaf15":
        return HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1., early_stopping=False,
            random_state=SEED)
    if name == "extra_depth8":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=8,
            max_features=.5, n_jobs=4, random_state=SEED)
    if name == "extra_depth12":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=8,
            max_features=.5, n_jobs=4, random_state=SEED)
    raise KeyError(name)


CANDIDATES = ("lr_C0.01", "lr_C0.1", "hist_leaf7", "hist_leaf15", "extra_depth8", "extra_depth12")


def build_features(meta, prepared):
    assert len(prepared) == 793
    oof = np.load(OOF_WHITEBOX, mmap_mode="r")
    generation = np.load(GENERATION, mmap_mode="r")
    local_nli = np.load(LOCAL_NLI, mmap_mode="r")
    citation = np.load(CITATION, mmap_mode="r")
    retrieved = np.load(RETRIEVED, mmap_mode="r")
    with np.load(RETRIEVED_SCORE, allow_pickle=False) as z:
        retrieved_score = z["claim_scores"].astype(np.float64)
    assert oof.shape == (210364, 2) and generation.shape == (210364, 1025)
    assert local_nli.shape == (210364, 12) and citation.shape == (210364, 8)
    assert retrieved.shape == (8845, 70) and retrieved_score.shape == (8845,)
    window_matrix = np.column_stack((oof, np.asarray(generation[:, -1]), local_nli, citation))
    base_names = (["oof_lookback", "oof_large", "generation_nll"] +
                  [f"local_nli_{i}" for i in range(12)] +
                  [f"citation_{i}" for i in range(8)])
    assert window_matrix.shape == (210364, len(base_names))

    feature_rows, response_ids, groups, partitions = [], [], [], []
    claim_offsets = {}
    cursor = 0
    lookup = {row["response_id"]: row for row in prepared}
    for row in prepared:
        rid = row["response_id"]
        claim_offsets[rid] = cursor
        owners = np.asarray(row["lexical_token_claim"], dtype=np.int32)
        memberships = [[] for _ in row["claims"]]
        for wi in meta["answer_windows"][rid]:
            claim_ids = {int(owners[token_index]) for token_index in meta["windows"][wi]["token_indices"]
                         if owners[token_index] >= 0}
            for cid in claim_ids:
                memberships[cid].append(wi)
        for claim, window_ids in zip(row["claims"], memberships):
            assert window_ids
            parts = [aggregate(window_matrix[window_ids, column])
                     for column in range(window_matrix.shape[1])]
            parts += [np.asarray(retrieved[cursor], dtype=np.float64),
                      np.asarray([retrieved_score[cursor]], dtype=np.float64),
                      text_features(row, claim)]
            feature_rows.append(np.concatenate(parts))
            response_ids.append(rid)
            groups.append(row["group_id"])
            partitions.append(row["partition"])
            cursor += 1
    assert cursor == 8845 and set(lookup) == {answer["response_id"] for answer in meta["answers"]}
    names = [f"{name}_{stat}" for name in base_names for stat in AGGREGATES]
    names += [f"retrieved_{i}" for i in range(70)] + ["retrieved_oof_score"] + list(TEXT_NAMES)
    x = np.asarray(feature_rows, dtype=np.float64)
    assert x.shape == (8845, len(names)) and np.isfinite(x).all()
    return x, names, response_ids, np.asarray(groups), np.asarray(partitions), claim_offsets


def run():
    assert not OUT.exists(), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    meta = q.metadata()
    prepared = rows(PREPARED)
    x, names, response_ids, groups, partitions, claim_offsets = build_features(meta, prepared)
    y = np.load(ROOT / "results/retrieved_evidence_nli_v1/claim_labels.npy", allow_pickle=False)
    assert y.shape == (8845,) and set(y.tolist()) == {0, 1}
    fit = np.flatnonzero(partitions == "fit")
    cal = np.flatnonzero(partitions == "calibration")
    lookup = {row["response_id"]: row for row in prepared}
    splitter = GroupKFold(FOLDS)
    folds = list(splitter.split(fit, y[fit], groups[fit]))
    fit_left, fit_right = meta["bounds"]["fit"]
    fit_answer_indices = np.asarray([i for i, answer in enumerate(meta["answers"])
                                     if answer["partition"] == "fit"], dtype=np.int64)
    fit_window_y = np.asarray([window["label"] for window in meta["windows"][fit_left:fit_right]])
    fit_answer_y = np.asarray([meta["answers"][i]["label"] for i in fit_answer_indices])
    histories, predictions, models = {}, {}, {}
    with threadpool_limits(limits=4):
        for name in CANDIDATES:
            oof_claim = np.full(len(y), np.nan, dtype=np.float64)
            fold_models = []
            for fold, (train_local, held_local) in enumerate(folds):
                train, held = fit[train_local], fit[held_local]
                weights = nli.nested_claim_weights(lookup, response_ids, y, train)
                model = make_model(name)
                model.fit(x[train], y[train], **({"sample_weight": weights[train]}
                          if not hasattr(model, "steps") else {"logisticregression__sample_weight": weights[train]}))
                oof_claim[held] = model.predict_proba(x[held])[:, 1]
                fold_models.append(model)
            assert np.isfinite(oof_claim[fit]).all()
            # Projection validates every answer; calibration values are a
            # harmless finite placeholder until the fit-only winner is frozen.
            oof_claim[cal] = 0.0
            fit_window = nli.project_claim_scores(meta, prepared, claim_offsets, oof_claim)
            fit_answer = q.answer_scores(meta, fit_window)
            wt = q.choose_threshold(fit_window_y, fit_window[fit_left:fit_right])
            at = q.choose_threshold(fit_answer_y, fit_answer[fit_answer_indices])
            key = [min(wt["f1"], at["f1"]), wt["f1"], at["f1"], wt["precision"]]
            histories[name] = {"fit_oof_thresholds": {"window": wt, "answer": at},
                               "fit_selection_key": key}
            predictions[name] = oof_claim
            models[name] = fold_models
            print("CLAIM_META_OOF", name, round(wt["f1"], 6), round(at["f1"], 6), flush=True)
    selected = max(CANDIDATES, key=lambda name: histories[name]["fit_selection_key"])
    save(OUT / "fit_only_selection.json", {"selected": selected, "candidates": histories,
         "selection_used_calibration": False, "official_test_opened": False})

    full_weights = nli.nested_claim_weights(lookup, response_ids, y, fit)
    model = make_model(selected)
    kwargs = ({"sample_weight": full_weights[fit]} if not hasattr(model, "steps")
              else {"logisticregression__sample_weight": full_weights[fit]})
    with threadpool_limits(limits=4):
        model.fit(x[fit], y[fit], **kwargs)
    claim_scores = predictions[selected]
    claim_scores[cal] = model.predict_proba(x[cal])[:, 1]
    assert np.isfinite(claim_scores).all()
    window_scores = nli.project_claim_scores(meta, prepared, claim_offsets, claim_scores)
    answer_scores = q.answer_scores(meta, window_scores)
    thresholds = histories[selected]["fit_oof_thresholds"]
    metrics = q.metrics(meta, window_scores, thresholds)
    np.savez_compressed(OUT / "scores.npz", claim_scores=claim_scores,
                        window_scores=window_scores, answer_scores=answer_scores)
    np.save(OUT / "features.npy", x.astype(np.float32))
    (OUT / "model.pkl").write_bytes(pickle.dumps({"model": model, "selected": selected,
        "feature_names": names, "fold_models": models[selected]}, protocol=5))
    summary = {
        "status": "development_only_complete", "role": "ours_method_candidate; baselines unchanged",
        "selected_by_fit_group_OOF_only": selected, "candidate_family": list(CANDIDATES),
        "claims": len(y), "features": len(names), "folds": FOLDS,
        "thresholds_from_fit_group_OOF": thresholds, "metrics": metrics,
        "fit_selection": histories, "seconds": time.perf_counter() - started,
        "calibration_used_for_model_or_threshold_selection": False,
        "official_test_opened": False, "final_test_claim": False,
    }
    save(OUT / "summary.json", summary)
    report = [
        "# Claim evidence meta v1", "",
        "六个候选只按 fit 的 source-group OOF 双层 F1 选择；calibration 在选择冻结后才计分。正式基线未改。", "",
        "| fit-only候选 | OOF窗口F1 | OOF整答F1 |", "|---|---:|---:|",
    ]
    for name in CANDIDATES:
        t = histories[name]["fit_oof_thresholds"]
        report.append(f"| {name} | {t['window']['f1']:.6f} | {t['answer']['f1']:.6f} |")
    report += ["", f"选中 `{selected}`。", "",
               "| 划分 | 窗口F1 | 整答F1 |", "|---|---:|---:|",
               f"| fit OOF | {metrics['fit']['windows']['f1']:.6f} | {metrics['fit']['answers']['f1']:.6f} |",
               f"| calibration | {metrics['calibration']['windows']['f1']:.6f} | {metrics['calibration']['answers']['f1']:.6f} |",
               "", "句段分数投影回原4-BPE窗口；测试仍封存。"]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("CLAIM_EVIDENCE_META_COMPLETE", selected,
          metrics["calibration"]["windows"]["f1"], metrics["calibration"]["answers"]["f1"], flush=True)


if __name__ == "__main__":
    run()
