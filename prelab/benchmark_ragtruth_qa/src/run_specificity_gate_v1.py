"""Fit-only-selected specificity gate over the incumbent and atomic NLI scores.

The experiment is deliberately small.  It compares three deterministic atomic
claim propagations and three compact group-cross-fitted gates.  Calibration is
opened only by the final report stage, after the method and thresholds are
frozen in ``fit_complete.json``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402


OUT = ROOT / "results/specificity_gate_v1"
ATOMIC = ROOT / "results/atomic_microclaim_nli_v1"
INC_DIR = ROOT / "results/large_fixed_convex_v1"
INC_SCORE = INC_DIR / "semantic_claim__old_tree__large_weight0.4_scores.npz"
INC_RESULT = INC_DIR / "semantic_claim__old_tree__large_weight0.4.json"
ROWS_PATH = ATOMIC / "inputs.jsonl"
FEATURE_PATH = ATOMIC / "evidence_features.npy"
FEATURE_NAMES_PATH = ATOMIC / "feature_names.json"
NLI_SCORE = ATOMIC / "scores.npz"
NLI_RESULT = ATOMIC / "summary.json"

SEED = 20261027
FOLDS = 5
MIN_ADD = 32
MIN_INCREMENTAL_PRECISION = 0.55
DETERMINISTIC = ("atomic_max", "atomic_q90", "atomic_q75")
GATES = ("gate_relative_lr", "gate_full_lr", "gate_full_hgb")
CANDIDATES = DETERMINISTIC + GATES

BASE_FEATURES = (
    "incumbent_window",
    "incumbent_atomic_max",
    "incumbent_atomic_q90",
    "incumbent_atomic_q75",
    "incumbent_atomic_mean",
    "atomic_nli_readout",
    "atomic_nli_raw_risk",
    "nli_global_max_entailment",
    "nli_global_max_contradiction",
    "nli_contradiction_minus_entailment",
    "retrieval_query_coverage",
    "relation_claim_coverage",
    "log1p_top_bm25",
    "cited_contradiction_minus_entailment",
    "explicit_or_parent_source_count",
)
RELATIVE_FEATURES = (
    "nli_minus_answer_median",
    "nli_minus_answer_max",
    "nli_within_answer_rank",
    "incumbent_minus_answer_median",
    "incumbent_minus_answer_max",
    "incumbent_within_answer_rank",
    "incumbent_answer_max",
    "incumbent_distance_to_fit_cutoff",
)
ALL_FEATURES = BASE_FEATURES + RELATIVE_FEATURES
RELATIVE_INDICES = tuple(ALL_FEATURES.index(name) for name in (
    "incumbent_window", "incumbent_atomic_max", "incumbent_atomic_q90",
    "atomic_nli_readout", "atomic_nli_raw_risk", *RELATIVE_FEATURES,
))


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_paths():
    return (
        Path(__file__),
        INC_SCORE, INC_RESULT, NLI_SCORE, NLI_RESULT, ROWS_PATH, FEATURE_PATH,
        FEATURE_NAMES_PATH, ATOMIC / "complete.json", ROOT / "data/gold_manifest.json",
    )


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in source_paths()}


def protocol():
    return {
        "version": "specificity-gate-v1",
        "purpose": "Filter or propagate localization evidence without changing any external baseline.",
        "scope": "RAGTruth QA fit634 plus calibration159 development data; official test is absent.",
        "inputs": {
            "incumbent": "Frozen semantic_claim__old_tree__large_weight0.4 window score. Its historical calibration-selected result remains a read-only comparison only.",
            "atomic_nli": "Frozen atomic_microclaim_nli_v1; fit entries are its stored source-group OOF predictions and calibration entries are its stored full-fit predictions.",
        },
        "atomic_propagation": {
            "seed_windows": "All eligible 4-BPE windows touching a deterministic label-blind atomic microclaim.",
            "claim_aggregates": ["max", "90th percentile", "75th percentile"],
            "projection": "Each 4-BPE window receives the maximum aggregate among atomic microclaims touched by its lexical tokens.",
        },
        "gate_pool": "All windows below the fit-selected incumbent window cutoff. No calibration statistic defines this pool.",
        "gate_features": {
            "same_answer_relative": list(RELATIVE_FEATURES),
            "nli_relation": ["entailment", "contradiction", "contradiction-entailment margin"],
            "retrieval": ["BM25", "lexical query coverage", "relation coverage", "cited/source consistency"],
            "incumbent": ["raw window", "atomic max/q90/q75/mean", "distance to cutoff"],
        },
        "candidates": list(CANDIDATES),
        "models": {
            "gate_relative_lr": "StandardScaler + L2 logistic regression, C=0.01",
            "gate_full_lr": "StandardScaler + L2 logistic regression, C=0.01",
            "gate_full_hgb": "HistGradientBoosting, 7 leaves, 150 iterations, min leaf 100, L2=2",
        },
        "crossfit": "Five-fold GroupKFold by locked source-connected group. Every learned fit gate score is held-group prediction.",
        "weights": "Equal source-group mass, equal answer mass inside group, equal window mass inside answer, then binary class balance.",
        "thresholds": {
            "incumbent": "Window and answer cutoffs independently maximize F1 on fit only.",
            "gate": f"Fit OOF only; maximize union F1 subject to at least {MIN_ADD} additions and incremental precision >= {MIN_INCREMENTAL_PRECISION}.",
            "deterministic": "Window and answer cutoffs independently maximize F1 on fit only.",
        },
        "selection": "Fit only: maximize min(window F1, answer F1), then window F1, window precision, incremental precision, then earlier candidate order.",
        "calibration": "Only the selected frozen candidate is strictly evaluated once. Calibration-F1-optimal values are a separately marked non-independent diagnostic.",
        "acceptance": "Replace neither incumbent nor baseline unless strict calibration window F1 exceeds 0.6902813989031736 and answer F1 is at least 0.8910891089108911.",
        "limitations": [
            "The incumbent source artifact itself was historically selected on calibration; this experiment does not retune or alter it.",
            "Its fit score provenance is fixed upstream and is not newly cross-fitted here; only learned gate predictions are source-group OOF.",
            "Calibration remains a repeatedly observed development split, not an independent test.",
        ],
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def load_fit_meta():
    # Deliberately do not open either calibration JSONL in this stage.
    answers = q.lines(ROOT / "data/answers_fit.jsonl")
    tokens = q.lines(ROOT / "data/tokens_fit.jsonl")
    windows = q.lines(ROOT / "data/windows_k4_fit.jsonl")
    assert len(answers) == len(tokens) == 634 and len(windows) == 168123
    by_response = {a["response_id"]: {"answer": a, "tokens": t} for a, t in zip(answers, tokens)}
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[window["response_id"]].append(index)
    return {
        "answers": answers, "windows": windows, "by_response": by_response,
        "answer_windows": dict(answer_windows), "bounds": {"fit": [0, len(windows)]},
    }


def partition_view(meta, partition):
    left, right = meta["bounds"][partition]
    answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == partition]
    answers = [meta["answers"][i] for i in answer_indices]
    windows = meta["windows"][left:right]
    local_aw = {answer["response_id"]: [i - left for i in meta["answer_windows"][answer["response_id"]]]
                for answer in answers}
    return answers, windows, local_aw, left


def empirical_ranks(values):
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1)
    return result


def build_partition(meta, partition, incumbent, nli_readout, nli_raw, evidence, names, atomic_rows,
                    claim_offsets, incumbent_cutoff=None):
    answers, windows, answer_windows, global_left = partition_view(meta, partition)
    local_n = len(windows)
    row_by_id = {row["response_id"]: row for row in atomic_rows if row["partition"] == partition}
    assert len(row_by_id) == len(answers)
    index = {name: i for i, name in enumerate(names)}

    claim_vectors = {}
    claim_aggregates = {}
    for answer in answers:
        rid = answer["response_id"]
        row = row_by_id[rid]
        owners = row["lexical_token_microclaims"]
        touched = [[] for _ in row["claims"]]
        for local_wi in answer_windows[rid]:
            global_wi = global_left + local_wi
            cids = {int(cid) for token_index in windows[local_wi]["token_indices"] for cid in owners[token_index]}
            for cid in cids:
                touched[cid].append(float(incumbent[global_wi]))
        aggs = np.empty((len(row["claims"]), 4), dtype=np.float64)
        for cid, values in enumerate(touched):
            assert values
            aggs[cid] = [max(values), np.quantile(values, .90), np.quantile(values, .75), np.mean(values)]
        claim_aggregates[rid] = aggs

        start = claim_offsets[rid]
        stop = start + len(row["claims"])
        one = evidence[start:stop]
        assert len(one) == len(row["claims"])
        passage_cov = one[:, [index[f"passage_{pid}_union_query_coverage"] for pid in (1, 2, 3)]].max(1)
        relation_cov = one[:, [index[f"passage_{pid}_relation_max__claim_token_coverage"] for pid in (1, 2, 3)]].max(1)
        top_bm25 = one[:, [index[f"passage_{pid}_top_bm25"] for pid in (1, 2, 3)]].max(1)
        entail = one[:, index["global_max_entailment"]]
        contradiction = one[:, index["global_max_contradiction"]]
        cited_margin = one[:, index["cited_max_contradiction"]] - one[:, index["cited_max_entailment"]]
        source_count = one[:, index["log1p_explicit_sources"]] + one[:, index["log1p_parent_sources"]]
        claim_vectors[rid] = np.column_stack((
            aggs[:, 0], aggs[:, 1], aggs[:, 2], aggs[:, 3],
            nli_readout[start:stop], nli_raw[start:stop], entail, contradiction,
            contradiction - entail, passage_cov, relation_cov,
            np.log1p(np.maximum(top_bm25, 0)), cited_margin, source_count,
        ))

    projected = np.empty((local_n, 14), dtype=np.float64)
    atomic_scores = {name: np.empty(local_n, dtype=np.float64) for name in DETERMINISTIC}
    inc_local = incumbent[global_left:global_left + local_n].astype(np.float64, copy=True)
    nli_window = np.empty(local_n, dtype=np.float64)
    raw_window = np.empty(local_n, dtype=np.float64)
    for answer in answers:
        rid = answer["response_id"]
        row = row_by_id[rid]
        owners = row["lexical_token_microclaims"]
        start = claim_offsets[rid]
        for local_wi in answer_windows[rid]:
            cids = {int(cid) for token_index in windows[local_wi]["token_indices"] for cid in owners[token_index]}
            ids = np.asarray(sorted(cids), dtype=np.int64)
            projected[local_wi] = claim_vectors[rid][ids].max(0)
            atomic_scores["atomic_max"][local_wi] = claim_aggregates[rid][ids, 0].max()
            atomic_scores["atomic_q90"][local_wi] = claim_aggregates[rid][ids, 1].max()
            atomic_scores["atomic_q75"][local_wi] = claim_aggregates[rid][ids, 2].max()
            nli_window[local_wi] = projected[local_wi, 4]
            raw_window[local_wi] = projected[local_wi, 5]

    # projected columns: atomic aggregates (4), NLI readout/raw (2), then evidence (8).
    assert np.allclose(projected[:, 4], nli_window)
    assert np.allclose(projected[:, 5], raw_window)
    base = np.column_stack((inc_local, projected))
    relative = np.empty((local_n, len(RELATIVE_FEATURES)), dtype=np.float64)
    for answer in answers:
        ids = np.asarray(answer_windows[answer["response_id"]], dtype=np.int64)
        ni = nli_window[ids]
        ii = inc_local[ids]
        nmed, nmax = np.median(ni), ni.max()
        imed, imax = np.median(ii), ii.max()
        relative[ids] = np.column_stack((
            ni - nmed, ni - nmax, empirical_ranks(ni), ii - imed, ii - imax,
            empirical_ranks(ii), np.full(len(ids), imax),
            ii - (float(incumbent_cutoff) if incumbent_cutoff is not None else 0.0),
        ))
    x = np.column_stack((base, relative))
    assert x.shape == (local_n, len(ALL_FEATURES)) and np.isfinite(x).all()
    return {
        "answers": answers, "windows": windows, "answer_windows": answer_windows,
        "global_left": global_left, "x": x, "incumbent": inc_local,
        "nli": nli_window, "raw_nli": raw_window, "atomic_scores": atomic_scores,
    }


def answer_scores(view, scores):
    return np.asarray([scores[view["answer_windows"][answer["response_id"]]].max()
                       for answer in view["answers"]], dtype=np.float64)


def metric_pair(view, scores, thresholds):
    wy = np.asarray([window["label"] for window in view["windows"]], dtype=np.int8)
    ay = np.asarray([answer["label"] for answer in view["answers"]], dtype=np.int8)
    return {
        "windows": q.count(wy, scores, thresholds["window"]["threshold"]),
        "answers": q.count(ay, answer_scores(view, scores), thresholds["answer"]["threshold"]),
    }


def fit_thresholds(view, scores):
    return {
        "window": q.choose_threshold([w["label"] for w in view["windows"]], scores),
        "answer": q.choose_threshold([a["label"] for a in view["answers"]], answer_scores(view, scores)),
    }


def sample_weights(view, active, labels):
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        window = view["windows"][index]
        tree[window["group_id"]][window["answer_id"]].append(index)
    weights = np.zeros(len(labels), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values():
            weights[indices] = 1 / (len(answers) * len(indices))
    mask = weights > 0
    weights[mask] /= weights[mask].mean()
    mass = np.bincount(labels[mask], weights=weights[mask], minlength=2)
    assert np.all(mass > 0)
    weights[mask] *= (mass.sum() / (2 * mass))[labels[mask]]
    for answers in tree.values():
        indices = [index for values in answers.values() for index in values]
        weights[indices] *= (mask.sum() / len(tree)) / weights[indices].sum()
    weights[mask] *= mask.sum() / weights[mask].sum()
    return weights


def make_model(name):
    if name in ("gate_relative_lr", "gate_full_lr"):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.01, solver="liblinear", penalty="l2", max_iter=3000, random_state=SEED))
    assert name == "gate_full_hgb"
    return HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=150, max_leaf_nodes=7, min_samples_leaf=100,
        l2_regularization=2., early_stopping=False, random_state=SEED)


def model_x(name, x):
    return x[:, RELATIVE_INDICES] if name == "gate_relative_lr" else x


def fit_model(model, x, y, weight):
    key = "logisticregression__sample_weight" if hasattr(model, "steps") else "sample_weight"
    model.fit(x, y, **{key: weight})
    return model


def choose_gate_threshold(y, base_prediction, active, scores):
    indices = np.flatnonzero(active)
    order = np.argsort(-scores[indices], kind="stable")
    sorted_indices = indices[order]
    sorted_scores = scores[sorted_indices]
    last = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(sorted_scores) - 1]
    add_tp = np.cumsum(y[sorted_indices])[last]
    add_n = last + 1
    add_fp = add_n - add_tp
    base_tp = int(np.count_nonzero(base_prediction & (y == 1)))
    base_fp = int(np.count_nonzero(base_prediction & (y == 0)))
    total_positive = int(y.sum())
    precision = add_tp / add_n
    f1 = 2 * (base_tp + add_tp) / (2 * (base_tp + add_tp) + base_fp + add_fp + total_positive - base_tp - add_tp)
    valid = (add_n >= MIN_ADD) & (precision >= MIN_INCREMENTAL_PRECISION)
    if not valid.any():
        return {"threshold": float(np.nextafter(sorted_scores.max(), np.inf)), "union_f1": q.count(y, base_prediction, .5)["f1"],
                "incremental_precision": 0., "additions": 0, "tp": 0, "fp": 0}
    choices = np.flatnonzero(valid)
    best = max(choices, key=lambda j: (f1[j], precision[j], -add_n[j], sorted_scores[last[j]]))
    return {"threshold": float(sorted_scores[last[best]]), "union_f1": float(f1[best]),
            "incremental_precision": float(precision[best]), "additions": int(add_n[best]),
            "tp": int(add_tp[best]), "fp": int(add_fp[best])}


def gated_score(incumbent, gate_probability, gate_threshold, base_window_threshold, base_answer_threshold):
    result = incumbent.astype(np.float64, copy=True)
    hit = (incumbent < base_window_threshold) & (gate_probability >= gate_threshold)
    # Put additions over the window cutoff but strictly below the answer cutoff.
    # Therefore the window set changes while no previously negative answer flips.
    assert base_answer_threshold > base_window_threshold
    if hit.any():
        probability = gate_probability[hit]
        scale = np.maximum(np.max(probability) - gate_threshold, np.finfo(np.float64).eps)
        fraction = np.clip((probability - gate_threshold) / scale, 0, 1)
        upper = np.nextafter(base_answer_threshold, -np.inf)
        result[hit] = base_window_threshold + (upper - base_window_threshold) * (.25 + .74 * fraction)
    assert np.array_equal(result >= base_window_threshold, (incumbent >= base_window_threshold) | hit)
    return result, hit


def initialize():
    assert not OUT.exists(), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True)
    save(OUT / "protocol.json", protocol())
    save(OUT / "initialized.json", {
        "status": "preregistered_before_fit_selection", "protocol_sha256": sha(OUT / "protocol.json"),
        "source_sha256": source_hashes(), "calibration_labels_opened": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("SPECIFICITY_GATE_V1_PREREGISTERED")


def load_sources():
    incumbent = np.load(INC_SCORE, allow_pickle=False)["window_scores"].astype(np.float64)
    with np.load(NLI_SCORE, allow_pickle=False) as data:
        nli_window = data["window_scores"].astype(np.float64)
        nli_raw_window = data["raw_window_scores"].astype(np.float64)
        nli_claim = data["claim_scores"].astype(np.float64)
        nli_raw_claim = data["raw_claim_scores"].astype(np.float64)
    atomic_rows = rows(ROWS_PATH)
    evidence = np.load(FEATURE_PATH, mmap_mode="r")
    names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))["evidence"]
    assert incumbent.shape == nli_window.shape == nli_raw_window.shape == (210364,)
    assert nli_claim.shape == nli_raw_claim.shape == (11322,) and evidence.shape[0] == 11322
    claim_offsets = {}
    cursor = 0
    for row in atomic_rows:
        claim_offsets[row["response_id"]] = cursor
        cursor += len(row["claims"])
    assert cursor == 11322
    return incumbent, nli_window, nli_raw_window, nli_claim, nli_raw_claim, atomic_rows, evidence, names, claim_offsets


def fit():
    initialized = q.read(OUT / "initialized.json")
    assert initialized["source_sha256"] == source_hashes()
    assert initialized["protocol_sha256"] == sha(OUT / "protocol.json")
    assert q.read(OUT / "protocol.json") == protocol()
    assert not (OUT / "fit_complete.json").exists()
    started = time.perf_counter()
    meta = load_fit_meta()
    (incumbent, nli_window, nli_raw_window, nli_claim, nli_raw_claim,
     atomic_rows, evidence, names, claim_offsets) = load_sources()
    # First pass obtains fit-only incumbent cutoffs.  Second pass includes distance to cutoff.
    placeholder = build_partition(meta, "fit", incumbent, nli_claim, nli_raw_claim, evidence, names,
                                  atomic_rows, claim_offsets, incumbent_cutoff=0.)
    base_thresholds = fit_thresholds(placeholder, placeholder["incumbent"])
    view = build_partition(meta, "fit", incumbent, nli_claim, nli_raw_claim, evidence, names,
                           atomic_rows, claim_offsets, incumbent_cutoff=base_thresholds["window"]["threshold"])
    assert np.allclose(view["nli"], nli_window[:168123])
    assert np.allclose(view["raw_nli"], nli_raw_window[:168123])
    y = np.asarray([window["label"] for window in view["windows"]], dtype=np.int8)
    groups = np.asarray([window["group_id"] for window in view["windows"]])
    base_prediction = view["incumbent"] >= base_thresholds["window"]["threshold"]
    active = ~base_prediction
    indices = np.flatnonzero(active)
    folds = list(GroupKFold(FOLDS).split(indices, y[indices], groups[indices]))
    history = {}
    stored_scores = {}

    for candidate_index, name in enumerate(DETERMINISTIC):
        scores = view["atomic_scores"][name]
        thresholds = fit_thresholds(view, scores)
        metrics = metric_pair(view, scores, thresholds)
        history[name] = {
            "kind": "deterministic_atomic_propagation", "fit_thresholds": thresholds,
            "fit_metrics": metrics, "incremental": None,
            "selection_key": [min(metrics["windows"]["f1"], metrics["answers"]["f1"]),
                              metrics["windows"]["f1"], metrics["windows"]["precision"], 0., -candidate_index],
        }
        stored_scores[name] = scores

    weight = sample_weights(view, indices, y)
    for gate_offset, name in enumerate(GATES, start=len(DETERMINISTIC)):
        oof = np.full(len(y), -np.inf, dtype=np.float64)
        fold_log = []
        for fold, (train_local, held_local) in enumerate(folds):
            train, held = indices[train_local], indices[held_local]
            model = fit_model(make_model(name), model_x(name, view["x"])[train], y[train], weight[train])
            oof[held] = model.predict_proba(model_x(name, view["x"])[held])[:, 1]
            fold_log.append({"fold": fold, "train_groups": len(set(groups[train])),
                             "held_groups": len(set(groups[held])), "train_windows": len(train),
                             "held_windows": len(held)})
        assert np.isfinite(oof[active]).all()
        gate = choose_gate_threshold(y, base_prediction, active, oof)
        scores, hit = gated_score(view["incumbent"], oof, gate["threshold"],
                                  base_thresholds["window"]["threshold"],
                                  base_thresholds["answer"]["threshold"])
        metrics = metric_pair(view, scores, base_thresholds)
        assert metrics["windows"]["f1"] == gate["union_f1"]
        history[name] = {
            "kind": "source_group_OOF_specificity_gate", "fit_thresholds": base_thresholds,
            "gate_threshold": gate, "fit_metrics": metrics, "incremental": {
                "added": int(hit.sum()), "tp": int(np.count_nonzero(hit & (y == 1))),
                "fp": int(np.count_nonzero(hit & (y == 0))),
            }, "folds": fold_log,
            "selection_key": [min(metrics["windows"]["f1"], metrics["answers"]["f1"]),
                              metrics["windows"]["f1"], metrics["windows"]["precision"],
                              gate["incremental_precision"], -gate_offset],
        }
        stored_scores[name] = scores

    selected = max(CANDIDATES, key=lambda name: history[name]["selection_key"])
    model_payload = None
    if selected in GATES:
        model = fit_model(make_model(selected), model_x(selected, view["x"])[indices], y[indices], weight[indices])
        model_payload = {"candidate": selected, "model": model, "feature_names":
                         [ALL_FEATURES[i] for i in (RELATIVE_INDICES if selected == "gate_relative_lr" else range(len(ALL_FEATURES)))]}
        (OUT / "model.pkl").write_bytes(pickle.dumps(model_payload, protocol=5))

    selected_scores = stored_scores[selected]
    nli_threshold = q.read(NLI_RESULT)["thresholds_from_fit_OOF"]["window"]["threshold"]
    nli_pred = view["nli"] >= nli_threshold
    selected_pred = selected_scores >= history[selected]["fit_thresholds"]["window"]["threshold"]
    unique = (y == 1) & nli_pred & ~base_prediction
    unique_fp = (y == 0) & nli_pred & ~base_prediction
    np.savez_compressed(OUT / "fit_scores.npz", selected_scores=selected_scores,
                        incumbent_scores=view["incumbent"], nli_scores=view["nli"], labels=y)
    fit_record = {
        "status": "selected_and_frozen_before_calibration_report", "selected": selected,
        "candidates": history, "base_fit_thresholds": base_thresholds,
        "nli_fit_threshold": nli_threshold,
        "fit_nli_unique": {
            "tp": int(unique.sum()), "fp": int(unique_fp.sum()),
            "selected_retained_tp": int(np.count_nonzero(unique & selected_pred)),
            "selected_retained_fp": int(np.count_nonzero(unique_fp & selected_pred)),
        },
        "model_sha256": sha(OUT / "model.pkl") if model_payload else None,
        "fit_scores_sha256": sha(OUT / "fit_scores.npz"),
        "protocol_sha256": sha(OUT / "protocol.json"), "source_sha256": source_hashes(),
        "fit_answers": 634, "fit_windows": 168123, "fit_groups": len(set(groups)),
        "calibration_files_opened": False, "calibration_used_for_selection": False,
        "formal_baselines_modified": False, "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    save(OUT / "fit_complete.json", fit_record)
    print("SPECIFICITY_GATE_V1_FIT_FROZEN", selected,
          history[selected]["fit_metrics"]["windows"]["f1"],
          history[selected]["fit_metrics"]["answers"]["f1"])


def cal_report():
    frozen = q.read(OUT / "fit_complete.json")
    assert frozen["source_sha256"] == source_hashes()
    assert frozen["protocol_sha256"] == sha(OUT / "protocol.json")
    assert frozen["fit_scores_sha256"] == sha(OUT / "fit_scores.npz")
    if frozen["model_sha256"]:
        assert frozen["model_sha256"] == sha(OUT / "model.pkl")
    assert not (OUT / "summary.json").exists()
    save(OUT / "calibration_report_started.json", {
        "status": "selected_candidate_and_thresholds_already_frozen",
        "fit_complete_sha256": sha(OUT / "fit_complete.json"), "time": time.time(),
    })
    started = time.perf_counter()
    meta = q.metadata()
    (incumbent, nli_window, nli_raw_window, nli_claim, nli_raw_claim,
     atomic_rows, evidence, names, claim_offsets) = load_sources()
    selected = frozen["selected"]
    base_thresholds = frozen["base_fit_thresholds"]
    view = build_partition(meta, "calibration", incumbent, nli_claim, nli_raw_claim, evidence, names,
                           atomic_rows, claim_offsets, incumbent_cutoff=base_thresholds["window"]["threshold"])
    assert np.allclose(view["nli"], nli_window[168123:])
    assert np.allclose(view["raw_nli"], nli_raw_window[168123:])
    if selected in DETERMINISTIC:
        scores = view["atomic_scores"][selected]
        hit = None
    else:
        payload = pickle.loads((OUT / "model.pkl").read_bytes())
        assert payload["candidate"] == selected
        active = view["incumbent"] < base_thresholds["window"]["threshold"]
        probability = np.full(len(view["windows"]), -np.inf, dtype=np.float64)
        probability[active] = payload["model"].predict_proba(model_x(selected, view["x"])[active])[:, 1]
        scores, hit = gated_score(view["incumbent"], probability,
                                  frozen["candidates"][selected]["gate_threshold"]["threshold"],
                                  base_thresholds["window"]["threshold"],
                                  base_thresholds["answer"]["threshold"])
    thresholds = frozen["candidates"][selected]["fit_thresholds"]
    strict_cal = metric_pair(view, scores, thresholds)
    diagnostic_thresholds = fit_thresholds(view, scores)
    diagnostic_cal = metric_pair(view, scores, diagnostic_thresholds)
    y = np.asarray([window["label"] for window in view["windows"]], dtype=np.int8)
    base_pred = view["incumbent"] >= base_thresholds["window"]["threshold"]
    selected_pred = scores >= thresholds["window"]["threshold"]
    nli_pred = view["nli"] >= frozen["nli_fit_threshold"]
    unique = (y == 1) & nli_pred & ~base_pred
    unique_fp = (y == 0) & nli_pred & ~base_pred
    incremental = selected_pred & ~base_pred
    full_scores = np.concatenate((np.load(OUT / "fit_scores.npz", allow_pickle=False)["selected_scores"], scores))
    full_answers = q.answer_scores(meta, full_scores)
    np.savez_compressed(OUT / "scores.npz", window_scores=full_scores, answer_scores=full_answers)
    incumbent_reference = q.read(INC_RESULT)["metrics"]["calibration"]
    accepted = (strict_cal["windows"]["f1"] > incumbent_reference["windows"]["f1"] and
                strict_cal["answers"]["f1"] >= incumbent_reference["answers"]["f1"])
    summary = {
        "status": "strict_calibration_report_complete", "selected_fit_only": selected,
        "fit": frozen["candidates"][selected]["fit_metrics"],
        "thresholds_frozen_from_fit": thresholds,
        "strict_calibration": strict_cal,
        "calibration_F1Opt_diagnostic_not_strict": {
            "thresholds": diagnostic_thresholds, "metrics": diagnostic_cal,
        },
        "calibration_incremental": {
            "added": int(incremental.sum()), "tp": int(np.count_nonzero(incremental & (y == 1))),
            "fp": int(np.count_nonzero(incremental & (y == 0))),
            "precision": float(y[incremental].mean()) if incremental.any() else 0.,
        },
        "calibration_nli_unique": {
            "tp": int(unique.sum()), "fp": int(unique_fp.sum()),
            "selected_retained_tp": int(np.count_nonzero(unique & selected_pred)),
            "selected_retained_fp": int(np.count_nonzero(unique_fp & selected_pred)),
        },
        "incumbent_historical_read_only_reference": incumbent_reference,
        "accepted_as_replacement": accepted,
        "decision": "accept" if accepted else "reject_and_stop_this_candidate",
        "scores_sha256": sha(OUT / "scores.npz"), "fit_complete_sha256": sha(OUT / "fit_complete.json"),
        "calibration_used_for_selection": False, "calibration_opened_after_freeze": True,
        "formal_baselines_modified": False, "official_test_opened": False,
        "final_test_claim": False, "seconds": time.perf_counter() - started,
    }
    save(OUT / "summary.json", summary)
    w, a = strict_cal["windows"], strict_cal["answers"]
    dw, da = diagnostic_cal["windows"], diagnostic_cal["answers"]
    lines = [
        "# Specificity gate v1", "",
        "候选和全部门槛只由 fit 决定；下表 strict cal 是冻结后一次报告。cal-F1Opt 只作非独立诊断。", "",
        "| 方法 | fit窗口F1 | fit整答F1 | strict cal窗口F1 | strict cal整答F1 | cal-F1Opt窗口 | cal-F1Opt整答 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {selected} | {summary['fit']['windows']['f1']:.6f} | {summary['fit']['answers']['f1']:.6f} | {w['f1']:.6f} | {a['f1']:.6f} | {dw['f1']:.6f} | {da['f1']:.6f} |",
        "", f"strict cal 增量：TP {summary['calibration_incremental']['tp']}，FP {summary['calibration_incremental']['fp']}，精度 {summary['calibration_incremental']['precision']:.6f}。",
        f"NLI独有候选中保留 TP {summary['calibration_nli_unique']['selected_retained_tp']}/{summary['calibration_nli_unique']['tp']}，同时保留 FP {summary['calibration_nli_unique']['selected_retained_fp']}/{summary['calibration_nli_unique']['fp']}。",
        "", f"与历史 incumbent 0.690281/0.891089 比较：{'接受' if accepted else '不接受，停止该候选'}。",
        "", "限制：incumbent 上游本身曾使用 calibration 选择；本实验没有改它。新门控及本轮阈值只使用 fit，正式 baseline 与 official test 均未改动或打开。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(OUT / "complete.json", {
        "status": "complete_development_only", "summary_sha256": sha(OUT / "summary.json"),
        "report_sha256": sha(OUT / "REPORT.md"), "scores_sha256": sha(OUT / "scores.npz"),
        "accepted_as_replacement": accepted, "formal_baselines_modified": False,
        "official_test_opened": False, "final_test_claim": False,
    })
    print("SPECIFICITY_GATE_V1_CAL_COMPLETE", selected, w["f1"], a["f1"], "accepted", accepted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "fit", "cal-report"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"initialize": initialize, "fit": fit, "cal-report": cal_report}[args.stage]()
