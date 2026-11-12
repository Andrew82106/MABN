"""Pair-aligned evidence head over frozen atomic microclaim NLI pairs.

Every slot keeps the NLI distribution and all lexical relation fields from one
actual evidence/hypothesis pair.  Fit model/threshold selection is performed by
source-group OOF predictions.  Calibration is unavailable until ``evaluate``.
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
import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402


OUT = ROOT / "results/aligned_evidence_head_v1"
UPSTREAM = ROOT / "results/atomic_microclaim_nli_v1"
ROWS_PATH = UPSTREAM / "inputs.jsonl"
PAIR_DIR = UPSTREAM / "pair_scores"
WHITEBOX_PATH = UPSTREAM / "whitebox_features.npy"
INC_SCORE = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
INC_RESULT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"

SEED = 20261103
FOLDS = 5
THREADS = 4
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
INCUMBENT_REFERENCE_F1 = {"window": 0.6902813989031736, "answer": 0.8910891089108911}

SLOT_NAMES = (
    "individual_top_entailment",
    "individual_top_contradiction",
    "individual_top_neutral",
    "individual_top_bm25",
    "joint_top_entailment",
    "joint_top_contradiction",
)
SUPPORT_RELATIONS = (
    "claim_token_coverage", "subject_token_coverage", "entity_token_coverage",
    "predicate_token_match", "negation_match", "comparator_match",
    "quantity_any_exact", "quantity_all_exact", "temporal_match", "condition_match",
)
CONFLICT_RELATIONS = (
    "negation_mismatch", "comparator_reversal", "comparator_missing",
    "quantity_same_unit_different_value", "quantity_missing", "temporal_reversal",
    "temporal_missing", "condition_mismatch",
)

PAIR_CORE_NAMES = (
    "present", "entailment", "neutral", "contradiction", "contradiction_minus_entailment",
    "lack_of_entailment", "nli_entropy", "log1p_bm25", "query_coverage",
    "union_query_coverage", "bm25_gap", "reciprocal_rank", "is_joint",
    *atomic_nli.PAIR_RELATION_NAMES,
    "is_explicit_source", "is_parent_source", "is_cited_source",
    "passage_1", "passage_2", "passage_3", "log1p_evidence_words",
)
PAIR_INTERACTION_NAMES = (
    *(f"entailment_x_{name}" for name in SUPPORT_RELATIONS),
    *(f"contradiction_x_{name}" for name in CONFLICT_RELATIONS),
    "entailment_x_cited_source", "contradiction_x_cited_source",
    "lack_x_missing_claim_coverage", "neutral_x_missing_claim_coverage",
)
PAIR_SLOT_NAMES = PAIR_CORE_NAMES + PAIR_INTERACTION_NAMES
GLOBAL_NAMES = (
    "log1p_pair_count", "log1p_individual_pair_count", "log1p_joint_pair_count",
    "best_individual_entailment", "best_individual_contradiction", "best_individual_neutral",
    "best_joint_entailment", "best_joint_contradiction", "best_entailment_gap_individual_joint",
    "best_contradiction_gap_individual_joint",
)
INCUMBENT_NAMES = (
    "incumbent_claim_max", "incumbent_claim_q90", "incumbent_claim_q75",
    "incumbent_claim_mean", "incumbent_q75_minus_answer_median",
    "incumbent_q75_within_answer_rank",
)
FEATURE_NAMES = (
    tuple(atomic_nli.CLAIM_FEATURE_NAMES)
    + tuple(f"{slot}__{name}" for slot in SLOT_NAMES for name in PAIR_SLOT_NAMES)
    + GLOBAL_NAMES
    + tuple(f"whitebox__{name}" for name in atomic_nli.WHITEBOX_FEATURE_NAMES)
    + INCUMBENT_NAMES
)

CANDIDATES = (
    "any_error__lr",
    "any_error__hgb",
    "conflict_only__lr",
    "conflict_only__hgb",
)


def load_jsonl(path: Path, partition: str | None = None):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if partition is None or row.get("partition") == partition:
                rows.append(row)
    return rows


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_npz(path: Path, **arrays):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_paths():
    return (
        Path(__file__), Path(atomic_nli.__file__), Path(q.__file__), ROWS_PATH,
        UPSTREAM / "complete.json", UPSTREAM / "extraction_complete.json",
        WHITEBOX_PATH, INC_SCORE, INC_RESULT, ROOT / "data/gold_manifest.json",
    )


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in source_paths()}


def protocol():
    return {
        "version": "aligned-evidence-head-v1",
        "purpose": "Repair cross-sentence feature mismatch in the 315-wide atomic NLI aggregate.",
        "scope": "Original RAGTruth QA fit634/calibration159 development data only; official test is absent.",
        "frozen_inputs": {
            "microclaims": "Exact label-blind atomic_microclaim_nli_v1 geometry.",
            "nli": "Every actual frozen ModernBERT NLI pair probability; no GPU inference or parameter update.",
            "incumbent": "Read-only semantic_claim__old_tree__large_weight0.4 window scores.",
        },
        "pair_alignment": {
            "slots": list(SLOT_NAMES),
            "slot_width": len(PAIR_SLOT_NAMES),
            "rule": "One slot is copied from one actual evidence/hypothesis pair. E/N/C, BM25, subject/entity/predicate, negation, comparison, quantity, time, condition, and source features never come from different sentences inside a slot.",
            "selection": "Deterministic label-free argmax with stable pair-order ties: top E/C/N/BM25 individual pair and top E/C joint pair.",
            "same_pair_interactions": list(PAIR_INTERACTION_NAMES),
        },
        "readout": {
            "candidates": list(CANDIDATES),
            "models": {
                "lr": "StandardScaler plus L2 logistic regression C=0.01.",
                "hgb": "HistGradientBoosting: 7 leaves, 180 iterations, min leaf 40, L2=2.",
            },
            "targets": {
                "any_error": "Any unchanged human risk token in the microclaim.",
                "conflict_only": "Any unchanged Evident/Subtle Conflict span in the microclaim.",
            },
            "crossfit": "Five-fold GroupKFold by locked source-connected group. Every fit claim score is held-group OOF.",
            "weights": "Equal source-group mass, answer mass, claim mass, then binary class balance.",
        },
        "incumbent_add_gate": {
            "base_thresholds": "Window and answer cutoffs selected on fit only.",
            "gate_threshold": "For each OOF head, fit-only threshold maximizes union window F1; no-add is an allowed choice.",
            "score_mapping": "Added windows are placed above the fit window cutoff and below the fit answer cutoff, preserving base answer decisions.",
            "veto": "None in v1; therefore all base TP and FP are retained, and additions are reported separately.",
        },
        "selection": "Fit only: maximize min(window F1, answer F1), then window F1, precision, incremental precision, then earlier candidate.",
        "projection": "Claim score takes max over claims touched by each unchanged eligible 4-raw-BPE stride-one window.",
        "calibration": "The selected model and thresholds are frozen in fit_complete.json before calibration labels are opened. Calibration is reported once; no calibration F1 optimization is computed.",
        "candidate_count": len(CANDIDATES),
        "formal_baselines_modified": False,
        "GPU_used": False,
        "official_test_opened": False,
    }


def synthetic_selfcheck():
    claim = {
        "text": "Revenue did not decrease to 12 percent", "children_in_parent": 1,
        "child_index": 0, "antecedent_subject": None,
        "feature_vector": {name: False for name in atomic_nli.atomic.FEATURE_NAMES},
        "word_count": 7, "predicate_count": 1, "entity_candidates": [{"text": "Revenue"}],
        "negation_cues": [{"kind": atomic_nli.NEG_KINDS[0]}],
        "comparator_cues": [{"kind": "decrease", "operator": "down"}],
        "quantities": [{"values": ["12"], "unit": "percent"}],
        "temporal_order_cues": [], "condition_cues": [], "explicit_passage_ids": [2],
        "parent_passage_ids": [], "passage_context_inherited": False,
        "effective_subject": {"text": "Revenue"}, "predicate": {"text": "decrease"},
    }
    relation = atomic_nli.relation_pair_features(claim, "Revenue increased to 15 percent.")
    ri = {name: i for i, name in enumerate(atomic_nli.PAIR_RELATION_NAMES)}
    assert relation[ri["subject_token_coverage"]] == 1
    assert relation[ri["comparator_reversal"]] == 1
    assert relation[ri["quantity_same_unit_different_value"]] == 1
    assert len(SLOT_NAMES) == 6 and len(CANDIDATES) == 4
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    return {
        "passed": True, "feature_width": len(FEATURE_NAMES),
        "slot_width": len(PAIR_SLOT_NAMES), "slots": len(SLOT_NAMES),
        "candidate_count": len(CANDIDATES), "same_pair_relation_check": True,
        "GPU_used": False, "official_test_opened": False,
    }


def load_fit_meta():
    answers = q.lines(ROOT / "data/answers_fit.jsonl")
    tokens = q.lines(ROOT / "data/tokens_fit.jsonl")
    windows = q.lines(ROOT / "data/windows_k4_fit.jsonl")
    assert len(answers) == len(tokens) == 634 and len(windows) == EXPECTED_WINDOWS["fit"]
    assert [row["response_id"] for row in answers] == [row["response_id"] for row in tokens]
    by_response = {answer["response_id"]: {"answer": answer, "tokens": token}
                   for answer, token in zip(answers, tokens)}
    answer_windows = defaultdict(list)
    for index, window in enumerate(windows):
        answer_windows[window["response_id"]].append(index)
    return {"answers": answers, "tokens": tokens, "windows": windows,
            "by_response": by_response, "answer_windows": dict(answer_windows),
            "bounds": {"fit": [0, len(windows)]}}


def rank_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1)
    return result


def pair_metadata(claim, source, owner, premise, probability):
    _, passage_id, rank, sentence_a, sentence_b = owner
    selected = source["selected"]
    is_joint = rank == 0
    if is_joint:
        assert len(selected) == 2 and sentence_a == selected[0]["sentence_id"] and sentence_b == selected[1]["sentence_id"]
        bm25 = max(float(item["bm25"]) for item in selected)
        query_coverage = float(source["selected_union_query_coverage"])
        reciprocal_rank = 1.0
    else:
        item = next(item for item in selected if int(item["rank"]) == rank)
        assert sentence_a == item["sentence_id"] and sentence_b == -1
        bm25 = float(item["bm25"])
        query_coverage = float(item["query_term_coverage"])
        reciprocal_rank = 1.0 / rank
    top_bm25 = float(selected[0]["bm25"])
    second_bm25 = float(selected[-1]["bm25"])
    relation = atomic_nli.relation_pair_features(claim, premise).astype(np.float64)
    e, n, c = map(float, probability)
    entropy = -sum(value * np.log(max(value, 1e-12)) for value in (e, n, c)) / np.log(3)
    cited = passage_id in set(claim["explicit_passage_ids"] or claim["parent_passage_ids"])
    values = [
        1.0, e, n, c, c - e, 1 - e, entropy, np.log1p(max(0.0, bm25)),
        query_coverage, float(source["selected_union_query_coverage"]), top_bm25 - second_bm25,
        reciprocal_rank, float(is_joint), *relation.tolist(),
        float(passage_id in claim["explicit_passage_ids"]),
        float(passage_id in claim["parent_passage_ids"]), float(cited),
        *(float(passage_id == pid) for pid in (1, 2, 3)), np.log1p(len(atomic_nli.WORD.findall(premise))),
    ]
    core = dict(zip(PAIR_CORE_NAMES, values))
    interactions = [
        *(e * core[name] for name in SUPPORT_RELATIONS),
        *(c * core[name] for name in CONFLICT_RELATIONS),
        e * core["is_cited_source"], c * core["is_cited_source"],
        (1 - e) * (1 - core["claim_token_coverage"]),
        n * (1 - core["claim_token_coverage"]),
    ]
    result = np.asarray(values + interactions, dtype=np.float32)
    assert result.shape == (len(PAIR_SLOT_NAMES),) and np.isfinite(result).all()
    return result


def select_slots(claim, row, probability, pairs=None, owners=None):
    if pairs is None or owners is None:
        pairs, owners = atomic_nli.row_pairs(row)
    indices = [i for i, owner in enumerate(owners) if owner[0] == claim["claim_id"]]
    assert indices
    sources = {source["passage_id"]: source for source in claim["retrieval"]}
    records = []
    for index in indices:
        owner = owners[index]
        source = sources[owner[1]]
        vector = pair_metadata(claim, source, owner, pairs[index][0], probability[index])
        core = dict(zip(PAIR_CORE_NAMES, vector[:len(PAIR_CORE_NAMES)]))
        records.append({"index": index, "joint": owner[2] == 0, "vector": vector,
                        "e": core["entailment"], "n": core["neutral"], "c": core["contradiction"],
                        "bm25": core["log1p_bm25"]})
    individual = [record for record in records if not record["joint"]]
    joint = [record for record in records if record["joint"]]
    assert individual

    def best(pool, field):
        if not pool:
            return np.zeros(len(PAIR_SLOT_NAMES), dtype=np.float32)
        # Earlier actual-pair order is the final deterministic tie break.
        chosen = max(pool, key=lambda record: (record[field], -record["index"]))
        return chosen["vector"]

    slots = [best(individual, "e"), best(individual, "c"), best(individual, "n"),
             best(individual, "bm25"), best(joint, "e"), best(joint, "c")]
    global_values = [
        np.log1p(len(records)), np.log1p(len(individual)), np.log1p(len(joint)),
        max(record["e"] for record in individual), max(record["c"] for record in individual),
        max(record["n"] for record in individual), max((record["e"] for record in joint), default=0.0),
        max((record["c"] for record in joint), default=0.0),
        max(record["e"] for record in individual) - max((record["e"] for record in joint), default=0.0),
        max(record["c"] for record in individual) - max((record["c"] for record in joint), default=0.0),
    ]
    return np.concatenate((*slots, np.asarray(global_values, dtype=np.float32))), float(global_values[3]), float(global_values[4])


def claim_incumbent_aggregates(meta, partition, row, incumbent, window_offset=0):
    values = [[] for _ in row["claims"]]
    owners = row["lexical_token_microclaims"]
    for wi in meta["answer_windows"][row["response_id"]]:
        global_wi = window_offset + wi
        cids = {int(cid) for ti in meta["windows"][wi]["token_indices"] for cid in owners[ti]}
        for cid in cids:
            values[cid].append(float(incumbent[global_wi]))
    assert all(values)
    return np.asarray([[max(one), np.quantile(one, .90), np.quantile(one, .75), np.mean(one)]
                       for one in values], dtype=np.float32)


def build_claim_matrix(meta, partition, atomic_rows, incumbent, whitebox, whitebox_offset, window_offset=0):
    feature_rows, response_ids, groups, top_e, top_c = [], [], [], [], []
    incumbent_rows = []
    cache_checks = {}
    wb_cursor = whitebox_offset
    for answer_index, row in enumerate(atomic_rows):
        probability, _ = atomic_nli.validate_cache(PAIR_DIR / f"{row['response_id']}.npz", row)
        pairs, owners = atomic_nli.row_pairs(row)
        cache_checks[row["response_id"]] = sha(PAIR_DIR / f"{row['response_id']}.npz")
        inc = claim_incumbent_aggregates(meta, partition, row, incumbent, window_offset)
        local_pair, local_self, local_e, local_c = [], [], [], []
        for claim in row["claims"]:
            pair_vector, one_e, one_c = select_slots(claim, row, probability, pairs, owners)
            local_pair.append(pair_vector); local_self.append(atomic_nli.claim_self_features(claim))
            local_e.append(one_e); local_c.append(one_c)
        local_pair = np.asarray(local_pair, dtype=np.float32)
        local_self = np.asarray(local_self, dtype=np.float32)
        local_e = np.asarray(local_e, dtype=np.float32); local_c = np.asarray(local_c, dtype=np.float32)
        one_wb = np.asarray(whitebox[wb_cursor:wb_cursor + len(row["claims"])], dtype=np.float32)
        wb_cursor += len(row["claims"])
        ranks = rank_fraction(inc[:, 2]).astype(np.float32)
        inc_extra = np.column_stack((inc, inc[:, 2] - np.median(inc[:, 2]), ranks)).astype(np.float32)
        matrix = np.column_stack((local_self, local_pair, one_wb, inc_extra)).astype(np.float32)
        assert matrix.shape == (len(row["claims"]), len(FEATURE_NAMES))
        feature_rows.append(matrix); incumbent_rows.append(inc)
        response_ids.extend([row["response_id"]] * len(row["claims"]))
        groups.extend([row["group_id"]] * len(row["claims"]))
        top_e.extend(local_e); top_c.extend(local_c)
        if (answer_index + 1) % 100 == 0:
            print("ALIGNED_FEATURE_ROWS", partition, answer_index + 1, len(atomic_rows), flush=True)
    result = np.vstack(feature_rows)
    assert result.shape == (EXPECTED_CLAIMS[partition], len(FEATURE_NAMES)) and np.isfinite(result).all()
    return {
        "x": result, "response_ids": np.asarray(response_ids), "groups": np.asarray(groups),
        "top_entailment": np.asarray(top_e), "top_contradiction": np.asarray(top_c),
        "incumbent_claim": np.vstack(incumbent_rows), "cache_sha256": cache_checks,
        "whitebox_stop": wb_cursor,
    }


def labels_for_claims(meta, atomic_rows):
    any_labels, conflict_labels = [], []
    for row in atomic_rows:
        token = meta["by_response"][row["response_id"]]["tokens"]
        risk = np.asarray(token["risk_mask"], dtype=bool)
        offsets = token["response_token_offsets"]
        conflict_mask = np.zeros(len(risk), dtype=bool)
        for label in token["original_labels"]:
            if "Conflict" not in label["label_type"]:
                continue
            left, right = int(label["start"]), int(label["end"])
            for ti, (a, b) in enumerate(offsets):
                if max(a, left) < min(b, right):
                    conflict_mask[ti] = True
        # Frozen risk_mask intentionally excludes punctuation-only BPEs even when
        # their character interval touches an annotated span.  Claims contain
        # lexical BPEs only; require exact agreement on that evaluated support.
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        assert np.all(~conflict_mask | risk | ~lexical)
        for claim in row["claims"]:
            ids = claim["lexical_token_indices"]
            any_labels.append(int(risk[ids].any()))
            conflict_labels.append(int(conflict_mask[ids].any()))
    any_y = np.asarray(any_labels, dtype=np.int8)
    conflict_y = np.asarray(conflict_labels, dtype=np.int8)
    assert len(any_y) == EXPECTED_CLAIMS[atomic_rows[0]["partition"]]
    assert np.all(conflict_y <= any_y) and any_y.sum() > conflict_y.sum() > 0
    return any_y, conflict_y


def claim_weights(rows_by_id, response_ids, labels, active):
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        rid = response_ids[index]
        tree[rows_by_id[rid]["group_id"]][rid].append(index)
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
        indices = [i for values in answers.values() for i in values]
        weights[indices] *= (mask.sum() / len(tree)) / weights[indices].sum()
    weights[mask] *= mask.sum() / weights[mask].sum()
    return weights


def make_model(candidate):
    if candidate.endswith("__lr"):
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.01, solver="liblinear", penalty="l2", max_iter=3000, random_state=SEED))
    return HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=180, max_leaf_nodes=7, min_samples_leaf=40,
        l2_regularization=2., early_stopping=False, random_state=SEED)


def fit_model(model, x, y, weights):
    key = "logisticregression__sample_weight" if hasattr(model, "steps") else "sample_weight"
    model.fit(x, y, **{key: weights})
    return model


def project_claims(meta, atomic_rows, claim_scores):
    output = np.empty(len(meta["windows"]), dtype=np.float64)
    cursor = 0
    for row in atomic_rows:
        owners = row["lexical_token_microclaims"]
        local = claim_scores[cursor:cursor + len(row["claims"])]
        cursor += len(row["claims"])
        for wi in meta["answer_windows"][row["response_id"]]:
            cids = {int(cid) for ti in meta["windows"][wi]["token_indices"] for cid in owners[ti]}
            assert cids
            output[wi] = max(local[cid] for cid in cids)
    assert cursor == len(claim_scores) and np.isfinite(output).all()
    return output


def answer_scores(meta, scores):
    return np.asarray([scores[meta["answer_windows"][answer["response_id"]]].max()
                       for answer in meta["answers"]], dtype=np.float64)


def fit_thresholds(meta, scores):
    return {
        "window": q.choose_threshold([row["label"] for row in meta["windows"]], scores),
        "answer": q.choose_threshold([row["label"] for row in meta["answers"]], answer_scores(meta, scores)),
    }


def metric_pair(meta, scores, thresholds):
    return {
        "windows": q.count([row["label"] for row in meta["windows"]], scores, thresholds["window"]["threshold"]),
        "answers": q.count([row["label"] for row in meta["answers"]], answer_scores(meta, scores), thresholds["answer"]["threshold"]),
    }


def choose_add_threshold(labels, base_prediction, head_scores):
    active = np.flatnonzero(~base_prediction)
    order = np.argsort(-head_scores[active], kind="stable")
    ids = active[order]
    scores = head_scores[ids]
    last = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    add_tp = np.cumsum(labels[ids])[last]
    additions = last + 1
    add_fp = additions - add_tp
    base_tp = int(np.count_nonzero(base_prediction & (labels == 1)))
    base_fp = int(np.count_nonzero(base_prediction & (labels == 0)))
    positives = int(labels.sum())
    f1 = 2 * (base_tp + add_tp) / (2 * (base_tp + add_tp) + base_fp + add_fp + positives - base_tp - add_tp)
    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + positives - base_tp)
    options = [(base_f1, 1.0, 0, float(np.nextafter(scores[0], np.inf)), 0, 0)]
    for j in range(len(last)):
        precision = float(add_tp[j] / additions[j])
        options.append((float(f1[j]), precision, -int(additions[j]), float(scores[last[j]]),
                        int(add_tp[j]), int(add_fp[j])))
    best = max(options, key=lambda item: item[:3] + (item[3],))
    return {"threshold": best[3], "union_f1": best[0], "incremental_precision": best[1],
            "additions": -best[2], "tp": best[4], "fp": best[5]}


def gated_score(incumbent, head_score, gate_threshold, base_thresholds):
    window_cut = float(base_thresholds["window"]["threshold"])
    answer_cut = float(base_thresholds["answer"]["threshold"])
    assert answer_cut > window_cut
    hit = (incumbent < window_cut) & (head_score >= gate_threshold)
    result = incumbent.astype(np.float64, copy=True)
    if hit.any():
        selected = head_score[hit]
        span = max(float(selected.max() - gate_threshold), np.finfo(np.float64).eps)
        fraction = np.clip((selected - gate_threshold) / span, 0, 1)
        upper = np.nextafter(answer_cut, -np.inf)
        result[hit] = window_cut + (upper - window_cut) * (.25 + .74 * fraction)
    assert np.array_equal(result >= window_cut, (incumbent >= window_cut) | hit)
    return result, hit


def initialize():
    assert not OUT.exists(), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True)
    write_json(OUT / "protocol.json", protocol())
    write_json(OUT / "initialized.json", {
        "status": "preregistered_before_fit", "protocol_sha256": sha(OUT / "protocol.json"),
        "source_sha256": source_hashes(), "selfcheck": synthetic_selfcheck(),
        "calibration_labels_opened": False, "GPU_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("ALIGNED_EVIDENCE_HEAD_V1_INITIALIZED", flush=True)


def check_initialized():
    initialized = q.read(OUT / "initialized.json")
    assert initialized["protocol_sha256"] == sha(OUT / "protocol.json")
    assert initialized["source_sha256"] == source_hashes()
    assert q.read(OUT / "protocol.json") == protocol()
    assert initialized["selfcheck"] == synthetic_selfcheck()
    return initialized


def fit():
    check_initialized()
    assert not (OUT / "fit_complete.json").exists()
    started = time.perf_counter()
    meta = load_fit_meta()
    rows = load_jsonl(ROWS_PATH, "fit")
    assert len(rows) == 634 and sum(len(row["claims"]) for row in rows) == EXPECTED_CLAIMS["fit"]
    incumbent_all = np.load(INC_SCORE, allow_pickle=False)["window_scores"].astype(np.float64)
    incumbent = incumbent_all[:EXPECTED_WINDOWS["fit"]]
    whitebox = np.load(WHITEBOX_PATH, mmap_mode="r")
    built = build_claim_matrix(meta, "fit", rows, incumbent_all, whitebox, 0, 0)
    assert built["whitebox_stop"] == EXPECTED_CLAIMS["fit"]
    any_y, conflict_y = labels_for_claims(meta, rows)
    window_y = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    base_thresholds = fit_thresholds(meta, incumbent)
    base_prediction = incumbent >= base_thresholds["window"]["threshold"]
    row_by_id = {row["response_id"]: row for row in rows}
    indices = np.arange(len(any_y))
    folds = list(GroupKFold(FOLDS).split(indices, any_y, built["groups"]))
    history, stored = {}, {}

    with threadpool_limits(limits=THREADS):
        for candidate_index, candidate in enumerate(CANDIDATES):
            target = any_y if candidate.startswith("any_error") else conflict_y
            oof = np.full(len(target), np.nan, dtype=np.float64)
            fold_log = []
            for fold, (train, held) in enumerate(folds):
                weights = claim_weights(row_by_id, built["response_ids"], target, train)
                model = fit_model(make_model(candidate), built["x"][train], target[train], weights[train])
                oof[held] = model.predict_proba(built["x"][held])[:, 1]
                fold_log.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                                 "train_groups": len(set(built["groups"][train])),
                                 "held_groups": len(set(built["groups"][held])),
                                 "train_positive": int(target[train].sum()),
                                 "held_positive": int(target[held].sum())})
            assert np.isfinite(oof).all()
            head_window = project_claims(meta, rows, oof)
            gate = choose_add_threshold(window_y, base_prediction, head_window)
            scores, hit = gated_score(incumbent, head_window, gate["threshold"], base_thresholds)
            metrics = metric_pair(meta, scores, base_thresholds)
            assert abs(metrics["windows"]["f1"] - gate["union_f1"]) < 1e-15
            history[candidate] = {
                "target": "any_error" if candidate.startswith("any_error") else "conflict_only",
                "model": "lr" if candidate.endswith("__lr") else "hgb",
                "fit_OOF_gate": gate, "fit_OOF_metrics": metrics,
                "incremental": {"added": int(hit.sum()),
                                "tp": int(np.count_nonzero(hit & (window_y == 1))),
                                "fp": int(np.count_nonzero(hit & (window_y == 0)))},
                "folds": fold_log,
                "selection_key": [min(metrics["windows"]["f1"], metrics["answers"]["f1"]),
                                  metrics["windows"]["f1"], metrics["windows"]["precision"],
                                  gate["incremental_precision"], -candidate_index],
            }
            stored[candidate] = {"claim": oof, "head_window": head_window, "scores": scores}
            print("ALIGNED_EVIDENCE_OOF", candidate, round(metrics["windows"]["f1"], 6),
                  round(metrics["answers"]["f1"], 6), gate["tp"], gate["fp"], flush=True)

    selected = max(CANDIDATES, key=lambda name: history[name]["selection_key"])
    target = any_y if selected.startswith("any_error") else conflict_y
    weights = claim_weights(row_by_id, built["response_ids"], target, indices)
    with threadpool_limits(limits=THREADS):
        model = fit_model(make_model(selected), built["x"], target, weights)
    model_path = OUT / "model.pkl"
    model_path.write_bytes(pickle.dumps({"candidate": selected, "model": model,
                                         "feature_names": FEATURE_NAMES}, protocol=5))
    write_npz(OUT / "fit_scores.npz",
              selected_claim_oof=stored[selected]["claim"],
              selected_head_window_oof=stored[selected]["head_window"],
              selected_gated_window_oof=stored[selected]["scores"],
              incumbent_window=incumbent,
              any_claim_labels=any_y, conflict_claim_labels=conflict_y)
    fit_complete = {
        "status": "selected_and_frozen_before_calibration", "selected": selected,
        "candidates": history, "base_fit_thresholds": base_thresholds,
        "base_fit_metrics": metric_pair(meta, incumbent, base_thresholds),
        "claim_counts": {"all": len(any_y), "any_error": int(any_y.sum()),
                         "conflict_only": int(conflict_y.sum())},
        "feature_width": len(FEATURE_NAMES), "slot_width": len(PAIR_SLOT_NAMES),
        "feature_names_sha256": hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest(),
        "model_sha256": sha(model_path), "fit_scores_sha256": sha(OUT / "fit_scores.npz"),
        "cache_sha256": built["cache_sha256"], "source_sha256": source_hashes(),
        "calibration_labels_opened": False, "fit_only_selection": True,
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False, "seconds": time.perf_counter() - started,
    }
    write_json(OUT / "fit_complete.json", fit_complete)
    print("ALIGNED_EVIDENCE_FIT_FROZEN", selected, flush=True)


def evaluate():
    check_initialized()
    fit_done = q.read(OUT / "fit_complete.json")
    assert fit_done["source_sha256"] == source_hashes()
    assert fit_done["model_sha256"] == sha(OUT / "model.pkl")
    assert fit_done["fit_scores_sha256"] == sha(OUT / "fit_scores.npz")
    assert not (OUT / "summary.json").exists()
    started = time.perf_counter()
    meta_all = q.metadata()
    left, right = meta_all["bounds"]["calibration"]
    assert (left, right) == (EXPECTED_WINDOWS["fit"], sum(EXPECTED_WINDOWS.values()))
    cal_answers = [answer for answer in meta_all["answers"] if answer["partition"] == "calibration"]
    cal_tokens = [meta_all["by_response"][answer["response_id"]]["tokens"] for answer in cal_answers]
    cal_windows = meta_all["windows"][left:right]
    cal_answer_windows = {
        answer["response_id"]: [wi - left for wi in meta_all["answer_windows"][answer["response_id"]]]
        for answer in cal_answers
    }
    cal_meta = {"answers": cal_answers, "tokens": cal_tokens, "windows": cal_windows,
                "by_response": {answer["response_id"]: meta_all["by_response"][answer["response_id"]]
                                for answer in cal_answers},
                "answer_windows": cal_answer_windows, "bounds": {"calibration": [0, len(cal_windows)]}}
    rows = load_jsonl(ROWS_PATH, "calibration")
    assert len(rows) == 159 and sum(len(row["claims"]) for row in rows) == EXPECTED_CLAIMS["calibration"]
    incumbent_all = np.load(INC_SCORE, allow_pickle=False)["window_scores"].astype(np.float64)
    incumbent_cal = incumbent_all[left:right]
    whitebox = np.load(WHITEBOX_PATH, mmap_mode="r")
    built = build_claim_matrix(cal_meta, "calibration", rows, incumbent_all, whitebox,
                               EXPECTED_CLAIMS["fit"], left)
    assert built["whitebox_stop"] == sum(EXPECTED_CLAIMS.values())
    payload = pickle.loads((OUT / "model.pkl").read_bytes())
    assert payload["candidate"] == fit_done["selected"] and tuple(payload["feature_names"]) == FEATURE_NAMES
    claim_scores = payload["model"].predict_proba(built["x"])[:, 1]
    head_window = project_claims(cal_meta, rows, claim_scores)
    gate_threshold = fit_done["candidates"][fit_done["selected"]]["fit_OOF_gate"]["threshold"]
    scores, hit = gated_score(incumbent_cal, head_window, gate_threshold, fit_done["base_fit_thresholds"])
    metrics_cal = metric_pair(cal_meta, scores, fit_done["base_fit_thresholds"])
    base_cal = metric_pair(cal_meta, incumbent_cal, fit_done["base_fit_thresholds"])
    wy = np.asarray([row["label"] for row in cal_windows], dtype=np.int8)
    selected_pred = scores >= fit_done["base_fit_thresholds"]["window"]["threshold"]
    base_pred = incumbent_cal >= fit_done["base_fit_thresholds"]["window"]["threshold"]
    assert np.all(~base_pred | selected_pred)
    additions = {
        "added": int(hit.sum()), "added_tp": int(np.count_nonzero(hit & (wy == 1))),
        "added_fp": int(np.count_nonzero(hit & (wy == 0))),
        "incremental_precision": float(wy[hit].mean()) if hit.any() else 0.0,
        "base_tp_retained": int(np.count_nonzero(base_pred & (wy == 1) & selected_pred)),
        "base_tp_total": int(np.count_nonzero(base_pred & (wy == 1))),
        "base_fp_retained": int(np.count_nonzero(base_pred & (wy == 0) & selected_pred)),
        "base_fp_total": int(np.count_nonzero(base_pred & (wy == 0))),
        "base_fp_removed": int(np.count_nonzero(base_pred & (wy == 0) & ~selected_pred)),
    }
    write_npz(OUT / "scores.npz", claim_scores=claim_scores, head_window_scores=head_window,
              gated_window_scores=scores, answer_scores=answer_scores(cal_meta, scores),
              incumbent_window_scores=incumbent_cal, additions=hit.astype(np.int8))
    summary = {
        "status": "development_only_complete", "selected_fit_only": fit_done["selected"],
        "fit_OOF": fit_done["candidates"][fit_done["selected"]]["fit_OOF_metrics"],
        "strict_calibration": metrics_cal, "strict_calibration_base_same_fit_thresholds": base_cal,
        "strict_calibration_additions": additions,
        "thresholds_from_fit_only": fit_done["base_fit_thresholds"],
        "gate_threshold_from_fit_OOF_only": gate_threshold,
        "historical_cal_selected_incumbent_read_only": q.read(INC_RESULT)["metrics"]["calibration"],
        "acceptance": {
            "beats_historical_window_F1": metrics_cal["windows"]["f1"] > INCUMBENT_REFERENCE_F1["window"],
            "preserves_historical_answer_F1": metrics_cal["answers"]["f1"] >= INCUMBENT_REFERENCE_F1["answer"],
            "accepted_as_incumbent_replacement": metrics_cal["windows"]["f1"] > INCUMBENT_REFERENCE_F1["window"] and metrics_cal["answers"]["f1"] >= INCUMBENT_REFERENCE_F1["answer"],
        },
        "calibration_used_for_selection": False, "calibration_F1Opt_computed": False,
        "calibration_evaluations": 1, "scores_sha256": sha(OUT / "scores.npz"),
        "source_sha256": source_hashes(), "GPU_used": False,
        "formal_baselines_modified": False, "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    write_json(OUT / "summary.json", summary)
    fm = fit_done["candidates"][fit_done["selected"]]["fit_OOF_metrics"]
    cm = metrics_cal
    report = [
        "# Aligned evidence head v1", "",
        "每个槽位只来自一个真实证据 pair；E/N/C 与 BM25、关系和来源特征不再跨句拼接。4 个 LR/HGB 候选和全部门槛只由 source-group fit OOF 决定；cal 只严格评测一次。", "",
        "| 选中候选 | fit OOF窗口F1 | fit OOF整答F1 | strict cal窗口F1 | strict cal整答F1 |",
        "|---|---:|---:|---:|---:|",
        f"| {fit_done['selected']} | {fm['windows']['f1']:.6f} | {fm['answers']['f1']:.6f} | {cm['windows']['f1']:.6f} | {cm['answers']['f1']:.6f} |",
        "",
        f"strict cal 相对同一 fit 门槛 incumbent：新增 TP {additions['added_tp']}、新增 FP {additions['added_fp']}，增量精度 {additions['incremental_precision']:.6f}。由于 v1 只做 add gate，原 TP 保留 {additions['base_tp_retained']}/{additions['base_tp_total']}，去除 FP {additions['base_fp_removed']}。",
        "",
        f"同一 fit 门槛 base strict cal 为 {base_cal['windows']['f1']:.6f}/{base_cal['answers']['f1']:.6f}；历史 cal 选型 incumbent 为 {INCUMBENT_REFERENCE_F1['window']:.6f}/{INCUMBENT_REFERENCE_F1['answer']:.6f}。",
        "",
        "未计算 cal-F1Opt；未用 GPU、未打开 official test、未改正式 baseline。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "complete.json", {
        "status": "complete_development_only", "summary_sha256": sha(OUT / "summary.json"),
        "report_sha256": sha(OUT / "REPORT.md"), "scores_sha256": sha(OUT / "scores.npz"),
        "fit_complete_sha256": sha(OUT / "fit_complete.json"), "model_sha256": sha(OUT / "model.pkl"),
        "calibration_used_for_selection": False, "calibration_F1Opt_computed": False,
        "GPU_used": False, "formal_baselines_modified": False, "official_test_opened": False,
    })
    print("ALIGNED_EVIDENCE_EVALUATION_COMPLETE", cm["windows"]["f1"], cm["answers"]["f1"], flush=True)


def check():
    check_initialized()
    if (OUT / "fit_complete.json").exists():
        fit_done = q.read(OUT / "fit_complete.json")
        assert fit_done["model_sha256"] == sha(OUT / "model.pkl")
        assert fit_done["fit_scores_sha256"] == sha(OUT / "fit_scores.npz")
        assert fit_done["selected"] in CANDIDATES
    if (OUT / "complete.json").exists():
        done = q.read(OUT / "complete.json")
        for name in ("summary", "report", "scores", "fit_complete", "model"):
            assert done[f"{name}_sha256"] == sha(OUT / ({"summary": "summary.json", "report": "REPORT.md", "scores": "scores.npz", "fit_complete": "fit_complete.json", "model": "model.pkl"}[name]))
        assert not done["calibration_used_for_selection"] and not done["calibration_F1Opt_computed"]
        assert not done["GPU_used"] and not done["formal_baselines_modified"] and not done["official_test_opened"]
    print("ALIGNED_EVIDENCE_HEAD_V1_CHECKED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("self-test", "initialize", "fit", "evaluate", "check"))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if args.stage == "self-test":
            print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif args.stage == "initialize":
            initialize()
        elif args.stage == "fit":
            fit()
        elif args.stage == "evaluate":
            evaluate()
        else:
            check()
