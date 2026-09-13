"""Citation-conditioned CPU readout over frozen retrieved-evidence NLI v1.

This candidate never changes retrieval, NLI pairs, checkpoint, probabilities,
or the formal baselines.  Preparation and checks are CPU-only and do not need
the probability cache to exist.  ``score`` becomes available only after v1
has completed frozen extraction; it trains on fit groups and reports
calibration without selecting on it.  Official test paths are never used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_development as q
import run_retrieved_evidence_nli_v1 as v1


ROOT = q.ROOT
OUT = ROOT / "results/retrieved_evidence_nli_citation_v2"
V1_OUT = ROOT / "results/retrieved_evidence_nli_v1"
CURRENT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
CURRENT_META = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"

ROLE = "ours_method_candidate; never a baseline"
FOLDS = 5
LR_C = 0.1
SEED = 20261014
FUSION_WEIGHT_CITATION = 0.5
FUSION_WEIGHT_WHITEBOX = 0.5
LOGIT_EPSILON = 1e-6

EXPECTED = {
    "v1_runner": "353129f81e194b87542ec60aa74a6db012cb437108384805ceaa1dd13d1bd64a",
    "v1_protocol": "a7e55e74cf87e734b6948f49aae47bc8ec8e8418fc742e81c797412598c56c52",
    "v1_inputs": "aaad043a5ecae7ff881758cc7b0717b2771168c8473a3f84096045f7fe151408",
    "v1_preparation": "4f5f34fd6404c33a03c46b52b6a8abc331a404143f36498bee1e16bf4cd334ec",
    "current_scores": "64b3972dd999c9655a302ad04a8709a1866b169a141e6167042e159bf601645f",
    "current_metadata": "c837e8fb289985d7d1579e51d77e7302348970b7665a50a704e3e1ca7dc46db6",
}


def citation_feature_names():
    names = [
        "citation_has_recognized_reference",
        "citation_has_valid_source",
        "citation_has_invalid_source",
        "citation_has_unknown_syntax",
        "citation_valid_source_count_div3",
        "citation_invalid_source_ratio",
        "citation_source_1",
        "citation_source_2",
        "citation_source_3",
    ]
    aggregate = (
        "max_entailment", "mean_entailment", "max_neutral", "mean_neutral",
        "max_contradiction", "mean_contradiction", "max_coverage", "mean_coverage",
        "max_bm25", "mean_bm25",
    )
    names += [f"cited_{name}" for name in aggregate]
    names += [f"uncited_{name}" for name in aggregate]
    names += [
        "cited_minus_uncited_max_entailment",
        "cited_minus_uncited_mean_entailment",
        "cited_minus_uncited_max_contradiction",
        "cited_minus_uncited_mean_contradiction",
        "citation_support_gap_all_minus_cited_max_entailment",
        "citation_conditioned_or_risk",
    ]
    return names


CITATION_FEATURE_NAMES = tuple(citation_feature_names())
FEATURE_NAMES = tuple(v1.FEATURE_NAMES) + CITATION_FEATURE_NAMES


def protocol():
    return {
        "version": "native-qa-retrieved-evidence-nli-citation-v2",
        "status": "frozen_before_v1_probability_cache_and_any_v2_fit",
        "role": ROLE,
        "upstream": {
            "policy": "Read-only reuse of v1 inputs, claim/evidence provenance, fixed BM25 selection and frozen ModernBERT E/N/C cache. Never rerun or alter retrieval/model inference.",
            "expected_sha256": EXPECTED,
            "expected_answers": 793,
            "expected_scored_claims": 8845,
            "expected_pairs": 51953,
        },
        "citation_interpretation": {
            "source_ids": "Use v1's already frozen citation_parse. Valid source ids are exactly 1,2,3; every other recognized id is invalid. Unique ids are used for counts.",
            "status": "Nine fixed values: recognized/valid/invalid/unknown flags, valid count divided by3, invalid unique-id ratio, and three valid source indicators.",
            "cited": "Across evidence pairs belonging to valid cited passages: max/mean E,N,C; max/mean selected-union query coverage by passage; max/mean selected-pair BM25.",
            "uncited_control": "The same ten values across passages not validly cited. If there is no valid citation, all three passages are the uncited control.",
            "empty_set": "An empty cited or uncited set maps to ten zeros; status/source flags distinguish this deterministic sentinel.",
            "contrasts": "Only when at least one valid source is cited: cited-minus-uncited max/mean E and C (empty uncited is zero), all-source maxE minus cited maxE, and max(cited maxC, support gap). Otherwise all six are zero.",
            "no_rewrite": "Citation markers remain in the exact claim hypothesis and original v1 retrieval query. v2 only conditions the readout; it does not alter pair text or scores.",
        },
        "features": {
            "base_v1": list(v1.FEATURE_NAMES),
            "citation": list(CITATION_FEATURE_NAMES),
            "width": len(FEATURE_NAMES),
        },
        "readout": {
            "model": "One StandardScaler and L2 liblinear logistic regression, fixed C=0.1 and seed. No feature/C/model search.",
            "label": "A claim is positive iff any assigned original lexical BPE has the unchanged fit human risk label.",
            "crossfit": "Five-fold GroupKFold by locked source-connected group gives every fit claim score. Full-fit model predicts calibration claims.",
            "weights": "Within each training fold: equal group mass, then equal answer mass, then equal claim mass; fit labels provide binary class balancing.",
            "projection": "Reuse v1 exact claim-to-original-BPE mapping; a unified four-BPE window takes max over its lexical claims; answer takes max over windows.",
            "threshold": "Window and answer thresholds are selected separately on fit only; learned fit values are group-OOF. Calibration is evaluated only after scores and fit thresholds are fixed.",
        },
        "fixed_fusion": {
            "formula": "sigmoid(0.5*logit(clamp(citation_probability,1e-6,1-1e-6)) + 0.5*logit(clamp(current_whitebox_probability,1e-6,1-1e-6))).",
            "weights": {"citation": FUSION_WEIGHT_CITATION, "current_whitebox": FUSION_WEIGHT_WHITEBOX},
            "selection": "One symmetric fusion fixed before v1 extraction; no grid, learned gate, calibration weight choice or fallback.",
            "source": str(CURRENT.resolve()),
            "scope_limit": "The bound current white-box candidate was selected during earlier repeated calibration development and its fit predictions are not end-to-end OOF. Fusion is therefore a disclosed development diagnostic, never the primary clean v2 result or a final-test claim.",
        },
        "stages": "initialize/prepare/check are CPU-only and label-free. score requires a complete hash-validated v1 frozen cache, then opens development gold in a separate CPU process. Official test is never opened.",
    }


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_paths():
    return [
        Path(__file__), Path(v1.__file__), Path(q.__file__),
        V1_OUT / "protocol.json", V1_OUT / "inputs.jsonl", V1_OUT / "preparation_complete.json",
        V1_OUT / "preparation_statistics.json", V1_OUT / "CPU_CHECK.json",
        V1_OUT / "INDEPENDENT_CPU_AUDIT.json", V1_OUT / "PROTOCOL_LEAKAGE_AUDIT.json",
        CURRENT, CURRENT_META,
    ]


def validate_bound_inputs():
    assert q.sha(v1.__file__) == EXPECTED["v1_runner"]
    assert q.sha(V1_OUT / "protocol.json") == EXPECTED["v1_protocol"]
    assert q.sha(V1_OUT / "inputs.jsonl") == EXPECTED["v1_inputs"]
    assert q.sha(V1_OUT / "preparation_complete.json") == EXPECTED["v1_preparation"]
    assert q.sha(CURRENT) == EXPECTED["current_scores"]
    assert q.sha(CURRENT_META) == EXPECTED["current_metadata"]
    rows, complete = v1.check_prepared()
    assert len(rows) == 793 and complete["scored_claims"] == 8845 and complete["pairs"] == 51953
    return rows, complete


def group_values(row, probability, claim, passage_ids):
    passage_ids = set(passage_ids)
    _, owners = v1.row_pairs(row)
    pair_indices = [index for index, owner in enumerate(owners)
                    if owner[0] == claim["claim_id"] and owner[1] in passage_ids]
    sources = [source for source in claim["retrieval"] if source["passage_id"] in passage_ids]
    if not pair_indices:
        assert not sources
        return np.zeros(10, dtype=np.float64)
    values = probability[pair_indices]
    coverage = np.asarray([source["selected_union_query_coverage"] for source in sources], dtype=np.float64)
    bm25 = np.asarray([selected["bm25"] for source in sources for selected in source["selected"]], dtype=np.float64)
    assert len(values) == len(bm25) and len(coverage) == len(sources)
    return np.asarray([
        values[:, 0].max(), values[:, 0].mean(),
        values[:, 1].max(), values[:, 1].mean(),
        values[:, 2].max(), values[:, 2].mean(),
        coverage.max(), coverage.mean(), bm25.max(), bm25.mean(),
    ], dtype=np.float64)


def citation_features(row, probability):
    base, _ = v1.aggregate_claim_features(row, probability)
    assert base.shape == (len(row["claims"]), len(v1.FEATURE_NAMES))
    _, owners = v1.row_pairs(row)
    output = np.empty((len(row["claims"]), len(FEATURE_NAMES)), dtype=np.float32)
    for claim in row["claims"]:
        parsed = claim["citation_parse"]
        recognized_ids = {int(source_id) for ref in parsed["references"] for source_id in ref["ids"]}
        valid = recognized_ids & set(v1.PASSAGE_IDS)
        invalid = recognized_ids - set(v1.PASSAGE_IDS)
        invalid_ratio = len(invalid) / len(recognized_ids) if recognized_ids else 0.0
        status = [
            float(bool(recognized_ids)), float(bool(valid)), float(bool(invalid)),
            float(bool(parsed["unknown"])), len(valid) / 3.0, invalid_ratio,
            *[float(source_id in valid) for source_id in v1.PASSAGE_IDS],
        ]
        cited = group_values(row, probability, claim, valid)
        uncited_ids = set(v1.PASSAGE_IDS) - valid
        uncited = group_values(row, probability, claim, uncited_ids)
        if valid:
            claim_indices = [index for index, owner in enumerate(owners) if owner[0] == claim["claim_id"]]
            all_max_entailment = float(probability[claim_indices, 0].max())
            support_gap = all_max_entailment - cited[0]
            assert support_gap >= -1e-7
            support_gap = max(0.0, support_gap)
            contrasts = [
                cited[0] - uncited[0], cited[1] - uncited[1],
                cited[4] - uncited[4], cited[5] - uncited[5],
                support_gap, max(cited[4], support_gap),
            ]
        else:
            contrasts = [0.0] * 6
        extra = np.asarray(status + cited.tolist() + uncited.tolist() + contrasts, dtype=np.float64)
        assert len(extra) == len(CITATION_FEATURE_NAMES)
        output[claim["claim_id"]] = np.concatenate((base[claim["claim_id"]], extra))
    assert np.isfinite(output).all()
    return output


def fixed_logit_fusion(citation_probability, whitebox_probability):
    left = np.clip(np.asarray(citation_probability, dtype=np.float64), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    right = np.clip(np.asarray(whitebox_probability, dtype=np.float64), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    result = expit(FUSION_WEIGHT_CITATION * logit(left) + FUSION_WEIGHT_WHITEBOX * logit(right))
    assert np.isfinite(result).all() and ((result > 0) & (result < 1)).all()
    return result


def synthetic_selfcheck():
    sentence_rows = [
        {"sentence_id": 0, "text": "one", "text_sha256": v1.digest("one")},
        {"sentence_id": 1, "text": "two", "text_sha256": v1.digest("two")},
    ]
    passages = [{"passage_id": source_id, "sentences": sentence_rows} for source_id in v1.PASSAGE_IDS]
    retrieval = []
    for source_id in v1.PASSAGE_IDS:
        retrieval.append({"passage_id": source_id, "selected_union_query_coverage": source_id / 10,
                          "selected": [{"rank": 1, "sentence_id": 0, "bm25": float(source_id), "evidence_sha256": v1.digest("one")},
                                       {"rank": 2, "sentence_id": 1, "bm25": float(source_id) / 2, "evidence_sha256": v1.digest("two")} ]})
    row = {"passages": passages, "claims": [{"claim_id": 0, "text": "claim", "retrieval": retrieval,
           "citation_parse": {"references": [{"ids": [2, 4]}], "unknown": [{"text": "passage four"}], "status": "recognized_and_unknown"}}]}
    probability = np.asarray([
        [.2, .7, .1], [.3, .6, .1],
        [.8, .1, .1], [.4, .2, .4],
        [.1, .2, .7], [.2, .3, .5],
    ], dtype=np.float32)
    features = citation_features(row, probability)
    assert features.shape == (1, len(FEATURE_NAMES)) and len(FEATURE_NAMES) == 70
    extra = features[0, len(v1.FEATURE_NAMES):]
    assert np.allclose(extra[:9], [1, 1, 1, 1, 1/3, 1/2, 0, 1, 0])
    cited, uncited = extra[9:19], extra[19:29]
    assert np.allclose(cited[:6], [.8, .6, .2, .15, .4, .25])
    assert np.allclose(uncited[:6], [.3, .2, .7, .45, .7, .35])
    assert np.allclose(cited[6:], [.2, .2, 2, 1.5])
    assert np.allclose(uncited[6:], [.3, .2, 3, 1.5])
    assert np.allclose(extra[29:], [.5, .4, -.3, -.1, 0, .4])
    p = np.asarray([.2, .5, .8])
    assert np.allclose(fixed_logit_fusion(p, p), p)
    assert np.allclose(fixed_logit_fusion(p, p[::-1]), .5)
    return {
        "status": "passed", "feature_width": len(FEATURE_NAMES),
        "base_width": len(v1.FEATURE_NAMES), "citation_width": len(CITATION_FEATURE_NAMES),
        "valid_invalid_unknown_citation": True, "cited_and_uncited_aggregation": True,
        "support_gap_and_conflict": True, "fixed_symmetric_logit_fusion": True,
        "NLI_cache_required": False, "model_loaded": False,
        "GPU_initialized": torch.cuda.is_initialized(), "labels_accessed": False,
        "official_test_opened": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), "Do not overwrite an existing frozen v2 path"
    validate_bound_inputs()
    OUT.mkdir(parents=True)
    q.save(OUT / "PREFLIGHT.json", synthetic_selfcheck())
    q.save(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Retrieved-evidence NLI citation v2\n\n"
        "v2 不改变 v1 的检索、claim、NLI 模型或概率，只新增引用条件读出。它分别汇总被引用来源与未引用来源的 E/N/C、覆盖和 BM25，"
        "再加入支持缺口、冲突差和引用状态。单一 C=0.1，source-group 五折 OOF；fit 定阈值，calibration 只报告。\n\n"
        "另预先冻结一个 0.5/0.5 的 logit 平均融合，不做权重网格。由于绑定的当前白盒候选有既往 calibration 选型和 fit 非完整 OOF 历史，"
        "融合只作开发诊断，citation v2 单模型才是主结果。CPU prepare/check 不需要 NLI 缓存；缓存完成后另行 score。\n",
        encoding="utf-8")
    print("RETRIEVED_EVIDENCE_NLI_CITATION_V2_PROTOCOL_FROZEN", flush=True)


def prepare():
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / "protocol.json") == protocol()
    assert not (OUT / "prepare_started.json").exists()
    rows, complete = validate_bound_inputs()
    snapshot = {str(path.resolve()): q.sha(path) for path in source_paths()}
    q.save(OUT / "prepare_started.json", {"status": "CPU_readout_preparation_started",
           "source_sha256": snapshot, "model_loaded": False, "GPU_used": False,
           "labels_accessed": False, "official_test_opened": False})
    cited_claims = invalid_claims = unknown_claims = 0
    valid_source_histogram = defaultdict(int)
    for row in rows:
        for claim in row["claims"]:
            parsed = claim["citation_parse"]
            ids = {int(source_id) for ref in parsed["references"] for source_id in ref["ids"]}
            valid = ids & set(v1.PASSAGE_IDS)
            cited_claims += int(bool(valid))
            invalid_claims += int(bool(ids - set(v1.PASSAGE_IDS)))
            unknown_claims += int(bool(parsed["unknown"]))
            valid_source_histogram[str(len(valid))] += 1
    statistics = {
        "answers": len(rows), "claims": complete["scored_claims"], "pairs": complete["pairs"],
        "claims_with_valid_citation": cited_claims,
        "claims_with_invalid_recognized_id": invalid_claims,
        "claims_with_unknown_citation_syntax": unknown_claims,
        "valid_source_count_histogram": dict(sorted(valid_source_histogram.items())),
        "feature_width": len(FEATURE_NAMES), "base_v1_width": len(v1.FEATURE_NAMES),
        "citation_width": len(CITATION_FEATURE_NAMES),
        "NLI_extraction_present_at_preparation": (V1_OUT / "extraction_complete.json").exists(),
        "labels_accessed": False, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(OUT / "citation_statistics.json", statistics)
    q.save(OUT / "feature_names.json", {"base_v1": list(v1.FEATURE_NAMES),
                                          "citation": list(CITATION_FEATURE_NAMES),
                                          "combined": list(FEATURE_NAMES)})
    q.save(OUT / "upstream_binding.json", {
        "expected_sha256": EXPECTED,
        "actual_sha256": {
            "v1_runner": q.sha(v1.__file__), "v1_protocol": q.sha(V1_OUT / "protocol.json"),
            "v1_inputs": q.sha(V1_OUT / "inputs.jsonl"),
            "v1_preparation": q.sha(V1_OUT / "preparation_complete.json"),
            "current_scores": q.sha(CURRENT), "current_metadata": q.sha(CURRENT_META),
        },
        "v1_cache_bound_later_only_after_complete_validation": True,
    })
    q.save(OUT / "source_snapshot.json", {"files_sha256": snapshot, "official_test_opened": False})
    assert snapshot == {str(path.resolve()): q.sha(path) for path in source_paths()}
    names = ["PREFLIGHT.json", "protocol.json", "PLAN.md", "citation_statistics.json",
             "feature_names.json", "upstream_binding.json", "source_snapshot.json"]
    q.save(OUT / "preparation_complete.json", {
        "status": "CPU_readout_prepared_waiting_for_v1_cache", "answers": 793,
        "claims": 8845, "pairs": 51953,
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "real_fits": 0, "labels_accessed": False, "model_loaded": False,
        "GPU_used": False, "official_test_opened": False,
    })
    print("RETRIEVED_EVIDENCE_NLI_CITATION_V2_PREPARED_CPU_ONLY", flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_readout_prepared_waiting_for_v1_cache"
    assert complete["answers"] == 793 and complete["claims"] == 8845 and complete["pairs"] == 51953
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    for name, expected in q.read(OUT / "source_snapshot.json")["files_sha256"].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / "protocol.json") == protocol()
    assert q.read(OUT / "upstream_binding.json")["expected_sha256"] == EXPECTED
    rows, _ = validate_bound_inputs()
    return rows, complete


def check():
    rows, complete = check_prepared()
    check_result = synthetic_selfcheck()
    with np.load(CURRENT, allow_pickle=False) as data:
        assert set(data.files) == {"window_scores", "answer_scores"}
        assert data["window_scores"].shape == (210364,) and data["answer_scores"].shape == (793,)
        assert np.isfinite(data["window_scores"]).all()
    q.save(OUT / "CPU_CHECK.json", {
        "status": "passed_waiting_for_v1_probability_cache", "selfcheck": check_result,
        "answers": len(rows), "claims": complete["claims"], "pairs": complete["pairs"],
        "v1_cache_currently_present": (V1_OUT / "extraction_complete.json").exists(),
        "current_whitebox_bound_exact": True, "fusion_formula_fixed_before_scores": True,
        "real_fits": 0, "labels_accessed": False, "model_loaded": False,
        "GPU_used": False, "official_test_opened": False,
    })
    print("RETRIEVED_EVIDENCE_NLI_CITATION_V2_CPU_CHECK_PASSED", flush=True)


def score():
    assert not torch.cuda.is_initialized()
    rows, _ = check_prepared()
    assert q.read(OUT / "CPU_CHECK.json")["status"] == "passed_waiting_for_v1_probability_cache"
    extraction = v1.check_extracted(rows)
    assert extraction["answers"] == 793 and extraction["claims"] == 8845 and extraction["pairs"] == 51953
    assert not (OUT / "score_started.json").exists(), "No silent score overwrite"
    q.save(OUT / "extraction_binding.json", {
        "v1_extraction_complete_sha256": q.sha(V1_OUT / "extraction_complete.json"),
        "v1_cache_file_sha256": extraction["files_sha256"],
        "v1_input_sha256": extraction["input_sha256"],
        "all_v1_caches_validated_before_gold": True,
    })
    q.save(OUT / "score_started.json", {
        "status": "CPU_development_readout_after_frozen_v1_cache",
        "extraction_binding_sha256": q.sha(OUT / "extraction_binding.json"),
        "official_test_opened": False,
    })
    # All NLI/citation features are built before development gold is opened.
    matrices, claim_response_ids, claim_offsets = [], [], {}
    cursor = 0
    for row in rows:
        probability = v1.validate_cache(V1_OUT / "pair_scores" / f"{row['response_id']}.npz", row)
        features = citation_features(row, probability)
        claim_offsets[row["response_id"]] = cursor
        cursor += len(row["claims"])
        matrices.append(features)
        claim_response_ids += [row["response_id"]] * len(row["claims"])
    x = np.concatenate(matrices).astype(np.float32)
    assert x.shape == (8845, len(FEATURE_NAMES)) and np.isfinite(x).all()
    meta = q.metadata()
    lookup = {row["response_id"]: row for row in rows}
    y_claim = np.empty(8845, dtype=np.int8)
    for row in rows:
        risk = np.asarray(meta["by_response"][row["response_id"]]["tokens"]["risk_mask"], dtype=bool)
        for claim in row["claims"]:
            y_claim[claim_offsets[row["response_id"]] + claim["claim_id"]] = int(risk[claim["lexical_token_indices"]].any())
    fit = np.asarray([index for index, response_id in enumerate(claim_response_ids)
                      if lookup[response_id]["partition"] == "fit"], dtype=np.int64)
    calibration = np.asarray([index for index, response_id in enumerate(claim_response_ids)
                              if lookup[response_id]["partition"] == "calibration"], dtype=np.int64)
    groups = np.asarray([lookup[claim_response_ids[index]]["group_id"] for index in fit])
    predictions = np.full(len(y_claim), np.nan, dtype=np.float64)
    folds = []
    for fold, (train_local, held_local) in enumerate(GroupKFold(n_splits=FOLDS).split(fit, y_claim[fit], groups)):
        train, held = fit[train_local], fit[held_local]
        weights = v1.nested_claim_weights(lookup, claim_response_ids, y_claim, train)
        scaler = StandardScaler().fit(x[train], sample_weight=weights[train])
        model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                                   random_state=SEED).fit(scaler.transform(x[train]), y_claim[train],
                                                          sample_weight=weights[train])
        assert model.n_iter_.max() < 2000
        predictions[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
        folds.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                      "train_groups": len(set(groups[train_local])), "held_groups": len(set(groups[held_local])),
                      "iterations": model.n_iter_.tolist()})
    assert np.isfinite(predictions[fit]).all()
    full_weights = v1.nested_claim_weights(lookup, claim_response_ids, y_claim, fit)
    scaler = StandardScaler().fit(x[fit], sample_weight=full_weights[fit])
    model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                               random_state=SEED).fit(scaler.transform(x[fit]), y_claim[fit],
                                                      sample_weight=full_weights[fit])
    predictions[calibration] = model.predict_proba(scaler.transform(x[calibration]))[:, 1]
    assert np.isfinite(predictions).all()
    citation_windows = v1.project_claim_scores(meta, rows, claim_offsets, predictions)
    with np.load(CURRENT, allow_pickle=False) as data:
        whitebox_windows = data["window_scores"].copy()
    assert whitebox_windows.shape == citation_windows.shape and np.isfinite(whitebox_windows).all()
    fused_windows = fixed_logit_fusion(citation_windows, whitebox_windows)
    left, right = meta["bounds"]["fit"]
    fit_answer_indices = [index for index, answer in enumerate(meta["answers"]) if answer["partition"] == "fit"]
    results = {}
    for name, window_scores in (("citation_lr", citation_windows), ("fixed_whitebox_fusion", fused_windows)):
        answer_scores = q.answer_scores(meta, window_scores)
        thresholds = {
            "window": q.choose_threshold([window["label"] for window in meta["windows"][left:right]],
                                         window_scores[left:right]),
            "answer": q.choose_threshold([meta["answers"][index]["label"] for index in fit_answer_indices],
                                         answer_scores[fit_answer_indices]),
        }
        metrics = q.metrics(meta, window_scores, thresholds)
        path = OUT / f"{name}_scores.npz"
        np.savez_compressed(path, window_scores=window_scores, answer_scores=answer_scores,
                            claim_scores=predictions if name == "citation_lr" else np.asarray([], np.float64))
        results[name] = {"thresholds": thresholds, "metrics": metrics, "scores_sha256": q.sha(path),
                         "primary_clean_v2_readout": name == "citation_lr",
                         "upstream_fit_not_end_to_end_OOF": name == "fixed_whitebox_fusion"}
    np.save(OUT / "claim_features.npy", x)
    np.save(OUT / "claim_labels.npy", y_claim)
    model_path = OUT / "citation_lr.pkl"
    model_path.write_bytes(pickle.dumps({"model": model, "scaler": scaler, "C": LR_C,
                                         "feature_names": FEATURE_NAMES, "folds": folds}, protocol=5))
    q.save(OUT / "summary.json", {
        "status": "development_only_complete", "role": ROLE,
        "claims": len(y_claim), "positive_claims": int(y_claim.sum()), "folds": folds,
        "results": results, "fixed_fusion_weights": [FUSION_WEIGHT_CITATION, FUSION_WEIGHT_WHITEBOX],
        "fit_only_model_and_thresholds": True, "calibration_reporting_only_for_v2": True,
        "fusion_upstream_calibration_history_disclosed": True,
        "official_test_opened": False, "final_test_claim": False,
    })
    report = [
        "# Citation-conditioned retrieved-evidence NLI v2", "",
        "v2 只改读出：区分被引用与未引用来源；单一 C、source-group OOF、fit 阈值。calibration 只报告。", "",
        "| 方法 | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("citation_lr", "fixed_whitebox_fusion"):
        metrics = results[name]["metrics"]
        report.append(f"| {name} | {metrics['fit']['windows']['f1']:.4f} | {metrics['calibration']['windows']['f1']:.4f} | {metrics['fit']['answers']['f1']:.4f} | {metrics['calibration']['answers']['f1']:.4f} |")
    report += ["", "固定融合权重在结果前冻结；其白盒上游有旧 calibration 选型和 fit 非完整 OOF 历史，只能作为开发诊断。正式 baseline 未修改，测试仍封存。"]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_names = ["extraction_binding.json", "score_started.json", "claim_features.npy", "claim_labels.npy",
                    "citation_lr.pkl", "citation_lr_scores.npz", "fixed_whitebox_fusion_scores.npz",
                    "summary.json", "REPORT.md"]
    q.save(OUT / "complete.json", {
        "status": "complete_development_only", "files_sha256": {name: q.sha(OUT / name) for name in output_names},
        "official_test_opened": False, "final_test_claim": False,
    })
    print("RETRIEVED_EVIDENCE_NLI_CITATION_V2_SCORE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "score"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            check()
        else:
            score()
