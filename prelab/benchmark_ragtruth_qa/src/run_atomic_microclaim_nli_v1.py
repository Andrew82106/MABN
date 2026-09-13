"""Atomic relation-aware retrieved-evidence NLI candidate for RAGTruth QA.

The runner refines the frozen sentence-like claim geometry into deterministic
microclaims, retrieves evidence independently inside each released passage,
scores evidence/hypothesis pairs with a frozen ModernBERT NLI checkpoint, and
fits only a source-group-cross-fitted development readout.  Formal baselines
and the sealed official test are never modified or opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/atomic_relation_audit_r32_v1"))
import run_development as q  # noqa: E402
import run_retrieved_evidence_nli_v1 as old_nli  # noqa: E402
import audit_atomic_relations as atomic  # noqa: E402


OUT = ROOT / "results/atomic_microclaim_nli_v1"
ATOMIC_DIR = ROOT / "research/atomic_relation_audit_r32_v1"
ATOMIC_FILES = {
    "fit": ATOMIC_DIR / "microclaims_fit.jsonl",
    "calibration": ATOMIC_DIR / "microclaims_calibration.jsonl",
}
OLD_INPUT = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
OLD_EXTRACT = ROOT / "results/retrieved_evidence_nli_v1/extraction_complete.json"
LAYOUT = ROOT / "data/feature_preparation/plans.jsonl"
MODEL = old_nli.MODEL
MODEL_REVISION = old_nli.MODEL_REVISION
MODEL_SHA256 = old_nli.MODEL_SHA256

# These optional white-box inputs are all group-OOF on fit and full-fit on cal.
OOF_WHITEBOX = ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy"
OOF_CONTROL = ROOT / "results/group_crossfit_lb_large_v1/group_oof.json"
GENERATION = ROOT / "results/development_v1/matrices/base.npy"
INCUMBENT_SCORES = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
INCUMBENT_RESULT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"

ROLE = "ours_atomic_microclaim_candidate; formal baselines untouched"
PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_MICROCLAIMS = {"fit": 9061, "calibration": 2268}
EXPECTED_SCORED_MICROCLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
PASSAGE_IDS = (1, 2, 3)
TOP_K = 2
PAIR_LIMIT = 2048
INFERENCE_BATCH = 16
THREADS = 4
FOLDS = 5
SEED = 20261020
BACKEND = "cuda_fp32_sdpa_math_batch16_atomic_v1"
PEAK_LIMIT = int(6.5 * 1024 ** 3)
MIN_FREE = 3 * 1024 ** 3

CONNECTOR = re.compile(r"^\s*(?:(?:and|or|but|yet|whereas|while|which|who)\b[,;:]?\s*)+", re.I)
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

NEG_KINDS = tuple(name for name, _ in atomic.NEGATION_PATTERNS)
COMP_KINDS = tuple(name for name, _, _ in atomic.COMPARATOR_PATTERNS)
TEMP_KINDS = tuple(name for name, _ in atomic.TEMPORAL_PATTERNS)
COND_KINDS = tuple(name for name, _ in atomic.CONDITION_PATTERNS)

CLAIM_FEATURE_NAMES = (
    tuple(atomic.FEATURE_NAMES)
    + ("log1p_words", "child_fraction", "is_split_parent", "log1p_children",
       "log1p_predicates", "log1p_entities", "log1p_negations", "log1p_comparators",
       "log1p_quantities", "log1p_temporal", "log1p_conditions", "log1p_explicit_sources",
       "log1p_parent_sources", "passage_context_inherited", "has_antecedent_subject")
    + tuple(f"neg_kind__{name}" for name in NEG_KINDS)
    + tuple(f"comp_kind__{name}" for name in COMP_KINDS)
    + tuple(f"temporal_kind__{name}" for name in TEMP_KINDS)
    + tuple(f"condition_kind__{name}" for name in COND_KINDS)
    + tuple(f"explicit_source__{pid}" for pid in PASSAGE_IDS)
    + tuple(f"parent_source__{pid}" for pid in PASSAGE_IDS)
)

PAIR_RELATION_NAMES = (
    "lexical_jaccard", "claim_token_coverage", "subject_token_coverage",
    "entity_token_coverage", "predicate_token_match",
    "claim_has_negation", "evidence_has_negation", "negation_match", "negation_mismatch",
    "claim_has_comparator", "evidence_has_comparator", "comparator_match",
    "comparator_reversal", "comparator_missing",
    "claim_has_quantity", "evidence_has_quantity", "quantity_any_exact",
    "quantity_all_exact", "quantity_unit_match", "quantity_same_unit_different_value",
    "quantity_missing", "claim_has_temporal", "evidence_has_temporal",
    "temporal_match", "temporal_reversal", "temporal_missing",
    "claim_has_condition", "evidence_has_condition", "condition_match", "condition_mismatch",
)


def passage_feature_names():
    names = []
    for pid in PASSAGE_IDS:
        prefix = f"passage_{pid}"
        names += [
            f"{prefix}_individual_max_entailment", f"{prefix}_individual_mean_entailment",
            f"{prefix}_individual_max_neutral", f"{prefix}_individual_mean_neutral",
            f"{prefix}_individual_max_contradiction", f"{prefix}_individual_mean_contradiction",
            f"{prefix}_joint_entailment", f"{prefix}_joint_neutral", f"{prefix}_joint_contradiction",
            f"{prefix}_union_query_coverage", f"{prefix}_top_bm25", f"{prefix}_second_bm25",
            f"{prefix}_bm25_gap", f"{prefix}_is_explicit_source", f"{prefix}_is_parent_source",
        ]
        for aggregation in ("max", "mean"):
            names += [f"{prefix}_relation_{aggregation}__{name}" for name in PAIR_RELATION_NAMES]
    return tuple(names)


GLOBAL_FEATURE_NAMES = (
    "global_max_entailment", "global_mean_entailment", "global_min_entailment",
    "global_max_neutral", "global_mean_neutral", "global_min_neutral",
    "global_max_contradiction", "global_mean_contradiction", "global_min_contradiction",
    "best_passage_entailment", "second_passage_entailment", "best_second_entailment_gap",
    "best_passage_contradiction", "second_passage_contradiction", "best_second_contradiction_gap",
    "best_entailment_source_1", "best_entailment_source_2", "best_entailment_source_3",
    "cited_max_entailment", "uncited_max_entailment", "cited_minus_uncited_entailment",
    "cited_max_contradiction", "uncited_max_contradiction", "cited_minus_uncited_contradiction",
    "lack_of_entailment", "or_risk", "contradiction_x_lack",
)

EVIDENCE_FEATURE_NAMES = CLAIM_FEATURE_NAMES + passage_feature_names() + GLOBAL_FEATURE_NAMES
WHITEBOX_BASE_NAMES = ("oof_lookback", "oof_large", "generation_nll")
WHITEBOX_FEATURE_NAMES = tuple(
    f"whitebox_{aggregation}__{name}"
    for aggregation in ("max", "mean", "min", "std")
    for name in WHITEBOX_BASE_NAMES
)

CANDIDATES = (
    "evidence__lr_C0.001", "evidence__lr_C0.01", "evidence__lr_C0.1",
    "evidence__hist_leaf7", "evidence__hist_leaf15", "evidence__extra_depth8",
    "fusion__lr_C0.001", "fusion__lr_C0.01", "fusion__lr_C0.1",
    "fusion__hist_leaf7", "fusion__hist_leaf15", "fusion__extra_depth8",
)


def json_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_jsonl(path, rows):
    path = Path(path)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protocol():
    return {
        "version": "atomic-microclaim-retrieved-nli-v1",
        "role": ROLE,
        "scope": {
            "dataset": "Only original RAGTruth QA fit634 + calibration159 development answers.",
            "microclaims": "Exact label-blind output of atomic_relation_audit_r32_v1 (9,061 fit; 2,268 calibration). Retain seven punctuation-only records in an audit list and score the remaining 9,055 + 2,267 factual microclaims.",
            "windows": "Unchanged eligible 4-original-BPE stride-one windows, used only in score stage.",
            "test": "Official test paths are absent from this runner and never opened.",
            "expansion_hook": "The loader is partition-manifest based; a future 3,680-answer run must use a new output/version and a separately frozen microclaim manifest. It may not be mixed into v1.",
        },
        "hypothesis": {
            "unsplit": "Exact microclaim text.",
            "split": "Remove only a leading discourse/attribution scaffold. For noninitial fragments with a deterministic previous-sibling antecedent, prepend that exact surface subject after removing a leading connector.",
            "labels": "No label, score, corpus statistic, or model output affects the transformation.",
        },
        "retrieval": {
            "query": "The deterministic contextualized microclaim hypothesis.",
            "per_passage": "Reuse the frozen v1 Unicode-token BM25 implementation and choose top two sentences independently in passage 1, 2 and 3.",
            "nli_pairs": "Each selected sentence separately plus their in-rank-order concatenation when two sentences exist. Premise=evidence, hypothesis=contextualized microclaim.",
        },
        "nli": {
            "checkpoint": "tasksource/ModernBERT-base-nli", "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256, "frozen": True, "class_order": list(old_nli.CLASSES),
            "maximum_pair_tokens": PAIR_LIMIT, "truncation": False, "backend": BACKEND,
            "reuse": "An exact premise+hypothesis pair may reuse a probability from the already frozen v1 cache only when every occurrence has bit-identical cached output. Ambiguous repeats are recomputed; all other pairs use the identical frozen checkpoint and numerical path.",
        },
        "features": {
            "evidence_relation_width": len(EVIDENCE_FEATURE_NAMES),
            "whitebox_optional_width": len(WHITEBOX_FEATURE_NAMES),
            "relation": "Label-blind claim flags plus lexical subject/entity/predicate, negation, comparator direction, quantity/unit, temporal direction, condition, and source-specific compatibility with retrieved evidence.",
            "whitebox": "Optional microclaim pooling of clean fit group-OOF Lookback/large probabilities and frozen generation NLL. Candidate family is selected on fit OOF only.",
        },
        "readout": {
            "label": "A microclaim is positive iff any lexical original BPE assigned to it has the unchanged human risk label; labels open only in score stage after NLI caches freeze.",
            "candidates": list(CANDIDATES),
            "crossfit": "Five-fold GroupKFold by locked source-connected group supplies every fit microclaim prediction. Full-fit selected model predicts calibration.",
            "weights": "Equal source-group mass, then answer mass, then microclaim mass, followed by binary class balance and restored equal group loss mass.",
            "selection": "Fit OOF only: maximize minimum(window F1, answer F1), then window F1, answer F1, precision, and simpler candidate order.",
            "projection": "Each lexical BPE inherits every microclaim whose alphanumeric characters it overlaps (normally one; a boundary-straddling BPE may map to both). Each unchanged four-BPE window and answer take the maximum overlapping score.",
            "strict": "Window and answer thresholds selected on fit OOF and frozen for calibration.",
            "diagnostic": "Also report calibration F1-optimal thresholds explicitly as a non-independent common development diagnostic.",
        },
        "baselines": "Read-only references only. No baseline file, structure, feature, parameter, threshold, or score is changed.",
        "stage_gate": "initialize -> CPU prepare -> CPU check -> explicit GPU smoke -> explicit extraction -> CPU score",
    }


def hypothesis_for(claim):
    raw = claim["text"].strip()
    if int(claim["children_in_parent"]) == 1:
        return raw
    core, _ = atomic.strip_scaffold(raw)
    core = CONNECTOR.sub("", core).strip(" \t\r\n,;:")
    if not core:
        core = raw
    antecedent = claim.get("antecedent_subject")
    if int(claim["child_index"]) > 0 and antecedent:
        subject = antecedent["text"].strip(" \t\r\n,;:")
        if subject and not re.match(rf"^{re.escape(subject)}\b", core, re.I):
            core = f"{subject} {core}"
    return core


def token_set(text):
    return {token.casefold() for token in WORD.findall(text)}


def kinds(rows, field="kind"):
    return {row[field] for row in rows}


def values_and_units(rows):
    values = {(value.casefold(), row.get("unit")) for row in rows for value in row.get("values", [])}
    units = {row.get("unit") for row in rows if row.get("unit")}
    return values, units


def claim_surface(claim):
    return {
        "negation": claim["negation_cues"], "comparator": claim["comparator_cues"],
        "quantities": claim["quantities"], "temporal_order": claim["temporal_order_cues"],
        "condition": claim["condition_cues"], "entity_candidates": claim["entity_candidates"],
        "local_subject": claim.get("effective_subject"), "predicate": claim.get("predicate"),
    }


def relation_pair_features(claim, evidence_text):
    left = claim_surface(claim)
    right = atomic.surface_features(evidence_text, 0)
    ct, et = token_set(hypothesis_for(claim)), token_set(evidence_text)
    overlap = ct & et
    subject = token_set(left["local_subject"]["text"]) if left.get("local_subject") else set()
    entities = set().union(*(token_set(row["text"]) for row in left["entity_candidates"])) if left["entity_candidates"] else set()
    predicate = token_set(left["predicate"]["text"]) if left.get("predicate") else set()

    cneg, eneg = bool(left["negation"]), bool(right["negation"])
    ccomp = {row.get("operator", row["kind"]) for row in left["comparator"]}
    ecomp = {row.get("operator", row["kind"]) for row in right["comparator"]}
    reverse = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "up": "down", "down": "up"}
    comp_reverse = any(reverse.get(value) in ecomp for value in ccomp)
    cvalues, cunits = values_and_units(left["quantities"])
    evalues, eunits = values_and_units(right["quantities"])
    exact = cvalues & evalues
    same_unit_diff = any(cu and cu == eu and cv != ev for cv, cu in cvalues for ev, eu in evalues)
    ctemp, etemp = kinds(left["temporal_order"]), kinds(right["temporal_order"])
    temporal_reverse_map = {"before": "after", "after": "before"}
    temporal_reverse = any(temporal_reverse_map.get(value) in etemp for value in ctemp)
    ccond, econd = bool(left["condition"]), bool(right["condition"])

    values = [
        len(overlap) / max(1, len(ct | et)), len(overlap) / max(1, len(ct)),
        len(subject & et) / max(1, len(subject)), len(entities & et) / max(1, len(entities)),
        float(bool(predicate & et)), float(cneg), float(eneg), float(cneg == eneg), float(cneg != eneg),
        float(bool(ccomp)), float(bool(ecomp)), float(bool(ccomp & ecomp)), float(comp_reverse),
        float(bool(ccomp) and not bool(ecomp)), float(bool(cvalues)), float(bool(evalues)),
        float(bool(exact)), float(bool(cvalues) and cvalues <= evalues), float(bool(cunits & eunits)),
        float(same_unit_diff), float(bool(cvalues) and not bool(evalues)),
        float(bool(ctemp)), float(bool(etemp)), float(bool(ctemp & etemp)), float(temporal_reverse),
        float(bool(ctemp) and not bool(etemp)), float(ccond), float(econd), float(ccond == econd),
        float(ccond != econd),
    ]
    result = np.asarray(values, dtype=np.float32)
    assert result.shape == (len(PAIR_RELATION_NAMES),) and np.isfinite(result).all()
    return result


def claim_self_features(claim):
    flags = [float(claim["feature_vector"][name]) for name in atomic.FEATURE_NAMES]
    children = int(claim["children_in_parent"])
    child = int(claim["child_index"])
    values = flags + [
        math.log1p(claim["word_count"]), child / max(1, children - 1), float(children > 1),
        math.log1p(children), math.log1p(claim["predicate_count"]),
        math.log1p(len(claim["entity_candidates"])), math.log1p(len(claim["negation_cues"])),
        math.log1p(len(claim["comparator_cues"])), math.log1p(len(claim["quantities"])),
        math.log1p(len(claim["temporal_order_cues"])), math.log1p(len(claim["condition_cues"])),
        math.log1p(len(claim["explicit_passage_ids"])), math.log1p(len(claim["parent_passage_ids"])),
        float(claim["passage_context_inherited"]), float(claim.get("antecedent_subject") is not None),
    ]
    for vocabulary, key in ((NEG_KINDS, "negation_cues"), (COMP_KINDS, "comparator_cues"),
                            (TEMP_KINDS, "temporal_order_cues"), (COND_KINDS, "condition_cues")):
        present = kinds(claim[key])
        values += [float(name in present) for name in vocabulary]
    values += [float(pid in claim["explicit_passage_ids"]) for pid in PASSAGE_IDS]
    values += [float(pid in claim["parent_passage_ids"]) for pid in PASSAGE_IDS]
    result = np.asarray(values, dtype=np.float32)
    assert result.shape == (len(CLAIM_FEATURE_NAMES),) and np.isfinite(result).all()
    return result


def assign_tokens(text, offsets, claims):
    owners = [[] for _ in offsets]
    inverse = [[] for _ in claims]
    for ti, (left, right) in enumerate(offsets):
        chars = [position for position in range(left, right) if text[position].isalnum()]
        if not chars:
            continue
        overlap = [sum(claim["start"] <= position < claim["end"] for position in chars) for claim in claims]
        owners[ti] = [cid for cid, count in enumerate(overlap) if count > 0]
        assert owners[ti]
        for owner in owners[ti]: inverse[owner].append(ti)
    assert all(inverse)
    lexical = [any(text[position].isalnum() for position in range(left, right)) for left, right in offsets]
    assert list(map(bool, owners)) == lexical
    return owners, inverse


def build_prepared_row(old_row, layout, all_claims):
    rid = old_row["response_id"]
    text = layout["original_response"]
    assert rid == layout["response_id"] and digest(text) == old_row["answer_sha256"]
    assert [claim["microclaim_index"] for claim in all_claims] == list(range(len(all_claims)))
    assert all(claim["response_id"] == rid and claim["partition"] == old_row["partition"] for claim in all_claims)
    claims = [claim for claim in all_claims if any(char.isalnum() for char in claim["text"])]
    discarded = [{"microclaim_index": claim["microclaim_index"], "microclaim_id": claim["microclaim_id"],
                  "start": claim["start"], "end": claim["end"], "text": claim["text"],
                  "reason": "no_unicode_alphanumeric_character"}
                 for claim in all_claims if not any(char.isalnum() for char in claim["text"])]
    owners, inverse = assign_tokens(text, layout["original"]["response_token_offsets"], claims)
    prepared_claims = []
    for cid, (claim, token_indices) in enumerate(zip(claims, inverse)):
        assert text[claim["start"]:claim["end"]] == claim["text"]
        hypothesis = hypothesis_for(claim)
        assert hypothesis and any(char.isalnum() for char in hypothesis)
        retrieval = [old_nli.select_evidence(hypothesis, passage) for passage in old_row["passages"]]
        item = dict(claim)
        item.update({"claim_id": cid, "hypothesis": hypothesis,
                     "hypothesis_sha256": digest(hypothesis),
                     "lexical_token_indices": token_indices, "retrieval": retrieval})
        prepared_claims.append(item)
    result = {
        "schema_version": "atomic-microclaim-retrieved-nli-v1",
        "response_id": rid, "source_id": old_row["source_id"], "group_id": old_row["group_id"],
        "partition": old_row["partition"], "answer_sha256": old_row["answer_sha256"],
        "response_token_offsets_sha256": old_row["response_token_offsets_sha256"],
        "passages": old_row["passages"], "claims": prepared_claims,
        "discarded_nonlexical_microclaims": discarded,
        "lexical_token_microclaims": owners, "labels_used": False, "official_test_opened": False,
    }
    old_nli.reject_annotation_keys(result)
    return result


def evidence_lookup(row):
    return {(passage["passage_id"], sentence["sentence_id"]): sentence
            for passage in row["passages"] for sentence in passage["sentences"]}


def row_pairs(row):
    lookup = evidence_lookup(row)
    pairs, owners = [], []
    for claim in row["claims"]:
        for source in claim["retrieval"]:
            selected_texts = []
            for item in source["selected"]:
                sentence = lookup[(source["passage_id"], item["sentence_id"])]
                assert sentence["text_sha256"] == item["evidence_sha256"]
                selected_texts.append(sentence["text"])
                pairs.append((sentence["text"], claim["hypothesis"]))
                owners.append((claim["claim_id"], source["passage_id"], item["rank"],
                               item["sentence_id"], -1))
            if len(selected_texts) == 2:
                joint = " ".join(selected_texts)
                pairs.append((joint, claim["hypothesis"]))
                owners.append((claim["claim_id"], source["passage_id"], 0,
                               source["selected"][0]["sentence_id"], source["selected"][1]["sentence_id"]))
    assert pairs and len(pairs) == len(owners)
    return pairs, owners


def source_paths(include_old_extract=False):
    paths = [Path(__file__), Path(old_nli.__file__), Path(atomic.__file__), Path(q.__file__),
             ATOMIC_DIR / "complete.json", ATOMIC_DIR / "protocol.json", *ATOMIC_FILES.values(),
             OLD_INPUT, LAYOUT, MODEL / "config.json", MODEL / "tokenizer.json",
             MODEL / "tokenizer_config.json", MODEL / "special_tokens_map.json",
             MODEL / "model.safetensors", MODEL / "download_manifest.json"]
    if include_old_extract:
        paths.append(OLD_EXTRACT)
    return {str(path.resolve()): q.sha(path) for path in paths}


def synthetic_selfcheck():
    claim = {
        "text": "and decreased to 12 percent", "children_in_parent": 2, "child_index": 1,
        "antecedent_subject": {"text": "Revenue"}, "feature_vector": {name: False for name in atomic.FEATURE_NAMES},
        "word_count": 5, "predicate_count": 1, "entity_candidates": [], "negation_cues": [],
        "comparator_cues": [{"kind": "decrease", "operator": "down"}],
        "quantities": [{"values": ["12"], "unit": "percent"}], "temporal_order_cues": [],
        "condition_cues": [], "explicit_passage_ids": [], "parent_passage_ids": [],
        "passage_context_inherited": False, "effective_subject": {"text": "Revenue"}, "predicate": {"text": "decreased"},
    }
    assert hypothesis_for(claim) == "Revenue decreased to 12 percent"
    same = relation_pair_features(claim, "Revenue decreased to 12 percent.")
    conflict = relation_pair_features(claim, "Revenue increased to 15 percent.")
    name = {key: index for index, key in enumerate(PAIR_RELATION_NAMES)}
    assert same[name["quantity_any_exact"]] == 1 and conflict[name["quantity_same_unit_different_value"]] == 1
    assert same[name["comparator_match"]] == 1 and conflict[name["comparator_reversal"]] == 1
    assert claim_self_features(claim).shape == (len(CLAIM_FEATURE_NAMES),)
    return {
        "status": "passed", "antecedent_completion": True, "quantity_conflict": True,
        "comparator_reversal": True, "claim_feature_width": len(CLAIM_FEATURE_NAMES),
        "evidence_feature_width": len(EVIDENCE_FEATURE_NAMES),
        "whitebox_feature_width": len(WHITEBOX_FEATURE_NAMES),
        "model_loaded": False, "GPU_initialized": torch.cuda.is_initialized(),
        "labels_accessed": False, "official_test_opened": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True)
    save_json(OUT / "PREFLIGHT.json", synthetic_selfcheck())
    save_json(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Atomic microclaim retrieved-NLI v1\n\n"
        "本实验只处理原始 QA fit634+cal159。确定性原子微主张在每个 passage 内独立 BM25 检索，冻结 ModernBERT-base-nli 对单句及 top-2 合并证据评分；未拆且文本完全一致的 pair 可复用冻结 v1 概率。\n\n"
        "证据关系特征在微主张级汇总；fit 按 source group 五折 OOF 选读出与阈值，calibration 只报告严格迁移结果及明确标注的 cal-F1Opt 共同诊断。正式 baseline、标签、4-BPE 窗口与 official test 不变。扩展到 3680 条必须另建版本和冻结清单，不能混入本目录。\n",
        encoding="utf-8")
    print("ATOMIC_MICROCLAIM_NLI_PROTOCOL_FROZEN", flush=True)


def prepare():
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / "protocol.json") == protocol()
    assert q.read(OUT / "PREFLIGHT.json")["status"] == "passed"
    assert not (OUT / "prepare_started.json").exists()
    snapshot = source_paths(False)
    save_json(OUT / "prepare_started.json", {"status": "CPU_label_free_preparation_started",
              "source_sha256": snapshot, "labels_accessed": False, "GPU_used": False,
              "official_test_opened": False})
    old_rows = json_lines(OLD_INPUT)
    layouts = {row["response_id"]: row for row in json_lines(LAYOUT)}
    by_response = defaultdict(list)
    for part in PARTITIONS:
        for claim in json_lines(ATOMIC_FILES[part]):
            by_response[claim["response_id"]].append(claim)
    assert len(old_rows) == len(layouts) == len(by_response) == 793
    assert Counter(row["partition"] for row in old_rows) == EXPECTED_ANSWERS
    assert Counter(claim["partition"] for claims in by_response.values() for claim in claims) == EXPECTED_MICROCLAIMS

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast and q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    prepared = []
    pair_count = joint_count = max_tokens = contextualized = 0
    pair_lengths = Counter()
    for answer_index, old_row in enumerate(old_rows):
        claims = sorted(by_response[old_row["response_id"]], key=lambda row: row["microclaim_index"])
        item = build_prepared_row(old_row, layouts[old_row["response_id"]], claims)
        pairs, owners = row_pairs(item)
        encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                            add_special_tokens=True, padding=False, truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        assert max(lengths) <= PAIR_LIMIT
        cursor = 0
        for claim in item["claims"]:
            contextualized += int(claim["hypothesis"] != claim["text"].strip())
            for source in claim["retrieval"]:
                for selected in source["selected"]:
                    selected["individual_pair_token_length"] = lengths[cursor]
                    cursor += 1
                if len(source["selected"]) == 2:
                    source["joint_pair_token_length"] = lengths[cursor]
                    cursor += 1
                else:
                    source["joint_pair_token_length"] = None
        assert cursor == len(lengths)
        prepared.append(item)
        pair_count += len(pairs)
        joint_count += sum(owner[2] == 0 for owner in owners)
        max_tokens = max(max_tokens, max(lengths))
        pair_lengths.update(lengths)
        if (answer_index + 1) % 100 == 0:
            print("ATOMIC_MICROCLAIM_PREPARED", answer_index + 1, 793, flush=True)
    del tokenizer
    save_jsonl(OUT / "inputs.jsonl", prepared)
    assert sum(len(row["claims"]) for row in prepared) == sum(EXPECTED_SCORED_MICROCLAIMS.values())
    assert sum(len(row["discarded_nonlexical_microclaims"]) for row in prepared) == 7
    assert snapshot == source_paths(False)
    lengths = np.repeat(np.asarray(sorted(pair_lengths), dtype=np.int32),
                        np.asarray([pair_lengths[n] for n in sorted(pair_lengths)], dtype=np.int64))
    stats = {
        "status": "CPU_prepared_waiting_for_check", "answers": len(prepared),
        "answers_by_partition": dict(Counter(row["partition"] for row in prepared)),
        "original_atomic_microclaims": sum(EXPECTED_MICROCLAIMS.values()),
        "microclaims": sum(len(row["claims"]) for row in prepared),
        "discarded_nonlexical_microclaims": 7,
        "microclaims_by_partition": dict(Counter(claim["partition"] for row in prepared for claim in row["claims"])),
        "contextualized_split_microclaims": contextualized, "NLI_pairs": pair_count,
        "joint_top2_pairs": joint_count,
        "pair_tokens": {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(max_tokens)},
        "input_sha256": q.sha(OUT / "inputs.jsonl"), "protocol_sha256": q.sha(OUT / "protocol.json"),
        "source_sha256": snapshot, "labels_accessed": False, "model_loaded": False,
        "GPU_used": False, "official_test_opened": False,
    }
    save_json(OUT / "preparation_statistics.json", stats)
    names = ("PREFLIGHT.json", "protocol.json", "PLAN.md", "inputs.jsonl",
             "prepare_started.json", "preparation_statistics.json")
    save_json(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_not_model_extracted", "answers": 793,
        "microclaims": stats["microclaims"], "pairs": pair_count,
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "labels_accessed": False, "GPU_used": False, "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("ATOMIC_MICROCLAIM_NLI_CPU_PREPARATION_COMPLETE", pair_count, flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_not_model_extracted"
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    assert q.read(OUT / "protocol.json") == protocol()
    stats = q.read(OUT / "preparation_statistics.json")
    for name, expected in stats["source_sha256"].items():
        assert q.sha(name) == expected, name
    rows = json_lines(OUT / "inputs.jsonl")
    assert len(rows) == 793 and sum(len(row["claims"]) for row in rows) == complete["microclaims"]
    assert sum(len(row_pairs(row)[0]) for row in rows) == complete["pairs"]
    return rows, complete


def check():
    rows, complete = check_prepared()
    samples = []
    for row in (rows[0], rows[len(rows) // 2], rows[-1]):
        pairs, owners = row_pairs(row)
        samples.append({"response_id": row["response_id"], "claims": len(row["claims"]),
                        "pairs": len(pairs), "sources": sorted({owner[1] for owner in owners}),
                        "joint_pairs": sum(owner[2] == 0 for owner in owners)})
    save_json(OUT / "CPU_CHECK.json", {
        "status": "passed_waiting_for_GPU", "selfcheck_replayed": synthetic_selfcheck(),
        "answers": len(rows), "microclaims": complete["microclaims"], "pairs": complete["pairs"],
        "samples": samples, "exact_input_hash": q.sha(OUT / "inputs.jsonl") == q.read(OUT / "preparation_statistics.json")["input_sha256"],
        "all_source_ids": all(sorted({owner[1] for owner in row_pairs(row)[1]}) == list(PASSAGE_IDS) for row in rows),
        "labels_accessed": False, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    })
    print("ATOMIC_MICROCLAIM_NLI_CPU_CHECK_PASSED", complete["pairs"], flush=True)


def runtime_signature():
    return {
        "backend": BACKEND, "source_sha256": q.sha(__file__),
        "protocol_sha256": q.sha(OUT / "protocol.json"), "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "model_sha256": MODEL_SHA256, "revision": MODEL_REVISION, "batch": INFERENCE_BATCH,
        "precision": "float32", "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
        "cuda_runtime": torch.version.cuda, "role": ROLE,
    }


def infer_pairs(tokenizer, model, pairs, device):
    output, shapes = [], []
    for left in range(0, len(pairs), INFERENCE_BATCH):
        batch = pairs[left:left + INFERENCE_BATCH]
        encoded = tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                            add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= PAIR_LIMIT
        shapes.append([len(batch), int(encoded["input_ids"].shape[1])])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), torch.autocast(device.type, enabled=False), context:
            logits = model(**encoded).logits
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            output.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(output).astype(np.float32, copy=False)
    assert result.shape == (len(pairs), 3) and np.allclose(result.sum(1), 1, rtol=0, atol=2e-6)
    return result, shapes


def load_cuda():
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert q.read(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE, f"Insufficient free GPU memory: {free}"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest"); torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False).eval().requires_grad_(False).to("cuda:0")
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cuda" and not parameter.requires_grad for parameter in model.parameters())
    return tokenizer, model, {"device": torch.cuda.get_device_name(0), "free_before_load": free,
                              "total_memory": total, "parameters": sum(p.numel() for p in model.parameters())}


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def memory_stats():
    result = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
              "reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}
    assert max(result.values()) <= PEAK_LIMIT, result
    return result


def old_pair_cache():
    assert q.read(OLD_EXTRACT)["status"] == "complete_frozen_probabilities_not_scored"
    mapping, ambiguous = {}, set()
    for old_row in json_lines(OLD_INPUT):
        probability = old_nli.validate_cache(old_nli.OUT / "pair_scores" / f"{old_row['response_id']}.npz", old_row)
        pairs, _ = old_nli.row_pairs(old_row)
        for pair, value in zip(pairs, probability):
            key = digest({"premise": pair[0], "hypothesis": pair[1]})
            if key in ambiguous:
                continue
            if key not in mapping:
                mapping[key] = value.copy()
            elif not np.array_equal(mapping[key], value):
                # Padding-batch context creates tiny FP32 variation for a few
                # repeated texts. Recompute those pairs instead of selecting
                # one old occurrence silently.
                del mapping[key]
                ambiguous.add(key)
    return mapping


def cache_signature(row):
    pairs, owners = row_pairs(row)
    return {
        "response_id": row["response_id"], "input_row_sha256": digest(row),
        "pair_definition_sha256": digest({"pairs": pairs, "owners": owners}),
        "input_file_sha256": q.sha(OUT / "inputs.jsonl"), "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": MODEL_SHA256, "execution_signature_sha256": digest(runtime_signature()),
        "execution_backend": BACKEND, "method_role": ROLE,
    }


def validate_cache(path, row):
    pairs, owners = row_pairs(row); signature = cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        expected = set(signature) | {"probabilities", "claim_ids", "passage_ids", "pair_kinds",
                                     "sentence_a", "sentence_b", "reused_old_pair"}
        assert set(data.files) == expected
        for key, value in signature.items():
            assert data[key].item() == value, (path.name, key)
        probability = data["probabilities"]
        assert probability.shape == (len(pairs), 3) and probability.dtype == np.float32
        assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
        assert data["claim_ids"].tolist() == [owner[0] for owner in owners]
        assert data["passage_ids"].tolist() == [owner[1] for owner in owners]
        assert data["pair_kinds"].tolist() == [owner[2] for owner in owners]
        assert data["sentence_a"].tolist() == [owner[3] for owner in owners]
        assert data["sentence_b"].tolist() == [owner[4] for owner in owners]
        return probability.copy(), data["reused_old_pair"].astype(bool).copy()


def gpu_smoke():
    rows, _ = check_prepared()
    assert q.read(OUT / "CPU_CHECK.json")["status"] == "passed_waiting_for_GPU"
    assert not (OUT / "GPU_SMOKE.json").exists()
    selected = [0, len(rows) // 2, len(rows) - 1]
    pairs = []
    for index in selected:
        pairs.extend(row_pairs(rows[index])[0][:12])
    tokenizer = model = None; started = time.perf_counter()
    try:
        tokenizer, model, loaded = load_cuda()
        first, shapes = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        second, shapes2 = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        assert shapes == shapes2 and np.array_equal(first, second)
        save_npz(OUT / "GPU_SMOKE_PROBABILITIES.npz", probabilities=first)
        save_json(OUT / "GPU_SMOKE.json", {
            "status": "passed_no_automatic_extract", "selected_answer_indices": selected,
            "pairs": len(pairs), "batch_shapes": shapes, "same_path_repeat_exact": True,
            "loaded": loaded, **memory_stats(), "seconds": time.perf_counter() - started,
            "execution_signature_sha256": digest(runtime_signature()), "trained": False,
            "GPU_used": True, "official_test_opened": False,
        })
        print("ATOMIC_MICROCLAIM_NLI_GPU_SMOKE_PASSED", flush=True)
    finally:
        del model, tokenizer; clean_gpu()


def extract():
    rows, complete = check_prepared()
    smoke = q.read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_automatic_extract"
    assert smoke["execution_signature_sha256"] == digest(runtime_signature())
    folder = OUT / "pair_scores"; folder.mkdir(exist_ok=True)
    old_cache = old_pair_cache()
    missing_rows = []
    for index, row in enumerate(rows):
        path = folder / f"{row['response_id']}.npz"
        if path.exists(): validate_cache(path, row)
        else: missing_rows.append(index)
    tokenizer = model = None; started = time.perf_counter(); newly_inferred = reused = 0
    try:
        if missing_rows:
            tokenizer, model, loaded = load_cuda()
            for done, index in enumerate(missing_rows, 1):
                row = rows[index]; pairs, owners = row_pairs(row)
                probability = np.empty((len(pairs), 3), dtype=np.float32)
                reuse_mask = np.zeros(len(pairs), dtype=np.int8)
                need, need_indices = [], []
                for pair_index, pair in enumerate(pairs):
                    key = digest({"premise": pair[0], "hypothesis": pair[1]})
                    if key in old_cache:
                        probability[pair_index] = old_cache[key]; reuse_mask[pair_index] = 1; reused += 1
                    else:
                        need.append(pair); need_indices.append(pair_index)
                if need:
                    values, _ = infer_pairs(tokenizer, model, need, torch.device("cuda:0"))
                    probability[need_indices] = values; newly_inferred += len(need)
                signature = {key: np.asarray(value) for key, value in cache_signature(row).items()}
                save_npz(folder / f"{row['response_id']}.npz", **signature, probabilities=probability,
                         claim_ids=np.asarray([owner[0] for owner in owners], np.int32),
                         passage_ids=np.asarray([owner[1] for owner in owners], np.int8),
                         pair_kinds=np.asarray([owner[2] for owner in owners], np.int8),
                         sentence_a=np.asarray([owner[3] for owner in owners], np.int32),
                         sentence_b=np.asarray([owner[4] for owner in owners], np.int32),
                         reused_old_pair=reuse_mask)
                validate_cache(folder / f"{row['response_id']}.npz", row); memory_stats()
                if done == 1 or done % 25 == 0 or done == len(missing_rows):
                    progress = {"answers_complete": len(rows) - len(missing_rows) + done,
                                "answers_total": len(rows), "new_pairs_inferred": newly_inferred,
                                "old_pairs_reused": reused, "seconds_this_run": time.perf_counter() - started,
                                "pid": os.getpid()}
                    path = OUT / "progress.json"
                    if path.exists(): path.unlink()
                    save_json(path, progress)
                    print("ATOMIC_MICROCLAIM_NLI_EXTRACT", progress, flush=True)
        files = {}; total_reused = total_pairs = 0
        for row in rows:
            path = folder / f"{row['response_id']}.npz"
            probability, mask = validate_cache(path, row)
            files[path.name] = q.sha(path); total_reused += int(mask.sum()); total_pairs += len(mask)
        save_json(OUT / "extraction_complete.json", {
            "status": "complete_frozen_probabilities_not_scored", "answers": len(rows),
            "microclaims": complete["microclaims"], "pairs": total_pairs,
            "reused_old_pairs": total_reused, "new_model_pairs": total_pairs - total_reused,
            "files_sha256": files, "input_sha256": q.sha(OUT / "inputs.jsonl"),
            "protocol_sha256": q.sha(OUT / "protocol.json"), "model_sha256": MODEL_SHA256,
            "execution_signature_sha256": digest(runtime_signature()),
            "seconds_this_run": time.perf_counter() - started, "trained": False,
            "GPU_used": bool(missing_rows), "labels_accessed": False, "official_test_opened": False,
        })
        print("ATOMIC_MICROCLAIM_NLI_EXTRACTION_COMPLETE", total_pairs, total_reused, flush=True)
    finally:
        del model, tokenizer; clean_gpu()


def check_extracted(rows):
    complete = q.read(OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_probabilities_not_scored"
    assert complete["input_sha256"] == q.sha(OUT / "inputs.jsonl")
    assert complete["protocol_sha256"] == q.sha(OUT / "protocol.json")
    assert complete["execution_signature_sha256"] == digest(runtime_signature())
    for row in rows:
        path = OUT / "pair_scores" / f"{row['response_id']}.npz"
        assert q.sha(path) == complete["files_sha256"][path.name]
        validate_cache(path, row)
    return complete


def aggregate_features(row, probability):
    pairs, owners = row_pairs(row)
    assert probability.shape == (len(pairs), 3)
    by_claim = defaultdict(list)
    for index, owner in enumerate(owners): by_claim[owner[0]].append((index, owner))
    matrix = np.empty((len(row["claims"]), len(EVIDENCE_FEATURE_NAMES)), dtype=np.float32)
    raw = np.empty(len(row["claims"]), dtype=np.float64)
    lookup = evidence_lookup(row)
    for claim in row["claims"]:
        cid = claim["claim_id"]; entries = by_claim[cid]
        values = claim_self_features(claim).tolist()
        passage_best_e, passage_best_c = [], []
        all_probabilities = []
        cited = set(claim["explicit_passage_ids"] or claim["parent_passage_ids"])
        cited_prob, uncited_prob = [], []
        for source in claim["retrieval"]:
            pid = source["passage_id"]
            individual_indices = [index for index, owner in entries if owner[1] == pid and owner[2] > 0]
            joint_indices = [index for index, owner in entries if owner[1] == pid and owner[2] == 0]
            assert len(individual_indices) == len(source["selected"]) in (1, 2)
            one = probability[individual_indices]
            joint = probability[joint_indices[0]] if joint_indices else one[0]
            values += [float(one[:, 0].max()), float(one[:, 0].mean()),
                       float(one[:, 1].max()), float(one[:, 1].mean()),
                       float(one[:, 2].max()), float(one[:, 2].mean()),
                       float(joint[0]), float(joint[1]), float(joint[2]),
                       float(source["selected_union_query_coverage"]),
                       float(source["selected"][0]["bm25"]),
                       float(source["selected"][1]["bm25"] if len(source["selected"]) == 2 else source["selected"][0]["bm25"]),
                       float(source["selected"][0]["bm25"] - source["selected"][-1]["bm25"]),
                       float(pid in claim["explicit_passage_ids"]), float(pid in claim["parent_passage_ids"])]
            relation = np.asarray([relation_pair_features(claim, lookup[(pid, selected["sentence_id"])]["text"])
                                   for selected in source["selected"]], dtype=np.float32)
            values += relation.max(0).tolist() + relation.mean(0).tolist()
            combined = np.vstack((one, joint[None, :])) if joint_indices else one
            all_probabilities.append(combined)
            (cited_prob if pid in cited else uncited_prob).append(combined)
            passage_best_e.append(float(combined[:, 0].max())); passage_best_c.append(float(combined[:, 2].max()))
        all_probability = np.vstack(all_probabilities)
        for class_index in range(3):
            column = all_probability[:, class_index]
            values += [float(column.max()), float(column.mean()), float(column.min())]
        sorted_e = sorted(passage_best_e, reverse=True); sorted_c = sorted(passage_best_c, reverse=True)
        values += [sorted_e[0], sorted_e[1], sorted_e[0] - sorted_e[1],
                   sorted_c[0], sorted_c[1], sorted_c[0] - sorted_c[1]]
        best_source = int(np.argmax(passage_best_e)) + 1
        values += [float(best_source == pid) for pid in PASSAGE_IDS]
        if cited_prob:
            cp = np.vstack(cited_prob); up = np.vstack(uncited_prob) if uncited_prob else cp
            ce, ue, cc, uc = float(cp[:, 0].max()), float(up[:, 0].max()), float(cp[:, 2].max()), float(up[:, 2].max())
            values += [ce, ue, ce - ue, cc, uc, cc - uc]
        else:
            values += [0.0] * 6
        max_e, max_c = float(all_probability[:, 0].max()), float(all_probability[:, 2].max())
        lack = 1 - max_e
        values += [lack, max(lack, max_c), lack * max_c]
        assert len(values) == len(EVIDENCE_FEATURE_NAMES), (len(values), len(EVIDENCE_FEATURE_NAMES))
        matrix[cid] = values; raw[cid] = max(lack, max_c)
    assert np.isfinite(matrix).all() and np.isfinite(raw).all()
    return matrix, raw


def make_model(candidate):
    name = candidate.split("__", 1)[1]
    if name.startswith("lr_C"):
        c = float(name.split("C", 1)[1])
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", penalty="l2",
                                                                   max_iter=3000, random_state=SEED))
    if name == "hist_leaf7":
        return HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=7,
                                              min_samples_leaf=50, l2_regularization=1., early_stopping=False,
                                              random_state=SEED)
    if name == "hist_leaf15":
        return HistGradientBoostingClassifier(learning_rate=.04, max_iter=250, max_leaf_nodes=15,
                                              min_samples_leaf=50, l2_regularization=2., early_stopping=False,
                                              random_state=SEED)
    if name == "extra_depth8":
        return ExtraTreesClassifier(n_estimators=400, max_depth=8, min_samples_leaf=12,
                                    max_features=.5, n_jobs=4, random_state=SEED)
    raise KeyError(candidate)


def fit_model(model, x, y, weight):
    key = "logisticregression__sample_weight" if hasattr(model, "steps") else "sample_weight"
    model.fit(x, y, **{key: weight}); return model


def claim_weights(rows_by_id, response_ids, labels, active):
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        rid = response_ids[index]; tree[rows_by_id[rid]["group_id"]][rid].append(index)
    weights = np.zeros(len(labels), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values(): weights[indices] = 1 / (len(answers) * len(indices))
    mask = weights > 0; weights[mask] /= weights[mask].mean()
    mass = np.bincount(labels[mask], weights=weights[mask], minlength=2); assert np.all(mass > 0)
    weights[mask] *= (mass.sum() / (2 * mass))[labels[mask]]
    for answers in tree.values():
        indices = [i for values in answers.values() for i in values]
        weights[indices] *= (mask.sum() / len(tree)) / weights[indices].sum()
    weights[mask] *= mask.sum() / weights[mask].sum()
    return weights


def project(meta, rows, offsets, claim_scores):
    result = np.empty(len(meta["windows"]), dtype=np.float64)
    lookup = {row["response_id"]: row for row in rows}
    for answer in meta["answers"]:
        rid = answer["response_id"]; row = lookup[rid]; owners = row["lexical_token_microclaims"]
        assert len(owners) == answer["token_count"]
        for wi in meta["answer_windows"][rid]:
            cids = {int(cid) for ti in meta["windows"][wi]["token_indices"] for cid in owners[ti]}
            assert cids
            result[wi] = max(claim_scores[offsets[rid] + cid] for cid in cids)
    assert np.isfinite(result).all(); return result


def microclaim_whitebox(meta, rows):
    oof = np.load(OOF_WHITEBOX, mmap_mode="r")
    base = np.load(GENERATION, mmap_mode="r")
    assert oof.shape == (len(meta["windows"]), 2) and base.shape == (len(meta["windows"]), 1025)
    signals = np.column_stack((oof, np.asarray(base[:, -1]))).astype(np.float32)
    parts = []
    for row in rows:
        for claim in row["claims"]:
            windows = []
            token_ids = set(claim["lexical_token_indices"])
            for wi in meta["answer_windows"][row["response_id"]]:
                if token_ids.intersection(meta["windows"][wi]["token_indices"]): windows.append(wi)
            assert windows
            one = signals[windows]
            parts.append(np.concatenate((one.max(0), one.mean(0), one.min(0), one.std(0))))
    result = np.asarray(parts, dtype=np.float32)
    assert result.shape == (sum(len(row["claims"]) for row in rows), len(WHITEBOX_FEATURE_NAMES))
    return result


def fit_thresholds(meta, window_scores):
    left, right = meta["bounds"]["fit"]
    answer_scores = q.answer_scores(meta, window_scores)
    answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == "fit"]
    return {
        "window": q.choose_threshold([window["label"] for window in meta["windows"][left:right]], window_scores[left:right]),
        "answer": q.choose_threshold([meta["answers"][i]["label"] for i in answer_indices], answer_scores[answer_indices]),
    }


def cal_diagnostic(meta, window_scores):
    left, right = meta["bounds"]["calibration"]
    answer_scores = q.answer_scores(meta, window_scores)
    answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == "calibration"]
    thresholds = {
        "window": q.choose_threshold([window["label"] for window in meta["windows"][left:right]], window_scores[left:right]),
        "answer": q.choose_threshold([meta["answers"][i]["label"] for i in answer_indices], answer_scores[answer_indices]),
    }
    return {"thresholds": thresholds, "metrics": q.metrics(meta, window_scores, thresholds)["calibration"]}


def score():
    assert not torch.cuda.is_initialized()
    rows, _ = check_prepared(); extraction = check_extracted(rows)
    assert not (OUT / "score_started.json").exists()
    save_json(OUT / "score_started.json", {"status": "development_gold_opened_after_frozen_extraction",
              "extraction_sha256": q.sha(OUT / "extraction_complete.json"), "official_test_opened": False})
    started = time.perf_counter(); meta = q.metadata()
    assert [row["response_id"] for row in rows] == [answer["response_id"] for answer in meta["answers"]]
    by_id = {row["response_id"]: row for row in rows}
    features, raw_parts, response_ids, partitions, groups, labels = [], [], [], [], [], []
    offsets = {}; cursor = 0; partial = 0
    for answer_index, row in enumerate(rows):
        probability, _ = validate_cache(OUT / "pair_scores" / f"{row['response_id']}.npz", row)
        one, raw = aggregate_features(row, probability)
        offsets[row["response_id"]] = cursor; cursor += len(row["claims"])
        token = meta["by_response"][row["response_id"]]["tokens"]
        risk = np.asarray(token["risk_mask"], dtype=bool)
        for claim in row["claims"]:
            values = risk[claim["lexical_token_indices"]]
            labels.append(int(values.any())); partial += int(values.any() and not values.all())
            response_ids.append(row["response_id"]); partitions.append(row["partition"]); groups.append(row["group_id"])
        features.append(one); raw_parts.append(raw)
        if (answer_index + 1) % 100 == 0: print("ATOMIC_MICROCLAIM_FEATURES", answer_index + 1, 793, flush=True)
    evidence = np.vstack(features).astype(np.float32); raw_claim = np.concatenate(raw_parts)
    y = np.asarray(labels, dtype=np.int8); response_ids = np.asarray(response_ids)
    partitions = np.asarray(partitions); groups = np.asarray(groups)
    assert evidence.shape == (sum(EXPECTED_SCORED_MICROCLAIMS.values()), len(EVIDENCE_FEATURE_NAMES))
    whitebox = microclaim_whitebox(meta, rows)
    matrices = {"evidence": evidence, "fusion": np.column_stack((evidence, whitebox)).astype(np.float32)}
    fit = np.flatnonzero(partitions == "fit"); cal = np.flatnonzero(partitions == "calibration")
    folds = list(GroupKFold(FOLDS).split(fit, y[fit], groups[fit]))
    history = {}; stored = {}
    with threadpool_limits(limits=THREADS):
        for candidate_index, candidate in enumerate(CANDIDATES):
            family = candidate.split("__", 1)[0]; x = matrices[family]
            prediction = np.full(len(y), np.nan, dtype=np.float64); fold_rows = []
            for fold, (train_local, held_local) in enumerate(folds):
                train, held = fit[train_local], fit[held_local]
                weight = claim_weights(by_id, response_ids, y, train)
                model = fit_model(make_model(candidate), x[train], y[train], weight[train])
                prediction[held] = model.predict_proba(x[held])[:, 1]
                fold_rows.append({"fold": fold, "train_microclaims": len(train), "held_microclaims": len(held),
                                  "train_groups": len(set(groups[train])), "held_groups": len(set(groups[held]))})
            assert np.isfinite(prediction[fit]).all(); prediction[cal] = 0
            ws = project(meta, rows, offsets, prediction)
            thresholds = fit_thresholds(meta, ws); metrics = q.metrics(meta, ws, thresholds)["fit"]
            key = [min(metrics["windows"]["f1"], metrics["answers"]["f1"]), metrics["windows"]["f1"],
                   metrics["answers"]["f1"], metrics["windows"]["precision"], -candidate_index]
            history[candidate] = {"family": family, "fit_OOF_thresholds": thresholds,
                                  "fit_OOF_metrics": metrics, "selection_key": key, "folds": fold_rows}
            stored[candidate] = prediction
            print("ATOMIC_MICROCLAIM_OOF", candidate, round(metrics["windows"]["f1"], 6),
                  round(metrics["answers"]["f1"], 6), flush=True)
    selected = max(CANDIDATES, key=lambda name: history[name]["selection_key"])
    family = selected.split("__", 1)[0]; x = matrices[family]
    full_weight = claim_weights(by_id, response_ids, y, fit)
    with threadpool_limits(limits=THREADS):
        full_model = fit_model(make_model(selected), x[fit], y[fit], full_weight[fit])
    selected_claim = stored[selected]; selected_claim[cal] = full_model.predict_proba(x[cal])[:, 1]
    window_scores = project(meta, rows, offsets, selected_claim)
    thresholds = history[selected]["fit_OOF_thresholds"]
    strict = q.metrics(meta, window_scores, thresholds)
    common = cal_diagnostic(meta, window_scores)
    raw_window = project(meta, rows, offsets, raw_claim)
    raw_thresholds = fit_thresholds(meta, raw_window)
    raw_strict = q.metrics(meta, raw_window, raw_thresholds)
    raw_common = cal_diagnostic(meta, raw_window)
    answers = q.answer_scores(meta, window_scores)
    save_npz(OUT / "scores.npz", claim_scores=selected_claim, window_scores=window_scores,
             answer_scores=answers, raw_claim_scores=raw_claim, raw_window_scores=raw_window)
    np.save(OUT / "evidence_features.npy", evidence); np.save(OUT / "whitebox_features.npy", whitebox); np.save(OUT / "microclaim_labels.npy", y)
    (OUT / "model.pkl").write_bytes(pickle.dumps({"model": full_model, "selected": selected,
                                                   "feature_names": EVIDENCE_FEATURE_NAMES + (WHITEBOX_FEATURE_NAMES if family == "fusion" else ())}, protocol=5))
    save_json(OUT / "feature_names.json", {"evidence": list(EVIDENCE_FEATURE_NAMES),
              "whitebox": list(WHITEBOX_FEATURE_NAMES), "selected": list(EVIDENCE_FEATURE_NAMES + (WHITEBOX_FEATURE_NAMES if family == "fusion" else ()))})
    summary = {
        "status": "development_only_complete", "method": "atomic microclaim retrieved-evidence NLI relation readout",
        "scope": "original fit634 + calibration159 only", "microclaims": len(y),
        "positive_microclaims": int(y.sum()), "partial_positive_microclaims": partial,
        "NLI_pairs": extraction["pairs"], "reused_old_pairs": extraction["reused_old_pairs"],
        "new_model_pairs": extraction["new_model_pairs"], "candidates": history,
        "selected_fit_OOF_only": selected, "thresholds_from_fit_OOF": thresholds,
        "strict_fit_threshold_to_calibration": strict,
        "common_calibration_F1Opt_diagnostic": common,
        "raw_NLI": {"thresholds_from_fit": raw_thresholds, "strict": raw_strict,
                    "common_calibration_F1Opt_diagnostic": raw_common},
        "incumbent_read_only_reference": q.read(INCUMBENT_RESULT)["metrics"]["calibration"],
        "source_sha256": {str(path.relative_to(ROOT)): q.sha(path) for path in
                          (OOF_WHITEBOX, OOF_CONTROL, GENERATION, INCUMBENT_SCORES, INCUMBENT_RESULT)},
        "fit_only_model_and_threshold_selection": True, "calibration_used_for_selection": False,
        "formal_baselines_modified": False, "official_test_opened": False,
        "final_test_claim": False, "seconds": time.perf_counter() - started,
    }
    save_json(OUT / "summary.json", summary)
    sm, cm = strict["fit"], strict["calibration"]
    report = [
        "# Atomic microclaim retrieved-NLI v1", "",
        "只用原始 QA fit634+cal159。原子微主张按每个 passage 独立检索；冻结 ModernBERT-base-nli 评分；读出和阈值只由 source-group fit OOF 决定。calibration 只报告。", "",
        "| 结果 | fit OOF窗口F1 | cal严格窗口F1 | fit OOF整答F1 | cal严格整答F1 | cal-F1Opt窗口 | cal-F1Opt整答 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {selected} | {sm['windows']['f1']:.6f} | {cm['windows']['f1']:.6f} | {sm['answers']['f1']:.6f} | {cm['answers']['f1']:.6f} | {common['metrics']['windows']['f1']:.6f} | {common['metrics']['answers']['f1']:.6f} |",
        f"| raw_NLI | {raw_strict['fit']['windows']['f1']:.6f} | {raw_strict['calibration']['windows']['f1']:.6f} | {raw_strict['fit']['answers']['f1']:.6f} | {raw_strict['calibration']['answers']['f1']:.6f} | {raw_common['metrics']['windows']['f1']:.6f} | {raw_common['metrics']['answers']['f1']:.6f} |",
        "", f"NLI pair 共 {extraction['pairs']}，其中精确复用旧冻结 pair {extraction['reused_old_pairs']}，新推理 {extraction['new_model_pairs']}。",
        "", "正式 baseline 未修改；official test 未打开。cal-F1Opt 是反复使用开发集上的共同诊断，不是独立测试。扩展到 fit3680 必须另建冻结版本。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    names = ("summary.json", "REPORT.md", "scores.npz", "model.pkl", "feature_names.json",
             "evidence_features.npy", "whitebox_features.npy", "microclaim_labels.npy")
    save_json(OUT / "complete.json", {"status": "complete_development_only",
              "files_sha256": {name: q.sha(OUT / name) for name in names},
              "formal_baselines_modified": False, "official_test_opened": False,
              "final_test_claim": False})
    print("ATOMIC_MICROCLAIM_NLI_SCORE_COMPLETE", selected,
          cm["windows"]["f1"], cm["answers"]["f1"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "gpu-smoke", "extract", "score"))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if args.stage == "initialize": initialize()
        elif args.stage == "self-test": print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif args.stage == "prepare": prepare()
        elif args.stage == "check": check()
        elif args.stage == "gpu-smoke": gpu_smoke()
        elif args.stage == "extract": extract()
        else: score()
