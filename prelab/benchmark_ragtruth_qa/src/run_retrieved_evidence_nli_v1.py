"""Retrieved-evidence NLI candidate for native RAGTruth QA.

The preparation stages are label-free and CPU-only.  They freeze automatic
claim geometry, passage-sentence provenance, deterministic BM25 top-2
retrieval, and exact NLI pair text before any annotation value is inspected.
Model extraction and supervised readout are deliberately separate commands.
Official test paths are never addressed by this runner.
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
import tempfile
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import build_citation_alignment as citation
import feature_qa as fq
import run_development as q


ROOT = q.ROOT
DATA = ROOT / "data"
OUT = ROOT / "results/retrieved_evidence_nli_v1"
CLAIM_PLANS = ROOT / "semantic_baseline/cuda_variant/plans.jsonl"
LAYOUT_PLANS = DATA / "feature_preparation/plans.jsonl"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"

ROLE = "ours_method_candidate; never a baseline"
PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
PASSAGE_IDS = (1, 2, 3)
CLASSES = ("entailment", "neutral", "contradiction")
TOP_K = 2
BM25_K1 = 1.2
BM25_B = 0.75
PAIR_LIMIT = 2048
INFERENCE_BATCH = 16
THREADS = 4
FOLDS = 5
LR_C = 0.1
SEED = 20261013
BACKEND = "cuda_fp32_sdpa_math_batch16_v1"
PEAK_LIMIT = int(6.5 * 1024 ** 3)
MIN_FREE = 3 * 1024 ** 3

PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
WORD = re.compile(r"[^\W_]+", re.UNICODE)
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def feature_names():
    names = []
    for passage_id in PASSAGE_IDS:
        prefix = f"passage_{passage_id}"
        names += [
            f"{prefix}_max_entailment", f"{prefix}_mean_entailment",
            f"{prefix}_max_neutral", f"{prefix}_mean_neutral",
            f"{prefix}_max_contradiction", f"{prefix}_mean_contradiction",
            f"{prefix}_selected_union_query_coverage", f"{prefix}_top_bm25",
        ]
    return names + [
        "global_max_entailment", "global_mean_entailment",
        "global_max_neutral", "global_mean_neutral",
        "global_max_contradiction", "global_mean_contradiction",
        "global_max_selected_union_query_coverage",
        "global_mean_selected_union_query_coverage",
        "global_lack_of_entailment", "global_or_risk",
        "global_contradiction_x_lack_of_entailment",
    ]


FEATURE_NAMES = tuple(feature_names())


def protocol():
    return {
        "version": "native-qa-retrieved-evidence-nli-v1",
        "status": "refrozen_after_label_free_preparation_failure_01_before_any_label_or_model_access",
        "role": ROLE,
        "scope": {
            "answers": "Exactly the existing 634 fit and 159 calibration llama-2-7b-chat QA answers.",
            "claims": "Reuse exact automatic claim text/spans from semantic_baseline/cuda_variant/plans.jsonl. Retain punctuation-only records in an explicit audit list but do not treat them as factual claims or NLI hypotheses; this fixed isalnum rule uses no annotation.",
            "windows": "Evaluation later uses the unchanged eligible four-original-BPE stride-one windows; preparation derives only label-free token-to-claim geometry.",
            "test": "No official validation/test file or path is opened.",
        },
        "evidence_units": {
            "passages": "Parse exactly the original line-start passage 1/2/3 headers and retain body plus original document coordinates and hashes.",
            "sentences": "Within each passage body, split at newline or .!?/CJK equivalents followed by whitespace/end; retain closing punctuation, protect fixed abbreviations, initials, dotted acronyms and list markers; trim only outer whitespace.",
            "provenance": "Every selected item retains passage id, sentence id, passage-relative and document-relative character ranges, exact text and SHA256.",
        },
        "retrieval": {
            "unit": "Independently rank sentences inside each of the three passages for every answer claim.",
            "tokenization": "Unicode alphanumeric words, casefolded; no stemming, stopword removal, synonym expansion, question expansion or citation deletion.",
            "formula": "Robertson BM25: idf=ln(1+(N-df+0.5)/(df+0.5)), k1=1.2, b=0.75; sum once per unique query term.",
            "selection": "Top min(2,N) sentences per passage by descending BM25; exact ties use earlier sentence id. Zero-overlap claims still receive the deterministic first two sentences.",
            "labels": "Question labels, hallucination spans, token risk, prior detector scores and NLI outputs are unavailable to retrieval.",
        },
        "nli": {
            "checkpoint": "tasksource/ModernBERT-base-nli",
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256,
            "frozen": True,
            "pair": "Premise is one exact selected evidence sentence; hypothesis is one exact automatic answer claim. Source markers are metadata, not added to text.",
            "pair_order": "answer order, claim id, passage id 1/2/3, retrieval rank 1/2.",
            "class_order": list(CLASSES),
            "maximum_pair_tokens": PAIR_LIMIT,
            "truncation": False,
            "extraction": f"Separate explicit CUDA FP32 eval/inference command, {BACKEND}; batch {INFERENCE_BATCH}; no training.",
        },
        "claim_features": {
            "per_passage": "Over its selected one/two evidence pairs: max/mean E, max/mean N, max/mean C, union query-term coverage, and top BM25 score (8 x 3).",
            "global": "Across every selected pair/source: max/mean E, N and C; max/mean per-passage query coverage; 1-maxE; max(maxC,1-maxE); and maxC*(1-maxE) (11).",
            "width": len(FEATURE_NAMES),
            "names": list(FEATURE_NAMES),
            "raw_risk": "max(global maximum contradiction, 1-global maximum entailment).",
        },
        "readout": {
            "label": "Opened only after all frozen NLI caches exist. A scored fit claim is positive iff any original lexical BPE assigned to it has the unchanged human risk label.",
            "learned": "One StandardScaler plus L2 logistic regression, liblinear, fixed C=0.1 and seed; no architecture or hyperparameter selection.",
            "crossfit": "Five-fold GroupKFold by locked source-connected group supplies every fit claim prediction; full-fit model predicts calibration claims.",
            "weights": "Within each training fold: equal source-group mass, then equal answer mass, then equal claim mass; labels supply binary class balance.",
            "projection": "Assign each lexical original BPE its claim score. A four-BPE window receives the maximum score among its lexical BPE claims; answer score is its maximum eligible window.",
            "thresholds": "For raw and learned scores separately, select window and answer F1 thresholds only on fit (learned uses OOF fit predictions); ties use precision then higher threshold; freeze for calibration.",
            "report": "Report both raw and learned fit/calibration window and answer AUROC/AP/F1. Calibration is reporting only, never model/threshold/feature selection.",
        },
        "stage_gate": "initialize -> CPU prepare -> CPU check. After code review only: gpu-smoke -> extract. A separate CPU score command then opens development gold and trains the readout.",
        "preparation_repair": "First label-free prepare stopped because 7/8852 inherited automatic records contain punctuation only (``` or \"\":\") and therefore cannot own a lexical BPE or form a BM25 query. The failure is retained. Before any label/model access, v1 was refrozen to retain those 7 records in an audit list and score the other 8845 alphanumeric claims.",
        "limitations": "Retrieval failure and NLI failure remain conflated. Sentence-level claim broadcasting is intentionally coarser than token attribution. This is a new candidate, not a modified formal baseline.",
    }


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), f"No overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def words(text):
    return [x.casefold() for x in WORD.findall(text)]


def sentence_spans(text):
    assert text and any(char.isalnum() for char in text)
    cuts, i, line_start = [], 0, 0
    while i < len(text):
        char = text[i]
        if char in "\r\n":
            if char == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
            cuts.append(i + 1)
            line_start = i + 1
        elif char in TERMINAL:
            end = i + 1
            while end < len(text) and (text[end] in TERMINAL or text[end] in CLOSERS):
                end += 1
            prefix = text[line_start:i + 1].strip()
            followed_by_break = end == len(text) or text[end].isspace()
            match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final_token = match.group(0).lower() if match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final_token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final_token))
            protected = end < len(text) and (final_token in ABBREVIATIONS or dotted or initial)
            if followed_by_break and not LIST_PREFIX.fullmatch(prefix) and not protected:
                cuts.append(end)
                i = end - 1
        i += 1
    cuts.append(len(text))
    spans, start = [], 0
    for stop in sorted(set(cuts)):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(char.isalnum() for char in text[left:right]):
            spans.append([left, right])
        start = stop
    assert spans
    coverage = np.zeros(len(text), dtype=np.int8)
    for left, right in spans:
        coverage[left:right] += 1
    lexical = np.fromiter((char.isalnum() for char in text), dtype=bool)
    assert np.all(coverage[lexical] == 1) and np.all(coverage <= 1)
    return spans


def parse_passages(text):
    matches = list(PASSAGE_HEADER.finditer(text))
    assert [int(match.group(1)) for match in matches] == list(PASSAGE_IDS)
    passages = []
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while body_start < body_end and text[body_start].isspace():
            body_start += 1
        while body_end > body_start and text[body_end - 1].isspace():
            body_end -= 1
        body = text[body_start:body_end]
        assert body and any(char.isalnum() for char in body)
        sentence_rows = []
        for sentence_id, (left, right) in enumerate(sentence_spans(body)):
            sentence_rows.append({
                "sentence_id": sentence_id,
                "passage_char_start": left,
                "passage_char_end": right,
                "document_char_start": body_start + left,
                "document_char_end": body_start + right,
                "text": body[left:right],
                "text_sha256": digest(body[left:right]),
            })
        passages.append({
            "passage_id": int(match.group(1)),
            "header_char_start": match.start(),
            "header_char_end": match.end(),
            "body_char_start": body_start,
            "body_char_end": body_end,
            "body_sha256": digest(body),
            "sentences": sentence_rows,
        })
    return passages


def bm25_rank(query_text, sentence_rows):
    assert sentence_rows
    corpus = [words(row["text"]) for row in sentence_rows]
    query = sorted(set(words(query_text)))
    assert query
    lengths = np.asarray([len(tokens) for tokens in corpus], dtype=np.float64)
    assert np.all(lengths > 0)
    average = float(lengths.mean())
    document_frequency = Counter()
    for tokens in corpus:
        document_frequency.update(set(tokens))
    n = len(corpus)
    ranked = []
    query_set = set(query)
    for sentence_id, tokens in enumerate(corpus):
        frequency = Counter(tokens)
        score = 0.0
        for term in query:
            tf = frequency[term]
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * len(tokens) / average)
            score += idf * tf * (BM25_K1 + 1.0) / denominator
        token_set = set(tokens)
        coverage = len(query_set & token_set) / len(query_set)
        ranked.append({"sentence_id": sentence_id, "bm25": score, "query_term_coverage": coverage})
    ranked.sort(key=lambda row: (-row["bm25"], row["sentence_id"]))
    return ranked


def select_evidence(claim_text, passage):
    ranked = bm25_rank(claim_text, passage["sentences"])
    chosen = []
    for rank, item in enumerate(ranked[:TOP_K], 1):
        sentence = passage["sentences"][item["sentence_id"]]
        chosen.append({
            "rank": rank,
            "sentence_id": item["sentence_id"],
            "bm25": float(item["bm25"]),
            "query_term_coverage": float(item["query_term_coverage"]),
            "evidence_sha256": sentence["text_sha256"],
        })
    query = set(words(claim_text))
    selected_words = set()
    for item in chosen:
        selected_words.update(words(passage["sentences"][item["sentence_id"]]["text"]))
    return {
        "passage_id": passage["passage_id"],
        "selected": chosen,
        "selected_union_query_coverage": len(query & selected_words) / len(query),
    }


def trace_claim_tokens(text, offsets, claims):
    owners = np.full(len(offsets), -1, dtype=np.int32)
    inverse = [[] for _ in claims]
    lexical = []
    for token_index, (left, right) in enumerate(offsets):
        chars = [position for position in range(left, right) if text[position].isalnum()]
        lexical.append(bool(chars))
        if not chars:
            continue
        overlaps = [sum(claim["start"] <= position < claim["end"] for position in chars)
                    for claim in claims]
        owner = max(range(len(claims)), key=lambda index: (overlaps[index], -index))
        assert overlaps[owner] > 0
        assert all(any(claim["start"] <= position < claim["end"] for claim in claims)
                   for position in chars)
        owners[token_index] = owner
        inverse[owner].append(token_index)
    assert all(inverse), "Every retained automatic claim must own a lexical BPE"
    assert all((owner >= 0) == flag for owner, flag in zip(owners, lexical))
    return owners.tolist(), inverse


def reject_annotation_keys(value):
    forbidden = {"label", "labels", "gold", "risk", "risk_mask", "original_labels"}
    if isinstance(value, dict):
        assert not (forbidden & set(value)), forbidden & set(value)
        for child in value.values():
            reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def load_unlabelled_rows():
    rows, manifest = fq.development_rows()
    layout = {row["response_id"]: row for row in q.lines(LAYOUT_PLANS)}
    claims = {row["response_id"]: row for row in q.lines(CLAIM_PLANS)}
    assert len(rows) == len(layout) == len(claims) == 793
    ids = {row["response_id"] for row in rows}
    assert set(layout) == set(claims) == ids
    assert [row["partition"] for row in rows] == ["fit"] * 634 + ["calibration"] * 159
    assert all(item["labels_used"] is False for item in layout.values())
    return rows, layout, claims, manifest


def build_input_row(row, layout, plan):
    response_id = row["response_id"]
    text = row["original_response"]
    assert response_id == layout["response_id"] == plan["response_id"]
    assert text == layout["original_response"]
    assert row["answer_sha256"] == layout["answer_sha256"] == plan["answer_sha256"] == digest(text)
    assert plan["document_sha256"] == digest(row["retrieved_passages"])
    original_claims = plan["claims"]
    assert original_claims and all(text[claim["start"]:claim["end"]] == claim["text"] for claim in original_claims)
    scored_claims = [claim for claim in original_claims if any(char.isalnum() for char in claim["text"])]
    discarded = [{"original_claim_index": index, "char_start": claim["start"],
                  "char_end": claim["end"], "text": claim["text"],
                  "reason": "no_unicode_alphanumeric_character"}
                 for index, claim in enumerate(original_claims)
                 if not any(char.isalnum() for char in claim["text"])]
    offsets = layout["original"]["response_token_offsets"]
    owners, inverse = trace_claim_tokens(text, offsets, scored_claims)
    passages = parse_passages(row["retrieved_passages"])
    prepared_claims = []
    original_indices = [index for index, claim in enumerate(original_claims)
                        if any(char.isalnum() for char in claim["text"])]
    for claim_id, (original_claim_index, claim, token_indices) in enumerate(zip(original_indices, scored_claims, inverse)):
        parsed = citation.parse_citations(claim["text"])
        retrieval = [select_evidence(claim["text"], passage) for passage in passages]
        prepared_claims.append({
            "claim_id": claim_id,
            "original_claim_index": original_claim_index,
            "char_start": claim["start"],
            "char_end": claim["end"],
            "text": claim["text"],
            "text_sha256": digest(claim["text"]),
            "lexical_token_indices": token_indices,
            "citation_parse": parsed,
            "retrieval": retrieval,
        })
    result = {
        "response_id": response_id,
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "partition": row["partition"],
        "question": row["question"],
        "answer_sha256": row["answer_sha256"],
        "retrieved_passages_sha256": digest(row["retrieved_passages"]),
        "passages": passages,
        "claims": prepared_claims,
        "discarded_nonlexical_claims": discarded,
        "lexical_token_claim": owners,
        "raw_token_count": len(offsets),
        "response_token_offsets_sha256": digest(offsets),
        "original_token_ids_sha256": digest(layout["original"]["answer_token_ids"]),
        "labels_used": False,
        "method_role": ROLE,
    }
    reject_annotation_keys({key: value for key, value in result.items() if key != "labels_used"})
    return result


def evidence_lookup(row):
    lookup = {}
    for passage in row["passages"]:
        for sentence in passage["sentences"]:
            lookup[(passage["passage_id"], sentence["sentence_id"])] = sentence
    return lookup


def row_pairs(row):
    lookup = evidence_lookup(row)
    pairs, owners = [], []
    for claim in row["claims"]:
        assert [item["passage_id"] for item in claim["retrieval"]] == list(PASSAGE_IDS)
        for source in claim["retrieval"]:
            for selected in source["selected"]:
                key = (source["passage_id"], selected["sentence_id"])
                sentence = lookup[key]
                assert sentence["text_sha256"] == selected["evidence_sha256"]
                pairs.append((sentence["text"], claim["text"]))
                owners.append((claim["claim_id"], source["passage_id"], selected["rank"], selected["sentence_id"]))
    assert pairs and len(pairs) == len(owners)
    return pairs, owners


def pair_lengths(tokenizer, pairs):
    encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                        add_special_tokens=True, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    assert lengths and max(lengths) <= PAIR_LIMIT
    return lengths


def source_files():
    paths = [
        Path(__file__), Path(fq.__file__), Path(q.__file__), Path(citation.__file__),
        OUT / "protocol.json", DATA / "development_manifest.json", DATA / "fit.jsonl",
        DATA / "calibration.jsonl", LAYOUT_PLANS, CLAIM_PLANS,
        MODEL / "config.json", MODEL / "tokenizer.json", MODEL / "tokenizer_config.json",
        MODEL / "special_tokens_map.json", MODEL / "model.safetensors", MODEL / "download_manifest.json",
    ]
    return {str(path.resolve()): q.sha(path) for path in paths}


def synthetic_selfcheck():
    sample = "Dr. A arrived. First line\nSecond line!"
    spans = sentence_spans(sample)
    assert [sample[left:right] for left, right in spans] == ["Dr. A arrived.", "First line", "Second line!"]
    passage = {"passage_id": 1, "sentences": [
        {"sentence_id": 0, "text": "red fox runs", "text_sha256": digest("red fox runs")},
        {"sentence_id": 1, "text": "blue bird flies", "text_sha256": digest("blue bird flies")},
        {"sentence_id": 2, "text": "red bird rests", "text_sha256": digest("red bird rests")},
    ]}
    chosen = select_evidence("red bird", passage)
    assert [item["sentence_id"] for item in chosen["selected"]] == [2, 0]
    tied = select_evidence("absent term", passage)
    assert [item["sentence_id"] for item in tied["selected"]] == [0, 1]
    text = "Alpha. Beta!"
    claims = [{"start": 0, "end": 6}, {"start": 7, "end": 12}]
    owners, inverse = trace_claim_tokens(text, [[0, 5], [5, 6], [6, 7], [7, 11], [11, 12]], claims)
    assert owners == [0, -1, -1, 1, -1] and inverse == [[0], [3]]
    fake = {
        "passages": [{"passage_id": 1, "sentences": passage["sentences"]},
                     {"passage_id": 2, "sentences": passage["sentences"]},
                     {"passage_id": 3, "sentences": passage["sentences"]}],
        "claims": [{"claim_id": 0, "text": "red bird", "retrieval": []}],
    }
    for passage_id in PASSAGE_IDS:
        one = dict(chosen); one["passage_id"] = passage_id
        fake["claims"][0]["retrieval"].append(one)
    pairs, pair_owners = row_pairs(fake)
    assert len(pairs) == 6 and pair_owners[0] == (0, 1, 1, 2) and pair_owners[-1] == (0, 3, 2, 0)
    return {
        "status": "passed", "sentence_boundaries": True, "BM25_exact_tie_order": True,
        "top2_per_passage": True, "token_claim_trace": True, "pair_order_and_provenance": True,
        "pretrained_model_loaded": False, "GPU_initialized": torch.cuda.is_initialized(),
        "labels_accessed": False, "official_test_opened": False,
    }


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), "Do not overwrite an existing frozen result path"
    OUT.mkdir(parents=True)
    check = synthetic_selfcheck()
    q.save(OUT / "PREFLIGHT.json", check)
    q.save(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Retrieved-evidence NLI v1（我们的方法候选）\n\n"
        "每条原回答陈述在 passage 1/2/3 内分别用固定 BM25 选最多两句。每个选中证据句单独与陈述送入冻结 ModernBERT NLI，保留 E/N/C。"
        "证据编号、坐标、原文和哈希全保留。\n\n"
        "prepare/check 只做 CPU、无标签预处理；审核后才可单独运行 gpu-smoke 和 extract。score 再另起 CPU 进程读取开发标签，"
        "以 source group 五折交叉预测训练固定逻辑回归，将 claim 风险映射回原 4-BPE 窗口。正式 baseline 不改，本目录只属于新候选。\n",
        encoding="utf-8")
    print("RETRIEVED_EVIDENCE_NLI_PROTOCOL_FROZEN", flush=True)


def prepare():
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / "protocol.json") == protocol()
    assert q.read(OUT / "PREFLIGHT.json")["status"] == "passed"
    assert not (OUT / "prepare_started.json").exists(), "No silent preparation overwrite"
    snapshot = source_files()
    q.save(OUT / "prepare_started.json", {
        "status": "CPU_label_free_preparation_started", "source_sha256": snapshot,
        "annotation_values_accessed": False, "GPU_used": False, "official_test_opened": False,
    })
    rows, layouts, claims, _ = load_unlabelled_rows()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert q.read(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    prepared = []
    pair_count = claim_count = original_claim_count = sentence_count = lexical_count = 0
    maximum_pair_tokens = 0
    maximum_pair = None
    answer_claim_counts, passage_sentence_counts, zero_overlap = [], [], 0
    lengths_histogram = Counter()
    for answer_index, row in enumerate(rows):
        item = build_input_row(row, layouts[row["response_id"]], claims[row["response_id"]])
        pairs, owners = row_pairs(item)
        lengths = pair_lengths(tokenizer, pairs)
        for length, owner in zip(lengths, owners):
            lengths_histogram[length] += 1
            if length > maximum_pair_tokens:
                maximum_pair_tokens = length
                maximum_pair = {"response_id": row["response_id"], "owner": list(owner)}
        cursor = 0
        for claim in item["claims"]:
            for source in claim["retrieval"]:
                for selected in source["selected"]:
                    selected["pair_token_length"] = lengths[cursor]
                    zero_overlap += int(selected["bm25"] == 0.0)
                    cursor += 1
        assert cursor == len(lengths)
        prepared.append(item)
        pair_count += len(pairs)
        claim_count += len(item["claims"])
        original_claim_count += len(item["claims"]) + len(item["discarded_nonlexical_claims"])
        sentence_count += sum(len(passage["sentences"]) for passage in item["passages"])
        lexical_count += sum(owner >= 0 for owner in item["lexical_token_claim"])
        answer_claim_counts.append(len(item["claims"]))
        passage_sentence_counts.extend(len(passage["sentences"]) for passage in item["passages"])
        if (answer_index + 1) % 100 == 0:
            print("RETRIEVED_EVIDENCE_PREPARED", answer_index + 1, len(rows), flush=True)
    del tokenizer
    atomic_jsonl(OUT / "inputs.jsonl", prepared)
    assert Counter(item["partition"] for item in prepared) == EXPECTED_ANSWERS
    assert original_claim_count == 8852 and claim_count == 8845
    assert snapshot == source_files()
    statistics = {
        "status": "prepared_waiting_for_code_review",
        "role": ROLE,
        "answers": len(prepared),
        "answers_by_partition": dict(Counter(item["partition"] for item in prepared)),
        "original_automatic_claim_records": original_claim_count,
        "scored_factual_claims": claim_count,
        "discarded_nonlexical_claim_records": original_claim_count - claim_count,
        "evidence_sentences_available_across_answer_passages": sentence_count,
        "NLI_pairs": pair_count,
        "selected_pairs_with_zero_BM25_overlap": zero_overlap,
        "lexical_BPEs_traced_exactly_once": lexical_count,
        "claims_per_answer": {"min": min(answer_claim_counts), "median": float(np.median(answer_claim_counts)), "max": max(answer_claim_counts)},
        "sentences_per_passage": {"min": min(passage_sentence_counts), "median": float(np.median(passage_sentence_counts)), "max": max(passage_sentence_counts)},
        "pair_token_length": {"min": min(lengths_histogram), "median": None, "max": maximum_pair_tokens},
        "longest_pair": maximum_pair,
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "annotation_values_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    ordered_lengths = np.repeat(np.asarray(sorted(lengths_histogram), dtype=np.int32),
                                np.asarray([lengths_histogram[x] for x in sorted(lengths_histogram)], dtype=np.int64))
    statistics["pair_token_length"]["median"] = float(np.median(ordered_lengths))
    q.save(OUT / "preparation_statistics.json", statistics)
    q.save(OUT / "source_snapshot.json", {"files_sha256": snapshot, "official_test_opened": False})
    names = ["PREFLIGHT.json", "protocol.json", "PLAN.md", "inputs.jsonl",
             "preparation_statistics.json", "source_snapshot.json"]
    q.save(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_checked_inputs_not_model_extracted",
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "answers": 793, "original_claim_records": original_claim_count,
        "scored_claims": claim_count, "pairs": pair_count,
        "GPU_used": False, "new_fits": 0, "annotation_values_accessed": False,
        "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("RETRIEVED_EVIDENCE_NLI_ALL793_PREPARED_CPU_ONLY", flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_checked_inputs_not_model_extracted"
    assert complete["answers"] == 793 and complete["original_claim_records"] == 8852
    assert complete["scored_claims"] == 8845
    for name, expected in complete["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    assert q.read(OUT / "protocol.json") == protocol()
    for name, expected in q.read(OUT / "source_snapshot.json")["files_sha256"].items():
        assert q.sha(name) == expected, name
    rows = q.lines(OUT / "inputs.jsonl")
    assert len(rows) == 793
    assert sum(len(row["claims"]) for row in rows) == complete["scored_claims"]
    assert sum(len(row["discarded_nonlexical_claims"]) for row in rows) == 7
    assert sum(len(row_pairs(row)[0]) for row in rows) == complete["pairs"]
    return rows, complete


def check():
    rows, complete = check_prepared()
    first, middle, last = rows[0], rows[len(rows) // 2], rows[-1]
    sample = []
    for row in (first, middle, last):
        pairs, owners = row_pairs(row)
        sample.append({"response_id": row["response_id"], "claims": len(row["claims"]),
                       "pairs": len(pairs), "first_pair_owner": list(owners[0]),
                       "last_pair_owner": list(owners[-1]),
                       "all_sources_explicit": sorted({owner[1] for owner in owners}) == list(PASSAGE_IDS)})
    report = {
        "status": "passed_waiting_for_GPU_code_review",
        "preflight_replayed": synthetic_selfcheck(),
        "answers": len(rows), "original_claim_records": complete["original_claim_records"],
        "scored_claims": complete["scored_claims"], "pairs": complete["pairs"],
        "sample_wiring": sample,
        "input_hash_exact": q.sha(OUT / "inputs.jsonl") == q.read(OUT / "preparation_statistics.json")["input_sha256"],
        "source_provenance_present_for_every_pair": True,
        "labels_accessed": False, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(OUT / "CPU_CHECK.json", report)
    assert not torch.cuda.is_initialized()
    print("RETRIEVED_EVIDENCE_NLI_CPU_CHECK_PASSED", flush=True)


def runtime_signature():
    return {
        "backend": BACKEND,
        "source_sha256": q.sha(__file__),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "model_sha256": MODEL_SHA256,
        "revision": MODEL_REVISION,
        "batch": INFERENCE_BATCH,
        "precision": "float32",
        "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
        "cuda_runtime": torch.version.cuda,
        "role": ROLE,
    }


def infer_pairs(tokenizer, model, pairs, device):
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    outputs, shapes = [], []
    for left in range(0, len(pairs), INFERENCE_BATCH):
        batch = pairs[left:left + INFERENCE_BATCH]
        encoded = tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                            add_special_tokens=True, padding=True, truncation=False,
                            return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= PAIR_LIMIT
        shapes.append([len(batch), int(encoded["input_ids"].shape[1])])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                   if device.type == "cuda" else nullcontext())
        with torch.inference_mode(), torch.autocast(device.type, enabled=False), context:
            logits = model(**encoded).logits
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            outputs.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    assert result.shape == (len(pairs), 3)
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1, rtol=0, atol=2e-6)
    return result, shapes


def load_cuda():
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert q.read(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE, f"Insufficient free GPU memory: {free}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False).eval().requires_grad_(False).to("cuda:0")
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert model.config.max_position_embeddings == PAIR_LIMIT
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cuda" and
               not parameter.requires_grad for parameter in model.parameters())
    return tokenizer, model, {"device": torch.cuda.get_device_name(0), "free_before_load": free,
                              "total_memory": total, "parameters": sum(p.numel() for p in model.parameters())}


def memory_gate():
    values = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
              "reserved_peak_bytes": torch.cuda.max_memory_reserved(0)}
    assert max(values.values()) <= PEAK_LIMIT, values
    return values


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cache_signature(row):
    pairs, owners = row_pairs(row)
    return {
        "response_id": row["response_id"],
        "input_row_sha256": digest(row),
        "pair_definition_sha256": digest({"pairs": pairs, "owners": owners}),
        "input_file_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": MODEL_SHA256,
        "execution_signature_sha256": digest(runtime_signature()),
        "execution_backend": BACKEND,
        "method_role": ROLE,
    }


def validate_cache(path, row):
    pairs, owners = row_pairs(row)
    signature = cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(signature) | {"probabilities", "claim_ids", "passage_ids", "ranks", "sentence_ids"}
        for key, value in signature.items():
            assert data[key].item() == value, (path.name, key)
        probability = data["probabilities"]
        assert probability.dtype == np.float32 and probability.shape == (len(pairs), 3)
        assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
        assert data["claim_ids"].tolist() == [owner[0] for owner in owners]
        assert data["passage_ids"].tolist() == [owner[1] for owner in owners]
        assert data["ranks"].tolist() == [owner[2] for owner in owners]
        assert data["sentence_ids"].tolist() == [owner[3] for owner in owners]
        return probability.copy()


def gpu_smoke():
    rows, _ = check_prepared()
    assert q.read(OUT / "CPU_CHECK.json")["status"] == "passed_waiting_for_GPU_code_review"
    assert not (OUT / "GPU_SMOKE.json").exists()
    selected = [0, len(rows) // 2, len(rows) - 1]
    longest_id = q.read(OUT / "preparation_statistics.json")["longest_pair"]["response_id"]
    selected.append(next(index for index, row in enumerate(rows) if row["response_id"] == longest_id))
    selected = list(dict.fromkeys(selected))
    tokenizer = model = None
    started = time.perf_counter()
    try:
        tokenizer, model, loaded = load_cuda()
        pairs, owners = [], []
        for index in selected:
            one_pairs, one_owners = row_pairs(rows[index])
            pairs.extend(one_pairs)
            owners.extend((rows[index]["response_id"], *owner) for owner in one_owners)
        first, shapes = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        second, second_shapes = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
        assert shapes == second_shapes and np.array_equal(first, second)
        atomic_npz(OUT / "GPU_SMOKE_PROBABILITIES.npz", probabilities=first,
                   owner_json=np.asarray(json.dumps(owners, separators=(",", ":"))),
                   execution_signature_sha256=np.asarray(digest(runtime_signature())))
        q.save(OUT / "GPU_SMOKE.json", {
            "status": "passed_no_automatic_extract", "selected_answer_indices": selected,
            "selected_response_ids": [rows[index]["response_id"] for index in selected],
            "pairs": len(pairs), "batch_shapes": shapes, "same_path_repeat_exact": True,
            "loaded": loaded, **memory_gate(), "seconds": time.perf_counter() - started,
            "probabilities_sha256": q.sha(OUT / "GPU_SMOKE_PROBABILITIES.npz"),
            "execution_signature_sha256": digest(runtime_signature()),
            "trained": False, "GPU_used": True, "official_test_opened": False,
        })
        print("RETRIEVED_EVIDENCE_NLI_GPU_SMOKE_PASSED_NO_AUTOMATIC_EXTRACT", flush=True)
    finally:
        del model, tokenizer
        clean_gpu()


def extract():
    rows, complete = check_prepared()
    smoke = q.read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_automatic_extract"
    assert smoke["execution_signature_sha256"] == digest(runtime_signature())
    folder = OUT / "pair_scores"
    folder.mkdir(exist_ok=True)
    missing = []
    for index, row in enumerate(rows):
        path = folder / f"{row['response_id']}.npz"
        if path.exists():
            validate_cache(path, row)
        else:
            missing.append(index)
    tokenizer = model = None
    started = time.perf_counter()
    try:
        if missing:
            tokenizer, model, loaded = load_cuda()
            for completed, index in enumerate(missing, 1):
                row = rows[index]
                pairs, owners = row_pairs(row)
                probability, _ = infer_pairs(tokenizer, model, pairs, torch.device("cuda:0"))
                signature = {key: np.asarray(value) for key, value in cache_signature(row).items()}
                path = folder / f"{row['response_id']}.npz"
                atomic_npz(path, **signature, probabilities=probability,
                           claim_ids=np.asarray([owner[0] for owner in owners], np.int32),
                           passage_ids=np.asarray([owner[1] for owner in owners], np.int8),
                           ranks=np.asarray([owner[2] for owner in owners], np.int8),
                           sentence_ids=np.asarray([owner[3] for owner in owners], np.int32))
                validate_cache(path, row)
                memory_gate()
                if completed == 1 or completed % 25 == 0 or completed == len(missing):
                    progress = {"complete": len(rows) - len(missing) + completed, "total": len(rows),
                                "seconds_this_run": time.perf_counter() - started, "pid": os.getpid()}
                    q.save(OUT / "progress.json", progress)
                    print("RETRIEVED_EVIDENCE_NLI_EXTRACT", progress, flush=True)
        files = {}
        for row in rows:
            path = folder / f"{row['response_id']}.npz"
            validate_cache(path, row)
            files[path.name] = q.sha(path)
        q.save(OUT / "extraction_complete.json", {
            "status": "complete_frozen_probabilities_not_scored", "answers": len(rows),
            "claims": complete["scored_claims"], "pairs": complete["pairs"], "files_sha256": files,
            "input_sha256": q.sha(OUT / "inputs.jsonl"), "protocol_sha256": q.sha(OUT / "protocol.json"),
            "model_sha256": MODEL_SHA256, "execution_signature_sha256": digest(runtime_signature()),
            "new_rows_this_run": len(missing), "seconds_this_run": time.perf_counter() - started,
            "trained": False, "GPU_used": bool(missing), "annotation_values_accessed": False,
            "official_test_opened": False,
        })
        print("RETRIEVED_EVIDENCE_NLI_EXTRACTION_COMPLETE_NO_AUTOMATIC_SCORE", flush=True)
    finally:
        del model, tokenizer
        clean_gpu()


def check_extracted(rows):
    complete = q.read(OUT / "extraction_complete.json")
    assert complete["status"] == "complete_frozen_probabilities_not_scored"
    assert complete["answers"] == 793 and complete["claims"] == 8845
    assert complete["input_sha256"] == q.sha(OUT / "inputs.jsonl")
    assert complete["protocol_sha256"] == q.sha(OUT / "protocol.json")
    assert complete["execution_signature_sha256"] == digest(runtime_signature())
    for row in rows:
        path = OUT / "pair_scores" / f"{row['response_id']}.npz"
        assert q.sha(path) == complete["files_sha256"][path.name]
        validate_cache(path, row)
    return complete


def aggregate_claim_features(row, probability):
    _, owners = row_pairs(row)
    assert probability.shape == (len(owners), 3)
    by_claim = defaultdict(list)
    for index, owner in enumerate(owners):
        by_claim[owner[0]].append((index, owner))
    result = np.empty((len(row["claims"]), len(FEATURE_NAMES)), dtype=np.float32)
    raw = np.empty(len(row["claims"]), dtype=np.float64)
    for claim in row["claims"]:
        claim_id = claim["claim_id"]
        entries = by_claim[claim_id]
        values = []
        per_passage_coverage = []
        for source in claim["retrieval"]:
            indices = [index for index, owner in entries if owner[1] == source["passage_id"]]
            assert len(indices) == len(source["selected"]) in (1, 2)
            one = probability[indices]
            values += [float(one[:, 0].max()), float(one[:, 0].mean()),
                       float(one[:, 1].max()), float(one[:, 1].mean()),
                       float(one[:, 2].max()), float(one[:, 2].mean()),
                       float(source["selected_union_query_coverage"]),
                       float(source["selected"][0]["bm25"])]
            per_passage_coverage.append(float(source["selected_union_query_coverage"]))
        all_indices = [index for index, _ in entries]
        all_probability = probability[all_indices]
        max_entailment = float(all_probability[:, 0].max())
        max_contradiction = float(all_probability[:, 2].max())
        lack = 1.0 - max_entailment
        values += [max_entailment, float(all_probability[:, 0].mean()),
                   float(all_probability[:, 1].max()), float(all_probability[:, 1].mean()),
                   max_contradiction, float(all_probability[:, 2].mean()),
                   max(per_passage_coverage), float(np.mean(per_passage_coverage)),
                   lack, max(max_contradiction, lack), max_contradiction * lack]
        assert len(values) == len(FEATURE_NAMES)
        result[claim_id] = values
        raw[claim_id] = max(max_contradiction, lack)
    assert np.isfinite(result).all() and np.isfinite(raw).all()
    return result, raw


def nested_claim_weights(rows, claim_response_ids, y, train_indices):
    tree = defaultdict(lambda: defaultdict(list))
    for index in train_indices:
        response_id = claim_response_ids[index]
        row = rows[response_id]
        tree[row["group_id"]][response_id].append(index)
    weights = np.zeros(len(y), dtype=np.float64)
    for answers in tree.values():
        for indices in answers.values():
            for index in indices:
                weights[index] = 1.0 / (len(answers) * len(indices))
    active = weights > 0
    weights[active] /= weights[active].mean()
    mass = np.bincount(y[active], weights=weights[active], minlength=2)
    assert np.all(mass > 0)
    factors = mass.sum() / (2.0 * mass)
    weights[active] *= factors[y[active]]
    weights[active] *= active.sum() / weights[active].sum()
    return weights


def project_claim_scores(meta, prepared, claim_offsets, claim_scores):
    windows = np.empty(len(meta["windows"]), dtype=np.float64)
    lookup = {row["response_id"]: row for row in prepared}
    for answer in meta["answers"]:
        response_id = answer["response_id"]
        row = lookup[response_id]
        token = meta["by_response"][response_id]["tokens"]
        owners = np.asarray(row["lexical_token_claim"], dtype=np.int32)
        assert len(owners) == token["token_count"]
        assert (owners >= 0).tolist() == list(map(bool, token["lexical_mask"]))
        offset = claim_offsets[response_id]
        for window_index in meta["answer_windows"][response_id]:
            window = meta["windows"][window_index]
            claim_ids = {int(owners[token_index]) for token_index in window["token_indices"]
                         if owners[token_index] >= 0}
            assert claim_ids
            windows[window_index] = max(claim_scores[offset + claim_id] for claim_id in claim_ids)
    assert np.isfinite(windows).all()
    return windows


def score():
    assert not torch.cuda.is_initialized()
    prepared, _ = check_prepared()
    check_extracted(prepared)
    assert not (OUT / "score_started.json").exists(), "No silent score overwrite"
    q.save(OUT / "score_started.json", {"status": "development_gold_opened_after_frozen_extraction",
                                         "official_test_opened": False})
    meta = q.metadata()
    lookup = {row["response_id"]: row for row in prepared}
    assert set(lookup) == {answer["response_id"] for answer in meta["answers"]}
    matrices, raw_parts, claim_response_ids = [], [], []
    claim_offsets = {}
    cursor = 0
    for row in prepared:
        probability = validate_cache(OUT / "pair_scores" / f"{row['response_id']}.npz", row)
        features, raw = aggregate_claim_features(row, probability)
        claim_offsets[row["response_id"]] = cursor
        cursor += len(row["claims"])
        matrices.append(features)
        raw_parts.append(raw)
        claim_response_ids += [row["response_id"]] * len(row["claims"])
    x = np.concatenate(matrices).astype(np.float32)
    raw_claim = np.concatenate(raw_parts)
    assert x.shape == (8845, len(FEATURE_NAMES)) and len(raw_claim) == 8845
    y_claim = np.empty(8845, dtype=np.int8)
    partial_claims = 0
    for row in prepared:
        token = meta["by_response"][row["response_id"]]["tokens"]
        risk_mask = np.asarray(token["risk_mask"], dtype=bool)
        for claim in row["claims"]:
            indices = claim["lexical_token_indices"]
            values = risk_mask[indices]
            index = claim_offsets[row["response_id"]] + claim["claim_id"]
            y_claim[index] = int(values.any())
            partial_claims += int(values.any() and not values.all())
    fit_claim_indices = np.asarray([index for index, response_id in enumerate(claim_response_ids)
                                    if lookup[response_id]["partition"] == "fit"], dtype=np.int64)
    cal_claim_indices = np.asarray([index for index, response_id in enumerate(claim_response_ids)
                                    if lookup[response_id]["partition"] == "calibration"], dtype=np.int64)
    fit_groups = np.asarray([lookup[claim_response_ids[index]]["group_id"] for index in fit_claim_indices])
    group_folds = GroupKFold(n_splits=FOLDS)
    learned_claim = np.full(len(y_claim), np.nan, dtype=np.float64)
    fold_records = []
    for fold, (train_local, held_local) in enumerate(group_folds.split(fit_claim_indices, y_claim[fit_claim_indices], fit_groups)):
        train = fit_claim_indices[train_local]
        held = fit_claim_indices[held_local]
        weights = nested_claim_weights(lookup, claim_response_ids, y_claim, train)
        scaler = StandardScaler().fit(x[train], sample_weight=weights[train])
        model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                                   random_state=SEED).fit(scaler.transform(x[train]), y_claim[train],
                                                          sample_weight=weights[train])
        assert model.n_iter_.max() < 2000
        learned_claim[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
        fold_records.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                             "train_groups": len(set(fit_groups[train_local])),
                             "held_groups": len(set(fit_groups[held_local])),
                             "iterations": model.n_iter_.tolist()})
    assert np.isfinite(learned_claim[fit_claim_indices]).all()
    full_weights = nested_claim_weights(lookup, claim_response_ids, y_claim, fit_claim_indices)
    scaler = StandardScaler().fit(x[fit_claim_indices], sample_weight=full_weights[fit_claim_indices])
    model = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2", max_iter=2000,
                               random_state=SEED).fit(scaler.transform(x[fit_claim_indices]),
                                                      y_claim[fit_claim_indices],
                                                      sample_weight=full_weights[fit_claim_indices])
    learned_claim[cal_claim_indices] = model.predict_proba(scaler.transform(x[cal_claim_indices]))[:, 1]
    assert np.isfinite(learned_claim).all()
    raw_window = project_claim_scores(meta, prepared, claim_offsets, raw_claim)
    learned_window = project_claim_scores(meta, prepared, claim_offsets, learned_claim)
    fit_left, fit_right = meta["bounds"]["fit"]
    fit_answer_indices = [index for index, answer in enumerate(meta["answers"]) if answer["partition"] == "fit"]
    results = {}
    for name, window_scores in (("raw_or_risk", raw_window), ("claim_lr", learned_window)):
        answer_scores = q.answer_scores(meta, window_scores)
        thresholds = {
            "window": q.choose_threshold([window["label"] for window in meta["windows"][fit_left:fit_right]],
                                         window_scores[fit_left:fit_right]),
            "answer": q.choose_threshold([meta["answers"][index]["label"] for index in fit_answer_indices],
                                         answer_scores[fit_answer_indices]),
        }
        metrics = q.metrics(meta, window_scores, thresholds)
        score_path = OUT / f"{name}_scores.npz"
        np.savez_compressed(score_path, claim_scores=raw_claim if name == "raw_or_risk" else learned_claim,
                            window_scores=window_scores, answer_scores=answer_scores)
        results[name] = {"thresholds": thresholds, "metrics": metrics,
                         "scores_sha256": q.sha(score_path)}
    model_path = OUT / "claim_lr.pkl"
    model_path.write_bytes(pickle.dumps({"model": model, "scaler": scaler, "C": LR_C,
                                         "feature_names": FEATURE_NAMES, "folds": fold_records}, protocol=5))
    np.save(OUT / "claim_features.npy", x)
    np.save(OUT / "claim_labels.npy", y_claim)
    q.save(OUT / "summary.json", {
        "status": "development_only_complete", "role": ROLE,
        "claims": len(y_claim), "positive_claims": int(y_claim.sum()), "partial_positive_claims": partial_claims,
        "features": list(FEATURE_NAMES), "folds": fold_records, "results": results,
        "fit_thresholds_only": True, "calibration_used_for_selection": False,
        "official_test_opened": False, "final_test_claim": False,
    })
    report = [
        "# Retrieved-evidence NLI v1 开发结果", "",
        "每条陈述先在三个来源内固定检索最多两句，再由冻结 NLI 评分。逻辑回归只在 fit 的 source-group 五折预测上定阈值；calibration 只报告。", "",
        "| 方法 | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw_or_risk", "claim_lr"):
        metrics = results[name]["metrics"]
        report.append(f"| {name} | {metrics['fit']['windows']['f1']:.4f} | {metrics['calibration']['windows']['f1']:.4f} | {metrics['fit']['answers']['f1']:.4f} | {metrics['calibration']['answers']['f1']:.4f} |")
    report += ["", "这属于我们的方法候选；正式 baseline 的结构和参数没有改。句级广播会牺牲边界精度，测试仍封存。"]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    q.save(OUT / "complete.json", {
        "status": "complete_development_only", "files_sha256": {
            name: q.sha(OUT / name) for name in ("summary.json", "REPORT.md", "claim_lr.pkl",
                                                  "claim_features.npy", "claim_labels.npy",
                                                  "raw_or_risk_scores.npz", "claim_lr_scores.npz")},
        "official_test_opened": False, "final_test_claim": False,
    })
    print("RETRIEVED_EVIDENCE_NLI_DEVELOPMENT_SCORE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "gpu-smoke", "extract", "score"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            print(json.dumps(synthetic_selfcheck(), ensure_ascii=False, indent=2))
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            check()
        elif arguments.stage == "gpu-smoke":
            gpu_smoke()
        elif arguments.stage == "extract":
            extract()
        else:
            score()
