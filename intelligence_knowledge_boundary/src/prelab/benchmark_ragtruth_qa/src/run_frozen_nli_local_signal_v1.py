"""Frozen sentence/local-item NLI signal on native RAGTruth QA development data.

Preparation and NLI extraction never inspect annotation values. Gold is opened
only by ``score`` after all fixed NLI probabilities exist. Official test data is
never addressed by this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import pickle
from pathlib import Path
import re
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import feature_qa as fq
import run_development as q


ROOT = q.ROOT
DATA = ROOT / "data"
OUT = ROOT / "results/nli_local_signal_v1"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"
CURRENT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
CURRENT_META = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
VIEWS = ("full", "passage_1", "passage_2", "passage_3")
CLASSES = ("entailment", "neutral", "contradiction")
PAIR_LIMIT = 2048
INFERENCE_BATCH = 8
THREADS = 4
FOLDS = 5
LR_C = 0.1
SEED = 20261012
ROLE = "ours_method_candidate; never a baseline"
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def protocol():
    return {
        "version": "native-qa-frozen-local-nli-v1",
        "status": "frozen_before_full_extraction",
        "role": ROLE,
        "scope": {
            "answers": "Exactly the original 634 fit and 159 calibration llama-2-7b-chat answers.",
            "windows": "The existing eligible four-original-BPE, stride-one windows: 168123 fit and 42241 calibration.",
            "test": "Official validation/test paths are never opened.",
        },
        "segmentation": {
            "inputs": "Only original answer characters and frozen original-BPE offsets; no label, gold span, evidence annotation, score, or model output is consulted.",
            "rule": "Split at every newline and at .!?/CJK equivalents followed by whitespace/end, including adjacent closing quote/bracket characters. A line-initial numeric/alphabetic list marker such as 1. or a), a fixed common abbreviation, an initial, or a dotted acronym is kept with the following text. Trim boundary whitespace and discard fragments without an alphanumeric character.",
            "trace": "Every original lexical BPE is assigned exactly once to the segment containing the greatest number of its alphanumeric characters; ties use the earlier segment. The inverse segment-to-BPE lists are saved.",
        },
        "premises": {
            "views": list(VIEWS),
            "full": "The byte-for-byte released retrieved_passages string.",
            "individual": "Exact body slice after each passage N: header, with only outer whitespace removed; all three are required and retained.",
            "hypothesis": "One deterministic answer fragment; question and prior answer text are not appended.",
        },
        "nli": {
            "checkpoint": "tasksource/ModernBERT-base-nli",
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SHA256,
            "class_order": list(CLASSES),
            "frozen": True,
            "device": "CPU only, FP32, eval/inference_mode",
            "batch_pairs": INFERENCE_BATCH,
            "truncation": False,
            "maximum_pair_tokens": PAIR_LIMIT,
            "saved": "For every fragment and all four premise views, save entailment, neutral and contradiction probabilities.",
            "independent_replay": "A separate process must reproduce the fixed three-answer probabilities with maximum absolute FP32 difference <=2e-6; the in-process repeat remains bit exact.",
        },
        "window_projection": {
            "nli_features": "For each of the 12 view/class probabilities, average segment values weighted by the count of lexical BPEs from that segment in the frozen four-BPE window.",
            "raw_primary": "Maximum contradiction probability over all segments touched by the window and all four premise views; fixed before evaluation.",
            "answer": "Maximum over every eligible frozen window in the answer.",
        },
        "readouts": {
            "raw": "No fitted model. Choose separate window/answer F1 thresholds on fit only, then freeze and report fit plus calibration AUROC/AP/F1.",
            "nli_lr": "Only 12 frozen NLI probabilities; StandardScaler plus one L2 logistic regression with fixed C=0.1.",
            "fusion_lr": "The same fixed readout over the unchanged current window score plus the 12 NLI probabilities; report separately as ours.",
            "crossfit": "Five-fold GroupKFold by source-connected group produces every fit prediction for the two learned readouts. Thresholds use these fit OOF scores. Full-fit readouts predict calibration.",
            "weights": "Within each training fold, source group, answer and window receive equal nested mass; fit-fold labels supply class balancing. Calibration never affects weights.",
            "selection": "No C, feature, view, aggregation, fold count, epoch or formula search. Calibration is used once for reporting and never for selection or thresholding.",
        },
        "current_score_limit": "The current score is a previously fixed development candidate that was selected on this calibration split. Fusion therefore remains a development diagnostic and inherits that upstream selection history; no clean final-test claim is allowed.",
        "execution_gate": "Run CPU preparation, synthetic check, and three-real-answer wiring check first; obtain code/protocol review before full 793-answer extraction.",
    }


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def count_nonempty_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def source_files():
    paths = [
        Path(__file__), OUT / "protocol.json", DATA / "development_manifest.json",
        DATA / "fit.jsonl", DATA / "calibration.jsonl",
        DATA / "feature_preparation/plans.jsonl", DATA / "answers_fit.jsonl",
        DATA / "answers_calibration.jsonl", DATA / "tokens_fit.jsonl",
        DATA / "tokens_calibration.jsonl", DATA / "windows_k4_fit.jsonl",
        DATA / "windows_k4_calibration.jsonl", DATA / "gold_manifest.json",
        MODEL / "config.json", MODEL / "tokenizer.json", MODEL / "tokenizer_config.json",
        MODEL / "special_tokens_map.json", MODEL / "model.safetensors",
        MODEL / "download_manifest.json", CURRENT, CURRENT_META,
        ROOT / "research/nli_ec_smoke_v1/README.md",
        ROOT / "research/nli_ec_smoke_v1/run.py",
        ROOT / "src/verify_frozen_nli_local_signal_v1.py",
    ]
    return {str(path.resolve()): q.sha(path) for path in paths}


def parse_passages(text):
    matches = list(PASSAGE_HEADER.finditer(text))
    assert [m.group(1) for m in matches] == ["1", "2", "3"]
    views, ranges = [], []
    for i, match in enumerate(matches):
        left = match.end()
        right = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        assert left < right and any(char.isalnum() for char in text[left:right])
        views.append(text[left:right])
        ranges.append([left, right])
    assert all(text[a:b] == view for view, (a, b) in zip(views, ranges))
    return views, ranges


def answer_segments(text):
    assert text and any(char.isalnum() for char in text)
    cuts = []
    i = 0
    line_start = 0
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
            token_match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final_token = token_match.group(0).lower() if token_match else ""
            dotted_acronym = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final_token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final_token))
            abbreviation = end < len(text) and (final_token in ABBREVIATIONS or dotted_acronym or initial)
            if followed_by_break and not LIST_PREFIX.fullmatch(prefix) and not abbreviation:
                cuts.append(end)
                i = end - 1
        i += 1
    cuts.append(len(text))
    spans = []
    start = 0
    for stop in sorted(set(cuts)):
        assert start <= stop
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
    lexical_characters = np.fromiter((char.isalnum() for char in text), dtype=bool)
    assert np.all(coverage[lexical_characters] == 1)
    assert np.all(coverage <= 1)
    return spans


def trace_tokens(text, offsets, spans):
    offsets = [list(map(int, item)) for item in offsets]
    lexical_owner = [-1] * len(offsets)
    inverse = [[] for _ in spans]
    lexical = []
    for token_index, (left, right) in enumerate(offsets):
        assert 0 <= left < right <= len(text)
        is_lexical = any(text[pos].isalnum() for pos in range(left, right))
        lexical.append(is_lexical)
        if not is_lexical:
            continue
        overlap = []
        for segment_index, (a, b) in enumerate(spans):
            mass = sum(text[pos].isalnum() for pos in range(max(left, a), min(right, b)))
            overlap.append(mass)
        assert max(overlap) > 0
        owner = max(range(len(spans)), key=lambda j: (overlap[j], -j))
        lexical_owner[token_index] = owner
        inverse[owner].append(token_index)
    assert all((owner >= 0) == is_lexical for owner, is_lexical in zip(lexical_owner, lexical))
    assert sorted(index for one in inverse for index in one) == [i for i, flag in enumerate(lexical) if flag]
    return lexical_owner, inverse


def reject_annotation_keys(value):
    forbidden = {"label", "labels", "gold", "risk", "risk_mask", "original_labels"}
    if isinstance(value, dict):
        assert not (forbidden & set(value)), forbidden & set(value)
        for child in value.values():
            reject_annotation_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_annotation_keys(child)


def synthetic_check():
    sample = "Intro line:\n1. Alpha is true. Beta is false!\n- Final item"
    spans = answer_segments(sample)
    pieces = [sample[a:b] for a, b in spans]
    assert pieces == ["Intro line:", "1. Alpha is true.", "Beta is false!", "- Final item"], pieces
    words = [match.span() for match in re.finditer(r"[A-Za-z]+", sample)]
    owners, inverse = trace_tokens(sample, words, spans)
    assert len(owners) == len(words) and all(owner >= 0 for owner in owners)
    assert sum(map(len, inverse)) == len(words)
    passages = "passage 1: first fact\n\npassage 2:second fact\n\npassage 3: third fact\n"
    bodies, ranges = parse_passages(passages)
    assert bodies == ["first fact", "second fact", "third fact"]
    assert [passages[a:b] for a, b in ranges] == bodies
    print("FROZEN_NLI_LOCAL_SYNTHETIC_CHECK_PASSED", flush=True)


def load_unlabelled_rows():
    rows, manifest = fq.development_rows()
    plans = {row["response_id"]: row for row in q.lines(DATA / "feature_preparation/plans.jsonl")}
    assert len(rows) == len(plans) == 793
    assert [row["partition"] for row in rows] == ["fit"] * 634 + ["calibration"] * 159
    assert set(plans) == {row["response_id"] for row in rows}
    assert all(plan["labels_used"] is False for plan in plans.values())
    return rows, plans, manifest


def build_input_row(row, plan):
    assert row["response_id"] == plan["response_id"]
    assert row["original_response"] == plan["original_response"]
    assert digest(row["original_response"]) == row["answer_sha256"] == plan["answer_sha256"]
    passage_views, passage_ranges = parse_passages(row["retrieved_passages"])
    spans = answer_segments(row["original_response"])
    offsets = plan["original"]["response_token_offsets"]
    assert len(offsets) == len(plan["original"]["answer_token_ids"])
    lexical_owner, inverse = trace_tokens(row["original_response"], offsets, spans)
    segments = []
    for segment_id, ((left, right), token_indices) in enumerate(zip(spans, inverse)):
        assert token_indices
        segments.append({
            "segment_id": segment_id,
            "char_start": left,
            "char_end": right,
            "hypothesis": row["original_response"][left:right],
            "lexical_token_indices": token_indices,
        })
    result = {
        "response_id": row["response_id"],
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "partition": row["partition"],
        "answer_sha256": row["answer_sha256"],
        "full_premise": row["retrieved_passages"],
        "passage_premises": passage_views,
        "passage_character_ranges": passage_ranges,
        "premise_sha256": [digest(row["retrieved_passages"])] + [digest(x) for x in passage_views],
        "segments": segments,
        "lexical_token_segment": lexical_owner,
        "raw_token_count": len(offsets),
        "response_token_offsets_sha256": digest(offsets),
        "original_token_ids_sha256": digest(plan["original"]["answer_token_ids"]),
        "labels_used": False,
        "method_role": ROLE,
    }
    reject_annotation_keys({k: v for k, v in result.items() if k != "labels_used"})
    return result


def load_nli():
    assert not torch.cuda.is_initialized()
    manifest = q.read(MODEL / "download_manifest.json")
    assert manifest["revision"] == MODEL_REVISION
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32).eval().cpu()
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert model.config.max_position_embeddings == PAIR_LIMIT
    model.requires_grad_(False)
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())
    return tokenizer, model


def row_pairs(row):
    premises = [row["full_premise"]] + row["passage_premises"]
    assert len(premises) == len(VIEWS)
    pairs, owners = [], []
    for segment in row["segments"]:
        for view_index, premise in enumerate(premises):
            pairs.append((premise, segment["hypothesis"]))
            owners.append((segment["segment_id"], view_index))
    assert len(pairs) == len(row["segments"]) * len(VIEWS)
    return pairs, owners


def pair_lengths(tokenizer, pairs):
    encoded = tokenizer([x[0] for x in pairs], [x[1] for x in pairs],
                        add_special_tokens=True, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    assert lengths and max(lengths) <= PAIR_LIMIT
    return lengths


def infer_pairs(tokenizer, model, pairs):
    probabilities = []
    maximum = 0
    for left in range(0, len(pairs), INFERENCE_BATCH):
        batch = pairs[left:left + INFERENCE_BATCH]
        encoded = tokenizer([x[0] for x in batch], [x[1] for x in batch],
                            add_special_tokens=True, padding=True, truncation=False,
                            return_tensors="pt")
        maximum = max(maximum, int(encoded["attention_mask"].sum(1).max()))
        assert encoded["input_ids"].shape[1] <= PAIR_LIMIT
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            probabilities.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float32)
    assert result.shape == (len(pairs), len(CLASSES))
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1.0, atol=2e-6)
    return result, maximum


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    assert not path.exists(), "Protocol already exists"
    q.save(path, protocol())
    print("FROZEN_NLI_LOCAL_PROTOCOL_INITIALIZED", flush=True)


def prepare():
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / "protocol.json") == protocol()
    assert not (OUT / "prepare_started.json").exists(), "No silent overwrite"
    synthetic_check()
    rows, plans, manifest = load_unlabelled_rows()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert q.read(MODEL / "download_manifest.json")["revision"] == MODEL_REVISION
    for partition in PARTITIONS:
        assert count_nonempty_lines(DATA / f"windows_k4_{partition}.jsonl") == EXPECTED_WINDOWS[partition]
    snapshot = source_files()
    q.save(OUT / "prepare_started.json", {
        "role": ROLE,
        "source_sha256": snapshot,
        "annotation_values_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    prepared = []
    total_pairs = total_segments = total_lexical = 0
    maximum_pair_tokens = 0
    segment_counts = []
    for index, row in enumerate(rows):
        item = build_input_row(row, plans[row["response_id"]])
        pairs, _ = row_pairs(item)
        lengths = pair_lengths(tokenizer, pairs)
        item["pair_token_lengths"] = lengths
        prepared.append(item)
        total_pairs += len(pairs)
        total_segments += len(item["segments"])
        total_lexical += sum(owner >= 0 for owner in item["lexical_token_segment"])
        maximum_pair_tokens = max(maximum_pair_tokens, max(lengths))
        segment_counts.append(len(item["segments"]))
        if (index + 1) % 100 == 0:
            print("FROZEN_NLI_LOCAL_PREPARED", index + 1, len(rows), flush=True)
    del tokenizer
    write_jsonl(OUT / "inputs.jsonl", prepared)
    assert [Counter(x["partition"] for x in prepared)[p] for p in PARTITIONS] == [634, 159]
    assert snapshot == source_files()
    summary = {
        "status": "prepared_waiting_for_code_review",
        "role": ROLE,
        "answers": len(prepared),
        "answers_by_partition": dict(Counter(x["partition"] for x in prepared)),
        "segments": total_segments,
        "NLI_pairs": total_pairs,
        "lexical_BPEs_traced_exactly_once": total_lexical,
        "minimum_segments_per_answer": min(segment_counts),
        "median_segments_per_answer": float(np.median(segment_counts)),
        "maximum_segments_per_answer": max(segment_counts),
        "maximum_pair_tokens": maximum_pair_tokens,
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "source_snapshot_sha256": q.sha(OUT / "source_snapshot.json") if (OUT / "source_snapshot.json").exists() else None,
        "annotation_values_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(OUT / "source_snapshot.json", {"role": ROLE, "files_sha256": snapshot, "official_test_opened": False})
    summary["source_snapshot_sha256"] = q.sha(OUT / "source_snapshot.json")
    q.save(OUT / "preparation_complete.json", summary)
    assert not torch.cuda.is_initialized()
    print("FROZEN_NLI_LOCAL_ALL793_PREPARED", flush=True)


def check_prepared():
    assert q.read(OUT / "protocol.json") == protocol()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "prepared_waiting_for_code_review"
    assert complete["answers"] == 793 and complete["answers_by_partition"] == {"fit": 634, "calibration": 159}
    assert q.sha(OUT / "inputs.jsonl") == complete["input_sha256"]
    assert q.sha(OUT / "protocol.json") == complete["protocol_sha256"]
    assert q.sha(OUT / "source_snapshot.json") == complete["source_snapshot_sha256"]
    assert q.read(OUT / "source_snapshot.json")["files_sha256"] == source_files()
    rows = q.lines(OUT / "inputs.jsonl")
    assert len(rows) == 793
    return rows, complete


def real_check():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(THREADS)
    assert not (OUT / "CPU_WIRING_CHECK.json").exists(), "No silent check overwrite"
    assert not (OUT / "CPU_WIRING_PROBABILITIES.npz").exists(), "No silent check overwrite"
    rows, complete = check_prepared()
    selected_indices = [0, len(rows) // 2, len(rows) - 1]
    selected = [rows[index] for index in selected_indices]
    tokenizer, model = load_nli()
    pairs = []
    owner = []
    for row in selected:
        one, one_owner = row_pairs(row)
        pairs.extend(one)
        owner.extend((row["response_id"], *entry) for entry in one_owner)
    first, maximum = infer_pairs(tokenizer, model, pairs)
    second, repeated_maximum = infer_pairs(tokenizer, model, pairs)
    assert np.array_equal(first, second)
    protocol_sha = q.sha(OUT / "protocol.json")
    input_sha = q.sha(OUT / "inputs.jsonl")
    selected_row_sha = [digest(row) for row in selected]
    probability_path = OUT / "CPU_WIRING_PROBABILITIES.npz"
    np.savez_compressed(
        probability_path,
        probabilities=first,
        owner_response_ids=np.asarray([item[0] for item in owner]),
        owner_segment_ids=np.asarray([item[1] for item in owner], dtype=np.int32),
        owner_view_indices=np.asarray([item[2] for item in owner], dtype=np.int8),
        selected_response_ids=np.asarray([row["response_id"] for row in selected]),
        selected_input_row_sha256=np.asarray(selected_row_sha),
        input_file_sha256=np.asarray(input_sha),
        protocol_sha256=np.asarray(protocol_sha),
        model_sha256=np.asarray(MODEL_SHA256),
        method_role=np.asarray(ROLE),
    )
    examples = []
    for response_id in [row["response_id"] for row in selected]:
        indices = [i for i, item in enumerate(owner) if item[0] == response_id and item[1] == 0]
        examples.append({
            "response_id": response_id,
            "first_segment_four_views": [
                {name: float(value) for name, value in zip(CLASSES, first[index])}
                for index in indices
            ],
        })
    report = {
        "status": "three_real_answers_wiring_passed_waiting_for_review",
        "role": ROLE,
        "response_ids": [row["response_id"] for row in selected],
        "selection_rule": "First, middle, and last prepared answer; labels unavailable to check.",
        "segments": sum(len(row["segments"]) for row in selected),
        "pairs": len(pairs),
        "maximum_pair_tokens": maximum,
        "repeated_maximum_pair_tokens": repeated_maximum,
        "repeat_max_abs_difference": float(np.max(np.abs(first - second))),
        "probability_min": float(first.min()),
        "probability_max": float(first.max()),
        "probability_row_sum_max_error": float(np.max(np.abs(first.sum(1) - 1))),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "all_model_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "selected_input_row_sha256": selected_row_sha,
        "input_file_sha256": input_sha,
        "protocol_sha256": protocol_sha,
        "model_sha256": MODEL_SHA256,
        "probability_cache_sha256": q.sha(probability_path),
        "examples": examples,
        "preparation_sha256": q.sha(OUT / "preparation_complete.json"),
        "annotation_values_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(OUT / "CPU_WIRING_CHECK.json", report)
    assert complete["maximum_pair_tokens"] <= PAIR_LIMIT
    del first, second, model, tokenizer
    assert not torch.cuda.is_initialized()
    print("FROZEN_NLI_LOCAL_REAL3_CPU_CHECK_PASSED", flush=True)


def score_cache_signature(row):
    pairs, owners = row_pairs(row)
    return {
        "response_id": row["response_id"],
        "input_row_sha256": digest(row),
        "pair_definition_sha256": digest({"pairs": pairs, "owners": owners}),
        "input_file_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": MODEL_SHA256,
    }


def validate_score_file(path, row):
    expected = score_cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        assert data["method_role"].item() == ROLE
        assert data["response_id"].item() == expected["response_id"]
        for name in ("input_row_sha256", "pair_definition_sha256", "input_file_sha256",
                     "protocol_sha256", "model_sha256"):
            assert data[name].item() == expected[name], (path.name, name)
        assert data["segment_ids"].tolist() == list(range(len(row["segments"])))
        assert data["view_names"].tolist() == list(VIEWS)
        arrays = [data[name] for name in CLASSES]
        assert all(x.shape == (len(row["segments"]), len(VIEWS)) and x.dtype == np.float32 for x in arrays)
        total = sum(arrays)
        assert np.isfinite(total).all() and np.allclose(total, 1.0, atol=2e-6)
    return True


def extract():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(THREADS)
    rows, _ = check_prepared()
    check = q.read(OUT / "CPU_WIRING_CHECK.json")
    assert check["status"] == "three_real_answers_wiring_passed_waiting_for_review"
    directory = OUT / "segment_scores"
    directory.mkdir(exist_ok=True)
    tokenizer, model = load_nli()
    started = time.perf_counter()
    maximum_pair_tokens = 0
    completed = 0
    for index, row in enumerate(rows):
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", row["response_id"])
        path = directory / (row["response_id"] + ".npz")
        if path.exists():
            validate_score_file(path, row)
            completed += 1
            continue
        pairs, owners = row_pairs(row)
        probability, maximum = infer_pairs(tokenizer, model, pairs)
        maximum_pair_tokens = max(maximum_pair_tokens, maximum)
        cube = np.empty((len(row["segments"]), len(VIEWS), len(CLASSES)), dtype=np.float32)
        for one, (segment_id, view_index) in zip(probability, owners):
            cube[segment_id, view_index] = one
        pending = path.with_suffix(".npz.pending")
        with pending.open("wb") as handle:
            signature = score_cache_signature(row)
            np.savez_compressed(handle,
                response_id=np.asarray(signature["response_id"]),
                input_row_sha256=np.asarray(signature["input_row_sha256"]),
                pair_definition_sha256=np.asarray(signature["pair_definition_sha256"]),
                input_file_sha256=np.asarray(signature["input_file_sha256"]),
                protocol_sha256=np.asarray(signature["protocol_sha256"]),
                model_sha256=np.asarray(signature["model_sha256"]),
                method_role=np.asarray(ROLE),
                segment_ids=np.arange(len(row["segments"]), dtype=np.int32),
                view_names=np.asarray(VIEWS),
                entailment=cube[:, :, 0], neutral=cube[:, :, 1], contradiction=cube[:, :, 2])
        pending.replace(path)
        validate_score_file(path, row)
        completed += 1
        if completed % 25 == 0:
            print("FROZEN_NLI_LOCAL_EXTRACTED", completed, len(rows), "seconds", round(time.perf_counter() - started, 1), flush=True)
    files = {row["response_id"] + ".npz": q.sha(directory / (row["response_id"] + ".npz")) for row in rows}
    assert len(files) == 793
    result = {
        "status": "complete_frozen_probabilities_not_scored",
        "role": ROLE,
        "answers": len(rows),
        "segments": sum(len(row["segments"]) for row in rows),
        "pairs": sum(len(row["segments"]) * len(VIEWS) for row in rows),
        "maximum_pair_tokens_observed_this_run": maximum_pair_tokens,
        "class_order": list(CLASSES),
        "view_order": list(VIEWS),
        "files_sha256": files,
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": q.sha(MODEL / "model.safetensors"),
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "annotation_values_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(OUT / "extraction_complete.json", result)
    del model, tokenizer
    assert not torch.cuda.is_initialized()
    print("FROZEN_NLI_LOCAL_ALL793_EXTRACTED", flush=True)


def check_extracted(rows):
    result = q.read(OUT / "extraction_complete.json")
    assert result["status"] == "complete_frozen_probabilities_not_scored"
    assert result["answers"] == 793 and result["input_sha256"] == q.sha(OUT / "inputs.jsonl")
    assert result["protocol_sha256"] == q.sha(OUT / "protocol.json")
    assert result["model_sha256"] == MODEL_SHA256
    directory = OUT / "segment_scores"
    for row in rows:
        name = row["response_id"] + ".npz"
        path = directory / name
        assert q.sha(path) == result["files_sha256"][name]
        validate_score_file(path, row)
    return result


def load_segment_features(row):
    path = OUT / "segment_scores" / (row["response_id"] + ".npz")
    with np.load(path, allow_pickle=False) as data:
        cube = np.stack([data[name] for name in CLASSES], axis=-1).astype(np.float32)
    assert cube.shape == (len(row["segments"]), len(VIEWS), len(CLASSES))
    return cube.reshape(len(row["segments"]), -1), cube[:, :, 2]


def build_window_design(meta, prepared):
    lookup = {row["response_id"]: row for row in prepared}
    assert set(lookup) == {answer["response_id"] for answer in meta["answers"]}
    n = len(meta["windows"])
    features = np.empty((n, len(VIEWS) * len(CLASSES)), dtype=np.float32)
    raw = np.empty(n, dtype=np.float64)
    for answer_index, answer in enumerate(meta["answers"]):
        response_id = answer["response_id"]
        row = lookup[response_id]
        token = meta["by_response"][response_id]["tokens"]
        owner = np.asarray(row["lexical_token_segment"], dtype=np.int32)
        assert len(owner) == token["token_count"]
        assert (owner >= 0).tolist() == list(map(bool, token["lexical_mask"]))
        segment_features, contradiction = load_segment_features(row)
        for window_index in meta["answer_windows"][response_id]:
            window = meta["windows"][window_index]
            owned = [int(owner[j]) for j in window["token_indices"] if owner[j] >= 0]
            assert owned
            counts = Counter(owned)
            segment_ids = np.asarray(sorted(counts), dtype=np.int64)
            weights = np.asarray([counts[j] for j in segment_ids], dtype=np.float64)
            weights /= weights.sum()
            features[window_index] = weights @ segment_features[segment_ids]
            raw[window_index] = float(contradiction[segment_ids].max())
        if (answer_index + 1) % 100 == 0:
            print("FROZEN_NLI_LOCAL_WINDOW_MAP", answer_index + 1, len(meta["answers"]), flush=True)
    assert np.isfinite(features).all() and np.isfinite(raw).all()
    return features, raw


def replay_token_window_identity(meta, prepared):
    """Bind prepared segment ownership back to exact frozen BPE/window metadata."""
    plans = {row["response_id"]: row for row in q.lines(DATA / "feature_preparation/plans.jsonl")}
    lookup = {row["response_id"]: row for row in prepared}
    assert set(plans) == set(lookup) == {row["response_id"] for row in meta["answers"]}
    for answer in meta["answers"]:
        response_id = answer["response_id"]
        prepared_row = lookup[response_id]
        original = plans[response_id]["original"]
        token = meta["by_response"][response_id]["tokens"]
        assert prepared_row["answer_sha256"] == token["answer_sha256"] == plans[response_id]["answer_sha256"]
        assert prepared_row["raw_token_count"] == token["token_count"] == len(original["answer_token_ids"])
        assert prepared_row["response_token_offsets_sha256"] == digest(original["response_token_offsets"])
        assert prepared_row["original_token_ids_sha256"] == digest(original["answer_token_ids"])
        assert token["token_ids"] == original["answer_token_ids"]
        assert token["answer_token_positions"] == original["answer_token_positions"]
        assert token["response_token_offsets"] == original["response_token_offsets"]
        assert token["response_token_offsets_raw"] == original["response_token_offsets_raw"]
        assert (np.asarray(prepared_row["lexical_token_segment"]) >= 0).tolist() == token["lexical_mask"]
        for window_index in meta["answer_windows"][response_id]:
            window = meta["windows"][window_index]
            indices = window["token_indices"]
            assert window["token_ids"] == [token["token_ids"][j] for j in indices]
            assert window["answer_token_positions"] == [token["answer_token_positions"][j] for j in indices]
            assert window["character_intervals"] == [token["response_token_offsets"][j] for j in indices]
    report = {
        "role": ROLE,
        "answers_exact": len(meta["answers"]),
        "windows_exact": len(meta["windows"]),
        "prepared_offset_and_token_ID_hashes_replayed": True,
        "plan_to_gold_token_metadata_exact": True,
        "window_token_ID_position_and_character_intervals_exact": True,
        "lexical_segment_mask_exact": True,
        "official_test_opened": False,
    }
    q.save(OUT / "TOKEN_WINDOW_IDENTITY.json", report)
    return report


def nested_weights(meta, indices):
    indices = np.asarray(indices, dtype=np.int64)
    tree = defaultdict(lambda: defaultdict(list))
    for local, global_index in enumerate(indices):
        row = meta["windows"][int(global_index)]
        tree[row["group_id"]][row["answer_id"]].append(local)
    base = np.empty(len(indices), dtype=np.float64)
    for answers in tree.values():
        for local_indices in answers.values():
            base[local_indices] = 1.0 / (len(answers) * len(local_indices))
    base /= base.mean()
    y = np.asarray([meta["windows"][int(index)]["label"] for index in indices], dtype=np.int64)
    mass = np.bincount(y, weights=base, minlength=2)
    assert np.all(mass > 0)
    factors = mass.sum() / (2.0 * mass)
    loss = base * factors[y]
    target_group_mass = len(indices) / len(tree)
    for answers in tree.values():
        local_indices = [item for items in answers.values() for item in items]
        loss[local_indices] *= target_group_mass / loss[local_indices].sum()
    loss *= len(indices) / loss.sum()
    return base, loss


def one_readout(x, meta, name):
    fit_end = EXPECTED_WINDOWS["fit"]
    fit_indices = np.arange(fit_end, dtype=np.int64)
    y = np.asarray([row["label"] for row in meta["windows"][:fit_end]], dtype=np.int64)
    groups = np.asarray([row["group_id"] for row in meta["windows"][:fit_end]])
    oof = np.empty(fit_end, dtype=np.float64)
    fold_id = np.full(fit_end, -1, dtype=np.int8)
    fold_records = []
    splitter = GroupKFold(n_splits=FOLDS)
    for fold, (train_local, held_local) in enumerate(splitter.split(x[:fit_end], y, groups)):
        train_global = fit_indices[train_local]
        base, loss = nested_weights(meta, train_global)
        scaler = StandardScaler().fit(x[train_global], sample_weight=base)
        classifier = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2",
                                        max_iter=2000, random_state=SEED)
        classifier.fit(scaler.transform(x[train_global]), y[train_local], sample_weight=loss)
        assert classifier.n_iter_.max() < 2000
        oof[held_local] = classifier.predict_proba(scaler.transform(x[held_local]))[:, 1]
        fold_id[held_local] = fold
        fold_records.append({
            "fold": fold,
            "train_windows": len(train_local),
            "held_windows": len(held_local),
            "train_groups": len(set(groups[train_local])),
            "held_groups": len(set(groups[held_local])),
            "iterations": classifier.n_iter_.tolist(),
            "coefficient": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        })
    assert np.all(fold_id >= 0) and np.isfinite(oof).all()
    base, loss = nested_weights(meta, fit_indices)
    scaler = StandardScaler().fit(x[:fit_end], sample_weight=base)
    classifier = LogisticRegression(C=LR_C, solver="liblinear", penalty="l2",
                                    max_iter=2000, random_state=SEED)
    classifier.fit(scaler.transform(x[:fit_end]), y, sample_weight=loss)
    assert classifier.n_iter_.max() < 2000
    scores = np.empty(len(x), dtype=np.float64)
    scores[:fit_end] = oof
    scores[fit_end:] = classifier.predict_proba(scaler.transform(x[fit_end:]))[:, 1]
    artifact = {
        "role": ROLE,
        "name": name,
        "features": int(x.shape[1]),
        "C": LR_C,
        "folds": FOLDS,
        "seed": SEED,
        "fit_prediction": "group-disjoint OOF",
        "calibration_prediction": "one full-fit model",
        "fold_id": fold_id,
        "fold_records": fold_records,
        "full_scaler": scaler,
        "full_classifier": classifier,
    }
    return scores, artifact


def fit_thresholds(meta, scores):
    fit_end = EXPECTED_WINDOWS["fit"]
    answer = q.answer_scores(meta, scores)
    return {
        "window": q.choose_threshold([row["label"] for row in meta["windows"][:fit_end]], scores[:fit_end]),
        "answer": q.choose_threshold([row["label"] for row in meta["answers"][:EXPECTED_ANSWERS["fit"]]],
                                     answer[:EXPECTED_ANSWERS["fit"]]),
    }


def result_record(meta, scores, thresholds, fit_prediction):
    return {
        "role": ROLE,
        "threshold_source": "fit only",
        "fit_prediction": fit_prediction,
        "thresholds": thresholds,
        "metrics": q.metrics(meta, scores, thresholds),
    }


def score():
    assert not torch.cuda.is_initialized()
    completion_path = OUT / "complete.json"
    if completion_path.exists():
        completed = q.read(completion_path)
        assert completed["status"] == "complete_development_only"
        for name, expected in completed["files_sha256"].items():
            assert q.sha(OUT / name) == expected, name
        assert completed["protocol_sha256"] == q.sha(OUT / "protocol.json")
        assert completed["input_sha256"] == q.sha(OUT / "inputs.jsonl")
        assert completed["extraction_sha256"] == q.sha(OUT / "extraction_complete.json")
        raise RuntimeError("Existing complete scoring output verified; refusing overwrite")
    score_outputs = ("window_nli_features.npy", "predictions.npz", "nli_only_fixed_lr.pkl",
                     "current_plus_nli_fixed_lr.pkl", "fit_freeze_before_calibration_report.json",
                     "TOKEN_WINDOW_IDENTITY.json", "summary.json", "REPORT.md")
    assert not any((OUT / name).exists() for name in score_outputs), "Partial score output exists; refusing overwrite"
    prepared, _ = check_prepared()
    extraction = check_extracted(prepared)
    meta = q.metadata()
    assert [row["response_id"] for row in prepared] == [row["response_id"] for row in meta["answers"]]
    token_window_identity = replay_token_window_identity(meta, prepared)
    features, raw = build_window_design(meta, prepared)
    current_file = np.load(CURRENT, allow_pickle=False)
    current = current_file["window_scores"].astype(np.float64)
    assert current.shape == raw.shape
    assert np.allclose(current_file["answer_scores"], q.answer_scores(meta, current), rtol=0, atol=0)
    current_file.close()
    nli_scores, nli_artifact = one_readout(features, meta, "nli_only_fixed_lr")
    fusion_features = np.column_stack((current, features)).astype(np.float32)
    fusion_scores, fusion_artifact = one_readout(fusion_features, meta, "current_plus_nli_fixed_lr")
    methods = {
        "raw_max_contradiction": (raw, "direct fit scores; no model"),
        "nli_only_fixed_lr": (nli_scores, "five-fold group OOF"),
        "current_plus_nli_fixed_lr": (fusion_scores, "five-fold group OOF readout; upstream current score is not crossfit"),
    }
    thresholds = {name: fit_thresholds(meta, scores) for name, (scores, _) in methods.items()}
    np.save(OUT / "window_nli_features.npy", features)
    np.savez_compressed(OUT / "predictions.npz", method_role=np.asarray(ROLE), raw_max_contradiction=raw,
                        nli_only_fixed_lr=nli_scores, current_plus_nli_fixed_lr=fusion_scores,
                        current_score=current)
    with (OUT / "nli_only_fixed_lr.pkl").open("wb") as handle:
        pickle.dump(nli_artifact, handle, protocol=5)
    with (OUT / "current_plus_nli_fixed_lr.pkl").open("wb") as handle:
        pickle.dump(fusion_artifact, handle, protocol=5)
    freeze = {
        "status": "thresholds_and_models_frozen_from_fit_before_calibration_reporting",
        "role": ROLE,
        "thresholds": thresholds,
        "model_files_sha256": {
            "nli_only_fixed_lr.pkl": q.sha(OUT / "nli_only_fixed_lr.pkl"),
            "current_plus_nli_fixed_lr.pkl": q.sha(OUT / "current_plus_nli_fixed_lr.pkl"),
        },
        "predictions_sha256": q.sha(OUT / "predictions.npz"),
        "features_sha256": q.sha(OUT / "window_nli_features.npy"),
        "calibration_used_for_selection": False,
        "official_test_opened": False,
    }
    q.save(OUT / "fit_freeze_before_calibration_report.json", freeze)
    results = {name: result_record(meta, scores, thresholds[name], fit_kind)
               for name, (scores, fit_kind) in methods.items()}
    summary = {
        "status": "complete_development_only",
        "role": "All three reported variants are ours method candidates; none is a baseline.",
        "scope": protocol()["scope"],
        "extraction": {key: extraction[key] for key in ("answers", "segments", "pairs")},
        "feature_order": [f"{view}_{label}" for view in VIEWS for label in CLASSES],
        "results": results,
        "token_window_identity": token_window_identity,
        "current_source": {
            "candidate": "semantic_claim__old_tree__large_weight0.4",
            "score_sha256": q.sha(CURRENT),
            "metadata_sha256": q.sha(CURRENT_META),
            "selection_limit": protocol()["current_score_limit"],
        },
        "calibration_used_for_new_selection": False,
        "official_test_opened": False,
        "final_test_claim": False,
    }
    q.save(OUT / "summary.json", summary)
    report = [
        "# 冻结NLI局部信号：原生QA开发结果",
        "",
        "所有阈值和两个固定LR均只由fit决定；calibration只报告一次。官方test未读取。",
        "",
        "| 方法 | cal窗口 AUROC | AP | F1 | cal整答 AUROC | AP | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in methods:
        window = results[name]["metrics"]["calibration"]["windows"]
        answer = results[name]["metrics"]["calibration"]["answers"]
        report.append(f"| {name} | {window['auroc']:.3f} | {window['average_precision']:.3f} | {window['f1']:.3f} | {answer['auroc']:.3f} | {answer['average_precision']:.3f} | {answer['f1']:.3f} |")
    report += [
        "",
        "三项均为我们的方法候选，不属于基线：raw是固定的跨片段/四资料视图最大矛盾概率；NLI LR只读12个冻结概率；融合LR再加入既有current窗口分数。",
        "融合继承current候选曾用本calibration选定的历史，只能作为开发诊断，不能视为独立最终成绩。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_names = ["window_nli_features.npy", "predictions.npz", "nli_only_fixed_lr.pkl",
                    "current_plus_nli_fixed_lr.pkl", "fit_freeze_before_calibration_report.json",
                    "TOKEN_WINDOW_IDENTITY.json", "summary.json", "REPORT.md"]
    q.save(OUT / "complete.json", {
        "status": "complete_development_only",
        "role": ROLE,
        "files_sha256": {name: q.sha(OUT / name) for name in output_names},
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "extraction_sha256": q.sha(OUT / "extraction_complete.json"),
        "calibration_used_for_new_selection": False,
        "official_test_opened": False,
        "final_test_claim": False,
    })
    assert not torch.cuda.is_initialized()
    print("FROZEN_NLI_LOCAL_SCORING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "self-test", "prepare", "check", "extract", "score"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            synthetic_check()
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            real_check()
        elif arguments.stage == "extract":
            extract()
        else:
            score()
