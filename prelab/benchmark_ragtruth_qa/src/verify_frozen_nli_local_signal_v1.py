"""Independent preparation audit and deterministic CPU replay for local NLI v1.

This verifier deliberately does not import the experiment runner. It rebuilds
passage slicing, answer segmentation, lexical-BPE ownership, NLI pair lengths,
cache signatures, and the three-real-answer forward pass from primary inputs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import feature_qa as fq
import run_development as q


ROOT = q.ROOT
DATA = ROOT / "data"
OUT = ROOT / "results/nli_local_signal_v1"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"
REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
VIEWS = ("full", "passage_1", "passage_2", "passage_3")
CLASSES = ("entailment", "neutral", "contradiction")
LIMIT = 2048
BATCH = 8
ROLE = "ours_method_candidate; never a baseline"
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slice_passages(text):
    matches = list(HEADER.finditer(text))
    assert [match.group(1) for match in matches] == ["1", "2", "3"]
    bodies, ranges = [], []
    for index, match in enumerate(matches):
        left = match.end()
        right = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        assert left < right and any(character.isalnum() for character in text[left:right])
        bodies.append(text[left:right])
        ranges.append([left, right])
    return bodies, ranges


def split_answer(text):
    cuts = []
    index = 0
    line_start = 0
    while index < len(text):
        character = text[index]
        if character in "\r\n":
            if character == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            cuts.append(index + 1)
            line_start = index + 1
        elif character in TERMINAL:
            end = index + 1
            while end < len(text) and (text[end] in TERMINAL or text[end] in CLOSERS):
                end += 1
            prefix = text[line_start:index + 1].strip()
            token_match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            token = token_match.group(0).lower() if token_match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", token))
            abbreviation = end < len(text) and (token in ABBREVIATIONS or dotted or initial)
            if (end == len(text) or text[end].isspace()) and not LIST_PREFIX.fullmatch(prefix) and not abbreviation:
                cuts.append(end)
                index = end - 1
        index += 1
    cuts.append(len(text))
    spans = []
    start = 0
    for stop in sorted(set(cuts)):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(character.isalnum() for character in text[left:right]):
            spans.append([left, right])
        start = stop
    coverage = np.zeros(len(text), dtype=np.int8)
    for left, right in spans:
        coverage[left:right] += 1
    lexical = np.fromiter((character.isalnum() for character in text), dtype=bool)
    assert spans and np.all(coverage[lexical] == 1) and np.all(coverage <= 1)
    return spans


def token_owners(text, offsets, spans):
    owners = [-1] * len(offsets)
    inverse = [[] for _ in spans]
    for token_index, item in enumerate(offsets):
        left, right = map(int, item)
        assert 0 <= left < right <= len(text)
        if not any(text[position].isalnum() for position in range(left, right)):
            continue
        masses = [sum(text[position].isalnum() for position in range(max(left, a), min(right, b)))
                  for a, b in spans]
        assert max(masses) > 0
        owner = max(range(len(spans)), key=lambda segment: (masses[segment], -segment))
        owners[token_index] = owner
        inverse[owner].append(token_index)
    assert all(inverse)
    return owners, inverse


def independent_row(row, plan, tokenizer):
    assert row["response_id"] == plan["response_id"]
    text = row["original_response"]
    assert text == plan["original_response"] and digest(text) == row["answer_sha256"]
    bodies, ranges = slice_passages(row["retrieved_passages"])
    spans = split_answer(text)
    offsets = plan["original"]["response_token_offsets"]
    owners, inverse = token_owners(text, offsets, spans)
    segments = [{
        "segment_id": index,
        "char_start": left,
        "char_end": right,
        "hypothesis": text[left:right],
        "lexical_token_indices": inverse[index],
    } for index, (left, right) in enumerate(spans)]
    premises = [row["retrieved_passages"]] + bodies
    pairs = [(premise, segment["hypothesis"]) for segment in segments for premise in premises]
    encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                        add_special_tokens=True, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    assert max(lengths) <= LIMIT
    return {
        "response_id": row["response_id"],
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "partition": row["partition"],
        "answer_sha256": row["answer_sha256"],
        "full_premise": row["retrieved_passages"],
        "passage_premises": bodies,
        "passage_character_ranges": ranges,
        "premise_sha256": [digest(x) for x in premises],
        "segments": segments,
        "lexical_token_segment": owners,
        "raw_token_count": len(offsets),
        "response_token_offsets_sha256": digest(offsets),
        "original_token_ids_sha256": digest(plan["original"]["answer_token_ids"]),
        "labels_used": False,
        "method_role": ROLE,
        "pair_token_lengths": lengths,
    }


def pairs_and_owners(row):
    premises = [row["full_premise"]] + row["passage_premises"]
    pairs, owners = [], []
    for segment in row["segments"]:
        for view_index, premise in enumerate(premises):
            pairs.append((premise, segment["hypothesis"]))
            owners.append((segment["segment_id"], view_index))
    return pairs, owners


def cache_signature(row):
    pairs, owners = pairs_and_owners(row)
    return {
        "response_id": row["response_id"],
        "input_row_sha256": digest(row),
        "pair_definition_sha256": digest({"pairs": pairs, "owners": owners}),
        "input_file_sha256": q.sha(OUT / "inputs.jsonl"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "model_sha256": MODEL_SHA256,
    }


def verify_probability_cache(path, row):
    expected = cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        assert data["method_role"].item() == ROLE
        assert data["response_id"].item() == expected["response_id"]
        for key in ("input_row_sha256", "pair_definition_sha256", "input_file_sha256",
                    "protocol_sha256", "model_sha256"):
            assert data[key].item() == expected[key]
        assert data["segment_ids"].tolist() == list(range(len(row["segments"])))
        assert data["view_names"].tolist() == list(VIEWS)
        probabilities = np.stack([data[name] for name in CLASSES], axis=-1)
        assert probabilities.shape == (len(row["segments"]), 4, 3)
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(2), 1, atol=2e-6)
    return probabilities


def model_replay(rows):
    check = q.read(OUT / "CPU_WIRING_CHECK.json")
    cache_path = OUT / "CPU_WIRING_PROBABILITIES.npz"
    assert q.sha(cache_path) == check["probability_cache_sha256"]
    selected_ids = [rows[index]["response_id"] for index in (0, len(rows) // 2, len(rows) - 1)]
    assert selected_ids == check["response_ids"]
    selected = [rows[index] for index in (0, len(rows) // 2, len(rows) - 1)]
    pairs, owners = [], []
    for row in selected:
        one_pairs, one_owners = pairs_and_owners(row)
        pairs.extend(one_pairs)
        owners.extend((row["response_id"], *owner) for owner in one_owners)
    assert q.sha(MODEL / "model.safetensors") == MODEL_SHA256
    assert q.read(MODEL / "download_manifest.json")["revision"] == REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32).eval().cpu()
    model.requires_grad_(False)
    replay = []
    for left in range(0, len(pairs), BATCH):
        batch = pairs[left:left + BATCH]
        encoded = tokenizer([x[0] for x in batch], [x[1] for x in batch],
                            add_special_tokens=True, padding=True, truncation=False,
                            return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= LIMIT
        with torch.inference_mode():
            replay.append(model(**encoded).logits.float().softmax(-1).cpu().numpy())
    replay = np.concatenate(replay).astype(np.float32)
    with np.load(cache_path, allow_pickle=False) as saved:
        assert saved["owner_response_ids"].tolist() == [item[0] for item in owners]
        assert saved["owner_segment_ids"].tolist() == [item[1] for item in owners]
        assert saved["owner_view_indices"].tolist() == [item[2] for item in owners]
        assert saved["selected_response_ids"].tolist() == selected_ids
        assert saved["selected_input_row_sha256"].tolist() == [digest(row) for row in selected]
        assert saved["input_file_sha256"].item() == q.sha(OUT / "inputs.jsonl")
        assert saved["protocol_sha256"].item() == q.sha(OUT / "protocol.json")
        assert saved["model_sha256"].item() == MODEL_SHA256
        assert saved["method_role"].item() == ROLE
        original = saved["probabilities"].copy()
    maximum_difference = float(np.max(np.abs(replay - original)))
    assert maximum_difference <= 2e-6
    del tokenizer, model, replay, original
    return {"response_ids": selected_ids, "pairs": len(pairs),
            "replay_within_fixed_2e-6_FP32_tolerance": True,
            "maximum_absolute_difference": maximum_difference}


def main():
    assert not torch.cuda.is_initialized()
    output = OUT / "INDEPENDENT_VERIFY_REPLAY.json"
    assert not output.exists(), "No silent verifier overwrite"
    protocol = q.read(OUT / "protocol.json")
    assert protocol["version"] == "native-qa-frozen-local-nli-v1"
    assert protocol["role"] == ROLE
    preparation = q.read(OUT / "preparation_complete.json")
    assert preparation["input_sha256"] == q.sha(OUT / "inputs.jsonl")
    assert preparation["protocol_sha256"] == q.sha(OUT / "protocol.json")
    snapshot = q.read(OUT / "source_snapshot.json")
    assert snapshot["role"] == ROLE
    assert all(q.sha(path) == expected for path, expected in snapshot["files_sha256"].items())
    raw_rows, _ = fq.development_rows()
    plans = {row["response_id"]: row for row in q.lines(DATA / "feature_preparation/plans.jsonl")}
    saved_rows = q.lines(OUT / "inputs.jsonl")
    assert len(raw_rows) == len(plans) == len(saved_rows) == 793
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rebuilt = []
    for index, (raw, saved) in enumerate(zip(raw_rows, saved_rows)):
        expected = independent_row(raw, plans[raw["response_id"]], tokenizer)
        assert expected == saved, (index, raw["response_id"])
        rebuilt.append(expected)
        if (index + 1) % 100 == 0:
            print("INDEPENDENT_NLI_INPUT_REBUILT", index + 1, 793, flush=True)
    del tokenizer
    extraction = None
    if (OUT / "extraction_complete.json").exists():
        manifest = q.read(OUT / "extraction_complete.json")
        for row in rebuilt:
            name = row["response_id"] + ".npz"
            path = OUT / "segment_scores" / name
            assert q.sha(path) == manifest["files_sha256"][name]
            verify_probability_cache(path, row)
        extraction = {"all_793_cache_signatures_exact": True,
                      "all_probability_shapes_and_sums_valid": True}
    replay = model_replay(rebuilt)
    report = {
        "status": "independent_preparation_and_real3_replay_passed",
        "role": ROLE,
        "prepared_rows_rebuilt_exactly": len(rebuilt),
        "partition_counts": dict(Counter(row["partition"] for row in rebuilt)),
        "segments": sum(len(row["segments"]) for row in rebuilt),
        "pairs": sum(4 * len(row["segments"]) for row in rebuilt),
        "lexical_BPEs_uniquely_traced": sum(sum(owner >= 0 for owner in row["lexical_token_segment"]) for row in rebuilt),
        "maximum_pair_tokens": max(max(row["pair_token_lengths"]) for row in rebuilt),
        "source_snapshot_exact": True,
        "real3_model_replay": replay,
        "full_extraction_cache_verification": extraction,
        "verifier_sha256": q.sha(__file__),
        "runner_sha256": q.sha(ROOT / "src/run_frozen_nli_local_signal_v1.py"),
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "input_sha256": q.sha(OUT / "inputs.jsonl"),
        "model_sha256": MODEL_SHA256,
        "annotation_values_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    q.save(output, report)
    assert not torch.cuda.is_initialized()
    print("INDEPENDENT_NLI_PREPARATION_AND_REPLAY_PASSED", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
