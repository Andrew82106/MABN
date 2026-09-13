"""Combined fit-only loader and exact 4-BPE projection index.

The module joins the 634 native semantic-attribution shards with the 3,046
expanded-fit shards for a future ``semantic_window_v2`` stage.  It never trains
a model, selects a feature set, tunes a threshold, opens calibration/test data,
or starts a GPU job.

Commands
--------
prepare
    Read only the frozen 3,680-answer fit labels and label-free microclaims;
    reproduce every eligible/excluded 4-BPE window and freeze projection axes.
status
    Report authoritative committed feature-cache coverage.
wait
    Poll cache status for a bounded period; never starts extraction.
audit
    Once both caches are complete, validate all 3,680 feature shards and write
    a compact, fixed-width combined feature matrix.  Otherwise return WAIT.

The public loader and projection functions at the end are intended to be
imported by later experiment runners.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "semantic-source-attribution-combined-fit-v1"
OUT = ROOT / "results/semantic_source_attribution_combined_fit_v1"

TOKENS_PATH = ROOT / "fit_expansion/data/tokens_fit.jsonl"
WINDOWS_PATH = ROOT / "fit_expansion/data/windows_k4_fit.jsonl"
EXCLUDED_WINDOWS_PATH = ROOT / "fit_expansion/data/windows_excluded_fit.jsonl"
MICROCLAIMS_PATH = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
MICROCLAIMS_COMPLETE_PATH = ROOT / "research/atomic_relation_expanded_fit_v1/complete.json"
NEW_PLANS_PATH = ROOT / "fit_expansion/data/new_token_plans.jsonl"
FIT_SOURCE_WHITELIST_PATH = ROOT / "fit_expansion/fit_source_whitelist.jsonl"

NATIVE_CACHE = ROOT / "results/semantic_source_attribution_v1"
EXPANDED_CACHE = ROOT / "results/semantic_source_attribution_expanded_fit_v1"
NATIVE_RUNNER = HERE / "run_semantic_source_attribution_v1.py"
EXPANDED_RUNNER = HERE / "run_semantic_source_attribution_expanded_fit_v1.py"

EXPECTED_HASHES = {
    TOKENS_PATH: "18501c2a53a22620f24827d3181d831f2aee77da2b335bd50270fbe939048ec2",
    WINDOWS_PATH: "cfaf5af088eaac422ee2685c9150a774171d930666426e936f86d0cee905eec3",
    EXCLUDED_WINDOWS_PATH: "88de2b92aadcaf499c4e83d4e10cee4563cdb7bafadefc12eac60069053b2b3f",
    MICROCLAIMS_PATH: "c1731ab6ca68569f7811db46f594431751995d65d2468cdec51f8d63f37d585a",
    MICROCLAIMS_COMPLETE_PATH: "3374bba512d6bdda37828a7f0e0ee66ce67b831e2a67e9f0db9b314d95f5c048",
    NEW_PLANS_PATH: "f22efc2cb7bcee00a9bbb6c5ef0a305b4fde40f92c2622b4fd67a8390734326c",
    FIT_SOURCE_WHITELIST_PATH: "e22e6412ee3455bff7fdcd2b4fbaab2197bf924a657819e23285f9b3c18edfa2",
    NATIVE_RUNNER: "a87b4c8bcc232cead05cdaa23deb67642c8551f11870b5fd1a640513e78b977b",
    EXPANDED_RUNNER: "924bbea7b59cc17bc7899e2a41c66f92b9d2f99130e08441473f31bb75112462",
    NATIVE_CACHE / "signature.json": "1c74873c6c6bd9882879309aa889a09d9990d9a91b5e888eaf56ac385d15aafa",
    NATIVE_CACHE / "feature_manifest.json": "37f50d7ad939272149688b24ac91ef68a8996aa398716de113cb1c5004ac0d8b",
    EXPANDED_CACHE / "signature.json": "0e96a2e9eb2792f7b161c62842a4a436980cb39f456812b7ba0e81d7657b7c97",
    EXPANDED_CACHE / "CPU_SELFCHECK.json": "6434f37735d50179547a9ca0a211faa4646a9c60676bdf52a51d8a114eec4a52",
}
EXPECTED_NATIVE_SIGNATURE_DIGEST = "cb13454bac5f2ccaaa873f74c935a88a7b954370497441a0bc10cb4a1a370804"
EXPECTED_EXPANDED_SIGNATURE_DIGEST = "6d5aa3cc47d6d6d8b1f44f4b2fa9beaac6698a1b3db6b363aa1763ed3f2382b1"

EXPECTED_RESPONSES = 3680
EXPECTED_NATIVE_RESPONSES = 634
EXPECTED_EXPANDED_RESPONSES = 3046
EXPECTED_SOURCES = 634
EXPECTED_GROUPS = 615
EXPECTED_TOKENS = 665708
EXPECTED_LEXICAL_TOKENS = 560300
EXPECTED_RISK_TOKENS = 47398
EXPECTED_RISK_ANSWERS = 1127
EXPECTED_RAW_CLAIMS = 34941
EXPECTED_CLAIMS = 34919
EXPECTED_POSITIVE_CLAIMS = 3474
EXPECTED_NONLEXICAL_CLAIMS = 22
EXPECTED_WINDOWS = 653979
EXPECTED_POSITIVE_WINDOWS = 58433
EXPECTED_EXCLUDED_WINDOWS = 692
EXPECTED_EDGES = 523930
EXPECTED_NATIVE_CLAIMS = 9055
EXPECTED_EXPANDED_CLAIMS = 25864
EXPECTED_NATIVE_EDGES = 133247
EXPECTED_EXPANDED_EDGES = 390683
LAYERS = 32
HEADS = 32
TOP_K = 15

FORBIDDEN_MICROCLAIM_KEYS = {
    "label", "labels", "gold", "risk", "risk_mask", "original_labels",
    "hallucination", "hallucination_label", "token_labels",
    "risk_token_indices", "risk_character_spans",
}


class FeatureCacheNotReady(RuntimeError):
    """Raised by strict loaders when expanded extraction is incomplete."""


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def json_lines(path: Path):
    return list(iter_jsonl(path))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, ("Frozen JSON changed", str(path))
    else:
        atomic_json(path, value)


def frozen_text(path: Path, value: str) -> None:
    if path.exists():
        assert path.read_text(encoding="utf-8") == value
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".pending")
        pending.write_text(value, encoding="utf-8")
        pending.replace(path)


def frozen_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    if path.exists():
        assert sha(path) == sha(pending), ("Frozen JSONL changed", str(path))
        pending.unlink()
    else:
        pending.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def frozen_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(arrays)
            assert all(np.array_equal(loaded[key], value)
                       for key, value in arrays.items())
    else:
        atomic_npz(path, arrays)


def reject_microclaim_annotations(value) -> None:
    if isinstance(value, dict):
        found = set(value) & FORBIDDEN_MICROCLAIM_KEYS
        assert not found, f"Gold-bearing microclaim field: {sorted(found)}"
        for child in value.values():
            reject_microclaim_annotations(child)
    elif isinstance(value, list):
        for child in value:
            reject_microclaim_annotations(child)


def assert_cpu_only() -> None:
    assert not torch.cuda.is_initialized(), "This combined loader is CPU-only"


def assert_input_hashes(include_labels: bool = True) -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not include_labels and path in (TOKENS_PATH, WINDOWS_PATH,
                                           EXCLUDED_WINDOWS_PATH):
            continue
        assert path.is_file() and sha(path) == expected, str(path)


def merge_intervals(intervals) -> list[list[int]]:
    result = []
    for left, right in sorted((int(left), int(right)) for left, right in intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def windows_for_count(count: int) -> list[tuple[int, int]]:
    assert count > 0
    return [(start, min(start + 4, count))
            for start in range(max(1, count - 4 + 1))]


def expected_window(token: dict, start: int, end: int) -> dict:
    text = token["original_response"]
    offsets = token["response_token_offsets"]
    indices = list(range(start, end))
    lexical = [index for index in indices if token["lexical_mask"][index]]
    risk = [index for index in indices if token["risk_mask"][index]]
    intervals = merge_intervals(offsets[start:end])
    assert intervals
    risk_intervals = merge_intervals([
        (span["start"], span["end"])
        for index in indices
        for span in token["token_risk_character_spans"][index]
    ])
    identity = {key: token[key] for key in
                ("response_id", "source_id", "group_id", "partition", "answer_id")}
    row = {
        **identity,
        "window_id": f"{token['response_id']}__k4_{start:05d}",
        "k": 4, "stride": 1,
        "token_start": start, "token_end": end,
        "token_indices": indices,
        "answer_token_positions": token["answer_token_positions"][start:end],
        "token_ids": token["token_ids"][start:end],
        "character_intervals": intervals,
        "char_start": intervals[0][0], "char_end": intervals[-1][1],
        "bounding_text": text[intervals[0][0]:intervals[-1][1]],
        "lexical_token_indices": lexical,
        "risk_token_indices": risk,
        "risk_character_spans": [
            {"start": left, "end": right, "text": text[left:right]}
            for left, right in risk_intervals
        ],
        "eligible": bool(lexical),
        "label": int(bool(risk)) if lexical else None,
    }
    if not lexical:
        row["exclusion_reason"] = "no_lexical_token"
    return row


def _microclaims_by_response(response_ids: set[str]):
    grouped = defaultdict(list)
    total = 0
    for row in iter_jsonl(MICROCLAIMS_PATH):
        assert row["partition"] == "fit"
        reject_microclaim_annotations(row)
        assert row["response_id"] in response_ids
        grouped[row["response_id"]].append(row)
        total += 1
    assert total == EXPECTED_RAW_CLAIMS and set(grouped) == response_ids
    return grouped


PROTOCOL = {
    "version": VERSION,
    "consumer": "future semantic_window_v2 training/scoring code",
    "scope": {
        "answers": "Frozen 3,680 fit answers only: native 634 followed by expanded 3,046.",
        "labels": "Only fit token/window annotations are opened in prepare.",
        "calibration": "No path constant and no read operation.",
        "official_test": "No path constant and no read operation.",
    },
    "identity": {
        "group": "Locked source-connected group_id; all response, claim and window units inherit it through response_index.",
        "counts": {"responses": EXPECTED_RESPONSES, "sources": EXPECTED_SOURCES,
                   "source_connected_groups": EXPECTED_GROUPS},
        "cache_order": "First the whitelist/native 634, then new_token_plans 3,046; exactly the fit token/window order.",
    },
    "gold": {
        "tokens": EXPECTED_TOKENS, "lexical_tokens": EXPECTED_LEXICAL_TOKENS,
        "risk_tokens": EXPECTED_RISK_TOKENS,
        "claims": EXPECTED_CLAIMS, "positive_claims": EXPECTED_POSITIVE_CLAIMS,
        "windows": EXPECTED_WINDOWS, "positive_windows": EXPECTED_POSITIVE_WINDOWS,
        "excluded_nonlexical_windows": EXPECTED_EXCLUDED_WINDOWS,
        "window_rule": "Every stride-one contiguous raw-BPE window of width 4; answers shorter than 4 yield one shortened window. Eligible iff at least one original alphanumeric-overlap BPE. Label is OR of original token risk over the window.",
    },
    "projection": {
        "claim_to_token": "Broadcast each claim scalar to its exact lexical BPE owners; punctuation tokens receive fill_value.",
        "claim_to_window": "Maximum of claim scores owning lexical BPEs in the exact window. This is a prediction projection and is not asserted to reproduce fine token gold.",
        "token_to_window": "Maximum over lexical BPE scores in the exact window.",
        "window_to_answer": "Maximum over every eligible window belonging to the answer.",
        "threshold": "No threshold exists in this module.",
    },
    "features": {
        "native": "Read-only semantic_source_attribution_v1 feature triplets for exact whitelist response IDs only.",
        "expanded": "Read-only semantic_source_attribution_expanded_fit_v1 feature triplets.",
        "fixed_width": ["source_total_relevance", "previous_answer_relevance",
                        "other_context_relevance", "passage_relevance",
                        "top_sentence_relevance", "top_sentence_indices"],
        "full_edges": "Remain in verified per-answer shards; no second 1.22-GiB CSR copy is made.",
    },
    "stage_gate": "prepare may run immediately; audit waits for both caches; no training, model selection, threshold selection, calibration, test, model load, or GPU operation",
}


def protocol_markdown() -> str:
    return """# Combined semantic-attribution fit interface\n\nThis is the frozen data interface for a later `semantic_window_v2` experiment. It combines the original 634 fit feature shards and 3,046 expanded fit shards without changing either extractor.\n\n- Gold: 3,680 fit answers, 665,708 raw BPEs, 34,919 scored microclaims, and 653,979 eligible 4-BPE windows.\n- Positive gold: 47,398 BPEs, 3,474 microclaims, and 58,433 windows.\n- Exclusions: 22 punctuation-only microclaims and 692 no-lexical-BPE windows.\n- Grouping: the locked 615 source-connected groups remain indivisible.\n- Isolation: no calibration/test path is opened.\n- This module only loads, verifies, and projects data; it has no model, loss, feature choice, threshold, or GPU command.\n"""


def prepare() -> dict:
    assert_cpu_only()
    assert_input_hashes(include_labels=True)
    tokens = json_lines(TOKENS_PATH)
    assert len(tokens) == EXPECTED_RESPONSES
    response_ids = [row["response_id"] for row in tokens]
    assert len(set(response_ids)) == EXPECTED_RESPONSES
    assert all(row["partition"] == "fit" for row in tokens)
    assert all(row["answer_id"] == row["response_id"] for row in tokens)

    whitelist = json_lines(FIT_SOURCE_WHITELIST_PATH)
    native_ids = [row["existing_response_id"] for row in whitelist]
    new_ids = [row["response_id"] for row in iter_jsonl(NEW_PLANS_PATH)]
    assert len(native_ids) == EXPECTED_NATIVE_RESPONSES
    assert len(new_ids) == EXPECTED_EXPANDED_RESPONSES
    assert response_ids == native_ids + new_ids
    assert not (set(native_ids) & set(new_ids))

    source_ids = sorted({row["source_id"] for row in tokens})
    group_ids = sorted({row["group_id"] for row in tokens})
    assert len(source_ids) == EXPECTED_SOURCES
    assert len(group_ids) == EXPECTED_GROUPS
    source_lookup = {value: index for index, value in enumerate(source_ids)}
    group_lookup = {value: index for index, value in enumerate(group_ids)}
    source_groups = defaultdict(set)
    for row in tokens:
        source_groups[row["source_id"]].add(row["group_id"])
    assert all(len(values) == 1 for values in source_groups.values())
    assert len(source_groups) == EXPECTED_SOURCES

    grouped_claims = _microclaims_by_response(set(response_ids))
    total_tokens = sum(int(row["token_count"]) for row in tokens)
    assert total_tokens == EXPECTED_TOKENS
    token_ids = np.empty(total_tokens, dtype=np.int32)
    token_absolute_positions = np.empty(total_tokens, dtype=np.int32)
    token_character_offsets = np.empty((total_tokens, 2), dtype=np.int32)
    token_lexical_mask = np.empty(total_tokens, dtype=np.uint8)
    token_labels = np.empty(total_tokens, dtype=np.int8)
    token_claim_index = np.full(total_tokens, -1, dtype=np.int32)

    response_token_indptr = [0]
    response_claim_indptr = [0]
    response_source_index, response_group_index = [], []
    response_is_native, response_labels = [], []
    claim_response_index, claim_group_index = [], []
    claim_local_ids, claim_microclaim_indices = [], []
    claim_character_ranges, claim_labels = [], []
    claim_token_indptr, claim_token_indices = [0], []
    claim_axes, nonlexical_claims = [], []
    token_cursor = 0
    global_claim = 0

    for response_index, token in enumerate(tokens):
        rid = token["response_id"]
        n = int(token["token_count"])
        assert n == len(token["token_ids"]) == len(token["answer_token_positions"])
        assert n == len(token["response_token_offsets"]) == len(token["lexical_mask"])
        assert n == len(token["risk_mask"]) == len(token["token_risk_character_spans"])
        left, right = token_cursor, token_cursor + n
        token_ids[left:right] = token["token_ids"]
        token_absolute_positions[left:right] = token["answer_token_positions"]
        token_character_offsets[left:right] = token["response_token_offsets"]
        token_lexical_mask[left:right] = token["lexical_mask"]
        token_labels[left:right] = token["risk_mask"]
        response_source_index.append(source_lookup[token["source_id"]])
        response_group_index.append(group_lookup[token["group_id"]])
        response_is_native.append(int(response_index < EXPECTED_NATIVE_RESPONSES))
        response_labels.append(int(token["answer_risk"]))

        raw = sorted(grouped_claims[rid], key=lambda row: row["microclaim_index"])
        assert [row["microclaim_index"] for row in raw] == list(range(len(raw)))
        inverse = [[] for _ in range(n)]
        local_claim = 0
        answer = token["original_response"]
        offsets = token["response_token_offsets"]
        for row in raw:
            assert row["source_id"] == token["source_id"]
            assert row["group_id"] == token["group_id"]
            start, end = int(row["start"]), int(row["end"])
            assert answer[start:end] == row["text"]
            local_tokens = []
            for local_index, (a, b) in enumerate(offsets):
                begin, finish = max(int(a), start), min(int(b), end)
                if begin < finish and any(char.isalnum()
                                          for char in answer[begin:finish]):
                    local_tokens.append(local_index)
            if not local_tokens:
                assert not any(char.isalnum() for char in row["text"])
                nonlexical_claims.append({
                    "schema_version": VERSION,
                    "response_index": response_index, "response_id": rid,
                    "source_id": token["source_id"], "group_id": token["group_id"],
                    "microclaim_id": row["microclaim_id"],
                    "microclaim_index": int(row["microclaim_index"]),
                    "start": start, "end": end, "text": row["text"],
                    "text_sha256": digest(row["text"]),
                    "reason": "no_alphanumeric_character_and_no_lexical_bpe",
                })
                continue
            for local_index in local_tokens:
                inverse[local_index].append(global_claim)
            global_token_indices = [left + index for index in local_tokens]
            claim_response_index.append(response_index)
            claim_group_index.append(group_lookup[token["group_id"]])
            claim_local_ids.append(local_claim)
            claim_microclaim_indices.append(int(row["microclaim_index"]))
            claim_character_ranges.append([start, end])
            claim_labels.append(int(any(token["risk_mask"][index]
                                        for index in local_tokens)))
            claim_token_indices.extend(global_token_indices)
            claim_token_indptr.append(len(claim_token_indices))
            claim_axes.append({
                "global_claim_index": global_claim,
                "response_index": response_index, "response_id": rid,
                "local_claim_id": local_claim,
                "microclaim_id": row["microclaim_id"],
                "microclaim_index": int(row["microclaim_index"]),
                "start": start, "end": end, "text": row["text"],
                "text_sha256": digest(row["text"]),
            })
            global_claim += 1
            local_claim += 1
        for local_index, is_lexical in enumerate(token["lexical_mask"]):
            owners = inverse[local_index]
            if is_lexical:
                assert len(owners) == 1
                token_claim_index[left + local_index] = owners[0]
            else:
                assert not owners
        response_claim_indptr.append(global_claim)
        token_cursor = right
        response_token_indptr.append(token_cursor)
        if (response_index + 1) % 500 == 0:
            print("COMBINED_PROJECTION_CLAIMS", response_index + 1,
                  EXPECTED_RESPONSES, flush=True)

    assert token_cursor == EXPECTED_TOKENS
    assert global_claim == EXPECTED_CLAIMS
    assert len(nonlexical_claims) == EXPECTED_NONLEXICAL_CLAIMS
    assert int(token_lexical_mask.sum()) == EXPECTED_LEXICAL_TOKENS
    assert int(token_labels.sum()) == EXPECTED_RISK_TOKENS
    assert int(np.sum(response_labels)) == EXPECTED_RISK_ANSWERS
    assert int(np.sum(claim_labels)) == EXPECTED_POSITIVE_CLAIMS
    assert len(claim_token_indices) == EXPECTED_LEXICAL_TOKENS
    assert np.count_nonzero(token_claim_index >= 0) == EXPECTED_LEXICAL_TOKENS
    assert response_claim_indptr[EXPECTED_NATIVE_RESPONSES] == EXPECTED_NATIVE_CLAIMS
    assert global_claim - EXPECTED_NATIVE_CLAIMS == EXPECTED_EXPANDED_CLAIMS

    window_response_index = np.empty(EXPECTED_WINDOWS, dtype=np.int32)
    window_token_ranges = np.empty((EXPECTED_WINDOWS, 2), dtype=np.int32)
    window_token_indices = np.full((EXPECTED_WINDOWS, 4), -1, dtype=np.int64)
    window_claim_indices = np.full((EXPECTED_WINDOWS, 4), -1, dtype=np.int32)
    window_labels = np.empty(EXPECTED_WINDOWS, dtype=np.int8)
    excluded_response_index = np.empty(EXPECTED_EXCLUDED_WINDOWS, dtype=np.int32)
    excluded_token_ranges = np.empty((EXPECTED_EXCLUDED_WINDOWS, 2), dtype=np.int32)
    excluded_token_indices = np.full((EXPECTED_EXCLUDED_WINDOWS, 4), -1, dtype=np.int64)
    response_window_indptr = [0]
    response_excluded_window_indptr = [0]
    eligible_iterator = iter(iter_jsonl(WINDOWS_PATH))
    excluded_iterator = iter(iter_jsonl(EXCLUDED_WINDOWS_PATH))
    window_cursor = excluded_cursor = 0

    for response_index, token in enumerate(tokens):
        token_base = response_token_indptr[response_index]
        response_windows = response_excluded = 0
        for start, end in windows_for_count(token["token_count"]):
            expected = expected_window(token, start, end)
            if expected["eligible"]:
                actual = next(eligible_iterator)
                assert actual == expected, actual["window_id"]
                row_index = window_cursor
                window_response_index[row_index] = response_index
                window_token_ranges[row_index] = [start, end]
                local = expected["token_indices"]
                window_token_indices[row_index, :len(local)] = [token_base + x for x in local]
                owners = []
                for local_index in expected["lexical_token_indices"]:
                    owner = int(token_claim_index[token_base + local_index])
                    assert owner >= 0
                    if owner not in owners:
                        owners.append(owner)
                assert 1 <= len(owners) <= 4
                window_claim_indices[row_index, :len(owners)] = owners
                window_labels[row_index] = expected["label"]
                window_cursor += 1
                response_windows += 1
            else:
                actual = next(excluded_iterator)
                assert actual == expected, actual["window_id"]
                row_index = excluded_cursor
                excluded_response_index[row_index] = response_index
                excluded_token_ranges[row_index] = [start, end]
                local = expected["token_indices"]
                excluded_token_indices[row_index, :len(local)] = [token_base + x for x in local]
                assert np.all(token_claim_index[
                    excluded_token_indices[row_index, :len(local)]] == -1)
                excluded_cursor += 1
                response_excluded += 1
        response_window_indptr.append(response_window_indptr[-1] + response_windows)
        response_excluded_window_indptr.append(
            response_excluded_window_indptr[-1] + response_excluded)
        if (response_index + 1) % 500 == 0:
            print("COMBINED_PROJECTION_WINDOWS", response_index + 1,
                  EXPECTED_RESPONSES, flush=True)
    try:
        next(eligible_iterator)
        raise AssertionError("Extra eligible window row")
    except StopIteration:
        pass
    try:
        next(excluded_iterator)
        raise AssertionError("Extra excluded window row")
    except StopIteration:
        pass
    assert window_cursor == EXPECTED_WINDOWS
    assert excluded_cursor == EXPECTED_EXCLUDED_WINDOWS
    assert int(window_labels.sum()) == EXPECTED_POSITIVE_WINDOWS
    assert response_window_indptr[EXPECTED_NATIVE_RESPONSES] == 168123
    assert window_cursor - 168123 == 485856

    arrays = {
        "response_source_index": np.asarray(response_source_index, dtype=np.int32),
        "response_group_index": np.asarray(response_group_index, dtype=np.int16),
        "response_is_native": np.asarray(response_is_native, dtype=np.uint8),
        "response_labels": np.asarray(response_labels, dtype=np.int8),
        "response_token_indptr": np.asarray(response_token_indptr, dtype=np.int64),
        "response_claim_indptr": np.asarray(response_claim_indptr, dtype=np.int64),
        "response_window_indptr": np.asarray(response_window_indptr, dtype=np.int64),
        "response_excluded_window_indptr": np.asarray(
            response_excluded_window_indptr, dtype=np.int64),
        "token_ids": token_ids,
        "token_absolute_positions": token_absolute_positions,
        "token_character_offsets": token_character_offsets,
        "token_lexical_mask": token_lexical_mask,
        "token_labels": token_labels,
        "token_claim_index": token_claim_index,
        "claim_response_index": np.asarray(claim_response_index, dtype=np.int32),
        "claim_group_index": np.asarray(claim_group_index, dtype=np.int16),
        "claim_local_ids": np.asarray(claim_local_ids, dtype=np.int32),
        "claim_microclaim_indices": np.asarray(claim_microclaim_indices, dtype=np.int32),
        "claim_character_ranges": np.asarray(claim_character_ranges, dtype=np.int32),
        "claim_labels": np.asarray(claim_labels, dtype=np.int8),
        "claim_token_indptr": np.asarray(claim_token_indptr, dtype=np.int64),
        "claim_token_indices": np.asarray(claim_token_indices, dtype=np.int64),
        "window_response_index": window_response_index,
        "window_token_ranges": window_token_ranges,
        "window_token_indices": window_token_indices,
        "window_claim_indices": window_claim_indices,
        "window_labels": window_labels,
        "excluded_response_index": excluded_response_index,
        "excluded_token_ranges": excluded_token_ranges,
        "excluded_token_indices": excluded_token_indices,
    }
    axes = {
        "schema_version": VERSION,
        "response_ids": response_ids,
        "response_answer_sha256": [row["answer_sha256"] for row in tokens],
        "response_source_ids": [row["source_id"] for row in tokens],
        "response_group_ids": [row["group_id"] for row in tokens],
        "source_ids": source_ids, "group_ids": group_ids,
        "window_id_recipe": "{response_id}__k4_{token_start:05d}",
        "partition": "fit",
        "labels_used": True, "calibration_opened": False,
        "official_test_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    projection_path = OUT / "projection.npz"
    axes_path = OUT / "axes.json"
    claims_path = OUT / "claim_axes.jsonl"
    exclusions_path = OUT / "nonlexical_claims.jsonl"
    frozen_npz(projection_path, arrays)
    frozen_json(axes_path, axes)
    frozen_jsonl(claims_path, claim_axes)
    frozen_jsonl(exclusions_path, nonlexical_claims)
    frozen_json(OUT / "protocol.json", PROTOCOL)
    frozen_text(OUT / "PROTOCOL.md", protocol_markdown())

    signature = {
        "version": VERSION, "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "input_sha256": {str(path.relative_to(ROOT)): expected
                         for path, expected in EXPECTED_HASHES.items()},
        "projection_sha256": sha(projection_path),
        "axes_sha256": sha(axes_path), "claim_axes_sha256": sha(claims_path),
        "nonlexical_claims_sha256": sha(exclusions_path),
        "labels_used": True, "calibration_opened": False,
        "official_test_opened": False,
    }
    frozen_json(OUT / "signature.json", signature)
    report = {
        "status": "projection_ready_waiting_for_complete_feature_cache",
        "responses": EXPECTED_RESPONSES,
        "native_responses": EXPECTED_NATIVE_RESPONSES,
        "expanded_responses": EXPECTED_EXPANDED_RESPONSES,
        "sources": EXPECTED_SOURCES, "source_connected_groups": EXPECTED_GROUPS,
        "tokens": EXPECTED_TOKENS, "lexical_tokens": EXPECTED_LEXICAL_TOKENS,
        "positive_tokens": EXPECTED_RISK_TOKENS,
        "claims": EXPECTED_CLAIMS, "positive_claims": EXPECTED_POSITIVE_CLAIMS,
        "nonlexical_claims": EXPECTED_NONLEXICAL_CLAIMS,
        "windows": EXPECTED_WINDOWS, "positive_windows": EXPECTED_POSITIVE_WINDOWS,
        "excluded_windows": EXPECTED_EXCLUDED_WINDOWS,
        "all_window_rows_reproduced_exactly": True,
        "every_lexical_bpe_has_exactly_one_claim_owner": True,
        "all_source_connected_groups_indivisible_by_index": True,
        "models_or_thresholds_selected": 0, "gpu_used": False,
        "labels_used": True, "label_partition": "fit",
        "calibration_opened": False, "official_test_opened": False,
        "signature_sha256": digest(signature),
    }
    frozen_json(OUT / "PREPARATION.json", report)
    frozen_text(OUT / "PREPARATION_REPORT.md", f"""# Combined fit projection preparation\n\nExact CPU reproduction passed for all {EXPECTED_WINDOWS:,} eligible windows ({EXPECTED_POSITIVE_WINDOWS:,} positive) and {EXPECTED_EXCLUDED_WINDOWS:,} excluded nonlexical windows. All {EXPECTED_LEXICAL_TOKENS:,} lexical BPEs have exactly one of {EXPECTED_CLAIMS:,} scored microclaim owners.\n\nThe 3,680 answers retain 634 sources and 615 source-connected groups. Only fit labels were read. Feature joining remains gated on completion of the independent expanded cache; no model, threshold, calibration/test record, or GPU was used.\n""")
    assert_cpu_only()
    print("COMBINED_FIT_PROJECTION_PREPARED", EXPECTED_WINDOWS,
          EXPECTED_POSITIVE_WINDOWS, EXPECTED_EXCLUDED_WINDOWS, flush=True)
    return report


def verify_prepared() -> tuple[dict, dict, dict[str, np.ndarray]]:
    assert_cpu_only()
    assert_input_hashes(include_labels=True)
    signature = read_json(OUT / "signature.json")
    preparation = read_json(OUT / "PREPARATION.json")
    assert signature["runner_sha256"] == sha(Path(__file__))
    assert signature["protocol_sha256"] == sha(OUT / "protocol.json")
    assert signature["projection_sha256"] == sha(OUT / "projection.npz")
    assert signature["axes_sha256"] == sha(OUT / "axes.json")
    assert signature["claim_axes_sha256"] == sha(OUT / "claim_axes.jsonl")
    assert signature["nonlexical_claims_sha256"] == sha(OUT / "nonlexical_claims.jsonl")
    assert preparation["signature_sha256"] == digest(signature)
    axes = read_json(OUT / "axes.json")
    assert axes["partition"] == "fit" and len(axes["response_ids"]) == EXPECTED_RESPONSES
    with np.load(OUT / "projection.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    assert arrays["window_labels"].shape == (EXPECTED_WINDOWS,)
    assert int(arrays["window_labels"].sum()) == EXPECTED_POSITIVE_WINDOWS
    assert arrays["claim_labels"].shape == (EXPECTED_CLAIMS,)
    assert arrays["token_labels"].shape == (EXPECTED_TOKENS,)
    return axes, signature, arrays


def _record_paths(cache: Path, response_id: str):
    feature = cache / "features" / f"{response_id}.npz"
    return feature, feature.with_suffix(".json"), feature.with_suffix(".commit.json")


def cache_status(write: bool = True) -> dict:
    assert_cpu_only()
    if (OUT / "axes.json").exists():
        response_ids = read_json(OUT / "axes.json")["response_ids"]
    else:
        native_ids = [row["existing_response_id"]
                      for row in iter_jsonl(FIT_SOURCE_WHITELIST_PATH)]
        new_ids = [row["response_id"] for row in iter_jsonl(NEW_PLANS_PATH)]
        response_ids = native_ids + new_ids
    native_ids = response_ids[:EXPECTED_NATIVE_RESPONSES]
    expanded_ids = response_ids[EXPECTED_NATIVE_RESPONSES:]

    def coverage(cache, ids):
        complete = partial = 0
        for rid in ids:
            present = [path.exists() for path in _record_paths(cache, rid)]
            complete += int(all(present))
            partial += int(any(present) and not all(present))
        return complete, partial

    native_complete, native_partial = coverage(NATIVE_CACHE, native_ids)
    expanded_complete, expanded_partial = coverage(EXPANDED_CACHE, expanded_ids)
    expanded_manifest = (EXPANDED_CACHE / "feature_manifest.json").exists()
    ready = (native_complete == EXPECTED_NATIVE_RESPONSES and
             expanded_complete == EXPECTED_EXPANDED_RESPONSES and
             expanded_manifest)
    report = {
        "status": "READY_FOR_FULL_CPU_AUDIT" if ready else "WAIT_FEATURE_CACHE",
        "projection_prepared": (OUT / "PREPARATION.json").exists(),
        "native": {"complete": native_complete, "total": EXPECTED_NATIVE_RESPONSES,
                   "partial_uncommitted": native_partial,
                   "manifest_complete": (NATIVE_CACHE / "feature_manifest.json").exists()},
        "expanded": {"complete": expanded_complete,
                     "total": EXPECTED_EXPANDED_RESPONSES,
                     "partial_uncommitted": expanded_partial,
                     "manifest_complete": expanded_manifest,
                     "gpu_started_elsewhere": (EXPANDED_CACHE / "extract_started.json").exists()},
        "combined_complete": native_complete + expanded_complete,
        "combined_total": EXPECTED_RESPONSES,
        "full_cpu_audit_complete": (OUT / "FEATURE_AUDIT_COMPLETE.json").exists(),
        "gpu_started_by_this_runner": False,
        "labels_opened_by_status": False,
        "calibration_opened": False, "official_test_opened": False,
    }
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def wait_for_cache(timeout_seconds: float, poll_seconds: float) -> dict:
    assert timeout_seconds >= 0 and poll_seconds > 0
    started = time.monotonic()
    while True:
        report = cache_status(write=True)
        if report["status"] == "READY_FOR_FULL_CPU_AUDIT":
            return report
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            return report
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def _load_feature_triplet(cache: Path, response: dict, expected_signature: str):
    feature, metadata_path, commit_path = _record_paths(cache, response["response_id"])
    assert feature.is_file() and metadata_path.is_file() and commit_path.is_file()
    commit, metadata = read_json(commit_path), read_json(metadata_path)
    assert commit["response_id"] == metadata["response_id"] == response["response_id"]
    assert commit["metadata_sha256"] == sha(metadata_path)
    assert commit["npz_sha256"] == metadata["npz_sha256"] == sha(feature)
    assert metadata["signature_sha256"] == expected_signature
    assert metadata["source_id"] == response["source_id"]
    assert metadata["group_id"] == response["group_id"]
    assert metadata["partition"] == "fit"
    assert metadata["labels_used"] is False
    assert metadata["official_test_opened"] is False
    if cache == EXPANDED_CACHE:
        assert metadata["calibration_opened"] is False
    with np.load(feature, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return arrays, metadata, {
        "response_id": response["response_id"],
        "cache": "native" if cache == NATIVE_CACHE else "expanded",
        "feature_path": str(feature.relative_to(ROOT)),
        "npz_sha256": commit["npz_sha256"],
        "metadata_sha256": commit["metadata_sha256"],
        "commit_sha256": sha(commit_path),
    }


def _validate_feature_arrays(arrays: dict[str, np.ndarray], expected_microclaims,
                             expected_answer_positions) -> dict:
    required = {
        "claim_sentence_mass", "claim_indptr", "claim_sentence_indices",
        "source_total_relevance", "previous_answer_relevance",
        "other_context_relevance", "passage_relevance", "sentence_relevance",
        "top_sentence_indices", "top_sentence_relevance", "claim_ids",
        "microclaim_indices", "sentence_indices", "sentence_passage_ids",
        "sentence_ids", "sentence_identity_sha256", "answer_token_positions",
    }
    assert set(arrays) == required
    c = len(expected_microclaims)
    s = len(arrays["sentence_indices"])
    assert c > 0 and s > 0
    assert np.array_equal(arrays["claim_ids"], np.arange(c, dtype=np.int32))
    assert np.array_equal(arrays["microclaim_indices"], expected_microclaims)
    assert np.array_equal(arrays["answer_token_positions"], expected_answer_positions)
    assert arrays["claim_sentence_mass"].shape == (c * s, LAYERS * HEADS)
    assert arrays["claim_sentence_mass"].dtype == np.float16
    assert np.array_equal(arrays["claim_indptr"], np.arange(c + 1, dtype=np.int64) * s)
    assert np.array_equal(arrays["claim_sentence_indices"],
                          np.tile(np.arange(s, dtype=np.int32), c))
    assert arrays["source_total_relevance"].shape == (c, LAYERS)
    assert arrays["previous_answer_relevance"].shape == (c, LAYERS)
    assert arrays["other_context_relevance"].shape == (c, LAYERS)
    assert arrays["passage_relevance"].shape == (c, LAYERS, 3)
    assert arrays["sentence_relevance"].shape == (c, LAYERS, s)
    assert arrays["top_sentence_indices"].shape == (c, LAYERS, TOP_K)
    assert arrays["top_sentence_relevance"].shape == (c, LAYERS, TOP_K)
    assert arrays["sentence_identity_sha256"].shape == (s, 32)
    floats = [value for value in arrays.values() if value.dtype.kind == "f"]
    assert all(np.isfinite(value).all() and np.all(value >= 0) for value in floats)
    reconstructed = arrays["claim_sentence_mass"].astype(np.float32).reshape(
        c, s, LAYERS, HEADS)
    quantized = reconstructed.mean(-1, dtype=np.float32).transpose(0, 2, 1)
    drift = float(np.max(np.abs(quantized - arrays["sentence_relevance"])))
    assert drift <= 0.02
    return {"claims": c, "sentences": s, "edges": c * s,
            "csr_float16_head_mean_max_abs_drift": drift}


def audit_features() -> dict:
    assert_cpu_only()
    state = cache_status(write=True)
    if state["status"] != "READY_FOR_FULL_CPU_AUDIT":
        wait = {
            "status": "WAIT_FEATURE_CACHE", "cache_status": state,
            "labels_opened": False, "gpu_started": False,
            "calibration_opened": False, "official_test_opened": False,
        }
        atomic_json(OUT / "FEATURE_AUDIT_WAIT.json", wait)
        return wait
    axes, signature, projection = verify_prepared()
    assert sha(NATIVE_CACHE / "feature_manifest.json") == EXPECTED_HASHES[
        NATIVE_CACHE / "feature_manifest.json"]
    expanded_manifest_sha = sha(EXPANDED_CACHE / "feature_manifest.json")

    c = EXPECTED_CLAIMS
    source = np.empty((c, LAYERS), dtype=np.float32)
    previous = np.empty((c, LAYERS), dtype=np.float32)
    other = np.empty((c, LAYERS), dtype=np.float32)
    passage = np.empty((c, LAYERS, 3), dtype=np.float32)
    top_relevance = np.empty((c, LAYERS, TOP_K), dtype=np.float32)
    top_indices = np.empty((c, LAYERS, TOP_K), dtype=np.int32)
    sentence_counts = np.empty(c, dtype=np.int16)
    edge_indptr = [0]
    records = []
    totals = Counter()
    max_drift = 0.0

    for response_index, rid in enumerate(axes["response_ids"]):
        begin = int(projection["response_claim_indptr"][response_index])
        end = int(projection["response_claim_indptr"][response_index + 1])
        token_begin = int(projection["response_token_indptr"][response_index])
        token_end = int(projection["response_token_indptr"][response_index + 1])
        response = {
            "response_id": rid,
            "source_id": axes["response_source_ids"][response_index],
            "group_id": axes["response_group_ids"][response_index],
        }
        if response_index < EXPECTED_NATIVE_RESPONSES:
            cache, expected_signature = NATIVE_CACHE, EXPECTED_NATIVE_SIGNATURE_DIGEST
        else:
            cache, expected_signature = EXPANDED_CACHE, EXPECTED_EXPANDED_SIGNATURE_DIGEST
        arrays, metadata, record = _load_feature_triplet(
            cache, response, expected_signature)
        check = _validate_feature_arrays(
            arrays,
            projection["claim_microclaim_indices"][begin:end],
            projection["token_absolute_positions"][token_begin:token_end])
        assert metadata["claims"] == check["claims"]
        assert metadata["sentences"] == check["sentences"]
        assert metadata["edges"] == check["edges"]
        source[begin:end] = arrays["source_total_relevance"]
        previous[begin:end] = arrays["previous_answer_relevance"]
        other[begin:end] = arrays["other_context_relevance"]
        passage[begin:end] = arrays["passage_relevance"]
        top_relevance[begin:end] = arrays["top_sentence_relevance"]
        top_indices[begin:end] = arrays["top_sentence_indices"]
        sentence_counts[begin:end] = check["sentences"]
        for _ in range(check["claims"]):
            edge_indptr.append(edge_indptr[-1] + check["sentences"])
        totals["claims"] += check["claims"]
        totals["edges"] += check["edges"]
        totals["native_claims" if cache == NATIVE_CACHE else "expanded_claims"] += check["claims"]
        totals["native_edges" if cache == NATIVE_CACHE else "expanded_edges"] += check["edges"]
        max_drift = max(max_drift, check["csr_float16_head_mean_max_abs_drift"])
        records.append({**record, **check})
        if (response_index + 1) % 250 == 0:
            print("COMBINED_FEATURE_AUDIT", response_index + 1,
                  EXPECTED_RESPONSES, flush=True)

    assert totals["claims"] == EXPECTED_CLAIMS
    assert totals["edges"] == EXPECTED_EDGES
    assert totals["native_claims"] == EXPECTED_NATIVE_CLAIMS
    assert totals["expanded_claims"] == EXPECTED_EXPANDED_CLAIMS
    assert totals["native_edges"] == EXPECTED_NATIVE_EDGES
    assert totals["expanded_edges"] == EXPECTED_EXPANDED_EDGES
    assert len(edge_indptr) == EXPECTED_CLAIMS + 1
    assert edge_indptr[-1] == EXPECTED_EDGES
    fixed = {
        "source_total_relevance": source,
        "previous_answer_relevance": previous,
        "other_context_relevance": other,
        "passage_relevance": passage,
        "top_sentence_relevance": top_relevance,
        "top_sentence_indices": top_indices,
        "sentence_counts": sentence_counts,
        "claim_sentence_edge_indptr": np.asarray(edge_indptr, dtype=np.int64),
    }
    frozen_npz(OUT / "compact_features.npz", fixed)
    frozen_jsonl(OUT / "feature_records.jsonl", records)
    complete = {
        "status": "complete",
        "responses": EXPECTED_RESPONSES, "claims": EXPECTED_CLAIMS,
        "claim_sentence_edges": EXPECTED_EDGES,
        "native_claims": totals["native_claims"],
        "expanded_claims": totals["expanded_claims"],
        "native_edges": totals["native_edges"],
        "expanded_edges": totals["expanded_edges"],
        "max_csr_float16_head_mean_abs_drift": max_drift,
        "projection_signature_sha256": digest(signature),
        "native_manifest_sha256": sha(NATIVE_CACHE / "feature_manifest.json"),
        "expanded_manifest_sha256": expanded_manifest_sha,
        "files_sha256": {
            "compact_features.npz": sha(OUT / "compact_features.npz"),
            "feature_records.jsonl": sha(OUT / "feature_records.jsonl"),
            "projection.npz": sha(OUT / "projection.npz"),
            "axes.json": sha(OUT / "axes.json"),
        },
        "models_or_thresholds_selected": 0, "gpu_used": False,
        "labels_used": True, "label_partition": "fit",
        "calibration_opened": False, "official_test_opened": False,
    }
    frozen_json(OUT / "FEATURE_AUDIT_COMPLETE.json", complete)
    frozen_text(OUT / "FEATURE_AUDIT_REPORT.md", f"""# Combined feature audit\n\nAll {EXPECTED_RESPONSES:,} fit feature shards passed commit/hash/schema/axis checks. The compact interface contains {EXPECTED_CLAIMS:,} claims and references {EXPECTED_EDGES:,} claim-sentence edges; the full layer-head CSR remains in the original read-only shards.\n\nNo model or threshold was selected, and no calibration/test data or GPU was used.\n""")
    assert_cpu_only()
    print("COMBINED_FEATURE_AUDIT_COMPLETE", EXPECTED_RESPONSES,
          EXPECTED_CLAIMS, EXPECTED_EDGES, flush=True)
    return complete


# ------------------------- reusable public interface -------------------------

def load_projection() -> tuple[dict, dict[str, np.ndarray]]:
    """Load verified fit identities, labels, and exact projection arrays."""
    axes, _, arrays = verify_prepared()
    return axes, arrays


def load_compact_features() -> dict[str, np.ndarray]:
    """Load combined fixed-width attribution features after the full audit."""
    if not (OUT / "FEATURE_AUDIT_COMPLETE.json").exists():
        raise FeatureCacheNotReady(cache_status(write=False)["status"])
    complete = read_json(OUT / "FEATURE_AUDIT_COMPLETE.json")
    assert complete["status"] == "complete"
    assert sha(OUT / "compact_features.npz") == complete["files_sha256"][
        "compact_features.npz"]
    with np.load(OUT / "compact_features.npz", allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def iter_feature_shards():
    """Yield verified per-answer full-CSR shards after cache completion."""
    axes, _, projection = verify_prepared()
    state = cache_status(write=False)
    if state["status"] != "READY_FOR_FULL_CPU_AUDIT":
        raise FeatureCacheNotReady(state["status"])
    for response_index, rid in enumerate(axes["response_ids"]):
        response = {"response_id": rid,
                    "source_id": axes["response_source_ids"][response_index],
                    "group_id": axes["response_group_ids"][response_index]}
        cache = NATIVE_CACHE if response_index < EXPECTED_NATIVE_RESPONSES else EXPANDED_CACHE
        expected = (EXPECTED_NATIVE_SIGNATURE_DIGEST if cache == NATIVE_CACHE
                    else EXPECTED_EXPANDED_SIGNATURE_DIGEST)
        arrays, metadata, _ = _load_feature_triplet(cache, response, expected)
        begin = int(projection["response_claim_indptr"][response_index])
        end = int(projection["response_claim_indptr"][response_index + 1])
        yield response_index, begin, end, arrays, metadata


def project_claim_scores_to_tokens(claim_scores, fill_value=np.nan) -> np.ndarray:
    """Broadcast one scalar per scored claim to its exact lexical BPE owners."""
    _, arrays = load_projection()
    scores = np.asarray(claim_scores)
    assert scores.shape == (EXPECTED_CLAIMS,) and np.isfinite(scores).all()
    result = np.full(EXPECTED_TOKENS, fill_value, dtype=np.result_type(scores, float))
    owners = arrays["token_claim_index"]
    valid = owners >= 0
    result[valid] = scores[owners[valid]]
    return result


def project_claim_scores_to_windows(claim_scores) -> np.ndarray:
    """Max-pool claim scalars over claims owning lexical BPEs in each window."""
    _, arrays = load_projection()
    scores = np.asarray(claim_scores)
    assert scores.shape == (EXPECTED_CLAIMS,) and np.isfinite(scores).all()
    owners = arrays["window_claim_indices"]
    valid = owners >= 0
    safe = np.where(valid, owners, 0)
    values = scores[safe].astype(np.result_type(scores, float), copy=False)
    values = np.where(valid, values, -np.inf)
    result = values.max(axis=1)
    assert np.isfinite(result).all()
    return result


def project_token_scores_to_windows(token_scores) -> np.ndarray:
    """Max-pool lexical-token scalars over every exact eligible 4-BPE window."""
    _, arrays = load_projection()
    scores = np.asarray(token_scores)
    assert scores.shape == (EXPECTED_TOKENS,)
    indices = arrays["window_token_indices"]
    valid = indices >= 0
    lexical = np.zeros_like(valid)
    lexical[valid] = arrays["token_lexical_mask"][indices[valid]].astype(bool)
    safe = np.where(valid, indices, 0)
    values = scores[safe].astype(np.result_type(scores, float), copy=False)
    values = np.where(valid & lexical, values, -np.inf)
    result = values.max(axis=1)
    assert np.isfinite(result).all()
    return result


def project_window_scores_to_answers(window_scores) -> np.ndarray:
    """Max-pool one scalar per eligible window to the 3,680 answer axis."""
    _, arrays = load_projection()
    scores = np.asarray(window_scores)
    assert scores.shape == (EXPECTED_WINDOWS,) and np.isfinite(scores).all()
    indptr = arrays["response_window_indptr"]
    result = np.empty(EXPECTED_RESPONSES, dtype=scores.dtype)
    for index in range(EXPECTED_RESPONSES):
        begin, end = map(int, indptr[index:index + 2])
        assert begin < end
        result[index] = scores[begin:end].max()
    return result


def group_indices(unit: str) -> np.ndarray:
    """Return locked source-connected group indices for a data axis."""
    _, arrays = load_projection()
    if unit == "response":
        return arrays["response_group_index"].astype(np.int32)
    if unit == "claim":
        return arrays["claim_group_index"].astype(np.int32)
    if unit == "token":
        response = np.repeat(np.arange(EXPECTED_RESPONSES),
                             np.diff(arrays["response_token_indptr"]))
        return arrays["response_group_index"][response].astype(np.int32)
    if unit == "window":
        return arrays["response_group_index"][
            arrays["window_response_index"]].astype(np.int32)
    raise ValueError("unit must be response, claim, token, or window")


def assert_source_connected_disjoint(train_indices, held_indices,
                                     unit: str = "window") -> None:
    """Fail if a downstream split puts one source-connected group on both sides."""
    groups = group_indices(unit)
    train = set(map(int, groups[np.asarray(train_indices, dtype=np.int64)]))
    held = set(map(int, groups[np.asarray(held_indices, dtype=np.int64)]))
    assert not (train & held), "Source-connected group leakage"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout-seconds", type=float, default=0.0)
    wait.add_argument("--poll-seconds", type=float, default=15.0)
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "status":
        cache_status(write=True)
    elif args.command == "wait":
        wait_for_cache(args.timeout_seconds, args.poll_seconds)
    else:
        audit_features()


if __name__ == "__main__":
    main()
