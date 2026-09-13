"""CPU-verifiable runner skeleton for the forced-evidence quote probe.

The executable surface intentionally contains CPU-only stages.  A later GPU
runner must generate greedily, then extract every white-box value by one full
sequence, ``use_cache=False`` replay.  Existing no-cache Q/K hooks must never be
attached to a KV-cache decode step.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/forced_evidence_quote_probe_v1"
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
MODEL = ROOT.parent / "models/Llama-2-7b-chat-hf"
PLAN_PATH = RESEARCH / "PLAN.json"
PROTOCOL_PATH = RESEARCH / "PROTOCOL.md"
REVIEW_PATH = RESEARCH / "INDEPENDENT_PROTOCOL_REVIEW.json"
INPUT_PATH = OUT / "label_free_inputs.jsonl"
MANIFEST_PATH = OUT / "MANIFEST.json"

EXPECTED_ROWS = 3_776
EXPECTED_ANSWERS = 256
MODEL_VALUE = "llama-2-7b-chat"
WHITEBOX_EXTRACTION_MODE = "post_generation_full_sequence_no_cache_replay"
PROMPT_TOKENIZER_FLAGS = {
    "add_special_tokens": False, "padding": False, "truncation": False,
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)%?")
NEGATIONS = frozenset((
    "no", "not", "never", "none", "neither", "nor", "without", "cannot",
    "can't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "didn't",
))
ASCII_EDGE = " \t\r\n"
OUTPUT_FIELDS = (
    "response_id", "source_id", "group_id", "model", "question",
    "passage_1", "passage_2", "passage_3", "microclaim_id",
    "microclaim_index", "claim_start", "claim_end", "claim_text_raw",
    "claim_prompt_text",
)
FORBIDDEN_KEYS = frozenset((
    "original_response", "quality", "labels", "gold_label", "risk_mask",
    "risk_bpe_indices", "risk_bpe_fraction", "label_type", "answer_risk",
    "existing_scores",
))


class GPUStillBlocked(RuntimeError):
    pass


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path: Path, value) -> None:
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def reject_forbidden(value) -> None:
    if isinstance(value, dict):
        overlap = set(value) & FORBIDDEN_KEYS
        assert not overlap, sorted(overlap)
        for child in value.values():
            reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            reject_forbidden(child)


def load_sanitized_rows():
    manifest = read_json(MANIFEST_PATH)
    assert manifest["status"] == "CPU_inputs_ready_GPU_and_scoring_blocked"
    assert manifest["sole_GPU_input"] == INPUT_PATH.name
    assert sha(INPUT_PATH) == manifest["files_sha256"][INPUT_PATH.name]
    rows = []
    with INPUT_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert tuple(row) == OUTPUT_FIELDS
            reject_forbidden(row)
            assert row["model"] == MODEL_VALUE
            rows.append(row)
    assert len(rows) == EXPECTED_ROWS
    assert len({row["response_id"] for row in rows}) == EXPECTED_ANSWERS
    return rows, manifest


def render_prompt_with_regions(template: str, row: dict):
    placeholders = ("{claim}", "{p1}", "{p2}", "{p3}")
    names = ("claim", "passage_1", "passage_2", "passage_3")
    values = (row["claim_prompt_text"], row["passage_1"],
              row["passage_2"], row["passage_3"])
    pieces, regions, cursor = [], {}, 0
    remaining = template
    for placeholder, name, value in zip(placeholders, names, values):
        assert remaining.count(placeholder) == 1
        left, remaining = remaining.split(placeholder, 1)
        pieces.extend((left, value)); cursor += len(left)
        regions[name] = [cursor, cursor + len(value)]
        cursor += len(value)
    pieces.append(remaining)
    text = "".join(pieces)
    assert text == template.format(claim=values[0], p1=values[1],
                                   p2=values[2], p3=values[3])
    ordered = [regions[name] for name in names]
    assert all(left < right for left, right in ordered)
    assert all(ordered[index][1] < ordered[index + 1][0]
               for index in range(len(ordered) - 1))
    return text, regions


def region_token_positions(text: str, offsets, regions: dict):
    """Assign every non-special token overlapping a marked character region.

    A tokenizer token may straddle a prompt delimiter.  It belongs to the
    region with the largest character overlap; an exact tie is rejected rather
    than silently giving the token to a region by iteration order.
    """
    output = {name: [] for name in regions}
    for token_index, (left, right) in enumerate(offsets):
        left, right = int(left), int(right)
        if right <= left:  # fast-tokenizer special tokens use (0, 0)
            continue
        overlaps = []
        for name, (begin, end) in regions.items():
            overlap = max(0, min(right, end) - max(left, begin))
            if overlap:
                overlaps.append((overlap, name))
        if not overlaps:
            continue
        best = max(overlap for overlap, _name in overlaps)
        owners = sorted(name for overlap, name in overlaps if overlap == best)
        assert len(owners) == 1, (
            "equal character overlap across prompt regions", token_index,
            left, right, owners,
        )
        output[owners[0]].append(token_index)
    assert all(output.values())
    return output


def exact_occurrences(needle: str, passage: str):
    if not needle:
        return []
    positions, cursor = [], 0
    while True:
        start = passage.find(needle, cursor)
        if start < 0:
            break
        positions.append((start, start + len(needle)))
        cursor = start + 1
    return positions


def parse_generated_quote(generated_text: str, stop_reason: str, passages):
    assert stop_reason in ("close", "eos", "max")
    close = generated_text.find("</quote>")
    missing_close = close < 0
    raw_inner = generated_text if missing_close else generated_text[:close]
    outside = "" if missing_close else generated_text[close + len("</quote>"):]
    logical = raw_inner.strip(ASCII_EDGE)
    nested_or_second = (
        "<quote>" in raw_inner or "<quote>" in outside or
        "</quote>" in outside
    )
    nonwhite_outside = bool(outside.strip(ASCII_EDGE))
    failures = []
    if not logical:
        failures.append("empty_quote")
    if missing_close:
        failures.append("missing_close")
    if nested_or_second:
        failures.append("nested_or_second_quote_tag")
    if nonwhite_outside:
        failures.append("nonwhitespace_outside_tag")
    if stop_reason == "eos":
        failures.append("eos_before_close")
    if stop_reason == "max":
        failures.append("max_tokens_before_close")
    matches = []
    for passage_id, passage in enumerate(passages, 1):
        for start, end in exact_occurrences(logical, passage):
            matches.append({"passage_id": passage_id, "char_start": start,
                            "char_end": end})
    matches.sort(key=lambda item: (item["passage_id"], item["char_start"]))
    return {
        "raw_inner": raw_inner, "logical_quote": logical,
        "outside_after_close": outside, "stop_reason": stop_reason,
        "parse_valid": not failures, "failure_reasons": sorted(set(failures)),
        "exact_matches": matches, "source_exact_substring": bool(matches),
    }


def terms(text: str):
    return {match.group(0).casefold() for match in WORD_RE.finditer(text)}


def numbers(text: str):
    return {match.group(0).replace(",", "") for match in NUMBER_RE.finditer(text)}


def has_negation(text: str):
    return bool(terms(text) & NEGATIONS)


def lcs_length(left: str, right: str) -> int:
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for char_left in left:
        current = [0]
        for index, char_right in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if char_left == char_right
                           else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def surface_features(parsed: dict, claim: str, passages, tokenizer,
                     sentence_candidates=()):
    quote = parsed["logical_quote"]
    matches = parsed["exact_matches"]
    primary = matches[0] if matches else None
    claim_terms, quote_terms = terms(claim), terms(quote)
    union = claim_terms | quote_terms
    common = claim_terms & quote_terms
    claim_numbers, quote_numbers = numbers(claim), numbers(quote)
    number_union = claim_numbers | quote_numbers
    lcs = lcs_length(claim, quote)
    exact_bm25 = [float(item["bm25"]) for item in sentence_candidates
                  if item["text"] == quote]
    relative = 0.0
    if primary is not None:
        plen = len(passages[primary["passage_id"] - 1])
        relative = primary["char_start"] / max(1, plen - len(quote))
    features = np.asarray([
        float(parsed["parse_valid"]), float(bool(matches)), float(len(matches) > 1),
        float(parsed["stop_reason"] == "close"),
        float(parsed["stop_reason"] == "eos"),
        float(parsed["stop_reason"] == "max"),
        float(len(tokenizer.encode(quote, add_special_tokens=False))), float(len(quote)),
        *[float(any(item["passage_id"] == passage_id for item in matches))
          for passage_id in (1, 2, 3)],
        relative, float(len(matches)),
        len(common) / max(1, len(claim_terms)),
        len(common) / max(1, len(quote_terms)),
        len(common) / max(1, len(union)), max(exact_bm25, default=0.0),
        lcs / max(1, len(claim)), lcs / max(1, len(quote)),
        len(claim_numbers ^ quote_numbers) / max(1, len(number_union)),
        float(has_negation(claim) != has_negation(quote)),
    ], dtype=np.float32)
    assert features.shape == (21,) and np.isfinite(features).all()
    return features


def claim_window_scores(response: str, token_offsets, claim_rows, claim_scores,
                        width: int = 4):
    assert len(claim_rows) == len(claim_scores)
    owners, lexical = [], []
    for left, right in token_offsets:
        chars = [position for position in range(left, right)
                 if 0 <= position < len(response) and response[position].isalnum()]
        lexical.append(bool(chars))
        owners.append([index for index, claim in enumerate(claim_rows)
                       if any(int(claim["claim_start"]) <= position <
                              int(claim["claim_end"]) for position in chars)])
        assert not chars or owners[-1]
    starts, scores = [], []
    for start in range(max(1, len(token_offsets) - width + 1)):
        stop = min(len(token_offsets), start + width)
        if not any(lexical[start:stop]):
            continue
        local = {owner for token_owners in owners[start:stop] for owner in token_owners}
        starts.append(start)
        scores.append(max((float(claim_scores[index]) for index in local), default=0.0))
    return np.asarray(starts, dtype=np.int32), np.asarray(scores, dtype=np.float64)


def full_no_cache_replay_contract(prompt_ids, generated_ids, relation_suffix_ids=()):
    """Return the sole allowed white-box sequence contract; no model call here."""
    full = list(prompt_ids) + list(generated_ids) + list(relation_suffix_ids)
    assert prompt_ids and generated_ids
    return {
        "input_ids": full,
        "use_cache": False,
        "past_key_value_allowed": False,
        "mode": WHITEBOX_EXTRACTION_MODE,
        "generated_token_start": len(prompt_ids),
        "generated_token_end": len(prompt_ids) + len(generated_ids),
        "relation_suffix_start": len(prompt_ids) + len(generated_ids),
    }


class GPUExecutionSkeleton:
    """Frozen high-level execution order; deliberately has no GPU entrypoint.

    1. Greedy quote generation may use KV cache only to obtain token IDs.
    2. Discard generation hooks/cache for white-box extraction.
    3. Replay prompt+generated tokens as one full sequence with use_cache=False.
    4. Attach the already-audited no-cache Q/K hook only to step 3.

    The frozen protocol now defines replay tolerances and malformed-quote
    relation handling.  This CPU preparation artifact still cannot initialize
    a model or CUDA: GPU execution belongs in a separately reviewed runner.
    """

    @staticmethod
    def gpu_smoke(*_args, **_kwargs):
        raise GPUStillBlocked(
            "GPU execution is absent from this CPU preparation artifact; add "
            "a separately reviewed executable GPU runner.")

    @staticmethod
    def extract(*_args, **_kwargs):
        raise GPUStillBlocked("Full GPU extraction and scoring are not exposed.")


def random_mapping_test(seed=20_261_023, trials=128):
    rng = random.Random(seed)
    checked_windows = 0
    for _ in range(trials):
        count = rng.randint(4, 20)
        words_local = [f"w{index}_{rng.randint(0, 999)}" for index in range(count)]
        response = " ".join(words_local)
        offsets, cursor = [], 0
        for word in words_local:
            start = response.index(word, cursor); end = start + len(word)
            offsets.append((start, end)); cursor = end
        cuts = sorted(set((0, count, *[rng.randint(1, count - 1)
                                      for _ in range(rng.randint(1, 4))])))
        claims = []
        for left_word, right_word in zip(cuts[:-1], cuts[1:]):
            claims.append({"claim_start": offsets[left_word][0],
                           "claim_end": offsets[right_word - 1][1]})
        scores = np.asarray([rng.random() for _ in claims])
        starts, actual = claim_window_scores(response, offsets, claims, scores)
        expected = []
        for start in starts:
            token_indices = range(int(start), min(count, int(start) + 4))
            active = set()
            for token_index in token_indices:
                for claim_index, claim in enumerate(claims):
                    if max(offsets[token_index][0], claim["claim_start"]) < min(
                            offsets[token_index][1], claim["claim_end"]):
                        active.add(claim_index)
            expected.append(max((scores[index] for index in active), default=0.0))
        np.testing.assert_array_equal(actual, np.asarray(expected))
        checked_windows += len(starts)
    return {"seed": seed, "trials": trials, "windows_checked": checked_windows}


def cpu_selfcheck():
    assert not torch.cuda.is_initialized()
    rows, manifest = load_sanitized_rows()
    plan = read_json(PLAN_PATH)
    review = read_json(REVIEW_PATH)
    assert review["execution_decision"]["cpu_manifest_implementation_may_proceed"]
    assert not review["execution_decision"]["gpu_smoke_allowed_now"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    assert tokenizer.is_fast
    template = plan["quote_prompt"]["utf8_template"]
    lengths = []
    for row in rows:
        prompt, _regions = render_prompt_with_regions(template, row)
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    order = sorted(range(len(rows)), key=lambda index: (lengths[index],
                   rows[index]["microclaim_id"]))
    quantile_positions = [0, len(order) // 7, 2 * len(order) // 7,
                          3 * len(order) // 7, 4 * len(order) // 7,
                          5 * len(order) // 7, 6 * len(order) // 7, len(order) - 1]
    selected = [order[position] for position in quantile_positions]
    prompt_checks = []
    for index in selected:
        row = rows[index]
        prompt, regions = render_prompt_with_regions(template, row)
        encoded = tokenizer(prompt, add_special_tokens=False,
                            return_offsets_mapping=True)
        positions = region_token_positions(prompt, encoded["offset_mapping"], regions)
        prompt_checks.append({
            "row_index": index, "microclaim_id": row["microclaim_id"],
            "prompt_tokens": len(encoded["input_ids"]),
            "prompt_sha256": digest(prompt),
            "region_token_counts": {key: len(value) for key, value in positions.items()},
        })

    punctuation_positions = region_token_positions(
        "A!?B", [(0, 1), (1, 2), (2, 3), (3, 4), (0, 0)],
        {"left": (0, 2), "right": (2, 4)},
    )
    assert punctuation_positions == {"left": [0, 1], "right": [2, 3]}
    boundary_positions = region_token_positions(
        "ab|cde", [(0, 1), (1, 5), (5, 6)],
        {"left": (0, 2), "right": (3, 6)},
    )
    assert boundary_positions == {"left": [0], "right": [1, 2]}
    equal_overlap_rejected = False
    try:
        region_token_positions(
            "ab|cd", [(0, 1), (1, 4), (4, 5)],
            {"left": (0, 2), "right": (3, 5)},
        )
    except AssertionError:
        equal_overlap_rejected = True
    assert equal_overlap_rejected

    synthetic_passages = [
        "Alpha has 12 units. Shared evidence.",
        "Beta has no units. Shared evidence.",
        "Gamma is separate.",
    ]
    valid = parse_generated_quote("Alpha has 12 units.</quote>", "close", synthetic_passages)
    assert valid["parse_valid"] and valid["source_exact_substring"]
    features = surface_features(valid, "Alpha has 10 units.", synthetic_passages,
                                tokenizer, [{"text": "Alpha has 12 units.", "bm25": 1.25}])
    duplicate = parse_generated_quote("Shared evidence.</quote>", "close", synthetic_passages)
    assert duplicate["parse_valid"] and len(duplicate["exact_matches"]) == 2
    invalid_cases = {
        "empty": parse_generated_quote(" </quote>", "close", synthetic_passages),
        "missing": parse_generated_quote("Alpha has 12 units.", "max", synthetic_passages),
        "nested": parse_generated_quote("<quote>Alpha</quote>", "close", synthetic_passages),
        "outside": parse_generated_quote("Alpha</quote>extra", "close", synthetic_passages),
    }
    assert all(not value["parse_valid"] for value in invalid_cases.values())
    replay = full_no_cache_replay_contract([1, 2, 3], [4, 5], [6])
    assert replay["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert replay["use_cache"] is False and replay["past_key_value_allowed"] is False
    mapping = random_mapping_test()
    result = {
        "status": "passed_CPU_only",
        "manifest_sha256": sha(MANIFEST_PATH),
        "sole_GPU_input_sha256": manifest["files_sha256"][INPUT_PATH.name],
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(PROTOCOL_PATH), "plan_sha256": sha(PLAN_PATH),
        "rows": len(rows), "answers": len({row["response_id"] for row in rows}),
        "prompt_length_min": min(lengths), "prompt_length_max": max(lengths),
        "prompt_quantile_checks": prompt_checks,
        "attention_region_mapping": {
            "punctuation_tokens_included": True,
            "boundary_crossing_rule": "largest_character_overlap",
            "equal_overlap_rejected": True,
            "special_zero_length_offsets_ignored": True,
        },
        "quote_parser_cases": {"valid_exact": True, "duplicate_exact": True,
                               "invalid_cases": sorted(invalid_cases)},
        "surface_feature_shape": list(features.shape),
        "random_claim_to_4BPE_mapping": mapping,
        "whitebox_extraction_mode": WHITEBOX_EXTRACTION_MODE,
        "full_no_cache_replay_contract_passed": True,
        "gpu_entrypoint_exposed": False,
        "GPU_used": False, "model_weights_loaded": False,
        "calibration_read": False, "official_test_read": False,
        "gold_bearing_source_JSON_opened_by_selfcheck": False,
        "sanitized_manifest_contains_gold_or_label_fields": False,
        "scoring_run": False,
        "original_review_items_closed_by_frozen_protocol_or_manifest": [
            item["id"] for item in review["blocking_items"]
        ],
        "GPU_execution_status": (
            "not_exposed_by_CPU_skeleton; separate reviewed runner required"
        ),
    }
    write_json_new(OUT / "CPU_SELFCHECK.json", result)
    assert not torch.cuda.is_initialized()
    print("FORCED_QUOTE_CPU_SELFCHECK_PASSED", len(rows), mapping["windows_checked"], flush=True)


def describe():
    plan = read_json(PLAN_PATH)
    print(json.dumps({
        "status": "CPU skeleton only; GPU and scoring blocked",
        "whitebox_extraction_mode": WHITEBOX_EXTRACTION_MODE,
        "resources": plan["resources"],
        "input": str(INPUT_PATH.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cpu-selfcheck", "describe"))
    args = parser.parse_args()
    {"cpu-selfcheck": cpu_selfcheck, "describe": describe}[args.stage]()
