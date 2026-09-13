"""Build the sealed 3,839-answer RAGognizer inference plan on CPU.

The only benchmark inputs are previously stripped, label-free feature text and
4-BPE geometry.  No model weights are loaded and no CUDA API is called.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from transformers import AutoTokenizer

from adapter import PLAN_VERSION, character_overlap_map, exact_token_map, sha256_file, sha256_json


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = HERE.parents[3]
FEATURE_INPUTS = PROJECT / "research" / "redeep_formal_baseline_v1" / "feature_inputs.jsonl"
GEOMETRY = PROJECT / "research" / "redeep_formal_baseline_v1" / "evaluation_geometry.jsonl"
LOCAL_BPE_TOKENIZER = WORKSPACE / "prelab" / "models" / "Llama-2-7b-chat-hf"
AUTHOR_TOKENIZER = HERE / "official_base_tokenizer_meta_f5db"
OUTPUT = HERE / "INFERENCE_PLAN.jsonl"
AUDIT = HERE / "CPU_AUDIT.json"

FEATURE_INPUTS_SHA256 = "cd44ebf504d86088ba7b039a1d7263994780ded9ea7a22ba95636e25c4862cb8"
GEOMETRY_SHA256 = "b87c570dd0536d61bd0d720fc7b46906a046f1f523dd83365fc71d4862a06b07"
EXPECTED = {
    "fit": {"answers": 3_680, "groups": 615, "windows": 653_979},
    "calibration": {"answers": 159, "groups": 154, "windows": 42_241},
}
EXPECTED_GENERATORS = {
    "llama-2-7b-chat": 793,
    "gpt-3.5-turbo-0613": 578,
    "mistral-7B-instruct": 594,
    "llama-2-13b-chat": 631,
    "llama-2-70b-chat": 629,
    "gpt-4-0613": 614,
}
EXPECTED_ANSWERS = 3_839
EXPECTED_WINDOWS = 696_220
WRAPPER_LEFT = "<s>[INST] "
WRAPPER_RIGHT = " [/INST] "
WINDOW_RE = re.compile(r"__k4_(\d+)$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def percentile(values: Sequence[int], q: float) -> int:
    ordered = sorted(map(int, values))
    if not ordered:
        raise ValueError("empty percentile input")
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def merge_intervals(intervals: Sequence[Sequence[int]]) -> list[list[int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def author_tokenization(tokenizer, prompt: str, response: str) -> dict:
    """Freeze native token positions and deterministic response coordinates.

    Fast-tokenizer offsets are external coordinates over the exact retokenized
    author input.  For ordinary text they must reproduce the author's
    ``_pack_probs`` positions and spans exactly.  The only accepted fallback is
    byte-fallback Unicode for which the public display packer decodes a single
    byte as U+FFFD; all native probability positions remain present.
    """
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    template_ids = list(map(int, tokenizer.apply_chat_template(
        chat, add_generation_prompt=False, tokenize=True
    )))
    visible_with_specials = tokenizer.decode(template_ids, skip_special_tokens=False)
    encoded = tokenizer(
        visible_with_specials,
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    model_input_ids = list(map(int, encoded["input_ids"]))
    model_offsets = [list(map(int, pair)) for pair in encoded["offset_mapping"]]
    try:
        visible_response_start = visible_with_specials.rindex(response)
        visible_response_lookup = "exact"
    except ValueError:
        stripped = response.strip()
        visible_response_start = visible_with_specials.rindex(stripped)
        visible_response_start -= len(response) - len(response.lstrip())
        visible_response_lookup = "stripped_fallback"
    visible_response_end = visible_response_start + len(response)
    individual_token_texts = tokenizer.batch_decode(
        [[token_id] for token_id in model_input_ids], skip_special_tokens=True
    )
    offset_positions = [
        index
        for index, (start, end) in enumerate(model_offsets)
        if end > visible_response_start
        and start < visible_response_end
        and end > start
        and individual_token_texts[index]
    ]
    raw_offset_intervals = [
        [
            max(0, model_offsets[index][0] - visible_response_start),
            min(len(response), model_offsets[index][1] - visible_response_start),
        ]
        for index in offset_positions
    ]
    if not offset_positions or any(end <= start for start, end in raw_offset_intervals):
        raise ValueError("author tokenizer produced invalid response offsets")
    # Match the author's readable-token convention: whitespace gaps carried by
    # empty single-token decodes attach to the following readable token.  When
    # multiple UTF-8 byte tokens share one character offset, keep the overlap so
    # every native probability remains represented.
    offset_intervals: list[list[int]] = []
    cursor = 0
    for raw_start, raw_end in raw_offset_intervals:
        adjusted_start = cursor if raw_start >= cursor else raw_start
        offset_intervals.append([adjusted_start, raw_end])
        cursor = max(cursor, raw_end)
    if cursor < len(response):
        offset_intervals[-1][1] = len(response)
    readable_full = tokenizer.decode(model_input_ids, skip_special_tokens=True)
    try:
        assistant_starts_at = readable_full.rindex(response)
        assistant_lookup = "exact"
    except ValueError:
        assistant_starts_at = readable_full.rindex(response.strip())
        assistant_lookup = "stripped_fallback"

    single_token_texts = individual_token_texts
    candidate_spans: list[dict] = []
    for probability_index, token_text in enumerate(single_token_texts):
        if not token_text:
            continue
        prefix_text = tokenizer.decode(
            model_input_ids[: probability_index + 1], skip_special_tokens=True
        )
        start = len(prefix_text) - len(token_text)
        if start >= assistant_starts_at:
            candidate_spans.append({
                "probability_index": probability_index,
                "token_id": model_input_ids[probability_index],
                "start": start - assistant_starts_at,
                "end": len(prefix_text) - assistant_starts_at,
                "text": token_text,
            })

    packed: list[dict] = []
    official_pack_failure = None
    try:
        left_over = response
        last_end = 0
        for candidate in candidate_spans:
            token_text = candidate["text"]
            if not token_text or not left_over:
                continue
            try:
                starts_at = left_over.index(token_text)
            except ValueError as exc:
                raise ValueError(
                    f"author _ensure_tokens_in_response cannot place {token_text!r}"
                ) from exc
            ends_at = starts_at + len(token_text)
            packed_text = left_over[:ends_at]
            packed.append({
                "probability_index": candidate["probability_index"],
                "token_id": candidate["token_id"],
                "start": last_end,
                "end": last_end + len(packed_text),
                "text": packed_text,
            })
            left_over = left_over[ends_at:]
            last_end += len(packed_text)
        if left_over:
            raise ValueError(f"author packing leaves response suffix {left_over[:80]!r}")
        if not packed:
            raise ValueError("author packing produced no response tokens")
    except ValueError as exc:
        official_pack_failure = str(exc)

    byte_fallback_local_indices = [
        index
        for index, probability_index in enumerate(offset_positions)
        if "\ufffd" in tokenizer.decode(
            [model_input_ids[probability_index]], skip_special_tokens=True
        )
    ]
    if official_pack_failure is None:
        if [item["probability_index"] for item in packed] != offset_positions:
            raise ValueError("fast offsets do not reproduce author probability positions")
        if [item["token_id"] for item in packed] != [
            model_input_ids[index] for index in offset_positions
        ]:
            raise ValueError("fast offsets do not reproduce author token IDs")
        packed_intervals = [[item["start"], item["end"]] for item in packed]
        raw_offset_span_differences = sum(
            left != right for left, right in zip(packed_intervals, offset_intervals)
        )
        # Author sequential packing is the native readable-token coordinate.
        # Fast offsets independently verify the probability positions/IDs; raw
        # offsets can assign long whitespace runs differently, so the ordinary
        # row keeps the exact author span rather than changing it.
        offset_intervals = packed_intervals
        coordinate_mode = "author_pack_exact_fast_positions_verified"
        coordinate_texts = [item["text"] for item in packed]
    else:
        if not byte_fallback_local_indices or "\ufffd" not in official_pack_failure:
            raise ValueError(
                "author packing failed for a reason other than audited byte fallback: "
                + official_pack_failure
            )
        coordinate_mode = "fast_offsets_author_pack_byte_fallback"
        coordinate_texts = [response[start:end] for start, end in offset_intervals]
        raw_offset_span_differences = None

    covered = [False] * len(response)
    overlap_depth = [0] * len(response)
    for start, end in offset_intervals:
        covered[start:end] = [True] * (end - start)
        for char_index in range(start, end):
            overlap_depth[char_index] += 1
    uncovered = [index for index, flag in enumerate(covered) if not flag]
    if uncovered:
        raise ValueError(f"author coordinates leave response characters uncovered: {uncovered[:5]}")

    return {
        "chat": chat,
        "template_ids": template_ids,
        "visible_with_specials_sha256": sha256_text(visible_with_specials),
        "model_input_ids": model_input_ids,
        "assistant_lookup": assistant_lookup,
        "visible_response_lookup": visible_response_lookup,
        "probability_indices": offset_positions,
        "token_ids": [model_input_ids[index] for index in offset_positions],
        "char_intervals": offset_intervals,
        "texts": coordinate_texts,
        "coordinate_mode": coordinate_mode,
        "official_pack_exact": official_pack_failure is None,
        "official_pack_failure_sha256": (
            sha256_text(official_pack_failure) if official_pack_failure is not None else None
        ),
        "byte_fallback_local_indices": byte_fallback_local_indices,
        "byte_fallback_probability_indices": [
            offset_positions[index] for index in byte_fallback_local_indices
        ],
        "raw_fast_offset_span_differences": raw_offset_span_differences,
        "response_character_coverage": {
            "length": len(response),
            "covered": sum(covered),
            "uncovered": len(uncovered),
            "characters_with_multiple_native_tokens": sum(
                depth > 1 for depth in overlap_depth
            ),
        },
    }


def shared_bpe_tokenization(tokenizer, prompt: str, response: str) -> dict:
    prefix = WRAPPER_LEFT + prompt + WRAPPER_RIGHT
    rendered = prefix + response
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    input_ids = list(map(int, encoded["input_ids"]))
    offsets = [list(map(int, pair)) for pair in encoded["offset_mapping"]]
    response_start = len(prefix)
    response_end = len(rendered)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > response_start and start < response_end and end > start
    ]
    token_ids = [input_ids[index] for index in positions]
    char_intervals = [
        [max(0, offsets[index][0] - response_start), min(len(response), offsets[index][1] - response_start)]
        for index in positions
    ]
    if not token_ids or any(end <= start for start, end in char_intervals):
        raise ValueError("invalid shared answer-token geometry")
    covered = [False] * len(response)
    for start, end in char_intervals:
        covered[start:end] = [True] * (end - start)
    if any(not response[index].isspace() and not covered[index] for index in range(len(response))):
        raise ValueError("shared tokenizer lost non-whitespace response characters")
    return {
        "rendered_sha256": sha256_text(rendered),
        "input_ids_sha256": sha256_json(input_ids),
        "token_positions": positions,
        "token_ids": token_ids,
        "char_intervals": char_intervals,
    }


def load_feature_rows() -> list[dict]:
    if sha256_file(FEATURE_INPUTS) != FEATURE_INPUTS_SHA256:
        raise ValueError("feature input hash changed")
    rows = []
    partition_counts: Counter[str] = Counter()
    partition_groups: dict[str, set[str]] = defaultdict(set)
    generator_counts: Counter[str] = Counter()
    for row in read_jsonl(FEATURE_INPUTS):
        partition = str(row.get("partition"))
        if partition not in EXPECTED:
            raise ValueError(f"unexpected feature partition {partition!r}")
        if row.get("labels_used") is not False:
            raise ValueError("feature input is not explicitly label-free")
        if row.get("official_split") != "train":
            raise ValueError("sealed/non-train RAGTruth material is forbidden")
        generator = str(row.get("generator_model"))
        if generator not in EXPECTED_GENERATORS:
            raise ValueError(f"unexpected generator model {generator!r}")
        if sha256_text(row["released_prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for {row['answer_id']}")
        if sha256_text(row["original_response"]) != row["answer_sha256"]:
            raise ValueError(f"answer hash mismatch for {row['answer_id']}")
        rows.append(row)
        partition_counts[partition] += 1
        partition_groups[partition].add(str(row["group_id"]))
        generator_counts[generator] += 1
    if len(rows) != EXPECTED_ANSWERS:
        raise ValueError(f"expected {EXPECTED_ANSWERS} total rows, got {len(rows)}")
    if len({str(row["answer_id"]) for row in rows}) != EXPECTED_ANSWERS:
        raise ValueError("duplicate answer_id")
    for partition, expected in EXPECTED.items():
        if partition_counts[partition] != expected["answers"]:
            raise ValueError(f"{partition} answer count changed")
        if len(partition_groups[partition]) != expected["groups"]:
            raise ValueError(f"{partition} group count changed")
    overlap = partition_groups["fit"] & partition_groups["calibration"]
    if overlap:
        raise ValueError(f"fit/calibration group leakage: {sorted(overlap)[:3]}")
    if dict(generator_counts) != EXPECTED_GENERATORS:
        raise ValueError(f"generator counts changed: {dict(generator_counts)}")
    return rows


def load_geometry() -> dict[str, list[dict]]:
    if sha256_file(GEOMETRY) != GEOMETRY_SHA256:
        raise ValueError("label-free geometry hash changed")
    allowed = {
        "version", "partition", "response_id", "answer_id", "group_id",
        "window_id", "character_intervals", "labels_present",
    }
    by_answer: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    partition_counts: Counter[str] = Counter()
    partition_groups: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(GEOMETRY):
        partition = str(row.get("partition"))
        if partition not in EXPECTED:
            raise ValueError(f"unexpected geometry partition {partition!r}")
        if set(row) != allowed or row.get("labels_present") is not False:
            raise ValueError("geometry schema is not the frozen label-free schema")
        window_id = str(row["window_id"])
        if window_id in seen:
            raise ValueError(f"duplicate window_id {window_id}")
        seen.add(window_id)
        partition_counts[partition] += 1
        partition_groups[partition].add(str(row["group_id"]))
        match = WINDOW_RE.search(window_id)
        if match is None:
            raise ValueError(f"malformed window_id {window_id}")
        by_answer[str(row["answer_id"])].append({
            "window_id": window_id,
            "token_start": int(match.group(1)),
            "character_intervals": row["character_intervals"],
            "partition": partition,
            "group_id": str(row["group_id"]),
        })
    if len(seen) != EXPECTED_WINDOWS:
        raise ValueError(f"expected {EXPECTED_WINDOWS} total windows, got {len(seen)}")
    for partition, expected in EXPECTED.items():
        if partition_counts[partition] != expected["windows"]:
            raise ValueError(f"{partition} window count changed")
        if len(partition_groups[partition]) != expected["groups"]:
            raise ValueError(f"{partition} geometry group count changed")
    if partition_groups["fit"] & partition_groups["calibration"]:
        raise ValueError("fit/calibration geometry groups overlap")
    return dict(by_answer)


def tokenizer_identity(path: Path, tokenizer) -> dict:
    names = (
        "config.json", "special_tokens_map.json", "tokenizer_config.json",
        "tokenizer.json", "tokenizer.model",
    )
    return {
        "path": str(path.relative_to(WORKSPACE)),
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocab_size": int(tokenizer.vocab_size),
        "length_with_added_tokens": len(tokenizer),
        "files_sha256": {name: sha256_file(path / name) for name in names},
    }


def prepare(output_path: Path, audit_path: Path) -> None:
    rows = load_feature_rows()
    geometry = load_geometry()
    if set(geometry) != {str(row["answer_id"]) for row in rows}:
        raise ValueError("feature/geometry answer sets differ")
    author_tokenizer = AutoTokenizer.from_pretrained(
        AUTHOR_TOKENIZER, use_fast=True, local_files_only=True
    )
    shared_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_BPE_TOKENIZER, use_fast=True, local_files_only=True
    )
    if not author_tokenizer.is_fast or not shared_tokenizer.is_fast:
        raise ValueError("fast tokenizers are required for the character audit")

    author_identity = tokenizer_identity(AUTHOR_TOKENIZER, author_tokenizer)
    shared_identity = tokenizer_identity(LOCAL_BPE_TOKENIZER, shared_tokenizer)
    same_tokenizer = author_identity["files_sha256"] == shared_identity["files_sha256"]
    # Mapping identity is decided before probabilities exist: identical frozen
    # tokenizers use one-to-one token alignment; otherwise every row uses
    # character overlap, even if one observed row happens to tokenize equally.

    plan_rows: list[dict] = []
    exact_rows = 0
    lookup_counts: Counter[str] = Counter()
    coordinate_mode_counts: Counter[str] = Counter()
    byte_fallback_rows: list[dict] = []
    short_window_rows: list[dict] = []
    model_lengths: list[int] = []
    author_lengths: list[int] = []
    shared_lengths: list[int] = []
    mapping_cardinalities: Counter[int] = Counter()
    max_author_token_local_slot_reuse = 0
    total_windows = 0
    partition_counts: dict[str, Counter[str]] = {
        partition: Counter() for partition in EXPECTED
    }
    partition_model_lengths: dict[str, list[int]] = {
        partition: [] for partition in EXPECTED
    }
    for row in rows:
        partition = str(row["partition"])
        answer_id = str(row["answer_id"])
        prompt = row["released_prompt"]
        response = row["original_response"]
        try:
            author = author_tokenization(author_tokenizer, prompt, response)
        except ValueError as exc:
            raise ValueError(f"author coordinate audit failed for {answer_id}: {exc}") from exc
        shared = shared_bpe_tokenization(shared_tokenizer, prompt, response)
        row_exact = False
        try:
            exact_token_map(
                author["token_ids"], author["char_intervals"],
                shared["token_ids"], shared["char_intervals"],
            )
            row_exact = True
            exact_rows += 1
        except ValueError:
            pass
        mapping = (
            exact_token_map(
                author["token_ids"], author["char_intervals"],
                shared["token_ids"], shared["char_intervals"],
            )
            if same_tokenizer
            else character_overlap_map(author["char_intervals"], shared["char_intervals"])
        )
        row_author_tokens_reused: Counter[int] = Counter()
        for indices in mapping:
            mapping_cardinalities[len(indices)] += 1
            for index in indices:
                row_author_tokens_reused[index] += 1
        max_author_token_local_slot_reuse = max(
            max_author_token_local_slot_reuse,
            max(row_author_tokens_reused.values()),
        )

        answer_windows = geometry.get(answer_id)
        if not answer_windows:
            raise ValueError(f"no eligible windows for {answer_id}")
        previous = -1
        if len(shared["token_ids"]) < 4:
            short_window_rows.append({
                "answer_id": answer_id,
                "partition": partition,
                "shared_bpe_tokens": len(shared["token_ids"]),
                "eligible_windows": len(answer_windows),
            })
        for window in answer_windows:
            if window["partition"] != partition or window["group_id"] != str(row["group_id"]):
                raise ValueError(f"feature/geometry partition or group mismatch for {answer_id}")
            start = int(window["token_start"])
            if start <= previous:
                raise ValueError(f"window order changed for {answer_id}")
            previous = start
            shared_count = len(shared["token_ids"])
            if shared_count < 4:
                if len(answer_windows) != 1 or start != 0:
                    raise ValueError(f"invalid project short-window geometry for {answer_id}")
                end = shared_count
            else:
                end = start + 4
            if start < 0 or end > shared_count:
                raise ValueError(f"window outside shared token range for {answer_id}")
            expected_intervals = merge_intervals(shared["char_intervals"][start:end])
            if expected_intervals != window["character_intervals"]:
                raise ValueError(f"window character geometry mismatch: {window['window_id']}")

        payload = {
            "version": PLAN_VERSION,
            "partition": partition,
            "official_split": str(row["official_split"]),
            "response_id": str(row["response_id"]),
            "answer_id": answer_id,
            "source_id": str(row["source_id"]),
            "group_id": str(row["group_id"]),
            "generator_model": str(row["generator_model"]),
            "released_prompt": prompt,
            "original_response": response,
            "prompt_sha256": row["prompt_sha256"],
            "answer_sha256": row["answer_sha256"],
            "chat": author["chat"],
            "author_roundtrip": {
                "apply_chat_template_add_generation_prompt": False,
                "decoded_with_special_tokens_sha256": author["visible_with_specials_sha256"],
                "template_input_ids_sha256": sha256_json(author["template_ids"]),
                "model_input_ids": author["model_input_ids"],
                "model_input_ids_sha256": sha256_json(author["model_input_ids"]),
                "assistant_lookup": author["assistant_lookup"],
            },
            "author_response_tokens": {
                "probability_indices": author["probability_indices"],
                "token_ids": author["token_ids"],
                "char_intervals": author["char_intervals"],
                "coordinate_texts": author["texts"],
                "coordinate_mode": author["coordinate_mode"],
                "official_pack_exact": author["official_pack_exact"],
                "official_pack_failure_sha256": author["official_pack_failure_sha256"],
                "byte_fallback_local_indices": author["byte_fallback_local_indices"],
                "byte_fallback_probability_indices": author["byte_fallback_probability_indices"],
                "raw_fast_offset_span_differences": author[
                    "raw_fast_offset_span_differences"
                ],
                "response_character_coverage": author["response_character_coverage"],
            },
            "shared_bpe_tokens": {
                "wrapper": WRAPPER_LEFT + "{released_prompt}" + WRAPPER_RIGHT + "{original_response}",
                "rendered_sha256": shared["rendered_sha256"],
                "input_ids_sha256": shared["input_ids_sha256"],
                "token_positions": shared["token_positions"],
                "token_ids": shared["token_ids"],
                "char_intervals": shared["char_intervals"],
            },
            "mapping": {
                "mode": "exact_token" if same_tokenizer else "character_overlap",
                "tokenizers_identical": same_tokenizer,
                "row_tokens_happen_to_be_exact": row_exact,
                "local_to_author_token_indices": mapping,
                "within_local_slot_reduction": "arithmetic_mean_over_each_intersecting_author_token_once",
                "overlap_rule": "max(local_start,author_start) < min(local_end,author_end)",
            },
            "eligible_windows": [
                {
                    "window_id": window["window_id"],
                    "token_start": window["token_start"],
                    "slot_count": (
                        4 if len(shared["token_ids"]) >= 4 else len(shared["token_ids"])
                    ),
                    "character_intervals": window["character_intervals"],
                }
                for window in answer_windows
            ],
            "window_reduction": "arithmetic_mean_of_four_shared_bpe_slots_or_all_N_slots_when_answer_N_lt_4",
            "answer_reduction": "maximum_over_all_eligible_shared_k4_windows",
        }
        payload["plan_row_sha256"] = sha256_json(payload)
        plan_rows.append(payload)
        model_lengths.append(len(author["model_input_ids"]))
        author_lengths.append(len(author["token_ids"]))
        shared_lengths.append(len(shared["token_ids"]))
        lookup_counts[author["assistant_lookup"]] += 1
        coordinate_mode_counts[author["coordinate_mode"]] += 1
        if not author["official_pack_exact"]:
            byte_fallback_rows.append({
                "answer_id": answer_id,
                "partition": partition,
                "probability_indices": author["byte_fallback_probability_indices"],
                "native_token_count": len(author["token_ids"]),
                "response_characters": len(response),
                "characters_with_multiple_native_tokens": author[
                    "response_character_coverage"
                ]["characters_with_multiple_native_tokens"],
                "official_pack_failure_sha256": author["official_pack_failure_sha256"],
            })
        total_windows += len(answer_windows)
        partition_counts[partition]["answers"] += 1
        partition_counts[partition]["windows"] += len(answer_windows)
        partition_counts[partition]["author_response_tokens"] += len(author["token_ids"])
        partition_counts[partition]["shared_bpe_tokens"] += len(shared["token_ids"])
        partition_counts[partition]["row_exact_tokenizations"] += int(row_exact)
        partition_model_lengths[partition].append(len(author["model_input_ids"]))

    if total_windows != EXPECTED_WINDOWS:
        raise ValueError("plan window total changed")
    for partition, expected in EXPECTED.items():
        if partition_counts[partition]["answers"] != expected["answers"]:
            raise ValueError(f"{partition} plan answer count changed")
        if partition_counts[partition]["windows"] != expected["windows"]:
            raise ValueError(f"{partition} plan window count changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    audit = {
        "version": "ragognizer-full-plan-cpu-audit-v1",
        "status": "cpu_plan_complete_gpu_not_started",
        "sealed_test_opened": False,
        "gpu_workload_started": False,
        "inputs": {
            "feature_inputs": {"sha256": FEATURE_INPUTS_SHA256, "partitions_used": ["fit", "calibration"]},
            "evaluation_geometry": {"sha256": GEOMETRY_SHA256, "partitions_used": ["fit", "calibration"]},
        },
        "counts": {
            "answers": len(plan_rows),
            "groups": len({row["group_id"] for row in plan_rows}),
            "eligible_windows": total_windows,
            "author_response_tokens": sum(author_lengths),
            "shared_bpe_tokens": sum(shared_lengths),
        },
        "partition_counts": {
            partition: {
                "answers": int(partition_counts[partition]["answers"]),
                "groups": len({row["group_id"] for row in plan_rows if row["partition"] == partition}),
                "eligible_windows": int(partition_counts[partition]["windows"]),
                "author_response_tokens": int(partition_counts[partition]["author_response_tokens"]),
                "shared_bpe_tokens": int(partition_counts[partition]["shared_bpe_tokens"]),
                "row_exact_tokenizations": int(partition_counts[partition]["row_exact_tokenizations"]),
                "author_model_input_tokens": {
                    "sum": sum(partition_model_lengths[partition]),
                    "min": min(partition_model_lengths[partition]),
                    "p50": percentile(partition_model_lengths[partition], 0.50),
                    "p95": percentile(partition_model_lengths[partition], 0.95),
                    "max": max(partition_model_lengths[partition]),
                },
            }
            for partition in EXPECTED
        },
        "group_isolation": {
            "fit_groups": EXPECTED["fit"]["groups"],
            "calibration_groups": EXPECTED["calibration"]["groups"],
            "intersection": 0,
        },
        "generator_counts": dict(sorted(Counter(row["generator_model"] for row in plan_rows).items())),
        "author_model_input_tokens": {
            "sum": sum(model_lengths),
            "min": min(model_lengths),
            "p50": percentile(model_lengths, 0.50),
            "p95": percentile(model_lengths, 0.95),
            "max": max(model_lengths),
            "llama2_context_limit": 4096,
            "all_within_limit": max(model_lengths) <= 4096,
        },
        "assistant_lookup": dict(sorted(lookup_counts.items())),
        "author_coordinate_audit": {
            "mode_counts": dict(sorted(coordinate_mode_counts.items())),
            "ordinary_rows_author_pack_exact_and_fast_positions_equal": coordinate_mode_counts[
                "author_pack_exact_fast_positions_verified"
            ],
            "byte_fallback_row_count": len(byte_fallback_rows),
            "byte_fallback_rows": byte_fallback_rows,
            "all_response_characters_covered": True,
            "all_author_readable_probability_positions_retained": True,
            "fallback_is_model_external_and_label_free": True,
        },
        "short_window_geometry": {
            "rule": "N>=4 uses four slots; N<4 keeps one start-0 window over all N slots",
            "row_count": len(short_window_rows),
            "rows": short_window_rows,
        },
        "tokenizer_identity": {
            "author_meta_llama_revision": "f5db02db724555f92da89c216ac04704f23d4590",
            "author": author_identity,
            "shared_bpe": shared_identity,
            "identical": same_tokenizer,
            "decisive_difference": None if same_tokenizer else "tokenizer.json SHA256 differs",
        },
        "mapping": {
            "global_mode": "exact_token" if same_tokenizer else "character_overlap",
            "rows_whose_observed_tokens_happen_to_match_exactly": exact_rows,
            "rows_requiring_nontrivial_overlap": len(plan_rows) - exact_rows,
            "local_slot_source_count_histogram": {
                str(key): value for key, value in sorted(mapping_cardinalities.items())
            },
            "max_local_slots_receiving_one_author_token": max_author_token_local_slot_reuse,
            "labels_read_by_plan_builder": False,
        },
        "software": {
            "python": __import__("sys").version.split()[0],
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "plan": {"path": output_path.name, "sha256": sha256_file(output_path)},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.audit.resolve())


if __name__ == "__main__":
    main()
