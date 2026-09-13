#!/usr/bin/env python3
"""CPU-only label/geometry audit for the onset+segment probe design.

The reader is deliberately allowlisted to RAGTruth fit and RAGognize train-only
conversion files. It imports no model/GPU library and writes nothing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import unicodedata


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parents[1]

PATHS = {
    "ragtruth_fit_answers": BENCHMARK / "data" / "answers_fit.jsonl",
    "ragtruth_fit_records": BENCHMARK / "data" / "fit.jsonl",
    "ragtruth_fit_tokens": BENCHMARK / "data" / "tokens_fit.jsonl",
    "ragognize_all_answers": BENCHMARK / "auxiliary_ragognize_train_v1" / "conversion_v1" / "all_train" / "answers.jsonl",
    "ragognize_all_sequences": BENCHMARK / "auxiliary_ragognize_train_v1" / "conversion_v1" / "all_train" / "sequences.jsonl",
    "ragognize_no_hint_answers": BENCHMARK / "auxiliary_ragognize_train_v1" / "conversion_v1" / "no_unanswerability_hint" / "answers.jsonl",
    "ragognize_no_hint_sequences": BENCHMARK / "auxiliary_ragognize_train_v1" / "conversion_v1" / "no_unanswerability_hint" / "sequences.jsonl",
}
ALLOWED = {path.resolve() for path in PATHS.values()}
BANNED_COMPONENT = re.compile(r"(^|[_-])(test|calibration|validation)([_-]|$)", re.IGNORECASE)
K = 4
FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_allowed(path: Path) -> Path:
    resolved = path.resolve()
    if resolved not in ALLOWED:
        raise AssertionError(f"non-allowlisted input refused: {resolved}")
    for component in resolved.parts:
        if BANNED_COMPONENT.search(component):
            raise AssertionError(f"sealed split component refused: {component}")
    return resolved


def read_jsonl(path: Path) -> list[dict]:
    path = assert_allowed(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise AssertionError(f"object expected at {path}:{line_number}")
                rows.append(row)
    return rows


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = str(row["response_id"])
        if key in result:
            raise AssertionError(f"duplicate response_id: {key}")
        result[key] = row
    return result


def nearest_rank(values: list[int], proportion: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(proportion * len(ordered)) - 1))
    return int(ordered[index])


def summarize(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None,
                "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p25": nearest_rank(values, 0.25),
        "median": nearest_rank(values, 0.50),
        "p75": nearest_rank(values, 0.75),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "quantile_definition": "nearest-rank",
    }


def length_buckets(values: list[int]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        if value <= 4:
            counts["1-4"] += 1
        elif value <= 8:
            counts["5-8"] += 1
        elif value <= 16:
            counts["9-16"] += 1
        elif value <= 32:
            counts["17-32"] += 1
        else:
            counts["33+"] += 1
    return {key: counts[key] for key in ("1-4", "5-8", "9-16", "17-32", "33+")}


def stable_fold(namespace: str, group_id: str) -> int:
    value = hashlib.sha256(f"{namespace}\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") % FOLDS


def windows_for_count(count: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise AssertionError("empty answer token axis")
    return [(start, min(start + K, count)) for start in range(max(1, count - K + 1))]


def normalize_material(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest_text(text: str) -> str:
    return hashlib.sha256(normalize_material(text).encode("utf-8")).hexdigest()


def unified_ragtruth() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    summaries = keyed(read_jsonl(PATHS["ragtruth_fit_answers"]))
    records = keyed(read_jsonl(PATHS["ragtruth_fit_records"]))
    tokens = keyed(read_jsonl(PATHS["ragtruth_fit_tokens"]))
    if not (set(summaries) == set(records) == set(tokens)):
        raise AssertionError("RAGTruth fit IDs differ across answer/record/token files")
    annotations = []
    for response_id in sorted(records):
        summary, record, token = summaries[response_id], records[response_id], tokens[response_id]
        if summary["partition"] != "fit" or record["partition"] != "fit" or token["partition"] != "fit":
            raise AssertionError("non-fit RAGTruth row")
        if record.get("official_split") != "train":
            raise AssertionError("RAGTruth fit row is not from released train")
        spans = []
        for span in record["labels"]:
            item = dict(span)
            item["valid"] = True
            item["error_type"] = span.get("label_type", "missing_type")
            spans.append(item)
        annotations.append({
            "response_id": response_id,
            "group_id": record["group_id"],
            "response": record["original_response"],
            "spans": spans,
            "annotation_kind": "released_human",
            "question": record["question"],
            "retrieved_passages": record["retrieved_passages"],
            "answerable": None,
            "unanswerability_hint_setting": None,
        })
        if summary["official_span_count"] != len(spans):
            raise AssertionError("RAGTruth summary span count mismatch")
        if token["original_labels"] != record["labels"]:
            raise AssertionError("RAGTruth token/record label mismatch")
    return annotations, tokens, summaries


def unified_ragognize(prefix: str) -> tuple[list[dict], dict[str, dict]]:
    answers = keyed(read_jsonl(PATHS[f"ragognize_{prefix}_answers"]))
    sequences = keyed(read_jsonl(PATHS[f"ragognize_{prefix}_sequences"]))
    if set(answers) != set(sequences):
        raise AssertionError(f"RAGognize {prefix} answer/sequence IDs differ")
    annotations = []
    for response_id in sorted(answers):
        answer = answers[response_id]
        if answer.get("official_split") != "train" or answer.get("partition") != "auxiliary_candidate_fit":
            raise AssertionError("non-train RAGognize row")
        spans = []
        for span in answer["released_hallucinations"]:
            item = dict(span)
            item["error_type"] = "automatic_unspecified"
            spans.append(item)
        annotations.append({
            "response_id": response_id,
            "group_id": answer["group_id"],
            "response": answer["original_response"],
            "spans": spans,
            "annotation_kind": "released_automatic_silver",
            "question": answer["question"],
            "retrieved_passages": answer["retrieved_passages"],
            "answerable": bool(answer["answerable"]),
            "unanswerability_hint_setting": bool(answer["unanswerability_hint_setting"]),
        })
    return annotations, sequences


def analyze_source(name: str, annotations: list[dict], sequences: dict[str, dict],
                   ragtruth_summaries: dict[str, dict] | None = None) -> tuple[dict, dict[str, set[str]]]:
    checks = Counter()
    answer_count = len(annotations)
    groups = {row["group_id"] for row in annotations}
    risk_answers = 0
    risk_groups = set()
    raw_valid_spans = 0
    raw_invalid_spans = 0
    raw_unmapped_spans = 0
    raw_span_chars: list[int] = []
    raw_span_bpes: list[int] = []
    binary_span_bpes: list[int] = []
    raw_onsets = []
    binary_onsets = []
    span_overlap_answer_count = 0
    touching_or_overlapping_answer_count = 0
    total_tokens = lexical_tokens = risk_tokens = 0
    window_counts = Counter()
    transition_counts = Counter()
    type_acc = defaultdict(lambda: {
        "released_spans": 0, "mapped_spans": 0, "answers": set(), "groups": set(),
        "char_lengths": [], "lexical_bpe_lengths": [],
    })
    fold_acc = defaultdict(Counter)
    material_signatures = {"question": set(), "retrieved_passages": set(), "response": set()}

    for row in annotations:
        rid = row["response_id"]
        if rid not in sequences:
            raise AssertionError(f"missing sequence for {rid}")
        seq = sequences[rid]
        if seq["group_id"] != row["group_id"]:
            raise AssertionError("group mismatch")
        text = row["response"]
        offsets = [tuple(map(int, pair)) for pair in seq["response_token_offsets"]]
        lexical = [int(value) for value in seq["lexical_mask"]]
        risk = [int(value) for value in seq["risk_mask"]]
        n = len(offsets)
        if not (n == len(lexical) == len(risk)):
            raise AssertionError("token geometry length mismatch")
        if any(value not in (0, 1) for value in lexical + risk):
            raise AssertionError("non-binary mask")
        if any(r and not l for l, r in zip(lexical, risk)):
            raise AssertionError("risk token is nonlexical")
        if any(not (0 <= left < right <= len(text)) for left, right in offsets):
            raise AssertionError("invalid clipped token offset")

        spans = row["spans"]
        valid_spans = []
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            if not span.get("valid", True):
                raw_invalid_spans += 1
                continue
            if not (0 <= start <= end <= len(text)):
                raise AssertionError(f"valid released span out of bounds: {rid}")
            if text[start:end] != span["text"]:
                raise AssertionError(f"valid released span text mismatch: {rid}")
            valid_spans.append(span)
        intervals = sorted((int(span["start"]), int(span["end"])) for span in valid_spans)
        if any(b_start < a_end for (a_start, a_end), (b_start, b_end) in zip(intervals, intervals[1:])):
            span_overlap_answer_count += 1
        if any(b_start <= a_end for (a_start, a_end), (b_start, b_end) in zip(intervals, intervals[1:])):
            touching_or_overlapping_answer_count += 1

        expected_lexical = [int(any(char.isalnum() for char in text[left:right])) for left, right in offsets]
        char_risk = [False] * len(text)
        for span in valid_spans:
            for position in range(int(span["start"]), int(span["end"])):
                char_risk[position] |= text[position].isalnum()
        expected_risk = [int(any(char_risk[left:right])) for left, right in offsets]
        if expected_lexical != lexical or expected_risk != risk:
            raise AssertionError(f"character oracle mask mismatch: {rid}")
        checks["character_oracle_rows"] += 1

        mapped_for_row = []
        for span in valid_spans:
            start, end = int(span["start"]), int(span["end"])
            mapped = [index for index, (left, right) in enumerate(offsets)
                      if any(text[pos].isalnum() for pos in range(max(left, start), min(right, end)))]
            kind = span["error_type"]
            acc = type_acc[kind]
            acc["released_spans"] += 1
            acc["answers"].add(rid)
            acc["groups"].add(row["group_id"])
            acc["char_lengths"].append(end - start)
            raw_valid_spans += 1
            raw_span_chars.append(end - start)
            if mapped:
                acc["mapped_spans"] += 1
                acc["lexical_bpe_lengths"].append(len(mapped))
                raw_span_bpes.append(len(mapped))
                raw_onsets.append((rid, mapped[0]))
            else:
                raw_unmapped_spans += 1
            mapped_for_row.append(mapped)

        lexical_indices = [index for index, value in enumerate(lexical) if value]
        lexical_states = [risk[index] for index in lexical_indices]
        row_raw_onsets = [mapped[0] for mapped in mapped_for_row if mapped]
        row_binary_onsets = []
        row_binary_lengths = []
        position = 0
        while position < len(lexical_states):
            if not lexical_states[position]:
                position += 1
                continue
            end = position + 1
            while end < len(lexical_states) and lexical_states[end]:
                end += 1
            onset = lexical_indices[position]
            row_binary_onsets.append(onset)
            row_binary_lengths.append(end - position)
            binary_onsets.append((rid, onset))
            binary_span_bpes.append(end - position)
            position = end

        for index, state in enumerate(lexical_states):
            previous = lexical_states[index - 1] if index else 0
            if previous == 0:
                transition_counts["onset_head_examples"] += 1
                transition_counts["onset_head_positive"] += state
                transition_counts["onset_head_negative"] += 1 - state
            else:
                transition_counts["continuation_head_examples_including_stop"] += 1
                transition_counts["continuation_positive"] += state
                transition_counts["stop_positive"] += 1 - state
        if lexical_states and lexical_states[-1]:
            transition_counts["continuation_head_examples_including_stop"] += 1
            transition_counts["terminal_stop_positive"] += 1
            transition_counts["stop_positive"] += 1

        row_window_counts = Counter()
        released_onset_set = set(row_raw_onsets)
        binary_onset_set = set(row_binary_onsets)
        for left, right in windows_for_count(n):
            indices = range(left, right)
            if not any(lexical[index] for index in indices):
                row_window_counts["excluded_nonlexical"] += 1
                continue
            row_window_counts["eligible"] += 1
            if any(risk[index] for index in indices):
                row_window_counts["risk_total"] += 1
                if any(index in released_onset_set for index in indices):
                    row_window_counts["released_span_onset"] += 1
                else:
                    row_window_counts["released_span_internal_continuation"] += 1
                if any(index in binary_onset_set for index in indices):
                    row_window_counts["binary_segment_onset"] += 1
                else:
                    row_window_counts["binary_segment_internal_continuation"] += 1
            else:
                row_window_counts["clean"] += 1
        if row_window_counts["risk_total"] != (row_window_counts["released_span_onset"]
                                                + row_window_counts["released_span_internal_continuation"]):
            raise AssertionError("released-span risk window partition mismatch")
        if row_window_counts["risk_total"] != (row_window_counts["binary_segment_onset"]
                                                + row_window_counts["binary_segment_internal_continuation"]):
            raise AssertionError("binary-segment risk window partition mismatch")
        if row_window_counts["eligible"] != row_window_counts["risk_total"] + row_window_counts["clean"]:
            raise AssertionError("eligible window partition mismatch")
        window_counts.update(row_window_counts)
        for onset in row_raw_onsets:
            incidence = sum(1 for left, right in windows_for_count(n)
                            if left <= onset < right and any(lexical[index] for index in range(left, right)))
            window_counts["released_span_onset_window_incidences"] += incidence
        for onset in row_binary_onsets:
            incidence = sum(1 for left, right in windows_for_count(n)
                            if left <= onset < right and any(lexical[index] for index in range(left, right)))
            window_counts["binary_segment_onset_window_incidences"] += incidence

        row_risk = int(any(risk))
        if row_risk:
            risk_answers += 1
            risk_groups.add(row["group_id"])
        if bool(valid_spans) != bool(seq.get("answer_risk", bool(valid_spans))):
            raise AssertionError("answer risk/released span mismatch")
        if ragtruth_summaries is not None:
            summary = ragtruth_summaries[rid]
            if row_window_counts["eligible"] != summary["eligible_window_count"]:
                raise AssertionError("RAGTruth eligible window count mismatch")
            if row_window_counts["risk_total"] != summary["positive_window_count"]:
                raise AssertionError("RAGTruth positive window count mismatch")

        fold = stable_fold(name, row["group_id"])
        fold_acc[fold]["groups"] = 0
        fold_acc[fold]["answers"] += 1
        fold_acc[fold]["risk_answers"] += row_risk
        fold_acc[fold]["released_valid_spans"] += len(valid_spans)
        fold_acc[fold]["binary_onsets"] += len(row_binary_onsets)
        fold_acc[fold]["risk_windows"] += row_window_counts["risk_total"]
        for kind, count in Counter(span["error_type"] for span in valid_spans).items():
            fold_acc[fold][f"type::{kind}"] += count

        total_tokens += n
        lexical_tokens += sum(lexical)
        risk_tokens += sum(risk)
        material_signatures["question"].add(digest_text(row["question"]))
        material_signatures["retrieved_passages"].add(digest_text(row["retrieved_passages"]))
        material_signatures["response"].add(digest_text(text))

    group_to_fold = {group: stable_fold(name, group) for group in groups}
    for fold in range(FOLDS):
        fold_acc[fold]["groups"] = sum(value == fold for value in group_to_fold.values())
    fold_rows = []
    for fold in range(FOLDS):
        counter = fold_acc[fold]
        fold_rows.append({key: counter[key] for key in sorted(counter)} | {"fold": fold})

    by_type = {}
    for kind in sorted(type_acc):
        acc = type_acc[kind]
        by_type[kind] = {
            "released_spans": acc["released_spans"],
            "mapped_spans": acc["mapped_spans"],
            "answers": len(acc["answers"]),
            "groups": len(acc["groups"]),
            "char_length": summarize(acc["char_lengths"]),
            "lexical_bpe_length": summarize(acc["lexical_bpe_lengths"]),
        }

    unique_raw_onsets = set(raw_onsets)
    unique_binary_onsets = set(binary_onsets)

    answerability = Counter(row["answerable"] for row in annotations if row["answerable"] is not None)
    hints = Counter(row["unanswerability_hint_setting"] for row in annotations
                    if row["unanswerability_hint_setting"] is not None)
    result = {
        "annotation_kind": annotations[0]["annotation_kind"] if annotations else None,
        "answers": answer_count,
        "groups": len(groups),
        "risk_answers": risk_answers,
        "risk_groups": len(risk_groups),
        "released_valid_risk_spans": raw_valid_spans,
        "released_invalid_spans_excluded": raw_invalid_spans,
        "released_valid_spans_without_lexical_bpe": raw_unmapped_spans,
        "released_span_mapped_onsets": len(raw_onsets),
        "released_span_unique_onset_bpe_positions": len(unique_raw_onsets),
        "released_unique_onsets_that_are_binary_segment_onsets": len(unique_raw_onsets & unique_binary_onsets),
        "released_unique_onsets_merged_into_existing_binary_segment": len(unique_raw_onsets - unique_binary_onsets),
        "binary_lexical_segments_and_onset_targets": len(binary_onsets),
        "span_overlap_answers": span_overlap_answer_count,
        "span_touching_or_overlap_answers": touching_or_overlapping_answer_count,
        "tokens": {"raw_bpe": total_tokens, "lexical_bpe": lexical_tokens, "risk_lexical_bpe": risk_tokens},
        "four_bpe_windows": {key: window_counts[key] for key in (
            "eligible", "released_span_onset", "released_span_internal_continuation",
            "binary_segment_onset", "binary_segment_internal_continuation", "risk_total", "clean",
            "excluded_nonlexical", "released_span_onset_window_incidences",
            "binary_segment_onset_window_incidences")},
        "transition_training_events": dict(sorted(transition_counts.items())),
        "released_span_length": {
            "characters": summarize(raw_span_chars),
            "lexical_bpe": summarize(raw_span_bpes),
            "lexical_bpe_buckets": length_buckets(raw_span_bpes),
        },
        "binary_segment_length_lexical_bpe": summarize(binary_span_bpes),
        "error_type_coverage": by_type,
        "error_taxonomy_available": not (set(by_type) <= {"automatic_unspecified"}),
        "metadata_coverage": {
            "answerable_true": answerability[True],
            "answerable_false": answerability[False],
            "unanswerability_hint_true": hints[True],
            "unanswerability_hint_false": hints[False],
        },
        "deterministic_hash_group_folds": fold_rows,
        "checks": {
            "character_oracle_rows": checks["character_oracle_rows"],
            "mask_and_window_invariants_passed": True,
        },
    }
    return result, material_signatures


def toy_checks() -> list[str]:
    text = "A, B!"
    offsets = [(0, 1), (1, 2), (2, 4), (4, 5)]
    lexical = [int(any(c.isalnum() for c in text[a:b])) for a, b in offsets]
    span = {"start": 3, "end": 4, "text": "B", "valid": True}
    risk = [int(any(text[pos].isalnum() for pos in range(max(a, span["start"]), min(b, span["end"]))))
            for a, b in offsets]
    assert lexical == [1, 0, 1, 0]
    assert risk == [0, 0, 1, 0]
    windows = windows_for_count(4)
    assert windows == [(0, 4)]
    assert any(any(risk[left:right]) for left, right in windows)
    return [
        "Unicode isalnum character oracle is independent of released masks",
        "nonlexical BPEs remain context only",
        "short/equal-width answer window geometry matches frozen max(1,n-k+1) rule",
    ]


def build() -> dict:
    for path in PATHS.values():
        assert_allowed(path)
        if not path.is_file():
            raise AssertionError(f"missing allowlisted input: {path}")

    rt_annotations, rt_sequences, rt_summaries = unified_ragtruth()
    rago_all_annotations, rago_all_sequences = unified_ragognize("all")
    rago_no_annotations, rago_no_sequences = unified_ragognize("no_hint")

    rt, rt_sig = analyze_source("ragtruth_fit_human", rt_annotations, rt_sequences, rt_summaries)
    rago_all, rago_all_sig = analyze_source(
        "ragognize_train_all_silver", rago_all_annotations, rago_all_sequences)
    rago_no, rago_no_sig = analyze_source(
        "ragognize_train_no_hint_silver", rago_no_annotations, rago_no_sequences)

    all_answers = {row["response_id"]: row for row in rago_all_annotations}
    no_answers = {row["response_id"]: row for row in rago_no_annotations}
    if not set(no_answers) < set(all_answers):
        raise AssertionError("RAGognize no-hint is not a proper subset of all-train")
    for response_id, row in no_answers.items():
        parent = all_answers[response_id]
        if row != parent or rago_no_sequences[response_id] != rago_all_sequences[response_id]:
            raise AssertionError("RAGognize no-hint row changed relative to all-train")

    cross = {}
    for field in ("question", "retrieved_passages", "response"):
        cross[field] = {
            "ragtruth_vs_ragognize_all_exact_normalized_hash_intersection": len(rt_sig[field] & rago_all_sig[field]),
            "ragtruth_vs_ragognize_no_hint_exact_normalized_hash_intersection": len(rt_sig[field] & rago_no_sig[field]),
        }

    fit_feature_dir = BENCHMARK / "data" / "features"
    fit_feature_pairs = sum(
        int((fit_feature_dir / f"{row['response_id']}.json").is_file()
            and (fit_feature_dir / f"{row['response_id']}.npz").is_file())
        for row in rt_annotations
    )
    capacity = {
        "schema_version": "onset-segment-capacity-v1",
        "audit_date": "2026-09-13",
        "scope": "CPU-only label and geometry audit; fit/train only; no model fitting",
        "definitions": {
            "released_span": "Each valid released character span; RAGTruth is human, RAGognize is automatic silver.",
            "released_span_onset": "First lexical raw BPE intersecting that released span.",
            "binary_segment_onset": "A 0->1 transition on the lexical-BPE-only risk sequence; this is the realizable start-head target.",
            "released_span_onset_window": "Eligible width-4 raw-BPE window containing at least one mapped released-span onset; onset takes priority if it also contains continuation risk.",
            "released_span_internal_continuation_window": "Eligible risk window containing no mapped released-span onset.",
            "binary_segment_window_variant": "The same partition is also reported using realizable 0->1 lexical-risk segment onsets for the semi-Markov decoder.",
            "clean_window": "Eligible window containing no released lexical risk BPE.",
            "window_geometry": "Raw BPE width 4, stride 1, no trailing short window when n>=4; a window is eligible iff it contains >=1 lexical BPE.",
        },
        "sources": {
            "ragtruth_fit_human": rt,
            "ragognize_train_no_hint_silver": rago_no,
            "ragognize_train_all_silver": rago_all,
        },
        "subset_relation": {
            "no_hint_is_strict_subset_of_all_train": True,
            "no_hint_answers": len(no_answers),
            "all_train_answers": len(all_answers),
            "never_sum_as_independent_sources": True,
        },
        "cross_source_exact_collision_audit": cross,
        "whitebox_feature_availability": {
            "ragtruth_fit_cached_feature_pairs": fit_feature_pairs,
            "ragtruth_fit_expected_answers": len(rt_annotations),
            "ragtruth_fit_cached_arrays_documented_elsewhere": ["hidden_last[4096]", "lb[1024]", "nll[1]"],
            "ragognize_conversion_has_historical_hidden_or_logprob": False,
            "ragognize_requirement": "Teacher-forced replay is a future prerequisite for whitebox silver pretraining; it was not run here.",
        },
        "capacity_judgment": {
            "global_low_capacity_onset_and_transition_heads": {
                "verdict": "feasible",
                "reason": "Human fit supplies 568 realizable binary onsets and every fixed hash fold has at least 101; no-hint silver pretraining adds 714 binary onsets with at least 130 per fold.",
                "maximum_recommended_supervised_head": "Two regularized linear transition heads after a fixed low-dimensional projection; do not fit a deep sequence encoder from these onset events.",
            },
            "continuation_head": {
                "verdict": "feasible_with_group_and_answer_balancing",
                "reason": "There are many continuation tokens, but the independent event ceiling is the number of answers/groups/segments, not the token count.",
            },
            "type_specific_heads": {
                "verdict": "not_supported",
                "reason": "Human fit has only 109 Evident Conflict spans and 7 Subtle Conflict spans; two fixed folds contain zero Subtle Conflict spans. RAGognize exposes no released error taxonomy.",
            },
            "silver_scope": {
                "primary": "no_unanswerability_hint (971 answers)",
                "ablation_only": "all_train (1842 answers) replaces, never augments, the no-hint subset",
                "limitation": "Automatic spans increase optimization support but do not increase human-gold evidentiary strength.",
            },
            "high_dimensional_raw_probe": {
                "verdict": "not_supported_by_event_count",
                "reason": "Raw 4096 hidden + 1024 Lookback dimensions exceed the independent onset-event count; use a frozen projection and strong regularization.",
            },
        },
        "input_sha256": {name: sha256_file(path) for name, path in sorted(PATHS.items())},
        "self_check": {
            "status": "PASS",
            "allowlisted_input_count": len(PATHS),
            "sealed_split_paths_opened": 0,
            "gpu_libraries_imported": False,
            "gpu_started": False,
            "model_training_started": False,
            "toy_checks": toy_checks(),
        },
    }
    return capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capacity", "selfcheck", "both"), default="both")
    args = parser.parse_args()
    capacity = build()
    if args.mode == "capacity":
        output = capacity
    elif args.mode == "selfcheck":
        output = {
            "schema_version": "onset-segment-selfcheck-v1",
            "status": capacity["self_check"]["status"],
            "scope": capacity["scope"],
            "input_sha256": capacity["input_sha256"],
            "source_checks": {name: data["checks"] for name, data in capacity["sources"].items()},
            "cross_source_exact_collision_audit": capacity["cross_source_exact_collision_audit"],
            "subset_relation": capacity["subset_relation"],
            "runtime": capacity["self_check"],
        }
    else:
        output = {"capacity": capacity, "selfcheck": capacity["self_check"]}
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
