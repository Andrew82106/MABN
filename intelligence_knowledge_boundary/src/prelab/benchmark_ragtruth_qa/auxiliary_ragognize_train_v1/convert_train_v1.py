"""Convert pinned RAGognize TRAIN/Llama-2 rows to the local unified K=4 schema.

Commands:
  design  Freeze source/code/protocol hashes before conversion.
  run     CPU-only conversion into conversion_v1/.
  check   Re-open every output and independently verify counts/invariants.

There is deliberately no network, model-forward, training, or GPU entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train-00000-of-00001.parquet"
GROUP_INDEX = ROOT / "GROUP_INDEX.jsonl"
ALL_INDEX = ROOT / "ALL_TRAIN_INDEX.jsonl"
NO_HINT_INDEX = ROOT / "NO_UNANSWERABILITY_HINT_INDEX.jsonl"
PROTOCOL = ROOT / "CONVERSION_PROTOCOL.json"
DESIGN = ROOT / "CONVERSION_DESIGN_FREEZE.json"
OUT = ROOT / "conversion_v1"
TOKENIZER_DIR = ROOT.parents[1] / "models" / "Llama-2-7b-chat-hf"
MODEL_KEY = "Llama-2-7b-chat-hf"
REVISION = "aab54518c2a7c0d25fff8bffbf5337d0321de142"
TRAIN_SHA256 = "6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44"
WINDOW = 4
EXPECTED = {"all_train": 1842, "no_unanswerability_hint": 971}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_paths() -> list[Path]:
    paths = [
        TRAIN,
        GROUP_INDEX,
        ALL_INDEX,
        NO_HINT_INDEX,
        ROOT / "AUDIT.json",
        ROOT / "SCHEMA.json",
        ROOT / "MANIFEST.json",
        PROTOCOL,
        ROOT / "CONVERSION_SCHEMA.md",
        Path(__file__),
    ]
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ):
        path = TOKENIZER_DIR / name
        if path.exists():
            paths.append(path)
    return paths


def source_hashes() -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in source_paths()}


def design() -> None:
    if DESIGN.exists():
        raise RuntimeError(f"Design already exists: {DESIGN}")
    if OUT.exists():
        raise RuntimeError(f"Output already exists before design: {OUT}")
    if file_sha256(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("Pinned train hash mismatch")
    forbidden = [
        path for pattern in ("*test*.parquet", "*validation*.parquet")
        for path in ROOT.rglob(pattern)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden split file present: {forbidden}")
    save_json(DESIGN, {
        "version": "ragognize-llama2-unified-k4-train-only-v1",
        "source_revision": REVISION,
        "source_hashes": source_hashes(),
        "test_or_validation_content_read": False,
        "test_or_validation_content_downloaded": False,
        "selection_used_labels_or_model_performance": False,
        "model_forward": False,
        "training": False,
        "GPU_used": False,
    })
    print("RAGOGNIZE_CONVERSION_DESIGN_FROZEN")


def check_design() -> dict:
    frozen = read_json(DESIGN)
    if frozen["source_revision"] != REVISION:
        raise RuntimeError("Revision changed")
    current = source_hashes()
    if frozen["source_hashes"] != current:
        changed = sorted(set(frozen["source_hashes"]) | set(current))
        changed = [key for key in changed if frozen["source_hashes"].get(key) != current.get(key)]
        raise RuntimeError(f"Frozen source changed: {changed}")
    if file_sha256(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("Train hash changed")
    forbidden = [
        path for pattern in ("*test*.parquet", "*validation*.parquet")
        for path in ROOT.rglob(pattern)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden split file present: {forbidden}")
    return frozen


def tokenizer_signature(tokenizer) -> dict:
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    )
    return {
        "path": str(TOKENIZER_DIR.resolve()),
        "class": type(tokenizer).__name__,
        "is_fast": tokenizer.is_fast,
        "vocab_size": tokenizer.vocab_size,
        "length_with_added_tokens": len(tokenizer),
        "files_sha256": {
            name: file_sha256(TOKENIZER_DIR / name)
            for name in names if (TOKENIZER_DIR / name).exists()
        },
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def exact_released_spans(response: dict) -> list[dict]:
    spans = [dict(span) for span in (response.get("hallucinations") or [])]
    text = response["text"]
    for span in spans:
        if set(span) != {"end", "start", "text", "valid"}:
            raise AssertionError(f"Unexpected released span fields: {set(span)}")
        if not isinstance(span["valid"], bool):
            raise AssertionError("Span valid must be bool")
        start, end = int(span["start"]), int(span["end"])
        if not (0 <= start < end <= len(text)):
            raise AssertionError("Released span out of bounds")
        if text[start:end] != span["text"]:
            raise AssertionError("Released span text/coordinate mismatch")
    return spans


def encode(tokenizer, full_prompt: str, response_text: str, valid_spans: list[dict]) -> dict:
    rendered = full_prompt + " " + response_text
    answer_begin = len(full_prompt) + 1
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
        padding=False,
        truncation=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded["attention_mask"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    answer_positions = [
        index for index, (left, right) in enumerate(offsets)
        if right > answer_begin and left < len(rendered)
    ]
    clipped, raw = [], []
    for position in answer_positions:
        left, right = offsets[position]
        raw.append([left - answer_begin, right - answer_begin])
        clipped.append([
            max(left, answer_begin) - answer_begin,
            min(right, len(rendered)) - answer_begin,
        ])
    coverage = np.zeros(len(response_text), dtype=bool)
    for left, right in clipped:
        coverage[left:right] = True
    nonspace = np.fromiter((not char.isspace() for char in response_text), dtype=bool)
    if np.any(nonspace & ~coverage):
        raise AssertionError("Tokenizer offsets lost a non-whitespace answer character")

    lexical_mask, risk_mask = [], []
    for left, right in clipped:
        lexical_mask.append(int(any(char.isalnum() for char in response_text[left:right])))
        risk_mask.append(int(any(
            span["valid"]
            and left < int(span["end"])
            and right > int(span["start"])
            and any(
                char.isalnum()
                for char in response_text[
                    max(left, int(span["start"])):min(right, int(span["end"]))
                ]
            )
            for span in valid_spans
        )))
    return {
        "rendered": rendered,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_positions": answer_positions,
        "response_ids": [input_ids[position] for position in answer_positions],
        "clipped_offsets": clipped,
        "raw_offsets": raw,
        "lexical_mask": lexical_mask,
        "risk_mask": risk_mask,
    }


class CohortWriter:
    def __init__(self, name: str, members: set[int]):
        self.name = name
        self.members = members
        self.path = OUT / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.handles = {
            stem: (self.path / f"{stem}.jsonl").open("w", encoding="utf-8", newline="\n")
            for stem in ("answers", "sequences", "tokens", "windows")
        }
        self.counts = Counter()
        self.group_members: defaultdict[str, list[dict]] = defaultdict(list)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def write(self, answer: dict, sequence: dict, tokens: list[dict], windows: list[dict]) -> None:
        self.handles["answers"].write(json.dumps(answer, ensure_ascii=False) + "\n")
        self.handles["sequences"].write(json.dumps(sequence, ensure_ascii=False) + "\n")
        for token in tokens:
            self.handles["tokens"].write(json.dumps(token, ensure_ascii=False) + "\n")
        for window in windows:
            self.handles["windows"].write(json.dumps(window, ensure_ascii=False) + "\n")
        self.counts["answers"] += 1
        self.counts["risk_answers"] += int(answer["answer_risk"])
        self.counts["valid_spans"] += sum(span["valid"] for span in answer["released_hallucinations"])
        self.counts["tokens"] += len(tokens)
        self.counts["lexical_tokens"] += sum(token["lexical"] for token in tokens)
        self.counts["risk_tokens"] += sum(token["risk"] for token in tokens)
        self.counts["windows"] += len(windows)
        self.counts["lexical_windows"] += sum(window["eligible_lexical"] for window in windows)
        self.counts["risk_windows"] += sum(window["risk"] for window in windows if window["eligible_lexical"])
        self.group_members[answer["group_id"]].append({
            "response_id": answer["response_id"],
            "train_row_index": answer["train_row_index"],
            "official_user_prompt_index": answer["official_user_prompt_index"],
            "source_id": answer["source_id"],
        })

    def finish(self, tokenizer_info: dict, design_hash: str) -> dict:
        groups_path = self.path / "groups.jsonl"
        with groups_path.open("w", encoding="utf-8", newline="\n") as handle:
            for group_id in sorted(self.group_members):
                members = sorted(self.group_members[group_id], key=lambda row: row["train_row_index"])
                handle.write(json.dumps({
                    "group_id": group_id,
                    "response_ids": [row["response_id"] for row in members],
                    "train_row_indices": [row["train_row_index"] for row in members],
                    "official_user_prompt_indices": sorted({row["official_user_prompt_index"] for row in members}),
                    "source_ids": sorted({row["source_id"] for row in members}),
                    "member_count": len(members),
                }, ensure_ascii=False) + "\n")
        output_names = ("answers.jsonl", "sequences.jsonl", "tokens.jsonl", "windows.jsonl", "groups.jsonl")
        files = {
            name: {
                "bytes": (self.path / name).stat().st_size,
                "sha256": file_sha256(self.path / name),
            }
            for name in output_names
        }
        counts = dict(self.counts)
        counts["groups"] = len(self.group_members)
        counts["risk_answer_rate"] = self.counts["risk_answers"] / self.counts["answers"]
        counts["risk_token_rate_among_lexical"] = self.counts["risk_tokens"] / self.counts["lexical_tokens"]
        counts["risk_window_rate_among_lexical"] = self.counts["risk_windows"] / self.counts["lexical_windows"]
        manifest = {
            "version": "ragognize-llama2-unified-k4-train-only-v1",
            "cohort": self.name,
            "official_split": "train",
            "source_revision": REVISION,
            "model_subset": MODEL_KEY,
            "membership_index": "ALL_TRAIN_INDEX.jsonl" if self.name == "all_train" else "NO_UNANSWERABILITY_HINT_INDEX.jsonl",
            "membership_is_label_blind": True,
            "group_split_required": True,
            "counts": counts,
            "tokenizer": tokenizer_info,
            "conversion_design_sha256": design_hash,
            "automatic_not_human_gold": True,
            "unmarked_positions_silver_not_verified_negative": True,
            "historical_native_generation_trace": False,
            "test_or_validation_content_read": False,
            "model_forward": False,
            "training": False,
            "GPU_used": False,
            "files": files,
        }
        save_json(self.path / "manifest.json", manifest)
        return manifest


def run() -> None:
    frozen = check_design()
    if OUT.exists():
        raise RuntimeError(f"Conversion output already exists: {OUT}")
    OUT.mkdir(parents=True)

    group_rows = {row["train_row_index"]: row for row in iter_jsonl(GROUP_INDEX)}
    all_members = {row["train_row_index"] for row in iter_jsonl(ALL_INDEX)}
    no_hint_members = {row["train_row_index"] for row in iter_jsonl(NO_HINT_INDEX)}
    if all_members != set(range(EXPECTED["all_train"])):
        raise AssertionError("Frozen all-train membership mismatch")
    if not no_hint_members < all_members or len(no_hint_members) != EXPECTED["no_unanswerability_hint"]:
        raise AssertionError("Frozen no-hint membership mismatch")
    if set(group_rows) != all_members:
        raise AssertionError("Group index does not cover train exactly once")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required")
    tokenizer_info = tokenizer_signature(tokenizer)
    writers = {
        "all_train": CohortWriter("all_train", all_members),
        "no_unanswerability_hint": CohortWriter("no_unanswerability_hint", no_hint_members),
    }
    canonical_token_mismatch = 0
    overlap_rows = 0
    detector_inputs = set()

    try:
        rows = pq.read_table(TRAIN).to_pylist()
        if len(rows) != EXPECTED["all_train"]:
            raise AssertionError("Unexpected train row count")
        for row_index, row in enumerate(rows):
            group = group_rows[row_index]
            response = row["responses"][MODEL_KEY]
            details = response["details"]
            response_text = response["text"]
            if details["output"] != response_text:
                raise AssertionError("Response text/output mismatch")
            if text_sha256(response_text) != group["llama_response_sha256"]:
                raise AssertionError("Response hash differs from frozen group index")
            labels = exact_released_spans(response)
            valid_spans = [span for span in labels if span["valid"]]
            ordered = sorted((int(span["start"]), int(span["end"])) for span in valid_spans)
            overlaps = any(left[1] > right[0] for left, right in zip(ordered, ordered[1:]))
            overlap_rows += overlaps

            encoded = encode(tokenizer, details["full_prompt"], response_text, valid_spans)
            local_starts = [offset[0] for offset in encoded["clipped_offsets"]]
            official_starts = [int(value) for value in (details.get("token_starts") or [])]
            canonical_official = official_starts[1:]
            if canonical_official and canonical_official[-1] == len(response_text):
                canonical_official = canonical_official[:-1]
            token_starts_match = canonical_official == local_starts
            canonical_token_mismatch += not token_starts_match

            response_id = f"ragognize_train_{row_index}__llama2_7b_chat"
            source_id = "ragognize_source_" + group["source_identity_sha256"]
            detector_input_hash = value_sha256([
                row["user_prompt"], row["documents_str"], response_text
            ])
            if detector_input_hash in detector_inputs:
                raise AssertionError("Duplicate detector input")
            detector_inputs.add(detector_input_hash)
            answer = {
                "schema_version": "ragognize-unified-answer-v1",
                "response_id": response_id,
                "train_row_index": row_index,
                "official_user_prompt_index": int(row["user_prompt_index"]),
                "source_dataset": "F4biian/RAGognize",
                "source_revision": REVISION,
                "official_split": "train",
                "generator": MODEL_KEY,
                "source_id": source_id,
                "group_id": group["recommended_group_id"],
                "partition": "auxiliary_candidate_fit",
                "task_type": "QA_automatic_natural_RAG_hallucination",
                "question": row["user_prompt"],
                "retrieved_passages": row["documents_str"],
                "released_prompt": details["full_prompt"],
                "original_response": response_text,
                "released_hallucinations": labels,
                "answer_risk": int(bool(valid_spans)),
                "answerable": bool(row["answerable"]),
                "unanswerability_hint_setting": bool(group["unanswerability_hint_setting"]),
                "explicit_abstention_instruction_in_full_prompt": bool(group["explicit_abstention_instruction_in_full_prompt"]),
                "automatic_not_human_gold": True,
                "unmarked_positions_silver_not_verified_negative": True,
                "natural_model_output_not_injected_hallucination": True,
                "source_record_sha256": value_sha256(row),
                "question_sha256": group["question_sha256"],
                "retrieved_passages_sha256": group["retrieved_context_sha256"],
                "response_sha256": group["llama_response_sha256"],
                "released_labels_sha256": value_sha256(labels),
                "detector_input_sha256": detector_input_hash,
                "answer_token_count": len(encoded["response_ids"]),
                "window_count": len(encoded["response_ids"]) - WINDOW + 1,
                "released_span_overlap": bool(overlaps),
                "canonical_official_token_starts_match_local": token_starts_match,
            }
            if answer["window_count"] <= 0:
                raise AssertionError("A short-answer policy would be required")
            sequence = {
                "schema_version": "ragognize-unified-sequence-v1",
                "response_id": response_id,
                "source_id": source_id,
                "group_id": group["recommended_group_id"],
                "full_input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "answer_token_positions": encoded["answer_positions"],
                "response_token_ids": encoded["response_ids"],
                "response_token_offsets": encoded["clipped_offsets"],
                "response_token_offsets_raw": encoded["raw_offsets"],
                "lexical_mask": encoded["lexical_mask"],
                "risk_mask": encoded["risk_mask"],
                "answer_risk": answer["answer_risk"],
                "answer_character_range_in_rendered_input": [len(details["full_prompt"]) + 1, len(encoded["rendered"])],
                "rendered_input_sha256": text_sha256(encoded["rendered"]),
                "response_sha256": answer["response_sha256"],
                "historical_native_generation_trace": False,
            }
            tokens = []
            for token_index, (token_id, position, clipped, raw, lexical, risk) in enumerate(zip(
                encoded["response_ids"], encoded["answer_positions"],
                encoded["clipped_offsets"], encoded["raw_offsets"],
                encoded["lexical_mask"], encoded["risk_mask"], strict=True
            )):
                tokens.append({
                    "schema_version": "ragognize-unified-token-v1",
                    "response_id": response_id,
                    "source_id": source_id,
                    "group_id": group["recommended_group_id"],
                    "answer_token_index": token_index,
                    "full_token_position": position,
                    "token_id": token_id,
                    "char_start": clipped[0],
                    "char_end": clipped[1],
                    "raw_char_start": raw[0],
                    "raw_char_end": raw[1],
                    "lexical": lexical,
                    "risk": risk,
                })
            windows = []
            for window_index in range(len(tokens) - WINDOW + 1):
                selected = tokens[window_index:window_index + WINDOW]
                lexical_count = sum(token["lexical"] for token in selected)
                risk_count = sum(token["risk"] for token in selected)
                windows.append({
                    "schema_version": "ragognize-unified-window-k4-v1",
                    "response_id": response_id,
                    "source_id": source_id,
                    "group_id": group["recommended_group_id"],
                    "window_index": window_index,
                    "token_start": window_index,
                    "token_end": window_index + WINDOW,
                    "token_ids": [token["token_id"] for token in selected],
                    "char_start": min(token["char_start"] for token in selected),
                    "char_end": max(token["char_end"] for token in selected),
                    "lexical_token_count": lexical_count,
                    "risk_token_count": risk_count,
                    "eligible_lexical": int(lexical_count > 0),
                    "risk": int(risk_count > 0),
                    "answer_risk": answer["answer_risk"],
                })

            for name, writer in writers.items():
                if row_index in writer.members:
                    writer.write(answer, sequence, tokens, windows)
    finally:
        for writer in writers.values():
            writer.close()

    if canonical_token_mismatch != 29 or overlap_rows != 1:
        raise AssertionError("Frozen token-start/span-overlap audit changed")
    design_hash = file_sha256(DESIGN)
    manifests = {
        name: writer.finish(tokenizer_info, design_hash)
        for name, writer in writers.items()
    }
    audit = verify_outputs(manifests)
    save_json(OUT / "AUDIT.json", audit)
    files = {
        str(path.relative_to(OUT)): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path != OUT / "MANIFEST.json"
    }
    manifest = {
        "version": "ragognize-llama2-unified-k4-train-only-v1",
        "status": "complete_converted_and_audited_not_trained",
        "source_revision": REVISION,
        "source_train_sha256": TRAIN_SHA256,
        "conversion_design_sha256": design_hash,
        "cohorts": {name: value["counts"] for name, value in manifests.items()},
        "test_or_validation_content_read": False,
        "model_forward": False,
        "training": False,
        "GPU_used": False,
        "baseline_modified": False,
        "files": files,
    }
    save_json(OUT / "MANIFEST.json", manifest)
    save_json(OUT / "complete.json", {
        "status": "complete_converted_and_audited_not_trained",
        "manifest_sha256": file_sha256(OUT / "MANIFEST.json"),
        "audit_sha256": file_sha256(OUT / "AUDIT.json"),
        "test_or_validation_content_read": False,
        "training": False,
        "GPU_used": False,
    })
    print(json.dumps({
        "status": manifest["status"],
        "cohorts": manifest["cohorts"],
        "token_start_mismatch_rows_preserved": canonical_token_mismatch,
        "overlapping_span_rows_preserved": overlap_rows,
    }, ensure_ascii=False, indent=2))


def verify_outputs(expected_manifests: dict | None = None) -> dict:
    check_design()
    if not OUT.exists():
        raise RuntimeError("Conversion output missing")
    source_rows = pq.read_table(TRAIN).to_pylist()
    frozen_members = {
        "all_train": {row["train_row_index"] for row in iter_jsonl(ALL_INDEX)},
        "no_unanswerability_hint": {row["train_row_index"] for row in iter_jsonl(NO_HINT_INDEX)},
    }
    results = {}
    for cohort, expected_rows in EXPECTED.items():
        path = OUT / cohort
        answers = list(iter_jsonl(path / "answers.jsonl"))
        sequences = list(iter_jsonl(path / "sequences.jsonl"))
        groups = list(iter_jsonl(path / "groups.jsonl"))
        if len(answers) != expected_rows or len(sequences) != expected_rows:
            raise AssertionError(f"{cohort}: answer/sequence count mismatch")
        if {row["train_row_index"] for row in answers} != frozen_members[cohort]:
            raise AssertionError(f"{cohort}: membership drift")
        if [row["response_id"] for row in answers] != [row["response_id"] for row in sequences]:
            raise AssertionError(f"{cohort}: answer/sequence ordering mismatch")
        answer_by_id = {row["response_id"]: row for row in answers}
        if len(answer_by_id) != len(answers):
            raise AssertionError(f"{cohort}: duplicate response IDs")
        for answer in answers:
            original = source_rows[answer["train_row_index"]]
            response = original["responses"][MODEL_KEY]
            if answer["question"] != original["user_prompt"]:
                raise AssertionError(f"{cohort}: question changed")
            if answer["retrieved_passages"] != original["documents_str"]:
                raise AssertionError(f"{cohort}: context changed")
            if answer["released_prompt"] != response["details"]["full_prompt"]:
                raise AssertionError(f"{cohort}: prompt changed")
            if answer["original_response"] != response["text"]:
                raise AssertionError(f"{cohort}: response changed")
            if answer["released_hallucinations"] != [dict(span) for span in (response["hallucinations"] or [])]:
                raise AssertionError(f"{cohort}: released labels changed")

        token_counts = Counter()
        lexical_tokens = risk_tokens = 0
        last_token_index = defaultdict(lambda: -1)
        for token in iter_jsonl(path / "tokens.jsonl"):
            rid = token["response_id"]
            if rid not in answer_by_id or token["group_id"] != answer_by_id[rid]["group_id"]:
                raise AssertionError(f"{cohort}: orphan/cross-group token")
            if token["answer_token_index"] != last_token_index[rid] + 1:
                raise AssertionError(f"{cohort}: token indices not contiguous")
            last_token_index[rid] += 1
            token_counts[rid] += 1
            lexical_tokens += token["lexical"]
            risk_tokens += token["risk"]

        window_counts = Counter()
        lexical_windows = risk_windows = 0
        last_window_index = defaultdict(lambda: -1)
        for window in iter_jsonl(path / "windows.jsonl"):
            rid = window["response_id"]
            if rid not in answer_by_id or window["group_id"] != answer_by_id[rid]["group_id"]:
                raise AssertionError(f"{cohort}: orphan/cross-group window")
            if window["window_index"] != last_window_index[rid] + 1:
                raise AssertionError(f"{cohort}: window indices not contiguous")
            if window["token_end"] - window["token_start"] != WINDOW or len(window["token_ids"]) != WINDOW:
                raise AssertionError(f"{cohort}: non-K4 window")
            if window["risk"] != int(window["risk_token_count"] > 0):
                raise AssertionError(f"{cohort}: window risk is not token OR")
            last_window_index[rid] += 1
            window_counts[rid] += 1
            lexical_windows += window["eligible_lexical"]
            risk_windows += window["risk"] if window["eligible_lexical"] else 0
        for answer, sequence in zip(answers, sequences, strict=True):
            rid = answer["response_id"]
            n = answer["answer_token_count"]
            if token_counts[rid] != n or len(sequence["response_token_ids"]) != n:
                raise AssertionError(f"{cohort}: token count mismatch")
            if window_counts[rid] != n - WINDOW + 1 or answer["window_count"] != n - WINDOW + 1:
                raise AssertionError(f"{cohort}: window count mismatch")
            if sequence["risk_mask"] and int(any(sequence["risk_mask"])) != answer["answer_risk"]:
                raise AssertionError(f"{cohort}: answer risk differs from token risk")

        group_answer_ids = [rid for group in groups for rid in group["response_ids"]]
        if len(group_answer_ids) != len(set(group_answer_ids)) or set(group_answer_ids) != set(answer_by_id):
            raise AssertionError(f"{cohort}: group membership is not a partition")
        if any(len({answer_by_id[rid]["group_id"] for rid in group["response_ids"]}) != 1 for group in groups):
            raise AssertionError(f"{cohort}: group contamination")
        result = {
            "answers": len(answers),
            "groups": len(groups),
            "tokens": sum(token_counts.values()),
            "lexical_tokens": lexical_tokens,
            "risk_tokens": risk_tokens,
            "windows": sum(window_counts.values()),
            "lexical_windows": lexical_windows,
            "risk_windows": risk_windows,
            "exact_source_input_and_label_replay": True,
            "unique_detector_inputs": len({row["detector_input_sha256"] for row in answers}) == len(answers),
            "token_indices_contiguous": True,
            "windows_exact_k4_stride1": True,
            "group_membership_exact_partition": True,
        }
        if expected_manifests is not None:
            expected = expected_manifests[cohort]["counts"]
            for key in ("answers", "groups", "tokens", "lexical_tokens", "risk_tokens", "windows", "lexical_windows", "risk_windows"):
                if result[key] != expected[key]:
                    raise AssertionError(f"{cohort}: manifest mismatch for {key}")
        results[cohort] = result
    return {
        "status": "passed",
        "cohorts": results,
        "cohort_memberships_match_frozen_indices": True,
        "released_inputs_and_labels_unchanged": True,
        "question_source_groups_isolated": True,
        "test_or_validation_content_read": False,
        "model_forward": False,
        "training": False,
        "GPU_used": False,
    }


def check() -> None:
    audit = verify_outputs()
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("design", "run", "check"))
    args = parser.parse_args()
    {"design": design, "run": run, "check": check}[args.command]()
