"""CPU-only audit of the pinned RAGognize TRAIN shard.

This script has no network code and refuses to run if a local test/validation
parquet is present. It writes aggregate quality/schema reports and a hash-only
group index. It never trains or loads model weights.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train-00000-of-00001.parquet"
TOKENIZER_DIR = ROOT.parents[1] / "models" / "Llama-2-7b-chat-hf"
MODEL_KEY = "Llama-2-7b-chat-hf"
EXPECTED_ROWS = 1842
WINDOW = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def describe(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "sum": int(array.sum()),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def canonical_spans(spans) -> list[tuple[int, int, str, bool]]:
    return sorted(
        (int(s["start"]), int(s["end"]), str(s["text"]), bool(s["valid"]))
        for s in (spans or [])
    )


def source_identity(row: dict) -> tuple:
    prompt_meta = row["details"]["user_prompt"]
    details = prompt_meta["details"]
    article = details["suitable_article"]
    return (
        details.get("suitable_article_index"),
        article.get("revision_id"),
        article.get("url"),
        article.get("title"),
    )


def main() -> None:
    forbidden = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*test*.parquet", "*validation*.parquet")
        for path in ROOT.rglob(pattern)
    )
    if forbidden:
        raise RuntimeError(f"Refusing audit because forbidden local split files exist: {forbidden}")
    if file_sha256(TRAIN) != "6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44":
        raise RuntimeError("Pinned train shard SHA-256 mismatch")

    parquet = pq.ParquetFile(TRAIN)
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} train rows, got {parquet.metadata.num_rows}")
    table = parquet.read()
    rows = table.to_pylist()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer with offset mapping is required")

    issues: Counter[str] = Counter()
    issue_rows: defaultdict[str, list[int]] = defaultdict(list)
    answerable = Counter()
    hallucinated_by_answerable = Counter()
    group_names = Counter()
    info_types = Counter()
    categories = Counter()
    spans_per_answer: list[int] = []
    span_lengths: list[int] = []
    answer_lengths: list[int] = []
    document_counts: list[int] = []
    full_token_counts: list[int] = []
    answer_token_counts: list[int] = []
    lexical_token_counts: list[int] = []
    risk_token_counts: list[int] = []
    window_counts: list[int] = []
    lexical_window_counts: list[int] = []
    risk_window_counts: list[int] = []
    document_containment = Counter()
    full_chat_variants = Counter()
    hint_settings = Counter()
    explicit_abstention_by_setting = Counter()
    subset_risk_counts: defaultdict[str, Counter] = defaultdict(Counter)
    official_token_starts_raw = Counter()
    official_token_starts_canonical_exact = 0
    official_token_starts_canonical_length_equal = 0
    official_token_count_deltas = Counter()
    span_token_boundaries = Counter()
    total_valid_spans = 0
    punctuation_only_spans = 0
    mapped_lexical_spans = 0

    qidxs, qnorms, source_keys, context_hashes, response_hashes = [], [], [], [], []
    group_rows = []

    def flag(name: str, row_index: int) -> None:
        issues[name] += 1
        if len(issue_rows[name]) < 100:
            issue_rows[name].append(row_index)

    for row_index, row in enumerate(rows):
        question = row.get("user_prompt")
        documents = row.get("documents")
        documents_str = row.get("documents_str")
        rag_prompt = row.get("rag_prompt")
        response = (row.get("responses") or {}).get(MODEL_KEY)

        if not isinstance(question, str) or not question.strip():
            flag("missing_question", row_index)
            question = question or ""
        if not isinstance(documents, list) or not documents:
            flag("missing_documents", row_index)
            documents = documents or []
        if not isinstance(documents_str, str) or not documents_str.strip():
            flag("missing_documents_str", row_index)
            documents_str = documents_str or ""
        if not isinstance(rag_prompt, list) or not rag_prompt:
            flag("missing_rag_prompt", row_index)
        if response is None:
            flag("missing_llama_response", row_index)
            continue

        for document in documents:
            document_containment["documents_total"] += 1
            if not isinstance(document.get("text"), str) or not document["text"].strip():
                flag("document_missing_text", row_index)
            if not isinstance(document.get("title"), str) or not document["title"].strip():
                flag("document_missing_title", row_index)
            document_text = document.get("text") or ""
            if document_text in documents_str:
                document_containment["exact"] += 1
            elif document_text.strip() and document_text.strip() in documents_str:
                document_containment["after_outer_whitespace_strip"] += 1
            else:
                document_containment["not_found_even_after_outer_whitespace_strip"] += 1
                flag("documents_str_missing_document_text_after_strip", row_index)

        text = response.get("text")
        details = response.get("details") or {}
        if not isinstance(text, str) or not text:
            flag("missing_response_text", row_index)
            continue
        if details.get("output") != text:
            flag("response_text_output_mismatch", row_index)
        if details.get("answerable") != row.get("answerable"):
            flag("answerable_mismatch", row_index)
        full_prompt = details.get("full_prompt")
        full_chat = details.get("full_chat")
        if not isinstance(full_prompt, str) or not full_prompt:
            flag("missing_full_prompt", row_index)
            continue
        if question not in full_prompt:
            flag("question_absent_from_full_prompt", row_index)
        if documents_str not in full_prompt:
            flag("documents_absent_from_full_prompt", row_index)
        if full_chat == full_prompt + " " + text + "</s>":
            full_chat_variants["full_prompt + one_space + output + eos"] += 1
        elif full_chat == full_prompt + " " + text:
            full_chat_variants["full_prompt + one_space + output; no eos"] += 1
        else:
            full_chat_variants["other"] += 1
            flag("full_chat_unexplained_layout", row_index)

        template_details = row["details"]["template_details"]
        settings = template_details["settings"]
        sample_log = template_details["sample_log"]
        hint_setting = bool(settings["unanswerability_hint"])
        hint_in_system_setting = bool(settings["unanswerability_hint_in_sys_msg"])
        raw_hint = sample_log.get("unansw_hint") or ""
        info_label = sample_log.get("info_label") or {}
        rendered_hint = raw_hint.replace(
            "{{info_label_plural}}", info_label.get("plural") or ""
        ).replace("{{info_label_singular}}", info_label.get("singular") or "")
        explicit_pattern = re.compile(
            r"cannot answer|cannot provide an answer|cannot assist|"
            r"acknowledge the limitation|not enough information|"
            r"insufficient (?:information|details|context)",
            flags=re.IGNORECASE,
        )
        explicit_abstention = bool(explicit_pattern.search(full_prompt))
        hint_settings[
            f"unanswerability_hint={str(hint_setting).lower()},"
            f"in_system={str(hint_in_system_setting).lower()}"
        ] += 1
        explicit_abstention_by_setting[
            f"unanswerability_hint={str(hint_setting).lower()},"
            f"explicit_in_full_prompt={str(explicit_abstention).lower()}"
        ] += 1
        if hint_setting and rendered_hint and rendered_hint in full_prompt:
            explicit_abstention_by_setting["exact_rendered_hint_found"] += 1

        outer = canonical_spans(response.get("hallucinations"))
        inner = canonical_spans(
            (((details.get("annotations") or {}).get("result") or {}).get("hallucinations"))
        )
        if outer != inner:
            flag("outer_inner_span_list_mismatch", row_index)
        valid_spans = []
        for start, end, span_text, valid in outer:
            if not valid:
                flag("released_span_valid_false", row_index)
                continue
            if not (0 <= start < end <= len(text)):
                flag("span_out_of_bounds", row_index)
                continue
            if text[start:end] != span_text:
                flag("span_text_coordinate_mismatch", row_index)
                continue
            valid_spans.append((start, end))
            total_valid_spans += 1
            span_lengths.append(end - start)
            if not any(char.isalnum() for char in text[start:end]):
                punctuation_only_spans += 1
        if any(left[1] > right[0] for left, right in zip(valid_spans, valid_spans[1:])):
            flag("overlapping_gold_spans", row_index)

        rendered = full_prompt + " " + text
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=True,
            padding=False,
            truncation=False,
        )
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
        answer_begin, answer_end = len(full_prompt) + 1, len(rendered)
        positions = np.flatnonzero(
            (offsets[:, 1] > answer_begin) & (offsets[:, 0] < answer_end)
        )
        relative = []
        for position in positions:
            left = max(int(offsets[position, 0]), answer_begin) - answer_begin
            right = min(int(offsets[position, 1]), answer_end) - answer_begin
            relative.append((left, right))
        coverage = np.zeros(len(text), dtype=bool)
        for left, right in relative:
            coverage[left:right] = True
        nonspace = np.fromiter((not char.isspace() for char in text), dtype=bool)
        if np.any(nonspace & ~coverage):
            flag("token_offsets_lost_nonspace_answer_character", row_index)

        lexical = []
        risk = []
        for left, right in relative:
            lexical.append(any(char.isalnum() for char in text[left:right]))
            risk.append(
                any(
                    any(char.isalnum() for char in text[max(left, start):min(right, end)])
                    for start, end in valid_spans
                    if left < end and right > start
                )
            )
        for start, end in valid_spans:
            if any(char.isalnum() for char in text[start:end]):
                if any(
                    left < end and right > start
                    and any(char.isalnum() for char in text[max(left, start):min(right, end)])
                    for left, right in relative
                ):
                    mapped_lexical_spans += 1
                else:
                    flag("lexical_gold_span_not_mapped_to_token", row_index)

        n_tokens = len(relative)
        starts = range(1) if n_tokens < WINDOW else range(n_tokens - WINDOW + 1)
        local_windows = 0
        local_lexical_windows = 0
        local_risk_windows = 0
        for start in starts:
            end = min(n_tokens, start + WINDOW)
            local_windows += 1
            if any(lexical[start:end]):
                local_lexical_windows += 1
                if any(risk[start:end]):
                    local_risk_windows += 1

        local_starts = [left for left, _ in relative]
        official_starts = [int(value) for value in (details.get("token_starts") or [])]
        official_token_starts_raw["rows"] += 1
        official_token_starts_raw["first_two_zero"] += bool(
            len(official_starts) >= 2 and official_starts[0] == official_starts[1] == 0
        )
        official_token_starts_raw["nondecreasing"] += all(
            left <= right for left, right in zip(official_starts, official_starts[1:])
        )
        official_token_starts_raw["terminal_equals_response_length"] += bool(
            official_starts and official_starts[-1] == len(text)
        )
        canonical_official = official_starts[1:]
        if canonical_official and canonical_official[-1] == len(text):
            canonical_official = canonical_official[:-1]
        official_token_count_deltas[len(canonical_official) - len(local_starts)] += 1
        official_token_starts_canonical_length_equal += (
            len(canonical_official) == len(local_starts)
        )
        if canonical_official == local_starts:
            official_token_starts_canonical_exact += 1
        else:
            flag("canonical_official_token_starts_differ_from_local_tokenizer", row_index)

        official_boundaries = set(canonical_official)
        official_end_boundaries = set(canonical_official[1:] + [len(text)])
        local_start_boundaries = set(local_starts)
        local_end_boundaries = {right for _, right in relative}
        for start, end in valid_spans:
            span_token_boundaries["valid_spans"] += 1
            span_token_boundaries["official_start_boundary"] += start in official_boundaries
            span_token_boundaries["official_end_boundary"] += end in official_end_boundaries
            span_token_boundaries["official_both_boundaries"] += (
                start in official_boundaries and end in official_end_boundaries
            )
            span_token_boundaries["local_start_boundary"] += start in local_start_boundaries
            span_token_boundaries["local_end_boundary"] += end in local_end_boundaries
            span_token_boundaries["local_both_boundaries"] += (
                start in local_start_boundaries and end in local_end_boundaries
            )

        answer_key = bool(row.get("answerable"))
        has_hallucination = bool(valid_spans)
        answerable[str(answer_key).lower()] += 1
        hallucinated_by_answerable[f"answerable={str(answer_key).lower()},hallucinated={str(has_hallucination).lower()}"] += 1
        group_names[str(details.get("group"))] += 1
        info_types[str(row.get("information_type"))] += 1
        categories[str(row.get("category"))] += 1
        spans_per_answer.append(len(valid_spans))
        answer_lengths.append(len(text))
        document_counts.append(len(documents))
        full_token_counts.append(len(encoded["input_ids"]))
        answer_token_counts.append(n_tokens)
        lexical_token_counts.append(sum(lexical))
        risk_token_counts.append(sum(risk))
        window_counts.append(local_windows)
        lexical_window_counts.append(local_lexical_windows)
        risk_window_counts.append(local_risk_windows)

        subset_name = "no_unanswerability_hint" if not hint_setting else "unanswerability_hint"
        subset_risk_counts[subset_name]["rows"] += 1
        subset_risk_counts[subset_name]["hallucinated_answers"] += has_hallucination
        subset_risk_counts[subset_name]["valid_spans"] += len(valid_spans)
        subset_risk_counts[subset_name]["lexical_tokens"] += sum(lexical)
        subset_risk_counts[subset_name]["risk_tokens"] += sum(risk)
        subset_risk_counts[subset_name]["lexical_windows"] += local_lexical_windows
        subset_risk_counts[subset_name]["risk_windows"] += local_risk_windows

        qidx = int(row["user_prompt_index"])
        qnorm = normalize_text(question)
        source = source_identity(row)
        context_hash = text_sha256(documents_str)
        response_hash = text_sha256(text)
        qidxs.append(qidx)
        qnorms.append(qnorm)
        source_keys.append(source)
        context_hashes.append(context_hash)
        response_hashes.append(response_hash)
        group_rows.append(
            {
                "train_row_index": row_index,
                "official_user_prompt_index": qidx,
                "question_sha256": text_sha256(question),
                "normalized_question_sha256": text_sha256(qnorm),
                "source_identity_sha256": value_sha256(source),
                "retrieved_context_sha256": context_hash,
                "llama_response_sha256": response_hash,
                "answerable": answer_key,
                "official_generation_group": details.get("group"),
                "unanswerability_hint_setting": hint_setting,
                "explicit_abstention_instruction_in_full_prompt": explicit_abstention,
            }
        )

    if len(group_rows) != EXPECTED_ROWS:
        raise RuntimeError("At least one Llama-2 train response is unusable; inspect AUDIT issues")

    dsu = DisjointSet(EXPECTED_ROWS)
    grouping_maps: dict[str, dict[object, int]] = {
        "official_question": {},
        "normalized_question": {},
        "source_identity": {},
        "exact_context": {},
    }
    values = {
        "official_question": qidxs,
        "normalized_question": qnorms,
        "source_identity": source_keys,
        "exact_context": context_hashes,
    }
    for key_type, row_values in values.items():
        seen = grouping_maps[key_type]
        for row_index, value in enumerate(row_values):
            if value in seen:
                dsu.union(row_index, seen[value])
            else:
                seen[value] = row_index

    component_members: defaultdict[int, list[int]] = defaultdict(list)
    for row_index in range(EXPECTED_ROWS):
        component_members[dsu.find(row_index)].append(row_index)
    component_ids = {
        root: "ragognize_train_group_" + value_sha256(members)[:16]
        for root, members in component_members.items()
    }
    for row in group_rows:
        row["recommended_group_id"] = component_ids[dsu.find(row["train_row_index"])]

    no_hint_rows = [row for row in group_rows if not row["unanswerability_hint_setting"]]
    question_hint_values: defaultdict[int, set[bool]] = defaultdict(set)
    for row in group_rows:
        question_hint_values[row["official_user_prompt_index"]].add(
            row["unanswerability_hint_setting"]
        )

    def duplicate_profile(values: list) -> dict:
        counts = Counter(values)
        return {
            "unique": len(counts),
            "duplicate_keys": sum(count > 1 for count in counts.values()),
            "rows_in_duplicate_keys": sum(count for count in counts.values() if count > 1),
            "max_multiplicity": max(counts.values()),
        }

    group_sizes = sorted((len(members) for members in component_members.values()), reverse=True)
    subset_descriptives = {}
    for name, counts in sorted(subset_risk_counts.items()):
        profile = dict(counts)
        profile["hallucinated_answer_rate"] = (
            counts["hallucinated_answers"] / counts["rows"]
        )
        profile["risk_token_rate_among_lexical"] = (
            counts["risk_tokens"] / counts["lexical_tokens"]
        )
        profile["risk_window_rate_among_lexical"] = (
            counts["risk_windows"] / counts["lexical_windows"]
        )
        subset_descriptives[name] = profile
    audit = {
        "scope": {
            "split_read": "train only",
            "model_subset": MODEL_KEY,
            "test_or_validation_content_read": False,
            "test_or_validation_content_downloaded": False,
            "model_forward": False,
            "training": False,
            "GPU_used": False,
        },
        "row_counts": {
            "train_prompt_rows": len(rows),
            "llama_2_response_rows": len(group_rows),
            "answerable": dict(sorted(answerable.items())),
            "hallucination_by_answerability": dict(sorted(hallucinated_by_answerable.items())),
            "answers_with_at_least_one_valid_span": sum(value > 0 for value in spans_per_answer),
            "answers_without_valid_span": sum(value == 0 for value in spans_per_answer),
        },
        "required_fields": {
            "question_present": EXPECTED_ROWS - issues["missing_question"],
            "documents_present": EXPECTED_ROWS - issues["missing_documents"],
            "documents_str_present": EXPECTED_ROWS - issues["missing_documents_str"],
            "rag_prompt_present": EXPECTED_ROWS - issues["missing_rag_prompt"],
            "llama_response_present": EXPECTED_ROWS - issues["missing_llama_response"],
            "response_text_present": EXPECTED_ROWS - issues["missing_response_text"],
            "full_prompt_present": EXPECTED_ROWS - issues["missing_full_prompt"],
            "span_list_present_in_schema": True,
        },
        "annotation_integrity": {
            "total_valid_spans": total_valid_spans,
            "punctuation_only_valid_spans": punctuation_only_spans,
            "mapped_lexical_spans": mapped_lexical_spans,
            "spans_per_answer": describe(spans_per_answer),
            "span_character_lengths": describe(span_lengths),
            "released_annotation_level": "automatic token/span-level labels; not human gold",
            "overlap_policy": "Preserve released overlapping valid spans; binary token/window labels use their union.",
        },
        "prompt_hint_audit": {
            "template_setting_counts": dict(sorted(hint_settings.items())),
            "explicit_abstention_instruction_counts": dict(
                sorted(explicit_abstention_by_setting.items())
            ),
            "definition": "Explicit means the released Llama full_prompt directly says cannot answer/provide an answer/assist, acknowledge the limitation, or that information is insufficient. Generic sole-source/do-not-infer wording is not counted.",
            "frozen_subsets_are_label_blind": True,
            "all_train_rows": EXPECTED_ROWS,
            "all_train_unique_questions": len({row["official_user_prompt_index"] for row in group_rows}),
            "all_train_recommended_groups": len({row["recommended_group_id"] for row in group_rows}),
            "no_unanswerability_hint_rows": len(no_hint_rows),
            "no_unanswerability_hint_unique_questions": len(
                {row["official_user_prompt_index"] for row in no_hint_rows}
            ),
            "no_unanswerability_hint_recommended_groups": len(
                {row["recommended_group_id"] for row in no_hint_rows}
            ),
            "official_question_groups_mixing_hint_settings": sum(
                len(values) > 1 for values in question_hint_values.values()
            ),
            "no_unanswerability_hint_definition": "template_details.settings.unanswerability_hint == false only; no response or label is consulted",
            "post_freeze_label_descriptives": subset_descriptives,
            "selection_warning": "These label rates are descriptive after the two index files were frozen; they must not be used to select the better-performing subset.",
        },
        "prompt_and_document_integrity": {
            "document_text_containment": dict(sorted(document_containment.items())),
            "full_chat_layouts": dict(sorted(full_chat_variants.items())),
            "full_prompt_plus_output_exact_without_separator_rows": 0,
            "separator_note": "The released Llama layout inserts exactly one space between full_prompt and output.",
        },
        "four_bpe_mapping": {
            "tokenizer_path": str(TOKENIZER_DIR),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_is_fast": tokenizer.is_fast,
            "window_width_raw_bpe": WINDOW,
            "stride": 1,
            "prompt_layout": "exact released Llama full_prompt + one space + exact response; add_special_tokens=False; no EOS for answer-axis audit",
            "full_sequence_tokens": describe(full_token_counts),
            "answer_raw_bpe_tokens": describe(answer_token_counts),
            "answer_lexical_tokens": describe(lexical_token_counts),
            "answer_risk_tokens": describe(risk_token_counts),
            "all_windows": describe(window_counts),
            "lexically_evaluable_windows": describe(lexical_window_counts),
            "risk_windows": describe(risk_window_counts),
            "official_token_starts_raw_structure": dict(sorted(official_token_starts_raw.items())),
            "official_token_starts_interpretation": "The raw array includes an initial BOS sentinel at 0 and usually a terminal EOS boundary at len(response). Remove the first value and a final len(response) before comparing answer-token starts.",
            "canonical_official_starts_exact_local_match_rows": official_token_starts_canonical_exact,
            "canonical_official_starts_local_length_equal_rows": official_token_starts_canonical_length_equal,
            "canonical_official_minus_local_token_count": {
                str(key): value for key, value in sorted(official_token_count_deltas.items())
            },
            "span_boundary_alignment": dict(sorted(span_token_boundaries.items())),
            "mapping_ready": not any(
                issues[name]
                for name in (
                    "span_out_of_bounds",
                    "span_text_coordinate_mismatch",
                    "token_offsets_lost_nonspace_answer_character",
                    "lexical_gold_span_not_mapped_to_token",
                )
            ),
        },
        "grouping": {
            "official_question_index": duplicate_profile(qidxs),
            "normalized_question": duplicate_profile(qnorms),
            "source_identity": duplicate_profile(source_keys),
            "exact_retrieved_context": duplicate_profile(context_hashes),
            "exact_llama_response": duplicate_profile(response_hashes),
            "recommended_components": {
                "definition": "transitive union over identical official question index, normalized question, source identity, or exact retrieved context",
                "count": len(component_members),
                "rows_in_non_singleton_components": sum(size for size in group_sizes if size > 1),
                "max_component_size": max(group_sizes),
                "component_size_distribution": dict(sorted(Counter(group_sizes).items())),
            },
            "cross_split_isolation_verified": False,
            "cross_split_limitation": "Test content/labels were intentionally not downloaded or read, so train-vs-test source/question overlap cannot be measured here.",
        },
        "lengths": {
            "question_characters": describe([len(row["user_prompt"]) for row in rows]),
            "documents_per_prompt": describe(document_counts),
            "documents_str_characters": describe([len(row["documents_str"]) for row in rows]),
            "answer_characters": describe(answer_lengths),
        },
        "distributions": {
            "information_type": dict(sorted(info_types.items())),
            "category": dict(sorted(categories.items())),
            "official_generation_group": dict(sorted(group_names.items())),
        },
        "issues": dict(sorted(issues.items())),
        "issue_train_row_indices_capped_100": dict(sorted(issue_rows.items())),
        "package_versions": {
            "python": __import__("sys").version.split()[0],
            "pyarrow": importlib.metadata.version("pyarrow"),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }

    schema = {
        "split": "train",
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "physical_columns": parquet.metadata.num_columns,
        "top_level_fields": [field.name for field in table.schema],
        "available_response_models": list(rows[0]["responses"].keys()),
        "required_paths_for_current_task": [
            "user_prompt_index",
            "user_prompt",
            "answerable",
            "documents[].{title,text}",
            "documents_str",
            "rag_prompt[].{role,content}",
            f"responses.{MODEL_KEY}.text",
            f"responses.{MODEL_KEY}.hallucinations[].{{start,end,text,valid}}",
            f"responses.{MODEL_KEY}.details.{{full_prompt,full_chat,output,token_starts,group,answerable}}",
            "details.user_prompt.details.suitable_article_index",
            "details.user_prompt.details.suitable_article.{revision_id,url,title}",
        ],
        "parquet_arrow_schema": str(table.schema),
    }

    (ROOT / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "SCHEMA.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "GROUP_INDEX.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in group_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    index_fields = (
        "train_row_index",
        "official_user_prompt_index",
        "recommended_group_id",
        "llama_response_sha256",
    )
    with (ROOT / "ALL_TRAIN_INDEX.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in group_rows:
            handle.write(json.dumps({key: row[key] for key in index_fields}, sort_keys=True) + "\n")
    with (ROOT / "NO_UNANSWERABILITY_HINT_INDEX.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in group_rows:
            if not row["unanswerability_hint_setting"]:
                handle.write(json.dumps({key: row[key] for key in index_fields}, sort_keys=True) + "\n")

    print(json.dumps({
        "rows": len(rows),
        "llama_rows": len(group_rows),
        "hallucinated_answers": audit["row_counts"]["answers_with_at_least_one_valid_span"],
        "valid_spans": total_valid_spans,
        "issues": audit["issues"],
        "groups": len(component_members),
        "mapping_ready": audit["four_bpe_mapping"]["mapping_ready"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
