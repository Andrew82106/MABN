"""Prepare traceable human-correction minimal pairs from auxiliary RAGTruth.

The only dataset opened here is auxiliary_human_v1/candidate_fit.jsonl.
No model weights or GPU APIs are used.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUX = ROOT / "auxiliary_human_v1"
OUT = ROOT / "results/human_correction_pair_pilot_v1"
ATOMIC_CODE = ROOT / "research/atomic_relation_audit_r32_v1/audit_atomic_relations.py"
MINICHECK_TOKENIZER = ROOT / "semantic_baseline/model"
MODERN_TOKENIZER = ROOT.parent / "models/ModernBERT-base-nli"
NLTK_DATA = ROOT / "semantic_baseline/nltk_data"

N_AUX = 9_678
N_CONFLICT_SPANS = 4_381
SEED = 20_261_024
PARENT_BPE_LIMIT = 96
MAX_PAIR_TOKENS = 2_048
CONFLICT_TYPES = frozenset(("Evident Conflict", "Subtle Conflict"))
KNOWN_TYPES = CONFLICT_TYPES | frozenset(("Evident Baseless Info", "Subtle Baseless Info"))
WORD = re.compile(r"[^\W_]+", re.UNICODE)
ORIGINAL_START = re.compile(r"(?im)^[ \t]*Original[ \t]*:[ \t]*")
BAD_START = re.compile(r"(?im)^[ \t]*(?:AIGC|Generated|Generative)[ \t]*:[ \t]*")
ANY_FIELD_START = re.compile(
    r"(?im)^[ \t]*(?:Original|AIGC|Generated|Generative)[ \t]*:[ \t]*")

TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


atomic = load_module("atomic_human_correction_pair_v1", ATOMIC_CODE)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def words_with_offsets(text):
    return [(match.group(0).casefold(), match.start(), match.end()) for match in WORD.finditer(text)]


def normalized_words(text):
    return [value for value, _left, _right in words_with_offsets(text)]


def find_sequences(needle, text):
    haystack = words_with_offsets(text)
    values = [value for value, _left, _right in haystack]
    if not needle or len(needle) > len(values):
        return []
    output = []
    width = len(needle)
    for start in range(len(values) - width + 1):
        if values[start:start + width] == needle:
            output.append((haystack[start][1], haystack[start + width - 1][2], start, start + width))
    return output


def parse_human_fields(meta):
    if not isinstance(meta, str):
        return None, None
    original_match = ORIGINAL_START.search(meta)
    if original_match is None:
        return None, None
    bad_match = BAD_START.search(meta, original_match.end())
    original_stop = bad_match.start() if bad_match is not None else len(meta)
    original = " ".join(meta[original_match.end():original_stop].split()).strip()
    bad = None
    if bad_match is not None:
        next_match = ANY_FIELD_START.search(meta, bad_match.end())
        bad_stop = next_match.start() if next_match is not None else len(meta)
        bad = " ".join(meta[bad_match.end():bad_stop].split()).strip()
    return original or "", bad or ""


def sentence_spans(text):
    """Exact source-sentence rule already used by the local NLI evidence code."""
    assert text and any(char.isalnum() for char in text)
    cuts, i, line_start = [], 0, 0
    while i < len(text):
        char = text[i]
        if char in "\r\n":
            if char == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
            cuts.append(i + 1); line_start = i + 1
        elif char in TERMINAL:
            end = i + 1
            while end < len(text) and (text[end] in TERMINAL or text[end] in CLOSERS):
                end += 1
            prefix = text[line_start:i + 1].strip()
            followed = end == len(text) or text[end].isspace()
            match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final = match.group(0).lower() if match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final))
            protected = end < len(text) and (final in ABBREVIATIONS or dotted or initial)
            if followed and not LIST_PREFIX.fullmatch(prefix) and not protected:
                cuts.append(end); i = end - 1
        i += 1
    cuts.append(len(text))
    output, start = [], 0
    for stop in sorted(set(cuts)):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(char.isalnum() for char in text[left:right]):
            output.append((left, right))
        start = stop
    coverage = np.zeros(len(text), dtype=np.int8)
    for left, right in output:
        coverage[left:right] += 1
    lexical = np.fromiter((char.isalnum() for char in text), dtype=bool)
    assert np.all(coverage[lexical] == 1) and np.all(coverage <= 1)
    return output


def trim(text, left, right):
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def sentence_parents(text, punkt, tokenizer):
    spans, offset = [], 0
    for line in text.splitlines(keepends=True):
        blocks = list(punkt.span_tokenize(line)); index = 0
        while index < len(blocks):
            left, right = blocks[index]
            if re.fullmatch(r"\s*(?:\d+[.)]|[-*•])\s*", line[left:right]) and index + 1 < len(blocks):
                right = blocks[index + 1][1]; index += 1
            left, right = trim(text, offset + left, offset + right)
            if left < right:
                spans.append((left, right))
            index += 1
        offset += len(line)
    assert offset == len(text)
    pieces = []
    for outer_left, outer_right in spans:
        left, right = outer_left, outer_right
        while left < right:
            left, right = trim(text, left, right)
            if left >= right:
                break
            encoded = tokenizer(text[left:right], add_special_tokens=False,
                                return_offsets_mapping=True, truncation=False)
            if len(encoded["input_ids"]) <= PARENT_BPE_LIMIT:
                pieces.append((left, right)); break
            stop = left + encoded["offset_mapping"][PARENT_BPE_LIMIT - 1][1]
            lower = left + encoded["offset_mapping"][3 * PARENT_BPE_LIMIT // 4 - 1][1]
            gaps = [match.start() + left for match in re.finditer(r"\s+", text[left:stop])
                    if match.start() + left >= lower]
            if gaps:
                stop = gaps[-1]
            while stop > left and len(tokenizer.encode(text[left:stop], add_special_tokens=False)) > PARENT_BPE_LIMIT:
                stop -= 1
            assert stop > left
            a, b = trim(text, left, stop)
            if a < b:
                pieces.append((a, b))
            left = stop
    covered = np.zeros(len(text), dtype=bool)
    for left, right in pieces:
        covered[left:right] = True
    assert all(char.isspace() or covered[index] for index, char in enumerate(text))
    return pieces


def microclaims(text, punkt, tokenizer, prefix):
    output = []
    for parent_index, (left, right) in enumerate(sentence_parents(text, punkt, tokenizer)):
        spans, reasons = atomic.atomic_split(text, left, right)
        for child_index, (start, end) in enumerate(spans):
            if not any(char.isalnum() for char in text[start:end]):
                continue
            output.append({
                "microclaim_id": f"{prefix}__atomic_{len(output):03d}",
                "microclaim_index": len(output), "parent_claim_index": parent_index,
                "child_index": child_index, "children_in_parent": len(spans),
                "parent_split_reasons": reasons, "start": start, "end": end,
                "text": text[start:end],
            })
    assert output
    covered = np.zeros(len(text), dtype=bool)
    for row in output:
        covered[row["start"]:row["end"]] = True
    assert all(not char.isalnum() or covered[index] for index, char in enumerate(text))
    return output


def lexical_overlap(text, left, right, other_left, other_right):
    lo, hi = max(left, other_left), min(right, other_right)
    return lo < hi and any(text[index].isalnum() for index in range(lo, hi))


def fold_for(group_id):
    return int(digest(f"{SEED}|{group_id}")[:16], 16) % 5


def quality_tier(original, best_coverage, exact_source, exact_bad, single_error, minimal_safe):
    if original is None:
        return "E_no_explicit_original"
    if best_coverage < 0.8:
        return "D_original_source_coverage_below_0.8"
    if not exact_source:
        return "C_coverage_only_not_exact_source_phrase"
    if not exact_bad or not single_error or not minimal_safe:
        return "B_exact_source_but_unsafe_minimal_edit"
    return "A_primary_minimal_pair"


def prepare_rows(punkt, mini_tokenizer):
    triples, audits = [], []
    span_types, tasks, failure_counts = Counter(), Counter(), Counter()
    sentence_cache = {}
    seen_sources = {}
    rows_read = 0
    with (AUX / "candidate_fit.jsonl").open(encoding="utf-8") as handle:
        for row_index in range(N_AUX):
            line = handle.readline()
            assert line, row_index
            source = json.loads(line); rows_read += 1
            assert source["partition"] == "auxiliary_candidate_fit"
            assert source["official_split"] == "train" and source["quality"] == "good"
            assert not source["new_labels_generated"]
            response, material = source["original_response"], source["retrieved_passages"]
            assert digest(response) == source["answer_sha256"]
            source_id = str(source["source_id"])
            if source_id not in seen_sources:
                seen_sources[source_id] = digest(material)
                sentence_cache[source_id] = [
                    {"sentence_id": index, "start": left, "end": right,
                     "text": material[left:right], "text_sha256": digest(material[left:right])}
                    for index, (left, right) in enumerate(sentence_spans(material))]
            else:
                assert seen_sources[source_id] == digest(material)
            conflict_indices = [index for index, label in enumerate(source["labels"])
                                if label["label_type"] in CONFLICT_TYPES]
            if not conflict_indices:
                continue
            claims = microclaims(response, punkt, mini_tokenizer, f"aux_{source['response_id']}")
            for span_index in conflict_indices:
                label = source["labels"][span_index]
                span_types[label["label_type"]] += 1; tasks[source["task_type"]] += 1
                span_left, span_right = int(label["start"]), int(label["end"])
                assert 0 <= span_left < span_right <= len(response)
                assert response[span_left:span_right] == label["text"]
                original, generated = parse_human_fields(label.get("meta"))
                failure = []
                if original is None:
                    failure.append("no_explicit_original_field")
                elif not normalized_words(original):
                    failure.append("original_has_no_words")
                if generated is None or not normalized_words(generated):
                    failure.append("no_explicit_generated_field")

                original_tokens = normalized_words(original or "")
                generated_tokens = normalized_words(generated or "")
                sentence_rows = sentence_cache[source_id]
                coverage_rows, source_exact = [], []
                original_set = set(original_tokens)
                for sentence in sentence_rows:
                    sentence_set = set(normalized_words(sentence["text"]))
                    coverage = len(original_set & sentence_set) / len(original_set) if original_set else 0.0
                    coverage_rows.append((coverage, sentence["sentence_id"]))
                    for occurrence in find_sequences(original_tokens, sentence["text"]):
                        source_exact.append((sentence, occurrence))
                coverage_rows.sort(key=lambda item: (-item[0], item[1]))
                best_coverage = coverage_rows[0][0] if coverage_rows else 0.0
                if best_coverage < 0.8:
                    failure.append("original_source_lexical_coverage_below_0.8")
                if not source_exact:
                    failure.append("original_not_exact_contiguous_source_phrase")
                distinct_sentence_hashes = {sentence["text_sha256"] for sentence, _ in source_exact}
                if len(distinct_sentence_hashes) > 1:
                    failure.append("original_phrase_ambiguous_across_distinct_source_sentences")
                selected_source = min(source_exact, key=lambda item: (item[0]["sentence_id"], item[1][0])) \
                    if source_exact and len(distinct_sentence_hashes) == 1 else None

                bad_exact = []
                if generated_tokens:
                    for claim in claims:
                        for occurrence in find_sequences(generated_tokens, claim["text"]):
                            rel_left, rel_right = occurrence[0], occurrence[1]
                            absolute_left, absolute_right = claim["start"] + rel_left, claim["start"] + rel_right
                            token_offsets = words_with_offsets(claim["text"])[occurrence[2]:occurrence[3]]
                            fully_inside = all(
                                span_left <= claim["start"] + left and claim["start"] + right <= span_right
                                for _word, left, right in token_offsets)
                            if fully_inside:
                                bad_exact.append((claim, occurrence, absolute_left, absolute_right))
                if not bad_exact:
                    failure.append("generated_field_not_exact_inside_human_span")
                if len(bad_exact) > 1:
                    failure.append("generated_field_ambiguous_inside_human_span")
                selected_bad = bad_exact[0] if len(bad_exact) == 1 else None

                other_errors = []
                if selected_bad is not None:
                    claim = selected_bad[0]
                    for other_index, other in enumerate(source["labels"]):
                        if other_index == span_index:
                            continue
                        assert other["label_type"] in KNOWN_TYPES
                        if lexical_overlap(response, claim["start"], claim["end"],
                                           int(other["start"]), int(other["end"])):
                            other_errors.append(other_index)
                    if other_errors:
                        failure.append("microclaim_touches_other_human_error_span")

                corrected = source_phrase = None
                context_words = bad_words = corrected_words = length_delta_limit = 0
                if selected_bad is not None and selected_source is not None:
                    claim, occurrence, _absolute_left, _absolute_right = selected_bad
                    sentence, source_occurrence = selected_source
                    source_phrase = sentence["text"][source_occurrence[0]:source_occurrence[1]]
                    rel_left, rel_right = occurrence[0], occurrence[1]
                    corrected = claim["text"][:rel_left] + source_phrase + claim["text"][rel_right:]
                    bad_words = len(normalized_words(claim["text"]))
                    corrected_words = len(normalized_words(corrected))
                    context_words = sum(
                        right <= rel_left or left >= rel_right
                        for _word, left, right in words_with_offsets(claim["text"]))
                    length_delta_limit = max(2, math.ceil(0.25 * bad_words))
                    if original_tokens == generated_tokens:
                        failure.append("correction_does_not_change_lexical_content")
                    if len(original_tokens) < 2:
                        failure.append("original_phrase_has_fewer_than_two_words")
                    if abs(corrected_words - bad_words) > length_delta_limit:
                        failure.append("corrected_microclaim_length_delta_too_large")
                    if context_words == 0 and len(original_tokens) != len(generated_tokens):
                        failure.append("full_replacement_with_length_change")

                exact_source_ok = selected_source is not None and best_coverage >= 0.8
                exact_bad_ok = selected_bad is not None
                single_error_ok = exact_bad_ok and not other_errors
                minimal_failures = {
                    "correction_does_not_change_lexical_content",
                    "original_phrase_has_fewer_than_two_words",
                    "corrected_microclaim_length_delta_too_large",
                    "full_replacement_with_length_change",
                }
                minimal_safe = corrected is not None and not (minimal_failures & set(failure))
                tier = quality_tier(original, best_coverage, exact_source_ok,
                                    exact_bad_ok, single_error_ok, minimal_safe)
                is_primary = tier == "A_primary_minimal_pair" and not failure
                if not is_primary and tier == "A_primary_minimal_pair":
                    tier = "B_exact_source_but_unsafe_minimal_edit"
                failure_counts.update(failure)
                audit = {
                    "origin": "auxiliary_human_v1", "source_id": source_id,
                    "group_id": source["group_id"], "response_id": str(source["response_id"]),
                    "task_type": source["task_type"], "span_index": span_index,
                    "label_type": label["label_type"], "span_start": span_left,
                    "span_end": span_right, "span_text": label["text"],
                    "span_text_sha256": digest(label["text"]),
                    "meta_sha256": digest(label.get("meta")),
                    "parsed_original": original, "parsed_generated": generated,
                    "original_word_count": len(original_tokens),
                    "generated_word_count": len(generated_tokens),
                    "best_original_source_lexical_coverage": best_coverage,
                    "exact_source_occurrences": len(source_exact),
                    "distinct_exact_source_sentence_hashes": len(distinct_sentence_hashes),
                    "exact_bad_occurrences_inside_span": len(bad_exact),
                    "other_error_spans_in_selected_microclaim": other_errors,
                    "bad_microclaim_word_count": bad_words,
                    "corrected_microclaim_word_count": corrected_words,
                    "unchanged_context_word_count": context_words,
                    "length_delta_limit": length_delta_limit,
                    "quality_tier": tier, "retained_primary": is_primary,
                    "failure_reasons": sorted(set(failure)),
                }
                audits.append(audit)
                if not is_primary:
                    continue

                claim, occurrence, absolute_left, absolute_right = selected_bad
                sentence, source_occurrence = selected_source
                triple_id = f"hcp_{source['response_id']}_{span_index:02d}"
                premise = f"Question: {source['question']}\nEvidence: {sentence['text']}"
                triples.append({
                    "origin": "auxiliary_human_v1", "triple_id": triple_id,
                    "source_id": source_id, "group_id": source["group_id"],
                    "fold": fold_for(source["group_id"]),
                    "response_id": str(source["response_id"]), "task_type": source["task_type"],
                    "question": source["question"], "question_sha256": digest(source["question"]),
                    "answer_sha256": source["answer_sha256"],
                    "material_sha256": digest(material),
                    "span_index": span_index, "label_type": label["label_type"],
                    "span_start": span_left, "span_end": span_right,
                    "span_text": label["text"], "span_text_sha256": digest(label["text"]),
                    "human_meta_sha256": digest(label.get("meta")),
                    "human_original": original, "human_generated": generated,
                    "bad_microclaim_id": claim["microclaim_id"],
                    "bad_microclaim_index": claim["microclaim_index"],
                    "bad_microclaim_start": claim["start"], "bad_microclaim_end": claim["end"],
                    "bad_microclaim": claim["text"],
                    "replacement_start_in_microclaim": occurrence[0],
                    "replacement_end_in_microclaim": occurrence[1],
                    "replacement_start_in_answer": absolute_left,
                    "replacement_end_in_answer": absolute_right,
                    "bad_replacement_text": claim["text"][occurrence[0]:occurrence[1]],
                    "source_correction_text": source_phrase,
                    "corrected_microclaim": corrected,
                    "source_sentence_id": sentence["sentence_id"],
                    "source_sentence_start": sentence["start"],
                    "source_sentence_end": sentence["end"],
                    "source_sentence": sentence["text"],
                    "source_sentence_sha256": sentence["text_sha256"],
                    "source_correction_start_in_sentence": source_occurrence[0],
                    "source_correction_end_in_sentence": source_occurrence[1],
                    "source_original_lexical_coverage": best_coverage,
                    "unchanged_context_word_count": context_words,
                    "bad_microclaim_word_count": bad_words,
                    "corrected_microclaim_word_count": corrected_words,
                    "premise": premise, "premise_sha256": digest(premise),
                })
            if (row_index + 1) % 1000 == 0:
                print("HUMAN_CORRECTION_PARSE", row_index + 1, N_AUX, len(audits), len(triples), flush=True)
    assert rows_read == N_AUX and len(audits) == N_CONFLICT_SPANS
    triples.sort(key=lambda row: (row["group_id"], row["response_id"], row["span_index"]))
    for index, row in enumerate(triples):
        row["triple_index"] = index
    return triples, audits, {
        "rows_read": rows_read, "released_conflict_spans": len(audits),
        "span_type_counts": dict(span_types), "task_counts": dict(tasks),
        "quality_tier_counts": dict(Counter(row["quality_tier"] for row in audits)),
        "failure_reason_counts": dict(failure_counts),
    }


def encode_pairs(triples, tokenizer):
    input_ids, indptr = array("i"), [0]
    roles, owners = array("b"), array("i")
    role_names = ("bad", "corrected", "source")
    for triple_index, row in enumerate(triples):
        hypotheses = (row["bad_microclaim"], row["corrected_microclaim"], row["source_sentence"])
        pair_indices, pair_lengths, pair_hashes = [], [], []
        for role_code, (role, hypothesis) in enumerate(zip(role_names, hypotheses)):
            encoded = tokenizer(row["premise"], hypothesis, add_special_tokens=True,
                                padding=False, truncation=False)
            assert len(encoded["input_ids"]) <= MAX_PAIR_TOKENS
            pair_index = len(roles); pair_indices.append(pair_index)
            pair_lengths.append(len(encoded["input_ids"]))
            pair_hashes.append(digest([row["premise"], hypothesis]))
            input_ids.extend(encoded["input_ids"]); indptr.append(len(input_ids))
            roles.append(role_code); owners.append(triple_index)
        row["pair_indices"] = pair_indices
        row["pair_input_lengths"] = pair_lengths
        row["pair_input_sha256"] = pair_hashes
    group_counts = Counter(row["group_id"] for row in triples)
    weights = np.asarray([1.0 / group_counts[row["group_id"]] for row in triples], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return {
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "indptr": np.asarray(indptr, dtype=np.int64),
        "pair_role_code": np.asarray(roles, dtype=np.int8),
        "pair_triple_index": np.asarray(owners, dtype=np.int32),
        "triple_fold": np.asarray([row["fold"] for row in triples], dtype=np.int8),
        "triple_weight": weights.astype(np.float32),
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "preparation_complete.json").exists()
    aux_manifest = json.loads((AUX / "manifest.json").read_text(encoding="utf-8"))
    assert sha(AUX / "candidate_fit.jsonl") == aux_manifest["artifacts_sha256"]["candidate_fit.jsonl"]
    upstream = json.loads((AUX / "INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))
    assert upstream["status"] == "passed"
    assert upstream["source_and_labels"]["test_answer_or_gold_JSON_parsed"] is False

    import nltk
    from nltk.tokenize.punkt import PunktTokenizer
    nltk.data.path.insert(0, str(NLTK_DATA))
    punkt = PunktTokenizer("english")
    mini_tokenizer = AutoTokenizer.from_pretrained(
        MINICHECK_TOKENIZER, local_files_only=True, use_fast=True, trust_remote_code=False)
    modern_tokenizer = AutoTokenizer.from_pretrained(
        MODERN_TOKENIZER, local_files_only=True, use_fast=True, trust_remote_code=False)

    triples, audits, parse_stats = prepare_rows(punkt, mini_tokenizer)
    assert triples
    arrays = encode_pairs(triples, modern_tokenizer)
    write_jsonl(OUT / "triples.jsonl", triples)
    write_jsonl(OUT / "span_audit.jsonl", audits)
    write_npz(OUT / "arrays.npz", **arrays)

    lengths = np.diff(arrays["indptr"])
    fold_counts = Counter(row["fold"] for row in triples)
    fold_groups = {str(fold): len({row["group_id"] for row in triples if row["fold"] == fold})
                   for fold in range(5)}
    task_counts = Counter((row["fold"], row["task_type"]) for row in triples)
    type_counts = Counter((row["fold"], row["label_type"]) for row in triples)
    complete = {
        "status": "cpu_preparation_complete",
        "source_audit": parse_stats,
        "retained": {
            "primary_triples": len(triples),
            "fraction_of_4381_spans": len(triples) / N_CONFLICT_SPANS,
            "groups": len({row["group_id"] for row in triples}),
            "sources": len({row["source_id"] for row in triples}),
            "fold_triples": {str(key): value for key, value in sorted(fold_counts.items())},
            "fold_groups": fold_groups,
            "fold_task_counts": {f"fold{fold}_{task}": value
                                 for (fold, task), value in sorted(task_counts.items())},
            "fold_type_counts": {f"fold{fold}_{kind}": value
                                 for (fold, kind), value in sorted(type_counts.items())},
        },
        "encoded": {
            "pairs": len(arrays["pair_role_code"]), "flat_tokens": len(arrays["input_ids"]),
            "max_pair_tokens": int(lengths.max()), "p50_pair_tokens": float(np.median(lengths)),
            "p95_pair_tokens": float(np.quantile(lengths, 0.95)), "no_truncation": True,
            "tokenizer_pad_id": int(modern_tokenizer.pad_token_id),
        },
        "isolation": {
            "assignment": "sha256(seed|material_group_id) modulo 5",
            "seed": SEED,
            "groups_in_multiple_folds": 0,
            "upstream_quarantined_sources": 62,
            "upstream_audit_sha256": sha(AUX / "INDEPENDENT_AUDIT.json"),
        },
        "scope": {
            "auxiliary_candidate_fit_rows_read": N_AUX,
            "other_dataset_rows_read": 0,
            "model_weights_loaded": False, "GPU_used": False,
            "published_baselines_modified": False, "prior_runners_modified": False,
        },
        "source_sha256": {
            "auxiliary_human_v1/manifest.json": sha(AUX / "manifest.json"),
            "auxiliary_human_v1/candidate_fit.jsonl": sha(AUX / "candidate_fit.jsonl"),
            "auxiliary_human_v1/INDEPENDENT_AUDIT.json": sha(AUX / "INDEPENDENT_AUDIT.json"),
            "research/atomic_relation_audit_r32_v1/audit_atomic_relations.py": sha(ATOMIC_CODE),
            "results/human_correction_pair_pilot_v1/PROTOCOL.md": sha(OUT / "PROTOCOL.md"),
        },
    }
    write_json(OUT / "preparation_complete.json", complete)
    write_json(OUT / "manifest.json", {
        "status": "cpu_data_ready_gpu_not_run",
        "files_sha256": {name: sha(OUT / name) for name in
                         ("PROTOCOL.md", "triples.jsonl", "span_audit.jsonl",
                          "arrays.npz", "preparation_complete.json")},
        "other_dataset_rows_read": 0, "model_weights_loaded": False, "GPU_used": False,
    })
    print("HUMAN_CORRECTION_PAIR_PREPARED", len(triples), flush=True)


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    complete = json.loads((OUT / "preparation_complete.json").read_text(encoding="utf-8"))
    with np.load(OUT / "arrays.npz", allow_pickle=False) as arrays:
        assert int(arrays["indptr"][0]) == 0
        assert int(arrays["indptr"][-1]) == len(arrays["input_ids"])
        assert len(arrays["pair_role_code"]) * 3 == len(arrays["indptr"][:-1]) * 3
        assert len(arrays["pair_role_code"]) == len(arrays["pair_triple_index"])
        assert len(arrays["triple_fold"]) == len(arrays["triple_weight"])
        assert len(arrays["pair_role_code"]) == 3 * len(arrays["triple_fold"])
    assert complete["scope"]["other_dataset_rows_read"] == 0
    assert complete["scope"]["GPU_used"] is False
    print("HUMAN_CORRECTION_PAIR_CPU_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.stage]()
