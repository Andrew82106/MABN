"""Direct, fit-only audit of RAGTruth annotator `Original:` notes.

The input is parsed here from candidate_fit.jsonl.  No preparation/training
runner, calibration file, test file, model, or GPU is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import ast
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "auxiliary_human_v1/candidate_fit.jsonl"
EXPECTED_INPUT_SHA256 = "4ad2008960abfd1835c3bddd49a1f0d794c729d051e70070d0b793525078f4c0"
REVIEW = HERE / "MANUAL_REVIEW.jsonl"
RESULT = HERE / "RESULTS.json"
REPORT = HERE / "REPORT.md"
CONFLICT_TYPES = ("Evident Conflict", "Subtle Conflict")
SAMPLE_PER_STRATUM = 25
SAMPLE_SEED = "human-original-pair-feasibility-v1|2026-09-12"

WORD_RE = re.compile(r"[a-z0-9]+(?:[.'/-][a-z0-9]+)*", re.I)
NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:[.:/-]\d+)*(?:%|am|pm)?", re.I)
MARKER_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 ()/_-]{0,40}:\s*")
STRICT_ORIGINAL_RE = re.compile(r"^\s*Original:\s*(.*)$", re.I)
VARIANT_ORIGINAL_RE = re.compile(r"^\s*Original\s*\([^)]{1,40}\):\s*(.*)$", re.I)
INLINE_ANNOTATOR_MARKER_RE = re.compile(
    r"\s*;\s*(?:AIGC|Generative|Generated|AOGC)\s*:\s*|"
    r"\s{2,}(?:AIGC|Generative|Generated|AOGC)\s*:\s*", re.I)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "in", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "were",
    "which", "who", "will", "with", "would", "said", "says", "stated",
}

EXPLANATORY_PATTERNS = {
    "source_reference": re.compile(r"\b(?:source|source content|article|business info|provided data)\b", re.I),
    "review_reference": re.compile(r"\b(?:review|reviewer|customer)\b", re.I),
    "not_mentioned": re.compile(r"\b(?:not|never|no)\s+(?:explicitly\s+)?(?:mention(?:ed)?|state(?:d)?|provide(?:d)?|indicate(?:d)?)\b", re.I),
    "correction_discourse": re.compile(r"\b(?:instead|rather than|not the|but not|the correct|should be|actually|in fact|according to)\b", re.I),
    "metalinguistic": re.compile(r"\b(?:aigc|generat(?:ive|ed)|claim|statement|information|content)\b", re.I),
    "schema_boolean": re.compile(r"(?:^|\b)(?:true|false|none|null|yes|no)\b|[\"'][A-Za-z][^\n]{0,50}[\"']\s*:", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("“", '"').replace("”", '"')


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_unicode(text).lower())


def content_words(text: str) -> list[str]:
    return [token for token in words(text) if token not in STOPWORDS]


def normalized_words(text: str) -> str:
    return " ".join(words(text))


def strip_wrappers(text: str) -> str:
    value = normalize_unicode(text).strip()
    value = re.sub(r"^\s*(?:\.\.\.|\[?original\]?)\s*", "", value, flags=re.I)
    while len(value) >= 2 and ((value[0], value[-1]) in {("\"", "\""), ("'", "'")}):
        value = value[1:-1].strip()
    return value.strip()


def parse_original(meta) -> dict:
    text = normalize_unicode(meta or "")
    lines = text.splitlines()
    exact, variants = [], []
    for index, line in enumerate(lines):
        match = STRICT_ORIGINAL_RE.match(line)
        kind = "strict"
        if not match:
            match = VARIANT_ORIGINAL_RE.match(line)
            kind = "variant"
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            continuation = []
            for following in lines[index + 1:]:
                if MARKER_RE.match(following):
                    break
                if following.strip():
                    continuation.append(following.strip())
                elif continuation:
                    break
            value = " ".join(continuation).strip()
        inline_marker = INLINE_ANNOTATOR_MARKER_RE.search(value)
        if inline_marker:
            value = value[:inline_marker.start()].strip()
        (exact if kind == "strict" else variants).append(value)
    chosen = exact[0] if len(exact) == 1 and exact[0] else None
    status = (
        "one_strict_nonempty" if chosen is not None else
        "multiple_strict" if len(exact) > 1 else
        "strict_empty" if len(exact) == 1 else
        "variant_only" if variants else "missing"
    )
    return {"status": status, "strict_values": exact,
            "variant_values": variants, "value": chosen,
            "inline_annotator_marker_removed": bool(
                INLINE_ANNOTATOR_MARKER_RE.search(text))}


def multiset_overlap(left: list[str], right: list[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def source_segments(source: str, task: str) -> list[str]:
    source = normalize_unicode(source)
    if task == "Summary":
        candidates = re.split(r"(?<=[.!?])\s+|\n+", source)
    else:
        candidates = re.split(r"(?<=[.!?}])\s+|,\s*(?=[\"']?[A-Za-z_])|\n+", source)
    output = []
    for segment in candidates:
        segment = segment.strip()
        if segment and len(words(segment)):
            output.append(segment)
    return output


def best_lexical_segment(original: str, source: str, task: str) -> dict:
    query = content_words(original)
    if not query:
        query = words(original)
    best = {"f1": 0.0, "recall": 0.0, "precision": 0.0, "segment": ""}
    for segment in source_segments(source, task):
        candidate = content_words(segment)
        if not candidate:
            candidate = words(segment)
        overlap = multiset_overlap(query, candidate)
        recall = overlap / len(query) if query else 0.0
        precision = overlap / len(candidate) if candidate else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        key = (f1, recall, -abs(len(candidate) - len(query)))
        old = (best["f1"], best["recall"],
               -abs(len(content_words(best["segment"])) - len(query)))
        if key > old:
            best = {"f1": f1, "recall": recall, "precision": precision,
                    "segment": segment}
    return best


def flatten_mapping(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_mapping(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_mapping(child, prefix + (str(index),))
    else:
        yield prefix, value


def structured_key_value_alignment(original: str, source: str, task: str) -> dict:
    result = {"parsed_source": False, "candidate": False, "matched": False,
              "key": None, "value": None}
    if task != "Data2txt":
        return result
    try:
        structure = ast.literal_eval(source)
    except (SyntaxError, ValueError):
        return result
    result["parsed_source"] = isinstance(structure, dict)
    if not result["parsed_source"]:
        return result
    text = strip_wrappers(original).strip().rstrip(",")
    patterns = [
        re.compile(r"^[\"']?([^:\"']+)[\"']?\s*:\s*[\"']?(.+?)[\"']?$", re.I),
        re.compile(r"^(true|false|none|null|yes|no)\s+for\s+(.+)$", re.I),
    ]
    key = value = None
    first = patterns[0].match(text)
    if first:
        key, value = first.group(1), first.group(2)
    else:
        second = patterns[1].match(text)
        if second:
            value, key = second.group(1), second.group(2)
    if key is None:
        return result
    key_tokens, value_tokens = words(key), words(value)
    result.update({"candidate": True, "key": key, "value": value})
    for path, leaf in flatten_mapping(structure):
        if not path:
            continue
        leaf_key = words(path[-1])
        leaf_value = words(str(leaf))
        key_match = normalized_words(" ".join(leaf_key)) == normalized_words(" ".join(key_tokens))
        value_match = normalized_words(" ".join(leaf_value)) == normalized_words(" ".join(value_tokens))
        if key_match and value_match:
            result["matched"] = True
            break
    return result


def evidence_alignment(original: str, source: str, task: str) -> dict:
    clean = strip_wrappers(original)
    raw_exact = bool(clean and clean.lower() in normalize_unicode(source).lower())
    query_norm, source_norm = normalized_words(clean), normalized_words(source)
    normalized_exact = bool(query_norm and len(words(clean)) >= 2 and query_norm in source_norm)
    occurrences = source_norm.count(query_norm) if query_norm else 0
    best = best_lexical_segment(clean, source, task)
    structured = structured_key_value_alignment(clean, source, task)
    original_numbers = NUMBER_RE.findall(clean.lower())
    source_numbers = NUMBER_RE.findall(source.lower())
    number_compatible = all(value in source_numbers for value in original_numbers)
    lexical_strong = bool(
        structured["matched"] or raw_exact or normalized_exact or
        (len(content_words(clean)) >= 2 and best["recall"] >= 0.80 and
         best["f1"] >= 0.58 and number_compatible)
    )
    semantic_candidate = bool(
        not lexical_strong and len(content_words(clean)) >= 2 and
        best["recall"] >= 0.35 and number_compatible
    )
    tier = ("exact_raw" if raw_exact else "exact_normalized" if normalized_exact else
            "structured_exact" if structured["matched"] else
            "lexical_strong" if lexical_strong else
            "semantic_review_candidate" if semantic_candidate else "not_aligned")
    return {
        "tier": tier, "raw_exact": raw_exact,
        "normalized_exact": normalized_exact, "normalized_occurrences": occurrences,
        "lexical_strong": lexical_strong, "semantic_review_candidate": semantic_candidate,
        "number_compatible": number_compatible,
        "best_segment_f1": best["f1"], "best_segment_recall": best["recall"],
        "best_segment": best["segment"], "structured": structured,
    }


def explanatory_flags(original: str) -> list[str]:
    return [name for name, pattern in EXPLANATORY_PATTERNS.items()
            if pattern.search(original)]


def token_edit(old: str, new: str) -> dict:
    left, right = words(old), words(new)
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    deleted = inserted = 0
    changed_left, changed_right = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            deleted += i2 - i1
            inserted += j2 - j1
            changed_left.extend(left[i1:i2]); changed_right.extend(right[j1:j2])
    changed = changed_left + changed_right
    numeric_only = bool(changed and all(NUMBER_RE.fullmatch(token) for token in changed))
    boolean_vocab = {"true", "false", "yes", "no", "none", "null"}
    boolean_only = bool(changed and set(changed).issubset(boolean_vocab))
    negation_vocab = {"not", "no", "never", "without", "with", "under", "over"}
    negation_only = bool(changed and set(changed).issubset(negation_vocab))
    return {
        "old_words": len(left), "good_words": len(right),
        "deleted_words": deleted, "inserted_words": inserted,
        "changed_bad": changed_left, "changed_good": changed_right,
        "one_token_each_or_less": deleted <= 1 and inserted <= 1,
        "numeric_only_change": numeric_only,
        "boolean_only_change": boolean_only,
        "negation_or_direction_only_change": negation_only,
        "token_similarity": matcher.ratio(),
    }


def surface_diagnostic(response: str, start: int, end: int, bad: str,
                       good: str, evidence: dict) -> dict:
    flags = explanatory_flags(good)
    bad_words, good_words = words(bad), words(good)
    word_ratio = len(good_words) / len(bad_words) if bad_words else math.inf
    char_ratio = len(good.strip()) / len(bad.strip()) if bad.strip() else math.inf
    before, after = response[:start], response[end:]
    previous = words(before[-50:]); following = words(after[:50])
    first, last = (good_words[0] if good_words else None,
                   good_words[-1] if good_words else None)
    duplicate_boundary = bool((previous and first == previous[-1]) or
                              (following and last == following[0]))
    schema_style = bool(EXPLANATORY_PATTERNS["schema_boolean"].search(good) or
                        re.search(r"[{}]|[\"'][A-Za-z][^\n]{0,50}[\"']\s*:", good))
    bracket_or_ellipsis = bool("..." in good or re.search(r"\[[^]]+\]", good))
    bad_sentence = bool(re.search(r"[.!?][\"']?\s*$", bad.strip()))
    good_sentence = bool(re.search(r"[.!?][\"']?\s*$", good.strip()))
    sentence_shape_mismatch = bad_sentence != good_sentence and max(len(bad_words), len(good_words)) >= 5
    replacement = before + good + after
    malformed_join = bool(re.search(r"\w{2}[,.!?]\w", replacement[max(0, start-5):start+len(good)+5]))
    strict_surface_viable = bool(
        not flags and not schema_style and not bracket_or_ellipsis and
        0.40 <= word_ratio <= 2.50 and not duplicate_boundary and
        not sentence_shape_mismatch and not malformed_join and
        evidence["tier"] in {"exact_raw", "exact_normalized", "structured_exact", "lexical_strong"}
    )
    return {
        "bad_chars": len(bad.strip()), "good_chars": len(good.strip()),
        "bad_words": len(bad_words), "good_words": len(good_words),
        "good_bad_char_ratio": char_ratio, "good_bad_word_ratio": word_ratio,
        "explanatory_flags": flags, "schema_style": schema_style,
        "bracket_or_ellipsis": bracket_or_ellipsis,
        "duplicate_boundary_token": duplicate_boundary,
        "sentence_shape_mismatch": sentence_shape_mismatch,
        "malformed_join": malformed_join,
        "strict_surface_viable_proxy": strict_surface_viable,
        "replacement_context": replacement[max(0, start - 90):min(len(replacement), start + len(good) + 90)],
    }


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def describe(values) -> dict:
    values = [float(value) for value in values if math.isfinite(value)]
    return {"n": len(values), "min": min(values), "p10": percentile(values, .10),
            "p25": percentile(values, .25), "median": percentile(values, .50),
            "p75": percentile(values, .75), "p90": percentile(values, .90),
            "max": max(values), "mean": statistics.fmean(values)} if values else {"n": 0}


def load_records() -> tuple[list[dict], dict]:
    assert sha256(INPUT) == EXPECTED_INPUT_SHA256
    records, all_rows = [], 0
    all_label_ranges = defaultdict(list)
    raw_rows = []
    with INPUT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); all_rows += 1; raw_rows.append(row)
            assert row["partition"] == "auxiliary_candidate_fit"
            assert row["official_split"] == "train"
            for label_index, label in enumerate(row["labels"]):
                start, end = int(label["start"]), int(label["end"])
                assert 0 <= start < end <= len(row["original_response"])
                assert row["original_response"][start:end] == label["text"]
                all_label_ranges[row["response_id"]].append(
                    (start, end, label["label_type"], label_index))
                if label["label_type"] not in CONFLICT_TYPES:
                    continue
                parsed = parse_original(label.get("meta"))
                record = {
                    "span_id": f"{row['response_id']}:{start}:{end}:{label_index}",
                    "source_id": row["source_id"], "group_id": row["group_id"],
                    "response_id": row["response_id"], "prompt_sha256": row["prompt_sha256"],
                    "answer_sha256": row["answer_sha256"], "task_type": row["task_type"],
                    "label_type": label["label_type"], "start": start, "end": end,
                    "bad": label["text"], "meta": label.get("meta"),
                    "original_parse": parsed,
                    "response": row["original_response"], "source": row["retrieved_passages"],
                }
                if parsed["value"] is not None:
                    evidence = evidence_alignment(parsed["value"], row["retrieved_passages"], row["task_type"])
                    record["evidence"] = evidence
                    record["surface"] = surface_diagnostic(
                        row["original_response"], start, end, label["text"], parsed["value"], evidence)
                    record["edit"] = token_edit(label["text"], parsed["value"])
                records.append(record)
    assert all_rows == 9678 and len(records) == 4381

    # Local contamination/overlap is computed against every released label.
    by_response_conflict = defaultdict(list)
    for record in records:
        by_response_conflict[record["response_id"]].append(record)
        overlaps = [item for item in all_label_ranges[record["response_id"]]
                    if max(record["start"], item[0]) < min(record["end"], item[1]) and
                    not (item[0] == record["start"] and item[1] == record["end"] and
                         item[2] == record["label_type"])]
        record["overlaps_other_released_label"] = bool(overlaps)
        record["other_overlap_types"] = sorted({item[2] for item in overlaps})
    for response_records in by_response_conflict.values():
        for record in response_records:
            record["conflict_spans_in_response"] = len(response_records)
            record["all_released_labels_in_response"] = len(
                all_label_ranges[record["response_id"]])

    audit = {
        "input_rows": all_rows,
        "unique_sources": len({row["source_id"] for row in raw_rows}),
        "unique_groups": len({row["group_id"] for row in raw_rows}),
        "unique_responses": len({row["response_id"] for row in raw_rows}),
        "task_rows": dict(Counter(row["task_type"] for row in raw_rows)),
        "all_rows_official_train": all(row["official_split"] == "train" for row in raw_rows),
        "all_rows_candidate_fit_partition": all(row["partition"] == "auxiliary_candidate_fit" for row in raw_rows),
        "all_rows_not_currently_used_in_training": all(
            row.get("currently_used_in_training") is False for row in raw_rows),
        "all_rows_not_newly_generated": all(
            row.get("new_labels_generated") is False for row in raw_rows),
        "raw_identity_rows": [
            {
                "source_id": row["source_id"], "group_id": row["group_id"],
                "response_id": row["response_id"],
                "prompt_sha256": row["prompt_sha256"],
                "answer_sha256": row["answer_sha256"],
                "task_type": row["task_type"],
            }
            for row in raw_rows
        ],
    }
    return records, audit


def deterministic_sample(records: list[dict]) -> list[dict]:
    strata = defaultdict(list)
    for record in records:
        strata[(record["task_type"], record["label_type"])].append(record)
    expected = {("Summary", "Evident Conflict"), ("Summary", "Subtle Conflict"),
                ("Data2txt", "Evident Conflict"), ("Data2txt", "Subtle Conflict")}
    assert set(strata) == expected
    selected = []
    for stratum in sorted(strata):
        ranked = sorted(strata[stratum], key=lambda record: hashlib.sha256(
            f"{SAMPLE_SEED}|{record['span_id']}".encode()).hexdigest())
        selected.extend(ranked[:SAMPLE_PER_STRATUM])
    assert len(selected) == 100
    return selected


def write_worksheet(records: list[dict]) -> None:
    sample = deterministic_sample(records)
    with REVIEW.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(sample, 1):
            parsed = record["original_parse"]
            item = {
                "review_index": index,
                "span_id": record["span_id"],
                "task_type": record["task_type"], "label_type": record["label_type"],
                "bad": record["bad"], "original": parsed["value"],
                "parse_status": parsed["status"],
                "bad_context": record["response"][max(0, record["start"]-90):min(len(record["response"]), record["end"]+90)],
                "best_source_segment": record.get("evidence", {}).get("best_segment"),
                "automatic_evidence_tier": record.get("evidence", {}).get("tier"),
                "automatic_surface_viable": record.get("surface", {}).get("strict_surface_viable_proxy"),
                "manual_evidence": None,
                "manual_raw_replacement_grammar": None,
                "manual_pair_disposition": None,
                "manual_shortcut": None,
                "manual_note": None,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"WROTE_REVIEW_WORKSHEET {REVIEW} 100", flush=True)


def grouped_counts(items, key):
    return dict(sorted(Counter(key(item) for item in items).items(),
                       key=lambda pair: str(pair[0])))


def ratio_buckets(values: list[float]) -> dict:
    return {
        "lt_0_5": sum(value < 0.5 for value in values),
        "between_0_5_and_2_inclusive": sum(0.5 <= value <= 2.0 for value in values),
        "gt_2": sum(value > 2.0 for value in values),
        "gt_4": sum(value > 4.0 for value in values),
    }


def collision_summary(rows: list[dict], identity_key: str,
                      group_key: str = "group_id") -> dict:
    groups = defaultdict(set)
    row_count = Counter()
    for row in rows:
        identity = row[identity_key]
        groups[identity].add(row[group_key])
        row_count[identity] += 1
    cross = {identity: values for identity, values in groups.items()
             if len(values) > 1}
    return {
        "unique_identities": len(groups),
        "identities_in_multiple_groups": len(cross),
        "rows_covered_by_cross_group_identities": sum(row_count[key] for key in cross),
        "max_groups_per_identity": max((len(value) for value in groups.values()), default=0),
        "examples": [
            {"identity": key, "groups": sorted(value)}
            for key, value in sorted(cross.items(), key=lambda item: (-len(item[1]), str(item[0])))[:10]
        ],
    }


def duplicate_and_group_audit(records: list[dict], raw_rows: list[dict]) -> dict:
    parsed = [record for record in records if record["original_parse"]["value"] is not None]

    pair_members = defaultdict(list)
    prompt_bad_to_goods = defaultdict(set)
    location_members = defaultdict(list)
    response_spans = defaultdict(list)
    for record in parsed:
        good = record["original_parse"]["value"]
        pair_key = (normalized_words(record["bad"]), normalized_words(good))
        pair_members[pair_key].append(record)
        prompt_bad_to_goods[(record["prompt_sha256"], normalized_words(record["bad"]))].add(
            normalized_words(good))
    for record in records:
        location_members[(record["response_id"], record["start"], record["end"])].append(record)
        response_spans[record["response_id"]].append(record)

    repeated_pair_keys = {key: members for key, members in pair_members.items()
                          if len(members) > 1}
    cross_group_pair_keys = {
        key: members for key, members in repeated_pair_keys.items()
        if len({member["group_id"] for member in members}) > 1
    }
    ambiguous = {key: goods for key, goods in prompt_bad_to_goods.items()
                 if len(goods) > 1}
    duplicate_locations = {key: members for key, members in location_members.items()
                           if len(members) > 1}

    overlapping_conflict_pairs = 0
    responses_with_conflict_overlap = set()
    for response_id, members in response_spans.items():
        ordered = sorted(members, key=lambda item: (item["start"], item["end"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if right["start"] >= left["end"]:
                    break
                if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                    overlapping_conflict_pairs += 1
                    responses_with_conflict_overlap.add(response_id)

    group_sources, group_tasks, group_prompts = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in raw_rows:
        group_sources[row["group_id"]].add(row["source_id"])
        group_tasks[row["group_id"]].add(row["task_type"])
        group_prompts[row["group_id"]].add(row["prompt_sha256"])

    return {
        "unique_span_ids": len({record["span_id"] for record in records}),
        "normalized_bad_good": {
            "parsed_pairs": len(parsed),
            "unique_pair_keys": len(pair_members),
            "repeated_pair_keys": len(repeated_pair_keys),
            "duplicate_excess_rows": sum(len(value) - 1 for value in repeated_pair_keys.values()),
            "pair_keys_repeated_across_groups": len(cross_group_pair_keys),
            "rows_in_cross_group_repeated_pairs": sum(len(value) for value in cross_group_pair_keys.values()),
        },
        "same_prompt_and_bad_with_multiple_good_values": {
            "keys": len(ambiguous),
            "good_value_count_max": max((len(value) for value in ambiguous.values()), default=0),
            "examples": [
                {"prompt_sha256": key[0], "bad": key[1], "good_values": sorted(goods)}
                for key, goods in sorted(ambiguous.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
            ],
        },
        "exact_location_duplicates": {
            "locations": len(duplicate_locations),
            "rows": sum(len(value) for value in duplicate_locations.values()),
        },
        "overlap": {
            "conflict_spans_overlapping_any_other_released_label": sum(
                record["overlaps_other_released_label"] for record in records),
            "overlapping_conflict_span_pairs": overlapping_conflict_pairs,
            "responses_with_conflict_span_overlap": len(responses_with_conflict_overlap),
            "responses_with_multiple_conflict_spans": sum(
                len(value) > 1 for value in response_spans.values()),
            "max_conflict_spans_in_one_response": max(map(len, response_spans.values()), default=0),
            "conflict_spans_in_multi_conflict_responses": sum(
                len(value) for value in response_spans.values() if len(value) > 1),
            "conflict_spans_in_answers_with_multiple_released_labels": sum(
                record["all_released_labels_in_response"] > 1 for record in records),
        },
        "fit_internal_identity_collisions": {
            "answer_sha256": collision_summary(raw_rows, "answer_sha256"),
            "prompt_sha256": collision_summary(raw_rows, "prompt_sha256"),
            "source_id": collision_summary(raw_rows, "source_id"),
            "groups_with_multiple_sources": sum(len(value) > 1 for value in group_sources.values()),
            "groups_with_multiple_task_types": sum(len(value) > 1 for value in group_tasks.values()),
            "groups_with_multiple_prompt_hashes": sum(len(value) > 1 for value in group_prompts.values()),
        },
        "external_cal_test_identity_intersection": {
            "status": "not_assessed_by_design",
            "reason": "The audit was forbidden from reading calibration or test data.",
        },
    }


def automatic_statistics(records: list[dict]) -> dict:
    parsed = [record for record in records if record["original_parse"]["value"] is not None]
    strata = lambda record: f"{record['task_type']}|{record['label_type']}"
    parse_by_stratum = {}
    for key in sorted({strata(record) for record in records}):
        subset = [record for record in records if strata(record) == key]
        parse_by_stratum[key] = {
            "n": len(subset),
            "status": grouped_counts(subset, lambda record: record["original_parse"]["status"]),
            "one_strict_nonempty": sum(
                record["original_parse"]["status"] == "one_strict_nonempty" for record in subset),
        }

    evidence_counts = grouped_counts(parsed, lambda record: record["evidence"]["tier"])
    evidence_by_stratum = {}
    for key in sorted({strata(record) for record in parsed}):
        subset = [record for record in parsed if strata(record) == key]
        evidence_by_stratum[key] = grouped_counts(subset, lambda record: record["evidence"]["tier"])

    bad_words = [record["surface"]["bad_words"] for record in parsed]
    good_words = [record["surface"]["good_words"] for record in parsed]
    bad_chars = [record["surface"]["bad_chars"] for record in parsed]
    good_chars = [record["surface"]["good_chars"] for record in parsed]
    good_bad_word = [record["surface"]["good_bad_word_ratio"] for record in parsed]
    good_bad_char = [record["surface"]["good_bad_char_ratio"] for record in parsed]
    bad_good_word = [1 / value for value in good_bad_word if value > 0 and math.isfinite(value)]
    bad_good_char = [1 / value for value in good_bad_char if value > 0 and math.isfinite(value)]

    phrase_patterns = {
        "source": re.compile(r"\bsource\b", re.I),
        # Restrict "states" to reporting constructions so "United States" is
        # not miscounted as annotator discourse.
        "states_stated_says_discourse": re.compile(
            r"\b(?:(?:source|article|review|data|it)\s+)?"
            r"(?:states?|stated|says?|said)\s+(?:that\b|[\"'])", re.I),
        "not_mentioned": re.compile(r"\b(?:not|never|no)\s+(?:explicitly\s+)?mention(?:ed|s)?\b", re.I),
        "instead": re.compile(r"\binstead\b", re.I),
        "rather_than": re.compile(r"\brather\s+than\b", re.I),
        "according_to": re.compile(r"\baccording\s+to\b", re.I),
        "correct_should_be": re.compile(r"\b(?:the\s+correct|should\s+be)\b", re.I),
    }
    phrase_counts = {
        name: sum(bool(pattern.search(record["original_parse"]["value"])) for record in parsed)
        for name, pattern in phrase_patterns.items()
    }
    flag_counts = Counter()
    for record in parsed:
        flag_counts.update(record["surface"]["explanatory_flags"])

    strong_tiers = {"exact_raw", "exact_normalized", "structured_exact", "lexical_strong"}
    good_source_exact = lambda record: record["evidence"]["raw_exact"] or record["evidence"]["normalized_exact"]
    bad_evidence = {
        record["span_id"]: evidence_alignment(record["bad"], record["source"], record["task_type"])
        for record in parsed
    }
    shortcut = {
        "one_token_each_or_less": sum(record["edit"]["one_token_each_or_less"] for record in parsed),
        "two_tokens_each_or_less": sum(
            record["edit"]["deleted_words"] <= 2 and record["edit"]["inserted_words"] <= 2
            for record in parsed),
        "numeric_only_change": sum(record["edit"]["numeric_only_change"] for record in parsed),
        "boolean_only_change": sum(record["edit"]["boolean_only_change"] for record in parsed),
        "negation_or_direction_only_change": sum(
            record["edit"]["negation_or_direction_only_change"] for record in parsed),
        "good_exact_source_bad_not_exact_source": sum(
            good_source_exact(record) and not (
                bad_evidence[record["span_id"]]["raw_exact"] or
                bad_evidence[record["span_id"]]["normalized_exact"])
            for record in parsed),
        "bad_itself_exact_in_source": sum(
            bad_evidence[record["span_id"]]["raw_exact"] or
            bad_evidence[record["span_id"]]["normalized_exact"]
            for record in parsed),
        "explanatory_or_schema_style": sum(
            bool(record["surface"]["explanatory_flags"]) or record["surface"]["schema_style"]
            for record in parsed),
        "extreme_word_length_mismatch_lt_0_4_or_gt_2_5": sum(
            record["surface"]["good_bad_word_ratio"] < 0.4 or
            record["surface"]["good_bad_word_ratio"] > 2.5
            for record in parsed),
    }

    changed = [record for record in parsed
               if normalized_words(record["bad"]) != normalized_words(record["original_parse"]["value"])]
    prefilter = [
        record for record in changed
        if not record["original_parse"]["inline_annotator_marker_removed"]
        and record["evidence"]["tier"] in strong_tiers
        and record["surface"]["strict_surface_viable_proxy"]
        and not record["overlaps_other_released_label"]
        and not (bad_evidence[record["span_id"]]["raw_exact"] or
                 bad_evidence[record["span_id"]]["normalized_exact"])
    ]
    pair_key = lambda record: (
        normalized_words(record["bad"]),
        normalized_words(record["original_parse"]["value"]),
    )
    unique_prefilter = {}
    for record in sorted(prefilter, key=lambda item: item["span_id"]):
        unique_prefilter.setdefault(pair_key(record), record)
    answer_clean = [record for record in prefilter
                    if record["all_released_labels_in_response"] == 1]
    unique_answer_clean = {}
    for record in sorted(answer_clean, key=lambda item: item["span_id"]):
        unique_answer_clean.setdefault(pair_key(record), record)

    return {
        "conflict_spans": len(records),
        "stratum_counts": grouped_counts(records, strata),
        "parse": {
            "status": grouped_counts(records, lambda record: record["original_parse"]["status"]),
            "one_strict_nonempty_rate": len(parsed) / len(records),
            "inline_annotator_marker_removed": sum(
                record["original_parse"]["inline_annotator_marker_removed"] for record in records),
            "by_stratum": parse_by_stratum,
        },
        "evidence_alignment": {
            "tier": evidence_counts,
            "by_stratum": evidence_by_stratum,
            "strong_automatic_alignment": sum(
                record["evidence"]["tier"] in strong_tiers for record in parsed),
            "semantic_review_candidate_is_not_verified": True,
        },
        "length": {
            "bad_words": describe(bad_words), "good_words": describe(good_words),
            "bad_chars": describe(bad_chars), "good_chars": describe(good_chars),
            "good_over_bad_word_ratio": describe(good_bad_word),
            "bad_over_good_word_ratio": describe(bad_good_word),
            "good_over_bad_char_ratio": describe(good_bad_char),
            "bad_over_good_char_ratio": describe(bad_good_char),
            "good_over_bad_word_ratio_buckets": ratio_buckets(good_bad_word),
            "bad_over_good_word_ratio_buckets": ratio_buckets(bad_good_word),
            "good_over_bad_char_ratio_buckets": ratio_buckets(good_bad_char),
            "bad_over_good_char_ratio_buckets": ratio_buckets(bad_good_char),
        },
        "wording_and_style": {
            "explanatory_flag_counts": dict(sorted(flag_counts.items())),
            "any_explanatory_flag": sum(
                bool(record["surface"]["explanatory_flags"]) for record in parsed),
            "literal_phrase_counts": phrase_counts,
            "schema_style": sum(record["surface"]["schema_style"] for record in parsed),
            "bracket_or_ellipsis": sum(record["surface"]["bracket_or_ellipsis"] for record in parsed),
        },
        "shortcut_diagnostics": shortcut,
        "surface_proxy": {
            "strict_surface_viable": sum(
                record["surface"]["strict_surface_viable_proxy"] for record in parsed),
            "strict_surface_viable_rate_among_parsed": sum(
                record["surface"]["strict_surface_viable_proxy"] for record in parsed) / len(parsed),
            "warning": "A string heuristic cannot certify grammar, entailment, or non-equivalence.",
        },
        "high_precision_prefilter_not_gold": {
            "span_candidates_before_pair_dedup": len(prefilter),
            "unique_normalized_pair_candidates": len(unique_prefilter),
            "single_released_label_answer_candidates": len(answer_clean),
            "single_released_label_unique_pair_candidates": len(unique_answer_clean),
            "requires_human_verification": True,
        },
    }


def load_and_validate_manual(records: list[dict]) -> tuple[list[dict], dict]:
    assert REVIEW.exists(), f"Missing manual review: {REVIEW}"
    with REVIEW.open(encoding="utf-8") as handle:
        reviewed = [json.loads(line) for line in handle if line.strip()]
    assert len(reviewed) == 100
    expected = deterministic_sample(records)
    assert [row["span_id"] for row in reviewed] == [row["span_id"] for row in expected]
    allowed = {
        "manual_evidence": {"exact", "lexical", "semantic", "unsupported", "unparseable"},
        "manual_raw_replacement_grammar": {"yes", "no", "uncertain"},
        "manual_pair_disposition": {"direct", "needs_rewrite", "reject"},
        "manual_shortcut": {"none", "numeric_or_entity", "boolean_schema", "explanatory_or_length", "style_copy"},
    }
    for index, row in enumerate(reviewed, 1):
        assert row["review_index"] == index
        for field, values in allowed.items():
            assert row[field] in values, (index, field, row[field])
        assert isinstance(row.get("manual_note"), str) and row["manual_note"].strip()

    strata = lambda row: f"{row['task_type']}|{row['label_type']}"
    by_stratum = {}
    for key in sorted({strata(row) for row in reviewed}):
        subset = [row for row in reviewed if strata(row) == key]
        by_stratum[key] = {
            "n": len(subset),
            "evidence": grouped_counts(subset, lambda row: row["manual_evidence"]),
            "raw_replacement_grammar": grouped_counts(
                subset, lambda row: row["manual_raw_replacement_grammar"]),
            "pair_disposition": grouped_counts(subset, lambda row: row["manual_pair_disposition"]),
            "shortcut": grouped_counts(subset, lambda row: row["manual_shortcut"]),
        }

    population = Counter(f"{record['task_type']}|{record['label_type']}" for record in records)
    projected = {}
    for disposition in ("direct", "needs_rewrite", "reject"):
        rate = 0.0
        for key, population_n in population.items():
            subset = [row for row in reviewed if strata(row) == key]
            rate += (population_n / len(records)) * (
                sum(row["manual_pair_disposition"] == disposition for row in subset) / len(subset))
        projected[disposition] = rate

    automatic_lookup = {record["span_id"]: record for record in records}
    tp = fp = fn = tn = 0
    for row in reviewed:
        predicted = bool(automatic_lookup[row["span_id"]].get("surface", {}).get(
            "strict_surface_viable_proxy", False))
        actual = row["manual_pair_disposition"] == "direct"
        if predicted and actual: tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    disposition_by_shortcut = defaultdict(Counter)
    for row in reviewed:
        disposition_by_shortcut[row["manual_shortcut"]][row["manual_pair_disposition"]] += 1

    summary = {
        "sample_design": {
            "seed": SAMPLE_SEED, "selection": "smallest sha256(seed|span_id)",
            "per_stratum": SAMPLE_PER_STRATUM, "n": len(reviewed),
            "strata": dict(sorted(Counter(strata(row) for row in reviewed).items())),
            "deterministic_not_probability_sample": True,
        },
        "review_protocol": {
            "evidence": (
                "exact/lexical/semantic only when the proposed correction is supported "
                "by retrieved_passages; unsupported when contradicted or incomplete; "
                "unparseable when no single strict Original target exists."),
            "raw_replacement_grammar": (
                "yes only if substituting Original at the exact labeled character span "
                "produces grammatical, non-duplicative answer prose without rewriting."),
            "pair_disposition": (
                "direct requires both source support and grammatical complete correction; "
                "needs_rewrite preserves a useful factual hint but needs verbalization or "
                "context repair; reject covers missing, unsupported, equivalent, ambiguous, "
                "or incomplete corrections."),
            "shortcut": (
                "dominant visible cue among numeric/entity, boolean/schema, explanatory/length, "
                "style/source-copy, or none."),
        },
        "overall": {
            "evidence": grouped_counts(reviewed, lambda row: row["manual_evidence"]),
            "raw_replacement_grammar": grouped_counts(
                reviewed, lambda row: row["manual_raw_replacement_grammar"]),
            "pair_disposition": grouped_counts(reviewed, lambda row: row["manual_pair_disposition"]),
            "shortcut": grouped_counts(reviewed, lambda row: row["manual_shortcut"]),
        },
        "by_stratum": by_stratum,
        "population_weighted_descriptive_projection": {
            "population_stratum_counts": dict(sorted(population.items())),
            "pair_disposition_rates": projected,
            "warning": "Descriptive post-stratification only; deterministic review and rule judgments do not justify a confidence interval.",
        },
        "automatic_surface_proxy_vs_manual_direct": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
        },
        "pair_disposition_by_manual_shortcut": {
            key: dict(sorted(value.items())) for key, value in sorted(disposition_by_shortcut.items())
        },
        "direct_review_indices": [row["review_index"] for row in reviewed
                                  if row["manual_pair_disposition"] == "direct"],
        "all_100_review_records": reviewed,
    }
    return reviewed, summary


def report_text(result: dict) -> str:
    auto = result["automatic_full_corpus"]
    manual = result["manual_rule_review_100"]
    dup = result["duplicates_conflicts_and_group_risk"]
    parsed = auto["parse"]["status"].get("one_strict_nonempty", 0)
    direct = manual["overall"]["pair_disposition"].get("direct", 0)
    rewrite = manual["overall"]["pair_disposition"].get("needs_rewrite", 0)
    reject = manual["overall"]["pair_disposition"].get("reject", 0)
    projection = manual["population_weighted_descriptive_projection"]["pair_disposition_rates"]["direct"]
    length = auto["length"]
    wording = auto["wording_and_style"]
    shortcut = auto["shortcut_diagnostics"]
    prefilter = auto["high_precision_prefilter_not_gold"]
    evidence = auto["evidence_alignment"]["tier"]
    overlap = dup["overlap"]
    proxy = manual["automatic_surface_proxy_vs_manual_direct"]
    strata = manual["by_stratum"]

    return f"""# `Original:` 人标说明转纠错 pair 的独立可行性审计

## 结论

**不能把 `Original:` 批量当成可直接替换的金标准。** 它通常是“纠错线索”或“证据说明”，不是与错误片段语法同型的正确文本。全量 4,381 个 conflict span 中，严格解析到一个非空 `Original:` 的有 {parsed:,} 个（{parsed/4381:.1%}）；但固定分层抽审 100 条只有 {direct} 条可原样替换，{rewrite} 条需要重写，{reject} 条应排除。

由于四层各抽 25 条，而总体被 Data2txt Evident Conflict 主导，按总体层权重回代的“可直接替换”描述性比例只有 **{projection:.1%}**。这不是置信区间或全量金标准，只说明“解析成功率”严重高估“可用 pair 率”。

## 数据和边界

- 只读输入：`auxiliary_human_v1/candidate_fit.jsonl`，SHA256 `{result['provenance']['input_sha256']}`。
- 共 9,678 个 fit 回答、4,381 个 conflict span；分层为 Data2txt EC 3,680、Data2txt SC 58、Summary EC 572、Summary SC 71。
- 本审计只读上述 fit 文件和固定人工复核表；未读取 calibration/test，未训练，未调用模型/GPU，未改 baseline。
- 因禁止读取 calibration/test，本报告只能审计 fit 内部的重复和分组风险，不能声称已排除外部分割泄漏。

## 全量机械检查

### 1. 解析与证据对齐

严格解析状态：

```json
{json.dumps(auto['parse']['status'], ensure_ascii=False, sort_keys=True)}
```

对已解析项，证据对齐层级为：

```json
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}
```

`exact_*` 是文字级命中；`structured_exact` 是 Data2txt 键值命中；`lexical_strong` 是冻结词汇阈值；`semantic_review_candidate` 只是待人工复核，**不代表语义已被证明**。

### 2. bad/good 长度差和说明性措辞

- bad/good 词数比中位数：{length['bad_over_good_word_ratio']['median']:.3f}，P10–P90 为 {length['bad_over_good_word_ratio']['p10']:.3f}–{length['bad_over_good_word_ratio']['p90']:.3f}；good/bad 词数比中位数：{length['good_over_bad_word_ratio']['median']:.3f}。
- bad 词数中位数 {length['bad_words']['median']:.1f}，`Original:` 词数中位数 {length['good_words']['median']:.1f}。
- good/bad 词数比低于 0.5：{length['good_over_bad_word_ratio_buckets']['lt_0_5']:,}；高于 2：{length['good_over_bad_word_ratio_buckets']['gt_2']:,}；高于 4：{length['good_over_bad_word_ratio_buckets']['gt_4']:,}。
- 含任一冻结说明/样式标记：{wording['any_explanatory_flag']:,}；schema/布尔键值样式：{wording['schema_style']:,}。
- 字面措辞计数：`source` {wording['literal_phrase_counts']['source']:,}，报告式 `states/stated/says ... that/quote` {wording['literal_phrase_counts']['states_stated_says_discourse']:,}，`not mentioned` {wording['literal_phrase_counts']['not_mentioned']:,}，`instead` {wording['literal_phrase_counts']['instead']:,}，`rather than` {wording['literal_phrase_counts']['rather_than']:,}，`according to` {wording['literal_phrase_counts']['according_to']:,}。

这些信号会形成明显捷径：分类器可能学到“正确端更长、像引用、含 schema 键值或说明词”，而不是学到事实是否被证据支持。

### 3. 最小替换捷径

- bad/good 各至多改 1 个词：{shortcut['one_token_each_or_less']:,}；各至多改 2 个词：{shortcut['two_tokens_each_or_less']:,}。
- 纯数字变化：{shortcut['numeric_only_change']:,}；纯布尔变化：{shortcut['boolean_only_change']:,}；纯否定/方向词变化：{shortcut['negation_or_direction_only_change']:,}。
- good 可从 source 精确复制、bad 不能：{shortcut['good_exact_source_bad_not_exact_source']:,}。若直接造 pair，模型可用“哪边更像 source”取巧。
- bad 本身也能在 source 精确找到：{shortcut['bad_itself_exact_in_source']:,}。这些项可能依赖上下文、标注边界或语义判断，不能机械当成单片段反事实。

### 4. 语法代理并不可靠

冻结字符串代理把 {auto['surface_proxy']['strict_surface_viable']:,} 条判为“表面可替换”；在 100 条人工复核上，其识别真正 direct pair 的 TP/FP/FN/TN 为 {proxy['tp']}/{proxy['fp']}/{proxy['fn']}/{proxy['tn']}，precision={proxy['precision']:.3f}，recall={proxy['recall']:.3f}。因此它只能做候选预筛，不能自动产金标。

## 100 条固定分层人工规则复核

抽样规则：四个 task×conflict 层各取 `sha256(seed|span_id)` 最小的 25 条，合计 100 条；样本与决定逐条保存在 `MANUAL_REVIEW.jsonl`，并完整嵌入 `RESULTS.json`。

人工规则要求把 `Original:` 原样替换回精确 offset：100 条中只有 {manual['overall']['raw_replacement_grammar'].get('yes',0)} 条语法可接受、{manual['overall']['raw_replacement_grammar'].get('no',0)} 条不可接受、{manual['overall']['raw_replacement_grammar'].get('uncertain',0)} 条不确定；再排除语义等价、证据不足和不完整纠错后，只剩 {direct} 条 direct pair。

| 层 | direct | needs rewrite | reject |
|---|---:|---:|---:|
| Data2txt EC | {strata['Data2txt|Evident Conflict']['pair_disposition'].get('direct',0)} | {strata['Data2txt|Evident Conflict']['pair_disposition'].get('needs_rewrite',0)} | {strata['Data2txt|Evident Conflict']['pair_disposition'].get('reject',0)} |
| Data2txt SC | {strata['Data2txt|Subtle Conflict']['pair_disposition'].get('direct',0)} | {strata['Data2txt|Subtle Conflict']['pair_disposition'].get('needs_rewrite',0)} | {strata['Data2txt|Subtle Conflict']['pair_disposition'].get('reject',0)} |
| Summary EC | {strata['Summary|Evident Conflict']['pair_disposition'].get('direct',0)} | {strata['Summary|Evident Conflict']['pair_disposition'].get('needs_rewrite',0)} | {strata['Summary|Evident Conflict']['pair_disposition'].get('reject',0)} |
| Summary SC | {strata['Summary|Subtle Conflict']['pair_disposition'].get('direct',0)} | {strata['Summary|Subtle Conflict']['pair_disposition'].get('needs_rewrite',0)} | {strata['Summary|Subtle Conflict']['pair_disposition'].get('reject',0)} |

主要失败形态：

1. Data2txt 的 `Original:` 经常是 `\"OutdoorSeating\": false` 一类 schema 证据。事实可能对，但不能直接插进自然语言回答。
2. Summary 常给整句 source 引文，而错误标签只覆盖短片段；直接替换会破坏语法、重复主语或重复上下文。
3. 一些 SC 是近义改写或仍被 source 支持，例如 `high winds/fierce winds`、`unnamed/unidentified`，不构成稳定纠错 pair。
4. `source states`、`not mentioned`、`instead` 等是给标注者看的解释，进入目标端会泄露标签。

## 重复、冲突和分组风险

- 标准化 bad/good 重复键：{dup['normalized_bad_good']['repeated_pair_keys']:,}；重复冗余行：{dup['normalized_bad_good']['duplicate_excess_rows']:,}；跨 group 重复键：{dup['normalized_bad_good']['pair_keys_repeated_across_groups']:,}。
- 同一 prompt+bad 对应多个 good 的键：{dup['same_prompt_and_bad_with_multiple_good_values']['keys']:,}。
- 与其他 released label 重叠的 conflict span：{overlap['conflict_spans_overlapping_any_other_released_label']:,}；处在多 conflict 回答中的 span：{overlap['conflict_spans_in_multi_conflict_responses']:,}。
- 处在含多个 released label 回答中的 conflict span：{overlap['conflict_spans_in_answers_with_multiple_released_labels']:,}。只改一个 span 就把整答当“纠正后答案”，会留下未修错误。
- 任何训练/测试划分都必须按 `group_id`，并同时封锁 source/prompt/answer 身份；不能随机拆 pair。

## 严格纳入规则

只有同时满足以下条件，才进入**待复核候选池**：

1. 保留原始 offset、response/source/group 身份；offset 与标签文字逐字一致。
2. 恰有一个严格、非空的 `Original:`；排除缺失、多值、variant-only 和内嵌 `Generative/AIGC` 标记污染。
3. good 被 source 明确支持，且姓名、数字、布尔值、否定和方向都一致；纯“语义候选”必须人工复核。
4. good 与 bad 在句法角色、时态、单复数、大小写和标点上可替换；插入原位置后自然、无重复。
5. good 修正整个 conflict span，不删除其他受证据支持的信息；bad 若也被 source 支持则排除或升级复核。
6. 与其他标签不重叠。若做整答 pair，回答中的所有错误必须同时修正；否则排除整答训练。
7. Data2txt schema 值不得原样作为自然语言目标；若使用，必须先经过独立冻结的 verbalizer，再重新做证据和语法复核。
8. 排除含 `source states/not mentioned/instead/correct/should be` 等说明性语言、括号占位、ellipsis 或 reviewer 元话语的目标。
9. 排除近义等价、标注含混和 questionable SC；不能为了数量把它们当反事实。
10. 标准化去重；按 group/source/prompt/answer 联合隔离。数字、实体、布尔、否定等最小编辑另行分层报告，并控制两端长度/样式。

按冻结机械条件，全量只剩 {prefilter['span_candidates_before_pair_dedup']:,} 个 span 候选、去重后 {prefilter['unique_normalized_pair_candidates']:,} 个；若额外要求整答只有一个 released label，则为 {prefilter['single_released_label_answer_candidates']:,} 个、去重后 {prefilter['single_released_label_unique_pair_candidates']:,} 个。**这些仍不是金标，必须人工确认语义和语法。**

## 可复现性

运行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/research/human_original_pair_feasibility_v1/audit.py final
```

脚本 SHA256：`{result['provenance']['audit_script_sha256']}`；人工复核表 SHA256：`{result['provenance']['manual_review_sha256']}`。
"""


def finalize(records: list[dict], base_audit: dict) -> None:
    raw_rows = base_audit.pop("raw_identity_rows")
    _, manual = load_and_validate_manual(records)
    result = {
        "schema_version": "human-original-pair-feasibility-v1",
        "conclusion": (
            "Original annotations are correction/evidence hints, not mechanically safe "
            "replacement gold. Strict prefiltering plus human entailment and surface-form "
            "review is required."),
        "provenance": {
            "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
            "input_sha256": sha256(INPUT),
            "manual_review": str(REVIEW.relative_to(ROOT)).replace("\\", "/"),
            "manual_review_sha256": sha256(REVIEW),
            "audit_script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "audit_script_sha256": sha256(Path(__file__)),
            "files_read": [str(INPUT), str(REVIEW)],
            "calibration_or_test_files_read": [],
            "gpu_used": False, "model_called": False, "training_run": False,
            "baseline_read_or_modified": False,
        },
        "input_audit": base_audit,
        "automatic_full_corpus": automatic_statistics(records),
        "manual_rule_review_100": manual,
        "duplicates_conflicts_and_group_risk": duplicate_and_group_audit(records, raw_rows),
        "interpretation_limits": [
            "Automatic lexical tiers do not certify semantic entailment.",
            "The 100-item audit is deterministic and rule-based; weighted rates are descriptive.",
            "Calibration/test identity leakage was not assessed because those files were forbidden.",
            "The automatic high-precision pool is a review queue, not a gold dataset.",
        ],
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(result), encoding="utf-8")
    hash_manifest = {
        "RESULTS.json": sha256(RESULT), "REPORT.md": sha256(REPORT),
        "MANUAL_REVIEW.jsonl": sha256(REVIEW), "audit.py": sha256(Path(__file__)),
        "candidate_fit.jsonl": sha256(INPUT),
    }
    (HERE / "SHA256.json").write_text(
        json.dumps(hash_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "FINALIZED", "result": str(RESULT), "report": str(REPORT),
        "hash_manifest": str(HERE / "SHA256.json"),
        "strict_parsed": result["automatic_full_corpus"]["parse"]["status"].get("one_strict_nonempty", 0),
        "manual_disposition": result["manual_rule_review_100"]["overall"]["pair_disposition"],
        "weighted_direct_rate": result["manual_rule_review_100"]["population_weighted_descriptive_projection"]["pair_disposition_rates"]["direct"],
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sample", "final"))
    args = parser.parse_args()
    records, _ = load_records()
    if args.stage == "sample":
        write_worksheet(records)
    else:
        finalize(records, base_audit=_)


if __name__ == "__main__":
    main()
