"""CPU-only feasibility audit for expanding semantic source attribution to fit3680.

This audit deliberately opens fit-only files.  It never imports the live v1
extractor, never loads a model, and never opens calibration or test rows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIT = ROOT / "fit_expansion/data/fit.jsonl"
NEW_FIT = ROOT / "fit_expansion/data/new_fit.jsonl"
NEW_PLANS = ROOT / "fit_expansion/data/new_token_plans.jsonl"
TOKENS = ROOT / "fit_expansion/data/tokens_fit.jsonl"
ANSWERS = ROOT / "fit_expansion/data/answers_fit.jsonl"
WINDOWS = ROOT / "fit_expansion/data/windows_k4_fit.jsonl"
EXCLUDED_WINDOWS = ROOT / "fit_expansion/data/windows_excluded_fit.jsonl"
CLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
EXPANSION_PREP = ROOT / "fit_expansion/llama_features_v3/preparation.json"
NATIVE_PROGRESS = ROOT / "results/semantic_source_attribution_v1/progress.json"
NATIVE_STATUS = ROOT / "results/semantic_source_attribution_v1/status.json"

# Explicit allow-list: no calibration/test path is present.
INPUTS = (
    FIT, NEW_FIT, NEW_PLANS, TOKENS, ANSWERS, WINDOWS, EXCLUDED_WINDOWS,
    CLAIMS, EXPANSION_PREP,
)

WRAPPER_LEFT = "<s>[INST] "
WRAPPER_RIGHT = " [/INST] "
PASSAGE_HEADER = re.compile(r"(?im)^passage[ \t]+([123]):")
TERMINAL = frozenset(".!?。！？")
CLOSERS = frozenset("\"'”’)]}」』】》")
LIST_PREFIX = re.compile(r"^(?:\(?[0-9]{1,3}[.)]|[A-Za-z][.)])$")
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "oz.",
    "lb.", "lbs.", "no.", "fig.", "dept.", "inc.", "ltd.", "co.",
})


def lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sentence_spans(text: str):
    assert text and any(char.isalnum() for char in text)
    cuts, index, line_start = [], 0, 0
    while index < len(text):
        char = text[index]
        if char in "\r\n":
            if char == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            cuts.append(index + 1)
            line_start = index + 1
        elif char in TERMINAL:
            end = index + 1
            while end < len(text) and (text[end] in TERMINAL or text[end] in CLOSERS):
                end += 1
            prefix = text[line_start:index + 1].strip()
            followed_by_break = end == len(text) or text[end].isspace()
            match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
            final_token = match.group(0).lower() if match else ""
            dotted = bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", final_token))
            initial = bool(re.fullmatch(r"[A-Za-z]\.", final_token))
            protected = end < len(text) and (
                final_token in ABBREVIATIONS or dotted or initial
            )
            if followed_by_break and not LIST_PREFIX.fullmatch(prefix) and not protected:
                cuts.append(end)
                index = end - 1
        index += 1
    cuts.append(len(text))
    spans, start = [], 0
    for stop in sorted(set(cuts)):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and any(char.isalnum() for char in text[left:right]):
            spans.append((left, right))
        start = stop
    coverage = [0] * len(text)
    for left, right in spans:
        for position in range(left, right):
            coverage[position] += 1
    assert all((not char.isalnum()) or coverage[index] == 1
               for index, char in enumerate(text))
    assert max(coverage, default=0) <= 1
    return spans


def parse_passages(document: str):
    matches = list(PASSAGE_HEADER.finditer(document))
    assert [int(match.group(1)) for match in matches] == [1, 2, 3]
    passages, sentences = [], []
    for passage_index, match in enumerate(matches):
        body_start = match.end()
        body_end = (matches[passage_index + 1].start()
                    if passage_index + 1 < len(matches) else len(document))
        while body_start < body_end and document[body_start].isspace():
            body_start += 1
        while body_end > body_start and document[body_end - 1].isspace():
            body_end -= 1
        body = document[body_start:body_end]
        assert body and any(char.isalnum() for char in body)
        local = []
        for sentence_id, (left, right) in enumerate(sentence_spans(body)):
            row = {
                "passage_id": passage_index + 1,
                "sentence_id": sentence_id,
                "document_char_start": body_start + left,
                "document_char_end": body_start + right,
                "text": body[left:right],
                "text_sha256": text_sha(body[left:right]),
            }
            local.append(row)
            sentences.append(row)
        passages.append(local)
    return passages, sentences


def map_span(rendered: str, offsets, left: int, right: int):
    positions, lexical, boundary = [], [], []
    covered_nonspace = [False] * (right - left)
    covered_alnum = [False] * (right - left)
    for token_position, pair in enumerate(offsets):
        token_left, token_right = map(int, pair)
        begin, end = max(token_left, left), min(token_right, right)
        if begin >= end:
            continue
        positions.append(token_position)
        if token_left < left or token_right > right:
            boundary.append(token_position)
        is_lexical = False
        for character in range(begin, end):
            if not rendered[character].isspace():
                covered_nonspace[character - left] = True
            if rendered[character].isalnum():
                covered_alnum[character - left] = True
                is_lexical = True
        if is_lexical:
            lexical.append(token_position)
    assert positions and lexical
    assert all(rendered[left + index].isspace() or covered_nonspace[index]
               for index in range(right - left))
    assert all((not rendered[left + index].isalnum()) or covered_alnum[index]
               for index in range(right - left))
    return positions, lexical, boundary


def assign_tokens(text: str, offsets, claims):
    owners = [[] for _ in offsets]
    inverse = [[] for _ in claims]
    for token_index, pair in enumerate(offsets):
        left, right = map(int, pair)
        chars = [position for position in range(left, right)
                 if text[position].isalnum()]
        if not chars:
            continue
        for claim_index, claim in enumerate(claims):
            if any(claim["start"] <= position < claim["end"] for position in chars):
                owners[token_index].append(claim_index)
                inverse[claim_index].append(token_index)
        assert owners[token_index]
    lexical = [any(text[position].isalnum() for position in range(left, right))
               for left, right in offsets]
    assert list(map(bool, owners)) == lexical
    assert all(inverse)
    return owners, inverse


def summary(values):
    values = sorted(values)
    assert values
    def percentile(q):
        return values[round((len(values) - 1) * q)]
    return {
        "min": values[0], "median": statistics.median(values),
        "p90": percentile(.90), "p95": percentile(.95),
        "p99": percentile(.99), "max": values[-1],
        "mean": statistics.fmean(values),
    }


def expected_array_bytes(claims: int, sentences: int, answer_tokens: int) -> int:
    # Exact uncompressed bytes for every array emitted by v1 finish/extract_one.
    edges = claims * sentences
    return (
        edges * 1024 * 2 +                 # claim_sentence_mass float16
        (claims + 1) * 8 +                 # claim_indptr int64
        edges * 4 +                         # claim_sentence_indices int32
        claims * 32 * 4 * 3 +              # source / previous / other
        claims * 32 * 3 * 4 +              # passage relevance
        claims * 32 * sentences * 4 +      # sentence relevance
        claims * 32 * 15 * 4 * 2 +         # top indices + values
        claims * 4 * 2 +                   # claim IDs + microclaim indices
        sentences * (4 + 1 + 4 + 32) +     # sentence IDs / passage / hashes
        answer_tokens * 8                   # absolute answer positions
    )


def main():
    started = time.perf_counter()
    for path in INPUTS:
        assert path.is_file(), path
        lowered = str(path).lower()
        assert "calibration" not in lowered and "test" not in lowered

    fit_rows = list(lines(FIT))
    new_rows = list(lines(NEW_FIT))
    plans = list(lines(NEW_PLANS))
    token_rows = list(lines(TOKENS))
    answer_rows = list(lines(ANSWERS))
    claim_rows = list(lines(CLAIMS))
    assert len(fit_rows) == len(token_rows) == len(answer_rows) == 3680
    assert len(new_rows) == len(plans) == 3046

    fit = {row["response_id"]: row for row in fit_rows}
    new = {row["response_id"]: row for row in new_rows}
    plan = {row["response_id"]: row for row in plans}
    tokens = {row["response_id"]: row for row in token_rows}
    answers = {row["response_id"]: row for row in answer_rows}
    assert len(fit) == 3680 and len(new) == len(plan) == 3046
    assert set(new) == set(plan) < set(fit)
    retained = set(fit) - set(new)
    assert len(retained) == 634
    assert len({(row["source_id"], row["original_response"]) for row in fit_rows}) == 3680
    assert len({row["source_id"] for row in fit_rows}) == 634
    assert len({row["group_id"] for row in fit_rows}) == 615
    assert (ROOT / "fit_expansion/data/label_conflicts.jsonl").stat().st_size == 0

    # Source-sentence mapping for every additional response.
    source_identity = {}
    new_sentence_counts, new_source_token_counts = [], []
    sentence_boundary_crossing_instances = 0
    sentence_boundary_crossing_unique_tokens = 0
    mapped_sentence_instances = 0
    mapped_sentence_lexical_tokens = 0
    source_passage_sentence_counts = Counter()
    sequence_lengths = []
    query_work = 0
    for index, rid in enumerate(new):
        row, item = new[rid], plan[rid]
        assert item["partition"] == row["partition"] == "fit"
        assert item["official_split"] == row["official_split"] == "train"
        assert item["source_id"] == row["source_id"]
        assert item["group_id"] == row["group_id"]
        assert item["released_prompt"] == row["released_prompt"]
        assert item["original_response"] == row["original_response"]
        assert item["answer_sha256"] == row["answer_sha256"] == text_sha(row["original_response"])
        view = item["original"]
        rendered = WRAPPER_LEFT + item["released_prompt"] + WRAPPER_RIGHT + item["original_response"]
        assert text_sha(rendered) == view["rendered_text_sha256"]
        assert len(view["input_ids"]) == len(view["input_token_offsets"])
        assert len(view["answer_token_positions"]) == len(view["response_token_offsets"])
        assert len(view["answer_token_positions"]) == tokens[rid]["token_count"]
        assert view["response_token_offsets"] == tokens[rid]["response_token_offsets"]
        assert view["answer_token_ids"] == tokens[rid]["token_ids"]
        assert view["answer_token_positions"] == tokens[rid]["answer_token_positions"]
        ref_left, ref_right = map(int, view["rendered_reference_character_range"])
        document = rendered[ref_left:ref_right]
        assert document == row["retrieved_passages"]
        passages, sentences = parse_passages(document)
        identity = (row["question"], row["released_prompt"], document,
                    tuple((s["passage_id"], s["sentence_id"], s["text_sha256"])
                          for s in sentences))
        if row["source_id"] in source_identity:
            assert source_identity[row["source_id"]] == identity
        else:
            source_identity[row["source_id"]] = identity
        context = set(map(int, view["context_token_positions"]))
        all_source_positions = set()
        per_passage = Counter()
        for sentence in sentences:
            left = ref_left + sentence["document_char_start"]
            right = ref_left + sentence["document_char_end"]
            assert rendered[left:right] == sentence["text"]
            positions, lexical_positions, boundary = map_span(
                rendered, view["input_token_offsets"], left, right)
            assert set(positions) <= context
            assert max(positions) < view["answer_token_positions"][0]
            all_source_positions.update(positions)
            mapped_sentence_instances += 1
            mapped_sentence_lexical_tokens += len(lexical_positions)
            sentence_boundary_crossing_instances += int(bool(boundary))
            sentence_boundary_crossing_unique_tokens += len(set(boundary))
            per_passage[sentence["passage_id"]] += 1
        assert all(per_passage[passage_id] for passage_id in (1, 2, 3))
        source_passage_sentence_counts.update(per_passage)
        new_sentence_counts.append(len(sentences))
        new_source_token_counts.append(len(all_source_positions))
        sequence_lengths.append(len(view["input_ids"]))
        query_work += len(view["input_ids"]) * sum(map(bool, tokens[rid]["lexical_mask"]))
        if (index + 1) % 500 == 0:
            print("SOURCE_MAPPING", index + 1, len(new), flush=True)
    assert len(source_identity) == 634

    # Fit-only claim geometry and token ownership for both retained and appended rows.
    claims_by_response = defaultdict(list)
    for claim in claim_rows:
        assert claim["partition"] == "fit"
        claims_by_response[claim["response_id"]].append(claim)
    assert set(claims_by_response) == set(fit)
    claim_stats = {
        "combined": Counter(), "new": Counter(), "retained": Counter(),
    }
    claim_counts_new, claim_counts_combined = [], []
    storage_new = storage_combined = 0
    edge_new = edge_combined = 0
    source_sentences_by_source = {
        source_id: len(identity[3]) for source_id, identity in source_identity.items()
    }
    for rid, row in fit.items():
        group = "new" if rid in new else "retained"
        raw_claims = sorted(claims_by_response[rid], key=lambda value: value["microclaim_index"])
        assert [value["microclaim_index"] for value in raw_claims] == list(range(len(raw_claims)))
        for claim in raw_claims:
            left, right = int(claim["start"]), int(claim["end"])
            assert row["original_response"][left:right] == claim["text"]
            assert claim["source_id"] == row["source_id"] and claim["group_id"] == row["group_id"]
        scored = [value for value in raw_claims if any(char.isalnum() for char in value["text"])]
        owners, inverse = assign_tokens(row["original_response"],
                                        tokens[rid]["response_token_offsets"], scored)
        lexical_token_count = sum(map(bool, tokens[rid]["lexical_mask"]))
        risk_token_count = sum(map(bool, tokens[rid]["risk_mask"]))
        assert sum(bool(value) for value in owners) == lexical_token_count
        risk = list(map(bool, tokens[rid]["risk_mask"]))
        positive = sum(any(risk[token_index] for token_index in owned) for owned in inverse)
        partial = sum(any(risk[token_index] for token_index in owned)
                      and not all(risk[token_index] for token_index in owned)
                      for owned in inverse)
        token_edges = sum(map(len, inverse))
        values = {
            "answers": 1, "raw_claims": len(raw_claims), "scored_claims": len(scored),
            "discarded_nonlexical_claims": len(raw_claims) - len(scored),
            "positive_claims": positive, "partial_positive_claims": partial,
            "claim_token_edges": token_edges,
            "lexical_tokens": lexical_token_count,
            "risk_tokens": risk_token_count,
            "risk_answers": int(tokens[rid]["answer_risk"]),
        }
        for scope in ("combined", group):
            claim_stats[scope].update(values)
        sentence_count = source_sentences_by_source[row["source_id"]]
        edges = len(scored) * sentence_count
        storage = expected_array_bytes(len(scored), sentence_count, tokens[rid]["token_count"])
        edge_combined += edges
        storage_combined += storage
        claim_counts_combined.append(len(scored))
        if group == "new":
            edge_new += edges
            storage_new += storage
            claim_counts_new.append(len(scored))
            item = plan[rid]["original"]
            for claim, local_indices in zip(scored, inverse):
                assert local_indices
                absolute = [item["answer_token_positions"][token_index]
                            for token_index in local_indices]
                assert absolute == sorted(set(absolute))
                assert max(absolute) < len(item["input_ids"])
                for token_index in local_indices:
                    token_left, token_right = tokens[rid]["response_token_offsets"][token_index]
                    begin, end = max(token_left, claim["start"]), min(token_right, claim["end"])
                    assert begin < end
                    assert any(char.isalnum() for char in row["original_response"][begin:end])
    assert claim_stats["combined"]["raw_claims"] == 34941
    assert claim_stats["combined"]["scored_claims"] == 34919
    assert claim_stats["combined"]["lexical_tokens"] == 560300

    # Verify every frozen fit window against unchanged BPE coordinates and risk masks.
    window_stats = {"combined": Counter(), "new": Counter(), "retained": Counter()}
    per_answer_windows = Counter()
    per_answer_positive_windows = Counter()
    for index, window in enumerate(lines(WINDOWS)):
        rid = window["response_id"]
        scope = "new" if rid in new else "retained"
        token = tokens[rid]
        indices = list(map(int, window["token_indices"]))
        assert window["partition"] == "fit" and window["eligible"] is True
        assert window["k"] == 4 and window["stride"] == 1
        assert indices == list(range(window["token_start"], window["token_end"]))
        assert 1 <= len(indices) <= 4
        assert window["answer_token_positions"] == [token["answer_token_positions"][i] for i in indices]
        assert window["token_ids"] == [token["token_ids"][i] for i in indices]
        assert window["lexical_token_indices"] == [i for i in indices if token["lexical_mask"][i]]
        expected_label = int(any(token["risk_mask"][i] for i in indices))
        assert window["label"] == expected_label
        per_answer_windows[rid] += 1
        per_answer_positive_windows[rid] += expected_label
        for target in ("combined", scope):
            window_stats[target]["eligible_windows"] += 1
            window_stats[target]["positive_windows"] += expected_label
        if (index + 1) % 100000 == 0:
            print("WINDOW_MAPPING", index + 1, flush=True)
    for excluded in lines(EXCLUDED_WINDOWS):
        rid = excluded["response_id"]
        scope = "new" if rid in new else "retained"
        token = tokens[rid]
        indices = list(map(int, excluded["token_indices"]))
        assert excluded["eligible"] is False and excluded["label"] is None
        assert excluded["exclusion_reason"] == "no_lexical_token"
        assert not any(token["lexical_mask"][i] for i in indices)
        for target in ("combined", scope):
            window_stats[target]["excluded_nonlexical_windows"] += 1
    for rid, answer in answers.items():
        assert per_answer_windows[rid] == answer["eligible_window_count"]
        assert per_answer_positive_windows[rid] == answer["positive_window_count"]

    expansion_prep = read(EXPANSION_PREP)
    assert expansion_prep["new_response_rows"] == 3046
    native_observation = None
    if NATIVE_PROGRESS.is_file():
        progress = read(NATIVE_PROGRESS)
        if progress.get("status") == "running" and progress.get("elapsed_seconds", 0) > 0:
            projected_native_minutes = (
                progress["elapsed_seconds"]
                / max(1, progress["completed_this_invocation"])
                * progress["scheduled_this_invocation"] / 60
            )
            ratio = float(expansion_prep["attention_query_work_ratio"])
            native_observation = {
                "snapshot_file": str(NATIVE_PROGRESS.relative_to(ROOT)),
                "snapshot": progress,
                "projected_native_793_minutes_from_live_partial_run": projected_native_minutes,
                "frozen_new_to_native_query_work_ratio": ratio,
                "projected_new3046_minutes_point": projected_native_minutes * ratio,
                "planning_range_minutes": [
                    projected_native_minutes * ratio * .9,
                    projected_native_minutes * ratio * 1.35,
                ],
                "status": "provisional; another authorized native extraction was already running; this audit did not start or control it",
            }

    report = {
        "version": "semantic-source-attribution-expansion-feasibility-v1",
        "status": "feasible_with_fit_only_conversion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "new_answers": len(new), "retained_native_fit_answers": len(retained),
            "combined_deduplicated_fit_answers": len(fit),
            "sources": len(source_identity), "source_connected_groups": 615,
            "calibration_rows_opened": 0, "official_test_rows_opened": 0,
            "model_loaded": False, "GPU_started_by_this_audit": False,
        },
        "identity_and_deduplication": {
            "unique_response_ids": len(fit),
            "unique_source_exact_answer_pairs": len({
                (row["source_id"], row["original_response"]) for row in fit_rows
            }),
            "new_ids_disjoint_from_retained_ids": not bool(set(new) & retained),
            "label_conflict_rows": 0,
            "all_634_sources_have_new_answers_and_stable_prompt_source_sentence_geometry": True,
        },
        "source_sentence_mapping_new3046": {
            "response_sentence_instances": mapped_sentence_instances,
            "unique_source_sentences": sum(source_sentences_by_source.values()),
            "sentence_count_per_answer": summary(new_sentence_counts),
            "source_union_token_count_per_answer": summary(new_source_token_counts),
            "mapped_sentence_lexical_token_instances": mapped_sentence_lexical_tokens,
            "sentence_spans_with_boundary_crossing_BPE": sentence_boundary_crossing_instances,
            "boundary_crossing_BPE_instances": sentence_boundary_crossing_unique_tokens,
            "passage_sentence_instances": {str(key): value for key, value in sorted(source_passage_sentence_counts.items())},
            "all_nonwhitespace_characters_covered": True,
            "all_alphanumeric_characters_covered": True,
            "all_source_tokens_precede_answer_and_belong_to_frozen_context": True,
            "exact_source_attention_query_work_sum_sequence_length_x_lexical_queries": query_work,
        },
        "microclaim_mapping": {
            scope: dict(values) for scope, values in claim_stats.items()
        } | {
            "new_scored_claims_per_answer": summary(claim_counts_new),
            "combined_scored_claims_per_answer": summary(claim_counts_combined),
            "all_claim_text_spans_exact": True,
            "all_new_claim_lexical_BPEs_mapped_to_absolute_replay_positions": True,
            "all_combined_lexical_BPEs_owned_by_at_least_one_claim": True,
        },
        "four_bpe_window_gold": {
            scope: dict(values) for scope, values in window_stats.items()
        } | {
            "k": 4, "stride": 1,
            "eligible_definition_verified": "contains at least one lexical BPE",
            "positive_definition_verified": "OR of unchanged fit risk mask over the raw-BPE window",
            "all_window_token_ids_positions_and_labels_recomputed_exact": True,
        },
        "claim_sentence_edges_and_storage": {
            "new3046_scored_claim_sentence_edges": edge_new,
            "combined_fit3680_scored_claim_sentence_edges": edge_combined,
            "new3046_exact_uncompressed_array_bytes": storage_new,
            "new3046_exact_uncompressed_array_GiB": storage_new / 2**30,
            "combined_fit3680_exact_uncompressed_array_bytes": storage_combined,
            "combined_fit3680_exact_uncompressed_array_GiB": storage_combined / 2**30,
            "main_CSR_tensor_bytes_per_edge": 2048,
            "recommended_free_space_GiB_for_new_cache": max(2.0, storage_new / 2**30 * 1.75),
            "storage_note": "Exact sum of uncompressed v1 NPZ arrays; compressed NPZ size is data dependent and sidecar/manifest overhead is small.",
        },
        "gpu_time_planning": {
            "frozen_expansion_query_work_ratio_vs_native793": expansion_prep["attention_query_work_ratio"],
            "prior_frozen_llama_expansion_range_minutes": expansion_prep["estimated_GPU_minutes_range"],
            "live_native_attribution_observation": native_observation,
            "recommended_reservation_minutes": [50, 70],
            "caveat": "Estimate only. No GPU job was launched; final timing should use completed native v1 elapsed time times the frozen query-work ratio.",
        },
        "required_conversion": {
            "needed": True,
            "reason": "No label-blind semantic-attribution layout currently joins the 3,046 replay plans to the fit3680 atomic claims and exact source-sentence token masks.",
            "inputs": [str(path.relative_to(ROOT)) for path in (
                NEW_PLANS, NEW_FIT, CLAIMS, TOKENS
            )],
            "steps": [
                "Filter microclaims_fit3680 by the frozen 3,046 response IDs and discard their 16 punctuation-only records (22 across the full fit3,680 pool).",
                "Parse each exact retrieved_passages block with the frozen sentence splitter and shift document spans by rendered_reference_character_range.",
                "Intersect source sentence spans and claim spans with frozen full-string/response BPE offsets; write label-blind layouts plus a global CSR index.",
                "Run the unchanged semantic attribution extractor only on these 3,046 layouts, then concatenate/reuse the existing native 634 fit caches for training.",
                "Open tokens_fit/windows_k4_fit only in the later scoring stage to attach fit labels and project scores to the unchanged four-BPE windows.",
            ],
            "do_not_use_as_extractor_input": "results/atomic_microclaim_relation_expanded_v4/examples.jsonl, because it mixes fit with calibration and contains gold-bearing fields.",
            "old_native_cache_policy": "Reuse the 634 native fit attribution caches; do not recompute or overwrite them.",
            "method_limit": "The 3,046 answers were generated by five other models but are teacher-forced through the common Llama-2-7B checkpoint, so their features are replay representations rather than native generator states.",
        },
        "files_sha256": {str(path.relative_to(ROOT)): sha(path) for path in INPUTS},
        "audit": {
            "elapsed_cpu_seconds": time.perf_counter() - started,
            "script_sha256_before_report_write": sha(Path(__file__)),
            "assertions_passed": True,
        },
    }
    (HERE / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "new_claims": claim_stats["new"]["scored_claims"],
        "new_edges": edge_new,
        "new_uncompressed_GiB": storage_new / 2**30,
        "elapsed_cpu_seconds": report["audit"]["elapsed_cpu_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
