"""Independent CPU-only audit for the auxiliary exact-span conflict pilot.

This file deliberately imports neither the preparation code nor the GPU runner.
It reconstructs every token target from released character spans and the stored
token offsets, and reads only the auxiliary candidate-fit files plus the exact
QA fit prefixes used by the pilot.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/auxiliary_conflict_token_pilot_v1"
AUX = ROOT / "auxiliary_human_v1"
QA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
RUNNER = HERE / "run_auxiliary_conflict_token_pilot_v1.py"

N_AUX = 9_678
N_QA_FIT = 34_919
N_QA_ANSWERS = 3_680
N_CONFLICT_SPANS = 4_381
QA_FOLD = 0
WINDOW_K = 4
KNOWN_TYPES = frozenset((
    "Evident Conflict", "Subtle Conflict",
    "Evident Baseless Info", "Subtle Baseless Info",
))
CONFLICT_TYPES = frozenset(("Evident Conflict", "Subtle Conflict"))
BASELESS_TYPES = KNOWN_TYPES - CONFLICT_TYPES
STAGE_CODE = {"aux": 0, "qa_train": 1, "qa_held": 2}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def label_tuple(label):
    return (
        int(label["span_index"]), label["label_type"], int(label["start"]), int(label["end"]),
        label["text_sha256"], label.get("meta_sha256"), bool(label.get("implicit_true", False)),
        bool(label.get("due_to_null", False)),
    )


def source_label_tuple(index, label, response):
    left, right = int(label["start"]), int(label["end"])
    assert 0 <= left < right <= len(response)
    text = response[left:right]
    if "text" in label:
        assert label["text"] == text
    return (
        index, label["label_type"], left, right, digest(text),
        digest(label.get("meta", "")), bool(label.get("implicit_true", False)),
        bool(label.get("due_to_null", False)),
    )


def overlap(left, right, other_left, other_right):
    return max(left, other_left) < min(right, other_right)


def classify_from_source(text, base_start, labels):
    conflict = np.zeros(len(text), dtype=bool)
    baseless = np.zeros(len(text), dtype=bool)
    for label in labels:
        assert label["label_type"] in KNOWN_TYPES
        left = max(base_start, int(label["start"])) - base_start
        right = min(base_start + len(text), int(label["end"])) - base_start
        if left >= right:
            continue
        destination = conflict if label["label_type"] in CONFLICT_TYPES else baseless
        destination[left:right] |= np.fromiter((c.isalnum() for c in text[left:right]), dtype=bool)
    c, b = int(conflict.sum()), int(baseless.sum())
    kind = "mixed" if c and b else "conflict" if c else "baseless" if b else "safe"
    return kind, conflict, baseless


def audit_sources_and_metadata(metadata, traces):
    aux_meta_by_response = defaultdict(list)
    qa_meta_by_example = {}
    for row in metadata:
        if row["origin"] == "auxiliary_human_v1":
            aux_meta_by_response[str(row["response_id"])].append(row)
        else:
            assert row["origin"] == "expanded_v4_fit"
            index = int(row["example_index"])
            assert index not in qa_meta_by_example
            qa_meta_by_example[index] = row

    trace_by_response = defaultdict(list)
    for row in traces:
        trace_by_response[str(row["response_id"])].append(row)

    source_text = {}
    expected_trace = Counter()
    aux_seen = 0
    aux_groups = set()
    aux_source_ids = set()
    with (AUX / "candidate_fit.jsonl").open(encoding="utf-8") as handle:
        for source_index in range(N_AUX):
            line = handle.readline()
            assert line, ("short auxiliary file", source_index)
            source = json.loads(line)
            aux_seen += 1
            assert source["partition"] == "auxiliary_candidate_fit"
            assert source["official_split"] == "train"
            assert source["quality"] == "good" and not source["new_labels_generated"]
            response_id = str(source["response_id"])
            response = source["original_response"]
            assert digest(response) == source["answer_sha256"]
            aux_groups.add(source["group_id"]); aux_source_ids.add(str(source["source_id"]))
            if response_id in aux_meta_by_response:
                source_text[("auxiliary_human_v1", response_id)] = response
                for row in aux_meta_by_response[response_id]:
                    left, right = int(row["start"]), int(row["end"])
                    assert response[left:right] == row["text"]
                    assert row["group_id"] == source["group_id"]
                    assert str(row["source_id"]) == str(source["source_id"])
                    assert row["task_type"] == source["task_type"]
                    assert row["answer_sha256"] == source["answer_sha256"]
                    assert row["hypothesis_sha256"] == digest(row["text"])
                    expected = [source_label_tuple(i, label, response)
                                for i, label in enumerate(source["labels"])
                                if overlap(left, right, int(label["start"]), int(label["end"]))]
                    assert Counter(map(label_tuple, row["trace_labels"])) == Counter(expected)
                    kind, _conflict, _baseless = classify_from_source(
                        row["text"], left, row["trace_labels"])
                    assert kind == row["kind"]
            for span_index, label in enumerate(source["labels"]):
                assert label["label_type"] in KNOWN_TYPES
                if label["label_type"] not in CONFLICT_TYPES:
                    continue
                left, right = int(label["start"]), int(label["end"])
                text = response[left:right]
                assert label["text"] == text
                expected_trace[(
                    str(source["source_id"]), source["group_id"], response_id, span_index,
                    label["label_type"], left, right, text, digest(text), label.get("meta", ""),
                    digest(label.get("meta", "")), sum(c.isalnum() for c in text),
                )] += 1
    assert aux_seen == N_AUX
    assert len(aux_groups) == 1_558 and len(aux_source_ids) == 1_614

    actual_trace = Counter()
    selected_aux_by_id = {row["microclaim_id"]: row for row in metadata if row["stage"] == "aux"}
    selected_trace_ids = set()
    for row in traces:
        key = (
            str(row["source_id"]), row["group_id"], str(row["response_id"]), int(row["span_index"]),
            row["label_type"], int(row["start"]), int(row["end"]), row["text"], row["text_sha256"],
            row["meta"], row["meta_sha256"], int(row["alnum_chars"]),
        )
        actual_trace[key] += 1
        assert row["overlapping_microclaim_ids"]
        assert len(row["overlapping_microclaim_ids"]) == len(set(row["overlapping_microclaim_ids"]))
        assert set(row["selected_microclaim_ids"]).issubset(row["overlapping_microclaim_ids"])
        for microclaim_id in row["selected_microclaim_ids"]:
            selected_trace_ids.add(microclaim_id)
            selected = selected_aux_by_id[microclaim_id]
            assert selected["kind"] == "conflict"
            assert str(selected["response_id"]) == str(row["response_id"])
            assert overlap(int(selected["start"]), int(selected["end"]), int(row["start"]), int(row["end"]))
            assert any(int(label["span_index"]) == int(row["span_index"])
                       for label in selected["trace_labels"])
    assert actual_trace == expected_trace
    assert selected_trace_ids == {row["microclaim_id"] for row in metadata
                                  if row["stage"] == "aux" and row["kind"] == "conflict"}

    qa_answers = []
    qa_answer_by_id = {}
    with (QA / "answers.jsonl").open(encoding="utf-8") as handle:
        for response_index in range(N_QA_ANSWERS):
            line = handle.readline()
            assert line, ("short QA answer fit prefix", response_index)
            answer = json.loads(line)
            assert answer["partition"] == "fit" and int(answer["response_index"]) == response_index
            response_id = str(answer["response_id"])
            assert response_id not in qa_answer_by_id
            assert digest(answer["original_response"]) == answer["answer_sha256"]
            qa_answers.append(answer); qa_answer_by_id[response_id] = answer

    qa_group_folds = defaultdict(set)
    selected_qa_seen = set()
    held_response_spans = defaultdict(dict)
    with (QA / "examples.jsonl").open(encoding="utf-8") as handle:
        for example_index in range(N_QA_FIT):
            line = handle.readline()
            assert line, ("short QA example fit prefix", example_index)
            source = json.loads(line)
            assert source["partition"] == "fit" and int(source["example_index"]) == example_index
            qa_group_folds[source["group_id"]].add(int(source["held_fold"]))
            if int(source["held_fold"]) == QA_FOLD:
                held_bucket = held_response_spans[str(source["response_id"])]
                for label in source["overlapping_gold_spans"]:
                    key = (label["label_type"], int(label["start"]), int(label["end"]))
                    held_bucket[key] = True
            row = qa_meta_by_example.get(example_index)
            if row is None:
                continue
            selected_qa_seen.add(example_index)
            expected_stage = "qa_held" if int(source["held_fold"]) == QA_FOLD else "qa_train"
            assert row["stage"] == expected_stage
            assert row["group_id"] == source["group_id"]
            assert str(row["source_id"]) == str(source["source_id"])
            assert str(row["response_id"]) == str(source["response_id"])
            assert int(row["held_fold"]) == int(source["held_fold"])
            assert int(row["start"]) == int(source["char_start"])
            assert int(row["end"]) == int(source["char_end"])
            assert row["text"] == source["claim_text"]
            assert row["hypothesis_sha256"] == digest(source["claim_text"])
            assert row["premise_sha256"] == digest(source["premise"])
            answer = qa_answer_by_id[str(source["response_id"])]
            response = answer["original_response"]
            assert response[int(row["start"]):int(row["end"])] == row["text"]
            assert row["answer_sha256"] == answer["answer_sha256"]
            source_text[("expanded_v4_fit", str(row["response_id"]))] = response
            expected = []
            for label in source["overlapping_gold_spans"]:
                left, right = int(label["start"]), int(label["end"])
                expected.append((int(label["span_index"]), label["label_type"], left, right,
                                 digest(response[left:right]), None,
                                 bool(label["implicit_true"]), bool(label["due_to_null"])))
            assert Counter(map(label_tuple, row["trace_labels"])) == Counter(expected)
            kind, _conflict, _baseless = classify_from_source(
                row["text"], int(row["start"]), row["trace_labels"])
            assert kind == row["kind"]
    assert selected_qa_seen == set(qa_meta_by_example)
    assert all(len(folds) == 1 for folds in qa_group_folds.values())

    held_groups = {row["group_id"] for row in metadata if row["stage"] == "qa_held"}
    train_groups = {row["group_id"] for row in metadata if row["stage"] == "qa_train"}
    assert held_groups.isdisjoint(train_groups)
    assert len(held_groups) == 122 and len(train_groups) == 142

    return {
        "source_text": source_text,
        "qa_answers": qa_answers,
        "qa_answer_by_id": qa_answer_by_id,
        "held_response_spans": held_response_spans,
        "stats": {
            "auxiliary_rows_read": aux_seen,
            "qa_fit_examples_read": N_QA_FIT,
            "qa_fit_answers_read": N_QA_ANSWERS,
            "calibration_rows_read": 0,
            "test_rows_read": 0,
            "released_conflict_spans_exactly_traced": sum(actual_trace.values()),
            "traced_spans_with_selected_clean_conflict": sum(bool(row["selected_microclaim_ids"])
                                                               for row in traces),
            "qa_all_fit_groups_single_fold": len(qa_group_folds),
            "qa_train_held_group_intersection": 0,
        },
    }


def audit_arrays(metadata, arrays, source_text):
    n_examples = len(metadata)
    flat = len(arrays["input_ids"])
    indptr = arrays["indptr"]
    assert len(indptr) == n_examples + 1 and int(indptr[0]) == 0 and int(indptr[-1]) == flat
    assert np.all(indptr[1:] > indptr[:-1])
    for name in ("targets", "char_counts", "sequence_code", "hypothesis_offset_start",
                 "hypothesis_offset_end"):
        assert len(arrays[name]) == flat
    assert len(arrays["stage_code"]) == len(arrays["example_weight"]) == n_examples
    assert np.isfinite(arrays["targets"]).all() and np.isfinite(arrays["example_weight"]).all()

    positive_tokens = soft_boundary_tokens = ignored_baseless_tokens = 0
    reconstructed_tokens = reconstructed_char_mass = 0
    conflict_claims_with_local_safe_tokens = 0
    fully_span_derived_claims = 0
    stage_supervised = Counter()
    stage_positive_mass = Counter()

    for index, row in enumerate(metadata):
        assert int(row["input_index"]) == index
        left, right = int(indptr[index]), int(indptr[index + 1])
        assert (left, right) == (int(row["flat_start"]), int(row["flat_end"]))
        assert right - left == int(row["input_tokens"])
        assert int(arrays["stage_code"][index]) == STAGE_CODE[row["stage"]]

        text = row["text"]
        assert digest(text) == row["hypothesis_sha256"]
        response = source_text[(row["origin"], str(row["response_id"]))]
        assert response[int(row["start"]):int(row["end"])] == text
        for label in row["trace_labels"]:
            a, b = int(label["start"]), int(label["end"])
            assert digest(response[a:b]) == label["text_sha256"]

        kind, conflict, baseless = classify_from_source(text, int(row["start"]), row["trace_labels"])
        assert kind == row["kind"]
        alnum = np.fromiter((c.isalnum() for c in text), dtype=bool)
        prefix_a = np.r_[0, np.cumsum(alnum, dtype=np.int64)]
        prefix_c = np.r_[0, np.cumsum(conflict, dtype=np.int64)]
        prefix_b = np.r_[0, np.cumsum(baseless, dtype=np.int64)]

        seq = arrays["sequence_code"][left:right]
        starts = arrays["hypothesis_offset_start"][left:right]
        ends = arrays["hypothesis_offset_end"][left:right]
        actual_target = arrays["targets"][left:right]
        actual_count = arrays["char_counts"][left:right]
        expected_target = np.full(right - left, -1.0, dtype=np.float64)
        expected_count = np.zeros(right - left, dtype=np.int64)
        assert set(map(int, np.unique(seq))).issubset({-1, 0, 1})
        non_hyp = seq != 1
        assert np.all(starts[non_hyp] == -1) and np.all(ends[non_hyp] == -1)
        hyp = np.flatnonzero(seq == 1)
        assert np.all(starts[hyp] >= 0) and np.all(ends[hyp] >= starts[hyp])
        assert np.all(ends[hyp] <= len(text))
        # Byte-fallback pieces can share the same source character, so exact
        # offsets may overlap; targets are reconstructed token by token.
        if len(hyp) > 1:
            assert np.all(starts[hyp][1:] >= starts[hyp][:-1])
        raw_counts = prefix_a[ends[hyp]] - prefix_a[starts[hyp]]
        conflict_counts = prefix_c[ends[hyp]] - prefix_c[starts[hyp]]
        baseless_counts = prefix_b[ends[hyp]] - prefix_b[starts[hyp]]
        valid_local = (raw_counts > 0) & (baseless_counts == 0)
        valid_positions = hyp[valid_local]
        expected_count[valid_positions] = raw_counts[valid_local]
        expected_target[valid_positions] = conflict_counts[valid_local] / raw_counts[valid_local]

        assert np.array_equal(actual_count.astype(np.int64), expected_count)
        assert np.allclose(actual_target, expected_target, rtol=0, atol=1e-7)
        assert np.all(actual_count[actual_target < 0] == 0)
        assert np.all((actual_target == -1) | ((actual_target >= 0) & (actual_target <= 1)))
        positive = actual_target > 0
        assert np.all(conflict_counts[valid_local][expected_target[valid_positions] > 0] > 0)
        positive_tokens += int(positive.sum())
        soft_boundary_tokens += int(((actual_target > 0) & (actual_target < 1)).sum())
        ignored_baseless_tokens += int(((raw_counts > 0) & (baseless_counts > 0)).sum())
        reconstructed_tokens += right - left
        reconstructed_char_mass += int(expected_count.sum())

        supervised = int((actual_target >= 0).sum())
        positive_mass = float(np.sum(actual_target * actual_count, where=actual_target >= 0))
        supervised_mass = float(actual_count.sum())
        assert supervised == int(row["supervised_tokens"])
        assert abs(positive_mass - float(row["positive_char_mass"])) < 1e-5
        assert abs(supervised_mass - float(row["supervised_char_mass"])) < 1e-5
        fraction = positive_mass / supervised_mass if supervised_mass else 0.0
        assert abs(fraction - float(row["positive_char_fraction"])) < 1e-7
        stage_supervised[row["stage"]] += int(supervised_mass)
        stage_positive_mass[row["stage"]] += positive_mass
        if row["kind"] == "conflict" and np.any(actual_target == 0):
            conflict_claims_with_local_safe_tokens += 1
        if row["kind"] == "conflict":
            assert positive_mass > 0
        if row["kind"] == "safe":
            assert positive_mass == 0
        fully_span_derived_claims += 1

    assert reconstructed_tokens == flat
    assert positive_tokens > 0 and conflict_claims_with_local_safe_tokens > 0
    return {
        "examples_reconstructed": fully_span_derived_claims,
        "flat_tokens_reconstructed": reconstructed_tokens,
        "supervised_character_mass_reconstructed": reconstructed_char_mass,
        "positive_tokens": positive_tokens,
        "soft_boundary_tokens": soft_boundary_tokens,
        "ignored_tokens_touching_baseless_spans": ignored_baseless_tokens,
        "conflict_microclaims_retaining_exact_local_safe_tokens": conflict_claims_with_local_safe_tokens,
        "stage_supervised_character_mass": dict(stage_supervised),
        "stage_positive_character_mass": dict(stage_positive_mass),
        "sentence_or_claim_label_broadcast_detected": False,
    }


def audit_held_windows(metadata, arrays, qa_answers, held_response_spans):
    label = arrays["held_window_label"]
    ec = arrays["held_window_ec"]
    sc = arrays["held_window_sc"]
    answer_index = arrays["held_window_answer_index"]
    window_indptr = arrays["held_window_token_indptr"]
    window_values = arrays["held_window_token_values"]
    answer_ids = list(map(str, arrays["held_answer_ids"]))
    n_windows = len(label)
    assert len(ec) == len(sc) == len(answer_index) == len(arrays["held_v4_window_score"]) == n_windows
    assert len(window_indptr) == n_windows + 1 and int(window_indptr[0]) == 0
    assert int(window_indptr[-1]) == len(window_values)
    assert np.all(window_indptr[1:] > window_indptr[:-1])
    held_flat_start = min(int(row["flat_start"]) for row in metadata if row["stage"] == "qa_held")
    assert int(window_values.min()) >= held_flat_start
    assert int(window_values.max()) < len(arrays["input_ids"])
    assert np.all(arrays["sequence_code"][window_values] == 1)
    assert np.all(arrays["hypothesis_offset_start"][window_values] >= 0)
    for window in range(n_windows):
        values = window_values[int(window_indptr[window]):int(window_indptr[window + 1])]
        assert np.all(values[1:] > values[:-1])
    assert set(map(int, np.unique(label))).issubset({0, 1})
    assert set(map(int, np.unique(ec))).issubset({0, 1})
    assert set(map(int, np.unique(sc))).issubset({0, 1})
    assert np.all((ec == 0) | (label == 1)) and np.all((sc == 0) | (label == 1))
    assert np.all(answer_index[1:] >= answer_index[:-1])
    assert int(answer_index.min()) == 0 and int(answer_index.max()) == len(answer_ids) - 1

    answer_by_id = {str(row["response_id"]): row for row in qa_answers}
    expected_answer_ids = []
    expected_label = []
    expected_ec = []
    expected_sc = []
    expected_answer_index = []
    expected_answer_risk = []
    for answer in qa_answers:
        rid = str(answer["response_id"])
        if rid not in held_response_spans:
            continue
        expected_answer_ids.append(rid)
        expected_answer_risk.append(int(answer["answer_risk"]))
        lexical = np.asarray(answer["lexical_mask"], dtype=bool)
        risk = np.asarray(answer["risk_mask"], dtype=bool)
        offsets = np.asarray(answer["response_token_offsets"], dtype=np.int64)
        text = answer["original_response"]
        type_mask = {name: np.zeros(len(lexical), dtype=bool) for name in CONFLICT_TYPES}
        for label_type, left, right in held_response_spans[rid]:
            if label_type not in CONFLICT_TYPES:
                continue
            for bpe, (a, b) in enumerate(offsets):
                lo, hi = max(int(a), left), min(int(b), right)
                if lo < hi and any(text[at].isalnum() for at in range(lo, hi)):
                    type_mask[label_type][bpe] = True
        eligible = [start for start in range(max(0, len(lexical) - WINDOW_K + 1))
                    if lexical[start:start + WINDOW_K].any()]
        local_index = len(expected_answer_ids) - 1
        for start in eligible:
            expected_label.append(int(risk[start:start + WINDOW_K].any()))
            expected_ec.append(int(type_mask["Evident Conflict"][start:start + WINDOW_K].any()))
            expected_sc.append(int(type_mask["Subtle Conflict"][start:start + WINDOW_K].any()))
            expected_answer_index.append(local_index)

    assert answer_ids == expected_answer_ids
    assert np.array_equal(arrays["held_answer_label"], np.asarray(expected_answer_risk, dtype=np.int8))
    assert np.array_equal(label, np.asarray(expected_label, dtype=np.int8))
    assert np.array_equal(ec, np.asarray(expected_ec, dtype=np.int8))
    assert np.array_equal(sc, np.asarray(expected_sc, dtype=np.int8))
    assert np.array_equal(answer_index, np.asarray(expected_answer_index, dtype=np.int32))
    assert len(answer_by_id) == N_QA_ANSWERS

    v4 = arrays["held_v4_window_score"].astype(np.float64)
    assert np.isfinite(v4).all() and np.all((v4 >= 0) & (v4 <= 1))
    ap = float(average_precision_score(label, v4))
    assert abs(ap - 0.6390503697171015) < 1e-12
    return {
        "answers": len(answer_ids),
        "windows": n_windows,
        "positive_windows": int(label.sum()),
        "evident_conflict_windows": int(ec.sum()),
        "subtle_conflict_windows": int(sc.sum()),
        "window_to_hypothesis_token_edges": len(window_values),
        "v4_window_average_precision_rebuilt": ap,
        "protocol": "same held fold-0, answer text, lexical eligibility, stride-1 4-BPE windows",
    }


def audit_selection(metadata):
    counts = Counter((row["stage"], row["kind"]) for row in metadata)
    assert counts == Counter({
        ("aux", "conflict"): 4_064, ("aux", "safe"): 4_064,
        ("qa_train", "conflict"): 341, ("qa_train", "safe"): 339,
        ("qa_held", "safe"): 6_280, ("qa_held", "baseless"): 636,
        ("qa_held", "conflict"): 64, ("qa_held", "mixed"): 3,
    })
    for stage in ("aux", "qa_train"):
        by_group = defaultdict(Counter)
        for row in metadata:
            if row["stage"] == stage:
                by_group[row["group_id"]][row["kind"]] += 1
                assert row["selection"] == ("all_clean_conflict" if row["kind"] == "conflict"
                                             else "within_group_matched_safe")
        assert all(value["safe"] <= value["conflict"] for value in by_group.values())
        if stage == "aux":
            assert len(by_group) == 1_205
            assert all(value["safe"] == value["conflict"] for value in by_group.values())
        else:
            assert len(by_group) == 142
            assert sum(value["conflict"] - value["safe"] for value in by_group.values()) == 2
    assert all(row["selection"] == "all_held_microclaims"
               for row in metadata if row["stage"] == "qa_held")
    return {f"{stage}_{kind}": count for (stage, kind), count in sorted(counts.items())}


def audit_runner_and_scope(manifest, complete):
    assert manifest["calibration_rows_read"] == manifest["test_rows_read"] == 0
    assert manifest["model_weights_loaded"] is False and manifest["GPU_used"] is False
    assert complete["scope"] == {
        "calibration_rows_read": 0, "test_rows_read": 0,
        "model_weights_loaded": False, "GPU_used": False,
        "published_baselines_modified": False,
    }
    upstream = load_json(AUX / "INDEPENDENT_AUDIT.json")
    assert upstream["status"] == "passed"
    source_audit = upstream["source_and_labels"]
    assert source_audit["test_answer_or_gold_JSON_parsed"] is False
    assert source_audit["retained_sources_material_linked_to_QA_fit"] == []
    assert source_audit["quarantined_train_sources"] == 62

    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        'V4_CHECKPOINT = ROOT / "results/microclaim_crossencoder_expanded_v4/fold_0/model.pt"',
        'OUT = DATA / "gpu_runs"',
        'control_pre = [qa_batches[index % len(qa_batches)] for index in range(len(aux_batches))]',
        'first = grouped_train(model, optimizer, arrays, aux, "auxiliary_exact_conflict"',
        'first = grouped_train(model, optimizer, arrays, control, "qa_same_update_replacement"',
        'second = grouped_train(model, optimizer, arrays, qa, "final_qa_conflict"',
        'combined = np.maximum(v4, conflict)',
    )
    assert all(fragment in source for fragment in required)
    assert 'ROOT / "results/atomic_microclaim_relation_expanded_v4' not in source
    assert 'ROOT / "calibration' not in source and 'ROOT / "test' not in source
    assert not (OUT / "GPU_SMOKE.json").exists()
    assert not (OUT / "gpu_runs").exists()
    cpu_check = load_json(OUT / "CPU_CHECK.json")
    gpu_plan = load_json(OUT / "GPU_PLAN.json")
    assert cpu_check["status"] == "passed" and cpu_check["GPU_used"] is False
    assert cpu_check["model_checkpoint_loaded"] is False
    assert gpu_plan["gpu_smoke_not_run"] is True and gpu_plan["GPU_used"] is False
    candidate_updates = gpu_plan["plans"]["candidate_aux"]["optimizer_updates"]
    control_updates = gpu_plan["plans"]["control_qa_replacement"]["optimizer_updates"]
    assert candidate_updates == control_updates == 395
    return {
        "calibration_rows_read": 0,
        "test_rows_read": 0,
        "GPU_used": False,
        "model_checkpoint_loaded_during_cpu_stage": False,
        "published_baselines_modified": False,
        "upstream_quarantined_sources": 62,
        "auxiliary_material_link_to_QA_fit": 0,
        "candidate_and_control_pre_stage_optimizer_updates": candidate_updates,
        "control_pre_stage_extra_padded_token_fraction": (
            gpu_plan["plans"]["control_qa_replacement"]["padded_tokens"] /
            gpu_plan["plans"]["candidate_aux"]["padded_tokens"] - 1.0),
        "runner_sha256": sha(RUNNER),
    }


def main():
    manifest = load_json(OUT / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    complete = load_json(OUT / "preparation_complete.json")
    for relative, expected in complete["sources_sha256"].items():
        assert sha(ROOT / relative) == expected, relative

    metadata = load_jsonl(OUT / "examples.jsonl")
    traces = load_jsonl(OUT / "conflict_span_trace.jsonl")
    assert len(metadata) == 15_791 and len(traces) == N_CONFLICT_SPANS
    with np.load(OUT / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    sources = audit_sources_and_metadata(metadata, traces)
    selection = audit_selection(metadata)
    targets = audit_arrays(metadata, arrays, sources["source_text"])
    windows = audit_held_windows(metadata, arrays, sources["qa_answers"],
                                 sources["held_response_spans"])
    scope = audit_runner_and_scope(manifest, complete)

    result = {
        "status": "passed",
        "scope": "Independent CPU-only audit; exact auxiliary and QA fit prefixes only; no model/GPU/calibration/test.",
        "script_sha256": sha(Path(__file__)),
        "core_manifest_sha256": sha(OUT / "manifest.json"),
        "source_and_group_isolation": sources["stats"],
        "selected_examples": selection,
        "exact_token_target_reconstruction": targets,
        "held_4bpe_reconstruction": windows,
        "runner_and_scope": scope,
    }
    target = OUT / "INDEPENDENT_AUDIT.json"
    pending = target.with_suffix(target.suffix + ".pending")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
