"""Independent CPU audit of atomic-microclaim-relation-expanded-v3.

This verifier does not import the preparation runner.  It independently
recomputes labels, BPE/window mappings, hashes, group-weight invariants, and
paired tokenizer lengths.  It never opens an official-test path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/atomic_microclaim_relation_expanded_v3"
FIT_RAW = ROOT / "fit_expansion/data/fit.jsonl"
FIT_TOKENS = ROOT / "fit_expansion/data/tokens_fit.jsonl"
FIT_MICROCLAIMS = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
CAL_TOKENS = ROOT / "data/tokens_calibration.jsonl"
CURRENT_ATOMIC_INPUT = ROOT / "results/atomic_microclaim_nli_v1/inputs.jsonl"
CURRENT_RETRIEVED_INPUT = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
WINDOW_K = 4
EXPECTED = {
    "fit_answers": 3680, "fit_groups": 615, "fit_sources": 634,
    "fit_atomic": 34941, "fit_scored": 34919, "fit_raw_bpes": 665708,
    "fit_windows": 653979, "fit_positive_windows": 58433,
    "cal_answers": 159, "cal_groups": 154, "cal_sources": 159,
    "cal_scored": 2267, "cal_raw_bpes": 42798,
    "cal_windows": 42241, "cal_positive_windows": 5984,
}
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def words(text):
    return [match.group(0).casefold() for match in WORD.finditer(text)]


def independent_rank(query_text, sentences):
    corpus = [words(row["text"]) for row in sentences]
    query = sorted(set(words(query_text)))
    assert query and corpus and all(corpus)
    lengths = np.asarray([len(tokens) for tokens in corpus], dtype=np.float64)
    average = float(lengths.mean())
    df = Counter()
    for tokens in corpus:
        df.update(set(tokens))
    ranked = []
    for sentence_id, tokens in enumerate(corpus):
        frequency = Counter(tokens)
        score = 0.0
        for term in query:
            tf = frequency[term]
            if not tf:
                continue
            inverse = math.log(1.0 + (len(corpus) - df[term] + 0.5) / (df[term] + 0.5))
            denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * len(tokens) / average)
            score += inverse * tf * 2.2 / denominator
        coverage = len(set(query) & set(tokens)) / len(set(query))
        ranked.append((sentence_id, score, coverage))
    return sorted(ranked, key=lambda row: (-row[1], row[0]))[:2]


def make_premise(question, evidence):
    sections = [f"Question: {question}", "Evidence:"]
    for source in evidence:
        sections.append(f"[Passage {source['passage_id']}] " + " ".join(row["text"] for row in source["selected"]))
    return "\n".join(sections)


def base_hierarchy(partitions, response_ids, group_ids):
    output = np.zeros(len(partitions), dtype=np.float64)
    for partition in ("fit", "calibration"):
        active = [index for index, value in enumerate(partitions) if value == partition]
        tree = defaultdict(lambda: defaultdict(list))
        for index in active:
            tree[group_ids[index]][response_ids[index]].append(index)
        for answers in tree.values():
            for indices in answers.values():
                output[indices] = 1.0 / (len(answers) * len(indices))
        output[active] *= len(active) / output[active].sum()
    return output


def train_weights(partitions, response_ids, group_ids, labels):
    output = np.zeros(len(labels), dtype=np.float64)
    active = np.flatnonzero(np.asarray(partitions) == "fit")
    tree = defaultdict(lambda: defaultdict(list))
    for index in active:
        tree[group_ids[index]][response_ids[index]].append(int(index))
    for answers in tree.values():
        for indices in answers.values():
            output[indices] = 1.0 / (len(answers) * len(indices))
    output[active] *= len(active) / output[active].sum()
    mass = np.bincount(labels[active], weights=output[active], minlength=2)
    output[active] *= (mass.sum() / (2 * mass))[labels[active]]
    target = len(active) / len(tree)
    for answers in tree.values():
        indices = [index for values in answers.values() for index in values]
        output[indices] *= target / output[indices].sum()
    output[active] *= len(active) / output[active].sum()
    return output


def save_new(path, value):
    path = Path(path)
    assert not path.exists(), path
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_CPU_dataset_waiting_for_independent_audit"
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert sha(name) == expected, name

    examples = lines(OUT / "examples.jsonl")
    answers = lines(OUT / "answers.jsonl")
    groups = lines(OUT / "groups.jsonl")
    discarded = lines(OUT / "discarded_microclaims.jsonl")
    arrays = np.load(OUT / "arrays.npz")
    assert [row["example_index"] for row in examples] == list(range(len(examples)))
    assert len(examples) == EXPECTED["fit_scored"] + EXPECTED["cal_scored"]
    assert len(answers) == EXPECTED["fit_answers"] + EXPECTED["cal_answers"]
    assert Counter(row["partition"] for row in groups) == {"fit": EXPECTED["fit_groups"], "calibration": EXPECTED["cal_groups"]}

    fit_raw = {row["response_id"]: row for row in lines(FIT_RAW)}
    fit_tokens = {row["response_id"]: row for row in lines(FIT_TOKENS)}
    cal_tokens = {row["response_id"]: row for row in lines(CAL_TOKENS)}
    current_atomic = {row["response_id"]: row for row in lines(CURRENT_ATOMIC_INPUT)}
    retrieved_cal = {row["response_id"]: row for row in lines(CURRENT_RETRIEVED_INPUT)
                     if row["partition"] == "calibration"}
    assert len(fit_raw) == len(fit_tokens) == EXPECTED["fit_answers"]
    assert sum(row["token_count"] for row in fit_tokens.values()) == EXPECTED["fit_raw_bpes"]
    assert sum(row["token_count"] for row in cal_tokens.values()) == EXPECTED["cal_raw_bpes"]

    fit_atomic_all = lines(FIT_MICROCLAIMS)
    assert len(fit_atomic_all) == EXPECTED["fit_atomic"]
    fit_atomic = [row for row in fit_atomic_all if any(char.isalnum() for char in row["text"])]
    expected_discarded = [row for row in fit_atomic_all if not any(char.isalnum() for char in row["text"])]
    assert len(discarded) == len(expected_discarded) == EXPECTED["fit_atomic"] - EXPECTED["fit_scored"]
    assert [(row["microclaim_id"], row["char_start"], row["char_end"], row["text"])
            for row in discarded] == [(row["microclaim_id"], row["start"], row["end"], row["text"])
                                      for row in expected_discarded]
    assert len(fit_atomic) == EXPECTED["fit_scored"]
    fit_examples = [row for row in examples if row["partition"] == "fit"]
    assert [(row["microclaim_id"], row["char_start"], row["char_end"], row["claim_text"])
            for row in fit_examples] == [(row["microclaim_id"], row["start"], row["end"], row["text"])
                                      for row in fit_atomic]
    cal_examples = [row for row in examples if row["partition"] == "calibration"]
    frozen_cal = [claim for answer in current_atomic.values() if answer["partition"] == "calibration"
                  for claim in answer["claims"]]
    assert [(row["microclaim_id"], row["claim_text"], row["hypothesis"])
            for row in cal_examples] == [(row["microclaim_id"], row["text"], row["hypothesis"])
                                        for row in frozen_cal]

    window_labels = arrays["window_label"]
    window_responses = arrays["window_response_index"]
    window_starts = arrays["window_token_start"]
    indptr = arrays["window_claim_indptr"]
    edge_values = arrays["window_claim_example_index"]
    all_labels, partitions, response_ids, group_ids = [], [], [], []
    fit_retrieval_checks = cal_reuse_checks = label_checks = mapping_checks = 0
    for answer in answers:
        response_index = answer["response_index"]
        token = fit_tokens[answer["response_id"]] if answer["partition"] == "fit" else cal_tokens[answer["response_id"]]
        raw = fit_raw[answer["response_id"]] if answer["partition"] == "fit" else None
        assert answer["original_response"] == token["original_response"]
        assert answer["answer_sha256"] == token["answer_sha256"] == digest(answer["original_response"])
        assert answer["token_ids"] == token["token_ids"]
        assert answer["response_token_offsets"] == token["response_token_offsets"]
        assert answer["lexical_mask"] == token["lexical_mask"] and answer["risk_mask"] == token["risk_mask"]
        if raw is not None:
            assert answer["question"] == raw["question"]
        else:
            assert answer["question"] == retrieved_cal[answer["response_id"]]["question"]
        local = examples[answer["example_start"]:answer["example_end"]]
        owners = [[] for _ in range(answer["token_count"])]
        for claim_id, example in enumerate(local):
            assert example["claim_id"] == claim_id and example["response_id"] == answer["response_id"]
            assert answer["original_response"][example["char_start"]:example["char_end"]] == example["claim_text"]
            expected_all = [index for index, (left, right) in enumerate(answer["response_token_offsets"])
                            if max(example["char_start"], left) < min(example["char_end"], right)]
            assert example["all_overlapping_bpe_indices"] == expected_all
            for token_index in example["lexical_bpe_indices"]:
                left, right = answer["response_token_offsets"][token_index]
                assert any(answer["original_response"][position].isalnum() and
                           example["char_start"] <= position < example["char_end"]
                           for position in range(left, right))
                owners[token_index].append(claim_id)
            expected_risk = [index for index in example["lexical_bpe_indices"] if answer["risk_mask"][index]]
            expected_label = int(bool(expected_risk))
            assert example["risk_bpe_indices"] == expected_risk and example["gold_label"] == expected_label
            assert abs(example["risk_bpe_fraction"] - len(expected_risk) / len(example["lexical_bpe_indices"])) < 1e-12
            assert example["premise"] == make_premise(example["question"], example["evidence"])
            assert example["input_pair_sha256"] == digest([example["premise"], example["hypothesis"]])
            if answer["partition"] == "fit":
                passage_lookup = {row["passage_id"]: row for row in answer["passages"]}
                for source in example["evidence"]:
                    ranked = independent_rank(example["hypothesis"], passage_lookup[source["passage_id"]]["sentences"])
                    assert [row["sentence_id"] for row in source["selected"]] == [row[0] for row in ranked]
                    assert all(abs(got["bm25"] - expected[1]) < 1e-12 and
                               abs(got["query_term_coverage"] - expected[2]) < 1e-12
                               for got, expected in zip(source["selected"], ranked))
                    fit_retrieval_checks += 1
            else:
                frozen = current_atomic[answer["response_id"]]["claims"][claim_id]
                assert [(source["passage_id"], [(row["rank"], row["sentence_id"], row["bm25"], row["query_term_coverage"])
                                                for row in source["selected"]]) for source in example["evidence"]] == [
                    (source["passage_id"], [(row["rank"], row["sentence_id"], row["bm25"], row["query_term_coverage"])
                                            for row in source["selected"]]) for source in frozen["retrieval"]]
                cal_reuse_checks += 1
            all_labels.append(expected_label); partitions.append(example["partition"])
            response_ids.append(example["response_id"]); group_ids.append(example["group_id"])
            label_checks += 1
        assert all(bool(owner) == bool(flag) for owner, flag in zip(owners, answer["lexical_mask"]))
        expected_eligible, expected_positive = [], []
        for start in range(max(1, answer["token_count"] - WINDOW_K + 1)):
            if any(answer["lexical_mask"][start:start + WINDOW_K]):
                expected_eligible.append(start)
                expected_positive.append(int(any(answer["risk_mask"][start:start + WINDOW_K])))
        left, right = answer["window_array_start"], answer["window_array_end"]
        assert np.all(window_responses[left:right] == response_index)
        assert np.array_equal(window_starts[left:right], np.asarray(expected_eligible, dtype=np.int32))
        assert np.array_equal(window_labels[left:right], np.asarray(expected_positive, dtype=np.int8))
        eligible_set = set(expected_eligible)
        for claim_id, example in enumerate(local):
            mapped = sorted({start for token_index in example["lexical_bpe_indices"]
                             for start in expected_eligible
                             if start <= token_index < min(answer["token_count"], start + WINDOW_K)})
            assert example["mapped_eligible_window_starts"] == mapped
        for window_index, start in enumerate(expected_eligible, left):
            expected_claims = sorted({answer["example_start"] + claim_id
                                      for token_index in range(start, start + WINDOW_K)
                                      for claim_id in owners[token_index]})
            assert edge_values[indptr[window_index]:indptr[window_index + 1]].tolist() == expected_claims
            mapping_checks += 1

    labels = np.asarray(all_labels, dtype=np.int8)
    assert np.array_equal(labels, arrays["labels"])
    assert np.array_equal(base_hierarchy(partitions, response_ids, group_ids), arrays["hierarchy_evaluation_weight"])
    assert np.allclose(train_weights(partitions, response_ids, group_ids, labels), arrays["fit_training_weight"], rtol=0, atol=1e-12)
    assert not ({row["group_id"] for row in fit_examples} & {row["group_id"] for row in cal_examples})
    assert not ({row["source_id"] for row in fit_examples} & {row["source_id"] for row in cal_examples})
    assert all(len({row["held_fold"] for row in fit_examples if row["group_id"] == group}) == 1
               for group in {row["group_id"] for row in fit_examples})

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    retokenized = []
    for left in range(0, len(examples), 256):
        batch = examples[left:left + 256]
        encoded = tokenizer([row["premise"] for row in batch], [row["hypothesis"] for row in batch],
                            add_special_tokens=True, padding=False, truncation=False)
        retokenized.extend(map(len, encoded["input_ids"]))
    assert np.array_equal(np.asarray(retokenized, dtype=np.int32), arrays["input_token_length"])

    fit_mask = arrays["partition"] == 0
    cal_mask = arrays["partition"] == 1
    assert (int(fit_mask.sum()), int(cal_mask.sum())) == (EXPECTED["fit_scored"], EXPECTED["cal_scored"])
    assert (int((window_responses < EXPECTED["fit_answers"]).sum()),
            int((window_responses >= EXPECTED["fit_answers"]).sum())) == (EXPECTED["fit_windows"], EXPECTED["cal_windows"])
    assert int(window_labels[window_responses < EXPECTED["fit_answers"]].sum()) == EXPECTED["fit_positive_windows"]
    assert int(window_labels[window_responses >= EXPECTED["fit_answers"]].sum()) == EXPECTED["cal_positive_windows"]
    result = {
        "status": "independent_CPU_audit_passed",
        "manifest_sha256": sha(OUT / "manifest.json"),
        "examples": len(examples), "answers": len(answers), "groups": len(groups),
        "fit_atomic_records_checked": EXPECTED["fit_atomic"],
        "fit_scored_microclaims_checked": EXPECTED["fit_scored"],
        "calibration_reused_microclaims_checked": EXPECTED["cal_scored"],
        "fit_raw_BPEs_checked": EXPECTED["fit_raw_bpes"],
        "calibration_raw_BPEs_checked": EXPECTED["cal_raw_bpes"],
        "gold_label_checks": label_checks, "window_mapping_checks": mapping_checks,
        "fit_BM25_source_checks": fit_retrieval_checks,
        "calibration_frozen_retrieval_checks": cal_reuse_checks,
        "paired_token_lengths_recomputed": len(retokenized),
        "fit_positive_microclaims": int(labels[fit_mask].sum()),
        "calibration_positive_microclaims": int(labels[cal_mask].sum()),
        "weights_recomputed_exactly": True, "source_connected_folds_indivisible": True,
        "fit_calibration_group_overlap": 0, "fit_calibration_source_overlap": 0,
        "model_loaded": False, "model_trained": False, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    }
    save_new(OUT / "INDEPENDENT_AUDIT.json", result)
    save_new(OUT / "complete.json", {
        "status": "complete_CPU_preparation_and_audit",
        "manifest_sha256": sha(OUT / "manifest.json"),
        "independent_audit_sha256": sha(OUT / "INDEPENDENT_AUDIT.json"),
        "audit_code_sha256": sha(Path(__file__)),
        "GPU_used": False, "official_test_opened": False,
    })
    print("EXPANDED_MICROCLAIM_RELATION_INDEPENDENT_AUDIT_PASSED", len(examples), flush=True)


if __name__ == "__main__":
    try:
        audit()
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = OUT / f"FAILURE_independent_audit_{stamp}.json"
        if OUT.exists():
            path.write_text(json.dumps({"status": "failed", "stage": "independent_audit",
                                        "error_type": type(exc).__name__, "error": str(exc),
                                        "record_retained": True, "GPU_used": False,
                                        "official_test_opened": False}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        raise
