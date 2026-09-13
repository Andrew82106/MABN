"""Prepare a fit-only within-answer contrastive continuation for expanded v4.

The production candidate warm-starts the matching expanded-v4 OOF/full-fit
checkpoint and performs one extra pair pass.  Every pair contains one risky and
one safe microclaim from the same generated answer.  This file deliberately
exposes only CPU preparation/check/tiny/finalize stages; it cannot start CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertForSequenceClassification


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_EXAMPLES = V4 / "examples.jsonl"
V4_MANIFEST = V4 / "manifest.json"
BASE = ROOT / "results/microclaim_crossencoder_expanded_v4"
BASE_PREP = BASE / "preparation_complete.json"
BASE_ARRAYS = BASE / "encoded_inputs.npz"
BASE_PROTOCOL = BASE / "protocol.json"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
OUT = ROOT / "results/microclaim_within_answer_contrastive_v5"
AUDITOR = HERE / "audit_microclaim_within_answer_contrastive_v5.py"

FIT_CLAIMS = 34_919
FIT_ANSWERS = 3_680
FIT_POSITIVES = 3_474
FIT_NEGATIVES = 31_445
FOLDS = 5
SEED = 20_261_109
MAX_NEGATIVES = 2
MIN_SAFE_CONTENT_WORDS = 2
PAIR_LAMBDA = 0.25
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
PAIR_TOKEN_BUDGET = 1536
PAIR_MAX_PAIRS = 4
ACCUM_MICROBATCHES = 4
THREADS = 4

V4_EXAMPLES_SHA256 = "6a8c46f805eae608dba4708fda9908acf774730d4ff358ff44144f8bc1ff1c54"
V4_MANIFEST_SHA256 = "35791a3b2881b098713d947273d7293f912f6f1a84ed2a50bedfe022a2c74370"
BASE_PREP_SHA256 = "0982dbbc548d5398be202e1ff2ad4566b011e8d7050bac7a55dbf1e9207e80ac"
BASE_ARRAYS_SHA256 = "d5c67a66e3ed16e3cd0faef4085f94ada92c085e7ca733e01b123d522914b7d6"
BASE_PROTOCOL_SHA256 = "5ec665bc7e92cb2cb00bcf5db65741e198f9195c2319e6617ea966b849f19973"
MODEL_SHA256 = "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
STOP = set(
    "a an the and or but if to of in on at for from with by as is are was were be been being "
    "it its this that these those some any may can could would should has have had do does did "
    "according based passage passages provided given seems here there their they them he she his "
    "her we us our you your i me my also however therefore thus".split()
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(path: Path, value):
    assert not path.exists(), f"Refuse overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_jsonl(path: Path, rows):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def save_npz(path: Path, **arrays):
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_fit_prefix():
    """Parse the fit prefix only; the calibration block is never parsed."""
    rows = []
    with V4_EXAMPLES.open(encoding="utf-8") as handle:
        for expected in range(FIT_CLAIMS):
            line = handle.readline()
            assert line
            row = json.loads(line)
            assert row["partition"] == "fit" and row["example_index"] == expected
            rows.append(row)
    return rows


def content_words(text: str):
    return frozenset(token.lower().replace("’", "'") for token in WORD_RE.findall(text)
                     if token.lower().replace("’", "'") not in STOP)


def character_trigrams(text: str):
    normalized = " ".join(token.lower().replace("’", "'") for token in WORD_RE.findall(text))
    return frozenset(normalized[index:index + 3] for index in range(max(0, len(normalized) - 2)))


def dice_parts(left, right):
    denominator = len(left) + len(right)
    return 2 * len(left & right), denominator if denominator else 1


def evidence_hashes(row):
    return frozenset(sentence["text_sha256"] for passage in row["evidence"]
                     for sentence in passage["selected"])


def span_gap(left, right):
    return max(0, max(int(right["char_start"]) - int(left["char_end"]),
                      int(left["char_start"]) - int(right["char_end"])))


def hardness(positive, negative):
    p_words, n_words = content_words(positive["hypothesis"]), content_words(negative["hypothesis"])
    p_chars, n_chars = character_trigrams(positive["hypothesis"]), character_trigrams(negative["hypothesis"])
    word_num, word_den = dice_parts(p_words, n_words)
    char_num, char_den = dice_parts(p_chars, n_chars)
    same_parent = int(positive["parent_claim_index"] == negative["parent_claim_index"])
    shared_evidence = len(evidence_hashes(positive) & evidence_hashes(negative))
    gap = span_gap(positive, negative)
    key = (same_parent, Fraction(word_num, word_den), Fraction(char_num, char_den),
           shared_evidence, -gap, -int(negative["example_index"]))
    detail = {
        "same_parent": same_parent,
        "word_dice_numerator": word_num, "word_dice_denominator": word_den,
        "word_dice": word_num / word_den,
        "char_trigram_dice_numerator": char_num, "char_trigram_dice_denominator": char_den,
        "char_trigram_dice": char_num / char_den,
        "shared_selected_evidence_sentences": shared_evidence,
        "answer_span_gap_chars": gap,
    }
    return key, detail


def build_pairs(rows):
    by_answer = defaultdict(list)
    for row in rows:
        by_answer[row["response_id"]].append(row)
    assert len(by_answer) == FIT_ANSWERS
    pairs = []
    pairable_positives = set()
    for response_id, claims in by_answer.items():
        positives = [row for row in claims if int(row["gold_label"]) == 1]
        negatives = [row for row in claims if int(row["gold_label"]) == 0
                     and len(content_words(row["hypothesis"])) >= MIN_SAFE_CONTENT_WORDS]
        for positive in positives:
            ranked = []
            for negative in negatives:
                key, detail = hardness(positive, negative)
                ranked.append((key, int(negative["example_index"]), negative, detail))
            ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            for pair_rank, (_, _, negative, detail) in enumerate(ranked[:MAX_NEGATIVES], 1):
                pairable_positives.add(int(positive["example_index"]))
                pairs.append({
                    "pair_index": len(pairs), "response_id": response_id,
                    "source_id": positive["source_id"], "group_id": positive["group_id"],
                    "held_fold": int(positive["held_fold"]),
                    "positive_index": int(positive["example_index"]),
                    "negative_index": int(negative["example_index"]),
                    "positive_microclaim_id": positive["microclaim_id"],
                    "negative_microclaim_id": negative["microclaim_id"],
                    "rank_within_positive": pair_rank,
                    **detail,
                })
    assert len({(row["positive_index"], row["negative_index"]) for row in pairs}) == len(pairs)
    return pairs, by_answer, pairable_positives


def pair_weight_matrix(pairs):
    weights = np.zeros((FOLDS + 1, len(pairs)), dtype=np.float32)
    for fold in range(FOLDS + 1):
        active = [i for i, pair in enumerate(pairs)
                  if fold == FOLDS or int(pair["held_fold"]) != fold]
        by_group = defaultdict(lambda: defaultdict(list))
        for index in active:
            pair = pairs[index]
            by_group[pair["group_id"]][int(pair["positive_index"])].append(index)
        group_mass = 1.0 / len(by_group)
        for positives in by_group.values():
            positive_mass = group_mass / len(positives)
            for indices in positives.values():
                for index in indices:
                    weights[fold, index] = positive_mass / len(indices)
        assert np.isclose(weights[fold, active].sum(), 1.0, rtol=0, atol=2e-7)
        inactive = set(range(len(pairs))) - set(active)
        if inactive:
            assert np.all(weights[fold, list(inactive)] == 0)
    return weights


def make_pair_batches(pair_indices, pairs, lengths, seed):
    pair_indices = list(map(int, pair_indices))
    ordered = sorted(pair_indices, key=lambda i: (
        max(int(lengths[pairs[i]["positive_index"]]), int(lengths[pairs[i]["negative_index"]])), i))
    batches, current = [], []
    for index in ordered:
        proposed = current + [index]
        endpoints = [claim for pair_index in proposed for claim in
                     (pairs[pair_index]["positive_index"], pairs[pair_index]["negative_index"])]
        padded = len(endpoints) * max(int(lengths[claim]) for claim in endpoints)
        if current and (padded > PAIR_TOKEN_BUDGET or len(proposed) > PAIR_MAX_PAIRS):
            batches.append(current); current = [index]
        else:
            current = proposed
    if current:
        batches.append(current)
    rng = np.random.default_rng(seed)
    for batch in batches:
        rng.shuffle(batch)
    rng.shuffle(batches)
    assert sorted(index for batch in batches for index in batch) == sorted(pair_indices)
    return batches


def source_files():
    paths = (Path(__file__), AUDITOR, V4_EXAMPLES, V4_MANIFEST, BASE_PREP,
             BASE_ARRAYS, BASE_PROTOCOL, MODEL / "model.safetensors")
    return {str(path.resolve()): sha(path) for path in paths}


def protocol():
    return {
        "version": "microclaim-within-answer-contrastive-v5",
        "scope": "Frozen expanded-v4 fit only; calibration labels and official test are absent from pair mining and training selection.",
        "pair_unit": "One risky microclaim and one safe microclaim in the exact same generated answer, hence the same question/source/group/fold.",
        "hard_negative_rule": {
            "eligible_safe": f"gold_label=0 and at least {MIN_SAFE_CONTENT_WORDS} non-stop content words",
            "maximum_per_positive": MAX_NEGATIVES,
            "ordered_key": ["same parent atomic claim", "content-word Dice", "character-trigram Dice",
                            "shared selected-evidence sentence count", "smallest answer-span gap", "lowest example index"],
            "scores": "Deterministic label-independent surface/context scores; no model score and no calibration statistic.",
        },
        "initialization": {
            "fold_f": "Load expanded-v4 fold_f/model.pt, which was trained without source-connected held fold f.",
            "full_fit": "Load expanded-v4 full_fit/model.pt for calibration inference.",
            "optimizer": "Fresh AdamW state because v4 checkpoints store model weights only; this is a warm-start stage, not optimizer-state continuation.",
        },
        "model": "Unchanged 149,607,171-parameter ModernBERT-base NLI encoder/head and unchanged risk logit logsumexp(neutral,contradiction)-entailment; no new head.",
        "continuation_objective": {
            "base_supervision": "The starting v4 checkpoint already received one all-fit weighted-BCE epoch.",
            "extra_pass": "One pass over selected pair endpoints only.",
            "endpoint_bce": "Original v4 claim weights restricted to selected endpoints; divide by endpoint reuse count before normalization.",
            "ranknet": "Group -> positive microclaim -> selected pair equal mass; softplus(risk_safe-risk_error).",
            "lambda": PAIR_LAMBDA,
            "loss": "weighted endpoint BCE + 0.25 * weighted RankNet",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs": 1,
        },
        "crossfit": "Pair inherits its source-connected held_fold. Fold f may use only pairs with held_fold != f; full-fit uses all fit pairs.",
        "evaluation": "Unchanged expanded-v4 4-BPE window projection, fit-OOF threshold selection and strict calibration evaluation.",
        "selection": "Single frozen candidate; no lambda/LR/pair-rule grid and no calibration-based choice.",
        "baselines": "Read-only; no baseline model, score, threshold, or artifact is modified.",
        "available_stages": "CPU prepare -> check -> cpu-tiny -> independent audit -> finalize. This runner has no CUDA stage.",
        "official_test_opened": False,
    }


def verify_sources():
    assert sha(V4_EXAMPLES) == V4_EXAMPLES_SHA256
    assert sha(V4_MANIFEST) == V4_MANIFEST_SHA256
    assert sha(BASE_PREP) == BASE_PREP_SHA256
    assert sha(BASE_ARRAYS) == BASE_ARRAYS_SHA256
    assert sha(BASE_PROTOCOL) == BASE_PROTOCOL_SHA256
    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    base = read_json(BASE_PREP)
    assert base["status"] == "CPU_prepared_not_trained"
    assert base["files_sha256"]["encoded_inputs.npz"] == BASE_ARRAYS_SHA256


def load_base_arrays():
    with np.load(BASE_ARRAYS, allow_pickle=False) as values:
        return {name: values[name].copy() for name in
                ("flat_input_ids", "bounds", "lengths", "fold_assignment", "weights")}


def prepare():
    assert not torch.cuda.is_initialized() and not OUT.exists()
    verify_sources(); snapshot = source_files()
    rows = read_fit_prefix()
    assert Counter(int(row["gold_label"]) for row in rows) == {0: FIT_NEGATIVES, 1: FIT_POSITIVES}
    pairs, by_answer, pairable = build_pairs(rows)
    base = load_base_arrays(); lengths = base["lengths"][:FIT_CLAIMS]
    folds = base["fold_assignment"][:FIT_CLAIMS]
    assert np.array_equal(folds, np.asarray([row["held_fold"] for row in rows], dtype=np.int8))
    for pair in pairs:
        p, n = rows[pair["positive_index"]], rows[pair["negative_index"]]
        assert p["gold_label"] == 1 and n["gold_label"] == 0
        assert p["response_id"] == n["response_id"] == pair["response_id"]
        assert p["source_id"] == n["source_id"] == pair["source_id"]
        assert p["group_id"] == n["group_id"] == pair["group_id"]
        assert p["held_fold"] == n["held_fold"] == pair["held_fold"]
    rank_weights = pair_weight_matrix(pairs)
    pair_fold = np.asarray([row["held_fold"] for row in pairs], dtype=np.int8)
    positive = np.asarray([row["positive_index"] for row in pairs], dtype=np.int32)
    negative = np.asarray([row["negative_index"] for row in pairs], dtype=np.int32)
    plans = []
    for fold in range(FOLDS + 1):
        active = np.flatnonzero((pair_fold != fold) if fold < FOLDS else np.ones(len(pairs), dtype=bool))
        batches = make_pair_batches(active, pairs, lengths, SEED + fold)
        logical = int(sum(int(lengths[positive[i]]) + int(lengths[negative[i]]) for i in active))
        padded = int(sum(2 * len(batch) * max(
            max(int(lengths[positive[i]]), int(lengths[negative[i]])) for i in batch) for batch in batches))
        plans.append({
            "fold": "full_fit" if fold == FOLDS else fold,
            "pairs": len(active), "groups": len({pairs[i]["group_id"] for i in active}),
            "positive_endpoints": len({int(positive[i]) for i in active}),
            "safe_endpoints": len({int(negative[i]) for i in active}),
            "logical_endpoint_tokens": logical, "padded_endpoint_tokens": padded,
            "pair_microbatches": len(batches),
            "optimizer_updates": math.ceil(len(batches) / ACCUM_MICROBATCHES),
            "rank_weight_sum": float(rank_weights[fold, active].sum()),
        })
    all_positive_answers = sum(bool(claims) and all(row["gold_label"] == 1 for row in claims)
                               for claims in by_answer.values())
    risky_answers = sum(any(row["gold_label"] == 1 for row in claims) for claims in by_answer.values())
    pairable_answers = len({row["response_id"] for row in pairs})
    unique_safe = {int(row["negative_index"]) for row in pairs}
    reuse = Counter(int(row["negative_index"]) for row in pairs)
    by_positive = Counter(int(row["positive_index"]) for row in pairs)
    stats = {
        "fit_claims": FIT_CLAIMS, "fit_answers": FIT_ANSWERS,
        "positive_claims": FIT_POSITIVES, "safe_claims": FIT_NEGATIVES,
        "risky_answers": risky_answers, "pairable_risky_answers": pairable_answers,
        "all_positive_unpairable_answers": all_positive_answers,
        "pairs": len(pairs), "pairable_positive_claims": len(pairable),
        "unpairable_positive_claims": FIT_POSITIVES - len(pairable),
        "positive_coverage": len(pairable) / FIT_POSITIVES,
        "unique_selected_safe_claims": len(unique_safe),
        "safe_coverage_all_fit": len(unique_safe) / FIT_NEGATIVES,
        "pairs_per_positive": dict(sorted(Counter(by_positive.values()).items())),
        "safe_reuse": {"maximum": max(reuse.values()),
                       "reused_more_than_twice": sum(value > 2 for value in reuse.values())},
        "same_parent_pairs": sum(row["same_parent"] for row in pairs),
        "shared_evidence_zero_pairs": sum(row["shared_selected_evidence_sentences"] == 0 for row in pairs),
        "fold_pair_counts": dict(sorted(Counter(int(row["held_fold"]) for row in pairs).items())),
        "word_dice": quantiles([row["word_dice"] for row in pairs]),
        "char_trigram_dice": quantiles([row["char_trigram_dice"] for row in pairs]),
        "span_gap_chars": quantiles([row["answer_span_gap_chars"] for row in pairs]),
    }
    base_smoke = read_json(BASE / "GPU_SMOKE.json") if (BASE / "GPU_SMOKE.json").exists() else None
    total_pair_padded = sum(row["padded_endpoint_tokens"] for row in plans)
    runtime = {
        "six_models_pair_padded_tokens": total_pair_padded,
        "six_models_pair_logical_tokens": sum(row["logical_endpoint_tokens"] for row in plans),
        "base_v4_six_model_padded_tokens": sum(row["padded_tokens"] for row in read_json(BASE / "preparation.json")["fold_batch_plan"]),
        "extra_pair_pass_fraction_of_original_v4_training": total_pair_padded / sum(
            row["padded_tokens"] for row in read_json(BASE / "preparation.json")["fold_batch_plan"]),
    }
    if base_smoke:
        seconds_per_padded = base_smoke["measured_train_seconds"] / base_smoke["train_report"]["padded_tokens"]
        runtime.update({"same_host_smoke_sha256": sha(BASE / "GPU_SMOKE.json"),
                        "point_extra_minutes": total_pair_padded * seconds_per_padded / 60,
                        "conservative_extra_minutes": total_pair_padded * seconds_per_padded * 1.35 / 60 + 5})
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_jsonl(OUT / "pairs.jsonl", pairs)
    save_npz(OUT / "pair_arrays.npz", positive_index=positive, negative_index=negative,
             held_fold=pair_fold, rank_within_positive=np.asarray(
                 [row["rank_within_positive"] for row in pairs], dtype=np.int8),
             rank_weights=rank_weights)
    save_json(OUT / "statistics.json", stats)
    save_json(OUT / "training_plan.json", {"folds": plans, "runtime": runtime,
              "checkpoint_contract": protocol()["initialization"],
              "checkpoint_readiness_at_prepare": {
                  **{f"fold_{fold}": (BASE / f"fold_{fold}/complete.json").exists() for fold in range(FOLDS)},
                  "full_fit": (BASE / "full_fit/complete.json").exists()},
              "GPU_used": False, "calibration_labels_used": False, "official_test_opened": False})
    save_json(OUT / "source_snapshot.json", snapshot)
    names = ("protocol.json", "pairs.jsonl", "pair_arrays.npz", "statistics.json",
             "training_plan.json", "source_snapshot.json")
    save_json(OUT / "preparation_complete.json", {
        "status": "CPU_pair_preparation_complete", "files_sha256": {name: sha(OUT / name) for name in names},
        "source_sha256": snapshot, "GPU_used": False, "calibration_rows_parsed": 0,
        "official_test_opened": False, "formal_baselines_modified": False,
    })
    assert snapshot == source_files() and not torch.cuda.is_initialized()
    print("WITHIN_ANSWER_CONTRASTIVE_V5_PREPARED", len(pairs), len(pairable), flush=True)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max())}


def check_prepared():
    assert not torch.cuda.is_initialized(); verify_sources()
    complete = read_json(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_pair_preparation_complete"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    assert complete["source_sha256"] == source_files() and read_json(OUT / "protocol.json") == protocol()
    pairs = read_jsonl(OUT / "pairs.jsonl")
    with np.load(OUT / "pair_arrays.npz", allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    assert len(pairs) == len(arrays["positive_index"]) and arrays["rank_weights"].shape == (FOLDS + 1, len(pairs))
    assert np.array_equal(arrays["positive_index"], [row["positive_index"] for row in pairs])
    assert np.array_equal(arrays["negative_index"], [row["negative_index"] for row in pairs])
    assert np.all(arrays["positive_index"] < FIT_CLAIMS) and np.all(arrays["negative_index"] < FIT_CLAIMS)
    for fold in range(FOLDS + 1):
        active = np.ones(len(pairs), dtype=bool) if fold == FOLDS else arrays["held_fold"] != fold
        assert np.isclose(arrays["rank_weights"][fold, active].sum(), 1.0, rtol=0, atol=2e-7)
        assert np.all(arrays["rank_weights"][fold, ~active] == 0)
    return pairs, arrays


def collate(base, indices, device):
    indices = list(map(int, indices)); lengths = base["lengths"][indices].astype(int)
    width = int(lengths.max()); ids = np.full((len(indices), width), 50283, dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=np.int64)
    for row, (index, length) in enumerate(zip(indices, lengths)):
        left, right = base["bounds"][index]
        ids[row, :length] = base["flat_input_ids"][left:right]; mask[row, :length] = 1
    return {"input_ids": torch.as_tensor(ids, device=device),
            "attention_mask": torch.as_tensor(mask, device=device)}


def risk_logit(logits):
    return torch.logsumexp(logits[:, 1:3].float(), dim=-1) - logits[:, 0].float()


def loss_parts(risk, endpoint_labels, endpoint_weights, positive_positions,
               negative_positions, pair_weights):
    bce_each = F.binary_cross_entropy_with_logits(risk, endpoint_labels, reduction="none")
    bce = (bce_each * endpoint_weights).sum() / endpoint_weights.sum()
    rank_each = F.softplus(risk[negative_positions] - risk[positive_positions])
    rank = (rank_each * pair_weights).sum() / pair_weights.sum()
    return bce, rank, bce + PAIR_LAMBDA * rank


def tiny_config():
    config = ModernBertConfig(vocab_size=50368, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              max_position_embeddings=768, local_attention=32,
                              pad_token_id=50283, bos_token_id=50281, eos_token_id=50282,
                              sep_token_id=50282, num_labels=3, classifier_pooling="mean",
                              reference_compile=False)
    config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    config._attn_implementation = "sdpa"
    return config


def cpu_tiny():
    pairs, arrays = check_prepared(); assert not (OUT / "CPU_TINY_TRAIN.json").exists()
    base = load_base_arrays(); torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    # Four shortest disjoint pairs exercise the exact endpoint layout cheaply.
    ordered = sorted(range(len(pairs)), key=lambda i: max(
        int(base["lengths"][pairs[i]["positive_index"]]), int(base["lengths"][pairs[i]["negative_index"]])))
    chosen, used = [], set()
    for pair_index in ordered:
        endpoints = {pairs[pair_index]["positive_index"], pairs[pair_index]["negative_index"]}
        if not (used & endpoints):
            chosen.append(pair_index); used |= endpoints
        if len(chosen) == 4:
            break
    assert len(chosen) == 4
    endpoints = [claim for pair_index in chosen for claim in
                 (pairs[pair_index]["positive_index"], pairs[pair_index]["negative_index"])]
    labels = torch.tensor([1., 0.] * len(chosen))
    endpoint_weights = torch.ones(len(endpoints))
    pos = torch.arange(0, len(endpoints), 2); neg = pos + 1
    pair_weights = torch.ones(len(chosen))
    model = ModernBertForSequenceClassification(tiny_config()).float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    batch = collate(base, endpoints, torch.device("cpu"))
    risk = risk_logit(model(**batch).logits)
    bce, rank, total = loss_parts(risk, labels, endpoint_weights, pos, neg, pair_weights)
    bce_gradient = torch.autograd.grad(bce, risk, retain_graph=True)[0]
    rank_gradient = torch.autograd.grad(rank, risk, retain_graph=True)[0]
    assert torch.all(bce_gradient[pos] < 0) and torch.all(bce_gradient[neg] > 0)
    assert torch.all(rank_gradient[pos] < 0) and torch.all(rank_gradient[neg] > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad(set_to_none=True); total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value)]
    ordered_loss = F.softplus(torch.tensor(-4.0)); reversed_loss = F.softplus(torch.tensor(4.0))
    assert changed and any(name.startswith("model.layers") for name in changed)
    assert ordered_loss < reversed_loss and torch.isfinite(total) and norm > 0
    report = {
        "status": "CPU_tiny_combined_loss_passed", "selected_pair_indices": chosen,
        "selected_claim_indices": endpoints, "bce": float(bce), "ranknet": float(rank),
        "combined": float(total), "lambda": PAIR_LAMBDA, "gradient_norm": float(norm),
        "bce_gradient_signs_correct": True, "ranknet_gradient_signs_correct": True,
        "changed_parameter_tensors": len(changed), "encoder_parameters_changed": True,
        "ordered_rank_loss": float(ordered_loss), "reversed_rank_loss": float(reversed_loss),
        "real_pretrained_model_loaded": False, "GPU_used": False,
        "calibration_labels_used": False, "official_test_opened": False,
    }
    save_json(OUT / "CPU_TINY_TRAIN.json", report)
    assert not torch.cuda.is_initialized()
    print("WITHIN_ANSWER_CONTRASTIVE_V5_CPU_TINY_PASSED", len(changed), flush=True)


def check():
    pairs, _ = check_prepared()
    save_json(OUT / "CPU_CHECK.json", {
        "status": "passed", "pairs": len(pairs), "fit_only_indices": True,
        "crossfold_pair_exclusion_checked": True, "rank_weights_checked": True,
        "calibration_rows_parsed": 0, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    })
    print("WITHIN_ANSWER_CONTRASTIVE_V5_CPU_CHECK_PASSED", len(pairs), flush=True)


def finalize():
    check_prepared(); assert read_json(OUT / "CPU_CHECK.json")["status"] == "passed"
    assert read_json(OUT / "CPU_TINY_TRAIN.json")["status"] == "CPU_tiny_combined_loss_passed"
    audit = read_json(OUT / "INDEPENDENT_AUDIT.json")
    assert audit["status"] == "independent_CPU_audit_passed"
    stats = read_json(OUT / "statistics.json"); plan = read_json(OUT / "training_plan.json")
    report = [
        "# Within-answer contrastive ModernBERT v5", "",
        "本阶段完成了 fit-only 配对、训练协议、成本核算和 CPU tiny；没有运行 GPU，也没有读取 calibration/test 标签。", "",
        "## 配对结果", "",
        f"- {stats['pairs']:,} 对；覆盖 {stats['pairable_positive_claims']:,}/{FIT_POSITIVES:,} 个错误微主张（{stats['positive_coverage']:.2%}）。",
        f"- 覆盖 {stats['pairable_risky_answers']:,}/{stats['risky_answers']:,} 个含错误回答；未配对的是 {stats['all_positive_unpairable_answers']} 个全错回答中的 {stats['unpairable_positive_claims']} 个错误微主张。",
        f"- 选中 {stats['unique_selected_safe_claims']:,} 个不同安全微主张；每个可配对错误微主张取 1-2 个最相似安全负例。",
        f"- 同一原子父主张 {stats['same_parent_pairs']} 对；只有 {stats['shared_evidence_zero_pairs']} 对不共享已选证据句。", "",
        "## 训练接法", "",
        "可直接加载每个对应的 expanded-v4 checkpoint 权重继续训练；v4 没保存优化器状态，所以 AdamW 状态会重新初始化。",
        "模型和 risk logit 不变，不增加头。额外做一轮 pair endpoint pass：原 v4 endpoint BCE + 0.25 × RankNet。",
        "OOF fold f 只用 held_fold != f 的 pair；full-fit 用全部 fit pair。最终仍按原 4-BPE 窗口和 fit 阈值评测。", "",
        "## 成本与限制", "",
        f"六个模型额外 pair pass 约为原 v4 训练 token 的 {plan['runtime']['extra_pair_pass_fraction_of_original_v4_training']:.1%}。",
        f"同机 smoke 外推约 {plan['runtime'].get('point_extra_minutes', float('nan')):.1f} 分钟，保守约 {plan['runtime'].get('conservative_extra_minutes', float('nan')):.1f} 分钟；尚未实跑 GPU。",
        "这里的‘困难’是同回答内的结构和表面相似，不是用 calibration 或模型分数挖掘。负例复用按 claim occurrence 校正，避免一个安全句反复出现而主导 BCE。",
        "正式 baseline 未改，official test 未打开。", "",
    ]
    report_path = OUT / "REPORT.md"; assert not report_path.exists()
    report_path.write_text("\n".join(report), encoding="utf-8")
    names = ("protocol.json", "pairs.jsonl", "pair_arrays.npz", "statistics.json",
             "training_plan.json", "preparation_complete.json", "CPU_CHECK.json",
             "CPU_TINY_TRAIN.json", "INDEPENDENT_AUDIT.json", "REPORT.md")
    save_json(OUT / "complete.json", {
        "status": "CPU_design_complete_waiting_for_GPU_authorization",
        "files_sha256": {name: sha(OUT / name) for name in names},
        "GPU_used": False, "calibration_labels_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    })
    print("WITHIN_ANSWER_CONTRASTIVE_V5_FINALIZED", stats["pairs"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "cpu-tiny", "finalize", "verify"))
    stage = parser.parse_args().stage
    if stage == "prepare": prepare()
    elif stage == "check": check()
    elif stage == "cpu-tiny": cpu_tiny()
    elif stage == "finalize": finalize()
    else:
        check_prepared()
        if (OUT / "complete.json").exists():
            complete = read_json(OUT / "complete.json")
            for name, expected in complete["files_sha256"].items(): assert sha(OUT / name) == expected
        print("WITHIN_ANSWER_CONTRASTIVE_V5_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
