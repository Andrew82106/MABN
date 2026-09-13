"""Gold-supervised full-evidence ModernBERT NLI microclaim cross-encoder.

This development runner uses only the released RAGTruth QA fit634 and
calibration159 partitions.  It never opens the official test and never writes
to a formal-baseline directory.  The single trained candidate is evaluated by
source-connected five-fold OOF predictions; a frozen NLI checkpoint on the
same inputs is the only control.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForSequenceClassification,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402
import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402


OUT = ROOT / "results/microclaim_crossencoder_v1"
ATOMIC = ROOT / "results/atomic_microclaim_nli_v1"
ATOMIC_INPUT = ATOMIC / "inputs.jsonl"
ATOMIC_LABELS = ATOMIC / "microclaim_labels.npy"
ATOMIC_SCORES = ATOMIC / "scores.npz"
QUESTION_INPUT = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
MODEL_SHA256 = atomic_nli.MODEL_SHA256
MODEL_REVISION = atomic_nli.MODEL_REVISION

PARTITIONS = ("fit", "calibration")
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
FOLDS = 5
SEED = 20261023
MAX_LENGTH = 768
TRAIN_TOKEN_BUDGET = 1536
EVAL_TOKEN_BUDGET = 4096
TRAIN_MAX_EXAMPLES = 8
EVAL_MAX_EXAMPLES = 16
ACCUM_MICROBATCHES = 4
EPOCHS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
HARD_NEGATIVE_ALPHA = 2.0
THREADS = 4
MIN_FREE = 5 * 1024**3
PEAK_LIMIT = int(7.75 * 1024**3)
INCUMBENT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
OLD_ATOMIC_AUDIT = ROOT / "research/atomic_microclaim_nli_v1_audit/REPORT.md"


def sha(path):
    return q.sha(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def save_json(path, value):
    path = Path(path)
    assert not path.exists(), f"Refuse overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_jsonl(path, rows):
    path = Path(path)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def source_files():
    paths = [
        Path(__file__), Path(q.__file__), Path(atomic_nli.__file__), ATOMIC_INPUT,
        ATOMIC_LABELS, ATOMIC_SCORES, ATOMIC / "complete.json",
        ATOMIC / "summary.json", QUESTION_INPUT, INCUMBENT,
        MODEL / "config.json", MODEL / "tokenizer.json",
        MODEL / "tokenizer_config.json", MODEL / "special_tokens_map.json",
        MODEL / "model.safetensors", MODEL / "download_manifest.json",
    ]
    return {str(path.resolve()): sha(path) for path in paths}


def protocol():
    return {
        "version": "microclaim-full-evidence-crossencoder-v1",
        "scope": "Only original RAGTruth QA fit634+calibration159 development answers; official test absent.",
        "unit": "One deterministic atomic microclaim; gold risk=1 iff any lexical original-answer BPE owned by the microclaim has a released human risk label.",
        "input": {
            "premise": "Question plus all three complete released retrieved passages, in released passage order.",
            "hypothesis": "The deterministic contextualized atomic microclaim.",
            "tokenizer": "tasksource/ModernBERT-base-nli tokenizer; paired encoding; no truncation.",
            "max_length_gate": MAX_LENGTH,
            "reason": "Full evidence removes BM25 miss as a false-positive cause. Question disambiguates what the evidence must answer.",
        },
        "trained_candidate": {
            "name": "fp_aware_full_evidence",
            "initialization": "Exact frozen tasksource/ModernBERT-base-nli 3-way checkpoint.",
            "architecture": "Unchanged 3-way NLI cross-encoder. Risk logit=logsumexp(neutral,contradiction)-entailment.",
            "parameters": "All encoder and NLI-head parameters fine-tuned; no added head.",
            "loss": "Binary BCE on microclaim gold. Existing group/answer/claim weights are retained; within each answer, clean claims are redistributed by 1+2*(frozen atomic-NLI risk)^2 and renormalized to preserve that answer's clean mass.",
            "fp_design": "The bounded 1..3 redistribution concentrates clean supervision on NLI-hard false-positive-like claims without reducing positive loss mass.",
            "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                          "clip_norm": 1.0, "epochs": EPOCHS, "scheduler": None},
            "precision": "FP32 parameters/gradients/optimizer; BF16 CUDA forward; non-reentrant gradient checkpointing; TF32 off.",
        },
        "minimal_control": {
            "name": "frozen_full_evidence_nli",
            "model": "Same untouched NLI checkpoint and exact same paired inputs.",
            "score": "1-softmax(entailment); no fit, feature learner, or parameter search.",
        },
        "crossfit": {
            "folds": FOLDS,
            "split": "GroupKFold over locked source-connected group_id; zero shared group, source_id, response_id, answer hash, or passage-body hash across train/held.",
            "fit_prediction": "Every fit claim predicted only by the fold model that excluded its source-connected group.",
            "calibration_prediction": "A sixth model trained once on all fit groups predicts calibration; no calibration label is used during GPU training/inference.",
        },
        "batching": {"train_padded_token_budget": TRAIN_TOKEN_BUDGET,
                     "train_max_examples": TRAIN_MAX_EXAMPLES,
                     "accumulated_microbatches": ACCUM_MICROBATCHES,
                     "eval_padded_token_budget": EVAL_TOKEN_BUDGET,
                     "eval_max_examples": EVAL_MAX_EXAMPLES,
                     "length_bucketing": "Deterministic sort-pack, then seeded batch and within-batch permutations."},
        "evaluation": "Unchanged 4-original-BPE stride-1 windows. Microclaim score is copied to owned lexical BPEs; a window and answer use max overlap. Thresholds chosen only on fit OOF and frozen for calibration.",
        "selection": "Exactly one trained candidate and one frozen control; no hyperparameter grid or calibration selection.",
        "baselines": "Read-only. No baseline implementation, score, threshold, or artifact is changed.",
        "stage_gate": "prepare -> check -> cpu-tiny -> gpu-smoke -> five train-fold calls + one train-full call + frozen control -> finalize -> verify.",
        "seed": SEED, "official_test_opened": False,
    }


def full_evidence(row):
    blocks = []
    for passage in row["passages"]:
        text = " ".join(sentence["text"] for sentence in passage["sentences"])
        blocks.append(f"[Passage {passage['passage_id']}] {text}")
    assert len(blocks) == 3
    return "\n".join(blocks)


def paired_text(question, evidence, hypothesis):
    return f"Question: {question}\nRetrieved evidence:\n{evidence}", f"Claim: {hypothesis}"


def load_arrays():
    with np.load(OUT / "encoded_inputs.npz", allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def make_batches(indices, lengths, seed, token_budget, max_examples):
    indices = np.asarray(indices, dtype=np.int64)
    ordered = indices[np.argsort(lengths[indices], kind="stable")]
    batches, current = [], []
    for raw in ordered:
        index = int(raw)
        proposed = current + [index]
        padded = max(int(lengths[j]) for j in proposed) * len(proposed)
        if current and (padded > token_budget or len(proposed) > max_examples):
            batches.append(current); current = [index]
        else:
            current = proposed
    if current:
        batches.append(current)
    rng = np.random.default_rng(seed)
    for batch in batches:
        rng.shuffle(batch)
    rng.shuffle(batches)
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == sorted(indices.tolist()) and len(flat) == len(indices)
    assert all(max(int(lengths[j]) for j in batch) * len(batch) <= token_budget for batch in batches)
    assert all(len(batch) <= max_examples for batch in batches)
    return batches


def base_and_hard_weights(rows, response_ids, labels, raw_risk, active):
    by_id = {row["response_id"]: row for row in rows}
    weights = atomic_nli.claim_weights(by_id, response_ids, labels, np.asarray(active, dtype=np.int64))
    before = weights.copy()
    active_set = set(map(int, active))
    by_answer = defaultdict(list)
    for index in active_set:
        by_answer[str(response_ids[index])].append(index)
    changed = 0
    for indices in by_answer.values():
        negative = [index for index in indices if labels[index] == 0]
        if len(negative) <= 1:
            continue
        mass = float(weights[negative].sum())
        factor = 1.0 + HARD_NEGATIVE_ALPHA * np.square(raw_risk[negative])
        weights[negative] *= factor
        weights[negative] *= mass / float(weights[negative].sum())
        changed += len(negative)
    assert np.all(weights[list(active_set)] > 0) and np.all(weights[[i for i in range(len(weights)) if i not in active_set]] == 0)
    assert np.isclose(weights.sum(), before.sum(), rtol=0, atol=2e-6)
    return weights.astype(np.float32), changed


def split_audit(rows, response_ids, groups, sources, answer_hashes, passage_hashes, fit, assignments):
    result = []
    for fold in range(FOLDS):
        held = fit[assignments[fit] == fold]
        train = fit[assignments[fit] != fold]
        fields = {}
        for name, values in (("groups", groups), ("sources", sources), ("responses", response_ids),
                             ("answer_hashes", answer_hashes)):
            overlap = set(values[train]) & set(values[held])
            assert not overlap, (fold, name, list(overlap)[:3])
            fields[name + "_overlap"] = 0
        train_responses, held_responses = set(response_ids[train]), set(response_ids[held])
        train_passages = {value for rid in train_responses for value in passage_hashes[rid]}
        held_passages = {value for rid in held_responses for value in passage_hashes[rid]}
        assert not (train_passages & held_passages)
        result.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                       "train_groups": len(set(groups[train])), "held_groups": len(set(groups[held])),
                       **fields, "passage_body_hash_overlap": 0})
    return result


def prepare():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), f"Preserve existing run: {OUT}"
    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    snapshot = source_files()
    rows = q.lines(ATOMIC_INPUT)
    questions = {row["response_id"]: row["question"] for row in q.lines(QUESTION_INPUT)}
    assert len(rows) == len(questions) == 793
    assert Counter(row["partition"] for row in rows) == EXPECTED_ANSWERS
    labels = np.load(ATOMIC_LABELS).astype(np.int8)
    with np.load(ATOMIC_SCORES, allow_pickle=False) as z:
        raw_risk = z["raw_claim_scores"].astype(np.float32)
    assert labels.shape == raw_risk.shape == (sum(EXPECTED_CLAIMS.values()),)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast and tokenizer.pad_token_id is not None
    records, flat, bounds, lengths = [], [], [], []
    response_ids, groups, sources, partitions, answer_hashes = [], [], [], [], []
    passage_hashes = {}
    cursor = 0
    for answer_index, row in enumerate(rows):
        evidence = full_evidence(row); question = questions[row["response_id"]]
        passage_hashes[row["response_id"]] = [p["body_sha256"] for p in row["passages"]]
        for claim in row["claims"]:
            premise, hypothesis = paired_text(question, evidence, claim["hypothesis"])
            encoded = tokenizer(premise, hypothesis, add_special_tokens=True, truncation=False)
            assert set(encoded) == {"input_ids", "attention_mask"}
            ids = encoded["input_ids"]
            assert encoded["attention_mask"] == [1] * len(ids) and len(ids) <= MAX_LENGTH
            begin = len(flat); flat.extend(ids); end = len(flat)
            index = len(records); bounds.append((begin, end)); lengths.append(len(ids))
            response_ids.append(row["response_id"]); groups.append(row["group_id"])
            sources.append(row["source_id"]); partitions.append(row["partition"])
            answer_hashes.append(row["answer_sha256"])
            records.append({"index": index, "partition": row["partition"],
                            "response_id": row["response_id"], "source_id": row["source_id"],
                            "group_id": row["group_id"], "answer_sha256": row["answer_sha256"],
                            "claim_id": claim["claim_id"], "microclaim_id": claim["microclaim_id"],
                            "input_tokens": len(ids), "premise_sha256": digest(premise),
                            "hypothesis_sha256": digest(hypothesis),
                            "label": int(labels[index]), "frozen_atomic_nli_risk": float(raw_risk[index]),
                            "lexical_token_indices": claim["lexical_token_indices"]})
        if (answer_index + 1) % 100 == 0:
            print("MICROCLAIM_CROSSENCODER_PREP", answer_index + 1, len(rows), flush=True)
    del tokenizer
    lengths = np.asarray(lengths, dtype=np.int16)
    partitions = np.asarray(partitions)
    response_ids = np.asarray(response_ids)
    groups = np.asarray(groups)
    sources = np.asarray(sources)
    answer_hashes = np.asarray(answer_hashes)
    bounds = np.asarray(bounds, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.int32)
    assert Counter(partitions.tolist()) == EXPECTED_CLAIMS and len(records) == len(labels)
    fit = np.flatnonzero(partitions == "fit")
    cal = np.flatnonzero(partitions == "calibration")
    assignments = np.full(len(records), -1, dtype=np.int8)
    for fold, (_, held_local) in enumerate(GroupKFold(FOLDS).split(fit, labels[fit], groups[fit])):
        assignments[fit[held_local]] = fold
    assert set(assignments[fit]) == set(range(FOLDS)) and np.all(assignments[cal] == -1)
    leakage = split_audit(rows, response_ids, groups, sources, answer_hashes, passage_hashes, fit, assignments)
    assert not (set(groups[fit]) & set(groups[cal])) and not (set(sources[fit]) & set(sources[cal]))
    weight_matrix = np.zeros((FOLDS + 1, len(records)), dtype=np.float32)
    batch_plan = []
    for fold in range(FOLDS):
        train = fit[assignments[fit] != fold]
        held = fit[assignments[fit] == fold]
        weight_matrix[fold], changed = base_and_hard_weights(rows, response_ids, labels, raw_risk, train)
        batches = make_batches(train, lengths, SEED + fold, TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
        padded = sum(max(int(lengths[j]) for j in batch) * len(batch) for batch in batches)
        logical = int(lengths[train].sum())
        batch_plan.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                           "positive_train": int(labels[train].sum()), "negative_train": int((labels[train] == 0).sum()),
                           "hard_clean_claims_redistributed": changed, "microbatches": len(batches),
                           "optimizer_updates": math.ceil(len(batches) / ACCUM_MICROBATCHES),
                           "logical_tokens": logical, "padded_tokens": padded,
                           "padding_ratio": padded / logical})
    weight_matrix[FOLDS], changed = base_and_hard_weights(rows, response_ids, labels, raw_risk, fit)
    batches = make_batches(fit, lengths, SEED + FOLDS, TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
    padded = sum(max(int(lengths[j]) for j in batch) * len(batch) for batch in batches)
    logical = int(lengths[fit].sum())
    batch_plan.append({"fold": "full_fit", "train_claims": len(fit), "held_claims": 0,
                       "positive_train": int(labels[fit].sum()), "negative_train": int((labels[fit] == 0).sum()),
                       "hard_clean_claims_redistributed": changed, "microbatches": len(batches),
                       "optimizer_updates": math.ceil(len(batches) / ACCUM_MICROBATCHES),
                       "logical_tokens": logical, "padded_tokens": padded,
                       "padding_ratio": padded / logical})
    length_summary = {"min": int(lengths.min()), "median": float(np.median(lengths)),
                      "p90": float(np.quantile(lengths, .90)), "p95": float(np.quantile(lengths, .95)),
                      "p99": float(np.quantile(lengths, .99)), "max": int(lengths.max()),
                      "over_768": int((lengths > MAX_LENGTH).sum())}
    prior = ROOT / "results/relation_pair_fast_v1/relation_rank025/complete.json"
    runtime_basis = None
    if prior.exists():
        old = q.read(prior); epochs = old["all_QA_epochs"]
        train_rate = sum(e["training"]["logical_input_tokens"] for e in epochs) / sum(e["training"]["seconds"] for e in epochs)
        eval_seconds = sum(e["seconds"] - e["training"]["seconds"] for e in epochs)
        eval_rate = sum(e["evaluation_logical_input_tokens"] for e in epochs) / eval_seconds
        total_train = sum(x["logical_tokens"] for x in batch_plan)
        total_eval = int(2 * lengths.sum())  # fit OOF once + full-fit cal once + frozen control all
        point = total_train / train_rate + total_eval / eval_rate + 5 * 50 + 120
        runtime_basis = {"path": str(prior.resolve()), "sha256": sha(prior),
                         "same_host_train_logical_tokens_per_second": train_rate,
                         "same_host_eval_logical_tokens_per_second": eval_rate,
                         "candidate_train_logical_tokens": total_train,
                         "candidate_eval_logical_tokens_including_control": total_eval,
                         "point_minutes_before_own_smoke": math.ceil(point / 60),
                         "conservative_minutes_before_own_smoke": math.ceil((point * 1.4 + 300) / 60)}
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_jsonl(OUT / "records.jsonl", records)
    save_npz(OUT / "encoded_inputs.npz", flat_input_ids=flat, bounds=bounds, lengths=lengths,
             labels=labels, raw_nli_risk=raw_risk, fold_assignment=assignments,
             weights=weight_matrix)
    preparation = {"status": "CPU_prepared_not_trained", "answers": len(rows), "claims": len(records),
                   "answers_by_partition": dict(Counter(row["partition"] for row in rows)),
                   "claims_by_partition": dict(Counter(partitions.tolist())),
                   "positive_claims": {part: int(labels[partitions == part].sum()) for part in PARTITIONS},
                   "fit_groups": len(set(groups[fit])), "calibration_groups": len(set(groups[cal])),
                   "fit_cal_group_overlap": 0, "fit_cal_source_overlap": 0,
                   "input_lengths": length_summary, "fold_leakage_audit": leakage,
                   "fold_batch_plan": batch_plan, "runtime_basis": runtime_basis,
                   "estimated_checkpoint_bytes_each": int((MODEL / "model.safetensors").stat().st_size),
                   "estimated_six_checkpoint_bytes": int(6 * (MODEL / "model.safetensors").stat().st_size),
                   "free_disk_bytes": shutil.disk_usage(OUT).free,
                   "source_sha256": snapshot, "model_sha256": MODEL_SHA256,
                   "labels_used_for_training_manifest": True, "calibration_labels_used_for_training": False,
                   "GPU_used": False, "official_test_opened": False}
    save_json(OUT / "preparation.json", preparation)
    names = ["protocol.json", "records.jsonl", "encoded_inputs.npz", "preparation.json"]
    save_json(OUT / "preparation_complete.json", {"status": "CPU_prepared_not_trained",
              "files_sha256": {name: sha(OUT / name) for name in names},
              "source_sha256": snapshot, "GPU_used": False, "official_test_opened": False})
    (OUT / "PLAN.md").write_text(
        "# Microclaim full-evidence cross-encoder v1\n\n"
        "主候选直接微调 ModernBERT-base-nli：输入为问题、三篇完整检索资料和一个原子微主张；输出该微主张的事实风险。五折按 source-connected group 严格隔离。"
        "训练损失在每个回答内部提高冻结 NLI 高风险但 gold 干净的微主张权重，并保持该回答的干净样本总权重不变，用来压低旧 NLI 的新增误报而不削弱正例总权重。\n\n"
        "唯一控制是不训练同一 NLI 模型，直接用 1-P(entailment)。统一映射回 4-BPE 窗口；fit OOF 定阈值，calibration 只报告。正式 baseline 和 official test 不动。\n",
        encoding="utf-8")
    assert snapshot == source_files() and not torch.cuda.is_initialized()
    print("MICROCLAIM_CROSSENCODER_CPU_PREPARED", len(records), length_summary, flush=True)


def check_prepared():
    assert not torch.cuda.is_initialized()
    complete = q.read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_prepared_not_trained"
    for name, value in complete["files_sha256"].items():
        assert sha(OUT / name) == value, name
    assert q.read(OUT / "protocol.json") == protocol()
    assert complete["source_sha256"] == source_files()
    records = q.lines(OUT / "records.jsonl"); arrays = load_arrays()
    n = sum(EXPECTED_CLAIMS.values())
    assert len(records) == n and arrays["bounds"].shape == (n, 2)
    assert arrays["lengths"].max() <= MAX_LENGTH and arrays["flat_input_ids"].dtype == np.int32
    assert np.array_equal(arrays["bounds"][:, 1] - arrays["bounds"][:, 0], arrays["lengths"])
    assert np.array_equal(arrays["labels"], np.asarray([row["label"] for row in records], np.int8))
    assert set(arrays["fold_assignment"][:EXPECTED_CLAIMS["fit"]]) == set(range(FOLDS))
    assert np.all(arrays["fold_assignment"][EXPECTED_CLAIMS["fit"]:] == -1)
    assert arrays["weights"].shape == (FOLDS + 1, n)
    for fold in range(FOLDS):
        train = np.flatnonzero((arrays["fold_assignment"] != fold) & (arrays["fold_assignment"] >= 0))
        held = np.flatnonzero(arrays["fold_assignment"] == fold)
        assert np.all(arrays["weights"][fold, train] > 0)
        assert np.all(arrays["weights"][fold, held] == 0)
        assert np.all(arrays["weights"][fold, EXPECTED_CLAIMS["fit"]:] == 0)
    assert np.all(arrays["weights"][FOLDS, :EXPECTED_CLAIMS["fit"]] > 0)
    assert np.all(arrays["weights"][FOLDS, EXPECTED_CLAIMS["fit"]:] == 0)
    return records, arrays


def collate(arrays, indices, device):
    indices = list(map(int, indices)); lengths = arrays["lengths"][indices].astype(int)
    width = int(lengths.max()); pad = 50283
    ids = np.full((len(indices), width), pad, dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=np.int64)
    for row, (index, length) in enumerate(zip(indices, lengths)):
        left, right = arrays["bounds"][index]
        ids[row, :length] = arrays["flat_input_ids"][left:right]
        mask[row, :length] = 1
    return {"input_ids": torch.as_tensor(ids, device=device),
            "attention_mask": torch.as_tensor(mask, device=device)}, lengths


def risk_logit(logits):
    assert logits.ndim == 2 and logits.shape[1] == 3
    values = torch.logsumexp(logits[:, 1:3].float(), dim=-1) - logits[:, 0].float()
    assert values.dtype == torch.float32 and torch.isfinite(values).all()
    return values


def forward_risk(model, batch, device, training):
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with context:
        logits = model(**batch).logits
    risk = risk_logit(logits)
    return risk, str(logits.dtype)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_epoch(model, optimizer, arrays, indices, weights, fold, device, batches_override=None, progress=True):
    batches = batches_override or make_batches(indices, arrays["lengths"], SEED + fold,
                                                TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
    model.train(); started = time.perf_counter(); updates = examples = logical = padded = 0
    weighted_loss = weighted_mass = 0.0; gradient_norms = []; dtypes = set()
    for first in range(0, len(batches), ACCUM_MICROBATCHES):
        group = batches[first:first + ACCUM_MICROBATCHES]
        denominator = float(sum(weights[index] for batch in group for index in batch))
        assert denominator > 0
        optimizer.zero_grad(set_to_none=True)
        for indices_one in group:
            encoded, lengths = collate(arrays, indices_one, device)
            risk, dtype = forward_risk(model, encoded, device, True); dtypes.add(dtype)
            y = torch.as_tensor(arrays["labels"][indices_one], device=device, dtype=torch.float32)
            w = torch.as_tensor(weights[indices_one], device=device, dtype=torch.float32)
            losses = F.binary_cross_entropy_with_logits(risk, y, reduction="none")
            numerator = (losses * w).sum()
            (numerator / denominator).backward()
            weighted_loss += float(numerator.detach()); weighted_mass += float(w.sum())
            examples += len(indices_one); logical += int(lengths.sum())
            padded += int(len(indices_one) * max(lengths))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step(); updates += 1; gradient_norms.append(float(norm))
        if progress and (updates % 100 == 0 or first + len(group) == len(batches)):
            print("MICROCLAIM_CROSSENCODER_TRAIN", fold, updates,
                  math.ceil(len(batches) / ACCUM_MICROBATCHES), round(time.perf_counter() - started, 1), flush=True)
    assert examples == len(indices)
    return {"fold": fold, "epochs": 1, "examples": examples, "microbatches": len(batches),
            "optimizer_updates": updates, "logical_tokens": logical, "padded_tokens": padded,
            "weighted_mean_online_bce": weighted_loss / weighted_mass,
            "gradient_norm_min": min(gradient_norms), "gradient_norm_max": max(gradient_norms),
            "forward_logits_dtypes": sorted(dtypes), "seconds": time.perf_counter() - started}


def score_indices(model, arrays, indices, device, progress=False):
    batches = make_batches(indices, arrays["lengths"], SEED + 9000,
                           EVAL_TOKEN_BUDGET, EVAL_MAX_EXAMPLES)
    scores = np.empty(len(indices), dtype=np.float32); positions = {int(v): j for j, v in enumerate(indices)}
    model.eval(); logical = padded = 0; started = time.perf_counter(); dtypes = set()
    with torch.inference_mode():
        for bi, one in enumerate(batches):
            encoded, lengths = collate(arrays, one, device)
            risk, dtype = forward_risk(model, encoded, device, False); dtypes.add(dtype)
            prob = torch.sigmoid(risk).cpu().numpy().astype(np.float32)
            for index, value in zip(one, prob): scores[positions[index]] = value
            logical += int(lengths.sum()); padded += int(len(one) * max(lengths))
            if progress and ((bi + 1) % 100 == 0 or bi + 1 == len(batches)):
                print("MICROCLAIM_CROSSENCODER_EVAL", bi + 1, len(batches), flush=True)
    assert np.isfinite(scores).all()
    return scores, {"examples": len(indices), "microbatches": len(batches), "logical_tokens": logical,
                    "padded_tokens": padded, "forward_logits_dtypes": sorted(dtypes),
                    "seconds": time.perf_counter() - started}


def tiny_config(vocab_size):
    config = ModernBertConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              max_position_embeddings=MAX_LENGTH, local_attention=32,
                              pad_token_id=50283, bos_token_id=50281, eos_token_id=50282,
                              sep_token_id=50282, num_labels=3, classifier_pooling="mean",
                              reference_compile=False)
    config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    config._attn_implementation = "sdpa"
    return config


def cpu_tiny():
    records, arrays = check_prepared(); assert not (OUT / "CPU_TINY_TRAIN.json").exists()
    torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    fold = 0; active = np.flatnonzero((arrays["fold_assignment"] != fold) & (arrays["fold_assignment"] >= 0))
    positive = [int(i) for i in active if arrays["labels"][i] == 1][:4]
    hard_clean = sorted((int(i) for i in active if arrays["labels"][i] == 0),
                        key=lambda i: (-float(arrays["raw_nli_risk"][i]), i))[:8]
    selected = np.asarray(positive + hard_clean, dtype=np.int64)
    assert len(selected) == 12 and set(arrays["labels"][selected]) == {0, 1}
    model = ModernBertForSequenceClassification(tiny_config(50368)).float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = optimizer_for(model)
    batches = make_batches(selected, arrays["lengths"], SEED, TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
    report = train_epoch(model, optimizer, arrays, selected, arrays["weights"][fold], fold,
                         torch.device("cpu"), batches_override=batches, progress=False)
    changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value)]
    assert changed and any(name.startswith("model.layers") for name in changed)
    sample_logits = torch.tensor([[2., 1., 0.], [0., 1., 2.]], dtype=torch.float32)
    expected = torch.log(torch.tensor([math.exp(1.) + math.exp(0.), math.exp(1.) + math.exp(2.)])) - torch.tensor([2., 0.])
    assert torch.allclose(risk_logit(sample_logits), expected)
    result = {"status": "passed", "production_loop_tiny_ModernBERT": True,
              "actual_manifest_examples": selected.tolist(), "labels": arrays["labels"][selected].tolist(),
              "hard_clean_examples": hard_clean, "train_report": report,
              "changed_parameter_tensors": len(changed), "encoder_parameters_changed": True,
              "risk_logit_identity_checked": True, "real_pretrained_model_loaded": False,
              "GPU_used": False, "official_test_opened": False,
              "preparation_sha256": sha(OUT / "preparation_complete.json")}
    save_json(OUT / "CPU_TINY_TRAIN.json", result)
    assert not torch.cuda.is_initialized()
    print("MICROCLAIM_CROSSENCODER_CPU_TINY_PASSED", report["optimizer_updates"], flush=True)


def configure_gpu(seed):
    assert torch.cuda.is_available(); torch.set_num_threads(THREADS); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, total = torch.cuda.mem_get_info(0); assert free >= MIN_FREE, f"Insufficient free GPU: {free}"
    return free, total


def load_model(device):
    assert sha(MODEL / "model.safetensors") == MODEL_SHA256
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False).to(device)
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert sum(p.numel() for p in model.parameters()) == 149607171
    return model


def cleanup_gpu(*objects):
    del objects; gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def gpu_smoke():
    _, arrays = check_prepared(); tiny = q.read(OUT / "CPU_TINY_TRAIN.json")
    assert tiny["status"] == "passed" and not (OUT / "GPU_SMOKE.json").exists()
    free, total = configure_gpu(SEED); device = torch.device("cuda:0")
    model = optimizer = None; started = time.perf_counter()
    try:
        model = load_model(device); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        fold = 0; train = np.flatnonzero((arrays["fold_assignment"] != fold) & (arrays["fold_assignment"] >= 0))
        all_batches = make_batches(train, arrays["lengths"], SEED + fold, TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
        ranked = sorted(all_batches, key=lambda b: max(int(arrays["lengths"][j]) for j in b) * len(b), reverse=True)
        smoke_batches = [ranked[0], ranked[len(ranked)//3], ranked[2*len(ranked)//3], ranked[-1]]
        smoke_indices = np.asarray([j for batch in smoke_batches for j in batch], dtype=np.int64)
        # Repeatability before the single disposable optimizer update.
        model.eval(); encoded, _ = collate(arrays, smoke_batches[0], device)
        with torch.inference_mode():
            a, _ = forward_risk(model, encoded, device, False); b, _ = forward_risk(model, encoded, device, False)
        repeat = float((a - b).abs().max()); assert repeat <= 2e-6
        del encoded, a, b
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = time.perf_counter()
        train_report = train_epoch(model, optimizer, arrays, smoke_indices, arrays["weights"][fold], fold,
                                   device, batches_override=smoke_batches, progress=False)
        torch.cuda.synchronize(); train_seconds = time.perf_counter() - tick
        eval_indices = np.asarray([j for batch in ranked[:8] for j in batch], dtype=np.int64)
        torch.cuda.synchronize(); tick = time.perf_counter()
        _, eval_report = score_indices(model, arrays, eval_indices, device)
        torch.cuda.synchronize(); eval_seconds = time.perf_counter() - tick
        peak_alloc = torch.cuda.max_memory_allocated(); peak_reserved = torch.cuda.max_memory_reserved()
        assert max(peak_alloc, peak_reserved) <= PEAK_LIMIT
        prep = q.read(OUT / "preparation.json"); total_train = sum(x["padded_tokens"] for x in prep["fold_batch_plan"])
        total_train_updates = sum(x["optimizer_updates"] for x in prep["fold_batch_plan"])
        train_per_padded = train_seconds / train_report["padded_tokens"]
        # Optimizer-step overhead is represented by the exact one-update smoke.
        train_est = total_train * train_per_padded
        fit_n = EXPECTED_CLAIMS["fit"]
        cal_indices = np.arange(fit_n, len(arrays["lengths"]))
        fit_indices = np.arange(fit_n)
        eval_sets = [np.flatnonzero(arrays["fold_assignment"] == fold) for fold in range(FOLDS)]
        eval_sets += [cal_indices, np.arange(len(arrays["lengths"]))]
        total_eval_padded = 0
        for values in eval_sets:
            batches = make_batches(values, arrays["lengths"], SEED + 9000, EVAL_TOKEN_BUDGET, EVAL_MAX_EXAMPLES)
            total_eval_padded += sum(max(int(arrays["lengths"][j]) for j in one) * len(one) for one in batches)
        eval_per_padded = eval_seconds / eval_report["padded_tokens"]
        point = train_est + total_eval_padded * eval_per_padded + 5 * 50 + 120
        report = {"status": "passed_no_formal_training", "device": torch.cuda.get_device_name(0),
                  "free_before_load": free, "total_memory": total, "parameters": sum(p.numel() for p in model.parameters()),
                  "smoke_examples": smoke_indices.tolist(), "smoke_lengths": arrays["lengths"][smoke_indices].tolist(),
                  "repeat_max_abs_difference": repeat, "exact_production_accumulation_update": True,
                  "train_report": train_report, "measured_train_seconds": train_seconds,
                  "eval_report": eval_report, "measured_eval_seconds": eval_seconds,
                  "peak_cuda_allocated_bytes": peak_alloc, "peak_cuda_reserved_bytes": peak_reserved,
                  "planned_train_optimizer_updates": total_train_updates,
                  "planned_train_padded_tokens": total_train, "planned_eval_padded_tokens": total_eval_padded,
                  "point_runtime_minutes_from_smoke": math.ceil(point / 60),
                  "conservative_runtime_minutes_from_smoke": math.ceil((point * 1.35 + 300) / 60),
                  "checkpoint_disk_estimate_bytes": prep["estimated_six_checkpoint_bytes"],
                  "formal_parameters_saved": False, "official_test_opened": False,
                  "preparation_sha256": sha(OUT / "preparation_complete.json"),
                  "seconds": time.perf_counter() - started}
        save_json(OUT / "GPU_SMOKE.json", report)
        print("MICROCLAIM_CROSSENCODER_GPU_SMOKE_PASSED", report["point_runtime_minutes_from_smoke"],
              report["conservative_runtime_minutes_from_smoke"], flush=True)
    finally:
        del model, optimizer; cleanup_gpu()


def save_checkpoint(model, path, fold):
    assert not path.exists()
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    torch.save({"model_state_dict": state, "fold": fold, "seed": SEED + fold,
                "preparation_sha256": sha(OUT / "preparation_complete.json"),
                "protocol_sha256": sha(OUT / "protocol.json")}, path)


def train_fold(fold):
    assert 0 <= fold < FOLDS
    _, arrays = check_prepared(); smoke = q.read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_formal_training"
    directory = OUT / f"fold_{fold}"; assert not directory.exists(), f"Preserve existing fold: {directory}"
    directory.mkdir(parents=True)
    save_json(directory / "started.json", {"fold": fold, "seed": SEED + fold,
              "preparation_sha256": sha(OUT / "preparation_complete.json"),
              "calibration_labels_used": False, "official_test_opened": False, "time": time.time()})
    free, total = configure_gpu(SEED + fold); device = torch.device("cuda:0")
    model = optimizer = None; started = time.perf_counter()
    try:
        train = np.flatnonzero((arrays["fold_assignment"] != fold) & (arrays["fold_assignment"] >= 0))
        held = np.flatnonzero(arrays["fold_assignment"] == fold)
        model = load_model(device); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats()
        training = train_epoch(model, optimizer, arrays, train, arrays["weights"][fold], fold, device)
        held_scores, held_eval = score_indices(model, arrays, held, device, progress=True)
        checkpoint = directory / "model.pt"; save_checkpoint(model, checkpoint, fold)
        save_npz(directory / "predictions.npz", held_indices=held, held_scores=held_scores)
        result = {"status": "fold_complete", "fold": fold, "seed": SEED + fold,
                  "train_claims": len(train), "held_claims": len(held),
                  "training": training, "held_inference": held_eval,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "free_before_load": free, "total_memory": total,
                  "checkpoint_sha256": sha(checkpoint),
                  "predictions_sha256": sha(directory / "predictions.npz"),
                  "calibration_labels_used": False, "threshold_selected": False,
                  "official_test_opened": False, "seconds": time.perf_counter() - started}
        save_json(directory / "complete.json", result)
        print("MICROCLAIM_CROSSENCODER_FOLD_COMPLETE", fold, round(result["seconds"], 1), flush=True)
    finally:
        del model, optimizer; cleanup_gpu()


def train_full():
    _, arrays = check_prepared(); smoke = q.read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_no_formal_training"
    directory = OUT / "full_fit"; assert not directory.exists(), f"Preserve existing full fit: {directory}"
    directory.mkdir(parents=True)
    save_json(directory / "started.json", {"role": "full_fit_for_calibration", "seed": SEED + FOLDS,
              "preparation_sha256": sha(OUT / "preparation_complete.json"),
              "calibration_labels_used": False, "official_test_opened": False, "time": time.time()})
    free, total = configure_gpu(SEED + FOLDS); device = torch.device("cuda:0")
    model = optimizer = None; started = time.perf_counter()
    try:
        fit = np.flatnonzero(arrays["fold_assignment"] >= 0)
        cal = np.flatnonzero(arrays["fold_assignment"] == -1)
        model = load_model(device); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats()
        training = train_epoch(model, optimizer, arrays, fit, arrays["weights"][FOLDS], FOLDS, device)
        cal_scores, cal_eval = score_indices(model, arrays, cal, device, progress=True)
        checkpoint = directory / "model.pt"; save_checkpoint(model, checkpoint, FOLDS)
        save_npz(directory / "predictions.npz", calibration_indices=cal, calibration_scores=cal_scores)
        result = {"status": "full_fit_complete", "seed": SEED + FOLDS,
                  "train_claims": len(fit), "calibration_claims": len(cal),
                  "training": training, "calibration_inference": cal_eval,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "free_before_load": free, "total_memory": total,
                  "checkpoint_sha256": sha(checkpoint),
                  "predictions_sha256": sha(directory / "predictions.npz"),
                  "calibration_labels_used": False, "threshold_selected": False,
                  "official_test_opened": False, "seconds": time.perf_counter() - started}
        save_json(directory / "complete.json", result)
        print("MICROCLAIM_CROSSENCODER_FULL_FIT_COMPLETE", round(result["seconds"], 1), flush=True)
    finally:
        del model, optimizer; cleanup_gpu()


def control():
    _, arrays = check_prepared(); assert q.read(OUT / "GPU_SMOKE.json")["status"] == "passed_no_formal_training"
    directory = OUT / "frozen_control"; assert not directory.exists(); directory.mkdir()
    free, total = configure_gpu(SEED); device = torch.device("cuda:0"); model = None
    try:
        model = load_model(device).eval().requires_grad_(False); torch.cuda.reset_peak_memory_stats()
        indices = np.arange(len(arrays["labels"]), dtype=np.int64)
        scores, inference = score_indices(model, arrays, indices, device, progress=True)
        save_npz(directory / "scores.npz", claim_scores=scores)
        save_json(directory / "complete.json", {"status": "frozen_control_complete", "inference": inference,
                  "free_before_load": free, "total_memory": total,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "scores_sha256": sha(directory / "scores.npz"), "trained": False,
                  "official_test_opened": False})
        print("MICROCLAIM_CROSSENCODER_FROZEN_CONTROL_COMPLETE", flush=True)
    finally:
        del model; cleanup_gpu()


def thresholds_from_fit(meta, window_scores):
    left, right = meta["bounds"]["fit"]
    answer_scores = q.answer_scores(meta, window_scores)
    answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == "fit"]
    return {"window": q.choose_threshold([w["label"] for w in meta["windows"][left:right]], window_scores[left:right]),
            "answer": q.choose_threshold([meta["answers"][i]["label"] for i in answer_indices], answer_scores[answer_indices])}


def cal_opt(meta, window_scores):
    left, right = meta["bounds"]["calibration"]
    answer_scores = q.answer_scores(meta, window_scores)
    answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == "calibration"]
    thresholds = {"window": q.choose_threshold([w["label"] for w in meta["windows"][left:right]], window_scores[left:right]),
                  "answer": q.choose_threshold([meta["answers"][i]["label"] for i in answer_indices], answer_scores[answer_indices])}
    return {"thresholds": thresholds, "metrics": q.metrics(meta, window_scores, thresholds)["calibration"]}


def assemble_main(arrays):
    n = len(arrays["labels"]); scores = np.full(n, np.nan, dtype=np.float32)
    for fold in range(FOLDS):
        directory = OUT / f"fold_{fold}"; complete = q.read(directory / "complete.json")
        assert complete["status"] == "fold_complete" and complete["checkpoint_sha256"] == sha(directory / "model.pt")
        assert complete["predictions_sha256"] == sha(directory / "predictions.npz")
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            held, values = z["held_indices"], z["held_scores"]
            assert np.all(arrays["fold_assignment"][held] == fold) and np.isnan(scores[held]).all()
            scores[held] = values
    fit_n = EXPECTED_CLAIMS["fit"]
    assert np.isfinite(scores[:fit_n]).all() and np.isnan(scores[fit_n:]).all()
    full = OUT / "full_fit"; complete = q.read(full / "complete.json")
    assert complete["status"] == "full_fit_complete" and complete["checkpoint_sha256"] == sha(full / "model.pt")
    assert complete["predictions_sha256"] == sha(full / "predictions.npz")
    with np.load(full / "predictions.npz", allow_pickle=False) as z:
        cal_indices, cal_scores = z["calibration_indices"], z["calibration_scores"]
        assert np.array_equal(cal_indices, np.arange(fit_n, n))
        scores[cal_indices] = cal_scores
    assert np.isfinite(scores).all()
    return scores


def method_summary(meta, rows, offsets, claim_scores):
    window = atomic_nli.project(meta, rows, offsets, claim_scores)
    thresholds = thresholds_from_fit(meta, window)
    return window, {"thresholds_from_fit_OOF": thresholds,
                    "strict_fit_threshold_to_calibration": q.metrics(meta, window, thresholds),
                    "common_calibration_F1Opt_diagnostic": cal_opt(meta, window)}


def finalize():
    records, arrays = check_prepared(); assert not (OUT / "summary.json").exists()
    rows = q.lines(ATOMIC_INPUT); meta = q.metadata()
    assert [row["response_id"] for row in rows] == [answer["response_id"] for answer in meta["answers"]]
    offsets = {}; cursor = 0
    for row in rows:
        offsets[row["response_id"]] = cursor; cursor += len(row["claims"])
    assert cursor == len(records)
    main_claim = assemble_main(arrays)
    control_complete = q.read(OUT / "frozen_control/complete.json")
    assert control_complete["status"] == "frozen_control_complete"
    assert control_complete["scores_sha256"] == sha(OUT / "frozen_control/scores.npz")
    with np.load(OUT / "frozen_control/scores.npz", allow_pickle=False) as z:
        control_claim = z["claim_scores"].copy()
    main_window, main = method_summary(meta, rows, offsets, main_claim)
    control_window, frozen = method_summary(meta, rows, offsets, control_claim)
    main_answer = q.answer_scores(meta, main_window); control_answer = q.answer_scores(meta, control_window)
    save_npz(OUT / "scores.npz", main_claim_scores=main_claim, main_window_scores=main_window,
             main_answer_scores=main_answer, frozen_claim_scores=control_claim,
             frozen_window_scores=control_window, frozen_answer_scores=control_answer)
    summary = {"status": "development_only_complete", "method": "gold-supervised full-evidence NLI microclaim cross-encoder",
               "trained_candidate": main, "minimal_frozen_control": frozen,
               "incumbent_read_only_reference": q.read(INCUMBENT)["metrics"]["calibration"],
               "atomic_NLI_read_only_reference": q.read(ATOMIC / "summary.json")["strict_fit_threshold_to_calibration"]["calibration"],
               "five_source_group_OOF_models": True, "separate_full_fit_calibration_model": True,
               "calibration_used_for_training_or_selection": False, "fit_only_thresholds": True,
               "formal_baselines_modified": False, "official_test_opened": False, "final_test_claim": False,
               "scores_sha256": sha(OUT / "scores.npz"),
               "fold_files": {**{f"fold_{fold}": sha(OUT / f"fold_{fold}/complete.json") for fold in range(FOLDS)},
                              "full_fit": sha(OUT / "full_fit/complete.json")}}
    save_json(OUT / "summary.json", summary)
    m = main["strict_fit_threshold_to_calibration"]; c = frozen["strict_fit_threshold_to_calibration"]
    report = ["# Microclaim full-evidence cross-encoder v1", "",
              "主候选直接微调原始三分类 ModernBERT NLI；输入包含问题、三篇完整资料和单个原子微主张。fit 为 source-connected 五折 OOF，cal 由独立的 full-fit 模型预测。", "",
              "| 方法 | fit OOF窗口F1 | cal严格窗口F1 | fit OOF整答F1 | cal严格整答F1 |",
              "|---|---:|---:|---:|---:|",
              f"| fp-aware full-evidence | {m['fit']['windows']['f1']:.6f} | {m['calibration']['windows']['f1']:.6f} | {m['fit']['answers']['f1']:.6f} | {m['calibration']['answers']['f1']:.6f} |",
              f"| frozen full-evidence NLI | {c['fit']['windows']['f1']:.6f} | {c['calibration']['windows']['f1']:.6f} | {c['fit']['answers']['f1']:.6f} | {c['calibration']['answers']['f1']:.6f} |", "",
              "统一4-BPE窗口和fit阈值未改；cal-F1Opt仅为共同开发诊断。正式baseline未改，official test未打开。", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    save_json(OUT / "complete.json", {"status": "complete_development_only",
              "summary_sha256": sha(OUT / "summary.json"), "scores_sha256": sha(OUT / "scores.npz"),
              "report_sha256": sha(OUT / "REPORT.md"), "official_test_opened": False})
    print("MICROCLAIM_CROSSENCODER_FINALIZED", m["calibration"]["windows"]["f1"],
          m["calibration"]["answers"]["f1"], flush=True)


def verify(final=False):
    records, arrays = check_prepared()
    assert len(records) == len(arrays["labels"]) == 11322
    assert q.read(OUT / "CPU_TINY_TRAIN.json")["status"] == "passed"
    if (OUT / "GPU_SMOKE.json").exists():
        assert q.read(OUT / "GPU_SMOKE.json")["status"] == "passed_no_formal_training"
    if final:
        complete = q.read(OUT / "complete.json")
        assert complete["status"] == "complete_development_only"
        for name in ("summary", "scores", "report"):
            suffix = ".md" if name == "report" else ".json" if name == "summary" else ".npz"
            assert sha(OUT / (name.upper() + suffix if name == "report" else name + suffix)) == complete[name + "_sha256"]
        assert not q.read(OUT / "summary.json")["formal_baselines_modified"]
    print("MICROCLAIM_CROSSENCODER_VERIFIED", "final" if final else "prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "check", "cpu-tiny", "gpu-smoke",
                                          "train-fold", "train-full", "control", "finalize", "verify", "verify-final"])
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.stage == "train-fold" and args.fold is None:
        parser.error("train-fold requires --fold 0..4")
    if args.stage != "train-fold" and args.fold is not None:
        parser.error("--fold is only valid with train-fold")
    if args.stage == "prepare": prepare()
    elif args.stage == "check": check_prepared(); print("MICROCLAIM_CROSSENCODER_CPU_CHECK_PASSED", flush=True)
    elif args.stage == "cpu-tiny": cpu_tiny()
    elif args.stage == "gpu-smoke": gpu_smoke()
    elif args.stage == "train-fold": train_fold(args.fold)
    elif args.stage == "train-full": train_full()
    elif args.stage == "control": control()
    elif args.stage == "finalize": finalize()
    elif args.stage == "verify": verify(False)
    else: verify(True)


if __name__ == "__main__":
    main()
