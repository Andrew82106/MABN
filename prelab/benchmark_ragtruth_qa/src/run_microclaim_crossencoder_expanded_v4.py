"""Expanded v4 microclaim cross-encoder, reusing the frozen v1 training loop.

This is a thin data/evaluation adapter around run_microclaim_crossencoder_v1.
The model, risk logit, optimizer, one-epoch loop, batching, checkpointing and
GPU smoke are reused from that runner.  The adapter binds them to the audited
expanded-v4 examples, source-group folds, weights and 4-BPE projection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_microclaim_crossencoder_v1 as base  # noqa: E402


OUT = ROOT / "results/microclaim_crossencoder_expanded_v4"
V4 = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_EXAMPLES = V4 / "examples.jsonl"
V4_ANSWERS = V4 / "answers.jsonl"
V4_ARRAYS = V4 / "arrays.npz"
V4_MANIFEST = V4 / "manifest.json"
V4_COMPLETE = V4 / "complete.json"
V4_AUDIT = V4 / "INDEPENDENT_AUDIT.json"
V4_MANIFEST_SHA256 = "35791a3b2881b098713d947273d7293f912f6f1a84ed2a50bedfe022a2c74370"
V4_COMPLETE_SHA256 = "d6d6658a6aa82cc10fd56116ebd938b92b896155d85c706c2282939f29cd6c4e"
BASE_RUNNER_SHA256 = "aaf08424d7f8b129bd21ae11100f0c13a0f6bcc025edbe8805b6c1a00c9ebaa8"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
EXPECTED_ANSWERS = {"fit": 3680, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 34919, "calibration": 2267}
EXPECTED_WINDOWS = {"fit": 653979, "calibration": 42241}
EXPECTED_POSITIVE_WINDOWS = {"fit": 58433, "calibration": 5984}
FOLDS = 5
SEED = base.SEED
MAX_LENGTH = base.MAX_LENGTH
EPOCHS = 1


def sha(path):
    return base.sha(path)


def lines(path):
    return base.q.lines(path)


def save_json(path, value):
    return base.save_json(path, value)


def save_jsonl(path, rows):
    return base.save_jsonl(path, rows)


def save_npz(path, **arrays):
    return base.save_npz(path, **arrays)


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_v4():
    assert sha(V4_MANIFEST) == V4_MANIFEST_SHA256
    assert sha(V4_COMPLETE) == V4_COMPLETE_SHA256
    manifest = json.loads(V4_MANIFEST.read_text(encoding="utf-8"))
    complete = json.loads(V4_COMPLETE.read_text(encoding="utf-8"))
    audit = json.loads(V4_AUDIT.read_text(encoding="utf-8"))
    assert manifest["version"] == "atomic-microclaim-relation-expanded-v4"
    assert complete["status"] == "complete_CPU_preparation_and_audit"
    assert complete["manifest_sha256"] == V4_MANIFEST_SHA256
    assert complete["independent_audit_sha256"] == sha(V4_AUDIT)
    assert audit["manifest_sha256"] == V4_MANIFEST_SHA256
    assert audit["status"] == "independent_CPU_audit_passed" and audit["examples"] == sum(EXPECTED_CLAIMS.values())
    for name, expected in manifest["files_sha256"].items():
        assert sha(V4 / name) == expected, name
    return manifest, complete, audit


def source_files():
    paths = (
        Path(__file__), Path(base.__file__), Path(base.q.__file__), Path(base.atomic_nli.__file__),
        V4_MANIFEST, V4_COMPLETE, V4_AUDIT, V4_EXAMPLES, V4_ANSWERS, V4_ARRAYS,
        MODEL / "config.json", MODEL / "tokenizer.json", MODEL / "tokenizer_config.json",
        MODEL / "special_tokens_map.json", MODEL / "model.safetensors",
        MODEL / "download_manifest.json", base.INCUMBENT,
    )
    return {str(path.resolve()): sha(path) for path in paths}


def protocol():
    return {
        "version": "microclaim-crossencoder-expanded-v4-v1",
        "scope": "Frozen expanded-v4 fit3680 plus unchanged calibration159; official test absent.",
        "data_binding": {"v4_manifest_sha256": V4_MANIFEST_SHA256,
                         "v4_complete_sha256": V4_COMPLETE_SHA256,
                         "base_runner_sha256": BASE_RUNNER_SHA256},
        "unit": "One audited factual atomic microclaim: 34,919 fit and exactly the reused 2,267 calibration microclaims.",
        "input": {
            "premise": "Exact frozen v4 question plus claim-selected top-2 evidence independently from each of passages 1, 2 and 3.",
            "hypothesis": "Exact frozen v4 contextualized atomic microclaim.",
            "tokenizer": "Exact local tasksource/ModernBERT-base-nli paired tokenizer; no truncation.",
            "maximum_tokens": MAX_LENGTH,
        },
        "only_trained_candidate": {
            "name": "expanded_top_evidence_crossencoder",
            "initialization": "Exact tasksource/ModernBERT-base-nli three-way checkpoint.",
            "architecture": "Unchanged three-way NLI cross-encoder; risk logit=logsumexp(neutral, contradiction)-entailment.",
            "parameters": "All original encoder and NLI-head parameters fine-tuned; no added head.",
            "loss": "Binary BCE with the frozen v4 source-group/answer/microclaim/class weights; fold weights are recomputed by the same canonical claim_weights function on each fold's training groups.",
            "optimizer": {"name": "AdamW", "lr": base.LEARNING_RATE,
                          "weight_decay": base.WEIGHT_DECAY, "clip_norm": 1.0,
                          "epochs": EPOCHS, "scheduler": None},
            "precision": "Same production loop: FP32 parameters/optimizer, BF16 CUDA forward, non-reentrant gradient checkpointing, TF32 off.",
        },
        "frozen_control": "The untouched initialization on identical inputs may be scored as a control; it is not a second trained candidate.",
        "crossfit": "Use the exact v4 held_fold: five source-connected fit OOF models plus one full-fit model for calibration. Calibration labels never enter GPU training or selection.",
        "known_fold_caveat": "Forty-nine unrelated fit responses share the generic refusal text 'Unable to answer based on given passages.' across folds. Exact paired inputs, response IDs, source IDs and passage bodies remain disjoint. This frozen-v4 template shortcut is reported and must be discussed with results.",
        "weights": "Same v4 hierarchy and class-balancing rule; no new hard-negative redistribution or score-derived weighting.",
        "batching": {"train_padded_token_budget": base.TRAIN_TOKEN_BUDGET,
                     "train_max_examples": base.TRAIN_MAX_EXAMPLES,
                     "accumulated_microbatches": base.ACCUM_MICROBATCHES,
                     "eval_padded_token_budget": base.EVAL_TOKEN_BUDGET,
                     "eval_max_examples": base.EVAL_MAX_EXAMPLES},
        "evaluation": "Use the frozen v4 ragged microclaim-to-original-4-BPE-window mapping. Fit OOF chooses window and answer thresholds; the full-fit model predicts calibration.",
        "reuse": "Training, inference, risk-logit, optimizer, batching, checkpoint and GPU-smoke functions are called directly from run_microclaim_crossencoder_v1 after binding only constants and manifests.",
        "selection": "Exactly one trained candidate; one epoch; no grid, alternate seed, checkpoint selection or calibration selection.",
        "baselines": "Read-only; no baseline structure, parameter, threshold or score is modified.",
        "stage_gate": "prepare -> check -> cpu-tiny -> gpu-smoke -> later explicit fold/full/control runs -> finalize.",
        "GPU_authorization": "This preparation task may expose gpu-smoke but must not start smoke or formal GPU work.",
        "seed": SEED, "official_test_opened": False,
    }


def bind_base():
    """Bind the audited expanded run while reusing the v1 implementation."""
    base.OUT = OUT
    base.EXPECTED_ANSWERS = EXPECTED_ANSWERS
    base.EXPECTED_CLAIMS = EXPECTED_CLAIMS
    base.FOLDS = FOLDS
    base.SEED = SEED
    base.MAX_LENGTH = MAX_LENGTH
    base.EPOCHS = EPOCHS
    base.MODEL = MODEL
    base.source_files = source_files
    base.protocol = protocol


def load_v4_arrays():
    with np.load(V4_ARRAYS, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def canonical_fold_weights(answers, examples, labels, assignments):
    rows_by_id = {row["response_id"]: row for row in answers}
    response_ids = np.asarray([row["response_id"] for row in examples])
    fit = np.flatnonzero(assignments >= 0)
    weights = np.zeros((FOLDS + 1, len(examples)), dtype=np.float32)
    for fold in range(FOLDS):
        active = fit[assignments[fit] != fold]
        weights[fold] = base.atomic_nli.claim_weights(rows_by_id, response_ids, labels, active).astype(np.float32)
    weights[FOLDS] = base.atomic_nli.claim_weights(rows_by_id, response_ids, labels, fit).astype(np.float32)
    return weights


def split_audit_expanded(answers, examples, response_ids, groups, sources, answer_hashes,
                         passage_hashes, fit, assignments):
    """Audit the frozen v4 folds without treating a generic refusal as leakage.

    Expanded fit contains the identical boilerplate answer ``Unable to answer
    based on given passages.`` for unrelated questions.  Its answer hash spans
    folds, while the paired model inputs, responses, sources and passages do
    not.  The original runner's answer-hash assertion would therefore reject a
    harmless repeated phrase and, more importantly, conflict with the frozen
    v4 source-connected fold assignment.
    """
    input_pairs = np.asarray([row["input_pair_sha256"] for row in examples])
    answer_text_by_id = {row["response_id"]: row["original_response"] for row in answers}
    answer_hash_by_id = {row["response_id"]: row["answer_sha256"] for row in answers}
    result = []
    for fold in range(FOLDS):
        held = fit[assignments[fit] == fold]
        train = fit[assignments[fit] != fold]
        fields = {}
        for name, values in (("groups", groups), ("sources", sources),
                             ("responses", response_ids), ("input_pairs", input_pairs)):
            overlap = set(values[train]) & set(values[held])
            assert not overlap, (fold, name, list(overlap)[:3])
            fields[name + "_overlap"] = 0
        train_responses, held_responses = set(response_ids[train]), set(response_ids[held])
        train_passages = {value for rid in train_responses for value in passage_hashes[rid]}
        held_passages = {value for rid in held_responses for value in passage_hashes[rid]}
        assert not (train_passages & held_passages)
        repeated_hashes = set(answer_hashes[train]) & set(answer_hashes[held])
        repeated_responses = {rid for rid in train_responses | held_responses
                              if answer_hash_by_id[rid] in repeated_hashes}
        repeated_texts = {answer_text_by_id[rid] for rid in repeated_responses}
        assert repeated_texts <= {"Unable to answer based on given passages."}
        result.append({"fold": fold, "train_claims": len(train), "held_claims": len(held),
                       "train_groups": len(set(groups[train])), "held_groups": len(set(groups[held])),
                       **fields, "passage_body_hash_overlap": 0,
                       "generic_refusal_answer_hash_overlap": len(repeated_hashes),
                       "generic_refusal_responses_in_both_sides": len(repeated_responses)})
    return result


def batch_plan(lengths, labels, assignments, weights):
    fit = np.flatnonzero(assignments >= 0)
    rows = []
    for fold in range(FOLDS + 1):
        active = fit if fold == FOLDS else fit[assignments[fit] != fold]
        held = np.empty(0, dtype=np.int64) if fold == FOLDS else fit[assignments[fit] == fold]
        batches = base.make_batches(active, lengths, SEED + fold,
                                    base.TRAIN_TOKEN_BUDGET, base.TRAIN_MAX_EXAMPLES)
        padded = sum(max(int(lengths[index]) for index in batch) * len(batch) for batch in batches)
        rows.append({
            "fold": "full_fit" if fold == FOLDS else fold,
            "train_claims": len(active), "held_claims": len(held),
            "positive_train": int(labels[active].sum()),
            "negative_train": int((labels[active] == 0).sum()),
            "microbatches": len(batches),
            "optimizer_updates": math.ceil(len(batches) / base.ACCUM_MICROBATCHES),
            "logical_tokens": int(lengths[active].sum()), "padded_tokens": int(padded),
            "padding_ratio": float(padded / lengths[active].sum()),
            "positive_weight_mass": float(weights[fold, active][labels[active] == 1].sum()),
            "negative_weight_mass": float(weights[fold, active][labels[active] == 0].sum()),
        })
    return rows


def eval_padded_tokens(lengths, assignments):
    fit = np.flatnonzero(assignments >= 0)
    cal = np.flatnonzero(assignments == -1)
    sets = [fit[assignments[fit] == fold] for fold in range(FOLDS)]
    sets += [cal, np.arange(len(lengths), dtype=np.int64)]
    total = 0
    for indices in sets:
        batches = base.make_batches(indices, lengths, SEED + 9000,
                                    base.EVAL_TOKEN_BUDGET, base.EVAL_MAX_EXAMPLES)
        total += sum(max(int(lengths[index]) for index in batch) * len(batch) for batch in batches)
    return int(total)


def runtime_estimate(plans, lengths, assignments):
    old_smoke = ROOT / "results/microclaim_crossencoder_v1/GPU_SMOKE.json"
    old = json.loads(old_smoke.read_text(encoding="utf-8"))
    train_rate = old["train_report"]["padded_tokens"] / old["measured_train_seconds"]
    eval_rate = old["eval_report"]["padded_tokens"] / old["measured_eval_seconds"]
    train_tokens = sum(row["padded_tokens"] for row in plans)
    eval_tokens = eval_padded_tokens(lengths, assignments)
    point_seconds = train_tokens / train_rate + eval_tokens / eval_rate + 5 * 50 + 120
    return {
        "basis": "Same model/loop/host GPU smoke from microclaim_crossencoder_v1; replace after this run's own smoke.",
        "basis_path": str(old_smoke.resolve()), "basis_sha256": sha(old_smoke),
        "train_padded_tokens_per_second": train_rate,
        "eval_padded_tokens_per_second": eval_rate,
        "planned_train_padded_tokens": train_tokens,
        "planned_eval_padded_tokens": eval_tokens,
        "point_minutes_before_own_smoke": math.ceil(point_seconds / 60),
        "conservative_minutes_before_own_smoke": math.ceil((point_seconds * 1.35 + 300) / 60),
    }


def prepare():
    bind_base()
    assert not torch.cuda.is_initialized() and not OUT.exists()
    verify_v4()
    assert sha(Path(base.__file__)) == BASE_RUNNER_SHA256
    snapshot = source_files()
    examples = lines(V4_EXAMPLES)
    answers = lines(V4_ANSWERS)
    v4 = load_v4_arrays()
    n = sum(EXPECTED_CLAIMS.values())
    assert len(examples) == n and len(answers) == sum(EXPECTED_ANSWERS.values())
    assert Counter(row["partition"] for row in examples) == EXPECTED_CLAIMS
    labels = v4["labels"].astype(np.int8)
    assignments = v4["held_fold"].astype(np.int8)
    assert np.array_equal(labels, np.asarray([row["gold_label"] for row in examples], dtype=np.int8))
    assert np.array_equal(assignments, np.asarray([row["held_fold"] if row["held_fold"] is not None else -1
                                                   for row in examples], dtype=np.int8))

    # Re-tokenize the exact frozen pairs. This is the only input transformation.
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.is_fast and tokenizer.pad_token_id == 50283
    flat, bounds, lengths, records = [], [], [], []
    for left in range(0, n, 256):
        batch = examples[left:left + 256]
        encoded = tokenizer([row["premise"] for row in batch], [row["hypothesis"] for row in batch],
                            add_special_tokens=True, truncation=False, padding=False)
        for row, ids in zip(batch, encoded["input_ids"]):
            index = len(records); assert row["example_index"] == index
            begin = len(flat); flat.extend(ids); end = len(flat)
            assert len(ids) == row["input_token_length"] <= MAX_LENGTH
            bounds.append((begin, end)); lengths.append(len(ids))
            retrieval_weakness = 1.0 - max(source["selected_union_query_coverage"] for source in row["evidence"])
            records.append({
                "index": index, "partition": row["partition"], "response_id": row["response_id"],
                "source_id": row["source_id"], "group_id": row["group_id"],
                "claim_id": row["claim_id"], "microclaim_id": row["microclaim_id"],
                "input_tokens": len(ids), "premise_sha256": digest_text(row["premise"]),
                "hypothesis_sha256": digest_text(row["hypothesis"]),
                "input_pair_sha256": row["input_pair_sha256"], "label": int(labels[index]),
                "lexical_token_indices": row["lexical_bpe_indices"],
                "v4_held_fold": row["held_fold"],
                "retrieval_weakness_for_tiny_selection_only": float(retrieval_weakness),
            })
        if min(left + 256, n) % 2560 == 0 or left + 256 >= n:
            print("EXPANDED_CROSSENCODER_TOKENIZED", min(left + 256, n), n, flush=True)
    del tokenizer
    lengths = np.asarray(lengths, dtype=np.int16)
    bounds = np.asarray(bounds, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.int32)
    assert np.array_equal(lengths.astype(np.int32), v4["input_token_length"])
    assert int(lengths.max()) <= MAX_LENGTH and len(flat) == int(lengths.sum())

    # Recompute each fold with the same canonical weighting function. The full
    # fit row must reproduce the already frozen v4 weight vector.
    weights = canonical_fold_weights(answers, examples, labels, assignments)
    fit = np.flatnonzero(assignments >= 0); cal = np.flatnonzero(assignments == -1)
    assert np.allclose(weights[FOLDS], v4["fit_training_weight"].astype(np.float32), rtol=0, atol=5e-7)
    assert np.all(weights[:, cal] == 0)
    plans = batch_plan(lengths, labels, assignments, weights)

    # Reproduce the frozen group folds and independently check all leakage keys.
    groups = np.asarray([row["group_id"] for row in examples])
    replay = np.full(n, -1, dtype=np.int8)
    for fold, (_, held_local) in enumerate(GroupKFold(FOLDS).split(fit, labels[fit], groups[fit])):
        replay[fit[held_local]] = fold
    assert np.array_equal(replay, assignments)
    response_ids = np.asarray([row["response_id"] for row in examples])
    sources = np.asarray([row["source_id"] for row in examples])
    answer_hash_by_id = {row["response_id"]: row["answer_sha256"] for row in answers}
    answer_hashes = np.asarray([answer_hash_by_id[row["response_id"]] for row in examples])
    passage_hashes = {row["response_id"]: [passage["body_sha256"] for passage in row["passages"]]
                      for row in answers}
    leakage = split_audit_expanded(answers, examples, response_ids, groups, sources,
                                   answer_hashes, passage_hashes, fit, assignments)
    refusal_text = "Unable to answer based on given passages."
    refusal_ids = {row["response_id"] for row in answers
                   if row["partition"] == "fit" and row["original_response"] == refusal_text}
    refusal_examples = [row for row in examples if row["response_id"] in refusal_ids]
    assert len(refusal_ids) == len(refusal_examples) == 49
    assert all(row["hypothesis"] == refusal_text and row["gold_label"] == 0
               for row in refusal_examples)
    assert len({row["input_pair_sha256"] for row in refusal_examples}) == 49
    template_shortcut = {
        "text": refusal_text, "fit_responses": 49, "microclaims": 49,
        "negative_labels": 49, "distinct_complete_input_pair_hashes": 49,
        "complete_input_pair_hash_overlap_across_folds": 0,
        "interpretation": "Repeated hypothesis wording but no repeated question+evidence+hypothesis input; report as a possible template shortcut.",
    }
    assert [row["response_index"] for row in answers] == list(range(len(answers)))
    assert [row["partition"] for row in answers[:EXPECTED_ANSWERS["fit"]]] == ["fit"] * EXPECTED_ANSWERS["fit"]
    assert [row["partition"] for row in answers[EXPECTED_ANSWERS["fit"]:]] == ["calibration"] * EXPECTED_ANSWERS["calibration"]
    assert np.all(v4["window_claim_indptr"][1:] > v4["window_claim_indptr"][:-1])
    first_window_owners = v4["window_claim_example_index"][v4["window_claim_indptr"][:-1]]
    assert np.array_equal(v4["partition"][first_window_owners],
                          (v4["window_response_index"] >= EXPECTED_ANSWERS["fit"]).astype(np.int8))

    runtime = runtime_estimate(plans, lengths, assignments)
    checkpoint_bytes = int((MODEL / "model.safetensors").stat().st_size)
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_jsonl(OUT / "records.jsonl", records)
    save_npz(OUT / "encoded_inputs.npz", flat_input_ids=flat, bounds=bounds, lengths=lengths,
             labels=labels, raw_nli_risk=np.asarray(
                 [row["retrieval_weakness_for_tiny_selection_only"] for row in records], dtype=np.float32),
             fold_assignment=assignments, weights=weights)
    preparation = {
        "status": "CPU_prepared_not_trained", "answers": len(answers), "claims": n,
        "answers_by_partition": EXPECTED_ANSWERS, "claims_by_partition": EXPECTED_CLAIMS,
        "positive_claims": {"fit": int(labels[fit].sum()), "calibration": int(labels[cal].sum())},
        "input_lengths": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                          "p90": float(np.quantile(lengths, .9)), "p95": float(np.quantile(lengths, .95)),
                          "p99": float(np.quantile(lengths, .99)), "max": int(lengths.max()),
                          "over_768": int((lengths > MAX_LENGTH).sum())},
        "fit_groups": len(set(groups[fit])), "calibration_groups": len(set(groups[cal])),
        "fit_cal_group_overlap": len(set(groups[fit]) & set(groups[cal])),
        "fit_cal_source_overlap": len(set(sources[fit]) & set(sources[cal])),
        "fold_leakage_audit": leakage, "fold_batch_plan": plans,
        "known_generic_refusal_template_shortcut": template_shortcut,
        "runtime_basis": runtime, "estimated_checkpoint_bytes_each": checkpoint_bytes,
        "estimated_six_checkpoint_bytes": 6 * checkpoint_bytes,
        "estimated_preparation_bytes": int(sum(path.stat().st_size for path in (V4_EXAMPLES, V4_ANSWERS, V4_ARRAYS))),
        "free_disk_bytes": shutil.disk_usage(OUT).free,
        "v4_manifest_sha256": sha(V4_MANIFEST), "v4_complete_sha256": sha(V4_COMPLETE),
        "base_runner_sha256": sha(Path(base.__file__)), "source_sha256": snapshot,
        "same_v4_full_fit_weights": True, "same_v4_fold_assignment": True,
        "single_candidate": True, "epochs": EPOCHS,
        "calibration_labels_used_for_training": False, "model_loaded": False,
        "GPU_used": False, "official_test_opened": False,
        "formal_baselines_modified": False,
    }
    save_json(OUT / "preparation.json", preparation)
    names = ("protocol.json", "records.jsonl", "encoded_inputs.npz", "preparation.json")
    save_json(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_not_trained", "files_sha256": {name: sha(OUT / name) for name in names},
        "source_sha256": snapshot, "v4_manifest_sha256": V4_MANIFEST_SHA256,
        "base_runner_sha256": BASE_RUNNER_SHA256, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    })
    (OUT / "PLAN.md").write_text(
        "# Expanded-v4 microclaim cross-encoder\n\n"
        "训练器直接复用原 crossencoder 的模型、risk logit、优化器、批处理、单轮训练、推理、checkpoint 和 GPU smoke。"
        "输入绑定到冻结 v4 的 question + three-passage top-2 evidence + contextualized claim；标签、source-connected 五折、"
        "层级/类别权重和 4-BPE 回投关系也绑定 v4。\n\n"
        "当前只完成 CPU prepare/check/tiny。GPU smoke 和六个正式训练进程必须另行显式启动。official test 与正式 baseline 均不动。\n"
        "\n已知口径限制：fit 中49条无关回答使用同一句固定拒答，且均为负例；完整的 question+evidence+hypothesis 输入哈希均不同并保持跨折零重叠。正式结果须把它作为潜在模板捷径报告。\n",
        encoding="utf-8")
    assert snapshot == source_files() and not torch.cuda.is_initialized()
    print("EXPANDED_CROSSENCODER_CPU_PREPARED", n, preparation["input_lengths"], flush=True)


def check():
    bind_base(); verify_v4()
    records, arrays = base.check_prepared()
    v4 = load_v4_arrays()
    n = sum(EXPECTED_CLAIMS.values())
    assert len(records) == n and np.array_equal(arrays["labels"], v4["labels"])
    assert np.array_equal(arrays["fold_assignment"], v4["held_fold"])
    assert np.array_equal(arrays["lengths"].astype(np.int32), v4["input_token_length"])
    assert np.allclose(arrays["weights"][FOLDS], v4["fit_training_weight"].astype(np.float32), rtol=0, atol=5e-7)
    answers = lines(V4_ANSWERS)
    assert [row["response_index"] for row in answers] == list(range(len(answers)))
    assert all(row["partition"] == "fit" for row in answers[:EXPECTED_ANSWERS["fit"]])
    assert all(row["partition"] == "calibration" for row in answers[EXPECTED_ANSWERS["fit"]:])
    first_window_owners = v4["window_claim_example_index"][v4["window_claim_indptr"][:-1]]
    assert np.array_equal(v4["partition"][first_window_owners],
                          (v4["window_response_index"] >= EXPECTED_ANSWERS["fit"]).astype(np.int8))
    shortcut = json.loads((OUT / "preparation.json").read_text(encoding="utf-8"))["known_generic_refusal_template_shortcut"]
    assert shortcut["fit_responses"] == 49 and shortcut["complete_input_pair_hash_overlap_across_folds"] == 0
    # Exercise the exact frozen ragged projection with deterministic dummy scores.
    dummy = np.linspace(0, 1, n, dtype=np.float32)
    projected = project_windows(v4, dummy)
    assert projected.shape == (sum(EXPECTED_WINDOWS.values()),) and np.isfinite(projected).all()
    save_json(OUT / "CPU_CHECK.json", {
        "status": "passed", "examples": n, "v4_labels_exact": True,
        "v4_folds_exact": True, "v4_full_fit_weights_exact": True,
        "v4_input_lengths_exact": True, "v4_window_projection_exercised": len(projected),
        "generic_refusal_complete_input_pair_overlap": 0,
        "base_training_loop_reused": True, "model_loaded": False, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    })
    print("EXPANDED_CROSSENCODER_CPU_CHECK_PASSED", n, flush=True)


def project_windows(v4, claim_scores):
    claim_scores = np.asarray(claim_scores, dtype=np.float32)
    indptr = v4["window_claim_indptr"]
    values = v4["window_claim_example_index"]
    assert len(indptr) == len(v4["window_label"]) + 1 and indptr[0] == 0 and indptr[-1] == len(values)
    output = np.empty(len(v4["window_label"]), dtype=np.float32)
    for left in range(0, len(output), 65536):
        right = min(len(output), left + 65536)
        for index in range(left, right):
            output[index] = claim_scores[values[indptr[index]:indptr[index + 1]]].max()
    return output


def answer_scores(answers, v4, window_scores):
    output = np.empty(len(answers), dtype=np.float32)
    for answer in answers:
        left, right = answer["window_array_start"], answer["window_array_end"]
        output[answer["response_index"]] = np.max(window_scores[left:right])
    return output


def metric_bundle(answers, v4, window_scores, thresholds):
    a_scores = answer_scores(answers, v4, window_scores)
    result = {}
    for partition, partition_value in (("fit", 0), ("calibration", 1)):
        wmask = v4["partition"][v4["window_claim_example_index"][v4["window_claim_indptr"][:-1]]] == partition_value
        # The first owner always has the same partition as its window.
        aidx = np.asarray([row["response_index"] for row in answers if row["partition"] == partition], dtype=np.int64)
        result[partition] = {
            "windows": base.q.count(v4["window_label"][wmask], window_scores[wmask], thresholds["window"]["threshold"]),
            "answers": base.q.count([answers[index]["answer_risk"] for index in aidx], a_scores[aidx], thresholds["answer"]["threshold"]),
        }
    return result, a_scores


def summarize_scores(answers, v4, claim_scores):
    windows = project_windows(v4, claim_scores)
    fit_windows = v4["window_response_index"] < EXPECTED_ANSWERS["fit"]
    fit_answers = np.arange(EXPECTED_ANSWERS["fit"])
    answers_all = answer_scores(answers, v4, windows)
    thresholds = {
        "window": base.q.choose_threshold(v4["window_label"][fit_windows], windows[fit_windows]),
        "answer": base.q.choose_threshold([answers[index]["answer_risk"] for index in fit_answers], answers_all[fit_answers]),
    }
    strict, _ = metric_bundle(answers, v4, windows, thresholds)
    cal_windows = ~fit_windows
    cal_answers = np.arange(EXPECTED_ANSWERS["fit"], len(answers))
    cal_thresholds = {
        "window": base.q.choose_threshold(v4["window_label"][cal_windows], windows[cal_windows]),
        "answer": base.q.choose_threshold([answers[index]["answer_risk"] for index in cal_answers], answers_all[cal_answers]),
    }
    cal_opt, _ = metric_bundle(answers, v4, windows, cal_thresholds)
    return windows, answers_all, {"thresholds_from_fit_OOF": thresholds,
                                  "strict_fit_threshold_to_calibration": strict,
                                  "common_calibration_F1Opt_diagnostic": {
                                      "thresholds": cal_thresholds, "metrics": cal_opt["calibration"]}}


def finalize():
    bind_base(); records, arrays = base.check_prepared()
    assert not (OUT / "summary.json").exists()
    verify_v4(); v4 = load_v4_arrays(); answers = lines(V4_ANSWERS)
    main_claim = base.assemble_main(arrays)
    control_complete = json.loads((OUT / "frozen_control/complete.json").read_text(encoding="utf-8"))
    assert control_complete["status"] == "frozen_control_complete"
    assert control_complete["scores_sha256"] == sha(OUT / "frozen_control/scores.npz")
    with np.load(OUT / "frozen_control/scores.npz", allow_pickle=False) as values:
        control_claim = values["claim_scores"].copy()
    main_window, main_answer, main = summarize_scores(answers, v4, main_claim)
    control_window, control_answer, control = summarize_scores(answers, v4, control_claim)
    save_npz(OUT / "scores.npz", main_claim_scores=main_claim, main_window_scores=main_window,
             main_answer_scores=main_answer, frozen_claim_scores=control_claim,
             frozen_window_scores=control_window, frozen_answer_scores=control_answer)
    summary = {
        "status": "development_only_complete", "method": "expanded-v4 top-evidence microclaim cross-encoder",
        "trained_candidate": main, "minimal_frozen_control": control,
        "incumbent_read_only_reference": json.loads(base.INCUMBENT.read_text(encoding="utf-8"))["metrics"]["calibration"],
        "v4_manifest_sha256": V4_MANIFEST_SHA256, "base_runner_sha256": BASE_RUNNER_SHA256,
        "single_trained_candidate": True, "epochs": EPOCHS,
        "five_source_group_OOF_models": True, "separate_full_fit_calibration_model": True,
        "calibration_used_for_training_or_selection": False, "fit_only_thresholds": True,
        "formal_baselines_modified": False, "official_test_opened": False, "final_test_claim": False,
        "scores_sha256": sha(OUT / "scores.npz"),
    }
    save_json(OUT / "summary.json", summary)
    metric = main["strict_fit_threshold_to_calibration"]
    report = [
        "# Expanded-v4 microclaim cross-encoder", "",
        "唯一训练候选直接微调原始三分类 ModernBERT NLI；fit为source-connected五折OOF，calibration由full-fit模型预测。", "",
        "| 层级 | fit OOF F1 | calibration严格F1 |", "|---|---:|---:|",
        f"| 4-BPE窗口 | {metric['fit']['windows']['f1']:.6f} | {metric['calibration']['windows']['f1']:.6f} |",
        f"| 整答 | {metric['fit']['answers']['f1']:.6f} | {metric['calibration']['answers']['f1']:.6f} |", "",
        "正式baseline未修改；official test未打开。calibration F1-opt仅作共同开发诊断。", "",
        "已知限制：fit中49条无关回答共享固定拒答文本且均为负例，但其完整question+evidence+hypothesis输入哈希互异、跨折零重叠；该模板仍可能带来分类捷径。", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    save_json(OUT / "complete.json", {"status": "complete_development_only",
              "summary_sha256": sha(OUT / "summary.json"), "scores_sha256": sha(OUT / "scores.npz"),
              "report_sha256": sha(OUT / "REPORT.md"), "official_test_opened": False})


def verify(final=False):
    bind_base(); records, arrays = base.check_prepared()
    assert len(records) == len(arrays["labels"]) == sum(EXPECTED_CLAIMS.values())
    assert json.loads((OUT / "CPU_CHECK.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert json.loads((OUT / "CPU_TINY_TRAIN.json").read_text(encoding="utf-8"))["status"] == "passed"
    if (OUT / "GPU_SMOKE.json").exists():
        assert json.loads((OUT / "GPU_SMOKE.json").read_text(encoding="utf-8"))["status"] == "passed_no_formal_training"
    if final:
        complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
        assert complete["status"] == "complete_development_only"
        assert complete["summary_sha256"] == sha(OUT / "summary.json")
        assert complete["scores_sha256"] == sha(OUT / "scores.npz")
        assert complete["report_sha256"] == sha(OUT / "REPORT.md")
    print("EXPANDED_CROSSENCODER_VERIFIED", "final" if final else "prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "cpu-tiny", "gpu-smoke",
                                          "train-fold", "train-full", "control", "finalize",
                                          "verify", "verify-final"))
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    bind_base()
    if args.stage == "train-fold" and args.fold is None:
        parser.error("train-fold requires --fold 0..4")
    if args.stage != "train-fold" and args.fold is not None:
        parser.error("--fold is only valid with train-fold")
    if args.stage == "prepare": prepare()
    elif args.stage == "check": check()
    elif args.stage == "cpu-tiny": base.cpu_tiny()
    elif args.stage == "gpu-smoke": base.gpu_smoke()
    elif args.stage == "train-fold": base.train_fold(args.fold)
    elif args.stage == "train-full": base.train_full()
    elif args.stage == "control": base.control()
    elif args.stage == "finalize": finalize()
    elif args.stage == "verify": verify(False)
    else: verify(True)


if __name__ == "__main__":
    main()
