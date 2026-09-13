"""Type-aware continuation of the expanded-v4 microclaim cross-encoder.

This runner prepares fit-only three-way targets and a frozen fold-0 GPU pilot.
Preparation, checking, and the tiny training test are CPU-only.  The pilot is
an explicit separate command and never opens calibration labels or test data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForSequenceClassification,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/microclaim_crossencoder_typeaware_v5"
V4_DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_RUN = ROOT / "results/microclaim_crossencoder_expanded_v4"
V4_EXAMPLES = V4_DATA / "examples.jsonl"
V4_ENCODED = V4_RUN / "encoded_inputs.npz"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"

N_FIT = 34_919
FOLDS = 5
FULL_FIT_ROW = 5
PILOT_FOLD = 0
SEED = 20_261_023
MAX_LENGTH = 768
PAD_TOKEN_ID = 50_283
TRAIN_TOKEN_BUDGET = 1_536
TRAIN_MAX_EXAMPLES = 8
ACCUM_MICROBATCHES = 4
EVAL_TOKEN_BUDGET = 4_096
EVAL_MAX_EXAMPLES = 16
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.01
TYPE_LOSS_LAMBDA = 0.25
EPOCHS = 1
MIN_FREE_GPU_BYTES = 4_500_000_000

CLASS_NAMES = ("entailment", "neutral", "contradiction")
CODE_NAMES = ("safe", "baseless_only", "conflict_only", "mixed_baseless_conflict")
BASELESS_TYPES = frozenset(("Evident Baseless Info", "Subtle Baseless Info"))
CONFLICT_TYPES = frozenset(("Evident Conflict", "Subtle Conflict"))
KNOWN_TYPES = BASELESS_TYPES | CONFLICT_TYPES

EXPECTED_SHA256 = {
    V4_EXAMPLES: "6a8c46f805eae608dba4708fda9908acf774730d4ff358ff44144f8bc1ff1c54",
    V4_ENCODED: "d5c67a66e3ed16e3cd0faef4085f94ada92c085e7ca733e01b123d522914b7d6",
    MODEL / "model.safetensors": "86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465",
}
V4_CHECKPOINT_SHA256 = {
    0: "4e7fc075b3a0ee3a0cbe9ebd0cd69182bdb49663496ebcfee7b551427103b048",
    1: "bfdc8e0ce46b440ecae7f34942d14940f7fd7755164c6c48cb3c84e39c05e8a5",
    2: "a072c38a3d9ee14f9d4e667eac2d90bc3f64d48f06ed38d548f645a152c9729a",
    3: "35e51efb8954ab368af0c87c2cb79f739899830011886bff3918d7cea043fdbf",
    4: "26959e3fe9838b405556322191626fcaabafec6605e725307fc6297356b7f166",
    5: "c301b3ce00e0556ed3c513630fdf262dda40b0138d315b89a854eeed6b368bbc",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def protocol() -> dict:
    return {
        "version": "microclaim-crossencoder-typeaware-v5",
        "scope": "Fit-only type target construction and a single frozen fold-0 continuation pilot; calibration/test labels remain unopened.",
        "parent": {
            "dataset": "atomic-microclaim-relation-expanded-v4",
            "model_run": "microclaim-crossencoder-expanded-v4",
            "initialization": "The matching completed v4 fold checkpoint, not the original NLI checkpoint.",
        },
        "three_way_target": {
            "safe_binary_gold_0": [1.0, 0.0, 0.0],
            "EBI_or_SBI_only": [0.0, 1.0, 0.0],
            "EC_or_SC_only": [0.0, 0.0, 1.0],
            "both_baseless_and_conflict": [0.0, 0.5, 0.5],
            "same_family_multiple_types": "Collapse to that family's one-hot target.",
            "precedence": "Binary gold=0 always maps to entailment, including five character-overlap-only cases with no risky lexical BPE.",
        },
        "loss": {
            "formula": "weighted_mean(BCEWithLogits(logsumexp(N,C)-E, binary_gold) + 0.25 * weighted_soft_CE([E,N,C], type_target))",
            "binary_risk_logit": "logsumexp(neutral, contradiction) - entailment",
            "lambda_type": TYPE_LOSS_LAMBDA,
            "example_weights": "Exact expanded-v4 canonical source-group/answer/claim binary weights, recomputed from fit only for each training fold.",
            "type_class_weights": "Within each fold, M/(3*m_c), where m_c is example-weighted soft-target mass and M=sum_c m_c; this gives equal aggregate CE target mass to E/N/C.",
            "mixed_target": "The 17 cross-family microclaims contribute half neutral and half contradiction before class weighting; none is discarded or given arbitrary precedence.",
        },
        "pilot": {
            "fold": PILOT_FOLD,
            "epochs": EPOCHS,
            "optimizer": "fresh AdamW state over all model parameters",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clip_norm": 1.0,
            "scheduler": None,
            "batching": {
                "train_padded_token_budget": TRAIN_TOKEN_BUDGET,
                "train_max_examples": TRAIN_MAX_EXAMPLES,
                "gradient_accumulation_microbatches": ACCUM_MICROBATCHES,
                "eval_padded_token_budget": EVAL_TOKEN_BUDGET,
                "eval_max_examples": EVAL_MAX_EXAMPLES,
            },
            "precision": "FP32 parameters/optimizer, BF16 CUDA forward, non-reentrant gradient checkpointing, TF32 off.",
            "selection": "One configuration, one epoch, one seed, no checkpoint/grid/calibration selection.",
        },
        "evaluation_output": "The risk score is unchanged from v4: sigmoid(logsumexp(N,C)-E). Three-way probabilities are diagnostics only.",
        "official_test_opened": False,
        "baselines_modified": False,
    }


def read_fit_examples(full: bool = False):
    """Read exactly the first frozen fit rows; never advance into calibration."""
    rows = []
    with V4_EXAMPLES.open("r", encoding="utf-8") as handle:
        for expected_index in range(N_FIT):
            line = handle.readline()
            assert line, expected_index
            row = json.loads(line)
            assert row["example_index"] == expected_index
            assert row["partition"] == "fit" and row["held_fold"] in range(FOLDS)
            if full:
                rows.append(row)
            else:
                raw_types = sorted({span["label_type"] for span in row["overlapping_gold_spans"]})
                assert set(raw_types) <= KNOWN_TYPES
                rows.append({
                    "index": expected_index,
                    "response_id": str(row["response_id"]),
                    "source_id": str(row["source_id"]),
                    "group_id": row["group_id"],
                    "held_fold": int(row["held_fold"]),
                    "binary_label": int(row["gold_label"]),
                    "raw_types": raw_types,
                    "fit_training_weight": float(row["fit_training_weight"]),
                })
    return rows


def target_for(binary_label: int, raw_types) -> tuple[int, np.ndarray]:
    types = set(raw_types)
    if binary_label == 0:
        return 0, np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    has_baseless = bool(types & BASELESS_TYPES)
    has_conflict = bool(types & CONFLICT_TYPES)
    assert has_baseless or has_conflict
    if has_baseless and has_conflict:
        return 3, np.asarray((0.0, 0.5, 0.5), dtype=np.float32)
    if has_baseless:
        return 1, np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    return 2, np.asarray((0.0, 0.0, 1.0), dtype=np.float32)


def canonical_weights(rows, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    response_ids = np.asarray([row["response_id"] for row in rows])
    group_by_response = {row["response_id"]: row["group_id"] for row in rows}
    result = np.zeros((FOLDS + 1, len(rows)), dtype=np.float32)
    for model_fold in range(FOLDS + 1):
        active = np.arange(len(rows)) if model_fold == FULL_FIT_ROW else np.flatnonzero(folds != model_fold)
        tree = defaultdict(lambda: defaultdict(list))
        for index in active:
            rid = response_ids[index]
            tree[group_by_response[rid]][rid].append(int(index))
        weights = np.zeros(len(rows), dtype=np.float64)
        for answers in tree.values():
            for indices in answers.values():
                weights[indices] = 1.0 / (len(answers) * len(indices))
        mask = weights > 0
        weights[mask] /= weights[mask].mean()
        mass = np.bincount(labels[mask], weights=weights[mask], minlength=2)
        assert np.all(mass > 0)
        weights[mask] *= (mass.sum() / (2.0 * mass))[labels[mask]]
        for answers in tree.values():
            indices = [index for values in answers.values() for index in values]
            weights[indices] *= (mask.sum() / len(tree)) / weights[indices].sum()
        weights[mask] *= mask.sum() / weights[mask].sum()
        assert np.all(weights[active] > 0)
        if model_fold < FOLDS:
            assert np.all(weights[folds == model_fold] == 0)
        result[model_fold] = weights.astype(np.float32)
    return result


def make_batches(indices, lengths, seed, token_budget, max_examples):
    indices = np.asarray(indices, dtype=np.int64)
    ordered = indices[np.argsort(lengths[indices], kind="stable")]
    batches, current = [], []
    for raw in ordered:
        index = int(raw)
        proposed = current + [index]
        padded = max(int(lengths[j]) for j in proposed) * len(proposed)
        if current and (padded > token_budget or len(proposed) > max_examples):
            batches.append(current)
            current = [index]
        else:
            current = proposed
    if current:
        batches.append(current)
    rng = np.random.default_rng(seed)
    for batch in batches:
        rng.shuffle(batch)
    rng.shuffle(batches)
    assert sorted(index for batch in batches for index in batch) == sorted(indices.tolist())
    return batches


def risk_logit(logits: torch.Tensor) -> torch.Tensor:
    assert logits.ndim == 2 and logits.shape[1] == 3
    value = torch.logsumexp(logits[:, 1:3].float(), dim=-1) - logits[:, 0].float()
    assert torch.isfinite(value).all()
    return value


def per_example_losses(logits, binary_labels, type_targets, class_weights):
    risk = risk_logit(logits)
    binary = F.binary_cross_entropy_with_logits(risk, binary_labels.float(), reduction="none")
    log_prob = F.log_softmax(logits.float(), dim=-1)
    typed = -(type_targets.float() * class_weights.float()[None, :] * log_prob).sum(dim=-1)
    joint = binary + TYPE_LOSS_LAMBDA * typed
    assert binary.shape == typed.shape == joint.shape and torch.isfinite(joint).all()
    return joint, binary, typed, risk


def type_class_weights(example_weights, targets, active):
    active = np.asarray(active, dtype=np.int64)
    mass = (example_weights[active, None].astype(np.float64) * targets[active]).sum(axis=0)
    assert np.all(mass > 0)
    total = mass.sum()
    weights = total / (len(CLASS_NAMES) * mass)
    effective = mass * weights
    assert np.allclose(effective, total / len(CLASS_NAMES), rtol=1e-10, atol=1e-8)
    return weights.astype(np.float32), mass, effective


def prepare():
    assert not OUT.exists()
    assert not torch.cuda.is_initialized()
    for path, expected in EXPECTED_SHA256.items():
        assert sha(path) == expected, path
    for fold, expected in V4_CHECKPOINT_SHA256.items():
        path = V4_RUN / ("full_fit" if fold == FULL_FIT_ROW else f"fold_{fold}") / "model.pt"
        assert sha(path) == expected, path

    rows = read_fit_examples(full=False)
    labels = np.asarray([row["binary_label"] for row in rows], dtype=np.int8)
    folds = np.asarray([row["held_fold"] for row in rows], dtype=np.int8)
    codes = np.empty(N_FIT, dtype=np.int8)
    targets = np.empty((N_FIT, 3), dtype=np.float32)
    for index, row in enumerate(rows):
        codes[index], targets[index] = target_for(row["binary_label"], row["raw_types"])
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert Counter(labels.tolist()) == {0: 31_445, 1: 3_474}
    assert Counter(codes.tolist()) == {0: 31_445, 1: 3_052, 2: 405, 3: 17}

    weights = canonical_weights(rows, labels, folds)
    given_full = np.asarray([row["fit_training_weight"] for row in rows], dtype=np.float32)
    assert np.array_equal(weights[FULL_FIT_ROW], given_full)
    class_weights = np.zeros((FOLDS + 1, 3), dtype=np.float32)
    weight_audit = []
    for model_fold in range(FOLDS + 1):
        active = np.arange(N_FIT) if model_fold == FULL_FIT_ROW else np.flatnonzero(folds != model_fold)
        class_weights[model_fold], mass, effective = type_class_weights(weights[model_fold], targets, active)
        weight_audit.append({
            "model": "full_fit" if model_fold == FULL_FIT_ROW else f"fold_{model_fold}",
            "training_examples": int(len(active)),
            "soft_target_weighted_mass_before": dict(zip(CLASS_NAMES, map(float, mass))),
            "type_class_weights": dict(zip(CLASS_NAMES, map(float, class_weights[model_fold]))),
            "soft_target_weighted_mass_after": dict(zip(CLASS_NAMES, map(float, effective))),
        })

    raw_combo = Counter(" + ".join(row["raw_types"]) if row["raw_types"] else "none" for row in rows)
    raw_membership = Counter()
    for row in rows:
        if row["binary_label"]:
            raw_membership.update(row["raw_types"])
    held_coverage = []
    for held_fold in range(FOLDS):
        held = np.flatnonzero(folds == held_fold)
        train = np.flatnonzero(folds != held_fold)
        held_groups = {rows[i]["group_id"] for i in held}
        train_groups = {rows[i]["group_id"] for i in train}
        assert not (held_groups & train_groups)
        held_coverage.append({
            "fold": held_fold,
            "held_examples": int(len(held)),
            "train_examples": int(len(train)),
            "held_code_counts": {CODE_NAMES[k]: int(np.sum(codes[held] == k)) for k in range(4)},
            "train_code_counts": {CODE_NAMES[k]: int(np.sum(codes[train] == k)) for k in range(4)},
            "held_positive_type_membership": {
                name: int(sum(labels[i] == 1 and name in rows[i]["raw_types"] for i in held))
                for name in sorted(KNOWN_TYPES)
            },
            "train_positive_type_membership": {
                name: int(sum(labels[i] == 1 and name in rows[i]["raw_types"] for i in train))
                for name in sorted(KNOWN_TYPES)
            },
            "held_groups": len(held_groups),
            "train_groups": len(train_groups),
            "group_overlap": 0,
        })
        assert all(value > 0 for value in held_coverage[-1]["held_code_counts"].values())
        assert all(value > 0 for value in held_coverage[-1]["train_code_counts"].values())

    type_rows = [{
        "index": row["index"],
        "response_id": row["response_id"],
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "held_fold": row["held_fold"],
        "binary_label": row["binary_label"],
        "raw_types": row["raw_types"],
        "exclusive_code": int(codes[i]),
        "exclusive_name": CODE_NAMES[int(codes[i])],
        "type_target_ENC": [float(v) for v in targets[i]],
    } for i, row in enumerate(rows)]

    pilot_train = np.flatnonzero(folds != PILOT_FOLD)
    pilot_held = np.flatnonzero(folds == PILOT_FOLD)
    with np.load(V4_ENCODED, allow_pickle=False) as encoded:
        lengths = encoded["lengths"][:N_FIT].astype(np.int32)
    train_batches = make_batches(pilot_train, lengths, SEED + PILOT_FOLD,
                                 TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
    train_padded = sum(max(int(lengths[i]) for i in batch) * len(batch) for batch in train_batches)
    held_batches = make_batches(pilot_held, lengths, SEED + 9000,
                                EVAL_TOKEN_BUDGET, EVAL_MAX_EXAMPLES)
    held_padded = sum(max(int(lengths[i]) for i in batch) * len(batch) for batch in held_batches)
    v4_fold0 = json.loads((V4_RUN / "fold_0/complete.json").read_text(encoding="utf-8"))
    v4_train_rate = v4_fold0["training"]["padded_tokens"] / v4_fold0["training"]["seconds"]
    v4_eval_rate = v4_fold0["held_inference"]["padded_tokens"] / v4_fold0["held_inference"]["seconds"]
    point_seconds = 1.05 * train_padded / v4_train_rate + held_padded / v4_eval_rate + 2.0
    conservative_seconds = 1.20 * point_seconds + 30.0
    pilot_plan = {
        "status": "frozen_not_run",
        "only_pilot": "fold_0",
        "command": "prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_microclaim_crossencoder_typeaware_v5.py pilot-fold0",
        "initial_checkpoint": str((V4_RUN / "fold_0/model.pt").resolve()),
        "initial_checkpoint_sha256": V4_CHECKPOINT_SHA256[0],
        "train_examples": int(len(pilot_train)),
        "held_examples": int(len(pilot_held)),
        "train_microbatches": len(train_batches),
        "optimizer_updates": math.ceil(len(train_batches) / ACCUM_MICROBATCHES),
        "train_padded_tokens": int(train_padded),
        "held_microbatches": len(held_batches),
        "held_padded_tokens": int(held_padded),
        "point_runtime_minutes": point_seconds / 60.0,
        "conservative_runtime_minutes": conservative_seconds / 60.0,
        "expected_peak_cuda_bytes": int(v4_fold0["peak_cuda_allocated_bytes"] * 1.03),
        "basis": "Measured expanded-v4 fold-0 training/evaluation throughput; 5% point overhead and 20%+30s conservative margin for the auxiliary CE.",
        "unique_configuration": protocol()["pilot"],
        "calibration_labels_used": False,
        "official_test_opened": False,
    }

    audit = {
        "status": "fit_type_coverage_passed",
        "fit_examples": N_FIT,
        "binary_counts": {"safe": int(np.sum(labels == 0)), "risky": int(np.sum(labels == 1))},
        "exclusive_code_counts": {CODE_NAMES[k]: int(np.sum(codes == k)) for k in range(4)},
        "positive_type_membership": {name: int(raw_membership[name]) for name in sorted(KNOWN_TYPES)},
        "raw_character_overlap_combinations_all_binary_labels": dict(sorted(raw_combo.items())),
        "binary_safe_with_character_span_overlap_but_no_risky_lexical_BPE": int(sum(
            row["binary_label"] == 0 and bool(row["raw_types"]) for row in rows)),
        "cross_family_mixed_positive": int(np.sum(codes == 3)),
        "fold_coverage": held_coverage,
        "class_weight_audit": weight_audit,
        "target_rows_sum_to_one": True,
        "full_fit_binary_weights_exactly_match_v4": True,
        "calibration_rows_read": 0,
        "calibration_label_arrays_accessed": False,
        "official_test_opened": False,
        "GPU_used": False,
        "baselines_modified": False,
    }

    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_jsonl(OUT / "type_examples.jsonl", type_rows)
    np.savez_compressed(OUT / "type_labels.npz", binary_labels=labels, type_targets=targets,
                        exclusive_codes=codes, fold_assignment=folds,
                        binary_weights=weights, type_class_weights=class_weights)
    save_json(OUT / "TYPE_COVERAGE_AUDIT.json", audit)
    save_json(OUT / "FOLD0_GPU_PILOT_PLAN.json", pilot_plan)
    names = ("protocol.json", "type_examples.jsonl", "type_labels.npz",
             "TYPE_COVERAGE_AUDIT.json", "FOLD0_GPU_PILOT_PLAN.json")
    save_json(OUT / "preparation_complete.json", {
        "status": "CPU_fit_labels_prepared_GPU_not_run",
        "files_sha256": {name: sha(OUT / name) for name in names},
        "source_code_sha256": sha(Path(__file__)),
        "source_inputs_sha256": {str(path.resolve()): expected for path, expected in EXPECTED_SHA256.items()},
        "v4_checkpoint_sha256": {str(key): value for key, value in V4_CHECKPOINT_SHA256.items()},
        "calibration_rows_read": 0,
        "calibration_labels_used": False,
        "official_test_opened": False,
        "GPU_used": False,
        "baselines_modified": False,
    })
    assert not torch.cuda.is_initialized()
    print("TYPEAWARE_V5_PREPARED", N_FIT, dict(Counter(codes.tolist())), flush=True)


def load_prepared():
    complete = json.loads((OUT / "preparation_complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "CPU_fit_labels_prepared_GPU_not_run"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    assert json.loads((OUT / "protocol.json").read_text(encoding="utf-8")) == protocol()
    with np.load(OUT / "type_labels.npz", allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    assert arrays["binary_labels"].shape == (N_FIT,)
    assert arrays["type_targets"].shape == (N_FIT, 3)
    assert arrays["binary_weights"].shape == (FOLDS + 1, N_FIT)
    assert arrays["type_class_weights"].shape == (FOLDS + 1, 3)
    return complete, arrays


def check():
    assert not torch.cuda.is_initialized()
    _, arrays = load_prepared()
    rows = read_fit_examples(full=False)
    replay_codes = np.empty(N_FIT, dtype=np.int8)
    replay_targets = np.empty((N_FIT, 3), dtype=np.float32)
    for index, row in enumerate(rows):
        replay_codes[index], replay_targets[index] = target_for(row["binary_label"], row["raw_types"])
    assert np.array_equal(replay_codes, arrays["exclusive_codes"])
    assert np.array_equal(replay_targets, arrays["type_targets"])
    assert np.array_equal(np.asarray([row["binary_label"] for row in rows], np.int8), arrays["binary_labels"])
    replay_weights = canonical_weights(rows, arrays["binary_labels"], arrays["fold_assignment"])
    assert np.array_equal(replay_weights, arrays["binary_weights"])
    for model_fold in range(FOLDS + 1):
        active = np.arange(N_FIT) if model_fold == FULL_FIT_ROW else np.flatnonzero(arrays["fold_assignment"] != model_fold)
        expected, _, _ = type_class_weights(arrays["binary_weights"][model_fold], arrays["type_targets"], active)
        assert np.allclose(expected, arrays["type_class_weights"][model_fold], rtol=0, atol=1e-7)
    save_json(OUT / "CPU_CHECK.json", {
        "status": "passed",
        "fit_rows_replayed": N_FIT,
        "type_targets_exact": True,
        "canonical_weights_exact": True,
        "fold_class_weights_replayed": True,
        "calibration_rows_read": 0,
        "calibration_labels_used": False,
        "official_test_opened": False,
        "GPU_used": False,
    })
    assert not torch.cuda.is_initialized()
    print("TYPEAWARE_V5_CPU_CHECK_PASSED", flush=True)


def tiny_config(vocab_size):
    config = ModernBertConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              max_position_embeddings=MAX_LENGTH, local_attention=32,
                              pad_token_id=PAD_TOKEN_ID, bos_token_id=50_281,
                              eos_token_id=50_282, sep_token_id=50_282,
                              num_labels=3, classifier_pooling="mean",
                              reference_compile=False)
    config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config.label2id = {name: index for index, name in config.id2label.items()}
    config._attn_implementation = "sdpa"
    return config


def cpu_tiny():
    assert not torch.cuda.is_initialized()
    _, arrays = load_prepared()
    assert (OUT / "CPU_CHECK.json").exists() and not (OUT / "CPU_TINY_TRAIN.json").exists()
    rows = read_fit_examples(full=True)
    active = np.flatnonzero(arrays["fold_assignment"] != PILOT_FOLD)
    selected = []
    for code in range(4):
        candidates = [int(index) for index in active if arrays["exclusive_codes"][index] == code]
        selected.extend(candidates[:3])
    selected = np.asarray(selected, dtype=np.int64)
    assert len(selected) == 12 and Counter(arrays["exclusive_codes"][selected].tolist()) == {0: 3, 1: 3, 2: 3, 3: 3}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    encoded = tokenizer([rows[i]["premise"] for i in selected],
                        [rows[i]["hypothesis"] for i in selected],
                        add_special_tokens=True, truncation=False, padding=True,
                        return_tensors="pt")
    assert int(encoded["attention_mask"].sum(1).max()) <= MAX_LENGTH
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = ModernBertForSequenceClassification(tiny_config(len(tokenizer))).float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(**encoded).logits
    y = torch.from_numpy(arrays["binary_labels"][selected]).float()
    target = torch.from_numpy(arrays["type_targets"][selected]).float()
    example_weight = torch.from_numpy(arrays["binary_weights"][PILOT_FOLD, selected]).float()
    class_weight = torch.from_numpy(arrays["type_class_weights"][PILOT_FOLD]).float()
    joint, binary, typed, risk = per_example_losses(logits, y, target, class_weight)
    objective = (joint * example_weight).sum() / example_weight.sum()
    objective.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    optimizer.step()
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(before[name], parameter)]
    assert changed and any(name.startswith("model.layers") for name in changed)
    assert any("classifier" in name for name in changed)

    synthetic = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float32)
    expected_risk = torch.logsumexp(synthetic[:, 1:3], dim=-1) - synthetic[:, 0]
    assert torch.allclose(risk_logit(synthetic), expected_risk)
    manual_type = -(target * class_weight[None, :] * F.log_softmax(logits.detach().float(), dim=-1)).sum(-1)
    assert torch.allclose(typed.detach(), manual_type)
    result = {
        "status": "passed",
        "actual_fit_indices": selected.tolist(),
        "exclusive_code_counts": {CODE_NAMES[k]: int(np.sum(arrays["exclusive_codes"][selected] == k)) for k in range(4)},
        "input_token_lengths": encoded["attention_mask"].sum(1).tolist(),
        "objective_before_step": float(objective.detach()),
        "binary_loss_mean": float(binary.detach().mean()),
        "type_loss_mean_before_lambda": float(typed.detach().mean()),
        "risk_logit_identity_checked": True,
        "soft_target_weighted_CE_identity_checked": True,
        "mixed_target_exercised": True,
        "gradient_norm_before_clip": float(grad_norm),
        "changed_parameter_tensors": len(changed),
        "encoder_parameters_changed": True,
        "classifier_parameters_changed": True,
        "real_pretrained_model_loaded": False,
        "calibration_rows_read": 0,
        "calibration_labels_used": False,
        "official_test_opened": False,
        "GPU_used": False,
    }
    save_json(OUT / "CPU_TINY_TRAIN.json", result)
    assert not torch.cuda.is_initialized()
    print("TYPEAWARE_V5_CPU_TINY_PASSED", len(changed), flush=True)


def report_text() -> str:
    audit = json.loads((OUT / "TYPE_COVERAGE_AUDIT.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "FOLD0_GPU_PILOT_PLAN.json").read_text(encoding="utf-8"))
    tiny = json.loads((OUT / "CPU_TINY_TRAIN.json").read_text(encoding="utf-8"))
    lines = [
        "# Expanded-v5 类型感知续训：CPU 准备报告",
        "",
        "v4 的 binary BCE 只要求 neutral/contradiction 任一变高，因此不会教模型区分“无依据”和“证据冲突”。v5 保留原风险分数，同时把 NLI 三类头重新对齐到安全、无依据、冲突。",
        "",
        "## Fit 标签",
        "",
        "| 互斥训练类 | 数量 | 三类软标签 E/N/C |",
        "|---|---:|---|",
        f"| 安全 | {audit['exclusive_code_counts']['safe']} | 1 / 0 / 0 |",
        f"| 仅无依据（EBI/SBI） | {audit['exclusive_code_counts']['baseless_only']} | 0 / 1 / 0 |",
        f"| 仅冲突（EC/SC） | {audit['exclusive_code_counts']['conflict_only']} | 0 / 0 / 1 |",
        f"| 同时无依据与冲突 | {audit['exclusive_code_counts']['mixed_baseless_conflict']} | 0 / 0.5 / 0.5 |",
        "",
        "多类型处理已经固定：同属无依据或同属冲突时合并为一个类；跨两类的17条使用各半软标签。5条只有字符跨度相交、但没有风险词元的样本按原 binary gold 保持安全。",
        "",
        "原始正例类型覆盖：EBI {ebi}、SBI {sbi}、EC {ec}、SC {sc}。五个 held fold 和五个训练补集都含四种互斥类，且来源组零交叉。".format(
            ebi=audit["positive_type_membership"]["Evident Baseless Info"],
            sbi=audit["positive_type_membership"]["Subtle Baseless Info"],
            ec=audit["positive_type_membership"]["Evident Conflict"],
            sc=audit["positive_type_membership"]["Subtle Conflict"]),
        "",
        "## 冻结损失",
        "",
        "`BCE(logsumexp(N,C)-E, 是否错误) + 0.25 × 三分类软标签CE`。仍用 v4 的逐样本权重；三分类部分在每个训练折内按加权标签质量做逆频率平衡，使 E/N/C 三类的总监督质量相等。最终风险分数仍是 v4 的 `sigmoid(logsumexp(N,C)-E)`。",
        "",
        "## 唯一 fold-0 GPU pilot",
        "",
        f"从 v4 fold-0 checkpoint 继续训练1轮；新 AdamW，学习率 {LEARNING_RATE:g}，weight decay {WEIGHT_DECAY:g}，无 scheduler，seed {SEED}。训练 {plan['train_examples']} 条、验证 {plan['held_examples']} 条，预计 {plan['point_runtime_minutes']:.1f} 分钟，保守 {plan['conservative_runtime_minutes']:.1f} 分钟，显存约 {plan['expected_peak_cuda_bytes']/1e9:.2f} GB。只允许这一套参数，不做网格或校准集选择。",
        "",
        "## CPU 验证",
        "",
        f"12条真实 fit 输入覆盖四类标签，tiny ModernBERT 完成一次联合损失更新；encoder 与 classifier 都发生更新（{tiny['changed_parameter_tensors']} 个参数张量），风险公式和软标签加权CE均逐值核对通过。",
        "",
        "本阶段读取 calibration 行数为0，没有访问 calibration/test 标签，没有使用 GPU，也没有修改 baseline。",
    ]
    return "\n".join(lines) + "\n"


def finalize():
    load_prepared()
    assert json.loads((OUT / "CPU_CHECK.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert json.loads((OUT / "CPU_TINY_TRAIN.json").read_text(encoding="utf-8"))["status"] == "passed"
    (OUT / "REPORT.md").write_text(report_text(), encoding="utf-8")
    names = ("protocol.json", "type_examples.jsonl", "type_labels.npz",
             "TYPE_COVERAGE_AUDIT.json", "FOLD0_GPU_PILOT_PLAN.json",
             "preparation_complete.json", "CPU_CHECK.json", "CPU_TINY_TRAIN.json", "REPORT.md")
    save_json(OUT / "manifest.json", {
        "version": "microclaim-crossencoder-typeaware-v5",
        "status": "CPU_prepared_checked_tiny_passed_fold0_GPU_pilot_not_run",
        "files_sha256": {name: sha(OUT / name) for name in names},
        "source_code_sha256": sha(Path(__file__)),
        "calibration_rows_read": 0,
        "calibration_labels_used": False,
        "official_test_opened": False,
        "GPU_used": False,
        "baselines_modified": False,
    })
    assert not torch.cuda.is_initialized()
    print("TYPEAWARE_V5_CPU_FINALIZED", flush=True)


def load_fit_encoded():
    """Load only fit inputs from v4; deliberately never access its label field."""
    assert sha(V4_ENCODED) == EXPECTED_SHA256[V4_ENCODED]
    with np.load(V4_ENCODED, allow_pickle=False) as values:
        bounds = values["bounds"][:N_FIT].copy()
        lengths = values["lengths"][:N_FIT].copy()
        fit_end = int(bounds[-1, 1])
        flat = values["flat_input_ids"][:fit_end].copy()
    assert bounds.shape == (N_FIT, 2) and len(flat) == int(bounds[-1, 1])
    return {"bounds": bounds, "lengths": lengths, "flat_input_ids": flat}


def collate(inputs, indices, device):
    indices = list(map(int, indices))
    lengths = inputs["lengths"][indices].astype(int)
    width = int(lengths.max())
    ids = np.full((len(indices), width), PAD_TOKEN_ID, dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=np.int64)
    for row, (index, length) in enumerate(zip(indices, lengths)):
        left, right = inputs["bounds"][index]
        ids[row, :length] = inputs["flat_input_ids"][left:right]
        mask[row, :length] = 1
    return {"input_ids": torch.as_tensor(ids, device=device),
            "attention_mask": torch.as_tensor(mask, device=device)}, lengths


def load_v4_checkpoint(fold, device):
    path = V4_RUN / ("full_fit" if fold == FULL_FIT_ROW else f"fold_{fold}") / "model.pt"
    assert sha(path) == V4_CHECKPOINT_SHA256[fold]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", reference_compile=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["fold"] == fold
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected
    assert model.config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    return model.to(device)


def train_epoch(model, optimizer, inputs, labels, train, device):
    fold = PILOT_FOLD
    batches = make_batches(train, inputs["lengths"], SEED + fold,
                           TRAIN_TOKEN_BUDGET, TRAIN_MAX_EXAMPLES)
    weights = labels["binary_weights"][fold]
    class_weights = torch.as_tensor(labels["type_class_weights"][fold], device=device)
    model.train()
    started = time.perf_counter()
    updates = examples = logical = padded = 0
    binary_sum = type_sum = joint_sum = weight_mass = 0.0
    gradient_norms = []
    for first in range(0, len(batches), ACCUM_MICROBATCHES):
        group = batches[first:first + ACCUM_MICROBATCHES]
        denominator = float(sum(weights[index] for batch in group for index in batch))
        optimizer.zero_grad(set_to_none=True)
        for one in group:
            encoded, lengths = collate(inputs, one, device)
            context = torch.autocast("cuda", dtype=torch.bfloat16)
            with context:
                logits = model(**encoded).logits
            y = torch.as_tensor(labels["binary_labels"][one], device=device)
            target = torch.as_tensor(labels["type_targets"][one], device=device)
            w = torch.as_tensor(weights[one], device=device)
            joint, binary, typed, _ = per_example_losses(logits, y, target, class_weights)
            numerator = (joint * w).sum()
            (numerator / denominator).backward()
            binary_sum += float((binary * w).sum().detach())
            type_sum += float((typed * w).sum().detach())
            joint_sum += float(numerator.detach())
            weight_mass += float(w.sum())
            examples += len(one)
            logical += int(lengths.sum())
            padded += int(len(one) * max(lengths))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        updates += 1
        gradient_norms.append(float(norm))
        if updates % 100 == 0 or first + len(group) == len(batches):
            print("TYPEAWARE_V5_TRAIN", updates, math.ceil(len(batches) / ACCUM_MICROBATCHES),
                  round(time.perf_counter() - started, 1), flush=True)
    assert examples == len(train)
    return {
        "examples": examples, "microbatches": len(batches), "optimizer_updates": updates,
        "logical_tokens": logical, "padded_tokens": padded,
        "weighted_binary_bce": binary_sum / weight_mass,
        "weighted_type_ce_before_lambda": type_sum / weight_mass,
        "weighted_joint_loss": joint_sum / weight_mass,
        "gradient_norm_min": min(gradient_norms), "gradient_norm_max": max(gradient_norms),
        "seconds": time.perf_counter() - started,
    }


def score_held(model, inputs, labels, held, device):
    batches = make_batches(held, inputs["lengths"], SEED + 9000,
                           EVAL_TOKEN_BUDGET, EVAL_MAX_EXAMPLES)
    position = {int(index): row for row, index in enumerate(held)}
    risk_scores = np.empty(len(held), dtype=np.float32)
    probabilities = np.empty((len(held), 3), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for bi, one in enumerate(batches):
            encoded, _ = collate(inputs, one, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits
            risk = torch.sigmoid(risk_logit(logits)).cpu().numpy().astype(np.float32)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float32)
            for index, value, prob in zip(one, risk, probs):
                risk_scores[position[index]] = value
                probabilities[position[index]] = prob
            if (bi + 1) % 100 == 0 or bi + 1 == len(batches):
                print("TYPEAWARE_V5_EVAL", bi + 1, len(batches), flush=True)
    assert np.isfinite(risk_scores).all() and np.isfinite(probabilities).all()
    return risk_scores, probabilities, {"examples": len(held), "microbatches": len(batches),
                                        "seconds": time.perf_counter() - started}


def pilot_fold0():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "CPU_prepared_checked_tiny_passed_fold0_GPU_pilot_not_run"
    _, labels = load_prepared()
    directory = OUT / "fold_0_pilot"
    assert not directory.exists()
    assert torch.cuda.is_available()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE_GPU_BYTES
    device = torch.device("cuda:0")
    inputs = load_fit_encoded()
    train = np.flatnonzero(labels["fold_assignment"] != PILOT_FOLD)
    held = np.flatnonzero(labels["fold_assignment"] == PILOT_FOLD)
    directory.mkdir(parents=True)
    save_json(directory / "started.json", {
        "status": "fold0_pilot_started", "seed": SEED,
        "manifest_sha256": sha(OUT / "manifest.json"),
        "calibration_labels_used": False, "official_test_opened": False,
        "time": time.time(),
    })
    model = optimizer = None
    started = time.perf_counter()
    try:
        model = load_v4_checkpoint(PILOT_FOLD, device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        torch.cuda.reset_peak_memory_stats()
        training = train_epoch(model, optimizer, inputs, labels, train, device)
        scores, probabilities, evaluation = score_held(model, inputs, labels, held, device)
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        checkpoint_path = directory / "model.pt"
        torch.save({"model_state_dict": state, "fold": PILOT_FOLD, "seed": SEED,
                    "parent_checkpoint_sha256": V4_CHECKPOINT_SHA256[0],
                    "protocol_sha256": sha(OUT / "protocol.json")}, checkpoint_path)
        np.savez_compressed(directory / "predictions.npz", held_indices=held,
                            held_risk_scores=scores, held_threeway_probabilities=probabilities)
        save_json(directory / "complete.json", {
            "status": "fold0_pilot_complete", "training": training, "held_inference": evaluation,
            "checkpoint_sha256": sha(checkpoint_path),
            "predictions_sha256": sha(directory / "predictions.npz"),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_before_load": free, "total_memory": total,
            "calibration_rows_read": 0, "calibration_labels_used": False,
            "official_test_opened": False, "baselines_modified": False,
            "seconds": time.perf_counter() - started,
        })
        print("TYPEAWARE_V5_FOLD0_PILOT_COMPLETE", round(time.perf_counter() - started, 1), flush=True)
    finally:
        del model, optimizer, inputs
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check", "cpu-tiny", "finalize", "pilot-fold0"))
    args = parser.parse_args()
    {
        "prepare": prepare,
        "check": check,
        "cpu-tiny": cpu_tiny,
        "finalize": finalize,
        "pilot-fold0": pilot_fold0,
    }[args.command]()


if __name__ == "__main__":
    main()
