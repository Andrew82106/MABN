"""GPU runner for the frozen within-answer contrastive-v5 continuation.

This runner never changes expanded-v4 or any formal baseline.  Fold f starts
from the matching expanded-v4 fold checkpoint, trains only fit pairs outside f,
and scores every held claim.  Full-fit starts from the matching v4 checkpoint,
trains only fit pairs, and scores calibration without reading its labels.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_microclaim_crossencoder_v1 as base  # noqa: E402
import run_microclaim_crossencoder_expanded_v4 as expanded  # noqa: E402
import run_microclaim_within_answer_contrastive_v5 as design  # noqa: E402


DESIGN = ROOT / "results/microclaim_within_answer_contrastive_v5"
V4_RUN = ROOT / "results/microclaim_crossencoder_expanded_v4"
V4_DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
OUT = ROOT / "results/microclaim_within_answer_contrastive_v5_gpu_v2"
AUDITOR = HERE / "audit_microclaim_within_answer_contrastive_v5_gpu.py"
DESIGN_COMPLETE_SHA256 = "5de78361f49fff89b49fc5923a493a1c16ab4111f2418b4bfc7a953d26d55967"
PAIR_ARRAYS_SHA256 = "28c9d1c4b0e49960773dcb35709075112ce3bd54622a85391d53fd6eb44ae669"
V4_COMPLETE_SHA256 = "fa2d4c631e560a7aa4b891f7b1279d1b9026893c8f5d1617cfe8ee268e4d2cb3"
FOLDS = 5
FIT_CLAIMS = 34_919
CAL_CLAIMS = 2_267
SEED = design.SEED
PAIR_LAMBDA = design.PAIR_LAMBDA
ACCUM_MICROBATCHES = design.ACCUM_MICROBATCHES
PEAK_LIMIT = int(7.75 * 1024**3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); assert not path.exists(), f"Refuse overwrite: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_npz(path, **arrays):
    path = Path(path); assert not path.exists(), f"Refuse overwrite: {path}"
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_rows():
    with (DESIGN / "pairs.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pair_arrays():
    with np.load(DESIGN / "pair_arrays.npz", allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def source_files(include_checkpoints=True):
    paths = [Path(__file__), AUDITOR, DESIGN / "complete.json", DESIGN / "pairs.jsonl",
             DESIGN / "pair_arrays.npz", DESIGN / "INDEPENDENT_AUDIT.json",
             V4_RUN / "preparation_complete.json", V4_RUN / "encoded_inputs.npz",
             V4_RUN / "complete.json", V4_RUN / "summary.json", V4_RUN / "scores.npz",
             V4_DATA / "arrays.npz", V4_DATA / "answers.jsonl",
             base.MODEL / "model.safetensors"]
    for fold in range(FOLDS):
        paths.append(V4_RUN / f"fold_{fold}/complete.json")
        if include_checkpoints: paths.append(V4_RUN / f"fold_{fold}/model.pt")
    paths.append(V4_RUN / "full_fit/complete.json")
    if include_checkpoints: paths.append(V4_RUN / "full_fit/model.pt")
    return {str(path.resolve()): sha(path) for path in paths}


def protocol():
    return {
        "version": "microclaim-within-answer-contrastive-v5-gpu-v2",
        "frozen_design_complete_sha256": DESIGN_COMPLETE_SHA256,
        "initialization": "Each OOF/full-fit model loads only its matching completed expanded-v4 model.pt; fresh AdamW state.",
        "architecture": "Unchanged ModernBERT-base NLI encoder/head and unchanged scalar risk logit; no added parameters.",
        "training": {
            "data": "Frozen fit-only pair manifest; no calibration/test labels.",
            "extra_epochs": 1, "scope": "Selected pair endpoints only",
            "loss": "occurrence-corrected v4 endpoint BCE + 0.25 * group/positive-balanced RankNet",
            "lambda": PAIR_LAMBDA, "optimizer": "AdamW", "lr": design.LEARNING_RATE,
            "weight_decay": design.WEIGHT_DECAY, "clip_norm": 1.0,
            "accumulated_pair_microbatches": ACCUM_MICROBATCHES,
            "precision": "FP32 parameters/optimizer and BF16 CUDA forward; TF32 off",
        },
        "crossfit": "Fold f excludes every pair and claim whose source-connected held_fold=f; full-fit trains all fit pairs.",
        "prediction": "Each fold scores every held claim; full-fit scores calibration claims. No label is needed for inference.",
        "evaluation": "After all predictions are frozen, reuse exact expanded-v4 4-BPE projection; select thresholds on fit OOF only and apply to calibration.",
        "selection": "One frozen candidate; calibration never selects pair rule, lambda, LR, epoch or checkpoint.",
        "control": "Completed expanded-v4 scores are a read-only warm-start reference.",
        "baselines": "No baseline model, parameter, score, or threshold is written.",
        "official_test_opened": False,
    }


def verify_frozen_inputs(hash_checkpoints=False):
    assert sha(DESIGN / "complete.json") == DESIGN_COMPLETE_SHA256
    assert sha(DESIGN / "pair_arrays.npz") == PAIR_ARRAYS_SHA256
    assert sha(V4_RUN / "complete.json") == V4_COMPLETE_SHA256
    assert read(DESIGN / "complete.json")["status"] == "CPU_design_complete_waiting_for_GPU_authorization"
    assert read(V4_RUN / "complete.json")["status"] == "complete_development_only"
    for fold in range(FOLDS):
        directory = V4_RUN / f"fold_{fold}"; item = read(directory / "complete.json")
        assert item["status"] == "fold_complete" and item["fold"] == fold
        if hash_checkpoints: assert sha(directory / "model.pt") == item["checkpoint_sha256"]
    item = read(V4_RUN / "full_fit/complete.json")
    assert item["status"] == "full_fit_complete"
    if hash_checkpoints: assert sha(V4_RUN / "full_fit/model.pt") == item["checkpoint_sha256"]


def prepare():
    assert not torch.cuda.is_initialized() and not OUT.exists()
    verify_frozen_inputs(True); snapshot = source_files(True)
    pairs = pair_rows(); parray = pair_arrays(); barrays = design.load_base_arrays()
    assert len(pairs) == len(parray["positive_index"]) == 6637
    assert len(barrays["lengths"]) == FIT_CLAIMS + CAL_CLAIMS
    plan = read(DESIGN / "training_plan.json")
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol())
    save_json(OUT / "preparation.json", {
        "status": "GPU_runner_prepared_not_started", "pairs": len(pairs),
        "fit_claims": FIT_CLAIMS, "calibration_claims": CAL_CLAIMS,
        "pair_plan": plan["folds"], "runtime_prior": plan["runtime"],
        "all_six_matching_v4_checkpoints_verified": True,
        "source_sha256": snapshot, "GPU_used": False,
        "calibration_labels_used": False, "official_test_opened": False,
        "formal_baselines_modified": False,
    })
    names = ("protocol.json", "preparation.json")
    save_json(OUT / "preparation_complete.json", {
        "status": "GPU_runner_prepared_not_started",
        "files_sha256": {name: sha(OUT / name) for name in names},
        "source_sha256": snapshot, "GPU_used": False,
        "calibration_labels_used": False, "official_test_opened": False,
        "formal_baselines_modified": False,
    })
    assert snapshot == source_files(True) and not torch.cuda.is_initialized()
    print("WITHIN_ANSWER_V5_GPU_RUNNER_PREPARED", len(pairs), flush=True)


def check_prepared(verify_checkpoint_files=False):
    verify_frozen_inputs(verify_checkpoint_files)
    complete = read(OUT / "preparation_complete.json")
    assert complete["status"] == "GPU_runner_prepared_not_started"
    for name, expected in complete["files_sha256"].items(): assert sha(OUT / name) == expected
    current = source_files(verify_checkpoint_files)
    for path, expected in complete["source_sha256"].items():
        if verify_checkpoint_files or not path.endswith("model.pt"):
            assert current[path] == expected, path
    assert read(OUT / "protocol.json") == protocol()
    pairs = pair_rows(); parray = pair_arrays(); barrays = design.load_base_arrays()
    assert np.array_equal(parray["positive_index"], [row["positive_index"] for row in pairs])
    assert np.array_equal(parray["negative_index"], [row["negative_index"] for row in pairs])
    assert np.all(barrays["fold_assignment"][parray["positive_index"]] == parray["held_fold"])
    assert np.all(barrays["fold_assignment"][parray["negative_index"]] == parray["held_fold"])
    return pairs, parray, barrays


def checkpoint_for(fold):
    return V4_RUN / ("full_fit" if fold == FOLDS else f"fold_{fold}") / "model.pt"


def load_warm_model(fold, device):
    path = checkpoint_for(fold)
    meta = read(path.parent / "complete.json")
    assert sha(path) == meta["checkpoint_sha256"]
    model = base.load_model(device)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert int(state["fold"]) == fold
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=True)
    assert not missing and not unexpected
    del state
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, meta["checkpoint_sha256"]


def occurrence_weights(fold, parray, barrays):
    active = np.flatnonzero(np.ones(len(parray["held_fold"]), dtype=bool) if fold == FOLDS
                            else parray["held_fold"] != fold)
    positive, negative = parray["positive_index"], parray["negative_index"]
    counts = np.bincount(np.concatenate((positive[active], negative[active])), minlength=len(barrays["lengths"]))
    base_weights = barrays["weights"][fold]
    pos_weight = np.divide(base_weights[positive], counts[positive],
                           out=np.zeros(len(positive), dtype=np.float32), where=counts[positive] > 0)
    neg_weight = np.divide(base_weights[negative], counts[negative],
                           out=np.zeros(len(negative), dtype=np.float32), where=counts[negative] > 0)
    assert np.all(pos_weight[active] > 0) and np.all(neg_weight[active] > 0)
    return active, pos_weight.astype(np.float32), neg_weight.astype(np.float32)


def configure_gpu(seed):
    free, total = base.configure_gpu(seed)
    return free, total, torch.device("cuda:0")


def pair_batches(active, pairs, barrays, fold):
    return design.make_pair_batches(active, pairs, barrays["lengths"], SEED + fold)


def train_pair_epoch(model, optimizer, pairs, parray, barrays, fold, device,
                     batches_override=None, progress=True):
    active, pos_weight, neg_weight = occurrence_weights(fold, parray, barrays)
    batches = batches_override or pair_batches(active, pairs, barrays, fold)
    expected = sorted(int(index) for batch in batches for index in batch)
    if batches_override is None: assert expected == sorted(active.tolist())
    model.train(); start = time.perf_counter(); updates = seen = logical = padded = 0
    bce_sum = bce_mass = rank_sum = rank_mass = correct = 0.0
    norms, dtypes = [], set()
    for first in range(0, len(batches), ACCUM_MICROBATCHES):
        group = batches[first:first + ACCUM_MICROBATCHES]
        group_indices = [int(index) for batch in group for index in batch]
        bce_den = float(sum(float(pos_weight[i] + neg_weight[i]) for i in group_indices))
        rank_den = float(sum(float(parray["rank_weights"][fold, i]) for i in group_indices))
        assert bce_den > 0 and rank_den > 0
        optimizer.zero_grad(set_to_none=True)
        for batch_indices in group:
            batch_indices = list(map(int, batch_indices))
            endpoints = [claim for pair_index in batch_indices for claim in
                         (int(parray["positive_index"][pair_index]), int(parray["negative_index"][pair_index]))]
            encoded, lengths = base.collate(barrays, endpoints, device)
            risk, dtype = base.forward_risk(model, encoded, device, True); dtypes.add(dtype)
            pos = torch.arange(0, len(endpoints), 2, device=device); neg = pos + 1
            labels = torch.zeros(len(endpoints), dtype=torch.float32, device=device); labels[pos] = 1
            endpoint_weights = torch.as_tensor(
                [value for pair_index in batch_indices for value in
                 (pos_weight[pair_index], neg_weight[pair_index])], device=device)
            rank_weights = torch.as_tensor(parray["rank_weights"][fold, batch_indices], device=device)
            bce_each = F.binary_cross_entropy_with_logits(risk, labels, reduction="none")
            rank_each = F.softplus(risk[neg] - risk[pos])
            bce_num = (bce_each * endpoint_weights).sum()
            rank_num = (rank_each * rank_weights).sum()
            (bce_num / bce_den + PAIR_LAMBDA * rank_num / rank_den).backward()
            bce_sum += float(bce_num.detach()); bce_mass += float(endpoint_weights.sum())
            rank_sum += float(rank_num.detach()); rank_mass += float(rank_weights.sum())
            correct += float((risk[pos] > risk[neg]).sum().detach()); seen += len(batch_indices)
            logical += int(lengths.sum()); padded += len(endpoints) * int(lengths.max())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step(); updates += 1; norms.append(float(norm))
        if progress and (updates % 100 == 0 or first + len(group) >= len(batches)):
            print("WITHIN_ANSWER_V5_TRAIN", fold, updates,
                  math.ceil(len(batches) / ACCUM_MICROBATCHES), round(time.perf_counter() - start, 1), flush=True)
    return {
        "fold": fold, "pairs": seen, "pair_microbatches": len(batches), "optimizer_updates": updates,
        "logical_endpoint_tokens": logical, "padded_endpoint_tokens": padded,
        "online_weighted_endpoint_bce": bce_sum / bce_mass,
        "online_weighted_ranknet": rank_sum / rank_mass,
        "online_pair_order_accuracy": correct / seen,
        "lambda": PAIR_LAMBDA, "gradient_norm_min": min(norms), "gradient_norm_max": max(norms),
        "forward_logits_dtypes": sorted(dtypes), "seconds": time.perf_counter() - start,
    }


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=design.LEARNING_RATE,
                             weight_decay=design.WEIGHT_DECAY)


def cleanup(*objects):
    del objects; gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache(); torch.cuda.synchronize()


def gpu_smoke():
    pairs, parray, barrays = check_prepared(True)
    assert not (OUT / "GPU_SMOKE.json").exists()
    free, total, device = configure_gpu(SEED); model = optimizer = None
    try:
        model, checkpoint_hash = load_warm_model(0, device)
        active, _, _ = occurrence_weights(0, parray, barrays)
        all_batches = pair_batches(active, pairs, barrays, 0)
        ranked = sorted(all_batches, key=lambda batch: 2 * len(batch) * max(
            max(int(barrays["lengths"][parray["positive_index"][i]]),
                int(barrays["lengths"][parray["negative_index"][i]])) for i in batch), reverse=True)
        smoke_batches = [ranked[0], ranked[len(ranked)//3], ranked[2*len(ranked)//3], ranked[-1]]
        endpoints = [claim for batch in smoke_batches[:1] for pair_index in batch for claim in
                     (int(parray["positive_index"][pair_index]), int(parray["negative_index"][pair_index]))]
        model.eval(); encoded, _ = base.collate(barrays, endpoints, device)
        with torch.inference_mode():
            one, _ = base.forward_risk(model, encoded, device, False)
            two, _ = base.forward_risk(model, encoded, device, False)
        repeat = float((one - two).abs().max()); assert repeat <= 2e-6
        del encoded, one, two
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        tick = time.perf_counter()
        training = train_pair_epoch(model, optimizer, pairs, parray, barrays, 0, device,
                                    batches_override=smoke_batches, progress=False)
        torch.cuda.synchronize(); measured = time.perf_counter() - tick
        held = np.flatnonzero(barrays["fold_assignment"] == 0)[:64]
        _, inference = base.score_indices(model, barrays, held, device)
        peak_alloc, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        assert max(peak_alloc, peak_reserved) <= PEAK_LIMIT
        plan = read(DESIGN / "training_plan.json")
        seconds_per_token = measured / training["padded_endpoint_tokens"]
        train_seconds = plan["runtime"]["six_models_pair_padded_tokens"] * seconds_per_token
        result = {
            "status": "passed_disposable_warm_start_no_formal_training",
            "device": torch.cuda.get_device_name(0), "parameters": sum(p.numel() for p in model.parameters()),
            "warm_start_fold": 0, "warm_start_checkpoint_sha256": checkpoint_hash,
            "smoke_pair_indices": [int(index) for batch in smoke_batches for index in batch],
            "repeat_max_abs_difference": repeat, "training": training,
            "measured_training_seconds": measured, "held_inference": inference,
            "peak_cuda_allocated_bytes": peak_alloc, "peak_cuda_reserved_bytes": peak_reserved,
            "free_before_load": free, "total_memory": total,
            "planned_pair_padded_tokens_six_models": plan["runtime"]["six_models_pair_padded_tokens"],
            "point_pair_training_minutes": train_seconds / 60,
            "conservative_total_minutes_including_inference_and_IO": train_seconds * 1.35 / 60 + 8,
            "formal_checkpoint_saved": False, "calibration_labels_used": False,
            "official_test_opened": False, "formal_baselines_modified": False,
            "preparation_sha256": sha(OUT / "preparation_complete.json"),
        }
        save_json(OUT / "GPU_SMOKE.json", result)
        print("WITHIN_ANSWER_V5_GPU_SMOKE_PASSED", round(result["point_pair_training_minutes"], 1), flush=True)
    finally:
        del model, optimizer; cleanup()


def save_checkpoint(model, path, fold, warm_hash):
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    torch.save({"model_state_dict": state, "fold": fold, "seed": SEED + fold,
                "warm_start_checkpoint_sha256": warm_hash,
                "preparation_sha256": sha(OUT / "preparation_complete.json"),
                "protocol_sha256": sha(OUT / "protocol.json")}, path)


def train_one(fold):
    assert 0 <= fold <= FOLDS
    pairs, parray, barrays = check_prepared(False)
    assert read(OUT / "GPU_SMOKE.json")["status"] == "passed_disposable_warm_start_no_formal_training"
    name = "full_fit" if fold == FOLDS else f"fold_{fold}"
    directory = OUT / name; assert not directory.exists(); directory.mkdir()
    save_json(directory / "started.json", {"fold": fold, "seed": SEED + fold,
              "warm_start": str(checkpoint_for(fold).resolve()), "calibration_labels_used": False,
              "official_test_opened": False, "time": time.time()})
    free, total, device = configure_gpu(SEED + fold); model = optimizer = None
    try:
        model, warm_hash = load_warm_model(fold, device)
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats()
        training = train_pair_epoch(model, optimizer, pairs, parray, barrays, fold, device)
        if fold < FOLDS:
            indices = np.flatnonzero(barrays["fold_assignment"] == fold)
        else:
            indices = np.flatnonzero(barrays["fold_assignment"] == -1)
        scores, inference = base.score_indices(model, barrays, indices, device, progress=True)
        checkpoint = directory / "model.pt"; save_checkpoint(model, checkpoint, fold, warm_hash)
        predictions = directory / "predictions.npz"
        if fold < FOLDS: save_npz(predictions, held_indices=indices, held_scores=scores)
        else: save_npz(predictions, calibration_indices=indices, calibration_scores=scores)
        result = {
            "status": "full_fit_complete" if fold == FOLDS else "fold_complete",
            "fold": fold, "seed": SEED + fold, "training": training, "inference": inference,
            "warm_start_checkpoint_sha256": warm_hash,
            "checkpoint_sha256": sha(checkpoint), "predictions_sha256": sha(predictions),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_before_load": free, "total_memory": total,
            "calibration_labels_used_for_training_or_selection": False,
            "threshold_selected": False, "official_test_opened": False,
        }
        save_json(directory / "complete.json", result)
        print("WITHIN_ANSWER_V5_MODEL_COMPLETE", fold, len(indices), flush=True)
    finally:
        del model, optimizer; cleanup()


def assemble_scores(barrays):
    scores = np.full(len(barrays["lengths"]), np.nan, dtype=np.float32)
    for fold in range(FOLDS):
        directory = OUT / f"fold_{fold}"; complete = read(directory / "complete.json")
        assert complete["status"] == "fold_complete"
        assert complete["predictions_sha256"] == sha(directory / "predictions.npz")
        with np.load(directory / "predictions.npz", allow_pickle=False) as values:
            indices, fold_scores = values["held_indices"], values["held_scores"]
        assert np.all(barrays["fold_assignment"][indices] == fold) and np.isnan(scores[indices]).all()
        scores[indices] = fold_scores
    directory = OUT / "full_fit"; complete = read(directory / "complete.json")
    assert complete["status"] == "full_fit_complete"
    with np.load(directory / "predictions.npz", allow_pickle=False) as values:
        indices, values_one = values["calibration_indices"], values["calibration_scores"]
    assert np.array_equal(indices, np.arange(FIT_CLAIMS, FIT_CLAIMS + CAL_CLAIMS))
    scores[indices] = values_one
    assert np.isfinite(scores).all()
    return scores


def finalize():
    pairs, parray, barrays = check_prepared(False); assert not (OUT / "summary.json").exists()
    claim_scores = assemble_scores(barrays)
    v4 = expanded.load_v4_arrays(); answers = expanded.lines(V4_DATA / "answers.jsonl")
    windows, answer_values, metrics = expanded.summarize_scores(answers, v4, claim_scores)
    with np.load(V4_RUN / "scores.npz", allow_pickle=False) as values:
        base_claim = values["main_claim_scores"].copy()
    base_windows, base_answers, base_metrics = expanded.summarize_scores(answers, v4, base_claim)
    save_npz(OUT / "scores.npz", claim_scores=claim_scores, window_scores=windows,
             answer_scores=answer_values, warm_start_claim_scores=base_claim,
             warm_start_window_scores=base_windows, warm_start_answer_scores=base_answers)
    strict = metrics["strict_fit_threshold_to_calibration"]
    reference = base_metrics["strict_fit_threshold_to_calibration"]
    summary = {
        "status": "development_only_complete", "method": "within-answer contrastive ModernBERT v5",
        "trained_candidate": metrics, "warm_start_read_only_reference": base_metrics,
        "strict_calibration_delta": {
            "window_f1": strict["calibration"]["windows"]["f1"] - reference["calibration"]["windows"]["f1"],
            "answer_f1": strict["calibration"]["answers"]["f1"] - reference["calibration"]["answers"]["f1"],
        },
        "five_source_group_OOF_models": True, "separate_full_fit_calibration_model": True,
        "calibration_used_for_training_or_selection": False, "fit_only_thresholds": True,
        "formal_baselines_modified": False, "official_test_opened": False,
        "scores_sha256": sha(OUT / "scores.npz"),
    }
    save_json(OUT / "summary.json", summary)
    report = ["# Within-answer contrastive ModernBERT v5: GPU result", "",
              "每折从对应 expanded-v4 checkpoint 热启动，额外训练同答错误/安全微主张对；模型结构不变。", "",
              "| 方法 | fit OOF窗口F1 | cal严格窗口F1 | fit OOF整答F1 | cal严格整答F1 |",
              "|---|---:|---:|---:|---:|",
              f"| v5 | {strict['fit']['windows']['f1']:.6f} | {strict['calibration']['windows']['f1']:.6f} | {strict['fit']['answers']['f1']:.6f} | {strict['calibration']['answers']['f1']:.6f} |",
              f"| v4 warm-start reference | {reference['fit']['windows']['f1']:.6f} | {reference['calibration']['windows']['f1']:.6f} | {reference['fit']['answers']['f1']:.6f} | {reference['calibration']['answers']['f1']:.6f} |", "",
              "阈值只由 fit OOF 选择；calibration 未参与训练或模型选择。正式 baseline 未改，official test 未打开。", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    save_json(OUT / "complete.json", {"status": "complete_development_only_waiting_for_independent_audit",
              "summary_sha256": sha(OUT / "summary.json"), "scores_sha256": sha(OUT / "scores.npz"),
              "report_sha256": sha(OUT / "REPORT.md"), "official_test_opened": False,
              "formal_baselines_modified": False})
    print("WITHIN_ANSWER_V5_FINALIZED", strict["calibration"]["windows"]["f1"],
          strict["calibration"]["answers"]["f1"], flush=True)


def verify(final=False):
    _, _, barrays = check_prepared(False)
    assert read(OUT / "GPU_SMOKE.json")["status"] == "passed_disposable_warm_start_no_formal_training"
    if final:
        complete = read(OUT / "complete.json")
        assert complete["status"] in ("complete_development_only_waiting_for_independent_audit",
                                      "complete_development_only_independently_audited")
        assert complete["summary_sha256"] == sha(OUT / "summary.json")
        assert complete["scores_sha256"] == sha(OUT / "scores.npz")
        assert complete["report_sha256"] == sha(OUT / "REPORT.md")
        assert np.isfinite(assemble_scores(barrays)).all()
    print("WITHIN_ANSWER_V5_GPU_VERIFIED", "final" if final else "prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "gpu-smoke", "train-fold",
                                          "train-full", "finalize", "verify", "verify-final"))
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.stage == "train-fold": assert args.fold is not None and 0 <= args.fold < FOLDS
    else: assert args.fold is None
    if args.stage == "prepare": prepare()
    elif args.stage == "check": check_prepared(True); print("WITHIN_ANSWER_V5_GPU_CHECK_PASSED", flush=True)
    elif args.stage == "gpu-smoke": gpu_smoke()
    elif args.stage == "train-fold": train_one(args.fold)
    elif args.stage == "train-full": train_one(FOLDS)
    elif args.stage == "finalize": finalize()
    elif args.stage == "verify": verify(False)
    else: verify(True)


if __name__ == "__main__":
    main()
