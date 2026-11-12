"""GPU runner for the pre-registered auxiliary exact-span conflict pilot.

The default/CPU commands never load the full checkpoint or initialize CUDA.
GPU commands are explicit and were not run during CPU preparation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, ModernBertConfig, ModernBertForSequenceClassification


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results/auxiliary_conflict_token_pilot_v1"
MODEL = ROOT.parent / "models/ModernBERT-base-nli"
V4_CHECKPOINT = ROOT / "results/microclaim_crossencoder_expanded_v4/fold_0/model.pt"
OUT = DATA / "gpu_runs"

SEED = 20_261_023
PAD_ID = 50_283
TOKEN_BUDGET = 1_536
MAX_EXAMPLES = 8
ACCUM = 4
EVAL_TOKEN_BUDGET = 4_096
EVAL_MAX_EXAMPLES = 16
LR = 3e-6
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
MIN_FREE_GPU_BYTES = 4_500_000_000
STAGE_CODES = {"aux": 0, "qa_train": 1, "qa_held": 2}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def load_data():
    manifest = read_json(DATA / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert sha(DATA / name) == expected, name
    complete = read_json(DATA / "preparation_complete.json")
    assert complete["scope"]["calibration_rows_read"] == 0
    assert complete["scope"]["test_rows_read"] == 0
    with np.load(DATA / "arrays.npz", allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    return complete, arrays


def make_batches(indices, lengths, seed, token_budget=TOKEN_BUDGET, max_examples=MAX_EXAMPLES):
    indices = np.asarray(indices, dtype=np.int64)
    ordered = indices[np.argsort(lengths[indices], kind="stable")]
    batches, current = [], []
    for raw in ordered:
        index = int(raw)
        proposed = current + [index]
        padded = max(int(lengths[value]) for value in proposed) * len(proposed)
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
    assert sorted(index for batch in batches for index in batch) == sorted(indices.tolist())
    return batches


def batch_tensors(arrays, batch, device, remap_vocab=None):
    indptr = arrays["indptr"]
    lengths = [int(indptr[index + 1] - indptr[index]) for index in batch]
    width = max(lengths)
    pad = 0 if remap_vocab is not None else PAD_ID
    ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
    targets = torch.full((len(batch), width), -1.0, dtype=torch.float32, device=device)
    chars = torch.zeros((len(batch), width), dtype=torch.float32, device=device)
    for row, (index, length) in enumerate(zip(batch, lengths)):
        left, right = int(indptr[index]), int(indptr[index + 1])
        value = arrays["input_ids"][left:right].astype(np.int64)
        if remap_vocab is not None:
            value %= remap_vocab
        ids[row, :length] = torch.from_numpy(value).to(device)
        attention[row, :length] = True
        targets[row, :length] = torch.from_numpy(arrays["targets"][left:right]).to(device)
        chars[row, :length] = torch.from_numpy(arrays["char_counts"][left:right].astype(np.float32)).to(device)
    weights = torch.from_numpy(arrays["example_weight"][batch].astype(np.float32)).to(device)
    return ids, attention, targets, chars, weights, lengths


def token_logits(model, input_ids, attention_mask):
    output = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = model.head(output.last_hidden_state)
    hidden = model.drop(hidden)
    return model.classifier(hidden)


def conflict_logit(logits):
    assert logits.shape[-1] == 3
    return logits[..., 2].float() - torch.logsumexp(logits[..., :2].float(), dim=-1)


def exact_span_loss(logits, targets, chars, example_weights, positive_factor, negative_factor):
    score = conflict_logit(logits)
    valid = targets >= 0
    assert torch.all(chars[~valid] == 0)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    token_loss = (positive_factor * safe_targets * F.softplus(-score) +
                  negative_factor * (1.0 - safe_targets) * F.softplus(score))
    mass = chars.sum(dim=1)
    assert torch.all(mass > 0)
    per_example = (token_loss * chars).sum(dim=1) / mass
    numerator = (per_example * example_weights).sum()
    return numerator, example_weights.sum(), per_example.detach()


def initialize_model(device):
    checkpoint = torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["fold"] == 0 and checkpoint["seed"] == SEED
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def batch_plan(arrays):
    lengths = np.diff(arrays["indptr"])
    aux = np.flatnonzero(arrays["stage_code"] == STAGE_CODES["aux"])
    qa = np.flatnonzero(arrays["stage_code"] == STAGE_CODES["qa_train"])
    held = np.flatnonzero(arrays["stage_code"] == STAGE_CODES["qa_held"])
    aux_batches = make_batches(aux, lengths, SEED)
    qa_batches = make_batches(qa, lengths, SEED + 1)
    held_batches = make_batches(held, lengths, SEED + 2, EVAL_TOKEN_BUDGET, EVAL_MAX_EXAMPLES)
    control_pre = [qa_batches[index % len(qa_batches)] for index in range(len(aux_batches))]
    return lengths, aux_batches, qa_batches, held_batches, control_pre


def plan_stats(batches, lengths):
    return {
        "microbatches": len(batches), "optimizer_updates": math.ceil(len(batches) / ACCUM),
        "examples_with_repetition": sum(len(batch) for batch in batches),
        "logical_tokens": int(sum(sum(int(lengths[index]) for index in batch) for batch in batches)),
        "padded_tokens": int(sum(max(int(lengths[index]) for index in batch) * len(batch) for batch in batches)),
        "max_sequence_tokens": int(max(max(int(lengths[index]) for index in batch) for batch in batches)),
    }


def estimate():
    complete, arrays = load_data()
    lengths, aux, qa, held, control = batch_plan(arrays)
    v4 = read_json(ROOT / "results/microclaim_crossencoder_expanded_v4/fold_0/complete.json")
    throughput = v4["training"]["padded_tokens"] / v4["training"]["seconds"]
    plans = {"candidate_aux": plan_stats(aux, lengths), "candidate_final_qa": plan_stats(qa, lengths),
             "control_qa_replacement": plan_stats(control, lengths), "control_final_qa": plan_stats(qa, lengths),
             "held_inference_each_arm": plan_stats(held, lengths)}
    total_padded = sum(plans[name]["padded_tokens"] for name in
                       ("candidate_aux", "candidate_final_qa", "control_qa_replacement", "control_final_qa"))
    inference_padded = 2 * plans["held_inference_each_arm"]["padded_tokens"]
    raw_seconds = (total_padded + inference_padded) / throughput
    output = {
        "status": "cpu_estimate_only", "plans": plans,
        "v4_measured_padded_tokens_per_second": throughput,
        "point_seconds_two_arms_plus_inference": raw_seconds,
        "conservative_seconds_x1_75_plus_load_save": raw_seconds * 1.75 + 60,
        "gpu_smoke_not_run": True, "GPU_used": False,
        "calibration_rows_read": 0, "test_rows_read": 0,
    }
    save_json(DATA / "GPU_PLAN.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cpu_check():
    _complete, arrays = load_data()
    lengths, aux, qa, _held, _control = batch_plan(arrays)
    batch = aux[0][:2] if len(aux[0]) >= 2 else (aux[0] + aux[1])[:2]
    config = ModernBertConfig(vocab_size=128, hidden_size=48, intermediate_size=96,
                              num_hidden_layers=2, num_attention_heads=4,
                              max_position_embeddings=2048, num_labels=3,
                              classifier_dropout=0.0, attention_dropout=0.0,
                              embedding_dropout=0.0, mlp_dropout=0.0,
                              local_attention=32, pad_token_id=0,
                              bos_token_id=1, eos_token_id=2, sep_token_id=2,
                              classifier_pooling="mean", reference_compile=False)
    config._attn_implementation = "sdpa"
    torch.manual_seed(SEED)
    model = ModernBertForSequenceClassification(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, mask, targets, chars, weights, _ = batch_tensors(arrays, batch, "cpu", remap_vocab=128)
    before_encoder = model.model.layers[0].attn.Wqkv.weight.detach().clone()
    before_classifier = model.classifier.weight.detach().clone()
    logits = token_logits(model, ids, mask)
    numerator, denominator, per_example = exact_span_loss(
        logits, targets, chars, weights, positive_factor=1.7, negative_factor=0.8)
    loss = numerator / denominator
    assert torch.isfinite(loss) and torch.isfinite(per_example).all()
    loss.backward(); optimizer.step()
    assert not torch.equal(before_encoder, model.model.layers[0].attn.Wqkv.weight)
    assert not torch.equal(before_classifier, model.classifier.weight)
    result = {
        "status": "passed", "tiny_examples": len(batch), "loss": float(loss),
        "encoder_updated": True, "nli_classifier_updated": True,
        "exact_hypothesis_token_mask_exercised": True,
        "aux_batches": len(aux), "qa_batches": len(qa),
        "model_checkpoint_loaded": False, "GPU_used": False,
        "calibration_rows_read": 0, "test_rows_read": 0,
    }
    save_json(DATA / "CPU_CHECK.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def grouped_train(model, optimizer, arrays, batches, stage_name, balance, device, log_prefix):
    model.train()
    started = time.perf_counter()
    losses, grad_norms, updates = [], [], 0
    for group_start in range(0, len(batches), ACCUM):
        group = batches[group_start:group_start + ACCUM]
        denominator = float(sum(arrays["example_weight"][index]
                                for batch in group for index in batch))
        assert denominator > 0
        optimizer.zero_grad(set_to_none=True)
        group_numerator = 0.0
        for batch in group:
            ids, mask, targets, chars, weights, _ = batch_tensors(arrays, batch, device)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                logits = token_logits(model, ids, mask)
                numerator, _local_denominator, _ = exact_span_loss(
                    logits, targets, chars, weights,
                    balance["positive_factor"], balance["negative_factor"])
                scaled = numerator / denominator
            scaled.backward()
            group_numerator += float(numerator.detach())
            del ids, mask, targets, chars, weights, logits, numerator, scaled
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM))
        assert math.isfinite(norm)
        optimizer.step(); updates += 1
        losses.append(group_numerator / denominator); grad_norms.append(norm)
        if updates % 100 == 0 or updates == math.ceil(len(batches) / ACCUM):
            print(log_prefix, stage_name, updates, math.ceil(len(batches) / ACCUM),
                  round(time.perf_counter() - started, 1), flush=True)
    return {"stage": stage_name, **plan_stats(batches, np.diff(arrays["indptr"])),
            "mean_update_loss": float(np.mean(losses)),
            "gradient_norm_min": min(grad_norms), "gradient_norm_max": max(grad_norms),
            "seconds": time.perf_counter() - started}


@torch.no_grad()
def held_inference(model, arrays, held_batches, device):
    model.eval()
    flat_scores = np.full(len(arrays["input_ids"]), np.nan, dtype=np.float32)
    indptr = arrays["indptr"]
    started = time.perf_counter()
    for batch_index, batch in enumerate(held_batches):
        ids, mask, _targets, _chars, _weights, lengths = batch_tensors(arrays, batch, device)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            logits = token_logits(model, ids, mask)
            scores = torch.softmax(logits.float(), dim=-1)[..., 2].cpu().numpy()
        for row, (index, length) in enumerate(zip(batch, lengths)):
            left, right = int(indptr[index]), int(indptr[index + 1])
            flat_scores[left:right] = scores[row, :length]
        if (batch_index + 1) % 100 == 0:
            print("AUX_CONFLICT_HELD", batch_index + 1, len(held_batches), flush=True)
    values, window_indptr = arrays["held_window_token_values"], arrays["held_window_token_indptr"]
    unique = np.unique(values)
    assert np.isfinite(flat_scores[unique]).all()
    window = np.empty(len(window_indptr) - 1, dtype=np.float32)
    for index in range(len(window)):
        window[index] = np.max(flat_scores[values[window_indptr[index]:window_indptr[index + 1]]])
    answers = np.zeros(len(arrays["held_answer_label"]), dtype=np.float32)
    for index, score in zip(arrays["held_window_answer_index"], window):
        answers[int(index)] = max(answers[int(index)], float(score))
    return window, answers, unique, flat_scores[unique], time.perf_counter() - started


def gpu_preflight():
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    assert free >= MIN_FREE_GPU_BYTES, (free, total)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    random.seed(SEED); np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    return free, total


def gpu_smoke():
    complete, arrays = load_data()
    lengths, aux, _qa, _held, _control = batch_plan(arrays)
    free, total = gpu_preflight(); device = torch.device("cuda")
    started = time.perf_counter(); model = initialize_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    balance = complete["stage_balance"]["aux"]
    batch = aux[0]
    ids, mask, targets, chars, weights, _ = batch_tensors(arrays, batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = token_logits(model, ids, mask)
        numerator, denominator, _ = exact_span_loss(logits, targets, chars, weights,
                                                     balance["positive_factor"], balance["negative_factor"])
        loss = numerator / denominator
    loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)); optimizer.step()
    result = {"status": "passed", "batch_examples": len(batch),
              "padded_tokens": int(max(lengths[batch]) * len(batch)),
              "loss": float(loss), "gradient_norm": norm,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "free_before_load": free, "total_memory": total,
              "seconds": time.perf_counter() - started,
              "calibration_rows_read": 0, "test_rows_read": 0}
    save_json(DATA / "GPU_SMOKE.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def train_arm(arm):
    assert arm in ("candidate", "control")
    complete, arrays = load_data()
    lengths, aux, qa, held, control = batch_plan(arrays)
    free, total = gpu_preflight(); device = torch.device("cuda")
    directory = OUT / arm
    assert not directory.exists(); directory.mkdir(parents=True)
    model = initialize_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if arm == "candidate":
        first = grouped_train(model, optimizer, arrays, aux, "auxiliary_exact_conflict",
                              complete["stage_balance"]["aux"], device, "AUX_CONFLICT_TRAIN")
    else:
        first = grouped_train(model, optimizer, arrays, control, "qa_same_update_replacement",
                              complete["stage_balance"]["qa_train"], device, "AUX_CONFLICT_CONTROL")
    second = grouped_train(model, optimizer, arrays, qa, "final_qa_conflict",
                           complete["stage_balance"]["qa_train"], device, "AUX_CONFLICT_TRAIN")
    window, answer, positions, position_scores, inference_seconds = held_inference(model, arrays, held, device)
    checkpoint = {"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "arm": arm, "seed": SEED, "data_manifest_sha256": sha(DATA / "manifest.json")}
    torch.save(checkpoint, directory / "model.pt")
    np.savez_compressed(directory / "predictions.npz", conflict_window_scores=window,
                        conflict_answer_scores=answer, held_flat_positions=positions,
                        held_flat_conflict_scores=position_scores)
    result = {"status": "complete", "arm": arm, "first_stage": first, "final_qa_stage": second,
              "held_inference_seconds": inference_seconds,
              "checkpoint_sha256": sha(directory / "model.pt"),
              "predictions_sha256": sha(directory / "predictions.npz"),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "free_before_load": free, "total_memory": total,
              "seconds": time.perf_counter() - started,
              "calibration_rows_read": 0, "test_rows_read": 0,
              "published_baselines_modified": False}
    save_json(directory / "complete.json", result)
    print("AUXILIARY_CONFLICT_ARM_COMPLETE", arm, round(result["seconds"], 1), flush=True)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()


def choose_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable"); ss, yy = scores[order], labels[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(yy)[last]]; n = np.r_[0, last + 1]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[last]]
    f1 = 2 * tp / (n + labels.sum())
    precision = np.divide(tp, n, out=np.zeros(len(n)), where=n > 0)
    best = max(range(len(n)), key=lambda index: (f1[index], precision[index], thresholds[index]))
    return float(thresholds[best])


def count(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1))); fp = int(np.count_nonzero(predicted & (labels == 0)))
    fn = int(np.count_nonzero(~predicted & (labels == 1))); tn = int(np.count_nonzero(~predicted & (labels == 0)))
    return {"n": len(labels), "positive": int(labels.sum()), "threshold": threshold,
            "predicted_positive": int(predicted.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}, predicted


def top_k(scores, k):
    order = np.argsort(-np.asarray(scores), kind="stable")
    mask = np.zeros(len(order), dtype=bool); mask[order[:k]] = True
    return mask


def finalize():
    _complete, arrays = load_data()
    y = arrays["held_window_label"]; conflict_y = np.maximum(arrays["held_window_ec"], arrays["held_window_sc"])
    v4 = arrays["held_v4_window_score"].astype(np.float64)
    v4_threshold = choose_threshold(y, v4)
    v4_metrics, v4_pred = count(y, v4, v4_threshold)
    budget = v4_metrics["predicted_positive"]
    output = {"v4": {"overall": v4_metrics,
                      "conflict_ap": float(average_precision_score(conflict_y, v4))}, "arms": {}}
    scores_by_arm = {}
    for arm in ("candidate", "control"):
        directory = OUT / arm; done = read_json(directory / "complete.json")
        assert sha(directory / "predictions.npz") == done["predictions_sha256"]
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            conflict = z["conflict_window_scores"].astype(np.float64)
        combined = np.maximum(v4, conflict)
        scores_by_arm[arm] = (conflict, combined)
        threshold = choose_threshold(y, combined)
        overall, _ = count(y, combined, threshold)
        matched = top_k(combined, budget)
        high = arrays["held_window_max_evidence_coverage"] >= 0.5
        output["arms"][arm] = {
            "combined_overall": overall,
            "conflict_only_ap": float(average_precision_score(conflict_y, conflict)),
            "combined_conflict_ap": float(average_precision_score(conflict_y, combined)),
            "high_overlap_conflict_ap": float(average_precision_score(conflict_y[high], conflict[high])),
            "matched_v4_budget": {"budget": budget,
                "tp": int(np.count_nonzero(matched & (y == 1))),
                "fp": int(np.count_nonzero(matched & (y == 0))),
                "ec_recall": float(np.count_nonzero(matched & (arrays["held_window_ec"] == 1)) /
                                   arrays["held_window_ec"].sum()),
                "sc_recall": float(np.count_nonzero(matched & (arrays["held_window_sc"] == 1)) /
                                   arrays["held_window_sc"].sum()),
                "combined_conflict_recall": float(np.count_nonzero(matched & (conflict_y == 1)) /
                                                  conflict_y.sum())},
        }
    candidate, control = output["arms"]["candidate"], output["arms"]["control"]
    base_conflict_recall = float(np.count_nonzero(v4_pred & (conflict_y == 1)) / conflict_y.sum())
    gate = (candidate["conflict_only_ap"] > control["conflict_only_ap"] and
            candidate["matched_v4_budget"]["combined_conflict_recall"] >= base_conflict_recall + 0.02 and
            candidate["matched_v4_budget"]["fp"] <= v4_metrics["fp"] and
            candidate["combined_overall"]["average_precision"] >= v4_metrics["average_precision"] - 0.002)
    output.update({"v4_alert_budget": budget, "v4_matched_conflict_recall": base_conflict_recall,
                   "pilot_gate_passed": bool(gate), "calibration_rows_read": 0,
                   "test_rows_read": 0, "published_baselines_modified": False})
    save_json(OUT / "summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("estimate", "cpu-check", "gpu-smoke", "train-arm", "finalize"))
    parser.add_argument("--arm", choices=("candidate", "control"))
    args = parser.parse_args()
    if args.stage == "train-arm":
        assert args.arm is not None
        train_arm(args.arm)
    else:
        assert args.arm is None
        {"estimate": estimate, "cpu-check": cpu_check, "gpu-smoke": gpu_smoke,
         "finalize": finalize}[args.stage]()
