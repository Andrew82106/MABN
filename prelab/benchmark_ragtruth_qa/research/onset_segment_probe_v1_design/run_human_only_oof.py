#!/usr/bin/env python3
"""CPU-only strict group-OOF diagnostic for frozen OSR-Linear-68.

This diagnostic reads only RAGTruth human fit rows and their corresponding
white-box feature cache entries.  It never opens a calibration/test label file,
does not import a GPU framework, and cannot mutate an existing baseline.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
ANSWERS = BENCH / "data" / "answers_fit.jsonl"
TOKENS = BENCH / "data" / "tokens_fit.jsonl"
FEATURES = BENCH / "data" / "features"
OUT = HERE / "human_only_oof_v1"

NAMESPACE = "ragtruth_fit_human"
FOLDS = 5
DIM = 68
WIDTH = 4
HIDDEN_SEED = 20260913
LOOKBACK_SEED = 20260914
TRAIN_SEED = 20260913
EPOCHS = 25
LEARNING_RATE = 0.0003
BATCH_ANSWERS = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl_gz(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if path.name not in {"answers_fit.jsonl", "tokens_fit.jsonl"}:
        raise RuntimeError(f"Refusing non-fit JSONL: {path}")
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("partition") != "fit":
                    raise AssertionError("non-fit row encountered")
                result.append(row)
    return result


def keyed(rows: list[dict]) -> dict[str, dict]:
    output = {str(row["response_id"]): row for row in rows}
    if len(output) != len(rows):
        raise AssertionError("duplicate response_id")
    return output


def stable_fold(group_id: str) -> int:
    raw = hashlib.sha256(f"{NAMESPACE}\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % FOLDS


def windows_for_count(count: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise AssertionError("empty raw token axis")
    return [(left, min(left + WIDTH, count)) for left in range(max(1, count - WIDTH + 1))]


def rademacher(width: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.integers(0, 2, size=(width, 32), dtype=np.int8)
    return ((2.0 * signs.astype(np.float32) - 1.0) / np.float32(math.sqrt(32.0))).astype(np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def released_onsets(token: dict) -> set[int]:
    onsets = set()
    for mapping in token["span_token_mapping"]:
        indices = [int(v) for v in mapping["risk_token_indices"]]
        if indices:
            onsets.add(indices[0])
    return onsets


def build_features(nll: np.ndarray, hidden: np.ndarray, lb: np.ndarray,
                   lexical_raw: np.ndarray, raw_count: int,
                   hidden_projection: np.ndarray,
                   lb_projection: np.ndarray) -> np.ndarray:
    if hidden.shape != (raw_count, 4096) or lb.shape != (raw_count, 1024) or nll.shape != (raw_count,):
        raise AssertionError("unexpected cached feature shape")
    lexical_nll = nll[lexical_raw].astype(np.float32)
    previous = np.r_[np.float32(lexical_nll[0]), lexical_nll[:-1]]
    delta = lexical_nll - previous
    delta[0] = 0.0
    cumulative = np.r_[np.float64(0.0), np.cumsum(lexical_nll, dtype=np.float64)]
    rolling = np.empty(len(lexical_nll), dtype=np.float32)
    for index in range(len(lexical_nll)):
        left = max(0, index - 3)
        rolling[index] = (cumulative[index + 1] - cumulative[left]) / (index + 1 - left)
    position = lexical_raw.astype(np.float32) / np.float32(max(1, raw_count - 1))
    projected_hidden = hidden[lexical_raw] @ hidden_projection
    projected_lb = lb[lexical_raw] @ lb_projection
    result = np.column_stack((projected_hidden, projected_lb, lexical_nll, delta, rolling, position))
    if result.shape != (len(lexical_raw), DIM) or result.dtype != np.float32:
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise AssertionError("non-finite derived feature")
    return result


def load_dataset(verify_npz_hashes: bool = True) -> tuple[np.ndarray, list[dict], dict]:
    answers = keyed(read_jsonl(ANSWERS))
    tokens = keyed(read_jsonl(TOKENS))
    if set(answers) != set(tokens) or len(answers) != 634:
        raise AssertionError("fit ID/count mismatch")
    hidden_projection = rademacher(4096, HIDDEN_SEED)
    lb_projection = rademacher(1024, LOOKBACK_SEED)
    chunks = []
    records = []
    feature_hash_rows = []
    cursor = 0
    for rid in sorted(answers):
        answer, token = answers[rid], tokens[rid]
        if answer["group_id"] != token["group_id"] or answer["original_response"] != token["original_response"]:
            raise AssertionError("answer/token mismatch")
        raw_count = int(token["token_count"])
        lexical = np.asarray(token["lexical_mask"], dtype=np.uint8)
        risk = np.asarray(token["risk_mask"], dtype=np.uint8)
        offsets = np.asarray(token["response_token_offsets"], dtype=np.int32)
        if not (len(lexical) == len(risk) == len(offsets) == raw_count):
            raise AssertionError("token geometry mismatch")
        if np.any((risk == 1) & (lexical == 0)):
            raise AssertionError("nonlexical risk token")
        lexical_raw = np.flatnonzero(lexical).astype(np.int32)
        states = risk[lexical_raw].astype(np.uint8)
        starts = released_onsets(token)
        if not starts.issubset(set(map(int, lexical_raw))):
            raise AssertionError("released onset is nonlexical")
        y_first = np.fromiter((int(int(raw) in starts) for raw in lexical_raw), dtype=np.uint8)

        npz_path = FEATURES / f"{rid}.npz"
        meta_path = FEATURES / f"{rid}.json"
        if not npz_path.is_file() or not meta_path.is_file():
            raise AssertionError(f"missing fit feature pair for {rid}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("partition") != "fit" or str(meta.get("response_id")) != rid:
            raise AssertionError("feature metadata escaped fit boundary")
        actual_hash = sha256_file(npz_path) if verify_npz_hashes else meta["npz_sha256"]
        if actual_hash != meta["npz_sha256"]:
            raise AssertionError("feature hash mismatch")
        with np.load(npz_path, allow_pickle=False) as opened:
            expected = {"lb", "nll", "hidden_last", "token_ids", "answer_token_positions",
                        "response_token_offsets", "response_token_offsets_raw", "token_start", "token_end"}
            if set(opened.files) != expected:
                raise AssertionError("unexpected feature keys")
            if not np.array_equal(opened["response_token_offsets"], offsets):
                raise AssertionError("feature/token offset mismatch")
            features = build_features(opened["nll"], opened["hidden_last"], opened["lb"],
                                      lexical_raw, raw_count, hidden_projection, lb_projection)
        chunks.append(features)
        lex_start, lex_end = cursor, cursor + len(lexical_raw)
        previous = np.r_[np.uint8(0), states[:-1]]
        onset_local = np.flatnonzero(previous == 0).astype(np.int32)
        cont_local = np.flatnonzero(previous == 1).astype(np.int32)
        # The cached axis has no EOS hidden state.  For this human-only diagnostic,
        # the terminal event reuses the final lexical representation; it is marked
        # explicitly so the final silver pipeline can replace it only via a v2 freeze.
        terminal_local = int(len(states) - 1) if len(states) and states[-1] else None
        records.append({
            "response_id": rid,
            "group_id": answer["group_id"],
            "fold": stable_fold(answer["group_id"]),
            "answer_label": int(answer["label"]),
            "error_types": [span.get("label_type", "missing_type") for span in answer["original_labels"]],
            "raw_count": raw_count,
            "lex_start": lex_start,
            "lex_end": lex_end,
            "lexical_raw": lexical_raw,
            "lexical_mask": lexical,
            "risk_mask": risk,
            "states": states,
            "y_first": y_first,
            "released_onset_raw": sorted(starts),
            "span_mappings": [[int(v) for v in item["risk_token_indices"]]
                              for item in token["span_token_mapping"] if item["risk_token_indices"]],
            "onset_local": onset_local,
            "cont_local": cont_local,
            "terminal_local": terminal_local,
        })
        cursor = lex_end
        feature_hash_rows.append(f"{rid}\0{actual_hash}\0{sha256_file(meta_path)}")
    matrix = np.concatenate(chunks, axis=0)
    if matrix.shape != (139518, DIM):
        raise AssertionError(f"unexpected lexical matrix shape {matrix.shape}")
    provenance = {
        "answers_fit_sha256": sha256_file(ANSWERS),
        "tokens_fit_sha256": sha256_file(TOKENS),
        "ordered_fit_feature_pairs_sha256": digest_bytes("\n".join(feature_hash_rows).encode("utf-8")),
        "fit_feature_pairs": len(feature_hash_rows),
        "hidden_projection_sha256": digest_bytes(hidden_projection.tobytes(order="C")),
        "lookback_projection_sha256": digest_bytes(lb_projection.tobytes(order="C")),
        "npz_hashes_recomputed": verify_npz_hashes,
    }
    return matrix, records, provenance


def examples(record: dict, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = record["lex_start"]
    if family == "onset":
        local = record["onset_local"]
        return start + local, record["states"][local].astype(np.float32), np.zeros(len(local), dtype=np.uint8)
    if family == "continuation":
        local = record["cont_local"]
        indices = start + local
        labels = record["states"][local].astype(np.float32)
        terminal = np.zeros(len(local), dtype=np.uint8)
        if record["terminal_local"] is not None:
            indices = np.r_[indices, start + record["terminal_local"]]
            labels = np.r_[labels, np.float32(0.0)]
            terminal = np.r_[terminal, np.uint8(1)]
        return indices.astype(np.int64), labels.astype(np.float32), terminal
    if family == "auxiliary":
        indices = np.arange(record["lex_start"], record["lex_end"], dtype=np.int64)
        return indices, record["y_first"].astype(np.float32), np.zeros(len(indices), dtype=np.uint8)
    raise KeyError(family)


def family_weight_map(records: list[dict], train_ids: list[int], family: str) -> dict[int, np.ndarray]:
    eligible_by_group = defaultdict(list)
    cached = {}
    for answer_index in train_ids:
        item = examples(records[answer_index], family)
        cached[answer_index] = item
        if len(item[0]):
            eligible_by_group[records[answer_index]["group_id"]].append(answer_index)
    if not eligible_by_group:
        raise AssertionError(f"no {family} examples")
    base = {}
    group_mass = 1.0 / len(eligible_by_group)
    for answer_indices in eligible_by_group.values():
        answer_mass = group_mass / len(answer_indices)
        for answer_index in answer_indices:
            n = len(cached[answer_index][0])
            base[answer_index] = np.full(n, answer_mass / n, dtype=np.float64)
    class_mass = np.zeros(2, dtype=np.float64)
    for answer_index, weight in base.items():
        labels = cached[answer_index][1].astype(np.int64)
        class_mass[0] += weight[labels == 0].sum()
        class_mass[1] += weight[labels == 1].sum()
    if np.any(class_mass <= 0):
        raise AssertionError(f"{family} lacks a class in training fold")
    factors = 0.5 / class_mass
    result = {}
    for answer_index, weight in base.items():
        labels = cached[answer_index][1].astype(np.int64)
        result[answer_index] = (weight * factors[labels]).astype(np.float32)
    total = sum(float(value.sum()) for value in result.values())
    if not math.isclose(total, 1.0, abs_tol=2e-6):
        raise AssertionError((family, total))
    return result


def batch_gradient(x: np.ndarray, records: list[dict], batch: np.ndarray, family: str,
                   weight_map: dict[int, np.ndarray], params: np.ndarray) -> tuple[np.ndarray, float, int]:
    all_indices, all_labels, all_weights = [], [], []
    for answer_index in map(int, batch):
        if answer_index not in weight_map:
            continue
        indices, labels, _ = examples(records[answer_index], family)
        all_indices.append(indices)
        all_labels.append(labels)
        all_weights.append(weight_map[answer_index])
    if not all_indices:
        return np.zeros_like(params), 0.0, 0
    indices = np.concatenate(all_indices)
    labels = np.concatenate(all_labels)
    weights = np.concatenate(all_weights).astype(np.float32)
    weights /= weights.sum()
    xb = x[indices]
    scores = sigmoid(xb @ params[:-1] + params[-1])
    error = (scores - labels) * weights
    gradient = np.r_[xb.T @ error, error.sum()].astype(np.float32)
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    loss = -float(np.sum(weights * (labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))))
    return gradient, loss, len(indices)


def full_loss(x: np.ndarray, records: list[dict], train_ids: list[int], family: str,
              weight_map: dict[int, np.ndarray], params: np.ndarray) -> float:
    total = 0.0
    for answer_index in train_ids:
        if answer_index not in weight_map:
            continue
        indices, labels, _ = examples(records[answer_index], family)
        score = sigmoid(x[indices] @ params[:-1] + params[-1])
        score = np.clip(score, 1e-7, 1.0 - 1e-7)
        weights = weight_map[answer_index]
        total += -float(np.sum(weights * (labels * np.log(score) + (1.0 - labels) * np.log(1.0 - score))))
    return total


def train_fold(matrix: np.ndarray, records: list[dict], fold: int) -> tuple[dict, dict]:
    train_ids = [i for i, row in enumerate(records) if row["fold"] != fold]
    held_ids = [i for i, row in enumerate(records) if row["fold"] == fold]
    train_groups = {records[i]["group_id"] for i in train_ids}
    held_groups = {records[i]["group_id"] for i in held_ids}
    if train_groups & held_groups:
        raise AssertionError("group leakage")
    train_lex = np.concatenate([np.arange(records[i]["lex_start"], records[i]["lex_end"]) for i in train_ids])
    mean = matrix[train_lex].mean(axis=0, dtype=np.float64)
    std = matrix[train_lex].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    x = ((matrix - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
    maps = {family: family_weight_map(records, train_ids, family)
            for family in ("onset", "continuation", "auxiliary")}
    onset = np.zeros(DIM + 1, dtype=np.float32)
    continuation = np.zeros(DIM + 1, dtype=np.float32)
    moment_on = np.zeros_like(onset); variance_on = np.zeros_like(onset)
    moment_cont = np.zeros_like(continuation); variance_cont = np.zeros_like(continuation)
    rng = np.random.Generator(np.random.PCG64(TRAIN_SEED + fold))
    step = 0
    history = []
    initial = {
        "onset_transition": full_loss(x, records, train_ids, "onset", maps["onset"], onset),
        "continuation_transition": full_loss(x, records, train_ids, "continuation", maps["continuation"], continuation),
        "released_onset_auxiliary": full_loss(x, records, train_ids, "auxiliary", maps["auxiliary"], onset),
    }
    for epoch in range(EPOCHS):
        order = rng.permutation(np.asarray(train_ids, dtype=np.int32))
        batch_losses = []
        for begin in range(0, len(order), BATCH_ANSWERS):
            batch = order[begin:begin + BATCH_ANSWERS]
            grad_on_transition, loss_on, n_on = batch_gradient(
                x, records, batch, "onset", maps["onset"], onset)
            grad_aux, loss_aux, n_aux = batch_gradient(
                x, records, batch, "auxiliary", maps["auxiliary"], onset)
            grad_cont, loss_cont, n_cont = batch_gradient(
                x, records, batch, "continuation", maps["continuation"], continuation)
            grad_on = np.float32(0.5) * grad_on_transition + np.float32(0.25) * grad_aux
            grad_cont = np.float32(0.5) * grad_cont
            joint_norm = float(math.sqrt(float(np.dot(grad_on, grad_on)) + float(np.dot(grad_cont, grad_cont))))
            if joint_norm > GRAD_CLIP:
                scale = np.float32(GRAD_CLIP / joint_norm)
                grad_on *= scale; grad_cont *= scale
            step += 1
            for parameter, gradient, moment, variance in (
                (onset, grad_on, moment_on, variance_on),
                (continuation, grad_cont, moment_cont, variance_cont),
            ):
                moment *= ADAM_BETA1; moment += (1.0 - ADAM_BETA1) * gradient
                variance *= ADAM_BETA2; variance += (1.0 - ADAM_BETA2) * gradient * gradient
                corrected_m = moment / (1.0 - ADAM_BETA1 ** step)
                corrected_v = variance / (1.0 - ADAM_BETA2 ** step)
                parameter[:-1] *= np.float32(1.0 - LEARNING_RATE * WEIGHT_DECAY)
                parameter -= np.float32(LEARNING_RATE) * corrected_m / (np.sqrt(corrected_v) + ADAM_EPS)
            batch_losses.append({"on": loss_on, "aux": loss_aux, "cont": loss_cont,
                                 "n_on": n_on, "n_aux": n_aux, "n_cont": n_cont})
        if epoch in {0, EPOCHS - 1}:
            history.append({
                "epoch": epoch + 1,
                "mean_batch_onset_transition_loss": float(np.mean([v["on"] for v in batch_losses])),
                "mean_batch_auxiliary_loss": float(np.mean([v["aux"] for v in batch_losses])),
                "mean_batch_continuation_loss": float(np.mean([v["cont"] for v in batch_losses])),
            })
    final = {
        "onset_transition": full_loss(x, records, train_ids, "onset", maps["onset"], onset),
        "continuation_transition": full_loss(x, records, train_ids, "continuation", maps["continuation"], continuation),
        "released_onset_auxiliary": full_loss(x, records, train_ids, "auxiliary", maps["auxiliary"], onset),
    }
    prediction = {}
    for answer_index in held_ids:
        row = records[answer_index]
        xb = x[row["lex_start"]:row["lex_end"]]
        p_on = sigmoid(xb @ onset[:-1] + onset[-1])
        p_cont = sigmoid(xb @ continuation[:-1] + continuation[-1])
        risk = np.empty(len(xb), dtype=np.float32)
        previous = np.float32(0.0)
        for i in range(len(xb)):
            current = (np.float32(1.0) - previous) * p_on[i] + previous * p_cont[i]
            risk[i] = current; previous = current
        prediction[answer_index] = {"p_on": p_on, "p_cont": p_cont, "risk": risk}
    model = {
        "fold": fold,
        "train_answers": len(train_ids),
        "held_answers": len(held_ids),
        "train_groups": len(train_groups),
        "held_groups": len(held_groups),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "onset_parameters": onset.tolist(),
        "continuation_parameters": continuation.tolist(),
        "initial_weighted_losses": initial,
        "final_weighted_losses": final,
        "loss_history": history,
        "optimizer_steps": step,
    }
    return prediction, model


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8); scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not len(labels):
        raise AssertionError("invalid threshold arrays")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]; sorted_labels = labels[order]
    positives = int(labels.sum()); tp = fp = 0
    best_key = (0.0, 0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    best = best_key[-1]
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        tp += int(sorted_labels[index:end].sum())
        fp += (end - index) - int(sorted_labels[index:end].sum())
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = (f1, precision, recall, float(sorted_scores[index]))
        if key > best_key:
            best_key = key; best = float(sorted_scores[index])
        index = end
    return best


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.uint8); scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum()); negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {"count": len(labels), "positive": positives, "prevalence": positives / len(labels),
                "average_precision": None, "auroc": None}
    order = np.argsort(scores, kind="stable")
    s = scores[order]; y = labels[order]
    neg_before = 0; concordance = 0.0
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        pos_group = int(y[index:end].sum()); neg_group = (end - index) - pos_group
        concordance += pos_group * (neg_before + 0.5 * neg_group)
        neg_before += neg_group; index = end
    auroc = concordance / (positives * negatives)
    order = np.argsort(-scores, kind="stable"); s = scores[order]; y = labels[order]
    tp = fp = 0; previous_recall = 0.0; ap = 0.0; index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and s[end] == s[index]:
            end += 1
        tp += int(y[index:end].sum()); fp += (end - index) - int(y[index:end].sum())
        recall = tp / positives; precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall; index = end
    return {"count": len(labels), "positive": positives, "prevalence": positives / len(labels),
            "average_precision": ap, "auroc": auroc}


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.uint8); selected = np.asarray(scores) >= threshold
    tp = int(np.sum(selected & (labels == 1))); fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1))); tn = int(np.sum(~selected & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            **ranking_metrics(labels, scores)}


def evaluate(records: list[dict], predictions: dict[int, dict]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    token_rows = []
    window_rows = []
    answer_rows = []
    onset_transition_y, onset_transition_s = [], []
    auxiliary_y, auxiliary_s = [], []
    continuation_y, continuation_s, continuation_terminal = [], [], []
    span_records = []
    for answer_index, row in enumerate(records):
        pred = predictions[answer_index]
        p_on, p_cont, risk = pred["p_on"], pred["p_cont"], pred["risk"]
        raw_score = np.full(row["raw_count"], np.nan, dtype=np.float32)
        raw_on = np.full(row["raw_count"], np.nan, dtype=np.float32)
        raw_cont = np.full(row["raw_count"], np.nan, dtype=np.float32)
        raw_score[row["lexical_raw"]] = risk
        raw_on[row["lexical_raw"]] = p_on
        raw_cont[row["lexical_raw"]] = p_cont
        for local, raw in enumerate(row["lexical_raw"]):
            token_rows.append({
                "response_id": row["response_id"], "group_id": row["group_id"], "fold": row["fold"],
                "raw_bpe_index": int(raw), "lexical_index": local,
                "gold_risk": int(row["states"][local]), "released_onset": int(row["y_first"][local]),
                "binary_onset": int(row["states"][local] == 1 and (local == 0 or row["states"][local - 1] == 0)),
                "p_on": float(p_on[local]), "p_cont": float(p_cont[local]), "risk_score": float(risk[local]),
            })
        onset_indices, onset_labels, _ = examples(row, "onset")
        onset_local = onset_indices - row["lex_start"]
        onset_transition_y.extend(onset_labels.tolist()); onset_transition_s.extend(p_on[onset_local].tolist())
        auxiliary_y.extend(row["y_first"].tolist()); auxiliary_s.extend(p_on.tolist())
        cont_indices, cont_labels, terminal = examples(row, "continuation")
        cont_local = cont_indices - row["lex_start"]
        continuation_y.extend(cont_labels.tolist()); continuation_s.extend(p_cont[cont_local].tolist())
        continuation_terminal.extend(terminal.tolist())
        answer_windows = []
        onset_set = set(row["released_onset_raw"])
        for left, right in windows_for_count(row["raw_count"]):
            lexical_raw = [i for i in range(left, right) if row["lexical_mask"][i]]
            if not lexical_raw:
                continue
            score = float(np.nanmax(raw_score[lexical_raw]))
            gold_risk = int(any(row["risk_mask"][i] for i in lexical_raw))
            is_onset = bool(gold_risk and any(i in onset_set for i in lexical_raw))
            category = "released_onset" if is_onset else "internal_continuation" if gold_risk else "clean"
            item = {"response_id": row["response_id"], "group_id": row["group_id"], "fold": row["fold"],
                    "raw_bpe_start": left, "raw_bpe_end": right, "gold_risk": gold_risk,
                    "gold_category": category, "score": score}
            window_rows.append(item); answer_windows.append(item)
        if not answer_windows:
            raise AssertionError("answer has no eligible window")
        answer_rows.append({"response_id": row["response_id"], "group_id": row["group_id"],
                            "fold": row["fold"], "gold_risk": row["answer_label"],
                            "score": max(item["score"] for item in answer_windows),
                            "eligible_windows": len(answer_windows)})
        for span_index, mapping in enumerate(row["span_mappings"]):
            lexical_mapping = [raw for raw in mapping if row["lexical_mask"][raw]]
            span_records.append({"answer_index": answer_index, "span_index": span_index,
                                 "raw_indices": lexical_mapping,
                                 "error_type": row["error_types"][span_index]})

    window_y = np.asarray([item["gold_risk"] for item in window_rows], dtype=np.uint8)
    window_s = np.asarray([item["score"] for item in window_rows], dtype=np.float64)
    answer_y = np.asarray([item["gold_risk"] for item in answer_rows], dtype=np.uint8)
    answer_s = np.asarray([item["score"] for item in answer_rows], dtype=np.float64)
    window_threshold = choose_threshold(window_y, window_s)
    answer_threshold = choose_threshold(answer_y, answer_s)
    window_main = binary_metrics(window_y, window_s, window_threshold)
    answer_main = binary_metrics(answer_y, answer_s, answer_threshold)

    category = {}
    selected_windows = window_s >= window_threshold
    for name in ("released_onset", "internal_continuation", "clean"):
        mask = np.asarray([item["gold_category"] == name for item in window_rows])
        category[name] = {"count": int(mask.sum()),
                          "selected": int(np.sum(selected_windows & mask)),
                          "selection_rate": float(np.mean(selected_windows[mask])) if mask.any() else None}

    onset_transition_y = np.asarray(onset_transition_y, dtype=np.uint8)
    onset_transition_s = np.asarray(onset_transition_s, dtype=np.float64)
    auxiliary_y = np.asarray(auxiliary_y, dtype=np.uint8)
    auxiliary_s = np.asarray(auxiliary_s, dtype=np.float64)
    continuation_y = np.asarray(continuation_y, dtype=np.uint8)
    continuation_s = np.asarray(continuation_s, dtype=np.float64)
    terminal_mask = np.asarray(continuation_terminal, dtype=bool)
    head_metrics = {}
    for name, y, score in (
        ("binary_onset_transition", onset_transition_y, onset_transition_s),
        ("released_onset_auxiliary", auxiliary_y, auxiliary_s),
        ("continuation", continuation_y, continuation_s),
        ("stop", 1 - continuation_y, 1.0 - continuation_s),
    ):
        threshold = choose_threshold(y, score)
        head_metrics[name] = binary_metrics(y, score, threshold)
    head_metrics["continuation"]["terminal_stop_examples"] = int(terminal_mask.sum())

    spans = []
    type_acc = defaultdict(Counter)
    for item in span_records:
        row = records[item["answer_index"]]; pred = predictions[item["answer_index"]]
        raw_to_local = {int(raw): local for local, raw in enumerate(row["lexical_raw"])}
        scores = [float(pred["risk"][raw_to_local[raw]]) for raw in item["raw_indices"]]
        hit = any(value >= window_threshold for value in scores)
        full = bool(scores) and all(value >= window_threshold for value in scores)
        spans.append({"response_id": row["response_id"], "fold": row["fold"],
                      "span_index": item["span_index"], "error_type": item["error_type"],
                      "lexical_bpe_count": len(scores), "any_hit": hit, "full_coverage": full,
                      "max_score": max(scores), "min_score": min(scores)})
        type_acc[item["error_type"]]["spans"] += 1
        type_acc[item["error_type"]]["any_hit"] += int(hit)
        type_acc[item["error_type"]]["full_coverage"] += int(full)

    fold_metrics = []
    for fold in range(FOLDS):
        wm = np.asarray([item["fold"] == fold for item in window_rows])
        am = np.asarray([item["fold"] == fold for item in answer_rows])
        fold_metrics.append({"fold": fold,
            "window": binary_metrics(window_y[wm], window_s[wm], window_threshold),
            "answer": binary_metrics(answer_y[am], answer_s[am], answer_threshold)})

    ap_uplift = window_main["average_precision"] / window_main["prevalence"]
    folds_above = sum(item["window"]["auroc"] is not None and item["window"]["auroc"] > 0.55
                      for item in fold_metrics)
    proceed = (window_main["auroc"] >= 0.60 and ap_uplift >= 1.5 and
               answer_main["auroc"] >= 0.60 and
               category["released_onset"]["selection_rate"] >= 0.50 and
               category["internal_continuation"]["selection_rate"] >= 0.50 and
               folds_above >= 4)
    clearly_ineffective = (window_main["auroc"] <= 0.55 or ap_uplift <= 1.25 or
                            answer_main["auroc"] <= 0.55)
    gate = {
        "status": "PROCEED_TO_971_LABEL_BLIND_REPLAY" if proceed else
                  "STOP_971_REPLAY" if clearly_ineffective else "HOLD_FOR_METHOD_REVIEW",
        "criteria_frozen_for_this_diagnostic": {
            "window_auroc_at_least": 0.60,
            "window_ap_over_prevalence_at_least": 1.5,
            "answer_auroc_at_least": 0.60,
            "released_onset_window_recall_at_least": 0.50,
            "internal_continuation_window_recall_at_least": 0.50,
            "folds_with_window_auroc_above_0_55_at_least": 4,
        },
        "observed": {"window_auroc": window_main["auroc"],
                     "window_ap_over_prevalence": ap_uplift,
                     "answer_auroc": answer_main["auroc"],
                     "released_onset_window_recall": category["released_onset"]["selection_rate"],
                     "internal_continuation_window_recall": category["internal_continuation"]["selection_rate"],
                     "folds_with_window_auroc_above_0_55": folds_above},
        "scope": "Investment diagnostic only; never substitute this human-only model for frozen silver-pretrain-to-human-finetune v1.",
    }
    metrics = {
        "schema_version": "osr-linear-68-human-only-group-oof-v1",
        "status": "COMPLETE",
        "scope": "RAGTruth human fit634 only; strict group OOF; CPU; diagnostic, not final v1 candidate",
        "window": window_main,
        "answer_max": answer_main,
        "window_categories_at_window_threshold": category,
        "heads": head_metrics,
        "spans_at_window_threshold": {
            "count": len(spans),
            "any_hit": sum(item["any_hit"] for item in spans),
            "any_hit_rate": float(np.mean([item["any_hit"] for item in spans])),
            "full_coverage": sum(item["full_coverage"] for item in spans),
            "full_coverage_rate": float(np.mean([item["full_coverage"] for item in spans])),
            "by_error_type": {name: {**dict(counts),
                "any_hit_rate": counts["any_hit"] / counts["spans"],
                "full_coverage_rate": counts["full_coverage"] / counts["spans"]}
                for name, counts in sorted(type_acc.items())},
        },
        "fold_metrics_at_pooled_oof_thresholds": fold_metrics,
        "investment_gate": gate,
        "counts": {"answers": len(answer_rows), "groups": len({r["group_id"] for r in records}),
                   "lexical_tokens": len(token_rows), "eligible_windows": len(window_rows)},
        "terminal_stop_feature_policy": "Reuse the final lexical representation because the frozen cache has no EOS hidden state; terminal examples are explicitly flagged.",
    }
    return metrics, token_rows, window_rows, answer_rows, spans


def toy_selfcheck() -> dict:
    scores = np.asarray([0.1, 0.8, 0.8, 0.2], dtype=np.float64)
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    threshold = choose_threshold(labels, scores)
    metric = binary_metrics(labels, scores, threshold)
    assert 0 <= metric["auroc"] <= 1 and 0 <= metric["average_precision"] <= 1
    assert rademacher(4096, HIDDEN_SEED).shape == (4096, 32)
    assert np.array_equal(rademacher(32, 9), rademacher(32, 9))
    assert stable_fold("x") == stable_fold("x")
    return {"status": "PASS", "device": "CPU", "gpu_framework_imported": False,
            "checks": ["deterministic PCG64 projections", "threshold tie handling",
                       "ranking metrics bounded", "stable group fold"]}


def run(verify_npz_hashes: bool) -> None:
    started = time.perf_counter()
    matrix, records, provenance = load_dataset(verify_npz_hashes)
    provenance["runner_sha256"] = sha256_file(Path(__file__))
    provenance["design_protocol_sha256"] = sha256_file(HERE / "PROTOCOL.json")
    predictions = {}
    models = []
    for fold in range(FOLDS):
        fold_predictions, model = train_fold(matrix, records, fold)
        overlap = set(predictions) & set(fold_predictions)
        if overlap:
            raise AssertionError("duplicate OOF prediction")
        predictions.update(fold_predictions); models.append(model)
        print("OSR_HUMAN_ONLY_FOLD", fold, "COMPLETE", flush=True)
    if set(predictions) != set(range(len(records))):
        raise AssertionError("OOF coverage incomplete")
    metrics, token_rows, window_rows, answer_rows, span_rows = evaluate(records, predictions)
    protocol = {
        "model": "OSR-Linear-68 human-only diagnostic",
        "folds": FOLDS,
        "fold_formula": "uint64_be(SHA256('ragtruth_fit_human' + NUL + group_id)[0:8]) mod 5",
        "input_dimension": DIM,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "answer_batch_size": BATCH_ANSWERS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "adam": {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": ADAM_EPS},
        "initialization": "all zeros; no silver pretraining",
        "loss": "0.5 onset-transition BCE + 0.5 continuation-transition BCE + 0.25 released-onset auxiliary BCE",
        "cell_weighting": "Within each family: equal group mass, equal eligible-answer mass per group, equal cell mass per answer, then equal positive/negative class mass using outer-training rows only; each answer batch renormalizes present family weights.",
        "normalization": "Unweighted lexical-token mean/std from outer-training human groups only",
        "thresholds": "Pooled human-fit OOF, tie break F1 then precision then recall then largest threshold",
        "silver_pretraining": False,
        "calibration_or_test_labels_read": False,
        "gpu_started": False,
        "baseline_mutated": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "METRICS.json", metrics)
    atomic_json(OUT / "FOLD_MODELS.json", {"protocol": protocol, "provenance": provenance, "folds": models})
    write_jsonl_gz(OUT / "TOKENS.jsonl.gz", token_rows)
    write_jsonl_gz(OUT / "WINDOWS.jsonl.gz", window_rows)
    write_jsonl_gz(OUT / "ANSWERS.jsonl.gz", answer_rows)
    write_jsonl_gz(OUT / "SPANS.jsonl.gz", span_rows)
    output_names = ["METRICS.json", "FOLD_MODELS.json", "TOKENS.jsonl.gz",
                    "WINDOWS.jsonl.gz", "ANSWERS.jsonl.gz", "SPANS.jsonl.gz"]
    manifest = {
        "schema_version": "osr-human-only-oof-output-manifest-v1",
        "status": "COMPLETE",
        "wall_seconds": time.perf_counter() - started,
        "outputs": {name: {"sha256": sha256_file(OUT / name), "bytes": (OUT / name).stat().st_size}
                    for name in output_names},
        "provenance": provenance,
        "selfcheck": toy_selfcheck(),
        "calibration_or_test_labels_read": False,
        "gpu_started": False,
        "baseline_mutated": False,
    }
    atomic_json(OUT / "MANIFEST.json", manifest)
    print(json.dumps({"status": "COMPLETE", "gate": metrics["investment_gate"],
                      "window": metrics["window"], "answer": metrics["answer_max"],
                      "wall_seconds": manifest["wall_seconds"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selfcheck", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--skip-npz-hash-verification", action="store_true",
                        help="Development convenience only; default run recomputes every fit NPZ SHA-256")
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(toy_selfcheck(), indent=2))
    else:
        run(not args.skip_npz_hash_verification)


if __name__ == "__main__":
    main()
