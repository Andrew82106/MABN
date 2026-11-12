"""Encode and predict without accepting any target labels.

Test preparation opens only test/inputs.jsonl, test/geometry.jsonl and the
label-blind split assignment list. It never calls the common evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import train_full_context as base

HERE = Path(__file__).resolve().parent
FORBIDDEN = {"risk_mask", "answer_risk", "released_hallucinations", "label", "labels", "target", "targets"}


def encode_unlabeled(answer, geometry, tokenizer, *, group_id=None):
    """Only textual inputs and token geometry enter this function."""
    assert not FORBIDDEN & set(answer) and not FORBIDDEN & set(geometry)
    response = answer["original_response"]
    prefix = answer["retrieved_passages"] + tokenizer.sep_token + answer["question"] + tokenizer.sep_token
    encoded = tokenizer(prefix + response, return_offsets_mapping=True, truncation=False)
    assert len(encoded["input_ids"]) <= tokenizer.model_max_length
    starts, ends = [], []
    for a, b in encoded["offset_mapping"]:
        if b > len(prefix) and a < len(prefix) + len(response):
            starts.append(max(a - len(prefix), 0)); ends.append(min(b - len(prefix), len(response)))
        else:
            starts.append(-1); ends.append(-1)
    offsets = geometry["response_token_offsets"]
    lexical = geometry["lexical_mask"]
    assert len(offsets) == len(lexical) and any(lexical)
    return {"response_id": answer["response_id"], "group_id": group_id,
        "input_ids": encoded["input_ids"], "raw_token_count": len(offsets),
        "mapping": base.character_map(response, offsets, starts, ends),
        "lexical_mask": lexical, "response_token_offsets": offsets,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest()}


def window_scores(record, probability):
    """Label-free 4-BPE lexical maximum and its original token-start positions."""
    probability = np.asarray(probability, np.float64)
    lexical = np.asarray(record["lexical_mask"], bool)
    assert len(probability) == len(lexical) == record["raw_token_count"]
    assert np.isfinite(probability).all()
    starts, values = [], []
    for lo in range(max(1, len(probability) - 3)):
        hi = min(lo + 4, len(probability))
        mask = lexical[lo:hi]
        if mask.any():
            starts.append(lo); values.append(float(probability[lo:hi][mask].max()))
    assert values
    return starts, values


@torch.no_grad()
def predict_records(model, records, device):
    model.eval()
    predictions = []
    for index, record in enumerate(records):
        assert not FORBIDDEN & set(record)
        probability = torch.sigmoid(base.logits(model, record, device)).cpu().numpy()
        starts, values = window_scores(record, probability)
        predictions.append({"response_id": record["response_id"], "group_id": record["group_id"],
            "response_sha256": record["response_sha256"], "token_scores": probability.tolist(),
            "window_starts": starts, "window_scores": values, "answer_score": max(values)})
        if (index + 1) % 100 == 0:
            print("PREDICT", index + 1, len(records), flush=True)
    return predictions


def prepare(args):
    assert not torch.cuda.is_initialized()
    meta = base.prepared(argparse.Namespace(out=args.checkpoint_dir))
    tokenizer = AutoTokenizer.from_pretrained(meta["protocol"]["model"], local_files_only=True)
    paths = [args.data / "test/inputs.jsonl", args.data / "test/geometry.jsonl", args.data / "split_assignments.jsonl"]
    aa, gg, assignments = (base.rows(p) for p in paths)
    assigned = {r["response_id"]: r for r in assignments if r["split"] == "test"}
    assert set(assigned) == {r["response_id"] for r in aa}
    geometry = {r["response_id"]: r for r in gg}
    assert set(geometry) == set(assigned)
    prepared = [encode_unlabeled(a, geometry[a["response_id"]], tokenizer,
                                 group_id=assigned[a["response_id"]]["group_id"]) for a in aa]
    args.out.mkdir(parents=True, exist_ok=True)
    base.write_rows(args.out / "inputs.jsonl", prepared)
    base.save(args.out / "prepared.json", {"answers": len(prepared),
        "max_input_tokens": max(len(r["input_ids"]) for r in prepared),
        "input_tokens": sum(len(r["input_ids"]) for r in prepared),
        "input_sha256": base.sha(args.out / "inputs.jsonl"),
        "source_sha256": {str(p.resolve()): base.sha(p) for p in paths},
        "encoding_model": meta["protocol"]["model"], "test_labels_read": False, "GPU_used": False})
    print("LABEL_FREE_TEST_PREPARED", len(prepared), flush=True)


def cpu_check(args):
    # Demonstrate equality to the already-tested training encoding on fit only.
    meta = base.prepared(argparse.Namespace(out=args.checkpoint_dir))
    tokenizer = AutoTokenizer.from_pretrained(meta["protocol"]["model"], local_files_only=True)
    answers = {r["response_id"]: r for r in base.rows(args.data / "fit/answers.jsonl")}
    geometries = {r["response_id"]: r for r in base.rows(args.data / "fit/sequences.jsonl")}
    prepared_fit = base.rows(args.checkpoint_dir / "fit_inputs.jsonl")
    selected = [prepared_fit[0], prepared_fit[-1], max(prepared_fit, key=lambda r: len(r["input_ids"]))]
    for old in selected:
        rid = old["response_id"]
        a = {k: answers[rid][k] for k in ("response_id", "question", "retrieved_passages", "original_response")}
        g = {k: geometries[rid][k] for k in ("response_token_offsets", "lexical_mask")}
        new = encode_unlabeled(a, g, tokenizer, group_id=old["group_id"])
        assert new["input_ids"] == old["input_ids"] and new["mapping"] == old["mapping"]
        p = np.random.default_rng(0).random(new["raw_token_count"])
        starts, values = window_scores(new, p)
        os, ov, _ = base.window_geometry(old, p)
        assert starts == os and values == ov
    meta_test = base.read(args.out / "prepared.json")
    assert base.sha(args.out / "inputs.jsonl") == meta_test["input_sha256"]
    for record in base.rows(args.out / "inputs.jsonl"):
        assert not FORBIDDEN & set(record)
        sums = np.bincount(record["mapping"][0], weights=record["mapping"][2], minlength=record["raw_token_count"])
        assert np.max(np.abs(sums[np.asarray(record["lexical_mask"], bool)] - 1)) < 1e-6
    assert not torch.cuda.is_initialized()
    base.save(args.out / "cpu_check.json", {"passed": True, "label_free_encoder_equals_training_encoder": True,
        "all_test_input_geometry_checked_without_targets": True, "test_labels_read": False, "GPU_used": False})
    print("LABEL_FREE_CPU_CHECK_PASSED", flush=True)


def predict(args):
    assert base.read(args.out / "cpu_check.json")["passed"]
    plan = base.read(args.out / "prepared.json")
    assert base.sha(args.out / "inputs.jsonl") == plan["input_sha256"]
    meta = base.prepared(argparse.Namespace(out=args.checkpoint_dir))
    checkpoint_path = args.checkpoint_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["protocol"]["model"] == plan["encoding_model"]
    base.configure_gpu()
    model = base.load_model(meta)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model = model.cuda()
    records = base.rows(args.out / "inputs.jsonl")
    predictions = predict_records(model, records, "cuda")
    base.write_rows(args.out / "predictions.jsonl", predictions)
    base.save(args.out / "complete.json", {"answers": len(predictions),
        "checkpoint_sha256": base.sha(checkpoint_path), "predictions_sha256": base.sha(args.out / "predictions.jsonl"),
        "test_labels_read": False, "evaluation_run": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cpu-check", "predict"])
    parser.add_argument("--data", type=Path, default=HERE.parent / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "modernbert_base_v1")
    parser.add_argument("--out", type=Path, default=HERE / "modernbert_base_test_v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"prepare": prepare, "cpu-check": cpu_check, "predict": predict}[args.stage](args)


if __name__ == "__main__":
    main()
