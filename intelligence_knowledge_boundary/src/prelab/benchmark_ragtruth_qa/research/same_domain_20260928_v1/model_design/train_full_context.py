"""Same-domain full-context token classifier; fit/validation only.

Fresh generic ModernBERT initialization. Character mapping and CUDA precision
follow src/tail_finetune.py and src/run_full_context_encoder_v2.py, but no
old data/model is imported. GPU stages are explicit, never chained by prepare.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForTokenClassification


HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2]
SPLITS = ("fit", "validation")
sys.path.insert(0, str(HERE.parent))
import metrics as common_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def character_map(text, raw_offsets, starts, ends):
    """Mean over non-whitespace characters, sharing duplicate encoder coverage."""
    owners = [[] for _ in text]
    for j, (a, b) in enumerate(zip(starts, ends)):
        if a < 0:
            assert a == b == -1
            continue
        assert 0 <= a < b <= len(text)
        for c in range(a, b):
            if not text[c].isspace():
                owners[c].append(j)
    assert all(owners[c] for c, ch in enumerate(text) if not ch.isspace())
    rr, cc, ww = [], [], []
    for i, (a, b) in enumerate(raw_offsets):
        assert 0 <= a <= b <= len(text)
        chars = [c for c in range(a, b) if not text[c].isspace()]
        weights = Counter()
        for c in chars:
            for j in owners[c]:
                weights[j] += 1 / (len(chars) * len(owners[c]))
        if chars:
            assert abs(sum(weights.values()) - 1) < 1e-12
        for j, weight in sorted(weights.items()):
            rr.append(i); cc.append(j); ww.append(weight)
    return [rr, cc, ww]


def encode(answer, sequence, tokenizer):
    response = answer["original_response"]
    # Question is a separate condition, not an asserted evidence document.
    prefix = answer["retrieved_passages"] + tokenizer.sep_token + answer["question"] + tokenizer.sep_token
    encoded = tokenizer(prefix + response, return_offsets_mapping=True, truncation=False)
    assert len(encoded["input_ids"]) <= tokenizer.model_max_length
    starts, ends = [], []
    for a, b in encoded["offset_mapping"]:
        if b > len(prefix) and a < len(prefix) + len(response):
            starts.append(max(a - len(prefix), 0)); ends.append(min(b - len(prefix), len(response)))
        else:
            starts.append(-1); ends.append(-1)
    offsets = sequence["response_token_offsets"]
    lexical = sequence["lexical_mask"]
    labels = sequence["risk_mask"]
    assert len(offsets) == len(lexical) == len(labels) and any(lexical)
    assert all(not y or x for x, y in zip(lexical, labels))
    assert answer["answer_risk"] == sequence["answer_risk"]
    return {
        "response_id": answer["response_id"], "group_id": answer["group_id"],
        "input_ids": encoded["input_ids"], "raw_token_count": len(offsets),
        "mapping": character_map(response, offsets, starts, ends),
        "lexical_mask": lexical, "risk_mask": labels,
        "answer_risk": int(answer["answer_risk"]),
        "response_token_offsets": offsets,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "question_sha256": hashlib.sha256(answer["question"].encode("utf-8")).hexdigest(),
        "documents_sha256": hashlib.sha256(answer["retrieved_passages"].encode("utf-8")).hexdigest(),
    }


def group_weights(records):
    counts = Counter(r["group_id"] for r in records)
    n_groups = len(counts)
    return [np.asarray(r["lexical_mask"], np.float32) /
            (n_groups * counts[r["group_id"]] * sum(r["lexical_mask"])) for r in records]


def window_geometry(record, probabilities):
    n = record["raw_token_count"]
    assert len(probabilities) == n
    lexical = np.asarray(record["lexical_mask"], bool)
    targets = np.asarray(record["risk_mask"], np.int64)
    start, scores, labels = [], [], []
    for lo in range(max(1, n - 3)):
        hi = min(lo + 4, n)
        mask = lexical[lo:hi]
        if not mask.any():
            continue
        start.append(lo)
        scores.append(float(np.max(np.asarray(probabilities)[lo:hi][mask])))
        labels.append(int(targets[lo:hi][mask].max()))
    assert scores, "Answers without scorable windows must not silently disappear"
    return start, scores, labels


def read_fit_ids(path):
    """A shared frozen ID subset supports fold-excluded refits without new splits."""
    path = Path(path)
    data = rows(path) if path.suffix == ".jsonl" else read(path)
    if isinstance(data, dict):
        data = data["response_ids"]
    return {r["response_id"] if isinstance(r, dict) else r for r in data}


def prepare(args):
    assert not torch.cuda.is_initialized()
    assert not (args.out / "prepared.json").exists(), "Use a new output directory"
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = ModernBertConfig.from_pretrained(args.model, local_files_only=True)
    assert config.model_type == "modernbert"
    assert args.model.name in ("ModernBERT-base", "ModernBERT-large"), "Use generic pretrained weights"
    generic = read(args.model / "download_manifest.json")
    assert (generic.get("repo") or generic.get("repo_id")) in ("answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large")
    source_hashes = {str(Path(common_metrics.__file__).resolve()): sha(common_metrics.__file__)}
    output_hashes, summary, split_groups = {}, {}, {}
    fit_ids = read_fit_ids(args.fit_ids) if args.fit_ids else None
    if args.fit_ids:
        source_hashes[str(args.fit_ids.resolve())] = sha(args.fit_ids)
    for split in SPLITS:
        aa_path = args.data / split / "answers.jsonl"
        ss_path = args.data / split / "sequences.jsonl"
        aa, ss = rows(aa_path), rows(ss_path)
        assert [a["response_id"] for a in aa] == [s["response_id"] for s in ss]
        assert all(a["partition"] == split for a in aa)
        prepared = [encode(a, s, tokenizer) for a, s in zip(aa, ss)]
        assert len({r["response_id"] for r in prepared}) == len(prepared)
        if split == "fit" and fit_ids is not None:
            assert fit_ids and fit_ids <= {r["response_id"] for r in prepared}
            excluded = [r for r in prepared if r["response_id"] not in fit_ids]
            prepared = [r for r in prepared if r["response_id"] in fit_ids]
            assert excluded and not {r["group_id"] for r in prepared} & {r["group_id"] for r in excluded}, "fit ID subset must hold out entire source groups"
            destination = args.out / "held_fit_inputs.jsonl"
            write_rows(destination, excluded)
            output_hashes[destination.name] = sha(destination)
            summary["held_fit"] = {"answers": len(excluded), "groups": len({r["group_id"] for r in excluded}), "use": "prediction only; never training or model selection"}
        split_groups[split] = {r["group_id"] for r in prepared}
        source_hashes[str(aa_path.resolve())] = sha(aa_path)
        source_hashes[str(ss_path.resolve())] = sha(ss_path)
        destination = args.out / f"{split}_inputs.jsonl"
        write_rows(destination, prepared)
        output_hashes[destination.name] = sha(destination)
        lengths = np.asarray([len(r["input_ids"]) for r in prepared])
        summary[split] = {"answers": len(prepared), "groups": len(split_groups[split]),
            "input_tokens": int(lengths.sum()), "max_input_tokens": int(lengths.max()),
            "median_input_tokens": float(np.median(lengths)),
            "raw_answer_tokens": sum(r["raw_token_count"] for r in prepared),
            "lexical_tokens": sum(sum(r["lexical_mask"]) for r in prepared)}
    assert not split_groups["fit"] & split_groups["validation"]
    model_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "download_manifest.json"]
    model_hashes = {name: sha(args.model / name) for name in model_files}
    protocol = {"model": str(args.model.resolve()), "seed": args.seed, "epochs": args.epochs,
        "learning_rate": args.lr, "effective_batch_answers": args.batch,
        "optimizer": "AdamW foreach=False, FP32 states, weight_decay=0.01, clip_norm=1",
        "precision": "CUDA BF16 forward/checkpoint recompute, FP32 logits/mapping/BCE/parameters/optimizer; CPU FP32",
        "input": "all documents + SEP + question + SEP + complete answer; no truncation",
        "loss": "lexical risk BCE; equal source-group, then answer, then lexical-token mass; no class reweighting",
        "evaluation": "width4 raw Llama BPE stride1; max lexical risk; short answer one window; answer=max eligible windows",
        "threshold": "shared metrics.py: validation no_hint max riskF1, ties precision then higher threshold; independent answer/window thresholds; report all/no_hint",
        "selection": "shared no_hint selection_key then earlier epoch; epochs1..6 only",
        "test_labels_read": False, "old_finetuned_checkpoint_loaded": False}
    protocol["fit_id_subset"] = str(args.fit_ids.resolve()) if args.fit_ids else None
    assert 1 <= args.epochs <= 6
    save(args.out / "prepared.json", {"status": "prepared", "summary": summary,
        "protocol": protocol, "source_sha256": source_hashes, "model_sha256": model_hashes,
        "prepared_sha256": output_hashes, "runner_sha256": sha(__file__), "GPU_used": False})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def prepared(args):
    meta = read(args.out / "prepared.json")
    for name, digest in meta["prepared_sha256"].items():
        assert sha(args.out / name) == digest
    return meta


def load_model(meta):
    torch.manual_seed(meta["protocol"]["seed"])
    return ModernBertForTokenClassification.from_pretrained(
        meta["protocol"]["model"], local_files_only=True, use_safetensors=True,
        num_labels=2, torch_dtype=torch.float32, attn_implementation="sdpa", reference_compile=False)


def mapped_logits(flat, record):
    rr, cc, ww = record["mapping"]
    rr = torch.tensor(rr, dtype=torch.long, device=flat.device)
    cc = torch.tensor(cc, dtype=torch.long, device=flat.device)
    ww = torch.tensor(ww, dtype=torch.float32, device=flat.device)
    return flat.new_zeros(record["raw_token_count"]).index_add(0, rr, flat[cc] * ww)


def logits(model, record, device):
    ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()
    with context:
        z = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    z = z.float()
    return mapped_logits(z[:, 1] - z[:, 0], record)


@torch.no_grad()
def infer(model, records, device):
    model.eval()
    predictions, wy, ws, ay, ans = [], [], [], [], []
    for r in records:
        probability = torch.sigmoid(logits(model, r, device)).cpu().numpy()
        assert np.isfinite(probability).all()
        starts, values, labels = window_geometry(r, probability)
        predictions.append({"response_id": r["response_id"], "group_id": r["group_id"],
            "response_sha256": r["response_sha256"], "token_scores": probability.tolist(),
            "window_starts": starts, "window_scores": values, "answer_score": max(values)})
        wy.extend(labels); ws.extend(values); ay.append(r["answer_risk"]); ans.append(max(values))
    return predictions, {"window": (wy, ws), "answer": (ay, ans)}


def configure_gpu():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def fp32_state(model, optimizer):
    assert all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32) for p in model.parameters())
    assert all(not isinstance(v, torch.Tensor) or not v.is_floating_point() or v.dtype == torch.float32
               for state in optimizer.state.values() for v in state.values())


def cpu_test(args):
    meta = prepared(args)
    assert not torch.cuda.is_initialized()
    # Byte/Unicode splits and duplicate character coverage retain unit mass.
    test_map = character_map("A 🙂中", [(0, 1), (1, 2), (2, 4)], [0, 2, 2, 3], [1, 3, 3, 4])
    dummy = {"mapping": test_map, "raw_token_count": 3}
    result = mapped_logits(torch.tensor([2., 4., 8., 10.], requires_grad=True), dummy)
    assert torch.allclose(result, torch.tensor([2., 0., 8.]))
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
        pad_token_id=0, num_labels=2, local_attention=16, reference_compile=False)
    config._attn_implementation = "sdpa"
    torch.manual_seed(20260928)
    model = ModernBertForTokenClassification(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    record = {"input_ids": [1, 3, 4, 5, 6, 7, 8, 2], "raw_token_count": 3,
        "mapping": [[0, 0, 1, 2], [4, 5, 6, 7], [.5, .5, 1., 1.]]}
    observed = []
    hook = model.model.layers[0].register_forward_pre_hook(lambda *_: observed.append(torch.is_autocast_enabled("cpu")))
    model.train()
    z = logits(model, record, "cpu")
    loss = F.binary_cross_entropy_with_logits(z, torch.tensor([1., 0., 1.]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    loss.backward()
    assert z.dtype == loss.dtype == torch.float32
    assert model.classifier.weight.grad.abs().sum() > 0
    assert model.model.embeddings.tok_embeddings.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step(); fp32_state(model, optimizer)
    assert len(observed) >= 2 and not any(observed)
    hook.remove()
    model.eval()
    with torch.no_grad():
        original = logits(model, record, "cpu")
        changed_docs = logits(model, dict(record, input_ids=[1, 13, 14, 5, 6, 7, 8, 2]), "cpu")
        changed_question = logits(model, dict(record, input_ids=[1, 3, 4, 15, 6, 7, 8, 2]), "cpu")
    assert not torch.equal(original, changed_docs) and not torch.equal(original, changed_question)
    actual_fit = rows(args.out / "fit_inputs.jsonl")
    weights = group_weights(actual_fit)
    masses = Counter()
    for r, w in zip(actual_fit, weights):
        masses[r["group_id"]] += float(w.sum())
        assert not np.any(w[~np.asarray(r["lexical_mask"], bool)])
    assert max(abs(v - 1 / len(masses)) for v in masses.values()) < 1e-8
    # Verify all fit/validation mapped character masses; no test path is opened.
    for split in SPLITS:
        for r in rows(args.out / f"{split}_inputs.jsonl"):
            sums = np.bincount(r["mapping"][0], weights=r["mapping"][2], minlength=r["raw_token_count"])
            assert np.max(np.abs(sums[np.asarray(r["lexical_mask"], bool)] - 1)) < 1e-6
            p = np.random.default_rng(0).random(r["raw_token_count"])
            starts, scores, _ = window_geometry(r, p)
            assert max(scores) == p[np.asarray(r["lexical_mask"], bool)].max()
            assert np.array_equal(scores, common_metrics.pool_windows(p, r["lexical_mask"], pooling="max"))
    short = {"raw_token_count": 2, "lexical_mask": [False, True], "risk_mask": [0, 1]}
    assert window_geometry(short, [.99, .3]) == ([0], [.3], [1])
    assert not torch.cuda.is_initialized()
    save(args.out / "cpu_check.json", {"passed": True, "Unicode_mapping_and_duplicate_coverage": True,
        "real_ModernBERT_checkpoint_gradient": True, "encoder_and_head_gradients": True,
        "documents_and_question_change_answer_scores": True, "group_loss_mass_equal": True,
        "all_fit_validation_mapping_and_answermax_checked": True,
        "CPU_FP32_no_autocast": True, "GPU_used": False, "test_labels_read": False,
        "prepared_sha256": sha(args.out / "prepared.json")})
    print("CPU_CHECK_PASSED", flush=True)


def gpu_smoke(args):
    meta = prepared(args)
    assert read(args.out / "cpu_check.json")["passed"]
    configure_gpu()
    record = max(rows(args.out / "fit_inputs.jsonl") + rows(args.out / "validation_inputs.jsonl"), key=lambda r: len(r["input_ids"]))
    model = load_model(meta).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    with torch.no_grad():
        a, b = logits(model, record, "cuda"), logits(model, record, "cuda")
    difference = float((a - b).abs().max())
    assert difference <= 2e-6 and torch.isfinite(a).all()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=meta["protocol"]["learning_rate"], weight_decay=.01, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    z = logits(model, record, "cuda")
    F.binary_cross_entropy_with_logits(z, torch.zeros_like(z)).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm) and norm > 0
    optimizer.step(); fp32_state(model, optimizer)
    torch.cuda.synchronize()
    save(args.out / "gpu_smoke.json", {"passed": True, "input_tokens": len(record["input_ids"]),
        "response_id": record["response_id"], "repeated_max_abs_difference": difference,
        "synthetic_target_only": True, "seconds": time.perf_counter() - tick,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "prepared_sha256": sha(args.out / "prepared.json"), "test_labels_read": False})
    print("GPU_SMOKE_PASSED", flush=True)


def train(args):
    meta = prepared(args)
    gate = read(args.out / "gpu_smoke.json")
    assert gate["passed"] and gate["prepared_sha256"] == sha(args.out / "prepared.json")
    assert not (args.out / "train_started.json").exists(), "Use a new run; no silent restart"
    configure_gpu()
    assert args.data.resolve() == common_metrics.DATA.resolve()
    protocol = meta["protocol"]
    fit, validation = (rows(args.out / f"{s}_inputs.jsonl") for s in SPLITS)
    weights = group_weights(fit)
    model = load_model(meta).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=.01, foreach=False)
    save(args.out / "train_started.json", {"started_at": time.time(), "prepared_sha256": sha(args.out / "prepared.json")})
    history, best = [], None
    start = time.perf_counter()
    for epoch in range(1, protocol["epochs"] + 1):
        model.train(); torch.cuda.reset_peak_memory_stats()
        tick, online = time.perf_counter(), 0.
        order = np.random.default_rng(protocol["seed"] + epoch).permutation(len(fit))
        batch = protocol["effective_batch_answers"]
        for first in range(0, len(fit), batch):
            chosen = order[first:first + batch]
            optimizer.zero_grad(set_to_none=True)
            for i in chosen:
                r = fit[i]
                z = logits(model, r, "cuda")
                y = torch.tensor(r["risk_mask"], dtype=torch.float32, device="cuda")
                w = torch.as_tensor(weights[i], dtype=torch.float32, device="cuda")
                objective = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() * len(fit) / len(chosen)
                assert torch.isfinite(objective)
                objective.backward(); online += float(objective.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            if (first // batch) % 25 == 0:
                print(f"TRAIN epoch={epoch} answers={first + len(chosen)}/{len(fit)} seconds={time.perf_counter() - tick:.1f}", flush=True)
        predictions, _ = infer(model, validation, "cuda")
        evaluation = common_metrics.evaluate_partition("validation", {r["response_id"]: r["token_scores"] for r in predictions}, pooling="max")
        thresholds = evaluation["thresholds"]
        score = evaluation["cohorts"]["no_hint"]
        key = evaluation["selection_key"] + [-epoch]
        entry = {"epoch": epoch, "validation": score, "thresholds": thresholds, "selection_key": key,
            "shared_evaluation": evaluation,
            "seconds": time.perf_counter() - tick, "online_objective_sum": online,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "test_labels_read": False}
        history.append(entry)
        save(args.out / f"epoch_{epoch:02d}.json", entry)
        write_rows(args.out / f"epoch_{epoch:02d}_validation_predictions.jsonl", predictions)
        if best is None or key > best["selection_key"]:
            best = entry
            checkpoint = {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "thresholds": thresholds, "protocol": protocol,
                "prepared_sha256": sha(args.out / "prepared.json")}
            torch.save(checkpoint, args.out / "best.pending.pt")
            (args.out / "best.pending.pt").replace(args.out / "best.pt")
            write_rows(args.out / "best_validation_predictions.jsonl", predictions)
            save(args.out / "best.json", entry)
            del checkpoint
        save(args.out / "progress.json", {"last_epoch": epoch, "best": best})
        print("EPOCH_COMPLETE", epoch, json.dumps(score), flush=True)
    save(args.out / "complete.json", {"status": "trained_fit_selected_validation", "best": best,
        "history": history, "seconds": time.perf_counter() - start,
        "checkpoint_sha256": sha(args.out / "best.pt"), "test_labels_read": False})


def predict(args):
    meta = prepared(args)
    configure_gpu()
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    model = load_model(meta)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.cuda()
    predictions, _ = infer(model, rows(args.out / f"{args.split}_inputs.jsonl"), "cuda")
    write_rows(args.out / f"best_{args.split}_predictions.jsonl", predictions)
    print("PREDICTED", args.split, len(predictions), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cpu-test", "gpu-smoke", "train", "predict"])
    parser.add_argument("--data", type=Path, default=HERE.parent / "data")
    parser.add_argument("--out", type=Path, default=HERE / "modernbert_base_v1")
    parser.add_argument("--model", type=Path, default=BENCH.parent / "models/ModernBERT-base")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fit-ids", type=Path, help="JSON list or JSONL response_id rows; retain complete groups; excluded fit rows saved for prediction")
    parser.add_argument("--split", choices=(*SPLITS, "held_fit"), default="fit")
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"prepare": prepare, "cpu-test": cpu_test, "gpu-smoke": gpu_smoke, "train": train, "predict": predict}[args.stage](args)


if __name__ == "__main__":
    main()
