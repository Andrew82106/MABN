"""Label-invisible prediction with frozen RAGognizer/Lookback and thresholds.

The only partition files opened are inputs.jsonl and geometry.jsonl. This file
does not call an evaluator, load annotations, train, or select thresholds.
RAGognizer requires --allow-gpu; Lookback consumes already-extracted signals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))
import ragognizer_same_domain as rag

FORBIDDEN = {"risk_mask", "answer_risk", "label", "labels", "original_labels",
             "released_hallucinations", "hallucinations", "answerable", "gold_answer"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()).hexdigest()


def visible_rows(data, partition):
    if partition not in ("fit", "validation", "test"):
        raise ValueError(partition)
    base = Path(data).resolve() / partition
    paths = {name: base / f"{name}.jsonl" for name in ("inputs", "geometry")}
    inputs, geometry = rag.rows(paths["inputs"]), rag.rows(paths["geometry"])
    assert len(inputs) == len(geometry)
    assert len({x["response_id"] for x in inputs}) == len(inputs)
    for x, g in zip(inputs, geometry):
        assert not (FORBIDDEN & set(x)) and not (FORBIDDEN & set(g)), "Target leakage in model-visible fields"
        assert x["response_id"] == g["response_id"]
        assert g["input_sha256"] == digest(x)
    return list(zip(inputs, geometry)), {str(p): rag.sha(p) for p in paths.values()}


def native_geometry(tokenizer, row, geometry):
    """Unlabeled counterpart of the author-template geometry used in training."""
    response = row["original_response"]
    chat = rag.prompt_messages(row["released_prompt"]) + [{"role": "assistant", "content": response}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    assert encoded["input_ids"] == tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    if len(encoded["input_ids"]) > 4096:
        raise ValueError("No truncation permitted")
    stripped = response.strip()
    if not stripped:
        raise ValueError("Empty response")
    begin = text.rindex(stripped)
    leading = len(response) - len(response.lstrip())
    end = begin + len(stripped)
    offsets, positions = [], []
    for i, (a, b) in enumerate(encoded["offset_mapping"]):
        if a < end and b > begin and b > a:
            offsets.append([max(0, a - begin + leading), min(len(response), b - begin + leading)])
            positions.append(i)
    assert positions
    offsets[0][0], offsets[-1][1] = 0, len(response)
    native_to_shared = []
    for a, b in geometry["response_token_offsets"]:
        matching = [i for i, (x, y) in enumerate(offsets) if max(a, x) < min(b, y)]
        if not matching:
            raise ValueError(f"Unmapped shared token: {row['response_id']}, {a,b}")
        native_to_shared.append(matching)
    return {"input_ids": encoded["input_ids"], "positions": positions, "offsets": offsets,
            "native_to_shared": native_to_shared}


def pool4(token_scores, lexical_mask, pooling="mean"):
    values, lexical = np.asarray(token_scores, np.float64), np.asarray(lexical_mask, bool)
    assert values.shape == lexical.shape and len(values) and np.isfinite(values).all()
    k = min(4, len(values))
    starts = [s for s in range(len(values)-k+1) if lexical[s:s+k].any()]
    if pooling == "mean":
        return [float(values[s:s+k].mean()) for s in starts]
    if pooling == "max":
        return [float(values[s:s+k][lexical[s:s+k]].max()) for s in starts]
    raise ValueError(pooling)


def span8_to_tokens(probabilities):
    probabilities = np.asarray(probabilities, np.float64)
    assert len(probabilities)
    totals, counts = np.zeros(len(probabilities)+7), np.zeros(len(probabilities)+7)
    for i, score in enumerate(probabilities):
        totals[i:i+8] += score
        counts[i:i+8] += 1
    return totals / counts


def checkpoint_files(path):
    path = Path(path).resolve()
    return {str(p.relative_to(path)): rag.sha(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix in (".safetensors", ".json", ".pt", ".bin")}


def frozen_artifacts(args):
    run_dir = args.run_dir.resolve()
    if args.method == "ragognizer":
        if not (run_dir / "COMPLETE.json").exists():
            raise RuntimeError("The fixed five-epoch training run is not complete")
        selected_path = run_dir / "SELECTED.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        checkpoint = Path(selected["checkpoint"]).resolve()
        checkpoint.relative_to(run_dir)  # A selected model must belong to this run.
        validation = json.loads((checkpoint / "VALIDATION.json").read_text(encoding="utf-8"))
        thresholds = selected["thresholds"]
        assert thresholds == validation["common_evaluation"]["thresholds"]
        pooling = selected["pooling"]
        assert pooling == validation["selected_pooling"] == validation["common_evaluation"]["pooling"]
        binding = json.loads((run_dir / "RUN_BINDING.json").read_text(encoding="utf-8"))
        assert binding["prepared_sha256"] == rag.sha(args.prepared / "PREPARED.json")
        artifacts = {"selected": rag.sha(selected_path), "training_binding": rag.sha(run_dir / "RUN_BINDING.json"),
                     "prepared": rag.sha(args.prepared / "PREPARED.json"), "checkpoint": checkpoint_files(checkpoint)}
        return checkpoint, thresholds, pooling, artifacts
    validation = json.loads((run_dir / "VALIDATION.json").read_text(encoding="utf-8"))
    thresholds = validation["thresholds"]
    pooling = validation["selected_pooling"]
    assert pooling == validation["pooling"]
    artifacts = {name: rag.sha(run_dir / name) for name in ("model.pkl", "PROTOCOL.json", "VALIDATION.json", "CACHE_BINDINGS.json")}
    return run_dir / "model.pkl", thresholds, pooling, artifacts


def prediction_row(source, tokens, geometry, thresholds, pooling, native=None):
    both = {name: pool4(tokens, geometry["lexical_mask"], pooling=name) for name in ("mean", "max")}
    windows = both[pooling]
    assert windows
    result = {"response_id": source["response_id"], "input_sha256": geometry["input_sha256"],
        "token_scores": list(map(float, tokens)), "pooling": pooling, "window_scores_by_pooling": both,
        "window_scores": windows, "answer_score": max(windows),
        "window_decisions": [score >= thresholds["window"] for score in windows],
        "answer_decision": max(windows) >= thresholds["answer"]}
    if native is not None:
        result["native"] = native
    return result


def run(args):
    if args.method == "ragognizer" and not args.allow_gpu:
        raise RuntimeError("GPU scheduling requires --allow-gpu")
    pairs, inputs_binding = visible_rows(args.data, args.partition)
    artifact, thresholds, pooling, models_binding = frozen_artifacts(args)
    assert pooling in ("mean", "max")
    assert set(thresholds) == {"window", "answer"} and all(np.isfinite(float(v)) for v in thresholds.values())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new prediction output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    binding = {"method": args.method, "partition": args.partition, "input_files": inputs_binding,
        "models": models_binding, "thresholds": thresholds, "thresholds_sha256": digest(thresholds),
        "pooling": pooling, "pooling_and_thresholds_sha256": digest({"pooling": pooling, "thresholds": thresholds}),
        "threshold_source": "frozen validation no-hint selection, no target partition optimization",
        "predictor_sha256": rag.sha(__file__), "annotations_opened": False, "test_labels_opened": False}
    rag.save(args.out / "PREDICTION_BINDING.json", binding)
    started = time.time()
    cache_bindings = {}
    if args.method == "lookback":
        from threadpoolctl import threadpool_limits
        protocol = json.loads((args.run_dir / "PROTOCOL.json").read_text(encoding="utf-8"))
        assert all(importlib.metadata.version(name) == version for name, version in protocol["software"].items())
        with artifact.open("rb") as f:
            model = pickle.load(f)
        assert model.n_features_in_ == 1024 and model.classes_.tolist() == [0, 1]
        assert model.C == protocol["C"]
        expected_source = json.loads((args.run_dir / "CACHE_BINDINGS.json").read_text(encoding="utf-8"))["fit"][0]["source_sha256"]
    else:
        if not args.allow_gpu:
            raise RuntimeError("GPU scheduling requires --allow-gpu")
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(args.prepared / "tokenizer", local_files_only=True)
        model, head = rag.load_training_model(args, torch, checkpoint=artifact)
        model.eval(); head.eval()
        # Prediction needs neither trainable parameters nor a training graph.
        model.requires_grad_(False); head.requires_grad_(False)
        model.gradient_checkpointing_disable()
    predictions_path = args.out / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for i, (source, geometry) in enumerate(pairs, 1):
            rid = source["response_id"]
            if args.method == "lookback":
                cache = args.signals / f"{rid}.npz"
                receipt = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
                assert receipt["input_sha256"] == geometry["input_sha256"]
                assert receipt["source_sha256"] == expected_source
                assert receipt["npz_sha256"] == rag.sha(cache)
                with np.load(cache, allow_pickle=False) as z:
                    assert np.array_equal(z["token_ids"], geometry["response_token_ids"])
                    assert np.array_equal(z["response_token_offsets"], geometry["response_token_offsets"])
                    x = z["lb_prefix_pre_header"]
                    if len(x) < 8:
                        raise ValueError("Original Lookback unsupported below8BPE; no fabricated score")
                    span = np.stack([x[j:j+8].mean(axis=0, dtype=np.float32) for j in range(len(x)-7)])
                with threadpool_limits(limits=4):
                    native = model.predict_proba(span)[:, 1]
                tokens = span8_to_tokens(native)
                row = prediction_row(source, tokens, geometry, thresholds, pooling,
                                     {"span8_probabilities": native.tolist()})
                cache_bindings[rid] = receipt["npz_sha256"]
            else:
                native = native_geometry(tokenizer, source, geometry)
                ids = torch.tensor([native["input_ids"]], dtype=torch.long, device="cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.get_base_model().model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, output_hidden_states=True, return_dict=True)
                    logits = head(result.hidden_states[-16][0]).float().view(-1)
                    probabilities = logits[native["positions"]].sigmoid().cpu().numpy()
                tokens = [float(probabilities[ix].mean()) for ix in native["native_to_shared"]]
                row = prediction_row(source, tokens, geometry, thresholds, pooling,
                    {"probabilities": probabilities.tolist(), "offsets": native["offsets"]})
                del result, logits, ids
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            if i % 25 == 0 or i == len(pairs):
                progress = {"completed": i, "total": len(pairs), "elapsed_seconds": time.time()-started,
                            "test_labels_opened": False}
                rag.save(args.out / "PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
    if cache_bindings:
        rag.save(args.out / "CACHE_BINDINGS.json", cache_bindings)
    rag.save(args.out / "PREDICTION_RECEIPT.json", {**binding, "responses": len(pairs),
        "prediction_sha256": rag.sha(predictions_path), "elapsed_seconds": time.time()-started,
        "fit_called": False, "threshold_selection_called": False, "gpu_used": args.method == "ragognizer"})


def preflight(args):
    """CPU only: check visible input identity and native/shared coordinate coverage."""
    from transformers import AutoTokenizer
    pairs, hashes = visible_rows(args.data, args.partition)
    tokenizer = AutoTokenizer.from_pretrained(args.prepared / "tokenizer", local_files_only=True)
    lengths, windows = [], 0
    for row, geometry in pairs:
        n = native_geometry(tokenizer, row, geometry)
        lengths.append(len(n["input_ids"]))
        windows += len(pool4(np.ones(len(n["native_to_shared"])), geometry["lexical_mask"]))
    args.out.mkdir(parents=True, exist_ok=True)
    rag.save(args.out / "PREFLIGHT.json", {"status": "PASS", "partition": args.partition,
        "responses": len(pairs), "windows": windows, "max_native_tokens": max(lengths),
        "input_files": hashes, "annotations_opened": False, "gpu_used": False})
    print(f"Label-invisible CPU preflight PASS: {len(pairs)} responses / {windows} windows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict", "preflight"))
    parser.add_argument("--method", choices=("ragognizer", "lookback"), default="ragognizer")
    parser.add_argument("--partition", choices=("fit", "validation", "test"), default="test")
    parser.add_argument("--data", type=Path, default=RUN / "data")
    parser.add_argument("--prepared", type=Path, default=HERE / "ragognizer_prepared_v2")
    parser.add_argument("--signals", type=Path, default=RUN / "signals")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.command == "preflight":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        preflight(args)
    else:
        if args.run_dir is None:
            parser.error("predict requires --run-dir")
        run(args)


if __name__ == "__main__":
    main()
