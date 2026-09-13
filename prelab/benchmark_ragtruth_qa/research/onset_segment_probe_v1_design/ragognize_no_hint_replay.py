#!/usr/bin/env python3
"""Fail-closed, label-blind RAGognize no-hint teacher-forced replay runner.

`--plan` and `--selfcheck` use only CPU and never import torch/transformers.
`--model-smoke` and `--run` are future actions: they first require a passing
human-fit investment gate, then lazily load a local NF4 Llama-2 on CPU.  The v1
gate currently blocks both actions, so this audit cannot load the model.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
ANSWERS = BENCH / "auxiliary_ragognize_train_v1" / "conversion_v1" / "no_unanswerability_hint" / "answers.jsonl"
SEQUENCES = BENCH / "auxiliary_ragognize_train_v1" / "conversion_v1" / "no_unanswerability_hint" / "sequences.jsonl"
FIT_ANSWERS = BENCH / "data" / "answers_fit.jsonl"
FIT_RECORDS = BENCH / "data" / "fit.jsonl"
FIT_FEATURES = BENCH / "data" / "features"
MODEL = BENCH.parent / "models" / "Llama-2-7b-chat-hf"
MODEL_MANIFEST = BENCH / "model_download_manifest.json"
FEATURE_CODE = BENCH / "src" / "feature_qa.py"
HUMAN_GATE = HERE / "human_only_oof_v1" / "METRICS.json"
OUT = HERE / "ragognize_no_hint_replay_v1"
FEATURE_OUT = OUT / "features"
MODEL_INPUTS = OUT / "MODEL_INPUTS.jsonl.gz"
FIT_COMPAT_INPUTS = OUT / "FIT_COMPAT_INPUTS.json"

EXPECTED_ROWS = 971
EXPECTED_RESPONSE_TOKENS = 102778
EXPECTED_LEXICAL_TOKENS = 88133
FEATURE_KEYS = ("lb", "nll", "hidden_last", "token_ids", "answer_token_positions",
                "response_token_offsets", "response_token_offsets_raw", "token_start", "token_end")
VISIBLE_ANSWER_FIELDS = ("response_id", "source_id", "group_id", "official_split", "partition",
                         "generator", "released_prompt", "retrieved_passages", "original_response",
                         "response_sha256")
VISIBLE_SEQUENCE_FIELDS = ("response_id", "source_id", "group_id", "full_input_ids", "attention_mask",
                           "answer_token_positions", "response_token_ids", "response_token_offsets",
                           "response_token_offsets_raw", "rendered_input_sha256", "response_sha256")
FORBIDDEN_MODEL_FIELDS = {"released_hallucinations", "released_labels_sha256", "answer_risk",
                          "risk_mask", "hallucination", "label", "labels"}
CPU_SMOKE_TOLERANCE = {"lb_max_abs": 0.005, "nll_max_abs": 0.05, "hidden_max_abs": 0.125}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def atomic_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def project_fields(row: dict, fields: tuple[str, ...]) -> dict:
    missing = set(fields) - set(row)
    if missing:
        raise AssertionError(f"missing fields: {sorted(missing)}")
    visible = {name: row[name] for name in fields}
    if set(visible) & FORBIDDEN_MODEL_FIELDS:
        raise AssertionError("annotation entered model-visible record")
    return visible


def load_visible_rows() -> list[dict]:
    answers = {}
    for raw in iter_jsonl(ANSWERS):
        if raw.get("official_split") != "train" or raw.get("partition") != "auxiliary_candidate_fit":
            raise AssertionError("non-train/no-hint answer")
        visible = project_fields(raw, VISIBLE_ANSWER_FIELDS)
        rid = visible["response_id"]
        if rid in answers:
            raise AssertionError("duplicate answer ID")
        answers[rid] = visible
    sequences = {}
    for raw in iter_jsonl(SEQUENCES):
        visible = project_fields(raw, VISIBLE_SEQUENCE_FIELDS)
        rid = visible["response_id"]
        if rid in sequences:
            raise AssertionError("duplicate sequence ID")
        sequences[rid] = visible
    if set(answers) != set(sequences) or len(answers) != EXPECTED_ROWS:
        raise AssertionError("no-hint source ID/count mismatch")
    result = []
    for rid in sorted(answers):
        answer, sequence = answers[rid], sequences[rid]
        if answer["source_id"] != sequence["source_id"] or answer["group_id"] != sequence["group_id"]:
            raise AssertionError("answer/sequence identity mismatch")
        if answer["response_sha256"] != sequence["response_sha256"] != digest(answer["original_response"]):
            raise AssertionError("response hash mismatch")
        if answer["response_sha256"] != digest(answer["original_response"]):
            raise AssertionError("response hash mismatch")
        if len(sequence["full_input_ids"]) != len(sequence["attention_mask"]):
            raise AssertionError("input/mask length mismatch")
        if any(int(v) != 1 for v in sequence["attention_mask"]):
            raise AssertionError("non-unit unpadded mask")
        n = len(sequence["answer_token_positions"])
        if not (n == len(sequence["response_token_ids"]) == len(sequence["response_token_offsets"]) ==
                len(sequence["response_token_offsets_raw"])):
            raise AssertionError("response geometry mismatch")
        if [sequence["full_input_ids"][int(i)] for i in sequence["answer_token_positions"]] != sequence["response_token_ids"]:
            raise AssertionError("answer IDs not aligned to full input")
        passages = answer["retrieved_passages"]
        prompt = answer["released_prompt"]
        if not passages or prompt.count(passages) != 1:
            raise AssertionError("retrieved passage block is not unique in released prompt")
        rendered = prompt + " " + answer["original_response"]
        if digest(rendered) != sequence["rendered_input_sha256"]:
            raise AssertionError("rendered input hash mismatch")
        begin = prompt.index(passages)
        end = begin + len(passages)
        result.append({**answer, **sequence,
                       "reference_character_range": [begin, end],
                       "labels_used": False})
    if sum(len(row["answer_token_positions"]) for row in result) != EXPECTED_RESPONSE_TOKENS:
        raise AssertionError("response token total mismatch")
    return result


def fit_feature_reference() -> dict:
    fit_ids = [str(row["response_id"]) for row in iter_jsonl(FIT_ANSWERS)]
    if len(fit_ids) != 634:
        raise AssertionError("unexpected fit count")
    raw_bytes = npz_bytes = feature_seconds = serialization_seconds = 0.0
    work = response_tokens = 0
    for rid in fit_ids:
        meta = json.loads((FIT_FEATURES / f"{rid}.json").read_text(encoding="utf-8"))
        if meta.get("partition") != "fit" or str(meta.get("response_id")) != rid:
            raise AssertionError("fit metadata boundary failure")
        raw_bytes += int(meta["raw_array_bytes"])
        npz_bytes += int(meta["npz_bytes"])
        feature_seconds += float(meta["feature_metadata"]["seconds"])
        serialization_seconds += float(meta["serialization_hash_seconds"])
        length = int(meta["feature_metadata"]["total_input_tokens"])
        response = int(meta["response_tokens"])
        response_tokens += response
        work += length * length + length * response
    return {
        "rows": len(fit_ids), "response_tokens": response_tokens,
        "raw_bytes": int(raw_bytes), "npz_bytes": int(npz_bytes),
        "npz_over_raw_ratio": npz_bytes / raw_bytes,
        "feature_seconds": feature_seconds,
        "serialization_and_hash_seconds": serialization_seconds,
        "attention_work_units": work,
        "hardware": "Historical NVIDIA RTX 3070 NF4 replay; reference only, not a CPU timing claim",
    }


def resource_estimate(rows: list[dict]) -> dict:
    lengths = [len(row["full_input_ids"]) for row in rows]
    response = [len(row["answer_token_positions"]) for row in rows]
    reference = fit_feature_reference()
    total_response = sum(response)
    coordinate_bytes_per_token = 8 + 8 + 8 + 8 + 4 + 4
    raw = {
        "lb_float32": total_response * 1024 * 4,
        "nll_float32": total_response * 4,
        "hidden_last_float32": total_response * 4096 * 4,
        "coordinate_arrays": total_response * coordinate_bytes_per_token,
    }
    work = sum(length * length + length * output for length, output in zip(lengths, response))
    compressed_estimate = int(sum(raw.values()) * reference["npz_over_raw_ratio"])
    gpu_reference_seconds = reference["feature_seconds"] * work / reference["attention_work_units"]
    serialization_reference_seconds = reference["serialization_and_hash_seconds"] * total_response / reference["response_tokens"]
    assets = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    checkpoint_bytes = sum(int(row.get("actual_bytes", 0)) for row in assets["files"])
    return {
        "rows": len(rows),
        "full_input_tokens": sum(lengths),
        "response_tokens": total_response,
        "full_input_length": {"median": percentile(lengths, 0.5), "p90": percentile(lengths, 0.9), "max": max(lengths)},
        "response_length": {"median": percentile(response, 0.5), "p90": percentile(response, 0.9), "max": max(response)},
        "analytic_attention_work_units_sum_L2_plus_LR": work,
        "lookback_query_key_scalar_scores": 32 * 32 * sum(l * r for l, r in zip(lengths, response)),
        "raw_output_bytes": raw,
        "raw_output_total_bytes": sum(raw.values()),
        "projected_68d_training_cache_bytes_float32": total_response * 68 * 4,
        "compressed_npz_point_estimate_bytes": compressed_estimate,
        "compression_basis": reference,
        "historical_same_formula_gpu_reference": {
            "feature_seconds_work_scaled": gpu_reference_seconds,
            "serialization_and_hash_seconds_token_scaled": serialization_reference_seconds,
            "not_a_cpu_estimate": True,
        },
        "cpu_time_status": "UNMEASURED_UNTIL_TWO_ROW_CPU_NF4_SMOKE",
        "cpu_time_extrapolation_after_smoke": "Scale measured short/long smoke seconds by both response count and sum(L^2+L*R); use 0.75*min to 1.5*max as a planning interval, not a confidence interval.",
        "checkpoint_files_bytes": checkpoint_bytes,
        "minimum_free_ram_gate_bytes": 24 * 2**30,
        "minimum_free_output_disk_gate_bytes": max(6 * 2**30, 3 * compressed_estimate),
        "ram_gate_reason": "Allows the 12.55-GiB local BF16 checkpoint shards, CPU NF4 conversion/runtime overhead, one unpadded example, and serialization buffers; actual peak must be measured by smoke.",
    }


def execution_gate() -> dict:
    if not HUMAN_GATE.is_file():
        return {"status": "BLOCKED", "reason": "human-only OOF gate file missing"}
    value = json.loads(HUMAN_GATE.read_text(encoding="utf-8"))
    status = value["investment_gate"]["status"]
    return {
        "status": "PASS" if status == "PROCEED_TO_971_LABEL_BLIND_REPLAY" else "BLOCKED",
        "human_only_gate_status": status,
        "human_only_metrics_sha256": sha256_file(HUMAN_GATE),
        "reason": "Human-only v1 must pass its predeclared investment gate before any model load or 971 replay.",
    }


def static_selfcheck(rows: list[dict]) -> dict:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for required in ("load_nf4_cpu", "materialize_model_plan", "extract_one", "model_smoke", "run_replay"):
        if required not in functions:
            raise AssertionError(f"missing runner function {required}")
    for guarded in ("model_smoke", "run_replay"):
        first = functions[guarded].body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and
                isinstance(first.value.func, ast.Name) and first.value.func.id == "require_execution_gate"):
            raise AssertionError(f"{guarded} must gate before any runtime/model import")
    model_plan_keys = set()
    for node in ast.walk(functions["materialize_model_plan"]):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            model_plan_keys.add(node.value)
    if model_plan_keys & FORBIDDEN_MODEL_FIELDS:
        raise AssertionError("forbidden label key appears in model plan builder")
    gate = execution_gate()
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "transformers", "bitsandbytes", "accelerate")}
    sanitized_ok = False
    if MODEL_INPUTS.is_file():
        with gzip.open(MODEL_INPUTS, "rt", encoding="utf-8") as handle:
            sanitized = [json.loads(line) for line in handle if line.strip()]
        expected = set(VISIBLE_ANSWER_FIELDS) | set(VISIBLE_SEQUENCE_FIELDS) | {
            "reference_character_range", "labels_used"}
        sanitized_ok = (len(sanitized) == EXPECTED_ROWS and
                        all(set(row) == expected and not (set(row) & FORBIDDEN_MODEL_FIELDS)
                            for row in sanitized))
        if not sanitized_ok:
            raise AssertionError("sanitized model input audit failed")
    return {
        "schema_version": "ragognize-no-hint-replay-static-selfcheck-v1",
        "status": "PASS",
        "rows_checked": len(rows),
        "response_tokens_checked": sum(len(row["answer_token_positions"]) for row in rows),
        "model_visible_fields": sorted(set(VISIBLE_ANSWER_FIELDS) | set(VISIBLE_SEQUENCE_FIELDS) |
                                       {"reference_character_range", "labels_used"}),
        "forbidden_annotation_fields_absent_from_model_plan_builder": True,
        "model_smoke_and_full_run_gate_before_runtime_import": True,
        "sanitized_model_inputs_checked": sanitized_ok,
        "full_input_ids_and_answer_coordinates_checked": True,
        "unique_reference_character_range_checked": True,
        "model_imported": False,
        "pretrained_model_loaded": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "current_python_dependencies": dependencies,
        "current_python_model_runtime_ready": all(dependencies.values()),
        "execution_gate": gate,
        "calibration_or_test_rows_read": False,
        "labels_used_for_feature_planning": False,
    }


def build_plan() -> dict:
    rows = load_visible_rows()
    atomic_jsonl_gz(MODEL_INPUTS, rows)
    atomic_json(FIT_COMPAT_INPUTS, fit_compatibility_rows())
    estimate = resource_estimate(rows)
    check = static_selfcheck(rows)
    model_assets = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    value = {
        "schema_version": "ragognize-no-hint-cpu-nf4-replay-plan-v1",
        "status": "PREPARED_MODEL_NOT_LOADED",
        "source": {
            "answers": str(ANSWERS.relative_to(BENCH)), "answers_sha256": sha256_file(ANSWERS),
            "sequences": str(SEQUENCES.relative_to(BENCH)), "sequences_sha256": sha256_file(SEQUENCES),
            "cohort": "official train / no_unanswerability_hint / Llama-2-7b-chat-hf",
            "rows": len(rows),
        },
        "sanitized_model_inputs": {
            "path": MODEL_INPUTS.name, "sha256": sha256_file(MODEL_INPUTS),
            "bytes": MODEL_INPUTS.stat().st_size, "rows": len(rows),
            "annotation_fields_present": False,
            "rule": "Future model-smoke/full-run opens this sanitized file; label-bearing source JSONL is used only by CPU planning and byte-level provenance hashing."
        },
        "sanitized_fit_compatibility_inputs": {
            "path": FIT_COMPAT_INPUTS.name, "sha256": sha256_file(FIT_COMPAT_INPUTS),
            "rows": 2, "selection": "lexicographically first and last response_id from RAGTruth fit only"
        },
        "feature_contract": {
            "formula_source": str(FEATURE_CODE.relative_to(BENCH)),
            "formula_source_sha256": sha256_file(FEATURE_CODE),
            "features": {"lb": "float32[T,1024]", "nll": "float32[T]", "hidden_last": "float32[T,4096]"},
            "lb": "Post-read answer query; per layer/head mean reference attention divided by mean-reference plus mean-prefix-answer attention; current answer token included.",
            "nll": "Pre-read position i-1 chosen-token negative log probability.",
            "hidden": "Post-read final model RMSNorm output.",
            "wrapper": "Use released_prompt + one literal space + original_response exactly as frozen by conversion_v1; no regeneration or truncation.",
            "storage": "Per-answer compressed NPZ plus atomic JSON metadata and SHA-256.",
            "original_generation_trace_claimed": False,
        },
        "model": {
            "repo_id": model_assets["repo_id"], "revision": model_assets["revision"],
            "local_directory": str(MODEL), "download_manifest_sha256": sha256_file(MODEL_MANIFEST),
            "checkpoint_asset_bytes": sum(int(row.get("actual_bytes", 0)) for row in model_assets["files"]),
            "load": "Local-only Llama-2; NF4 double quantization; BF16 compute; CPU device_map; eval/frozen; SDPA; no cache.",
            "asset_hash_recheck_deferred_until_model_smoke": True,
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "execution_sequence": [
            "Run --plan/--selfcheck without torch or a model.",
            "Require human-only investment gate PASS; current v1 is blocked.",
            "In a CPU NF4-capable environment, run --model-smoke on deterministic shortest/longest RAGognize rows and two fit-cache compatibility rows.",
            "Measure peak RAM and elapsed time; require coordinate identity, finite arrays, NF4 CPU backend, and frozen tolerances.",
            "Only after a passing smoke manifest, run --run; resume only hash-verified complete rows and verify all 971 outputs at the end.",
        ],
        "model_smoke_tolerances": CPU_SMOKE_TOLERANCE,
        "resource_estimate": estimate,
        "execution_gate": check["execution_gate"],
        "no_model_loaded_in_plan": True,
        "calibration_or_test_rows_read": False,
        "labels_used_for_feature_planning": False,
        "baseline_mutated": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "PLAN.json", value)
    atomic_json(OUT / "SELF_CHECK.json", check)
    return value


def require_execution_gate() -> None:
    gate = execution_gate()
    if gate["status"] != "PASS":
        raise RuntimeError(f"REPLAY_BLOCKED_BEFORE_MODEL_LOAD: {gate}")


def frozen_model_inputs() -> list[dict]:
    plan_path = OUT / "PLAN.json"
    if not plan_path.is_file() or not MODEL_INPUTS.is_file():
        raise RuntimeError("Run --plan before any model action")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256_file(MODEL_INPUTS) != plan["sanitized_model_inputs"]["sha256"]:
        raise AssertionError("sanitized model input hash mismatch")
    with gzip.open(MODEL_INPUTS, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    expected = set(VISIBLE_ANSWER_FIELDS) | set(VISIBLE_SEQUENCE_FIELDS) | {
        "reference_character_range", "labels_used"}
    if len(rows) != EXPECTED_ROWS or any(set(row) != expected for row in rows):
        raise AssertionError("sanitized model input schema mismatch")
    if any(set(row) & FORBIDDEN_MODEL_FIELDS for row in rows):
        raise AssertionError("annotation leaked into sanitized model inputs")
    return rows


def lazy_runtime():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    sys.path.insert(0, str(BENCH / "src"))
    import feature_qa
    return np, torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, feature_qa


def verify_model_assets() -> dict:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("all_files_source_hash_matched"):
        raise AssertionError("model download manifest not complete")
    checked = {}
    for entry in manifest["files"]:
        path = MODEL / entry["filename"]
        if not path.is_file() or path.stat().st_size != int(entry["actual_bytes"]):
            raise AssertionError(f"model asset size mismatch: {path}")
        actual = sha256_file(path)
        if actual != entry["actual_sha256"]:
            raise AssertionError(f"model asset hash mismatch: {path}")
        checked[entry["filename"]] = {"sha256": actual, "bytes": path.stat().st_size}
    return checked


def load_nf4_cpu(torch, AutoModelForCausalLM, BitsAndBytesConfig, feature_qa):
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU replay requires CUDA unavailable and uninitialized")
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.manual_seed(20260913)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=["lm_head"])
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        trust_remote_code=False, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}, attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise AssertionError("CPU model is not actually 4-bit")
    spec = feature_qa.architecture(model)
    if next(model.parameters()).device.type != "cpu" or spec != {
        "model_type": "llama", "layers": 32, "heads": 32, "kv_heads": 32,
        "head_dim": 128, "hidden_size": 4096, "lb_width": 1024, "vocab_size": 32000}:
        raise AssertionError("CPU device or architecture mismatch")
    return model, {"seconds": time.perf_counter() - started, "device": "cpu", "architecture": spec,
                   "quantization": "NF4 double quantization, BF16 compute"}


def materialize_model_plan(row: dict, tokenizer) -> dict:
    rendered = row["released_prompt"] + " " + row["original_response"]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=True, padding=False, truncation=False)
    ids = [int(v) for v in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if ids != row["full_input_ids"] or list(map(int, encoded["attention_mask"])) != row["attention_mask"]:
        raise AssertionError("tokenizer no longer reproduces conversion_v1")
    answer_positions = [int(v) for v in row["answer_token_positions"]]
    if [ids[i] for i in answer_positions] != row["response_token_ids"]:
        raise AssertionError("answer ID alignment mismatch")
    begin, end = row["reference_character_range"]
    context_positions = [i for i, (left, right) in enumerate(offsets)
                         if right > begin and left < end and right > left]
    if not context_positions or max(context_positions) >= min(answer_positions):
        raise AssertionError("reference token range invalid")
    view = {
        "input_ids": ids,
        "attention_mask": row["attention_mask"],
        "answer_token_positions": answer_positions,
        "answer_token_ids": row["response_token_ids"],
        "response_token_offsets": row["response_token_offsets"],
        "response_token_offsets_raw": row["response_token_offsets_raw"],
        "context_token_positions": context_positions,
    }
    return {"partition": "fit", "official_split": "train", "response_id": row["response_id"],
            "source_partition": "auxiliary_candidate_fit", "original": view,
            "labels_used": False, "response_regenerated": False}


def extract_one(model, plan: dict, feature_qa):
    arrays, metadata = feature_qa.extract_features(model, plan, include_delta=False)
    metadata["partition"] = "auxiliary_candidate_fit"
    metadata["adapter_partition_for_formula_guard"] = "fit"
    metadata["labels_used"] = False
    return arrays, metadata


def validate_arrays(arrays, row: dict, np) -> int:
    count = len(row["answer_token_positions"])
    if set(arrays) != set(FEATURE_KEYS):
        raise AssertionError("feature key mismatch")
    if arrays["lb"].shape != (count, 1024) or arrays["nll"].shape != (count,) or arrays["hidden_last"].shape != (count, 4096):
        raise AssertionError("feature shape mismatch")
    if not all(np.isfinite(arrays[name]).all() for name in ("lb", "nll", "hidden_last")):
        raise AssertionError("non-finite feature")
    if not np.all((arrays["lb"] >= 0) & (arrays["lb"] <= 1)) or not np.all(arrays["nll"] >= 0):
        raise AssertionError("feature range mismatch")
    for key, source in (("token_ids", "response_token_ids"),
                        ("answer_token_positions", "answer_token_positions"),
                        ("response_token_offsets", "response_token_offsets"),
                        ("response_token_offsets_raw", "response_token_offsets_raw")):
        if not np.array_equal(arrays[key], np.asarray(row[source])):
            raise AssertionError(f"coordinate mismatch: {key}")
    return count


def save_npz(path: Path, arrays, np) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def deterministic_smoke_rows(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (len(row["full_input_ids"]), row["response_id"]))
    return [ordered[0], ordered[-1]]


def fit_compatibility_rows() -> list[dict]:
    visible_fields = ("response_id", "source_id", "group_id", "partition", "official_split",
                      "released_prompt", "retrieved_passages", "original_response")
    rows = []
    for raw in iter_jsonl(FIT_RECORDS):
        if raw.get("partition") != "fit" or raw.get("official_split") != "train":
            raise AssertionError("non-fit compatibility row")
        rows.append({name: raw[name] for name in visible_fields})
    if len(rows) != 634:
        raise AssertionError("fit compatibility count mismatch")
    ordered = sorted(rows, key=lambda row: row["response_id"])
    return [ordered[0], ordered[-1]]


def compare_fit_cache(model, tokenizer, feature_qa, np) -> list[dict]:
    checks = []
    rows = json.loads(FIT_COMPAT_INPUTS.read_text(encoding="utf-8"))
    if sha256_file(FIT_COMPAT_INPUTS) != json.loads((OUT / "PLAN.json").read_text(encoding="utf-8"))["sanitized_fit_compatibility_inputs"]["sha256"]:
        raise AssertionError("fit compatibility input hash mismatch")
    for row in rows:
        plan = feature_qa.prepare_row(tokenizer, row, include_no_context=False)
        arrays, metadata = feature_qa.extract_features(model, plan, include_delta=False)
        with np.load(FIT_FEATURES / f"{row['response_id']}.npz", allow_pickle=False) as cached:
            differences = {
                "lb_max_abs": float(np.max(np.abs(arrays["lb"].astype(np.float64) - cached["lb"].astype(np.float64)))),
                "nll_max_abs": float(np.max(np.abs(arrays["nll"].astype(np.float64) - cached["nll"].astype(np.float64)))),
                "hidden_max_abs": float(np.max(np.abs(arrays["hidden_last"].astype(np.float64) - cached["hidden_last"].astype(np.float64)))),
            }
            coordinates_exact = all(np.array_equal(arrays[name], cached[name]) for name in
                                    ("token_ids", "answer_token_positions", "response_token_offsets",
                                     "response_token_offsets_raw", "token_start", "token_end"))
        passed = coordinates_exact and all(differences[name] <= CPU_SMOKE_TOLERANCE[name]
                                           for name in CPU_SMOKE_TOLERANCE)
        if not passed:
            raise AssertionError(("CPU NF4 differs from frozen fit cache", row["response_id"], differences,
                                  coordinates_exact))
        checks.append({"response_id": row["response_id"], "coordinates_exact": coordinates_exact,
                       "differences": differences, "tolerances": CPU_SMOKE_TOLERANCE,
                       "feature_seconds": metadata["seconds"], "passed": passed})
    return checks


def model_smoke() -> None:
    require_execution_gate()
    rows = frozen_model_inputs()
    np, torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, feature_qa = lazy_runtime()
    assets = verify_model_assets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True,
                                              trust_remote_code=False)
    model = None; started = time.perf_counter(); checks = []
    try:
        model, loading = load_nf4_cpu(torch, AutoModelForCausalLM, BitsAndBytesConfig, feature_qa)
        for row in deterministic_smoke_rows(rows):
            plan = materialize_model_plan(row, tokenizer)
            arrays, metadata = extract_one(model, plan, feature_qa)
            validate_arrays(arrays, row, np)
            checks.append({"response_id": row["response_id"], "input_tokens": len(row["full_input_ids"]),
                           "response_tokens": len(row["answer_token_positions"]),
                           "feature_seconds": metadata["seconds"], "finite_and_coordinates_exact": True})
        compatibility = compare_fit_cache(model, tokenizer, feature_qa, np)
        value = {"schema_version": "ragognize-no-hint-cpu-nf4-smoke-v1", "status": "PASS",
                 "runner_sha256": sha256_file(Path(__file__)), "source_sha256": {"answers": sha256_file(ANSWERS),
                 "sequences": sha256_file(SEQUENCES)}, "model_assets": assets, "loading": loading,
                 "ragognize_checks": checks, "ragtruth_fit_cache_compatibility": compatibility,
                 "wall_seconds": time.perf_counter() - started,
                 "pretrained_model_loaded": True, "device": "CPU", "labels_used": False,
                 "calibration_or_test_rows_read": False}
        atomic_json(OUT / "MODEL_SMOKE.json", value)
    finally:
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_initialized():
            raise AssertionError("CUDA initialized during CPU smoke")


def cached(row: dict, signature_sha: str, np):
    path = FEATURE_OUT / f"{row['response_id']}.npz"
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        if path.exists():
            raise AssertionError("orphan NPZ")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete") or meta["signature_sha256"] != signature_sha:
        raise AssertionError("cached metadata signature mismatch")
    if sha256_file(path) != meta["npz_sha256"]:
        raise AssertionError("cached NPZ hash mismatch")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {name: opened[name] for name in opened.files}
    validate_arrays(arrays, row, np)
    return meta


def run_replay() -> None:
    require_execution_gate()
    smoke_path = OUT / "MODEL_SMOKE.json"
    if not smoke_path.is_file():
        raise RuntimeError("REPLAY_BLOCKED: CPU model smoke missing")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("REPLAY_BLOCKED: stale or failed CPU model smoke")
    rows = frozen_model_inputs()
    np, torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, feature_qa = lazy_runtime()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True,
                                              trust_remote_code=False)
    signature = {"schema_version": "ragognize-no-hint-cpu-nf4-feature-signature-v1",
                 "runner_sha256": sha256_file(Path(__file__)), "feature_formula_sha256": sha256_file(FEATURE_CODE),
                 "source_sha256": {"answers": sha256_file(ANSWERS), "sequences": sha256_file(SEQUENCES)},
                 "model_smoke_sha256": sha256_file(smoke_path), "response_ids": [row["response_id"] for row in rows],
                 "features": list(FEATURE_KEYS), "labels_used": False,
                 "calibration_or_test_rows_read": False, "device": "CPU"}
    signature_sha = digest(signature)
    FEATURE_OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(FEATURE_OUT).free < 6 * 2**30:
        raise RuntimeError("Need at least 6 GiB free output disk")
    lock = OUT / ".runner.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii")); os.close(descriptor)
    model = None; records = []; started = time.perf_counter()
    try:
        for row in rows:
            entry = cached(row, signature_sha, np)
            if entry is None:
                break
            records.append(entry)
        model, loading = load_nf4_cpu(torch, AutoModelForCausalLM, BitsAndBytesConfig, feature_qa)
        for row in rows[len(records):]:
            plan = materialize_model_plan(row, tokenizer)
            arrays, metadata = extract_one(model, plan, feature_qa)
            count = validate_arrays(arrays, row, np)
            path = FEATURE_OUT / f"{row['response_id']}.npz"
            save_npz(path, arrays, np)
            entry = {"complete": True, "response_id": row["response_id"], "group_id": row["group_id"],
                     "partition": "auxiliary_candidate_fit", "official_split": "train",
                     "signature_sha256": signature_sha, "response_tokens": count,
                     "npz_sha256": sha256_file(path), "npz_bytes": path.stat().st_size,
                     "raw_array_bytes": sum(value.nbytes for value in arrays.values()),
                     "feature_metadata": metadata, "labels_used": False,
                     "calibration_or_test_rows_read": False}
            atomic_json(path.with_suffix(".json"), entry); records.append(entry)
            if len(records) % 10 == 0:
                print("RAGOGNIZE_CPU_REPLAY", len(records), "/", len(rows), flush=True)
        audited = [cached(row, signature_sha, np) for row in rows]
        if any(item is None for item in audited):
            raise AssertionError("final cache audit incomplete")
        atomic_json(OUT / "MANIFEST.json", {"schema_version": "ragognize-no-hint-cpu-nf4-manifest-v1",
            "status": "COMPLETE", "signature": signature, "signature_sha256": signature_sha,
            "rows": len(records), "response_tokens": sum(row["response_tokens"] for row in records),
            "model_loading": loading, "wall_seconds": time.perf_counter() - started,
            "all_hashes_shapes_coordinates_and_finite_values_checked": True,
            "labels_used": False, "calibration_or_test_rows_read": False, "device": "CPU"})
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            raise AssertionError("CUDA initialized during CPU replay")
        if lock.exists():
            lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--selfcheck", action="store_true")
    action.add_argument("--model-smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan:
        value = build_plan()
        print(json.dumps({"status": value["status"], "execution_gate": value["execution_gate"],
                          "resource_estimate": value["resource_estimate"]}, indent=2))
    elif args.selfcheck:
        rows = load_visible_rows()
        print(json.dumps(static_selfcheck(rows), indent=2))
    elif args.model_smoke:
        model_smoke()
    else:
        run_replay()


if __name__ == "__main__":
    main()
