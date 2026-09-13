"""Hardened provenance and deterministic replay check for ReDeEP extraction.

This verifier never opens a gold-label file or any test artifact.  It first binds
every executable/model/upstream identity needed by the completed run, then
re-extracts four fixed fit/calibration rows spanning short, median, and long
sequences.  Fresh tensors must equal the persisted raw tensors element by element.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from accelerate import cpu_offload
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "research" / "redeep_formal_baseline_v1"
MODEL = PROJECT.parent / "models" / "Llama-2-7b-chat-hf"
UPSTREAM = PROJECT / "third_party" / "ReDEeP-ICLR"
INPUTS = ROOT / "feature_inputs.jsonl"
RESULT = ROOT / "EXTRACTION_IDENTITY_VERIFICATION.json"
RUN_EVIDENCE = ROOT / "RUN_IDENTITY_EVIDENCE.json"
HEADS_FILE = UPSTREAM / "ReDeEP" / "log" / "test_llama2_7B" / "topk_heads.json"

EXPECTED_COMMIT = "4d081915b8fb4430fda65c411da61540cc73cc57"
EXPECTED_HASHES = {
    MODEL / "model-00001-of-00002.safetensors": "66dec18c9f1705b9387d62f8485f4e7d871ca388718786737ed3c72dbfaac9fb",
    MODEL / "model-00002-of-00002.safetensors": "0fd6895090da1b2ccffdb93964847709a3b31e6b69fe7dc5a480dce37c811b1d",
    MODEL / "model.safetensors.index.json": "11b694a9d4bffdac71733db7f782be0df1f87a65fb9ef05bcf6d317029c22364",
    MODEL / "config.json": "3c66a81e29ce617a74ff2e4055e7fb275b5c9457103864745693fbce59b368a2",
    MODEL / "tokenizer.json": "f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7",
    MODEL / "tokenizer.model": "9e556afd44213b6bd1be2b850ebbBD98f5481437a8021afaf58ee7fb1818d347".lower(),
    MODEL / "tokenizer_config.json": "37951d5a1e0c6a77f88c61f79972f17e889b62258aadff4c61eacd3fb567fe72",
    MODEL / "special_tokens_map.json": "2ea88203b955648951a454e73a73967ff77ba4e79ae8ceb4400d4ab2cb040777",
    PROJECT / "src" / "run_redeep_formal_baseline_v1.py": "63e350bfaa77a614cfdec88218c31d5a8cd692c363c00976551adc91983191a2",
    PROJECT / "src" / "redeep_formal_core_v1.py": "8fded50579a701c27d43be2edd83fe0303cc7f4df9cae7561028181e26d5efdb",
    PROJECT / "src" / "score_redeep_formal_baseline_v1.py": "c5ffa368b6e91780f863bfb2ff523941bb0273651ef2f9475b2df66f4585417f",
    INPUTS: "cd44ebf504d86088ba7b039a1d7263994780ded9ea7a22ba95636e25c4862cb8",
    HEADS_FILE: "633a37382dfe46242028cea3d4f28c6bb2102da1859572f6077a39a612228a4b",
    UPSTREAM / "ReDeEP" / "token_level_detect.py": "333c833a8a8f334a430a6a20159b993bf12328ecc40025c8c958b4a70d60da55",
    UPSTREAM / "ReDeEP" / "token_level_reg.py": "e508d443a75f19a48ec4ceaaee5ded26481379d49fe07867fecb1ac70315e13f",
    UPSTREAM / "ReDeEP" / "log" / "test_llama2_7B" / "token_hyperparameter.json": "31a78d840ae19e28b7b27c3c833876c0d59b3cc5e927bf30074b06b02f136892",
}
EXPECTED_HEADS = [
    [1, 14], [1, 21], [1, 27], [2, 5], [2, 22], [3, 0], [3, 19], [5, 13],
    [5, 29], [10, 20], [13, 20], [15, 7], [16, 1], [18, 9], [18, 10],
    [18, 13], [19, 1], [20, 1], [20, 5], [20, 15], [20, 17], [20, 22],
    [22, 10], [23, 8], [23, 30], [25, 0], [27, 9], [28, 18], [31, 18],
    [31, 19], [31, 24], [31, 28],
]
TARGETS = [
    {"answer_id": "16656", "partition": "fit", "length_role": "shortest_fit", "full_tokens": 284},
    {"answer_id": "17628", "partition": "fit", "length_role": "median_fit", "full_tokens": 631},
    {"answer_id": "17751", "partition": "fit", "length_role": "longest_fit", "full_tokens": 1252},
    {"answer_id": "11883", "partition": "calibration", "length_role": "median_calibration", "full_tokens": 732},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_static_identities() -> dict:
    observed = {}
    failures = []
    for path, expected in EXPECTED_HASHES.items():
        value = sha256_file(path)
        key = path.relative_to(PROJECT.parent).as_posix()
        observed[key] = value
        if value != expected:
            failures.append({"path": key, "expected": expected, "observed": value})

    commit = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        failures.append({"upstream_commit_expected": EXPECTED_COMMIT, "observed": commit})
    if tracked_status:
        failures.append({"upstream_tracked_status_expected": "clean", "observed": tracked_status})

    heads = sorted(json.loads(HEADS_FILE.read_text(encoding="utf-8")))
    if heads != EXPECTED_HEADS:
        failures.append({"candidate_heads_expected": EXPECTED_HEADS, "observed": heads})

    evidence = json.loads(RUN_EVIDENCE.read_text(encoding="utf-8-sig"))
    runner_path = PROJECT / "src" / "run_redeep_formal_baseline_v1.py"
    runner_stat = runner_path.stat()
    if evidence["runner_sha256"] != EXPECTED_HASHES[runner_path]:
        failures.append({"run_evidence_runner_sha": evidence["runner_sha256"]})
    recorded_mtime = datetime.fromisoformat(evidence["runner_last_write_utc"].replace("Z", "+00:00")).timestamp()
    if abs(runner_stat.st_mtime - recorded_mtime) > 1e-5:
        failures.append({
            "runner_mtime_changed_since_active_run_capture": {
                "recorded": evidence["runner_last_write_utc"],
                "current_unix": runner_stat.st_mtime,
            }
        })
    if not evidence.get("processes") or not evidence.get("gpu_lock"):
        failures.append({"run_process_identity_missing": True})

    if failures:
        raise RuntimeError(f"identity binding failed: {failures}")
    return {
        "hashes": observed,
        "upstream_commit": commit,
        "upstream_tracked_files_clean": True,
        "candidate_heads": heads,
        "candidate_head_count": len(heads),
        "run_evidence_sha256": sha256_file(RUN_EVIDENCE),
        "run_process_start_evidence_present": True,
        "runner_last_write_utc_at_active_run_capture": evidence["runner_last_write_utc"],
    }


def load_target_rows() -> list[dict]:
    wanted = {x["answer_id"]: x for x in TARGETS}
    found = {}
    with INPUTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            answer_id = str(row["answer_id"])
            if answer_id in wanted:
                found[answer_id] = row
    if set(found) != set(wanted):
        raise RuntimeError("representative row identity changed")
    rows = []
    for target in TARGETS:
        row = found[target["answer_id"]]
        if row["partition"] != target["partition"] or int(row["official_full_token_count"]) != target["full_tokens"]:
            raise RuntimeError(f"representative row metadata drift: {target['answer_id']}")
        rows.append(row)
    return rows


def array_comparison(fresh: np.ndarray, persisted: np.ndarray) -> dict:
    if fresh.shape != persisted.shape:
        return {"shape_equal": False, "fresh_shape": list(fresh.shape), "persisted_shape": list(persisted.shape)}
    difference = np.abs(fresh.astype(np.float64) - persisted.astype(np.float64))
    return {
        "shape_equal": True,
        "dtype_equal": str(fresh.dtype) == str(persisted.dtype),
        "elementwise_exact": bool(np.array_equal(fresh, persisted)),
        "max_abs_difference": float(difference.max(initial=0.0)),
        "finite_both": bool(np.isfinite(fresh).all() and np.isfinite(persisted).all()),
        "elements": int(fresh.size),
    }


def replay(rows: list[dict], heads: list[list[int]]) -> list[dict]:
    # Import only after the exact hashes of the runner and its core dependency pass.
    from run_redeep_formal_baseline_v1 import (
        ReDeEPActivationCapture,
        exclusive_gpu_lock,
        llama2_official_chat_prompt,
        project_pks,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("representative full-FP16 replay requires CUDA")
    device = torch.device("cuda:0")
    comparisons = []
    with exclusive_gpu_lock("redeep_extraction_identity_verification_v1"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        lm_head_weight = model.lm_head.weight.detach().clone().to(device=device, dtype=torch.float16)
        final_norm_weight = model.model.norm.weight.detach().clone().to(device=device, dtype=torch.float16)
        cpu_offload(model, execution_device=device, offload_buffers=True)
        capture = ReDeEPActivationCapture(model, heads)
        try:
            with torch.inference_mode():
                for row, target in zip(rows, TARGETS):
                    rendered = llama2_official_chat_prompt(row["released_prompt"])
                    encoded = tokenizer(
                        rendered + row["original_response"],
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                    capture.reset(
                        int(row["official_prefix_token_count"]),
                        int(row["official_full_token_count"]),
                        list(map(int, row["paper_reference_token_positions"])),
                    )
                    outputs = model.model(
                        input_ids=encoded.input_ids.to(device),
                        attention_mask=encoded.attention_mask.to(device),
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    ecs_code, ecs_paper = capture.ecs(device)
                    pks_code, pks_paper = project_pks(
                        capture,
                        lm_head_weight,
                        final_norm_weight,
                        float(model.config.rms_norm_eps),
                        1024,
                    )
                    fresh = {
                        "ecs_code": ecs_code,
                        "ecs_paper": ecs_paper,
                        "pks_code": pks_code,
                        "pks_paper": pks_paper,
                    }
                    raw_path = ROOT / "raw_features" / row["partition"] / f"{row['answer_id']}.npz"
                    with np.load(raw_path, allow_pickle=False) as data:
                        record = {
                            **target,
                            "raw_sha256": sha256_file(raw_path),
                            "arrays": {name: array_comparison(value, data[name]) for name, value in fresh.items()},
                            "candidate_heads_exact": bool(np.array_equal(data["candidate_heads"], np.asarray(heads))),
                            "answer_sha256_exact": str(data["answer_sha256"].item()) == row["answer_sha256"],
                        }
                    comparisons.append(record)
                    del outputs, encoded, ecs_code, ecs_paper, pks_code, pks_paper
                    torch.cuda.empty_cache()
        finally:
            capture.close()
            del capture, model, lm_head_weight, final_norm_weight, tokenizer
            torch.cuda.empty_cache()
    return comparisons


def main() -> None:
    started = time.time()
    payload = {
        "version": "redeep-formal-extraction-identity-verification-v1",
        "status": "running",
        "labels_read": False,
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
    }
    try:
        identities = verify_static_identities()
        rows = load_target_rows()
        comparisons = replay(rows, identities["candidate_heads"])
        exact = all(
            row["candidate_heads_exact"]
            and row["answer_sha256_exact"]
            and all(item.get("elementwise_exact", False) for item in row["arrays"].values())
            for row in comparisons
        )
        if not exact:
            raise RuntimeError("one or more representative raw arrays were not elementwise identical")
        payload.update({
            "status": "passed",
            "identity_binding": identities,
            "representative_replays": comparisons,
            "all_four_feature_arrays_elementwise_exact_for_all_rows": True,
            "representative_rows": len(comparisons),
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "verifier_sha256": sha256_file(Path(__file__)),
        })
    except Exception as exc:
        payload.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - started,
        })
        RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "representative_rows": payload["representative_rows"], "elapsed_seconds": payload["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
