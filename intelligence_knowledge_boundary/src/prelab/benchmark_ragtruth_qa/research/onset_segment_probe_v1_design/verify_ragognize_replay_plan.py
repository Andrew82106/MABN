#!/usr/bin/env python3
"""CPU-only independent check of the label-blind RAGognize replay plan."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
OUT = HERE / "ragognize_no_hint_replay_v1"
RUNNER = HERE / "ragognize_no_hint_replay.py"
FORBIDDEN = {"released_hallucinations", "released_labels_sha256", "answer_risk",
             "risk_mask", "hallucination", "label", "labels"}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify() -> dict:
    plan = json.loads((OUT / "PLAN.json").read_text(encoding="utf-8"))
    check = json.loads((OUT / "SELF_CHECK.json").read_text(encoding="utf-8"))
    inputs = OUT / plan["sanitized_model_inputs"]["path"]
    compatibility = OUT / plan["sanitized_fit_compatibility_inputs"]["path"]
    if sha(RUNNER) != plan["runner_sha256"] or sha(inputs) != plan["sanitized_model_inputs"]["sha256"]:
        raise AssertionError("runner/model-input hash mismatch")
    if sha(compatibility) != plan["sanitized_fit_compatibility_inputs"]["sha256"]:
        raise AssertionError("compatibility-input hash mismatch")
    with gzip.open(inputs, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 971 or any(set(row) & FORBIDDEN for row in rows):
        raise AssertionError("model inputs contain wrong rows or annotation fields")
    if any(row.get("labels_used") is not False or row.get("official_split") != "train" or
           row.get("partition") != "auxiliary_candidate_fit" for row in rows):
        raise AssertionError("model input scope mismatch")
    full_tokens = sum(len(row["full_input_ids"]) for row in rows)
    response_tokens = sum(len(row["answer_token_positions"]) for row in rows)
    if (full_tokens, response_tokens) != (684655, 102778):
        raise AssertionError("token totals mismatch")
    expected_raw = response_tokens * (1024 * 4 + 4 + 4096 * 4 + 40)
    estimate = plan["resource_estimate"]
    if estimate["raw_output_total_bytes"] != expected_raw or estimate["projected_68d_training_cache_bytes_float32"] != response_tokens * 68 * 4:
        raise AssertionError("resource byte arithmetic mismatch")
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("model_smoke", "run_replay"):
        first = functions[name].body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and
                isinstance(first.value.func, ast.Name) and first.value.func.id == "require_execution_gate"):
            raise AssertionError("model action is not fail-closed")
    if plan["execution_gate"]["status"] != "BLOCKED" or plan["execution_gate"]["human_only_gate_status"] != "STOP_971_REPLAY":
        raise AssertionError("v1 stop gate missing")
    if check["pretrained_model_loaded"] or check["model_imported"] or not check["sanitized_model_inputs_checked"]:
        raise AssertionError("static selfcheck state mismatch")
    forbidden_outputs = [OUT / "MODEL_SMOKE.json", OUT / "MANIFEST.json", OUT / "features"]
    if any(path.exists() for path in forbidden_outputs):
        raise AssertionError("model smoke or 971 replay unexpectedly started")
    result = {
        "schema_version": "ragognize-no-hint-replay-plan-independent-verification-v1",
        "status": "PASS",
        "sanitized_rows": len(rows), "full_input_tokens": full_tokens,
        "response_tokens": response_tokens, "annotation_fields_in_model_inputs": 0,
        "resource_byte_arithmetic_recomputed": True,
        "model_actions_gate_before_runtime_import": True,
        "execution_gate": "BLOCKED_BY_STOP_971_REPLAY",
        "pretrained_model_loaded": False, "feature_replay_started": False,
        "calibration_or_test_rows_read": False, "gpu_started": False,
    }
    pending = OUT / "VERIFY.json.pending"
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(OUT / "VERIFY.json")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
