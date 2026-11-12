"""CPU-only unit and full-plan checks for the frozen RAGognizer adapter."""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from adapter import (
    PLAN_VERSION,
    RAW_VERSION,
    ZERO_CHAIN,
    adapt_probabilities,
    add_chain_fields,
    character_overlap_map,
    exact_token_map,
    freeze_predictions,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_chained_row,
)


HERE = Path(__file__).resolve().parent
PLAN = HERE / "INFERENCE_PLAN.jsonl"
OUTPUT = HERE / "CPU_SELFTEST.json"
EXPECTED_PLAN_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
EXPECTED_ROWS = {"fit": 3_680, "calibration": 159}
EXPECTED_GROUPS = {"fit": 615, "calibration": 154}
EXPECTED_WINDOWS = {"fit": 653_979, "calibration": 42_241}


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"{actual} != {expected}")


def make_plan(
    mapping: list[list[int]], native_count: int, local_count: int, starts: list[int]
) -> dict:
    core = {
        "version": PLAN_VERSION,
        "partition": "fit",
        "answer_id": "synthetic",
        "answer_sha256": "0" * 64,
        "author_response_tokens": {
            "token_ids": list(range(native_count)),
            "coordinate_mode": "synthetic",
            "official_pack_exact": True,
        },
        "shared_bpe_tokens": {"token_ids": list(range(local_count))},
        "mapping": {
            "mode": "character_overlap",
            "local_to_author_token_indices": mapping,
        },
        "eligible_windows": [
            {
                "window_id": f"synthetic__k4_{start:05d}",
                "token_start": start,
                "slot_count": 4 if local_count >= 4 else local_count,
            }
            for start in starts
        ],
    }
    core["plan_row_sha256"] = sha256_json(core)
    return core


def must_raise(callable_, phrase: str) -> None:
    try:
        callable_()
    except ValueError as exc:
        if phrase not in str(exc):
            raise AssertionError(f"wrong failure: {exc}") from exc
    else:
        raise AssertionError("expected ValueError")


def check_no_forbidden_keys(value: object, location: str = "root") -> None:
    forbidden = ("label", "risk", "gold", "annotation")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(term in lowered for term in forbidden):
                raise AssertionError(f"forbidden plan key {location}.{key}")
            check_no_forbidden_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_no_forbidden_keys(child, f"{location}[{index}]")


def main() -> None:
    checks: list[str] = []

    direct = exact_token_map(
        [10, 11, 12, 13, 14],
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        [10, 11, 12, 13, 14],
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
    )
    if direct != [[0], [1], [2], [3], [4]]:
        raise AssertionError("one-to-one exact map changed")
    result = adapt_probabilities(
        make_plan(direct, 5, 5, [0, 1]), [0.1, 0.2, 0.3, 0.4, 0.9]
    )
    close(result["shared_k4_window_probabilities"][0], 0.25)
    close(result["shared_k4_window_probabilities"][1], 0.45)
    close(result["shared_answer_probability"], 0.45)
    close(result["author_native_answer_probability"], 0.9)
    if result["partition"] != "fit":
        raise AssertionError("adapter did not preserve plan partition")
    checks.append("exact-token one-to-one, partition, four-slot, and answer reductions")

    fallback = character_overlap_map(
        [[0, 2], [2, 4], [4, 5]],
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
    )
    if fallback != [[0], [0], [1], [1], [2]]:
        raise AssertionError("split-token character map changed")
    result = adapt_probabilities(
        make_plan(fallback, 3, 5, [0, 1]), [0.2, 0.8, 0.6]
    )
    close(result["shared_k4_window_probabilities"][0], 0.5)
    close(result["shared_k4_window_probabilities"][1], 0.6)
    close(result["shared_answer_probability"], 0.6)
    checks.append("author token split across multiple local BPE slots")

    many_to_one = character_overlap_map(
        [[0, 2], [2, 4], [4, 5]],
        [[0, 1], [1, 3], [3, 4], [4, 5]],
    )
    if many_to_one != [[0], [0, 1], [1], [2]]:
        raise AssertionError("multi-overlap character map changed")
    result = adapt_probabilities(
        make_plan(many_to_one, 3, 4, [0]), [0.2, 0.8, 0.6]
    )
    close(result["shared_bpe_token_probabilities"][1], 0.5)
    close(result["shared_k4_window_probabilities"][0], 0.525)
    checks.append("multiple intersecting author tokens receive equal arithmetic weight")

    short = adapt_probabilities(make_plan([[0]], 1, 1, [0]), [0.7])
    close(short["shared_k4_window_probabilities"][0], 0.7)
    close(short["shared_answer_probability"], 0.7)
    checks.append("project N<4 short-answer window averages its actual slots without pad/drop")

    must_raise(
        lambda: character_overlap_map([[0, 1]], [[2, 3]]),
        "no intersecting author token",
    )
    must_raise(
        lambda: exact_token_map([1], [[0, 1]], [2], [[0, 1]]),
        "token IDs differ",
    )
    must_raise(
        lambda: adapt_probabilities(
            make_plan([[0], [0], [0], [0]], 1, 4, [0]), [1.1]
        ),
        "finite probability",
    )
    checks.append("uncovered slots, false exact alignment, and invalid probabilities fail closed")

    if EXPECTED_PLAN_SHA256 == "__FULL_PLAN_SHA256__":
        raise AssertionError("selfcheck plan hash placeholder remains")
    if sha256_file(PLAN) != EXPECTED_PLAN_SHA256:
        raise AssertionError("real plan hash changed")
    plans = list(read_jsonl(PLAN))
    row_counts = Counter(str(row["partition"]) for row in plans)
    groups: dict[str, set[str]] = defaultdict(set)
    windows = Counter()
    for row in plans:
        groups[str(row["partition"])].add(str(row["group_id"]))
        windows[str(row["partition"])] += len(row["eligible_windows"])
        check_no_forbidden_keys(row)
    if dict(row_counts) != EXPECTED_ROWS:
        raise AssertionError("real plan answer counts changed")
    if {key: len(groups[key]) for key in EXPECTED_GROUPS} != EXPECTED_GROUPS:
        raise AssertionError("real plan group counts changed")
    if groups["fit"] & groups["calibration"]:
        raise AssertionError("fit/calibration groups overlap")
    if dict(windows) != EXPECTED_WINDOWS:
        raise AssertionError("real plan window counts changed")
    if any(row["mapping"]["mode"] != "character_overlap" for row in plans):
        raise AssertionError("real plan contains a non-frozen mapping mode")
    if any(
        not indices
        for row in plans
        for indices in row["mapping"]["local_to_author_token_indices"]
    ):
        raise AssertionError("real plan contains an unmapped local BPE slot")
    checks.append("3,839-answer full plan is label-free, isolated, and fully mapped")

    first = plans[0]
    synthetic_run_identity = {
        "route": "model_card_default_transformer_heads_with_accelerate_cpu_offload",
        "device_placement_change_only": True,
        "checkpoint_revision": "2" * 40,
        "ragognizer_commit": "3" * 40,
        "transformer_heads_commit": "4" * 40,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "runner_script_sha256": "5" * 64,
        "model_manifest_sha256": "6" * 64,
        "source_audit_sha256": "7" * 64,
        "runtime_alias_audit_sha256": "8" * 64,
        "torch_dtype_argument": "torch.bfloat16",
        "quantization": None,
        "postprocessor": False,
        "batch_size": 1,
        "author_probability": "sigmoid(integrated hallu_head_neg_16 logit)",
        "python": "synthetic",
        "versions": {"torch": "synthetic"},
    }
    synthetic_run_identity_sha256 = sha256_json(synthetic_run_identity)
    raw_core = {
        "version": RAW_VERSION,
        "route": "model_card_default_transformer_heads",
        "partition": first["partition"],
        "answer_id": first["answer_id"],
        "answer_sha256": first["answer_sha256"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "plan_row_sha256": first["plan_row_sha256"],
        "run_identity_sha256": synthetic_run_identity_sha256,
        "author_response_token_ids": first["author_response_tokens"]["token_ids"],
        "author_response_token_char_intervals": first["author_response_tokens"][
            "char_intervals"
        ],
        "author_response_token_probabilities": [0.25]
        * len(first["author_response_tokens"]["token_ids"]),
        "author_native_preds_0_6523_appendix_only": [0]
        * len(first["author_response_tokens"]["token_ids"]),
        "postprocessor": False,
        "quantization": None,
        "threshold_applied": False,
        "author_coordinate_mode": first["author_response_tokens"]["coordinate_mode"],
        "author_public_packer_exact": first["author_response_tokens"]["official_pack_exact"],
    }
    raw = add_chain_fields(raw_core, ZERO_CHAIN, "raw_row_sha256")
    if validate_chained_row(raw, ZERO_CHAIN, "raw_row_sha256", "synthetic") != raw[
        "chain_sha256"
    ]:
        raise AssertionError("raw chain did not validate")
    with tempfile.TemporaryDirectory() as directory:
        raw_path = Path(directory) / "raw.jsonl"
        gpu_audit_path = Path(directory) / "gpu_audit.json"
        output_path = Path(directory) / "adapted.jsonl"
        raw_path.write_text(
            json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        gpu_audit = {
            "version": "synthetic-gpu-audit",
            "status": "smoke_complete",
            "run_identity": synthetic_run_identity,
            **synthetic_run_identity,
            "run_identity_sha256": synthetic_run_identity_sha256,
            "rows": 1,
            "run_scope": "smoke",
            "selection": {
                "partition": None,
                "limit": 1,
                "answer_ids_sha256": sha256_json([str(first["answer_id"])]),
            },
            "output_sha256": sha256_file(raw_path),
            "raw_final_chain_sha256": raw["chain_sha256"],
            "labels_read": False,
            "test_opened": False,
        }
        gpu_audit_path.write_text(
            json.dumps(gpu_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        subset_audit = freeze_predictions(
            PLAN,
            raw_path,
            output_path,
            gpu_audit_path=gpu_audit_path,
            allow_subset=True,
        )
        subset_output = list(read_jsonl(output_path))
        tampered_gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
        tampered_gpu_audit["run_identity"]["batch_size"] = 2
        gpu_audit_path.write_text(
            json.dumps(tampered_gpu_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        must_raise(
            lambda: freeze_predictions(
                PLAN,
                raw_path,
                output_path,
                gpu_audit_path=gpu_audit_path,
                allow_subset=True,
            ),
            "does not match raw rows",
        )
    if (
        subset_audit["rows"] != 1
        or not subset_audit["subset_mode"]
        or not subset_audit["gpu_audit_verified"]
        or len(subset_output) != 1
    ):
        raise AssertionError("smoke subset mode failed")
    if any(
        score != 0.25
        for score in subset_output[0]["shared_k4_window_probabilities"]
    ):
        raise AssertionError("constant-probability smoke mapping changed")
    if subset_output[0]["raw_chain_sha256"] != raw["chain_sha256"]:
        raise AssertionError("raw-to-adapted chain link changed")
    checks.append("one-answer smoke binds runner identity, GPU audit, raw, and adapted chains")

    tampered = dict(raw)
    tampered["answer_id"] = "tampered"
    must_raise(
        lambda: validate_chained_row(
            tampered, ZERO_CHAIN, "raw_row_sha256", "tampered"
        ),
        "row hash mismatch",
    )
    checks.append("per-row chain fails closed under payload tampering")

    audit = {
        "version": "ragognizer-adapter-cpu-selftest-v2",
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "real_plan_sha256": sha256_file(PLAN),
        "answers": len(plans),
        "partition_answers": dict(row_counts),
        "groups": {key: len(groups[key]) for key in EXPECTED_GROUPS},
        "fit_calibration_group_intersection": 0,
        "eligible_windows": sum(windows.values()),
        "partition_windows": dict(windows),
        "labels_read": False,
        "gpu_workload_started": False,
        "test_opened": False,
    }
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
