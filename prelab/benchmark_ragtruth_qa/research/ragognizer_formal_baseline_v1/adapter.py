"""Label-free RAGognizer token-to-shared-4-BPE score adapter.

The adapter starts at the author's unchanged response-token probabilities.  It
has no learned parameters, never reads labels, and preserves a per-row hash
chain from native output through shared-window output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


VERSION = "ragognizer-zero-parameter-adapter-v2"
PLAN_VERSION = "ragognizer-full-inference-plan-v1"
RAW_VERSION = "ragognizer-native-probabilities-v2"
FROZEN_VERSION = "ragognizer-shared-k4-probabilities-v2"
ZERO_CHAIN = "0" * 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_no}")
            yield value


def add_chain_fields(core: dict, previous_chain: str, row_hash_key: str) -> dict:
    if len(previous_chain) != 64:
        raise ValueError("previous chain hash must have 64 hex characters")
    row_hash = sha256_json(core)
    chain_hash = sha256_text(previous_chain + "\n" + row_hash)
    return {
        **core,
        row_hash_key: row_hash,
        "previous_chain_sha256": previous_chain,
        "chain_sha256": chain_hash,
    }


def validate_chained_row(
    row: dict, previous_chain: str, row_hash_key: str, where: str
) -> str:
    stored_row_hash = str(row.get(row_hash_key, ""))
    stored_previous = str(row.get("previous_chain_sha256", ""))
    stored_chain = str(row.get("chain_sha256", ""))
    core = {
        key: value
        for key, value in row.items()
        if key not in {row_hash_key, "previous_chain_sha256", "chain_sha256"}
    }
    if stored_previous != previous_chain:
        raise ValueError(f"{where} previous chain mismatch")
    if stored_row_hash != sha256_json(core):
        raise ValueError(f"{where} row hash mismatch")
    expected_chain = sha256_text(previous_chain + "\n" + stored_row_hash)
    if stored_chain != expected_chain:
        raise ValueError(f"{where} chain hash mismatch")
    return stored_chain


def validate_plan_row(row: dict, where: str) -> None:
    if row.get("version") != PLAN_VERSION:
        raise ValueError(f"{where} has unexpected plan version")
    stored = str(row.get("plan_row_sha256", ""))
    core = {key: value for key, value in row.items() if key != "plan_row_sha256"}
    if stored != sha256_json(core):
        raise ValueError(f"{where} plan row hash mismatch")


def _finite_probability(value: object, where: str) -> float:
    try:
        score = float(value)
    except Exception as exc:
        raise ValueError(f"{where} is not numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{where} must be a finite probability in [0, 1]")
    return score


def character_overlap_map(
    native_intervals: Sequence[Sequence[int]],
    local_intervals: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Map each local BPE slot to all intersecting author-token indices."""
    native = [tuple(map(int, span)) for span in native_intervals]
    local = [tuple(map(int, span)) for span in local_intervals]
    for name, spans in (("native", native), ("local", local)):
        for index, (start, end) in enumerate(spans):
            if start < 0 or end <= start:
                raise ValueError(f"{name} interval {index} is not nonempty half-open")
    mapped: list[list[int]] = []
    for local_i, (local_start, local_end) in enumerate(local):
        indices = [
            native_i
            for native_i, (native_start, native_end) in enumerate(native)
            if max(local_start, native_start) < min(local_end, native_end)
        ]
        if not indices:
            raise ValueError(f"local BPE slot {local_i} has no intersecting author token")
        mapped.append(indices)
    return mapped


def exact_token_map(
    native_token_ids: Sequence[int],
    native_intervals: Sequence[Sequence[int]],
    local_token_ids: Sequence[int],
    local_intervals: Sequence[Sequence[int]],
) -> list[list[int]]:
    """One-to-one mapping allowed only for identical IDs and character spans."""
    if list(map(int, native_token_ids)) != list(map(int, local_token_ids)):
        raise ValueError("token IDs differ; exact-token alignment is forbidden")
    if [list(map(int, x)) for x in native_intervals] != [
        list(map(int, x)) for x in local_intervals
    ]:
        raise ValueError("token character spans differ; exact-token alignment is forbidden")
    return [[index] for index in range(len(local_token_ids))]


def adapt_probabilities(plan: dict, native_probabilities: Sequence[object]) -> dict:
    """Apply one pre-frozen plan row to unchanged author probabilities."""
    validate_plan_row(plan, str(plan.get("answer_id", "unknown")))
    probabilities = [
        _finite_probability(value, f"native_probabilities[{index}]")
        for index, value in enumerate(native_probabilities)
    ]
    expected_native = len(plan["author_response_tokens"]["token_ids"])
    if len(probabilities) != expected_native:
        raise ValueError(f"native probability length {len(probabilities)} != {expected_native}")

    mapping = plan["mapping"]["local_to_author_token_indices"]
    expected_local = len(plan["shared_bpe_tokens"]["token_ids"])
    if len(mapping) != expected_local:
        raise ValueError("mapping/local-token length mismatch")
    local_probabilities: list[float] = []
    for local_i, source_indices in enumerate(mapping):
        if not source_indices:
            raise ValueError(f"empty mapping at local BPE slot {local_i}")
        if len(set(map(int, source_indices))) != len(source_indices):
            raise ValueError(f"duplicate author token in local BPE slot {local_i}")
        try:
            values = [probabilities[int(index)] for index in source_indices]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"out-of-range mapping at local BPE slot {local_i}") from exc
        local_probabilities.append(math.fsum(values) / len(values))

    window_ids: list[str] = []
    window_probabilities: list[float] = []
    for window in plan["eligible_windows"]:
        start = int(window["token_start"])
        slot_count = int(window.get("slot_count", 4))
        if slot_count not in (1, 2, 3, 4):
            raise ValueError(f"invalid slot count for {window['window_id']}")
        if len(local_probabilities) >= 4 and slot_count != 4:
            raise ValueError(f"non-short answer changed four-slot geometry for {window['window_id']}")
        if len(local_probabilities) < 4 and (
            start != 0 or slot_count != len(local_probabilities) or len(plan["eligible_windows"]) != 1
        ):
            raise ValueError(f"invalid project short-window geometry for {window['window_id']}")
        end = start + slot_count
        if start < 0 or end > len(local_probabilities):
            raise ValueError(f"invalid shared window {window['window_id']}")
        window_ids.append(str(window["window_id"]))
        window_probabilities.append(
            math.fsum(local_probabilities[start:end]) / slot_count
        )
    if not window_probabilities:
        raise ValueError("answer has no eligible shared 4-BPE window")

    return {
        "version": FROZEN_VERSION,
        "partition": str(plan["partition"]),
        "answer_id": str(plan["answer_id"]),
        "answer_sha256": str(plan["answer_sha256"]),
        "plan_row_sha256": str(plan["plan_row_sha256"]),
        "mapping_mode": str(plan["mapping"]["mode"]),
        "author_coordinate_mode": str(
            plan["author_response_tokens"]["coordinate_mode"]
        ),
        "author_public_packer_exact": bool(
            plan["author_response_tokens"]["official_pack_exact"]
        ),
        "author_response_token_probabilities": probabilities,
        "shared_bpe_token_probabilities": local_probabilities,
        "window_ids": window_ids,
        "shared_k4_window_probabilities": window_probabilities,
        "shared_answer_probability": max(window_probabilities),
        "author_native_answer_probability": max(probabilities),
        "author_native_threshold_appendix_only": 0.6523,
        "shared_thresholds_applied": False,
        "labels_read": False,
    }


def _load_and_validate_gpu_audit(
    gpu_audit_path: Path,
    *,
    plan_sha256: str,
    raw_sha256: str,
    raw_rows: int,
    raw_final_chain: str,
    raw_run_identity_sha256: str,
    expected_status: str,
    expected_scope: str,
    expected_partition: str | None,
    expected_answer_ids_sha256: str,
) -> dict:
    audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != expected_status:
        raise ValueError(f"GPU audit status must be {expected_status}")
    checks = {
        "plan_sha256": plan_sha256,
        "output_sha256": raw_sha256,
        "rows": raw_rows,
        "raw_final_chain_sha256": raw_final_chain,
    }
    for key, expected in checks.items():
        if audit.get(key) != expected:
            raise ValueError(f"GPU audit {key} mismatch")
    if audit.get("run_scope") != expected_scope:
        raise ValueError("GPU audit run scope mismatch")
    selection = audit.get("selection")
    if not isinstance(selection, dict) or selection.get("partition") != expected_partition:
        raise ValueError("GPU audit partition selection mismatch")
    if selection.get("answer_ids_sha256") != expected_answer_ids_sha256:
        raise ValueError("GPU audit selected answer identity mismatch")
    run_identity = audit.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("GPU audit lacks its canonical run identity payload")
    if sha256_json(run_identity) != raw_run_identity_sha256:
        raise ValueError("GPU audit run identity does not match raw rows")
    if audit.get("run_identity_sha256") != raw_run_identity_sha256:
        raise ValueError("GPU audit run identity hash changed")
    for key, value in run_identity.items():
        if audit.get(key) != value:
            raise ValueError(f"GPU audit top-level run identity field changed: {key}")
    required_identity = {
        "route": "model_card_default_transformer_heads_with_accelerate_cpu_offload",
        "device_placement_change_only": True,
        "plan_sha256": plan_sha256,
        "torch_dtype_argument": "torch.bfloat16",
        "quantization": None,
        "postprocessor": False,
        "batch_size": 1,
        "author_probability": "sigmoid(integrated hallu_head_neg_16 logit)",
    }
    for key, expected in required_identity.items():
        if run_identity.get(key) != expected:
            raise ValueError(f"GPU run identity {key} mismatch")
    if not isinstance(run_identity.get("versions"), dict):
        raise ValueError("GPU run identity lacks dependency versions")
    if len(str(run_identity.get("runner_script_sha256", ""))) != 64:
        raise ValueError("GPU run identity lacks runner script hash")
    if audit.get("quantization") is not None or audit.get("postprocessor") is not False:
        raise ValueError("GPU audit changed the formal numerical path")
    if audit.get("labels_read") is not False or audit.get("test_opened") is not False:
        raise ValueError("GPU audit crossed the data boundary")
    return audit


def freeze_predictions(
    plan_path: Path,
    raw_path: Path,
    output_path: Path,
    *,
    gpu_audit_path: Path | None = None,
    allow_subset: bool = False,
    partition: str | None = None,
) -> dict:
    plans_all = list(read_jsonl(plan_path))
    raw_rows = list(read_jsonl(raw_path))
    for index, plan in enumerate(plans_all):
        validate_plan_row(plan, f"plan[{index}]")
    if not plans_all or len({str(row["answer_id"]) for row in plans_all}) != len(plans_all):
        raise ValueError("empty plan or duplicate plan answer_id")
    if partition not in (None, "calibration"):
        raise ValueError("only the frozen calibration provisional partition is supported")
    plans = (
        [row for row in plans_all if row["partition"] == partition]
        if partition
        else plans_all
    )
    if not raw_rows or len({str(row["answer_id"]) for row in raw_rows}) != len(raw_rows):
        raise ValueError("empty raw file or duplicate raw answer_id")
    expected_plans = plans[: len(raw_rows)] if allow_subset else plans
    if not allow_subset and len(raw_rows) != len(plans):
        raise ValueError("adapter requires complete selected-plan/raw coverage")
    if [str(row["answer_id"]) for row in raw_rows] != [
        str(row["answer_id"]) for row in expected_plans
    ]:
        raise ValueError("raw rows must be an ordered plan prefix")

    plan_sha256 = sha256_file(plan_path)
    previous_raw_chain = ZERO_CHAIN
    raw_run_identity_hashes = {
        str(row.get("run_identity_sha256", "")) for row in raw_rows
    }
    if len(raw_run_identity_hashes) != 1:
        raise ValueError("raw rows contain mixed run identities")
    raw_run_identity_sha256 = next(iter(raw_run_identity_hashes))
    if len(raw_run_identity_sha256) != 64:
        raise ValueError("raw rows lack a valid run identity hash")
    outputs: list[dict] = []
    previous_output_chain = ZERO_CHAIN
    partition_counts: Counter[str] = Counter()
    partition_windows: Counter[str] = Counter()
    for index, (plan, raw) in enumerate(zip(expected_plans, raw_rows)):
        answer_id = str(plan["answer_id"])
        if raw.get("version") != RAW_VERSION:
            raise ValueError(f"unexpected raw version for {answer_id}")
        if raw.get("partition") != plan["partition"]:
            raise ValueError(f"partition mismatch for {answer_id}")
        if raw.get("plan_sha256") != plan_sha256:
            raise ValueError(f"plan file hash mismatch for {answer_id}")
        if raw.get("plan_row_sha256") != plan["plan_row_sha256"]:
            raise ValueError(f"plan row hash mismatch for {answer_id}")
        if str(raw.get("answer_sha256")) != str(plan["answer_sha256"]):
            raise ValueError(f"answer hash mismatch for {answer_id}")
        if raw.get("author_response_token_ids") != plan["author_response_tokens"]["token_ids"]:
            raise ValueError(f"author token IDs changed for {answer_id}")
        if raw.get("author_response_token_char_intervals") != plan["author_response_tokens"]["char_intervals"]:
            raise ValueError(f"author token spans changed for {answer_id}")
        if raw.get("author_coordinate_mode") != plan["author_response_tokens"]["coordinate_mode"]:
            raise ValueError(f"author coordinate mode changed for {answer_id}")
        if raw.get("author_public_packer_exact") is not plan["author_response_tokens"]["official_pack_exact"]:
            raise ValueError(f"author public packer status changed for {answer_id}")
        if raw.get("route") != "model_card_default_transformer_heads":
            raise ValueError(f"non-primary route for {answer_id}")
        if raw.get("postprocessor") is not False or raw.get("quantization") is not None:
            raise ValueError(f"formal numerical path changed for {answer_id}")
        if raw.get("threshold_applied") is not False:
            raise ValueError(f"native probabilities were thresholded for {answer_id}")
        previous_raw_chain = validate_chained_row(
            raw, previous_raw_chain, "raw_row_sha256", f"raw[{index}]"
        )
        core = adapt_probabilities(plan, raw["author_response_token_probabilities"])
        core["raw_chain_sha256"] = previous_raw_chain
        output = add_chain_fields(core, previous_output_chain, "adapted_row_sha256")
        previous_output_chain = output["chain_sha256"]
        outputs.append(output)
        row_partition = str(plan["partition"])
        partition_counts[row_partition] += 1
        partition_windows[row_partition] += len(output["window_ids"])

    raw_sha256 = sha256_file(raw_path)
    gpu_audit = None
    gpu_audit_sha256 = None
    if partition == "calibration" and not allow_subset:
        adapter_scope = "provisional_calibration"
        expected_gpu_status = "provisional_complete"
        adapter_status = "provisional_complete"
    elif partition is None and not allow_subset:
        adapter_scope = "formal_full"
        expected_gpu_status = "complete"
        adapter_status = "complete"
    else:
        adapter_scope = "smoke"
        expected_gpu_status = "smoke_complete"
        adapter_status = "smoke_complete"
    if gpu_audit_path is not None:
        gpu_audit = _load_and_validate_gpu_audit(
            gpu_audit_path,
            plan_sha256=plan_sha256,
            raw_sha256=raw_sha256,
            raw_rows=len(raw_rows),
            raw_final_chain=previous_raw_chain,
            raw_run_identity_sha256=raw_run_identity_sha256,
            expected_status=expected_gpu_status,
            expected_scope=adapter_scope,
            expected_partition=partition,
            expected_answer_ids_sha256=sha256_json(
                [str(row["answer_id"]) for row in expected_plans]
            ),
        )
        gpu_audit_sha256 = sha256_file(gpu_audit_path)
    else:
        raise ValueError("adaptation requires --gpu-audit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in outputs:
            handle.write(canonical_json(row) + "\n")
    output_sha256 = sha256_file(output_path)
    return {
        "version": "ragognizer-adapter-run-audit-v2",
        "status": adapter_status,
        "adapter_scope": adapter_scope,
        "partition_filter": partition,
        "rows": len(outputs),
        "partition_rows": dict(sorted(partition_counts.items())),
        "windows": sum(partition_windows.values()),
        "partition_windows": dict(sorted(partition_windows.items())),
        "plan_sha256": plan_sha256,
        "gpu_run_audit_sha256": gpu_audit_sha256,
        "raw_sha256": raw_sha256,
        "raw_final_chain_sha256": previous_raw_chain,
        "adapter_script_sha256": sha256_file(Path(__file__).resolve()),
        "output_sha256": output_sha256,
        "adapted_final_chain_sha256": previous_output_chain,
        "labels_read": False,
        "thresholds_applied": False,
        "subset_mode": allow_subset,
        "gpu_audit_verified": gpu_audit is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gpu-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--partition", choices=("calibration",))
    args = parser.parse_args()
    audit = freeze_predictions(
        args.plan.resolve(),
        args.raw.resolve(),
        args.output.resolve(),
        gpu_audit_path=args.gpu_audit.resolve() if args.gpu_audit else None,
        allow_subset=args.allow_subset,
        partition=args.partition,
    )
    args.audit.resolve().write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
