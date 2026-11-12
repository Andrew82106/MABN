"""Prepare label-free ReDeEP(Token) inputs and audit exact fit/cal joins.

This entry point is intentionally hard-coded to the shared fit and calibration
partitions.  It refuses any row whose official split is not RAGTruth train.  It
never opens the historical official-test artifact bundled in the upstream repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

from redeep_formal_core_v1 import VERSION as CORE_VERSION
from redeep_formal_core_v1 import llama2_official_chat_prompt


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "research" / "redeep_formal_baseline_v1"
MODEL = PROJECT.parent / "models" / "Llama-2-7b-chat-hf"
PARTITIONS = {
    "fit": {
        "source": PROJECT / "fit_expansion" / "data" / "fit.jsonl",
        "answers": PROJECT / "fit_expansion" / "data" / "answers_fit.jsonl",
        "tokens": PROJECT / "fit_expansion" / "data" / "tokens_fit.jsonl",
        "windows": PROJECT / "fit_expansion" / "data" / "windows_k4_fit.jsonl",
        "excluded_windows": PROJECT / "fit_expansion" / "data" / "windows_excluded_fit.jsonl",
    },
    "calibration": {
        "source": PROJECT / "data" / "calibration.jsonl",
        "answers": PROJECT / "data" / "answers_calibration.jsonl",
        "tokens": PROJECT / "data" / "tokens_calibration.jsonl",
        "windows": PROJECT / "data" / "windows_k4_calibration.jsonl",
        "excluded_windows": PROJECT / "data" / "windows_excluded_calibration.jsonl",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def key(row: dict) -> tuple[str, str]:
    return str(row["response_id"]), str(row["answer_sha256"])


def audit_partition(name: str, files: dict[str, Path]) -> tuple[list[dict], dict]:
    source_rows = list(read_jsonl(files["source"]))
    answer_rows = list(read_jsonl(files["answers"]))
    token_rows = list(read_jsonl(files["tokens"]))
    expected_partition = name
    for kind, rows in (("source", source_rows), ("answers", answer_rows), ("tokens", token_rows)):
        observed = Counter(str(row.get("partition")) for row in rows)
        if observed != Counter({expected_partition: len(rows)}):
            raise ValueError(f"{kind} partition mismatch for {name}: {dict(observed)}")
    for row in source_rows:
        if str(row.get("official_split")) != "train":
            raise ValueError("sealed/non-train RAGTruth material is forbidden in this preparation")
        if sha256_text(row["original_response"]) != row["answer_sha256"]:
            raise ValueError(f"answer hash mismatch: {row['response_id']}")
        if sha256_text(row["released_prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['response_id']}")

    source_index = {key(row): row for row in source_rows}
    answer_index = {key(row): row for row in answer_rows}
    token_index = {key(row): row for row in token_rows}
    if len(source_index) != len(source_rows):
        raise ValueError(f"duplicate source key in {name}")
    if set(source_index) != set(answer_index) or set(source_index) != set(token_index):
        raise ValueError(f"source/answer/token key mismatch in {name}")

    answer_ids = {str(x["answer_id"]) for x in answer_rows}
    windows_by_answer: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "excluded": 0, "all": 0})
    seen_window_ids: set[str] = set()
    for file_kind in ("windows", "excluded_windows"):
        expected_eligible = file_kind == "windows"
        for row in read_jsonl(files[file_kind]):
            if str(row.get("partition")) != expected_partition:
                raise ValueError(f"window partition mismatch in {name}")
            if bool(row.get("eligible")) != expected_eligible:
                raise ValueError(f"window eligibility/file mismatch: {row.get('window_id')}")
            answer_id = str(row["answer_id"])
            if answer_id not in answer_ids:
                raise ValueError(f"orphan window answer_id={answer_id}")
            window_id = str(row["window_id"])
            if window_id in seen_window_ids:
                raise ValueError(f"duplicate window_id={window_id}")
            seen_window_ids.add(window_id)
            windows_by_answer[answer_id]["all"] += 1
            windows_by_answer[answer_id]["eligible" if row.get("eligible") else "excluded"] += 1

    for row in answer_rows:
        counts = windows_by_answer[str(row["answer_id"])]
        if counts["eligible"] != int(row["eligible_window_count"]):
            raise ValueError(f"eligible-window count mismatch: {row['answer_id']}")
        if counts["excluded"] != int(row["excluded_window_count"]):
            raise ValueError(f"excluded-window count mismatch: {row['answer_id']}")
        tok = token_index[key(row)]
        if int(row["token_count"]) != int(tok["token_count"]):
            raise ValueError(f"token-count mismatch: {row['answer_id']}")

    group_ids = {str(row["group_id"]) for row in source_rows}
    return source_rows, {
        "answers": len(source_rows),
        "groups": len(group_ids),
        "eligible_windows": sum(x["eligible"] for x in windows_by_answer.values()),
        "excluded_windows": sum(x["excluded"] for x in windows_by_answer.values()),
        "all_candidate_windows": len(seen_window_ids),
        "source_answer_token_exact_join": True,
        "official_splits": {"train": len(source_rows)},
        "input_sha256": {kind: sha256_file(path) for kind, path in files.items()},
    }


def percentile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("empty percentile input")
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return int(values[index])


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("a fast tokenizer is required for the label-free character audit")

    rows_by_partition: dict[str, list[dict]] = {}
    audit: dict[str, object] = {
        "version": "redeep-formal-input-audit-v1",
        "core_version": CORE_VERSION,
        "labels_used_to_build_feature_inputs": False,
        "allowed_partitions": ["fit", "calibration"],
        "test_artifacts_accessed_by_preparation_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
        "partitions": {},
    }
    for name, files in PARTITIONS.items():
        rows, part_audit = audit_partition(name, files)
        rows_by_partition[name] = rows
        audit["partitions"][name] = part_audit

    fit_groups = {str(row["group_id"]) for row in rows_by_partition["fit"]}
    cal_groups = {str(row["group_id"]) for row in rows_by_partition["calibration"]}
    overlap = fit_groups & cal_groups
    if overlap:
        raise ValueError(f"fit/cal group leakage: {len(overlap)} groups")
    audit["fit_cal_group_overlap"] = 0

    # Strip every gold/risk field before the later score-mapping stage.  This file
    # is the only shared-window input accepted by that stage; labels are opened by
    # a separate fit-selection/evaluation boundary.
    geometry_path = out_dir / "evaluation_geometry.jsonl"
    geometry_rows = 0
    with geometry_path.open("w", encoding="utf-8", newline="\n") as output:
        for partition in ("fit", "calibration"):
            for row in read_jsonl(PARTITIONS[partition]["windows"]):
                payload = {
                    "version": "redeep-common-k4-geometry-v1",
                    "partition": partition,
                    "response_id": str(row["response_id"]),
                    "answer_id": str(row["answer_id"]),
                    "group_id": str(row["group_id"]),
                    "window_id": str(row["window_id"]),
                    "character_intervals": row["character_intervals"],
                    "labels_present": False,
                }
                output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                geometry_rows += 1
    audit["label_free_evaluation_geometry"] = {
        "path": str(geometry_path.relative_to(PROJECT)),
        "sha256": sha256_file(geometry_path),
        "rows": geometry_rows,
        "allowed_fields": [
            "version", "partition", "response_id", "answer_id", "group_id",
            "window_id", "character_intervals", "labels_present",
        ],
        "gold_or_risk_fields_present": False,
    }

    feature_path = out_dir / "feature_inputs.jsonl"
    full_lengths: list[int] = []
    prefix_lengths: list[int] = []
    response_lengths: list[int] = []
    reference_counts: list[int] = []
    prefix_identity_failures: list[str] = []
    boundary_crossers: list[str] = []
    unscored_prefix_chars: dict[str, int] = {}
    with feature_path.open("w", encoding="utf-8", newline="\n") as output:
        for partition in ("fit", "calibration"):
            for row in rows_by_partition[partition]:
                prompt = row["released_prompt"][:12000]
                response = row["original_response"]
                retrieved = row["retrieved_passages"]
                if prompt.count(retrieved) != 1:
                    raise ValueError(f"retrieved passage is not an exact unique prompt substring: {row['response_id']}")
                rendered = llama2_official_chat_prompt(prompt)
                input_text = rendered + response
                prefix_enc = tokenizer(rendered, add_special_tokens=True)
                full_enc = tokenizer(input_text, add_special_tokens=True, return_offsets_mapping=True)
                prefix_len = len(prefix_enc["input_ids"])
                full_len = len(full_enc["input_ids"])
                if prefix_enc["input_ids"] != full_enc["input_ids"][:prefix_len]:
                    prefix_identity_failures.append(str(row["response_id"]))
                if full_len > int(tokenizer.model_max_length) and int(tokenizer.model_max_length) < 1_000_000:
                    raise ValueError(f"model context limit exceeded: {row['response_id']} has {full_len}")
                if full_enc["input_ids"][:2] != [int(tokenizer.bos_token_id)] * 2:
                    raise ValueError(f"official double-BOS behavior changed: {row['response_id']}")

                answer_offsets: list[list[int]] = []
                for start, end in full_enc["offset_mapping"][prefix_len:]:
                    rel_start = int(start) - len(rendered)
                    rel_end = int(end) - len(rendered)
                    if rel_start < 0 < rel_end:
                        boundary_crossers.append(str(row["response_id"]))
                    answer_offsets.append([max(0, rel_start), min(len(response), max(0, rel_end))])
                if not answer_offsets or len(answer_offsets) != full_len - prefix_len:
                    raise ValueError(f"empty/misaligned native answer tokens: {row['response_id']}")
                first_scored_char = min(start for start, end in answer_offsets if end > start)
                if first_scored_char:
                    unscored_prefix_chars[str(row["response_id"])] = first_scored_char

                raw_retrieval_start = prompt.index(retrieved)
                stripped_prompt = prompt.strip()
                left_trim = len(prompt) - len(prompt.lstrip())
                if raw_retrieval_start < left_trim:
                    raise ValueError(f"retrieval removed by template stripping: {row['response_id']}")
                retrieval_start_in_stripped = raw_retrieval_start - left_trim
                template_user_start = rendered.index(stripped_prompt)
                retrieval_full_start = template_user_start + retrieval_start_in_stripped
                retrieval_full_end = retrieval_full_start + len(retrieved)
                ref_positions = [
                    i
                    for i, (start, end) in enumerate(full_enc["offset_mapping"][:prefix_len])
                    if max(int(start), retrieval_full_start) < min(int(end), retrieval_full_end)
                ]
                if len(ref_positions) < 10:
                    raise ValueError(
                        f"paper top-10% ECS undefined: response {row['response_id']} has only {len(ref_positions)} retrieved tokens"
                    )

                payload = {
                    "version": "redeep-formal-feature-input-v1",
                    "partition": partition,
                    "response_id": str(row["response_id"]),
                    "answer_id": str(row["response_id"]),
                    "source_id": str(row["source_id"]),
                    "group_id": str(row["group_id"]),
                    "generator_model": str(row["model"]),
                    "official_split": "train",
                    "released_prompt": row["released_prompt"],
                    "retrieved_passages": retrieved,
                    "original_response": response,
                    "prompt_sha256": row["prompt_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "official_rendered_prompt_sha256": sha256_text(rendered),
                    "official_prefix_token_count": prefix_len,
                    "official_full_token_count": full_len,
                    "paper_reference_token_positions": ref_positions,
                    "native_response_token_char_intervals": answer_offsets,
                    "prefix_is_exact_full_token_prefix": prefix_enc["input_ids"] == full_enc["input_ids"][:prefix_len],
                    "native_response_unscored_prefix_chars": first_scored_char,
                    "labels_used": False,
                    "exact_original_generation_trace": False,
                    "replay_definition": "full-FP16 Llama-2-7b-chat teacher-forced replay under the released ReDeEP prompt/tokenization contract",
                }
                output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                full_lengths.append(full_len)
                prefix_lengths.append(prefix_len)
                response_lengths.append(full_len - prefix_len)
                reference_counts.append(len(ref_positions))

    if boundary_crossers:
        raise ValueError(f"token crossed prompt/response character boundary for {len(boundary_crossers)} answers")
    audit["feature_inputs"] = {
        "path": str(feature_path.relative_to(PROJECT)),
        "sha256": sha256_file(feature_path),
        "rows": len(full_lengths),
        "full_tokens": {
            "sum": sum(full_lengths),
            "min": min(full_lengths),
            "p50": percentile(full_lengths, 0.50),
            "p95": percentile(full_lengths, 0.95),
            "max": max(full_lengths),
        },
        "prefix_tokens": {"min": min(prefix_lengths), "max": max(prefix_lengths)},
        "native_response_tokens": {"sum": sum(response_lengths), "min": min(response_lengths), "max": max(response_lengths)},
        "paper_reference_tokens": {"min": min(reference_counts), "max": max(reference_counts)},
        "prefix_is_exact_full_token_prefix": len(prefix_identity_failures) == 0,
        "prefix_identity_failure_count": len(prefix_identity_failures),
        "prefix_identity_failure_response_ids": prefix_identity_failures,
        "native_unscored_prefix_character_rows": unscored_prefix_chars,
        "prompt_response_boundary_crossing_tokens": 0,
        "double_bos_all_rows": True,
    }
    audit["tokenizer"] = {
        "path": str(MODEL),
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
        "bos_token_id": int(tokenizer.bos_token_id),
        "tokenizer_json_sha256": sha256_file(MODEL / "tokenizer.json"),
        "tokenizer_model_sha256": sha256_file(MODEL / "tokenizer.model"),
    }
    audit_path = out_dir / "DATA_JOIN_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"feature_inputs": str(feature_path), "audit": str(audit_path), "rows": len(full_lengths)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.output_dir.resolve())


if __name__ == "__main__":
    main()
