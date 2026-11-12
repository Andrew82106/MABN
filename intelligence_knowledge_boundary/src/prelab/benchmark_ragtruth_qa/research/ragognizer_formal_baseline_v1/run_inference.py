"""Guarded route-A runner for the frozen full RAGognizer inference plan.

The default target is all 3,839 fit+calibration answers.  --limit is only a
smoke/debug prefix and --resume validates the existing per-row hash chain
before appending.  The author model, checkpoint, head, sigmoid probabilities,
BF16 request, and postprocessor setting are unchanged; Accelerate changes only
device placement.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from adapter import (
    PLAN_VERSION,
    RAW_VERSION,
    ZERO_CHAIN,
    add_chain_fields,
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_chained_row,
    validate_plan_row,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = HERE.parents[3]
PLAN = HERE / "INFERENCE_PLAN.jsonl"
MODEL_MANIFEST = HERE / "MODEL_MANIFEST.json"
SOURCE_AUDIT = HERE / "SOURCE_AUDIT.json"
RUNTIME_ALIAS_AUDIT = HERE / "RUNTIME_ALIAS_AUDIT.json"
RUNTIME_CHECKPOINT = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-offline-runtime"
RUNTIME_BASE = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf-base-f5db-runtime"
OFFICIAL_MODEL = WORKSPACE / "prelab" / "models" / "RAGognizer-Llama-2-7b-chat-hf"
RAGOGNIZER_REPO = PROJECT / "third_party" / "RAGognizer"
TRANSFORMER_HEADS_REPO = PROJECT / "third_party" / "transformer-heads"
DEFAULT_OUTPUT = HERE / "NATIVE_PROBABILITIES.jsonl"
DEFAULT_AUDIT = HERE / "GPU_RUN_AUDIT.json"

# Filled only after the deterministic full CPU plan is regenerated and audited.
PLAN_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
EXPECTED_ROWS = 3_839
EXPECTED_PARTITIONS = {"fit": 3_680, "calibration": 159}
RAGOGNIZER_COMMIT = "1d3e8fdd6de638dc2d06265829a4d9ced387e0ba"
TRANSFORMER_HEADS_COMMIT = "6a2ca2a12a25035ea290b0b0e03839fc16348e44"
CHECKPOINT_REVISION = "2b58ab9aa73f6d499ec72a38ced0976caf78b26f"
AUTHORIZATION_TOKEN = "REDEEP_GPU_RELEASED"
RUNTIME_HASHES = {
    "base/model-00001-of-00002.safetensors": "66dec18c9f1705b9387d62f8485f4e7d871ca388718786737ed3c72dbfaac9fb",
    "base/model-00002-of-00002.safetensors": "0fd6895090da1b2ccffdb93964847709a3b31e6b69fe7dc5a480dce37c811b1d",
    "base/model.safetensors.index.json": "11b694a9d4bffdac71733db7f782be0df1f87a65fb9ef05bcf6d317029c22364",
    "base/config.json": "c904bd84ff892a2a0a454a35a4031914ccb65a82974ac8ab0fe34d3744656fc2",
    "base/generation_config.json": "4eb769d945b7c9a9d9b5ee0cff40ac23a62e218ac683f87ac68d8ad9cd18dd7f",
    "base/special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "base/tokenizer_config.json": "04aaa6eedd97412f3e63dd2493d2388b82347dd512b89192d3e9b4973a176417",
    "base/tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    "base/tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "checkpoint/ft_llm/adapter_model.safetensors": "63dbd3cdb009f2e71cda0d95c92998818b4d4d943cdb64c6ff8f85ca6c3e1abc",
    "checkpoint/ft_llm/hallu_head_neg_16.safetensors": "27917e89aee9510979d7dc77fcb668efcb83accefd6067d91ea081d1a4be4ab8",
    "checkpoint/ft_llm/head_configs.json": "0fb22c85c82f3691779888e641cf7e4269885ef0ae129a6dc883dcf7b3a14cf6",
}
SOURCE_HASHES = {
    "ragognizer/ragognizer/ragognizer/detectors/RAGognizer.py": "6a2e913f8a5b1de5239e2f55f8793c25d4d793c59e202e2b70f610b26458c7b5",
    "ragognizer/ragognizer/ragognizer/detectors/detector.py": "eee8d847db64680a4bc2e9197ad022858f5e19c5634181e2b43d1022ce7cea8c",
    "transformer_heads/transformer_heads/util/load_model.py": "5e7873a33827b8801bf7560f49883a9ca15bb5373bf901427fbbccc867a9c3ad",
    "transformer_heads/transformer_heads/util/helpers.py": "32bd213c66a350828adaae31484f1ab185a50d35018e2266131daced4e107679",
    "transformer_heads/transformer_heads/util/prepare_model.py": "1d95ff7d62c341131b189ee662db24f2022d783aff298bfa4e5587f163806d4c",
    "transformer_heads/transformer_heads/model/model.py": "7cd0a155bc0fc1b0d1d3fd0e0f372b720f0eee7669200ecf147e29487bfacf84",
    "transformer_heads/transformer_heads/model/head.py": "a24576251e115bd18652200c07636256086dedcecc88626fda89d37a68e66243",
    "transformer_heads/transformer_heads/config.py": "e253380ee53926a1f3180b915f374412bc7f196d6d7c332cdf866ba13966e867",
    "transformer_heads/transformer_heads/constants.py": "371af802784fb78ca282e61cc67af215d28bc1a717084fd05d7450e0b257e25c",
}


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean_git_tree(path: Path, name: str) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"{name} source tree is dirty; frozen commit identity is insufficient")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required package is missing: {name}") from exc


def validate_runtime_files() -> None:
    roots = {"base": RUNTIME_BASE, "checkpoint": RUNTIME_CHECKPOINT}
    for relative, expected in RUNTIME_HASHES.items():
        root_name, remainder = relative.split("/", 1)
        actual = sha256_file(roots[root_name] / remainder)
        if actual != expected:
            raise RuntimeError(f"runtime file hash changed: {relative}")
    if sha256_file(OFFICIAL_MODEL / "ft_llm" / "adapter_config.json") != (
        "4564e39c2aa87355b6a941038478580caec0a489facb7a8a4095f9a17dcf94d4"
    ):
        raise RuntimeError("official adapter_config hash changed")
    if sha256_file(OFFICIAL_MODEL / "mlp_other_data.json") != (
        "ffa816bd439bd1dc96462458910d99aae1f0426ac384cf9b13b7d464dde2d5ec"
    ):
        raise RuntimeError("official native-threshold metadata hash changed")
    original_config = json.loads(
        (OFFICIAL_MODEL / "ft_llm" / "adapter_config.json").read_text(encoding="utf-8")
    )
    runtime_config = json.loads(
        (RUNTIME_CHECKPOINT / "ft_llm" / "adapter_config.json").read_text(encoding="utf-8")
    )
    expected_runtime = dict(original_config)
    expected_runtime["base_model_name_or_path"] = str(RUNTIME_BASE.resolve())
    if runtime_config != expected_runtime:
        raise RuntimeError("runtime adapter_config changed beyond local base path binding")


def validate_source_files() -> None:
    roots = {"ragognizer": RAGOGNIZER_REPO, "transformer_heads": TRANSFORMER_HEADS_REPO}
    for relative, expected in SOURCE_HASHES.items():
        root_name, remainder = relative.split("/", 1)
        if sha256_file(roots[root_name] / remainder) != expected:
            raise RuntimeError(f"audited source file changed: {relative}")


def external_response_coordinates(tokenizer, visible: str, response: str) -> dict:
    """Recompute the frozen label-free coordinate fallback from fast offsets."""
    encoded = tokenizer(
        visible,
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    input_ids = list(map(int, encoded["input_ids"]))
    offsets = [list(map(int, pair)) for pair in encoded["offset_mapping"]]
    response_start = visible.rindex(response)
    response_end = response_start + len(response)
    token_texts = tokenizer.batch_decode(
        [[token_id] for token_id in input_ids], skip_special_tokens=True
    )
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > response_start
        and start < response_end
        and end > start
        and token_texts[index]
    ]
    raw_intervals = [
        [
            max(0, offsets[index][0] - response_start),
            min(len(response), offsets[index][1] - response_start),
        ]
        for index in positions
    ]
    intervals: list[list[int]] = []
    cursor = 0
    for raw_start, raw_end in raw_intervals:
        intervals.append([cursor if raw_start >= cursor else raw_start, raw_end])
        cursor = max(cursor, raw_end)
    if cursor < len(response):
        intervals[-1][1] = len(response)
    byte_fallback_local_indices = [
        index
        for index, probability_index in enumerate(positions)
        if "\ufffd" in tokenizer.decode(
            [input_ids[probability_index]], skip_special_tokens=True
        )
    ]
    covered = [False] * len(response)
    for start, end in intervals:
        covered[start:end] = [True] * (end - start)
    if not positions or not all(covered):
        raise RuntimeError("external author coordinate fallback is incomplete")
    return {
        "model_input_ids": input_ids,
        "probability_indices": positions,
        "token_ids": [input_ids[index] for index in positions],
        "char_intervals": intervals,
        "coordinate_texts": [response[start:end] for start, end in intervals],
        "byte_fallback_local_indices": byte_fallback_local_indices,
        "byte_fallback_probability_indices": [
            positions[index] for index in byte_fallback_local_indices
        ],
    }


def load_plan() -> list[dict]:
    if PLAN_SHA256 == "__FULL_PLAN_SHA256__":
        raise RuntimeError("runner is not frozen: full plan SHA256 placeholder remains")
    if sha256_file(PLAN) != PLAN_SHA256:
        raise RuntimeError("full inference plan hash changed")
    rows = list(read_jsonl(PLAN))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"full plan must contain {EXPECTED_ROWS} answers")
    counts = Counter(str(row.get("partition")) for row in rows)
    if dict(counts) != EXPECTED_PARTITIONS:
        raise RuntimeError(f"full plan partitions changed: {dict(counts)}")
    for index, row in enumerate(rows):
        validate_plan_row(row, f"plan[{index}]")
        if row.get("official_split") != "train":
            raise RuntimeError("non-train official split is forbidden")
    return rows


def validate_existing_raw(
    rows: list[dict], plan_rows: list[dict], run_identity_sha256: str
) -> str:
    if len(rows) > len(plan_rows):
        raise RuntimeError("resume output has more rows than target")
    previous = ZERO_CHAIN
    for index, (raw, plan) in enumerate(zip(rows, plan_rows)):
        if raw.get("version") != RAW_VERSION:
            raise RuntimeError(f"resume raw version mismatch at row {index}")
        if raw.get("answer_id") != plan.get("answer_id"):
            raise RuntimeError(f"resume order mismatch at row {index}")
        if raw.get("partition") != plan.get("partition"):
            raise RuntimeError(f"resume partition mismatch at row {index}")
        if raw.get("answer_sha256") != plan.get("answer_sha256"):
            raise RuntimeError(f"resume answer hash mismatch at row {index}")
        if raw.get("plan_sha256") != PLAN_SHA256:
            raise RuntimeError(f"resume plan file hash mismatch at row {index}")
        if raw.get("plan_row_sha256") != plan.get("plan_row_sha256"):
            raise RuntimeError(f"resume plan row hash mismatch at row {index}")
        if raw.get("run_identity_sha256") != run_identity_sha256:
            raise RuntimeError(f"resume run identity mismatch at row {index}")
        previous = validate_chained_row(raw, previous, "raw_row_sha256", f"raw[{index}]")
    return previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-authorization", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="ordered selected-plan prefix size for smoke/debug",
    )
    parser.add_argument(
        "--partition",
        choices=("calibration",),
        help="calibration-only provisional run; omit for the formal full plan",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invocation_started = time.perf_counter()
    if args.gpu_authorization != AUTHORIZATION_TOKEN:
        raise RuntimeError(
            "GPU execution remains locked. Run only after the GPU owner explicitly releases it, "
            f"then pass --gpu-authorization {AUTHORIZATION_TOKEN}."
        )

    plan_rows_all = load_plan()
    selected_rows = (
        [row for row in plan_rows_all if row["partition"] == args.partition]
        if args.partition
        else plan_rows_all
    )
    target_rows = len(selected_rows) if args.limit is None else args.limit
    if not 1 <= target_rows <= len(selected_rows):
        raise ValueError(f"--limit must be in [1, {len(selected_rows)}]")
    plan_rows = selected_rows[:target_rows]
    if args.partition == "calibration" and args.limit is None:
        run_scope = "provisional_calibration"
    elif args.partition is None and args.limit is None:
        run_scope = "formal_full"
    else:
        run_scope = "smoke"
    output_path = args.output.resolve()
    audit_path = args.audit.resolve()
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError("--resume requires an existing raw output")
        if audit_path.exists():
            raise FileExistsError("resume audit already exists; use a new target or validate it")
    elif (output_path.exists() or audit_path.exists()) and not args.overwrite:
        raise FileExistsError("output/audit already exists; pass --overwrite deliberately")
    elif args.overwrite:
        # --overwrite is the explicit authority to replace both members of the
        # output/audit pair.  Removing the old audit first prevents a failed new
        # run from being mistaken for a completed run over the new raw file.
        if audit_path.exists():
            audit_path.unlink()
        if output_path.exists():
            output_path.unlink()

    if git_head(RAGOGNIZER_REPO) != RAGOGNIZER_COMMIT:
        raise RuntimeError("RAGognizer source commit changed")
    if git_head(TRANSFORMER_HEADS_REPO) != TRANSFORMER_HEADS_COMMIT:
        raise RuntimeError("transformer-heads source commit changed")
    require_clean_git_tree(RAGOGNIZER_REPO, "RAGognizer")
    require_clean_git_tree(TRANSFORMER_HEADS_REPO, "transformer-heads")
    validate_source_files()
    validate_runtime_files()
    versions = {
        name: package_version(name)
        for name in ("torch", "transformers", "tokenizers", "peft", "accelerate", "safetensors")
    }
    if versions["transformers"] != "4.57.1":
        raise RuntimeError("author inference package requires transformers==4.57.1")
    if versions["peft"] != "0.18.1":
        raise RuntimeError("frozen local runtime requires peft==0.18.1")
    if versions["accelerate"] != "1.13.0":
        raise RuntimeError("frozen local offload runtime requires accelerate==1.13.0")

    run_identity = {
        "route": "model_card_default_transformer_heads_with_accelerate_cpu_offload",
        "device_placement_change_only": True,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "ragognizer_commit": RAGOGNIZER_COMMIT,
        "transformer_heads_commit": TRANSFORMER_HEADS_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "runner_script_sha256": sha256_file(Path(__file__).resolve()),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT),
        "runtime_alias_audit_sha256": sha256_file(RUNTIME_ALIAS_AUDIT),
        "torch_dtype_argument": "torch.bfloat16",
        "quantization": None,
        "postprocessor": False,
        "batch_size": 1,
        "author_probability": "sigmoid(integrated hallu_head_neg_16 logit)",
        "python": sys.version,
        "versions": versions,
    }
    run_identity_sha256 = sha256_json(run_identity)
    existing_rows = list(read_jsonl(output_path)) if args.resume else []
    previous_chain = validate_existing_raw(existing_rows, plan_rows, run_identity_sha256)
    completed_before_resume = len(existing_rows)
    if completed_before_resume >= target_rows:
        raise RuntimeError("resume target already complete; preserve and validate its existing audit")

    # Heavy imports and every CUDA call remain behind the explicit authorization gate.
    model_load_started = time.perf_counter()
    sys.path.insert(0, str(TRANSFORMER_HEADS_REPO))
    sys.path.insert(0, str(RAGOGNIZER_REPO / "ragognizer"))
    import torch
    from accelerate import cpu_offload
    from transformers import AutoTokenizer
    from transformer_heads import load_lora_with_heads
    from transformer_heads.util.helpers import get_model_params
    from ragognizer.detectors.RAGognizer import RAGognizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(RUNTIME_BASE, local_files_only=True, use_fast=True)
    checkpoint_ft = RUNTIME_CHECKPOINT / "ft_llm"
    model_params = get_model_params(str(RUNTIME_BASE))
    model = load_lora_with_heads(
        model_params["model_class"],
        str(checkpoint_ft),
        new_emb_size=len(tokenizer) + 1,
        quantization_config=None,
        only_inference=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    model.requires_grad_(False)
    model.eval()
    dtype_before_offload = Counter(str(parameter.dtype) for parameter in model.parameters())
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("quantized model is forbidden for the formal baseline")
    cpu_offload(model, execution_device=device, offload_buffers=True)
    model_ready = time.perf_counter()
    packer = RAGognizer.__new__(RAGognizer)
    packer.tokenizer = tokenizer
    packer.binarization_threshold = 0.6523

    output_path.parent.mkdir(parents=True, exist_ok=True)
    peak_allocated = 0
    peak_reserved = 0
    mode = "a" if args.resume else "w"
    partition_counts = Counter(str(row["partition"]) for row in existing_rows)
    inference_started = time.perf_counter()
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        for plan_index in range(completed_before_resume, target_rows):
            plan = plan_rows[plan_index]
            chat = plan["chat"]
            template_ids = list(
                map(
                    int,
                    tokenizer.apply_chat_template(
                        chat, add_generation_prompt=False, tokenize=True
                    ),
                )
            )
            visible = tokenizer.decode(template_ids, skip_special_tokens=False)
            model_input_ids = list(
                map(
                    int,
                    tokenizer(
                        visible,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                    )["input_ids"],
                )
            )
            if model_input_ids != plan["author_roundtrip"]["model_input_ids"]:
                raise RuntimeError(f"author tokenization changed for {plan['answer_id']}")

            input_ids = torch.tensor([model_input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                matching = [key for key in outputs.preds_by_head if "hallu" in key]
                if matching != ["hallu_head_neg_16"]:
                    raise RuntimeError(f"unexpected hallucination heads: {matching}")
                full_probabilities_tensor = (
                    torch.sigmoid(outputs.preds_by_head[matching[0]].flatten())
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                full_probabilities = full_probabilities_tensor.numpy()

            expected_tokens = plan["author_response_tokens"]
            external = external_response_coordinates(
                tokenizer, visible, plan["original_response"]
            )
            if external["model_input_ids"] != model_input_ids:
                raise RuntimeError(f"fast-offset tokenization changed for {plan['answer_id']}")
            probability_indices = expected_tokens["probability_indices"]
            token_ids = expected_tokens["token_ids"]
            if external["probability_indices"] != probability_indices:
                raise RuntimeError(f"frozen readable probability positions changed for {plan['answer_id']}")
            if external["token_ids"] != token_ids:
                raise RuntimeError(f"frozen readable token IDs changed for {plan['answer_id']}")
            if [model_input_ids[index] for index in probability_indices] != token_ids:
                raise RuntimeError(f"frozen probability indices changed for {plan['answer_id']}")
            native_probabilities = [float(full_probabilities[index]) for index in probability_indices]
            if not all(0.0 <= value <= 1.0 for value in native_probabilities):
                raise RuntimeError(f"invalid probability for {plan['answer_id']}")

            if expected_tokens["official_pack_exact"]:
                packed = packer._pack_probs(
                    probs=full_probabilities,
                    input_ids=torch.tensor([model_input_ids], dtype=torch.long),
                    original_chat=chat,
                    return_assistant_only=True,
                    only_readable_tokens=True,
                )
                if [item["text"] for item in packed] != expected_tokens["coordinate_texts"]:
                    raise RuntimeError(f"author packed token text changed for {plan['answer_id']}")
                if [[item["start"], item["end"]] for item in packed] != expected_tokens["char_intervals"]:
                    raise RuntimeError(f"author packed token spans changed for {plan['answer_id']}")
                if native_probabilities != [float(item["prob"]) for item in packed]:
                    raise RuntimeError(f"author packing changed probability order for {plan['answer_id']}")
                appendix_predictions = [int(item["pred"]) for item in packed]
            else:
                if expected_tokens["coordinate_mode"] != "fast_offsets_author_pack_byte_fallback":
                    raise RuntimeError(f"unknown coordinate fallback for {plan['answer_id']}")
                for key in (
                    "char_intervals",
                    "coordinate_texts",
                    "byte_fallback_local_indices",
                    "byte_fallback_probability_indices",
                ):
                    if external[key] != expected_tokens[key]:
                        raise RuntimeError(f"byte-fallback {key} changed for {plan['answer_id']}")
                appendix_predictions = [
                    int(probability >= 0.6523) for probability in native_probabilities
                ]

            core = {
                "version": RAW_VERSION,
                "route": "model_card_default_transformer_heads",
                "partition": str(plan["partition"]),
                "answer_id": str(plan["answer_id"]),
                "answer_sha256": str(plan["answer_sha256"]),
                "plan_sha256": PLAN_SHA256,
                "plan_row_sha256": str(plan["plan_row_sha256"]),
                "run_identity_sha256": run_identity_sha256,
                "author_response_token_ids": token_ids,
                "author_response_token_char_intervals": expected_tokens["char_intervals"],
                "author_response_token_probabilities": native_probabilities,
                "author_native_preds_0_6523_appendix_only": [
                    int(value) for value in appendix_predictions
                ],
                "author_coordinate_mode": expected_tokens["coordinate_mode"],
                "author_public_packer_exact": expected_tokens["official_pack_exact"],
                "postprocessor": False,
                "quantization": None,
                "threshold_applied": False,
            }
            payload = add_chain_fields(core, previous_chain, "raw_row_sha256")
            previous_chain = payload["chain_sha256"]
            output.write(canonical_json(payload) + "\n")
            output.flush()
            partition_counts[str(plan["partition"])] += 1

            peak_allocated = max(
                peak_allocated, int(torch.cuda.max_memory_allocated(device))
            )
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(device))
            )
            del outputs, full_probabilities, full_probabilities_tensor
            del external, input_ids, attention_mask
            if "packed" in locals():
                del packed
            torch.cuda.empty_cache()
            print(
                canonical_json(
                    {
                        "completed": plan_index + 1,
                        "target": target_rows,
                        "answer_id": plan["answer_id"],
                        "partition": plan["partition"],
                    }
                )
            )

    expected_target_counts = Counter(str(row["partition"]) for row in plan_rows)
    if partition_counts != expected_target_counts:
        raise RuntimeError("written partition coverage differs from target plan prefix")
    inference_finished = time.perf_counter()
    invocation_rows = target_rows - completed_before_resume
    audit = {
        "version": "ragognizer-route-a-gpu-run-audit-v2",
        "status": {
            "formal_full": "complete",
            "provisional_calibration": "provisional_complete",
            "smoke": "smoke_complete",
        }[run_scope],
        "run_identity": run_identity,
        **run_identity,
        "run_identity_sha256": run_identity_sha256,
        "runner_script_sha256": sha256_file(Path(__file__).resolve()),
        "plan_version": PLAN_VERSION,
        "rows": target_rows,
        "required_full_rows": EXPECTED_ROWS,
        "run_scope": run_scope,
        "selection": {
            "partition": args.partition,
            "limit": args.limit,
            "answer_ids_sha256": sha256_json(
                [str(row["answer_id"]) for row in plan_rows]
            ),
        },
        "partition_rows": dict(sorted(partition_counts.items())),
        "completed_before_resume": completed_before_resume,
        "rows_computed_this_invocation": invocation_rows,
        "resume_used": args.resume,
        "output_sha256": sha256_file(output_path),
        "raw_final_chain_sha256": previous_chain,
        "loaded_parameter_dtype_tensor_counts": dict(sorted(dtype_before_offload.items())),
        "torch_no_grad": True,
        "tf32_matmul_runtime_default": bool(torch.backends.cuda.matmul.allow_tf32),
        "versions": versions,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "timing_seconds": {
            "model_load_and_offload": model_ready - model_load_started,
            "inference_this_invocation": inference_finished - inference_started,
            "per_answer_this_invocation": (
                (inference_finished - inference_started) / invocation_rows
            ),
            "total_invocation": inference_finished - invocation_started,
        },
        "author_threshold_applied": False,
        "shared_threshold_applied": False,
        "labels_read": False,
        "test_opened": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
