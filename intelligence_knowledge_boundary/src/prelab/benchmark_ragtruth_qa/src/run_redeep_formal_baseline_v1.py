"""Extract untouched ReDeEP(Token) signals on the shared fit/cal inputs.

The script preserves full FP16 Llama-2-7B weights and the released token features.
It avoids retaining dense attention matrices by recomputing Q/K only in the 32
published candidate-head locations and retaining their top-10% prefix indices.
That is an engineering implementation of the same attention ranking, not a new
feature.  No label or gold-window file is opened here.

This entry point requires CUDA.  It was prepared and CPU-tested, but must not be
started while another project owns the shared GPU lock.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import socket
import time
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import cpu_offload
from transformers import AutoModelForCausalLM, AutoTokenizer

from redeep_formal_core_v1 import (
    code_reverse_mixture_divergence,
    ecs_from_top_positions,
    llama2_official_chat_prompt,
    paper_standard_js_divergence,
    top_fraction_indices,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT / "research" / "redeep_formal_baseline_v1"
INPUTS = OUT_ROOT / "feature_inputs.jsonl"
FEATURE_ROOT = OUT_ROOT / "raw_features"
MODEL_PATH = PROJECT.parent / "models" / "Llama-2-7b-chat-hf"
UPSTREAM = PROJECT / "third_party" / "ReDEeP-ICLR"
COPY_HEADS_PATH = UPSTREAM / "ReDeEP" / "log" / "test_llama2_7B" / "topk_heads.json"
GLOBAL_GPU_LOCK = PROJECT / "results" / ".exclusive_gpu_runner.lock"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_inputs(path: Path) -> Iterable[dict]:
    # Physical safety boundary: the extractor accepts this one prepared file only.
    if path.resolve() != INPUTS.resolve():
        raise ValueError(f"extractor input must be the frozen label-free file {INPUTS}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("partition") not in {"fit", "calibration"} or row.get("official_split") != "train":
                raise ValueError(f"forbidden split at feature input line {line_no}")
            if row.get("labels_used") is not False:
                raise ValueError(f"feature input line {line_no} is not label-free")
            yield row


@contextlib.contextmanager
def exclusive_gpu_lock(stage: str):
    token = {
        "lease_nonce": uuid.uuid4().hex,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "stage": stage,
        "created_unix": time.time(),
    }
    GLOBAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(GLOBAL_GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another GPU runner owns {GLOBAL_GPU_LOCK}; the lock is never auto-broken") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(token, handle)
        yield token
    finally:
        if GLOBAL_GPU_LOCK.exists():
            try:
                current = json.loads(GLOBAL_GPU_LOCK.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if current.get("lease_nonce") == token["lease_nonce"]:
                GLOBAL_GPU_LOCK.unlink()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rms_norm_like_llama(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(dtype)


class ReDeEPActivationCapture:
    """Read-only hooks for ReDeEP ECS/PKS sufficient statistics."""

    def __init__(self, model, candidate_heads: list[list[int]]):
        self.model = model
        self.candidate_heads = [tuple(map(int, x)) for x in sorted(candidate_heads)]
        self.feature_index = {head: i for i, head in enumerate(self.candidate_heads)}
        self.handles = []
        self.predictor_positions_cpu: torch.Tensor | None = None
        self.code_allowed_cpu: torch.Tensor | None = None
        self.paper_allowed_cpu: torch.Tensor | None = None
        self.mid: dict[int, torch.Tensor] = {}
        self.after: dict[int, torch.Tensor] = {}
        self.top_code: dict[int, torch.Tensor] = {}
        self.top_paper: dict[int, torch.Tensor] = {}
        self.final_hidden_cpu: torch.Tensor | None = None
        heads_by_layer: dict[int, list[int]] = {}
        for layer, head in self.candidate_heads:
            heads_by_layer.setdefault(layer, []).append(head)
        for layer_idx, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._mid_hook(layer_idx), with_kwargs=True
                )
            )
            self.handles.append(layer.register_forward_hook(self._after_hook(layer_idx), with_kwargs=True))
            if layer_idx in heads_by_layer:
                self.handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        self._attention_hook(layer_idx, heads_by_layer[layer_idx]), with_kwargs=True
                    )
                )
        self.handles.append(model.model.norm.register_forward_hook(self._final_norm_hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self, prefix_len: int, full_len: int, paper_positions: list[int]) -> None:
        if full_len <= prefix_len:
            raise ValueError("response has no native tokens")
        self.predictor_positions_cpu = torch.arange(prefix_len - 1, full_len - 1, dtype=torch.long)
        self.code_allowed_cpu = torch.arange(prefix_len, dtype=torch.long)
        self.paper_allowed_cpu = torch.tensor(paper_positions, dtype=torch.long)
        if self.paper_allowed_cpu.numel() < 10 or self.code_allowed_cpu.numel() < 10:
            raise ValueError("top-10% context set would be empty")
        self.mid.clear()
        self.after.clear()
        self.top_code.clear()
        self.top_paper.clear()
        self.final_hidden_cpu = None

    def _positions(self, device: torch.device) -> torch.Tensor:
        assert self.predictor_positions_cpu is not None
        return self.predictor_positions_cpu.to(device=device)

    def _mid_hook(self, layer_idx: int):
        def hook(_module, args, _kwargs):
            hidden = args[0]
            positions = self._positions(hidden.device)
            self.mid[layer_idx] = hidden[0].index_select(0, positions).detach().to("cpu", dtype=hidden.dtype)

        return hook

    def _after_hook(self, layer_idx: int):
        def hook(_module, _args, _kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = self._positions(hidden.device)
            self.after[layer_idx] = hidden[0].index_select(0, positions).detach().to("cpu", dtype=hidden.dtype)

        return hook

    def _attention_hook(self, layer_idx: int, selected_heads: list[int]):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden is None or position_embeddings is None:
                raise RuntimeError("LlamaAttention hook did not receive hidden_states/position_embeddings")
            batch, seq_len, _ = hidden.shape
            if batch != 1:
                raise ValueError("formal extractor is frozen at batch_size=1")
            query = module.q_proj(hidden).view(batch, seq_len, -1, module.head_dim).transpose(1, 2)
            key = module.k_proj(hidden).view(batch, seq_len, -1, module.head_dim).transpose(1, 2)
            query, key = apply_rope(query, key, *position_embeddings)
            if key.shape[1] != query.shape[1]:
                groups = query.shape[1] // key.shape[1]
                key = key[:, :, None, :, :].expand(batch, key.shape[1], groups, seq_len, module.head_dim)
                key = key.reshape(batch, query.shape[1], seq_len, module.head_dim)
            predictor_positions = self._positions(hidden.device)
            assert self.code_allowed_cpu is not None and self.paper_allowed_cpu is not None
            code_allowed = self.code_allowed_cpu.to(hidden.device)
            paper_allowed = self.paper_allowed_cpu.to(hidden.device)
            for head in selected_heads:
                q = query[0, head].index_select(0, predictor_positions)
                all_scores = torch.matmul(q, key[0, head].transpose(0, 1)) * float(module.scaling)
                feature_idx = self.feature_index[(layer_idx, head)]
                self.top_code[feature_idx] = top_fraction_indices(all_scores, code_allowed).detach().cpu()
                self.top_paper[feature_idx] = top_fraction_indices(all_scores, paper_allowed).detach().cpu()

        return hook

    def _final_norm_hook(self, _module, _args, _kwargs, output):
        positions = self._positions(output.device)
        self.final_hidden_cpu = output[0].detach().to("cpu", dtype=output.dtype)
        # The released custom model stores the final normalized hidden state as
        # hidden_states[num_layers], so layer 31's "after" LogitLens input is normalized.
        last = len(self.model.model.layers) - 1
        self.after[last] = output[0].index_select(0, positions).detach().to("cpu", dtype=output.dtype)

    def ecs(self, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
        if self.final_hidden_cpu is None:
            raise RuntimeError("final hidden state was not captured")
        final_hidden = self.final_hidden_cpu.to(device)
        predictors = self._positions(device)
        n = predictors.numel()
        code = torch.empty((n, len(self.candidate_heads)), dtype=torch.float32, device="cpu")
        paper = torch.empty_like(code)
        for feature_idx in range(len(self.candidate_heads)):
            code[:, feature_idx] = ecs_from_top_positions(
                final_hidden, predictors, self.top_code[feature_idx]
            ).float().cpu()
            paper[:, feature_idx] = ecs_from_top_positions(
                final_hidden, predictors, self.top_paper[feature_idx]
            ).float().cpu()
        return code.numpy(), paper.numpy()


def project_pks(
    capture: ReDeEPActivationCapture,
    lm_head_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    norm_eps: float,
    chunk_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    layers = len(capture.model.model.layers)
    if set(capture.mid) != set(range(layers)) or set(capture.after) != set(range(layers)):
        raise RuntimeError("incomplete before/after FFN capture")
    n = next(iter(capture.mid.values())).shape[0]
    code = np.empty((n, layers), dtype=np.float32)
    paper = np.empty((n, layers), dtype=np.float32)
    device = lm_head_weight.device
    for layer_idx in range(layers):
        for start in range(0, n, chunk_tokens):
            end = min(n, start + chunk_tokens)
            before = capture.mid[layer_idx][start:end].to(device)
            after = capture.after[layer_idx][start:end].to(device)
            before_norm = rms_norm_like_llama(before, final_norm_weight, norm_eps)
            # Official code avoids applying final RMSNorm twice at the last layer.
            after_norm = after if layer_idx == layers - 1 else rms_norm_like_llama(after, final_norm_weight, norm_eps)
            before_logits = F.linear(before_norm, lm_head_weight)
            after_logits = F.linear(after_norm, lm_head_weight)
            code[start:end, layer_idx] = code_reverse_mixture_divergence(after_logits, before_logits).cpu().numpy()
            paper[start:end, layer_idx] = paper_standard_js_divergence(after_logits, before_logits).cpu().numpy()
            del before, after, before_norm, after_norm, before_logits, after_logits
    return code, paper


def validate_existing(path: Path, row: dict, heads: list[list[int]]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            n = len(row["native_response_token_char_intervals"])
            return (
                data["ecs_code"].shape == (n, len(heads))
                and data["ecs_paper"].shape == (n, len(heads))
                and data["pks_code"].shape == (n, 32)
                and data["pks_paper"].shape == (n, 32)
                and str(data["answer_sha256"].item()) == row["answer_sha256"]
            )
    except Exception:
        return False


def save_features(path: Path, row: dict, heads: list[list[int]], arrays: tuple[np.ndarray, ...]) -> None:
    ecs_code, ecs_paper, pks_code, pks_paper = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary,
        version=np.asarray("redeep-formal-raw-feature-v1"),
        response_id=np.asarray(row["response_id"]),
        answer_sha256=np.asarray(row["answer_sha256"]),
        candidate_heads=np.asarray(heads, dtype=np.int16),
        predictor_positions=np.arange(
            int(row["official_prefix_token_count"]) - 1,
            int(row["official_full_token_count"]) - 1,
            dtype=np.int32,
        ),
        target_positions=np.arange(
            int(row["official_prefix_token_count"]),
            int(row["official_full_token_count"]),
            dtype=np.int32,
        ),
        token_char_intervals=np.asarray(row["native_response_token_char_intervals"], dtype=np.int32),
        ecs_code=ecs_code.astype(np.float32, copy=False),
        ecs_paper=ecs_paper.astype(np.float32, copy=False),
        pks_code=pks_code.astype(np.float32, copy=False),
        pks_paper=pks_paper.astype(np.float32, copy=False),
    )
    os.replace(temporary, path)


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal full-FP16 extraction requires CUDA")
    if sha256_file(INPUTS) != args.expected_inputs_sha256:
        raise RuntimeError("feature_inputs.jsonl changed after protocol freeze")
    if sha256_file(COPY_HEADS_PATH) != "633a37382dfe46242028cea3d4f28c6bb2102da1859572f6077a39a612228a4b":
        raise RuntimeError("published candidate-head file changed")
    heads = sorted(json.loads(COPY_HEADS_PATH.read_text(encoding="utf-8")))
    if len(heads) != 32 or len({tuple(x) for x in heads}) != 32:
        raise RuntimeError("expected 32 unique published candidate heads")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if model.config.num_hidden_layers != 32 or model.config.num_attention_heads != 32:
        raise RuntimeError("frozen method requires Llama-2-7B's 32 layers and 32 heads")
    # Keep one exact FP16 LogitLens projection on GPU; the transformer itself is
    # streamed one module at a time from CPU so the unquantized model fits 8 GiB.
    lm_head_weight = model.lm_head.weight.detach().clone().to(device=device, dtype=torch.float16)
    final_norm_weight = model.model.norm.weight.detach().clone().to(device=device, dtype=torch.float16)
    cpu_offload(model, execution_device=device, offload_buffers=True)
    capture = ReDeEPActivationCapture(model, heads)
    completed = skipped = 0
    try:
        with torch.inference_mode():
            for row_index, row in enumerate(iter_inputs(INPUTS)):
                if args.partition != "all" and row["partition"] != args.partition:
                    continue
                feature_path = FEATURE_ROOT / row["partition"] / f"{row['answer_id']}.npz"
                if args.resume and validate_existing(feature_path, row, heads):
                    skipped += 1
                    continue
                rendered = llama2_official_chat_prompt(row["released_prompt"])
                if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != row["official_rendered_prompt_sha256"]:
                    raise RuntimeError(f"rendered prompt drift: {row['response_id']}")
                encoded = tokenizer(rendered + row["original_response"], add_special_tokens=True, return_tensors="pt")
                if encoded.input_ids.shape[1] != int(row["official_full_token_count"]):
                    raise RuntimeError(f"tokenization drift: {row['response_id']}")
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
                    args.logit_chunk_tokens,
                )
                save_features(feature_path, row, heads, (ecs_code, ecs_paper, pks_code, pks_paper))
                completed += 1
                del outputs, encoded, ecs_code, ecs_paper, pks_code, pks_paper
                torch.cuda.empty_cache()
                if args.limit is not None and completed >= args.limit:
                    break
                if completed % 10 == 0:
                    print(json.dumps({"completed_this_run": completed, "resumed": skipped, "last": row["response_id"]}))
    finally:
        capture.close()
    run_meta = {
        "version": "redeep-formal-extraction-run-v1",
        "completed_this_run": completed,
        "resumed": skipped,
        "partition": args.partition,
        "limit": args.limit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
    }
    print(json.dumps(run_meta, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["fit", "calibration", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    # Every project answer has <=533 native tokens, so 1024 preserves one GEMM per
    # layer (closest to the released full-sequence projection) without a large VRAM
    # allocation.  Smaller values are an explicit emergency memory option only.
    parser.add_argument("--logit-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--expected-inputs-sha256",
        default="cd44ebf504d86088ba7b039a1d7263994780ded9ea7a22ba95636e25c4862cb8",
    )
    args = parser.parse_args()
    with exclusive_gpu_lock("redeep_formal_baseline_v1"):
        run(args)


if __name__ == "__main__":
    main()
