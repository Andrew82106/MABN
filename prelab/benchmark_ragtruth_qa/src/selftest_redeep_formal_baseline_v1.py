"""CPU-only numerical and geometry self-test for the ReDeEP migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from redeep_formal_core_v1 import (
    answer_max_window,
    code_reverse_mixture_divergence,
    ecs_from_top_positions,
    f1_opt_threshold,
    map_native_tokens_to_windows,
    paper_standard_js_divergence,
)
from run_redeep_formal_baseline_v1 import (
    ReDeEPActivationCapture,
    project_pks,
    rms_norm_like_llama,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "redeep_formal_baseline_v1" / "CPU_SELFTEST.json"


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def direct_reference(model, input_ids, prefix_len, paper_positions, candidate_heads, captured_mid):
    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    final_hidden = output.hidden_states[-1][0]
    predictors = torch.arange(prefix_len - 1, input_ids.shape[1] - 1)
    code_ecs = torch.empty((predictors.numel(), len(candidate_heads)), dtype=torch.float32)
    paper_ecs = torch.empty_like(code_ecs)
    for feature_idx, (layer, head) in enumerate(candidate_heads):
        attn = output.attentions[layer][0, head].index_select(0, predictors)
        code_allowed = torch.arange(prefix_len)
        paper_allowed = torch.tensor(paper_positions)
        code_k = int(prefix_len * 0.1)
        paper_k = int(len(paper_positions) * 0.1)
        code_local = torch.argsort(attn.index_select(-1, code_allowed), dim=-1, descending=True)[:, :code_k]
        paper_local = torch.argsort(attn.index_select(-1, paper_allowed), dim=-1, descending=True)[:, :paper_k]
        code_top = code_allowed[code_local]
        paper_top = paper_allowed[paper_local]
        code_ecs[:, feature_idx] = ecs_from_top_positions(final_hidden, predictors, code_top).float()
        paper_ecs[:, feature_idx] = ecs_from_top_positions(final_hidden, predictors, paper_top).float()

    layers = model.config.num_hidden_layers
    code_pks = torch.empty((predictors.numel(), layers), dtype=torch.float32)
    paper_pks = torch.empty_like(code_pks)
    weight = model.lm_head.weight.detach()
    norm_weight = model.model.norm.weight.detach()
    for layer in range(layers):
        before = captured_mid[layer][0].index_select(0, predictors)
        after = output.hidden_states[layer + 1][0].index_select(0, predictors)
        before_norm = rms_norm_like_llama(before, norm_weight, model.config.rms_norm_eps)
        after_norm = after if layer == layers - 1 else rms_norm_like_llama(after, norm_weight, model.config.rms_norm_eps)
        before_logits = F.linear(before_norm, weight)
        after_logits = F.linear(after_norm, weight)
        code_pks[:, layer] = code_reverse_mixture_divergence(after_logits, before_logits)
        paper_pks[:, layer] = paper_standard_js_divergence(after_logits, before_logits)
    return (
        code_ecs.numpy(),
        paper_ecs.numpy(),
        code_pks.numpy(),
        paper_pks.numpy(),
        final_hidden.numpy(),
        output,
    )


def main() -> None:
    torch.manual_seed(194)
    config = LlamaConfig(
        vocab_size=47,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 4, 8, 3, 9, 11, 7, 2, 12, 15, 18, 20, 6, 5, 31, 7, 14, 21, 9, 2]])
    prefix_len = 12
    paper_positions = list(range(1, 11))
    candidate_heads = [[0, 0], [1, 2], [2, 3]]

    reference_mid: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(index):
            def hook(_module, args, _kwargs):
                reference_mid[index] = args[0].detach().clone()
            return hook
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
    direct = direct_reference(model, input_ids, prefix_len, paper_positions, candidate_heads, reference_mid)
    for handle in handles:
        handle.remove()

    capture = ReDeEPActivationCapture(model, candidate_heads)
    capture.reset(prefix_len, input_ids.shape[1], paper_positions)
    with torch.inference_mode():
        streamed_output = model.model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    stream_ecs_code, stream_ecs_paper = capture.ecs(torch.device("cpu"))
    stream_pks_code, stream_pks_paper = project_pks(
        capture,
        model.lm_head.weight.detach(),
        model.model.norm.weight.detach(),
        model.config.rms_norm_eps,
        chunk_tokens=100,
    )
    capture.close()

    direct_ecs_code, direct_ecs_paper, direct_pks_code, direct_pks_paper, direct_hidden, _ = direct
    diffs = {
        "ecs_official_code_vs_dense_attention": max_abs(stream_ecs_code, direct_ecs_code),
        "ecs_paper_vs_dense_attention": max_abs(stream_ecs_paper, direct_ecs_paper),
        "pks_official_code_vs_dense_projection": max_abs(stream_pks_code, direct_pks_code),
        "pks_paper_vs_dense_projection": max_abs(stream_pks_paper, direct_pks_paper),
        "final_hidden_hook_vs_model_output": max_abs(capture.final_hidden_cpu.numpy(), streamed_output.last_hidden_state[0].numpy()),
        "separate_dense_runs_final_hidden": max_abs(direct_hidden, streamed_output.last_hidden_state[0].numpy()),
    }

    rng = torch.Generator().manual_seed(91)
    after_logits = torch.randn(5, 17, generator=rng)
    before_logits = torch.randn(5, 17, generator=rng)
    p = torch.softmax(after_logits, -1)
    q = torch.softmax(before_logits, -1)
    m = 0.5 * (p + q)
    official_direct = 0.5 * (
        F.kl_div(torch.log_softmax(after_logits, -1), m, reduction="none").mean(-1)
        + F.kl_div(torch.log_softmax(before_logits, -1), m, reduction="none").mean(-1)
    ) * 1_000_000.0
    paper_direct = 0.5 * (
        torch.sum(p * (torch.log(p) - torch.log(m)), -1)
        + torch.sum(q * (torch.log(q) - torch.log(m)), -1)
    )
    diffs["official_code_divergence_formula"] = max_abs(
        code_reverse_mixture_divergence(after_logits, before_logits).numpy(), official_direct.numpy()
    )
    diffs["paper_jsd_formula"] = max_abs(
        paper_standard_js_divergence(after_logits, before_logits).numpy(), paper_direct.numpy()
    )

    synthetic_windows = [
        {"window_id": "w0", "character_intervals": [[0, 5]], "label": 999},
        {"window_id": "w1", "character_intervals": [[4, 9]], "label": -999},
    ]
    mapped = map_native_tokens_to_windows(
        [1.0, 2.0, 3.0, 4.0], [[0, 2], [2, 5], [5, 8], [8, 10]], synthetic_windows
    )
    expected_mapped = np.asarray([1.5, 3.0])
    diffs["label_blind_character_overlap_mapping"] = max_abs(mapped, expected_mapped)
    if answer_max_window(mapped) != 3.0:
        raise AssertionError("answer=max(window) adapter failed")
    threshold_check = f1_opt_threshold([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.6])
    if threshold_check["threshold"] != 0.7 or threshold_check["f1"] != 1.0:
        raise AssertionError("F1 threshold rule failed")

    tolerance = 3e-5
    passed = all(value <= tolerance for value in diffs.values())
    if not passed:
        raise AssertionError(f"CPU self-test failed: {diffs}")
    payload = {
        "version": "redeep-formal-cpu-selftest-v1",
        "passed": True,
        "device": "cpu",
        "tiny_model": {
            "random_seed": 194,
            "layers": 3,
            "heads": 4,
            "hidden_size": 32,
            "vocab_size": 47,
            "sequence_tokens": 20,
            "prefix_tokens": 12,
        },
        "checks": diffs,
        "absolute_tolerance": tolerance,
        "gold_independence_check": "synthetic window labels were contradictory sentinel values and were not read by the mapper",
        "torch_version": torch.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
