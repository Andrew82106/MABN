"""Label-free diagnosis of cached generation versus no-cache replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_forced_evidence_quote_probe_v1_gpu as runner


TARGET = runner.OUT / "CACHE_REPLAY_DIAGNOSIS_SMOKE1.json"


def stats(left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p50_abs": float(np.percentile(delta, 50)),
        "p95_abs": float(np.percentile(delta, 95)),
        "max_index": int(delta.argmax()),
        "cached_at_max": float(left[delta.argmax()]),
        "replay_at_max": float(right[delta.argmax()]),
    }


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    rows, _manifest = runner.load_inputs()
    plan = runner.load_plan_and_check()
    tokenizer = runner.tokenizer_load()
    pca = runner.load_pca()
    chosen, lengths = runner.fixed_smoke_indices(rows, tokenizer, plan)
    index = chosen[0]
    row = rows[index]
    _prompt, prompt_ids, positions = runner.prompt_ids_and_regions(
        tokenizer, plan, row)
    model = None
    with runner.exclusive_gpu("cache-replay-diagnosis-smoke1") as lease:
        try:
            model, model_meta = runner.load_nf4()
            cached, generation = runner.cached_greedy_generate(
                model, tokenizer, prompt_ids, plan)
            decoded, offsets = runner.generated_token_offsets(
                tokenizer, cached["generated_token_ids"],
                generation["terminal_eos_id"])
            passages = [row[f"passage_{number}"] for number in (1, 2, 3)]
            parsed = runner.cpu.parse_generated_quote(
                decoded, generation["stop_reason"], passages)
            content = runner.generated_content_indices(parsed, offsets)
            query_positions = [len(prompt_ids) + value for value in content]
            full_ids = prompt_ids + cached[
                "generated_token_ids"].astype(int).tolist()
            passage_positions = [positions[f"passage_{number}"]
                                 for number in (1, 2, 3)]
            final, hidden64, raw_attention, attention = (
                runner.no_cache_quote_forward(
                    model, full_ids, query_positions, passage_positions,
                    positions["claim"], pca))
            generated_positions = np.arange(
                len(prompt_ids), len(full_ids), dtype=np.int64)
            replay = runner.statistics_from_hidden(
                model, final, generated_positions,
                cached["generated_token_ids"].astype(np.int64))
            del final
            mismatch = np.flatnonzero(
                replay["argmax_token_ids"] != cached["generated_token_ids"])
            result = {
                "status": "label_free_cache_replay_diagnosis_after_failed_smoke",
                "runner_sha256": runner.sha(Path(runner.__file__)),
                "record_index": index,
                "microclaim_id": row["microclaim_id"],
                "prompt_tokens": lengths[index],
                "generated_tokens": len(cached["generated_token_ids"]),
                "content_tokens": len(content),
                "decoded": decoded,
                "stop_reason": generation["stop_reason"],
                "parse": parsed,
                "argmax_mismatch_count": int(len(mismatch)),
                "argmax_mismatch_indices": mismatch.astype(int).tolist(),
                "differences": {
                    name: stats(cached[name], replay[name])
                    for name in ("selected_logprob", "vocab_entropy",
                                 "signed_margin")
                },
                "cached": {
                    name: cached[name].astype(float).tolist()
                    for name in ("generated_token_ids", "selected_logprob",
                                 "vocab_entropy", "signed_margin")
                },
                "replay": {
                    name: replay[name].astype(float).tolist()
                    for name in ("argmax_token_ids", "selected_logprob",
                                 "vocab_entropy", "signed_margin")
                },
                "hidden64_shape": list(hidden64.shape),
                "attention_shape": list(attention.shape),
                "raw_attention_shapes": {
                    key: list(value.shape) for key, value in raw_attention.items()
                },
                "lease": lease, "model": model_meta,
                "GPU_used": True, "gold_read": False,
                "calibration_or_test_read": False, "scoring_run": False,
            }
            runner.atomic_json(TARGET, result)
        finally:
            if model is not None:
                del model
            runner.clean_gpu()
    print(json.dumps({
        "target": str(TARGET.resolve()),
        "argmax_mismatch_count": result["argmax_mismatch_count"],
        "differences": result["differences"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
