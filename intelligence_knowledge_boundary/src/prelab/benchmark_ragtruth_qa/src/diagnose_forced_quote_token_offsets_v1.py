"""Label-free one-claim diagnosis for non-canonical generated tokenization."""
from __future__ import annotations

import json
from pathlib import Path

import run_forced_evidence_quote_probe_v1_gpu as runner


TARGET = (runner.OUT / "OFFSET_DIAGNOSIS_SMOKE1.json")


def common_prefix(left: str, right: str) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    rows, _manifest = runner.load_inputs()
    plan = runner.load_plan_and_check()
    tokenizer = runner.tokenizer_load()
    chosen, lengths = runner.fixed_smoke_indices(rows, tokenizer, plan)
    index = chosen[0]
    row = rows[index]
    prompt, prompt_ids, _regions = runner.prompt_ids_and_regions(
        tokenizer, plan, row)
    model = None
    with runner.exclusive_gpu("offset-diagnosis-smoke1") as lease:
        try:
            model, model_meta = runner.load_nf4()
            cached, generation = runner.cached_greedy_generate(
                model, tokenizer, prompt_ids, plan)
            ids = cached["generated_token_ids"].astype(int).tolist()
            active = ids[:-1] if generation["terminal_eos_id"] is not None else ids
            decoded = tokenizer.decode(
                active, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            encoded = tokenizer(decoded, add_special_tokens=False,
                                return_offsets_mapping=True)
            reencoded = list(map(int, encoded["input_ids"]))
            prefixes = [tokenizer.decode(
                active[:end], skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
                for end in range(1, len(active) + 1)]
            result = {
                "status": "label_free_offset_diagnosis_after_failed_smoke",
                "runner_sha256": runner.sha(Path(runner.__file__)),
                "record_index": index,
                "microclaim_id": row["microclaim_id"],
                "prompt_tokens": lengths[index],
                "prompt_sha256": runner.digest(prompt),
                "generated_ids": ids,
                "active_ids": active,
                "generated_token_strings": tokenizer.convert_ids_to_tokens(active),
                "decoded": decoded,
                "decoded_sha256": runner.digest(decoded),
                "reencoded_ids": reencoded,
                "reencoded_token_strings": tokenizer.convert_ids_to_tokens(reencoded),
                "reencoded_offsets": [list(map(int, pair))
                                      for pair in encoded["offset_mapping"]],
                "ids_equal": active == reencoded,
                "prefix_decodes": prefixes,
                "prefix_lengths": [len(value) for value in prefixes],
                "prefix_lcp_with_final": [common_prefix(value, decoded)
                                          for value in prefixes],
                "stop_reason": generation["stop_reason"],
                "terminal_eos_id": generation["terminal_eos_id"],
                "lease": lease,
                "model": model_meta,
                "GPU_used": True,
                "gold_read": False,
                "calibration_or_test_read": False,
                "scoring_run": False,
            }
            runner.atomic_json(TARGET, result)
        finally:
            if model is not None:
                del model
            runner.clean_gpu()
    print(json.dumps({
        "target": str(TARGET.resolve()),
        "ids_equal": result["ids_equal"],
        "generated_tokens": len(active),
        "decoded": decoded,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
