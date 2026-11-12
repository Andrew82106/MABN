"""Diagnostic-only exact comparison of main and separate-MLP native outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-heads", type=Path, required=True)
    parser.add_argument("--separate-mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_rows = read_jsonl(args.transformer_heads.resolve())
    mlp_rows = read_jsonl(args.separate_mlp.resolve())
    if [row["answer_id"] for row in main_rows] != [row["answer_id"] for row in mlp_rows]:
        raise ValueError("route answer order differs")

    unequal = 0
    comparisons = 0
    max_abs_difference = 0.0
    first_unequal = None
    for main_row, mlp_row in zip(main_rows, mlp_rows):
        for key in ("answer_sha256", "author_response_token_ids", "author_response_token_char_intervals"):
            if main_row[key] != mlp_row[key]:
                raise ValueError(f"native geometry differs for {main_row['answer_id']}: {key}")
        main_probs = list(map(float, main_row["author_response_token_probabilities"]))
        mlp_probs = list(map(float, mlp_row["author_response_token_probabilities"]))
        if len(main_probs) != len(mlp_probs):
            raise ValueError(f"probability length differs for {main_row['answer_id']}")
        for token_index, (left, right) in enumerate(zip(main_probs, mlp_probs)):
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("non-finite probability")
            difference = abs(left - right)
            max_abs_difference = max(max_abs_difference, difference)
            comparisons += 1
            if left != right:
                unequal += 1
                if first_unequal is None:
                    first_unequal = {
                        "answer_id": main_row["answer_id"], "token_index": token_index,
                        "transformer_heads": left, "separate_mlp": right,
                    }
    result = {
        "version": "ragognizer-route-identity-diagnostic-v1",
        "role": "diagnostic_only_never_result_selected",
        "rows": len(main_rows),
        "token_probability_comparisons": comparisons,
        "exactly_equal": unequal == 0,
        "unequal_values": unequal,
        "max_absolute_difference": max_abs_difference,
        "first_unequal": first_unequal,
        "primary_route_rule": "separate MLP may replace neither the model-card default nor its scores unless every native probability is exactly equal",
    }
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
