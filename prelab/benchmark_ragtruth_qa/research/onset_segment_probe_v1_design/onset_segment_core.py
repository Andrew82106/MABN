#!/usr/bin/env python3
"""Pure-CPU reference core for onset/continuation targets and decoding.

This module has no dataset reader and no optimizer. Running it executes only
deterministic toy checks; it cannot train a model or access a sealed split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""


def derive_targets(
    lexical_mask: Sequence[int],
    risk_mask: Sequence[int],
    released_span_token_indices: Iterable[Sequence[int]],
) -> dict:
    """Build lexical-axis onset, union-state, continuation, and stop targets."""
    if len(lexical_mask) != len(risk_mask):
        raise ValueError("mask lengths differ")
    if any(value not in (0, 1, False, True) for value in lexical_mask):
        raise ValueError("lexical mask must be binary")
    if any(value not in (0, 1, False, True) for value in risk_mask):
        raise ValueError("risk mask must be binary")
    if any(bool(risk) and not bool(lexical) for lexical, risk in zip(lexical_mask, risk_mask)):
        raise ValueError("risk token cannot be nonlexical")

    lexical_raw_indices = [index for index, value in enumerate(lexical_mask) if value]
    raw_to_lexical = {raw: lexical for lexical, raw in enumerate(lexical_raw_indices)}
    states = [int(bool(risk_mask[raw])) for raw in lexical_raw_indices]

    released_onset_raw_indices = set()
    for mapped in released_span_token_indices:
        mapped = list(mapped)
        if not mapped:
            continue
        onset = int(mapped[0])
        if onset not in raw_to_lexical:
            raise ValueError("released onset is not lexical")
        released_onset_raw_indices.add(onset)
    released_first = [int(raw in released_onset_raw_indices) for raw in lexical_raw_indices]

    binary_onset = []
    transition_head = []
    transition_target = []
    for index, state in enumerate(states):
        previous = states[index - 1] if index else 0
        binary_onset.append(int(state == 1 and previous == 0))
        transition_head.append("continuation" if previous else "onset")
        transition_target.append(state)
    terminal_stop = bool(states and states[-1])
    return {
        "lexical_raw_indices": lexical_raw_indices,
        "union_risk_state": states,
        "released_first_target": released_first,
        "binary_onset_target": binary_onset,
        "transition_head": transition_head,
        "transition_target": transition_target,
        "terminal_stop": terminal_stop,
    }


def soft_risk_recurrence(p_on: Sequence[float], p_cont: Sequence[float]) -> list[float]:
    """Return causal risk probabilities from asymmetric transition heads."""
    if len(p_on) != len(p_cont):
        raise ValueError("head lengths differ")
    output = []
    previous = 0.0
    for onset, continuation in zip(p_on, p_cont):
        if not (0.0 <= onset <= 1.0 and 0.0 <= continuation <= 1.0):
            raise ValueError("probabilities must lie in [0,1]")
        current = (1.0 - previous) * onset + previous * continuation
        output.append(current)
        previous = current
    return output


def viterbi_path(p_on: Sequence[float], p_cont: Sequence[float], epsilon: float = 1e-7) -> list[int]:
    """Most likely binary path under the row-normalized transition model."""
    if len(p_on) != len(p_cont):
        raise ValueError("head lengths differ")
    if not p_on:
        return []

    def clipped_log(value: float) -> float:
        return math.log(min(1.0 - epsilon, max(epsilon, value)))

    score = [clipped_log(1.0 - p_on[0]), clipped_log(p_on[0])]
    backpointers: list[list[int]] = [[0, 0]]
    for index in range(1, len(p_on)):
        transition = (
            (clipped_log(1.0 - p_on[index]), clipped_log(p_on[index])),
            (clipped_log(1.0 - p_cont[index]), clipped_log(p_cont[index])),
        )
        new_score = []
        back = []
        for current in (0, 1):
            candidates = [score[previous] + transition[previous][current] for previous in (0, 1)]
            previous = 1 if candidates[1] > candidates[0] else 0
            new_score.append(candidates[previous])
            back.append(previous)
        score = new_score
        backpointers.append(back)
    state = 1 if score[1] > score[0] else 0
    path = [state]
    for index in range(len(p_on) - 1, 0, -1):
        state = backpointers[index][state]
        path.append(state)
    return list(reversed(path))


def map_to_four_bpe(
    lexical_mask: Sequence[int],
    raw_token_scores: Sequence[float],
    raw_risk_mask: Sequence[int] | None = None,
    released_onset_raw_indices: Iterable[int] = (),
    width: int = 4,
) -> list[dict]:
    """Map raw-token scores and optional gold geometry to frozen windows."""
    if len(lexical_mask) != len(raw_token_scores):
        raise ValueError("score/mask lengths differ")
    if raw_risk_mask is not None and len(raw_risk_mask) != len(lexical_mask):
        raise ValueError("risk/mask lengths differ")
    if width <= 0 or not lexical_mask:
        raise ValueError("positive width and nonempty token axis required")
    onset_set = {int(value) for value in released_onset_raw_indices}
    result = []
    for left in range(max(1, len(lexical_mask) - width + 1)):
        right = min(left + width, len(lexical_mask))
        lexical_indices = [index for index in range(left, right) if lexical_mask[index]]
        if not lexical_indices:
            continue
        row = {
            "token_start": left,
            "token_end": right,
            "score": max(float(raw_token_scores[index]) for index in lexical_indices),
        }
        if raw_risk_mask is not None:
            risk = any(raw_risk_mask[index] for index in lexical_indices)
            onset = risk and any(index in onset_set for index in lexical_indices)
            row["gold"] = "onset" if onset else "internal_continuation" if risk else "clean"
            row["risk"] = int(risk)
        result.append(row)
    return result


def lexical_scores_to_raw(
    lexical_mask: Sequence[int], lexical_scores: Sequence[float], nonlexical_value: float = 0.0
) -> list[float]:
    if sum(bool(value) for value in lexical_mask) != len(lexical_scores):
        raise ValueError("lexical score count mismatch")
    iterator = iter(lexical_scores)
    return [float(next(iterator)) if lexical else float(nonlexical_value) for lexical in lexical_mask]


def selfcheck() -> dict:
    # Two released spans touch through punctuation on the lexical-only axis. The
    # raw-start auxiliary keeps both first-token events; the binary chain has one.
    lexical = [1, 0, 1, 1, 0, 1, 1]
    risk = [0, 0, 1, 1, 0, 1, 0]
    mapped = [[2, 3], [5]]
    target = derive_targets(lexical, risk, mapped)
    assert target["lexical_raw_indices"] == [0, 2, 3, 5, 6]
    assert target["union_risk_state"] == [0, 1, 1, 1, 0]
    assert target["released_first_target"] == [0, 1, 0, 1, 0]
    assert target["binary_onset_target"] == [0, 1, 0, 0, 0]
    assert target["transition_head"] == ["onset", "onset", "continuation", "continuation", "continuation"]
    assert target["transition_target"] == [0, 1, 1, 1, 0]
    assert not target["terminal_stop"]

    p_on = [0.01, 0.95, 0.05, 0.05, 0.01]
    p_cont = [0.01, 0.50, 0.95, 0.95, 0.05]
    soft = soft_risk_recurrence(p_on, p_cont)
    hard = viterbi_path(p_on, p_cont)
    assert all(0.0 <= value <= 1.0 for value in soft)
    assert hard == [0, 1, 1, 1, 0]

    raw_scores = lexical_scores_to_raw(lexical, soft)
    windows = map_to_four_bpe(lexical, raw_scores, risk, [2, 5])
    assert [row["gold"] for row in windows] == ["onset", "onset", "onset", "onset"]
    assert max(row["score"] for row in windows) == max(soft)

    ending = derive_targets([1, 1], [0, 1], [[1]])
    assert ending["terminal_stop"]
    short = map_to_four_bpe([1, 0], [0.2, 0.0], [0, 0], [], width=4)
    assert len(short) == 1 and short[0]["token_end"] == 2
    return {
        "schema_version": "onset-segment-core-selfcheck-v1",
        "status": "PASS",
        "device": "CPU",
        "gpu_libraries_imported": False,
        "training_run": False,
        "checks": [
            "released starts remain distinct while binary union segments stay realizable",
            "onset and continuation/stop transition targets are correct",
            "soft recurrence remains within probability bounds",
            "Viterbi recovers the toy risk segment",
            "4-raw-BPE mapping uses lexical max and answer=max is preserved",
            "terminal stop and short-answer window geometry are explicit"
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true", help="run deterministic CPU toy checks")
    args = parser.parse_args()
    if not args.selfcheck:
        parser.error("this reference core only exposes --selfcheck; it does not train")
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
