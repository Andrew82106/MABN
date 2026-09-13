"""Frozen, label-free math and geometry for the ReDeEP(Token) migration.

This module contains no dataset loader.  In particular, the mapping functions cannot
inspect gold labels.  The two score identities are kept separate:

* ``paper``: retrieved-context ECS and the standard Jensen-Shannon divergence.
* ``official_code``: full-prefix ECS and the reverse-KL mixture implemented by the
  released repository.

The official repository calls the second quantity JSD, but it is not the formula
written in the paper.  We preserve both without choosing between them.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


VERSION = "redeep-formal-core-v1"


def llama2_official_chat_prompt(user_prompt: str) -> str:
    """Reproduce the one-system/one-user Llama-2 template used by ReDeEP.

    The released detector calls ``apply_chat_template(..., tokenize=False,
    add_generation_prompt=True)`` and then tokenizes the rendered string again with
    the tokenizer default ``add_special_tokens=True``.  Consequently the detector
    receives two BOS tokens.  The second tokenization is deliberately performed by
    the caller; this function only renders the first ``<s>``.
    """

    content = user_prompt[:12000].strip()
    return (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n"
        f"{content} [/INST]"
    )


def stable_minmax_fit(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot fit min-max bounds without a finite value")
    return float(finite.min()), float(finite.max())


def stable_minmax_apply(values: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Apply fit-only min-max scaling, including a deterministic constant case."""

    lo, hi = map(float, bounds)
    values = np.asarray(values, dtype=np.float64)
    if hi < lo:
        raise ValueError(f"invalid min-max bounds: min={lo}, max={hi}")
    if hi == lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def code_reverse_mixture_divergence(
    after_logits: torch.Tensor, before_logits: torch.Tensor
) -> torch.Tensor:
    """Exactly preserve the released token detector's PKS formula.

    PyTorch ``kl_div(log_p, target=m)`` computes ``KL(m || p)``.  The released
    code averages over vocabulary and multiplies by ``10e5`` (one million), so its
    value is ``1e6 / |V|`` times a reverse-KL mixture.  Computation is float32, as
    in the released detector after the model returns FP16 logits.
    """

    after_logits = after_logits.float()
    before_logits = before_logits.float()
    after_prob = F.softmax(after_logits, dim=-1)
    before_prob = F.softmax(before_logits, dim=-1)
    mixture = 0.5 * (after_prob + before_prob)
    left = F.kl_div(F.log_softmax(after_logits, dim=-1), mixture, reduction="none").mean(-1)
    right = F.kl_div(F.log_softmax(before_logits, dim=-1), mixture, reduction="none").mean(-1)
    return 0.5 * (left + right) * 1_000_000.0


def paper_standard_js_divergence(
    after_logits: torch.Tensor, before_logits: torch.Tensor
) -> torch.Tensor:
    """Equation (5): conventional JSD(P || Q), using natural logarithms."""

    after_logits = after_logits.float()
    before_logits = before_logits.float()
    log_after = F.log_softmax(after_logits, dim=-1)
    log_before = F.log_softmax(before_logits, dim=-1)
    after_prob = log_after.exp()
    before_prob = log_before.exp()
    mixture = 0.5 * (after_prob + before_prob)
    log_mixture = torch.log(mixture.clamp_min(torch.finfo(mixture.dtype).tiny))
    left = torch.sum(after_prob * (log_after - log_mixture), dim=-1)
    right = torch.sum(before_prob * (log_before - log_mixture), dim=-1)
    return 0.5 * (left + right)


def top_fraction_indices(
    scores: torch.Tensor,
    allowed_positions: torch.Tensor,
    fraction: float = 0.10,
) -> torch.Tensor:
    """Return absolute positions of the top floor(fraction*n) allowed scores.

    This deliberately does not silently use ``max(1, ...)``.  ReDeEP's released
    code uses ``int(n * 0.1)``.  Inputs with fewer than ten eligible context tokens
    therefore have no defined score and must be reported N/A by the caller.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, sequence]")
    allowed_positions = allowed_positions.to(device=scores.device, dtype=torch.long)
    count = int(allowed_positions.numel() * fraction)
    if count < 1:
        raise ValueError("ReDeEP top-10% set is empty because fewer than 10 positions are eligible")
    allowed_scores = scores.index_select(-1, allowed_positions)
    # The released detector calls argsort without an explicit stable-tie option.
    local = torch.argsort(allowed_scores, dim=-1, descending=True)[..., :count]
    return allowed_positions[local]


def ecs_from_top_positions(
    final_hidden: torch.Tensor,
    predictor_positions: torch.Tensor,
    top_positions: torch.Tensor,
) -> torch.Tensor:
    """Equation (3), for one head and every predictor position."""

    if final_hidden.ndim != 2:
        raise ValueError("final_hidden must have shape [sequence, hidden]")
    predictor_positions = predictor_positions.to(final_hidden.device, dtype=torch.long)
    top_positions = top_positions.to(final_hidden.device, dtype=torch.long)
    if top_positions.ndim != 2 or top_positions.shape[0] != predictor_positions.numel():
        raise ValueError("top_positions must have shape [predictors, selected_context_tokens]")
    selected = final_hidden[top_positions]
    attended_mean = selected.mean(dim=1)
    current = final_hidden[predictor_positions]
    # Preserve the model activation dtype; the released FP16 detector does not cast
    # these two tensors before cosine_similarity.
    return F.cosine_similarity(attended_mean, current, dim=-1)


def overlap_length(interval_a: Sequence[int], interval_b: Sequence[int]) -> int:
    return max(0, min(int(interval_a[1]), int(interval_b[1])) - max(int(interval_a[0]), int(interval_b[0])))


def map_native_tokens_to_windows(
    token_scores: Sequence[float],
    token_char_intervals: Sequence[Sequence[int]],
    windows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Zero-parameter ReDeEP-token -> shared-4-BPE mapping.

    Each shared window receives the arithmetic mean of the native ReDeEP token
    scores whose response-character interval overlaps any of that window's frozen
    character intervals.  A native token is counted once even when it touches more
    than one disjoint interval.  This mirrors ReDeEP's own arithmetic token pooling
    on a smaller, fixed span.  Gold fields in ``windows`` are neither required nor
    read.
    """

    scores = np.asarray(token_scores, dtype=np.float64)
    intervals = np.asarray(token_char_intervals, dtype=np.int64)
    if intervals.shape != (scores.size, 2):
        raise ValueError("token scores and [start,end) character intervals do not align")
    result = np.empty(len(windows), dtype=np.float64)
    for wi, window in enumerate(windows):
        char_intervals = window.get("character_intervals")
        if not isinstance(char_intervals, list) or not char_intervals:
            raise ValueError(f"window {window.get('window_id', wi)!r} lacks character_intervals")
        selected: list[int] = []
        for ti, token_interval in enumerate(intervals):
            if int(token_interval[1]) <= int(token_interval[0]):
                continue
            if any(overlap_length(token_interval, c) > 0 for c in char_intervals):
                selected.append(ti)
        if not selected:
            raise ValueError(f"window {window.get('window_id', wi)!r} has no overlapping ReDeEP token")
        result[wi] = float(scores[np.asarray(selected, dtype=np.int64)].mean())
    return result


def answer_max_window(window_scores: Iterable[float]) -> float:
    values = np.asarray(list(window_scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("an eligible answer must have at least one window")
    return float(values.max())


def f1_opt_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | int]:
    """Project rule: max F1, then max precision, then higher threshold."""

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors")
    if y.size == 0:
        raise ValueError("empty evaluation")
    thresholds = np.unique(s[np.isfinite(s)])
    if thresholds.size == 0:
        raise ValueError("all scores are non-finite")
    best: tuple[float, float, float, int, int, int] | None = None
    for threshold in thresholds:
        pred = s >= threshold
        tp = int(np.sum((y == 1) & pred))
        fp = int(np.sum((y == 0) & pred))
        fn = int(np.sum((y == 1) & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, precision, float(threshold), tp, fp, fn)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    f1, precision, threshold, tp, fp, fn = best
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
