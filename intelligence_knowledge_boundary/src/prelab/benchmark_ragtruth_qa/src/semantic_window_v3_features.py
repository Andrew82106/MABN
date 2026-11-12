"""Label-free feature composition for semantic-window v3.

The v2 whitebox+geometry backbone is copied exactly.  Evidence-union NLI
claim rows are pooled only over the lexical tokens owned inside each native
four-BPE window.  No claim-level detector score is projected by maximum.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

import semantic_window_v2_features as v2


EVIDENCE_UNION_WIDTH = 44
BACKBONE_COLUMNS = v2.variant_indices("whitebox_geometry")
ATTRIBUTION_COLUMNS = np.asarray(
    [index for index, name in enumerate(v2.UNION_NAMES)
     if name in set(v2.ATTRIBUTION_NAMES)],
    dtype=np.int32,
)

VARIANT_BLOCKS = {
    "backbone_union": ("backbone", "evidence_union"),
    "backbone_union_attribution": ("backbone", "evidence_union", "attribution"),
}


def feature_names(evidence_names: Iterable[str]) -> tuple[str, ...]:
    evidence_names = tuple(map(str, evidence_names))
    assert len(evidence_names) == EVIDENCE_UNION_WIDTH
    assert len(set(evidence_names)) == len(evidence_names)
    names = (
        tuple(v2.UNION_NAMES[index] for index in BACKBONE_COLUMNS)
        + tuple(f"claim_token_weighted_mean__evidence_union__{name}"
                for name in evidence_names)
        + tuple(v2.UNION_NAMES[index] for index in ATTRIBUTION_COLUMNS)
    )
    assert len(names) == 26 + EVIDENCE_UNION_WIDTH + 36
    assert len(set(names)) == len(names)
    return names


def variant_indices(name: str) -> np.ndarray:
    widths = {
        "backbone": len(BACKBONE_COLUMNS),
        "evidence_union": EVIDENCE_UNION_WIDTH,
        "attribution": len(ATTRIBUTION_COLUMNS),
    }
    offsets, cursor = {}, 0
    for block in ("backbone", "evidence_union", "attribution"):
        offsets[block] = np.arange(cursor, cursor + widths[block], dtype=np.int32)
        cursor += widths[block]
    result = np.concatenate([offsets[block] for block in VARIANT_BLOCKS[name]])
    assert len(result) == sum(widths[block] for block in VARIANT_BLOCKS[name])
    return result


def _claim_positions(row: dict) -> tuple[list[dict[int, int]], np.ndarray]:
    claims = row["claims"]
    positions = []
    lengths = np.empty(len(claims), dtype=np.int32)
    assert [int(claim["claim_id"]) for claim in claims] == list(range(len(claims)))
    for claim in claims:
        claim_id = int(claim["claim_id"])
        token_ids = [int(value) for value in claim["lexical_token_indices"]]
        assert token_ids == sorted(set(token_ids)) and token_ids
        positions.append({token_id: rank for rank, token_id in enumerate(token_ids)})
        lengths[claim_id] = len(token_ids)
    return positions, lengths


def pool_claim_features_to_windows(
    claim_features: np.ndarray,
    claim_response_ids: Iterable[str],
    rows: list[dict],
    meta: dict,
) -> np.ndarray:
    """Token-ownership weighted claim mean in native window order.

    Labels are deliberately absent from the API.  Evidence maxima inside the
    supplied claim descriptors remain ordinary input columns; the pooling
    operation itself is a weighted mean and never broadcasts a claim risk.
    """
    claim_features = np.asarray(claim_features, dtype=np.float32)
    claim_response_ids = np.asarray(list(claim_response_ids)).astype(str)
    assert claim_features.ndim == 2
    assert len(claim_features) == len(claim_response_ids)
    assert np.isfinite(claim_features).all()
    output = np.empty((len(meta["windows"]), claim_features.shape[1]), dtype=np.float32)
    visited = np.zeros(len(output), dtype=bool)
    cursor = 0
    for row in rows:
        count = len(row["claims"])
        assert np.all(claim_response_ids[cursor:cursor + count] == row["response_id"])
        positions, lengths = _claim_positions(row)
        for window_index in meta["answer_windows"][row["response_id"]]:
            window = meta["windows"][window_index]
            _, counts = v2._window_geometry(row, window, positions, lengths)
            output[window_index] = v2._weighted_claim_mean(
                claim_features, cursor, counts
            )
            visited[window_index] = True
        cursor += count
    assert cursor == len(claim_features)
    assert visited.all() and np.isfinite(output).all()
    return output


def compose_window_features(
    v2_window_union: np.ndarray,
    evidence_claim_features: np.ndarray,
    claim_response_ids: Iterable[str],
    rows: list[dict],
    meta: dict,
) -> np.ndarray:
    """Return backbone26 + evidence-union44 + attribution36."""
    v2_window_union = np.asarray(v2_window_union, dtype=np.float32)
    assert v2_window_union.shape == (len(meta["windows"]), len(v2.UNION_NAMES))
    evidence = pool_claim_features_to_windows(
        evidence_claim_features, claim_response_ids, rows, meta
    )
    assert evidence.shape == (len(meta["windows"]), EVIDENCE_UNION_WIDTH)
    result = np.column_stack((
        v2_window_union[:, BACKBONE_COLUMNS],
        evidence,
        v2_window_union[:, ATTRIBUTION_COLUMNS],
    )).astype(np.float32, copy=False)
    assert result.shape == (len(meta["windows"]), 106)
    assert np.isfinite(result).all()
    return result


def synthetic_selfcheck() -> dict:
    rows = [{
        "response_id": "x",
        "claims": [
            {"claim_id": 0, "lexical_token_indices": [0, 1]},
            {"claim_id": 1, "lexical_token_indices": [2, 3]},
        ],
        "lexical_token_microclaims": [[0], [0], [1], [1]],
    }]
    meta = {
        "windows": [
            {"window_id": "x0", "k": 4, "lexical_token_indices": [0, 1]},
            {"window_id": "x1", "k": 4, "lexical_token_indices": [1, 2]},
        ],
        "answer_windows": {"x": [0, 1]},
    }
    claims = np.vstack((
        np.ones(EVIDENCE_UNION_WIDTH, dtype=np.float32),
        np.full(EVIDENCE_UNION_WIDTH, 3, dtype=np.float32),
    ))
    pooled = pool_claim_features_to_windows(claims, ["x", "x"], rows, meta)
    assert np.array_equal(pooled[0], claims[0])
    assert np.array_equal(pooled[1], np.full(EVIDENCE_UNION_WIDTH, 2))
    assert not np.array_equal(pooled[1], claims[1])
    return {
        "status": "passed",
        "union_width": 106,
        "variant_widths": {
            name: len(variant_indices(name)) for name in VARIANT_BLOCKS
        },
        "claim_score_max_projection_used": False,
        "labels_accepted_by_builder": False,
    }


assert len(BACKBONE_COLUMNS) == 26
assert len(ATTRIBUTION_COLUMNS) == 36
assert len(variant_indices("backbone_union")) == 70
assert len(variant_indices("backbone_union_attribution")) == 106
