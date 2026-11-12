"""Low-dimensional claim-to-window features for semantic window v2.

The functions in this module never inspect labels.  They consume the frozen
claim rows from semantic source attribution v1 and map them to the project's
native four-BPE windows through exact lexical-token ownership.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


LAYERS = 32
HEADS = 32
RAW_WIDTH = LAYERS * HEADS
REGION_KINDS = 9

RAW = slice(0, 1024)
REGION = slice(1024, 1312)
TOP1_NLI = slice(1312, 1315)
TOP3_NLI = slice(1345, 1348)
WHITEBOX = slice(1382, 1394)
SEMANTIC_WIDTH = 1394

LAYER_BANDS = ((0, 8), (8, 16), (16, 24), (24, 32))
REGION_SELECTED = (
    (0, "source_share"),
    (1, "strict_previous_answer_share"),
    (2, "other_context_share"),
    (6, "top1_sentence_concentration"),
    (7, "top3_sentence_concentration"),
    (8, "sentence_relevance_entropy"),
)

WHITEBOX_NAMES = tuple(
    f"claim_token_weighted_mean__whitebox_{aggregation}__{signal}"
    for aggregation in ("max", "mean", "min", "std")
    for signal in ("oof_lookback", "oof_large", "generation_nll")
)

ATTRIBUTION_NAMES = tuple(
    name
    for band, _ in enumerate(LAYER_BANDS)
    for name in (
        f"claim_token_weighted_mean__attribution_band_{band}__raw_logmass_mean",
        f"claim_token_weighted_mean__attribution_band_{band}__raw_logmass_head_std",
    )
) + tuple(
    f"claim_token_weighted_mean__attribution_band_{band}__{kind}"
    for _, kind in REGION_SELECTED
    for band, _ in enumerate(LAYER_BANDS)
) + tuple(
    f"claim_token_weighted_mean__attribution_band_{band}__passage_hhi"
    for band, _ in enumerate(LAYER_BANDS)
)

NLI_NAMES = tuple(
    f"claim_token_weighted_mean__{scope}_{label}"
    for scope in ("attribution_top1_nli", "attribution_top3_weighted_nli")
    for label in ("entailment", "neutral", "contradiction")
)

GEOMETRY_NAMES = (
    "geometry__lexical_token_fraction_of_k4",
    "geometry__ownerless_lexical_fraction",
    "geometry__log1p_distinct_claims",
    "geometry__claim_boundary_crossing",
    "geometry__dominant_claim_owner_mass_fraction",
    "geometry__mean_claim_fraction_covered",
    "geometry__dominant_claim_fraction_covered",
    "geometry__mean_relative_position_in_claim",
    "geometry__mean_center_distance_in_claim",
    "geometry__fraction_at_claim_start",
    "geometry__fraction_at_claim_end",
    "geometry__fraction_within_two_lexical_tokens_of_start",
    "geometry__fraction_within_two_lexical_tokens_of_end",
    "geometry__log1p_dominant_claim_length",
)

LOCAL_NAMES = (
    "local_clean__lookback_probability",
    "local_clean__large_probability",
    "local_clean__generation_nll",
)

BLOCK_NAMES = {
    "whitebox": WHITEBOX_NAMES,
    "attribution": ATTRIBUTION_NAMES,
    "nli": NLI_NAMES,
    "geometry": GEOMETRY_NAMES,
    "local": LOCAL_NAMES,
}

UNION_BLOCKS = ("whitebox", "attribution", "nli", "geometry", "local")
UNION_NAMES = tuple(name for block in UNION_BLOCKS for name in BLOCK_NAMES[block])

VARIANT_BLOCKS = {
    "whitebox_geometry": ("whitebox", "geometry"),
    "whitebox_attribution_geometry": ("whitebox", "attribution", "geometry"),
    "whitebox_attribution_nli_geometry": (
        "whitebox", "attribution", "nli", "geometry",
    ),
    "whitebox_attribution_geometry_local": (
        "whitebox", "attribution", "geometry", "local",
    ),
}


def variant_indices(name: str) -> np.ndarray:
    """Return stable union-column indices for one preregistered variant."""
    blocks = VARIANT_BLOCKS[name]
    wanted = {feature for block in blocks for feature in BLOCK_NAMES[block]}
    indices = np.asarray(
        [index for index, feature in enumerate(UNION_NAMES) if feature in wanted],
        dtype=np.int32,
    )
    expected = sum(len(BLOCK_NAMES[block]) for block in blocks)
    assert len(indices) == expected and len(set(indices.tolist())) == expected
    return indices


def claim_low_dimensional_blocks(semantic: np.ndarray) -> dict[str, np.ndarray]:
    """Compress frozen 1,394-D claim rows by fixed, label-free formulas."""
    semantic = np.asarray(semantic, dtype=np.float32)
    assert semantic.ndim == 2 and semantic.shape[1] == SEMANTIC_WIDTH
    assert np.isfinite(semantic).all()
    count = len(semantic)

    raw = semantic[:, RAW].reshape(count, LAYERS, HEADS).astype(np.float64)
    raw_parts = []
    for left, right in LAYER_BANDS:
        band = raw[:, left:right]
        raw_parts.extend((band.mean(axis=(1, 2)), band.std(axis=(1, 2))))

    region = semantic[:, REGION].reshape(count, REGION_KINDS, LAYERS).astype(np.float64)
    region_parts = []
    for kind_index, _ in REGION_SELECTED:
        for left, right in LAYER_BANDS:
            region_parts.append(region[:, kind_index, left:right].mean(axis=1))

    # Passage shares are already normalized within source mass.  HHI retains
    # diffuse-versus-concentrated routing without selecting a passage or head.
    passage_hhi = np.square(region[:, 3:6]).sum(axis=1)
    passage_parts = [
        passage_hhi[:, left:right].mean(axis=1) for left, right in LAYER_BANDS
    ]
    attribution = np.column_stack((*raw_parts, *region_parts, *passage_parts))
    nli = np.column_stack((semantic[:, TOP1_NLI], semantic[:, TOP3_NLI]))
    whitebox = semantic[:, WHITEBOX]
    result = {
        "whitebox": np.asarray(whitebox, dtype=np.float32),
        "attribution": np.asarray(attribution, dtype=np.float32),
        "nli": np.asarray(nli, dtype=np.float32),
    }
    assert result["whitebox"].shape == (count, len(WHITEBOX_NAMES))
    assert result["attribution"].shape == (count, len(ATTRIBUTION_NAMES))
    assert result["nli"].shape == (count, len(NLI_NAMES))
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def _weighted_claim_mean(
    block: np.ndarray, claim_offset: int, counts: dict[int, float]
) -> np.ndarray:
    claim_ids = np.asarray(sorted(counts), dtype=np.int32)
    weights = np.asarray([counts[int(cid)] for cid in claim_ids], dtype=np.float64)
    weights /= weights.sum()
    return np.average(block[claim_offset + claim_ids], axis=0, weights=weights)


def _window_geometry(
    row: dict,
    window: dict,
    claim_positions: list[dict[int, int]],
    claim_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    lexical = [int(index) for index in window["lexical_token_indices"]]
    assert lexical
    counts: dict[int, float] = defaultdict(float)
    observations: list[tuple[int, int, float]] = []
    ownerless = 0
    for token_index in lexical:
        owners = [int(value) for value in row["lexical_token_microclaims"][token_index]]
        if not owners:
            ownerless += 1
            continue
        mass = 1.0 / len(owners)
        for claim_id in owners:
            assert 0 <= claim_id < len(row["claims"])
            assert token_index in claim_positions[claim_id]
            counts[claim_id] += mass
            observations.append((claim_id, token_index, mass))
    assert counts, (row["response_id"], window["window_id"])
    owner_mass = float(sum(counts.values()))
    dominant = max(sorted(counts), key=lambda cid: counts[cid])
    claim_weight = {cid: value / owner_mass for cid, value in counts.items()}
    fractions = {
        cid: min(float(value / claim_lengths[cid]), 1.0)
        for cid, value in counts.items()
    }

    relative, center_distance = 0.0, 0.0
    at_start = at_end = near_start = near_end = 0.0
    for claim_id, token_index, mass in observations:
        rank = claim_positions[claim_id][token_index]
        length = int(claim_lengths[claim_id])
        position = 0.5 if length == 1 else rank / (length - 1)
        relative += mass * position
        center_distance += mass * abs(position - 0.5) * 2
        at_start += mass * int(rank == 0)
        at_end += mass * int(rank == length - 1)
        near_start += mass * int(rank < 2)
        near_end += mass * int(rank >= length - 2)

    k = int(window.get("k", 4))
    assert k == 4 and owner_mass > 0
    values = np.asarray(
        [
            len(lexical) / k,
            ownerless / len(lexical),
            np.log1p(len(counts)),
            float(len(counts) > 1),
            counts[dominant] / owner_mass,
            sum(claim_weight[cid] * fractions[cid] for cid in counts),
            fractions[dominant],
            relative / owner_mass,
            center_distance / owner_mass,
            at_start / owner_mass,
            at_end / owner_mass,
            near_start / owner_mass,
            near_end / owner_mass,
            np.log1p(claim_lengths[dominant]),
        ],
        dtype=np.float32,
    )
    assert values.shape == (len(GEOMETRY_NAMES),) and np.isfinite(values).all()
    return values, counts


def build_window_feature_matrix(
    semantic_claim_features: np.ndarray,
    claim_response_ids: Iterable[str],
    rows: list[dict],
    meta: dict,
    local_window_signals: np.ndarray,
) -> np.ndarray:
    """Build the union feature matrix in the exact ``meta['windows']`` order.

    No label value is accepted by the API.  Claim rows are averaged according
    to their actual lexical-token ownership inside each window; no claim risk
    score and no max propagation is used.
    """
    claim_blocks = claim_low_dimensional_blocks(semantic_claim_features)
    claim_response_ids = np.asarray(list(claim_response_ids)).astype(str)
    local_window_signals = np.asarray(local_window_signals, dtype=np.float32)
    windows = meta["windows"]
    assert local_window_signals.shape == (len(windows), len(LOCAL_NAMES))
    assert len(claim_response_ids) == len(semantic_claim_features)

    output = np.empty((len(windows), len(UNION_NAMES)), dtype=np.float32)
    cursor = 0
    visited = np.zeros(len(windows), dtype=bool)
    for row in rows:
        claims = row["claims"]
        count = len(claims)
        assert np.all(claim_response_ids[cursor:cursor + count] == row["response_id"])
        assert [int(claim["claim_id"]) for claim in claims] == list(range(count))
        claim_positions = []
        claim_lengths = np.empty(count, dtype=np.int32)
        for claim in claims:
            token_ids = [int(value) for value in claim["lexical_token_indices"]]
            assert token_ids == sorted(set(token_ids)) and token_ids
            claim_positions.append({token_id: rank for rank, token_id in enumerate(token_ids)})
            claim_lengths[int(claim["claim_id"])] = len(token_ids)

        for window_index in meta["answer_windows"][row["response_id"]]:
            window = windows[window_index]
            geometry, counts = _window_geometry(
                row, window, claim_positions, claim_lengths,
            )
            pieces = [
                _weighted_claim_mean(claim_blocks["whitebox"], cursor, counts),
                _weighted_claim_mean(claim_blocks["attribution"], cursor, counts),
                _weighted_claim_mean(claim_blocks["nli"], cursor, counts),
                geometry,
                local_window_signals[window_index],
            ]
            output[window_index] = np.concatenate(pieces)
            visited[window_index] = True
        cursor += count
    assert cursor == len(semantic_claim_features)
    assert visited.all() and np.isfinite(output).all()
    return output


def synthetic_selfcheck() -> dict:
    """Check ordering, weighted mean pooling, and boundary geometry."""
    semantic = np.zeros((2, SEMANTIC_WIDTH), dtype=np.float32)
    semantic[0, WHITEBOX] = 1
    semantic[1, WHITEBOX] = 3
    semantic[:, 1312:1315] = np.asarray([.7, .2, .1], dtype=np.float32)
    semantic[:, 1345:1348] = np.asarray([.6, .3, .1], dtype=np.float32)
    row = {
        "response_id": "synthetic",
        "claims": [
            {"claim_id": 0, "lexical_token_indices": [0, 1]},
            {"claim_id": 1, "lexical_token_indices": [2, 3]},
        ],
        "lexical_token_microclaims": [[0], [0], [1], [1]],
    }
    windows = [
        {
            "window_id": "a", "response_id": "synthetic", "k": 4,
            "token_indices": [0, 1], "lexical_token_indices": [0, 1],
        },
        {
            "window_id": "b", "response_id": "synthetic", "k": 4,
            "token_indices": [1, 2], "lexical_token_indices": [1, 2],
        },
    ]
    meta = {
        "windows": windows,
        "answer_windows": {"synthetic": [0, 1]},
    }
    matrix = build_window_feature_matrix(
        semantic, ["synthetic", "synthetic"], [row], meta,
        np.zeros((2, len(LOCAL_NAMES)), dtype=np.float32),
    )
    assert np.array_equal(matrix[0, :len(WHITEBOX_NAMES)], np.ones(len(WHITEBOX_NAMES)))
    assert np.array_equal(matrix[1, :len(WHITEBOX_NAMES)], np.full(len(WHITEBOX_NAMES), 2))
    assert not np.array_equal(matrix[1, :len(WHITEBOX_NAMES)], semantic[1, WHITEBOX])
    assert all(len(variant_indices(name)) < SEMANTIC_WIDTH for name in VARIANT_BLOCKS)
    return {
        "status": "passed",
        "union_width": len(UNION_NAMES),
        "variant_widths": {
            name: len(variant_indices(name)) for name in VARIANT_BLOCKS
        },
        "claim_score_max_used": False,
        "labels_accepted_by_builder": False,
    }


assert len(WHITEBOX_NAMES) == 12
assert len(ATTRIBUTION_NAMES) == 36
assert len(NLI_NAMES) == 6
assert len(GEOMETRY_NAMES) == 14
assert len(LOCAL_NAMES) == 3
assert len(UNION_NAMES) == 71
