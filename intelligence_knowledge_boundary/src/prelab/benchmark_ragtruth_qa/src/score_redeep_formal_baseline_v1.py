"""Fit-only ReDeEP selection, label-free 4-BPE mapping, and cal evaluation.

Stages are physically separated:

1. ``check`` only checks label-free inputs and raw feature completeness.
2. ``freeze`` opens shared *fit* labels to rank concrete heads/layers, then writes
   both fit and calibration scores using label-free geometry.  It never opens a
   calibration gold file.
3. ``evaluate`` opens only frozen calibration scores and calibration gold files.

No stage accepts a test path or an arbitrary data path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from redeep_formal_core_v1 import (
    answer_max_window,
    f1_opt_threshold,
    stable_minmax_apply,
    stable_minmax_fit,
)


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "research" / "redeep_formal_baseline_v1"
INPUTS = ROOT / "feature_inputs.jsonl"
GEOMETRY = ROOT / "evaluation_geometry.jsonl"
RAW_ROOT = ROOT / "raw_features"
FROZEN_SCORES = ROOT / "frozen_scores.npz"
SCORE_FREEZE = ROOT / "SCORE_FREEZE.json"
RESULTS = ROOT / "CAL_RESULTS.json"
STATUS = ROOT / "SCORING_STATUS.json"
FIT_WINDOWS = PROJECT / "fit_expansion" / "data" / "windows_k4_fit.jsonl"
CAL_WINDOWS = PROJECT / "data" / "windows_k4_calibration.jsonl"
CAL_ANSWERS = PROJECT / "data" / "answers_calibration.jsonl"

EXPECTED_INPUT_SHA = "cd44ebf504d86088ba7b039a1d7263994780ded9ea7a22ba95636e25c4862cb8"
EXPECTED_GEOMETRY_SHA = "b87c570dd0536d61bd0d720fc7b46906a046f1f523dd83365fc71d4862a06b07"
IDENTITIES = ("paper", "official_code")
FIXED_K_HEADS = 1
FIXED_K_LAYERS = 10
FIXED_ALPHA_PKS = 1.0
FIXED_BETA_ECS = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def input_rows() -> list[dict]:
    if sha256_file(INPUTS) != EXPECTED_INPUT_SHA:
        raise RuntimeError("label-free feature input drift")
    rows = list(read_jsonl(INPUTS))
    if any(row.get("partition") not in {"fit", "calibration"} for row in rows):
        raise RuntimeError("forbidden partition in feature inputs")
    return rows


def raw_path(row: dict) -> Path:
    return RAW_ROOT / row["partition"] / f"{row['answer_id']}.npz"


def validate_raw(path: Path, row: dict) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=False) as data:
            n = len(row["native_response_token_char_intervals"])
            prefix = int(row["official_prefix_token_count"])
            full = int(row["official_full_token_count"])
            feature_names = ("ecs_code", "ecs_paper", "pks_code", "pks_paper")
            checks = [
                str(data["version"].item()) == "redeep-formal-raw-feature-v1",
                data["ecs_code"].shape == (n, 32),
                data["ecs_paper"].shape == (n, 32),
                data["pks_code"].shape == (n, 32),
                data["pks_paper"].shape == (n, 32),
                data["token_char_intervals"].shape == (n, 2),
                np.array_equal(data["predictor_positions"], np.arange(prefix - 1, full - 1)),
                np.array_equal(data["target_positions"], np.arange(prefix, full)),
                np.array_equal(
                    data["token_char_intervals"],
                    np.asarray(row["native_response_token_char_intervals"], dtype=np.int32),
                ),
                all(np.isfinite(data[name]).all() for name in feature_names),
                str(data["answer_sha256"].item()) == row["answer_sha256"],
            ]
            return (all(checks), "ok" if all(checks) else "shape_or_identity")
    except Exception as exc:
        return False, f"unreadable:{type(exc).__name__}"


def completeness(write: bool = True) -> dict:
    rows = input_rows()
    reasons: dict[str, int] = defaultdict(int)
    complete = 0
    for row in rows:
        ok, reason = validate_raw(raw_path(row), row)
        if ok:
            complete += 1
        else:
            reasons[reason] += 1
    payload = {
        "version": "redeep-formal-scoring-status-v1",
        "status": "ready" if complete == len(rows) else "N/A_pending_full_FP16_GPU_extraction",
        "required": len(rows),
        "complete": complete,
        "missing_or_invalid": len(rows) - complete,
        "reasons": dict(sorted(reasons.items())),
        "window_metrics": None,
        "answer_metrics": None,
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
    }
    if write:
        STATUS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def geometry_by_partition() -> dict[str, dict[str, list[dict]]]:
    if sha256_file(GEOMETRY) != EXPECTED_GEOMETRY_SHA:
        raise RuntimeError("label-free window geometry drift")
    result: dict[str, dict[str, list[dict]]] = {
        "fit": defaultdict(list),
        "calibration": defaultdict(list),
    }
    allowed = {
        "version", "partition", "response_id", "answer_id", "group_id",
        "window_id", "character_intervals", "labels_present",
    }
    for row in read_jsonl(GEOMETRY):
        if set(row) - allowed or row.get("labels_present") is not False:
            raise RuntimeError("gold-like field entered the mapping geometry")
        result[row["partition"]][str(row["answer_id"])].append(row)
    return result


def overlap_index_lists(token_intervals: np.ndarray, windows: Sequence[dict]) -> list[np.ndarray]:
    intervals = np.asarray(token_intervals, dtype=np.int64)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("native token character intervals must be [N,2]")
    starts, ends = intervals[:, 0], intervals[:, 1]
    mappings: list[np.ndarray] = []
    for window in windows:
        mask = np.zeros(intervals.shape[0], dtype=bool)
        for char_start, char_end in window["character_intervals"]:
            mask |= (ends > int(char_start)) & (starts < int(char_end)) & (ends > starts)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise RuntimeError(f"window has no native ReDeEP token: {window['window_id']}")
        mappings.append(indices)
    return mappings


def pool_matrix(matrix: np.ndarray, mappings: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.stack([matrix[index].mean(axis=0) for index in mappings], axis=0)


def fit_labels() -> dict[str, int]:
    # This is the only gold loader reachable by the freeze stage.
    labels: dict[str, int] = {}
    for row in read_jsonl(FIT_WINDOWS):
        window_id = str(row["window_id"])
        if window_id in labels:
            raise RuntimeError(f"duplicate fit window {window_id}")
        labels[window_id] = int(row["label"])
    return labels


def pearson_score(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if np.std(values) == 0.0 or np.std(labels) == 0.0:
        return -math.inf
    return float(np.corrcoef(values, labels)[0, 1])


def rank_desc(scores: Sequence[float], identities: Sequence[object]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), identities[i]))


def raw_arrays(path: Path, identity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        suffix = "paper" if identity == "paper" else "code"
        return (
            data[f"ecs_{suffix}"].astype(np.float64),
            data[f"pks_{suffix}"].astype(np.float64),
            data["token_char_intervals"].astype(np.int64),
            data["candidate_heads"].astype(np.int64),
        )


def select_parameters(identity: str, rows: list[dict], geometry, labels_by_id) -> dict:
    ecs_windows: list[np.ndarray] = []
    pks_windows: list[np.ndarray] = []
    labels: list[int] = []
    candidate_heads: np.ndarray | None = None
    for row in rows:
        if row["partition"] != "fit":
            continue
        ecs, pks, intervals, heads = raw_arrays(raw_path(row), identity)
        if candidate_heads is None:
            candidate_heads = heads
        elif not np.array_equal(candidate_heads, heads):
            raise RuntimeError("candidate-head order drift across raw feature files")
        windows = geometry["fit"][str(row["answer_id"])]
        mappings = overlap_index_lists(intervals, windows)
        ecs_windows.append(pool_matrix(ecs, mappings))
        pks_windows.append(pool_matrix(pks, mappings))
        labels.extend(labels_by_id[str(window["window_id"])] for window in windows)
    if candidate_heads is None:
        raise RuntimeError("no fit features")
    ecs_all = np.concatenate(ecs_windows, axis=0)
    pks_all = np.concatenate(pks_windows, axis=0)
    y = np.asarray(labels, dtype=np.int8)
    if ecs_all.shape[0] != 653979 or pks_all.shape[0] != 653979:
        raise RuntimeError("fit window coverage changed")

    if identity == "paper":
        head_values = [pearson_score(ecs_all[:, i], 1 - y) for i in range(ecs_all.shape[1])]
        layer_values = [pearson_score(pks_all[:, i], y) for i in range(pks_all.shape[1])]
        rank_measure = "Pearson(ECS,1-risk) and Pearson(PKS,risk), as stated by paper Appendix J"
    else:
        head_values = [float(roc_auc_score(1 - y, ecs_all[:, i])) for i in range(ecs_all.shape[1])]
        layer_values = [float(roc_auc_score(y, pks_all[:, i])) for i in range(pks_all.shape[1])]
        rank_measure = "AUROC(ECS,1-risk) and AUROC(PKS,risk), as executed by released token_level_reg.py"
    head_order = rank_desc(head_values, [tuple(map(int, x)) for x in candidate_heads.tolist()])
    layer_order = rank_desc(layer_values, list(range(32)))
    selected_head_indices = head_order[:FIXED_K_HEADS]
    selected_layers = layer_order[:FIXED_K_LAYERS]
    return {
        "identity": identity,
        "rank_measure": rank_measure,
        "fixed_k_heads": FIXED_K_HEADS,
        "fixed_k_layers": FIXED_K_LAYERS,
        "fixed_alpha_pks": FIXED_ALPHA_PKS,
        "fixed_beta_ecs": FIXED_BETA_ECS,
        "selected_head_indices": selected_head_indices,
        "selected_heads": [candidate_heads[i].tolist() for i in selected_head_indices],
        "selected_layers": selected_layers,
        "head_ranking": [
            {"head": candidate_heads[i].tolist(), "score": head_values[i], "rank": head_order.index(i) + 1}
            for i in range(len(head_values))
        ],
        "layer_ranking": [
            {"layer": i, "score": layer_values[i], "rank": layer_order.index(i) + 1}
            for i in range(len(layer_values))
        ],
        "fit_window_rows": int(y.size),
        "selection_labels": "shared fit 4-BPE window risk labels only",
        "k_beta_search": "not rerun; fixed to published RAGTruth Llama2-7B Token values because the paper omits greedy search order/objective/ties and released search line raises KeyError",
    }


def fit_bounds(identity: str, rows: list[dict], params: dict) -> dict:
    ecs_values: list[np.ndarray] = []
    pks_values: list[np.ndarray] = []
    hi = params["selected_head_indices"]
    li = params["selected_layers"]
    for row in rows:
        if row["partition"] != "fit":
            continue
        ecs, pks, _intervals, _heads = raw_arrays(raw_path(row), identity)
        ecs_values.append(ecs[:, hi].sum(axis=1))
        pks_values.append(pks[:, li].sum(axis=1))
    ecs_bounds = stable_minmax_fit(np.concatenate(ecs_values))
    pks_bounds = stable_minmax_fit(np.concatenate(pks_values))
    return {"ecs_min_max": list(ecs_bounds), "pks_min_max": list(pks_bounds)}


def score_identity(identity: str, rows: list[dict], geometry, params: dict, bounds: dict) -> dict[str, np.ndarray]:
    window_ids: list[str] = []
    window_partitions: list[str] = []
    window_scores: list[float] = []
    answer_ids: list[str] = []
    answer_partitions: list[str] = []
    answer_scores: list[float] = []
    native_answer_scores: list[float] = []
    hi = params["selected_head_indices"]
    li = params["selected_layers"]
    for row in rows:
        ecs, pks, intervals, _heads = raw_arrays(raw_path(row), identity)
        ecs_sum = ecs[:, hi].sum(axis=1)
        pks_sum = pks[:, li].sum(axis=1)
        h_token = (
            FIXED_ALPHA_PKS * stable_minmax_apply(pks_sum, bounds["pks_min_max"])
            - FIXED_BETA_ECS * stable_minmax_apply(ecs_sum, bounds["ecs_min_max"])
        )
        windows = geometry[row["partition"]][str(row["answer_id"])]
        mappings = overlap_index_lists(intervals, windows)
        local_window_scores = np.asarray([h_token[index].mean() for index in mappings], dtype=np.float64)
        window_ids.extend(str(window["window_id"]) for window in windows)
        window_partitions.extend([row["partition"]] * len(windows))
        window_scores.extend(local_window_scores.tolist())
        answer_ids.append(str(row["answer_id"]))
        answer_partitions.append(row["partition"])
        answer_scores.append(answer_max_window(local_window_scores))
        native_answer_scores.append(float(h_token.mean()))
    return {
        "window_id": np.asarray(window_ids),
        "window_partition": np.asarray(window_partitions),
        "window_score": np.asarray(window_scores, dtype=np.float64),
        "answer_id": np.asarray(answer_ids),
        "answer_partition": np.asarray(answer_partitions),
        "answer_score": np.asarray(answer_scores, dtype=np.float64),
        "native_answer_score": np.asarray(native_answer_scores, dtype=np.float64),
    }


def freeze_scores() -> dict:
    status = completeness(write=True)
    if status["status"] != "ready":
        raise RuntimeError(f"raw extraction incomplete: {status['complete']}/{status['required']}")
    rows = input_rows()
    geometry = geometry_by_partition()
    labels_by_id = fit_labels()  # no calibration gold is opened in this function
    expected_fit_ids = {
        str(window["window_id"])
        for windows in geometry["fit"].values()
        for window in windows
    }
    if set(labels_by_id) != expected_fit_ids:
        raise RuntimeError("fit labels and label-free geometry do not join exactly")

    selections = {}
    scores = {}
    for identity in IDENTITIES:
        params = select_parameters(identity, rows, geometry, labels_by_id)
        bounds = fit_bounds(identity, rows, params)
        params["fit_only_minmax"] = bounds
        selections[identity] = params
        scores[identity] = score_identity(identity, rows, geometry, params, bounds)

    first = scores[IDENTITIES[0]]
    for identity in IDENTITIES[1:]:
        current = scores[identity]
        for key in ("window_id", "window_partition", "answer_id", "answer_partition"):
            if not np.array_equal(first[key], current[key]):
                raise RuntimeError(f"score identity alignment mismatch: {key}")
    np.savez_compressed(
        FROZEN_SCORES,
        version=np.asarray("redeep-formal-frozen-scores-v1"),
        window_id=first["window_id"],
        window_partition=first["window_partition"],
        answer_id=first["answer_id"],
        answer_partition=first["answer_partition"],
        paper_window_score=scores["paper"]["window_score"],
        official_code_window_score=scores["official_code"]["window_score"],
        paper_answer_score=scores["paper"]["answer_score"],
        official_code_answer_score=scores["official_code"]["answer_score"],
        paper_native_answer_score=scores["paper"]["native_answer_score"],
        official_code_native_answer_score=scores["official_code"]["native_answer_score"],
    )
    payload = {
        "version": "redeep-formal-score-freeze-v1",
        "status": "scores_frozen_before_calibration_gold",
        "frozen_scores_sha256": sha256_file(FROZEN_SCORES),
        "method_identities": selections,
        "mapping": {
            "native_token_to_shared_k4": "arithmetic mean over ReDeEP target-token scores whose response character interval overlaps the frozen eligible window character intervals",
            "whole_answer": "maximum of all eligible shared 4-BPE window scores",
            "parameters": "none; character overlap and arithmetic mean/max only",
            "calibration_labels_opened": False,
        },
        "counts": {
            "window_rows": int(first["window_id"].size),
            "answer_rows": int(first["answer_id"].size),
        },
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
    }
    SCORE_FREEZE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def metric_bundle(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "f1_opt": f1_opt_threshold(labels, scores),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "rows": int(labels.size),
        "positives": int(labels.sum()),
    }


def evaluate() -> dict:
    freeze = json.loads(SCORE_FREEZE.read_text(encoding="utf-8"))
    if sha256_file(FROZEN_SCORES) != freeze["frozen_scores_sha256"]:
        raise RuntimeError("frozen score artifact changed before evaluation")
    with np.load(FROZEN_SCORES, allow_pickle=False) as data:
        cal_window_mask = data["window_partition"] == "calibration"
        cal_answer_mask = data["answer_partition"] == "calibration"
        frozen = {key: data[key].copy() for key in data.files}

    window_gold = {str(row["window_id"]): int(row["label"]) for row in read_jsonl(CAL_WINDOWS)}
    answer_gold = {str(row["answer_id"]): int(row["label"]) for row in read_jsonl(CAL_ANSWERS)}
    window_ids = frozen["window_id"][cal_window_mask]
    answer_ids = frozen["answer_id"][cal_answer_mask]
    if set(map(str, window_ids)) != set(window_gold) or set(map(str, answer_ids)) != set(answer_gold):
        raise RuntimeError("calibration score/gold exact join failed")
    wy = np.asarray([window_gold[str(x)] for x in window_ids], dtype=np.int8)
    ay = np.asarray([answer_gold[str(x)] for x in answer_ids], dtype=np.int8)
    results = {}
    for identity in IDENTITIES:
        ws = frozen[f"{identity}_window_score"][cal_window_mask]
        ans = frozen[f"{identity}_answer_score"][cal_answer_mask]
        native = frozen[f"{identity}_native_answer_score"][cal_answer_mask]
        results[identity] = {
            "shared_k4_window": metric_bundle(wy, ws),
            "shared_answer_max_window": metric_bundle(ay, ans),
            "author_native_answer_mean": {
                "auroc": float(roc_auc_score(ay, native)),
                "pearson": pearson_score(native, ay),
                "rows": int(ay.size),
            },
        }
    payload = {
        "version": "redeep-formal-cal-results-v1",
        "status": "complete",
        "scores_frozen_before_calibration_gold": True,
        "identities": results,
        "calibration": {"windows": int(wy.size), "answers": int(ay.size)},
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
        "warning": "Calibration F1Opt follows the shared development protocol and is optimized/evaluated on the same repeatedly used calibration partition.",
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "freeze", "evaluate"])
    args = parser.parse_args()
    if args.stage == "check":
        payload = completeness(write=True)
    elif args.stage == "freeze":
        payload = freeze_scores()
    else:
        payload = evaluate()
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
