"""Independent calibration-score verification for the formal ReDeEP migration.

This file deliberately does not import the scorer or its helper module.  It checks
fit-only scaling bounds from raw tensors, independently remaps every calibration
token score to the shared 4-BPE windows, recomputes answer maxima and metrics, and
compares them with the frozen artifacts.  It never accepts or opens a test path.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "research" / "redeep_formal_baseline_v1"
INPUTS = ROOT / "feature_inputs.jsonl"
GEOMETRY = ROOT / "evaluation_geometry.jsonl"
FREEZE = ROOT / "SCORE_FREEZE.json"
SCORES = ROOT / "frozen_scores.npz"
RESULTS = ROOT / "CAL_RESULTS.json"
CAL_WINDOWS = PROJECT / "data" / "windows_k4_calibration.jsonl"
CAL_ANSWERS = PROJECT / "data" / "answers_calibration.jsonl"
OUT = ROOT / "SCORING_INDEPENDENT_VERIFICATION.json"
EXPECTED_SCORER_SHA = "c5ffa368b6e91780f863bfb2ff523941bb0273651ef2f9475b2df66f4585417f"
EXPECTED_INPUT_SHA = "cd44ebf504d86088ba7b039a1d7263994780ded9ea7a22ba95636e25c4862cb8"
EXPECTED_GEOMETRY_SHA = "b87c570dd0536d61bd0d720fc7b46906a046f1f523dd83365fc71d4862a06b07"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def independent_f1_opt(y: np.ndarray, score: np.ndarray) -> dict:
    """Vectorized implementation of max F1, then precision, then threshold."""

    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order].astype(np.int64)
    tp_all = np.cumsum(sorted_y)
    fp_all = np.cumsum(1 - sorted_y)
    group_ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = tp_all[group_ends]
    fp = fp_all[group_ends]
    fn = int(y.sum()) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    thresholds = sorted_score[group_ends]
    # np.lexsort uses the final key as primary.  Negation gives descending priority.
    winner = np.lexsort((-thresholds, -precision, -f1))[0]
    return {
        "threshold": float(thresholds[winner]),
        "f1": float(f1[winner]),
        "precision": float(precision[winner]),
        "recall": float(recall[winner]),
        "tp": int(tp[winner]),
        "fp": int(fp[winner]),
        "fn": int(fn[winner]),
    }


def close_float(a, b, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def metric_bundle(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "f1_opt": independent_f1_opt(y, score),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "rows": int(y.size),
        "positives": int(y.sum()),
    }


def verify_metric_dict(observed: dict, expected: dict, path: str) -> None:
    for key in ("auroc", "average_precision"):
        if not close_float(observed[key], expected[key]):
            raise RuntimeError(f"{path}.{key} mismatch")
    for key in ("threshold", "f1", "precision", "recall"):
        if not close_float(observed["f1_opt"][key], expected["f1_opt"][key]):
            raise RuntimeError(f"{path}.f1_opt.{key} mismatch")
    for key in ("tp", "fp", "fn", "rows", "positives"):
        source = observed["f1_opt"] if key in {"tp", "fp", "fn"} else observed
        target = expected["f1_opt"] if key in {"tp", "fp", "fn"} else expected
        if int(source[key]) != int(target[key]):
            raise RuntimeError(f"{path}.{key} mismatch")


def overlap_mappings(intervals: np.ndarray, windows: list[dict]) -> list[np.ndarray]:
    mappings = []
    for window in windows:
        selected = []
        for token_index, (token_start, token_end) in enumerate(intervals):
            if token_end <= token_start:
                continue
            for char_start, char_end in window["character_intervals"]:
                if min(int(token_end), int(char_end)) > max(int(token_start), int(char_start)):
                    selected.append(token_index)
                    break
        if not selected:
            raise RuntimeError(f"no ReDeEP token for {window['window_id']}")
        mappings.append(np.asarray(selected, dtype=np.int64))
    return mappings


def verify_fit_bounds(rows: list[dict], freeze: dict) -> dict:
    result = {}
    for identity in ("paper", "official_code"):
        suffix = "paper" if identity == "paper" else "code"
        params = freeze["method_identities"][identity]
        head_indices = np.asarray(params["selected_head_indices"], dtype=np.int64)
        layers = np.asarray(params["selected_layers"], dtype=np.int64)
        ecs_min, ecs_max = np.inf, -np.inf
        pks_min, pks_max = np.inf, -np.inf
        files = tokens = 0
        for row in rows:
            if row["partition"] != "fit":
                continue
            path = ROOT / "raw_features" / "fit" / f"{row['answer_id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                ecs = data[f"ecs_{suffix}"][:, head_indices].astype(np.float64).sum(axis=1)
                pks = data[f"pks_{suffix}"][:, layers].astype(np.float64).sum(axis=1)
            ecs_min = min(ecs_min, float(ecs.min()))
            ecs_max = max(ecs_max, float(ecs.max()))
            pks_min = min(pks_min, float(pks.min()))
            pks_max = max(pks_max, float(pks.max()))
            files += 1
            tokens += int(ecs.size)
        observed = {"ecs_min_max": [ecs_min, ecs_max], "pks_min_max": [pks_min, pks_max]}
        expected = params["fit_only_minmax"]
        if any(not close_float(a, b) for key in observed for a, b in zip(observed[key], expected[key])):
            raise RuntimeError(f"{identity} fit-only MinMax mismatch")
        result[identity] = {"bounds": observed, "fit_files": files, "fit_native_tokens": tokens, "exact": True}
    return result


def main() -> None:
    if sha256_file(INPUTS) != EXPECTED_INPUT_SHA or sha256_file(GEOMETRY) != EXPECTED_GEOMETRY_SHA:
        raise RuntimeError("prepared input/geometry hash mismatch")
    scorer_sha = sha256_file(PROJECT / "src" / "score_redeep_formal_baseline_v1.py")
    if scorer_sha != EXPECTED_SCORER_SHA:
        raise RuntimeError("scorer changed after identity verification/freeze")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256_file(SCORES) != freeze["frozen_scores_sha256"]:
        raise RuntimeError("frozen score hash mismatch")
    released = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = list(read_jsonl(INPUTS))
    fit_bound_checks = verify_fit_bounds(rows, freeze)

    cal_geometry: dict[str, list[dict]] = defaultdict(list)
    for window in read_jsonl(GEOMETRY):
        if window["partition"] == "calibration":
            cal_geometry[str(window["answer_id"])].append(window)

    with np.load(SCORES, allow_pickle=False) as frozen_file:
        frozen = {key: frozen_file[key].copy() for key in frozen_file.files}
    cal_window_mask = frozen["window_partition"] == "calibration"
    cal_answer_mask = frozen["answer_partition"] == "calibration"
    frozen_window_index = {
        str(window_id): i for i, window_id in enumerate(frozen["window_id"])
        if frozen["window_partition"][i] == "calibration"
    }
    frozen_answer_index = {
        str(answer_id): i for i, answer_id in enumerate(frozen["answer_id"])
        if frozen["answer_partition"][i] == "calibration"
    }

    remap_checks = {}
    for identity in ("paper", "official_code"):
        suffix = "paper" if identity == "paper" else "code"
        params = freeze["method_identities"][identity]
        head_indices = np.asarray(params["selected_head_indices"], dtype=np.int64)
        layers = np.asarray(params["selected_layers"], dtype=np.int64)
        ecs_lo, ecs_hi = map(float, params["fit_only_minmax"]["ecs_min_max"])
        pks_lo, pks_hi = map(float, params["fit_only_minmax"]["pks_min_max"])
        maximum_window_difference = 0.0
        maximum_answer_difference = 0.0
        maximum_native_difference = 0.0
        windows_checked = answers_checked = 0
        for row in rows:
            if row["partition"] != "calibration":
                continue
            answer_id = str(row["answer_id"])
            raw_path = ROOT / "raw_features" / "calibration" / f"{answer_id}.npz"
            with np.load(raw_path, allow_pickle=False) as data:
                ecs = data[f"ecs_{suffix}"][:, head_indices].astype(np.float64).sum(axis=1)
                pks = data[f"pks_{suffix}"][:, layers].astype(np.float64).sum(axis=1)
                intervals = data["token_char_intervals"].astype(np.int64)
            h = (pks - pks_lo) / (pks_hi - pks_lo) - 0.2 * (ecs - ecs_lo) / (ecs_hi - ecs_lo)
            mappings = overlap_mappings(intervals, cal_geometry[answer_id])
            local = np.asarray([h[index].mean() for index in mappings], dtype=np.float64)
            expected_local = np.asarray([
                frozen[f"{identity}_window_score"][frozen_window_index[str(window["window_id"])]]
                for window in cal_geometry[answer_id]
            ])
            maximum_window_difference = max(maximum_window_difference, float(np.max(np.abs(local - expected_local))))
            ai = frozen_answer_index[answer_id]
            maximum_answer_difference = max(
                maximum_answer_difference,
                abs(float(local.max()) - float(frozen[f"{identity}_answer_score"][ai])),
            )
            maximum_native_difference = max(
                maximum_native_difference,
                abs(float(h.mean()) - float(frozen[f"{identity}_native_answer_score"][ai])),
            )
            windows_checked += len(local)
            answers_checked += 1
        if max(maximum_window_difference, maximum_answer_difference, maximum_native_difference) > 1e-12:
            raise RuntimeError(f"{identity} independent remap mismatch")
        remap_checks[identity] = {
            "calibration_windows": windows_checked,
            "calibration_answers": answers_checked,
            "max_window_abs_difference": maximum_window_difference,
            "max_answer_abs_difference": maximum_answer_difference,
            "max_native_mean_abs_difference": maximum_native_difference,
        }

    window_gold = {str(x["window_id"]): int(x["label"]) for x in read_jsonl(CAL_WINDOWS)}
    answer_gold = {str(x["answer_id"]): int(x["label"]) for x in read_jsonl(CAL_ANSWERS)}
    window_ids = frozen["window_id"][cal_window_mask]
    answer_ids = frozen["answer_id"][cal_answer_mask]
    wy = np.asarray([window_gold[str(x)] for x in window_ids], dtype=np.int8)
    ay = np.asarray([answer_gold[str(x)] for x in answer_ids], dtype=np.int8)
    metric_checks = {}
    for identity in ("paper", "official_code"):
        observed_window = metric_bundle(wy, frozen[f"{identity}_window_score"][cal_window_mask])
        observed_answer = metric_bundle(ay, frozen[f"{identity}_answer_score"][cal_answer_mask])
        native = frozen[f"{identity}_native_answer_score"][cal_answer_mask]
        observed_native = {
            "auroc": float(roc_auc_score(ay, native)),
            "pearson": float(np.corrcoef(native.astype(np.float64), ay.astype(np.float64))[0, 1]),
            "rows": int(ay.size),
        }
        verify_metric_dict(observed_window, released["identities"][identity]["shared_k4_window"], f"{identity}.window")
        verify_metric_dict(observed_answer, released["identities"][identity]["shared_answer_max_window"], f"{identity}.answer")
        expected_native = released["identities"][identity]["author_native_answer_mean"]
        if (
            not close_float(observed_native["auroc"], expected_native["auroc"])
            or not close_float(observed_native["pearson"], expected_native["pearson"])
            or observed_native["rows"] != expected_native["rows"]
        ):
            raise RuntimeError(f"{identity}.author_native_answer_mean mismatch")
        metric_checks[identity] = {
            "shared_k4_window": observed_window,
            "shared_answer_max_window": observed_answer,
            "author_native_answer_mean": observed_native,
        }

    all_positive_f1 = 2 * int(ay.sum()) / (int(ay.size) + int(ay.sum()))
    payload = {
        "version": "redeep-formal-independent-scoring-verification-v1",
        "status": "passed",
        "label_access": {
            "fit_labels_read_by_this_verifier": False,
            "calibration_labels_read_only_after_frozen_score_hash_check": True,
            "test_labels_read": False,
        },
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
        "frozen_scores_sha256": sha256_file(SCORES),
        "scorer_sha256": scorer_sha,
        "fit_bound_checks": fit_bound_checks,
        "calibration_remap_checks": remap_checks,
        "metric_checks": metric_checks,
        "answer_label_prevalence": float(ay.mean()),
        "all_positive_answer_f1": float(all_positive_f1),
        "interpretation_guard": "The answer F1 must be interpreted beside AUROC/AP and the all-positive F1 because 100/159 calibration answers are positive.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "fit_files_checked": fit_bound_checks["paper"]["fit_files"],
        "calibration_windows_remapped": remap_checks["paper"]["calibration_windows"],
        "all_positive_answer_f1": all_positive_f1,
    }))


if __name__ == "__main__":
    main()
