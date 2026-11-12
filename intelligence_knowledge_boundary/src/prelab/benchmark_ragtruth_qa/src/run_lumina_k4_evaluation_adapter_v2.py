"""Gold-isolated replay of frozen LUMINA token scores onto the common QA task.

``map`` opens only label-free feature inputs and frozen feature/score artifacts.
``score`` runs later and is the only stage that opens human labels.
No model is loaded or changed in either stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import run_development as q


ROOT = q.ROOT
PREP = ROOT / "results/lumina_qa_preparation_v1"
FEATURES = ROOT / "results/lumina_qa_features_v1"
LEGACY = ROOT / "results/lumina_qa_scoring_v1"
OUT = ROOT / "results/lumina_k4_evaluation_adapter_v2"
FEATURE_INPUTS = PREP / "feature_inputs.jsonl"
METHODS = ("lumina", "ipr", "negative_mmd")
N_ANSWERS = 3839
N_FIT_ANSWERS = 3680
N_FIT_WINDOWS = 653979
N_CAL_WINDOWS = 42241
N_WINDOWS = N_FIT_WINDOWS + N_CAL_WINDOWS
N_RAW_TOKENS = 708506
K = 4


def protocol():
    return {
        "version": "lumina-frozen-token-to-common-k4-v2-gold-isolated",
        "role": "evaluation_adapter_for_frozen_lumina_formula; no model change",
        "native_output": (
            "Frozen per-answer-token float32 LUMINA score: "
            "0.5*IPR-0.5*MMD_squared. IPR and -MMD are retained only as fixed ablations."
        ),
        "mapping": {
            "window": (
                "Reconstruct the project's eligible stride-one 4-raw-BPE geometry from "
                "label-free answer text and frozen token offsets; take the float64 arithmetic "
                "mean of the four token scores (all actual tokens for an answer shorter than four)."
            ),
            "answer": "Maximum over every eligible mapped 4-BPE window in the answer.",
            "parameters": 0,
            "training": False,
            "deterministic": True,
        },
        "gold_isolation": {
            "map_stage": (
                "May open only feature_inputs.jsonl (which has labels_used=false), frozen "
                "LUMINA feature artifacts, and legacy score arrays. It never opens answers_*, "
                "tokens_*, windows_k4_*, gold manifests, metrics, thresholds, or test data."
            ),
            "score_stage": (
                "Runs only after mapped arrays are frozen and byte-bound; it then opens project "
                "answer/window labels solely for thresholds and metrics."
            ),
        },
        "evaluation": (
            "On calibration, choose separate window and answer thresholds by F1, then precision, "
            "then higher threshold; predict score>=threshold and report F1/AUROC/AP."
        ),
        "legacy_replay": (
            "Every v2 mapped value must equal the corresponding frozen v1 score value exactly; "
            "v1 remains historical because its orchestration validated gold before score freeze."
        ),
        "official_test_opened": False,
        "model_loaded": False,
        "new_model_fits": 0,
    }


def ensure_new_output():
    assert not OUT.exists(), f"Refusing to overwrite {OUT}"
    OUT.mkdir(parents=True)
    q.save(OUT / "protocol.json", protocol())


def label_free_feature_gate():
    done = q.read(FEATURES / "features_complete.json")
    assert done["status"] == "complete" and done["records"] == N_ANSWERS
    assert done["raw_answer_tokens"] == N_RAW_TOKENS and done["all_records_validated"]
    assert done["no_test"] and done["trained"] is False
    manifest = q.read(FEATURES / "feature_manifest.json")
    assert manifest["status"] == "complete" and len(manifest["entries"]) == N_ANSWERS
    assert manifest["raw_answer_tokens"] == N_RAW_TOKENS
    assert manifest["all_records_validated"] and not manifest["labels_used"]
    assert not manifest["trained"] and not manifest["test_opened"]
    assert manifest["answer_order"] == [row["response_id"] for row in q.lines(FEATURE_INPUTS)]
    return done, manifest


def label_free_rows():
    rows = q.lines(FEATURE_INPUTS)
    assert len(rows) == N_ANSWERS
    allowed_label_key = "labels_used"
    for row in rows:
        forbidden = [key for key in row if
                     ("label" in key.lower() and key != allowed_label_key)
                     or "risk" in key.lower() or "gold" in key.lower()]
        assert not forbidden, (row["response_id"], forbidden)
        assert row[allowed_label_key] is False
        assert row["partition"] in ("fit", "calibration")
    assert [row["partition"] for row in rows] == (
        ["fit"] * N_FIT_ANSWERS + ["calibration"] * (N_ANSWERS - N_FIT_ANSWERS)
    )
    fit_groups = {row["group_id"] for row in rows[:N_FIT_ANSWERS]}
    cal_groups = {row["group_id"] for row in rows[N_FIT_ANSWERS:]}
    assert len(fit_groups) == 615 and len(cal_groups) == 154 and fit_groups.isdisjoint(cal_groups)
    return rows


def eligible_windows(row):
    text = row["original_response"]
    token_ids = row["answer_token_ids"]
    offsets = row["response_token_offsets"]
    positions = row["original_answer_positions"]
    n = len(token_ids)
    assert n > 0 and len(offsets) == len(positions) == n
    lexical = []
    for left, right in offsets:
        assert 0 <= left < right <= len(text)
        lexical.append(any(char.isalnum() for char in text[left:right]))
    windows = []
    for start in range(max(1, n - K + 1)):
        end = min(start + K, n)
        indices = list(range(start, end))
        if any(lexical[index] for index in indices):
            windows.append(indices)
    assert windows
    return windows


def token_columns(values):
    assert values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 7
    assert np.isfinite(values[:, :3]).all()
    assert np.array_equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
    return np.column_stack((values[:, 2], values[:, 0], -values[:, 1])).astype(np.float64)


def map_stage():
    ensure_new_output()
    done, manifest = label_free_feature_gate()
    rows = label_free_rows()
    windows = np.empty((N_WINDOWS, len(METHODS)), dtype=np.float64)
    answers = np.empty((N_ANSWERS, len(METHODS)), dtype=np.float64)
    native_answer_means = np.empty_like(answers)
    answer_window_offsets = np.empty((N_ANSWERS, 2), dtype=np.int64)
    window_ids = []
    cursor = 0
    raw_tokens = 0
    partition_windows = Counter()

    for index, (row, entry) in enumerate(zip(rows, manifest["entries"])):
        rid = row["response_id"]
        assert entry["record_index"] == str(index) and entry["response_id"] == rid
        assert entry["record_sha256"] == q.digest(row)
        assert entry["labels_used"] is False and entry["trained"] is False
        path = FEATURES / "features" / entry["file"]
        assert path.name == f"{index:05d}.npz" and q.sha(path) == entry["npz_sha256"]
        assert q.read(path.with_suffix(".json")) == entry
        with np.load(path, allow_pickle=False) as source:
            assert str(source["response_id"].item()) == rid
            assert str(source["record_index"].item()) == str(index)
            assert str(source["record_sha256"].item()) == q.digest(row)
            assert np.array_equal(source["token_ids"], np.asarray(row["answer_token_ids"]))
            assert np.array_equal(source["token_start"], np.asarray(row["response_token_offsets"])[:, 0])
            assert np.array_equal(source["token_end"], np.asarray(row["response_token_offsets"])[:, 1])
            scores = token_columns(source["lumina_features"])
        geometry = eligible_windows(row)
        left, right = cursor, cursor + len(geometry)
        answer_window_offsets[index] = [left, right]
        mapped = np.asarray([scores[indices].mean(axis=0, dtype=np.float64)
                             for indices in geometry], dtype=np.float64)
        windows[left:right] = mapped
        answers[index] = mapped.max(axis=0)
        native_answer_means[index] = scores.mean(axis=0, dtype=np.float64)
        window_ids.extend(f"{rid}__k4_{indices[0]:05d}" for indices in geometry)
        cursor = right
        raw_tokens += len(scores)
        partition_windows[row["partition"]] += len(geometry)

    assert cursor == N_WINDOWS and raw_tokens == N_RAW_TOKENS
    assert partition_windows == {"fit": N_FIT_WINDOWS, "calibration": N_CAL_WINDOWS}
    assert len(window_ids) == len(set(window_ids)) == N_WINDOWS
    assert np.isfinite(windows).all() and np.isfinite(answers).all()
    assert np.isfinite(native_answer_means).all()

    np.savez_compressed(
        OUT / "adapted_scores.npz",
        method_names=np.asarray(METHODS),
        window_scores=windows,
        answer_scores=answers,
        native_answer_mean_scores=native_answer_means,
        response_ids=np.asarray([row["response_id"] for row in rows]),
        window_ids=np.asarray(window_ids),
        answer_window_offsets=answer_window_offsets,
    )

    replay = {}
    for method_index, method in enumerate(METHODS):
        with np.load(LEGACY / f"{method}_scores.npz", allow_pickle=False) as old:
            checks = {
                "window_scores": np.array_equal(windows[:, method_index], old["window_scores"]),
                "answer_scores": np.array_equal(answers[:, method_index], old["answermax_scores"]),
                "native_answer_mean_scores": np.array_equal(
                    native_answer_means[:, method_index], old["paper_answer_mean_scores"]),
            }
        assert all(checks.values()), (method, checks)
        replay[method] = {**checks, "all_values_exact": True,
                          "legacy_scores_sha256": q.sha(LEGACY / f"{method}_scores.npz")}

    source_paths = [Path(__file__), FEATURE_INPUTS, FEATURES / "features_complete.json",
                    FEATURES / "feature_manifest.json", *[LEGACY / f"{name}_scores.npz" for name in METHODS]]
    q.save(OUT / "mapping_complete.json", {
        "status": "complete_label_free_mapping_frozen_before_gold",
        "counts": {"answers": N_ANSWERS, "fit_answers": N_FIT_ANSWERS,
                   "calibration_answers": N_ANSWERS - N_FIT_ANSWERS,
                   "windows": N_WINDOWS, "fit_windows": N_FIT_WINDOWS,
                   "calibration_windows": N_CAL_WINDOWS, "raw_tokens": N_RAW_TOKENS},
        "fit_groups": 615,
        "calibration_groups": 154,
        "group_overlap": 0,
        "legacy_exact_replay": replay,
        "source_files_sha256": {str(path.resolve()): q.sha(path) for path in source_paths},
        "protocol_sha256": q.sha(OUT / "protocol.json"),
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "gold_bearing_files_opened": False,
        "labels_accessed": False,
        "model_loaded": False,
        "new_model_fits": 0,
        "official_test_opened": False,
        "feature_complete_sha256": q.sha(FEATURES / "features_complete.json"),
        "feature_signature_sha256": done["signature_sha256"],
    })
    print("LUMINA_V2_LABEL_FREE_MAPPING_COMPLETE", flush=True)


def verify_mapping():
    complete = q.read(OUT / "mapping_complete.json")
    assert complete["status"] == "complete_label_free_mapping_frozen_before_gold"
    assert complete["gold_bearing_files_opened"] is False and complete["labels_accessed"] is False
    assert complete["protocol_sha256"] == q.sha(OUT / "protocol.json")
    assert complete["adapted_scores_sha256"] == q.sha(OUT / "adapted_scores.npz")
    for path, expected in complete["source_files_sha256"].items():
        assert q.sha(path) == expected, path
    assert q.read(OUT / "protocol.json") == protocol()
    return complete


def gold_rows():
    answers = q.lines(ROOT / "fit_expansion/data/answers_fit.jsonl")
    answers += q.lines(ROOT / "data/answers_calibration.jsonl")
    windows = q.lines(ROOT / "fit_expansion/data/windows_k4_fit.jsonl")
    windows += q.lines(ROOT / "data/windows_k4_calibration.jsonl")
    assert len(answers) == N_ANSWERS and len(windows) == N_WINDOWS
    return answers, windows


def score_stage():
    mapping = verify_mapping()
    assert not (OUT / "complete.json").exists(), "Refusing to overwrite completed evaluation"
    answers, windows = gold_rows()  # First gold access, after adapted_scores is hash-frozen.
    with np.load(OUT / "adapted_scores.npz", allow_pickle=False) as source:
        method_names = source["method_names"].tolist()
        scores = source["window_scores"].copy()
        answer_scores = source["answer_scores"].copy()
        response_ids = source["response_ids"].tolist()
        window_ids = source["window_ids"].tolist()
    assert method_names == list(METHODS)
    assert response_ids == [row["response_id"] for row in answers]
    assert window_ids == [row["window_id"] for row in windows]
    answer_labels = np.asarray([row["label"] for row in answers], dtype=np.int8)
    window_labels = np.asarray([row["label"] for row in windows], dtype=np.int8)
    assert answer_labels[N_FIT_ANSWERS:].sum() == 100
    assert window_labels[N_FIT_WINDOWS:].sum() == 5984

    all_results = {}
    for method_index, method in enumerate(METHODS):
        ws = scores[:, method_index]
        aa = answer_scores[:, method_index]
        thresholds = {
            "window": q.choose_threshold(window_labels[N_FIT_WINDOWS:], ws[N_FIT_WINDOWS:]),
            "answer": q.choose_threshold(answer_labels[N_FIT_ANSWERS:], aa[N_FIT_ANSWERS:]),
        }
        all_results[method] = {
            "thresholds_calibration_F1Opt": thresholds,
            "calibration": {
                "windows": q.count(window_labels[N_FIT_WINDOWS:], ws[N_FIT_WINDOWS:],
                                   thresholds["window"]["threshold"]),
                "answers": q.count(answer_labels[N_FIT_ANSWERS:], aa[N_FIT_ANSWERS:],
                                   thresholds["answer"]["threshold"]),
            },
        }

    old_lumina = q.read(LEGACY / "lumina_result.json")
    assert all_results["lumina"]["calibration"] == old_lumina["window_then_answermax_metrics"]["calibration"]
    assert all_results["lumina"]["thresholds_calibration_F1Opt"] == old_lumina["thresholds"]
    summary = {
        "status": "complete_calibration_F1Opt_development_only",
        "role": protocol()["role"],
        "mapping": protocol()["mapping"],
        "gold_isolation": protocol()["gold_isolation"],
        "primary": "lumina",
        "all_fixed_readouts": all_results,
        "shared_counts": {"calibration_answers": 159, "calibration_groups": 154,
                          "calibration_windows": N_CAL_WINDOWS,
                          "positive_answers": 100, "positive_windows": 5984},
        "legacy_primary_metrics_exact": True,
        "legacy_all_score_values_exact": True,
        "mapping_complete_sha256": q.sha(OUT / "mapping_complete.json"),
        "adapted_scores_sha256": mapping["adapted_scores_sha256"],
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
    }
    q.save(OUT / "summary.json", summary)
    primary = all_results["lumina"]["calibration"]
    report = (
        "# LUMINA 统一评测适配 v2\n\n"
        "映射阶段只读取无标签输入、冻结 token 特征和历史分数数组；映射结果落盘并绑定哈希后，"
        "评测阶段才读取人工标签。新适配逐值复放旧 LUMINA/IPR/−MMD 三组分数，全部完全相等；"
        "模型、公式和既有特征均未改变。\n\n"
        "| 粒度 | AUROC | AP | F1 |\n|---|---:|---:|---:|\n"
        f"| 4-BPE 窗口 | {primary['windows']['auroc']:.6f} | "
        f"{primary['windows']['average_precision']:.6f} | {primary['windows']['f1']:.6f} |\n"
        f"| 回答（窗口 max） | {primary['answers']['auroc']:.6f} | "
        f"{primary['answers']['average_precision']:.6f} | {primary['answers']['f1']:.6f} |\n\n"
        "口径为同一 cal159/154 组/42,241 窗口；阈值分别按 F1、precision、较高阈值选取。"
        "这些仍是反复使用的开发集结果，official test 未读取。\n"
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    q.save(OUT / "complete.json", {
        "status": "complete",
        "files_sha256": {name: q.sha(OUT / name) for name in
                         ("protocol.json", "adapted_scores.npz", "mapping_complete.json",
                          "summary.json", "REPORT.md")},
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    print("LUMINA_V2_COMMON_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("map", "score", "check"))
    stage = parser.parse_args().stage
    {"map": map_stage, "score": score_stage, "check": verify_mapping}[stage]()
