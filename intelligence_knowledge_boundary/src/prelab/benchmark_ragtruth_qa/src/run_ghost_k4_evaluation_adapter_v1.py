"""Map the frozen whole-answer GHOST score to the project's common QA metrics.

This is an evaluation adapter only.  It never changes, refits, calibrates, or
otherwise post-processes the frozen GHOST random forest.  A whole-answer score
is copied unchanged to every eligible four-BPE window in that answer so that a
method without a localization output is still measured on the localization
task required by this project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import run_development as q


ROOT = q.ROOT
SOURCE = ROOT / "results/ghost_official_answer_rf_v1"
OUT = ROOT / "results/ghost_k4_evaluation_adapter_v1"
ANSWER_FILE = SOURCE / "answer_scores.npz"
ORDER_FILE = SOURCE / "answer_order.json"


def protocol():
    return {
        "version": "ghost-frozen-answer-to-common-k4-evaluation-adapter-v1",
        "role": "evaluation_adapter_for_frozen_ghost_baseline; no model change",
        "frozen_method": (
            "The existing four whole-answer feature means and the existing 750-tree "
            "GHOST-structure RF probability are byte-bound and remain unchanged."
        ),
        "mapping": (
            "For each common answer, copy its one frozen RF risk probability unchanged "
            "to every eligible four-original-BPE stride-one window belonging to that answer."
        ),
        "why": (
            "The project requires both answer detection and local-window localization. "
            "Uniform broadcast exposes, rather than repairs, the native answer-only granularity."
        ),
        "mapping_parameters": 0,
        "mapping_training": False,
        "labels_in_mapping": False,
        "thresholds": (
            "Same common development reporting as the current formal table: separate "
            "calibration-F1Opt thresholds for window and answer; AUROC/AP are threshold-free."
        ),
        "scope": "Original native 634 fit plus 159 calibration answers; 210364 common windows.",
        "limits": (
            "Broadcast is a model-external evaluation convention, not a claim that GHOST "
            "natively localizes tokens. The calibration set is repeatedly used development data."
        ),
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
    }


def verify_frozen_source():
    complete = q.read(SOURCE / "complete.json")
    assert complete["status"] == "complete" and complete["real_fits"] == 1
    for name, expected in complete["files_sha256"].items():
        assert q.sha(SOURCE / name) == expected, name
    prepared = q.read(SOURCE / "preparation_complete.json")
    assert prepared["status"] == "CPU_ready_not_fitted"
    for name, expected in prepared["files_sha256"].items():
        assert q.sha(SOURCE / name) == expected, name
    summary = q.read(SOURCE / "summary.json")
    assert summary["method"] == "ghost_paper_structure_answer_rf"
    assert summary["real_fits"] == 1 and summary["parameters"]["n_estimators"] == 750
    assert summary["scores_sha256"] == q.sha(ANSWER_FILE)
    return complete, prepared, summary


def frozen_answer_lookup():
    order = q.read(ORDER_FILE)
    with np.load(ANSWER_FILE, allow_pickle=False) as data:
        values = data["answer_scores"].astype(np.float64, copy=True)
    assert len(order) == len(values) == 3839
    identifiers = [row["response_id"] for row in order]
    assert len(set(identifiers)) == len(identifiers)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return dict(zip(identifiers, map(float, values)))


def mapped_scores(meta):
    lookup = frozen_answer_lookup()
    answer_ids = [row["response_id"] for row in meta["answers"]]
    assert len(answer_ids) == 793 and len(set(answer_ids)) == 793
    answer_scores = np.asarray([lookup[rid] for rid in answer_ids], dtype=np.float64)
    answer_position = {rid: i for i, rid in enumerate(answer_ids)}
    window_scores = np.asarray(
        [answer_scores[answer_position[row["response_id"]]] for row in meta["windows"]],
        dtype=np.float64,
    )
    assert len(window_scores) == 210364
    for answer in meta["answers"]:
        ix = meta["answer_windows"][answer["response_id"]]
        assert ix and np.all(window_scores[ix] == lookup[answer["response_id"]])
    return answer_scores, window_scores


def prepare():
    assert not OUT.exists(), "Preserve any prior evaluation adapter output."
    OUT.mkdir(parents=True)
    verify_frozen_source()
    meta = q.metadata()
    answer_scores, window_scores = mapped_scores(meta)
    q.save(OUT / "protocol.json", protocol())
    np.savez_compressed(
        OUT / "adapted_scores.npz",
        answer_scores=answer_scores,
        window_scores=window_scores,
        response_ids=np.asarray([row["response_id"] for row in meta["answers"]]),
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
    )
    source_paths = [
        Path(__file__), Path(q.__file__), SOURCE / "complete.json",
        SOURCE / "preparation_complete.json", SOURCE / "summary.json",
        SOURCE / "answer_rf.pkl", ANSWER_FILE, ORDER_FILE,
        ROOT / "data/answers_fit.jsonl", ROOT / "data/answers_calibration.jsonl",
        ROOT / "data/windows_k4_fit.jsonl", ROOT / "data/windows_k4_calibration.jsonl",
    ]
    q.save(OUT / "source_snapshot.json", {
        "files_sha256": {str(path.resolve()): q.sha(path) for path in source_paths},
        "frozen_baseline_score_sha256": q.sha(ANSWER_FILE),
        "frozen_baseline_model_sha256": q.sha(SOURCE / "answer_rf.pkl"),
        "labels_or_metrics_used_to_change_mapping": False,
        "new_model_fits": 0,
        "official_test_opened": False,
    })
    text = (
        "# GHOST统一4-BPE评测适配\n\n"
        "冻结GHOST整答RF及其概率完全不变。为执行本项目统一的定位评测，"
        "把每条回答的同一个冻结概率原值复制到该回答所有4-BPE窗口；该映射无参数、"
        "不训练、不读取标签来决定位置。它会如实暴露整答模型无法定位局部错误。\n\n"
        "随后才按与正式对照表相同的cal-F1Opt口径分别计算窗口和整答指标。"
        "本适配器不是GHOST模型的一部分，也不把GHOST写成原生词元检测器。\n"
    )
    (OUT / "PLAN.md").write_text(text, encoding="utf-8")
    names = ["protocol.json", "adapted_scores.npz", "source_snapshot.json", "PLAN.md"]
    q.save(OUT / "preparation_complete.json", {
        "status": "mapped_before_metric_scoring",
        "answers": len(answer_scores),
        "windows": len(window_scores),
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "mapping_parameters": 0,
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
    })
    print("GHOST_COMMON_MAPPING_PREPARED", flush=True)


def check_prepared():
    verify_frozen_source()
    done = q.read(OUT / "preparation_complete.json")
    assert done["status"] == "mapped_before_metric_scoring"
    for name, expected in done["files_sha256"].items():
        assert q.sha(OUT / name) == expected, name
    for name, expected in q.read(OUT / "source_snapshot.json")["files_sha256"].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / "protocol.json") == protocol()
    meta = q.metadata()
    answer_scores, window_scores = mapped_scores(meta)
    with np.load(OUT / "adapted_scores.npz", allow_pickle=False) as data:
        assert np.array_equal(data["answer_scores"], answer_scores)
        assert np.array_equal(data["window_scores"], window_scores)
        assert data["response_ids"].tolist() == [row["response_id"] for row in meta["answers"]]
        assert data["window_ids"].tolist() == [row["window_id"] for row in meta["windows"]]
    return meta, answer_scores, window_scores


def score():
    assert not (OUT / "summary.json").exists(), "Refusing to overwrite scored output."
    meta, answer_scores, window_scores = check_prepared()
    lo, hi = meta["bounds"]["calibration"]
    thresholds = {
        "window": q.choose_threshold(
            [row["label"] for row in meta["windows"][lo:hi]], window_scores[lo:hi]
        ),
        "answer": q.choose_threshold(
            [row["label"] for row in meta["answers"][634:]], answer_scores[634:]
        ),
    }
    metrics = q.metrics(meta, window_scores, thresholds)
    original = q.read(SOURCE / "summary.json")
    assert metrics["calibration"]["answers"] == original["answer_metrics"]["calibration"]
    summary = {
        "status": "complete_calibration_F1Opt_development_only",
        "role": protocol()["role"],
        "mapping": protocol()["mapping"],
        "thresholds_calibration_F1Opt": thresholds,
        "common_metrics": metrics,
        "native_answer_result_preserved_unchanged": original["answer_metrics"]["calibration"],
        "frozen_baseline_scores_sha256": q.sha(ANSWER_FILE),
        "frozen_baseline_model_sha256": q.sha(SOURCE / "answer_rf.pkl"),
        "adapted_scores_sha256": q.sha(OUT / "adapted_scores.npz"),
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "calibration_used_for_threshold": True,
        "official_test_opened": False,
        "final_test_claim": False,
    }
    q.save(OUT / "summary.json", summary)
    window = metrics["calibration"]["windows"]
    answer = metrics["calibration"]["answers"]
    report = (
        "# GHOST冻结整答模型：统一评测结果\n\n"
        "GHOST原RF、四维特征、训练和整答概率均未改变。统一评测器只把每答概率"
        "原值广播到其所有4-BPE窗口，然后按本项目同一cal-F1Opt规则计分。\n\n"
        "| 统一粒度 | AUROC | AP | F1 |\n|---|---:|---:|---:|\n"
        f"| 4-BPE窗口 | {window['auroc']:.6f} | {window['average_precision']:.6f} | {window['f1']:.6f} |\n"
        f"| 整答 | {answer['auroc']:.6f} | {answer['average_precision']:.6f} | {answer['f1']:.6f} |\n\n"
        "窗口广播没有赋予模型局部定位能力；它只是让整答基线接受本项目要求的同一评测。"
        "结果仍是反复使用的开发校准集，官方测试未读取。\n"
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    names = ["summary.json", "REPORT.md"]
    q.save(OUT / "complete.json", {
        "status": "complete",
        "files_sha256": {name: q.sha(OUT / name) for name in names},
        "preparation_sha256": q.sha(OUT / "preparation_complete.json"),
        "new_model_fits": 0,
        "baseline_parameters_changed": False,
        "official_test_opened": False,
    })
    print("GHOST_COMMON_EVALUATION_COMPLETE", window["f1"], answer["f1"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "check", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "check": check_prepared, "score": score}[args.stage]()
