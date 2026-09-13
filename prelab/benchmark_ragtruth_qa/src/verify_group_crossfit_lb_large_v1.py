"""Independent arithmetic and wiring replay for group-crossfit LB+large."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q  # noqa: E402


OUT = ROOT / "results/group_crossfit_lb_large_v1"
WINDOW_MASS = 168123


def exact_nested(left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact_nested(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_nested(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return bool(np.isclose(left, right, rtol=0, atol=1e-15, equal_nan=True))
    return left == right


def main():
    assert (OUT / "complete.json").exists() and (OUT / "summary.json").exists()
    meta = q.metadata()
    folds = q.read(OUT / "folds.json")
    assert len(folds) == 3
    hold_answers = np.zeros(3680, dtype=np.int8)
    hold_windows = np.zeros(WINDOW_MASS, dtype=np.int8)
    for fold in folds:
        assert not set(fold["fit_groups"]) & set(fold["hold_groups"])
        hold_answers[np.asarray(fold["hold_answer_indices"], int)] += 1
        hold_windows[np.asarray(fold["hold_window_indices"], int)] += 1
    assert np.all(hold_answers == 1) and np.all(hold_windows == 1)

    old = np.load(OUT / "old_input_scores.npy")
    saved_oof = np.load(OUT / "oof_input_scores.npy")
    assert old.shape == saved_oof.shape == (210364, 2)
    replay = old.copy()
    replay[:WINDOW_MASS] = np.nan
    for fold in folds:
        folder = OUT / f"fold_{fold['fold']}"
        hold = np.asarray(fold["hold_window_indices"], int)
        with np.load(folder / "lb_hold_scores.npz", allow_pickle=False) as data:
            assert np.array_equal(data["window_indices"], hold)
            replay[hold, 0] = data["scores"]
        with np.load(folder / "large_hold_token_probabilities.npz", allow_pickle=False) as data:
            for window_index in hold:
                window = meta["windows"][int(window_index)]
                replay[window_index, 1] = np.max(
                    data[window["response_id"]][np.asarray(window["lexical_token_indices"], int)]
                )
    assert np.isfinite(replay).all()
    assert np.array_equal(replay, saved_oof)
    assert np.array_equal(replay[WINDOW_MASS:], old[WINDOW_MASS:])

    summary = q.read(OUT / "summary.json")
    audits = {}
    for name, design in (("old_in_sample", old), ("group_oof", saved_oof)):
        saved = summary["methods"][name]
        model = pickle.loads((OUT / f"{name}.pkl").read_bytes())
        predicted = model.predict_proba(design)[:, 1]
        with np.load(OUT / f"{name}_scores.npz", allow_pickle=False) as data:
            window_scores = data["window_scores"].copy()
            answer_scores = data["answer_scores"].copy()
        max_probability_error = float(np.max(np.abs(predicted - window_scores)))
        assert max_probability_error <= 1e-15
        replay_answers = q.answer_scores(meta, window_scores)
        assert np.array_equal(replay_answers, answer_scores)
        thresholds = {
            "window": q.choose_threshold(
                [window["label"] for window in meta["windows"][WINDOW_MASS:]],
                window_scores[WINDOW_MASS:],
            ),
            "answer": q.choose_threshold(
                [answer["label"] for answer in meta["answers"][634:]],
                answer_scores[634:],
            ),
        }
        metrics = q.metrics(meta, window_scores, thresholds)
        assert exact_nested(thresholds, saved["thresholds"])
        assert exact_nested(metrics, saved["metrics"])
        audits[name] = {
            "saved_model_probability_max_abs_error": max_probability_error,
            "answer_max_exact": True,
            "thresholds_exact": True,
            "metrics_exact": True,
            "calibration_windows_f1": metrics["calibration"]["windows"]["f1"],
            "calibration_answers_f1": metrics["calibration"]["answers"]["f1"],
        }

    result = {
        "status": "passed_independent_arithmetic_and_wiring_replay",
        "fold_answer_coverage_exactly_once": True,
        "fold_window_coverage_exactly_once": True,
        "fit_hold_groups_disjoint": True,
        "oof_columns_reconstructed_exactly": True,
        "calibration_columns_same_between_modes_exactly": True,
        "methods": audits,
        "official_test_opened": False,
    }
    q.save(OUT / "INDEPENDENT_AUDIT.json", result)
    (OUT / "INDEPENDENT_AUDIT.md").write_text(
        "# Group-crossfit LB+large independent audit\n\n"
        "三折资料组隔离、一次覆盖、OOF两列、已存模型概率、整答max、cal阈值和全部指标均独立复算通过。"
        "仅使用开发fit/calibration，未读取官方test。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
