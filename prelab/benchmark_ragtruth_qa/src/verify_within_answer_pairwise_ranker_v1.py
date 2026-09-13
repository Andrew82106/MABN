"""Independent arithmetic replay for within_answer_pairwise_ranker_v1."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import GroupKFold

import run_development as q
import run_within_answer_pairwise_ranker_v1 as r


OUT = r.OUT


def close_dict(a, b, path="root", atol=1e-12):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            close_dict(a[k], b[k], f"{path}.{k}", atol)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close_dict(x, y, f"{path}[{i}]", atol)
    elif isinstance(a, float):
        assert abs(a - float(b)) <= atol, (path, a, b)
    else:
        assert a == b, (path, a, b)


def main():
    prep = q.read(OUT / "preparation_complete.json")
    complete = q.read(OUT / "complete.json")
    summary = q.read(OUT / "summary.json")
    freeze = q.read(OUT / "fit_freeze_before_calibration.json")
    assert complete["summary_sha256"] == q.sha(OUT / "summary.json")
    assert complete["scores_sha256"] == q.sha(OUT / "scores.npz")
    assert q.read(OUT / "protocol.json") == r.protocol()
    for name, h in prep["files_sha256"].items():
        assert q.sha(OUT / name) == h, name
    for name, h in q.read(OUT / "source_binding.json")["files_sha256"].items():
        assert q.sha(name) == h, name
    for name, h in freeze["models_sha256"].items():
        assert q.sha(OUT / name) == h, name

    x = np.load(OUT / "window_features.npy", mmap_mode="r")
    with np.load(OUT / "pairs.npz") as z:
        differences = z["differences"]; pair_y = z["labels"]; pair_w = z["weights"]
    pair_index = q.read(OUT / "pair_index.json")
    assert len(pair_index) == len(differences) == 9780
    assert np.array_equal(differences[0::2], -differences[1::2])
    assert np.array_equal(pair_y[0::2], 1-pair_y[1::2])
    assert np.array_equal(pair_w[0::2], pair_w[1::2])

    # Recheck the hierarchical mass definition from saved pair metadata.
    group_mass = defaultdict(float); answer_mass = defaultdict(float); run_mass = defaultdict(float)
    for item, weight in zip(pair_index, pair_w):
        g = item["group_id"]; a = item["response_id"]; run = item["run_id"]
        group_mass[g] += float(weight); answer_mass[(g, a)] += float(weight); run_mass[(g, a, run)] += float(weight)
    assert max(group_mass.values()) - min(group_mass.values()) < 1e-12
    for g in group_mass:
        vals = [v for (gg, _), v in answer_mass.items() if gg == g]
        assert max(vals) - min(vals) < 1e-12
    for g, a in answer_mass:
        vals = [v for (gg, aa, _), v in run_mass.items() if gg == g and aa == a]
        assert max(vals) - min(vals) < 1e-12

    meta = q.metadata()
    y = np.asarray([w["label"] for w in meta["windows"]], np.int8)
    groups = np.asarray([w["group_id"] for w in meta["windows"][:r.NFIT]])
    pair_groups = np.asarray([p["group_id"] for p in pair_index])
    replay_oof = np.full(r.NFIT, np.nan)
    fold_details = []
    for fold, (_, held) in enumerate(GroupKFold(r.FOLDS).split(np.arange(r.NFIT), y[:r.NFIT], groups)):
        held_groups = set(groups[held].tolist())
        assert not set(pair_groups[[g not in held_groups for g in pair_groups]]) & held_groups
        scaler = pickle.loads((OUT / f"fold_{fold}_scaler.pkl").read_bytes())
        model = pickle.loads((OUT / f"fold_{fold}_model.pkl").read_bytes())
        replay_oof[held] = r.decision(scaler, model, x[held])
        fold_details.append({"fold": fold, "held_windows": len(held), "held_groups": len(held_groups)})
    scaler = pickle.loads((OUT / "full_scaler.pkl").read_bytes())
    model = pickle.loads((OUT / "full_model.pkl").read_bytes())
    replay_score = r.decision(scaler, model, x)
    with np.load(OUT / "scores.npz") as z:
        stored_oof = z["fit_oof_window_scores"]
        stored_score = z["window_scores"]
        stored_answers = z["derived_answer_scores"]
        stored_current_answers = z["preserved_current_answer_scores"]
    max_oof = float(np.max(np.abs(replay_oof-stored_oof)))
    max_score = float(np.max(np.abs(replay_score-stored_score)))
    assert max_oof <= 1e-12 and max_score <= 1e-12
    replay_answers = q.answer_scores(meta, replay_score)
    assert np.array_equal(replay_answers, stored_answers)

    lo, hi = meta["bounds"]["calibration"]
    wt = freeze["window_threshold"]["threshold"]
    cal_opt = summary["thresholds"]["common_cal_F1Opt_reporting_only"]["threshold"]
    checks = {
        "fit_OOF": r.count(y[:r.NFIT], replay_oof, wt),
        "cal_fit_threshold": r.count(y[lo:hi], replay_score[lo:hi], wt),
        "cal_common_F1Opt": r.count(y[lo:hi], replay_score[lo:hi], cal_opt),
    }
    close_dict(checks["fit_OOF"], summary["fit_OOF"])
    close_dict(checks["cal_fit_threshold"], summary["calibration"]["fit_threshold_window"])
    close_dict(checks["cal_common_F1Opt"], summary["calibration"]["common_evaluator_F1Opt_window"])
    current = np.load(r.SOURCES["current"])["answer_scores"]
    assert np.array_equal(current, stored_current_answers)
    assert summary["calibration_used_for_model_or_variant_selection"] is False
    assert complete["official_test_opened"] is False

    report = {
        "status": "independent_replay_passed",
        "source_and_model_hashes_exact": True,
        "pair_reverse_and_hierarchical_mass_exact": True,
        "group_fold_isolation_exact": True,
        "fold_details": fold_details,
        "max_abs_OOF_score_difference": max_oof,
        "max_abs_full_score_difference": max_score,
        "answer_max_exact": True,
        "metric_replay_exact_at_1e-12": True,
        "calibration_used_for_model_or_variant_selection": False,
        "official_test_opened": False,
    }
    q.save(OUT / "INDEPENDENT_VERIFY.json", report)
    print("PAIRWISE_RANKER_INDEPENDENT_VERIFY_PASSED", report, flush=True)


if __name__ == "__main__":
    main()
