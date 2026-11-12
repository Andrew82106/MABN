"""Group-cross-fitted within-answer pairwise ranker over completed scores.

This is an ours-only development experiment.  It never reads the official
test split and never changes any paper baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_development as q


ROOT = q.ROOT
OUT = ROOT / "results/within_answer_pairwise_ranker_v1"
NFIT = 168123
NANS_FIT = 634
SEED = 20261013
FOLDS = 5
C = 0.01

SOURCES = {
    "current": ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz",
    "large": ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight1_scores.npz",
    "nli_init": ROOT / "results/nli_fixed_convex_v1/semantic_claim__old_tree__nli_weight1_scores.npz",
    "fava": ROOT / "results/fava_fixed_convex_v1/semantic_claim__old_tree__fava_weight1_scores.npz",
    "tail2": ROOT / "results/minicheck_tail_all_docs_v3/tail2/epoch_02_scores.npz",
    "semantic_claim": ROOT / "results/claim_pooling_v1/minicheck_hidden64_risk_tcn_w32_alpha0.5_scores.npz",
    "harp_claim": ROOT / "results/claim_pooling_v1/full_lb_harp64_tcn_alpha0.75_scores.npz",
    "lookback": ROOT / "results/lookback_regularization_v2/lb_prefix_pre_header_C0.0001_scores.npz",
    "minicheck": ROOT / "semantic_baseline/cuda_variant/results/minicheck_calibrated_scores.npz",
    "local_nli": ROOT / "results/nli_local_signal_cuda_scoring_v1/predictions.npz",
    "raw_nll": ROOT / "results/ghost_nll_nonlinear_v1/window_features.npy",
}


def protocol():
    return {
        "version": "within-answer-pairwise-ranker-v1",
        "identity": "ours_method_candidate; never a baseline",
        "question": "Can within-answer risk ranking remove the main localization error that whole-answer gating cannot remove?",
        "scope": {
            "fit": "Original native 634 answers, 615 material-connected groups and 168123 eligible 4-raw-BPE windows.",
            "calibration": "Original 159 answers and 42241 windows, evaluated once only after model and fit threshold freeze.",
            "test": "Official test is never read.",
        },
        "inputs": list(SOURCES),
        "input_semantics": {
            "absolute": "Clipped logit of each completed risk score.",
            "relative": "Within-answer mid-rank of each score, centered at 0.5; ties receive their average rank.",
            "direction": "Every source is oriented so larger means more hallucination risk.",
            "reuse": "No new language-model inference and no source model refit.",
        },
        "pairs": {
            "unit": "Each contiguous positive-window run inside a risky fit answer.",
            "positive_representatives": "First, middle and last window of the run, deduplicated.",
            "negative_representatives": "Nearest negative on the left, nearest negative on the right and highest-current-score negative in the answer, deduplicated.",
            "rows": "Cartesian product of those representatives plus its reversed copy.",
            "weight": "Equal total mass per material group, then answer, then positive run, then unordered pair; reverse copies split the pair mass equally.",
            "reason": "Directly learns which detail is riskier inside the same response and avoids counting thousands of overlapping windows as independent examples.",
        },
        "model": {
            "type": "StandardScaler + LogisticRegression",
            "C": C,
            "penalty": "L2",
            "solver": "liblinear",
            "fit_intercept": False,
            "max_iter": 2000,
            "random_state": SEED,
            "feature_width": 2 * len(SOURCES),
        },
        "crossfit": {
            "folds": FOLDS,
            "split": "GroupKFold by material-connected group; every fit answer/window gets one OOF score.",
            "selection": "The sole window threshold is selected from all fit OOF window scores. No C, source, pair, fold or feature search.",
            "final": "One same-configuration full-fit model scores calibration after the model and OOF threshold are saved and hashed.",
        },
        "answer": {
            "primary_two_head": "Keep the already-fixed current answer score and threshold exactly; the new ranker changes localization only.",
            "secondary": "Also report max ranker score per answer with the fit-OOF-derived answer threshold.",
        },
        "known_limit": "The meta ranker is group-cross-fitted, but most cached upstream fit scores were themselves trained in-sample. The unfinished upstream large crossfit must still be completed for a fully honest stack.",
        "not_a_conflict_detector": "This v1 adds no new source-conditioned evidence. It is a localization/overfitting diagnostic and cannot by itself repair the very low recall on explicit or coverage-gap Evident Conflict.",
        "GPU_used": False,
        "official_test_opened": False,
    }


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def average_midranks(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = ((i + j - 1) / 2 + .5) / len(values)
        i = j
    return out


def score_array(name, path):
    if name == "raw_nll":
        x = np.load(path, mmap_mode="r")
        assert x.shape == (696220, 5)
        # Expanded fit keeps the native 634 answers first; calibration follows
        # all 653979 expanded-fit windows.
        return np.r_[np.asarray(x[:NFIT, 0], np.float64),
                     np.asarray(x[653979:, 0], np.float64)]
    with np.load(path, allow_pickle=False) as z:
        if name == "tail2":
            return np.r_[z["fit_window_scores"][:NFIT], z["cal_window_scores"]]
        if name == "local_nli":
            return z["nli_only_fixed_lr"].astype(np.float64)
        return z["window_scores"].astype(np.float64)


def feature_matrix(meta):
    raw = []
    source_hashes = {}
    for name, path in SOURCES.items():
        assert path.exists(), path
        source_hashes[str(path.resolve())] = q.sha(path)
        v = score_array(name, path)
        assert v.shape == (210364,) and np.isfinite(v).all()
        if name != "raw_nll":
            assert ((v >= 0) & (v <= 1)).all(), name
            v = np.log(np.clip(v, 1e-6, 1 - 1e-6) / np.clip(1 - v, 1e-6, 1))
        else:
            # NLL is already an unbounded higher-is-riskier scalar.
            v = np.clip(v, np.quantile(v[:NFIT], .001), np.quantile(v[:NFIT], .999))
        raw.append(v)
    raw = np.column_stack(raw)
    relative = np.empty_like(raw)
    cursor = 0
    for answer in meta["answers"]:
        n = answer["eligible_window_count"]
        assert n > 0
        for j in range(raw.shape[1]):
            relative[cursor:cursor+n, j] = average_midranks(raw[cursor:cursor+n, j]) - .5
        cursor += n
    assert cursor == len(raw)
    x = np.column_stack((raw, relative)).astype(np.float32)
    names = [f"{n}__absolute" for n in SOURCES] + [f"{n}__within_answer_rank" for n in SOURCES]
    return x, names, source_hashes


def runs(indices, windows):
    if not len(indices):
        return []
    out = []
    left = 0
    for j in range(1, len(indices)):
        a, b = indices[j-1], indices[j]
        if windows[b]["token_start"] != windows[a]["token_start"] + 1:
            out.append(indices[left:j]); left = j
    out.append(indices[left:])
    return out


def build_pairs(meta, x, y):
    by_answer = defaultdict(list)
    for i, w in enumerate(meta["windows"][:NFIT]):
        by_answer[w["response_id"]].append(i)
    current = x[:NFIT, list(SOURCES).index("current")]
    records = []
    groups_to_answers = defaultdict(set)
    answer_runs = {}
    for answer in meta["answers"][:NANS_FIT]:
        rid = answer["response_id"]; indices = np.asarray(by_answer[rid], np.int64)
        assert len(indices) == answer["eligible_window_count"]
        positive = indices[y[indices] == 1]
        rr = runs(positive, meta["windows"])
        if not rr:
            continue
        negative = indices[y[indices] == 0]
        # A few answers are completely covered by a gold span and therefore
        # contain no within-answer ordering pair.  They still remain in the
        # OOF threshold and final evaluation populations.
        if not len(negative):
            continue
        groups_to_answers[answer["group_id"]].add(rid)
        answer_runs[rid] = rr
        for run_id, run in enumerate(rr):
            pos = sorted(set([int(run[0]), int(run[len(run)//2]), int(run[-1])]))
            left = negative[negative < run[0]]
            right = negative[negative > run[-1]]
            neg = []
            if len(left): neg.append(int(left[-1]))
            if len(right): neg.append(int(right[0]))
            neg.append(int(negative[np.argmax(current[negative])]))
            neg = sorted(set(neg))
            for a in pos:
                for b in neg:
                    records.append((answer["group_id"], rid, run_id, a, b))
    group_mass = 1 / len(groups_to_answers)
    pair_counts = defaultdict(int)
    for group, rid, run_id, _, _ in records:
        pair_counts[(group, rid, run_id)] += 1
    rows = []; labels = []; weights = []; pair_meta = []
    for group, rid, run_id, pos, neg in records:
        nruns = len(answer_runs[rid])
        mass = group_mass / len(groups_to_answers[group]) / nruns / pair_counts[(group, rid, run_id)] / 2
        delta = x[pos] - x[neg]
        rows.extend([delta, -delta]); labels.extend([1, 0]); weights.extend([mass, mass])
        pair_meta.extend([(group, rid, run_id, pos, neg, 1), (group, rid, run_id, neg, pos, 0)])
    rows = np.asarray(rows, np.float32); labels = np.asarray(labels, np.int8); weights = np.asarray(weights, np.float64)
    assert len(rows) == len(labels) == len(weights) == len(pair_meta)
    assert set(labels) == {0, 1} and np.isfinite(rows).all() and (weights > 0).all()
    # Reverse rows are exact opposites with equal mass.
    assert np.array_equal(rows[0::2], -rows[1::2])
    assert np.array_equal(weights[0::2], weights[1::2])
    return rows, labels, weights, pair_meta


def fit_model(rows, labels, weights):
    scaler = StandardScaler().fit(rows, sample_weight=weights)
    xx = scaler.transform(rows).astype(np.float32)
    model = LogisticRegression(C=C, penalty="l2", solver="liblinear", fit_intercept=False,
                               max_iter=2000, random_state=SEED)
    model.fit(xx, labels, sample_weight=weights / weights.mean())
    assert model.n_iter_[0] < 2000 and model.classes_.tolist() == [0, 1]
    return scaler, model


def decision(scaler, model, x):
    # Pairwise P(pos>neg) is monotonic in this signed linear score.
    return model.decision_function(scaler.transform(x).astype(np.float32)).astype(np.float64)


def count(y, score, threshold):
    y = np.asarray(y, bool); p = np.asarray(score) >= threshold
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    return {"n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "auroc": float(roc_auc_score(y, score)), "average_precision": float(average_precision_score(y, score))}


def answer_scores(meta, window_scores):
    return q.answer_scores(meta, window_scores)


def prepare():
    assert not (OUT / "preparation_complete.json").exists(), "Never overwrite a prepared experiment"
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        assert q.read(OUT / "protocol.json") == protocol()
    else:
        q.save(OUT / "protocol.json", protocol())
    meta = q.metadata()
    assert meta["bounds"] == {"fit": [0, NFIT], "calibration": [NFIT, 210364]}
    x, names, source_hashes = feature_matrix(meta)
    y = np.asarray([w["label"] for w in meta["windows"]], np.int8)
    rows, labels, weights, pair_meta = build_pairs(meta, x, y)
    np.save(OUT / "window_features.npy", x)
    np.savez_compressed(OUT / "pairs.npz", differences=rows, labels=labels, weights=weights)
    q.save(OUT / "pair_index.json", [dict(group_id=g, response_id=r, run_id=run, first_window=a,
                                              second_window=b, label=lab)
                                         for g, r, run, a, b, lab in pair_meta])
    q.save(OUT / "feature_names.json", names)
    # A real tiny numerical fit, but never on project data.
    rng = np.random.default_rng(SEED)
    toy = rng.normal(size=(400, len(names))).astype(np.float32)
    toy_y = (toy[:, 0] + .5*toy[:, -1] > 0).astype(np.int8)
    toy_w = rng.uniform(.5, 1.5, len(toy))
    ss, mm = fit_model(toy, toy_y, toy_w)
    assert np.isfinite(decision(ss, mm, toy)).all()
    q.save(OUT / "CPU_SELFCHECK.json", {"status": "passed", "synthetic_rows": len(toy),
                                           "real_project_fits": 0, "feature_width": len(names)})
    bindings = {str(Path(__file__).resolve()): q.sha(Path(__file__)),
                str((ROOT / "src/run_development.py").resolve()): q.sha(ROOT / "src/run_development.py"), **source_hashes}
    q.save(OUT / "source_binding.json", {"files_sha256": bindings})
    q.save(OUT / "preparation_complete.json", {
        "status": "CPU_prepared_not_real_fit", "fit_answers": NANS_FIT, "fit_windows": NFIT,
        "calibration_windows_packaged_not_scored": 42241, "pair_rows": len(rows),
        "unordered_pairs": len(rows)//2, "positive_runs": len(set((m[1], m[2]) for m in pair_meta)),
        "feature_width": len(names), "source_sha256": source_hashes,
        "files_sha256": {p.name: q.sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "preparation_complete.json"},
        "GPU_used": False, "official_test_opened": False})
    (OUT / "PLAN.md").write_text(
        "# 同答细节成对排序器 v1\n\n"
        "当前整答头已足够强，而窗口误报主要落在本身确有错误的回答内。本实验把每个风险段的首、中、末窗口与同答最近及高分正常窗口成对，"
        "用11个已缓存信号的绝对值和同答排名训练一个固定L2线性排序器。材料组五折产生全部fit OOF分数并只在OOF上定阈值；"
        "之后冻结模型才一次性报告cal。整答主头逐值保留current，不修改正式baseline，不读取official test。\n\n"
        "该模型只解决细节排序，既不能消除上游fit分数已见本折标签的问题，也没有新增原子陈述到来源的冲突证据，不能称作冲突检测器。"
        "因此无论结果好坏，large第三折交叉拟合与source-conditioned conflict头仍是后续关键验证。\n",
        encoding="utf-8")
    print("PAIRWISE_RANKER_CPU_PREPARED_NO_REAL_FIT", q.read(OUT / "preparation_complete.json"), flush=True)


def check():
    done = q.read(OUT / "preparation_complete.json")
    assert done["status"] == "CPU_prepared_not_real_fit" and not done["official_test_opened"]
    assert q.read(OUT / "protocol.json") == protocol()
    for name, h in done["files_sha256"].items():
        assert q.sha(OUT / name) == h, name
    for name, h in q.read(OUT / "source_binding.json")["files_sha256"].items():
        assert q.sha(name) == h, name
    x = np.load(OUT / "window_features.npy", mmap_mode="r")
    with np.load(OUT / "pairs.npz") as z:
        d, lab, w = z["differences"], z["labels"], z["weights"]
        assert d.shape[1] == 2*len(SOURCES) and len(d) == done["pair_rows"]
        assert np.array_equal(d[0::2], -d[1::2]) and np.array_equal(lab[0::2], 1-lab[1::2])
        assert np.array_equal(w[0::2], w[1::2])
    assert x.shape == (210364, 2*len(SOURCES)) and np.isfinite(x).all()
    print("PAIRWISE_RANKER_CHECK_PASSED_NO_REAL_FIT", flush=True)


def train():
    check()
    assert not (OUT / "started.json").exists(), "Never silently refit"
    q.save(OUT / "started.json", {"time": time.time(), "preparation_sha256": q.sha(OUT / "preparation_complete.json")})
    meta = q.metadata(); x = np.load(OUT / "window_features.npy", mmap_mode="r")
    with np.load(OUT / "pairs.npz") as z:
        d, pair_y, pair_w = z["differences"], z["labels"], z["weights"]
    pair_index = q.read(OUT / "pair_index.json")
    pair_groups = np.asarray([p["group_id"] for p in pair_index])
    y = np.asarray([w["label"] for w in meta["windows"]], np.int8)
    win_groups = np.asarray([w["group_id"] for w in meta["windows"][:NFIT]])
    oof = np.full(NFIT, np.nan); models = []; logs = []
    splitter = GroupKFold(FOLDS)
    start = time.perf_counter()
    with threadpool_limits(4):
        for fold, (fit_ix, held_ix) in enumerate(splitter.split(np.arange(NFIT), y[:NFIT], win_groups)):
            held_groups = set(win_groups[held_ix].tolist())
            pair_fit = np.asarray([g not in held_groups for g in pair_groups])
            assert pair_fit.any() and not set(pair_groups[pair_fit]) & held_groups
            scaler, model = fit_model(d[pair_fit], pair_y[pair_fit], pair_w[pair_fit])
            oof[held_ix] = decision(scaler, model, x[held_ix])
            sp = OUT / f"fold_{fold}_scaler.pkl"; mp = OUT / f"fold_{fold}_model.pkl"
            sp.write_bytes(pickle.dumps(scaler, protocol=5)); mp.write_bytes(pickle.dumps(model, protocol=5))
            models += [sp, mp]
            logs.append({"fold": fold, "held_groups": len(held_groups), "held_windows": len(held_ix),
                         "training_pair_rows": int(pair_fit.sum()), "iterations": int(model.n_iter_[0])})
        assert np.isfinite(oof).all()
        window_threshold = q.choose_threshold(y[:NFIT], oof)
        oof_answer = answer_scores({**meta, "answers": meta["answers"][:NANS_FIT],
                                    "windows": meta["windows"][:NFIT]}, oof)
        ay = np.asarray([a["label"] for a in meta["answers"]], np.int8)
        answer_threshold = q.choose_threshold(ay[:NANS_FIT], oof_answer)
        scaler, model = fit_model(d, pair_y, pair_w)
        fs = OUT / "full_scaler.pkl"; fm = OUT / "full_model.pkl"
        fs.write_bytes(pickle.dumps(scaler, protocol=5)); fm.write_bytes(pickle.dumps(model, protocol=5)); models += [fs, fm]
        q.save(OUT / "fit_freeze_before_calibration.json", {
            "window_threshold": window_threshold, "derived_answer_threshold": answer_threshold,
            "models_sha256": {p.name: q.sha(p) for p in models}, "OOF_scores_sha256": q.digest(oof.tolist()),
            "calibration_scored": False, "official_test_opened": False})
        score = decision(scaler, model, x)
    answers = answer_scores(meta, score)
    lo, hi = meta["bounds"]["calibration"]
    fit_threshold_cal = count(y[lo:hi], score[lo:hi], window_threshold["threshold"])
    common_cal_opt = q.choose_threshold(y[lo:hi], score[lo:hi])
    common_cal_metrics = count(y[lo:hi], score[lo:hi], common_cal_opt["threshold"])
    derived_answer_cal = count(ay[NANS_FIT:], answers[NANS_FIT:], answer_threshold["threshold"])
    current_meta = q.read(ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json")
    current_scores = np.load(SOURCES["current"])["answer_scores"]
    current_answer_cal = count(ay[NANS_FIT:], current_scores[NANS_FIT:], current_meta["thresholds"]["answer"]["threshold"])
    np.savez_compressed(OUT / "scores.npz", fit_oof_window_scores=oof, window_scores=score,
                        derived_answer_scores=answers, preserved_current_answer_scores=current_scores)
    summary = {
        "status": "complete_development_only", "fit_OOF": count(y[:NFIT], oof, window_threshold["threshold"]),
        "calibration": {"fit_threshold_window": fit_threshold_cal,
                        "common_evaluator_F1Opt_window": common_cal_metrics,
                        "derived_answer_fit_threshold": derived_answer_cal,
                        "preserved_current_answer_head": current_answer_cal},
        "thresholds": {"window_fit_OOF": window_threshold, "derived_answer_fit_OOF": answer_threshold,
                       "common_cal_F1Opt_reporting_only": common_cal_opt},
        "fit_logs": logs, "seconds": time.perf_counter()-start,
        "calibration_used_for_model_or_variant_selection": False, "GPU_used": False, "official_test_opened": False,
        "upstream_in_sample_limit": protocol()["known_limit"]}
    q.save(OUT / "summary.json", summary)
    (OUT / "REPORT.md").write_text(
        "# 同答细节成对排序器 v1：开发结果\n\n"
        f"fit组留出窗口F1：{summary['fit_OOF']['f1']:.6f}。fit阈值直接迁移到cal：{fit_threshold_cal['f1']:.6f}；"
        f"共同评测F1Opt：{common_cal_metrics['f1']:.6f}。派生整答F1：{derived_answer_cal['f1']:.6f}；"
        f"保留的current整答头F1：{current_answer_cal['f1']:.6f}。\n\n"
        "模型、输入和fit阈值冻结后才读取cal分数；共同F1Opt仅是既定评测指标，不用于回改模型。上游多数fit分数仍非OOF，故本轮不能替代尚缺large fold2的完整交叉拟合。\n",
        encoding="utf-8")
    q.save(OUT / "complete.json", {"status": "complete_development_only", "summary_sha256": q.sha(OUT / "summary.json"),
                                     "scores_sha256": q.sha(OUT / "scores.npz"), "official_test_opened": False})
    print("PAIRWISE_RANKER_COMPLETE", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "check", "train"))
    args = parser.parse_args()
    {"prepare": prepare, "check": check, "train": train}[args.stage]()
