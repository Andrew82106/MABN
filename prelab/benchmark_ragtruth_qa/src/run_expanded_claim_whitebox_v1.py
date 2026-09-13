"""Expanded claim-level probe over frozen Llama white-box traces.

The formal baselines are untouched.  This candidate pools the tested answer
model's PCA hidden state, Lookback attention tensor and token NLL inside fixed
automatic claims, selects one fixed classifier family by source-group OOF on
the 3,680-answer fit split, then evaluates the original 159-answer calibration
split and unchanged four-BPE windows.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "fit_expansion"))
import run_probe_expansion as expansion  # noqa: E402
import run_development as q  # noqa: E402


OUT = ROOT / "results/expanded_claim_whitebox_v1"
CLAIM_INPUTS = ROOT / "results/exact_subset_attribution_v1/prepared_inputs.jsonl"
TRACE = ROOT / "fit_expansion/llama_baselines_v1"
BASE = TRACE / "matrices/base.npy"
PCA = TRACE / "matrices/token_pca64.npy"
TOKEN_INDEX = TRACE / "token_index.json"

SEED = 20261017
FOLDS = 5
CANDIDATES = ("lr_C0.001", "lr_C0.01", "lr_C0.1", "hist_leaf7", "extra_depth10")


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return np.asarray([values.min(), np.quantile(values, .25), values.mean(),
                       np.median(values), np.quantile(values, .75),
                       values.max(), values.std()], dtype=np.float64)


def text_features(text, cid, count, refs):
    alnum = [c for c in text if c.isalnum()]
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    lower = f" {text.lower()} "
    return np.asarray([
        np.log1p(len(text)), np.log1p(len(words)), cid / max(1, count - 1),
        (cid + 1) / count, sum(c.isdigit() for c in text) / max(1, len(alnum)),
        sum(c.isupper() for c in text) / max(1, len(alnum)),
        float(any(c.isdigit() for c in text)), float("%" in text),
        float(any(c in text for c in "$€£")), float(bool(refs)),
        float(" not " in lower or "n't" in lower),
        float(any(term in lower for term in (" more ", " less ", " higher ", " lower ", " than "))),
    ], dtype=np.float64)


def make_model(name):
    if name.startswith("lr_C"):
        c = float(name.split("C", 1)[1])
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, solver="liblinear", max_iter=3000, random_state=SEED))
    if name == "hist_leaf7":
        return HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, max_leaf_nodes=7,
            min_samples_leaf=50, l2_regularization=1., early_stopping=False,
            random_state=SEED)
    if name == "extra_depth10":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=12,
            max_features=.5, n_jobs=4, random_state=SEED)
    raise KeyError(name)


def model_fit(model, x, y, weights):
    key = "logisticregression__sample_weight" if hasattr(model, "steps") else "sample_weight"
    return model.fit(x, y, **{key: weights})


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=np.float64)
    pred = score >= threshold
    tp = int(np.count_nonzero(pred & (y == 1))); fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1))); tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score))}


def claim_weights(groups, responses, native, y, active):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index in active:
        tree[groups[index]][int(responses[index] in native)][responses[index]].append(index)
    weight = np.zeros(len(y), dtype=np.float64)
    for strata in tree.values():
        for answers in strata.values():
            for ids in answers.values():
                weight[ids] = 1 / (len(strata) * len(answers) * len(ids))
    mask = weight > 0; weight[mask] /= weight[mask].mean()
    mass = np.bincount(y[mask], weights=weight[mask], minlength=2)
    weight[mask] *= (mass.sum() / (2 * mass))[y[mask]]
    for strata in tree.values():
        ids = [i for answers in strata.values() for values in answers.values() for i in values]
        weight[ids] *= (mask.sum() / len(tree)) / weight[ids].sum()
    weight[mask] *= mask.sum() / weight[mask].sum()
    return weight


def project(claim_scores, window_claim_ids):
    ids = np.asarray(window_claim_ids, dtype=np.int64)
    valid = ids >= 0
    values = claim_scores[np.maximum(ids, 0)]
    values[~valid] = -np.inf
    result = values.max(1)
    assert np.isfinite(result).all()
    return result


def answer_scores(meta, window_scores, partition):
    indices = [i for i, a in enumerate(meta["answers"]) if a["partition"] == partition]
    return np.asarray([window_scores[meta["answer_windows"][meta["answers"][i]["response_id"]]].max()
                       for i in indices]), np.asarray([meta["answers"][i]["label"] for i in indices])


def prepare():
    assert not OUT.exists() or not any(OUT.iterdir()), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, meta = expansion.metadata()
    claim_rows = lines(CLAIM_INPUTS)
    assert [row["response_id"] for row in claim_rows] == [a["response_id"] for a in meta["answers"]]
    index = {row["response_id"]: row for row in q.read(TOKEN_INDEX)["answers"]}
    token_pca = np.load(PCA, mmap_mode="r")
    base = np.load(BASE, mmap_mode="r")
    assert token_pca.shape == (708506, 64) and base.shape == (696220, 1025)

    features, labels, responses, groups, partitions = [], [], [], [], []
    claim_offsets, window_claim_ids = {}, np.full((len(meta["windows"]), 4), -1, dtype=np.int32)
    cursor = 0
    for answer_no, (answer, token, row) in enumerate(zip(meta["answers"], meta["tokens"], claim_rows)):
        rid = answer["response_id"]; claim_offsets[rid] = cursor
        owners = np.asarray(row["citation_geometry"]["token_claim_index"], dtype=np.int32)
        assert len(owners) == token["token_count"] and np.array_equal(owners >= 0, token["lexical_mask"])
        claims = row["citation_geometry"]["claims"]
        assert owners.max() < len(claims)
        active_cids = [cid for cid in range(len(claims)) if np.any(owners == cid)]
        dense_id = {cid: cursor + offset for offset, cid in enumerate(active_cids)}
        win_by_claim = [[] for _ in claims]
        for wi in meta["answer_windows"][rid]:
            local = sorted({int(owners[t]) for t in meta["windows"][wi]["token_indices"] if owners[t] >= 0})
            assert 1 <= len(local) <= 4
            window_claim_ids[wi, :len(local)] = np.asarray([dense_id[cid] for cid in local], dtype=np.int32)
            for cid in local: win_by_claim[cid].append(wi)
        token_left = index[rid]["left"]
        risk = np.asarray(token["risk_mask"], dtype=bool)
        for cid in active_cids:
            claim, wis = claims[cid], win_by_claim[cid]
            local_ids = np.flatnonzero(owners == cid); assert len(local_ids) and wis
            hidden = np.asarray(token_pca[token_left + local_ids], dtype=np.float64)
            hidden_features = np.concatenate((hidden.mean(0), hidden.std(0), hidden[0], hidden[-1], hidden[-1] - hidden[0]))
            attention = np.asarray(base[wis, :1024], dtype=np.float64).mean(0).reshape(32, 32)
            attention_features = np.concatenate((attention.mean(1), attention.max(1), attention.std(1),
                                                  attention.mean(0), attention.max(0), attention.std(0),
                                                  attention[-4:].reshape(-1)))
            nll_features = stats(base[wis, -1])
            text = answer["original_response"][claim["char_start"]:claim["char_end"]]
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == claim["text_sha256"]
            features.append(np.concatenate((hidden_features, attention_features, nll_features,
                                            text_features(text, cid, len(claims), claim["valid_cited_passage_ids"]))))
            labels.append(int(risk[local_ids].any())); responses.append(rid)
            groups.append(answer["group_id"]); partitions.append(answer["partition"]); cursor += 1
        if (answer_no + 1) % 250 == 0:
            print("EXPANDED_CLAIM_WHITEBOX_PREPARED", answer_no + 1, 3839, flush=True)
    x = np.asarray(features, dtype=np.float32); y = np.asarray(labels, dtype=np.int8)
    assert x.shape == (28528, 659) and window_claim_ids.shape == (696220, 4)
    assert np.isfinite(x).all() and np.all(window_claim_ids[:, 0] >= 0)
    np.save(OUT / "features.npy", x); np.save(OUT / "labels.npy", y)
    np.save(OUT / "window_claim_ids.npy", window_claim_ids)
    save(OUT / "claims.json", {"response_ids": responses, "groups": groups,
         "partitions": partitions, "claim_offsets": claim_offsets})
    save(OUT / "preparation.json", {"status": "complete", "answers": 3839,
         "claims": len(y), "positive_claims": int(y.sum()), "features": x.shape[1],
         "windows": len(window_claim_ids), "seconds": time.perf_counter() - started,
         "formal_baselines_modified": False, "official_test_opened": False})
    print("EXPANDED_CLAIM_WHITEBOX_PREPARATION_COMPLETE", x.shape, int(y.sum()), flush=True)


def train():
    prep = q.read(OUT / "preparation.json"); assert prep["status"] == "complete"
    _, meta = expansion.metadata()
    x = np.load(OUT / "features.npy", mmap_mode="r"); y = np.load(OUT / "labels.npy")
    window_claim_ids = np.load(OUT / "window_claim_ids.npy", mmap_mode="r")
    claims = q.read(OUT / "claims.json")
    responses = np.asarray(claims["response_ids"]); groups = np.asarray(claims["groups"])
    partitions = np.asarray(claims["partitions"])
    fit = np.flatnonzero(partitions == "fit"); cal = np.flatnonzero(partitions == "calibration")
    native = {a["response_id"] for a in meta["answers"][:634]}
    splitter = GroupKFold(FOLDS); folds = list(splitter.split(fit, y[fit], groups[fit]))
    fit_window_slice = slice(*meta["bounds"]["fit"]); cal_window_slice = slice(*meta["bounds"]["calibration"])
    fit_wy = np.asarray([w["label"] for w in meta["windows"][fit_window_slice]], dtype=np.int8)
    cal_wy = np.asarray([w["label"] for w in meta["windows"][cal_window_slice]], dtype=np.int8)
    history, stored, fold_models = {}, {}, {}
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        for name in CANDIDATES:
            pred = np.full(len(y), np.nan, dtype=np.float64); models = []
            for train_local, held_local in folds:
                train_ids, held_ids = fit[train_local], fit[held_local]
                weight = claim_weights(groups, responses, native, y, train_ids)
                model = make_model(name); model_fit(model, x[train_ids], y[train_ids], weight[train_ids])
                pred[held_ids] = model.predict_proba(x[held_ids])[:, 1]; models.append(model)
            assert np.isfinite(pred[fit]).all(); pred[cal] = 0
            ws = project(pred, window_claim_ids)
            fit_as, fit_ay = answer_scores(meta, ws, "fit")
            wt = q.choose_threshold(fit_wy, ws[fit_window_slice]); at = q.choose_threshold(fit_ay, fit_as)
            key = [min(wt["f1"], at["f1"]), wt["f1"], at["f1"], wt["precision"]]
            history[name] = {"thresholds": {"window": wt, "answer": at}, "selection_key": key}
            stored[name] = pred; fold_models[name] = models
            print("EXPANDED_CLAIM_WHITEBOX_OOF", name, round(wt["f1"], 6), round(at["f1"], 6), flush=True)
    selected = max(CANDIDATES, key=lambda name: history[name]["selection_key"])
    save(OUT / "fit_only_selection.json", {"selected": selected, "candidates": history,
         "calibration_used_for_selection": False, "official_test_opened": False})
    weight = claim_weights(groups, responses, native, y, fit)
    model = make_model(selected)
    with threadpool_limits(limits=4): model_fit(model, x[fit], y[fit], weight[fit])
    pred = stored[selected]; pred[cal] = model.predict_proba(x[cal])[:, 1]
    ws = project(pred, window_claim_ids); ans_fit, ay_fit = answer_scores(meta, ws, "fit")
    ans_cal, ay_cal = answer_scores(meta, ws, "calibration")
    threshold = history[selected]["thresholds"]
    strict = {"fit": {"windows": metrics(fit_wy, ws[fit_window_slice], threshold["window"]["threshold"]),
                       "answers": metrics(ay_fit, ans_fit, threshold["answer"]["threshold"])},
              "calibration": {"windows": metrics(cal_wy, ws[cal_window_slice], threshold["window"]["threshold"]),
                              "answers": metrics(ay_cal, ans_cal, threshold["answer"]["threshold"])}}
    common = {"window": q.choose_threshold(cal_wy, ws[cal_window_slice]),
              "answer": q.choose_threshold(ay_cal, ans_cal)}
    np.savez_compressed(OUT / "scores.npz", claim_scores=pred, window_scores=ws,
                        answer_scores_fit=ans_fit, answer_scores_calibration=ans_cal)
    (OUT / "model.pkl").write_bytes(pickle.dumps({"model": model, "fold_models": fold_models[selected],
        "selected": selected}, protocol=5))
    result = {"status": "development_only_complete", "selected_by_fit_group_OOF_only": selected,
              "fit_selection": history, "strict_fit_threshold_transfer": strict,
              "common_calibration_F1Opt_diagnostic": common, "seconds": time.perf_counter() - started,
              "formal_baselines_modified": False, "calibration_used_for_model_or_threshold_selection": False,
              "official_test_opened": False, "final_test_claim": False}
    save(OUT / "summary.json", result)
    report = ["# Expanded claim white-box v1", "",
        "3,680份fit回答、615个来源组；只按fit的group-OOF结果选模型和阈值。正式基线未改。", "",
        "| 候选 | fit OOF窗口F1 | fit OOF整答F1 |", "|---|---:|---:|"]
    for name in CANDIDATES:
        t = history[name]["thresholds"]; report.append(f"| {name} | {t['window']['f1']:.6f} | {t['answer']['f1']:.6f} |")
    report += ["", f"选中 `{selected}`。", "", "| 口径 | 窗口F1 | 整答F1 |", "|---|---:|---:|",
        f"| fit阈值迁移到cal | {strict['calibration']['windows']['f1']:.6f} | {strict['calibration']['answers']['f1']:.6f} |",
        f"| 统一cal-F1Opt诊断 | {common['window']['f1']:.6f} | {common['answer']['f1']:.6f} |", "",
        "两项都来自同一陈述分数；原4-BPE标签与测试集均未改变。"]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("EXPANDED_CLAIM_WHITEBOX_COMPLETE", selected,
          strict["calibration"]["windows"]["f1"], strict["calibration"]["answers"]["f1"],
          common["window"]["f1"], common["answer"]["f1"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "train"}:
        raise SystemExit("usage: run_expanded_claim_whitebox_v1.py {prepare|train}")
    {"prepare": prepare, "train": train}[sys.argv[1]]()
