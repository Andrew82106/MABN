"""CPU-only, freeze-before-test unified Round 7 evaluation.

`fit` opens only annotations_train.jsonl and annotations_validation.jsonl.
`test` opens held-out labels only after every detector has been frozen.
The three MLP seeds are separate predictors, never a probability ensemble.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260910
MLP_SEEDS = (20260910, 20260911, 20260912)
EXTERNAL_SPLIT = "external_test"
LAYERS = (7, 14, 21, 28)
CS = (.01, .1, 1.)
LRS = (.0001, .0003)
HEAD_KS = (1, 2, 4, 8, 16, 32)
LAYER_KS = (1, 2, 4, 8, 14, 28)
BETAS = (.1, .2, .4, .6, 1., 1.2, 1.6, 1.9)
MLP_CONFIG = {"widths": [256, 128, 64], "dropout": .1, "weight_decay": .01,
              "batch_size": 32, "max_epochs": 100, "patience": 10, "min_delta": 1e-6}
RAW_METHODS = {"lumina": "lumina_score", "nll": "mean_nll", "entropy": "mean_entropy",
               "self_confidence": "self_risk", "direct_check": "direct_risk"}
METHODS = ["hidden_probe", *[f"hallurag_mlp_seed_{s}" for s in MLP_SEEDS],
           "lookback_lens", "redeep", "lumina", "nll", "entropy", "surface",
           "self_confidence", "direct_check", "all_positive", "all_negative"]
ARRAYS = {"features": [*[f"hidden_{l}" for l in LAYERS], "mean_nll", "mean_entropy", "surface"],
          "attention": ["lookback_features", "redeep_ecs", "redeep_pks"],
          "lumina": ["lumina_score", "lumina_mmd", "lumina_ipr"]}


def readl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def savel(path, records):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(clean(r), ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def finite(value):
    return value is not None and np.isfinite(value)


def metadata(root):
    inputs = readl(root / "data/inputs.jsonl")
    generated = readl(root / "data/generated.jsonl")
    source = {r["row_id"]: r for r in inputs}
    assert len(source) == len(inputs) and len({r["row_id"] for r in generated}) == len(generated)
    assert {r["row_id"] for r in generated} == set(source), "All requested generations must be complete"
    items = []
    for row in generated:
        src = source[row["row_id"]]
        assert row["question_id"] == src["question_id"] and row["split"] == src["split"]
        expected = src.get("expected_items", len(src.get("questions", [])))
        assert expected and len(row["items"]) == expected, "Missing predeclared slots in parser metadata"
        for item in row["items"]:
            items.append({**item, "row_id": row["row_id"], "question_id": row["question_id"],
                          "group_id": src.get("group_id", src.get("question_group_id", row["question_id"])),
                          "split": src["split"], "condition": src["condition"],
                          "dataset": src.get("dataset", "main"), "expected_items": expected})
    assert len({i["item_id"] for i in items}) == len(items)
    for group_key in ("question_id", "group_id"):
        groups = defaultdict(set)
        for item in items:
            groups[item[group_key]].add(item["split"])
        assert all(len(s) == 1 for s in groups.values()), group_key + " crosses splits"
    return inputs, generated, items


def load_labels(root, items, splits):
    """Explicit split files; fitting cannot accidentally open the held-out file."""
    wanted = {i["item_id"]: i for i in items if i["split"] in splits}
    result = {}
    generations = {}
    for split in splits:
        path = root / "data" / f"annotations_{split}.jsonl"
        for a in readl(path):
            assert a["item_id"] in wanted, f"Unexpected annotation in {path}: {a['item_id']}"
            assert wanted[a["item_id"]]["split"] == split
            assert a["item_id"] not in result
            expected_item = wanted[a["item_id"]]
            row_id = expected_item["row_id"]
            assert Path(row_id).name == row_id
            if row_id not in generations:
                generation_path = root/"data/generation_records"/(row_id+".json")
                record = json.loads(generation_path.read_text(encoding="utf-8"))
                assert record["row_id"] == row_id and record["question_id"] == expected_item["question_id"]
                assert record["split"] == split
                record_items = {i["item_id"]: i for i in record["items"]}
                assert len(record_items) == len(record["items"]), "Duplicate saved generation items"
                generations[row_id] = record, record_items, sha(generation_path)
            record, record_items, generation_hash = generations[row_id]
            assert a.get("source_generation_sha256") == generation_hash, "Stale annotation source_generation_sha256: " + a["item_id"]
            for field in ("row_id", "question_id", "group_id", "split", "dataset", "condition"):
                assert a.get(field) == expected_item[field], "Annotation identity mismatch " + field + ": " + a["item_id"]
            saved = record_items[a["item_id"]]
            for field in ("item_id", "item_index", "text", "start", "end", "parse_ok"):
                assert field in a and a[field] == saved[field] == expected_item[field], "Annotation/generation mismatch " + field + ": " + a["item_id"]
            start, end = saved["start"], saved["end"]
            if start is None or end is None:
                assert start is None and end is None and saved["text"] == "" and not saved["parse_ok"]
            else:
                assert isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(record["response"])
                assert record["response"][start:end] == saved["text"], "Saved generation text does not match exact character slice"
            relation, stance = a["evidence_relation"], a["stance"]
            expected = int(relation != "supported") if stance == "asserted" and relation in {
                "supported", "unsupported", "contradicted"} else None
            assert a.get("risk") == expected, "Inconsistent semantic label " + a["item_id"]
            result[a["item_id"]] = a
    assert set(result) == set(wanted), "Every predeclared slot requires an annotation"
    return result


def resolved(a):
    return a["stance"] == "asserted" and a.get("risk") in (0, 1)


class Bank:
    """Read feature rows for the requested split only; preserve absent item features."""
    def __init__(self, root, items, strict_shapes=True, area=None):
        self.items, self.vectors, self.surface_names = items, {}, None
        area = Path(area) if area is not None else root / "data"
        wanted = {i["item_id"] for i in items}
        for row_id in sorted({i["row_id"] for i in items}):
            assert Path(row_id).name == row_id
            row_ids = {i["item_id"] for i in items if i["row_id"] == row_id}
            for stage, keys in ARRAYS.items():
                path = area / stage / (row_id + ".npz")
                assert path.exists(), "Unfinished method extraction: " + str(path)
                with np.load(path, allow_pickle=False) as z:
                    ids = [str(s) for s in z["item_ids"]]
                    assert len(set(ids)) == len(ids) and set(ids) <= row_ids
                    if stage == "features":
                        names = z["surface_names"].tolist()
                        assert self.surface_names is None or self.surface_names == names
                        self.surface_names = names
                    for key in keys:
                        if key not in z and not ids:
                            continue
                        values = z[key]
                        assert values.shape[0] == len(ids), (path, key)
                        if strict_shapes and ids:
                            width = 3584 if key.startswith("hidden_") else 784 if key in {"lookback_features", "redeep_ecs"} else 28 if key == "redeep_pks" else None
                            if width:
                                assert values.shape == (len(ids), width), (path, key, values.shape)
                        for n, item_id in enumerate(ids):
                            self.vectors.setdefault(item_id, {})[key] = np.asarray(values[n], dtype=np.float32)
        baseline_file = area / "baselines.jsonl"
        assert baseline_file.exists(), "Self-confidence/direct baseline stage incomplete"
        seen = set()
        for b in readl(baseline_file):
            if b["item_id"] not in wanted:
                continue
            assert b["item_id"] not in seen
            seen.add(b["item_id"])
            for key in ("self_risk", "direct_risk"):
                v = b.get(key)
                assert v is None or finite(v) and 0 <= v <= 1
                self.vectors.setdefault(b["item_id"], {})[key] = np.float32(v) if v is not None else np.float32(np.nan)

    def matrix(self, items, key):
        width = next((np.asarray(v[key]).size for v in self.vectors.values() if key in v), None)
        if width is None:
            width = 3584 if key.startswith("hidden_") else 784 if key in {"lookback_features", "redeep_ecs"} else 28 if key == "redeep_pks" else len(self.surface_names or []) if key == "surface" else 1
        X = np.full((len(items), width), np.nan, dtype=np.float32)
        for n, item in enumerate(items):
            value = self.vectors.get(item["item_id"], {}).get(key)
            if item.get("parse_ok", False) and value is not None:
                vector = np.asarray(value).reshape(-1)
                assert vector.size == width
                X[n] = vector
        return X


def threshold_search(labels, scores):
    """Sort once, evaluate all >= unique-score thresholds with cumulative counts."""
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    total_n, total_positive = len(y), int(y.sum())
    both_classes = len(np.unique(y)) == 2
    good = np.isfinite(s)
    y, s = y[good], s[good]
    if not len(s) or not both_classes:
        return {"threshold": None, "status": "unavailable", "reason": "Validation lacks both classes or any usable score", "n": total_n, "scorable_n": len(s)}
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[ends].astype(float)
    fp = (ends+1)-tp
    # Missing positives remain false negatives at every threshold. This makes
    # candidate selection use the same all-resolved denominator as final F1.
    fn = total_positive-tp
    f1 = 2*tp/(2*tp+fp+fn)
    precision = tp/(tp+fp)
    # Include both declared endpoints explicitly. The no-alert candidate has F1=0.
    ts = np.r_[np.nextafter(s[0], np.inf), s[ends], np.nextafter(s[-1], -np.inf)]
    f1 = np.r_[0., f1, f1[-1]]
    precision = np.r_[0., precision, precision[-1]]
    best = max(range(len(ts)), key=lambda k: (f1[k], precision[k], ts[k]))
    return {"threshold": float(ts[best]), "status": "ready", "n": total_n, "scorable_n": len(s),
            "validation_f1": float(f1[best]), "validation_precision": float(precision[best]),
            "candidate_count": len(ts), "comparison": ">=",
            "tie_break": "F1, precision, higher threshold"}


def selection_key(threshold, *simplicity):
    return (threshold.get("validation_f1", -1), threshold.get("validation_precision", -1), *simplicity)


def lr_fit(X, y, C):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
        solver="liblinear", penalty="l2", max_iter=2000, random_state=SEED))
    model.fit(X, y)
    return model


def model_scores(model, X):
    score = np.full(len(X), np.nan)
    good = np.isfinite(X).all(1)
    if model is not None and good.any():
        score[good] = model.predict_proba(X[good])[:, 1]
    return score


def select_lr(bank, train, validation, ytrain, yval, keys):
    candidates, best = [], None
    for feature in keys:
        X, V = bank.matrix(train, feature), bank.matrix(validation, feature)
        good = np.isfinite(X).all(1)
        for C in CS:
            model = lr_fit(X[good], ytrain[good], C) if len(np.unique(ytrain[good])) == 2 else None
            threshold = threshold_search(yval, model_scores(model, V))
            layer = int(feature.split("_")[-1]) if feature.startswith("hidden_") else 0
            entry = {"feature": feature, "C": C, "threshold": threshold,
                     "n_train": int(good.sum()), "train_item_ids": [i["item_id"] for i, ok in zip(train, good) if ok]}
            candidates.append(entry)
            rank = selection_key(threshold, -C, -layer)
            if best is None or rank > best[0]:
                best = (rank, {"kind": "lr", "model": model, "feature": feature, "selection": entry})
    return best[1], candidates


class MLP(torch.nn.Module):
    def __init__(self, width, widths=(256, 128, 64), dropout=.1):
        super().__init__()
        layers = []
        for target in widths:
            layers.extend([torch.nn.Linear(width, target), torch.nn.ReLU(), torch.nn.Dropout(dropout)])
            width = target
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(-1)


def train_mlp(X, y, V, vy, lr, seed, config=None):
    """Best checkpoint chosen only by train-weighted validation BCE."""
    config = dict(MLP_CONFIG if config is None else config)
    scaler = StandardScaler().fit(X)
    X = torch.tensor(scaler.transform(X).astype(np.float32), device="cpu")
    V = torch.tensor(scaler.transform(V).astype(np.float32), device="cpu")
    y = torch.tensor(y, dtype=torch.float32, device="cpu")
    vy = torch.tensor(vy, dtype=torch.float32, device="cpu")
    assert y.sum() > 0 and y.sum() < len(y) and len(vy)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = MLP(X.shape[1], config["widths"], config["dropout"]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    positive_weight = (len(y)-y.sum())/y.sum()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history, best_loss, wait, best_state, best_epoch = [], math.inf, 0, None, None
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"]+1):
        model.train()
        permutation = torch.randperm(len(X), generator=generator)
        total = 0.
        for batch in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(V), vy))
        history.append({"epoch": epoch, "train_bce": total/len(X), "validation_bce": valid_loss})
        if valid_loss < best_loss-config["min_delta"]:
            best_loss, wait, best_epoch = valid_loss, 0, epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= config["patience"]:
            break
    assert best_state is not None
    return {"kind": "mlp", "scaler": scaler, "state_dict": best_state, "input_width": X.shape[1],
            "config": config, "seed": seed, "lr": lr,
            "training": {"history": history, "best_epoch": best_epoch, "best_validation_bce": best_loss,
                         "train_positive_weight": float(positive_weight), "n_train": len(X), "n_validation": len(V),
                         "seconds": time.perf_counter()-start, "device": "cpu"}}


def mlp_scores(model, X):
    scores = np.full(len(X), np.nan)
    good = np.isfinite(X).all(1)
    if model is None or not good.any():
        return scores
    network = MLP(model["input_width"], model["config"]["widths"], model["config"]["dropout"]).cpu()
    network.load_state_dict(model["state_dict"])
    network.eval()
    value = torch.tensor(model["scaler"].transform(X[good]).astype(np.float32), device="cpu")
    with torch.no_grad():
        scores[good] = network(value).sigmoid().numpy()
    return scores


def select_mlp(bank, train, validation, ytrain, yval, cache, signature):
    cache.mkdir(parents=True, exist_ok=True)
    candidates, best = [], None
    for layer in LAYERS:
        feature = f"hidden_{layer}"
        X, V = bank.matrix(train, feature), bank.matrix(validation, feature)
        tgood, vgood = np.isfinite(X).all(1), np.isfinite(V).all(1)
        for lr in LRS:
            models, thresholds, seed_entries = {}, {}, []
            for seed in MLP_SEEDS:
                path = cache/f"mlp_l{layer}_lr{lr:g}_s{seed}.pkl"
                key = digest([signature, feature, lr, seed, MLP_CONFIG])
                if path.exists():
                    stored = pickle.loads(path.read_bytes())
                    assert stored["signature"] == key, "Stale MLP checkpoint; do not silently reuse"
                    model = stored["model"]
                else:
                    model = train_mlp(X[tgood], ytrain[tgood], V[vgood], yval[vgood], lr, seed) if (
                        len(np.unique(ytrain[tgood])) == 2 and vgood.any()) else None
                    path.write_bytes(pickle.dumps({"signature": key, "model": model}, protocol=5))
                threshold = threshold_search(yval, mlp_scores(model, V))
                if model is not None:
                    model.update(feature=feature, selection={"threshold": threshold, "layer": layer,
                        "train_item_ids": [i["item_id"] for i, ok in zip(train, tgood) if ok],
                        "early_stopping": "Training-positive-weighted validation BCE; no test data"})
                models[seed], thresholds[seed] = model, threshold
                seed_entries.append({"seed": seed, "threshold": threshold, "checkpoint_sha256": sha(path),
                                     "training": model["training"] if model is not None else None})
                print("MLP_CANDIDATE", layer, lr, seed, threshold.get("validation_f1"), flush=True)
            fs = [thresholds[s].get("validation_f1", -1.) for s in MLP_SEEDS]
            ps = [thresholds[s].get("validation_precision", -1.) for s in MLP_SEEDS]
            entry = {"layer": layer, "lr": lr, "seeds": seed_entries,
                     "mean_validation_f1": float(np.mean(fs)), "mean_validation_precision": float(np.mean(ps)),
                     "selection_uses": "Arithmetic mean of individual-seed validation F1; not ensemble probabilities"}
            candidates.append(entry)
            rank = (entry["mean_validation_f1"], entry["mean_validation_precision"], -lr, -layer)
            if best is None or rank > best[0]:
                best = rank, models, thresholds, entry
    return best[1], best[2], best[3], candidates


def pearson_rank(X, target):
    centered = np.asarray(X, float)-np.asarray(X, float).mean(0)
    target = np.asarray(target, float)-np.mean(target)
    denominator = np.sqrt((centered**2).sum(0)*(target**2).sum())
    correlation = np.divide(centered.T@target, denominator, out=np.zeros(X.shape[1]), where=denominator > 0)
    order = np.lexsort((np.arange(X.shape[1]), -correlation))
    return order, correlation


def redeep_scores(model, ecs, pks):
    scores = np.full(len(ecs), np.nan)
    good = np.isfinite(ecs).all(1) & np.isfinite(pks).all(1)
    if model is None or not good.any():
        return scores
    E = ecs[good][:, model["heads"]].sum(1, dtype=np.float64)
    P = pks[good][:, model["layers"]].sum(1, dtype=np.float64)
    scores[good] = (P-model["p_min"])/model["p_range"] - model["beta"]*(E-model["e_min"])/model["e_range"]
    return scores


def select_redeep(bank, train, validation, ytrain, yval):
    E, P = bank.matrix(train, "redeep_ecs"), bank.matrix(train, "redeep_pks")
    VE, VP = bank.matrix(validation, "redeep_ecs"), bank.matrix(validation, "redeep_pks")
    good = np.isfinite(E).all(1) & np.isfinite(P).all(1)
    if len(np.unique(ytrain[good])) != 2:
        return None, [], {"reason": "Training ReDeEP features lack both classes"}
    heads, ec = pearson_rank(E[good], 1-ytrain[good])
    layers, pc = pearson_rank(P[good], ytrain[good])
    candidates, best = [], None
    for kh in HEAD_KS:
        for kl in LAYER_KS:
            h, l = heads[:kh], layers[:kl]
            ev, pv = E[good][:, h].sum(1, dtype=np.float64), P[good][:, l].sum(1, dtype=np.float64)
            erange, prange = float(np.ptp(ev)), float(np.ptp(pv))
            for beta in BETAS:
                model = {"kind": "redeep", "heads": h, "layers": l, "beta": beta,
                         "e_min": float(ev.min()), "p_min": float(pv.min()),
                         "e_range": erange if erange else 1., "p_range": prange if prange else 1.}
                threshold = threshold_search(yval, redeep_scores(model, VE, VP))
                entry = {"k_heads": kh, "k_layers": kl, "beta": beta, "threshold": threshold}
                candidates.append(entry)
                rank = selection_key(threshold, -kh, -kl, -beta)
                if best is None or rank > best[0]:
                    model["selection"] = entry
                    best = rank, model
    ranking = {"head_order": heads, "head_correlations_to_1_minus_risk": ec,
               "layer_order": layers, "layer_correlations_to_risk": pc,
               "train_item_ids": [i["item_id"] for i, ok in zip(train, good) if ok],
               "ranking_ties": "lower flat head/layer index", "constant_feature_correlation": 0,
               "scaling": "Train min/max of sums; no clipping on validation/test; constant range denominator 1"}
    return best[1], candidates, ranking


def metrics(y, pred, score):
    y, pred, score = np.asarray(y, int), np.asarray(pred, int), np.asarray(score, float)
    tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tn = int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 0)).sum())
    good = np.isfinite(score)
    reasons = {}
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*tp/(2*tp+fp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    auc = float(roc_auc_score(y[good], score[good])) if len(np.unique(y[good])) == 2 else None
    ap = float(average_precision_score(y[good], score[good])) if y[good].sum() else None
    for key, value in {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr,
                       "auroc": auc, "average_precision": ap}.items():
        if value is None:
            reasons[key] = "Required class, predictions or denominator absent; not estimated"
    return {"n": len(y), "positive": tp+fn, "negative": fp+tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prevalence": float(y.mean()) if len(y) else None, "precision": precision, "recall": recall,
            "f1": f1, "false_positive_rate": fpr, "auroc": auc, "average_precision": ap,
            "ranking_n": int(good.sum()), "undefined_reasons": reasons}


def score_method(model, bank, items):
    if model is None:
        return np.full(len(items), np.nan)
    kind = model["kind"]
    if kind == "constant":
        return np.full(len(items), model["score"], dtype=float)
    if kind == "redeep":
        return redeep_scores(model, bank.matrix(items, "redeep_ecs"), bank.matrix(items, "redeep_pks"))
    X = bank.matrix(items, model["feature"])
    if kind == "raw":
        return X[:, 0].astype(float)
    return mlp_scores(model, X) if kind == "mlp" else model_scores(model["model"], X)


def predict(items, labels, bank, models, thresholds):
    scores = {method: score_method(models[method], bank, items) for method in METHODS}
    result = []
    for n, item in enumerate(items):
        values = {}
        for method in METHODS:
            score, threshold = scores[method][n], thresholds[method]["threshold"]
            ok = finite(score) and finite(threshold)
            values[method] = {"score": float(score) if finite(score) else None,
                              "threshold": threshold, "prediction": int(score >= threshold) if ok else None,
                              "missing_reason": None if ok else "feature_parse_model_or_validation_threshold_unavailable"}
        result.append({**item, "annotation": labels[item["item_id"]], "methods": values})
    return result


def behavior(records):
    def part(xs):
        return {"slots": len(xs), "question_groups": len({r["group_id"] for r in xs}),
                "questions": len({r["question_id"] for r in xs}), "responses": len({r["row_id"] for r in xs}),
                "stance": dict(Counter(r["annotation"]["stance"] for r in xs)),
                "evidence_relation": dict(Counter(r["annotation"]["evidence_relation"] for r in xs)),
                "reference_correctness": dict(Counter(r["annotation"]["reference_correctness"] for r in xs)),
                "resolved_asserted": sum(resolved(r["annotation"]) for r in xs),
                "risk_items": sum(r["annotation"].get("risk") == 1 for r in xs),
                "parse_failures": sum(not r.get("parse_ok", False) for r in xs),
                "multiple_facts": sum(bool(r["annotation"].get("multi_claim", r["annotation"].get("multiple_facts"))) for r in xs),
                "multiple_facts_recorded": sum("multi_claim" in r["annotation"] or "multiple_facts" in r["annotation"] for r in xs)}
    return {"all": part(records), "conditions": {c: part([r for r in records if r["condition"] == c])
                                                 for c in sorted({r["condition"] for r in records})}}


def evaluate_method(records, method):
    rows = [r for r in records if resolved(r["annotation"])]
    def measure(xs):
        return metrics([r["annotation"]["risk"] for r in xs],
                       [r["methods"][method]["prediction"] or 0 for r in xs],
                       [np.nan if r["methods"][method]["score"] is None else r["methods"][method]["score"] for r in xs])
    usable = [r for r in rows if r["methods"][method]["prediction"] is not None]
    missing = [r for r in rows if r["methods"][method]["prediction"] is None]
    result = {"all_resolved": measure(rows), "scorable_only": measure(usable),
              "coverage": {"resolved": len(rows), "scorable_resolved": len(usable), "missing_resolved": len(missing),
                           "missing_risk": sum(r["annotation"]["risk"] for r in missing), "all_slots": len(records),
                           "scorable_all_slots": sum(r["methods"][method]["prediction"] is not None for r in records)},
              "conditions": {c: measure([r for r in rows if r["condition"] == c]) for c in sorted({r["condition"] for r in records})}}
    subtypes = {}
    for name, relation, correct in [("unsupported_correct", "unsupported", "correct"),
                                    ("unsupported_incorrect", "unsupported", "incorrect"),
                                    ("unsupported_unresolved", "unsupported", "unresolved"),
                                    ("contradicted", "contradicted", None)]:
        xs = [r for r in rows if r["annotation"]["evidence_relation"] == relation and (
            correct is None or r["annotation"]["reference_correctness"] == correct)]
        detected = sum(r["methods"][method]["prediction"] == 1 for r in xs)
        subtypes[name] = {"n": len(xs), "detected": detected, "recall": detected/len(xs) if xs else None}
    result["subtypes"] = subtypes
    budget = math.ceil(.2*len(rows))
    top = sorted(usable, key=lambda r: (-r["methods"][method]["score"], r["item_id"]))[:budget]
    found, total = sum(r["annotation"]["risk"] for r in top), sum(r["annotation"]["risk"] for r in rows)
    result["top20_resolved_only"] = {"budget": budget, "selected": len(top), "found": found,
                       "precision": found/len(top) if top else None, "recall": found/total if total else None,
                       "item_ids": [r["item_id"] for r in top],
                       "denominator": "Oracle annotation-filtered secondary: resolved asserted items only; excludes abstained, missing and unresolved using labels"}
    # Operational selection cannot consult annotations, including refusal status.
    budget = math.ceil(.2*len(records))
    scored = [r for r in records if finite(r["methods"][method]["score"])]
    top = sorted(scored, key=lambda r: (-r["methods"][method]["score"], r["item_id"]))[:budget]
    found = sum(r["annotation"].get("risk") == 1 for r in top)
    total = sum(r["annotation"].get("risk") == 1 for r in records)
    result["top20"] = {
        "budget": budget, "all_slots": len(records), "scored_all_slots": len(scored),
        "missing_scores": len(records)-len(scored),
        "missing_predictions": sum(r["methods"][method]["prediction"] is None for r in records),
        "selected": len(top), "unfilled_budget": budget-len(top), "found": found, "known_risk_total": total,
        "selected_resolved": sum(resolved(r["annotation"]) for r in top),
        "selected_supported": sum(r["annotation"].get("evidence_relation") == "supported" for r in top),
        "selected_abstained": sum(r["annotation"]["stance"] == "abstained" for r in top),
        "selected_missing": sum(r["annotation"]["stance"] == "missing" for r in top),
        "selected_unresolved": sum(not resolved(r["annotation"]) and r["annotation"]["stance"] not in ("abstained", "missing") for r in top),
        "confirmed_risk_yield": found/len(top) if top else None,
        "recall": found/total if total else None, "item_ids": [r["item_id"] for r in top],
        "denominator": "All predetermined answer slots; rank all finite scores without label or prediction-threshold filtering; budget ceil(0.2*N)",
        "interpretation": "Yield is confirmed risk / selected, not precision against unknown truth; selected_resolved includes supported and risky assertions; missing means missing answer slot"}
    byrow = defaultdict(list)
    for r in rows:
        byrow[r["row_id"]].append(r)
    mixed = [xs for xs in byrow.values() if {r["annotation"]["risk"] for r in xs} == {0, 1}]
    pair_scores, answer_scores, eligible_pairs = [], [], 0
    for xs in mixed:
        local = []
        for a in [r for r in xs if r["annotation"]["risk"]]:
            for b in [r for r in xs if not r["annotation"]["risk"]]:
                eligible_pairs += 1
                av, bv = a["methods"][method], b["methods"][method]
                if av["prediction"] is not None and bv["prediction"] is not None:
                    local.append(1. if av["score"] > bv["score"] else .5 if av["score"] == bv["score"] else 0.)
        pair_scores.extend(local)
        if local:
            answer_scores.append({"row_id": xs[0]["row_id"], "auc": float(np.mean(local))})
    result["within_answer"] = {"mixed_responses": len(mixed), "mixed_groups": len({xs[0]["group_id"] for xs in mixed}),
                               "scored_responses": len(answer_scores), "eligible_pairs": eligible_pairs, "scored_pairs": len(pair_scores),
                               "mean_response_auc": float(np.mean([r["auc"] for r in answer_scores])) if answer_scores else None,
                               "pair_ranking": float(np.mean(pair_scores)) if pair_scores else None, "responses": answer_scores}
    paired = defaultdict(dict)
    for r in rows:
        paired[(r["question_id"], r["item_index"])][r["condition"]] = r
    controls = []
    for (qid, index), pair in paired.items():
        if len(pair) != 2 or "complete" not in pair or any(r["annotation"]["risk"] for r in pair.values()):
            continue
        a = pair["complete"]
        b = next(r for c, r in pair.items() if c != "complete")
        av, bv = a["methods"][method], b["methods"][method]
        if finite(av["score"]) and finite(bv["score"]):
            controls.append({"question_id": qid, "item_index": index, "complete_score": av["score"], "partial_score": bv["score"],
                             "delta": bv["score"]-av["score"], "complete_alert": av["prediction"], "partial_alert": bv["prediction"]})
    result["supported_in_both_conditions"] = {"n": len(controls), "pairs": controls,
        "mean_partial_minus_complete": float(np.mean([r["delta"] for r in controls])) if controls else None,
        "note": "Same target item remains supported in both conditions; wording may differ; not a mechanism proof"}
    return result


def mean_sd(values):
    good = [v for v in values if v is not None and finite(v)]
    return {"mean": float(np.mean(good)) if good else None,
            "sd": float(np.std(good, ddof=1)) if len(good) > 1 else None, "n_defined": len(good)}


def bootstrap_groups(records, methods, draws=2000, seed=SEED):
    """Shared group resamples; fixed predictors, no refitting or retuning."""
    rows = [r for r in records if resolved(r["annotation"])]
    # Include groups without resolved assertions: they are sampled but contribute 0 counts.
    groups = sorted({r["group_id"] for r in records})
    if not groups:
        return {"draws": draws, "groups": 0, "methods": {}}
    index = {g: n for n, g in enumerate(groups)}
    counts = np.zeros((len(groups), len(methods), 4), dtype=np.int64)
    for r in rows:
        for m, method in enumerate(methods):
            y, pred = r["annotation"]["risk"], r["methods"][method]["prediction"] or 0
            cell = 0 if y and pred else 1 if not y and pred else 2 if y else 3
            counts[index[r["group_id"]], m, cell] += 1
    rng = np.random.default_rng(seed)
    draws_index = rng.integers(0, len(groups), size=(draws, len(groups)))
    weights = np.stack([np.bincount(row, minlength=len(groups)) for row in draws_index])
    aggregated = np.einsum("bg,gmc->bmc", weights, counts, optimize=True)
    tp, fp, fn = [aggregated[:, :, i] for i in range(3)]
    denominator = 2*tp+fp+fn
    f1 = np.divide(2*tp, denominator, out=np.full(tp.shape, np.nan, dtype=float), where=(tp+fn) > 0)
    reference = f1[:, methods.index("hidden_probe")]
    def interval(v):
        v = v[np.isfinite(v)]
        return {"ci95": np.quantile(v, [.025, .975]).tolist() if len(v) else None, "defined_draws": len(v)}
    values = {method: {"f1": interval(f1[:, n]), "f1_minus_hidden_probe": interval(f1[:, n]-reference)}
              for n, method in enumerate(methods)}
    seeds = [methods.index(f"hallurag_mlp_seed_{s}") for s in MLP_SEEDS]
    # Arithmetic mean of seed-specific F1 for each common group resample.
    family = np.mean(f1[:, seeds], axis=1)
    values["hallurag_mlp_mean"] = {"f1": interval(family), "f1_minus_hidden_probe": interval(family-reference)}
    return {"draws": draws, "seed": seed, "groups": len(groups), "group_ids": groups,
            "unit": "group_id; all paired conditions and items sampled together",
            "inference": "Conditional on frozen fitted detectors and thresholds; no retraining; seed family uses mean F1, not averaged scores",
            "methods": values}


def evaluate(records, bootstrap=False):
    result = {"behavior": behavior(records), "methods": {m: evaluate_method(records, m) for m in METHODS}}
    seeds = [f"hallurag_mlp_seed_{s}" for s in MLP_SEEDS]
    result["hallurag_mlp_summary"] = {
        "seeds": list(MLP_SEEDS), "meaning": "Mean and sample SD of independently thresholded seed metrics; no ensemble predictor",
        "all_resolved": {k: mean_sd([result["methods"][m]["all_resolved"][k] for m in seeds])
                         for k in ("f1", "precision", "recall", "tp", "fp", "fn", "tn", "auroc", "average_precision")},
        "top20": {k: mean_sd([result["methods"][m]["top20"][k] for m in seeds]) for k in ("found", "confirmed_risk_yield", "recall")},
        "top20_resolved_only": {k: mean_sd([result["methods"][m]["top20_resolved_only"][k] for m in seeds]) for k in ("found", "precision", "recall")}}
    common = [r for r in records if resolved(r["annotation"]) and all(r["methods"][m]["prediction"] is not None for m in METHODS)]
    result["common_scorable"] = {"n": len(common), "item_ids": [r["item_id"] for r in common],
        "methods": {m: evaluate_method(common, m)["all_resolved"] for m in METHODS}}
    if bootstrap:
        result["bootstrap"] = bootstrap_groups(records, METHODS)
    return result


def data_hashes(root):
    paths = [root/"data/inputs.jsonl", root/"data/generated.jsonl", root/"data/baselines.jsonl"]
    expected_rows = {r["row_id"] for r in readl(root/"data/inputs.jsonl")}
    for row_id in sorted(expected_rows):
        assert Path(row_id).name == row_id
        generation_path = root/"data/generation_records"/(row_id+".json")
        assert generation_path.exists(), "Exact generation record missing"
        paths.append(generation_path)
    def check_manifest(m):
        assert m["complete"], "All methods must finish before fit/test"
        assert m["expected_rows"] == m["completed_rows"] == len(expected_rows), "Stage manifest covers a different cohort"
        assert set(m["record_files"]) == {row+".json" for row in expected_rows}, "Stage manifest row coverage differs"
    for stage in ARRAYS:
        paths.extend(sorted((root/"data"/stage).glob("*.npz")))
        manifest = root/"data"/stage/"manifest.json"
        assert manifest.exists(), "Stage manifest missing: " + str(manifest)
        m = json.loads(manifest.read_text(encoding="utf-8"))
        check_manifest(m)
        assert all((root/"data"/stage/(row+".npz")).exists() for row in expected_rows)
        paths.append(manifest)
    baseline_manifest = root/"data/baselines/manifest.json"
    assert baseline_manifest.exists(), "Baseline stage manifest missing"
    check_manifest(json.loads(baseline_manifest.read_text(encoding="utf-8")))
    for rel in ["protocol.json", "PLAN.md", "data/references.jsonl", "data/generation_manifest.json", "data/baselines/manifest.json"]:
        if (root/rel).exists():
            paths.append(root/rel)
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in paths}


def fit(root):
    out = root/"results"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/"freeze.json").exists(), "Already frozen; preserve the run"
    started = time.perf_counter()
    _, _, all_items = metadata(root)
    items = [i for i in all_items if i["split"] in {"train", "validation"}]
    labels = load_labels(root, items, ["train", "validation"])
    hashes = data_hashes(root)
    code_hash = sha(Path(__file__))
    label_hashes = {s: sha(root/"data"/f"annotations_{s}.jsonl") for s in ("train", "validation")}
    signature = digest([hashes, code_hash, label_hashes, MLP_CONFIG, MLP_SEEDS])
    bank = Bank(root, items)
    train = [i for i in items if i["split"] == "train" and resolved(labels[i["item_id"]])]
    validation = [i for i in items if i["split"] == "validation" and resolved(labels[i["item_id"]])]
    ytrain = np.array([labels[i["item_id"]]["risk"] for i in train])
    yval = np.array([labels[i["item_id"]]["risk"] for i in validation])
    models, thresholds, search = {}, {}, {}
    with threadpool_limits(limits=4):
        for method, features in [("hidden_probe", [f"hidden_{l}" for l in LAYERS]),
                                 ("lookback_lens", ["lookback_features"]), ("surface", ["surface"])]:
            models[method], search[method] = select_lr(bank, train, validation, ytrain, yval, features)
            thresholds[method] = models[method]["selection"]["threshold"]
            print("SELECTED", method, models[method]["selection"], flush=True)
        mlps, mlp_thresholds, mlp_choice, search["hallurag_mlp"] = select_mlp(
            bank, train, validation, ytrain, yval, out/"candidates", signature)
        for seed in MLP_SEEDS:
            name = f"hallurag_mlp_seed_{seed}"
            models[name], thresholds[name] = mlps[seed], mlp_thresholds[seed]
        models["redeep"], search["redeep"], ranking = select_redeep(bank, train, validation, ytrain, yval)
        thresholds["redeep"] = models["redeep"]["selection"]["threshold"] if models["redeep"] else threshold_search([], [])
        for name, feature in RAW_METHODS.items():
            models[name] = {"kind": "raw", "feature": feature}
            thresholds[name] = threshold_search(yval, score_method(models[name], bank, validation))
        for name, value in [("all_positive", 1.), ("all_negative", 0.)]:
            models[name] = {"kind": "constant", "score": value}
            thresholds[name] = {"threshold": .5, "status": "ready", "reason": "fixed constant baseline"}
        assert set(models) == set(METHODS) == set(thresholds)
        predictions = predict(items, labels, bank, models, thresholds)
    weights = out/"frozen_models.pkl"
    weights.write_bytes(pickle.dumps(models, protocol=5))
    savel(out/"development_predictions.jsonl", predictions)
    save(out/"validation.json", evaluate([p for p in predictions if p["split"] == "validation"]))
    save(out/"selection.json", {"candidates": search, "redeep_train_ranking": ranking,
                              "mlp_common_configuration": mlp_choice, "surface_names": bank.surface_names})
    frozen = {"stage": "fit_complete_no_test_labels_opened", "utc": datetime.now(timezone.utc).isoformat(),
              "primary_method": "hidden_probe", "method_names": METHODS, "thresholds": thresholds,
              "mlp_seeds": MLP_SEEDS, "mlp_configuration": MLP_CONFIG, "mlp_selected": mlp_choice,
              "selected_lr": {m: models[m]["selection"] for m in ("hidden_probe", "surface", "lookback_lens")},
              "selected_redeep": models["redeep"]["selection"] if models["redeep"] else None,
              "frozen_model_sha256": sha(weights), "selection_sha256": sha(out/"selection.json"),
              "data_sha256": hashes, "development_annotation_sha256": label_hashes, "evaluator_sha256": code_hash,
              "development_item_ids": [i["item_id"] for i in items],
              "test_item_ids": [i["item_id"] for i in all_items if i["split"] == "test"],
              "external_item_ids": [i["item_id"] for i in all_items if i["split"] == EXTERNAL_SPLIT],
              "train_group_ids": sorted({i["group_id"] for i in train}),
              "validation_group_ids": sorted({i["group_id"] for i in validation}),
              "fit_seconds": time.perf_counter()-started, "cpu_threads_max": 4,
              "threshold_grid": "Distinct validation scores plus all/no-alert endpoints; no held-out threshold selection",
              "mlp_aggregation": "Common hyperparameters by mean three-seed validation F1; individual thresholds; test metric means and SD, no probability ensemble",
              "redeep_adaptation": "Training Pearson ranking and minmax; validation 288 Khead/Klayer/beta candidates; standardized JSD features supplied by extractor",
              "lumina": "Fixed official lambda=.5 supplied score; only threshold fitted on validation",
              "failed_item_policy": "All resolved asserted labels remain in main denominator; absent predictions mean no alert and risk counted FN. Scorable-only separately reported."}
    save(out/"freeze.json", frozen)
    print("ALL_METHODS_FROZEN", frozen["fit_seconds"], flush=True)


def test(root):
    out = root/"results"
    frozen = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert not (out/"test_complete.json").exists(), "Unified held-out test already completed; preserve results"
    assert set(frozen["method_names"]) == set(METHODS)
    assert frozen["evaluator_sha256"] == sha(Path(__file__))
    assert frozen["data_sha256"] == data_hashes(root)
    assert frozen["frozen_model_sha256"] == sha(out/"frozen_models.pkl")
    assert frozen["selection_sha256"] == sha(out/"selection.json")
    assert all(sha(root/"data"/f"annotations_{s}.jsonl") == h for s, h in frozen["development_annotation_sha256"].items())
    _, _, all_items = metadata(root)
    cohorts = {split: [i for i in all_items if i["split"] == split] for split in ("test", EXTERNAL_SPLIT)}
    assert all(cohorts.values()), "Both main and external cohorts are required for the unified run"
    for split, items in cohorts.items():
        assert [i["item_id"] for i in items] == frozen[f'{"test" if split == "test" else "external"}_item_ids']
        assert (root/"data"/f"annotations_{split}.jsonl").exists(), "Finalize all held-out labels first"
    # No test metrics are exposed until both complete cohorts have been evaluated.
    labels = {s: load_labels(root, items, [s]) for s, items in cohorts.items()}
    label_hashes = {s: sha(root/"data"/f"annotations_{s}.jsonl") for s in cohorts}
    if (out/"test_started.json").exists():
        prior = json.loads((out/"test_started.json").read_text(encoding="utf-8"))
        assert prior["freeze_sha256"] == sha(out/"freeze.json")
        assert prior["held_out_annotation_sha256"] == label_hashes, "Cannot change labels after test started"
        print("RESUMING_IDENTICAL_FROZEN_TEST", flush=True)
    else:
        save(out/"test_started.json", {"utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha(out/"freeze.json"),
                                      "held_out_annotation_sha256": label_hashes, "retuning_allowed": False})
    models = pickle.loads((out/"frozen_models.pkl").read_bytes())
    completed = {}
    with threadpool_limits(limits=4):
        for split, items in cohorts.items():
            bank = Bank(root, items)
            preds = predict(items, labels[split], bank, models, frozen["thresholds"])
            met = evaluate(preds, bootstrap=True)
            met.update(split=split, utc=datetime.now(timezone.utc).isoformat(),
                       unit="answer-item risk F1, not word/token or whole-response F1",
                       freeze_sha256=sha(out/"freeze.json"), annotation_sha256=label_hashes[split])
            completed[split] = preds, met
    for split, (preds, met) in completed.items():
        name = "main" if split == "test" else "external"
        savel(out/f"predictions_{name}.jsonl", preds)
        save(out/f"metrics_{name}.json", met)
        errors = [{**p, "error_methods": [m for m in METHODS if (p["methods"][m]["prediction"] or 0) != p["annotation"]["risk"]]}
                  for p in preds if resolved(p["annotation"]) and any((p["methods"][m]["prediction"] or 0) != p["annotation"]["risk"] for m in METHODS)]
        savel(out/f"errors_{name}.jsonl", errors)
    save(out/"test_complete.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha(out/"freeze.json"), "metrics_sha256": {n: sha(out/f"metrics_{n}.json") for n in ("main", "external")},
        "all_methods_and_both_cohorts_evaluated": True, "held_out_annotation_sha256": label_hashes})
    print("UNIFIED_TEST_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "test"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (fit if args.stage == "fit" else test)(args.root.resolve())


if __name__ == "__main__":
    main()
