"""Round9 group-balanced coarse/fine token supervision. Never runs on import.

fit parses train/validation gold only; test requires immutable frozen artifacts.
Round8's pure alignment/counting functions are reused, never its data loader.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "4")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
METRIC_PATH = ROOT.parent/"round8_token_localization/src/evaluate8.py"
EXTERNAL_SOURCES = {f"../round7_evidence_grounding/src/{name}": ROOT.parent/"round7_evidence_grounding/src"/name
                    for name in ("model7.py", "attention7.py", "lumina7.py")}
EXTERNAL_SOURCES["../round8_token_localization/src/evaluate8.py"] = METRIC_PATH
_spec = importlib.util.spec_from_file_location("round8_pure_metrics", METRIC_PATH)
metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metric)
readl, save, savel, sha = metric.readl, metric.save, metric.savel, metric.sha
SCHEMA = "round9-evidence-binding-evaluation-v1"
SEEDS = (20260910, 20260911, 20260912)
LR_C = (.1, 1.)
LRS = (1e-4, 3e-4)
MLP_CONFIG = {"widths": (128, 64), "dropout": .1, "weight_decay": .01,
              "batch_size": 128, "max_epochs": 60, "patience": 8, "min_delta": 1e-6}
LR_METHODS = {"lb_coarse": ("lookback_features", "coarse"),
              "lb_fine": ("lookback_features", "fine"),
              "binding_coarse": ("bound", "coarse"), "binding_fine": ("bound", "fine"),
              "hidden_lr": ("hidden_28", "fine")}
MLP_NAMES = tuple(f"hidden_mlp_seed_{s}" for s in SEEDS)
BROADCAST = {"lb_coarse_broadcast": "lb_coarse", "binding_coarse_broadcast": "binding_coarse"}
METHODS = tuple(LR_METHODS)+MLP_NAMES+("nll", "entropy", "redeep", "lumina", "all_positive", "all_negative")+tuple(BROADCAST)
CONTRASTS = {"fine_minus_coarse_global": ("lb_fine", "lb_coarse"),
             "fine_minus_coarse_binding": ("binding_fine", "binding_coarse"),
             "binding_minus_global_fine": ("binding_fine", "lb_fine"),
             "binding_minus_global_coarse": ("binding_coarse", "lb_coarse"),
             "coarse_global_minus_broadcast": ("lb_coarse", "lb_coarse_broadcast"),
             "coarse_binding_minus_broadcast": ("binding_coarse", "binding_coarse_broadcast")}


def utc():
    return datetime.now(timezone.utc).isoformat()


def metadata(root):
    inputs = readl(root/"data/inputs.jsonl")
    generated = {g["row_id"]: g for g in readl(root/"data/generated.jsonl")}
    assert len(generated) == len(inputs) and len({r["row_id"] for r in inputs}) == len(inputs)
    group_splits, items, records = {}, [], {}
    for row in inputs:
        rid = row["row_id"]
        assert Path(rid).name == rid and row["split"] in {"train", "validation", "test"}
        group = row.get("group_id", row["question_id"])
        assert group_splits.setdefault(group, row["split"]) == row["split"], "Question group crosses split"
        assert row["expected_items"] == 1 and len(row["questions"]) == 1
        g = generated[rid]
        path = root/"data/generation_records"/(rid+".json")
        actual = json.loads(path.read_text(encoding="utf-8"))
        assert actual == g, "Aggregate generation differs from frozen row record"
        for key in ("row_id", "question_id", "split", "condition"):
            assert g[key] == row[key]
        assert len(g["items"]) == 1
        records[rid] = g, sha(path)
        for item in g["items"]:
            assert item["item_index"] == 1
            if item["start"] is not None:
                assert g["response"][item["start"]:item["end"]] == item["text"]
            items.append({**item, **{k: row[k] for k in ("row_id", "question_id", "split", "condition")},
                          "group_id": group, "category": row.get("category", "unclassified")})
    assert len({i["item_id"] for i in items}) == len(items)
    assert {r["split"] for r in inputs} == {"train", "validation", "test"}
    return inputs, items, records


def load_gold(root, split, items, records):
    """Read only the explicitly requested split, binding labels to actual text/hash."""
    assert split in {"train", "validation", "test"}
    expected = {i["item_id"]: i for i in items if i["split"] == split}
    gold = {}
    for a in readl(root/"data"/f"annotations_{split}.jsonl"):
        iid = a["item_id"]
        assert iid in expected and iid not in gold, "Unexpected/duplicate annotation"
        item = expected[iid]
        g, digest = records[item["row_id"]]
        for key in ("item_id", "row_id", "question_id", "split", "text", "start", "end"):
            assert a.get(key) == item.get(key), "Annotation identity mismatch: "+key+" "+iid
        assert a["source_generation_sha256"] == digest, "Stale annotation generation hash"
        assert a.get("original_stance") in {"asserted", "tentative", "abstained", "missing"}
        assert a.get("original_risk") in (0, 1, None)
        assert a["localization_status"] in {"resolved", "unresolved", "excluded"}
        assert not a.get("token_scores_viewed", False), "Gold must be blind to new scores"
        assert isinstance(a["risk_spans"], list)
        if a["localization_status"] == "resolved":
            assert a["original_stance"] == "asserted" and a["original_risk"] in (0, 1)
            assert item["parse_ok"] and item["start"] is not None
            assert bool(a["risk_spans"]) == bool(a["original_risk"])
        if a["localization_status"] == "excluded":
            assert a["original_stance"] != "asserted" or a["original_risk"] is None
        for span in a["risk_spans"]:
            start, end = span["start"], span["end"]
            assert type(start) is int and type(end) is int
            assert item["start"] is not None and item["start"] <= start < end <= item["end"]
            assert g["response"][start:end] == span["text"]
            assert metric.positions(g["response"], start, end)
        gold[iid] = a
    assert set(gold) == set(expected), "Every predeclared slot must have a label/status"
    return gold


def cohort(root, split, meta=None):
    inputs, all_items, records = meta or metadata(root)
    items = [i for i in all_items if i["split"] == split]
    gold = load_gold(root, split, items, records)
    original = {iid: {"annotation": {"risk": a["original_risk"], "stance": a["original_stance"]}} for iid, a in gold.items()}
    tokens, regions = [], []
    for item in items:
        g, _ = records[item["row_id"]]
        tt, rr = metric.align_row(g, [item], gold, original)
        for token in tt:
            token["coarse_gold"] = gold[item["item_id"]]["original_risk"] if token["main_eligible"] else None
            token["attribute_group"] = item["category"]
        tokens.extend(tt); regions.extend(rr)
    coverage = {"groups": len({i["group_id"] for i in items}), "rows": len(items),
                "items": len(items), "statuses": dict(Counter(a["localization_status"] for a in gold.values())),
                "stances": dict(Counter(a["original_stance"] for a in gold.values())),
                "all_tokens": len(tokens), "eligible_tokens": sum(t["main_eligible"] for t in tokens),
                "risk_tokens": sum(t["gold"] == 1 for t in tokens),
                "failed_parse_items": sum(not i["parse_ok"] for i in items)}
    return tokens, regions, coverage


class Bank:
    """No labels are read. Row matrices align to original generated token positions."""
    def __init__(self, root, records):
        self.root, self.records, self.cache = root, records, {}
        self.binding_names = None

    def row(self, rid):
        if rid in self.cache:
            return self.cache[rid]
        g, digest = self.records[rid]
        n = len(g["response_token_ids"])
        data = {}
        for folder in ("features", "attention", "lumina"):
            path = self.root/"data"/folder/(rid+".npz")
            side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert side["source_generation_sha256"] == digest, "Stale feature generation"
            assert side["arrays_sha256"] == sha(path), "Feature array hash mismatch"
            assert side.get("row_id", rid) == rid
            with np.load(path, allow_pickle=False) as arrays:
                if folder == "features":
                    for key in ("hidden_28", "lookback_features", "binding_features", "token_nll", "token_entropy", "token_ids", "token_start", "token_end"):
                        data[key] = arrays[key].copy()
                    names = side.get("binding_feature_names")
                    if names is None and "binding_feature_names" in arrays:
                        names = arrays["binding_feature_names"].tolist()
                    assert isinstance(names, list) and len(names) == data["binding_features"].shape[1]
                    assert len(names) == len(set(names)), "Duplicate binding feature name"
                    if self.binding_names is None:
                        self.binding_names = names
                    assert names == self.binding_names, "Binding feature axes changed"
                elif folder == "attention":
                    data["redeep_ecs"] = arrays["token_redeep_ecs"].copy()
                    data["redeep_pks"] = arrays["token_redeep_pks"].copy()
                else:
                    data["lumina"] = arrays["token_lumina_score"].copy()
                if "token_ids" in arrays:
                    assert arrays["token_ids"].tolist() == g["response_token_ids"]
        offsets = np.asarray(g["response_token_offsets"])
        assert data["token_ids"].tolist() == g["response_token_ids"]
        assert np.array_equal(data["token_start"], offsets[:, 0]) and np.array_equal(data["token_end"], offsets[:, 1])
        for key, width in (("hidden_28", 3584), ("lookback_features", 784), ("redeep_ecs", 784), ("redeep_pks", 28)):
            assert data[key].shape == (n, width), key+" shape"
        assert data["binding_features"].ndim == 2 and data["binding_features"].shape[0] == n
        for key in ("token_nll", "token_entropy", "lumina"):
            assert data[key].shape == (n,), key+" shape"
        data["bound"] = np.concatenate((data["lookback_features"], data["binding_features"]), axis=1)
        self.cache[rid] = data
        return data

    def matrix(self, tokens, feature):
        return np.asarray([self.row(t["row_id"])[feature][t["token_index"]] for t in tokens], dtype=np.float32)


def base_weights(tokens):
    """Equal group, condition, item mass; no label or sentence-length feature."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for j, token in enumerate(tokens):
        assert token["main_eligible"] and len(token["item_ids"]) == 1
        tree[token["group_id"]][token["condition"]][token["item_ids"][0]].append(j)
    assert tokens
    weights = np.zeros(len(tokens), dtype=float)
    for conditions in tree.values():
        for items in conditions.values():
            for indices in items.values():
                weights[indices] = 1/(len(conditions)*len(items)*len(indices))
    return weights/weights.mean()


def class_factors(y, base):
    mass = np.bincount(np.asarray(y, int), weights=base, minlength=2)
    assert (mass > 0).all(), "Both train classes required"
    return mass.sum()/(2*mass)


def loss_weights(tokens, y, base, factors):
    weights = base*np.asarray(factors)[np.asarray(y, int)]
    groups = defaultdict(list)
    for j, t in enumerate(tokens):
        groups[t["group_id"]].append(j)
    for indices in groups.values():
        weights[indices] *= (len(tokens)/len(groups))/weights[indices].sum()
    assert np.isclose(weights.mean(), 1.)
    return weights


def threshold_search(y, scores):
    return metric.choose_threshold([{"gold": int(v), "scores": {"x": float(s)}} for v, s in zip(y, scores)], "x")


def choice(entry, *simplicity):
    assert entry.get("threshold") is not None, "Validation must have both classes and usable predictions"
    return entry["validation_f1"], entry["validation_precision"], *simplicity


class MLP(torch.nn.Module):
    def __init__(self, width, config):
        super().__init__()
        layers = []
        for target in config["widths"]:
            layers.extend((torch.nn.Linear(width, target), torch.nn.ReLU(), torch.nn.Dropout(config["dropout"])))
            width = target
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(-1)


def train_mlp(X, y, V, vy, weight, vweight, lr, seed, config=None):
    config = dict(MLP_CONFIG if config is None else config)
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    net = MLP(X.shape[1], config).cpu()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=config["weight_decay"])
    tx, ty, tw, vx, vyy, vw = [torch.tensor(a, dtype=torch.float32, device="cpu") for a in (X, y, weight, V, vy, vweight)]
    history, best, wait, best_state, best_epoch = [], math.inf, 0, None, None
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"]+1):
        net.train(); total = 0.
        for ids in torch.randperm(len(X), generator=gen).split(config["batch_size"]):
            opt.zero_grad(set_to_none=True)
            # Mean-1 global weights; per-batch renormalization would change the objective.
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(tx[ids]), ty[ids], reduction="none")*tw[ids]).mean()
            loss.backward(); opt.step()
            total += float(loss.detach())*len(ids)
        net.eval()
        with torch.no_grad():
            valid = float((torch.nn.functional.binary_cross_entropy_with_logits(net(vx), vyy, reduction="none")*vw).mean())
        history.append({"epoch": epoch, "train_bce": total/len(X), "validation_bce": valid})
        if valid < best-config["min_delta"]:
            best, wait, best_epoch = valid, 0, epoch
            best_state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if wait >= config["patience"]:
            break
    assert best_state is not None
    return {"kind": "mlp", "input_width": X.shape[1], "state_dict": best_state, "config": config,
            "seed": seed, "lr": lr, "training": {"history": history, "best_epoch": best_epoch,
            "best_validation_bce": best, "seconds": time.perf_counter()-start}}


def weighted_rank(X, target, weight):
    X = np.asarray(X, float); w = weight/weight.sum()
    centered = X-np.sum(X*w[:, None], axis=0)
    target = np.asarray(target, float)-np.dot(target, w)
    denom = np.sqrt(np.sum(centered**2*w[:, None], axis=0)*np.dot(target**2, w))
    corr = np.divide(centered.T@(target*w), denom, out=np.zeros(X.shape[1]), where=denom > 0)
    return np.lexsort((np.arange(X.shape[1]), -corr)), corr


def score_array(model, X):
    scores = np.full(len(X), np.nan)
    good = np.isfinite(X).all(1)
    if not good.any():
        return scores
    transformed = model["scaler"].transform(X[good]).astype(np.float32)
    if model["kind"] == "lr":
        scores[good] = model["model"].predict_proba(transformed)[:, 1]
    else:
        net = MLP(model["input_width"], model["config"]).cpu()
        net.load_state_dict(model["state_dict"]); net.eval()
        with torch.no_grad():
            v = torch.tensor(transformed, device="cpu")
            scores[good] = torch.cat([net(chunk).sigmoid() for chunk in v.split(1024)]).numpy()
    return scores


def redeep_scores(model, E, P):
    scores = np.full(len(E), np.nan)
    good = np.isfinite(E).all(1) & np.isfinite(P).all(1)
    e = E[good][:, model["heads"]].sum(1, dtype=np.float64)
    p = P[good][:, model["layers"]].sum(1, dtype=np.float64)
    scores[good] = (p-model["p_min"])/model["p_range"]-model["beta"]*(e-model["e_min"])/model["e_range"]
    return scores


def score_method(model, bank, tokens):
    if model["kind"] in {"lr", "mlp"}:
        return score_array(model, bank.matrix(tokens, model["feature"]))
    if model["kind"] == "redeep":
        return redeep_scores(model, bank.matrix(tokens, "redeep_ecs"), bank.matrix(tokens, "redeep_pks"))
    if model["kind"] == "constant":
        return np.full(len(tokens), model["value"], dtype=float)
    assert model["kind"] == "raw"
    return bank.matrix(tokens, model["feature"]).astype(float)


def apply_broadcast(tokens):
    """Annotation-independent mean: every lexical token overlapping the item, not gold-eligible only."""
    items = defaultdict(list)
    for t in tokens:
        if t["lexical"]:
            for iid in t["item_ids"]:
                items[iid].append(t)
    for dest, source in BROADCAST.items():
        scores = {}
        for iid, tt in items.items():
            values = [t["scores"].get(source) for t in tt]
            scores[iid] = float(np.mean(values)) if values and all(metric.finite(v) for v in values) else None
        for t in tokens:
            values = [scores[i] for i in t["item_ids"]]
            t["scores"][dest] = max(values) if values and all(metric.finite(v) for v in values) else None


def predict(bank, tokens, models):
    for name, model in models.items():
        score = score_method(model, bank, tokens)
        assert len(score) == len(tokens)
        for token, value in zip(tokens, score):
            token["scores"][name] = float(value) if metric.finite(value) else None
    apply_broadcast(tokens)
    assert all(set(t["scores"]) == set(METHODS) for t in tokens)


def fit_models(bank, train, validation, mlp_config=None):
    tt = [t for t in train if t["main_eligible"]]
    vv = [t for t in validation if t["main_eligible"]]
    assert tt and vv
    yt = np.asarray([t["gold"] for t in tt], int); yv = np.asarray([t["gold"] for t in vv], int)
    yc = np.asarray([t["coarse_gold"] for t in tt], int)
    assert len(set(yt)) == len(set(yv)) == len(set(yc)) == 2
    b, vb = base_weights(tt), base_weights(vv)
    models, selected, candidates, scaling = {}, {}, {}, {}
    factors = {"fine": class_factors(yt, b), "coarse": class_factors(yc, b)}
    weights = {"fine": loss_weights(tt, yt, b, factors["fine"]), "coarse": loss_weights(tt, yc, b, factors["coarse"])}
    features = {}
    for key in ("lookback_features", "bound", "hidden_28", "redeep_ecs", "redeep_pks"):
        X, V = bank.matrix(tt, key), bank.matrix(vv, key)
        assert np.isfinite(X).all() and np.isfinite(V).all(), "Repair feature extraction before fit; do not select/drop tokens: "+key
        features[key] = X, V
        if key in {"lookback_features", "bound", "hidden_28"}:
            scaling[key] = StandardScaler().fit(X, sample_weight=b)
    for name, (feature, target) in LR_METHODS.items():
        X, V = features[feature]; y = yt if target == "fine" else yc
        scaler = scaling[feature]; xx = scaler.transform(X).astype(np.float32)
        best, candidates[name] = None, []
        for C in LR_C:
            estimator = LogisticRegression(C=C, penalty="l2", solver="liblinear", class_weight=None,
                                           max_iter=2000, random_state=SEEDS[0])
            estimator.fit(xx, y, sample_weight=weights[target])
            model = {"kind": "lr", "model": estimator, "scaler": scaler, "feature": feature,
                     "supervision": target, "C": C}
            threshold = threshold_search(yv, score_array(model, V))
            entry = {"C": C, "threshold": threshold}
            candidates[name].append(entry)
            rank = choice(threshold, -C)
            if best is None or rank > best[0]:
                best = rank, model, threshold
        _, models[name], selected[name] = best
    X, V = features["hidden_28"]; scaler = scaling["hidden_28"]
    xx, vx = [scaler.transform(a).astype(np.float32) for a in (X, V)]
    vweight = loss_weights(vv, yv, vb, factors["fine"])
    best, candidates["hidden_mlp"] = None, []
    for lr in LRS:
        ms, thresholds, entries = {}, {}, []
        for seed, name in zip(SEEDS, MLP_NAMES):
            model = train_mlp(xx, yt, vx, yv, weights["fine"], vweight, lr, seed, mlp_config)
            model.update(scaler=scaler, feature="hidden_28", supervision="fine")
            threshold = threshold_search(yv, score_array(model, V))
            choice(threshold)
            ms[name], thresholds[name] = model, threshold
            entries.append({"seed": seed, "threshold": threshold, "training": model["training"]})
        avgf = float(np.mean([v["validation_f1"] for v in thresholds.values()]))
        avgp = float(np.mean([v["validation_precision"] for v in thresholds.values()]))
        entry = {"lr": lr, "mean_validation_f1": avgf, "mean_validation_precision": avgp, "seeds": entries}
        candidates["hidden_mlp"].append(entry)
        rank = avgf, avgp, -lr
        if best is None or rank > best[0]:
            best = rank, ms, thresholds
    models.update(best[1]); selected.update(best[2])
    E, VE = features["redeep_ecs"]; P, VP = features["redeep_pks"]
    heads, ec = weighted_rank(E, 1-yt, b); layers, pc = weighted_rank(P, yt, b)
    best, candidates["redeep"] = None, []
    for kh in (1, 4):
        for kl in (4, 8):
            h, l = heads[:kh], layers[:kl]
            ev, pv = E[:, h].sum(1, dtype=np.float64), P[:, l].sum(1, dtype=np.float64)
            for beta in (.1, .5, 1.):
                model = {"kind": "redeep", "heads": h, "layers": l, "beta": beta,
                         "e_min": float(ev.min()), "e_range": float(np.ptp(ev)) or 1.,
                         "p_min": float(pv.min()), "p_range": float(np.ptp(pv)) or 1.}
                threshold = threshold_search(yv, redeep_scores(model, VE, VP))
                entry = {"k_heads": kh, "k_layers": kl, "beta": beta, "threshold": threshold}
                candidates["redeep"].append(entry)
                rank = choice(threshold, -kh, -kl, -beta)
                if best is None or rank > best[0]:
                    best = rank, model, threshold
    _, models["redeep"], selected["redeep"] = best
    candidates["redeep_ranking"] = {"heads": heads, "ecs_correlation_to_1_minus_risk": ec,
                                     "layers": layers, "pks_correlation_to_risk": pc,
                                     "weight": "label-independent training group weights"}
    for name, feature in (("nll", "token_nll"), ("entropy", "token_entropy"), ("lumina", "lumina")):
        models[name] = {"kind": "raw", "feature": feature}
        values = score_method(models[name], bank, vv)
        assert np.isfinite(values).all(), "Validation raw features incomplete"
        selected[name] = threshold_search(yv, values); choice(selected[name])
    for name, value in (("all_positive", 1.), ("all_negative", 0.)):
        models[name] = {"kind": "constant", "value": value}
        selected[name] = {"threshold": .5, "source": "Fixed constant"}
    predict(bank, validation, models)
    for name in BROADCAST:
        selected[name] = metric.choose_threshold(vv, name); choice(selected[name])
    assert set(models) == set(METHODS)-set(BROADCAST) and set(selected) == set(METHODS)
    audit = {"train_token_keys": [t["token_key"] for t in tt], "validation_token_keys": [t["token_key"] for t in vv],
             "train_base_weights": b, "class_factors": factors, "train_loss_weights": weights,
             "weighted_class_mass": {k: np.bincount(yt if k == "fine" else yc, weights=w, minlength=2) for k, w in weights.items()},
             "scaler_weight": "Shared label-independent training base weights", "coarse_validation": "Fine token gold calibrates both rows"}
    return models, selected, candidates, audit


def bootstrap(tokens, results, draws=2000, seed=20260911):
    groups = sorted({t["group_id"] for t in tokens}); gi = {g: j for j, g in enumerate(groups)}
    row_group = {t["row_id"]: t["group_id"] for t in tokens}
    sampled = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    w = np.stack([np.bincount(s, minlength=len(groups)) for s in sampled])
    def interval(v):
        v = np.asarray(v); good = v[np.isfinite(v)]
        return {"ci95": np.quantile(good, [.025, .975]).tolist() if len(good) else None, "defined_draws": len(good)}
    out = {"groups": len(groups), "draws": draws, "seed": seed, "unit": "group_id: both conditions and all tokens kept together", "subsets": {}}
    for subset in ("all_resolved_items", "risk_items_only"):
        counts = np.zeros((len(groups), len(METHODS), 3), dtype=np.int64)
        for j, name in enumerate(METHODS):
            for rid, m in results[name][subset]["per_answer"].items():
                counts[gi[row_group[rid]], j] += [m["tp"], m["fp"], m["fn"]]
        total = np.einsum("bg,gmc->bmc", w, counts, optimize=True)
        tp, fp, fn = (total[:, :, i] for i in range(3))
        f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.full(tp.shape, np.nan), where=tp+fn > 0)
        values = {name: f1[:, j] for j, name in enumerate(METHODS)}
        contrasts = {key: interval(values[a]-values[b]) for key, (a, b) in CONTRASTS.items()}
        contrasts["interaction"] = interval(values["binding_fine"]-values["binding_coarse"]-values["lb_fine"]+values["lb_coarse"])
        out["subsets"][subset] = {"methods": {n: interval(v) for n, v in values.items()}, "contrasts": contrasts,
                                 "mlp_seed_metric_mean": interval(np.mean([values[n] for n in MLP_NAMES], axis=0))}
    return out


def within_answer_ranking(tokens, method):
    """Only mixed gold-label answers identify within-answer discrimination.

    Averaging local AUROC removes between-answer score offsets. All-red/constant
    broadcasting has AUROC .5 even if the right risky answers were selected.
    """
    groups = defaultdict(list)
    for t in tokens:
        if t["main_eligible"]:
            groups[t["row_id"]].append(t)
    per_answer = {}
    for rid, rows in groups.items():
        if {r["gold"] for r in rows} != {0, 1}:
            continue
        m = metric.confusion(rows, method, .5)
        per_answer[rid] = {k: m[k] for k in ("tokens", "risk_tokens", "missing_predictions", "ranking_tokens", "auroc", "average_precision")}
    return {"mixed_answers": len(per_answer), "per_answer": per_answer,
            "mean_auroc": metric.average([m["auroc"] for m in per_answer.values()]),
            "mean_average_precision": metric.average([m["average_precision"] for m in per_answer.values()]),
            "interpretation": "Ranking within each answer containing risk and nonspan tokens; constant item broadcast AUROC is .5"}


def evaluate(tokens, regions, thresholds, do_bootstrap=False):
    results = {}
    for name in METHODS:
        threshold = thresholds[name]["threshold"]
        results[name] = {"threshold": threshold,
                         "scope": "posthoc_item_mean_broadcast" if name in BROADCAST else "native_token",
                         "all_resolved_items": metric.localization_metrics(tokens, regions, name, threshold),
                         "risk_items_only": metric.localization_metrics(tokens, regions, name, threshold, risk_only=True),
                         "within_answer_ranking": within_answer_ranking(tokens, name),
                         "all_text_description": metric.describe_alerts(tokens, name, threshold), "strata": {}}
        for axis in ("condition", "attribute_group"):
            results[name]["strata"][axis] = {}
            for value in sorted({t[axis] for t in tokens}):
                tt = [t for t in tokens if t[axis] == value]
                results[name]["strata"][axis][value] = {"groups": len({t["group_id"] for t in tt}),
                    "all_resolved_items": metric.confusion([t for t in tt if t["main_eligible"]], name, threshold),
                    "risk_items_only": metric.confusion([t for t in tt if t["risk_item_eligible"]], name, threshold)}
    summary = {"interpretation": "Mean and sample SD of individually thresholded seed metrics; not score ensemble", "subsets": {}}
    for subset in ("all_resolved_items", "risk_items_only"):
        summary["subsets"][subset] = {}
        for key in ("f1", "precision", "recall", "auroc", "average_precision"):
            v = [results[n][subset]["micro"][key] for n in MLP_NAMES]
            v = [x for x in v if x is not None]
            summary["subsets"][subset][key] = {"mean": float(np.mean(v)) if v else None,
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n_defined": len(v)}
    out = {"methods": results, "mlp_seed_summary": summary}
    if do_bootstrap:
        out["group_bootstrap"] = bootstrap(tokens, results)
    return out


def data_freeze_links(root):
    """Connect current artifacts to the original pre-generation freeze, not a new snapshot."""
    root = Path(root).resolve()
    path = root/"data/freeze.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen.get("schema") == "round9-input-freeze-v1" and frozen.get("status") == "frozen"
    files = frozen["files_sha256"]
    assert {"data/inputs.jsonl", "data/lumina_random_manifest.json"} <= set(files)
    def local(name):
        dest = (root/name).resolve()
        assert dest == root or root in dest.parents, "Local frozen path escapes data root"
        return dest
    for name, digest in files.items():
        assert sha(local(name)) == digest, "Original data/code freeze changed: "+name
    actual_sources = sorted(p.relative_to(root).as_posix() for p in (root/"src").rglob("*.py"))
    assert actual_sources == frozen["locked_source_files"], "Frozen local code inventory changed"
    protocol = local(frozen["protocol_file"])
    assert protocol == root/"protocol.json"
    assert frozen["protocol_sha256"] == sha(protocol), "Original protocol freeze changed"
    dependency_hashes = {name: sha(p) for name, p in EXTERNAL_SOURCES.items()}
    assert frozen["external_source_sha256"] == dependency_hashes, "Frozen Round7/8 dependency changed"
    rows = readl(root/"data/inputs.jsonl")
    assert len(rows) == frozen["expected_rows"] and len({r["row_id"] for r in rows}) == len(rows)
    assert len({r["group_id"] for r in rows}) == frozen["expected_groups"]
    expected = {"data_freeze_sha256": sha(path), "protocol_sha256": sha(protocol)}
    for row in rows:
        rid = row["row_id"]
        assert Path(rid).name == rid and rid not in (".", "..")
        gp = root/"data/generation_records"/(rid+".json")
        generated = json.loads(gp.read_text(encoding="utf-8"))
        for key, value in expected.items():
            assert generated.get(key) == value, "Generation belongs to a different original freeze: "+rid+" "+key
        assert generated["row_id"] == rid
        visible = {"system": str(row["system"]), "prompt": str(row["prompt"]),
                   "questions": [str(q) for q in row["questions"]],
                   "passages": [{"title": str(p["title"]), "text": str(p["text"])} for p in row["passages"]]}
        visible_hash = hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert generated.get("visible_input_sha256") == visible_hash, "Generation does not match currently displayed input"
        for folder in ("features", "attention", "lumina"):
            sp = root/"data"/folder/(rid+".json")
            side = json.loads(sp.read_text(encoding="utf-8"))
            for key, value in {**expected, "source_generation_sha256": sha(gp), "visible_input_sha256": visible_hash}.items():
                assert side.get(key) == value, "Feature belongs to a different original freeze: "+rid+" "+folder+" "+key
            assert side.get("arrays_sha256") == sha(sp.with_suffix(".npz")), "Frozen feature array differs from its sidecar"
    return frozen, rows, dependency_hashes


def source_hashes(root):
    """Hash all split labels as bytes, never parse held-out annotation content."""
    data_lock, rows, dependencies = data_freeze_links(root)
    lock_path = root/"data/annotation_freeze.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "frozen"
    expected = lock["canonical_spans_sha256"]
    assert set(expected) == {"train", "validation", "test"}
    for split, digest in expected.items():
        assert sha(root/"data"/f"annotations_{split}.jsonl") == digest, "Gold changed after blind annotation freeze"
    paths = [root/p for p in ("data/inputs.jsonl", "data/generated.jsonl", "data/freeze.json", "data/annotation_freeze.json",
                             "protocol.json", "PLAN.md", "ANNOTATION_GUIDE.md")]
    paths += [root/"data"/f"annotations_{s}.jsonl" for s in expected]
    paths += [root/name for name in data_lock["files_sha256"]]
    for row in rows:
        rid = row["row_id"]
        paths.append(root/"data/generation_records"/(rid+".json"))
        for folder in ("features", "attention", "lumina"):
            for suffix in (".npz", ".json"):
                paths.append(root/"data"/folder/(rid+suffix))
    return {"files": {p.relative_to(root).as_posix(): sha(p) for p in paths},
            "evaluator_sha256": sha(Path(__file__)), "pure_metric_source_sha256": sha(METRIC_PATH),
            "external_source_sha256": dependencies}


def fit(root):
    out = root/"results"
    assert not (out/"freeze9.json").exists(), "Never overwrite frozen model selection"
    sources = source_hashes(root)
    meta = metadata(root)
    train, _, traincoverage = cohort(root, "train", meta)
    validation, regions, valcoverage = cohort(root, "validation", meta)
    bank = Bank(root, meta[2]); start = time.perf_counter()
    models, thresholds, candidates, audit = fit_models(bank, train, validation)
    seconds = time.perf_counter()-start
    assert sources == source_hashes(root), "Inputs changed during fit"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out/"frozen_models.pkl"
    model_path.write_bytes(pickle.dumps(models, protocol=5))
    save(out/"selection.json", candidates)
    save(out/"training_weights.json", audit)
    save(out/"validation_metrics.json", {"coverage": valcoverage, **evaluate(validation, regions, thresholds)})
    savel(out/"validation_token_scores.jsonl", validation)
    frozen = {"schema": SCHEMA, "utc": utc(), "stage": "train_validation_fit_frozen_no_test_labels_read",
              "source_hashes": sources, "method_names": METHODS, "thresholds": thresholds,
              "model_sha256": sha(model_path), "selection_sha256": sha(out/"selection.json"),
              "training_weights_sha256": sha(out/"training_weights.json"), "binding_feature_names": bank.binding_names,
              "train_coverage": traincoverage, "validation_coverage": valcoverage,
              "fit_wall_seconds": seconds, "cpu_threads": 4, "mlp_config": MLP_CONFIG,
              "mlp_seeds": SEEDS, "review_strategy_changed": False,
              "threshold_target": "unweighted validation token micro F1; precision then higher threshold",
              "broadcast_mean": "All lexical tokens of completed item; independent of annotation status"}
    save(out/"freeze9.json", frozen)
    print("ROUND9_FROZEN_NO_TEST_LABELS_READ", flush=True)


def test(root):
    out = root/"results"
    assert not (out/"test_complete9.json").exists(), "Do not rerun completed held-out evaluation"
    frozen = json.loads((out/"freeze9.json").read_text(encoding="utf-8"))
    assert frozen["source_hashes"] == source_hashes(root), "Frozen data/code/labels changed"
    assert sha(out/"frozen_models.pkl") == frozen["model_sha256"]
    assert sha(out/"selection.json") == frozen["selection_sha256"]
    assert sha(out/"training_weights.json") == frozen["training_weights_sha256"]
    started = {"freeze9_sha256": sha(out/"freeze9.json"), "retuning_allowed": False}
    if (out/"test_started9.json").exists():
        assert json.loads((out/"test_started9.json").read_text(encoding="utf-8")) == started
    else:
        save(out/"test_started9.json", started)
    load_start = time.perf_counter()
    models = pickle.loads((out/"frozen_models.pkl").read_bytes())
    meta = metadata(root); tokens, regions, coverage = cohort(root, "test", meta)
    bank = Bank(root, meta[2])
    for rid in sorted({t["row_id"] for t in tokens}):
        bank.row(rid)
    assert bank.binding_names == frozen["binding_feature_names"]
    load_seconds = time.perf_counter()-load_start
    start = time.perf_counter(); predict(bank, tokens, models); score_seconds = time.perf_counter()-start
    result = {"schema": SCHEMA, "split": "test", "coverage": coverage, "freeze9_sha256": started["freeze9_sha256"],
              "load_seconds": load_seconds, "cpu_score_seconds": score_seconds,
              "interpretation": "Token localization on new held-out groups; nonspan words are not separately verified truths. Broadcast controls use completed items.",
              **evaluate(tokens, regions, frozen["thresholds"], do_bootstrap=True)}
    assert frozen["source_hashes"] == source_hashes(root)
    save(out/"metrics_test.json", result); savel(out/"token_scores_test.jsonl", tokens)
    # Include every item with alerts, misses or exclusions, rather than curated best cases.
    savel(out/"error_tokens_test.jsonl", [t for t in tokens if not t["main_eligible"] or
          any(bool(metric.finite(t["scores"][n]) and t["scores"][n] >= frozen["thresholds"][n]["threshold"]) != bool(t["gold"]) for n in METHODS)])
    save(out/"test_complete9.json", {**started, "utc": utc(), "all_16_predictors": True,
         "metrics_sha256": sha(out/"metrics_test.json"), "token_scores_sha256": sha(out/"token_scores_test.jsonl"),
         "review_strategy_changed": False})
    print("ROUND9_TEST_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "test"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        (fit if args.stage == "fit" else test)(args.root.resolve())


if __name__ == "__main__":
    main()
