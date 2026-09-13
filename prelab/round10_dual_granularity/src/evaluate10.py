"""Round10: independently trained item-mean and fine-token Lookback heads.

Import is inert. fit reads train/validation labels only; test requires the original
source/data/gold lock and immutable model selection. No Round9 test cohort is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
METRIC_PATH = ROOT.parent/"round8_token_localization/src/evaluate8.py"
_spec = importlib.util.spec_from_file_location("round10_pure_round8_metrics", METRIC_PATH)
metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metric)
readl, save, savel, sha = metric.readl, metric.save, metric.savel, metric.sha
SCHEMA = "round10-dual-granularity-v1"
FEATURES = ("lb", "lb_surface", "lb_whitebox", "lb_full", "lb_old_binding")
ITEM_METHODS = tuple(f+"__item" for f in FEATURES)
TOKEN_METHODS = tuple(f+"__token" for f in FEATURES)
BROADCAST = {f+"__broadcast": f+"__item" for f in FEATURES}
LOCALIZATION_METHODS = TOKEN_METHODS+tuple(BROADCAST)
LR_C = (.1, 1.)
SEED, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20260912, 20260912, 2000
EXTERNAL_SOURCES = {f"../round7_evidence_grounding/src/{name}": ROOT.parent/"round7_evidence_grounding/src"/name
                    for name in ("model7.py", "attention7.py", "lumina7.py")}
EXTERNAL_SOURCES.update({"../round8_token_localization/src/evaluate8.py": METRIC_PATH,
    **{f"../round9_evidence_binding/src/{name}": ROOT.parent/"round9_evidence_binding/src"/name
       for name in ("binding9.py", "run9.py", "annotation9.py")}})


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def metadata(root):
    rows = readl(root/"data/inputs.jsonl")
    generated = readl(root/"data/generated.jsonl")
    assert len(rows) == len(generated)
    assert len({r["row_id"] for r in rows}) == len(rows)
    assert len({g["row_id"] for g in generated}) == len(generated)
    by_id = {g["row_id"]: g for g in generated}
    splits, pairs, records, items = {}, defaultdict(list), {}, []
    for row in rows:
        rid = row["row_id"]
        assert Path(rid).name == rid and rid not in (".", "..")
        assert row["split"] in ("train", "validation", "test")
        gid = row.get("group_id", row["question_id"])
        assert splits.setdefault(gid, row["split"]) == row["split"], "Group crosses splits"
        pairs[gid].append(row["condition"])
        assert row["expected_items"] == len(row["questions"]) == 1
        gpath = root/"data/generation_records"/(rid+".json")
        g = json.loads(gpath.read_text(encoding="utf-8"))
        assert g == by_id[rid], "Aggregate/row generation disagreement"
        for key in ("row_id", "question_id", "split", "condition"):
            assert g[key] == row[key]
        assert len(g["items"]) == 1
        item = g["items"][0]
        assert item["item_index"] == 1
        if item["start"] is not None:
            assert g["response"][item["start"]:item["end"]] == item["text"]
        items.append({**item, **{k: row[k] for k in ("row_id", "question_id", "split", "condition")},
                      "group_id": gid, "category": row.get("category", "unclassified")})
        records[rid] = g, sha(gpath)
    assert all(sorted(p) == ["complete", "partial"] for p in pairs.values()), "Expected paired conditions"
    assert set(splits.values()) == {"train", "validation", "test"}
    assert len({i["item_id"] for i in items}) == len(items)
    return rows, items, records


def load_gold(root, split, items, records):
    """Exact identity/span/hash checks; only the requested split file is parsed."""
    assert split in ("train", "validation", "test")
    expected = {i["item_id"]: i for i in items if i["split"] == split}
    gold = {}
    for a in readl(root/"data"/f"annotations_{split}.jsonl"):
        iid = a["item_id"]
        assert iid in expected and iid not in gold, "Unexpected/duplicate label"
        item = expected[iid]; g, checksum = records[item["row_id"]]
        for key in ("item_id", "row_id", "question_id", "split", "text", "start", "end"):
            assert a.get(key) == item.get(key), "Annotation identity mismatch: "+key
        assert a["source_generation_sha256"] == checksum, "Stale annotation generation hash"
        assert a["original_stance"] in ("asserted", "tentative", "abstained", "missing")
        assert a["original_risk"] in (0, 1, None)
        assert a["localization_status"] in ("resolved", "excluded", "unresolved")
        assert not a.get("token_scores_viewed", False), "Annotation must be blind"
        assert isinstance(a["risk_spans"], list)
        if a["localization_status"] == "resolved":
            assert a["original_stance"] == "asserted" and a["original_risk"] in (0, 1)
            assert item["parse_ok"] and item["start"] is not None
            assert bool(a["risk_spans"]) == bool(a["original_risk"])
        for span in a["risk_spans"]:
            start, end = span["start"], span["end"]
            assert type(start) is int and type(end) is int
            assert item["start"] is not None and item["start"] <= start < end <= item["end"]
            assert g["response"][start:end] == span["text"]
            assert metric.positions(g["response"], start, end), "Empty/nonlexical risk span"
        gold[iid] = a
    assert set(gold) == set(expected), "Every planned answer needs a label/status"
    return gold


def question_policy(root, split, gold):
    """Safe refusals need a pre-score explicit review list, never a text heuristic."""
    policy = json.loads((root/"data/question_label_policy.json").read_text(encoding="utf-8"))
    assert policy["status"] == "reviewed_frozen"
    path = root/"data"/f"safe_refusals_{split}.json"
    assert sha(path) == policy["reviewed_safe_refusal_files_sha256"][split]
    decisions = json.loads(path.read_text(encoding="utf-8"))
    if "source_annotation_sha256" in decisions:
        assert decisions["source_annotation_sha256"] == sha(root/"data"/f"annotations_{split}.jsonl"), "Stale safe-refusal review"
    assert decisions.get("split", split) == split and decisions.get("status", "reviewed_frozen") == "reviewed_frozen"
    ids = decisions["safe_refusal_item_ids"]
    assert len(ids) == len(set(ids)), "Duplicate safe-refusal decision"
    safe = set(ids)
    assert safe <= set(gold), "Safe-refusal review belongs to another split/output"
    for iid in safe:
        a = gold[iid]
        assert a["split"] == split and a["original_stance"] == "abstained"
        assert a["localization_status"] == "excluded" and a["original_risk"] is None
        assert not a["risk_spans"], "Safe-refusal list contradicts risk spans"
    return safe


def cohort(root, split, meta):
    _, all_items, records = meta
    items = [dict(i) for i in all_items if i["split"] == split]
    gold = load_gold(root, split, items, records)
    safe = question_policy(root, split, gold)
    original = {iid: {"annotation": {"risk": a["original_risk"], "stance": a["original_stance"]}}
                for iid, a in gold.items()}
    tokens, regions = [], []
    for item in items:
        a = gold[item["item_id"]]
        # A known item label can remain evaluable even when its exact span is unresolved.
        asserted = a["original_stance"] == "asserted" and a["original_risk"] in (0, 1)
        safe_refusal = item["item_id"] in safe
        eligible = asserted or safe_refusal
        item.update(gold=a["original_risk"] if asserted else 0 if safe_refusal else None,
                    main_eligible=eligible, asserted_eligible=asserted, reviewed_safe_refusal=safe_refusal,
                    original_stance=a["original_stance"], localization_status=a["localization_status"],
                    scores={}, annotation=a)
        tt, rr = metric.align_row(records[item["row_id"]][0], [item], gold, original)
        for t in tt:
            t["attribute_group"] = item["category"]
        tokens.extend(tt); regions.extend(rr)
    coverage = {"groups": len({i["group_id"] for i in items}), "planned_answers": len(items),
        "item_evaluable": sum(i["main_eligible"] for i in items),
        "risk_answers": sum(i["gold"] == 1 for i in items),
        "asserted_evaluable": sum(i["asserted_eligible"] for i in items),
        "reviewed_safe_refusals": sum(i["reviewed_safe_refusal"] for i in items),
        "refusals_without_explicit_safe_review": [i["item_id"] for i in items if i["original_stance"] == "abstained" and not i["reviewed_safe_refusal"]],
        "stances": dict(Counter(a["original_stance"] for a in gold.values())),
        "localization_statuses": dict(Counter(a["localization_status"] for a in gold.values())),
        "total_bpe_tokens": len(tokens), "eligible_tokens": sum(t["main_eligible"] for t in tokens),
        "risk_tokens": sum(t["gold"] == 1 for t in tokens),
        "failed_parse_answers": sum(not i["parse_ok"] for i in items)}
    return items, tokens, regions, coverage


class Bank:
    """Feature-only loader; no labels or reference/condition fields select features."""
    def __init__(self, root, records):
        self.root, self.records, self.cache, self.names = root, records, {}, {}

    def row(self, rid):
        if rid in self.cache:
            return self.cache[rid]
        g, checksum = self.records[rid]
        n = len(g["response_token_ids"]); offsets = np.asarray(g["response_token_offsets"])
        data = {}
        specs = {"features": {"lookback_features": 784, "binding_features": 32},
                 "soft_features": {"new_features": 16, "surface_features": 8}}
        for folder, widths in specs.items():
            path = self.root/"data"/folder/(rid+".npz")
            side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert side["source_generation_sha256"] == checksum, "Stale feature generation"
            assert side["arrays_sha256"] == sha(path), "Feature arrays hash mismatch"
            assert side.get("row_id", rid) == rid
            with np.load(path, allow_pickle=False) as arrays:
                assert arrays["token_ids"].tolist() == g["response_token_ids"]
                assert np.array_equal(arrays["token_start"], offsets[:, 0])
                assert np.array_equal(arrays["token_end"], offsets[:, 1]), "Token phase/offset mismatch"
                for key, width in widths.items():
                    a = arrays[key].copy()
                    assert a.shape == (n, width), key+" shape"
                    data[key] = a
                    if key != "lookback_features":
                        name_key = {"binding_features": "binding_feature_names", "new_features": "new_feature_names",
                                    "surface_features": "surface_feature_names"}[key]
                        names = side.get(name_key)
                        if names is None and name_key in arrays:
                            names = arrays[name_key].tolist()
                        assert isinstance(names, list) and len(names) == width and len(set(names)) == width, name_key
                        assert self.names.setdefault(key, names) == names, "Feature axes changed"
        lb, sf, white = data["lookback_features"], data["surface_features"], data["new_features"]
        data.update(lb=lb, lb_surface=np.concatenate((lb, sf), axis=1),
                    lb_whitebox=np.concatenate((lb, white), axis=1),
                    lb_full=np.concatenate((lb, sf, white), axis=1),
                    lb_old_binding=np.concatenate((lb, data["binding_features"]), axis=1))
        self.cache[rid] = data
        return data

    def token_matrix(self, rows, feature):
        return np.asarray([self.row(r["row_id"])[feature][r["token_index"]] for r in rows], dtype=np.float32)

    def item_matrix(self, rows, feature):
        """Original R7 adapter: mean ALL overlapping item BPE, including punctuation.

        Mean raw features before LR, not mean probabilities, not gold-filtered tokens.
        """
        values = []
        for item in rows:
            g, _ = self.records[item["row_id"]]; x = self.row(item["row_id"])[feature]
            offsets = np.asarray(g["response_token_offsets"])
            if not item.get("parse_ok") or item["start"] is None:
                values.append(np.full(x.shape[1], np.nan)); continue
            indices = np.flatnonzero((offsets[:, 1] > item["start"]) & (offsets[:, 0] < item["end"]))
            values.append(x[indices].mean(0) if len(indices) else np.full(x.shape[1], np.nan))
        return np.asarray(values, dtype=np.float32)


def base_weights(rows, granularity):
    """Each group equal mass; conditions equal before class weighting; tokens share item mass."""
    assert granularity in ("item", "token") and rows
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for j, r in enumerate(rows):
        iid = r["item_id"] if granularity == "item" else r["item_ids"][0]
        if granularity == "token":
            assert len(r["item_ids"]) == 1
        tree[r["group_id"]][r["condition"]][iid].append(j)
    w = np.zeros(len(rows), float)
    for conditions in tree.values():
        for items in conditions.values():
            for indices in items.values():
                w[indices] = 1/(len(conditions)*len(items)*len(indices))
    return w/w.mean()


def loss_weights(rows, y, base):
    mass = np.bincount(y, weights=base, minlength=2)
    assert (mass > 0).all(), "Both training classes required"
    factors = mass.sum()/(2*mass)
    w = base*factors[y]
    groups = defaultdict(list)
    for j, r in enumerate(rows):
        groups[r["group_id"]].append(j)
    for ix in groups.values():
        w[ix] *= (len(rows)/len(groups))/w[ix].sum()
    return w, factors


def score_array(model, x):
    values = np.full(len(x), np.nan)
    good = np.isfinite(x).all(axis=1)
    if good.any():
        values[good] = model["model"].predict_proba(model["scaler"].transform(x[good]).astype(np.float32))[:, 1]
    return values


def threshold_search(y, values):
    return metric.choose_threshold([{"gold": int(a), "scores": {"x": float(s)}} for a, s in zip(y, values)], "x")


def fit_models(bank, train_items, train_tokens, val_items, val_tokens):
    models, thresholds, candidates, audit = {}, {}, {}, {}
    for granularity, tr, va in (("item", train_items, val_items), ("token", train_tokens, val_tokens)):
        tt = [r for r in tr if r["main_eligible"]]; vv = [r for r in va if r["main_eligible"]]
        y = np.asarray([r["gold"] for r in tt], int); vy = np.asarray([r["gold"] for r in vv], int)
        assert set(y) == set(vy) == {0, 1}, "Both classes required in train and validation"
        b = base_weights(tt, granularity); w, factors = loss_weights(tt, y, b)
        matrix = bank.item_matrix if granularity == "item" else bank.token_matrix
        audit[granularity] = {"observations": len(tt), "groups": len({r["group_id"] for r in tt}),
            "class_factors_train_only": factors, "base_weight_sum": float(b.sum()),
            "weighted_class_mass": np.bincount(y, weights=w, minlength=2),
            "group_final_mass": {g: float(w[[r["group_id"] == g for r in tt]].sum()) for g in sorted({r["group_id"] for r in tt})},
            "scaler_weight": "Label-independent training base weights only",
            "loss_weight": "Train class factors then renormalize each group to equal mass; mean weight 1"}
        for feature in FEATURES:
            name = feature+"__"+granularity
            x, v = matrix(tt, feature), matrix(vv, feature)
            assert np.isfinite(x).all() and np.isfinite(v).all(), "Repair features before fit; no selective dropping"
            scaler = StandardScaler().fit(x, sample_weight=b)
            transformed = scaler.transform(x).astype(np.float32)
            best, candidates[name] = None, []
            for c in LR_C:
                head = LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=2000, random_state=SEED)
                head.fit(transformed, y, sample_weight=w)
                model = {"kind": "lr", "model": head, "scaler": scaler, "feature": feature,
                         "granularity": granularity, "C": c}
                entry = threshold_search(vy, score_array(model, v))
                assert entry["threshold"] is not None
                entry.update(C=c, validation_unit=granularity)
                candidates[name].append(entry)
                # Threshold tie handling is inside threshold_search; lower C breaks model ties.
                key = (entry["validation_f1"], entry["validation_precision"], -c)
                if best is None or key > best[0]:
                    best = key, model, entry
            models[name], thresholds[name] = best[1:]
    for dest, source in BROADCAST.items():
        thresholds[dest] = {**thresholds[source], "threshold_origin": source,
            "calibration": "Unmodified native item threshold; no token-threshold search for broadcast"}
    return models, thresholds, candidates, audit


def predict(bank, items, tokens, models):
    for name, model in models.items():
        rows = items if model["granularity"] == "item" else tokens
        matrix = bank.item_matrix if model["granularity"] == "item" else bank.token_matrix
        scores = score_array(model, matrix(rows, model["feature"]))
        for row, value in zip(rows, scores):
            row["scores"][name] = float(value) if metric.finite(value) else None
    by_id = {i["item_id"]: i for i in items}
    for t in tokens:
        for dest, source in BROADCAST.items():
            values = [by_id[i]["scores"].get(source) for i in t["item_ids"]]
            t["scores"][dest] = max(values) if values and all(metric.finite(v) for v in values) else None
    assert all(set(i["scores"]) == set(ITEM_METHODS) for i in items)
    assert all(set(t["scores"]) == set(LOCALIZATION_METHODS) for t in tokens)


def item_confusion(items, method, threshold):
    value = metric.confusion(items, method, threshold)
    renames = {"tokens": "answers", "risk_tokens": "risk_answers", "nonspan_tokens": "nonrisk_answers",
               "ranking_tokens": "ranking_answers", "nonspan_false_positive_rate": "nonrisk_false_positive_rate",
               "nonspan_share_of_alerts": "nonrisk_share_of_alerts"}
    return {renames.get(k, k): v for k, v in value.items()}


def within_answer(tokens, method):
    grouped = defaultdict(list)
    for t in tokens:
        if t["main_eligible"]:
            grouped[t["row_id"]].append(t)
    values = {rid: metric.confusion(tt, method, .5) for rid, tt in grouped.items() if {t["gold"] for t in tt} == {0, 1}}
    return {"mixed_answers": len(values), "mean_auroc": metric.average([v["auroc"] for v in values.values()]),
        "mean_average_precision": metric.average([v["average_precision"] for v in values.values()]),
        "per_answer": values, "interpretation": "Within mixed answers; constant whole-item broadcast has AUROC 0.5"}


def ci(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "defined_draws": len(finite)}


def bootstrap(items, tokens, thresholds, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    groups = sorted({i["group_id"] for i in items}); gi = {g: j for j, g in enumerate(groups)}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), (draws, len(groups)))
    weights = np.stack([np.bincount(s, minlength=len(groups)) for s in sampled])
    out = {"groups": len(groups), "draws": draws, "seed": seed,
           "unit": "group_id; both conditions and all tokens sampled together, same draws at both granularities",
           "interpretation": "Conditional on frozen development selection and assistant labels; no retraining or multiplicity correction",
           "subsets": {}}
    subsets = {"answer_items": ([i for i in items if i["main_eligible"]], ITEM_METHODS),
               "answer_asserted_only": ([i for i in items if i.get("asserted_eligible")], ITEM_METHODS),
               "all_resolved_items": ([t for t in tokens if t["main_eligible"]], LOCALIZATION_METHODS),
               "risk_items_only": ([t for t in tokens if t["risk_item_eligible"]], LOCALIZATION_METHODS)}
    for subset, (rows, methods) in subsets.items():
        counts = np.zeros((len(groups), len(methods), 3), dtype=np.int64)
        for r in rows:
            for j, name in enumerate(methods):
                pred = metric.finite(r["scores"].get(name)) and r["scores"][name] >= thresholds[name]["threshold"]
                counts[gi[r["group_id"]], j] += [int(pred and r["gold"]), int(pred and not r["gold"]), int(not pred and r["gold"])]
        total = np.einsum("bg,gmc->bmc", weights, counts, optimize=True)
        tp, fp, fn = (total[:, :, k] for k in range(3))
        def divide(a, b, ok):
            return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=ok)
        arrays = {"precision": divide(tp, tp+fp, tp+fp > 0), "recall": divide(tp, tp+fn, tp+fn > 0),
                  "f1": divide(2*tp, 2*tp+fp+fn, tp+fn > 0)}
        values = {name: {k: a[:, j] for k, a in arrays.items()} for j, name in enumerate(methods)}
        is_item = subset.startswith("answer_")
        suffix = "__item" if is_item else "__token"
        contrasts = {"primary_full_minus_lb": ("lb_full"+suffix, "lb"+suffix),
                     "whitebox_increment_over_surface": ("lb_full"+suffix, "lb_surface"+suffix),
                     "whitebox_only_minus_lb": ("lb_whitebox"+suffix, "lb"+suffix),
                     "surface_only_increment": ("lb_surface"+suffix, "lb"+suffix),
                     "old_binding_minus_lb": ("lb_old_binding"+suffix, "lb"+suffix)}
        if not is_item:
            contrasts.update({f+"_native_minus_item_broadcast": (f+"__token", f+"__broadcast") for f in FEATURES})
        out["subsets"][subset] = {"methods": {n: {k: ci(v) for k, v in vv.items()} for n, vv in values.items()},
            "contrasts": {name: {k: ci(values[a][k]-values[b][k]) for k in arrays} for name, (a, b) in contrasts.items()}}
    return out


def evaluate(items, tokens, regions, thresholds, with_bootstrap=False):
    answer, local = {}, {}
    for name in ITEM_METHODS:
        threshold = thresholds[name]["threshold"]
        eligible = [i for i in items if i["main_eligible"]]
        answer[name] = {"threshold": threshold, "scope": "Completed single-question answer; mean raw item BPE features",
            "micro": item_confusion(eligible, name, threshold),
            "asserted_only": item_confusion([i for i in items if i.get("asserted_eligible")], name, threshold),
            "reviewed_safe_refusals": item_confusion([i for i in items if i.get("reviewed_safe_refusal")], name, threshold),
            "strata": {}, "excluded_alerts": {}}
        for axis in ("condition", "category"):
            answer[name]["strata"][axis] = {v: item_confusion([i for i in eligible if i[axis] == v], name, threshold)
                                             for v in sorted({i[axis] for i in items})}
        for stance in ("abstained", "tentative", "missing", "asserted_unresolved"):
            rr = [i for i in items if not i["main_eligible"] and
                  (i["original_stance"] == stance or stance == "asserted_unresolved" and i["original_stance"] == "asserted")]
            answer[name]["excluded_alerts"][stance] = {"answers": len(rr),
                "scorable": sum(metric.finite(i["scores"].get(name)) for i in rr),
                "alerts": sum(metric.finite(i["scores"].get(name)) and i["scores"][name] >= threshold for i in rr),
                "meaning": "Descriptive only; not confirmed false positives"}
    for name in LOCALIZATION_METHODS:
        threshold = thresholds[name]["threshold"]
        local[name] = {"threshold": threshold, "scope": "posthoc_item_broadcast" if name in BROADCAST else "native_causal_token",
            "all_resolved_items": metric.localization_metrics(tokens, regions, name, threshold),
            "risk_items_only": metric.localization_metrics(tokens, regions, name, threshold, risk_only=True),
            "within_answer_ranking": within_answer(tokens, name),
            "all_text_description": metric.describe_alerts(tokens, name, threshold), "strata": {}}
        for axis in ("condition", "attribute_group"):
            local[name]["strata"][axis] = {}
            for value in sorted({t[axis] for t in tokens}):
                tt = [t for t in tokens if t[axis] == value]
                local[name]["strata"][axis][value] = {"all_resolved_items": metric.confusion([t for t in tt if t["main_eligible"]], name, threshold),
                    "risk_items_only": metric.confusion([t for t in tt if t["risk_item_eligible"]], name, threshold)}
    out = {"answer_methods": answer, "localization_methods": local,
        "primary_method_fixed_before_test": "lb_full", "primary_baseline": "lb",
        "whitebox_increment_baseline": "lb_surface", "review_strategy_changed": False,
        "interpretation": "Answer and token heads are separately trained/calibrated. Main answer denominator includes explicitly reviewed safe refusals as risk 0; token denominator excludes refusals. Asserted-only answer result is secondary. Native token features use the current token and prefix; item means/broadcasts use the completed answer. Nonspan tokens mean outside annotated risk spans, not individually certified true facts."}
    if with_bootstrap:
        out["group_bootstrap"] = bootstrap(items, tokens, thresholds)
    return out


def verify_signature(side, dependencies, root, stage):
    signature = side["stage_signature"]
    assert signature["stage"] == stage
    assert side["stage_signature_sha256"] == digest(signature), "Stage signature digest changed"
    actual = {str(p.resolve()): sha(p) for p in (root/"src").glob("*.py")}
    actual.update({str(EXTERNAL_SOURCES[k].resolve()): v for k, v in dependencies.items()})
    expected = [root/"src/run10.py", ROOT.parent/"round7_evidence_grounding/src/model7.py",
                ROOT.parent/"round7_evidence_grounding/src/attention7.py",
                ROOT.parent/"round9_evidence_binding/src/binding9.py", ROOT.parent/"round9_evidence_binding/src/run9.py"]
    if stage == "soft":
        expected.append(root/"src/feature10.py")
    assert set(signature["code_sha256"]) == {str(p.resolve()) for p in expected}, "Incomplete runtime signature"
    for name, checksum in signature["code_sha256"].items():
        assert actual.get(str(Path(name).resolve())) == checksum, "Extraction dependency changed: "+name


def validate_protocol(protocol):
    assert protocol["schema"] == SCHEMA
    assert tuple(protocol["feature_sets"]) == FEATURES
    assert protocol["primary_method"] == "lb_full"
    assert tuple(protocol["lr"]["C"]) == LR_C
    assert protocol["lr"]["seed"] == SEED and protocol["lr"]["solver"] == "liblinear"
    assert protocol["lr"]["penalty"] == "l2" and protocol["lr"]["max_iter"] == 2000
    assert protocol["bootstrap"]["seed"] == BOOTSTRAP_SEED and protocol["bootstrap"]["draws"] == BOOTSTRAP_DRAWS
    for key, width in (("lookback_dimensions", 784), ("new_dimensions", 16), ("surface_dimensions", 8), ("old_binding_dimensions", 32)):
        assert protocol["features"][key] == width


def source_hashes(root):
    """Connect pre-generation freeze, exact reused bytes, runtime dependencies and blind gold.

    Test annotation files are hashed as opaque bytes here, never JSON-parsed.
    """
    frozen = json.loads((root/"data/freeze.json").read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen" and frozen["created_before_test_generation"]
    validate_protocol(json.loads((root/"protocol.json").read_text(encoding="utf-8")))
    required = {"data/inputs.jsonl", "data/reuse_manifest.json", "protocol.json", "PLAN.md", "ANNOTATION_GUIDE.md"}
    sources = {p.relative_to(root).as_posix() for p in (root/"src").rglob("*.py")}
    assert required | sources <= set(frozen["files_sha256"]), "Pre-generation freeze omitted data/protocol/source"
    paths = {root/name for name in frozen["files_sha256"]}
    for name, checksum in frozen["files_sha256"].items():
        path = (root/name).resolve()
        assert root.resolve() in path.parents, "Frozen local path escapes root"
        assert sha(path) == checksum, "Pre-generation data/code freeze changed: "+name
    dependencies = {k: sha(p) for k, p in EXTERNAL_SOURCES.items()}
    assert frozen["external_source_sha256"] == dependencies, "External frozen dependency changed"
    reuse = json.loads((root/"data/reuse_manifest.json").read_text(encoding="utf-8"))
    assert reuse["excludes_round9_test"]
    oldroot = ROOT.parent/"round9_evidence_binding"
    for name, key in (("data/freeze.json", "source_data_freeze_sha256"),
                      ("data/annotation_freeze.json", "source_annotation_freeze_sha256")):
        assert sha(oldroot/name) == reuse[key], "Original reuse source freeze changed"
    assert sha(root/"data/dev_inputs.jsonl") == reuse["dev_inputs_sha256"]
    rows = readl(root/"data/inputs.jsonl")
    assert {r["row_id"] for r in rows if r["split"] in ("train", "validation")} == set(reuse["rows"])
    for name, checksum in reuse["annotation_files"].items():
        assert sha(root/name) == sha(oldroot/name) == checksum, "Development labels no longer exact copies"
    for row in rows:
        rid = row["row_id"]; assert Path(rid).name == rid and rid not in (".", "..")
        gp = root/"data/generation_records"/(rid+".json")
        g = json.loads(gp.read_text(encoding="utf-8")); paths.add(gp)
        reused = reuse["rows"].get(rid)
        if reused:
            assert digest(row) == reused["input_row_sha256"]
            for name, checksum in reused["files_sha256"].items():
                assert sha(root/name) == sha(oldroot/name) == checksum, "Changed development reuse artifact"
        else:
            assert row["split"] == "test" and g["input_row_sha256"] == digest(row)
            verify_signature(g, dependencies, root, "generate")
        for folder in ("features", "soft_features"):
            sp = root/"data"/folder/(rid+".json")
            side = json.loads(sp.read_text(encoding="utf-8")); paths.update((sp, sp.with_suffix(".npz")))
            assert side["source_generation_sha256"] == sha(gp)
            assert side["arrays_sha256"] == sha(sp.with_suffix(".npz"))
            if not reused or folder == "soft_features":
                assert side["input_row_sha256"] == digest(row)
                verify_signature(side, dependencies, root, "core" if folder == "features" else "soft")
    lock = json.loads((root/"data/annotation_freeze.json").read_text(encoding="utf-8"))
    assert lock["status"] == "frozen" and set(lock["canonical_spans_sha256"]) == {"train", "validation", "test"}
    for split, checksum in lock["canonical_spans_sha256"].items():
        path = root/"data"/f"annotations_{split}.jsonl"
        assert sha(path) == checksum, "Labels changed after blind annotation freeze"
        paths.add(path)
    assert lock["question_label_policy_sha256"] == sha(root/"data/question_label_policy.json"), "Safe-refusal policy changed"
    policy = json.loads((root/"data/question_label_policy.json").read_text(encoding="utf-8"))
    assert policy["status"] == "reviewed_frozen"
    assert set(policy["reviewed_safe_refusal_files_sha256"]) == {"train", "validation", "test"}
    for split, checksum in policy["reviewed_safe_refusal_files_sha256"].items():
        path = root/"data"/f"safe_refusals_{split}.json"
        assert sha(path) == checksum, "Safe-refusal decisions changed"
        paths.add(path)
    paths.update(root/p for p in ("data/annotation_freeze.json", "data/freeze.json", "data/dev_inputs.jsonl", "data/generated.jsonl", "data/question_label_policy.json"))
    return {"files": {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)},
            "evaluator_sha256": sha(Path(__file__)), "external_source_sha256": dependencies}


def fit(root):
    out = root/"results"
    assert not (out/"freeze10.json").exists(), "Never overwrite frozen selection"
    sources = source_hashes(root); meta = metadata(root)
    ti, tt, _, tc = cohort(root, "train", meta)
    vi, vt, vr, vc = cohort(root, "validation", meta)
    bank = Bank(root, meta[2]); start = time.perf_counter()
    models, thresholds, candidates, audit = fit_models(bank, ti, tt, vi, vt)
    seconds = time.perf_counter()-start
    predict(bank, vi, vt, models)
    assert source_hashes(root) == sources, "Sources changed during fit"
    out.mkdir(parents=True, exist_ok=True)
    (out/"frozen_models.pkl").write_bytes(pickle.dumps(models, protocol=5))
    save(out/"selection.json", candidates); save(out/"training_weights.json", audit)
    save(out/"validation_metrics.json", {"coverage": vc, **evaluate(vi, vt, vr, thresholds)})
    savel(out/"answer_scores_validation.jsonl", vi); savel(out/"token_scores_validation.jsonl", vt)
    frozen = {"schema": SCHEMA, "utc": utc(), "stage": "fit_frozen_no_test_labels_parsed", "source_hashes": sources,
        "model_sha256": sha(out/"frozen_models.pkl"), "selection_sha256": sha(out/"selection.json"),
        "training_weights_sha256": sha(out/"training_weights.json"), "thresholds": thresholds,
        "models": tuple(models), "feature_names": bank.names, "train_coverage": tc, "validation_coverage": vc,
        "fit_wall_seconds": seconds, "cpu_threads": 4, "C_candidates": LR_C, "seed": SEED,
        "primary_method": "lb_full", "primary_baseline": "lb", "whitebox_increment_baseline": "lb_surface",
        "native_threshold_targets": "Each head uses its own granularity's unweighted validation risk F1; ties precision, then higher threshold; candidate C ties lower C",
        "broadcast_threshold": "Original item head threshold, no extra calibration",
        "review_strategy_changed": False}
    save(out/"freeze10.json", frozen)
    print("ROUND10_FROZEN_NO_TEST_LABELS_PARSED", flush=True)


def test(root):
    out = root/"results"
    assert not (out/"test_complete10.json").exists(), "Do not rerun completed heldout test"
    frozen = json.loads((out/"freeze10.json").read_text(encoding="utf-8"))
    assert frozen["schema"] == SCHEMA and frozen["source_hashes"] == source_hashes(root)
    for name, key in (("frozen_models.pkl", "model_sha256"), ("selection.json", "selection_sha256"),
                      ("training_weights.json", "training_weights_sha256")):
        assert sha(out/name) == frozen[key], "Frozen model/selection changed"
    started = {"freeze10_sha256": sha(out/"freeze10.json"), "retuning_allowed": False}
    sp = out/"test_started10.json"
    if sp.exists():
        assert json.loads(sp.read_text(encoding="utf-8")) == started
    else:
        save(sp, started)
    begin = time.perf_counter()
    models = pickle.loads((out/"frozen_models.pkl").read_bytes())
    assert set(models) == set(ITEM_METHODS+TOKEN_METHODS)
    meta = metadata(root); items, tokens, regions, coverage = cohort(root, "test", meta)
    bank = Bank(root, meta[2])
    for rid in sorted({i["row_id"] for i in items}):
        bank.row(rid)
    assert bank.names == frozen["feature_names"]
    load_seconds = time.perf_counter()-begin
    begin = time.perf_counter(); predict(bank, items, tokens, models); score_seconds = time.perf_counter()-begin
    result = {"schema": SCHEMA, "split": "test", "coverage": coverage, **started,
        "load_seconds": load_seconds, "cpu_score_seconds": score_seconds,
        **evaluate(items, tokens, regions, frozen["thresholds"], with_bootstrap=True)}
    assert source_hashes(root) == frozen["source_hashes"]
    save(out/"metrics_test.json", result)
    savel(out/"answer_scores_test.jsonl", items); savel(out/"token_scores_test.jsonl", tokens)
    save(out/"test_complete10.json", {**started, "utc": utc(), "models": 10, "broadcast_controls": 5,
        "metrics_sha256": sha(out/"metrics_test.json"), "answer_scores_sha256": sha(out/"answer_scores_test.jsonl"),
        "token_scores_sha256": sha(out/"token_scores_test.jsonl"), "review_strategy_changed": False})
    print("ROUND10_DUAL_GRANULARITY_TEST_COMPLETE", flush=True)


def report(root):
    """Automatic factual tables only; no training, scoring or posthoc winner selection."""
    out = root/"results"; done = json.loads((out/"test_complete10.json").read_text(encoding="utf-8"))
    assert sha(out/"metrics_test.json") == done["metrics_sha256"]
    m = json.loads((out/"metrics_test.json").read_text(encoding="utf-8"))
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    lines = ["# Round10 双粒度固定评测", "", "主方法预定为 LB+full；回答级与词元级是分别训练、分别验证阈值的探针。", "",
             "| 特征 | 回答级 P | R | F1 | 词元级 P | R | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FEATURES:
        a = m["answer_methods"][f+"__item"]["micro"]
        t = m["localization_methods"][f+"__token"]["all_resolved_items"]["micro"]
        lines.append("| "+f+" | "+" | ".join(fmt(x[k]) for x in (a, t) for k in ("precision", "recall", "f1"))+" |")
    lines += ["", "回答级：一条输入对应一个短回答，任何已裁定风险即为正，逐项复核过的纯拒答作为无编造负例；未决与缺失排除并公开覆盖。词元级：原始 BPE 精确位置，正常事实回答中的误报保留，拒答不自动生成为全零词元标签。", "",
              "整项广播只作事后定位对照，使用完成后的整项信息；不能当成实时词元定位。拒答与未决单列覆盖。", "",
              "完整计数、两级配对置信区间、分层、正常回答误报、句内排序和全部逐项分数见 metrics_test.json 与两个 scores_test.jsonl。",
              "", "本次测试题组未进入开发；训练/验证沿用 Round9，已有开发集被反复研究，结果只针对本次模型、资料与助手审核标签。"]
    (out/"AUTOMATIC_TABLES.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("fit", "test", "report"))
    p.add_argument("--root", type=Path, default=ROOT); a = p.parse_args()
    with threadpool_limits(limits=4):
        {"fit": fit, "test": test, "report": report}[a.stage](a.root.resolve())


if __name__ == "__main__":
    main()
