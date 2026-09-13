"""Frozen item-detector token-localization diagnostic; never trains a classifier.

fit_thresholds reads validation spans only. test requires freeze8.json, then
evaluates the previously used main/external questions in two threshold regimes.
The original review strategy and all Round 7 files remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R7_ROOT = ROOT.parent/"round7_evidence_grounding"
TOKEN_METHODS = ("hidden_probe", "hallurag_mlp_seed_20260910", "hallurag_mlp_seed_20260911",
                 "hallurag_mlp_seed_20260912", "lookback_lens", "redeep", "lumina", "nll", "entropy")
SCHEMA = "round8-token-localization-v1"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260911
MLP_SEEDS = (20260910, 20260911, 20260912)


def readl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def savel(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text("".join(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def finite(v):
    return v is not None and math.isfinite(float(v))


def positions(text, start, end):
    """Unicode letters/digits, including function words; no punctuation filtering by word type."""
    return {n for n in range(start, end) if text[n].isalnum()}


def merge_spans(spans):
    """Canonical error regions: duplicate, overlapping and touching intervals merge."""
    regions = []
    for start, end in sorted((s["start"], s["end"]) for s in spans):
        if regions and start <= regions[-1][1]:
            regions[-1][1] = max(end, regions[-1][1])
        else:
            regions.append([start, end])
    return regions


def load_gold(path, items, originals, records):
    """Validate every item, exact text/ranges and its immutable original generation."""
    expected = {i["item_id"]: i for i in items}
    gold = {}
    for a in readl(path):
        iid = a["item_id"]
        assert iid in expected and iid not in gold, "Unexpected/duplicate span annotation: "+iid
        item, original = expected[iid], originals[iid]["annotation"]
        generated, generation_hash = records[item["row_id"]]
        for key in ("item_id", "row_id", "question_id", "split", "text", "start", "end"):
            assert a.get(key) == item[key], "Exact span annotation identity mismatch: "+key+" "+iid
        assert a.get("source_generation_sha256") == generation_hash, "Stale generation hash: "+iid
        assert a["original_risk"] == original.get("risk")
        assert a["original_stance"] == original["stance"]
        if "token_scores_viewed" in a:
            assert a["token_scores_viewed"] is False, "Localization gold must remain blind to token scores"
        assert a["localization_status"] in {"resolved", "unresolved", "excluded"}
        if a["localization_status"] == "excluded":
            assert original["stance"] != "asserted" or original.get("risk") not in (0, 1)
        assert isinstance(a["risk_spans"], list)
        response = generated["response"]
        for span in a["risk_spans"]:
            start, end = span["start"], span["end"]
            assert type(start) is int and type(end) is int
            assert item["start"] is not None and item["start"] <= start < end <= item["end"]
            assert span["text"] == response[start:end], "Gold span text mismatch: "+iid
            assert positions(response, start, end), "A risk span must contain a letter or digit"
        if a["localization_status"] == "resolved":
            assert original["stance"] == "asserted" and original.get("risk") in (0, 1)
            assert bool(a["risk_spans"]) == bool(original["risk"]), "Resolved risky items require exhaustive nonempty spans; normal items require none"
        gold[iid] = a
    assert set(gold) == set(expected), "Every predetermined item needs a localization status"
    return gold


def align_row(generated, items, gold, original):
    """One record per original BPE token. Cross-item labels OR, unknown overlap excludes.

    Non-span token label 0 means outside annotated risk locations, not separately
    verified factual truth. Crossing excluded alphanumeric text is never labeled 0.
    """
    text = generated["response"]
    ids, offsets = generated["response_token_ids"], generated["response_token_offsets"]
    assert len(ids) == len(offsets)
    item_chars, risk_chars, eligible_items, regions = {}, {}, set(), []
    for item in items:
        iid = item["item_id"]
        item_chars[iid] = positions(text, item["start"], item["end"]) if item["start"] is not None else set()
        a, old = gold[iid], original[iid]["annotation"]
        resolved = old["stance"] == "asserted" and old.get("risk") in (0, 1) and a["localization_status"] == "resolved"
        if resolved:
            eligible_items.add(iid)
        risk_chars[iid] = set()
        for n, (start, end) in enumerate(merge_spans(a["risk_spans"])):
            chars = positions(text, start, end)
            risk_chars[iid] |= chars
            if resolved:
                regions.append({"span_id": iid+f"__span{n+1}", "item_id": iid, "row_id": item["row_id"],
                                "start": start, "end": end, "text": text[start:end], "characters": sorted(chars), "token_keys": []})
    all_known_chars = set().union(*(item_chars[iid] for iid in eligible_items)) if eligible_items else set()
    all_risk_chars = set().union(*(risk_chars[iid] for iid in eligible_items)) if eligible_items else set()
    tokens = []
    for n, (token_id, offset) in enumerate(zip(ids, offsets)):
        start, end = map(int, offset)
        assert 0 <= start <= end <= len(text)
        chars = positions(text, start, end)
        owners = [item["item_id"] for item in items if chars & item_chars[item["item_id"]]]
        # A cross-boundary BPE token cannot contribute one TP and another TN.
        main = bool(chars and owners) and all(iid in eligible_items for iid in owners) and chars <= all_known_chars
        risky_owner = any(original[iid]["annotation"].get("risk") == 1 for iid in owners)
        token = {"token_key": generated["row_id"]+f"__token{n}", "token_index": n, "token_id": int(token_id),
                 "row_id": generated["row_id"], "question_id": generated["question_id"],
                 "group_id": items[0]["group_id"], "split": items[0]["split"], "condition": items[0]["condition"],
                 "start": start, "end": end, "text": text[start:end], "item_ids": owners,
                 "lexical": bool(chars), "main_eligible": main, "risk_item_eligible": main and risky_owner,
                 "gold": int(bool(chars & all_risk_chars)) if main else None,
                 "characters": sorted(chars), "risk_characters": sorted(chars & all_risk_chars) if main else [],
                 "cross_item": len(owners) > 1, "scores": {},
                 "abstention_overlap": any(original[iid]["annotation"]["stance"] == "abstained" for iid in owners),
                 "unresolved_overlap": any(iid not in eligible_items and original[iid]["annotation"]["stance"] != "abstained" for iid in owners),
                 "outside_items": bool(chars) and not owners}
        tokens.append(token)
        if main:
            for region in regions:
                if chars & set(region["characters"]):
                    region["token_keys"].append(token["token_key"])
    # Regions lacking an evaluable token remain explicit misses, not silently dropped.
    return tokens, regions


def confusion(rows, method, threshold, ranking=True):
    tp = fp = fn = tn = missing = 0
    y, scores = [], []
    for r in rows:
        score = r["scores"].get(method)
        ok = finite(score) and finite(threshold)
        predicted = bool(ok and score >= threshold)
        if not ok:
            missing += 1
        if r["gold"]:
            tp += int(predicted); fn += int(not predicted)
        else:
            fp += int(predicted); tn += int(not predicted)
        if finite(score):
            y.append(r["gold"]); scores.append(score)
    return {"tokens": len(rows), "risk_tokens": tp+fn, "nonspan_tokens": fp+tn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if tp+fn else None,
            "nonspan_false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "nonspan_share_of_alerts": fp/(tp+fp) if tp+fp else None,
            "missing_predictions": missing, "ranking_tokens": len(y),
            "auroc": float(roc_auc_score(y, scores)) if ranking and len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, scores)) if ranking and sum(y) else None}


def choose_threshold(rows, method):
    """Validation risk F1; include all/no-positive endpoints; ties precision then higher threshold."""
    scored = [(float(r["scores"][method]), int(r["gold"])) for r in rows if finite(r["scores"].get(method))]
    positives = sum(r["gold"] for r in rows)
    if not scored or positives == 0 or positives == len(rows):
        return {"threshold": None, "reason": "Both validation token classes and finite scores required", "tokens": len(rows)}
    scored.sort(reverse=True)
    candidates = [(0., 0., math.nextafter(scored[0][0], math.inf))]
    tp = fp = 0
    n = 0
    while n < len(scored):
        value = scored[n][0]
        while n < len(scored) and scored[n][0] == value:
            tp += scored[n][1]; fp += 1-scored[n][1]; n += 1
        f1 = 2*tp/(tp+fp+positives)
        candidates.append((f1, tp/(tp+fp), value))
    candidates.append((2*tp/(tp+fp+positives), tp/(tp+fp), math.nextafter(scored[-1][0], -math.inf)))
    f1, precision, threshold = max(candidates)
    return {"threshold": threshold, "validation_f1": f1, "validation_precision": precision,
            "tokens": len(rows), "scorable_tokens": len(scored), "risk_tokens": positives}


def average(values):
    good = [v for v in values if v is not None]
    return {"mean": float(np.mean(good)) if good else None, "defined_answers": len(good)}


def localization_metrics(tokens, regions, method, threshold, risk_only=False):
    rows = [r for r in tokens if r["risk_item_eligible"]] if risk_only else [r for r in tokens if r["main_eligible"]]
    by_answer = defaultdict(list)
    for r in rows:
        by_answer[r["row_id"]].append(r)
    per_answer = {rid: confusion(rr, method, threshold, ranking=False) for rid, rr in by_answer.items()}
    all_answer_rows = defaultdict(list)
    for r in tokens:
        if r["lexical"]:
            all_answer_rows[r["row_id"]].append(r)
    # A partly excluded answer is not certified to be an entirely normal answer.
    clean_normal_answers = [rid for rid, rr in all_answer_rows.items()
                            if any(r["main_eligible"] for r in rr) and
                            not any(r["unresolved_overlap"] or r["abstention_overlap"] for r in rr) and
                            not any(region["row_id"] == rid for region in regions)]
    if risk_only:
        clean_normal_answers = []
    eligible = {r["token_key"]: r for r in rows}
    alerts = {r["token_key"] for r in rows if finite(threshold) and finite(r["scores"].get(method)) and r["scores"][method] >= threshold}
    region_details = []
    for region in regions:
        keys = set(region["token_keys"]) & set(eligible)
        found = keys & alerts
        predicted_chars = set().union(*(set(eligible[k]["characters"]) for k in found)) if found else set()
        region_details.append({**region, "eligible_tokens": len(keys), "alert_tokens": len(found),
                               "hit": bool(found), "token_coverage": len(found)/len(keys) if keys else 0.,
                               "character_coverage": len(predicted_chars & set(region["characters"]))/len(region["characters"])})
    total_chars = defaultdict(set); gold_chars = defaultdict(set); alert_chars = defaultdict(set)
    for r in rows:
        total_chars[r["row_id"]].update(r["characters"])
        gold_chars[r["row_id"]].update(r["risk_characters"])
        if r["token_key"] in alerts:
            alert_chars[r["row_id"]].update(r["characters"])
    for region in regions:
        gold_chars[region["row_id"]].update(region["characters"])
    char_tp = sum(len(alert_chars[rid] & gold_chars[rid]) for rid in set(total_chars) | set(gold_chars))
    char_alert = sum(len(x) for x in alert_chars.values())
    char_gold = sum(len(x) for x in gold_chars.values())
    normal_with_alert = sum(per_answer.get(rid, {}).get("fp", 0) > 0 for rid in clean_normal_answers)
    normal_tokens = sum(per_answer.get(rid, {}).get("tokens", 0) for rid in clean_normal_answers)
    normal_fp_tokens = sum(per_answer.get(rid, {}).get("fp", 0) for rid in clean_normal_answers)
    return {"micro": confusion(rows, method, threshold),
            "macro_by_answer": {"f1": average([m["f1"] for m in per_answer.values()]),
                                "precision": average([m["precision"] for m in per_answer.values()]),
                                "recall": average([m["recall"] for m in per_answer.values()]),
                                "note": "Macro F1/recall average only answers with gold risk tokens; macro precision only answers with alerts. Denominators differ and are explicit."},
            "fully_normal_answers": {"n": len(clean_normal_answers), "with_false_alert": normal_with_alert,
                                     "any_false_alert_rate": normal_with_alert/len(clean_normal_answers) if clean_normal_answers else None,
                                     "tokens": normal_tokens, "false_positive_tokens": normal_fp_tokens,
                                     "token_false_positive_rate": normal_fp_tokens/normal_tokens if normal_tokens else None},
            "span_regions": {"n": len(region_details), "hit": sum(s["hit"] for s in region_details),
                             "any_hit_recall": sum(s["hit"] for s in region_details)/len(region_details) if region_details else None,
                             "mean_token_coverage": float(np.mean([s["token_coverage"] for s in region_details])) if region_details else None,
                             "mean_character_coverage": float(np.mean([s["character_coverage"] for s in region_details])) if region_details else None,
                             "unscorable_regions": sum(s["eligible_tokens"] == 0 for s in region_details), "details": region_details},
            "character_coverage": {"risk_characters": char_gold, "alert_characters": char_alert,
                                   "risk_characters_alerted": char_tp, "extra_alert_characters": char_alert-char_tp,
                                   "precision": char_tp/char_alert if char_alert else None,
                                   "recall": char_tp/char_gold if char_gold else None},
            "extra_alert_tokens_per_answer": average([m["fp"] for m in per_answer.values()]),
            "per_answer": per_answer}


def describe_alerts(tokens, method, threshold):
    result = {}
    for name, predicate in {"all_lexical_text": lambda r: r["lexical"],
                            "refusal_text": lambda r: r["lexical"] and r["abstention_overlap"],
                            "unresolved_text": lambda r: r["lexical"] and r["unresolved_overlap"],
                            "outside_items": lambda r: r["lexical"] and r["outside_items"]}.items():
        rows = [r for r in tokens if predicate(r)]
        n = sum(finite(r["scores"].get(method)) and finite(threshold) for r in rows)
        count = sum(finite(threshold) and finite(r["scores"].get(method)) and r["scores"][method] >= threshold for r in rows)
        result[name] = {"tokens": len(rows), "scorable": n, "alerts": count,
                        "alert_rate_all_tokens": count/len(rows) if rows else None,
                        "interpretation": "Descriptive alert rate, not false-positive rate against unknown token truth"}
    return result


def group_bootstrap(tokens, methods, metrics, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    groups = sorted({r["group_id"] for r in tokens})
    if not groups:
        return {"groups": 0, "draws": draws, "subsets": {}}
    group_index = {g: i for i, g in enumerate(groups)}
    row_group = {r["row_id"]: r["group_id"] for r in tokens}
    index = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    weights = np.stack([np.bincount(row, minlength=len(groups)) for row in index])
    result = {"groups": len(groups), "draws": draws, "seed": seed, "subsets": {},
              "unit": "Original group_id; paired conditions and every token sampled together",
              "interpretation": "Conditional on frozen heads, thresholds and assistant span labels; no retraining; no multiplicity correction"}
    def interval(values):
        good = np.asarray(values)[np.isfinite(values)]
        return {"ci95": np.quantile(good, [.025, .975]).tolist() if len(good) else None, "defined_draws": len(good)}
    for subset in ("all_resolved_items", "risk_items_only"):
        counts = np.zeros((len(groups), len(methods), 3), dtype=np.int64)
        for j, name in enumerate(methods):
            for row_id, m in metrics[name][subset]["per_answer"].items():
                counts[group_index[row_group[row_id]], j] += [m["tp"], m["fp"], m["fn"]]
        total = np.einsum("bg,gmc->bmc", weights, counts, optimize=True)
        tp, fp, fn = (total[:, :, k] for k in range(3))
        f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.full(tp.shape, np.nan, dtype=float), where=(tp+fn) > 0)
        scores = {name: f1[:, j] for j, name in enumerate(methods)}
        values = {}
        for name, score in scores.items():
            values[name] = {"micro_f1": interval(score)}
            if name.endswith("__token"):
                broadcast = name[:-7]+"__broadcast"
                if broadcast in scores:
                    values[name]["micro_f1_minus_own_broadcast"] = interval(score-scores[broadcast])
        for mode in ("token", "broadcast"):
            names = [f"hallurag_mlp_seed_{s}__{mode}" for s in MLP_SEEDS]
            if all(name in scores for name in names):
                values["mlp_mean__"+mode] = {"micro_f1": interval(np.mean([scores[name] for name in names], axis=0)),
                                               "meaning": "Mean of three separately thresholded F1 values in each group resample, not an ensemble"}
        result["subsets"][subset] = values
    return result


def evaluate(tokens, regions, thresholds, bootstrap=False):
    result = {}
    for method, entry in thresholds.items():
        threshold = entry["threshold"]
        result[method] = {"threshold": threshold,
                          "signal_scope": "native_token" if method.endswith("__token") else "posthoc_item_broadcast",
                          "availability": "Current token and earlier context only; original feature phase is preserved" if method.endswith("__token") else "Uses the completed item's score; may include later tokens within the item, so this is not a causal current-token detector",
                          "all_resolved_items": localization_metrics(tokens, regions, method, threshold),
                          "risk_items_only": localization_metrics(tokens, regions, method, threshold, risk_only=True),
                          "all_text_description": describe_alerts(tokens, method, threshold), "attributes": {}}
        for category in sorted({r.get("attribute_group", "unclassified") for r in tokens}):
            rows = [r for r in tokens if r.get("attribute_group", "unclassified") == category]
            result[method]["attributes"][category] = {
                "all_resolved_items": confusion([r for r in rows if r["main_eligible"]], method, threshold),
                "risk_items_only": confusion([r for r in rows if r["risk_item_eligible"]], method, threshold),
                "original_groups": len({r["group_id"] for r in rows})}
    wrapped = {"methods": result, "mlp_seed_summary": {}}
    for mode in ("token", "broadcast"):
        names = [f"hallurag_mlp_seed_{s}__{mode}" for s in MLP_SEEDS]
        if all(name in result for name in names):
            summary = {"meaning": "Mean and sample SD of three individually thresholded predictors; no score ensemble", "subsets": {}}
            for subset in ("all_resolved_items", "risk_items_only"):
                values = {}
                for key in ("f1", "precision", "recall"):
                    vv = [result[n][subset]["micro"][key] for n in names]
                    vv = [x for x in vv if x is not None]
                    values[key] = {"mean": float(np.mean(vv)) if vv else None,
                                   "sd": float(np.std(vv, ddof=1)) if len(vv) > 1 else None, "n_defined": len(vv)}
                summary["subsets"][subset] = values
            wrapped["mlp_seed_summary"][mode] = summary
    if bootstrap:
        wrapped["group_bootstrap"] = group_bootstrap(tokens, list(thresholds), result)
    return wrapped


def import_r7(r7root):
    path = r7root/"src/evaluate7.py"
    spec = importlib.util.spec_from_file_location("round7_frozen_evaluator_for_round8", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenBank:
    def __init__(self, arrays):
        self.arrays = arrays

    def matrix(self, items, key):
        values = np.asarray(self.arrays[key], dtype=np.float32)
        return values.reshape(len(items), -1)


def load_r7(r7root):
    frozen = json.loads((r7root/"results/freeze.json").read_text(encoding="utf-8"))
    complete = json.loads((r7root/"results/test_complete.json").read_text(encoding="utf-8"))
    assert complete["freeze_sha256"] == sha(r7root/"results/freeze.json")
    assert frozen["frozen_model_sha256"] == sha(r7root/"results/frozen_models.pkl")
    assert frozen["evaluator_sha256"] == sha(r7root/"src/evaluate7.py")
    for key in ("data/inputs.jsonl", "data/generated.jsonl", "data/references.jsonl"):
        assert frozen["data_sha256"][key] == sha(r7root/key)
    module = import_r7(r7root)
    return module, frozen, pickle.loads((r7root/"results/frozen_models.pkl").read_bytes())


def cohort(root, r7root, split, e7, models):
    """Only the explicitly requested split's fine-grained gold is opened."""
    _, generated, all_items = e7.metadata(r7root)
    references = {r["question_id"]: r for r in readl(r7root/"data/references.jsonl")}
    items = [i for i in all_items if i["split"] == split]
    assert items
    rowids = {i["row_id"] for i in items}
    rows = {r["row_id"]: r for r in generated if r["row_id"] in rowids}
    records, paths = {}, []
    for rid in rowids:
        p = r7root/"data/generation_records"/(rid+".json")
        actual = json.loads(p.read_text(encoding="utf-8"))
        for field in ("response", "items", "response_token_ids", "response_token_offsets"):
            assert actual[field] == rows[rid][field]
        records[rid] = actual, sha(p)
    predname = "development_predictions" if split == "validation" else "predictions_main" if split == "test" else "predictions_external"
    originals = {r["item_id"]: r for r in readl(r7root/"results"/(predname+".jsonl")) if r["split"] == split}
    assert set(originals) == {i["item_id"] for i in items}
    goldpath = root/"data"/("spans_"+split+".jsonl")
    gold = load_gold(goldpath, items, originals, records)
    original_annotation_hash = sha(r7root/"data"/("annotations_"+split+".jsonl"))
    for a in gold.values():
        if "source_annotation_sha256" in a:
            assert a["source_annotation_sha256"] == original_annotation_hash, "Stale original item-label file"
    paths.append(goldpath)
    tokens, regions = [], []
    with e7.threadpool_limits(limits=4):
        for rid in sorted(rowids):
            npzpath = root/"data/token_features"/(rid+".npz")
            metapath = npzpath.with_suffix(".json")
            meta = json.loads(metapath.read_text(encoding="utf-8"))
            assert meta["source_generation_sha256"] == records[rid][1]
            assert meta["arrays_sha256"] == sha(npzpath)
            assert meta["r7_freeze_sha256"] == sha(r7root/"results/freeze.json")
            assert meta["extractor_sha256"] == sha(root/"src/extract8.py")
            assert meta["source_model_code_sha256"] == sha(r7root/"src/model7.py")
            assert meta["labels_or_detector_scores_read"] is False
            for stage, checksum in meta["source_r7_arrays_sha256"].items():
                assert stage in {"features", "attention", "lumina"}
                assert sha(r7root/"data"/stage/(rid+".npz")) == checksum
            if "row_id" in meta:
                assert meta["row_id"] == rid
            row_items = [i for i in items if i["row_id"] == rid]
            if "items" in meta:
                assert len(meta["items"]) == len(row_items)
                for m, i in zip(meta["items"], row_items):
                    for key in ("item_id", "start", "end"):
                        assert m[key] == i[key]
            with np.load(npzpath, allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            gen = rows[rid]
            n = len(gen["response_token_ids"])
            assert np.array_equal(arrays["token_indices"], np.arange(n))
            assert np.array_equal(arrays["token_ids"], gen["response_token_ids"])
            assert np.array_equal(np.stack([arrays["token_start"], arrays["token_end"]], axis=1), gen["response_token_offsets"])
            widths = {"hidden_21": 3584, "hidden_28": 3584, "lookback_features": 784,
                      "redeep_ecs": 784, "redeep_pks": 28, "lumina_score": None, "mean_nll": None, "mean_entropy": None}
            for key, width in widths.items():
                assert key in arrays and arrays[key].shape == ((n, width) if width else (n,)), (rid, key)
            local, row_regions = align_row(gen, row_items, gold, originals)
            attribute = references[gen["question_id"]].get("attribute")
            for token in local:
                token.update(attribute=attribute, attribute_group="birth_year" if attribute == "birth year" else "other" if attribute else "unclassified")
            bank = TokenBank(arrays)
            for method in TOKEN_METHODS:
                values = e7.score_method(models[method], bank, local)
                assert values.shape == (n,)
                for t, score in zip(local, values):
                    t["scores"][method+"__token"] = float(score) if finite(score) else None
            for t in local:
                for method in e7.METHODS:
                    scores = [originals[iid]["methods"][method]["score"] for iid in t["item_ids"]]
                    # A token intersecting two items is counted once; broadcast uses max item risk.
                    t["scores"][method+"__broadcast"] = max(scores) if scores and all(finite(x) for x in scores) else None
            tokens.extend(local); regions.extend(row_regions)
            paths.extend((npzpath, metapath))
    coverage = {"items": len(items), "answers": len(rowids), "groups": len({i["group_id"] for i in items}),
                "original_stance": dict(Counter(a["original_stance"] for a in gold.values())),
                "localization_status": dict(Counter(a["localization_status"] for a in gold.values())),
                "original_risk_items": sum(a["original_risk"] == 1 for a in gold.values()),
                "risk_items_without_resolved_locations": sum(a["original_risk"] == 1 and a["localization_status"] != "resolved" for a in gold.values()),
                "all_bpe_tokens": len(tokens), "lexical_tokens": sum(t["lexical"] for t in tokens),
                "main_tokens": sum(t["main_eligible"] for t in tokens),
                "cross_item_tokens": sum(t["cross_item"] for t in tokens), "canonical_error_regions": len(regions),
                "original_span_count": sum(len(a["risk_spans"]) for a in gold.values())}
    return tokens, regions, coverage, {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in paths}


def annotation_lock(root):
    """Require all three span files frozen before scoring; hash bytes, never parse held-out spans."""
    path = root/"data/annotation_freeze.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen", "All localization annotations must be frozen before any scoring"
    expected = manifest["canonical_spans_sha256"]
    assert set(expected) == {"validation", "test", "external_test"}
    for split, checksum in expected.items():
        assert sha(root/"data"/("spans_"+split+".jsonl")) == checksum, "Frozen localization gold changed: "+split
    return {"annotation_freeze_sha256": sha(path), "canonical_spans_sha256": expected}


def source_hashes(root, r7root):
    # Hash test artifacts as bytes without parsing their labels or token-local scores.
    paths = ["results/freeze.json", "results/test_complete.json", "results/frozen_models.pkl", "src/evaluate7.py",
             "data/inputs.jsonl", "data/generated.jsonl", "data/references.jsonl", "results/development_predictions.jsonl",
             "results/predictions_main.jsonl", "results/predictions_external.jsonl",
             "data/annotations_validation.jsonl", "data/annotations_test.jsonl", "data/annotations_external_test.jsonl"]
    manifest_path = root/"data/token_features/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {r["row_id"]+".json" for r in readl(r7root/"data/inputs.jsonl") if r["split"] in {"validation", "test", "external_test"}}
    assert manifest["status"] == "complete" and manifest["expected_rows"] == manifest["completed_rows"] == len(expected)
    assert set(manifest["record_files"]) == expected
    for name, checksum in manifest["record_files"].items():
        assert Path(name).name == name and sha(manifest_path.parent/name) == checksum
    return {"round7": {p: sha(r7root/p) for p in paths}, "protocol_sha256": sha(root/"protocol.json"),
            "annotation_guide_sha256": sha(root/"ANNOTATION_GUIDE.md"), "plan_sha256": sha(root/"PLAN.md"),
            "token_manifest_sha256": sha(manifest_path), **annotation_lock(root)}


def fit_thresholds(root, r7root):
    out = root/"results"
    assert not (out/"freeze8.json").exists(), "Preserve frozen token thresholds"
    sources = source_hashes(root, r7root)
    e7, frozen7, models = load_r7(r7root)
    tokens, regions, coverage, hashes = cohort(root, r7root, "validation", e7, models)
    methods = [m+"__token" for m in TOKEN_METHODS]+[m+"__broadcast" for m in e7.METHODS]
    original, selected = {}, {}
    eligible = [r for r in tokens if r["main_eligible"]]
    for name in methods:
        base = name.rsplit("__", 1)[0]
        original[name] = {"threshold": frozen7["thresholds"][base]["threshold"], "source": "Round7 frozen item threshold"}
        selected[name] = {"threshold": .5, "reason": "Fixed constant baseline"} if base in {"all_positive", "all_negative"} else choose_threshold(eligible, name)
    freeze = {"schema": SCHEMA, "utc": datetime.now(timezone.utc).isoformat(), "stage": "validation_only_thresholds_frozen",
              "method_names": methods, "original_thresholds": original, "validation_token_thresholds": selected,
              "validation_files_sha256": hashes, "source_hashes": sources,
              "evaluator_sha256": sha(Path(__file__)), "validation_coverage": coverage,
              "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "group_id"},
              "heads_retrained": False, "review_strategy_changed": False,
              "timing": {"native": "hidden/Lookback/ReDeEP post-read t; probability/LUMINA predict t at P+t-1; no temporal shift",
                         "broadcast": "Completed-item score broadcast to earlier tokens is a retrospective noncausal control"},
              "threshold_target": "Micro risk-token F1 on all validation items with resolved exhaustive locations; ties precision then higher threshold",
              "diagnostic_scope": "Previously used Round7 held-out questions; not a new independent benchmark or native-method SOTA comparison"}
    assert sources == source_hashes(root, r7root), "Sources changed during validation scoring"
    save(out/"validation_metrics.json", {"coverage": coverage,
          "original_threshold": evaluate(tokens, regions, original), "validation_token_threshold": evaluate(tokens, regions, selected)})
    savel(out/"validation_token_scores.jsonl", tokens)
    save(out/"freeze8.json", freeze)
    print("TOKEN_THRESHOLDS_FROZEN_NO_TEST_GOLD_OPENED", flush=True)


def test(root, r7root):
    out = root/"results"
    assert not (out/"test_complete8.json").exists(), "Preserve completed diagnostic"
    frozen = json.loads((out/"freeze8.json").read_text(encoding="utf-8"))
    assert frozen["evaluator_sha256"] == sha(Path(__file__))
    assert frozen["source_hashes"] == source_hashes(root, r7root)
    assert all(sha(root/p) == h for p, h in frozen["validation_files_sha256"].items())
    e7, _, models = load_r7(r7root)
    gold_hashes = {s: sha(root/"data"/("spans_"+s+".jsonl")) for s in ("test", "external_test")}
    started = {"freeze8_sha256": sha(out/"freeze8.json"), "heldout_gold_sha256": gold_hashes, "retuning_allowed": False}
    if (out/"test_started8.json").exists():
        assert json.loads((out/"test_started8.json").read_text(encoding="utf-8")) == started
    else:
        save(out/"test_started8.json", started)
    completed = {}
    for split, name in (("test", "main"), ("external_test", "external")):
        tokens, regions, coverage, hashes = cohort(root, r7root, split, e7, models)
        result = {"schema": SCHEMA, "split": split, "unit": "Original BPE token containing Unicode letters/digits; no token shift/tolerance",
                  "coverage": coverage, "input_sha256": hashes, "freeze8_sha256": started["freeze8_sha256"],
                  "original_threshold": evaluate(tokens, regions, frozen["original_thresholds"], bootstrap=True),
                  "validation_token_threshold": evaluate(tokens, regions, frozen["validation_token_thresholds"], bootstrap=True),
                  "interpretation": "Localization diagnostic on previously used questions. Non-span label means outside annotated risk span, not independently verified fact. Broadcast scores are item controls, not token-local algorithms."}
        completed[name] = result, tokens
    assert frozen["source_hashes"] == source_hashes(root, r7root), "Sources changed during held-out scoring"
    for name, (result, tokens) in completed.items():
        save(out/("metrics_"+name+".json"), result)
        savel(out/("token_scores_"+name+".jsonl"), tokens)
    save(out/"test_complete8.json", {**started, "utc": datetime.now(timezone.utc).isoformat(),
         "all_methods_both_cohorts_and_threshold_regimes": True,
         "metrics_sha256": {n: sha(out/("metrics_"+n+".json")) for n in completed},
         "classifier_training_performed": False, "review_strategy_changed": False})
    print("TOKEN_LOCALIZATION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit_thresholds", "test"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--r7-root", type=Path, default=R7_ROOT)
    args = parser.parse_args()
    (fit_thresholds if args.stage == "fit_thresholds" else test)(args.root.resolve(), args.r7_root.resolve())
