"""Independent Round10 gold/count/paired-CI audit. No evaluator/model imports.

gold reads frozen source labels; predictions additionally requires test_complete10.
self-test is entirely synthetic. Only new audit files under results are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
FEATURES = ("lb", "lb_surface", "lb_whitebox", "lb_full", "lb_old_binding")
ITEM_METHODS = tuple(f+"__item" for f in FEATURES)
TOKEN_METHODS = tuple(f+"__token" for f in FEATURES)+tuple(f+"__broadcast" for f in FEATURES)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def direct_gold(row, generation, annotation, safe_ids):
    """Loop over original Unicode character positions, without evaluator set alignment."""
    require(len(generation["items"]) == 1, "Exactly one planned answer required")
    item = generation["items"][0]; iid = item["item_id"]
    text = generation["response"]; start, end = item["start"], item["end"]
    stance, risk = annotation["original_stance"], annotation["original_risk"]
    asserted = stance == "asserted" and risk in (0, 1)
    safe = iid in safe_ids
    if safe:
        require(stance == "abstained" and risk is None and annotation["localization_status"] == "excluded"
                and not annotation["risk_spans"], "Safe review conflicts with annotation")
    resolved = asserted and annotation["localization_status"] == "resolved"
    answer = {**{k: row[k] for k in ("row_id", "question_id", "group_id", "condition", "split", "category")},
              "item_id": iid, "gold": risk if asserted else 0 if safe else None,
              "main_eligible": asserted or safe, "asserted_eligible": asserted, "reviewed_safe_refusal": safe,
              "original_stance": stance, "localization_status": annotation["localization_status"],
              "start": start, "end": end, "text": item["text"], "parse_ok": item["parse_ok"]}
    offsets, ids = generation["response_token_offsets"], generation["response_token_ids"]
    require(len(offsets) == len(ids), "BPE lengths differ")
    tokens = []
    for index, (token_id, pair) in enumerate(zip(ids, offsets)):
        left, right = pair
        require(type(left) is int and type(right) is int and 0 <= left <= right <= len(text), "Invalid token offset")
        letters = [j for j in range(left, right) if text[j].isalnum()]
        inside = [start is not None and start <= j < end for j in letters]
        owner = bool(letters and any(inside))
        main = bool(resolved and letters and all(inside))
        risky = any(s["start"] <= j < s["end"] for j in letters for s in annotation["risk_spans"])
        tokens.append({**{k: row[k] for k in ("row_id", "question_id", "group_id", "condition", "split")},
            "attribute_group": row["category"], "token_key": row["row_id"]+f"__token{index}",
            "token_index": index, "token_id": token_id, "start": left, "end": right, "text": text[left:right],
            "item_ids": [iid] if owner else [], "lexical": bool(letters), "main_eligible": main,
            "risk_item_eligible": main and risk == 1, "gold": int(risky) if main else None,
            "abstention_overlap": owner and stance == "abstained",
            "unresolved_overlap": owner and not resolved and stance != "abstained",
            "outside_items": bool(letters) and not owner})
    return answer, tokens


def coverage(items, tokens):
    return {"groups": len({i["group_id"] for i in items}), "planned_answers": len(items),
        "item_evaluable": sum(i["main_eligible"] for i in items), "risk_answers": sum(i["gold"] == 1 for i in items),
        "asserted_evaluable": sum(i["asserted_eligible"] for i in items),
        "reviewed_safe_refusals": sum(i["reviewed_safe_refusal"] for i in items),
        "refusals_without_explicit_safe_review": [i["item_id"] for i in items if i["original_stance"] == "abstained" and not i["reviewed_safe_refusal"]],
        "stances": dict(Counter(i["original_stance"] for i in items)),
        "localization_statuses": dict(Counter(i["localization_status"] for i in items)),
        "total_bpe_tokens": len(tokens), "eligible_tokens": sum(t["main_eligible"] for t in tokens),
        "risk_tokens": sum(t["gold"] == 1 for t in tokens),
        "failed_parse_answers": sum(not i["parse_ok"] for i in items)}


def collect_gold(root):
    lock = read_json(root/"data/annotation_freeze.json")
    require(lock["status"] == "frozen", "Wait for frozen labels")
    require(set(lock["canonical_spans_sha256"]) == set(SPLITS), "Incomplete splits")
    policy_path = root/"data/question_label_policy.json"
    require(sha(policy_path) == lock["question_label_policy_sha256"], "Question policy changed")
    policy = read_json(policy_path); require(policy["status"] == "reviewed_frozen", "Safe reviews not final")
    labels, safe, source_hashes = {}, set(), {}
    for split in SPLITS:
        ap = root/"data"/f"annotations_{split}.jsonl"; rp = root/"data"/f"safe_refusals_{split}.json"
        require(sha(ap) == lock["canonical_spans_sha256"][split], "Annotation SHA differs")
        require(sha(rp) == policy["reviewed_safe_refusal_files_sha256"][split], "Safe review SHA differs")
        aa = read_lines(ap); rr = read_json(rp); listed = rr["safe_refusal_item_ids"]
        require(len(listed) == len(set(listed)), "Duplicate safe review")
        require(set(listed) <= {a["item_id"] for a in aa}, "Wrong-split safe review")
        if "source_annotation_sha256" in rr:
            require(rr["source_annotation_sha256"] == sha(ap), "Stale safe refusal review")
        safe.update(listed)
        for a in aa:
            require(a["split"] == split and a["item_id"] not in labels, "Label split/duplicate")
            labels[a["item_id"]] = a
        source_hashes[f"data/annotations_{split}.jsonl"] = sha(ap)
        source_hashes[f"data/safe_refusals_{split}.json"] = sha(rp)
    frozen = read_json(root/"data/freeze.json")
    require(frozen["status"] == "frozen", "Input freeze incomplete")
    for name, digest in frozen["files_sha256"].items():
        require(sha(root/name) == digest, "Original source/data freeze changed: "+name)
    for name, digest in frozen["external_source_sha256"].items():
        require(sha(root/name) == digest, "External dependency changed: "+name)
    inputs = read_lines(root/"data/inputs.jsonl"); gg = read_lines(root/"data/generated.jsonl")
    generations = {g["row_id"]: g for g in gg}
    require(len(generations) == len(gg) == len(inputs), "Duplicate or missing generations")
    require(len({r["row_id"] for r in inputs}) == len(inputs), "Duplicate input")
    require(set(generations) == {r["row_id"] for r in inputs}, "Generation/input IDs differ")
    group_rows, used, answers, tokens = defaultdict(list), set(), [], []
    for row in inputs:
        rid = row["row_id"]; group_rows[row["group_id"]].append(row)
        require(row["split"] in SPLITS and row["expected_items"] == len(row["questions"]) == 1, "Unsupported slot schema")
        gp = root/"data/generation_records"/(rid+".json"); g = read_json(gp)
        require(g == generations[rid], "Aggregate generation differs")
        for key in ("row_id", "question_id", "condition", "split"):
            require(g[key] == row[key], "Generation identity differs")
        require(len(g["items"]) == 1, "Not single answer")
        item = g["items"][0]; iid = item["item_id"]
        require(iid in labels and iid not in used, "Missing/duplicate item label")
        used.add(iid); a = labels[iid]
        for key in ("row_id", "question_id", "split"):
            require(a[key] == row[key], "Annotation row identity differs")
        for key in ("item_id", "text", "start", "end"):
            require(a[key] == item[key], "Annotation exact span identity differs")
        require(a["source_generation_sha256"] == sha(gp), "Stale generation label")
        require(not a.get("token_scores_viewed", False), "Not blind labels")
        if item["start"] is not None:
            require(g["response"][item["start"]:item["end"]] == item["text"], "Raw item text differs")
        for span in a["risk_spans"]:
            left, right = span["start"], span["end"]
            require(item["start"] <= left < right <= item["end"], "Risk span outside item")
            require(g["response"][left:right] == span["text"] and any(c.isalnum() for c in span["text"]), "Invalid exact risk span")
        answer, tt = direct_gold(row, g, a, safe); answers.append(answer); tokens.extend(tt)
    require(used == set(labels), "Extra labels")
    for gid, rows in group_rows.items():
        require(len({r["split"] for r in rows}) == 1, "Group crosses train/validation/test: "+gid)
        require(sorted(r["condition"] for r in rows) == ["complete", "partial"], "Unpaired condition group")
    reuse = read_json(root/"data/reuse_manifest.json")
    require(reuse["excludes_round9_test"], "Old test reuse disallowed")
    require(set(reuse["rows"]) == {r["row_id"] for r in inputs if r["split"] != "test"}, "Development reuse split differs")
    report = {"schema": "round10-independent-gold-count-audit-v1", "status": "passed", "labels_reconstructed_from_raw": True,
        "group_overlap": False, "splits": {}, "source_hashes": source_hashes}
    for split in SPLITS:
        ii = [i for i in answers if i["split"] == split]; tt = [t for t in tokens if t["split"] == split]
        report["splits"][split] = {"coverage": coverage(ii, tt),
            "token_nonspan": sum(t["gold"] == 0 for t in tt),
            "risk_only_tokens": sum(t["risk_item_eligible"] for t in tt),
            "safe_refusal_ids": [i["item_id"] for i in ii if i["reviewed_safe_refusal"]]}
    return answers, tokens, report


def count(rows, name, threshold):
    table = [[0, 0], [0, 0]]  # [gold][prediction]
    missing = 0
    for row in rows:
        value = row["scores"].get(name)
        usable = finite(value) and finite(threshold)
        missing += int(not usable)
        table[int(row["gold"])][int(usable and value >= threshold)] += 1
    tn, fp = table[0]; fn, tp = table[1]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "f1": 2*tp/(2*tp+fp+fn) if tp+fn else None,
        "missing_predictions": missing}


def compare_values(actual, expected, context):
    for key, value in actual.items():
        require(key in expected, context+" missing key "+key)
        other = expected[key]
        if value is None:
            require(other is None, context+" differs "+key)
        elif isinstance(value, float):
            require(finite(other) and abs(value-other) <= 1e-11, context+" differs "+key)
        else:
            require(value == other, context+" differs "+key)


def attach_scores(raw, saved, key, methods):
    index = {r[key]: r for r in saved}
    require(len(index) == len(saved) == len(raw), "Prediction coverage/duplicates")
    require(set(index) == {r[key] for r in raw}, "Prediction identities differ")
    for r in raw:
        s = index[r[key]]
        for name, value in r.items():
            require(s.get(name) == value, "Predicted record differs from independently reconstructed raw gold: "+r[key]+" "+name)
        require(set(s["scores"]) == set(methods), "Missing method scores")
        r["scores"] = s["scores"]


def direct_intervals(items, tokens, thresholds, frozen_intervals):
    """Explicit group-index resampling/summing, separate from evaluator's einsum."""
    import numpy as np
    groups = sorted({i["group_id"] for i in items})
    require(frozen_intervals["groups"] == len(groups), "Bootstrap group count differs")
    require(frozen_intervals["draws"] == 2000 and frozen_intervals["seed"] == 20260912, "Unexpected CI protocol")
    draws = np.random.default_rng(20260912).integers(0, len(groups), (2000, len(groups)))
    subsets = {"answer_items": ([i for i in items if i["main_eligible"]], ITEM_METHODS),
        "answer_asserted_only": ([i for i in items if i["asserted_eligible"]], ITEM_METHODS),
        "all_resolved_items": ([t for t in tokens if t["main_eligible"]], TOKEN_METHODS),
        "risk_items_only": ([t for t in tokens if t["risk_item_eligible"]], TOKEN_METHODS)}
    result = {}
    for subset, (rows, methods) in subsets.items():
        cell = np.empty((len(groups), len(methods), 3), dtype=np.int64)
        for gi, gid in enumerate(groups):
            local = [r for r in rows if r["group_id"] == gid]
            for j, name in enumerate(methods):
                c = count(local, name, thresholds[name]["threshold"])
                cell[gi, j] = [c["tp"], c["fp"], c["fn"]]
        sampled = np.asarray([cell[index].sum(axis=0) for index in draws])
        values = {}
        for j, name in enumerate(methods):
            values[name] = {k: [] for k in ("precision", "recall", "f1")}
            for tp, fp, fn in sampled[:, j]:
                values[name]["precision"].append(float(tp/(tp+fp)) if tp+fp else math.nan)
                values[name]["recall"].append(float(tp/(tp+fn)) if tp+fn else math.nan)
                values[name]["f1"].append(float(2*tp/(2*tp+fp+fn)) if tp+fn else math.nan)
        def interval(sequence):
            a = np.asarray(sequence, dtype=float); good = a[np.isfinite(a)]
            return {"ci95": np.quantile(good, [.025, .975]).tolist() if len(good) else None, "defined_draws": len(good)}
        def check_interval(actual, expected, ctx):
            require(actual["defined_draws"] == expected["defined_draws"], ctx+" valid draw count")
            if actual["ci95"] is None:
                require(expected["ci95"] is None, ctx+" undefined CI")
            else:
                require(np.allclose(actual["ci95"], expected["ci95"], rtol=0, atol=1e-11), ctx+" CI differs")
        reference = frozen_intervals["subsets"][subset]
        result[subset] = {"methods": {}, "contrasts": {}}
        for name, seqs in values.items():
            result[subset]["methods"][name] = {k: interval(v) for k, v in seqs.items()}
            for k, entry in result[subset]["methods"][name].items():
                check_interval(entry, reference["methods"][name][k], subset+" "+name+" "+k)
        suf = "__item" if subset.startswith("answer_") else "__token"
        contrasts = {"primary_full_minus_lb": ("lb_full"+suf, "lb"+suf),
            "whitebox_increment_over_surface": ("lb_full"+suf, "lb_surface"+suf),
            "whitebox_only_minus_lb": ("lb_whitebox"+suf, "lb"+suf),
            "surface_only_increment": ("lb_surface"+suf, "lb"+suf),
            "old_binding_minus_lb": ("lb_old_binding"+suf, "lb"+suf)}
        if suf == "__token":
            contrasts.update({f+"_native_minus_item_broadcast": (f+"__token", f+"__broadcast") for f in FEATURES})
        require(set(contrasts) == set(reference["contrasts"]), "CI contrast inventory differs")
        for name, (a, b) in contrasts.items():
            entries = {k: interval(np.asarray(values[a][k])-np.asarray(values[b][k])) for k in ("precision", "recall", "f1")}
            for k, entry in entries.items():
                check_interval(entry, reference["contrasts"][name][k], subset+" "+name+" "+k)
            result[subset]["contrasts"][name] = entries
    return result


def predictions(root):
    out = root/"results"; done = read_json(out/"test_complete10.json")
    frozen = read_json(out/"freeze10.json")
    require(sha(out/"freeze10.json") == done["freeze10_sha256"], "Model/threshold freeze changed")
    for name, key in (("metrics_test.json", "metrics_sha256"), ("answer_scores_test.jsonl", "answer_scores_sha256"),
                      ("token_scores_test.jsonl", "token_scores_sha256")):
        require(sha(out/name) == done[key], "Completed test artifact changed")
    for name, key in (("frozen_models.pkl", "model_sha256"), ("selection.json", "selection_sha256"),
                      ("training_weights.json", "training_weights_sha256")):
        require(sha(out/name) == frozen[key], "Frozen fitting artifact changed")
    for name, checksum in frozen["source_hashes"]["files"].items():
        require(sha(root/name) == checksum, "Source differs from fit freeze: "+name)
    aa, tt, gold_report = collect_gold(root)
    limits = frozen["thresholds"]
    require(set(limits) == set(ITEM_METHODS+TOKEN_METHODS), "Expected 10 heads and 5 broadcasts")
    for f in FEATURES:
        require(limits[f+"__item"]["threshold"] == limits[f+"__broadcast"]["threshold"], "Broadcast was recalibrated")
    for split, key in (("train", "train_coverage"), ("validation", "validation_coverage")):
        require(gold_report["splits"][split]["coverage"] == frozen[key], "Development denominator changed")
    items = [i for i in aa if i["split"] == "test"]; tokens = [t for t in tt if t["split"] == "test"]
    attach_scores(items, read_lines(out/"answer_scores_test.jsonl"), "item_id", ITEM_METHODS)
    attach_scores(tokens, read_lines(out/"token_scores_test.jsonl"), "token_key", TOKEN_METHODS)
    result = read_json(out/"metrics_test.json")
    require(result["coverage"] == gold_report["splits"]["test"]["coverage"], "Test denominator differs")
    require(set(result["answer_methods"]) == set(ITEM_METHODS), "Question method inventory differs")
    require(set(result["localization_methods"]) == set(TOKEN_METHODS), "Localization inventory differs")
    report = {"schema": "round10-independent-prediction-audit-v1", "status": "passed",
        "gold_audit": gold_report, "question_heads": {}, "localization": {}, "missing_predictions": {},
        "independence": "Raw annotation and explicit refusal decisions + direct character loops over original BPE; no evaluator/model/feature imports"}
    for name in ITEM_METHODS:
        threshold = limits[name]["threshold"]; ref = result["answer_methods"][name]
        table = {}
        for subset, pred in (("micro", lambda i: i["main_eligible"]), ("asserted_only", lambda i: i["asserted_eligible"]),
                             ("reviewed_safe_refusals", lambda i: i["reviewed_safe_refusal"])):
            rr = [i for i in items if pred(i)]; values = count(rr, name, threshold)
            compare_values(values, ref[subset], name+" "+subset); require(ref[subset]["answers"] == len(rr), "Answer count differs")
            table[subset] = values
        for axis in ("condition", "category"):
            for value in sorted({i[axis] for i in items}):
                rr = [i for i in items if i["main_eligible"] and i[axis] == value]
                compare_values(count(rr, name, threshold), ref["strata"][axis][value], name+" "+axis+" "+value)
        report["question_heads"][name] = table
        report["missing_predictions"][name] = table["micro"]["missing_predictions"]
    for name in TOKEN_METHODS:
        threshold = limits[name]["threshold"]; ref = result["localization_methods"][name]; table = {}
        for subset, flag in (("all_resolved_items", "main_eligible"), ("risk_items_only", "risk_item_eligible")):
            rr = [t for t in tokens if t[flag]]; values = count(rr, name, threshold)
            compare_values(values, ref[subset]["micro"], name+" "+subset)
            require(ref[subset]["micro"]["tokens"] == len(rr), "Token denominator differs")
            for rid in {t["row_id"] for t in rr}:
                compare_values(count([t for t in rr if t["row_id"] == rid], name, threshold),
                               ref[subset]["per_answer"][rid], name+" per answer "+rid)
            table[subset] = values
            for axis in ("condition", "attribute_group"):
                for value in sorted({t[axis] for t in tokens}):
                    compare_values(count([t for t in rr if t[axis] == value], name, threshold),
                                   ref["strata"][axis][value][subset], name+" "+axis+" "+value)
        for label, predicate in (("all_lexical_text", lambda t: t["lexical"]),
                                 ("refusal_text", lambda t: t["lexical"] and t["abstention_overlap"]),
                                 ("unresolved_text", lambda t: t["lexical"] and t["unresolved_overlap"]),
                                 ("outside_items", lambda t: t["lexical"] and t["outside_items"])):
            rr = [t for t in tokens if predicate(t)]
            count_alerts = sum(finite(t["scores"].get(name)) and t["scores"][name] >= threshold for t in rr)
            observed = {"tokens": len(rr), "scorable": sum(finite(t["scores"].get(name)) for t in rr), "alerts": count_alerts,
                        "alert_rate_all_tokens": count_alerts/len(rr) if rr else None}
            compare_values(observed, ref["all_text_description"][label], name+" "+label)
            table[label] = observed
        report["localization"][name] = table
        report["missing_predictions"][name] = table["all_resolved_items"]["missing_predictions"]
    # Every broadcast is exactly its own completed item's score, regardless of labels.
    by_id = {i["item_id"]: i for i in items}
    for t in tokens:
        for f in FEATURES:
            expected = by_id[t["item_ids"][0]]["scores"][f+"__item"] if t["item_ids"] else None
            require(t["scores"][f+"__broadcast"] == expected, "Broadcast is not the original item score")
    report["paired_ci_audit"] = direct_intervals(items, tokens, limits, result["group_bootstrap"])
    report["test_metrics_sha256"] = sha(out/"metrics_test.json")
    write_json(out/"INDEPENDENT_COUNT_AUDIT10.json", report)
    print(json.dumps({"status": "passed", "question_heads": 5, "localization_predictors": 10,
        "all_missing_predictions": report["missing_predictions"], "paired_ci": "passed"}, ensure_ascii=False))
    return report


def self_test():
    row = {"row_id": "r", "question_id": "q", "group_id": "q", "split": "test", "condition": "partial", "category": "time"}
    text = "It was 2020."
    item = {"item_id": "r__1", "text": text, "start": 0, "end": len(text), "parse_ok": True}
    g = {"response": text, "items": [item], "response_token_ids": list(range(5)),
         "response_token_offsets": [[0, 2], [2, 6], [6, 9], [9, 11], [11, 12]]}
    a = {"original_stance": "asserted", "original_risk": 1, "localization_status": "resolved",
         "risk_spans": [{"start": 7, "end": 11, "text": "2020"}]}
    answer, tokens = direct_gold(row, g, a, set())
    require([t["gold"] for t in tokens if t["main_eligible"]] == [0, 0, 1, 1], "Split number or function-word failure")
    for t, value in zip(tokens, [.1, .9, .9, None, .9]):
        t["scores"] = {"x": value}
    require(count([t for t in tokens if t["main_eligible"]], "x", .5) ==
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": .5, "recall": .5, "f1": .5, "missing_predictions": 1}, "Count failure")
    refusal = {"original_stance": "abstained", "original_risk": None, "localization_status": "excluded", "risk_spans": []}
    safe_answer, safe_tokens = direct_gold(row, g, refusal, {"r__1"})
    require(safe_answer["gold"] == 0 and safe_answer["main_eligible"] and not any(t["main_eligible"] for t in safe_tokens), "Refusal granularity conflated")
    unreviewed, _ = direct_gold(row, g, refusal, set())
    require(unreviewed["gold"] is None and not unreviewed["main_eligible"], "Unreviewed refusal auto-negative")
    answer["scores"] = {"x": .9}; safe_answer["scores"] = {"x": .9}
    require(count([answer, safe_answer], "x", .5)["f1"] == 2/3, "Safe refusal FP omitted")
    # One BPE crossing unknown alphanumeric text outside the item is excluded.
    g2 = {"response": "XA", "items": [{**item, "start": 1, "end": 2, "text": "A"}],
          "response_token_ids": [1], "response_token_offsets": [[0, 2]]}
    a2 = {**a, "original_risk": 0, "risk_spans": []}
    _, crossing = direct_gold(row, g2, a2, set())
    require(not crossing[0]["main_eligible"], "Cross-boundary token became false normal")
    # Separate BPE IDs with identical Unicode offsets are still separate tokens.
    g3 = {"response": "年", "items": [{**item, "text": "年", "start": 0, "end": 1}],
          "response_token_ids": [1, 2], "response_token_offsets": [[0, 1], [0, 1]]}
    _, duplicate = direct_gold(row, g3, {**a, "risk_spans": [{"start": 0, "end": 1, "text": "年"}]}, set())
    require([t["gold"] for t in duplicate] == [1, 1], "Duplicate offset BPE deduplicated")
    # A one-group, two-condition sample must keep the perfect and missed half together.
    ii, tt = [], []
    for condition, score in (("complete", .9), ("partial", .1)):
        ii.append({"group_id": "one", "condition": condition, "gold": 1, "main_eligible": True,
                   "asserted_eligible": True, "scores": {n: score for n in ITEM_METHODS}})
        tt.append({"group_id": "one", "condition": condition, "gold": 1, "main_eligible": True,
                   "risk_item_eligible": True, "scores": {n: score for n in TOKEN_METHODS}})
    reference = {"groups": 1, "draws": 2000, "seed": 20260912, "subsets": {}}
    ordinary = {k: {"ci95": [v, v], "defined_draws": 2000} for k, v in (("precision", 1.), ("recall", .5), ("f1", 2/3))}
    zero = {k: {"ci95": [0., 0.], "defined_draws": 2000} for k in ordinary}
    contrast_names = ["primary_full_minus_lb", "whitebox_increment_over_surface", "whitebox_only_minus_lb", "surface_only_increment", "old_binding_minus_lb"]
    for subset in ("answer_items", "answer_asserted_only", "all_resolved_items", "risk_items_only"):
        is_item = subset.startswith("answer_"); methods = ITEM_METHODS if is_item else TOKEN_METHODS
        contrasts = contrast_names+([] if is_item else [f+"_native_minus_item_broadcast" for f in FEATURES])
        reference["subsets"][subset] = {"methods": {m: ordinary for m in methods}, "contrasts": {c: zero for c in contrasts}}
    direct_intervals(ii, tt, {n: {"threshold": .5} for n in ITEM_METHODS+TOKEN_METHODS}, reference)
    print("SELF_TEST_PASSED: numeric BPE, normal words, missing FN, safe-refusal FP, independent denominators, boundary, duplicate Unicode offsets, paired group CIs")


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("self-test", "gold", "predictions"))
    p.add_argument("--root", type=Path, default=ROOT); args = p.parse_args(); root = args.root.resolve()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "gold":
        _, _, report = collect_gold(root); write_json(root/"results/INDEPENDENT_GOLD_AUDIT10.json", report)
        print(json.dumps(report["splits"], ensure_ascii=False))
    else:
        predictions(root)


if __name__ == "__main__":
    main()
