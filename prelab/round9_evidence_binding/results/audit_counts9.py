"""Independent, read-only Round9 gold and fixed-prediction count audit.

Never imports an evaluator, model, feature extractor, or annotation helper.
`gold` requires the final annotation freeze; `predictions` requires completed test.
`self-test` uses synthetic strings only. Writes audit artifacts under results/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
METHODS = ("lb_coarse", "lb_fine", "binding_coarse", "binding_fine", "hidden_lr",
           "hidden_mlp_seed_20260910", "hidden_mlp_seed_20260911", "hidden_mlp_seed_20260912",
           "nll", "entropy", "redeep", "lumina", "all_positive", "all_negative",
           "lb_coarse_broadcast", "binding_coarse_broadcast")


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8-sig").splitlines() if line.strip()]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def direct_tokens(generation, item, annotation, row):
    """Single predeclared item: test characters directly, without evaluator sets.

    A lexical token needs a Unicode letter/number. Every such character must be
    inside the resolved item; any intersection with a risk span makes it positive.
    Overlapping BPE offsets (e.g. split Unicode bytes) remain separate tokens.
    """
    response = generation["response"]
    offsets = generation["response_token_offsets"]
    ids = generation["response_token_ids"]
    require(len(offsets) == len(ids), "Token ids/offsets length mismatch")
    resolved = (annotation["original_stance"] == "asserted"
                and annotation["original_risk"] in (0, 1)
                and annotation["localization_status"] == "resolved")
    spans = annotation["risk_spans"]
    start, end = item["start"], item["end"]
    result = []
    for index, (token_id, interval) in enumerate(zip(ids, offsets)):
        left, right = interval
        require(type(left) is int and type(right) is int
                and 0 <= left <= right <= len(response), "Invalid token offset")
        letters = [i for i in range(left, right) if response[i].isalnum()]
        membership = [start is not None and start <= i < end for i in letters]
        owner = bool(letters and any(membership))
        main = bool(resolved and letters and all(membership))
        positive = any(s["start"] <= i < s["end"] for i in letters for s in spans)
        result.append({
            "token_key": f"{row['row_id']}__token{index}", "token_index": index,
            "token_id": token_id, "row_id": row["row_id"], "question_id": row["question_id"],
            "group_id": row["group_id"], "split": row["split"], "condition": row["condition"],
            "attribute_group": row["category"], "start": left, "end": right,
            "text": response[left:right], "lexical": bool(letters),
            "main_eligible": main,
            "risk_item_eligible": main and annotation["original_risk"] == 1,
            "gold": int(positive) if main else None,
            "item_ids": [item["item_id"]] if owner else [],
            "outside_items": bool(letters) and not owner,
            "abstention_overlap": owner and annotation["original_stance"] == "abstained",
            "unresolved_overlap": owner and not resolved and annotation["original_stance"] != "abstained",
            "cross_item": False,
        })
    return result


def summarize(tokens):
    selected = [t for t in tokens if t["main_eligible"]]
    risky = [t for t in tokens if t["risk_item_eligible"]]
    return {
        "all_tokens": len(tokens), "lexical_tokens": sum(t["lexical"] for t in tokens),
        "eligible_tokens": len(selected), "risk_tokens": sum(t["gold"] for t in selected),
        "nonspan_tokens": sum(t["gold"] == 0 for t in selected),
        "risk_items_only_tokens": len(risky),
        "risk_items_only_positive": sum(t["gold"] for t in risky),
        "risk_items_only_nonspan": sum(t["gold"] == 0 for t in risky),
        "abstention_lexical_tokens": sum(t["lexical"] and t["abstention_overlap"] for t in tokens),
        "other_excluded_lexical_tokens": sum(t["lexical"] and t["unresolved_overlap"] for t in tokens),
        "outside_item_lexical_tokens": sum(t["outside_items"] for t in tokens),
        "boundary_crossing_lexical_tokens": sum(t["lexical"] and bool(t["item_ids"])
            and not t["outside_items"] and not t["main_eligible"]
            and not t["abstention_overlap"] and not t["unresolved_overlap"] for t in tokens),
    }


def collect_gold(root):
    lock_path = root / "data/annotation_freeze.json"
    lock = read_json(lock_path)
    require(lock.get("status") == "frozen", "Wait for the final annotation freeze")
    require(set(lock["canonical_spans_sha256"]) == set(SPLITS), "Missing canonical splits")
    inputs = read_lines(root / "data/inputs.jsonl")
    generation_list = read_lines(root / "data/generated.jsonl")
    generations = {g["row_id"]: g for g in generation_list}
    require(len(generations) == len(generation_list) == len(inputs), "Generation coverage/duplicates")
    require(len({r["row_id"] for r in inputs}) == len(inputs), "Duplicate input rows")
    require(set(generations) == {r["row_id"] for r in inputs}, "Generation row ids differ")
    input_freeze = read_json(root / "data/freeze.json")
    require(input_freeze["status"] == "frozen", "Input freeze not final")
    require(digest(root / "data/inputs.jsonl") == input_freeze["files_sha256"]["data/inputs.jsonl"],
            "Inputs changed after freeze")
    require(digest(root / "protocol.json") == input_freeze["protocol_sha256"], "Protocol changed")
    require(len(inputs) == input_freeze["expected_rows"], "Unexpected row count")
    require(set(read_json(root / "protocol.json")["methods"]) == set(METHODS), "Method protocol differs")
    labels = {}
    files = [lock_path, root / "data/inputs.jsonl", root / "data/generated.jsonl",
             root / "data/freeze.json", root / "protocol.json", root / "ANNOTATION_GUIDE.md"]
    for split in SPLITS:
        path = root / "data" / f"annotations_{split}.jsonl"
        require(digest(path) == lock["canonical_spans_sha256"][split], "Canonical annotation changed: " + split)
        files.append(path)
        for a in read_lines(path):
            require(a["split"] == split, "Annotation in wrong split")
            require(a["item_id"] not in labels, "Duplicated label")
            labels[a["item_id"]] = a
    group_rows, question_groups = defaultdict(list), defaultdict(set)
    tokens, reports, prefix_exceptions = [], [], []
    used_items = set()
    generation_hashes = {}
    for row in inputs:
        rid = row["row_id"]
        require(row["split"] in SPLITS and row["expected_items"] == len(row["questions"]) == 1,
                "Unsupported input slot schema")
        group_rows[row["group_id"]].append(row)
        question_groups[row["question_id"]].add(row["group_id"])
        g = generations[rid]
        path = root / "data/generation_records" / (rid + ".json")
        require(read_json(path) == g, "Aggregate generation differs: " + rid)
        generation_hashes[rid] = digest(path)
        require(g["data_freeze_sha256"] == digest(root / "data/freeze.json"), "Generation input freeze differs")
        require(g["protocol_sha256"] == input_freeze["protocol_sha256"], "Generation protocol differs")
        require(len(g["items"]) == 1, "Expected one item per row")
        item = g["items"][0]
        iid = item["item_id"]
        require(iid not in used_items and iid in labels, "Item coverage mismatch: " + iid)
        used_items.add(iid)
        a = labels[iid]
        for key in ("row_id", "question_id", "split", "condition"):
            require(g[key] == row[key] == a[key], "Row metadata mismatch: " + key)
        require(a["group_id"] == row["group_id"], "Annotation group mismatch")
        for key in ("item_id", "text", "start", "end"):
            require(a[key] == item[key], "Annotation text/offset mismatch: " + iid + " " + key)
        require(a["source_generation_sha256"] == generation_hashes[rid], "Stale generation label: " + iid)
        require(not a.get("token_scores_viewed", False), "Non-blind label")
        require(a["original_stance"] in ("asserted", "abstained", "tentative", "missing"), "Bad stance")
        require(a["localization_status"] in ("resolved", "unresolved", "excluded"), "Bad status")
        require(a["original_risk"] in (0, 1, None), "Bad risk")
        if item["start"] is not None:
            require(g["response"][item["start"]:item["end"]] == item["text"], "Parsed item differs from response")
        if a["localization_status"] == "resolved":
            require(item["parse_ok"] and item["start"] is not None
                    and a["original_stance"] == "asserted" and a["original_risk"] in (0, 1), "Invalid resolved item")
            require(bool(a["risk_spans"]) == bool(a["original_risk"]), "Risk/span inconsistency")
        for scope in a.get("claim_scope", []):
            if scope["start"] is not None:
                require(g["response"][scope["start"]:scope["end"]] == scope["text"], "Claim scope mismatch")
        for s in a["risk_spans"]:
            require(type(s["start"]) is int and type(s["end"]) is int
                    and item["start"] <= s["start"] < s["end"] <= item["end"], "Span outside item")
            require(g["response"][s["start"]:s["end"]] == s["text"], "Span quote mismatch")
            require(any(c.isalnum() for c in s["text"]), "Non-lexical gold span")
        tt = direct_tokens(g, item, a, row)
        tokens.extend(tt)
        # Inspect actual frozen rule: an unrecognized bracketed ordinal can be lexical.
        marker = re.match(r"\s*(?:\[\d+\]|\(\d+\)|\d+[.)])\s+", g["response"])
        if marker:
            affected = [t["token_key"] for t in tt if t["main_eligible"]
                        and any(g["response"][j].isalnum() for j in range(t["start"], min(t["end"], marker.end())))]
            if affected:
                prefix_exceptions.append({"row_id": rid, "item_id": iid, "marker": marker.group(),
                                          "counted_token_keys": affected, "parse_reason": item.get("parse_reason")})
        uncovered = []
        for s in a["risk_spans"]:
            for j in range(s["start"], s["end"]):
                if g["response"][j].isalnum() and not any(t["main_eligible"] and t["start"] <= j < t["end"] for t in tt):
                    uncovered.append(j)
        reports.append({"row_id": rid, "item_id": iid, "group_id": row["group_id"],
                        "split": row["split"], "condition": row["condition"], "category": row["category"],
                        "stance": a["original_stance"], "status": a["localization_status"],
                        "risk": a["original_risk"], "parse_ok": item["parse_ok"],
                        "uncovered_risk_character_indices": sorted(set(uncovered)), **summarize(tt)})
    require(used_items == set(labels), "Extra annotations outside inputs")
    require(len(group_rows) == input_freeze["expected_groups"], "Group count differs")
    require(all(len(v) == 1 for v in question_groups.values()), "Same question split into distinct groups")
    for group, rr in group_rows.items():
        require(len({r["split"] for r in rr}) == 1, "Group crosses split: " + group)
        require(len(rr) == 2 and {r["condition"] for r in rr} == {"complete", "partial"}, "Incomplete condition pair")
        require(len({r["question_id"] for r in rr}) == 1, "Multiple questions share a group")
    split_reports = {}
    for split in SPLITS:
        rr = [r for r in reports if r["split"] == split]
        tt = [t for t in tokens if t["split"] == split]
        split_reports[split] = {"groups": len({r["group_id"] for r in rr}), "rows": len(rr), "items": len(rr),
            "statuses": dict(Counter(r["status"] for r in rr)), "stances": dict(Counter(r["stance"] for r in rr)),
            "item_risk_counts": dict(Counter(str(r["risk"]) for r in rr)),
            "failed_parse_items": sum(not r["parse_ok"] for r in rr), **summarize(tt)}
    result = {"status": "passed", "audit": "independent direct Unicode character/BPE intersection",
              "file_hashes": {str(f.relative_to(root)): digest(f) for f in files},
              "generation_records_sha256": generation_hashes,
              "groups_disjoint_across_splits": True, "paired_conditions_kept_together": True,
              "splits": split_reports, "items": reports, "numbering_count_exceptions": prefix_exceptions,
              "risk_characters_without_eligible_token": sum(len(r["uncovered_risk_character_indices"]) for r in reports),
              "scores_read": False, "source_code_modified": False,
              "negative_label_meaning": "Outside annotated risk span, not independently verified factual truth"}
    return result, tokens


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def counts(rows, scores, method, threshold):
    table = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    missing = 0
    for token in rows:
        value = scores[token["token_key"]][method]
        present = finite(value) and finite(threshold)
        missing += not present
        alert = present and value >= threshold
        cell = ("tp" if alert else "fn") if token["gold"] else ("fp" if alert else "tn")
        table[cell] += 1
    tp, fp, fn, tn = (table[k] for k in ("tp", "fp", "fn", "tn"))
    return {**table, "tokens": len(rows), "risk_tokens": tp + fn, "nonspan_tokens": fp + tn,
            "missing_predictions": missing, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None}


def compare_counts(actual, reported, where):
    for key, value in actual.items():
        expected = reported[key]
        same = value == expected if value is None or expected is None else math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
        require(same, f"Count disagreement {where}/{key}: independent={value}, reported={expected}")


def audit_predictions(root):
    completed = read_json(root / "results/test_complete9.json")
    freeze_path = root / "results/freeze9.json"
    require(digest(freeze_path) == completed["freeze9_sha256"], "Test freeze changed")
    frozen = read_json(freeze_path)
    gold_report, all_tokens = collect_gold(root)
    for rel in ("data/annotation_freeze.json", "data/annotations_train.jsonl",
                "data/annotations_validation.jsonl", "data/annotations_test.jsonl",
                "data/inputs.jsonl", "data/generated.jsonl", "protocol.json", "ANNOTATION_GUIDE.md"):
        require(digest(root / rel) == frozen["source_hashes"]["files"][rel], "Fit-frozen input changed: " + rel)
    require(digest(root / "results/frozen_models.pkl") == frozen["model_sha256"], "Frozen models changed")
    score_path = root / "results/token_scores_test.jsonl"
    metrics_path = root / "results/metrics_test.json"
    require(digest(score_path) == completed["token_scores_sha256"], "Saved predictions changed")
    require(digest(metrics_path) == completed["metrics_sha256"], "Saved metrics changed")
    require(set(frozen["method_names"]) == set(METHODS), "Predictor set differs")
    expected = {t["token_key"]: t for t in all_tokens if t["split"] == "test"}
    saved = read_lines(score_path)
    require(len(saved) == len(expected), "Saved token count differs")
    scores = {}
    for t in saved:
        key = t["token_key"]
        require(key in expected and key not in scores, "Missing/duplicate/extra saved token")
        for field, value in expected[key].items():
            require(t[field] == value, f"Saved alignment/gold mismatch {key}/{field}")
        require(set(t["scores"]) == set(METHODS), "Missing/extra method scores")
        scores[key] = t["scores"]
    require(set(scores) == set(expected), "Missing saved token")
    metrics = read_json(metrics_path)
    for split, official in (("train", frozen["train_coverage"]),
                            ("validation", frozen["validation_coverage"]), ("test", metrics["coverage"])):
        for key in ("groups", "rows", "items", "statuses", "stances", "all_tokens", "eligible_tokens", "risk_tokens", "failed_parse_items"):
            require(official[key] == gold_report["splits"][split][key], "Coverage mismatch: " + split + "/" + key)
    summary = {}
    for method in METHODS:
        threshold = frozen["thresholds"][method]["threshold"]
        require(metrics["methods"][method]["threshold"] == threshold, "Reported threshold differs")
        summary[method] = {}
        for subset, flag in (("all_resolved_items", "main_eligible"), ("risk_items_only", "risk_item_eligible")):
            chosen = [t for t in expected.values() if t[flag]]
            c = counts(chosen, scores, method, threshold)
            compare_counts(c, metrics["methods"][method][subset]["micro"], method + "/" + subset)
            summary[method][subset] = c
            by_row = defaultdict(list)
            for t in chosen:
                by_row[t["row_id"]].append(t)
            require(set(by_row) == set(metrics["methods"][method][subset]["per_answer"]), "Per-answer coverage differs")
            for rid, rr in by_row.items():
                compare_counts(counts(rr, scores, method, threshold), metrics["methods"][method][subset]["per_answer"][rid], method + "/" + rid)
            for axis in ("condition", "attribute_group"):
                for value in {t[axis] for t in expected.values()}:
                    cc = counts([t for t in chosen if t[axis] == value], scores, method, threshold)
                    compare_counts(cc, metrics["methods"][method]["strata"][axis][value][subset], method + "/" + axis + "/" + value)
    return {"status": "passed", "all_16_methods_recounted": True, "subsets": 2,
            "per_answer_and_category_condition_counts_checked": True, "methods": summary,
            "freeze9_sha256": digest(freeze_path), "token_scores_sha256": digest(score_path),
            "metrics_sha256": digest(metrics_path), "gold_audit": gold_report["splits"],
            "models_retrained": False, "thresholds_changed": False,
            "interpretation": "Two mean item broadcasts use completed answers; they are not causal token-local predictors."}


def self_test():
    response = "1. A 12 cats."
    g = {"response": response, "response_token_ids": [1, 2, 3, 4, 5, 6],
         "response_token_offsets": [[0, 1], [1, 3], [3, 4], [5, 6], [6, 7], [8, 13]]}
    item = {"item_id": "r__1", "start": 3, "end": 13}
    a = {"original_stance": "asserted", "original_risk": 1, "localization_status": "resolved",
         "risk_spans": [{"start": 5, "end": 7, "text": "12"}]}
    row = {"row_id": "r", "question_id": "q", "group_id": "q", "split": "train",
           "condition": "partial", "category": "quantity"}
    t = direct_tokens(g, item, a, row)
    require([x["gold"] for x in t] == [None, None, 0, 1, 1, 0], "Synthetic split-number/nonspan/numbering error")
    require(sum(x["main_eligible"] for x in t) == 4, "Normal words must remain in denominator")
    require(summarize(t)["nonspan_tokens"] == 2 and summarize(t)["outside_item_lexical_tokens"] == 1,
            "Independent denominator summary")
    a0 = {**a, "original_risk": 0, "risk_spans": []}
    require(sum(x["gold"] or 0 for x in direct_tokens(g, item, a0, row)) == 0, "Supported item label error")
    for stance, status in (("abstained", "excluded"), ("asserted", "unresolved")):
        ax = {**a, "original_risk": None, "original_stance": stance, "localization_status": status, "risk_spans": []}
        require(not any(x["main_eligible"] for x in direct_tokens(g, item, ax, row)), "Excluded status entered denominator")
    gx = {**g, "response_token_ids": [0], "response_token_offsets": [[0, 4]]}
    require(not direct_tokens(gx, item, a, row)[0]["main_eligible"], "Cross-item lexical character leaked in")
    item_all = {**item, "start": 0}
    require(direct_tokens(g, item_all, a, row)[0]["main_eligible"], "Whole-response numeric marker audit exception")
    unicode_g = {"response": "é 9", "response_token_ids": [7, 8, 9, 10],
                 "response_token_offsets": [[0, 1], [0, 1], [1, 2], [2, 3]]}
    unicode_item = {"item_id": "r__1", "start": 0, "end": 3}
    unicode_a = {**a, "risk_spans": [{"start": 2, "end": 3, "text": "9"}]}
    require([x["gold"] for x in direct_tokens(unicode_g, unicode_item, unicode_a, row)] == [0, 0, None, 1],
            "Unicode BPE duplicate offsets/whitespace must preserve exact token counts")
    chosen = [x for x in t if x["main_eligible"]]
    scores = {x["token_key"]: {"x": v} for x, v in zip(chosen, [.8, .9, None, .1])}
    c = counts(chosen, scores, "x", .5)
    require((c["tp"], c["fp"], c["fn"], c["tn"], c["missing_predictions"], c["f1"]) == (1, 1, 1, 1, 1, .5), "Independent confusion arithmetic")
    print("Independent synthetic count checks passed; no actual labels or scores read.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("self-test", "gold", "predictions"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = args.root.resolve()
    if args.stage == "gold":
        result, _ = collect_gold(root)
        target = root / "results/GOLD_COUNT_AUDIT.json"
    else:
        result = audit_predictions(root)
        target = root / "results/PREDICTION_COUNT_AUDIT.json"
    write_json(target, result)
    print("INDEPENDENT_AUDIT_PASSED", str(target))


if __name__ == "__main__":
    main()
