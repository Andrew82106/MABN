"""Read-only independent Round11 audit; reuse only the independent Round10 auditor.

No evaluator, model or feature extractor is imported. `run` requires completed
exploratory scoring. `self-test` uses synthetic records only. No fitting/GPU calls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT.parent/"round10_dual_granularity"
spec = importlib.util.spec_from_file_location("independent_audit10_helpers", R10/"results/audit10_counts.py")
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
FEATURES = ("nll", "lb", "lb_nll")
ITEM = tuple(f+"__item" for f in FEATURES)
TOKEN = tuple(f+"__token" for f in FEATURES)


def baseline_equal(current, previous, key, name, current_threshold, old_threshold):
    old = {r[key]: r for r in previous}
    a.require(len(old) == len(previous) == len(current), "Lookback baseline row count changed")
    a.require(set(old) == {r[key] for r in current}, "Lookback baseline identities changed")
    differences = []
    for r in current:
        before = old[r[key]]
        for field in ("gold", "main_eligible", "group_id", "split", "condition", "start", "end", "text"):
            a.require(r[field] == before[field], "Lookback denominator/identity changed: "+field)
        now_score, old_score = r["scores"][name], before["scores"][name]
        if now_score is None or old_score is None:
            a.require(now_score is old_score, "Lookback missing score changed")
            continue
        delta = abs(now_score-old_score); differences.append(delta)
        a.require(delta <= 1e-10, "Lookback baseline score not reproduced")
        a.require((now_score >= current_threshold) == (old_score >= old_threshold), "Lookback baseline decision changed")
    a.require(abs(current_threshold-old_threshold) <= 1e-10, "Lookback threshold changed")
    return {"rows": len(current), "max_absolute_score_difference": max(differences, default=0.),
            "threshold_equal_atol_1e_10": True, "all_binary_decisions_equal": True}


def check_cache(root, cache_folder=None):
    """Check bytes/axes and NLL provenance without recomputing probabilities.

    If Round11 materializes lightweight caches, retain token_nll and
    lookback_features names plus token_ids/start/end. Otherwise reuse R10 directly.
    Optional lb/nll/lb_nll matrices are checked against the original arrays too.
    """
    inputs = a.read_lines(R10/"data/inputs.jsonl")
    folder = Path(cache_folder) if cache_folder else root/"data/features"
    materialized = folder.exists() and any(folder.glob("*.npz"))
    records, tokens = 0, 0
    for row in inputs:
        rid = row["row_id"]
        gp = R10/"data/generation_records"/(rid+".json"); g = a.read_json(gp)
        source = R10/"data/features"/(rid+".npz"); side = a.read_json(source.with_suffix(".json"))
        a.require(side["source_generation_sha256"] == a.sha(gp), "NLL source generation changed")
        a.require(side["arrays_sha256"] == a.sha(source), "NLL source arrays changed")
        a.require("P+j-1" in side["probability_timing"], "NLL phase is not previous-state prediction")
        source_code = R10.parent/"round9_evidence_binding/src/binding9.py"
        a.require(side["source_code_sha256"] == a.sha(source_code), "NLL extractor source changed")
        with np.load(source, allow_pickle=False) as z:
            original = {k: z[k].copy() for k in ("token_nll", "lookback_features", "token_ids", "token_start", "token_end")}
        n = len(g["response_token_ids"]); offsets = np.asarray(g["response_token_offsets"])
        a.require(original["token_nll"].shape == (n,) and np.isfinite(original["token_nll"]).all(), "NLL length/nonfinite")
        a.require(original["lookback_features"].shape == (n, 784), "LB width changed")
        a.require(np.array_equal(original["token_ids"], g["response_token_ids"]), "Source token IDs changed")
        a.require(np.array_equal(original["token_start"], offsets[:, 0]) and np.array_equal(original["token_end"], offsets[:, 1]), "Source token offsets changed")
        if materialized:
            target = folder/(rid+".npz")
            a.require(target.exists(), "Missing Round11 cache")
            with np.load(target, allow_pickle=False) as z:
                for k, before in original.items():
                    a.require(k in z.files and np.array_equal(z[k], before), "New cache differs: "+rid+" "+k)
                expected = {"nll": original["token_nll"][:, None], "lb": original["lookback_features"],
                            "lb_nll": np.column_stack((original["lookback_features"], original["token_nll"]))}
                for k, before in expected.items():
                    if k in z.files:
                        a.require(np.array_equal(z[k], before), "Concatenation/order differs: "+k)
        records += 1; tokens += n
    return {"rows": records, "tokens": tokens, "axis_and_finite_checks": "passed",
            "materialized_round11_cache": materialized, "source_phase": "P+j-1 logits predict observed token j",
            "meaning": "Source code/hash and exact cached arrays checked; no new GPU probability recomputation"}


def paired_f1(rows, methods, limits, all_groups, draws=2000):
    """Resample original group IDs explicitly, keeping both conditions together."""
    groups = sorted(all_groups)
    sampled = np.random.default_rng(20260912).integers(0, len(groups), (draws, len(groups)))
    counts = np.empty((len(groups), len(methods), 3), dtype=np.int64)
    for i, group in enumerate(groups):
        group_rows = [r for r in rows if r["group_id"] == group]
        for j, method in enumerate(methods):
            c = a.count(group_rows, method, limits[method]["threshold"])
            counts[i, j] = [c["tp"], c["fp"], c["fn"]]
    totals = np.asarray([counts[ids].sum(axis=0) for ids in sampled])
    values = {}
    for j, method in enumerate(methods):
        tp, fp, fn = totals[:, j, 0], totals[:, j, 1], totals[:, j, 2]
        values[method] = np.divide(2*tp, 2*tp+fp+fn, out=np.full(draws, np.nan), where=tp+fn > 0)
    def interval(v):
        good = v[np.isfinite(v)]
        return {"ci95": np.quantile(good, [.025, .975]).tolist() if len(good) else None, "defined_draws": len(good)}
    suffix = "__item" if methods[0].endswith("__item") else "__token"
    return {"groups": len(groups), "draws": draws, "seed": 20260912,
            "methods": {n: interval(v) for n, v in values.items()},
            "lb_nll_minus_lb": interval(values["lb_nll"+suffix]-values["lb"+suffix])}


def check_interval(actual, expected, label):
    a.require(actual["defined_draws"] == expected["defined_draws"], label+" valid draws differ")
    if actual["ci95"] is None:
        a.require(expected["ci95"] is None, label+" undefined CI differs")
    else:
        a.require(np.allclose(actual["ci95"], expected["ci95"], atol=1e-11, rtol=0), label+" CI differs")


def decision_changes(rows, kind, limits):
    """Compare the actual frozen operating points, including excluded records."""
    old, joint = "lb__"+kind, "lb_nll__"+kind
    changed = []
    for r in rows:
        x, y = r["scores"][old], r["scores"][joint]
        if not a.finite(x) or not a.finite(y):
            continue
        before, after = int(x >= limits[old]["threshold"]), int(y >= limits[joint]["threshold"])
        if before == after:
            continue
        known = r["main_eligible"]
        def cell(pred):
            if not known: return "excluded"
            return ("TP" if pred else "FN") if r["gold"] else ("FP" if pred else "TN")
        changed.append({"id": r["item_id" if kind == "item" else "token_key"], "row_id": r["row_id"],
            "text": r["text"], "start": r["start"], "end": r["end"], "main_eligible": known,
            "gold": r["gold"], "lb_score": x, "lb_nll_score": y,
            "lb_prediction": before, "lb_nll_prediction": after, "transition": cell(before)+"->"+cell(after)})
    return {"main_changes": [r for r in changed if r["main_eligible"]],
            "excluded_changes": [r for r in changed if not r["main_eligible"]]}


def run(root, cache_folder=None):
    out = root/"results"; done = a.read_json(out/"test_complete11.json")
    freeze = a.read_json(out/"freeze11.json"); metrics = a.read_json(out/"metrics_test.json")
    a.require(a.sha(out/"freeze11.json") == done["freeze11_sha256"], "Round11 model/threshold freeze changed")
    a.require(not freeze["test_labels_used_in_fit"], "Test labels used in fit")
    a.require(a.sha(out/"source_snapshot11.json") == freeze["source_snapshot_sha256"], "Source snapshot changed")
    snapshot = a.read_json(out/"source_snapshot11.json")
    for name, checksum in snapshot["local_files_sha256"].items():
        a.require(a.sha(root/name) == checksum, "Round11 frozen source changed: "+name)
    for name, checksum in snapshot["round10_frozen_graph"]["files"].items():
        a.require(a.sha(R10/name) == checksum, "R10 source cache/label changed: "+name)
    for name, key in (("frozen_models.pkl", "model_sha256"), ("selection.json", "selection_sha256"),
                      ("training_weights.json", "training_weights_sha256")):
        a.require(a.sha(out/name) == freeze[key], "Frozen model/selection changed")
    for name, key in (("metrics_test.json", "metrics_sha256"), ("answer_scores_test.jsonl", "answer_scores_sha256"),
                      ("token_scores_test.jsonl", "token_scores_sha256")):
        a.require(a.sha(out/name) == done[key], "Completed exploratory result changed")
    aa, tt, gold_report = a.collect_gold(R10)
    items = [r for r in aa if r["split"] == "test"]; tokens = [r for r in tt if r["split"] == "test"]
    a.attach_scores(items, a.read_lines(out/"answer_scores_test.jsonl"), "item_id", ITEM)
    a.attach_scores(tokens, a.read_lines(out/"token_scores_test.jsonl"), "token_key", TOKEN)
    limits = freeze["thresholds"]
    a.require(set(limits) == set(ITEM+TOKEN), "Expected exactly six frozen heads")
    a.require(metrics["coverage"] == gold_report["splits"]["test"]["coverage"], "R10/R11 denominator differs")
    results = {"schema": "round11-independent-audit-v1", "status": "passed", "exploratory_reuse_of_seen_test": True,
        "gold": gold_report["splits"], "answer_counts": {}, "token_counts": {}, "baseline_replication": {}, "paired_f1": {}}
    oldfreeze = a.read_json(R10/"results/freeze10.json")
    olddone = a.read_json(R10/"results/test_complete10.json")
    a.require(olddone["freeze10_sha256"] == a.sha(R10/"results/freeze10.json"), "Original baseline freeze changed")
    a.require(snapshot["round10_frozen_graph"] == oldfreeze["source_hashes"], "Different R10 frozen graph reused")
    for split, key in (("train", "train_coverage"), ("validation", "validation_coverage")):
        a.require(gold_report["splits"][split]["coverage"] == freeze[key] == oldfreeze[key], "Development denominator changed")
    for rows, kind, methods, section in ((items, "item", ITEM, "answer_methods"), (tokens, "token", TOKEN, "localization_methods")):
        oldpath = R10/"results"/("answer_scores_test.jsonl" if kind == "item" else "token_scores_test.jsonl")
        a.require(a.sha(oldpath) == olddone["answer_scores_sha256" if kind == "item" else "token_scores_sha256"], "Original baseline predictions changed")
        results["baseline_replication"][kind] = baseline_equal(rows, a.read_lines(oldpath), "item_id" if kind == "item" else "token_key",
            "lb__"+kind, limits["lb__"+kind]["threshold"], oldfreeze["thresholds"]["lb__"+kind]["threshold"])
        for name in methods:
            selected = [r for r in rows if r["main_eligible"]]
            count = a.count(selected, name, limits[name]["threshold"])
            reference = metrics[section][name]["micro"] if kind == "item" else metrics[section][name]["all_resolved_items"]["micro"]
            a.compare_values(count, reference, name+" primary")
            dest = results["answer_counts"] if kind == "item" else results["token_counts"]
            dest[name] = {"primary": count}
            secondary = [r for r in rows if r["asserted_eligible"]] if kind == "item" else [r for r in rows if r["risk_item_eligible"]]
            key = "asserted_only" if kind == "item" else "risk_items_only"
            secondary_count = a.count(secondary, name, limits[name]["threshold"])
            ref = metrics[section][name][key] if kind == "item" else metrics[section][name][key]["micro"]
            a.compare_values(secondary_count, ref, name+" secondary"); dest[name][key] = secondary_count
            if kind == "item":
                safe = [r for r in rows if r["reviewed_safe_refusal"]]
                safe_count = a.count(safe, name, limits[name]["threshold"])
                a.compare_values(safe_count, metrics[section][name]["reviewed_safe_refusals"], name+" safe refusals")
                dest[name]["reviewed_safe_refusals"] = safe_count
        results["paired_f1"][kind] = paired_f1([r for r in rows if r["main_eligible"]], methods, limits, {i["group_id"] for i in items})
    reported = metrics["group_bootstrap"]
    a.require(reported["groups"] == 60 and reported["draws"] == 2000 and reported["seed"] == 20260912, "Pairing CI unit differs")
    for kind, rebuilt in results["paired_f1"].items():
        subset = reported["subsets"]["answer_items" if kind == "item" else "all_resolved_items"]
        for name, interval in rebuilt["methods"].items():
            check_interval(interval, subset["methods"][name]["f1"], name)
        check_interval(rebuilt["lb_nll_minus_lb"], subset["contrasts"]["lb_nll_minus_lb"]["f1"], kind+" joint-minus-LB")
    # Refit control should also reproduce the exact development validation scores.
    results["baseline_validation_replication"] = {}
    for raw, kind, key, file_prefix in ((aa, "item", "item_id", "answer"), (tt, "token", "token_key", "token")):
        rows = [r for r in raw if r["split"] == "validation"]
        a.attach_scores(rows, a.read_lines(out/(file_prefix+"_scores_validation.jsonl")), key, ITEM if kind == "item" else TOKEN)
        results["baseline_validation_replication"][kind] = baseline_equal(rows,
            a.read_lines(R10/"results"/(file_prefix+"_scores_validation.jsonl")), key, "lb__"+kind,
            limits["lb__"+kind]["threshold"], oldfreeze["thresholds"]["lb__"+kind]["threshold"])
    results["cache_provenance"] = check_cache(root, cache_folder)
    results["actual_operating_point_changes"] = {"item": decision_changes(items, "item", limits),
                                                   "token": decision_changes(tokens, "token", limits)}
    results["unresolved_descriptive_only"] = []
    original_annotations = {r["item_id"]: r for r in a.read_lines(R10/"data/annotations_test.jsonl")}
    for item in items:
        if item["localization_status"] != "unresolved":
            continue
        lexical = [t for t in tokens if t["row_id"] == item["row_id"] and t["lexical"] and t["unresolved_overlap"]]
        results["unresolved_descriptive_only"].append({"item_id": item["item_id"], "text": item["text"],
            "rationale": original_annotations[item["item_id"]]["rationale"], "excluded_from_primary_metrics": True,
            "item_alerts": {name: int(item["scores"][name] >= limits[name]["threshold"]) for name in ITEM},
            "lexical_tokens": len(lexical),
            "token_alerts": {name: sum(t["scores"][name] >= limits[name]["threshold"] for t in lexical) for name in TOKEN}})
    results["metrics_sha256"] = a.sha(out/"metrics_test.json")
    a.write_json(out/"INDEPENDENT_AUDIT11.json", results)
    print(json.dumps({"status": "passed", "heads": 6, "baseline": results["baseline_replication"], "paired_f1": results["paired_f1"]}, ensure_ascii=False))


def self_test():
    rows = []
    for condition, score in (("complete", .9), ("partial", .1)):
        rows.append({"item_id": condition, "group_id": "g", "condition": condition, "split": "test", "start": 0, "end": 1,
                     "text": "x", "gold": 1, "main_eligible": True, "scores": {n: score for n in ITEM}})
    baseline_equal(rows, rows, "item_id", "lb__item", .5, .5)
    changed = json.loads(json.dumps(rows)); changed[0]["scores"]["lb__item"] += 1e-6
    try:
        baseline_equal(changed, rows, "item_id", "lb__item", .5, .5)
    except AssertionError:
        pass
    else:
        raise AssertionError("Baseline drift went unnoticed")
    boot = paired_f1(rows, ITEM, {n: {"threshold": .5} for n in ITEM}, {"g"}, draws=31)
    np.testing.assert_allclose(boot["methods"]["lb__item"]["ci95"], [2/3, 2/3])
    np.testing.assert_allclose(boot["lb_nll_minus_lb"]["ci95"], [0., 0.])
    print("SELF_TEST_PASSED: independent counts, exact baseline drift detection, paired-condition bootstrap")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("self-test", "run"))
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--cache-folder", type=Path)
    args = p.parse_args()
    if args.stage == "self-test": self_test()
    else: run(args.root.resolve(), args.cache_folder)
