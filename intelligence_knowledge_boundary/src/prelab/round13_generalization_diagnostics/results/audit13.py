"""Independent, read-only Round13 group/label/count audit. No extra fitting.

Only original training labels are parsed. Original validation/test labels are
neither loaded nor used. Reuses independent audit helpers, never the runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
R12 = ROOT.parent/"round12_conditional_fusion"
spec = importlib.util.spec_from_file_location("independent_audit12", R12/"results/audit12.py")
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
a = previous.a; R10 = previous.R10
METHODS = ("lb_c01", "lb_c001", "lb_c0001", "pca16", "pca64", "layer_mean28", "fit24", "fit48")
CS = dict(zip(METHODS, (.1, .01, .001, .1, .1, .1, .1, .1)))


def training_gold():
    lock = a.read_json(R10/"data/annotation_freeze.json")
    ap = R10/"data/annotations_train.jsonl"
    a.require(a.sha(ap) == lock["canonical_spans_sha256"]["train"], "Training labels changed")
    pp = R10/"data/question_label_policy.json"
    a.require(a.sha(pp) == lock["question_label_policy_sha256"], "Question label policy changed")
    policy = a.read_json(pp); rp = R10/"data/safe_refusals_train.json"
    a.require(a.sha(rp) == policy["reviewed_safe_refusal_files_sha256"]["train"], "Training refusal review changed")
    safe = set(a.read_json(rp)["safe_refusal_item_ids"])
    labels = {r["item_id"]: r for r in a.read_lines(ap)}
    rows = [r for r in a.read_lines(R10/"data/inputs.jsonl") if r["split"] == "train"]
    groups = defaultdict(list); items = []; tokens = []; used = set()
    for r in rows:
        gp = R10/"data/generation_records"/(r["row_id"]+".json"); g = a.read_json(gp)
        a.require(g["split"] == "train" and len(g["items"]) == 1, "Wrong training generation")
        item = g["items"][0]; label = labels[item["item_id"]]; used.add(item["item_id"])
        for name in ("item_id", "text", "start", "end"):
            a.require(label[name] == item[name], "Exact training annotation mismatch")
        a.require(label["source_generation_sha256"] == a.sha(gp), "Stale training annotation")
        a.require(label["split"] == "train" and not label.get("token_scores_viewed", False), "Label protocol mismatch")
        a.require(g["response"][item["start"]:item["end"]] == item["text"], "Original answer span differs")
        for s in label["risk_spans"]:
            a.require(item["start"] <= s["start"] < s["end"] <= item["end"], "Risk span outside original item")
            a.require(g["response"][s["start"]:s["end"]] == s["text"], "Risk quote differs")
        ii, tt = a.direct_gold(r, g, label, safe)
        items.append(ii); tokens.extend(tt); groups[r["group_id"]].append(r["condition"])
        fp = R10/"data/features"/(r["row_id"]+".npz"); meta = a.read_json(fp.with_suffix(".json"))
        a.require(meta["source_generation_sha256"] == a.sha(gp) and meta["arrays_sha256"] == a.sha(fp), "Training feature bytes changed")
        a.require(meta["lookback_axes"] == "token x layer-major/head-minor", "Wrong layer/head averaging axis")
    a.require(used == set(labels) and len(rows) == 240 and len(groups) == 120, "Training coverage differs")
    a.require(all(sorted(v) == ["complete", "partial"] for v in groups.values()), "Unpaired training conditions")
    usable = [r for r in tokens if r["main_eligible"]]
    a.require(len(usable) == 3854 and sum(r["gold"] for r in usable) == 339, "Original token denominator changed")
    return items, usable, {"coverage": a.coverage(items, tokens), "labels_parsed": ["train"], "layer_axis_checked_rows": len(rows)}


def expected_splits(items):
    groups = sorted({r["group_id"] for r in items})
    outer = a.read_json(R12/"results/fold_assignment.json")
    a.require(set(outer) == set(groups), "Round12 outer groups changed")
    result = {}
    for fold in range(5):
        evaluation = sorted(g for g in groups if outer[g] == fold)
        inner = sorted(g for g in groups if outer[g] != fold)
        shuffled = np.random.default_rng(20260913+fold).permutation(inner).tolist()
        result[fold] = {"evaluation_groups": evaluation, "calibration_groups": shuffled[:24],
                        "fit_groups": shuffled[24:], "fit24": shuffled[24:48], "fit48": shuffled[24:72]}
        a.require(len(evaluation) == 24 and len(shuffled) == 96, "Outer split size differs")
    return result


def direct_metric(rows, name, threshold=None):
    """Threshold=None means each row uses its already verified fold decision."""
    binary = [{"gold": r["gold"], "scores": {name: int(r["predictions"][name]) if threshold is None else r["scores"][name]}} for r in rows]
    return a.count(binary, name, .5 if threshold is None else threshold)


def ranking(y, scores):
    """Exact tied-score AUROC and non-interpolated AP, without sklearn metrics."""
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    pos, neg = int(y.sum()), int((1-y).sum())
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0)+1]
    size = np.diff(np.r_[starts, len(y)]); p = np.add.reduceat(y[order], starts); n = size-p
    favourable = float((p*(np.cumsum(n)-.5*n)).sum())
    ap = float(((p[::-1]/pos)*(np.cumsum(p[::-1])/np.cumsum(size[::-1]))).sum()) if pos else None
    return {"auroc": favourable/(pos*neg) if pos and neg else None, "average_precision": ap if pos else None}


def independent_threshold(rows, name):
    values = np.asarray([r["scores"][name] for r in rows]); y = np.asarray([r["gold"] for r in rows])
    candidates = np.unique(values).tolist()+[np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)]
    choices = []
    for threshold in candidates:
        pred = values >= threshold; tp = int(y[pred].sum()); alerts = int(pred.sum())
        choices.append((2*tp/(alerts+y.sum()), tp/alerts if alerts else 0., threshold))
    f1, precision, threshold = max(choices)
    return {"threshold": threshold, "validation_f1": f1, "validation_precision": precision}


def group_intervals(rows, group_ids, draws=2000):
    groups = sorted(group_ids)
    sample = np.random.default_rng(20260913).integers(0, len(groups), (draws, len(groups)))
    counts = np.zeros((len(groups), len(METHODS), 3), int)
    for k, g in enumerate(groups):
        rr = [r for r in rows if r["group_id"] == g]
        for j, name in enumerate(METHODS):
            c = direct_metric(rr, name)
            counts[k, j] = c["tp"], c["fp"], c["fn"]
    total = np.asarray([counts[ix].sum(0) for ix in sample])
    tp, fp, fn = (total[:, :, k] for k in range(3))
    scores = np.divide(2*tp, 2*tp+fp+fn, out=np.full(tp.shape, np.nan, float), where=tp+fn > 0)
    def ci(v):
        good = v[np.isfinite(v)]
        return {"ci95": np.quantile(good, [.025, .975]).tolist() if len(good) else None, "defined_draws": len(good)}
    return {"methods": {name: ci(scores[:, j]) for j, name in enumerate(METHODS)},
            "minus_lb_c01": {name: ci(scores[:, j]-scores[:, 0]) for j, name in enumerate(METHODS[1:], 1)}}


def probability(model, x):
    raw = x.reshape(-1, 28, 28).mean(2) if model["feature"] == "layer_mean28" else x
    z = previous.transform_only(model["scaler"], raw).astype(np.float32)
    if model["components"] is not None: z = (z @ model["components"].T).astype(np.float32)
    # Use the frozen LR's sigmoid kernel: a one-ULP alternative implementation
    # can change an equality at a calibration score used as the exact threshold.
    return expit((z @ model["model"].coef_.T+model["model"].intercept_).reshape(-1))


def compare_metrics(rows, name, reported, threshold=None):
    actual = direct_metric(rows, name, threshold)
    a.require(actual.pop("missing_predictions") == 0, "Missing prediction")
    actual.update(ranking([r["gold"] for r in rows], [r["scores"][name] for r in rows]))
    actual["tokens"] = len(rows); actual["risk_tokens"] = sum(r["gold"] for r in rows)
    actual["risk_rate"] = actual["risk_tokens"]/len(rows)
    actual["alert_rate"] = (actual["tp"]+actual["fp"])/len(rows)
    a.compare_values(actual, reported, name)
    return actual


def verify_pca(model, x, base, label):
    components = model["components"]
    if components is None: return None
    dim = int(model["feature"][3:])
    a.require(components.shape == (dim, 784), "PCA width differs")
    np.testing.assert_allclose(components@components.T, np.eye(dim), atol=1e-10, rtol=0)
    z = previous.transform_only(model["scaler"], x).astype(np.float32).astype(np.float64)
    weights = base/base.sum(); projection = z@components.T
    variances = np.sum(weights[:, None]*projection**2, axis=0)
    action = z.T@(weights[:, None]*projection)
    np.testing.assert_allclose(action, components.T*variances, atol=1e-7, rtol=1e-6, err_msg=label+" training covariance eigenvectors")
    a.require((np.diff(variances) <= 1e-7).all(), "PCA components not descending")
    explained = float(variances.sum()/np.sum(weights[:, None]*z**2))
    np.testing.assert_allclose(explained, model["pca_explained_variance"], atol=1e-10, rtol=0)
    return {"dimensions": dim, "orthogonality_and_fit_only_covariance_eigenvectors": "passed", "explained_variance": explained}


def verify_localization(rows, summary):
    grouped = defaultdict(list)
    for row in rows: grouped[row["item_ids"][0]].append(row)
    result = {}
    for name in METHODS:
        ranks = {}; positions = Counter(); categories = defaultdict(list)
        for iid, rr in grouped.items():
            rr = sorted(rr, key=lambda r:r["token_index"])
            risk = [j for j, r in enumerate(rr) if r["gold"] == 1]
            if {r["gold"] for r in rr} == {0, 1}:
                ranks[iid] = ranking([r["gold"] for r in rr], [r["scores"][name] for r in rr])
            for j, r in enumerate(rr):
                categories[r["attribute_group"]].append(r)
                if r["gold"] == 0 and r["predictions"][name]:
                    label = "FP_in_fully_nonrisk_answer" if not risk else (
                        "FP_one_eligible_token_from_gold_risk" if any(abs(j-k) == 1 for k in risk)
                        else "FP_farther_inside_risk_answer")
                    positions[label] += 1
        reference = summary["localization_diagnostics"][name]
        a.require(dict(positions) == reference["false_positive_locations"], "FP boundary location count differs")
        expected_rank = reference["within_answer_ranking"]
        a.require(len(ranks) == expected_rank["answers_with_both_labels"], "Within-answer denominator differs")
        a.require(set(ranks) == {r["item_id"] for r in expected_rank["per_answer"]}, "Within-answer identities differ")
        for row in expected_rank["per_answer"]: a.compare_values(ranks[row["item_id"]], row, "within "+name)
        for metric in ("auroc", "average_precision"):
            np.testing.assert_allclose(np.mean([r[metric] for r in ranks.values()]), expected_rank["macro_"+metric], atol=1e-12, rtol=0)
        for category, rr in categories.items(): compare_metrics(rr, name, reference["strata"][category])
        result[name] = {"mixed_answers": len(ranks), "false_positive_locations": dict(positions), "strata": "passed"}
    return result


def run(root):
    out = root/"results"; done = a.read_json(out/"complete13.json")
    a.require(done["fits"] == 40 and not done["original_val_or_test_labels_used"] and not done["test_retuning"], "Diagnostic scope differs")
    for name, checksum in done["files_sha256"].items(): a.require(a.sha(out/name) == checksum, "Completed artifact changed: "+name)
    snapshot = a.read_json(out/"source_snapshot13.json")
    started = a.read_json(out/"started13.json")
    a.require(started["source_snapshot_sha256"] == a.sha(out/"source_snapshot13.json"), "Pre-run source snapshot changed")
    a.require(not started["original_val_or_test_labels_used"], "Original holdout labels used")
    for name, checksum in snapshot["local_files_sha256"].items(): a.require(a.sha(root/name) == checksum, "Frozen source changed: "+name)
    a.require(snapshot["prior"] == a.read_json(R12/"results/source_snapshot12.json"), "Original source graph differs")
    a.require(snapshot["outer_folds_sha256"] == a.sha(R12/"results/fold_assignment.json"), "Outer groups changed")
    p12 = snapshot["prior"]; p11 = p12["round11_frozen_graph"]
    for base, hashes in ((R12, p12["local_files_sha256"]), (previous.R11, p11["local_files_sha256"]),
                         (R10, p11["round10_frozen_graph"]["files"])):
        for name, checksum in hashes.items(): a.require(a.sha(base/name) == checksum, "Source dependency changed: "+name)
    protocol = a.read_json(root/"protocol.json")
    a.require(protocol["methods"] == list(METHODS) and protocol["C"] == CS, "Candidate matrix differs")
    items, raw, gold = training_gold(); expected = expected_splits(items)
    output = a.read_lines(out/"oof_scores.jsonl"); summary = a.read_json(out/"summary.json")
    models = pickle.loads((out/"fitted_folds.pkl").read_bytes())
    by_key = {r["token_key"]:r for r in output}
    a.require(len(output) == len(by_key) == len(raw), "OOF coverage/duplicates")
    a.require(set(by_key) == {r["token_key"] for r in raw}, "OOF identities differ")
    a.require(summary["coverage"] == gold["coverage"], "Original denominator differs")
    check_fields = ("token_key", "token_index", "token_id", "row_id", "question_id", "group_id", "condition", "item_ids", "start", "end", "text", "gold", "attribute_group")
    for row in raw:
        saved = by_key[row["token_key"]]
        a.require(all(row[k] == saved[k] for k in check_fields), "OOF gold/offset identity mismatch")
        a.require(set(saved["scores"]) == set(saved["predictions"]) == set(METHODS), "Missing candidate")
        a.require(all(a.finite(v) for v in saved["scores"].values()), "Nonfinite OOF score")
    x, _ = previous.raw_matrix(raw, "token"); ids = np.asarray([r["group_id"] for r in raw]); assigned = np.zeros(len(raw), int)
    report = {"schema": "round13-independent-audit-v1", "status": "passed", "gold": gold,
              "no_refitting_or_gpu": True, "hashes": "passed", "models": {}, "pooled": {}}
    for fold, block in models.items():
        plan = expected[fold]
        for field in ("fit_groups", "calibration_groups", "evaluation_groups"):
            a.require(block[field] == plan[field], "Fixed split/order differs")
        a.require(not (set(plan["fit_groups"]) & set(plan["calibration_groups"]) or
                       set(plan["fit_groups"]) & set(plan["evaluation_groups"]) or
                       set(plan["calibration_groups"]) & set(plan["evaluation_groups"])), "Group leakage")
        full = np.flatnonzero(np.isin(ids, plan["fit_groups"]))
        ca = np.flatnonzero(np.isin(ids, plan["calibration_groups"])); ev = np.flatnonzero(np.isin(ids, plan["evaluation_groups"]))
        np.testing.assert_array_equal(block["calibration_indices"], ca); np.testing.assert_array_equal(block["evaluation_indices"], ev)
        assigned[ev] += 1; report["models"][str(fold)] = {}
        a.require(set(block["models"]) == set(METHODS), "Wrong candidate count")
        np.testing.assert_array_equal(block["models"]["pca16"]["components"], block["models"]["pca64"]["components"][:16])
        for name, model in block["models"].items():
            chosen_groups = plan[name] if name in ("fit24", "fit48") else plan["fit_groups"]
            ix = np.flatnonzero(np.isin(ids, chosen_groups)); np.testing.assert_array_equal(model["fit_indices"], ix)
            fit_rows = [raw[j] for j in ix]; b, w, factors = previous.direct_weights(fit_rows, "token")
            w *= len(full)/w.sum()
            np.testing.assert_allclose(model["base_weights"], b, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(model["loss_weights"], w, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(model["class_factors"], factors, atol=1e-12, rtol=1e-12)
            a.require(model["target_loss_mass"] == len(full) and model["C"] == model["model"].C == CS[name], "Changed loss mass/C")
            raw_x = x[ix].reshape(-1, 28, 28).mean(2) if name == "layer_mean28" else x[ix]
            previous.check_scaler(model["scaler"], raw_x, b, str(fold)+" "+name)
            pca = verify_pca(model, x[ix], b, str(fold)+" "+name)
            stages = {}
            for label, indices in (("train", ix), ("calibration", ca), ("outer", ev)):
                scores = probability(model, x[indices])
                rr = [{"gold":raw[j]["gold"], "scores":{name:float(s)}} for j,s in zip(indices,scores)]
                stages[label] = rr
                compare_metrics(rr, name, summary["folds"][str(fold)][name][label], model["threshold"])
            selection = independent_threshold(stages["calibration"], name)
            a.compare_values(selection, model["threshold_selection"], "calibration-only selection")
            a.require(selection["threshold"] == model["threshold"] == summary["folds"][str(fold)][name]["threshold"], "Wrong threshold provenance")
            oracle = independent_threshold(stages["outer"], name)
            oracle_report = summary["folds"][str(fold)][name]["posthoc_outer_oracle_not_deployable"]
            a.compare_values(oracle, oracle_report, "posthoc-only oracle")
            compare_metrics(stages["outer"], name, oracle_report["metrics"], oracle["threshold"])
            errors = []
            for j, row in zip(ev, stages["outer"]):
                saved = by_key[raw[j]["token_key"]]
                a.require(saved["fold"] == fold, "OOF assigned to wrong model")
                error = abs(saved["scores"][name]-row["scores"][name]); errors.append(error)
                a.require(error <= 1e-9, "Saved OOF score not reproduced from frozen model")
                a.require(saved["predictions"][name] == (saved["scores"][name] >= model["threshold"]), "Wrong fold threshold prediction")
            report["models"][str(fold)][name] = {"fit_groups":len(chosen_groups), "fit_tokens":len(ix),
                "calibration_groups":24, "evaluation_groups":24, "disjoint":True,
                "target_loss_mass":len(full), "threshold":model["threshold"], "calibration_threshold_verified":True,
                "max_heldout_probability_error":max(errors), "pca":pca}
    a.require((assigned == 1).all() and len(models) == 5, "Not held out exactly once")
    for name in METHODS:
        report["pooled"][name] = compare_metrics(output, name, summary["pooled"][name])
        for stage in ("train", "calibration", "outer"):
            for metric in ("f1", "auroc", "average_precision", "alert_rate", "risk_rate"):
                expected_mean = np.mean([summary["folds"][str(f)][name][stage][metric] for f in range(5)])
                np.testing.assert_allclose(expected_mean, summary["fold_mean"][name][stage][metric], atol=1e-12, rtol=0)
    boot = group_intervals(output, {r["group_id"] for r in output})
    saved_boot = summary["paired_bootstrap"]
    a.require((saved_boot["groups"], saved_boot["draws"], saved_boot["seed"]) == (120,2000,20260913), "Bootstrap group scope differs")
    for branch, other in (("methods", "f1"), ("minus_lb_c01", "f1_difference_from_lb_c01")):
        for name, interval in boot[branch].items(): previous.prior.check_interval(interval, saved_boot[other][name], name)
    report["paired_bootstrap"] = boot
    report["localization"] = verify_localization(output, summary)
    report["summary_sha256"] = a.sha(out/"summary.json")
    a.write_json(out/"INDEPENDENT_AUDIT13.json", report)
    print(json.dumps({"status":"passed", "models":40,"tokens":len(output),"labels_parsed":["train"],
                      "pooled_f1":{n:v["f1"] for n,v in report["pooled"].items()},
                      "c001_minus_baseline":boot["minus_lb_c01"]["lb_c001"]},ensure_ascii=False))


def self_test():
    r = ranking([0, 1, 0, 1], [.1, .1, .8, .9])
    np.testing.assert_allclose([r["auroc"], r["average_precision"]], [.625, .75])
    rows = [{"gold": y, "scores": {"x": s}, "predictions": {"x": bool(p)}} for y, s, p in ((0,.9,0),(1,.8,1))]
    a.require(direct_metric(rows, "x")["f1"] == 1., "Fold decisions accidentally replaced by global threshold")
    a.require(direct_metric(rows, "x", .5)["fp"] == 1, "Threshold diagnostic check failed")
    print("SELF_TEST_PASSED: tied-score ranking and fold-specific decisions")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["self-test", "run"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.stage == "self-test": self_test()
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4): run(args.root.resolve())
