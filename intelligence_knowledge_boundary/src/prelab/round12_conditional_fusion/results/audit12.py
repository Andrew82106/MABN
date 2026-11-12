"""Independent Round12 audit: no fitting, GPU, or official evaluator imports.

Raw gold is reconstructed by the independent Round10 auditor. Cross-fitting
checks use cached input arrays and explicit fitted parameters, not refitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R11 = ROOT.parent / "round11_logprob"
spec = importlib.util.spec_from_file_location("independent_audit11", R11 / "results/audit11.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
a = prior.a
R10 = prior.R10
FEATURES = ("lb", "lb_nll", "score_only", "additive", "interaction")
ITEM = tuple(x + "__item" for x in FEATURES)
TOKEN = tuple(x + "__token" for x in FEATURES)
CONTRASTS = {"interaction_minus_additive": ("interaction", "additive"),
             "interaction_minus_lb": ("interaction", "lb"),
             "interaction_minus_lb_nll": ("interaction", "lb_nll"),
             "additive_minus_score_only": ("additive", "score_only")}


def direct_weights(rows, kind):
    """Rebuild equal-group/condition/item mass and train-only class factors."""
    paths = [(r["group_id"], r["condition"], r["item_id"] if kind == "item" else r["item_ids"][0]) for r in rows]
    sizes = Counter(paths)
    conditions = defaultdict(set)
    items = defaultdict(set)
    for group, condition, iid in paths:
        conditions[group].add(condition)
        items[group, condition].add(iid)
    base = np.asarray([1 / (len(conditions[g]) * len(items[g, c]) * sizes[g, c, iid]) for g, c, iid in paths])
    base *= len(base) / base.sum()
    y = np.asarray([r["gold"] for r in rows], int)
    mass = np.asarray([base[y == c].sum() for c in (0, 1)])
    a.require((mass > 0).all(), "Cross-fit training fold lacks a class")
    factors = mass.sum() / (2 * mass)
    loss = base * factors[y]
    for g in conditions:
        mask = np.asarray([p[0] == g for p in paths])
        loss[mask] *= (len(rows) / len(conditions)) / loss[mask].sum()
    return base, loss, factors


def raw_matrix(rows, kind):
    """Original BPE item means or individual token rows, no label-based averaging."""
    arrays = {}
    result, nll = [], []
    for r in rows:
        rid = r["row_id"]
        if rid not in arrays:
            with np.load(R10 / "data/features" / (rid + ".npz"), allow_pickle=False) as z:
                arrays[rid] = {k: z[k].copy() for k in ("lookback_features", "token_nll", "token_start", "token_end")}
        z = arrays[rid]
        if kind == "item":
            ix = (z["token_end"] > r["start"]) & (z["token_start"] < r["end"])
            a.require(ix.any(), "Item has no overlapping BPE")
            result.append(z["lookback_features"][ix].mean(0))
            nll.append(z["token_nll"][ix].mean(0))
        else:
            ix = r["token_index"]
            a.require((int(z["token_start"][ix]), int(z["token_end"][ix])) == (r["start"], r["end"]), "Token phase mismatch")
            result.append(z["lookback_features"][ix])
            nll.append(z["token_nll"][ix])
    return np.asarray(result, np.float32), np.asarray(nll, np.float32)


def weighted_stats(x, weights):
    x = np.asarray(x, np.float64)
    mean = np.average(x, axis=0, weights=weights)
    var = np.average((x - mean) ** 2, axis=0, weights=weights)
    return mean, var


def check_scaler(scaler, x, weights, label):
    mean, var = weighted_stats(x, weights)
    np.testing.assert_allclose(scaler.mean_, mean, atol=1e-9, rtol=1e-7, err_msg=label+" mean")
    np.testing.assert_allclose(scaler.var_, var, atol=1e-9, rtol=1e-7, err_msg=label+" variance")
    bound = weights.sum() * np.finfo(float).eps * var + (weights.sum()*mean*np.finfo(float).eps)**2
    expected_scale = np.where(var <= bound, 1., np.sqrt(var))
    np.testing.assert_allclose(scaler.scale_, expected_scale, atol=1e-9, rtol=1e-7, err_msg=label+" scale")


def explicit_logit(model, x):
    scaler, head = model["scaler"], model["model"]
    # Preserve intermediate input dtype, as the frozen sklearn transformer does.
    transformed = np.asarray(x).copy()
    transformed -= scaler.mean_
    transformed /= scaler.scale_
    transformed = transformed.astype(np.float32)
    return (transformed @ head.coef_.T + head.intercept_).reshape(-1)


def transform_only(scaler, x):
    result = np.asarray(x).copy()
    result -= scaler.mean_
    result /= scaler.scale_
    return result


def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1/(1+np.exp(-x[positive]))
    e = np.exp(x[~positive]); out[~positive] = e/(1+e)
    return out


def fusion_matrix(model, logits, nll):
    z = transform_only(model["scaler"], np.column_stack((logits, nll)))
    if model["feature"] == "score_only": return z[:, :1]
    if model["feature"] == "additive": return z
    a.require(model["feature"] == "interaction", "Unknown fusion")
    product = transform_only(model["product_scaler"], (z[:, 0]*z[:, 1]).reshape(-1, 1))
    return np.column_stack((z, product))


def audit_oof(out, all_items, all_tokens, models, frozen):
    assignments = a.read_json(out/"fold_assignment.json")
    groups = sorted({r["group_id"] for r in all_items if r["split"] == "train"})
    shuffled = np.random.default_rng(20260912).permutation(groups)
    expected = {str(g): k for k, chunk in enumerate(np.array_split(shuffled, 5)) for g in chunk}
    a.require(len(groups) == 120 and assignments == expected, "Fold assignment differs from fixed 120 training groups")
    fold_models = pickle.loads((out/"fold_models.pkl").read_bytes())
    reported = a.read_json(out/"training_weights.json")["fusion"]
    drift = a.read_json(out/"distribution_diagnostic.json")
    report = {}
    for kind, rows in (("item", all_items), ("token", all_tokens)):
        tr = [r for r in rows if r["split"] == "train" and r["main_eligible"]]
        va = [r for r in rows if r["split"] == "validation" and r["main_eligible"]]
        key = "item_id" if kind == "item" else "token_key"
        x, n = raw_matrix(tr, kind); vx, vn = raw_matrix(va, kind)
        b, w, factors = direct_weights(tr, kind)
        vb, _, _ = direct_weights(va, kind)
        with np.load(out/("oof_"+kind+".npz"), allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        a.require(values["row_ids"].tolist() == [r[key] for r in tr], "OOF identities/order differ")
        a.require(values["group_ids"].tolist() == [r["group_id"] for r in tr], "OOF group identity differs")
        np.testing.assert_array_equal(values["y"], [r["gold"] for r in tr])
        np.testing.assert_array_equal(values["n"], n)
        np.testing.assert_allclose(values["base_weights"], b, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(values["loss_weights"], w, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reported[kind]["class_factors_full_train"], factors, atol=1e-12, rtol=1e-12)
        fids = np.asarray([assignments[r["group_id"]] for r in tr])
        np.testing.assert_array_equal(values["fold_ids"], fids)
        recomputed = np.full(len(tr), np.nan); seen = np.zeros(len(tr), int)
        fold_report = []
        a.require(len(fold_models[kind]) == 5, "Five saved base folds required")
        for k, model in enumerate(fold_models[kind]):
            fit_ix, held_ix = np.flatnonzero(fids != k), np.flatnonzero(fids == k)
            fit_rows = [tr[j] for j in fit_ix]
            fb, fw, ff = direct_weights(fit_rows, kind)
            training = sorted({r["group_id"] for r in fit_rows})
            held = sorted({tr[j]["group_id"] for j in held_ix})
            stated = reported[kind]["folds"][k]
            a.require(stated["training_groups"] == training and stated["heldout_groups"] == held, "Incorrect fold manifest")
            a.require(not set(training) & set(held), "OOF leakage across groups")
            a.require(stated["training_rows"] == len(fit_ix) and stated["heldout_rows"] == len(held_ix), "OOF fold coverage differs")
            np.testing.assert_allclose(stated["training_class_factors"], ff, atol=1e-12, rtol=1e-12)
            a.require(model["C"] == model["model"].C == .1, "OOF base C changed")
            check_scaler(model["scaler"], x[fit_ix], fb, kind+" fold "+str(k))
            recomputed[held_ix] = explicit_logit(model, x[held_ix]); seen[held_ix] += 1
            fold_report.append({"fold": k, "training_groups": len(training), "heldout_groups": len(held),
                                "training_rows": len(fit_ix), "heldout_rows": len(held_ix), "no_group_overlap": True,
                                "class_factors": ff.tolist(), "fold_weight_sum": float(fw.sum())})
        a.require((seen == 1).all(), "OOF rows not held out exactly once")
        np.testing.assert_allclose(recomputed, values["a"], atol=1e-8, rtol=1e-10)
        validation_error = {}
        for feature in FEATURES[2:]:
            name = feature+"__"+kind; model = models[name]
            a.require(model["base"]["C"] == model["base"]["model"].C == .1, "Full-training base C changed")
            a.require(model["C"] == model["model"].C == frozen["thresholds"][name]["C"], "Meta C differs from frozen selection")
            check_scaler(model["base"]["scaler"], x, b, name+" full base")
            np.testing.assert_array_equal(model["base"]["model"].coef_, models["lb__"+kind]["model"].coef_)
            np.testing.assert_array_equal(model["base"]["model"].intercept_, models["lb__"+kind]["model"].intercept_)
            an = np.column_stack((values["a"], n))
            check_scaler(model["scaler"], an, b, name+" OOF meta")
            z = transform_only(model["scaler"], an)
            check_scaler(model["product_scaler"], (z[:, 0]*z[:, 1]).reshape(-1, 1), b, name+" product")
            expected_width = {"score_only": 1, "additive": 2, "interaction": 3}[feature]
            a.require(model["model"].coef_.shape == (1, expected_width), "Fusion input width differs")
            av = explicit_logit(model["base"], vx)
            fused = fusion_matrix(model, av, vn)
            score = sigmoid((fused @ model["model"].coef_.T + model["model"].intercept_).reshape(-1))
            saved = np.asarray([r["scores"][name] for r in va])
            np.testing.assert_allclose(score, saved, atol=1e-9, rtol=1e-9)
            validation_error[name] = float(np.max(np.abs(score-saved)))
            for label, v, weights in (("train_oof", values["a"], b),
                                      ("train_in_fit_descriptive_only", explicit_logit(model["base"], x), b),
                                      ("validation_full_refit", av, vb)):
                mean, var = weighted_stats(v, weights)
                np.testing.assert_allclose([mean, np.sqrt(var)],
                    [drift[kind][label]["weighted_mean"], drift[kind][label]["weighted_sd"]], atol=1e-8, rtol=1e-9)
        report[kind] = {"rows": len(tr), "folds": fold_report, "each_row_heldout_once": True,
                        "oof_logit_max_absolute_error": float(np.max(np.abs(recomputed-values["a"]))),
                        "full_refit_base_reproduces_original_lb": True,
                        "independent_validation_probability_max_errors": validation_error,
                        "weighted_preprocessing_and_class_factors": "passed",
                        "oof_and_deployment_distribution_statistics": "passed"}
    return report


def paired_intervals(rows, methods, thresholds, group_ids, draws=2000):
    groups = sorted(group_ids)
    sampled = np.random.default_rng(20260912).integers(0, len(groups), (draws, len(groups)))
    counts = np.zeros((len(groups), len(methods), 3), np.int64)
    for i, g in enumerate(groups):
        rr = [r for r in rows if r["main_eligible"] and r["group_id"] == g]
        for j, name in enumerate(methods):
            c = a.count(rr, name, thresholds[name]["threshold"])
            counts[i, j] = c["tp"], c["fp"], c["fn"]
    totals = np.asarray([counts[ix].sum(axis=0) for ix in sampled])
    tp, fp, fn = (totals[:, :, j] for j in range(3))
    def divide(num, den, mask):
        return np.divide(num, den, out=np.full(num.shape, np.nan, float), where=mask)
    values = {"precision": divide(tp, tp+fp, tp+fp > 0), "recall": divide(tp, tp+fn, tp+fn > 0),
              "f1": divide(2*tp, 2*tp+fp+fn, tp+fn > 0)}
    def interval(v):
        ok = v[np.isfinite(v)]
        return {"ci95": np.quantile(ok, [.025, .975]).tolist() if len(ok) else None, "defined_draws": len(ok)}
    suffix = "__item" if methods == ITEM else "__token"
    return {"methods": {name: {k: interval(v[:, j]) for k, v in values.items()} for j, name in enumerate(methods)},
            "contrasts": {label: {k: interval(v[:, methods.index(p+suffix)] - v[:, methods.index(q+suffix)])
                                  for k, v in values.items()} for label, (p, q) in CONTRASTS.items()}}


def audit_counts(items, tokens, metrics, thresholds):
    report = {"item": {}, "token": {}, "group_bootstrap": {}}
    for rows, kind, names, section, subset in ((items, "item", ITEM, "answer_methods", "answer_items"),
            (tokens, "token", TOKEN, "localization_methods", "all_resolved_items")):
        for name in names:
            value = a.count([r for r in rows if r["main_eligible"]], name, thresholds[name]["threshold"])
            expected = metrics[section][name]["micro"] if kind == "item" else metrics[section][name]["all_resolved_items"]["micro"]
            a.compare_values(value, expected, name)
            a.require(value["missing_predictions"] == 0, "Missing predictions: "+name)
            report[kind][name] = {"primary": value}
            tag = "asserted_only" if kind == "item" else "risk_items_only"
            eligible = "asserted_eligible" if kind == "item" else "risk_item_eligible"
            second = a.count([r for r in rows if r[eligible]], name, thresholds[name]["threshold"])
            expected = metrics[section][name][tag] if kind == "item" else metrics[section][name][tag]["micro"]
            a.compare_values(second, expected, name+" "+tag)
            report[kind][name][tag] = second
        rebuilt = paired_intervals(rows, names, thresholds, {r["group_id"] for r in items})
        reported = metrics["group_bootstrap"]["subsets"][subset]
        for branch in ("methods", "contrasts"):
            for name, intervals in rebuilt[branch].items():
                for metric, interval in intervals.items():
                    prior.check_interval(interval, reported[branch][name][metric], name+" "+metric)
        report["group_bootstrap"][subset] = rebuilt
    return report


def check_validation_selection(rows, names, thresholds, selection):
    eligible = [r for r in rows if r["main_eligible"]]
    y = np.asarray([r["gold"] for r in eligible]); positives = int(y.sum())
    for name in names:
        entries = selection[name]
        a.require([e["C"] for e in entries] == [.1, 1.], "Unplanned C candidates")
        best = max(entries, key=lambda e: (e["validation_f1"], e["validation_precision"], -e["C"]))
        a.require(best == thresholds[name], "Frozen C not chosen by validation rule")
        scores = np.asarray([r["scores"][name] for r in eligible])
        choices = np.unique(scores).tolist() + [np.nextafter(scores.max(), np.inf), np.nextafter(scores.min(), -np.inf)]
        objective = []
        for threshold in choices:
            pred = scores >= threshold
            tp, alerts = int(y[pred].sum()), int(pred.sum())
            objective.append((2*tp/(alerts+positives), tp/alerts if alerts else 0., threshold))
        f1, precision, threshold = max(objective)
        np.testing.assert_allclose([threshold, f1, precision],
            [best["threshold"], best["validation_f1"], best["validation_precision"]], atol=1e-12, rtol=0)


def decision_delta(rows, before, after, thresholds, kind):
    changes = []
    for r in rows:
        old, new = (int(r["scores"][n] >= thresholds[n]["threshold"]) for n in (before, after))
        if old == new: continue
        def cell(p):
            if not r["main_eligible"]: return "excluded"
            return ("TP" if p else "FN") if r["gold"] else ("FP" if p else "TN")
        changes.append({"id": r["item_id" if kind == "item" else "token_key"], "row_id": r["row_id"],
                        "text": r["text"], "start": r["start"], "end": r["end"], "gold": r["gold"],
                        "main_eligible": r["main_eligible"], "before": old, "after": new,
                        "before_score": r["scores"][before], "after_score": r["scores"][after],
                        "transition": cell(old)+"->"+cell(new)})
    main = [c for c in changes if c["main_eligible"]]
    return {"main_changes": main, "transitions": dict(Counter(c["transition"] for c in main)),
            "excluded_changes": [c for c in changes if not c["main_eligible"]]}


def run(root):
    out = root/"results"
    done, frozen = a.read_json(out/"test_complete12.json"), a.read_json(out/"freeze12.json")
    a.require(done["freeze12_sha256"] == a.sha(out/"freeze12.json"), "Model freeze changed")
    a.require(not frozen["test_labels_used_in_fit"], "Test labels used in fit")
    for manifest in (frozen, done):
        for name, checksum in manifest["files_sha256"].items():
            a.require(a.sha(out/name) == checksum, "Frozen artifact changed: "+name)
    snapshot = a.read_json(out/"source_snapshot12.json")
    for name, checksum in snapshot["local_files_sha256"].items():
        a.require(a.sha(root/name) == checksum, "Round12 source changed: "+name)
    old_snapshot = a.read_json(R11/"results/source_snapshot11.json")
    a.require(snapshot["round11_frozen_graph"] == old_snapshot, "Round11 snapshot differs")
    for name, checksum in old_snapshot["local_files_sha256"].items():
        a.require(a.sha(R11/name) == checksum, "Round11 source changed: "+name)
    for name, checksum in snapshot["old_results_sha256"].items():
        a.require(a.sha(R11/"results"/name) == checksum, "Original baseline artifact changed: "+name)
    old_done = a.read_json(R11/"results/test_complete11.json")
    a.require(old_done["freeze11_sha256"] == a.sha(R11/"results/freeze11.json"), "Original freeze changed")
    protocol = a.read_json(root/"protocol.json")
    a.require(protocol["methods"] == list(FEATURES) and protocol["contrasts"] == list(CONTRASTS), "Method matrix changed")
    a.require(protocol["base_C"] == .1 and protocol["folds"] == 5 and protocol["seed"] == 20260912, "Cross-fitting protocol changed")
    a.require(frozen["comparison_status"] == "exploratory_retest_of_previously_exposed_round10_test", "Misstated independence")
    aa, tt, gold = a.collect_gold(R10)
    for split, key in (("train", "train_coverage"), ("validation", "validation_coverage")):
        a.require(gold["splits"][split]["coverage"] == frozen[key], "Development denominator changed")
    models = pickle.loads((out/"frozen_models.pkl").read_bytes())
    a.require(set(models) == set(ITEM+TOKEN) == set(frozen["thresholds"]), "Expected exactly ten heads")
    thresholds = frozen["thresholds"]; old_freeze = a.read_json(R11/"results/freeze11.json")
    selection = a.read_json(out/"selection.json")
    baseline = {}
    for split in ("validation", "test"):
        baseline[split] = {}
        for raw, names, kind, prefix, key in ((aa, ITEM, "item", "answer", "item_id"), (tt, TOKEN, "token", "token", "token_key")):
            rows = [r for r in raw if r["split"] == split]
            name = prefix+"_scores_"+split+".jsonl"
            a.attach_scores(rows, a.read_lines(out/name), key, names)
            a.require(all(a.finite(r["scores"][n]) for r in rows for n in names), "Nonfinite prediction")
            old_rows = a.read_lines(R11/"results"/name)
            for feature in ("lb", "lb_nll"):
                method = feature+"__"+kind
                baseline[split][method] = prior.baseline_equal(rows, old_rows, key, method,
                    thresholds[method]["threshold"], old_freeze["thresholds"][method]["threshold"])
            if split == "validation": check_validation_selection(rows, names, thresholds, selection)
    items = [r for r in aa if r["split"] == "test"]; tokens = [r for r in tt if r["split"] == "test"]
    metrics = a.read_json(out/"metrics_test.json")
    a.require(metrics["coverage"] == gold["splits"]["test"]["coverage"], "Test denominator changed")
    boot = metrics["group_bootstrap"]
    a.require((boot["groups"], boot["draws"], boot["seed"]) == (60, 2000, 20260912), "Incorrect paired bootstrap")
    report = {"schema": "round12-independent-audit-v1", "status": "passed", "scope": "exploratory reuse of seen test",
              "no_fitting_or_gpu": True, "source_and_artifact_hashes": "passed", "gold": gold["splits"],
              "baseline_replication": baseline, "validation_selection": "passed for all ten selected heads",
              "oof": audit_oof(out, aa, tt, models, frozen),
              "metrics": audit_counts(items, tokens, metrics, thresholds), "decision_changes": {}}
    for kind, rows in (("item", items), ("token", tokens)):
        report["decision_changes"][kind] = {}
        for p, q in (("score_only", "lb"), ("interaction", "additive"), ("interaction", "lb")):
            report["decision_changes"][kind][p+"_minus_"+q] = decision_delta(rows, q+"__"+kind, p+"__"+kind, thresholds, kind)
    report["cache_provenance"] = prior.check_cache(root)
    report["metrics_sha256"] = a.sha(out/"metrics_test.json")
    a.write_json(out/"INDEPENDENT_AUDIT12.json", report)
    print(json.dumps({"status": "passed", "heads": 10, "oof_rows": {k: v["rows"] for k, v in report["oof"].items()},
                      "test_baseline_max_score_error": {n: v["max_absolute_score_difference"] for n, v in baseline["test"].items()},
                      "decision_changes": {k: {n: v["transitions"] for n, v in d.items()} for k, d in report["decision_changes"].items()}}, ensure_ascii=False))


def self_test():
    rows = []
    for group, cond, y, score in (("g1", "complete", 0, .8), ("g1", "partial", 1, .9),
                                  ("g2", "complete", 0, .1), ("g2", "partial", 1, .7)):
        rows.append({"group_id": group, "condition": cond, "item_id": group+cond, "gold": y,
                     "main_eligible": True, "scores": {name: score for name in ITEM}})
    base, loss, _ = direct_weights(rows, "item")
    np.testing.assert_allclose(base, np.ones(4))
    np.testing.assert_allclose(loss, np.ones(4))
    intervals = paired_intervals(rows, ITEM, {n: {"threshold": .5} for n in ITEM}, {"g1", "g2"}, draws=51)
    np.testing.assert_array_equal(intervals["contrasts"]["interaction_minus_additive"]["f1"]["ci95"], [0., 0.])
    print("SELF_TEST_PASSED: weights and paired-condition contrasts")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["self-test", "run"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.stage == "self-test": self_test()
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4): run(args.root.resolve())
