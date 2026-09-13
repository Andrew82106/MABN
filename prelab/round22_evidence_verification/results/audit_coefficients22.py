"""Read-only independent R22 frozen coefficient audit. No fitting or GPU.

Reads only protocol22.json, run22.py bytes, answer_index22.jsonl,
verifier_designs22.npz, and the five frozen pickle / score NPZ pairs.
No project runner import; no model/scaler fit, transform, or predict call.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parent
REPORT = RESULTS / "COEFFICIENT_AUDIT22.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64))))


def assert_close(a, b, tag, rtol=1e-10, atol=1e-12):
    assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False), (tag, maxerr(a, b))
    return maxerr(a, b)


def weights_from_metadata(rows, y, loss_mass):
    group_rows = defaultdict(list)
    group_conditions = defaultdict(set)
    gc_counts = Counter()
    for i, row in enumerate(rows):
        g, c = row["group_id"], row["condition"]
        group_rows[g].append(i)
        group_conditions[g].add(c)
        gc_counts[g, c] += 1
    n, ng = len(rows), len(group_rows)
    base = np.asarray([(n/ng)/(len(group_conditions[r["group_id"]])*gc_counts[r["group_id"], r["condition"]]) for r in rows], np.float64)
    class_mass = np.asarray([base[y == k].sum() for k in [0, 1]])
    factors = base.sum()/(2*class_mass)
    unnormalized = base*factors[y]
    loss = np.empty(n, np.float64)
    for indices in group_rows.values():
        loss[indices] = (loss_mass/ng)*unnormalized[indices]/unnormalized[indices].sum()
    base_group_error, base_condition_error, within_condition_spread = 0., 0., 0.
    for group, indices in group_rows.items():
        base_group_error = max(base_group_error, abs(float(base[indices].sum())-n/ng))
        for condition in group_conditions[group]:
            ii = [i for i in indices if rows[i]["condition"] == condition]
            base_condition_error = max(base_condition_error, abs(float(base[ii].sum())-(n/ng)/len(group_conditions[group])))
            within_condition_spread = max(within_condition_spread, float(np.ptp(base[ii])))
    return base, factors, loss, group_rows, {
        "base_group_equal_mass_error": base_group_error,
        "base_condition_equal_mass_error": base_condition_error,
        "base_within_group_condition_answer_weight_spread": within_condition_spread,
    }


def sigmoid(logits):
    out = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    out[positive] = 1/(1+np.exp(-logits[positive]))
    e = np.exp(logits[~positive])
    out[~positive] = e/(1+e)
    return out


def audit(report):
    protocol = json.loads((ROOT / "protocol22.json").read_text(encoding="utf-8"))
    answers = [json.loads(line) for line in (RESULTS / "answer_index22.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(answers) == 602 and len({a["item_id"] for a in answers}) == 602
    assert all(a["split"] == "train" for a in answers)
    groups = {a["group_id"] for a in answers}
    assert len(groups) == 278 and len({a["question_id"] for a in answers}) == 301
    eligible = np.asarray([a["main_eligible"] for a in answers], bool)
    assert eligible.sum() == 598
    frozen = [pickle.loads((RESULTS / f"fold_{fold}_frozen22.pkl").read_bytes()) for fold in range(5)]
    outer = [set(f["evaluation_groups"]) for f in frozen]
    assert set.union(*outer) == groups and sum(map(len, outer)) == len(groups)
    assert protocol["fit_counts"]["new_lr"] == 15 and protocol["fit_counts"]["pca"] == 5
    assert protocol["loss_mass"] == 3854. and protocol["C"] == [.001, .01, .1]
    with np.load(RESULTS / "verifier_designs22.npz", allow_pickle=False) as bank:
        hidden = bank["hidden"].astype(np.float64)
        extra = bank["extra"].copy()
    assert hidden.shape == (602, 3584) and extra.shape == (602, 2) and extra.dtype == np.float32
    assert np.isfinite(hidden).all() and np.isfinite(extra).all()
    appearances = np.zeros(602, np.int64)
    fold_reports = []
    for fold, f in enumerate(frozen):
        fit_groups = set(f["fit_groups"])
        cal_groups = set(f["calibration_groups"])
        assert cal_groups == outer[(fold+1) % 5]
        assert fit_groups == groups-outer[fold]-cal_groups
        assert not (fit_groups & cal_groups or fit_groups & outer[fold] or cal_groups & outer[fold])
        ix = np.asarray([i for i, a in enumerate(answers) if a["group_id"] in fit_groups and eligible[i]], np.int64)
        assert np.array_equal(ix, f["fit_answer_indices"])
        appearances[ix] += 1
        rows = [answers[i] for i in ix]
        keys = [r["item_id"] for r in rows]
        actual_fit_groups = sorted({r["group_id"] for r in rows})
        y = np.asarray([r["gold"] for r in rows], np.int64)
        assert set(y.tolist()) == {0, 1}
        base, factors, loss, group_rows, fairness = weights_from_metadata(rows, y, protocol["loss_mass"])
        pc = f["projection"]
        assert np.array_equal(pc["fit_ix"], ix) and pc["fit_keys"] == keys and pc["fit_groups"] == actual_fit_groups
        weight_error = assert_close(pc["base_weights"], base, "pca_base_weights")
        pca_weights = base/base.sum()
        pca_weight_error = assert_close(pc["pca_weights"], pca_weights, "pca_normalized_weights", atol=1e-15)
        weighted_mean = np.sum(hidden[ix]*pca_weights[:, None], axis=0)
        mean_error = assert_close(pc["mean"], weighted_mean, "pca_fit_only_weighted_mean")
        components = np.asarray(pc["components"], np.float64)
        sv = np.asarray(pc["singular_values"], np.float64)
        assert components.shape == (64, 3584) and sv.shape == (64,)
        assert np.isfinite(components).all() and np.isfinite(sv).all()
        orthogonality_error = assert_close(components @ components.T, np.eye(64), "pca_orthonormal", atol=1e-10)
        assert pc["seed"] == 20260922+fold and pc["n_iter"] == 3 and pc["whiten"] is False
        total_variance = np.sum(((hidden[ix]-weighted_mean)**2)*pca_weights[:, None])
        ratio = float(np.sum(sv**2)/total_variance)
        variance_ratio_error = assert_close(ratio, pc["explained_variance_ratio_sum"], "pca_variance_ratio")
        # Apply the frozen basis, never recompute/refit an SVD/PCA basis.
        projected = ((hidden-pc["mean"]) @ components.T).astype(np.float32)
        design = np.column_stack((projected, extra)).astype(np.float32)
        with np.load(RESULTS / f"fold_{fold}_scores22.npz", allow_pickle=False) as saved:
            assert np.array_equal(projected, saved["projected_verifier"]), (fold, "projection_not_exact")
            assert np.array_equal(design, saved["probe_design"]), (fold, "design_not_exact")
            candidates = f["all_lr_candidates"]
            assert len(candidates) == 3
            lr_reports = []
            for j, candidate in enumerate(candidates):
                assert np.array_equal(candidate["fit_ix"], ix)
                assert candidate["fit_keys"] == keys and candidate["fit_groups"] == actual_fit_groups
                assert np.array_equal(candidate["fit_y"], y)
                b_error = assert_close(candidate["base_weights"], base, "lr_base_weights")
                factor_error = assert_close(candidate["class_factors"], factors, "fit_only_class_factors")
                loss_error = assert_close(candidate["loss_weights"], loss, "group_rebalanced_loss")
                actual_loss = np.asarray(candidate["loss_weights"])
                loss_mass_error = assert_close(actual_loss.sum(), 3854., "loss_mass")
                group_loss_error = max(abs(float(actual_loss[ii].sum())-3854./len(group_rows)) for ii in group_rows.values())
                assert group_loss_error < 1e-10
                assert candidate["target_loss_mass"] == 3854. and candidate["width"] == 66
                assert candidate["C"] == protocol["C"][j]
                scaler, model = candidate["scaler"], candidate["model"]
                assert scaler.n_features_in_ == 66 and scaler.with_mean and scaler.with_std
                # StandardScaler validates sample_weight with dtype=X.dtype.
                # Input X was float32, so effective weights are cast before
                # its float64 population-statistic accumulation.
                effective_weights = base.astype(np.float32).astype(np.float64)
                weight_mass = effective_weights.sum()
                fit_design = design[ix].astype(np.float64)
                scaler_mean = np.sum(fit_design*effective_weights[:, None], axis=0)/weight_mass
                scaler_var = np.sum((fit_design-scaler_mean)**2*effective_weights[:, None], axis=0)/weight_mass
                scaler_mean_error = assert_close(scaler.mean_, scaler_mean, "fit_only_scaler_mean")
                scaler_var_error = assert_close(scaler.var_, scaler_var, "fit_only_scaler_variance", atol=1e-20)
                scaler_var_relative_error = float(np.max(np.abs((scaler.var_-scaler_var)/scaler_var)))
                assert np.all(scaler_var > 0)
                scaler_scale_error = assert_close(scaler.scale_, np.sqrt(scaler_var), "scaler_scale", atol=1e-15)
                n_seen_error = assert_close(scaler.n_samples_seen_, weight_mass, "scaler_fit_weight_mass")
                assert model.C == protocol["C"][j] and model.solver == "liblinear" and model.penalty == "l2"
                assert model.max_iter == 2000 and np.max(model.n_iter_) < 2000
                assert model.class_weight is None and model.random_state == protocol["seed"]
                assert np.array_equal(model.classes_, [0, 1]) and model.n_features_in_ == 66
                assert model.coef_.shape == (1, 66) and model.intercept_.shape == (1,)
                # Explicit two-step float32 scaling mirrors arithmetic dtype,
                # without calling scaler.transform or any prediction method.
                scaled = design.copy()
                scaled -= scaler.mean_
                scaled /= scaler.scale_
                logits = scaled @ model.coef_[0] + model.intercept_[0]
                replayed = sigmoid(logits)
                observed = saved[f"probe_C{j}_answer"]
                assert observed.shape == (602,) and np.isfinite(observed).all()
                probability_error = assert_close(replayed, observed, "manual_sigmoid_all_602", rtol=0, atol=1e-12)
                lr_reports.append({"C": candidate["C"], "all_602_probabilities_replayed": True,
                    "max_probability_abs_error": probability_error, "base_weights_max_abs_error": b_error,
                    "class_factors_max_abs_error": factor_error, "loss_weights_max_abs_error": loss_error,
                    "loss_mass": float(actual_loss.sum()), "loss_mass_abs_error": loss_mass_error,
                    "group_loss_equal_mass_max_abs_error": group_loss_error,
                    "scaler_effective_weight_dtype": "float32 before float64 accumulation",
                    "scaler_fit_weight_mass": float(weight_mass), "scaler_n_samples_seen_abs_error": n_seen_error,
                    "scaler_mean_max_abs_error": scaler_mean_error, "scaler_var_max_abs_error": scaler_var_error,
                    "scaler_var_max_relative_error": scaler_var_relative_error, "scaler_scale_max_abs_error": scaler_scale_error,
                    "n_iter": model.n_iter_.tolist()})
        fold_reports.append({"fold": fold, "fit_answers": len(ix), "fit_groups": len(actual_fit_groups),
            "fit_class_counts": {str(k): int(np.sum(y == k)) for k in [0, 1]},
            "calibration_groups": len(cal_groups), "outer_groups": len(outer[fold]),
            "fit_indices_keys_and_groups_exact": True, **fairness,
            "pca_base_weights_max_abs_error": weight_error, "pca_weights_max_abs_error": pca_weight_error,
            "pca_fit_only_mean_max_abs_error": mean_error, "pca_orthogonality_max_abs_error": orthogonality_error,
            "pca_explained_variance_ratio_abs_error": variance_ratio_error,
            "projected_all_602_rows_bitwise_exact": True, "all_602_design_rows_bitwise_exact": True,
            "lr_candidates": lr_reports})
    assert np.all(appearances[eligible] == 3) and np.all(appearances[~eligible] == 0)
    all_candidates = [candidate for f in fold_reports for candidate in f["lr_candidates"]]
    report.update(status="passed", answer_count=602, original_r16_split_train_count=602,
                  question_count=301, group_count=278, eligible_answer_count=598, excluded_fit_answer_count=4,
                  pca_checked=5, lr_checked=15, replayed_probability_count=15*602,
                  every_eligible_answer_fit_appearances=3, every_ineligible_answer_fit_appearances=0,
                  max_probability_abs_error=max(c["max_probability_abs_error"] for c in all_candidates),
                  max_pca_fit_mean_abs_error=max(f["pca_fit_only_mean_max_abs_error"] for f in fold_reports),
                  max_scaler_mean_abs_error=max(c["scaler_mean_max_abs_error"] for c in all_candidates),
                  max_scaler_var_relative_error=max(c["scaler_var_max_relative_error"] for c in all_candidates),
                  folds=fold_reports,
                  limitations=["No PCA/SVD or LR refit was performed; fitted PCA basis provenance is checked through frozen metadata, fit-only mean, code, orthogonality and exact forward projection.",
                               "Fold boundaries are cross-checked across five frozen artifacts; no external R18/R19 assignment or original R16 validation/test source was opened."],
                  independent_weight_subreview="/root/evaluation/event_links independently confirmed the same fit metadata and weight formulas without importing runner code.")


def main():
    files = [ROOT / "protocol22.json", ROOT / "src/run22.py", RESULTS / "answer_index22.jsonl", RESULTS / "verifier_designs22.npz"]
    files += [RESULTS / f"fold_{fold}_{name}22.{ext}" for fold in range(5) for name, ext in [("frozen", "pkl"), ("scores", "npz")]]
    report = {"status": "running", "reviewer": "/root/data_build/extract_review",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Frozen R22 actual-train602 numerical coefficient audit only",
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
              "no_fit_or_retuning": True, "no_model_gpu_loaded": True,
              "no_original_r16_validation_test_opened": True,
              "base_weight_formula": "b_i=(N/G)/(K_g*n_gc)",
              "class_factor_formula": "f_y=sum(b)/(2*sum(b[fit_y==y]))",
              "loss_weight_formula": "u_i=b_i*f_y; w_i=(3854/G)*u_i/sum(u_j for j in group_i)"}
    try:
        with threadpool_limits(limits=4):
            audit(report)
        assert {str(p.relative_to(ROOT)): digest(p) for p in files} == report["source_sha256"], "Source mutated during audit"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_detail=str(error))
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), audit_script_sha256=digest(Path(__file__)))
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "pca_checked", "lr_checked", "replayed_probability_count", "max_probability_abs_error", "max_pca_fit_mean_abs_error", "max_scaler_mean_abs_error", "max_scaler_var_relative_error"]}))


if __name__ == "__main__":
    main()
