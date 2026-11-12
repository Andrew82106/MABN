"""Freeze and fit a fixed-DAG development pilot. Never generates test outputs.

Scratch-trained BN/classifiers use the same eight training runs. The frozen
source BN is separately identified as using historical external training.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import sha, write, now
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN
from paperalpha_monitor.privacy_model_independent import ModelIndependentPrivacyBN
from paperalpha_monitor.branch_benchmark import sample_from_prefix, field_features, metrics, rule_distribution, classifier_distribution

PROTOCOL = {
    "regularization": [.001, .03], "seeds": [0, 17], "bn_max_iter": 500,
    "bn_families": ["branch", "branch_uncoupled", "pooled"],
    "logistic_C": [.1, 1., 10.], "histgb_leaves": [2, 3],
    "histgb": {"max_iter": 80, "learning_rate": .1, "min_samples_leaf": 2, "l2_regularization": 1., "early_stopping": False, "random_state": 0},
    "training": "Scratch for all fitted candidates, per-field final labels. BN also models observed W; no inherited source training advantage for main comparison",
    "selection": "Best converged training loss initial seed per family/lambda, then minimum validation primary k=1 Brier; tie by candidate ID. Classifiers minimum same validation metric",
    "threshold": .5, "nll_clip": 1e-12,
    "baseline_evidence": "Full two-field context, worker occurrence and ownership matrix, visible cues; identical current-authority shared-world gate. No case/condition/template ID, timestamp or future text",
    "uncertainty_prior": "Unknown authority logits initially zero; current pilot has no unknown facts, so does NOT learn or validate these priors",
    "frozen_source": "Historical identity-free BN, pooled with zero visible coefficients; external-trained transfer reference only, not matched training history",
    "heldout": "No test data read or generated. Eight planned test runs remain separate until model/protocol audit. This run only fits and selects on 8 train / 8 validation runs",
    "scope": "Small related reconciliation pilot, not original MAS method reproduction, calibration or structure-benefit evidence",
}
NEW_SOURCES = ('code/commands/train_branch_publisher.py', 'code/commands/verify_branch_publisher.py', 'code/commands/branch_graph_evidence.py',
    'code/commands/branch_publisher_fixtures.py', 'code/commands/scalar_binary_graph.py', 'code/src/paperalpha_monitor/branch_publisher_bn.py',
    'code/src/paperalpha_monitor/branch_benchmark.py', 'code/tests/test_branch_publisher_bn.py', 'code/tests/test_branch_benchmark.py')


def prepare(input_run, source_fit, out):
    if out.exists():
        raise ValueError("new training directory required")
    parent = json.loads((input_run / "PLAN.json").read_text(encoding="utf-8"))
    audited = Path(parent["audited"])
    for name, expected in parent["input_sha256"].items():
        if sha(audited / name) != expected:
            raise ValueError("development input changed: " + name)
    files = {p.relative_to(input_run / "source_snapshot").as_posix(): p for p in (input_run / "source_snapshot").rglob("*") if p.is_file()}
    for name, expected in parent["source_sha256"].items():
        if sha(files[name]) != expected:
            raise ValueError("parent source changed: " + name)
    files.update({name: ROOT / name for name in NEW_SOURCES})
    for name, source in files.items():
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for split in ("train", "validation"):
        dest = out / "inputs" / (split + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audited / (split + ".json"), dest)
    frozen = json.loads((source_fit / "FROZEN.json").read_text(encoding="utf-8"))
    name = frozen["selected"]["bn"]["bn_coupled"]
    if sha(source_fit / name) != frozen["model_files_sha256"][name]:
        raise ValueError("historical source BN changed")
    (out / "source_prior").mkdir()
    shutil.copyfile(source_fit / name, out / "source_prior/model.json")
    shutil.copyfile(source_fit / "FROZEN.json", out / "source_prior/original_manifest.json")
    write(out / "PLAN.json", {"created_at": now(), "protocol": PROTOCOL,
        "data_run": parent["data_run"], "development_input_run": str(input_run),
        "data_plan_sha256": sha(Path(parent["data_run"]) / "PLAN.json"), "audited_report_sha256": sha(audited / "REPORT.json"),
        "input_sha256": {s: sha(out / s) for s in ("inputs/train.json", "inputs/validation.json", "source_prior/model.json", "source_prior/original_manifest.json")},
        "source_sha256": {s: sha(out / "source_snapshot" / s) for s in files},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}})
    print(json.dumps({"prepared": str(out), "sources": len(files), "protocol": PROTOCOL}), flush=True)


def load_data(out):
    rows = {s: json.loads((out / "inputs" / (s + ".json")).read_text(encoding="utf-8")) for s in ("train", "validation")}
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    data_plan = json.loads((Path(plan["data_run"]) / "PLAN.json").read_text(encoding="utf-8"))
    all_ids = []
    for split, subset in rows.items():
        if len(subset) != 8 or any(r["split"] != split for r in subset):
            raise ValueError("exact development split required")
        expected = set(data_plan["split_template_ids"][split])
        if {r["template_id"] for r in subset} != expected:
            raise ValueError("source template grouping changed")
        all_ids.extend(r["id"] for r in subset)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("overlapping development records")
    samples = {s: [sample_from_prefix(r["prefix"]) for r in subset] for s, subset in rows.items()}
    return rows, samples


def restore_bn(path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    cls = BranchPublisherPrivacyBN if saved["model"] == BranchPublisherPrivacyBN.MODEL_NAME else PublisherViewPrivacyBN
    return cls.restore(saved)


def distributions(model, samples):
    return [model.predict(s if isinstance(model, BranchPublisherPrivacyBN) else s.release)["violation_count_probabilities"] for s in samples]


def choose(candidates):
    chosen = {}
    for family in sorted({c["family"] for c in candidates}):
        subset = [c for c in candidates if c["family"] == family]
        if family in PROTOCOL["bn_families"]:
            subset = [min((c for c in subset if c["lambda"] == regularization and c["optimizer"]["success"]),
                          key=lambda c: (c["optimizer"]["loss"], c["id"]), default=None)
                      for regularization in PROTOCOL["regularization"]]
            subset = [c for c in subset if c is not None]
        if not subset:
            raise ValueError("no eligible converged candidate for " + family)
        chosen[family] = min(subset, key=lambda c: (c["metrics"]["validation"]["1"]["brier"], c["id"]))["id"]
    return chosen


def run(out):
    if Path(__file__).resolve() != out / "source_snapshot/code/commands/train_branch_publisher.py":
        raise ValueError("execute frozen training source")
    import paperalpha_monitor.branch_publisher_bn as implementation
    assert Path(implementation.__file__).resolve() == out / "source_snapshot/code/src/paperalpha_monitor/branch_publisher_bn.py"
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    assert plan["protocol"] == PROTOCOL
    for name, expected in plan["source_sha256"].items():
        assert sha(out / "source_snapshot" / name) == expected, name
    for name, expected in plan["input_sha256"].items():
        assert sha(out / name) == expected, name
    assert sha(Path(plan["data_run"]) / "PLAN.json") == plan["data_plan_sha256"]
    if (out / "models").exists():
        raise ValueError("training cannot overwrite a previous or partial run")
    rows, samples = load_data(out)
    fs = [r["outcome"]["appearance"] for r in rows["train"]]
    authority = [r["outcome"]["authority_failure"] for r in rows["train"]]
    assert all(len(s.release.graph.facts) == 0 for split in samples.values() for s in split)
    candidates = []

    def register(identity, family, model_file, dist, **extra):
        entry = {"id": identity, "family": family, "file": model_file, **extra,
            "metrics": {s: metrics(dist[s], [r["outcome"] for r in rows[s]]) for s in rows},
            "distributions": dist}
        candidates.append(entry)
        write(out / "candidates" / (identity + ".json"), entry)
        print(json.dumps({"candidate": identity, "validation_brier": entry["metrics"]["validation"]["1"]["brier"],
                          "converged": extra.get("optimizer", {}).get("success")}), flush=True)

    for family in PROTOCOL["bn_families"]:
        xs = samples["train"] if family != "pooled" else [s.release for s in samples["train"]]
        cls = PublisherViewPrivacyBN if family == "pooled" else BranchPublisherPrivacyBN
        for regularization in PROTOCOL["regularization"]:
            for seed in PROTOCOL["seeds"]:
                identity = f"{family}_lambda{regularization}_seed{seed}"
                bn = cls(ModelIndependentPrivacyBN(coupled=family != "branch_uncoupled", seed=seed), regularization=regularization)
                bn.fit(xs, fs, authority, max_iter=PROTOCOL["bn_max_iter"])
                model_file = "models/" + identity + ".json"
                write(out / model_file, bn.parameters())
                register(identity, family, model_file, {s: distributions(bn, samples[s]) for s in rows},
                    **{"lambda": regularization, "seed": seed, "optimizer": bn.optimizer})
    features = np.concatenate([field_features(s) for s in samples["train"]])
    labels = np.asarray(fs).ravel()
    for family, grid in (("logistic", PROTOCOL["logistic_C"]), ("histgb", PROTOCOL["histgb_leaves"])):
        for value in grid:
            identity = f"{family}_grid{value}"
            estimator = (make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=2000)) if family == "logistic"
                else HistGradientBoostingClassifier(max_leaf_nodes=value, **PROTOCOL["histgb"]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(features, labels)
            model_file = "models/" + identity + ".joblib"
            joblib.dump(estimator, out / model_file)
            dist = {s: [classifier_distribution(sample, estimator).tolist() for sample in samples[s]] for s in rows}
            register(identity, family, model_file, dist, fit_warnings=[str(w.message) for w in caught])
    for method in ("persistence_shared_gate", "visible_compliance_shared_gate"):
        dist = {s: [rule_distribution(sample, method).tolist() for sample in samples[s]] for s in rows}
        register(method, method, None, dist)
    source = ModelIndependentPrivacyBN.restore(json.loads((out / "source_prior/model.json").read_text(encoding="utf-8")))
    source = PublisherViewPrivacyBN(source)
    write(out / "models/frozen_source_pooled.json", source.parameters())
    register("frozen_source_pooled", "frozen_source_pooled", "models/frozen_source_pooled.json",
        {s: distributions(source, samples[s]) for s in rows}, scope="External source-trained, no pilot fitting; not same-history comparison")
    selected = choose(candidates)
    selected_entries = {family: next(c for c in candidates if c["id"] == identity) for family, identity in selected.items()}
    write(out / "SELECTED.json", {"created_at": now(), "selected": selected,
        "selection_rule": PROTOCOL["selection"], "test_used": False,
        "entries": {family: {k: c[k] for k in ("id", "family", "file", "metrics")} for family, c in selected_entries.items()}})
    write(out / "FROZEN.json", {"created_at": now(), "schema": "fixed-branch-pilot-monitor-v1", "selected": selected,
        "entries": {family: {k: c[k] for k in ("id", "family", "file")} for family, c in selected_entries.items()},
        "files_sha256": {name: sha(out / name) for name in ("PLAN.json", "SELECTED.json", *[c["file"] for c in selected_entries.values() if c["file"]])},
        "source_sha256": plan["source_sha256"], "gate_prior": .5, "threshold": .5,
        "scope": PROTOCOL["scope"], "test_acquisition_authorized_by_this_file": False})
    report = {"selected_metrics": {family: c["metrics"] for family, c in selected_entries.items()},
        "candidate_count": len(candidates), "bn_fits": 12, "classifier_fits": 5,
        "train_runs": 8, "validation_runs": 8, "test_runs_generated": 0, "new_api_calls": 0,
        "training_feature_shape": list(features.shape), "training_design_rank_with_intercept": int(np.linalg.matrix_rank(np.c_[np.ones(len(features)), features])),
        "unique_observed_worker_matrices": len({tuple(s.worker_presence.ravel()) for subset in samples.values() for s in subset}),
        "unknown_authority_facts_in_this_cohort": 0,
        "model_selection_not_independent_test": True, "scope": PROTOCOL["scope"],
        "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}}
    write(out / "REPORT.json", report)
    print(json.dumps({"selected": selected, "selected_validation": {f: c["metrics"]["validation"] for f, c in selected_entries.items()},
                      "candidates": len(candidates), "test_used": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--source-fit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.input_run.resolve(), args.source_fit.resolve(), args.out.resolve())
    else:
        run(args.out.resolve())
