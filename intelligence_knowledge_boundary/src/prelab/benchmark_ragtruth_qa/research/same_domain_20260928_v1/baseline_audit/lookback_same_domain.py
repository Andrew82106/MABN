"""Same-fit original Lookback Lens 1024-feature / 8-token / default-LR baseline.

CPU only. Consumes the root's label-blind whitebox_signals.py cache. Never
loads the old fitted LR or any test labels. Report fixed C=1 plus a separately
named, predeclared four-C validation-selected contrast; no scaler or weighting.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import shutil
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))
from metrics import evaluate_partition


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean8(x):
    if len(x) < 8:
        raise ValueError("Original Lookback requires at least 8 BPE; no padding or fabricated score")
    return np.stack([x[i:i+8].mean(axis=0, dtype=np.float32) for i in range(len(x)-7)])


def native_to_tokens(probabilities):
    p = np.asarray(probabilities, np.float64)
    total, n = np.zeros(len(p)+7, np.float64), np.zeros(len(p)+7, np.int32)
    for i, value in enumerate(p):
        total[i:i+8] += value
        n[i:i+8] += 1
    assert (n > 0).all()
    return total / n


def load_partition(args, split):
    values = rows(args.data / split / "sequences.jsonl")
    features, labels, ledger = [], [], []
    for row in values:
        rid = row["response_id"]
        path = args.signals / f"{rid}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert receipt["response_id"] == rid and receipt["partition"] == split
        assert receipt["npz_sha256"] == sha(path)
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["token_ids"], row["response_token_ids"])
            assert np.array_equal(z["response_token_offsets"], row["response_token_offsets"])
            x = z["lb_prefix_pre_header"].copy()
        assert x.shape == (len(row["risk_mask"]), 1024) and np.isfinite(x).all()
        assert np.all((x >= 0) & (x <= 1))
        span = mean8(x)
        risk = np.asarray(row["risk_mask"], bool)
        y = np.array([risk[i:i+8].any() for i in range(len(risk)-7)], np.int8)
        features.append(span); labels.append(y)
        ledger.append({"response_id": rid, "group_id": row["group_id"], "spans": len(span),
                       "cache_sha256": receipt["npz_sha256"], "source_sha256": receipt["source_sha256"]})
    return features, labels, ledger


def fit_one(c, x, y, xv, yv, lv, destination, protocol):
    destination.mkdir(parents=True, exist_ok=True)
    attempts = []
    # Increasing solver budget changes neither C, objective, features nor architecture.
    for budget in (1000, 3000, 10000):
        model = LogisticRegression(C=c, max_iter=budget)
        begun = time.time()
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            model.fit(x, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
        attempts.append({"max_iter": budget, "iterations": model.n_iter_.tolist(), "converged": converged,
                         "warnings": [str(w.message) for w in captured], "seconds": time.time()-begun})
        save(destination / "CONVERGENCE.json", attempts)
        if converged:
            break
    if not converged:
        raise RuntimeError(f"C={c} remains unconverged; no final baseline published")
    assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == 1024
    with (destination / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    token_scores, native_scores = {}, {}
    with threadpool_limits(limits=4):
        for value, row in zip(xv, lv):
            probability = model.predict_proba(value)[:, 1]
            native_scores[row["response_id"]] = probability.tolist()
            token_scores[row["response_id"]] = native_to_tokens(probability)
    save(destination / "VALIDATION_NATIVE_SCORES.json", native_scores)
    save(destination / "VALIDATION_TOKEN_SCORES.json", {k: v.tolist() for k, v in token_scores.items()})
    pooling_evaluations = {pooling: evaluate_partition("validation", token_scores, pooling=pooling)
                           for pooling in ("mean", "max")}
    pooling = max(("mean", "max"), key=lambda p: tuple(pooling_evaluations[p]["selection_key"]))
    report = dict(pooling_evaluations[pooling])
    report["pooling_evaluations"] = pooling_evaluations
    report["selected_pooling"] = pooling
    for name, evaluation in pooling_evaluations.items():
        save(destination / f"VALIDATION_{name.upper()}.json", evaluation)
    report.update({"C": c, "iterations": model.n_iter_.tolist(), "convergence_attempts": attempts,
                   "converged": True, "fit_spans": len(y),
                   "validation_spans": sum(len(v) for v in yv), "test_opened": False})
    save(destination / "VALIDATION.json", report)
    save(destination / "PROTOCOL.json", {**protocol, "C": c, "classifier": model.get_params()})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=RUN / "data")
    p.add_argument("--signals", type=Path, default=RUN / "signals")
    p.add_argument("--out", type=Path, default=HERE / "lookback_run")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Preserve the earlier fit; use a fresh output directory")
    started = time.time()
    xf, yf, lf = load_partition(args, "fit")
    xv, yv, lv = load_partition(args, "validation")
    assert not {x["group_id"] for x in lf} & {x["group_id"] for x in lv}
    assert len({x["source_sha256"] for x in lf + lv}) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"identity": "Lookback Lens original architecture / same-fit / NF4 replay",
        "features": "native lb_prefix_pre_header, 32 layers x32 heads; unweighted mean over8 complete raw BPE",
        "classifier": LogisticRegression(max_iter=1000).get_params(), "scaler": None, "sample_weight": None,
        "baseline_C": 1., "tuned_C_grid": [.01, .1, 1., 10.],
        "tuned_selection": "maximum common no-hint validation selection_key; tie prefer C=1 then ascending C",
        "solver_budget": "1000; if ConvergenceWarning rerun unchanged objective at3000 then10000; fail if still unconverged",
        "adaptation": "8-window probability coverage-mean per BPE -> report mean4 and lexical max4 -> answer max",
        "pooling_selection": "validation selection_key maximum; tie mean; frozen together with thresholds",
        "fit_answers": len(lf), "validation_answers": len(lv), "fit_spans": sum(len(v) for v in yf),
        "data_manifest_sha256": sha(args.data / "MANIFEST.json"), "code_sha256": sha(__file__),
        "metrics_sha256": sha(RUN / "metrics.py"),
        "validation_only_thresholds": "common no-hint validation threshold rule",
        "software": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "scipy")},
        "test_opened": False, "gpu_used": False}
    save(args.out / "PROTOCOL.json", protocol)
    cache = {"fit": lf, "validation": lv}
    save(args.out / "CACHE_BINDINGS.json", cache)
    x, y = np.concatenate(xf), np.concatenate(yf)
    del xf, yf
    reports, selected, best_key = {}, None, None
    # The C=1 baseline also participates in the predeclared tuned comparison.
    for c in (1., .01, .1, 10.):
        candidate = args.out / "candidates" / f"C{c:g}"
        report = fit_one(c, x, y, xv, yv, lv, candidate, protocol)
        save(candidate / "CACHE_BINDINGS.json", cache)
        reports[str(c)] = report
        key = tuple(report["selection_key"])
        if best_key is None or key > best_key:
            best_key, selected = key, candidate
        if c == 1.:
            for name in ("model.pkl", "PROTOCOL.json", "VALIDATION.json", "VALIDATION_MEAN.json", "VALIDATION_MAX.json", "VALIDATION_NATIVE_SCORES.json", "VALIDATION_TOKEN_SCORES.json", "CONVERGENCE.json"):
                shutil.copyfile(candidate / name, args.out / name)
        save(args.out / "ALL_CANDIDATE_RESULTS.json", reports)
        print(json.dumps({"C": c, "validation_key": list(key), "convergence": report["convergence_attempts"],
                          "elapsed_seconds": time.time()-started}), flush=True)
    tuned = args.out / "tuned"
    tuned.mkdir()
    for name in ("model.pkl", "PROTOCOL.json", "VALIDATION.json", "VALIDATION_NATIVE_SCORES.json",
                 "VALIDATION_TOKEN_SCORES.json", "VALIDATION_MEAN.json", "VALIDATION_MAX.json", "CONVERGENCE.json", "CACHE_BINDINGS.json"):
        shutil.copyfile(selected / name, tuned / name)
    tuned_report = json.loads((tuned / "VALIDATION.json").read_text(encoding="utf-8"))
    save(args.out / "TUNED_SELECTED.json", {"candidate": str(selected), "run_dir": str(tuned), "selection_key": list(best_key),
         "pooling": tuned_report["selected_pooling"], "thresholds": tuned_report["thresholds"]})
    save(args.out / "COMPLETE.json", {"baseline": str(args.out), "tuned": str(tuned), "elapsed_seconds": time.time()-started,
                                      "all_candidates_converged": True, "test_opened": False})


if __name__ == "__main__":
    main()
