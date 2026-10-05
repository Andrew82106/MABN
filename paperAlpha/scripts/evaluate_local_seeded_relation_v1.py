"""Small architecture audit: seed runtime propagation with Layer-1 node risk.

The released two-line monitor keeps the dynamic state initialized from the
untrusted-input indicator and adds the local BN score only in the final head.
This audit tests a stricter system-level composition: a named local template
score is a node state and may pass through an observed message/delegation edge.

This is deliberately a diagnostic variant.  It does not overwrite the frozen
submission result and it never reads hidden labels as features.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def seeded_propagate(two, theta, parsed, bn):
    """Propagate named Layer-1 node evidence through observed edges.

    Every explicit edge is a possible transport relation; the learned gate
    still uses the edge's semantic type, provenance confidence, and taint
    status.  This differs from v2 only in that clean edges can transport a
    *pre-existing node risk state*; they do not create risk without one.
    """
    names = parsed["names"]
    n = len(names)
    source = parsed["source"]
    ew = theta[:7]
    b0, bpath, blocal, bbn, bhard, bdist = theta[7:13]
    l1, node_scores = two.layer1_score(parsed, bn)
    state = np.maximum(source.astype(float), np.asarray([node_scores[a] for a in names], dtype=float))
    edges = []
    for u, v, f, info in parsed["edges"]:
        gate = float(sigmoid(np.dot(ew, f)))
        # Explicit messages/delegations transport an already existing node
        # state.  Confidence controls the point estimate; no hidden outcome is
        # used to infer edge reliability.
        g = gate * max(0.0, min(1.0, float(f[3])))
        edges.append((u, v, g, info, f))
    for _ in range(max(1, n)):
        incoming = [[] for _ in range(n)]
        for u, v, g, _, _ in edges:
            incoming[v].append((u, g))
        nxt = state.copy()
        for v in range(n):
            prod = 1.0
            for u, g in incoming[v]:
                prod *= 1.0 - state[u] * g
            nxt[v] = 1.0 - (1.0 - state[v]) * (1.0 - (1.0 - prod))
        state = nxt
    sink = parsed["sink"]
    path = float(1.0 - np.prod(1.0 - state * sink))
    path_norm = path ** (1.0 / max(1, n - 1))
    local = float(1.0 - np.prod(1.0 - source * sink))
    hard_path = float(path > 0.0 and np.any(sink > 0.0))
    score = float(sigmoid(b0 + bpath * path_norm + blocal * local + bbn * l1 + bhard * hard_path))
    return score


def fit_seeded(two, eps, y, seed):
    bn = two.fit_bn(eps, y)
    rng = np.random.default_rng(seed)
    x0 = np.array([-2., 3., .2, .8, -.4, .1, .1, -2., 4., .2, 1., 1., .3]) + rng.normal(0, .03, 13)

    def objective(theta):
        p = np.asarray([seeded_propagate(two, theta, two.event_features(e), bn) for e in eps])
        bce = -np.mean(y * np.log(np.clip(p, 1e-7, 1 - 1e-7)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)))
        return float(bce + 1e-3 * np.sum(theta * theta))

    res = minimize(objective, x0, method="L-BFGS-B", bounds=[(-8., 8.)] * 13,
                   options={"maxiter": 250, "ftol": 1e-9})
    return bn, res.x


def metrics(y, p, threshold=.5):
    return {
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/independent_mas_v3/traces_public.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/independent_mas_v3/labels.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/submission/development/local_seeded_relation_v1")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    two = load_module("local_seeded_two", ROOT / "scripts/evaluate_two_layer_v2.py")
    traces = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {str(json.loads(line)["episode_id"]): json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()}
    traces = traces[: args.limit] if args.limit else traces
    eps = []
    for trace in traces:
        row = labels[str(trace["episode_id"])]
        # Only public fields are passed to both variants.  Label is held here
        # solely for the split/evaluation and never enters event_features.
        eps.append({**trace, "label": int(row["label"])})
    y = np.asarray([e["label"] for e in eps], dtype=int)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps))
    cut = max(1, int(round(len(eps) * .7)))
    train, test = idx[:cut], idx[cut:]
    train_eps = [eps[int(i)] for i in train]
    test_eps = [eps[int(i)] for i in test]
    bn, theta = fit_seeded(two, train_eps, y[train], args.seed)
    bn_v2 = two.fit_bn(train_eps, y[train])
    theta_v2, _ = two.fit_model(train_eps, y[train], bn_v2, args.seed)
    p_seed = np.asarray([seeded_propagate(two, theta, two.event_features(e), bn) for e in test_eps])
    p_seed_train = np.asarray([seeded_propagate(two, theta, two.event_features(e), bn) for e in train_eps])
    p_v2 = np.asarray([two._propagate(theta_v2, two.event_features(e), bn_v2) for e in test_eps])
    p_v2_train = np.asarray([two._propagate(theta_v2, two.event_features(e), bn_v2) for e in train_eps])
    threshold_seed = two.best_threshold(y[train], p_seed_train)
    threshold_v2 = two.best_threshold(y[train], p_v2_train)
    seeded_metrics = metrics(y[test], p_seed, threshold_seed)
    v2_metrics = metrics(y[test], p_v2, threshold_v2)
    result = {
        "protocol": {"dataset": "independent_mas_v3", "n": len(eps), "limit": args.limit, "split": {"train": len(train), "test": len(test)}, "seed": args.seed},
        "architecture": {"variant": "Layer-1 local BN node-state seeded runtime propagation", "baseline": "TwoLayerBNGraphMonitor-v2 separately fitted under the same split", "causal_claim": False},
        "seeded_relation": seeded_metrics,
        "v2_separately_fitted": v2_metrics,
        "delta": {"auroc": float(seeded_metrics["auroc"] - v2_metrics["auroc"]), "auprc": float(seeded_metrics["auprc"] - v2_metrics["auprc"]), "brier": float(seeded_metrics["brier"] - v2_metrics["brier"])},
        "fit": {"seeded_parameters": [float(v) for v in theta], "v2_parameters": [float(v) for v in theta_v2]},
        "limitations": ["single development split", "diagnostic result only; not added to frozen submission tables", "synthetic labels and public traces remain the same as independent_mas_v3"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "REPORT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "REPORT.md").write_text("# Local-seeded relation audit\n\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
