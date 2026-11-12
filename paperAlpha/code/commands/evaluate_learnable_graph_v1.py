"""Learnable, interpretable message-passing monitor for compositional MAS.

The model learns edge gates from *event semantics* (explicit message edge,
data class, and edge kind), then propagates an untrusted source signal over
the observed graph.  It never consumes the episode label, the generated
``label_evidence``, or a terminal violation field as an input.  The learned
parameters are fitted on the training partition only.  This is intentionally
small and auditable: the point is to test whether a learnable graph layer can
retain the explicit path explanation used by the rule monitor.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

PRIV_TOOLS = {"payments.write", "filesystem.write", "shell.exec"}
AGENT_ORDER = ["ingress", "planner", "reviewer", "executor"]
AGENT_INDEX = {a: i for i, a in enumerate(AGENT_ORDER)}


def read_eps(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_arrays(ep):
    """Return source, sink and edge feature tensors without using labels."""
    n = len(AGENT_ORDER)
    source = np.zeros(n, dtype=float)
    sink = np.zeros(n, dtype=float)
    # Edge rows: source index, target index, [bias, untrusted, unexpected].
    edges = []
    for e in ep.get("events", []):
        actor = e.get("actor")
        if actor not in AGENT_INDEX:
            continue
        if (e.get("event_type") == "input_received"
                and e.get("data_class") == "untrusted"):
            source[AGENT_INDEX[actor]] = 1.0
        if (e.get("event_type") == "tool_call"
                and e.get("tool") in PRIV_TOOLS):
            sink[AGENT_INDEX[actor]] = 1.0
        if (e.get("event_type") not in {"message_sent", "delegation"}
                or e.get("target") not in AGENT_INDEX):
            continue
        # Feature meanings are deliberately documented and human readable.
        # No temporal adjacency is inferred as an edge.
        kind = e.get("edge_kind") or "other"
        edges.append((AGENT_INDEX[actor], AGENT_INDEX[e["target"]],
                      np.array([1.0, float(e.get("data_class") == "untrusted"),
                                float(kind == "unexpected")], dtype=float),
                      {"source": actor, "target": e["target"],
                       "data_class": e.get("data_class"),
                       "edge_kind": kind, "sequence": e.get("sequence")}))
    return source, sink, edges


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def forward(theta, ep, return_explanation=False):
    """Differentiable noisy-OR message passing and calibrated risk head.

    theta[:3] are edge-gate coefficients; theta[3:6] are risk-head
    coefficients (bias, propagated-path weight, local co-occurrence weight).
    Three rounds suffice for the fixed four-agent workflow but are written as
    ``n`` rounds to make the implementation valid for arbitrary DAG depth.
    """
    edge_w = theta[:3]
    b0, b_path, b_local = theta[3:6]
    source, sink, edges = episode_arrays(ep)
    gates = [float(sigmoid(np.dot(edge_w, feat))) for _, _, feat, _ in edges]
    state = source.copy()
    history = [state.copy()]
    for _ in range(len(AGENT_ORDER)):
        incoming = [[] for _ in AGENT_ORDER]
        for (u, v, _, _), g in zip(edges, gates):
            incoming[v].append((u, g))
        nxt = state.copy()  # source evidence remains available
        for v in range(len(AGENT_ORDER)):
            # noisy-OR over incoming evidence; differentiable wrt gates.
            prod = 1.0
            for u, g in incoming[v]:
                prod *= 1.0 - state[u] * g
            nxt[v] = max(nxt[v], 1.0 - prod)
        state = nxt
        history.append(state.copy())
    path_evidence = float(1.0 - np.prod(1.0 - state * sink))
    local = float(1.0 - np.prod(1.0 - source * sink))
    risk = float(sigmoid(b0 + b_path * path_evidence + b_local * local))
    if not return_explanation:
        return risk
    # Rank observed paths by product of learned edge gates.  This explanation
    # is evidence attribution, not a causal-discovery claim.
    adj = defaultdict(list)
    for (u, v, _, info), g in zip(edges, gates):
        adj[u].append((v, g, info))
    paths = []
    for s in np.where(source > 0)[0]:
        stack = [(s, [s], 1.0, [])]
        while stack:
            cur, nodes, score, pedges = stack.pop()
            if sink[cur] > 0 and len(nodes) > 1:
                paths.append({"nodes": [AGENT_ORDER[x] for x in nodes],
                              "edge_score": float(score), "edges": pedges})
            if len(nodes) >= len(AGENT_ORDER):
                continue
            for nxt, g, info in adj[cur]:
                if nxt not in nodes:
                    stack.append((nxt, nodes + [nxt], score * g,
                                  pedges + [info]))
    paths.sort(key=lambda p: p["edge_score"], reverse=True)
    return risk, {"path_evidence": path_evidence, "local_evidence": local,
                  "propagated_state": {AGENT_ORDER[i]: float(state[i])
                                        for i in range(len(AGENT_ORDER))},
                  "top_paths": paths[:3]}


def loss(theta, eps, y, l2=1e-3):
    p = np.asarray([forward(theta, ep) for ep in eps])
    # Stable binary cross entropy plus mild regularization for identifiable
    # gates.  Labels are only used by this training objective.
    bce = -np.mean(y * np.log(np.clip(p, 1e-8, 1 - 1e-8))
                   + (1 - y) * np.log(np.clip(1 - p, 1e-8, 1 - 1e-8)))
    return float(bce + l2 * np.sum(theta * theta))


def fit(eps, y, seed):
    rng = np.random.default_rng(seed)
    # Small random initialization avoids a symmetric all-edge solution.
    x0 = np.array([-2.0, 0.5, 0.0, -2.0, 3.0, 0.5]) + rng.normal(0, .05, 6)
    result = minimize(loss, x0, args=(eps, y), method="L-BFGS-B",
                      bounds=[(-8, 8)] * 6,
                      options={"maxiter": 300, "ftol": 1e-10})
    return result


def best_threshold(y, scores):
    vals = np.unique(np.r_[0., scores, 1.])
    return float(max(vals, key=lambda t: f1_score(y, scores >= t, zero_division=0)))


def evaluate(y, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, scores)) if len(np.unique(scores)) > 1 else None,
            "auprc": float(average_precision_score(y, scores)),
            "threshold": threshold, "predicted_positive": int(pred.sum())}


def ablation_audit(theta, eps, indices):
    """Audit whether explanations are sensitive to the cited edge evidence.

    For every positive test episode, remove one explicit untrusted edge at a
    time and measure the change in score.  This is a model-sensitivity check,
    not a causal intervention claim about the original MAS.
    """
    tainted_deltas, clean_deltas = [], []
    for i in indices:
        ep = eps[int(i)]
        if int(ep.get("label", 0)) != 1:
            continue
        base = forward(theta, ep)
        for j, e in enumerate(ep.get("events", [])):
            if e.get("event_type") not in {"message_sent", "delegation"} or not e.get("target"):
                continue
            cf = copy.deepcopy(ep)
            del cf["events"][j]
            delta = base - forward(theta, cf)
            (tainted_deltas if e.get("data_class") == "untrusted" else clean_deltas).append(float(delta))
    def summary(values):
        if not values:
            return {"n": 0, "mean_delta": None, "median_delta": None,
                    "positive_delta_fraction": None}
        a = np.asarray(values)
        return {"n": int(len(a)), "mean_delta": float(a.mean()),
                "median_delta": float(np.median(a)),
                "positive_delta_fraction": float(np.mean(a > 1e-6))}
    return {"untrusted_edge_removal": summary(tainted_deltas),
            "clean_edge_removal": summary(clean_deltas),
            "interpretation": "sensitivity only; not a causal intervention"}


def run_split(eps, idx_train, idx_test, seed):
    y = np.asarray([int(e["label"]) for e in eps])
    fit_result = fit([eps[i] for i in idx_train], y[idx_train], seed)
    theta = fit_result.x
    train_p = np.asarray([forward(theta, eps[i]) for i in idx_train])
    test_p = np.asarray([forward(theta, eps[i]) for i in idx_test])
    threshold = best_threshold(y[idx_train], train_p)
    return theta, fit_result, evaluate(y[idx_test], test_p, threshold), train_p, test_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/compositional_mas_v1/episodes.jsonl"))
    ap.add_argument("--out", type=Path, default=(Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/learnable_graph_v1"))
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--test-size", type=float, default=.30)
    args = ap.parse_args()
    eps = read_eps(args.input)
    y = np.asarray([int(e["label"]) for e in eps])
    idx = np.arange(len(eps))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size,
                                            random_state=args.seed, stratify=y)
    theta, result, metric, train_p, test_p = run_split(eps, train_idx, test_idx, args.seed)
    # Also report a stricter leave-one-family-out protocol. The held-out family
    # is selected deterministically (direct_attack), and never used in fitting.
    hold_family = "direct_attack"
    tr2 = np.asarray([i for i, e in enumerate(eps) if e.get("scenario_family") != hold_family])
    te2 = np.asarray([i for i, e in enumerate(eps) if e.get("scenario_family") == hold_family])
    theta_loo, res_loo, metric_loo, _, _ = run_split(eps, tr2, te2, args.seed + 1)
    preds = []
    for i in test_idx:
        p, exp = forward(theta, eps[i], return_explanation=True)
        preds.append({"episode_id": eps[i]["episode_id"], "label": int(eps[i]["label"]),
                      "risk_probability": p, "explanation": exp})
    ablation = ablation_audit(theta, eps, test_idx)
    baseline_path = (Path(__file__).resolve().parents[2] / "experiments/synthetic_mas/runs/mas_baselines_v1/metrics.json")
    baseline_comparison = None
    if baseline_path.exists():
        try:
            baseline_comparison = json.loads(baseline_path.read_text(encoding="utf-8")).get("metrics")
        except (OSError, json.JSONDecodeError):
            baseline_comparison = None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in preds) + "\n", encoding="utf-8")
    report = {
        "model": {"name": "LearnableNoisyORGraph", "edge_gate": "sigmoid(w0 + w1*untrusted + w2*unexpected)",
                   "propagation": "explicit message/delegation edges only; noisy-OR for up to |V| rounds",
                   "risk_head": "sigmoid(b0 + b_path*path_evidence + b_local*same_agent_evidence)",
                   "interpretation": "top paths rank observed evidence; no causal-discovery claim"},
        "dataset": {"input": str(args.input), "n": len(eps), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
                    "label_source": "independent structural reachability rule; no terminal risk event in input"},
        "random_split": {"train": len(train_idx), "test": len(test_idx), "seed": args.seed,
                          "metrics": metric, "optimizer": {"success": bool(result.success), "message": result.message,
                                                              "iterations": int(result.nit), "train_loss": float(result.fun)},
                          "parameters": {"edge_bias": float(theta[0]), "untrusted_weight": float(theta[1]),
                                         "unexpected_weight": float(theta[2]), "risk_bias": float(theta[3]),
                                         "path_weight": float(theta[4]), "local_weight": float(theta[5])}},
        "ablation_audit": ablation,
        "mas_baseline_comparison": baseline_comparison,
        "leave_one_family_out": {"held_out_family": hold_family, "train": len(tr2), "test": len(te2),
                                  "metrics": metric_loo, "optimizer": {"success": bool(res_loo.success), "iterations": int(res_loo.nit)},
                                  "parameters": [float(x) for x in theta_loo]},
        "comparison_context": "Compare with compositional_mas_v1 and mas_baselines_v1; all thresholds fit on train only."
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Learnable Graph Monitor v1", "", "该模型学习显式消息/委派边的门控参数，再对不可信输入做可解释 noisy-OR 传播。训练不读取 label_evidence 或终态违规字段。", "",
          "## Random stratified split", "", "| method | F1 | precision | recall | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    if baseline_comparison:
        for name, bm in baseline_comparison.items():
            auc = "n/a" if bm.get("auroc") is None else f"{bm['auroc']:.3f}"
            md.append(f"| {name} (existing MAS baseline) | {bm['f1']:.3f} | {bm['precision']:.3f} | {bm['recall']:.3f} | {auc} | {bm['auprc']:.3f} |")
    m = metric; md.append(f"| LearnableNoisyORGraph (ours) | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} |")
    md += ["", "## Strict leave-one-family-out", f"Held out `{hold_family}`; train={len(tr2)}, test={len(te2)}.", "", "| method | F1 | precision | recall | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    m = metric_loo; md.append(f"| LearnableNoisyORGraph | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} |")
    md += ["", "## Learned parameters", "- `untrusted_weight` controls whether an explicit edge carries untrusted data.", "- `unexpected_weight` captures unexpected direct cross-agent edges.", "- Risk is explained by propagated state and top-ranked observed paths.", "- Temporal order is metadata only; no causal claim is made.", "", "## Caveat", "该数据集是机制验证用的可控合成基准；结果不能直接宣称真实部署 SOTA。"]
    md += ["", "## Edge-ablation audit", f"- Removing an untrusted edge: mean score drop={ablation['untrusted_edge_removal']['mean_delta']}", f"- Removing a clean edge: mean score drop={ablation['clean_edge_removal']['mean_delta']}", "- This is a sensitivity check for explanation faithfulness, not a causal intervention claim."]
    (args.out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
