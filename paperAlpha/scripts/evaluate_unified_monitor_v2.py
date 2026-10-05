"""Unified two-layer MAS monitor (controlled compositional benchmark).

Layer 1 estimates interpretable per-agent local risk from event semantics.
Layer 2 learns gates on explicit runtime message/delegation edges and
propagates local risk across the observed graph.  Labels are used only in the
training objective; label_evidence and terminal fields are never features.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PRIV_TOOLS = {"payments.write", "filesystem.write", "shell.exec"}


def read_eps(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def encode(ep, treat_unknown_as_untrusted=False):
    names = sorted(set(e["actor"] for e in ep.get("events", [])) | {e["target"] for e in ep.get("events", []) if e.get("target")})
    idx = {a: i for i, a in enumerate(names)}
    # local features: source, privileged sink, sanitization, unexpected input,
    # outgoing untrusted, incoming untrusted.
    X = np.zeros((len(names), 6), dtype=float)
    edges = []
    for e in ep.get("events", []):
        a = e.get("actor"); i = idx.get(a)
        if i is None: continue
        if e.get("event_type") == "input_received" and e.get("data_class") == "untrusted": X[i, 0] = 1
        if e.get("event_type") == "tool_call" and e.get("tool") in PRIV_TOOLS: X[i, 1] = 1
        if e.get("event_type") == "sanitization": X[i, 2] = 1
        if e.get("event_type") in {"message_sent", "delegation"} and e.get("edge_kind") == "unexpected": X[i, 3] = 1
        if e.get("event_type") in {"message_sent", "delegation"} and e.get("data_class") == "untrusted": X[i, 4] = 1
        if e.get("event_type") in {"message_sent", "delegation"} and e.get("target") in idx:
            unknown = e.get("data_class") in {"unknown", "redacted"} or e.get("edge_kind") in {"redacted", "unknown"}
            untrusted = e.get("data_class") == "untrusted" or (treat_unknown_as_untrusted and unknown)
            if untrusted: X[idx[e["target"]], 5] = 1
            conf = e.get("edge_confidence", 1.0)
            conf = 1.0 if treat_unknown_as_untrusted and unknown else (0.0 if unknown else conf)
            edges.append((i, idx[e["target"]], np.array([1., float(untrusted), float(e.get("edge_kind") == "unexpected"), float(conf)]), {"source": a, "target": e["target"], "data_class": e.get("data_class"), "edge_kind": e.get("edge_kind"), "sequence": e.get("sequence"), "edge_confidence": e.get("edge_confidence")}))
    return names, X, edges


def forward(theta, ep, explain=False, unknown_edges=False):
    names, X, edges = encode(ep, treat_unknown_as_untrusted=unknown_edges)
    # Layer 1: local BN-style logit and noisy-OR conversion.
    local = sigmoid(theta[:7][0] + X @ theta[:7][1:])
    # Layer 2: semantic edge gates, with optional unknown edges treated as
    # possible rather than clean for an upper risk bound.
    ew = theta[7:11]
    gates = [float(sigmoid(f @ ew)) for _, _, f, _ in edges]
    state = local.copy()
    for _ in range(max(1, len(names))):
        nxt = state.copy()
        incoming = [[] for _ in names]
        for (u, v, _, info), g in zip(edges, gates): incoming[v].append((u, g))
        for v in range(len(names)):
            p = 1.0
            for u, g in incoming[v]: p *= 1 - state[u] * g
            nxt[v] = 1 - (1 - nxt[v]) * p
        state = nxt
    sinks = np.where(X[:, 1] > 0)[0]
    raw_path = float(max((state[i] for i in sinks), default=0.0))
    path = raw_path ** (1.0 / max(1, len(names) - 1))
    local_max = float(max(local, default=0.0))
    risk = float(sigmoid(theta[10] + theta[11] * path + theta[12] * local_max))
    if not explain: return risk
    # Unknown-edge upper bound: add a conservative complete directed graph
    # between agents, but do not call it recovered evidence.
    upper = risk
    if explain and not unknown_edges:
        upper = float(forward(theta, ep, False, True))
    return risk, {"agents": names, "local_risk": {n: float(local[i]) for i, n in enumerate(names)}, "propagated_risk": {n: float(state[i]) for i, n in enumerate(names)}, "path_evidence": path, "top_edges": [info for _, _, _, info in sorted(edges, key=lambda z: z[2][1], reverse=True)[:5]], "upper_risk_if_unknown_edges": upper}


def loss(theta, eps, y):
    p = np.asarray([forward(theta, e) for e in eps])
    return float(-np.mean(y*np.log(np.clip(p,1e-7,1-1e-7)) + (1-y)*np.log(np.clip(1-p,1e-7,1-1e-7))) + 1e-3*np.sum(theta*theta))


def metric(y, p, thr):
    pred = np.asarray(p) >= thr
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if np.any(mask): ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"f1": float(f1_score(y,pred,zero_division=0)), "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)), "auroc": float(roc_auc_score(y,p)) if len(np.unique(p))>1 else None, "auprc": float(average_precision_score(y,p)), "brier": float(np.mean((p-y)**2)), "ece_10bin": float(ece), "threshold": float(thr)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=Path("paperAlpha/results/compositional_mas_v1/episodes.jsonl")); ap.add_argument("--labels", type=Path, default=None, help="optional separate episode_id/label file"); ap.add_argument("--out", type=Path, default=Path("paperAlpha/results/unified_monitor_v2")); ap.add_argument("--seed", type=int, default=20260914); ap.add_argument("--limit", type=int, default=0, help="optional deterministic prefix of episodes for quick audits"); ap.add_argument("--group-split", action="store_true", help="hold out complete scenario_family groups"); args = ap.parse_args()
    eps = read_eps(args.input)
    if args.labels:
        label_rows = read_eps(args.labels); labels = {r["episode_id"]: r["label"] for r in label_rows}; groups_meta = {r["episode_id"]: r.get("scenario_family") for r in label_rows}
        for e in eps:
            if e["episode_id"] not in labels: raise ValueError(f"missing label for {e['episode_id']}")
            e["label"] = labels[e["episode_id"]]
            if groups_meta.get(e["episode_id"]) is not None: e["scenario_family"] = groups_meta[e["episode_id"]]
    eps = eps[:args.limit] if args.limit else eps; y = np.asarray([int(e["label"]) for e in eps]); idx = np.arange(len(eps))
    if args.group_split and eps and any("scenario_family" in e for e in eps):
        groups = sorted({e.get("scenario_family", "unknown") for e in eps}); hold = set(groups[-max(1, int(len(groups)*.2)):]); te = np.asarray([i for i,e in enumerate(eps) if e.get("scenario_family", "unknown") in hold]); tr = np.asarray([i for i in idx if i not in set(te)])
    else:
        tr, te = train_test_split(idx,test_size=.3,random_state=args.seed,stratify=y)
    rng=np.random.default_rng(args.seed); theta0=np.zeros(14); theta0[:7]=rng.normal(0,.1,7); theta0[7:11]=np.array([-1.,2.,.5,.5]); theta0[11:]=[-2.,4.,.5]
    fit=minimize(loss,theta0,args=([eps[i] for i in tr],y[tr]),method="L-BFGS-B",bounds=[(-8,8)]*14,options={"maxiter":400,"ftol":1e-10}); theta=fit.x
    p_tr=np.asarray([forward(theta,eps[i]) for i in tr]); p_te=np.asarray([forward(theta,eps[i]) for i in te]); vals=np.unique(np.r_[0.,p_tr,1.]); thr=float(max(vals,key=lambda t:f1_score(y[tr],p_tr>=t,zero_division=0)))
    preds=[]
    for i,p in zip(te,p_te):
        _,ex=forward(theta,eps[i],True); preds.append({"episode_id":eps[i]["episode_id"],"label":int(y[i]),"risk_probability":float(p),"explanation":ex})
    metrics=metric(y[te],p_te,thr)
    # Prefix behavior and uncertainty audit: remove suffix events.
    prefix=[]
    for frac in (.25,.5,.75):
        pp=[]; yy=[]
        for i in te:
            ev=eps[i].get("events",[]); cut=max(1,int(len(ev)*frac)); ep=dict(eps[i]); ep["events"]=ev[:cut]; pp.append(forward(theta,ep)); yy.append(y[i])
        prefix.append({"fraction":frac,**metric(np.asarray(yy),np.asarray(pp),thr)})
    args.out.mkdir(parents=True,exist_ok=True); (args.out/"predictions.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True) for x in preds)+"\n",encoding="utf-8")
    label_source = ("hidden external unauthorized privileged side effect (labels supplied separately)" if args.labels else (eps[0].get("label_source", "unspecified") if eps else "unspecified"))
    independent = "external" in label_source.lower() or "hidden" in label_source.lower()
    limitations = (["This benchmark uses a hidden external-effect label and noisy runtime observations; it is closer to an independent monitor test, but still synthetic."] if independent else ["Labels are generated by a reachability rule, so high scores test mechanism consistency rather than real-world SOTA."])
    limitations += ["Unknown/redacted edges are reported through a conservative upper bound; missing API events are not reconstructed.","Explanation is evidence attribution, not causal discovery."]
    report={"model":{"name":"UnifiedLocalBNGraphMonitor-v2","layer1":"per-agent local BN-style sigmoid risk from six event-semantic features","layer2":"learned semantic edge gates + noisy-OR message passing; confidence-aware and length-normalized","risk_head":"sigmoid(system_bias + path_weight*normalized_sink_path + local_weight*max_local)","inputs_exclude":["label","label_evidence","label_rule","external_effect","hidden_fields","terminal risk/violation"]},"dataset":{"n":len(eps),"positive":int(y.sum()),"negative":int(len(y)-y.sum()),"label_source":label_source},"split":{"train":len(tr),"test":len(te),"seed":args.seed,"group_split":bool(args.group_split)},"metrics":metrics,"prefix_metrics":prefix,"fit":{"success":bool(fit.success),"iterations":int(fit.nit),"train_loss":float(fit.fun),"parameters":[float(x) for x in theta]},"limitations":limitations}
    (args.out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    md=["# Unified MAS Monitor v2","", "两层端到端监测器：第一层对每个 Agent 输出局部风险，第二层在运行时显式通信图上传播风险，再输出系统风险与解释。训练不读取标签证据字段。", "", f"- n={len(eps)}, positive={int(y.sum())}, test={len(te)}", "", "| metric | value |", "|---|---:|"]
    for k in ("f1","precision","recall","auroc","auprc","brier","ece_10bin"): md.append(f"| {k} | {metrics[k]:.3f} |")
    md += ["", "## Prefix", "", "| fraction | F1 | AUROC |", "|---:|---:|---:|"]+[f"| {r['fraction']:.2f} | {r['f1']:.3f} | {r['auroc']:.3f} |" for r in prefix]+["", "## Interpretation", "- Layer 1 is readable because every local feature maps to a named node variable.", "- Layer 2 only propagates along observed message/delegation edges; it does not claim to discover causality."]
    md.append("- This hidden-effect benchmark is still synthetic; external execution traces and human labels are required before deployment claims." if independent else "- A perfect result on the controlled benchmark is only mechanism validation and must be followed by independent human-labeled traces.")
    (args.out/"REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
