"""Strict prefix/early-warning evaluation on the canonical A2ASecBench adapter.

Every score at fraction f is computed from events whose sequence is <= the
prefix cut only.  The episode-level observed_violation is a *post-hoc* outcome
label, not an event-time label; therefore these numbers are an early-warning
sanity check and not causal or deployment evidence.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
EP_PATH = ROOT / 'data/processed/mas_adapter/a2asecbench/episodes.jsonl'
EV_PATH = ROOT / 'data/processed/mas_adapter/a2asecbench/events.jsonl'
SPLIT_PATH = ROOT / 'data/processed/mas_adapter/a2asecbench/splits_v1.json'
OUT = ROOT / 'experiments/a2asecbench/runs/a2a_prefix_v1'
PREFIXES = (0.25, 0.50, 0.75, 1.00)
LOCAL = ("anomaly_path", "permission_mismatch", "tool_side_effect", "untrusted_input", "memory_poison")
GLOBAL = ("cross_agent_propagation", "risk_accumulation", "delegation_edge", "cycle_detected", "max_depth_exceeded", "task_store_pressure", "half_open")
OBSERVABLE = ("resource_ref_present", "artifact_present", "task_request_present", "delegation_depth", "repeated_edge")

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

def nb_fit(xs: Sequence[Tuple[dict, int]], keys: Sequence[str], alpha: float = 1.0):
    prior = (sum(y for _, y in xs) + alpha) / (len(xs) + 2 * alpha)
    cond = {}
    for k in keys:
        cond[k] = {}
        for y in (0, 1):
            n = sum(yy == y for _, yy in xs)
            one = sum(int(bool(x.get(k, 0))) for x, yy in xs if yy == y)
            cond[k][y] = (one + alpha) / (n + 2 * alpha)
    return prior, cond

def nb_predict(model, x: dict, keys: Sequence[str]) -> float:
    prior, cond = model; logs = []
    for y in (0, 1):
        z = math.log(max(prior if y else 1-prior, 1e-12))
        for k in keys:
            q = cond[k][y]; z += math.log(max(q if x.get(k, 0) else 1-q, 1e-12))
        logs.append(z)
    return sigmoid(logs[1] - logs[0])

def event_seq(e: dict) -> float:
    c = e.get("context") or {}
    try: return float(c.get("sequence", e.get("source", {}).get("event_index", 0)))
    except (TypeError, ValueError): return 0.0

def load(input_path: Path) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    ep_path = input_path / "episodes.jsonl" if input_path.is_dir() else input_path
    event_path = ep_path.parent / "events.jsonl"
    eps = {}
    for line in ep_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("observed_violation") is not None: eps[r["episode_id"]] = r
    events = defaultdict(list)
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line); eid = e.get("episode_id")
                if eid in eps: events[eid].append(e)
    for eid in events: events[eid].sort(key=event_seq)
    return eps, events

def prefix_vector(evs: List[dict], frac: float) -> dict:
    if not evs: return {k: 0 for k in LOCAL + GLOBAL + OBSERVABLE} | {"event_count": 0, "agent_count": 0, "edge_count": 0}
    cut = max(1, math.ceil(len(evs) * frac)); pre = evs[:cut]
    x = {k: 0 for k in LOCAL + GLOBAL + OBSERVABLE}
    for e in pre:
        for k in LOCAL:
            x[k] = max(x[k], int(bool((e.get("features") or {}).get(k, 0))))
        typ = e.get("event_type", "")
        ef = e.get("features") or {}
        for k in ("cross_agent_propagation", "risk_accumulation", "half_open"):
            x[k] = max(x[k], int(bool(ef.get(k, 0))))
        if typ == "delegation_edge": x["delegation_edge"] = 1
        if typ == "cycle_detected": x["cycle_detected"] = 1
        if typ == "max_depth_exceeded": x["max_depth_exceeded"] = 1
        if typ == "task_store_pressure": x["task_store_pressure"] = 1
        evidence = ((e.get("context") or {}).get("evidence") or {})
        x["resource_ref_present"] = max(x["resource_ref_present"], int(evidence.get("resource_ref_count", 0) > 0))
        x["artifact_present"] = max(x["artifact_present"], int(evidence.get("artifact_count", 0) > 0))
        x["task_request_present"] = max(x["task_request_present"], int(evidence.get("task_request_count", 0) > 0))
    agents = {e.get("agent_id") for e in pre if e.get("agent_id")}
    edges = set()
    for e in pre:
        s, t = e.get("message_source"), e.get("message_target")
        if s and t and s != t: edges.add((s, t))
        for p in e.get("parent") or []:
            if isinstance(p, dict) and p.get("source") and p.get("target"): edges.add((p["source"], p["target"]))
    edge_counts = defaultdict(int)
    for e in pre:
        s, t = e.get("message_source"), e.get("message_target")
        if s and t and s != t: edge_counts[(s, t)] += 1
        for p in e.get("parent") or []:
            if isinstance(p, dict) and p.get("source") and p.get("target"):
                edge_counts[(p["source"], p["target"])] += 1
    x["repeated_edge"] = int(any(n > 1 for n in edge_counts.values()))
    adj = defaultdict(set)
    for s, t in edges: adj[s].add(t)
    memo = {}
    def depth(node, active=()):
        if node in memo: return memo[node]
        if node in active: return 0
        memo[node] = 1 + max((depth(n, active + (node,)) for n in adj.get(node, ())), default=0)
        return memo[node]
    x["delegation_depth"] = max((depth(n) for n in agents), default=0)
    x["event_count"] = len(pre); x["agent_count"] = len(agents); x["edge_count"] = len(edges)
    return x

def count_score(x: dict) -> float: return min(1.0, math.log1p(x["event_count"]) / 5.0)
def rule_score(x: dict) -> float:
    # transparent dynamic/global evidence, with no terminal outcome fields
    return min(1.0, .22*x["cross_agent_propagation"] + .18*x["delegation_edge"] + .18*x["permission_mismatch"] + .16*x["tool_side_effect"] + .12*x["cycle_detected"] + .08*x["max_depth_exceeded"] + .06*x["task_store_pressure"])

def metrics(scores: Sequence[float], ys: Sequence[int], t: float) -> dict:
    p = [int(s >= t) for s in scores]; tp=sum(a==b==1 for a,b in zip(p,ys)); tn=sum(a==b==0 for a,b in zip(p,ys)); fp=sum(a==1 and b==0 for a,b in zip(p,ys)); fn=sum(a==0 and b==1 for a,b in zip(p,ys))
    prec=tp/(tp+fp) if tp+fp else 0.; rec=tp/(tp+fn) if tp+fn else 0.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.; spec=tn/(tn+fp) if tn+fp else 0.
    pos=[s for s,y in zip(scores,ys) if y]; neg=[s for s,y in zip(scores,ys) if not y]
    auc=sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None
    return {"threshold":t,"tp":tp,"tn":tn,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"specificity":spec,"f1":f1,"roc_auc":auc}

def best_threshold(scores: Sequence[float], ys: Sequence[int]) -> float:
    best=(.5,-1.)
    for t in sorted({0.,.5,1.,*scores}):
        m=metrics(scores,ys,t)
        if m["f1"]>best[1]: best=(t,m["f1"])
    return best[0]

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate strict A2A prefix early-warning models.")
    ap.add_argument("--input", type=Path, default=EP_PATH, help="episodes.jsonl file or adapter directory")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = ap.parse_args()
    input_path = args.input
    eps, events = load(input_path)
    local_split = (input_path / "splits_v1.json" if input_path.is_dir() else input_path.parent / "splits_v1.json")
    split_path = local_split if local_split.exists() else SPLIT_PATH
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    # Fresh API-run IDs do not match the canonical split. Reconstruct the
    # leave-one-attack-family-out protocol from the rows in that case.
    if not any(set(spec.get("train", [])) & set(eps) and set(spec.get("test", [])) & set(eps) for spec in splits.values()):
        families = sorted({r.get("attack_id", "unknown") for r in eps.values()})
        splits = {
            f"leave_{fam}_out": {
                "train": [eid for eid, r in eps.items() if r.get("attack_id", "unknown") != fam],
                "test": [eid for eid, r in eps.items() if r.get("attack_id", "unknown") == fam],
            }
            for fam in families
        }
    out={}
    for frac in PREFIXES:
        fold_out={}; allp=defaultdict(list); ally=[]
        for fold,spec in splits.items():
            tr=[eps[i] for i in spec["train"] if i in eps]; te=[eps[i] for i in spec["test"] if i in eps]
            if not tr or not te: continue
            def xy(r): return prefix_vector(events.get(r["episode_id"],[]),frac),int(bool(r["observed_violation"]))
            trxy=[xy(r) for r in tr]; texy=[xy(r) for r in te]; ys_tr=[y for _,y in trxy]; ys_te=[y for _,y in texy]
            observable_keys = LOCAL + OBSERVABLE
            dynamic_keys = GLOBAL + OBSERVABLE
            local_model=nb_fit(trxy,observable_keys); global_model=nb_fit(trxy,dynamic_keys); all_model=nb_fit(trxy,observable_keys+GLOBAL)
            c_tr=[count_score(x) for x,_ in trxy]; c_te=[count_score(x) for x,_ in texy]; l_tr=[nb_predict(local_model,x,observable_keys) for x,_ in trxy]; l_te=[nb_predict(local_model,x,observable_keys) for x,_ in texy]; g_tr=[nb_predict(global_model,x,dynamic_keys) for x,_ in trxy]; g_te=[nb_predict(global_model,x,dynamic_keys) for x,_ in texy]
            # Fusion alpha is selected on training only; dynamic/global and local BN are both inspectable.
            alpha = min(
                (i / 20 for i in range(21)),
                key=lambda z: sum((z * a + (1 - z) * b - y) ** 2 for a, b, y in zip(l_tr, g_tr, ys_tr)),
            )
            f_te=[alpha*a+(1-alpha)*b for a,b in zip(l_te,g_te)]; names={"event_count":(c_tr,c_te),"local_nb":(l_tr,l_te),"dynamic_global_nb":(g_tr,g_te),"fusion":([alpha*a+(1-alpha)*b for a,b in zip(l_tr,g_tr)],f_te)}
            fold_out[fold]={"train":len(trxy),"test":len(texy),"fusion_alpha":alpha,"metrics":{n:metrics(b,ys_te,best_threshold(a,ys_tr) if n!="event_count" else .5) for n,(a,b) in names.items()}}
            for n,(a,b) in names.items(): allp[n].extend(b)
            ally.extend(ys_te)
        # Aggregate uses a fixed operational threshold (0.5), never tuned on
        # the held-out predictions. Per-fold thresholds are reported above.
        out["prefix_"+str(int(frac*100))]={"fraction":frac,"folds":fold_out,"aggregate":{n:metrics(s,ally,.5) for n,s in allp.items()}}
    payload={"dataset":"A2ASecBench canonical adapter","evaluated":len(eps),"prefixes":list(PREFIXES),"split":"splits_v1.json leave-one-attack-family-out","metrics":out,"caveat":"Scores use only prefix runtime events. observed_violation is a post-hoc benchmark outcome label; results are an early-warning sanity check, not causal discovery or independent deployment evidence."}
    output_path = args.out
    output_path.mkdir(parents=True,exist_ok=True); (output_path/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# A2ASecBench strict prefix / early-warning evaluation v1","","每个 prefix 只使用该比例之前的事件；标签是事后 observed_violation，仅用于提前预警 sanity check。","","| prefix | event-count F1 | local NB F1 | dynamic/global NB F1 | fusion F1 |","|---:|---:|---:|---:|---:|"]
    for k,v in out.items():
        a=v["aggregate"]; lines.append(f"| {v['fraction']:.2f} | {a['event_count']['f1']:.3f} | {a['local_nb']['f1']:.3f} | {a['dynamic_global_nb']['f1']:.3f} | {a['fusion']['f1']:.3f} |")
    lines += ["","说明：local NB 只看局部事件特征；dynamic/global NB 看跨 Agent、委派、循环、深度和任务存储等动态图证据；fusion 在训练折内选择权重。没有使用 episode 的终态字段作为输入。","","限制：A2ASecBench 的事件与标签来自同一离线 reference SUT，仍可能存在定义耦合；AS 配置无运行时标签，未进入本表。"]
    (output_path/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v["aggregate"] for k,v in out.items()},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
