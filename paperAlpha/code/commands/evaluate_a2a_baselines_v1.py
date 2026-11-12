"""First reproducible baseline comparison on the canonical A2ASecBench adapter.

This is an adapter-level benchmark, not a claim of causal discovery.  Labels in
the offline benchmark are explicit runtime violation outcomes; therefore all
scores must be interpreted as a sanity check until independently collected MAS
traces are available.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def resolve_input(path: Path) -> Path:
    """Resolve either an episodes.jsonl file or an adapter directory."""
    return path / "episodes.jsonl" if path.is_dir() else path


def load_rows(input_path: Path) -> Tuple[List[dict], Dict[str, dict]]:
    ep_path = resolve_input(input_path)
    rows = [json.loads(x) for x in ep_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    events_path = ep_path.parent / "events.jsonl"
    by: Dict[str, List[dict]] = defaultdict(list)
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                by[e["episode_id"]].append(e)
    graph = {}
    for eid, evs in by.items():
        agents = {e.get("agent_id") for e in evs if e.get("agent_id")}
        edges = set()
        for e in evs:
            s, t = e.get("message_source"), e.get("message_target")
            if s and t and s != t:
                edges.add((s, t))
            for p in e.get("parent") or []:
                if isinstance(p, dict):
                    ps, pt = p.get("source"), p.get("target")
                    if ps and pt and ps != pt:
                        edges.add((ps, pt))
        graph[eid] = {
            "agent_count": len(agents),
            "edge_count": len(edges),
            "event_count": len(evs),
            "cross_agent_edges": int(len(edges) > 0),
            "multi_agent": int(len(agents) > 1),
        }
    return rows, graph


LOCAL = ["anomaly_path", "permission_mismatch", "tool_side_effect", "untrusted_input", "memory_poison"]
GLOBAL = ["cross_agent_propagation", "risk_accumulation", "multi_agent", "cross_agent_edges"]


def vector(row: dict, graph: dict) -> Dict[str, int]:
    f = row.get("features") or {}
    g = graph.get(row["episode_id"], {})
    out = {k: int(bool(f.get(k, 0))) for k in LOCAL + ["cross_agent_propagation", "risk_accumulation"]}
    out.update({k: int(bool(g.get(k, 0))) for k in ["multi_agent", "cross_agent_edges"]})
    out["has_events"] = int(row.get("event_count", 0) > 0)
    out["high_event_count"] = int(row.get("event_count", 0) >= 30)
    return out


def nb_fit(train: Sequence[Tuple[dict, int]], keys: Iterable[str], alpha: float = 1.0):
    keys = list(keys)
    prior = (sum(y for _, y in train) + alpha) / (len(train) + 2 * alpha)
    cond = {}
    for k in keys:
        cond[k] = {}
        for y in (0, 1):
            n = sum(1 for _, yy in train if yy == y)
            one = sum(x[k] for x, yy in train if yy == y)
            cond[k][y] = (one + alpha) / (n + 2 * alpha)
    return prior, cond


def nb_predict(model, xs: Sequence[Dict[str, int]], keys: Iterable[str]) -> List[float]:
    prior, cond = model
    keys = list(keys)
    out = []
    for x in xs:
        logs = []
        for y in (0, 1):
            p = prior if y else 1 - prior
            z = math.log(max(p, 1e-12))
            for k in keys:
                q = cond[k][y]
                z += math.log(max(q if x[k] else 1 - q, 1e-12))
            logs.append(z)
        m = max(logs)
        out.append(math.exp(logs[1] - m) / (math.exp(logs[0] - m) + math.exp(logs[1] - m)))
    return out


def rule_score(x: Dict[str, int], row: dict) -> float:
    # Transparent Layer-2 evidence score: topology is only additional evidence.
    return min(1.0, 0.35 * x["anomaly_path"] + 0.25 * x["permission_mismatch"]
               + 0.20 * x["tool_side_effect"] + 0.10 * x["cross_agent_propagation"]
               + 0.10 * x["risk_accumulation"])


def best_threshold(scores: Sequence[float], ys: Sequence[int]) -> float:
    cands = sorted({0.0, 0.5, 1.0, *scores})
    best = (0.5, -1.0)
    for t in cands:
        pred = [s >= t for s in scores]
        tp = sum(a and b for a, b in zip(pred, ys)); fp = sum(a and not b for a, b in zip(pred, ys)); fn = sum((not a) and b for a, b in zip(pred, ys))
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best[1]: best = (t, f)
    return best[0]


def metrics(scores: Sequence[float], ys: Sequence[int], threshold: float) -> dict:
    pred = [int(s >= threshold) for s in scores]
    tp = sum(a == b == 1 for a, b in zip(pred, ys)); tn = sum(a == b == 0 for a, b in zip(pred, ys))
    fp = sum(a == 1 and b == 0 for a, b in zip(pred, ys)); fn = sum(a == 0 and b == 1 for a, b in zip(pred, ys))
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    # Rank AUC (ties receive half credit), implemented without sklearn dependency.
    pos = [s for s, y in zip(scores, ys) if y]; neg = [s for s, y in zip(scores, ys) if not y]
    auc = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else None
    return {"threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": p, "recall": r, "specificity": spec, "f1": f1, "roc_auc": auc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / 'data/processed/mas_adapter/a2asecbench', help="episodes.jsonl file or adapter directory")
    ap.add_argument("--out", type=Path, default=Path('experiments/a2asecbench/runs/a2a_baselines_v1'), help="output directory")
    args = ap.parse_args()
    rows, graph = load_rows(args.input)
    total_rows = len(rows)
    rows = [r for r in rows if r.get("observed_violation") is not None]
    # Case-level split stratified by attack family: held-out case IDs avoid
    # duplicate-trial leakage while keeping every family in train and test.
    fam_cases = defaultdict(list)
    for r in rows:
        # CTRL_001 and EVAL_001 are a matched control/attack pair.
        suffix = r["case_id"].rsplit("_", 1)[-1]
        fam_cases[r["attack_id"]].append(suffix)
    train_cases = set()
    for fam, cs in fam_cases.items():
        cs = sorted(set(cs)); train_cases.update(cs[:max(1, int(len(cs) * 0.7))])
    tr, te = [], []
    for r in rows:
        item = (vector(r, graph), int(bool(r["observed_violation"])))
        suffix = r["case_id"].rsplit("_", 1)[-1]
        (tr if suffix in train_cases else te).append(item)
    ys_tr, ys_te = [y for _, y in tr], [y for _, y in te]
    models = {}
    # count baseline (fixed family-agnostic event-count evidence)
    # preserve item order directly to avoid case lookup ambiguity
    count_tr = [min(1.0, math.log1p(r.get("event_count", 0)) / 5) for r in rows if r["case_id"].rsplit("_", 1)[-1] in train_cases]
    count_te = [min(1.0, math.log1p(r.get("event_count", 0)) / 5) for r in rows if r["case_id"].rsplit("_", 1)[-1] not in train_cases]
    models["event_count"] = (count_tr, count_te, best_threshold(count_tr, ys_tr))
    # Naive event-presence baseline, included to expose how much information is
    # already encoded by the adapter's event projection.
    any_tr = [float(x["has_events"]) for x, _ in tr]; any_te = [float(x["has_events"]) for x, _ in te]
    models["any_event"] = (any_tr, any_te, 0.5)
    # local-only NB approximates a single-agent monitor: no topology/global evidence.
    local_keys = LOCAL + ["has_events", "high_event_count"]
    local_model = nb_fit(tr, local_keys); models["single_agent_nb"] = (nb_predict(local_model, [x for x, _ in tr], local_keys), nb_predict(local_model, [x for x, _ in te], local_keys), None)
    # Diagnostic ablation: remove the three flags most directly derived from
    # explicit violation events. This is not a replacement model; it exposes
    # the amount of target/evidence coupling in the offline adapter.
    ablation_keys = [k for k in local_keys if k not in {"anomaly_path", "permission_mismatch", "tool_side_effect"}]
    ablation_model = nb_fit(tr, ablation_keys); models["single_agent_nb_no_direct_violation_flags"] = (nb_predict(ablation_model, [x for x, _ in tr], ablation_keys), nb_predict(ablation_model, [x for x, _ in te], ablation_keys), None)
    all_keys = local_keys + GLOBAL
    hier_model = nb_fit(tr, all_keys); models["hierarchical_bn"] = (nb_predict(hier_model, [x for x, _ in tr], all_keys), nb_predict(hier_model, [x for x, _ in te], all_keys), None)
    rule_tr = [rule_score(x, {}) for x, _ in tr]; rule_te = [rule_score(x, {}) for x, _ in te]
    models["layer2_dynamic_rule"] = (rule_tr, rule_te, None)
    results = {}
    for name, (s_tr, s_te, fixed) in models.items():
        threshold = fixed if fixed is not None else best_threshold(s_tr, ys_tr)
        results[name] = metrics(s_te, ys_te, threshold)
    # Leave-one-family-out audit (the stricter generalisation view).
    loo = {}
    families = sorted({r["attack_id"] for r in rows})
    for fam in families:
        a, b = [], []
        for r in rows:
            item = (vector(r, graph), int(bool(r["observed_violation"])))
            (b if r["attack_id"] == fam else a).append(item)
        ya, yb = [y for _, y in a], [y for _, y in b]
        fam_out = {}
        for name in ["single_agent_nb", "hierarchical_bn", "layer2_dynamic_rule"]:
            keys = local_keys if name == "single_agent_nb" else all_keys if name == "hierarchical_bn" else None
            if keys is None:
                sa = [rule_score(x, {}) for x, _ in a]; sb = [rule_score(x, {}) for x, _ in b]
            else:
                mdl = nb_fit(a, keys); sa = nb_predict(mdl, [x for x, _ in a], keys); sb = nb_predict(mdl, [x for x, _ in b], keys)
            t = best_threshold(sa, ya); fam_out[name] = metrics(sb, yb, t)
        loo[fam] = fam_out
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "A2ASecBench adapter", "evaluated": len(rows), "train": len(tr), "test": len(te), "train_cases": sorted(train_cases), "excluded_null_labels": total_rows - len(rows), "baselines": results, "leave_one_family_out": loo, "caveat": "Offline reference SUT labels are explicit runtime outcomes; this is an adapter-level sanity comparison, not independent evidence of causal discovery or deployment performance."}
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# A2ASecBench baseline comparison v1", "", f"按 case_id 做 70/30 划分，避免同一 case 的 trial 泄漏。本次集合共 {len(rows)} 个有明确 observed_violation 标签的 episode；无标签样本数为 {total_rows - len(rows)}。", "", "| baseline | precision | recall | specificity | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for n, m in results.items(): lines.append(f"| {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |" if m['roc_auc'] is not None else f"| {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | {m['f1']:.3f} | n/a |")
    lines += ["", "模型说明：event_count 只看事件量；single_agent_nb 只使用节点/事件局部证据；layer2_dynamic_rule 使用透明动态图证据；hierarchical_bn 将局部模板证据与跨 Agent/边证据用带 Laplace 平滑的可解释 Naive-Bayes 结构融合。另有 leave-one-family-out 审计结果写入 metrics.json。", "", "限制：A2ASecBench 当前适配的是离线 reference SUT，标签与适配器中的显式运行时证据存在定义上的耦合。因此这些数字用于检查管线和比较基线，不应直接写成真实部署性能或因果发现结果。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
