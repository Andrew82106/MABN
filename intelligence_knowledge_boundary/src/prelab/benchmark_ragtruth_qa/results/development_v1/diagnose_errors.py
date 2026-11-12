"""Post hoc error description of frozen calibration predictions; no fitting."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "data"
DEST = OUT / "error_diagnosis"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confusion(y, pred, indices):
    ix = np.asarray(indices, dtype=int)
    yy, pp = y[ix], pred[ix]
    tp, fp = int(np.sum(yy & pp)), int(np.sum(~yy & pp))
    fn, tn = int(np.sum(yy & ~pp)), int(np.sum(~yy & ~pp))
    return {"n": len(ix), "positive": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def bucket_span(n):
    return "01-04" if n <= 4 else "05-16" if n <= 16 else "17-64" if n <= 64 else "65+"


def bucket_answer(n):
    return "001-128" if n <= 128 else "129-256" if n <= 256 else "257-512" if n <= 512 else "513+"


def main():
    complete, summary = read(OUT / "complete.json"), read(OUT / "summary.json")
    assert complete["status"] == "complete_development_only" and not summary["test_opened"]
    for filename, expected in complete["files_sha256"].items():
        assert sha(OUT / filename) == expected, filename
    answers = readl(DATA / "answers_calibration.jsonl")
    tokens = {a["response_id"]: a for a in readl(DATA / "tokens_calibration.jsonl")}
    windows = readl(DATA / "windows_k4_calibration.jsonl")
    source = {a["response_id"]: a for a in readl(DATA / "calibration.jsonl")}
    index = read(OUT / "score_index.json")
    selected_positions = [i for i, w in enumerate(index["windows"]) if w["partition"] == "calibration"]
    assert [index["windows"][i]["window_id"] for i in selected_positions] == [w["window_id"] for w in windows]
    assert len(answers) == 159 and len(windows) == 42241
    aa = {a["response_id"]: a for a in answers}
    by_answer = defaultdict(list)
    spaninfo = {}
    for a in answers:
        rid = a["response_id"]
        t = tokens[rid]
        assert t["original_labels"] == a["original_labels"] == source[rid]["labels"]
        for mapped in t["span_token_mapping"]:
            j = mapped["span_index"]
            lab = a["original_labels"][j]
            assert a["original_response"][lab["start"]:lab["end"]] == lab["text"]
            spaninfo[(rid, j)] = {"response_id": rid, "span_index": j, "label": lab,
                "risk_tokens": set(mapped["risk_token_indices"]),
                "length_bucket": bucket_span(len(mapped["risk_token_indices"]))}
    strata = {k: defaultdict(list) for k in ("answer_length_raw_bpe", "risk_token_count", "positive_span_type", "positive_span_length", "nearest_span_type", "nearest_span_length", "answer_risk")}
    facts, span_windows = [], defaultdict(list)
    for i, w in enumerate(windows):
        rid, ix = w["response_id"], w["token_indices"]
        t, a = tokens[rid], aa[rid]
        assert ix == list(range(w["token_start"], w["token_end"])) and len(ix) <= 4
        risk_ix = [j for j in ix if t["risk_mask"][j]]
        assert risk_ix == w["risk_token_indices"] and bool(risk_ix) == bool(w["label"])
        relevant = [(key, sp) for key, sp in spaninfo.items() if key[0] == rid]
        overlaps = [(key, sp) for key, sp in relevant if sp["risk_tokens"].intersection(ix)]
        for key, _ in overlaps:
            span_windows[key].append(i)
        nearest, distance = None, None
        if relevant:
            distance, _, nearest = min((min(abs(j - r) for j in ix for r in sp["risk_tokens"]), key[1], sp) for key, sp in relevant)
        inner_single_span = any(all(lab["start"] <= t["response_token_offsets"][j][0] and t["response_token_offsets"][j][1] <= lab["end"] for j in ix) for lab in a["original_labels"])
        fact = {"window_id": w["window_id"], "response_id": rid,
                "raw_tokens": len(ix), "risk_tokens": len(risk_ix),
                "all_four_raw_tokens_risk": len(ix) == len(risk_ix) == 4,
                "all_lexical_tokens_risk": all(t["risk_mask"][j] for j in ix if t["lexical_mask"][j]),
                "inside_one_span_char_bounds": inner_single_span,
                "nearest_risk_token_distance_raw_bpe": distance,
                "overlap_span_indices": [key[1] for key, _ in overlaps],
                "nearest_span_index": nearest["span_index"] if nearest else None}
        facts.append(fact)
        by_answer[rid].append(i)
        strata["answer_length_raw_bpe"][bucket_answer(t["token_count"])].append(i)
        strata["risk_token_count"][str(len(risk_ix))].append(i)
        strata["answer_risk"][str(a["label"])].append(i)
        for typ in {sp["label"]["label_type"] for _, sp in overlaps}:
            strata["positive_span_type"][typ].append(i)
        for typ in {sp["length_bucket"] for _, sp in overlaps}:
            strata["positive_span_length"][typ].append(i)
        strata["nearest_span_type"][nearest["label"]["label_type"] if nearest else "NO_GOLD_SPAN_IN_ANSWER"].append(i)
        strata["nearest_span_length"][nearest["length_bucket"] if nearest else "NO_GOLD_SPAN_IN_ANSWER"].append(i)
    y = np.asarray([w["label"] for w in windows], dtype=bool)
    assert y.sum() == 5984 and len(spaninfo) == 200
    results, examples = {}, []
    for method, chosen in summary["selected"].items():
        with np.load(OUT / (chosen["candidate"] + "_scores.npz")) as z:
            scores = z["window_scores"][selected_positions]
        threshold = chosen["thresholds"]["window"]["threshold"]
        assert np.isfinite(scores).all()
        pred = scores >= threshold
        total = confusion(y, pred, range(len(y)))
        for key in ("n", "positive", "tp", "fp", "fn", "tn", "f1"):
            assert abs(total[key] - chosen["metrics"]["calibration"]["windows"][key]) < 1e-12, (method, key)
        fn_indices = np.flatnonzero(y & ~pred).tolist()
        fp_indices = np.flatnonzero(~y & pred).tolist()
        fn_full = [i for i in fn_indices if facts[i]["all_four_raw_tokens_risk"]]
        full_indices = [i for i, f in enumerate(facts) if f["all_four_raw_tokens_risk"]]
        inner_indices = [i for i, f in enumerate(facts) if f["inside_one_span_char_bounds"] and y[i]]
        fp_distance = Counter()
        for i in fp_indices:
            distance = facts[i]["nearest_risk_token_distance_raw_bpe"]
            category = "answer_has_no_gold_span" if distance is None else "within_1" if distance <= 1 else "2_to_4" if distance <= 4 else "5_to_8" if distance <= 8 else "farther_than_8"
            fp_distance[category] += 1
        spans_report = []
        for key, sp in spaninfo.items():
            ix = span_windows[key]
            hits = sum(bool(pred[j]) for j in ix)
            spans_report.append({"response_id": key[0], "span_index": key[1], "label_type": sp["label"]["label_type"],
                "risk_tokens": len(sp["risk_tokens"]), "char_length": sp["label"]["end"] - sp["label"]["start"],
                "positive_windows_overlapping_span": len(ix), "detected_windows": hits,
                "any_overlap_hit": bool(hits), "all_overlap_windows_hit": hits == len(ix)})
        result = {"candidate": chosen["candidate"], "threshold": threshold, "overall": total,
            "strata": {family: {k: confusion(y, pred, ix) for k, ix in sorted(groups.items())} for family, groups in strata.items()},
            "strict_all_four_risk_windows": confusion(y, pred, full_indices),
            "strict_all_four_risk_fn_fraction_of_all_fn": len(fn_full) / len(fn_indices),
            "all_four_inside_one_official_span": confusion(y, pred, inner_indices),
            "fp_distance": dict(fp_distance),
            "fp_within_4_of_risk_fraction": sum(fp_distance[k] for k in ("within_1", "2_to_4")) / len(fp_indices),
            "span_any_window_hit": sum(r["any_overlap_hit"] for r in spans_report),
            "span_count": len(spans_report), "spans": spans_report,
            "answers": [{"response_id": rid, "raw_token_count": tokens[rid]["token_count"], "answer_risk": aa[rid]["label"], **confusion(y, pred, ix)} for rid, ix in by_answer.items()]}
        results[method] = result
        # Stable ID-order examples, never chosen by scores or narrative appeal.
        groups = {"four_risk_tokens_but_fn": fn_full,
                  "fp_in_human_negative_answer": [i for i in fp_indices if not aa[facts[i]["response_id"]]["label"]],
                  "fp_far_from_risk_span": [i for i in fp_indices if (facts[i]["nearest_risk_token_distance_raw_bpe"] or 0) > 8]}
        for kind, ix in groups.items():
            for i in sorted(ix, key=lambda j: facts[j]["window_id"])[:3]:
                f, w = facts[i], windows[i]
                rid = f["response_id"]
                examples.append({"method": method, "kind": kind, **f, "score": float(scores[i]), "threshold": threshold,
                    "text": w["bounding_text"], "question": source[rid]["question"],
                    "retrieved_passages": source[rid]["retrieved_passages"], "original_response": aa[rid]["original_response"],
                    "original_labels": aa[rid]["original_labels"]})
    footprint = [DATA / n for n in ("gold_manifest.json", "answers_calibration.jsonl", "tokens_calibration.jsonl", "windows_k4_calibration.jsonl", "calibration.jsonl")]
    footprint += [OUT / "complete.json", OUT / "summary.json", OUT / "score_index.json"]
    output = {"scope": "Post hoc calibration error description; thresholds and labels unchanged; no test opened or model fitted",
              "calibration_was_used_for_model_selection": True,
              "boundary_nearness_definition": "For gold-negative windows in risk answers, minimum raw BPE index distance to any gold-risk lexical token; near=distance<=4. No gold span in answer is a separate class, never near.",
              "span_strata_definition": "Positive-span type/length tables use actual intersected span(s); multiple distinct classes can overlap and are not additive. Negative windows have no true risk type: nearest-span tables provide descriptive proximity attribution only, ties lower original span index.",
              "span_length_definition": "Number of risk lexical BPE tokens in the unchanged official span; bins1-4,5-16,17-64,65+. Raw answer length bins<=128,129-256,257-512,>512.",
              "strict_interior_definition": "All4 actual raw BPE tokens are gold-risk lexical tokens. Separate character-bound interior allows punctuation but requires entire window inside one official span.",
              "limitations": "Function words may be human-labeled risk when whole claims are spanned. Interior FN proves these misses are not only exact boundary displacement, but does not establish a causal mechanism or inability to understand every semantic fact. Any-window span hit is loose overlap, not full recovery.",
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in footprint},
              "official_span_types": dict(Counter(sp["label"]["label_type"] for sp in spaninfo.values())),
              "methods": results}
    DEST.mkdir(exist_ok=True)
    (DEST / "diagnosis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DEST / "examples.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in examples), encoding="utf-8")
    lines = ["固定校准预测的事后错误诊断：不训练，不改变金标或阈值，未读取测试集。校准集参与过选参，因此不能当独立测试成绩。", "",
        "| 方法 | 窗口F1 | 总FN | 4个词元全是风险仍FN | 该类FN/全FN | 该类窗口漏报率 | 总FP | 距风险≤4词元的FP | 无风险答案中的FP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in results.items():
        full, allm = r["strict_all_four_risk_windows"], r["overall"]
        lines.append(f"| {method} | {allm['f1']:.4f} | {allm['fn']} | {full['fn']}/{full['n']} | {r['strict_all_four_risk_fn_fraction_of_all_fn']:.1%} | {1-full['recall']:.1%} | {allm['fp']} | {r['fp_within_4_of_risk_fraction']:.1%} | {r['fp_distance'].get('answer_has_no_gold_span',0)} |")
    lines.extend(["", "4个词元全为风险的窗口仍被漏掉，说明问题不能只归结为边缘画得不准。另一方面，人类标签可能覆盖整段无依据陈述，其中也包括功能词；这些统计不能单独证明某个具体机制失败。", "",
        "| 官方span类型 | span数 | LB风险窗口召回 | 顺序风险窗口召回 | LB至少命中一窗的span |",
        "|---|---:|---:|---:|---:|"])
    base, slots = results["lookback_mean"], results["layerband_slots"]
    for typ, count in output["official_span_types"].items():
        hit = sum(sp["label_type"] == typ and sp["any_overlap_hit"] for sp in base["spans"])
        rb = base["strata"]["positive_span_type"][typ]["recall"]
        rs = slots["strata"]["positive_span_type"][typ]["recall"]
        lines.append(f"| {typ} | {count} | {rb:.1%} | {rs:.1%} | {hit}/{count} |")
    lines.extend(["", "明确冲突（Evident Conflict）是突出短板：LB仅识别180/997个风险窗口，43个原始span中22个连一个相交窗口都未命中。顺序模型也没有解决；它主要改善部分无依据陈述的召回。Subtle Conflict仅5段、来自3份答案，不宜下稳定结论。", "",
        "| span风险BPE长度 | LB阳性窗口数 | LB漏报数 | LB召回 | 顺序召回 |",
        "|---|---:|---:|---:|---:|"])
    for name, counts in base["strata"]["positive_span_length"].items():
        rs = slots["strata"]["positive_span_length"][name]["recall"]
        lines.append(f"| {name} | {counts['positive']} | {counts['fn']} | {counts['recall']:.1%} | {rs:.1%} |")
    lines.extend(["", "| 答案raw BPE长度 | LB窗口数 | LB FP | LB FN | LB窗口F1 | 顺序窗口F1 |",
        "|---|---:|---:|---:|---:|---:|"])
    for name, counts in base["strata"]["answer_length_raw_bpe"].items():
        sf = slots["strata"]["answer_length_raw_bpe"][name]["f1"]
        lines.append(f"| {name} | {counts['n']} | {counts['fp']} | {counts['fn']} | {counts['f1']:.4f} | {sf:.4f} |")
    lines.extend(["", "较短风险span更易漏掉；但较长答案的窗口F1在此校准集反而较高，不能把下降笼统归因于回答太长。这里是相关性描述，类型、跨度与答案长度之间没有被随机控制，不能据此声称因果关系。", "",
        "边界附近FP定义：窗口自身没有风险词元，但与同一答案中最近风险词元的原始BPE距离≤4；也保存≤1、5–8和>8的分组。完全无风险答案中的FP单列，不能解释为风险边界偏移。", "",
        "官方类型、风险span长度、答案长度的完整TP/FP/FN/TN在diagnosis.json。阳性窗口按真正相交span分类；负窗口没有真实错误类型，单独按最近span进行描述归属，不能当作该错误类型的真标签。跨多个类型的阳性窗口可重复计入，类型表不简单相加。", "",
        "examples.jsonl只列固定ID顺序的少量FN/FP，包含完整当前资料、原答案、原标注和冻结分数；没有改标或删除难例。"])
    (DEST / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"overall": r["overall"], "four_risk": r["strict_all_four_risk_windows"], "fn_full_fraction": r["strict_all_four_risk_fn_fraction_of_all_fn"], "fp_distance": r["fp_distance"], "fp_near4_fraction": r["fp_within_4_of_risk_fraction"], "span_hit": r["span_any_window_hit"]} for name, r in results.items()}))


if __name__ == "__main__":
    main()
