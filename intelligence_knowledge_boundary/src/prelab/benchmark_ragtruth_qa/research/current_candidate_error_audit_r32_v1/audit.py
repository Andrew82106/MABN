"""Read-only error audit for the frozen current QA development candidate.

This script only opens the original fit/calibration artifacts.  It never opens
the sealed official test and never fits, selects, or rewrites a model.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import bisect
import json
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q  # noqa: E402


OUT = HERE
CANDIDATE = "semantic_claim__old_tree__large_weight0.4"
CURRENT_META = ROOT / "results/large_fixed_convex_v1" / f"{CANDIDATE}.json"
CURRENT_SCORES = ROOT / "results/large_fixed_convex_v1" / f"{CANDIDATE}_scores.npz"
OLD_META = ROOT / "results/completed_score_combiner_v1/semantic_claim.json"
OLD_SCORES = ROOT / "results/completed_score_combiner_v1/semantic_claim_scores.npz"
LARGE = ROOT / "results/large_matched_combination_v2/large_window_probability.npy"
CITATION = ROOT / "results/citation_alignment_v1/window_features.npy"
NLI = ROOT / "results/nli_local_signal_cuda_scoring_v1/window_nli_features.npy"
BASE_MATRIX = ROOT / "results/development_v1/matrices/base.npy"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def divide(a: int | float, b: int | float) -> float | None:
    return float(a / b) if b else None


def confusion(indices, y, pred, score):
    ix = np.asarray(indices, dtype=np.int64)
    yy, pp, ss = y[ix], pred[ix], score[ix]
    tp = int(np.count_nonzero((yy == 1) & pp))
    fp = int(np.count_nonzero((yy == 0) & pp))
    fn = int(np.count_nonzero((yy == 1) & ~pp))
    tn = int(np.count_nonzero((yy == 0) & ~pp))
    return {
        "n": int(len(ix)), "positive": int(yy.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "predicted_positive_rate": divide(tp + fp, len(ix)),
        "mean_score": float(ss.mean()) if len(ss) else None,
    }


def mask_confusion(mask, y, pred, score):
    return confusion(np.flatnonzero(mask), y, pred, score)


def span_bin(n: int) -> str:
    if n <= 4:
        return "01-04"
    if n <= 12:
        return "05-12"
    if n <= 32:
        return "13-32"
    return "33+"


def pos5(value: float) -> str:
    j = min(4, int(value * 5))
    return ("00-20%", "20-40%", "40-60%", "60-80%", "80-100%")[j]


def pos3(value: float) -> str:
    if value < .2:
        return "sentence_first_20%"
    if value < .8:
        return "sentence_middle_60%"
    return "sentence_last_20%"


def distance_bin(n: int) -> str:
    if n <= 3:
        return "01-03_raw_tokens"
    if n <= 8:
        return "04-08_raw_tokens"
    if n <= 32:
        return "09-32_raw_tokens"
    return "33+_raw_tokens"


def sentence_intervals(text: str):
    # Boundaries are deterministic punctuation/newline endpoints.  This is a
    # diagnostic geometry, not a linguistic sentence parser or model feature.
    ends = [m.end() for m in re.finditer(r"(?:[.!?]+(?:\s+|$)|\n+)", text)]
    bounds = [0] + sorted(set(e for e in ends if 0 < e < len(text))) + [len(text)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def signal_audit(indices, y, signals):
    ix = np.asarray(indices, dtype=np.int64)
    yy = y[ix]
    assert set(yy.tolist()) == {0, 1}
    result = {}
    for name, values in signals.items():
        vv = np.asarray(values)[ix]
        auc = float(roc_auc_score(yy, vv))
        result[name] = {
            "risk_AUROC_prespecified_direction": auc,
            "positive_mean": float(vv[yy == 1].mean()),
            "negative_mean": float(vv[yy == 0].mean()),
            "mean_gap_positive_minus_negative": float(vv[yy == 1].mean() - vv[yy == 0].mean()),
        }
    return result


def concentration(values, total):
    counts = sorted(values, reverse=True)
    out = {"nonzero_answers": len(counts), "max_in_one_answer": counts[0] if counts else 0}
    for frac in (.5, .8):
        target = total * frac
        running = 0
        k = 0
        for v in counts:
            running += v
            k += 1
            if running >= target:
                break
        out[f"answers_covering_at_least_{int(frac * 100)}pct"] = k
    return out


def main(refresh_derived: bool = False):
    if (OUT / "complete.json").exists():
        assert refresh_derived, "Preserve an existing audit unless --refresh-derived is explicit"
        previous = q.read(OUT / "complete.json")
        assert previous["status"] == "complete"
        for key, name in (("audit_sha256", "AUDIT.json"), ("report_sha256", "REPORT.md"),
                          ("span_records_sha256", "span_records.jsonl")):
            assert q.sha(OUT / name) == previous[key], f"Refuse to refresh a modified {name}"
    meta = q.metadata()
    candidate = q.read(CURRENT_META)
    assert candidate["candidate"] == CANDIDATE
    assert candidate["scores_sha256"] == q.sha(CURRENT_SCORES)
    assert candidate["new_fits"] == 0
    assert candidate["large_weight"] == .4 and candidate["base_mode"] == "old_tree"
    with np.load(CURRENT_SCORES, allow_pickle=False) as z:
        current_all = z["window_scores"].astype(np.float64)
        current_answer = z["answer_scores"].astype(np.float64)
    with np.load(OLD_SCORES, allow_pickle=False) as z:
        old_all = z["window_scores"].astype(np.float64)
    large_all = np.load(LARGE, mmap_mode="r")
    citations_all = np.load(CITATION, mmap_mode="r")
    nli_all = np.load(NLI, mmap_mode="r")
    base_all = np.load(BASE_MATRIX, mmap_mode="r")
    assert current_all.shape == old_all.shape == large_all.shape == (210364,)
    assert citations_all.shape == (210364, 8) and nli_all.shape == (210364, 12)
    assert base_all.shape == (210364, 1025)
    assert np.max(np.abs(current_all - (.6 * old_all + .4 * large_all))) < 1e-14
    assert np.array_equal(q.answer_scores(meta, current_all), current_answer)
    assert q.metrics(meta, current_all, candidate["thresholds"]) == candidate["metrics"]

    lo, hi = meta["bounds"]["calibration"]
    assert (lo, hi) == (168123, 210364)
    windows = meta["windows"][lo:hi]
    answers = [a for a in meta["answers"] if a["partition"] == "calibration"]
    assert len(windows) == 42241 and len(answers) == 159
    y = np.asarray([w["label"] for w in windows], dtype=np.int8)
    score = current_all[lo:hi]
    threshold = candidate["thresholds"]["window"]["threshold"]
    pred = score >= threshold
    old = old_all[lo:hi]
    large = np.asarray(large_all[lo:hi])
    citation = np.asarray(citations_all[lo:hi])
    nli = np.asarray(nli_all[lo:hi])
    nll = np.asarray(base_all[lo:hi, -1])
    nll_fit = np.asarray(base_all[:lo, -1])
    assert int(y.sum()) == 5984 and int(np.count_nonzero(pred & (y == 1))) == 3839

    local_by_global = {g: g - lo for g in range(lo, hi)}
    answer_local_windows = {}
    answer_by_id = {a["response_id"]: a for a in answers}
    for a in answers:
        ids = [local_by_global[g] for g in meta["answer_windows"][a["response_id"]]]
        answer_local_windows[a["response_id"]] = ids
    token_by_id = {t["response_id"]: t for t in meta["tokens"] if t["partition"] == "calibration"}

    # Window position within answer and within deterministic sentence geometry.
    answer_position = defaultdict(list)
    sentence_position = defaultdict(list)
    for rid, ids in answer_local_windows.items():
        answer = answer_by_id[rid]
        text = answer["original_response"]
        segments = sentence_intervals(text)
        segment_starts = [a for a, _ in segments]
        for j in ids:
            w = windows[j]
            center_token = (w["token_start"] + w["token_end"]) / 2
            answer_position[pos5(center_token / max(1, answer["token_count"]))].append(j)
            center_char = (w["char_start"] + w["char_end"]) / 2
            k = min(len(segments) - 1, max(0, bisect.bisect_right(segment_starts, center_char) - 1))
            a, b = segments[k]
            sentence_position[pos3((center_char - a) / max(1, b - a))].append(j)

    # Gold type assignment and span-level detection.
    type_windows = defaultdict(set)
    span_records = []
    span_by_length = defaultdict(list)
    span_by_position = defaultdict(list)
    span_by_type = defaultdict(list)
    for rid, ids in answer_local_windows.items():
        answer = answer_by_id[rid]
        token = token_by_id[rid]
        mappings = token["span_token_mapping"]
        assert len(mappings) == len(answer["original_labels"])
        for mapping, label in zip(mappings, answer["original_labels"]):
            risk = set(mapping["risk_token_indices"])
            assert risk
            touching = [j for j in ids if risk.intersection(windows[j]["token_indices"])]
            assert touching
            for j in touching:
                type_windows[label["label_type"]].add(j)
            hits = sum(bool(pred[j]) for j in touching)
            rec = {
                "response_id": rid,
                "span_index": int(mapping["span_index"]),
                "label_type": label["label_type"],
                "risk_raw_tokens": len(risk),
                "touching_windows": len(touching),
                "detected_windows": hits,
                "window_recall": hits / len(touching),
                "any_window_detected": bool(hits),
                "fully_missed": not bool(hits),
                "mean_score": float(score[touching].mean()),
                "max_score": float(score[touching].max()),
                "answer_position": (label["start"] + label["end"]) / 2 / max(1, len(answer["original_response"])),
            }
            span_records.append(rec)
            span_by_length[span_bin(len(risk))].append(rec)
            span_by_position[pos5(rec["answer_position"])].append(rec)
            span_by_type[label["label_type"]].append(rec)

    def summarize_spans(rows):
        return {
            "spans": len(rows),
            "fully_missed": sum(r["fully_missed"] for r in rows),
            "any_hit_rate": divide(sum(r["any_window_detected"] for r in rows), len(rows)),
            "mean_window_recall": float(np.mean([r["window_recall"] for r in rows])) if rows else None,
            "mean_max_score": float(np.mean([r["max_score"] for r in rows])) if rows else None,
        }

    # Number of gold risk tokens inside each positive four-BPE window is a
    # direct boundary-density diagnostic.
    risk_density = defaultdict(list)
    for j, w in enumerate(windows):
        if not y[j]:
            continue
        token = token_by_id[w["response_id"]]
        density = sum(token["risk_mask"][k] for k in w["token_indices"])
        assert 1 <= density <= 4
        risk_density[str(density)].append(j)

    # False-positive distance from the nearest gold risk token.  This is
    # strictly post-hoc diagnosis and never a deployable feature.
    fp_distance = defaultdict(list)
    for rid, ids in answer_local_windows.items():
        token = token_by_id[rid]
        risk = np.flatnonzero(token["risk_mask"])
        for j in ids:
            if y[j]:
                continue
            if not len(risk):
                fp_distance["clean_answer"].append(j)
            else:
                raw = np.asarray(windows[j]["token_indices"])
                distance = int(np.min(np.abs(raw[:, None] - risk[None, :])))
                assert distance >= 1
                fp_distance[distance_bin(distance)].append(j)

    # Attribution/citation strata use frozen, label-free lexical features.
    citation_masks = {
        "claim_without_explicit_reference": citation[:, 4] == 0,
        "claim_with_explicit_reference": citation[:, 4] > 0,
        "direct_citation_text": citation[:, 5] > 0,
        "cited_source_coverage_gap": citation[:, 2] > 1e-8,
        "any_source_coverage_below_0.20": citation[:, 0] < .20,
        "any_source_coverage_0.20_to_0.40": (citation[:, 0] >= .20) & (citation[:, 0] < .40),
        "any_source_coverage_at_least_0.40": citation[:, 0] >= .40,
    }

    type_by_source = {}
    for name, members in type_windows.items():
        ids = np.asarray(sorted(members), dtype=np.int64)
        explicit = citation[ids, 4] > 0
        gap = citation[ids, 2] > 1e-8
        type_by_source[name] = {
            "all": confusion(ids, y, pred, score),
            "with_explicit_reference": confusion(ids[explicit], y, pred, score),
            "without_explicit_reference": confusion(ids[~explicit], y, pred, score),
            "with_cited_source_coverage_gap": confusion(ids[gap], y, pred, score),
        }

    # Use fit-only, label-free generation-NLL quantiles as fixed bucket edges.
    nll_edges = np.quantile(nll_fit, [.5, .75, .9])
    nll_masks = {
        "bottom_50pct": nll <= nll_edges[0],
        "50_to_75pct": (nll > nll_edges[0]) & (nll <= nll_edges[1]),
        "75_to_90pct": (nll > nll_edges[1]) & (nll <= nll_edges[2]),
        "top_10pct": nll > nll_edges[2],
    }

    contradiction = nli[:, [2, 5, 8, 11]].max(axis=1)
    max_entailment = nli[:, [0, 3, 6, 9]].max(axis=1)
    signals = {
        "current_score": score,
        "old_semantic_tail_tree": old,
        "full_context_large": large,
        "generation_nll": nll,
        "frozen_nli_max_contradiction": contradiction,
        "one_minus_frozen_nli_max_entailment": 1 - max_entailment,
        "citation_coverage_gap": citation[:, 2],
        "one_minus_any_source_lexical_coverage": 1 - citation[:, 0],
        "old_large_absolute_disagreement": np.abs(old - large),
    }
    current_negative = np.flatnonzero(~pred)
    current_positive = np.flatnonzero(pred)

    fn_per_answer, fp_per_answer = [], []
    for ids in answer_local_windows.values():
        yy, pp = y[ids], pred[ids]
        fn_per_answer.append(int(np.count_nonzero((yy == 1) & ~pp)))
        fp_per_answer.append(int(np.count_nonzero((yy == 0) & pp)))
    fn_nonzero = [v for v in fn_per_answer if v]
    fp_nonzero = [v for v in fp_per_answer if v]

    # Arithmetic target diagnoses.  They do not assume that an oracle rescue or
    # prune is learnable.
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    target = .75
    tp_needed_fixed_fp = int(np.ceil(target * (fp + int(y.sum())) / (2 - target)))
    fp_allowed_fixed_tp = int(np.floor(2 * tp / target - 2 * tp - fn))
    ec = np.zeros(len(y), dtype=bool)
    for name, ids in type_windows.items():
        if name == "Evident Conflict":
            ec[list(ids)] = True
    ec_total, ec_hit = int(ec.sum()), int(np.count_nonzero(ec & pred))
    ec_target_hit = int(np.ceil(.75 * ec_total))
    ec_rescue = max(0, ec_target_hit - ec_hit)
    hypothetical_tp = tp + ec_rescue
    hypothetical_fn = fn - ec_rescue
    hypothetical_f1 = 2 * hypothetical_tp / (2 * hypothetical_tp + fp + hypothetical_fn)

    result = {
        "status": "complete_read_only_development_audit",
        "candidate": CANDIDATE,
        "candidate_construction": {
            "formula_verified_max_abs_error": float(np.max(np.abs(current_all - (.6 * old_all + .4 * large_all)))),
            "formula": "0.6 * frozen semantic_claim+tail2 monotone-tree probability + 0.4 * frozen full-context ModernBERT-large probability",
            "old_tree": {
                "inputs": ["semantic_claim window risk probability", "tail2 window risk probability"],
                "model": "HistGradientBoostingClassifier, 100 iterations, depth2, <=4 leaves, monotonic [1,1]",
                "training": "original 634 fit answers / 168123 windows; upstream fit predictions are in-sample",
            },
            "large": "Fully finetuned generic ModernBERT-large over passages + question + complete answer; selected epoch3",
            "final_combination_training": "none; alpha and both thresholds were selected on repeatedly viewed calibration",
        },
        "overall": candidate["metrics"]["calibration"],
        "position_in_answer": {k: confusion(v, y, pred, score) for k, v in sorted(answer_position.items())},
        "position_in_sentence": {k: confusion(v, y, pred, score) for k, v in sorted(sentence_position.items())},
        "risk_tokens_per_positive_window": {k: confusion(v, y, pred, score) for k, v in sorted(risk_density.items())},
        "human_spans": {
            "count": len(span_records),
            "by_risk_raw_token_length": {k: summarize_spans(v) for k, v in sorted(span_by_length.items())},
            "by_answer_position": {k: summarize_spans(v) for k, v in sorted(span_by_position.items())},
            "by_label_type": {k: summarize_spans(v) for k, v in sorted(span_by_type.items())},
        },
        "gold_type_window_recall": {k: confusion(sorted(v), y, pred, score) for k, v in sorted(type_windows.items())},
        "gold_type_by_source_attribution": type_by_source,
        "normal_window_distance_to_nearest_gold_risk": {k: confusion(v, y, pred, score) for k, v in sorted(fp_distance.items())},
        "citation_and_source_attribution": {k: mask_confusion(v, y, pred, score) for k, v in citation_masks.items()},
        "generation_nll_fit_only_edges": [float(v) for v in nll_edges],
        "generation_nll_buckets": {k: mask_confusion(v, y, pred, score) for k, v in nll_masks.items()},
        "conditional_rescue_ranking_within_current_negatives_FN_vs_TN": signal_audit(current_negative, y, signals),
        "conditional_prune_ranking_within_current_positives_TP_vs_FP": signal_audit(current_positive, y, signals),
        "error_concentration": {
            "false_negative_windows": concentration(fn_nonzero, fn),
            "false_positive_windows": concentration(fp_nonzero, fp),
        },
        "target_arithmetic": {
            "target_window_f1": target,
            "current_tp_fp_fn": [tp, fp, fn],
            "with_current_fp_unchanged_minimum_tp": tp_needed_fixed_fp,
            "additional_tp_needed_if_fp_unchanged": tp_needed_fixed_fp - tp,
            "with_current_tp_unchanged_maximum_fp": fp_allowed_fixed_tp,
            "fp_to_remove_if_tp_unchanged": fp - fp_allowed_fixed_tp,
            "evident_conflict_current_hit_total": [ec_hit, ec_total],
            "if_EC_recall_reached_0.75_with_no_new_FP_additional_TP": ec_rescue,
            "resulting_window_f1": hypothetical_f1,
            "warning": "Gold-conditioned arithmetic only; not a model result.",
        },
        "limits": [
            "Post-selection diagnosis on repeatedly viewed calibration159; no independent-test claim.",
            "Gold span/type/distance buckets are analysis only and cannot be model inputs at deployment.",
            "Citation coverage is lexical overlap, not entailment or proof of source correctness.",
            "Sentence geometry is deterministic punctuation/newline splitting, not a linguistic parser.",
            "Conditional AUROC describes ranking inside the frozen model's errors; it is not an achieved F1 or a trained gate.",
        ],
        "new_fits": 0,
        "new_thresholds": 0,
        "GPU_used": False,
        "official_test_opened": False,
        "source_sha256": {str(p.resolve()): q.sha(p) for p in (
            Path(__file__), CURRENT_META, CURRENT_SCORES, OLD_META, OLD_SCORES,
            LARGE, CITATION, NLI, BASE_MATRIX, ROOT / "data/gold_manifest.json")},
    }
    write_json(OUT / "AUDIT.json", result)
    with (OUT / "span_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in span_records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def pct(v):
        return "N/A" if v is None else f"{100*v:.2f}%"

    lines = [
        "# 当前公共QA候选R32误差审计", "",
        "只读分析原cal159/42241窗；未拟合、未调阈值、未用GPU、未打开官方test。", "",
        "## 最关键结果", "",
        f"- 当前窗口F1 **{candidate['metrics']['calibration']['windows']['f1']:.6f}**：TP/FP/FN={tp}/{fp}/{fn}。FP不变时至少再补{tp_needed_fixed_fp-tp}个TP才到.75；TP不变时则需删掉{fp-fp_allowed_fixed_tp}个FP。",
        f"- Evident Conflict仅检出{ec_hit}/{ec_total}（{pct(ec_hit/ec_total)}）。若只把该类召回提到75%、且不新增FP，算术F1为{hypothetical_f1:.6f}；这是gold oracle，不是模型成绩。",
        f"- 200个人工span中完全漏检{sum(r['fully_missed'] for r in span_records)}个。长度≤4 raw BPE的span命中率为{pct(summarize_spans(span_by_length['01-04'])['any_hit_rate'])}，33+为{pct(summarize_spans(span_by_length['33+'])['any_hit_rate'])}。",
        f"- 答案首20%位置的风险窗召回仅{pct(confusion(answer_position['00-20%'], y, pred, score)['recall'])}，60–80%位置为{pct(confusion(answer_position['60-80%'], y, pred, score)['recall'])}。",
        f"- 当前候选fit窗口F1 {candidate['metrics']['fit']['windows']['f1']:.6f}，cal降至{candidate['metrics']['calibration']['windows']['f1']:.6f}；旧树用的是上游模型在同一fit上的in-sample分数。",
        "", "## 风险窗内部密度", "", "| 4-BPE窗内风险词元数 | 正窗 | 召回 |", "|---:|---:|---:|",
    ]
    for k, ids in sorted(risk_density.items()):
        m = confusion(ids, y, pred, score)
        lines.append(f"| {k} | {m['positive']} | {pct(m['recall'])} |")
    lines += ["", "## 正常窗距离真实风险的距离", "", "| 区域 | 正常窗 | 误报 | 误报率 |", "|---|---:|---:|---:|"]
    for k, ids in sorted(fp_distance.items()):
        m = confusion(ids, y, pred, score)
        lines.append(f"| {k} | {m['n']} | {m['fp']} | {pct(m['false_positive_rate'])} |")
    lines += ["", "## 人工错误类型与来源归属", "", "| 类型 | 风险窗 | 总召回 | 显式引用窗召回 | 无引用窗召回 | 引用来源覆盖缺口窗召回 | span数 | span完全漏检 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(type_windows):
        m = confusion(sorted(type_windows[k]), y, pred, score)
        sm = summarize_spans(span_by_type[k])
        ts = type_by_source[k]
        lines.append(f"| {k} | {m['positive']} | {pct(m['recall'])} | {pct(ts['with_explicit_reference']['recall'])} ({ts['with_explicit_reference']['positive']}) | {pct(ts['without_explicit_reference']['recall'])} ({ts['without_explicit_reference']['positive']}) | {pct(ts['with_cited_source_coverage_gap']['recall'])} ({ts['with_cited_source_coverage_gap']['positive']}) | {sm['spans']} | {sm['fully_missed']} |")
    rescue = result["conditional_rescue_ranking_within_current_negatives_FN_vs_TN"]
    prune = result["conditional_prune_ranking_within_current_positives_TP_vs_FP"]
    lines += ["", "## 冻结信号在现有错误区内的排序", "", "AUROC越高，越可能把FN从TN中补出，或把TP从FP中保住；没有训练新门。", "", "| 信号 | FN vs TN（补漏） | TP vs FP（去误报） |", "|---|---:|---:|"]
    for name in signals:
        lines.append(f"| {name} | {rescue[name]['risk_AUROC_prespecified_direction']:.4f} | {prune[name]['risk_AUROC_prespecified_direction']:.4f} |")
    lines += [
        "", "## 结论与下一结构", "",
        "1. **优先做关系/冲突专头。** 当前显性无依据召回较高，而显性冲突极低；下一头应保留主体、动作、条件、数量、顺序和来源编号，并用只改一个关系槽的最小对比样本训练。输出仍映射回原4-BPE窗。",
        "2. **检测与边界分两阶段。** 先按claim判断事实是否有风险，再用逐词元序列头细化范围。建议用多视角attention特征+小Transformer+部分标注CRF；短span与边界窗不能继续只靠同一个独立窗口阈值。",
        "3. **来源条件化，而非继续做总覆盖率。** 为每个claim保留三份passage的独立支持/冲突分数和引用作用域，再由门控头决定是来源归属错、关系错或无依据。现有词面覆盖只适合作控制信号。",
        "", "这些结构均属于我们的方法升级，不改Lookback、LUMINA、GHOST等正式baseline。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "complete.json", {
        "status": "complete", "audit_sha256": q.sha(OUT / "AUDIT.json"),
        "report_sha256": q.sha(OUT / "REPORT.md"), "span_records_sha256": q.sha(OUT / "span_records.jsonl"),
        "new_fits": 0, "new_thresholds": 0, "GPU_used": False, "official_test_opened": False,
    })
    print("R32_CURRENT_CANDIDATE_READ_ONLY_AUDIT_COMPLETE", len(span_records), flush=True)


if __name__ == "__main__":
    main(refresh_derived="--refresh-derived" in sys.argv[1:])
