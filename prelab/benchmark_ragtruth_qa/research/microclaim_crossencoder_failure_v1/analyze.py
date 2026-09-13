"""CPU-only stratified failure analysis for microclaim_crossencoder_v1."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q
import run_microclaim_crossencoder_v1 as method


def save(path, value):
    path = Path(path); assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=bool); score = np.asarray(score, dtype=np.float64); pred = score >= threshold
    tp = int(np.count_nonzero(y & pred)); fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred)); tn = int(np.count_nonzero(~y & ~pred))
    return {"n": len(y), "positive": int(y.sum()), "negative": int((~y).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mean_score": float(score.mean()) if len(score) else None}


def stratify(values, order, y, score, threshold):
    values = np.asarray(values, dtype=object); result = {}
    for name in order:
        active = values == name
        if active.any(): result[name] = metrics(y[active], score[active], threshold)
    assert sum(row["n"] for row in result.values()) == len(values)
    return result


def word_bin(n):
    if n <= 8: return "01_1-8"
    if n <= 16: return "02_9-16"
    if n <= 24: return "03_17-24"
    return "04_25+"


def input_bin(n):
    if n <= 256: return "01_<=256"
    if n <= 384: return "02_257-384"
    if n <= 512: return "03_385-512"
    return "04_>512"


def coverage_bin(x):
    if x == 0: return "01_zero"
    if x <= .25: return "02_(0,.25]"
    if x <= .50: return "03_(.25,.50]"
    return "04_>.50"


def entailment_bin(x):
    if x < .25: return "01_<.25"
    if x < .50: return "02_[.25,.50)"
    if x < .75: return "03_[.50,.75)"
    return "04_>=.75"


def source_bucket(claim):
    ids = claim["explicit_passage_ids"] or claim["parent_passage_ids"]
    ids = sorted(set(map(int, ids)))
    if not ids: return "none"
    if len(ids) > 1: return "multiple"
    return f"passage_{ids[0]}"


def claim_position(index, count):
    value = (index + .5) / count
    if value <= 1/3: return "first_third"
    if value <= 2/3: return "middle_third"
    return "last_third"


def main():
    summary = q.read(method.OUT / "summary.json")
    assert summary["status"] == "development_only_complete"
    records = q.lines(method.OUT / "records.jsonl"); atomic_rows = q.lines(method.ATOMIC_INPUT)
    with np.load(method.OUT / "scores.npz", allow_pickle=False) as z:
        scores = z["main_claim_scores"].astype(np.float64)
    arrays = method.load_arrays(); labels = arrays["labels"].astype(np.int8)
    feature_names = q.read(method.ATOMIC / "feature_names.json")["evidence"]
    evidence_features = np.load(method.ATOMIC / "evidence_features.npy", mmap_mode="r")
    assert len(records) == len(scores) == len(labels) == evidence_features.shape[0]
    threshold = summary["trained_candidate"]["thresholds_from_fit_OOF"]["window"]["threshold"]
    cal_indices = np.flatnonzero(np.asarray([row["partition"] for row in records]) == "calibration")
    y = labels[cal_indices]; score = scores[cal_indices]
    atomic_by_id = {row["response_id"]: row for row in atomic_rows}
    answers = {row["response_id"]: row for row in q.lines(ROOT / "data/answers_calibration.jsonl")}
    token_rows = {row["response_id"]: row for row in q.lines(ROOT / "data/tokens_calibration.jsonl")}
    name_to_feature = {name: i for i, name in enumerate(feature_names)}
    global_entail = np.asarray(evidence_features[:, name_to_feature["global_max_entailment"]])[cal_indices]
    passage_entail = np.column_stack([
        np.asarray(evidence_features[:, name_to_feature[f"passage_{pid}_individual_max_entailment"]])[cal_indices]
        for pid in (1, 2, 3)])

    word_bins = []; input_bins = []; coverage_bins = []; entail_bins = []; support_proxy = []
    answer_kind = []; source_locations = []; response_positions = []; best_passages = []
    label_types = defaultdict(list); risk_fraction = []
    for local, global_index in enumerate(cal_indices):
        record = records[global_index]; answer = atomic_by_id[record["response_id"]]
        claim = answer["claims"][record["claim_id"]]
        assert claim["microclaim_id"] == record["microclaim_id"]
        word_bins.append(word_bin(int(claim["word_count"])))
        input_bins.append(input_bin(int(record["input_tokens"])))
        coverage = max(float(source["selected_union_query_coverage"]) for source in claim["retrieval"])
        coverage_bins.append(coverage_bin(coverage)); entail_bins.append(entailment_bin(global_entail[local]))
        support_proxy.append("yes_entail>=.5" if global_entail[local] >= .5 else "no_entail<.5")
        answer_kind.append("risky_answer" if answers[record["response_id"]]["label"] else "clean_answer")
        source_locations.append(source_bucket(claim))
        response_positions.append(claim_position(record["claim_id"], len(answer["claims"])))
        best_passages.append(f"passage_{int(np.argmax(passage_entail[local])) + 1}")
        token_risk = np.asarray(token_rows[record["response_id"]]["risk_mask"], dtype=bool)
        owned = claim["lexical_token_indices"]
        risk_fraction.append(float(token_risk[owned].mean()))
        overlaps = []
        for annotation in answers[record["response_id"]]["original_labels"]:
            if max(int(claim["start"]), int(annotation["start"])) < min(int(claim["end"]), int(annotation["end"])):
                overlaps.append(annotation["label_type"])
        for kind in set(overlaps): label_types[kind].append(local)

    # Replay the risk-fraction calculation independently before accepting it.
    replay_fraction = []
    for global_index in cal_indices:
        record = records[global_index]; claim = atomic_by_id[record["response_id"]]["claims"][record["claim_id"]]
        risk = np.asarray(token_rows[record["response_id"]]["risk_mask"], dtype=bool)
        replay_fraction.append(float(risk[claim["lexical_token_indices"]].mean()))
    assert np.array_equal(np.asarray(risk_fraction), np.asarray(replay_fraction))

    result = {
        "status": "complete_CPU_only_development_failure_analysis",
        "strict_claim_threshold_inherited_from_fit_OOF_window": threshold,
        "overall_claims": metrics(y, score, threshold),
        "strata": {
            "claim_word_length": stratify(word_bins, ["01_1-8", "02_9-16", "03_17-24", "04_25+"], y, score, threshold),
            "full_evidence_input_tokens": stratify(input_bins, ["01_<=256", "02_257-384", "03_385-512", "04_>512"], y, score, threshold),
            "top2_lexical_query_coverage": stratify(coverage_bins, ["01_zero", "02_(0,.25]", "03_(.25,.50]", "04_>.50"], y, score, threshold),
            "top2_frozen_NLI_entailment": stratify(entail_bins, ["01_<.25", "02_[.25,.50)", "03_[.50,.75)", "04_>=.75"], y, score, threshold),
            "top2_contains_support_sentence_proxy": stratify(support_proxy, ["no_entail<.5", "yes_entail>=.5"], y, score, threshold),
            "answer_gold_status": stratify(answer_kind, ["clean_answer", "risky_answer"], y, score, threshold),
            "claim_source_location": stratify(source_locations, ["none", "passage_1", "passage_2", "passage_3", "multiple"], y, score, threshold),
            "best_NLI_support_passage": stratify(best_passages, ["passage_1", "passage_2", "passage_3"], y, score, threshold),
            "claim_position_in_answer": stratify(response_positions, ["first_third", "middle_third", "last_third"], y, score, threshold),
        },
        "positive_claim_label_type_recall": {
            kind: {"claims": len(indices), "hits": int(np.count_nonzero(score[indices] >= threshold)),
                   "recall": float(np.mean(score[indices] >= threshold)), "mean_score": float(score[indices].mean())}
            for kind, indices in sorted(label_types.items())
        },
        "partial_positive": {
            "positive_claims": int(y.sum()),
            "partially_positive_claims": int(np.count_nonzero((np.asarray(risk_fraction) > 0) & (np.asarray(risk_fraction) < 1))),
            "fully_positive_claims": int(np.count_nonzero(np.asarray(risk_fraction) == 1)),
            "partial_recall": float(np.mean((score >= threshold)[(np.asarray(risk_fraction) > 0) & (np.asarray(risk_fraction) < 1)])),
            "full_recall": float(np.mean((score >= threshold)[np.asarray(risk_fraction) == 1])),
        },
        "support_proxy_definition": "A top-2 sentence is counted as present only by a model-derived proxy: frozen ModernBERT NLI max entailment >=0.5 over the already selected evidence pairs. RAGTruth has no human sentence-level support links, so this is not gold support coverage.",
        "expanded_top2_causal_assessment": {
            "changes_vs_v1": ["fit microclaims 9,055 -> 34,919", "full three passages -> claim-selected top-2 per passage",
                              "hard-negative redistribution -> canonical hierarchy/class weights", "fit positive rate 14.26% -> 9.95%"],
            "identifiability": "A score from expanded-v4 cannot by itself distinguish sample-size gain from evidence-packing or weighting gain.",
            "required_clean_ablation": ["native634 top-2 with v1 weights versus native634 full evidence",
                                        "expanded3680 full evidence with canonical weights versus expanded3680 top-2"],
        },
        "calibration_only_posthoc": True, "no_new_GPU": True,
        "formal_baselines_modified": False, "official_test_opened": False,
        "source_sha256": {"candidate_summary": q.sha(method.OUT / "summary.json"),
                          "candidate_scores": q.sha(method.OUT / "scores.npz"),
                          "atomic_input": q.sha(method.ATOMIC_INPUT),
                          "evidence_features": q.sha(method.ATOMIC / "evidence_features.npy"),
                          "feature_names": q.sha(method.ATOMIC / "feature_names.json")},
    }
    save(HERE / "ANALYSIS.json", result)

    s = result["strata"]; support = s["top2_contains_support_sentence_proxy"]
    lengths = s["full_evidence_input_tokens"]; answers_s = s["answer_gold_status"]
    lines = ["# Full-evidence microclaim cross-encoder failure analysis", "",
             f"严格claim阈值下：{result['overall_claims']['tp']} TP / {result['overall_claims']['fp']} FP / {result['overall_claims']['fn']} FN，召回 {result['overall_claims']['recall']:.3f}。", "",
             "## 关键分层", "",
             "| 分层 | 样本 | 正例 | 召回 | FPR |", "|---|---:|---:|---:|---:|"]
    for title, table in (("输入<=256", lengths["01_<=256"]), ("输入>512", lengths["04_>512"]),
                         ("top2无支持代理", support["no_entail<.5"]), ("top2有支持代理", support["yes_entail>=.5"]),
                         ("正常回答", answers_s["clean_answer"]), ("风险回答", answers_s["risky_answer"])):
        recall = "-" if table["recall"] is None else f"{table['recall']:.3f}"
        fpr = "-" if table["false_positive_rate"] is None else f"{table['false_positive_rate']:.3f}"
        lines.append(f"| {title} | {table['n']} | {table['positive']} | {recall} | {fpr} |")
    lines += ["", "## 判断", "",
              "expanded top-2 同时改变训练量、证据打包、权重和类别比例，因此它若变好，不能直接归因于 top-2。"
              "只有同规模、同权重的配对消融才能分开判断。RAGTruth没有人工支持句链接；文中的“支持句”是冻结NLI代理。", "",
              "本分析只使用 calibration 做事后诊断，没有新GPU、没有修改baseline、没有打开official test。", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("MICROCLAIM_CROSSENCODER_FAILURE_ANALYSIS_COMPLETE", json.dumps({
        "claim_recall": result["overall_claims"]["recall"],
        "short_recall": lengths["01_<=256"]["recall"], "long_recall": lengths["04_>512"]["recall"],
        "support_proxy_recall": support["yes_entail>=.5"]["recall"]}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
