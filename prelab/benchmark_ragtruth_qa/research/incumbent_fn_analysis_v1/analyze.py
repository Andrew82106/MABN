"""CPU-only post-hoc diagnosis of the frozen R32 incumbent.

Only the public fit/calibration development artifacts are opened.  The script
does not fit a model, select a threshold, modify a baseline, or open test data.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q  # noqa: E402
import run_atomic_microclaim_nli_v1 as atomic_nli  # noqa: E402


INCUMBENT_SCORE = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
INCUMBENT_META = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
NLI_DIR = ROOT / "results/atomic_microclaim_nli_v1"
CROSS_DIR = ROOT / "results/microclaim_crossencoder_v1"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def quantiles(values: np.ndarray):
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "q10": float(np.quantile(values, .1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, .9)),
    }


def auc(y: np.ndarray, score: np.ndarray):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def metrics(y: np.ndarray, pred: np.ndarray):
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    tn = int(np.count_nonzero(~y & ~pred))
    return {
        "n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp,
        "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": f1(tp, fp, fn) if tp + fp + fn else None,
    }


def pair_aligned_features(rows, records, evidence):
    """Keep relation slots tied to the exact highest-entailment sentence.

    This avoids the existing 315-vector's independent max pooling, under which
    the entailment maximum and a relation mismatch may come from two different
    evidence sentences.
    """
    names = list(atomic_nli.PAIR_RELATION_NAMES)
    name = {value: index for index, value in enumerate(names)}
    count = len(records)
    probability = np.zeros((count, 3), dtype=np.float32)
    relation = np.zeros((count, len(names)), dtype=np.float32)
    best_passage = np.zeros(count, dtype=np.int8)
    cited_gap = np.zeros(count, dtype=np.float32)
    cursor = 0
    for row in rows:
        with np.load(NLI_DIR / "pair_scores" / f"{row['response_id']}.npz", allow_pickle=False) as data:
            probs = data["probabilities"]
            claim_ids = data["claim_ids"]
            passage_ids = data["passage_ids"]
            kinds = data["pair_kinds"]
            sentence_a = data["sentence_a"]
        lookup = {
            (passage["passage_id"], sentence["sentence_id"]): sentence["text"]
            for passage in row["passages"] for sentence in passage["sentences"]
        }
        for claim in row["claims"]:
            cid = claim["claim_id"]
            individual = np.flatnonzero((claim_ids == cid) & (kinds > 0))
            selected = individual[int(np.argmax(probs[individual, 0]))]
            probability[cursor] = probs[selected]
            pid = int(passage_ids[selected])
            best_passage[cursor] = pid
            relation[cursor] = atomic_nli.relation_pair_features(
                claim, lookup[(pid, int(sentence_a[selected]))]
            )
            cited = set(claim["explicit_passage_ids"] or claim["parent_passage_ids"])
            if cited:
                cited_rows = individual[np.isin(passage_ids[individual], list(cited))]
                uncited_rows = individual[~np.isin(passage_ids[individual], list(cited))]
                cited_e = float(probs[cited_rows, 0].max()) if len(cited_rows) else 0.
                uncited_e = float(probs[uncited_rows, 0].max()) if len(uncited_rows) else cited_e
                cited_gap[cursor] = max(0., uncited_e - cited_e)
            cursor += 1
    assert cursor == count
    flags = {
        "subject_low_coverage": (evidence[:, 8] > 0) & (relation[:, name["subject_token_coverage"]] < .5),
        "entity_incomplete": (evidence[:, 9] > 0) & (relation[:, name["entity_token_coverage"]] < .999),
        "predicate_absent": (evidence[:, 16] > 0) & (relation[:, name["predicate_token_match"]] < .5),
        "negation_mismatch": relation[:, name["negation_mismatch"]] > 0,
        "comparator_reversal": relation[:, name["comparator_reversal"]] > 0,
        "comparator_missing": relation[:, name["comparator_missing"]] > 0,
        "quantity_different": relation[:, name["quantity_same_unit_different_value"]] > 0,
        "quantity_missing": relation[:, name["quantity_missing"]] > 0,
        "temporal_reversal": relation[:, name["temporal_reversal"]] > 0,
        "temporal_missing": relation[:, name["temporal_missing"]] > 0,
        "condition_mismatch": relation[:, name["condition_mismatch"]] > 0,
        "cited_source_gap_015": cited_gap >= .15,
    }
    flags["hard_relation_or_source"] = (
        flags["negation_mismatch"] | flags["comparator_reversal"] |
        flags["quantity_different"] | flags["temporal_reversal"] |
        flags["condition_mismatch"] | flags["cited_source_gap_015"]
    )
    return probability, relation, cited_gap, flags


def main():
    meta = q.metadata()
    records = load_jsonl(CROSS_DIR / "records.jsonl")
    atomic_rows = load_jsonl(NLI_DIR / "inputs.jsonl")
    evidence = np.load(NLI_DIR / "evidence_features.npy", mmap_mode="r")
    with np.load(NLI_DIR / "scores.npz", allow_pickle=False) as data:
        nli_claim = data["claim_scores"].copy()
        nli_window = data["window_scores"].copy()
        raw_nli_window = data["raw_window_scores"].copy()
    with np.load(CROSS_DIR / "scores.npz", allow_pickle=False) as data:
        cross_claim = data["main_claim_scores"].copy()
        cross_window = data["main_window_scores"].copy()
    with np.load(INCUMBENT_SCORE, allow_pickle=False) as data:
        incumbent = data["window_scores"].copy()
    incumbent_meta = q.read(INCUMBENT_META)
    threshold = incumbent_meta["thresholds"]["window"]["threshold"]
    windows = meta["windows"]
    y = np.asarray([row["label"] for row in windows], dtype=bool)
    pred = incumbent >= threshold
    assert evidence.shape == (11322, 315)
    assert len(records) == len(nli_claim) == len(cross_claim) == 11322
    assert len(windows) == len(incumbent) == len(nli_window) == len(cross_window) == 210364
    assert [row["response_id"] for row in atomic_rows] == [row["response_id"] for row in meta["answers"]]
    assert all(int(row["label"]) == int(value) for row, value in zip(records, np.load(NLI_DIR / "microclaim_labels.npy")))

    pair_probability, pair_relation, cited_gap, claim_flags = pair_aligned_features(
        atomic_rows, records, evidence
    )
    claim_y = np.asarray([row["label"] for row in records], dtype=bool)
    claim_partition = np.asarray([row["partition"] for row in records])
    claim_by_response = defaultdict(list)
    for index, row in enumerate(records):
        claim_by_response[row["response_id"]].append(index)

    answer_by_id = {row["response_id"]: row for row in meta["answers"]}
    token_by_id = {row["response_id"]: row for row in meta["tokens"]}
    type_by_response_token = {}
    claim_types = []
    for response_id, token in token_by_id.items():
        mapping = defaultdict(set)
        for span, label in zip(token["span_token_mapping"], answer_by_id[response_id]["original_labels"]):
            for token_index in span["risk_token_indices"]:
                mapping[int(token_index)].add(label["label_type"])
        type_by_response_token[response_id] = mapping
    for record in records:
        kinds = set()
        mapping = type_by_response_token[record["response_id"]]
        for token_index in record["lexical_token_indices"]:
            kinds.update(mapping.get(token_index, ()))
        claim_types.append(kinds)

    # Project claim-local, label-free diagnostics to the unchanged 4-BPE windows.
    projected = {
        "global_entailment": np.zeros(len(windows), dtype=np.float32),
        "pair_entailment": np.zeros(len(windows), dtype=np.float32),
        "pair_contradiction": np.zeros(len(windows), dtype=np.float32),
        "cited_gap": np.zeros(len(windows), dtype=np.float32),
    }
    projected_flags = {name: np.zeros(len(windows), dtype=bool) for name in claim_flags}
    gold_claim_oracle = np.zeros(len(windows), dtype=bool)
    window_types = []
    window_claims = []
    for window_index, window in enumerate(windows):
        token_indices = set(window["token_indices"])
        owners = [
            index for index in claim_by_response[window["response_id"]]
            if token_indices.intersection(records[index]["lexical_token_indices"])
        ]
        assert owners
        window_claims.append(owners)
        projected["global_entailment"][window_index] = float(np.max(evidence[owners, 288]))
        projected["pair_entailment"][window_index] = float(np.max(pair_probability[owners, 0]))
        projected["pair_contradiction"][window_index] = float(np.max(pair_probability[owners, 2]))
        projected["cited_gap"][window_index] = float(np.max(cited_gap[owners]))
        gold_claim_oracle[window_index] = bool(np.any(claim_y[owners]))
        for name, values in claim_flags.items():
            projected_flags[name][window_index] = bool(np.any(values[owners]))
        kinds = set()
        token_type_map = type_by_response_token[window["response_id"]]
        for token_index in window["token_indices"]:
            kinds.update(token_type_map.get(token_index, ()))
        window_types.append(kinds)

    # Exclusive type assignment permits exact totals; conflict takes priority.
    priority = ("Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info")
    # Recompute type-specific relation projections so an adjacent safe claim
    # cannot lend its high entailment or mismatch flag to a gold error span.
    type_entailment = {kind: np.zeros(len(windows), dtype=np.float32) for kind in priority}
    type_flags = {
        kind: {name: np.zeros(len(windows), dtype=bool) for name in claim_flags}
        for kind in priority
    }
    for window_index, owners in enumerate(window_claims):
        for kind in priority:
            relevant = [owner for owner in owners if kind in claim_types[owner]]
            if not relevant:
                continue
            type_entailment[kind][window_index] = float(np.max(evidence[relevant, 288]))
            for name, values in claim_flags.items():
                type_flags[kind][name][window_index] = bool(np.any(values[relevant]))
    exclusive_type = np.asarray([
        next((kind for kind in priority if kind in kinds), "none") for kinds in window_types
    ])

    lo, hi = meta["bounds"]["calibration"]
    cal = np.arange(lo, hi)
    cal_y, cal_pred = y[cal], pred[cal]
    fn_mask = cal_y & ~cal_pred
    fp_mask = ~cal_y & cal_pred

    signal_values = {
        "incumbent": incumbent[cal],
        "atomic_NLI315_readout": nli_window[cal],
        "crossencoder": cross_window[cal],
        "raw_NLI_risk": raw_nli_window[cal],
        "global_entailment": projected["global_entailment"][cal],
        "pair_contradiction": projected["pair_contradiction"][cal],
        "cited_gap": projected["cited_gap"][cal],
    }

    def cluster_summary(mask):
        return {
            "windows": int(mask.sum()),
            "signals": {name: quantiles(values[mask]) for name, values in signal_values.items()},
            "observable_flag_fraction": {
                name: float(values[cal][mask].mean()) if mask.any() else None
                for name, values in projected_flags.items()
            },
        }

    fn_clusters = {}
    for kind in priority:
        mask = fn_mask & (exclusive_type[cal] == kind)
        summary = cluster_summary(mask)
        typed_entailment = type_entailment[kind][cal]
        summary["global_entailment_bins"] = {
            "below_050": int(np.count_nonzero(mask & (typed_entailment < .5))),
            "050_to_075": int(np.count_nonzero(mask & (typed_entailment >= .5) & (typed_entailment < .75))),
            "at_least_075": int(np.count_nonzero(mask & (typed_entailment >= .75))),
        }
        high_mask = mask & (typed_entailment >= .75)
        summary["observable_flag_counts_all_FN"] = {
            name: int(np.count_nonzero(mask & type_flags[kind][name][cal]))
            for name in claim_flags
        }
        summary["observable_flag_counts_high_entailment_FN"] = {
            name: int(np.count_nonzero(high_mask & type_flags[kind][name][cal]))
            for name in claim_flags
        }
        fn_clusters[kind] = summary
    assert sum(row["windows"] for row in fn_clusters.values()) == int(fn_mask.sum())

    answer_risky = np.asarray([bool(answer_by_id[w["response_id"]]["label"]) for w in windows[lo:hi]])
    fp_cluster_masks = {
        "inside_gold_positive_microclaim": fp_mask & gold_claim_oracle[cal],
        "outside_positive_microclaim_in_risky_answer": fp_mask & ~gold_claim_oracle[cal] & answer_risky,
        "clean_answer": fp_mask & ~answer_risky,
    }
    assert sum(int(mask.sum()) for mask in fp_cluster_masks.values()) == int(fp_mask.sum())
    fp_clusters = {name: cluster_summary(mask) for name, mask in fp_cluster_masks.items()}

    # Conditional ranking is reported on fit and calibration separately.  No
    # threshold or model is selected from calibration.
    risk_signals = {
        "atomic_NLI315_readout": nli_window,
        "crossencoder": cross_window,
        "raw_NLI_risk": raw_nli_window,
        "one_minus_global_entailment": 1 - projected["global_entailment"],
        "pair_contradiction": projected["pair_contradiction"],
        "cited_gap": projected["cited_gap"],
        "entity_incomplete": projected_flags["entity_incomplete"].astype(float),
        "quantity_different": projected_flags["quantity_different"].astype(float),
        "hard_relation_or_source": projected_flags["hard_relation_or_source"].astype(float),
    }
    conditional_ranking = {}
    for partition in ("fit", "calibration"):
        left, right = meta["bounds"][partition]
        yy, pp = y[left:right], pred[left:right]
        conditional_ranking[partition] = {"rescue_FN_vs_TN": {}, "prune_TP_vs_FP": {}}
        for name, values in risk_signals.items():
            vv = values[left:right]
            conditional_ranking[partition]["rescue_FN_vs_TN"][name] = auc(yy[~pp], vv[~pp])
            conditional_ranking[partition]["prune_TP_vs_FP"][name] = auc(yy[pp], vv[pp])

    # Fixed, pre-existing fit thresholds and semantic flags.  These are checks,
    # not newly optimized combinations.
    fixed = {
        "atomic_NLI315_strict": nli_window >= .6173424472146823,
        "crossencoder_strict": cross_window >= .4897472858428955,
    }
    fixed["both_learned_detectors"] = fixed["atomic_NLI315_strict"] & fixed["crossencoder_strict"]
    fixed["cited_gap_015"] = projected_flags["cited_source_gap_015"]
    fixed["quantity_different"] = projected_flags["quantity_different"]
    fixed_trigger_checks = {}
    for name, trigger in fixed.items():
        fixed_trigger_checks[name] = {}
        for partition in ("fit", "calibration"):
            left, right = meta["bounds"][partition]
            yy, pp, tt = y[left:right], pred[left:right], trigger[left:right]
            rescue = ~pp & tt
            union = pp | tt
            fixed_trigger_checks[name][partition] = {
                "rescued_TP": int(np.count_nonzero(rescue & yy)),
                "added_FP": int(np.count_nonzero(rescue & ~yy)),
                "rescue_precision": float(np.mean(yy[rescue])) if rescue.any() else None,
                "union_metrics": metrics(yy, union),
            }

    # Claim-level high-entailment hard cases and cross-partition stability.
    high_entailment = evidence[:, 288] >= .75
    conflict_claim = np.asarray([
        bool({"Evident Conflict", "Subtle Conflict"}.intersection(kinds)) for kinds in claim_types
    ])
    claim_signals = {
        "atomic_NLI315_readout": nli_claim,
        "crossencoder": cross_claim,
        "one_minus_global_entailment": 1 - np.asarray(evidence[:, 288]),
        "global_contradiction": np.asarray(evidence[:, 294]),
    }
    high_entailment_claims = {}
    for partition in ("fit", "calibration"):
        active = claim_partition == partition
        hard = active & high_entailment
        positive = hard & claim_y
        negative = hard & ~claim_y
        high_entailment_claims[partition] = {
            "claims": int(hard.sum()),
            "positive": int(positive.sum()),
            "positive_rate": float(claim_y[hard].mean()),
            "conflict_positive": int(np.count_nonzero(positive & conflict_claim)),
            "score_separation": {
                name: {
                    "AUROC": auc(claim_y[hard], values[hard]),
                    "positive": quantiles(values[positive]),
                    "negative": quantiles(values[negative]),
                }
                for name, values in claim_signals.items()
            },
            "flag_prevalence_positive_vs_negative": {
                name: {
                    "positive": float(values[positive].mean()) if positive.any() else None,
                    "negative": float(values[negative].mean()) if negative.any() else None,
                }
                for name, values in claim_flags.items()
            },
        }

    tp = int(np.count_nonzero(cal_y & cal_pred))
    fp = int(np.count_nonzero(~cal_y & cal_pred))
    fn = int(np.count_nonzero(cal_y & ~cal_pred))
    conflict_fn = fn_mask & np.isin(exclusive_type[cal], ["Evident Conflict", "Subtle Conflict"])
    high_entail_conflict_fn = (
        (fn_mask & (exclusive_type[cal] == "Evident Conflict") & (type_entailment["Evident Conflict"][cal] >= .75)) |
        (fn_mask & (exclusive_type[cal] == "Subtle Conflict") & (type_entailment["Subtle Conflict"][cal] >= .75))
    )
    inside_claim_fp = fp_cluster_masks["inside_gold_positive_microclaim"]
    outside_claim_fp = fp_mask & ~gold_claim_oracle[cal]
    reachability = {
        "current": {"tp": tp, "fp": fp, "fn": fn, "f1": f1(tp, fp, fn)},
        "perfect_rescue_all_conflict_no_added_FP": {
            "additional_TP": int(conflict_fn.sum()),
            "f1": f1(tp + int(conflict_fn.sum()), fp, fn - int(conflict_fn.sum())),
        },
        "perfect_rescue_high_entailment_conflict_only_no_added_FP": {
            "additional_TP": int(high_entail_conflict_fn.sum()),
            "f1": f1(tp + int(high_entail_conflict_fn.sum()), fp, fn - int(high_entail_conflict_fn.sum())),
        },
        "perfect_prune_FP_inside_positive_microclaims_only": {
            "removed_FP": int(inside_claim_fp.sum()),
            "f1": f1(tp, fp - int(inside_claim_fp.sum()), fn),
        },
        "perfect_prune_FP_outside_positive_microclaims_only": {
            "removed_FP": int(outside_claim_fp.sum()),
            "f1": f1(tp, fp - int(outside_claim_fp.sum()), fn),
        },
        "gold_atomic_microclaim_geometry_reference": {
            "tp": 5984, "fp": 1489, "fn": 0, "f1": f1(5984, 1489, 0)
        },
    }

    result = {
        "status": "complete_CPU_only_posthoc_development_diagnosis",
        "incumbent": metrics(cal_y, cal_pred),
        "false_negative_exclusive_gold_type_clusters": fn_clusters,
        "false_positive_clusters": fp_clusters,
        "conditional_signal_AUROC_fit_vs_calibration": conditional_ranking,
        "fixed_trigger_checks_no_new_thresholds": fixed_trigger_checks,
        "high_entailment_claims": high_entailment_claims,
        "reachability_gold_oracles_not_model_results": reachability,
        "interpretation": {
            "main": "The stable hard class is a rare positive among highly entailed claims. Existing scalar NLI and cross-encoder scores retain some rank information, but their absolute risk is suppressed. Surface slot flags are too noisy and the existing 315-vector independently max-pools entailment and relation features across sentences.",
            "candidate_1": "Entailment-conditioned, evidence-pair-aligned relation expert: preserve each evidence sentence as one item containing E/N/C plus subject/entity/predicate/quantity/negation/source interactions; route high-entailment claims to a class-balanced hard-conflict head and pool only after pair scoring.",
            "candidate_1_minimum_cost_validation": "On frozen original-fit pair features, train a group-OOF two-layer or logistic hard-conflict head only on fit, freeze its threshold on fit OOF, and evaluate calibration. Require improvement specifically on high-entailment conflict without more than one FP per rescued TP before GPU fine-tuning.",
            "candidate_2": "Within-answer safe-claim contrast head: pair each risky microclaim with safe microclaims from the same answer/material, subtract the shared answer-risk component, and learn only from fully group-OOF claim-specific evidence features. This directly targets safe claims inside risky answers that inherit a high global score.",
            "candidate_2_minimum_cost_validation": "Use the frozen NLI315/cross-encoder claim scores plus pair-aligned features in a fit-only grouped pairwise logistic head. Freeze on fit OOF, then require calibration to prune at least two incumbent false positives per lost true positive. The earlier within-answer ranker is not sufficient evidence against this test because its upstream fit scores were in-sample and it lacked pair-aligned evidence conflict.",
        },
        "limits": [
            "Calibration is repeatedly viewed development data; every gold/type/oracle calculation is post-hoc diagnosis.",
            "No model or threshold was fitted or selected here.",
            "Conditional fit AUROC is affected by the incumbent's in-sample upstream fit predictions and is used only as a consistency check.",
            "Surface subject/entity flags are proxies, not semantic role labels.",
            "Official test was not opened and formal baselines were not modified.",
        ],
        "integrity": {
            "new_models_fit": 0, "new_thresholds_selected": 0, "GPU_used": False,
            "official_test_opened": False, "formal_baselines_modified": False,
            "source_sha256": {
                str(path.relative_to(ROOT)): sha(path) for path in (
                    INCUMBENT_SCORE, INCUMBENT_META, NLI_DIR / "scores.npz",
                    NLI_DIR / "evidence_features.npy", NLI_DIR / "inputs.jsonl",
                    CROSS_DIR / "scores.npz", CROSS_DIR / "records.jsonl"
                )
            },
        },
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def p(value):
        return "N/A" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Incumbent FN/FP relation diagnosis", "",
        "只读 CPU 事后分析 fit634/cal159；未拟合模型或阈值，未打开 test，未修改 baseline。", "",
        f"当前 cal 窗口 TP/FP/FN={tp}/{fp}/{fn}，F1={f1(tp, fp, fn):.6f}。", "",
        "## FN 的互斥错误簇", "",
        "| Gold类型 | FN窗 | entail<.50 | .50-.75 | >=.75 | NLI315中位数 | Cross中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in priority:
        row = fn_clusters[kind]
        bins = row["global_entailment_bins"]
        lines.append(
            f"| {kind} | {row['windows']} | {bins['below_050']} | {bins['050_to_075']} | {bins['at_least_075']} | "
            f"{row['signals']['atomic_NLI315_readout']['median']:.3f} | {row['signals']['crossencoder']['median']:.3f} |"
        )
    ec = fn_clusters["Evident Conflict"]
    ec_high_count = ec["global_entailment_bins"]["at_least_075"]
    lines += ["", "### Evident Conflict 中的可观测关系代理", "",
              f"| 代理 | 全部{ec['windows']}个FN | 其中{ec_high_count}个高entail FN |", "|---|---:|---:|"]
    for name in ("subject_low_coverage", "entity_incomplete", "predicate_absent",
                 "negation_mismatch", "comparator_reversal", "comparator_missing",
                 "quantity_different", "quantity_missing", "condition_mismatch",
                 "cited_source_gap_015", "hard_relation_or_source"):
        lines.append(f"| {name} | {ec['observable_flag_counts_all_FN'][name]} | {ec['observable_flag_counts_high_entailment_FN'][name]} |")
    lines += ["", "## FP 的来源", "", "| FP簇 | 窗口 | NLI315中位数 | Cross中位数 |", "|---|---:|---:|---:|"]
    for name, row in fp_clusters.items():
        lines.append(f"| {name} | {row['windows']} | {row['signals']['atomic_NLI315_readout']['median']:.3f} | {row['signals']['crossencoder']['median']:.3f} |")
    lines += ["", "## 可迁移信号检查", "",
              "AUROC是在 incumbent 当前负判中分 FN/TN；fit 与 cal 同向才值得继续。", "",
              "| 信号 | fit | cal |", "|---|---:|---:|"]
    for name in risk_signals:
        fa = conditional_ranking["fit"]["rescue_FN_vs_TN"][name]
        ca = conditional_ranking["calibration"]["rescue_FN_vs_TN"][name]
        lines.append(f"| {name} | {fa:.3f} | {ca:.3f} |")
    fit_h = high_entailment_claims["fit"]
    cal_h = high_entailment_claims["calibration"]
    lines += ["", "## 高entailment难例", "",
              f"- fit：{fit_h['positive']}/{fit_h['claims']} 为错误（{p(fit_h['positive_rate'])}），其中冲突 {fit_h['conflict_positive']} 条。",
              f"- cal：{cal_h['positive']}/{cal_h['claims']} 为错误（{p(cal_h['positive_rate'])}），其中冲突 {cal_h['conflict_positive']} 条。",
              f"- NLI315 在该簇的 claim AUROC：fit {fit_h['score_separation']['atomic_NLI315_readout']['AUROC']:.3f} / cal {cal_h['score_separation']['atomic_NLI315_readout']['AUROC']:.3f}；Cross：fit {fit_h['score_separation']['crossencoder']['AUROC']:.3f} / cal {cal_h['score_separation']['crossencoder']['AUROC']:.3f}。",
              "- 现有表面关系 flag 在 fit/cal 不稳定；尤其否定、比较和条件 mismatch 在 cal 高entailment正例中几乎不触发。它们不能直接当规则。", "",
              "## 两个下一候选", "",
              "1. **证据对齐的高entailment关系专头**：每个证据句保留 E/N/C 与主体、实体、谓词、数量、否定、来源的同一对交互，再做聚合；高entailment样本走单独的类平衡冲突头。最低成本先在冻结特征上做 fit-group OOF 小头，阈值只取 fit。",
              "2. **同答安全微主张对比头**：在同一个风险回答内，把错误主张与安全主张配对，减掉共享的整答风险，只用完全 OOF 的主张证据特征学习排序。最低成本先用 NLI315、Cross 和证据对齐特征做 fit-only 分组逻辑回归。", "",
              "## 可达增益（gold oracle，仅算术）", "",
              f"- 完美补回全部冲突 FN：+{reachability['perfect_rescue_all_conflict_no_added_FP']['additional_TP']} TP，F1={reachability['perfect_rescue_all_conflict_no_added_FP']['f1']:.3f}。",
              f"- 只补高entailment冲突：+{reachability['perfect_rescue_high_entailment_conflict_only_no_added_FP']['additional_TP']} TP，F1={reachability['perfect_rescue_high_entailment_conflict_only_no_added_FP']['f1']:.3f}。",
              f"- 完美删掉主张边界内 FP：-{reachability['perfect_prune_FP_inside_positive_microclaims_only']['removed_FP']} FP，F1={reachability['perfect_prune_FP_inside_positive_microclaims_only']['f1']:.3f}。",
              "", "这些是开发集诊断，不是新模型成绩。"
              ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete = {
        "status": "complete", "analysis_sha256": sha(HERE / "ANALYSIS.json"),
        "report_sha256": sha(HERE / "REPORT.md"), "new_models_fit": 0,
        "new_thresholds_selected": 0, "GPU_used": False,
        "official_test_opened": False, "formal_baselines_modified": False,
    }
    (HERE / "complete.json").write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print("INCUMBENT_FN_ANALYSIS_COMPLETE", tp, fp, fn)


if __name__ == "__main__":
    main()
