"""Paired source-group bootstrap of already-selected calibration predictions.

No fitting, score inference, threshold/candidate selection, or test-file access.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
SEED = 20261009
REPEATS = 5000
METHODS = ("harp_tail_fixed", "harp_claim", "tail2", "semantic_tail_fixed", "harp_tree")
CONTRASTS = (
    ("harp_tail_fixed_minus_harp_claim", "harp_tail_fixed", "harp_claim"),
    ("harp_tail_fixed_minus_tail2", "harp_tail_fixed", "tail2"),
    ("harp_tail_fixed_minus_semantic_tail_fixed", "harp_tail_fixed", "semantic_tail_fixed"),
    ("harp_tree_minus_harp_tail_fixed", "harp_tree", "harp_tail_fixed"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f1(counts):
    tp, fp, fn = np.moveaxis(np.asarray(counts), -1, 0)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0)


def metric(y, scores, threshold):
    positive, predicted = y == 1, scores >= threshold
    tp = int(np.sum(positive & predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    tn = int(np.sum(~positive & ~predicted))
    return {"n": len(y), "positive": int(positive.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0., "f1": float(f1([tp, fp, fn]))}


def group_counts(y, scores, threshold, owners, n_groups):
    positive, predicted = y == 1, scores >= threshold
    masks = (positive & predicted, ~positive & predicted, positive & ~predicted)
    return np.stack([np.bincount(owners[m], minlength=n_groups) for m in masks], axis=1)


def run():
    data, results = QA / "data", QA / "results"
    source_hashes = {}

    def tracked(path):
        source_hashes[str(path.relative_to(QA)).replace("\\", "/")] = sha(path)
        return read(path)

    aw_path = data / "answers_calibration.jsonl"
    ww_path = data / "windows_k4_calibration.jsonl"
    answers, windows = rows(aw_path), rows(ww_path)
    for path in (aw_path, ww_path):
        source_hashes[str(path.relative_to(QA)).replace("\\", "/")] = sha(path)
    assert len(answers) == 159 and len(windows) == 42241
    assert all(x["partition"] == "calibration" and x["eligible"] for x in answers + windows)
    assert all(x["label"] in (0, 1) for x in answers + windows)
    by_id = {x["response_id"]: x for x in answers}
    assert len(by_id) == 159
    assert all(x["group_id"] == by_id[x["response_id"]]["group_id"] for x in windows)
    group_ids = sorted({x["group_id"] for x in answers})
    assert len(group_ids) == 154
    group_index = {g: i for i, g in enumerate(group_ids)}
    owners = {"windows": np.array([group_index[x["group_id"]] for x in windows]),
              "answers": np.array([group_index[x["group_id"]] for x in answers])}
    gold = {"windows": np.array([x["label"] for x in windows], dtype=np.int8),
            "answers": np.array([x["label"] for x in answers], dtype=np.int8)}
    assert gold["windows"].sum() == 5984 and gold["answers"].sum() == 100

    protocol = {
        "scope": "Already-exposed calibration only; fixed previously-selected models and thresholds",
        "seed": SEED, "replicates": REPEATS, "groups": 154, "answers": 159, "windows": 42241,
        "group_key": "Original group_id, sorted lexicographically; all answers/windows in a sampled group move together",
        "sampling": "numpy default_rng(20261009), PCG64; integers(0,154,size=(5000,154)); same draws for all methods and units",
        "statistic": "Pooled micro F1 = 2TP/(2TP+FP+FN); counts summed within groups then resampled",
        "interval": "Paired F1 difference, 2.5th/97.5th percentile with NumPy linear interpolation",
        "zero_denominator_f1": 0, "methods": list(METHODS), "contrasts": list(CONTRASTS),
        "thresholds_or_models_reselected": False, "test_opened": False,
        "limitation": "Conditional uncertainty on repeatedly-used development calibration; does not account for threshold, model, epoch, alpha or other selection bias. Not an independent test interval.",
    }
    write(OUT / "protocol.json", protocol)

    fusion_dir = results / "completed_score_fusion_v1"
    fusion = tracked(fusion_dir / "summary.json")
    fusion_complete = tracked(fusion_dir / "complete.json")
    assert sha(fusion_dir / "summary.json") == fusion_complete["summary_sha256"]
    claim_dir = results / "claim_pooling_v1"
    claim = tracked(claim_dir / "summary.json")["selected"]["full_lb_harp64_tcn"]
    tail_dir = results / "minicheck_tail_all_docs_v3/tail2"
    tail = tracked(tail_dir / "complete.json")["selected"]
    tree_dir = results / "completed_score_combiner_v1"
    tree = tracked(tree_dir / "summary.json")["methods"]["harp_claim"]
    tree_complete = tracked(tree_dir / "complete.json")
    assert sha(tree_dir / "summary.json") == tree_complete["summary_sha256"]

    entries = {
        "harp_tail_fixed": (fusion["selected"]["harp_claim"], fusion_dir, "selected alpha=0.25"),
        "harp_claim": (claim, claim_dir, "original HARP claim propagation alpha=0.75"),
        "tail2": (tail, tail_dir, "selected epoch=2, no new claim propagation"),
        "semantic_tail_fixed": (fusion["selected"]["semantic_claim"], fusion_dir, "selected alpha=0.25, same five-alpha budget"),
        "harp_tree": (tree, tree_dir, "saved fixed depth-2 monotone tree, 100 iterations"),
    }
    assert entries["harp_tail_fixed"][0]["tail_weight"] == .25
    assert entries["semantic_tail_fixed"][0]["tail_weight"] == .25
    assert claim["alpha"] == .75 and tail["epoch"] == 2
    frozen = {}
    counts = np.zeros((len(METHODS), 2, len(group_ids), 3), dtype=np.int64)
    answer_index = {x["response_id"]: i for i, x in enumerate(answers)}
    window_answer = np.array([answer_index[x["response_id"]] for x in windows])
    for mi, name in enumerate(METHODS):
        entry, directory, description = entries[name]
        if name == "tail2":
            path = directory / f'epoch_{entry["epoch"]:02d}_scores.npz'
            expected_hash = entry["artifacts_sha256"]["_scores.npz"]
            with np.load(path, allow_pickle=False) as z:
                scores = {"windows": z["cal_window_scores"].copy(), "answers": z["cal_answer_scores"].copy()}
                assert np.array_equal(z["cal_window_labels"], gold["windows"])
                assert np.array_equal(z["cal_answer_labels"], gold["answers"])
            metrics = entry["calibration"]
        else:
            path = directory / ("harp_claim_scores.npz" if name == "harp_tree" else entry["candidate"] + "_scores.npz")
            expected_hash = entry["scores_sha256"]
            with np.load(path, allow_pickle=False) as z:
                assert z["window_scores"].shape == (210364,) and z["answer_scores"].shape == (793,)
                # Existing combined archives carry fit first; no fit labels or fit-score analyses are used.
                scores = {"windows": z["window_scores"][168123:].copy(), "answers": z["answer_scores"][634:].copy()}
            metrics = entry["metrics"]["calibration"]
        actual_hash = sha(path)
        assert actual_hash == expected_hash
        source_hashes[str(path.relative_to(QA)).replace("\\", "/")] = actual_hash
        maxima = np.full(159, -np.inf)
        np.maximum.at(maxima, window_answer, scores["windows"])
        assert np.array_equal(maxima, scores["answers"])
        frozen[name] = {"description": description, "thresholds": entry["thresholds"], "metrics": {}}
        for ui, unit in enumerate(("windows", "answers")):
            threshold = entry["thresholds"]["window" if unit == "windows" else "answer"]["threshold"]
            s = scores[unit]
            assert s.shape == gold[unit].shape and np.isfinite(s).all()
            m = metric(gold[unit], s, threshold)
            assert all(v == metrics[unit][k] for k, v in m.items()), (name, unit)
            frozen[name]["metrics"][unit] = m
            counts[mi, ui] = group_counts(gold[unit], s, threshold, owners[unit], len(group_ids))
            assert np.array_equal(counts[mi, ui].sum(axis=0), [m["tp"], m["fp"], m["fn"]])

    rng = np.random.default_rng(SEED)
    assert type(rng.bit_generator).__name__ == "PCG64"
    draws = rng.integers(0, len(group_ids), size=(REPEATS, len(group_ids)))
    # Keep one answer's overlapping windows correlated by resampling group-level counts only.
    resampled_counts = counts[:, :, draws, :].sum(axis=3)
    assert resampled_counts.shape == (len(METHODS), 2, REPEATS, 3)
    boot = f1(resampled_counts)
    pooled = f1(counts.sum(axis=2))
    degenerate = (2 * resampled_counts[..., 0] + resampled_counts[..., 1] + resampled_counts[..., 2]) == 0
    # Small direct repetition check of the count-aggregation path, no model/threshold choice.
    for r in (0, 1, REPEATS - 1):
        expected = sum((counts[:, :, int(g), :] for g in draws[r]), np.zeros((len(METHODS), 2, 3), dtype=np.int64))
        assert np.array_equal(expected, resampled_counts[:, :, r, :])
    intervals = {}
    for label, left, right in CONTRASTS:
        li, ri = METHODS.index(left), METHODS.index(right)
        intervals[label] = {"left": left, "right": right}
        for ui, unit in enumerate(("windows", "answers")):
            differences = boot[li, ui] - boot[ri, ui]
            low, high = np.percentile(differences, [2.5, 97.5], method="linear")
            intervals[label][unit] = {"point_f1_difference": float(pooled[li, ui] - pooled[ri, ui]),
                                      "percentile95": [float(low), float(high)],
                                      "contains_zero": bool(low <= 0 <= high)}
    np.savez_compressed(OUT / "group_bootstrap.npz", group_ids=np.array(group_ids), methods=np.array(METHODS),
                        units=np.array(["windows", "answers"]), count_order=np.array(["tp", "fp", "fn"]),
                        group_counts=counts, draws=draws.astype(np.int16), bootstrap_f1=boot)
    group_sizes = Counter(x["group_id"] for x in answers)
    result = {"status": "passed", "protocol": protocol, "source_sha256": source_hashes,
              "answer_count_per_group_histogram": dict(Counter(group_sizes.values())),
              "fixed_methods": frozen, "contrasts": intervals,
              "degenerate_f1_denominators": int(degenerate.sum()),
              "checks": {"frozen_score_hashes": True, "calibration_gold_and_group_identity": True,
                         "pooled_counts_and_f1_exact": True, "answer_maxima_exact": True,
                         "same_paired_draws_across_methods_and_units": True, "model_or_threshold_selection": False,
                         "model_forward": False, "GPU_used": False, "test_opened": False},
              "numpy_version": np.__version__, "script_sha256": sha(__file__),
              "bootstrap_arrays_sha256": sha(OUT / "group_bootstrap.npz")}
    write(OUT / "PAIRED_GROUP_INTERVALS.json", result)
    labels = {"harp_tail_fixed_minus_harp_claim": "HARP+tail − 原 HARP_claim",
              "harp_tail_fixed_minus_tail2": "HARP+tail − tail2",
              "harp_tail_fixed_minus_semantic_tail_fixed": "HARP+tail − semantic+tail",
              "harp_tree_minus_harp_tail_fixed": "HARP 单调树 − HARP+tail"}
    lines = ["# 固定结果的资料组配对区间", "", "只使用已反复开发的 159 条校准回答、154 个资料组，共 42,241 个四词元窗口。所有模型、轮次、组合权重和两级阈值保持原选择；按资料组有放回抽样 5,000 次，固定种子 20261009。同组的回答及全部重叠窗口一起重采样。", "",
             "下表为 F1 差值及 95% 百分位区间（原始 F1 单位，非百分数）。", "",
             "| 对比 | 四词元窗口 ΔF1 [95% 区间] | 整答 ΔF1 [95% 区间] |", "|---|---:|---:|"]
    for label, _, _ in CONTRASTS:
        fields = []
        for unit in ("windows", "answers"):
            v = intervals[label][unit]
            fields.append(f'{v["point_f1_difference"]:+.5f} [{v["percentile95"][0]:+.5f}, {v["percentile95"][1]:+.5f}]')
        lines.append(f'| {labels[label]} | {fields[0]} | {fields[1]} |')
    lines += ["", "这些是固定已选结果在该开发集上的条件区间，没有计入反复选择模型、轮次、阈值及组合权重带来的偏差，不能替代未见测试集证据。区间跨零表示本次重采样不能清楚区分差异，不代表方法等效。", "",
              "已核对保存分数哈希、原始计数和 F1、全部整答最大窗口分数；未拟合、未重新推理、未读取测试。各资料组计数、共用抽样索引与每轮 F1 保存在 group_bootstrap.npz。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "complete.json", {"status": "complete_conditional_calibration_diagnostic", "result_sha256": sha(OUT / "PAIRED_GROUP_INTERVALS.json"),
                                "report_sha256": sha(OUT / "REPORT.md"), "protocol_sha256": sha(OUT / "protocol.json"), "test_opened": False})
    print(json.dumps({"status": "passed", "groups": len(group_ids), "replicates": REPEATS, "contrasts": intervals}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
