"""Compare frozen validation predictions only; no new model execution."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs = {name: HERE / f"modernbert_{name}_v1" for name in ("base", "large")}
    completed = {name: read(path / "complete.json") for name, path in runs.items()}
    predictions = {name: {r["response_id"]: r for r in metrics.readl(path / "best_validation_predictions.jsonl")}
                   for name, path in runs.items()}
    answers = metrics.readl(HERE.parent / "data/validation/answers.jsonl")
    sequences = {r["response_id"]: r for r in metrics.readl(HERE.parent / "data/validation/sequences.jsonl")}
    assert all(set(p) == set(sequences) for p in predictions.values())
    identical = {split: sha(runs["base"] / f"{split}_inputs.jsonl") == sha(runs["large"] / f"{split}_inputs.jsonl")
                 for split in ("fit", "validation")}
    assert all(identical.values())
    cohorts = {}
    for cohort in ("no_hint", "all"):
        counts = {name: Counter() for name in runs}
        changes = Counter()
        for answer in answers:
            if cohort == "no_hint" and answer["unanswerability_hint_setting"]:
                continue
            rid = answer["response_id"]
            seq = sequences[rid]
            lexical, risk = (np.asarray(seq[k], bool) for k in ("lexical_mask", "risk_mask"))
            risk = risk & lexical
            intervals = [(0, len(risk))] if len(risk) < 4 else [(i, i + 4) for i in range(len(risk) - 3)]
            intervals = [(a, b) for a, b in intervals if lexical[a:b].any()]
            y = np.asarray([risk[a:b].any() for a, b in intervals], bool)
            token_risk = np.flatnonzero(risk)
            near = np.asarray([bool(len(token_risk) and np.maximum(np.maximum(a - token_risk, token_risk - (b - 1)), 0).min() <= 4)
                               for a, b in intervals], bool)
            interior = np.asarray([risk[a:b][lexical[a:b]].all() for a, b in intervals], bool)
            predicted = {}
            for name in runs:
                scores = metrics.pool_windows(predictions[name][rid]["token_scores"], lexical, "max")
                thresholds = completed[name]["best"]["thresholds"]
                predicted[name] = p = scores >= thresholds["window"]
                fp, fn = p & ~y, ~p & y
                counts[name].update(window_tp=int((p & y).sum()), window_fp=int(fp.sum()), window_fn=int(fn.sum()),
                    fp_near_risk_within_4_bpe=int((fp & near).sum()), fp_far_or_clean=int((fp & ~near).sum()),
                    fp_in_clean_answers=int(fp.sum()) if not answer["answer_risk"] else 0,
                    fn_risk_only_windows=int((fn & interior).sum()), fn_mixed_boundary_windows=int((fn & ~interior).sum()))
                ap, ay = bool(scores.max() >= thresholds["answer"]), bool(answer["answer_risk"])
                counts[name].update(answer_tp=int(ap and ay), answer_fp=int(ap and not ay), answer_fn=int(not ap and ay))
            b, l = predicted["base"], predicted["large"]
            corrected_fp, introduced_fp = b & ~l & ~y, ~b & l & ~y
            corrected_fn, introduced_fn = ~b & l & y, b & ~l & y
            changes.update(corrected_fp=int(corrected_fp.sum()), introduced_fp=int(introduced_fp.sum()),
                corrected_fp_near=int((corrected_fp & near).sum()), corrected_fp_far_or_clean=int((corrected_fp & ~near).sum()),
                introduced_fp_near=int((introduced_fp & near).sum()), introduced_fp_far_or_clean=int((introduced_fp & ~near).sum()),
                corrected_fn=int(corrected_fn.sum()), introduced_fn=int(introduced_fn.sum()),
                corrected_fn_risk_only=int((corrected_fn & interior).sum()), introduced_fn_risk_only=int((introduced_fn & interior).sum()))
        for name in runs:
            frozen = completed[name]["best"]["shared_evaluation"]["cohorts"][cohort]
            for level in ("answer", "window"):
                for kind in ("tp", "fp", "fn"):
                    assert counts[name][f"{level}_{kind}"] == frozen[level][kind]
        cohorts[cohort] = {"counts": {k: dict(v) for k, v in counts.items()}, "base_to_large_changes": dict(changes)}
    large = completed["large"]
    duration = {"trainer_training_validation_selection_and_checkpoint_seconds": large["seconds"],
        "epoch_training_validation_selection_seconds": [e["seconds"] for e in large["history"]],
        "epoch_sum_seconds": sum(e["seconds"] for e in large["history"]),
        "smoke_adam_step_seconds": read(runs["large"] / "gpu_smoke.json")["seconds"],
        "scope": "Trainer timer starts after model loading and includes six epochs, validation threshold selection and best checkpoint writing; excludes smoke/model loading and final checkpoint hashing. Per-epoch timer excludes checkpoint writing."}
    result = {"selected_epochs": {n: r["best"]["epoch"] for n, r in completed.items()},
        "identical_prepared_input_hashes": identical, "cohorts": cohorts, "duration": duration,
        "source_sha256": {str(p / f): sha(p / f) for p in runs.values() for f in ("complete.json", "best_validation_predictions.jsonl")},
        "annotation": "Existing released automatic labels, not new human gold.", "test_labels_read": False, "GPU_used": False}
    (HERE / "LARGE_VALIDATION_COMPARISON.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = ["# Large 与 base 的冻结验证结果", "", "沿用各模型原先选定轮次与阈值，只比较已保存预测；没有新增训练、调阈值或修改标签。标签仍为发布的自动标注。", "",
            "|验证队列/级别|base FP/FN|large FP/FN|base F1|large F1|", "|---|---:|---:|---:|---:|"]
    for cohort in ("no_hint", "all"):
        for level in ("window", "answer"):
            b, l = (completed[n]["best"]["shared_evaluation"]["cohorts"][cohort][level] for n in runs)
            text.append(f"|{cohort}/{level}|{b['fp']}/{b['fn']}|{l['fp']}/{l['fn']}|{b['f1']:.5f}|{l['f1']:.5f}|")
    nh = cohorts["no_hint"]
    change = nh["base_to_large_changes"]
    bw, lw = (completed[n]["best"]["validation"]["window"] for n in runs)
    text.extend(["", f"Base 选择第 {completed['base']['best']['epoch']} 轮，large 选择第 {large['best']['epoch']} 轮。fit 和 validation 的编码输入、字符映射、原始 BPE 窗口几何逐字节相同，差异不是更换切词或边界对齐实现产生的。",
        "", f"在 no_hint 队列，large 修正 {change['corrected_fp']} 个原 base 误报窗口，其中 {change['corrected_fp_far_or_clean']} 个离风险标注超过 4 个原始 BPE 或所在回答没有风险标注，{change['corrected_fp_near']} 个在 4 BPE 以内；同时新增 {change['introduced_fp']} 个误报。修正 {change['corrected_fn']} 个漏报、同时新增 {change['introduced_fn']} 个漏报。",
        "", f"主要收益是误报减少，代价是漏报小幅增加。两者使用各自已选阈值；窗口 AP 从 {bw['average_precision']:.5f} 升至 {lw['average_precision']:.5f}，AUROC 从 {bw['auroc']:.5f} 升至 {lw['auroc']:.5f}，表明排序能力也改善，不能仅归因于阈值变动。",
        "", "距离定义为负窗口到最近风险词元的最短原始 BPE 距离；没有风险词元的回答归为远处/无风险组。该统计说明变化是否集中在边界附近，不构成人工正确性重标。",
        "", f"Large 六轮训练、逐轮验证和选择、最佳权重保存共 {large['seconds']:.2f} 秒（{large['seconds']/60:.2f} 分钟）；逐轮训练和验证计时合计 {duration['epoch_sum_seconds']:.2f} 秒。前者不含模型加载、独立 smoke 和结束时 checkpoint 哈希，后者还不含 checkpoint 保存。Smoke 的单次 Adam 步为 {duration['smoke_adam_step_seconds']:.3f} 秒。",
        "", "仅验证结果可用于本轮方案选择；OOF 和独立测试结论另行报告。"])
    (HERE / "LARGE_VALIDATION_COMPARISON.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"selected_epochs": result["selected_epochs"], "no_hint": nh, "duration": duration}, ensure_ascii=False))


if __name__ == "__main__":
    main()
