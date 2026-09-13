"""Read frozen development outputs; add arithmetic baselines without refitting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    complete = load("complete.json")
    for name, expected in complete["files_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, name
    summary = load("summary.json")
    naive = load("naive_baselines.json")
    rows = {}
    for name, chosen in summary["selected"].items():
        candidates = summary["all_candidates"][name]
        assert chosen["candidate"] == max(candidates, key=lambda x: x["selection_key"])["candidate"]
        rows[name] = chosen["metrics"]["calibration"]
    for name in ("all_positive", "all_negative"):
        rows[name] = {g: naive["partitions"]["calibration"][g][name] for g in ("windows", "answers")}
    for row in rows.values():
        for g, metrics in row.items():
            n = summary["coverage"]["calibration"]["eligible_windows" if g == "windows" else "answers"]
            assert metrics["n"] == n
            tp, fp, fn, tn = (metrics[k] for k in ("tp", "fp", "fn", "tn"))
            assert tp + fp + fn + tn == n
            assert abs(metrics["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    lines = [
        "12个候选已完整训练并冻结，使用全部168123个fit窗口。用时789.4秒（约13.2分钟，含构建矩阵及评分），未打开测试集。",
        "",
        "以下159份回答/42241个窗口来自用于选参的校准集，不能当最终测试成绩。整答风险数100，窗口风险数5984。",
        "",
        "| 方法 | 窗口P | 窗口R | 窗口F1 | 窗口AP | 整答P | 整答R | 整答F1 | 整答AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        values = [row[g][m] for g in ("windows", "answers") for m in ("precision", "recall", "f1", "average_precision")]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend([
        "",
        "全报风险的整答F1已有0.7722，超过0.75本身不足以证明效果好。纯Lookback在本校准集把误报从59降至23，同时漏掉7份风险回答；这体现了开发集上的区分能力，仍须在封存测试集确认。",
        "",
        "加入NLL或内部状态没有明显改善窗口定位；层带顺序的窗口F1最高为0.5632，也远未达到0.75。四个家族均按预定规则选择C=0.001。窗口是4个原始BPE词元的范围，窗口F1不是逐词元F1。",
        "",
        "人工good答案与拒答文字均保留；风险仍按公开资料与人工span衡量。内部状态由匹配Llama2 NF4模型对原答案重建，不能声称获得了数据首次生成时保存的状态。",
        "",
        "本文件仅汇总冻结结果和无模型算术基线；没有改模型、阈值或金标。这里的哈希与计数核对由实现者完成，不替代独立复核。",
    ])
    (ROOT / "COMPARISON_WITH_NAIVE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = {"status": "frozen_output_consistency_checked", "independent_audit": False,
              "test_opened": False, "elapsed_seconds": summary["elapsed_seconds"],
              "frozen_files_checked": len(complete["files_sha256"]), "calibration": rows}
    (ROOT / "comparison_with_naive.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "files_checked": output["frozen_files_checked"],
                      "elapsed_seconds": output["elapsed_seconds"],
                      "calibration": {k: {g: [v[g][m] for m in ("precision", "recall", "f1", "average_precision")] for g in ("windows", "answers")} for k, v in rows.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
