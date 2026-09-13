"""Automatic factual tables only; the research interpretation is written separately."""
import argparse
import json
from pathlib import Path

from evaluate7 import ROOT, METHODS, MLP_SEEDS, save, sha

NAMES = {"hidden_probe": "内部状态逻辑回归", "lookback_lens": "Lookback Lens 项均值适配",
         "redeep": "ReDeEP-token Qwen／标准JSD适配", "lumina": "LUMINA 固定λ=0.5",
         "nll": "平均负对数概率", "entropy": "平均预测熵", "surface": "表面特征逻辑回归",
         "self_confidence": "同一Qwen自评把握", "direct_check": "同一Qwen直接核查",
         "all_positive": "全部报风险", "all_negative": "全部报正常"}


def fmt(v, digits=3):
    return "不可估计" if v is None else f"{v:.{digits}f}"


def spread(v):
    return fmt(v["mean"])+" ± "+fmt(v["sd"])


def interval(v):
    return "不可估计" if v["ci95"] is None else "["+", ".join(fmt(x) for x in v["ci95"])+"]"


def make_tables(root):
    out = root/"results"
    complete = json.loads((out/"test_complete.json").read_text(encoding="utf-8"))
    freeze = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert complete["freeze_sha256"] == sha(out/"freeze.json")
    lines = ["# Round 7：冻结评测自动结果表", "",
             "本表仅呈现已经冻结的一次统一评测。指标是回答项风险类F1，不是逐词或整份回答F1。标签由助手依据来源审阅，未冒称独立人工金标准。", "",
             "所有可判定事实断言均进入主分母；无可用预测的风险项按漏检计数。AUROC/AP只在可评分项计算，覆盖数另列。AP使用average precision。MLP报告各随机种子单独判定的指标及其均值、样本标准差，没有平均概率组成的新检测器。", ""]
    for cohort, title in [("main", "主实验留出"), ("external", "RAGognize外部留出")]:
        path = out/f"metrics_{cohort}.json"
        assert sha(path) == complete["metrics_sha256"][cohort]
        data = json.loads(path.read_text(encoding="utf-8"))
        behavior, boot = data["behavior"]["all"], data["bootstrap"]
        lines += [f"## {title}", "",
                  f'{behavior["question_groups"]} 个独立组，{behavior["questions"]} 个问题，{behavior["responses"]} 份回答，{behavior["slots"]} 个预定项；其中 {behavior["resolved_asserted"]} 个可判定事实断言、{behavior["risk_items"]} 个风险项。', "",
                  "| 方法 | F1 | 精确率 | 召回率 | TP | FP | FN | TN | AUROC | AP | 可评分/可判定 | F1 95%区间 | F1减线性探针95%区间 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for method in METHODS:
            if method.startswith("hallurag_mlp_seed_"):
                continue
            z = data["methods"][method]
            m, c, b = z["all_resolved"], z["coverage"], boot["methods"][method]
            lines.append(f'| {NAMES[method]} | {fmt(m["f1"])} | {fmt(m["precision"])} | {fmt(m["recall"])} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["tn"]} | {fmt(m["auroc"])} | {fmt(m["average_precision"])} | {c["scorable_resolved"]}/{c["resolved"]} | {interval(b["f1"])} | {interval(b["f1_minus_hidden_probe"])} |')
        m = data["hallurag_mlp_summary"]["all_resolved"]
        b = boot["methods"]["hallurag_mlp_mean"]
        lines += ["", "HalluRAG风格隐藏状态MLP（均值 ± 样本标准差，非完整原版复现）：", "",
                  "| F1 | 精确率 | 召回率 | AUROC | AP | 平均F1的组bootstrap区间 | 相对线性探针差值区间 |",
                  "|---:|---:|---:|---:|---:|---|---|",
                  f'| {spread(m["f1"])} | {spread(m["precision"])} | {spread(m["recall"])} | {spread(m["auroc"])} | {spread(m["average_precision"])} | {interval(b["f1"])} | {interval(b["f1_minus_hidden_probe"])} |', "",
                  "| MLP种子 | F1 | 精确率 | 召回率 | TP | FP | FN | TN | 可评分/可判定 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for seed in MLP_SEEDS:
            z = data["methods"][f"hallurag_mlp_seed_{seed}"]
            m, c = z["all_resolved"], z["coverage"]
            lines.append(f'| {seed} | {fmt(m["f1"])} | {fmt(m["precision"])} | {fmt(m["recall"])} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["tn"]} | {c["scorable_resolved"]}/{c["resolved"]} |')
        lines += ["", "| 方法 | 前20%实际复查/预算 | 找出风险数 | 已确认风险产出率 | 已知风险召回率 | 混合回答内平均AUROC | 混合回答/独立组 |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for method in METHODS:
            z = data["methods"][method]
            t, w = z["top20"], z["within_answer"]
            name = NAMES.get(method, "MLP seed "+method.rsplit("_", 1)[-1])
            lines.append(f'| {name} | {t["selected"]}/{t["budget"]} | {t["found"]} | {fmt(t["confirmed_risk_yield"])} | {fmt(t["recall"])} | {fmt(w["mean_response_auc"])} | {w["mixed_responses"]}/{w["mixed_groups"]} |')
        lines += ["", "前20%主预算按全部预定回答项向上取整，从所有具有有限风险分数的项中排序，同分按固定item_id排序；选择过程不使用标签，也不按拒答、未决或判定阈值过滤。已确认风险产出率=选中已确认风险数/实际选中数；未决项的真实风险未知，因此该值不冒称完整真值下的精确率。风险召回分母包含全部已知风险。", "",
                  "| 方法 | 全项缺分数 | 全项缺判定 | 预算未填满 | 选中可判定 | 其中有依据 | 选中拒答 | 选中未决 | 选中缺项 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            t = data["methods"][method]["top20"]
            name = NAMES.get(method, "MLP seed "+method.rsplit("_", 1)[-1])
            lines.append(f'| {name} | {t["missing_scores"]} | {t["missing_predictions"]} | {t["unfilled_budget"]} | {t["selected_resolved"]} | {t["selected_supported"]} | {t["selected_abstained"]} | {t["selected_unresolved"]} | {t["selected_missing"]} |')
        lines += ["", "选中可判定包括风险和有依据的断言；缺项指未生成该回答项，缺分数指检测器无有限分数。旧的先按标注过滤可判定断言再排名的指标仅保存在 JSON 的 top20_resolved_only，属于使用金标筛选的次要诊断（oracle annotation-filtered secondary），不作为实际复查效果。", "",
                  f'区间按 {boot["groups"]} 个group_id进行 {boot["draws"]} 次共同重采样，完整／缺失及同组各项一起抽取。区间以本次已经训练的模型和阈值为条件，不重新训练。主实验和外部检查分开，不能合并成一个F1。', "",
                  "| 条件 | 预定项 | 可判定断言 | 风险项 | 拒答 | 缺项 | 未决依据 | 自动解析失败 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for condition, z in data["behavior"]["conditions"].items():
            lines.append(f'| {condition} | {z["slots"]} | {z["resolved_asserted"]} | {z["risk_items"]} | {z["stance"].get("abstained",0)} | {z["stance"].get("missing",0)} | {z["evidence_relation"].get("unresolved",0)} | {z["parse_failures"]} |')
        lines += ["", f'[全部指标](metrics_{cohort}.json) · [逐项预测](predictions_{cohort}.jsonl) · [各方法错误项](errors_{cohort}.jsonl)', ""]
    lines += ["## 已冻结设置", "",
              f'本次拟合调用墙钟耗时：{fmt(freeze["fit_seconds"],1)} 秒（包含载入、校验、参数选择及验证；缓存恢复时不含此前调用）；最多4线程。MLP共同选择层{freeze["mlp_selected"]["layer"]}、学习率{freeze["mlp_selected"]["lr"]}；三个种子均保留。', "",
              "[参数和阈值](freeze.json) · [全部验证候选](selection.json) · [验证指标](validation.json)", ""]
    target = out/"TABLES.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    save(out/"tables_manifest.json", {"script_sha256": sha(Path(__file__)), "table_sha256": sha(target),
        "freeze_sha256": sha(out/"freeze.json"), "source_metrics_sha256": complete["metrics_sha256"],
        "research_interpretation": "Not auto-generated here; compare strong cheap baselines and label/wording confounds separately"})
    print("TABLES_COMPLETE", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    make_tables(parser.parse_args().root.resolve())
