"""Render frozen localization results; never rescore or select a threshold."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    complete = load("test_complete8.json")
    data = {name: load(f"metrics_{name}.json") for name in ("main", "external")}
    for name in data:
        checksum = hashlib.sha256((OUT / f"metrics_{name}.json").read_bytes()).hexdigest()
        assert checksum == complete["metrics_sha256"][name]
    lines = ["# Round8 完整定位成绩", "", "所有数值从冻结结果生成；未重新选阈值、重训或重跑模型。"
             "风险表示当前资料无依据或相矛盾，不等于世界事实必定错误。", "",
             "原生词元方法读取当前位置的内部信号；广播对照把整项原分数赋给每个词元，属于事后对照，不能称逐词元算法。", ""]
    rows = []
    for name, d in data.items():
        c = d["coverage"]
        m = d["original_threshold"]["methods"]["hidden_probe__token"]["all_resolved_items"]["micro"]
        rows.append([name, c["groups"], c["answers"], c["items"], c["localization_status"]["resolved"],
                     c["main_tokens"], m["risk_tokens"], c["canonical_error_regions"]])
    lines += [table(["集合", "题组", "回答", "预定项", "可判定项", "计分词元", "风险词元", "风险片段"], rows), "",
              "同一风险片段可能含多个词元。分母保留正常回答项和风险项内的非风险位置；纯标点/空白等排除。", ""]
    for regime, title in (("original_threshold", "A：沿用原回答项阈值"),
                          ("validation_token_threshold", "B：仅验证集选择词元阈值")):
        lines += ["## " + title, "", "F1均为风险类。句内F1只计算已知含风险的回答项中的所有可计分词元，保留其中的非风险词元。", ""]
        for kind, suffix in (("原生词元评分", "__token"), ("整项分数广播对照", "__broadcast")):
            rows = []
            for method in data["main"][regime]["methods"]:
                if not method.endswith(suffix):
                    continue
                a = data["main"][regime]["methods"][method]
                b = data["external"][regime]["methods"][method]
                rows.append([method.removesuffix(suffix), fmt(a["all_resolved_items"]["micro"]["precision"]),
                             fmt(a["all_resolved_items"]["micro"]["recall"]), fmt(a["all_resolved_items"]["micro"]["f1"]),
                             fmt(a["risk_items_only"]["micro"]["f1"]), fmt(b["all_resolved_items"]["micro"]["precision"]),
                             fmt(b["all_resolved_items"]["micro"]["recall"]), fmt(b["all_resolved_items"]["micro"]["f1"]),
                             fmt(b["risk_items_only"]["micro"]["f1"])])
            lines += ["### " + kind, "", table(["方法", "主P", "主R", "主F1", "主句内F1", "外P", "外R", "外F1", "外句内F1"], rows), ""]
        rows = []
        for name in ("main", "external"):
            for mode in ("token", "broadcast"):
                stats = data[name][regime]["mlp_seed_summary"][mode]["subsets"]
                rows.append([name, mode] + [fmt(stats[s]["f1"]["mean"]) + " ± " + fmt(stats[s]["f1"]["sd"])
                                           for s in ("all_resolved_items", "risk_items_only")])
        lines += ["### MLP三种子汇总", "", "每个种子独立预测，报告F1均值和样本标准差；没有挑最佳种子或平均概率。", "",
                  table(["集合", "方式", "全部项F1", "句内F1"], rows), ""]
        for name, cn in (("main", "主测试"), ("external", "外部测试")):
            rows = []
            for method, result in data[name][regime]["methods"].items():
                if not method.endswith("__token"):
                    continue
                a = result["all_resolved_items"]
                ci = data[name][regime]["group_bootstrap"]["subsets"]["all_resolved_items"][method]
                delta = ci["micro_f1_minus_own_broadcast"]["ci95"]
                interval = ci["micro_f1"]["ci95"]
                sp = a["span_regions"]
                rows.append([method.removesuffix("__token"), "–".join(map(fmt, interval)),
                             "[" + ", ".join(map(fmt, delta)) + "]", fmt(a["micro"]["auroc"]),
                             fmt(result["risk_items_only"]["micro"]["auroc"]), f"{sp['hit']}/{sp['n']}",
                             fmt(sp["mean_token_coverage"]), a["micro"]["fp"],
                             fmt(a["fully_normal_answers"]["token_false_positive_rate"])])
            lines += ["### " + cn + "：区间、排序与误报", "",
                      table(["方法", "F1的95%区间", "相对自身广播F1差值区间", "全部项AUROC", "句内AUROC", "命中片段", "片段平均词元覆盖", "额外报警词元", "全正常回答词元误报率"], rows), ""]
        rows = []
        for method, result in data["main"][regime]["methods"].items():
            if method.endswith("__token"):
                rows.append([method.removesuffix("__token")] + [fmt(result["attributes"][c][scope]["f1"])
                             for c in ("birth_year", "other") for scope in ("all_resolved_items", "risk_items_only")])
        lines += ["### 主测试属性分层", "", "出生年29组，其他属性11组；沿用同一阈值，未按属性调参。", "",
                  table(["方法", "出生年F1", "出生年句内F1", "其他F1", "其他句内F1"], rows), ""]
    lines += ["## 口径", "", "区间按原题组抽样2000次，seed=20260911；同题的两个条件和全部词元一起抽取。"
              "它只描述冻结模型、阈值与助手标注条件下的抽样波动，没有包含重训、标注分歧或多重比较校正。", "",
              "F1 = 2TP / (2TP + FP + FN)。TP是正确标中的风险词元；FP是在非风险位置报警；FN是漏掉的风险词元。"
              "AUROC/AP只反映风险排序。sigmoid输出不是已校准的事实错误概率；ReDeEP等分数也不一定处于0–1。", "",
              "完整TP/FP/FN/TN、每答macro F1、逐片段覆盖、字符覆盖、排除项报警描述、两个分母区间及全部分数保存在"
              " metrics_main.json、metrics_external.json、token_scores_main.jsonl、token_scores_external.jsonl。", "",
              "本轮复用已看过回答项成绩的Round7题目，是定位诊断，不能作为新留出集或原论文完整复现的排行榜结果。", ""]
    (OUT / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print("TABLES_RENDERED_FROM_FROZEN_RESULTS")


if __name__ == "__main__":
    main()
