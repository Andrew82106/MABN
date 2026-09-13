"""Produce a self-contained Chinese report and all-case HTML from frozen results."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from evaluate6 import ROOT, METHODS, readl, resolved, sha, writej


NAMES = {"hidden21_probe": "第21层内部状态探针", "surface_logistic": "表面信息逻辑回归",
         "mean_nll": "生成平均负对数概率", "mean_entropy": "生成平均预测熵",
         "self_confidence": "同一Qwen自评把握", "direct_check": "同一Qwen直接核查",
         "all_positive": "全部报风险", "all_negative": "全部报正常"}
STANCE = {"asserted": "事实断言", "tentative": "明确推测", "abstained": "主动承认不足", "missing": "缺项"}
RELATION = {"supported": "材料支持", "unsupported": "材料无依据", "contradicted": "与材料冲突", "unresolved": "依据未判清"}
CORRECT = {"correct": "符合完整参考", "incorrect": "不符合完整参考", "unresolved": "参考对错未判清", "not_applicable": "不作事实对错判断"}
SPLITS = {"train": "训练", "validation": "验证", "test": "留出检查"}
CONDITIONS = {"complete": "完整资料", "partial": "缺失资料", "one_entity_evidence_missing": "缺失资料"}


def fmt(value, digits=3):
    return "不可估计" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "不可估计" if value is None else f"{value:.1%}"


def esc(value):
    return html.escape(str(value))


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def label_text(annotation):
    return "；".join([STANCE.get(annotation["stance"], annotation["stance"]),
                      RELATION.get(annotation["evidence_relation"], annotation["evidence_relation"]),
                      CORRECT.get(annotation["reference_correctness"], annotation["reference_correctness"])])


def status(prediction, method="hidden21_probe"):
    a = prediction["annotation"]
    flag = prediction["methods"][method]["prediction"]
    if not resolved(a):
        return "主分类外"
    if flag is None:
        return "未能监测／风险漏检" if a["risk"] else "未能监测／有依据"
    return {(1, 1): "检出风险", (1, 0): "漏检风险", (0, 1): "误报有依据", (0, 0): "保留有依据"}[(a["risk"], flag)]


def main():
    root = ROOT
    out = root / "results"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    predictions = readl(out / "development_predictions.jsonl") + readl(out / "predictions.jsonl")
    generated = readl(root / "data/generated.jsonl")
    inputs = {r["row_id"]: r for r in readl(root / "data/inputs.jsonl")}
    references = {r["question_id"]: r for r in readl(root / "data/references.jsonl")}
    baseline_path = root / "data/baselines.jsonl"
    baselines = {r["item_id"]: r for r in readl(baseline_path)}
    assert len(predictions) == len({p["item_id"] for p in predictions})
    assert {p["item_id"] for p in predictions} == {i["item_id"] for r in generated for i in r["items"]}
    assert metrics["freeze_sha256"] == sha(out / "freeze.json")
    by_row = defaultdict(list)
    for p in predictions:
        by_row[p["row_id"]].append(p)
    by_question = defaultdict(list)
    for row in generated:
        by_question[row["question_id"]].append(row)
    probe = metrics["methods"]["hidden21_probe"]
    primary = probe["end_to_end"]
    direct = metrics["methods"]["direct_check"]["end_to_end"]
    behavior = metrics["behavior"]["overall"]
    primary_f1, direct_f1 = primary["f1"], direct["f1"]
    if primary_f1 is None:
        finding = "留出部分缺少可估计主指标的条件，不能报告一个有效的探针F1。"
    else:
        finding = (f'第21层探针的**事实项风险F1为 {primary_f1:.3f}**，'
                   f'检出 {primary["tp"]}/{primary["positive"]} 个风险项，'
                   f'误报 {primary["fp"]}/{primary["negative"]} 个有依据项。')
    if primary_f1 is not None and direct_f1 is not None:
        comparison = (f'同一Qwen直接核查F1为 {direct_f1:.3f}。' + (
            '探针在这批小样本上的F1点估计较高，但还不能证明有稳定优势。' if primary_f1 > direct_f1 else
            '本轮没有证明探针在F1上超过直接核查。'))
    else:
        comparison = "探针与直接核查的F1不能在当前有效标签条件下完整比较。"
    entropy_f1 = metrics['methods']['mean_entropy']['end_to_end']['f1']
    comparison += f' 更简单的生成预测熵对照F1为 {fmt(entropy_f1)}；不能只比较直接核查就宣称探针明显更好。'
    split_counts = Counter(r["split"] for r in generated)
    group_counts = {split: len({r["question_id"] for r in generated if r["split"] == split}) for split in split_counts}
    all_label_counts = Counter((p["annotation"]["stance"], p["annotation"]["evidence_relation"],
                                p["annotation"]["reference_correctness"]) for p in predictions)
    row_counts = Counter(r["condition"] for r in generated)
    total_generated_tokens = sum(r['generated_tokens'] for r in generated)
    primary_generation_seconds = sum(r['seconds'] for r in generated)
    confidence_seconds = sum(b.get('self_confidence_seconds', 0) for b in baselines.values())
    direct_seconds = sum(b.get('direct_seconds', 0) for b in baselines.values())
    lines = ["# 搜索资料不完整时的事实项风险监测：第六轮预实验结果", "",
             finding + comparison, "",
             f'本轮完成 {len(by_question)} 个 HotpotQA 比较题、{len(generated)} 份模型自己生成的回答、{len(predictions)} 个预定回答项。'
             f'留出检查只有 {group_counts.get("test", 0)} 个独立题组、{behavior["slots"]} 个预定项，'
             f'其中 {primary["n"]} 个实际断言可判定依据风险。**这是小样本试跑，不能称为稳定达到0.7，也不能称为逐词定位结果。**', "",
             "## 实际做了什么", "",
             "每题要求回答两个主体各自的一个属性，再给出比较结论。同题设置完整与缺失资料两种条件；两种条件均显示四段搜索结果，缺失条件替换其中一个主体的有效资料。使用本地 Qwen2.5-7B-Instruct NF4 自己生成答案，模型权重冻结。共同提示正常提问，未加入‘资料不足可以不答’或‘不要编造’等提示。模型主动承认不足的输出照实保留。", "",
             f'实际生成 {total_generated_tokens} 个 token，回答生成累计 {primary_generation_seconds:.1f} 秒；另有 {len(baselines)} 次同模型自评（累计 {confidence_seconds:.1f} 秒）和 {len(baselines)} 次直接核查（累计 {direct_seconds:.1f} 秒）。这些调用在原回答冻结后独立执行。时间不含全部加载与准备开销，未调用外部付费模型。', "",
             "主探针读取模型已经读入该事实项最后内容token后的第21层3584维状态。StandardScaler和逻辑回归只拟合训练集，参数固定为L2、C=0.1、balanced、liblinear、max_iter=2000、seed=20260910。Qwen不训练；没有选择部分句子的入口、没有把句级分数覆盖成逐词分数，也没有本轮测试选层或调C。", "",
             "| 划分 | 独立题组 | 回答份数 | 预定项数 | 可判定事实断言 | 风险项 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ["train", "validation", "test"]:
        part = [p for p in predictions if p["split"] == split]
        lines.append(f'| {SPLITS[split]} | {group_counts.get(split, 0)} | {split_counts.get(split, 0)} | {len(part)} | {sum(resolved(p["annotation"]) for p in part)} | {sum(p["annotation"].get("risk") == 1 for p in part)} |')
    lines += ["", "同题两个资料版本放在同一划分；共享主体、关键资料及展示背景的分组与来源检查见数据协议。训练与验证标签快照、特征文件、模型权重、阈值在测试评测前冻结。试跑只用固定第21层；第7、14、28层仅预留保存，没有在留出结果上择优。", "",
              "## 模型自然产生了什么", "",
              "‘材料无依据’与‘事实错误’分别标注：没有资料却碰巧答对，仍是需要复查的无依据陈述。主动拒答、明确推测、缺项与依据未判清不混入正常负类。多事实项以是否包含风险陈述作为项标签。", "",
              "下表是**全部试跑数据**的实际输出标签分布，不只是测试集。", "",
              "| 输出行为 | 当前材料关系 | 完整参考对错 | 项数 |", "|---|---|---|---:|"]
    for (stance, relation, correctness), n in sorted(all_label_counts.items()):
        lines.append(f'| {STANCE.get(stance, stance)} | {RELATION.get(relation, relation)} | {CORRECT.get(correctness, correctness)} | {n} |')
    lines += ['', '主分析排除了14项自然拒答与7项依据未决，共保留159项事实断言。7项未决中，5项涉及“both ... not ...”否定范围歧义，2项来自验证题Marsilea：二选一“哪个种数更多”的问法可能与另一属只有1种共同暗示答案。这些问题均在拟合或查看检测分数之前发现并记录，原输入和回答没有改写。未决并不表示正常，仍应交人工复核。详见[标注冻结说明](../data/annotation_manifest.json)与[验证标签复核](../data/validation_annotation_review.json)。', '']
    lines += ["", "以下是留出检查的所有预定项统计，分母包含主动拒答及缺项。", "",
              "| 条件 | 预定项 | 事实断言 | 主动承认不足 | 明确推测 | 缺项 | 风险断言 | 解析失败 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, b in metrics["behavior"]["by_condition"].items():
        lines.append(f'| {CONDITIONS.get(condition,condition)} | {b["slots"]} | {b["stance"].get("asserted",0)} | {b["stance"].get("abstained",0)} | {b["stance"].get("tentative",0)} | {b["stance"].get("missing",0)} | {b["risk_items"]} | {b["parse_failures"]} |')
    lines += ["", "## 留出检查：事实项风险检测", "",
              "主表覆盖所有可判定的事实断言。若方法没有可用分数／自动边界失败，按未触发复查处理，其中风险项计为漏检；缺失项数另列。AUROC和AP只在有分数的部分计算，因此必须一起看覆盖率。AP指average precision，不是用梯形法积分的PR曲线面积。无相应分母时写‘不可估计’，不填造一个0。", "",
              "| 方法 | F1 | 精确率 | 召回率 | TP | FP | FN | TN | 有效分数/可判定项 | AUROC | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for method in METHODS:
        m = metrics["methods"][method]["end_to_end"]
        c = metrics["methods"][method]["coverage"]
        lines.append(f'| {NAMES[method]} | {fmt(m["f1"])} | {fmt(m["precision"])} | {fmt(m["recall"])} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["tn"]} | {c["scorable_resolved_items"]}/{c["resolved_items"]} | {fmt(m["auroc"])} | {fmt(m["average_precision"])} |')
    common = metrics["common_scorable_comparison"]
    lines += ["", f'所有方法共同可评分的可判定项共有 {common["n"]} 个；共同子集的指标完整保存在 `metrics.json`。任何方法的自评分数无法解析都保留为缺失，没有默认为0或100分。', "",
              "| 方法 | 验证集选定阈值 | 验证F1 | 仅可评分项的留出F1 | 留出假阳性率 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        threshold = frozen["thresholds"][method]["threshold"]
        vm = validation["methods"][method]["end_to_end"]
        m = metrics["methods"][method]
        lines.append(f'| {NAMES[method]} | {fmt(threshold,8)} | {fmt(vm["f1"])} | {fmt(m["scorable_only"]["f1"])} | {fmt(m["end_to_end"]["false_positive_rate"])} |')
    lines += ["", "阈值候选为验证分数的所有不同值及全报／全不报端点；按风险F1、精确率、较高阈值依次选择。全报和全不报为固定常数对照，不调阈值。主探针在测试上没有再调参。", "",
              "## 主探针漏在哪里、误报在哪里", "",
              "| 条件 | F1 | 精确率 | 召回率 | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, m in probe["by_condition"].items():
        lines.append(f'| {CONDITIONS.get(condition,condition)} | {fmt(m["f1"])} | {fmt(m["precision"])} | {fmt(m["recall"])} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["tn"]} |')
    subtype_names = {"unsupported_reference_correct": "无依据但符合完整参考", "unsupported_reference_incorrect": "无依据且不符合完整参考",
                     "unsupported_reference_unresolved": "无依据、参考对错未判清", "contradicted": "与当前资料冲突"}
    lines += ["", "| 风险类型 | 检出数/该类项数 | 检出率 |", "|---|---|---:|"]
    for kind, m in probe["risk_subtypes"].items():
        lines.append(f'| {subtype_names[kind]} | {m["detected"]}/{m["n"]} | {pct(m["recall"])} |')
    lines += ["", "## 复查预算与同一回答内的排序", "",
              "| 方法 | 复查项/预算 | 找出风险项 | 复查精确率 | 覆盖全部风险 | 混合回答内平均AUROC |", "|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        m = metrics["methods"][method]
        top = m["top20_review"]
        mixed = m["within_mixed_answer"]
        lines.append(f'| {NAMES[method]} | {top["reviewed"]}/{top["requested_budget"]} | {top["risk_found"]} | {pct(top["precision"])} | {pct(top["recall"])} | {fmt(mixed["mean_answer_auroc"])} |')
    control = probe["both_conditions_supported_control"]
    lines += ["", "复查预算取可判定断言总数的20%向上取整，只在已评分项中排序；同分按item_id固定顺序。分母、选中项ID及覆盖率保存于指标文件。混合回答指同一份回答内同时有正常与风险项；不会把只含单类的回答硬算成AUROC。", "",
              f'主探针在两个条件均有依据、可匹配且可评分的 {control["n"]} 对回答项上，缺失条件减完整条件的平均风险分数为 {fmt(control["mean_partial_minus_complete"])}。这用于检查是否所有项的风险都随资料缺失一起升高；同一目标的实际措辞可以变化，不能直接当作内部机制因果证据。', "",
              "## 逐项误报、漏报复核", "",
              "下面保留留出部分的全部误报与漏报，包括原始回答、标签理由和依据指针；并不删除难例来提高分数。", "",
              "| 项ID | 条件 | 类型 | 模型实际回答 | 主探针分数 | 标签理由 |", "|---|---|---|---|---:|---|"]
    mistakes = [p for p in predictions if p["split"] == "test" and resolved(p["annotation"]) and
                (p["methods"]["hidden21_probe"]["prediction"] or 0) != p["annotation"]["risk"]]
    for p in mistakes:
        lines.append(f'| {md(p["item_id"])} | {CONDITIONS.get(p["condition"],p["condition"])} | {status(p)} | {md(p["text"])} | {fmt(p["methods"]["hidden21_probe"]["score"])} | {md(p["annotation"].get("rationale",""))} |')
    if not mistakes:
        lines.append("| — | — | 当前留出中未出现可判定错误（不代表泛化无误） | — | — | — |")
    # Preserve runtime/provenance files verbatim without guessing their field schema.
    manifest_paths = sorted({p for directory in [root, root / "data", out] for p in directory.glob("*.json")
                             if any(term in p.stem for term in ["manifest", "audit", "protocol", "cost", "timing"])
                             and p.name != "report_manifest.json"})
    lines += ["", "## 成本、检查与解释边界", "",
              "生成与探针读取使用同一本地Qwen，未调用额外付费大模型。自评把握和直接核查分别是对同一Qwen的额外独立请求；它们不进入原始答案生成上下文，也不用于制作标签。直接核查的三选一相对概率不是已校准的幻觉概率。", "",
              "表面特征对照包含长度、位置、项目角色、资料长度、词语重合和主体名称匹配。移除主体资料可能留下明显的名字匹配线索，探针有信号不等于读出了‘模型在猜’；若没有超过表面对照或直接核查，就不能据此宣称白盒方法更有价值。", "",
              '本次最需要防止过度解释的结果：探针F1为0.762，预测熵为0.750；只复查分数最高的7项时，探针找出5个风险项，预测熵找出6个，同模型自评找出7个。同一回答内部的10对风险/正常排序，探针、预测熵和负对数概率都全部排对；这6份混合回答实际来自5个题组。因此本轮显示可区分信号，但没有建立探针优于简单指标的稳定优势。自评F1低但排序可用，说明阈值从极小验证集迁移时也可能失效，不能直接断言自评没有检测信息。', '',
              '评分后误差诊断发现：训练中仅有2条以Yes开头的回答项，均为风险；测试中2条正确的Yes回答均被探针误报。这提示探针可能学到表述方式与标签的偶然关联，尚不能证明因果关系。测试分数、标签和划分保持原样。相关逐项记录见[事后诊断](diagnostics.json)。探针分数也没有做概率校准，0.99不能解释成99%的真实幻觉率。', '',
              "标签来自助手依据数据集材料逐项核对及辅助复核，尚不构成独立研究者人工金标准。完整参考的一致性也不等于独立核实世界事实。开放情报场景、中文材料、真正联网检索及多主体研判仍需后续独立数据验证。", "",
              f'留出标签中多事实标记已记录 {behavior["multi_fact_field_recorded"]}/{behavior["slots"]} 项，标为多事实的有 {behavior["multiple_fact_items"]} 项；若未完整记录，不能据缺省值宣称没有多事实。全部解析失败 {behavior["parse_failures"]} 项。', "",
              "本轮为30题试跑，没有执行300题扩展，也未计算把180个事实项视为独立样本的虚假精确区间。后续若扩展，当前试跑题及共享资料不得进入新的独立测试。", "",
              '下一轮优先改数据：去掉可能暗示答案的二选一问法，覆盖有依据的肯定回答与无依据的否定回答，再按主体和来源分组留出新题；安排独立研究者复核标签。之后在同一数据上比较探针、概率和熵，使用新验证集选择层与正则化参数，并在新测试集报告置信区间。当前问题不会靠在这6个测试题上继续调阈值来解决。', '',
              "实际运行与工程检查记录："]
    for path in manifest_paths:
        relative = path.relative_to(out) if path.is_relative_to(out) else Path("..") / path.relative_to(root)
        lines.append(f'- [{path.name}]({relative.as_posix()})')
    lines += ["", "## 全部事实项索引", "",
              "下表包含训练、验证和留出全部项；训练指标仅用于排错，不能用作效果结论。点击完整案例文件可按题目并排查看完整／缺失资料、原始回答、所有标签、每种方法分数和实际提示。", "",
              "[打开全部题目的证据与回答](examples.html) · [冻结记录](freeze.json) · [留出完整指标](metrics.json) · [验证指标](validation.json)", "",
              "| 项ID | 划分 | 条件 | 实际回答 | 标注 | 探针分数 | 判定 |", "|---|---|---|---|---|---:|---|"]
    for p in sorted(predictions, key=lambda p: (p["split"], p["question_id"], p["condition"], p["item_index"])):
        lines.append(f'| {md(p["item_id"])} | {SPLITS.get(p["split"],p["split"])} | {CONDITIONS.get(p["condition"],p["condition"])} | {md(p["text"])} | {label_text(p["annotation"])} | {fmt(p["methods"]["hidden21_probe"]["score"])} | {status(p)} |')
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_html(out, generated, predictions, inputs, references, baselines, frozen, finding, comparison)
    writej(out / "report_manifest.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "questions": len(by_question), "responses": len(generated), "items": len(predictions),
        "test_error_items": [p["item_id"] for p in mistakes],
        "source_results_sha256": {name: sha(out / name) for name in ["metrics.json", "freeze.json", "predictions.jsonl", "development_predictions.jsonl"]},
        "report_sha256": sha(out / "REPORT.md"), "html_sha256": sha(out / "examples.html"),
        "script_sha256": sha(Path(__file__)), "all_item_ids": sorted(p["item_id"] for p in predictions)})
    print(f"REPORT COMPLETE: {len(by_question)} questions, {len(predictions)} items; {len(mistakes)} held-out errors", flush=True)


def render_html(out, generated, predictions, inputs, references, baselines, frozen, finding, comparison):
    by_row = defaultdict(list)
    for p in predictions:
        by_row[p["row_id"]].append(p)
    grouped = defaultdict(list)
    for row in generated:
        grouped[row["question_id"]].append(row)
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1"><title>第六轮：全部证据与模型回答</title>',
             '''<style>body{max-width:1480px;margin:26px auto;padding:0 18px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#1f2937;background:#f8fafc}h1,h2,h3{line-height:1.3}p{white-space:pre-wrap}a{color:#075985}header{max-width:1050px}section.question{background:white;padding:22px;margin:28px 0;border:1px solid #cbd5e1;border-radius:8px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.condition{min-width:0}.item{padding:12px 15px;margin:14px 0;background:#f8fafc;border-left:5px solid #64748b}.item.risk{border-color:#be123c;background:#fff1f2}.item.supported{border-color:#15803d;background:#f0fdf4}.item.mistake{outline:2px solid #c2410c}.badge{display:inline-block;padding:1px 8px;border-radius:4px;background:#e2e8f0;font-size:13px;margin-right:6px}.alert{background:#ffedd5;color:#9a3412}details{margin:12px 0}summary{cursor:pointer;color:#0c4a6e}pre{white-space:pre-wrap;word-break:break-word;background:#f1f5f9;padding:14px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #cbd5e1;padding:6px;text-align:left;word-break:break-word}select,input{font:inherit;margin:6px;padding:5px}blockquote{margin:10px 0;padding:5px 14px;border-left:3px solid #94a3b8;white-space:pre-wrap}.muted{font-size:13px;color:#475569}.toolbar{position:sticky;top:0;background:#f8fafcf2;padding:8px;border-bottom:1px solid #cbd5e1;z-index:2}.hidden{display:none}@media(max-width:950px){.columns{grid-template-columns:1fr}}@media print{.toolbar{display:none}.columns{display:block}details{display:block}section.question{break-before:page}}</style></head><body>''',
             '<header><h1>搜索资料不完整：全部证据与模型回答</h1>',
             f'<p>{esc(finding.replace("**", ""))}{esc(comparison)}</p>',
             '<p>绿色左边框＝有依据；红色左边框＝风险断言；灰色＝主分类外。橙色外框＝探针误报或漏报。颜色依据评测标签，不能理解为模型自己生成的解释。完整／缺失条件仅在评测页面显示，没有告诉生成模型。</p>',
             '<p>完整参考与输入资料分开显示：“无依据但答对”仍可报风险。标签由助手依据材料核对，尚不是独立研究者金标准。所有方法原始值和提示均可展开查看；本页面无外部资源或联网请求。</p></header>',
             '<div class="toolbar"><label>划分 <select id="split"><option value="all">全部</option><option value="train">训练</option><option value="validation">验证</option><option value="test">留出检查</option></select></label><label><input id="errors" type="checkbox">仅看有探针误报／漏报的题目</label><label>搜索 <input id="search" placeholder="主体、题目或ID"></label><span id="count"></span></div>']
    for qid, rows in sorted(grouped.items(), key=lambda kv: (kv[1][0]["split"], kv[0])):
        reference = references[qid]
        split = rows[0]["split"]
        question_items = [p for row in rows for p in by_row[row["row_id"]]]
        mistakes = any(resolved(p["annotation"]) and (p["methods"]["hidden21_probe"]["prediction"] or 0) != p["annotation"]["risk"] for p in question_items)
        parts.append(f'<section class="question" id="q-{esc(qid)}" data-split="{esc(split)}" data-errors="{int(mistakes)}"><h2>{esc(reference.get("original_question",qid))}</h2>')
        parts.append(f'<p class="muted">题目ID：{esc(qid)}　划分：{esc(SPLITS.get(split,split))}　属性：{esc(reference.get("attribute",""))}</p>')
        parts.append('<details><summary>评测侧完整参考（没有给生成模型额外展示）</summary>')
        parts.append(f'<p>原题答案：{esc(reference.get("original_answer",""))}；主体：{esc(reference.get("subjects",[]))}；移除主体索引：{esc(reference.get("removed_subject_index",""))}</p>')
        for ref_item in reference.get("items", []):
            parts.append(f'<h3>回答项 {esc(ref_item.get("item_index",""))}：参考 {esc(ref_item.get("reference_answer",""))}</h3>')
            for ev in ref_item.get("evidence", []):
                parts.append(f'<blockquote>{esc(ev.get("text",""))}</blockquote><p class="muted">{esc(ev.get("title",""))}，句号索引 {esc(ev.get("sent_id",""))}</p>')
            parts.append(f'<p>{esc(ref_item.get("rationale",""))}</p>')
        parts.append(f'<details><summary>完整参考原始记录</summary><pre>{esc(json.dumps(reference,ensure_ascii=False,indent=2))}</pre></details></details><div class="columns">')
        for row in sorted(rows, key=lambda row: (row["condition"] != "complete", row["condition"])):
            source = inputs[row["row_id"]]
            parts.append(f'<div class="condition"><h3>{esc(CONDITIONS.get(row["condition"],row["condition"]))}</h3><p class="muted">{esc(row["row_id"])}</p>')
            parts.append('<details><summary>查看本次输入的四段搜索资料</summary>')
            for n, passage in enumerate(source.get("passages", []), 1):
                parts.append(f'<h4>搜索结果 {n}：{esc(passage.get("title",""))}</h4><p>{esc(passage.get("text", " ".join(passage.get("sentences",[]))))}</p>')
            parts.append('</details><details><summary>查看完整原始提示与原始回答</summary>')
            parts.append(f'<pre>System: {esc(source.get("system",""))}\n\n{esc(source.get("prompt",""))}</pre><h4>原始回答</h4><pre>{esc(row["response"])}</pre></details>')
            for p in sorted(by_row[row["row_id"]], key=lambda p: p["item_index"]):
                a = p["annotation"]
                r = a.get("risk")
                score = p["methods"]["hidden21_probe"]
                mistake = resolved(a) and (score["prediction"] or 0) != r
                css = ("risk" if r == 1 else "supported" if r == 0 else "") + (" mistake" if mistake else "")
                parts.append(f'<article class="item {css}" id="item-{esc(p["item_id"])}"><span class="badge">项 {p["item_index"]}</span><span class="badge {"alert" if score["prediction"] == 1 else ""}">{esc(status(p))}</span>')
                parts.append(f'<p><strong>{esc(p["text"] or "（没有可解析的回答内容）")}</strong></p><p>{esc(label_text(a))}</p>')
                parts.append(f'<p>探针风险分数：<strong>{fmt(score["score"],6)}</strong>；阈值：{fmt(score["threshold"],6)}；自动边界可用：{esc(p.get("parse_ok",False))}</p>')
                parts.append(f'<p>标注依据：{esc(a.get("rationale",""))}</p><p class="muted">依据指针：{esc(json.dumps(a.get("evidence_refs",[]),ensure_ascii=False))}；标注者：{esc(a.get("annotator","未记录"))}</p>')
                parts.append('<details><summary>所有方法的分数与判定</summary><table><tr><th>方法</th><th>分数</th><th>阈值</th><th>结果</th></tr>')
                for method in METHODS:
                    value = p["methods"][method]
                    result = "未能评分：" + str(value.get("missing_reason")) if value["prediction"] is None else "报风险" if value["prediction"] else "不报风险"
                    parts.append(f'<tr><td>{esc(NAMES[method])}</td><td>{fmt(value["score"],8)}</td><td>{fmt(value["threshold"],8)}</td><td>{esc(result)}</td></tr>')
                parts.append('</table></details>')
                raw = {"item": p, "baseline_raw": baselines.get(p["item_id"])}
                parts.append(f'<details><summary>原始标注、字符边界与基线输出</summary><pre>{esc(json.dumps(raw,ensure_ascii=False,indent=2))}</pre></details></article>')
            parts.append('</div>')
        parts.append('</div></section>')
    parts.append(f'<details><summary>冻结阈值与训练记录</summary><pre>{esc(json.dumps(frozen,ensure_ascii=False,indent=2))}</pre></details>')
    parts.append('''<script>const groups=[...document.querySelectorAll('section.question')];function filter(){const s=document.getElementById('split').value,e=document.getElementById('errors').checked,q=document.getElementById('search').value.trim().toLowerCase();let n=0;for(const g of groups){const visible=(s==='all'||g.dataset.split===s)&&(!e||g.dataset.errors==='1')&&(!q||g.textContent.toLowerCase().includes(q));g.classList.toggle('hidden',!visible);if(visible)n++;}document.getElementById('count').textContent=`显示 ${n}/${groups.length} 个题目`;}for(const id of ['split','errors','search'])document.getElementById(id).addEventListener('input',filter);filter();</script></body></html>''')
    (out / "examples.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
