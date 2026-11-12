import json,html
from collections import Counter
from common5 import R5,R4,readl

LABEL={'automatic':'原自动选点','oracle_sentence':'人工指定错误句子','oracle_span':'再圈准完整错误片段'}
METHOD={'original_attention':'原注意力风险','direct_B':'同模型直接自检分数','frozen_internal_probe':'现有内部状态探针'}

def main():
    pm=json.loads((R5/'results/pipeline_metrics.json').read_text());um=json.loads((R5/'results/unit_metrics.json').read_text());changes=json.loads((R5/'results/changes.json').read_text());lengths=json.loads((R5/'results/length_controls.json').read_text());verdict=json.loads((R5/'results/direct_verdict_counts.json').read_text());records=readl(R5/'data/readouts/records.jsonl');counts=Counter(r['origin'] for r in records);audit=json.loads((R5/'results/audit.json').read_text());assert audit['passed']
    auto,sen,span=pm;probe=next(m for m in um if m['mode']=='span' and m['method']=='frozen_internal_probe')
    lines=['# 人工选点诊断：选择位置，还是判断对错？','',
      f'人工指定错误所在句子后，同一批50条回答的F1从 **{auto["f1"]:.3f}提升到{sen["f1"]:.3f}**，检出错误从{auto["tp"]}/20提升到{sen["tp"]}/20，正常误报维持{sen["fp"]}/30。进一步圈准完整错误片段后仍为{span["f1"]:.3f}。这支持“当前自动选点漏掉重要信号”这一诊断，但没有证明后续判断已经解决。',
      '', '**这是利用人工错误范围辅助选点的诊断，不是自动检测达标，也不是逐token F1升至0.737。** 原自动检测结果仍为回答级F1 0.588、逐token F1 0.130。本轮没有测一份新的独立泛化测试。',
      '', '## 同一批回答，只改变选点','',
      '复用上一轮50条新闻（20错误、30无错误）。Qwen、探针权重、内部状态PCA、回答级汇总方式、判定阈值、每条核查次数均固定。对错误回答，用最早一个人工错误标注所在句子替换原选句中最低风险的一句（若已选中则保持）；再把该句的事实目标改为完整人工错误范围。正常回答输入完全不变。模型看到原文、待核查句子和中性目标提示，不看到正确/错误标签或标准答案。',
      '', '| 条件 | 回答级F1 | 精确率 | 召回率 | 检出错误/20 | 误报正常/30 | 每条平均核查次数 |','|---|---:|---:|---:|---:|---:|---:|']
    for m in pm:lines.append(f'| {LABEL[m["mode"]]} | {m["f1"]:.3f} | {m["precision"]:.1%} | {m["recall"]:.1%} | {m["tp"]}/20 | {m["fp"]}/30 | {m["mean_calls"]:.2f} |')
    lines+=['',f'人工选句后新增检出的样例ID为 {", ".join(changes["oracle_sentence"]["recovered_error_ids"])}，原来检出的错误没有丢失。仍漏掉{sen["fn"]}条，说明识别、分数汇总与阈值仍有改进空间。人工目标只使用每条回答的最早错误，不宣称这是所有可能人工辅助方式的理论上限。',
      '', '## 选准位置后，能否区分正确与错误？','',
      '另取这批数据的20个最早人工错误片段，为每个片段匹配一个来自完全无错回答、原生成器相同、长度尽量接近的正确对照片段；每个来源仅出现一次。匹配不使用探针风险或新模型得分。核查其所在句子及指定片段两种输入。正确片段沿用“回答没有幻觉标注”的数据集判定，不是本轮独立人工逐事实核验。',
      '', '这40个单位是20错＋20对。单位判定阈值分别在之前的验证集90个匹配单位（45错＋45对）确定，没有重新训练探针或用本批结果调阈值。它们的F1是“给定位置后判断对错”的F1，不可与逐token F1或回答级F1直接比较。全部判错的单位F1已有0.667，所以还需看AUROC及精确率、召回率。',
      '', '| 给定范围 | 方法 | 单位F1 | 精确率 | 召回率 | AUROC | AUROC 95%区间 | 错片段风险高于配对正确片段 |','|---|---|---:|---:|---:|---:|---|---:|']
    for m in um:
        ci=m['auc_95pct'];lines.append(f'| {"整句" if m["mode"]=="sentence" else "指定片段"} | {METHOD[m["method"]]} | {m["f1"]:.3f} | {m["precision"]:.1%} | {m["recall"]:.1%} | {m["auc"]:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {m["paired_ranking"]:.1%} |')
    lines+=['', f'指定片段后，现有内部探针识别{probe["tp"]}/20个错误片段，误报{probe["fp"]}/20个正确片段，单位F1为{probe["f1"]:.3f}、AUROC为{probe["auc"]:.3f}。说明存在可读出的区分信号；不等于模型已经准确理解了每个事实。',
      '', '同模型直接自检分数的AUROC在整句和片段上均不低于内部探针，因此这次没有证明内部探针拥有独有优势。两者片段F1有差异，也可能来自阈值选择和校准，不能据此归因于内部状态。',
      '', f'若直接取A/B/C概率最高的一项作为模型回答，20个错误中，整句核查仅{verdict["sentence"]["1"]["contradicted"]}个、片段核查仅{verdict["span"]["1"]["contradicted"]}个选择“冲突”；两种情况下20个正常对照都选择“支持”。这解释了为什么风险排序有信号，不代表模型直接给出的判断可靠。',
      '', '## 长度与实现检查','', '| 范围 | 错误平均字符数 | 正确平均字符数 | 只按长度排序的AUROC |','|---|---:|---:|---:|']
    for m in lengths:lines.append(f'| {"整句" if m["mode"]=="sentence" else "片段"} | {m["mean_error_length"]:.1f} | {m["mean_clean_length"]:.1f} | {m["length_only_auc"]:.3f} |')
    lines+=['', '长度对照接近随机，未发现简单的长度差异能够解释当前区分度。配对还按原生成器控制；仍可能有标注片段与规则选出的正确片段在语义上的差异，不作完全因果隔离或跨任务泛化结论。区间按20组匹配对做2000次重采样，保留同一对的依赖。',
      '',f'共{len(records)}个唯一查询，其中{counts["exact prior prompt cache"]}个复用完全相同提示的缓存，{counts["fresh same-model forward"]}个新前向计算。只使用本地Qwen2.5-7B NF4；无其他大模型调用、无自由生成、无新依赖。缓存数组、提示、原始标签、字符范围、旧权重、相同核查次数、正常输入不变、全部指标均已复核；新读数抽查重放一致，读取状态没有改变模型输出。',
      '', '## 本轮结论','',
      '自动选点是已观察到的瓶颈之一：人工选句补回4条漏检。下一步优先改进候选句覆盖和完整事实关系的筛选，同时检查风险汇总与阈值；人工精确圈词在本次固定流水线中没有额外收益。仍需保留简单自检作为对照，检验白盒探针是否带来额外价值。',
      '', '[所有40个配对单位及50条回答判定](examples.html) · [回答级诊断指标](pipeline_metrics.json) · [给定位置后的单位指标](unit_metrics.json) · [冻结协议](../protocol.json) · [阈值记录](unit_thresholds.json) · [复核记录](audit.json)']
    (R5/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    rows={r['id']:r for r in readl(R5/'data/rows.jsonl')};qs={q['query_id']:q for q in readl(R5/'data/queries.jsonl')};cases=readl(R5/'results/unit_predictions.jsonl');pipeline=readl(R5/'results/pipeline_predictions.jsonl')
    parts=['<!doctype html><meta charset="utf-8"><title>人工选点诊断全部样例</title><style>body{max-width:1050px;margin:28px auto;font:16px/1.8 system-ui;padding:0 18px;color:#234}article{border-top:1px solid #bbc;padding:12px 0}p{white-space:pre-wrap}mark{background:#ffe0a8}td,th{padding:6px 12px;text-align:left}summary{cursor:pointer}</style><h1>人工选点诊断</h1><p>高亮是指定核查范围，不代表模型自己找到了该位置。标签仅用于评测，不进入模型提示。</p><h2>全部40个配对单位</h2>']
    for c in sorted(cases,key=lambda a:(a['pair_id'],-a['label'])):
        r=rows[c['id']];q=qs[c['span_query']];lo,hi=q['target_span'];a,b=q['sentence_span'];text=html.escape(r['response'][a:lo])+'<mark>'+html.escape(r['response'][lo:hi])+'</mark>'+html.escape(r['response'][hi:b]);v=c['methods']['span_frozen_internal_probe'];parts.append(f'<article><h3>{html.escape(c["pair_id"])} / ID {r["id"]} / 标签{c["label"]}</h3><p>{text}</p><p>片段探针风险{v["score"]:.4f}，单位判定{int(v["pred"])}</p><details><summary>完整新闻原文</summary><p>{html.escape(r["evidence"])}</p></details></article>')
    parts+=['<h2>全部50条回答判定</h2><table><tr><th>ID</th><th>真实标签</th><th>自动选点</th><th>人工选句</th><th>人工圈片段</th></tr>']
    for p in pipeline:parts.append('<tr>'+''.join(f'<td>{v}</td>' for v in [p['id'],p['label']]+[int(p['methods'][m]['pred']) for m in ['automatic','oracle_sentence','oracle_span']])+'</tr>')
    parts.append('</table>');(R5/'results/examples.html').write_text('\n'.join(parts),encoding='utf8');print('ORACLE REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
