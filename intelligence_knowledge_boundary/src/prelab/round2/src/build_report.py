"""Assemble observed results, provenance and complete heldout risk views."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,readl

LABELS={'before':'HaMI ori / before','after':'HaMI ori / after','lookback':'Attention / weak',
 'support':'Evidence change / weak','support_lookback':'Evidence + attention / weak','uncertainty':'Uncertainty / weak',
 'supervised_before':'States before / local labels','supervised_after':'States after / local labels','supervised_lookback':'Attention / local labels',
 'last_linear':'Last state / linear','mean_linear':'Mean state / linear','nll_topk':'Surprisal','source_copy':'Source string match','text':'Text TF-IDF',
 'few_local_10':'Attention / 10% local training labels','few_local_25':'Attention / 25% local training labels',
 'few_local_50':'Attention / 50% local training labels','few_local_100':'Attention / all local labels, bag validation',
 'fact_after':'Three-fact MIL / states','fact_support_lookback':'Three-fact MIL / evidence + attention',
 'local_mlp':'Attention MLP / local labels','local_mlp_alarm':'Attention MLP / local labels + alarm loss'}
WANTED=['before','after','last_linear','mean_linear','lookback','support_lookback','fact_after','fact_support_lookback','nll_topk','source_copy','text','few_local_10','few_local_25','few_local_50','few_local_100','supervised_before','supervised_after','supervised_lookback','local_mlp','local_mlp_alarm']
def fmt(x): return '—' if x is None else f'{x:.3f}'

def heatmap(name):
    dest=ROOT/f'results/{name}'; run=ROOT/f'data/{name}'
    rows={r['id']:r for r in readl(run/'labeled.jsonl')}; anns={a['id']:a for a in readl(run/'local_annotations.jsonl')}
    ci=json.loads((dest/'confidence_intervals.json').read_text()); selected=ci['selected_names']
    style='body{font:16px/1.8 system-ui;max-width:1080px;margin:35px auto;padding:0 25px;color:#203040}article{border-top:1px solid #bbc;padding:24px 0}mark{color:inherit;padding:2px 0}details{margin:12px 0}small{color:#567}summary{cursor:pointer}p{white-space:pre-wrap}'
    pieces=[f'<!doctype html><meta charset="utf-8"><title>{name} 全部留出结果</title><style>{style}</style>',
        f'<h1>{name}：全部留出结果</h1><p>红色越深，探针分数越高；蓝色下划线表示参考标注的错误位置。分数尚未校准，不是真实错误概率。展示方法由验证集整段指标选定：{html.escape(ci["selected_by_validation"])}，固定三个种子平均。按 ID 排序，未挑选好看的案例。</p>']
    for pred in sorted(readl(dest/'predictions.jsonl'),key=lambda r:r['id']):
        r=rows[pred['id']]; ann=anns[r['id']]; scores=np.mean([pred['predictions'][n]['tokens'] for n in selected],axis=0)
        n=len(r['response']); colors=np.zeros(n); truth=np.zeros(n,bool)
        for (a,b),s in zip(pred['offsets'],scores): colors[a:b]=np.maximum(colors[a:b],s)
        for span in ann['spans']: truth[span['start']:span['end']]=True
        rendered=[]; i=0
        while i<n:
            j=i+1; color=int(220*np.clip(colors[i],0,1))
            while j<n and int(220*np.clip(colors[j],0,1))==color and truth[j]==truth[i]: j+=1
            underline='text-decoration:underline 3px #236fa8;' if truth[i] else ''
            rendered.append(f'<mark style="background:rgb(255,{255-color//2},{255-color//2});{underline}">{html.escape(r["response"][i:j])}</mark>'); i=j
        text=r.get('question') or '\n'.join(q['question'] for q in r.get('qas',[]))
        evidence=r.get('evidence') or '无外部证据；仅按基准目标答案评估。'
        pieces.append(f'<article><h2>{html.escape(r["id"])}</h2><small>整段标签：{r["label"]}；{html.escape(r["label_status"])}</small><p>{html.escape(text)}</p><p>{"".join(rendered)}</p><details><summary>参考证据／标注</summary><p>{html.escape(evidence)}</p><pre>{html.escape(json.dumps(ann,ensure_ascii=False,indent=2))}</pre></details></article>')
    (dest/'token_risk_report.html').write_text('\n'.join(pieces),encoding='utf8')

def main():
    runs=[p.parent.name for p in sorted((ROOT/'results').glob('*/summary.json'))]
    tables=[]; summaries={}
    for name in runs:
        values=json.loads((ROOT/f'results/{name}/summary.json').read_text()); summaries[name]={m['method']:m for m in values}
        manifest=json.loads((ROOT/f'data/{name}/label_manifest.json').read_text()); data=readl(ROOT/f'data/{name}/labeled.jsonl')
        counts={s:{'total':sum(r['split']==s for r in data),'usable':sum(r['split']==s and r['label'] is not None for r in data),'errors':sum(r['split']==s and r['label']==1 for r in data)} for s in ['train','val','test']}
        tables.extend(['',f'## {name}','',f'划分与可用标签：`{json.dumps(counts,ensure_ascii=False)}`。','',
            '| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |',
            '|---|---|---:|---:|---:|---:|'])
        for key in WANTED:
            if key not in summaries[name]: continue
            m=summaries[name][key]; supervision='局部位置' if key.startswith(('supervised','few_local','local_mlp')) else '整段／无需训练'
            tables.append(f'| {LABELS.get(key,key)} | {supervision} | {fmt(m["bag_auc"])} | {fmt(m.get("local_auc"))} | {fmt(m.get("fact_auc"))} | {fmt(m.get("top10_precision"))} |')
        tables.extend(['',f'完整结果：[metrics.csv]({name}/metrics.csv)，[置信区间]({name}/confidence_intervals.json)。',''])
        if not name.startswith('trivia'):
            tables.extend(['所有正常和错误回答一起评估：仅使用验证集正常回答的最高 token 分数，将验证误报控制在约 5% 以内，再固定阈值测试。以下检出率是“能否对错误回答发出至少一次警报”，与前表的条件定位指标不同。','',
                '| 方法 | 全部 token AUROC | 正常回答误报率 | 错误回答检出率 | 全局最高 10% token 精确率 |',
                '|---|---:|---:|---:|---:|'])
            for key in ['before','support_lookback','fact_after','fact_support_lookback','supervised_after','supervised_lookback','local_mlp','local_mlp_alarm']:
                m=summaries[name].get(key)
                if m is None: continue
                tables.append(f'| {LABELS[key]} | {fmt(m.get("global_token_auc"))} | {fmt(m.get("test_clean_answer_alarm"))} | {fmt(m.get("test_error_answer_recall"))} | {fmt(m.get("global_top10_precision"))} |')
            tables.append('')
            heatmap(name); tables.append(f'[全部测试回答及错误位置]({name}/token_risk_report.html)。')
    fig,axs=plt.subplots(len(runs),2,figsize=(13,max(4,len(runs)*3.5)),squeeze=False)
    methods=['before','after','support_lookback','nll_topk','supervised_after','supervised_lookback']
    for i,name in enumerate(runs):
        for j,metric in enumerate(['bag_auc','local_auc']):
            valid=[k for k in methods if k in summaries[name] and summaries[name][k].get(metric) is not None]
            if not valid: axs[i,j].axis('off'); continue
            vals=[summaries[name][k][metric] for k in valid]
            axs[i,j].barh([LABELS[k] for k in valid],vals,color=['#de9b4b' if k.startswith('supervised') else '#317f93' for k in valid])
            display=name+' (matching labels only)' if name.startswith('rag') else name
            axs[i,j].axvline(.5,ls='--',color='#888'); axs[i,j].set_xlim(0,1); axs[i,j].set_title(display+' — '+metric)
            for yy,value in enumerate(vals): axs[i,j].text(value+.01,yy,f'{value:.3f}',va='center',fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT/'results/comparison.png',dpi=160); plt.close(fig)
    required={'news_small','news_large','trivia_small','trivia_large','rag_large'}
    pending=list(sorted(required-set(runs)))
    for name in runs:
        if not name.startswith('trivia') and 'supervised_lookback' not in summaries[name]: pending.append(name+' 局部监督')
        if name.startswith('news') and 'few_local_100' not in summaries[name]: pending.append(name+' 标注预算对照')
    state='五组主实验已完成；完整性核验见报告末尾' if not pending else '进行中；缺少：'+', '.join(pending)
    findings=[]
    if 'news_large' in summaries and 'trivia_large' in summaries:
        large=summaries['news_large']; trivia=summaries['trivia_large']
        findings.extend(['## 最重要的结果','',
            f'1. 短问答的整段正确性可以检测：7B 的读取答案后状态探针 AUROC 为 {fmt(trivia["after"]["bag_auc"])}，简单词概率对照为 {fmt(trivia["nll_topk"]["bag_auc"])}。这是参考答案匹配任务，不是逐 token 幻觉定位；不能与论文不同数据和模型的数值直接比较。',
            f'2. 人工新闻上的弱监督定位仍弱：7B HaMI-before 的回答内 token AUROC 为 {fmt(large["before"]["local_auc"])}；直接提供错误位置标签后的状态线性探针为 {fmt(large["supervised_after"]["local_auc"])}。有局部可读信号，但当前整段监督未稳定提取它。',
            f'3. 定位排序有效不代表能可靠报警：同一个监督状态探针在原测试集正常回答误报率为 {large["supervised_after"]["test_clean_answer_alarm"]:.1%}，错误回答检出率仅 {large["supervised_after"]["test_error_answer_recall"]:.1%}。该集只有 25 条错误回答，不能把小样本点估计当成部署表现。',
            '4. 三事实自动标签不够可靠：逐条对照材料审阅测试集全部 14 条候选错误，只有 4 条明确事实冲突，3 条实际受材料支持，7 条含糊、答非所问或问题不当。因此这组仅作自动匹配诊断；即使表中定位指标高，也不能证明事实性幻觉检测有效。审阅由执行助手完成，未取得独立双人标注。',
            '', 'AUROC 是排序指标，0.5 约为随机；0.69 不代表 69% 准确率。',''])
        findings.extend(['## 可以继续推进的切口','',
            '最有依据的方向是“有限位置标注下的证据冲突定位”，先辅助人工复核，再验证自动报警。不要把现阶段工作命名为已经实现了可靠的联网幻觉监测。',
            '', '| 训练回答中使用位置标注的数量 | 0.5B 定位 AUROC | 7B 定位 AUROC |', '|---:|---:|---:|'])
        for fraction,n in [(10,24),(25,60),(50,120),(100,240)]:
            findings.append(f'| {n} | {fmt(summaries["news_small"][f"few_local_{fraction}"]["local_auc"])} | {fmt(large[f"few_local_{fraction}"]["local_auc"])} |')
        findings.extend(['','上述对照仅使用训练集所选回答的位置标签，验证集只用整段标签，三种子平均。当前更一致的变化来自增加位置标注，扩大模型没有同样稳定的收益。这是探索线索，尚非新算法或已证实的显著提升。',
            '', '下一阶段应先建“可信材料—自然生成回答—事实冲突位置”的小型人工核验集，并按事件隔离。先限定人物、地点、时间、数量等明确字段；把材料支持、材料冲突、材料不足分开，后两者不能混标。之后固定标注预算，对比零位置标注、少量位置标注及完整位置标注；以固定误报下的检出率和人工复核命中率为主要目标。此为后续研究建议，本轮没有再启动这些实验。',''])
    confirmation=ROOT/'results/confirmation/confirmation_summary.json'
    if confirmation.exists():
        findings.extend(['## 未使用过的新闻复核','',
            '额外冻结 110 条人工标注新闻（80 正常、30 错误），来源与原 394 条训练／验证／测试均无重叠。它们来自官方训练部分，但在本研究中全程留出；不是额外的官方测试集。',
            '', '在查看原测试结果后，仅增加两个受监督对照：同一个注意力 MLP 学错误位置，以及额外惩罚正常回答的最高风险。系数固定为 0／0.5，三种子；用原验证集的低误报检出率选训练步。新留出集不用于拟合、选参数、选择方法或校准阈值。这个扩展是探索性的，不声称算法原创。',
            '', '| 方法 | 回答内定位 AUROC | 全体 token AUROC | 正常回答误报率 | 错误回答检出率 |', '|---|---:|---:|---:|---:|'])
        for m in json.loads(confirmation.read_text()):
            findings.append(f'| {LABELS.get(m["method"],m["method"])} | {fmt(m["local_auc"])} | {fmt(m["global_token_auc"])} | {m["test_clean_answer_alarm"]:.1%} | {m["test_error_answer_recall"]:.1%} |')
        findings.extend(['', '复核结果：位置监督 MLP 的定位 AUROC 为 0.709，同特征整段监督为 0.570；按来源重采样的差值 95% 区间约为 [0.060, 0.216]。这支持位置监督有用，但两者的监督信息不同，不能作为新算法公平胜出。与已经使用位置标签的线性探针相比，差值区间约为 [-0.009, 0.050]，没有证明 MLP 稳定更好。',
            '', '额外误报惩罚未形成可靠突破：错误回答检出率由 7.8% 升至 17.8%，但正常回答误报也从 5.4% 升至 9.2%；局部排序改善的差值区间包含 0。不能仅挑检出率提高来宣称抗幻觉监测成功。',
            '', '[每个种子的完整数值](confirmation/metrics.json)，[来源重采样区间](confirmation/confidence_intervals.json)，[冻结的模型哈希](confirmation/frozen_models.json)，[数据来源及隔离](../data/news_confirmation/manifest.json)。训练种子均值不等于独立重复抽取的数据集；30 条错误的检出率波动较大。验证集仅 40 条正常回答，约 5% 的经验阈值不保证现实误报率也不超过 5%。',''])
    audits=['final_audit.json','alert_metric_audit.json','generation_audit_large.json','generation_audit_small.json','token_repair_rag_large.json']
    audit_lines=['## 完整性与追溯','']
    for name in audits:
        path=ROOT/'results'/name
        status='待完成' if not path.exists() else ('通过' if json.loads(path.read_text()).get('passed') else '未完全通过，见具体差异')
        audit_lines.append(f'- [{name}]({name})：{status}。')
    audit_lines.extend(['', 'RAG 全部 1,050 条保存生成时实际 token ID，并按原 ID 重放；11 条因解码后重新切词不同而修复。此前 Trivia 没有完整保存生成轨迹，只进行了固定抽样的同种子再生成检查；不能把抽样通过写成所有回答实时状态已逐一验证。',
        '', '模型权重与数据哈希见 [模型清单](model_manifest.json) 和各数据清单；代码哈希包含在最终完整性核验中。指标核验不等于标签真实性、因果机制或现实部署有效性证明。',''])
    intro=f'''# 第二轮：更大模型、自己生成的回答和基线审计

状态：{state}。

本轮将问题拆开验证：模型和数据是否不匹配，token 时刻是否正确，以及整段标签能否提供足够的定位监督。基线实现范围见 [BASELINES.md](../BASELINES.md)。没有把本地适配称为顶会原始实验的完整复现。

{chr(10).join(findings)}

![各实验指标](comparison.png)

- 新闻：394 条 RAGTruth 人工标注回答，同一批输入分别由 0.5B 和 7B 重放；属于受控表征比较。
- TriviaQA：每个模型独立生成 2,700 个短答，按问题隔离。此项评估参考目标答案的正确性，没有可靠的逐 token 错误金标。
- 三事实 RAG：7B 按 1,050 段文章、每段三个问题生成。训练只用整段候选错误标签；局部标签来自单题答案匹配，和人工事实核验分开解释。0.5B 只做了流程试跑，由于格式与标签可用性差，没有完成此项对比，见 [停跑依据](rag_small_pilot_decision.json)。
- 7B 使用第三方 Unsloth NF4 量化版，模型规模与精度同时变化，不能把差异完全归因于参数量。
- 弱监督方法报告三种子的均值；直接局部监督是单独的诊断对照。局部金标用于监督训练的结果不能冒称只用了整段标签。
- 高分是相对风险，未作概率校准。定位是在完整回答上离线排名；生成前和生成后状态分开报告。
- NLL 与证据概率对照依赖已经选出的 token；这些组合方法属于输出后监测，不是提前预测具体错误词。

## 标签与评估限制

新闻的 Evident Conflict 是相对提供证据的明显冲突；不等于重新核验了全部现实事实。固定材料模拟联网检索后的阅读阶段，没有运行实时搜索代理。

自动匹配已排除问题复述导致的别名误匹配、部分词形歧义、拒答及多句附带断言，但仍可能误判语义等价、含糊问题和错误参考。抽查说明见 [标签审计](label_rule_audit.json)、[三事实逐条审阅](rag_large_label_review.json)、[大模型短答审阅](trivia_large_label_review.json)。不能将 QA 候选错误的高检出率直接等同于开源情报事实性幻觉检出率。

TriviaQA 是历史问答基准，其参考答案不等于 2026 年的现时事实；未逐题重做时间有效性核验。

整段分类和回答内部定位分别统计。后者在同时含正常与错误 token 的回答中计算，随机排序约为 0.5；RAG 只评估可判定答案字符串内部，排除 JSON 标点。三事实排序在同一回答既有正确事实又有候选错误时计算。置信区间按来源／文章／问题聚类重采样，不能把 token 当作独立样本扩大显著性。

另报告全体回答的 token 排序与固定阈值报警。条件定位高，不代表能够区分哪一段回答真的含错；误报控制与错误回答检出率必须一起看。报警阈值只使用验证集的正常整段标签，不读取验证集错误位置。事实片段方法需要等该片段完整输出后才能计算分数。

RAG 的局部金标把错误答案字符串整体标为错误，是事实片段粒度；新闻则使用原始错误字符区间。两者不能作为同一定位难度直接横向比较，也不能把事实片段内的统一分数解释为已经找出了最小错误词。

更换特征、局部监督的比较属于探索性分析；不预设正结果，不将多个对照里偶然最高的一项包装为稳定突破。

'''
    (ROOT/'results/REPORT.md').write_text(intro+'\n'.join(tables)+'\n'+'\n'.join(audit_lines),encoding='utf8')
    print('REPORT BUILT',runs,flush=True)

if __name__=='__main__': main()
