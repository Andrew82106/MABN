import json,pickle,html
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common4 import ROOT,readl,save

NAMES={'base_scores':'原始注意力探针','support_scores':'加入证据差异的探针','single_sentence':'核查一句','sentences3':'核查最多三句','facts3':'核查最多三个事实点',
 'whole':'整篇核查','whole_sentences':'整篇＋句子','whole_facts':'整篇＋事实点','sentences_facts':'句子＋事实点','all_checks':'整篇＋句子＋事实点','selected':'验证集预选主方法','always_error':'全部判为有错'}

def main():
    dest=ROOT/'results';ms=json.loads((dest/'metrics.json').read_text());m={r['name']:r for r in ms};selected=m['selected'];selection=json.loads((dest/'selection.json').read_text());ci=json.loads((dest/'confidence.json').read_text())['selected_F1_95pct'];coverage=json.loads((dest/'coverage.json').read_text());manifest=json.loads((ROOT/'data/manifest.json').read_text())
    sel=selection['selected'];method=NAMES[sel['family']];met=selected['f1']>.7
    lines=['# 第四轮：事实点核查与F1优化','',f'验证集选定的方法是**{method}**。新测试集的回答级错误类F1为**{selected["f1"]:.3f}**，'+('超过0.70目标。' if met else '尚未超过0.70目标。'),
        '',f'精确率{selected["precision"]:.1%}，召回率{selected["recall"]:.1%}；40条错误中检出{selected["tp"]}条、漏掉{selected["fn"]}条；60条正常中误报{selected["fp"]}条。固定模型的来源重采样95%区间为[{ci[0]:.3f}, {ci[1]:.3f}]。该区间较宽，尚不能证明跨数据集稳定超过0.70。',
        '', '**这个F1衡量整条回答是否有事实错误，不是逐词F1，也不是AUROC。** '+f'相应逐词F1为{selected["token_metrics"]["f1"]:.3f}（另外在验证集选位置阈值），回答内部的错误位置排序AUROC为{selected["within_answer_auc"]:.3f}。',
        '', '## 数据与选择规则','',
        '所有此前看过的584个新闻来源均明确作为开发数据，按原生成器和标签分层重分为464训练、120验证。新测试100个来源与此前584个全部隔离，含40条人工明显事实冲突、60条完全无幻觉标注的摘要；没有合成标签或按模型表现筛选样本。测试原摘要来自Llama-2-7B/13B/70B，各生成器错误比例均为40%。这是抽样构造的新闻预实验，不能外推真实情报错误率。',
        '', '全部计算使用同一本地冻结Qwen2.5-7B NF4，重读现成新闻原文与摘要；没有调用其他大模型或裁判。数据集摘要的生成器仅是原始文本来源。未运行真实联网搜索或Qwen自行生成摘要的在线监测。',
        '', '参数、核查范围、汇总方式和判定阈值均按验证集的错误类F1选择，再冻结到新测试。共24个轻量单位探针配置，加上直接读取B概率及B/(A+B)的免训练对照，合计630个验证候选；同分时先看精确率，再看调用数，最后固定顺序。没有按测试集最优方法或阈值报主结果。',
        '', '本轮阈值为F1优化，前轮阈值偏重低误报；本轮还扩大训练数据并更换测试来源。因此不能把两轮F1差值直接归因于某一个算法改动。同一张表里的对照才有相同本轮数据与选择目标。',
        '', '## 这次升级了什么','',
        '1. 覆盖：从只核查风险最高的一句，扩展到最多三句，并加入整篇核查对照。训练样本选句用严格按来源三折的折外探针预测，验证和测试用全训练集拟合的探针。',
        '2. 细化：在选中句子中，按固定规则和风险挑选人名、数字、角色或高风险词作为目标，明确要求Qwen只判断该目标在句中是否与原文冲突。这个规则还不是完整的事实关系抽取。',
        '3. 对应位置：句子核查分数覆盖该句，事实点核查分数再覆盖准确的原字符范围；其他位置保留原探针分数。整篇核查只提供回答级信号，本身不新增局部定位信息。',
        '4. 学习信号：读取支持／冲突／证据不足三个选项的概率、原探针风险、目标文字是否在原文出现等特征；另外比较加入16维压缩内部状态。单位标签仍来自原人工错误范围。模型自检分数只是特征。',
        '', '## 同一新测试集的完整对照','', '| 方法 | 回答级F1 | 精确率 | 召回率 | 正常误报 | 检出/错误40条 | 每条核查次数（验证均值） |','|---|---:|---:|---:|---:|---:|---:|']
    for r in ms:
        calls=r.get('config',{}).get('mean_calls',0)
        lines.append(f'| {NAMES[r["name"]]} | {r["f1"]:.3f} | {r["precision"]:.1%} | {r["recall"]:.1%} | {r["false_alarm"]:.1%} | {r["tp"]}/40 | {calls:.2f} |')
    lines+=['', '“全部判为有错”的F1为0.571，说明F1本身受错误比例影响。主结果必须同时看精确率、召回率和误报。表中其他方法即使测试分数更高，也不替换预先选定的主方法。',
        '', '## 第一批的发现','',
        '核查三句的家族优胜配置为直接读取B自检分数，F1为0.709；句子＋事实点家族使用内部状态探针，F1为0.703。两者是预设对照，不是原先在验证集选定的主方法；不能事后把原主方法0.645隐藏掉。白盒方法在这一批也未超过更简单的自检对照。',
        '', '核查最多三句覆盖36/40条错误回答，单句仅19/40；但精简成具体词语范围后只覆盖25/40。扩大范围确实减少了选漏，当前词语规则还会丢掉关键错误关系。回答级过0.70没有转化为逐词定位过0.70。',
        '', '另用50个新来源对其中两种替代方案进行了冻结复核，详见[独立复核报告](../confirmation/results/REPORT.md)。复核保持原权重和阈值；其混合生成器分布与本批Llama分布不同，两个批次不能合并当作一份未参与任何方法选择的测试。',
        '', '## 选中的位置覆盖了多少错误','', '| 范围 | 覆盖的错误回答 | 覆盖的错误片段 | 平均核查次数 |','|---|---:|---:|---:|']
    for name,c in coverage.items():lines.append(f'| {NAMES[name]} | {c["error_answers_covered"]}/{c["error_answers"]} | {c["spans_covered"]}/{c["all_error_spans"]} | {c["mean_calls"]:.2f} |')
    records=readl(ROOT/'data/readouts/records.jsonl');counts=Counter(r['origin'] for r in records)
    lines += ['', '覆盖表示选到了人工错误所在位置，不等于最终已正确识别。具体事实点选漏时，句子或整篇核查仍可帮助回答级判断。',
        '', f'共保存{len(records)}次单位读数，其中{sum(v for k,v in counts.items() if k.startswith("round3"))}次复用前轮完全相同的来源、句子和提示缓存。无需自由生成解释；每个新核查只做一次前向计算。调用数按方法需要的冷缓存次数解释，实际批次和摊销耗时保存在每条记录中，未作为不同算法的严格速度对照。',
        '', '## 可复现结果','',f'固定选择：`{sel["model_id"]}`，范围`{sel["scope"]}`，汇总`{sel["aggregate"]}`，判定为分数 **>= {sel["threshold"]:.8g}**。其验证F1为{sel["val_metrics"]["f1"]:.3f}。',
        '', '配置名以raw开头表示直接使用同模型自检分数，没有训练额外的单位分类器；abc等配置表示训练了轻量探针。两类均只在验证集定阈值，不能把免训练读数的收益说成学习了新探针的收益。',
        '', '[全部数值](metrics.json) · [冻结选择](selection.json) · [全部验证候选](validation_trials.jsonl) · [全部100条新闻及风险位置](all_test_examples.html) · [运行协议](../configs/protocol.json)']
    for file in ['audit.json','readout_audit.json']:
        p=dest/file;status='待完成' if not p.exists() else ('通过' if json.loads(p.read_text()).get('passed') else '未通过');lines += ['',f'[{file}]({file})：{status}。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    ps=readl(dest/'test_predictions.jsonl');rows={r['id']:r for r in readl(ROOT/'data/rows.jsonl')};ann={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')};queries=readl(ROOT/'data/queries.jsonl')
    pieces=['<!doctype html><meta charset="utf-8"><title>第四轮全部100条测试</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.8 system-ui;padding:0 20px;color:#203040}article{border-top:1px solid #bbc;margin-top:28px}p{white-space:pre-wrap}summary{cursor:pointer}mark{color:inherit}</style><h1>全部100条测试新闻</h1><p>按ID排序；红色为风险分数，蓝色下划线为人工错误范围。列表同时给出核查的事实点。原模型概率与风险分数都不是正确答案。</p>']
    for p in sorted(ps,key=lambda r:int(r['id'])):
        r=rows[p['id']];truth=np.zeros(len(r['response']),bool)
        for s in ann[r['id']]['spans']:truth[s['start']:s['end']]=True
        pieces.append(f'<article><h2>{r["id"]}：真实标签{r["label"]}，主方法判定{int(p["methods"]["selected"]["pred"])}</h2>')
        for name in ['base_scores','single_sentence','sentences3','facts3','all_checks','selected']:
            values=np.zeros(len(truth))
            for (lo,hi),risk in zip(p['offsets'],p['methods'][name]['token_risks']):values[lo:hi]=np.maximum(values[lo:hi],risk)
            bins=(100*np.clip(values,0,1)).astype(int);text=[];i=0
            while i<len(truth):
                j=i+1
                while j<len(truth) and truth[j]==truth[i] and bins[j]==bins[i]:j+=1
                extra='text-decoration:underline 3px #2676aa;' if truth[i] else ''
                text.append(f'<mark style="background:rgb(255,{255-bins[i]},{255-bins[i]});{extra}">{html.escape(r["response"][i:j])}</mark>');i=j
            pieces.append(f'<details><summary>{NAMES[name]}</summary><p>{"".join(text)}</p></details>')
        facts=[q['target_text'] for q in queries if q['id']==r['id'] and q['kind']=='fact'];pieces.append(f'<p>核查事实点：{html.escape(" / ".join(facts))}</p><details><summary>原新闻</summary><p>{html.escape(r["evidence"])}</p></details></article>')
    (dest/'all_test_examples.html').write_text('\n'.join(pieces),encoding='utf8')
    names=['base_scores','single_sentence','sentences3','facts3','whole','all_checks','selected'];labels=['Base probe','One sentence','Up to 3 sentences','Up to 3 factual spans','Whole summary','Whole + sentences + spans','Validation-selected method']
    fig,ax=plt.subplots(figsize=(10,5));v=[m[n]['f1'] for n in names];ax.barh(labels,v,color='#327e91');ax.axvline(.7,color='#ae563c',ls='--');ax.set_xlim(0,1);ax.set_xlabel('Answer-level error-class F1 on 100 new sources')
    for i,x in enumerate(v):ax.text(x+.01,i,f'{x:.3f}',va='center')
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig);print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
