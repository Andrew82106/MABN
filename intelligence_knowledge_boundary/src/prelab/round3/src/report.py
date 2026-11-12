"""Readable full experiment report; no selection by fresh test performance."""
import json,html,pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import ROOT,readl

LABELS={'linear_attention':'线性／注意力','linear_hybrid':'线性／组合特征','attention_mlp_ensemble':'MLP／注意力',
 'attention_conv_ensemble':'因果卷积／注意力','attention_gru_ensemble':'GRU／注意力','hybrid_mlp_ensemble':'MLP／组合特征',
 'hybrid_conv_ensemble':'因果卷积／组合特征','hybrid_gru_ensemble':'GRU／组合特征','architecture_selected':'验证集预选结构',
 'hami_ensemble':'HaMI ori 本地适配／整段监督','fusion_baseline_active':'原探针分数校准',
 'fusion_selection_only_active':'仅标记选中的句子','fusion_direct_probs_active':'直接核查的 A/B/C 概率',
 'fusion_direct_hidden_active':'直接核查提示的内部状态','fusion_verify_prompt_active':'引用证据提示，尚未生成',
 'fusion_verify_prompt_wide_active':'尚未生成，104维状态对照',
 'fusion_verify_text_active':'补充回答的口头结论等文本特征','fusion_verify_hidden_active':'补充回答的内部状态',
 'fusion_verify_joint_active':'补充回答的状态＋文本特征','fusion_expand_hidden_active':'普通扩写的内部状态',
 'fusion_verify_joint_random':'随机选句后补充核查'}

def table(metrics,names):
    lines=['| 方法 | 全体 token AUROC | 回答内定位 AUROC | 正常回答误报 | 错误回答检出 |', '|---|---:|---:|---:|---:|']
    for n in names:
        m=metrics[n];lines.append(f'| {LABELS.get(n,n)} | {m["global_token_auc"]:.3f} | {m["local_auc"]:.3f} | {m["clean_alarm"]:.1%} | {m["error_recall"]:.1%} |')
    return '\n'.join(lines)

def structure_grid_report(selection):
    dest=ROOT/'results';grid=json.loads((ROOT/'configs/structure_grid.json').read_text())
    lines=['# 结构与参数：全部验证结果','',
      '此表只读取已保存的训练检查点，列出全部18个预定配置，不按新测试表现筛选。每配置训练400步，每25步检查验证集；保留验证全体token AUROC最高的检查点。',
      '', '所有网络每批取8条正常、8条错误摘要。balanced对正负token损失各给一半权重；natural对该批有效token直接求平均，并不恢复原数据的摘要比例；ranking在balanced基础上增加同一回答中正负token平均分数的间隔损失。',
      '', '卷积只读取当前位置与前4个位置，GRU为单向；具体实现见[networks.py](../src/networks.py)。',
      '', '验证阶段观察：注意力MLP从宽64加到256，AUROC由0.6938降至0.6745；组合特征MLP加宽仅由0.7017变为0.7021，加深降至0.6877。两类GRU分别为0.7189、0.7266。所有18个配置在第400步均低于较早保存的验证峰值，因此不能把训练更久或网络更大当作可靠改进。这些是同一验证集内的探索现象，独立效果以新测试为准。',
      '', '| 配置 | 特征 | 结构 | 宽度/深度 | 学习率 | dropout/权重衰减 | 损失 | 参数数 | 验证AUROC | 保留步数 |',
      '|---|---|---|---|---:|---|---|---:|---:|---:|']
    for c in grid:
        cp=torch.load(dest/f'checkpoints/{c["id"]}_42.pt',map_location='cpu',weights_only=False)
        lines.append(f'| {c["id"]} | {c["feature"]} | {c["architecture"]} | {c["width"]}/{c["depth"]} | {c["lr"]} | {c["dropout"]}/{c["weight_decay"]} | {c["loss"]} | {cp["parameters"]:,} | {cp["val_global_token_auc"]:.4f} | {cp["step"]} |')
    lines.extend(['', '每类先按种子42选配置，再固定配置用43、44训练。各次训练仍只按验证集选择保留步数，不能把这些分数当独立测试成绩。',
        '', '| 类别 | 选定配置 | 种子42验证AUROC | 种子43验证AUROC | 种子44验证AUROC |', '|---|---|---:|---:|---:|'])
    for family,w in selection['family_winners'].items():
        values=[torch.load(dest/f'checkpoints/{w["config_id"]}_{s}.pt',map_location='cpu',weights_only=False)['val_global_token_auc'] for s in [42,43,44]]
        lines.append(f'| {family} | {w["config_id"]} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines.extend(['',f'验证集预选主结构：{selection["overall_seed42_winner"]}。新测试结果见[主报告](REPORT.md)。'])
    (dest/'STRUCTURE_SEARCH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def heatmap(metrics):
    rows={r['id']:r for r in readl(ROOT/'data/rows.jsonl')};ann={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')};ps=readl(ROOT/'results/test_predictions.jsonl')
    policies={(r['id'],r['policy']):r for r in readl(ROOT/'data/query_policies.jsonl')};queries={q['query_id']:q for q in readl(ROOT/'data/queries.jsonl')};reply={r['query_id']:r for r in readl(ROOT/'data/supplement/verify.jsonl')}
    pieces=['<!doctype html><meta charset="utf-8"><title>全部新测试新闻</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#203040}article{border-top:1px solid #bbc;padding:20px 0}p{white-space:pre-wrap}mark{color:inherit}small{color:#567}summary{cursor:pointer}</style>',
      '<h1>全部80条新测试新闻</h1><p>按ID排序，未挑选好看案例。蓝色下划线：原人工错误区间。红色：相对风险分数，未校准为真实概率。每条展示原分数、仅选句、短核查概率、32维提示、104维提示、追加核查六个对照。补充内容由Qwen生成，不是金标。</p>']
    names=['fusion_baseline_active','fusion_selection_only_active','fusion_direct_probs_active','fusion_verify_prompt_active','fusion_verify_prompt_wide_active','fusion_verify_joint_active']
    for p in sorted(ps,key=lambda x:int(x['id'])):
        r=rows[p['id']];truth=np.zeros(len(r['response']),bool)
        for span in ann[r['id']]['spans']:truth[span['start']:span['end']]=True
        pieces.append(f'<article><h2>{r["id"]}：整段标签 {r["label"]}</h2>')
        for name in names:
            colors=np.zeros(len(truth))
            for (lo,hi),s in zip(p['offsets'],p['predictions'][name]):colors[lo:hi]=np.maximum(colors[lo:hi],s)
            text=[];i=0;bins=(100*np.clip(colors,0,1)).astype(int)
            while i<len(truth):
                j=i+1;c=bins[i]
                while j<len(truth) and bins[j]==c and truth[j]==truth[i]:j+=1
                style='text-decoration:underline 3px #2676aa;' if truth[i] else ''
                text.append(f'<mark style="background:rgb(255,{255-c},{255-c});{style}">{html.escape(r["response"][i:j])}</mark>');i=j
            pieces.append(f'<details><summary>{LABELS[name]}</summary><p>{"".join(text)}</p></details>')
        qid=policies[(r['id'],'active')]['query_id'];q=queries[qid]
        pieces.append(f'<p><b>选中句子：</b>{html.escape(q["statement"])}</p><p><b>Qwen补充核查：</b>{html.escape(reply[qid]["response"])}</p><details><summary>原新闻及人工标注</summary><p>{html.escape(r["evidence"])}</p><pre>{html.escape(json.dumps(ann[r["id"]],ensure_ascii=False,indent=2))}</pre></details></article>')
    (ROOT/'results/all_test_examples.html').write_text('\n'.join(pieces),encoding='utf8')

def main():
    dest=ROOT/'results';ms=json.loads((dest/'metrics.json').read_text());m={r['name']:r for r in ms};ci=json.loads((dest/'confidence_intervals.json').read_text());cost=json.loads((dest/'policy_costs.json').read_text());selection=json.loads((dest/'structure_selection.json').read_text())
    structure_grid_report(selection)
    structure=['linear_attention','linear_hybrid','attention_mlp_ensemble','attention_conv_ensemble','attention_gru_ensemble','hybrid_mlp_ensemble','hybrid_conv_ensemble','hybrid_gru_ensemble','architecture_selected','hami_ensemble']
    information=['fusion_baseline_active','fusion_selection_only_active','fusion_direct_probs_active','fusion_direct_hidden_active','fusion_verify_prompt_active','fusion_verify_prompt_wide_active','fusion_verify_text_active','fusion_verify_hidden_active','fusion_verify_joint_active','fusion_expand_hidden_active','fusion_verify_joint_random']
    lines=['# 第三轮：探针结构与补充回答实验','', '状态：两条路线已运行并在同一批新来源新闻测试。核验状态见末尾；具体结论必须同时看检出和误报。','',
      '## 本轮发现','',
      '1. **复杂网络没有稳定胜过简单探针。** 验证集预选的组合特征GRU，新测试全体token AUROC为0.724，组合特征线性探针为0.734；成对差值95%区间为[-0.048, 0.033]。组合特征MLP的点估计为0.746，但正常回答误报达12.5%，且不是验证集预选主结构。加宽、加深和延长训练没有形成可靠改进。',
      '', '2. **这版长解释没有提供有效增量。** 仅引用证据提示、尚未生成时为0.731；加入后续状态与文本为0.646，成对差值区间[-0.187, -0.007]。同为104维内部特征时，提示状态和生成状态都约0.670。这说明当前状态压缩与线性融合方案未受益，不能推广为任何追加生成都无用。',
      '', '3. **短核查值得继续，但更细定位尚未改善。** 让同一Qwen判断A支持、B冲突、C证据不足，直接读取三个候选的输出分数再与原探针融合，AUROC为0.763；它在验证集的补充方案中也最高。仅选句对照已达0.737，回答内定位为0.736，短核查为0.738，局部增量很小。整条回答排序则由0.609变为0.773。短核查是同模型自检分数基线，尚不能称作新的探针算法。',
      '', '4. **还不能可靠检出错误。** 主阈值下，短核查检出7/40条错误摘要，正常摘要误报0/40；仅选句检出2/40、误报0/40。主动选句覆盖22/40条错误摘要和23/44个错误片段，随机选句覆盖10/40条和10/44个片段。选句覆盖与风险判定都还有缺口。0/40不等于真实误报为零，其二项分布双侧95%精确区间上限约8.8%。',
      '', '短核查与仅选句的额外成对分析在查看测试结果后补做，属于探索性描述：全体token AUROC增量区间[-0.030, 0.071]，包含0；整条回答排序增量区间[0.063, 0.259]。因此目前更有依据的是回答级风险筛选的改善迹象，逐词定位增益仍需新数据确认。见[后补分析与原预测哈希](exploratory_direct_checks.json)。未重训、改阈值或追加模型调用。',
      '', '**建议后续优先做：围绕具体疑点的短核查，加上更好的选句覆盖。** 保留组合特征线性探针和三选一自检作为强对照，在全新来源上检验能否减少漏检并维持低误报。本轮不继续依据这80条测试结果调参，也不把简单提示或三选一自检当作原创点。','',
      '## 数据与比较条件','',
      '训练240条、验证65条，沿用人工标注的Mistral新闻摘要；新测试80条（40正常、40明显冲突）为Llama-2-7B摘要，来源文本哈希与此前504条新闻全部隔离。新测试是跨生成器复核，仍为英文新闻，不等于中文情报或Qwen自己生成时的效果。未做全面的事件语义去重。',
      '', '所有重放和追问均由同一个本地Qwen2.5-7B NF4完成，模型参数不更新，无其他大模型API或裁判。标签全部来自新闻原有人工标注；不把追加回答作为新证据真值或金标。',
      '', '## 结构与参数','',
      '固定位置监督，比较注意力特征与加入四层状态压缩、证据移除差异的组合特征。18个配置覆盖MLP宽度64/256、深度1/2、学习率、正则、位置损失，以及因果卷积和GRU。每配置先用种子42，按验证全体token AUROC选每类配置，再固定43/44复跑。下表神经网络是三种子分数平均后的结果；各单种子在完整指标文件。',
      '',f'验证集预选的主结构为 `{selection["overall_seed42_winner"]}`；不能根据新测试最高值再次选方法。HaMI额外列作旧的整段监督参考，监督信息不同。','',table(m,structure),'',
      '## 补充信息','',
      '先用固定注意力线性探针选风险最高的一句话，每条摘要最多核查一句；随机选句作为相同调用次数对照。训练集选句使用按来源三折的折外预测，防止模型在见过标签的训练回答上选择得特别准。',
      '', '追加核查要求引用原文再给结论；普通扩写使用同样64-token上限。分别比较输出前提示状态、口头结论等文本特征、生成过程中内部状态及其组合。融合器仍用原来的逐token标注训练；未选句子的token也全部参与测试。',
      '',table(m,information),'',
      '“仅标记选中的句子”控制选句本身带来的信息。“引用证据提示，尚未生成”使用同一次调用的输出前状态，不需要后面的回答；它与追加状态的比较用于判断新生成内容是否真的有增量。直接A/B/C概率是三个选项内归一化的分数，不是已校准错误概率。',
      '', '额外的104维提示状态对照，与生成状态方案提供相同数量的内部特征，排除简单的输入维度差异。该对照在补充融合器拟合及新测试查看之前加入；PCA仍仅在训练查询上拟合。状态加文本方案再增加9维文本特征，另与只读文本的方案比较。',
      '', '## 追加成本与选句覆盖','', '| 选句方式 | 追加方式 | 平均新增token | 每条摊销秒数 | 截断比例 | 选中句覆盖错误片段 |', '|---|---|---:|---:|---:|---:|']
    for c in cost:lines.append(f'| {c["policy"]} | {c["arm"]} | {c["mean_generated_tokens"]:.1f} | {c["mean_amortized_seconds"]:.2f} | {c["truncated_fraction"]:.1%} | {c["selected_span_recall"]:.1%} |')
    lines.extend(['', '各策略每条摘要一次额外请求；两种生成提示上限相同，但实际输出长度不一定相同。统计保留截断、缺结论等情况。运行批次由4改为2，剩余普通扩写改为1以使用量化库的单向量生成内核。时间仅为不同批次与内核下的实际摊销记录，不能用于比较算法速度，也不是单次交互延迟；不含已缓存的原摘要特征提取成本。prompt-only消融复用生成前状态，未单独测量其整批时延。',
       '', '## 成对不确定性分析','', '| 固定比较：前者减后者 | 全体token AUROC差值95%区间 | 回答内定位差值95%区间 | 检出率差值95%区间 |', '|---|---|---|---|'])
    for name,v in ci['paired_differences'].items():
        fmt=lambda key:'['+', '.join(f'{x:.3f}' for x in v[key])+']'
        lines.append(f'| {name} | {fmt("global_token_auc")} | {fmt("local_auc")} | {fmt("error_recall")} |')
    lines.extend(['', '按来源重采样1000次；区间描述固定检查点在这些来源上的抽样不确定性，不是跨数据集或重新训练的不确定性。差值区间含0时，不应宣称已证明提升。',
       '', '## 解读边界','',
       '- AUROC是排序指标，0.5约为随机，0.7不等于70%准确率。回答内定位只对同时含正确和错误token的回答统计。',
       '- 报警阈值固定为验证集正常回答最大分数的经验95%分位数；验证仅40条正常，不能保证新数据或真实场景误报率不超过5%。完整指标还保留top-10%均分报警作为次要对照。',
      '- 输入特征对已经输出的摘要计算，追加核查也在句子形成后进行；未实现阻止错误token输出的在线控制。',
      '- 本轮追加特征由选中句子的各token共享，线性融合主要调整整句风险；尚未为每个词分别构建补充证据。',
       '- 摘取证据和自述结论仍可能错误；所学的是原始人工冲突标签，未独立重做所有事实核验。',
       '- 抽样核对3个训练查询的3种提示，共9组状态重放；相对差异最高2.86%，低于预设5%量化／批次容差。挂钩前后logits逐项相同，但跨批次缓存与重放并非逐项相同；这不能替代统一批次的新数据复核。',
       '- 提示改变内部状态、多次回答及事实粒度检测已有研究，见[相关工作](../RELATED_WORK.md)。本轮是本地适配和探索对照，不声称复现论文完整实验或算法原创。',
       '', '## 完整结果与核验','', '[所有方法和种子指标](metrics.json) · [全部18种结构与参数的验证记录](STRUCTURE_SEARCH.md) · [全部80条新闻的输出和风险](all_test_examples.html) · [成对区间](confidence_intervals.json) · [运行协议](../configs/protocol.json)'])
    for name in ['final_audit.json','supplement_state_audit.json']:
        p=dest/name;status='待完成' if not p.exists() else ('通过' if json.loads(p.read_text()).get('passed') else '未通过')
        lines.extend(['',f'[{name}]({name})：{status}。'])
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');heatmap(m)
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    panels=[(['linear_attention','linear_hybrid','architecture_selected'],['Attention LR','Hybrid LR','Selected architecture']),
            (['fusion_baseline_active','fusion_selection_only_active','fusion_direct_probs_active','fusion_verify_prompt_active','fusion_verify_prompt_wide_active','fusion_verify_text_active','fusion_verify_joint_active','fusion_expand_hidden_active','fusion_verify_joint_random'],['Base risk','Sentence selection only','Short-check A/B/C scores','Prompt only (32 features)','Prompt only (104 features)','Generated text','Generated states + text','Neutral expansion','Random sentence review'])]
    for a,(names,labels) in zip(ax,panels):
        values=[m[n]['global_token_auc'] for n in names];a.barh(labels,values,color='#327e91');a.axvline(.5,ls='--',color='#888');a.set_xlim(0,1);a.set_xlabel('Global token AUROC, fresh-source transfer test')
        for i,v in enumerate(values):a.text(v+.01,i,f'{v:.3f}',va='center')
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig);print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
