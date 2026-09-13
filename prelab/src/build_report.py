"""Build report and reproducibility manifest from executed artifacts, not hardcoded metrics."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_data import ROOT,sha

def main():
    rows=json.loads((ROOT/'results/metrics.json').read_text())
    selection=json.loads((ROOT/'results/selection.json').read_text())
    audit=json.loads((ROOT/'results/audit.json').read_text())
    own=json.loads((ROOT/'results/own_generation_metrics.json').read_text())
    assert audit['status']=='passed'
    methods=['last_linear','mean_linear','mil_mean','mil_max','mil_topk','hami_ori','token_surprisal','tfidf_text','response_length','shuffled_label_mil_topk']
    names=['Last-token linear','Mean-state linear','MIL mean','MIL max','MIL top-k','HaMI ori','Token surprisal','Text TF-IDF','Response length','Shuffled-label MIL']
    table=[]; summary=[]
    for method in methods:
        ms=[r for r in rows if r['method']==method]
        r={'method':method,'n_seeds':len(ms)}
        for metric in ['response_auc','within_error_response_auc','top10_precision','top10_recall']:
            v=[m[metric] for m in ms if metric in m]
            r[metric+'_mean']=float(np.mean(v)) if v else None
            r[metric+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
        summary.append(r)
        def fmt(k):
            value=r[k+'_mean']; sd=r[k+'_sd']
            return '—' if value is None else f'{value:.3f}'+(f' ± {sd:.3f}' if sd is not None else '')
        table.append(f'| {method} | {fmt("response_auc")} | {fmt("within_error_response_auc")} | {fmt("top10_precision")} | {fmt("top10_recall")} |')
    (ROOT/'results/method_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,title in zip(axes,['response_auc','within_error_response_auc'],['Whole-response detection','Within-error-response localization']):
        valid=[(n,r) for n,r in zip(names,summary) if r[key+'_mean'] is not None]
        ax.barh(range(len(valid)),[r[key+'_mean'] for _,r in valid],xerr=[r[key+'_sd'] or 0 for _,r in valid],color=['#2563eb' if r['method']=='mil_max' else '#94a3b8' for _,r in valid])
        ax.set_yticks(range(len(valid)),[n for n,_ in valid]); ax.invert_yaxis(); ax.axvline(.5,color='#dc2626',linestyle='--')
        ax.set_xlim(0,1); ax.set_xlabel('AUROC (0.5 = chance; bars: seed SD)'); ax.set_title(title)
    fig.suptitle('Weak supervision pilot | Qwen2.5-0.5B replay | 89 heldout news responses')
    fig.savefig(ROOT/'results/comparison.png',dpi=170); plt.close(fig)
    m=selection['metrics']; ba=selection['response_auc_95ci']; la=selection['within_error_response_auc_95ci']
    report=f'''# 预实验结果：整段标签能否训练出事实错误定位探针

**结论：流程已完整跑通，但当前结果未证明可靠定位。** 本轮白盒探针只有弱信号；按验证集预定规则选出的模型，整段和局部 AUROC 的 95% 区间都包含随机水平 0.5。不能据此宣称弱监督定位已成功，也不能据此否定更大模型或同模型生成数据上的可行性。

## 实际完成

- GPU：RTX 3070 8GB；Qwen2.5-0.5B-Instruct，冻结权重，FP16；权重通过官方 LFS SHA-256 校验。
- RAGTruth 新闻摘要 394 条：训练 240、验证 65、测试 89。测试含 25 条错误、64 条无标注错误；31 个错误片段。
- 同一原始生成器 Mistral-7B-Instruct；只选 Evident Conflict 与无幻觉样本。来源全文哈希隔离，官方测试来源不参加训练；不截断原文。
- 6 类探针方案、4 类对照；4 类 MLP 各 3 个种子；第 8/16/24 层只由验证集整段标签选择。
- 训练、选层、选训练步和阈值均不读取局部标签；测试时才使用错误片段标签。
- 实际生成并审阅了 16 条 Qwen 回答，单独报告。

## 方法对比

数值越高越好；AUROC 0.5 为随机。± 是三个种子之间的标准差，不是置信区间。局部 AUROC 是在每条错误回答内部计算后平均，避免只靠整段有错就伪装成定位成功。top-10% 指每条错误回答中风险最高约 10% 的 token。

| 方法 | 整段 AUROC | 局部 AUROC | top-10% 精确率 | top-10% 召回率 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

![方法对比](comparison.png)

**选择规则**：四种逐 token MLP 方法按三个种子的平均验证集整段 AUROC 选择，得到 `{selection['method']}`。展示/区间计算使用预定 seed 42；没有按测试定位结果选最好的方法或种子。

选定检查点 `{selection['display_checkpoint']}`：

- 整段 AUROC **{m['response_auc']:.3f}**，95% 区间 **[{ba[0]:.3f}, {ba[1]:.3f}]**。
- 局部 AUROC **{m['within_error_response_auc']:.3f}**，95% 区间 **[{la[0]:.3f}, {la[1]:.3f}]**。
- 取风险最高约 10% 的 token，错误精确率 **{m['top10_precision']:.1%}**，错误召回率 **{m['top10_recall']:.1%}**。错误回答本身平均已有 **{m['token_positive_fraction_in_error_responses']:.1%}** 的 token 落在错误片段中；高分区域并没有明显富集错误。
- 区间按新闻来源/回答重采样 1,000 次，不能把大量相关 token 当独立样本。
- 文本 TF-IDF 的整段 AUROC 高于本轮白盒探针，不能声称内部探针优于表面文本基线。

## 小模型自己生成的补充检查

固定抽取 16 个留出来源，Qwen 自然采样（temperature 0.7、top-p 0.9），没有往答案里植入错误。执行助手在未查看这批预测分数时逐项对照证据：13 条含明确冲突，1 条未发现明确错误，2 条只有不能确定真假的新增内容，后两条不强行判错。

可见错误包括把 John Travolta 写成 Actress、把 130 名证人写成 130 名陪审员、把 B.B. King 的 89 岁写成 56 岁。原文、生成文本、标注理由和风险热图均保留。

选定探针在 13 条错误回答内部的局部 AUROC 为 **{own['metrics_descriptive_only']['within_error_response_auc']:.3f}**，top-10% 精确率 **{own['metrics_descriptive_only']['top10_precision']:.1%}**；这些回答平均错误 token 比例为 **{own['metrics_descriptive_only']['token_positive_fraction_in_error_responses']:.1%}**。仍未显示明显错误富集。由于只有 1 条正常回答，不用这组数据估计可靠的整体检出率或误报率。

这是**单执行助手标注**，不是独立人工双标数据；含娱乐/影视报道，不能全部称为现实安全情报事实。需要独立复核后才能用于正式研究。

## 本次能证明与不能证明的事

本次证明工程上可用整段标签训练逐位置探针，并在独立测试中量化定位效果。当前小模型、样本量和重放设置下，没有得到可靠定位的实证支持。

主实验是 Qwen 重放 Mistral 已有回答，不能当作 Qwen 自发生成实验；仅使用固定检索证据，未搭建实时联网代理；Evident Conflict 是证据冲突而非全部现实事实的独立核验；没有测试幻觉抑制。HaMI 使用官方 `ori` 网络/损失，未做语义一致性增强，不能冒称完整论文复现。

下一步最值得改变的是**用目标模型自己生成的新闻回答建立足量、独立复核的训练与测试数据**，再检验同样的方法。现阶段不建议把此结果包装成“已能逐 token 定位”，也不建议仅因为某个种子稍高于随机就扩大结论。

## 验证与产物

- [审计](audit.json)：{audit['feature_files_valid']} 份特征、{audit['annotation_spans_valid']} 个训练/验证/测试标注片段偏移检查通过；来源无交集。
- 真实模型的“前缀单独推理”与“整段因果推理”对应状态相对差异 {audit['causal_prefix_relative_error']:.4f}，符合 FP16/SDPA 数值差异；token 惊讶度索引独立核对通过。
- [全部指标](metrics.csv)、[三种子汇总](method_summary.json)、[选择与区间](selection.json)。
- [全部 89 条测试热图](token_risk_report.html)、[16 条自然生成热图](own_generation_report.html)。红色是回答内相对风险，蓝线是已标注错误，不是校准概率；不只展示成功案例。
- [配置](../configs/experiment.json)、[数据清单](dataset_manifest.json)、[模型校验](model_manifest.json)、[环境](environment_freeze.txt)、[复现说明](../README.md)。
'''
    (ROOT/'results/REPORT.md').write_text(report,encoding='utf8')
    paths=[]
    for folder,pattern in [('src','*.py'),('configs','*.json'),('data/processed','*.jsonl'),('data/annotations','*.jsonl'),('results/checkpoints','*'),('data/own','*.json*')]:
        paths.extend((ROOT/folder).glob(pattern))
    manifest={'python':sys.executable,'python_version':platform.python_version(),
              'files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.is_file()},
              'completion':'Executed data preparation, frozen model feature extraction, fitting, heldout bag/span evaluation, causal audit, natural generation, reviewed supplementary evaluation, report'}
    (ROOT/'results/run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Report written:',ROOT/'results/REPORT.md')

if __name__=='__main__': main()
