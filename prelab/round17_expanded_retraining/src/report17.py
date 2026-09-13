"""Write concise comparison from frozen, independently checked outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text('utf-8'))
def fmt(x):return '—' if x is None else f'{x:.3f}'
def ci(value):return '未定义' if value['ci95'] is None else '['+', '.join(f'{v:+.3f}' for v in value['ci95'])+']'
def link(label,p):return f'[{label}]({(ROOT/p).resolve().as_posix()})'


def run():
    audit=read('results/INDEPENDENT_AUDIT17.json');assert audit['status']=='passed'
    result=read('results/metrics_test.json');fit=read('results/fit_metrics.json');val=read('results/validation_metrics.json')
    frozen=read('results/fit_freeze17.json');new=result['methods']['expanded_lb'];old=result['methods']['legacy_lb']
    names={'legacy_lb':'旧训练集／Lookback','expanded_lb':'扩充训练集／Lookback（预定主方法）',
           'legacy_nll':'旧训练集／NLL','expanded_nll':'扩充训练集／NLL',
           'legacy_lb_nll':'旧训练集／Lookback＋NLL','expanded_lb_nll':'扩充训练集／Lookback＋NLL'}
    table='\n'.join(f"|{names[n]}|{fmt(m['answers']['f1'])}|{fmt(m['windows']['f1'])}|{fmt(m['highlight_tokens']['f1'])}|"
                    for n,m in result['methods'].items())
    precision='\n'.join(f"|{label}|{fmt(new[k]['precision'])}|{fmt(new[k]['recall'])}|{new[k]['tp']}|{new[k]['fp']}|{new[k]['fn']}|"
                        for k,label in [('answers','回答'),('windows','4词元窗口'),('highlight_tokens','高亮词元')])
    training='\n'.join(f"|{names[n]}|{fmt(fit['own_training'][n]['windows']['f1'])}|{fmt(val['methods'][n]['windows']['f1'])}|{fmt(result['methods'][n]['windows']['f1'])}|"
                       for n in ('legacy_lb','expanded_lb'))
    delta_w=new['windows']['f1']-old['windows']['f1'];delta_a=new['answers']['f1']-old['answers']['f1']
    interval=result['paired_bootstrap']
    cov=result['coverage'];ec=fit['coverage']['expanded_train'];lc=fit['coverage']['legacy_train']
    text=f'''# Round 17：扩充数据后重新训练

已完成六个小探针的拟合和新测试集评测。预定主方法保持Lookback、4词元窗口、C=0.01；扩充后的窗口F1为{fmt(new['windows']['f1'])}，旧训练规模在相同新测试集上为{fmt(old['windows']['f1'])}，差值{delta_w:+.3f}。回答级F1为{fmt(new['answers']['f1'])}，相对旧训练规模差值{delta_a:+.3f}。这是本轮可直接比较的结果，不与旧实验另一批题目的F1直接相减。

本轮没有证据表明单纯扩大数据改善了窗口定位，也没有达到F1 0.7。主方法的误报窗口从{old['windows']['fp']}减至{new['windows']['fp']}，但漏报窗口从{old['windows']['fn']}增至{new['windows']['fn']}。加入NLL后的扩充探针窗口F1为{fmt(result['methods']['expanded_lb_nll']['windows']['f1'])}，仍低于同一融合方法旧训练规模的{fmt(result['methods']['legacy_lb_nll']['windows']['f1'])}。差值的重采样范围跨0，不能把点估计下降进一步说成已证实的总体退化。

## 同一新测试集上的全部结果

|训练规模与信号|回答F1|4词元窗口F1|合并高亮词元F1|
|---|---:|---:|---:|
{table}

这三列评测对象不同：回答F1判断整份回答是否包含无依据或矛盾陈述；窗口F1判断最多4个词元的小段是否包含风险；高亮词元F1把报警窗口合并后，核对实际标记的每个有效词元。高亮表示需检查的范围，不表示其中每个词都有错误。

新测试集为{cov['questions']}题、{cov['groups']}个事件或主体组、{cov['planned_answers']}份回答，包含{cov['risk_answers']}份风险回答和{cov['reviewed_safe_refusals']}份已审核安全拒答。窗口评测{cov['eligible_windows']}个，其中{cov['risk_windows']}个风险窗口；{cov['eligible_tokens']}个可评词元。重叠窗口不是独立样本。

主方法的精确率、召回率和计数：

|粒度|精确率|召回率|检出风险TP|误报FP|漏报FN|
|---|---:|---:|---:|---:|---:|
{precision}

主方法标记了可评文本词元的{new['highlight_tokens']['marked_token_fraction']:.1%}；回答级误报{new['answers']['fp']}份，正常回答总数{new['answers']['tn']+new['answers']['fp']}份（包含安全拒答）。

## 是否缓解过拟合

|主方法训练规模|各自训练内窗口F1|新验证窗口F1|新测试窗口F1|
|---|---:|---:|---:|
{training}

训练内成绩是乐观诊断，两个训练集的内容和类别比例也不同，不能单凭训练与测试的差距缩小就说过拟合得到解决。完整结果还保存了两个模型在同一旧训练集上的诊断值。若要归因扩充的帮助，应优先看同一新测试集上的配对差异。

按{cov['groups']}个测试事件或主体组进行2000次配对重采样，Lookback扩充减旧训练的F1差值95%范围：窗口{ci(interval['windows']['expanded_minus_legacy']['lb'])}；回答{ci(interval['answers']['expanded_minus_legacy']['lb'])}。该范围固定已训练模型及阈值，不包含重新拟合的不确定性，也不校正多方案比较。

## 训练与评测规则

旧规模{lc['questions']}道训练题，扩充规模{ec['questions']}道；可拟合窗口由{lc['eligible_windows']}增至{ec['eligible_windows']}。大模型Qwen2.5-7B-Instruct NF4参数冻结，只拟合逻辑回归探针。Lookback用784个注意力头比值；NLL是模型给已生成词元的负对数概率；融合对照拼接为785维，并按训练数据逐维标准化。本轮没有更换为复杂融合网络。

训练按组、条件、回答平衡，再计算训练专属类别权重；两种规模都固定总损失权重3854，避免样本增加同时改变正则强度。没有用旧验证或旧测试数据拟合。新验证49题分别选择窗口与回答阈值，模型和阈值冻结后才读取新测试标签及评测；未依据测试分数再调参。各方法的阈值见fit_freeze17.json。

窗口候选来自实际输出范围和原始BPE位置，不依金标挑选。回答分数取所有候选窗口风险的最大值；安全拒答也正常经过探针，不使用拒答金标把分数强制归零。已审核安全拒答仅在回答级作为负例，定位未决与拒答不补全零词元标签。缺预测不得从已知标签分母悄悄删除。

## 文件与限制

- {link('冻结探针模型','results/frozen_models.pkl')}、{link('模型及阈值冻结记录','results/fit_freeze17.json')}。
- {link('测试指标与分题型结果','results/metrics_test.json')}、{link('逐回答分数','results/answer_scores_test.jsonl')}、{link('逐窗口分数','results/window_scores_test.jsonl')}。
- {link('内部信号提取核验','data/extraction_selfcheck.json')}、{link('独立评测审计','results/INDEPENDENT_AUDIT17.json')}。

标签由助手标注和交叉复核，并非研究者人工金标；任务仍是英文短回答和受控证据缺失。旧训练标签规则保持原样，因此扩充也包含来源及标签构成变化。新旧特征均采用完整答案的因果回放；定长后缀替换用于检查未读取未来词元内容，低精度计算在缩短回放长度时可能改变数值，不能据此宣称逐步在线提取与完整回放严格数值一致。全部旧数据和旧实验保持原样。
'''
    (ROOT/'results/REPORT.md').write_text(text,'utf-8')
    print('REPORT_WRITTEN',json.dumps({'new_window_f1':new['windows']['f1'],'old_window_f1':old['windows']['f1'],
          'new_answer_f1':new['answers']['f1'],'old_answer_f1':old['answers']['f1']},ensure_ascii=False))


if __name__=='__main__':run()
