"""Describe the completed dataset using verified counts, not detector scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text('utf-8'))
def link(label,p): return f'[{label}]({(ROOT/p).resolve().as_posix()})'

def run():
    freeze=read('data/annotation_freeze.json'); assert freeze['answers']==800
    audit=read('results/INDEPENDENT_DATASET_AUDIT16.json'); assert audit['status']=='passed'
    stats=read('results/dataset_statistics.json'); combined=read('data/combined_manifest.json')
    assembly=read('data/curation/assembly_report.json'); new=stats['new']
    sums={k:sum(s[k] for s in new.values()) for k in ['answers','supported_answers','risky_answers','safe_refusals',
          'unresolved_or_other','localization_answers','eligible_tokens','risk_tokens','windows_4','risk_windows_4']}
    rows='\n'.join(f"|{name}|{new[s]['questions']}|{new[s]['answers']}|{new[s]['event_or_subject_groups']}|{new[s]['localization_answers']}|"
                   for s,name in [('train','训练'),('validation','验证'),('test','测试')])
    topics='、'.join(f'{k} {v}题' for k,v in stats['source_topic_questions'].items())
    text=f'''# Round 16：数据集扩充

已新增400道题、800份同一Qwen模型的真实回答。旧数据原样保留；合计{combined['total_questions']}道题、{combined['total_answers']}份回答，其中训练题由120增至{combined['split_questions']['train']}。

|新增划分|题目|回答|事件或主体组|可作局部定位的回答|
|---|---:|---:|---:|---:|
{rows}

新增题型为数量100、关系120、时间80、行动80、地点20。共{assembly['event_or_subject_groups']}个事件或主体组；同组不跨划分。25道与旧材料关联到同一事件的新题只进训练。组数表示已识别的关联分组，不是统计独立性的证明。

## 实际标签

800份回答中：有据陈述{sums['supported_answers']}份、存在无依据或矛盾陈述{sums['risky_answers']}份、安全拒答{sums['safe_refusals']}份、未决或其他排除{sums['unresolved_or_other']}份。

可定位回答{sums['localization_answers']}份；有效词元{sums['eligible_tokens']}个，其中风险词元{sums['risk_tokens']}个；4词元窗口{sums['windows_4']}个，其中风险窗口{sums['risk_windows_4']}个。窗口最多4个原始BPE词元，短回答可不足4个，步长1；窗口内任一有效词元有风险即标为风险窗口。重叠窗口不能当成独立题目。

初标与独立复核覆盖全部800份回答，{stats['initial_independent_disagreements']}份在状态或风险坐标上有分歧，均由第三位助手裁决。标注由助手完成，human_gold=false，不称研究者人工金标。标签描述的是当前展示资料对实际陈述的支持程度；现实可能正确而当前无证据的补充仍属风险。答非所问但内容有据，与编造事实分开。拒答、解析失败和语义未决不自动补成全零词元标签。

## 来源与生成

使用[RAGognize](https://huggingface.co/datasets/F4biian/RAGognize)固定版本aab54518c2a7c0d25fff8bffbf5337d0321de142的文章快照；原数据页标注CC-BY-SA 4.0。每题保留URL、版本、全文哈希和引文坐标，并配独占背景来源。全文、目标事实、配对和划分在生成前冻结，来源筛选不使用模型是否犯错或检测器分数。

每题有完整资料和缺少目标证据两个版本，正常提问，不提示“资料不足”。本地Qwen2.5-7B-Instruct NF4以greedy生成，最多256个新词元，保留原始输出IDs及字符偏移。未使用其他模型的现成答案或继承其标签。

本批为英文、短答案、受控资料缺失问答。来源主题：{topics}。它仍是预实验数据，不能代表任意长篇联网研判场景。

## 文件与后续使用

- {link('训练数据','data/dataset_train.jsonl')}、{link('验证数据','data/dataset_validation.jsonl')}、{link('测试数据','data/dataset_test.jsonl')}：每行含输入、实际回答、原始词元与标注。
- {link('新旧合并索引','data/combined_index.jsonl')}引用旧数据，不覆盖旧文件；索引中的相对路径以round16_dataset_expansion目录为基准。{link('合并清单','data/combined_manifest.json')}记录数量及哈希。
- {link('详细统计','results/dataset_statistics.json')}、{link('来源配对审计','data/curation/paired_source_review.json')}、{link('标签冻结清单','data/annotation_freeze.json')}。
- {link('独立导出核验','results/INDEPENDENT_DATASET_AUDIT16.json')}已通过：原始回答、字符坐标、词元和窗口标签、各划分统计与冻结哈希相符，旧文件未改。该核验检查数据一致性，不能代替研究者对标注语义的复审。

本轮仅扩充数据，没有重训探针、提取新白盒特征或报告新的F1。后续训练可使用新旧train；新的validation/test应单独使用，旧留出数据已经经历前期实验。代码和原始生成记录足以继续按原始词元回放提取特征。
'''
    (ROOT/'results/REPORT.md').write_text(text,'utf-8')
    example=[json.loads(s) for s in (ROOT/'data/dataset_train.jsonl').read_text('utf-8').splitlines()[:2]]
    (ROOT/'results/EXAMPLE.json').write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('REPORT_WRITTEN',sums)

if __name__=='__main__': run()
