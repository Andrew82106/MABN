"""No-model baselines; only frozen development counts, no test or fitting."""
from pathlib import Path
import json, hashlib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
path=ROOT/'data/gold_manifest.json'
g=json.loads(path.read_text(encoding='utf-8'));assert g['complete'] and g['official_test_opened'] is False
result={}
for part,counts in g['counts'].items():
    assert part in ('fit','calibration')
    result[part]={}
    for unit,nkey,pkey in [('windows','eligible_windows','positive_windows'),('answers','answers','positive_answers')]:
        n,p=counts[nkey],counts[pkey];prevalence=p/n
        result[part][unit]={
          'all_positive':{'n':n,'positive':p,'tp':p,'fp':n-p,'fn':0,'tn':0,'precision':prevalence,'recall':1.,'f1':2*p/(n+p),
            'average_precision':prevalence,'auroc':.5,'score':1.},
          'all_negative':{'n':n,'positive':p,'tp':0,'fp':0,'fn':p,'tn':n-p,'precision':0.,'precision_denominator_nonzero':False,'recall':0.,'f1':0.,
            'average_precision':prevalence,'auroc':.5,'score':0.}}
output={'status':'arithmetic_baselines','source_gold_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'test_opened':False,'trained_models':0,'risk1_convention':True,'partitions':result,
  'note':'AP for either constant score equals prevalence; AP is ranking-based, unlike the all-negative hard-label F1.'}
(OUT/'naive_baselines.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n','utf-8')
report=['无模型基线：全部报风险／全部报正常。仅用冻结人工QA开发分母计算，不改变模型、参数或阈值。','',
  '| 划分 | 粒度 | 风险数/总数 | 全报风险P/AP | 全报风险R | 全报风险F1 | 全报正常F1 |',
  '|---|---|---:|---:|---:|---:|---:|']
for part,values in result.items():
    for unit,v in values.items():
        a=v['all_positive'];report.append(f"| {part} | {unit} | {a['positive']}/{a['n']} | {a['precision']:.4f} | 1.0000 | {a['f1']:.4f} | 0.0000 |")
report+=['','校准集100/159份回答含人工风险span，所以不看输入、全部报风险也有整答F1=200/259≈0.7722。仅超过0.75不能证明模型很准，必须结合此基线以及precision、recall、AP判断。',
  '恒定分数的AP等于风险比例、AUROC为0.5。全报正常时precision分母为0，表中按0记并在JSON明确标志。校准结果还被用于选参，不是最终测试成绩。']
(OUT/'NAIVE_BASELINES.md').write_text('\n'.join(report)+'\n','utf-8')
print(json.dumps(output,ensure_ascii=False))
