"""Report the unchanged-calibration error tradeoff of the fixed weight control."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
import run_development as q
OUT=HERE/'type_balanced_v1'
meta=q.metadata();wins=meta['windows'][168123:];by={t['response_id']:t for t in meta['tokens'][634:]}
conflict=np.zeros(len(wins),bool)
for i,w in enumerate(wins):
    t=by[w['response_id']]
    for s,m in zip(t['original_labels'],t['span_token_mapping']):
        if s['label_type']=='Evident Conflict' and set(m['risk_token_indices'])&set(w['token_indices']):conflict[i]=True
assert conflict.sum()==997
results=q.read(OUT/'summary.json')['results'];prior=q.read(HERE/'probe_v1/summary.json')['selected']
rows=[];lines=['提高“与资料冲突”样本的训练权重没有改善本轮总体定位。仍是原159答开发校准结果，未打开测试。','','| 特征 | 原二分类平衡 窗口/整答F1 | 类型加权 窗口/整答F1 | 类型加权显性冲突召回 |','|---|---:|---:|---:|']
for method,record in results.items():
    e=record['selected'];old=prior['expanded3680_'+method]
    with np.load(OUT/(e['candidate']+'_scores.npz')) as z:p=z['window_scores'][-42241:]>=e['thresholds']['window']['threshold']
    oldm=old['metrics']['calibration'];newm=e['metrics']['calibration'];recall=float(p[conflict].mean())
    rows.append({'method':method,'candidate':e['candidate'],'baseline':old['candidate'],'metrics':newm,
                 'evident_conflict_detected':int((p&conflict).sum()),'evident_conflict_positive':997,'evident_conflict_recall':recall})
    lines.append(f"| {method} | {oldm['windows']['f1']:.4f}/{oldm['answers']['f1']:.4f} | {newm['windows']['f1']:.4f}/{newm['answers']['f1']:.4f} | {recall:.3f} |")
lines+=['','同3680答、原PCA64、原fit-only scaler、同三档C、同损失总质量168123。唯一模型训练改动为正样本的类别权重；标签仍为原二分类。兼有两种错误的窗口平分权重，不创建新的风险标签。','',
        '基础目标质量设为正常50%/无依据25%/矛盾25%，随后保持既有来源组等质量，所以最终总质量比例不再恰为该目标。实际矛盾部分的损失质量由10156提高至21701，总体定位反而下降；不能简单把问题归为矛盾样本权重不足。','',
        '这是6次新CPU拟合的负结果，不替换既有较强模型。增权改变了不同错误的取舍，单类召回不等于总体F1。独立数值核查另存；开发结果存在选型乐观，不是封存测试。']
q.save(OUT/'ERROR_TRADEOFF.json',{'rows':rows,'diagnostic_only':True,'test_opened':False,'source_sha256':q.sha(OUT/'summary.json')})
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(rows)
