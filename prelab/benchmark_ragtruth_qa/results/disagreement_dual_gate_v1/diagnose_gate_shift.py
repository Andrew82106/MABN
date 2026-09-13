"""Describe frozen OOF/cal gate errors without modifying or selecting anything."""
from pathlib import Path
import sys
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_development as q


def counts(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def distribution(x):
    return dict(n=len(x),mean=float(np.mean(x)),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75))) if len(x) else dict(n=0)


def run():
    assert not (OUT/'GATE_SHIFT_DIAGNOSIS.json').exists()
    summary=q.read(OUT/'summary.json');selected=summary['selected'];table=q.read(OUT/'fit_only_threshold_grid.json')
    identity=table[0];assert identity['mode']=='identity'
    with np.load(OUT/'frozen_inputs.npz') as z:
        y=z['window_labels'].astype(bool);old=z['current_positive'].copy();eligible=z['rescue_eligible'].copy()
    with np.load(OUT/'fit_OOF_gate_probabilities.npz') as z: ok=z['keep_probability'].copy();orr=z['rescue_probability'].copy()
    with np.load(OUT/'final_two_head_predictions.npz') as z:
        fk=z['keep_probability'].copy();fr=z['rescue_probability'].copy();final=z['window_decisions'].copy()
    features=np.load(OUT/'window_features.npy',mmap_mode='r');assert features.shape==(210364,40)
    if selected['mode']=='identity': oof=old[:168123].copy()
    else: oof=(old[:168123]&(ok>=selected['keep_threshold'])) | (eligible[:168123]&(orr>=selected['rescue_threshold']))
    rows={}
    for part,sl,p,pk,pr in [('fit_OOF',slice(0,168123),oof,ok,orr),
                            ('calibration',slice(168123,None),final[168123:],fk[168123:],fr[168123:])]:
        yy=y[sl];cur=old[sl];res=eligible[sl]
        keep_loss=cur&~p;rescue_add=res&p
        r=dict(whole_window_counts=counts(yy,p),
            keep=dict(eligible_rows=int(cur.sum()),source_TP=int((cur&yy).sum()),source_FP=int((cur&~yy).sum()),
                retained_TP=int((cur&p&yy).sum()),retained_FP=int((cur&p&~yy).sum()),
                deleted_TP_became_FN=int((keep_loss&yy).sum()),deleted_FP_became_TN=int((keep_loss&~yy).sum()),
                conditional_probability_by_gold={str(k):distribution(pk[cur&(yy==k)]) for k in (0,1)}),
            rescue=dict(eligible_rows=int(res.sum()),eligible_risk=int((res&yy).sum()),eligible_normal=int((res&~yy).sum()),
                added_TP=int((rescue_add&yy).sum()),added_FP=int((rescue_add&~yy).sum()),
                risk_not_rescued=int((res&~p&yy).sum()),normal_not_rescued=int((res&~p&~yy).sum()),
                conditional_probability_by_gold={str(k):distribution(pr[res&(yy==k)]) for k in (0,1)}),
            answermax_feature_descriptive={})
        assert r['keep']['retained_TP']+r['rescue']['added_TP']==r['whole_window_counts']['tp']
        assert r['keep']['retained_FP']+r['rescue']['added_FP']==r['whole_window_counts']['fp']
        for gate,mask in [('keep',cur),('rescue',res)]:
            r['answermax_feature_descriptive'][gate]={name:{str(k):distribution(features[sl,36+j][mask&(yy==k)]) for k in (0,1)}
                for j,name in enumerate(('current','semantic_claim','tail2','large'))}
        rows[part]=r
    result=dict(status='complete',OOF_identity_window_metrics=identity['metrics'],OOF_selected_window_metrics=selected['metrics'],
        selected_thresholds={k:selected[k] for k in ('mode','keep_threshold','rescue_threshold')},
        parts=rows,current_and_final_calibration=summary['source_current_window_metrics']['calibration'],
        answer_metrics_unchanged=summary['answer_metrics_unchanged']['calibration'],
        inference='Gate-only OOF does not remove upstream fit-label exposure. Learned fit-side separation failed to transfer. Answermax is among frozen inputs and may contribute to this mismatch, but no ablation or changed upstream OOF scores exists here to isolate that cause.',
        source_sha256={n:q.sha(OUT/n) for n in ('summary.json','fit_only_threshold_grid.json','frozen_inputs.npz','window_features.npy','fit_OOF_gate_probabilities.npz','final_two_head_predictions.npz','diagnose_gate_shift.py')},
        new_fits=0,new_thresholds=0,GPU_used=False,official_test_opened=False)
    q.save(OUT/'GATE_SHIFT_DIAGNOSIS.json',result)
    a=rows['fit_OOF'];b=rows['calibration']
    text=['# 双门控分歧解码器：负结果保留','',
        '**这是我们的候选方法，任何正式baseline均未修改。** 固定8个HGB已实际exit0，内部训练/计分3.92秒；OOF只在fit选择keep=0.20、rescue=0.59，未用cal重选。','',
        '| 阶段 | 窗口F1 | TP/FP/FN/TN |','|---|---:|---|',
        '| fit原identity | 0.808219 | 17091/3725/4386/142921 |',
        '| fit门层OOF门控 | 0.859222 | 18777/3453/2700/143193 |',
        '| 最终门模型fit内重测 | 0.889604 | 19364/2693/2113/143953 |',
        '| cal原current | 0.690281 | 3839/1300/2145/34957 |',
        '| cal门控 | 0.613639 | 3190/1223/2794/35034 |','',
        '**整答原头逐值保持**：cal F1 0.891089、阈值0.4566737821091461，TP/FP/FN/TN=90/12/10/47。它不是门控后窗口max；两个head可以不一致。窗口下降，因此不替换current。','',
        '| 门的改变 | fit OOF | cal |','|---|---:|---:|',
        f"| keep删掉的真阳性→FN | {a['keep']['deleted_TP_became_FN']} | {b['keep']['deleted_TP_became_FN']} |",
        f"| keep删掉的误报→TN | {a['keep']['deleted_FP_became_TN']} | {b['keep']['deleted_FP_became_TN']} |",
        f"| rescue补中风险→TP | {a['rescue']['added_TP']} | {b['rescue']['added_TP']} |",
        f"| rescue新增误报→FP | {a['rescue']['added_FP']} | {b['rescue']['added_FP']} |",'',
        'cal中keep删掉741个原TP，只减少485个FP；rescue仅补回92个TP，却新增408个FP。净少649个TP，误报仅净减77，故定位明显下降。','',
        '这与上游fit分数已见过自身标签造成的迁移差相符，但本次不能证明唯一原因：3折只使新gate看不到该折训练标签，无法撤销冻结semantic/tail2/large/current已有的fit监督。当前底座本身fit F1 0.808219与cal0.690281已有差距，新门层进一步依赖这种fit侧分数结构。','',
        '四路answermax只是输入特征，整答输出头完全冻结；不能把窗口下降归因于整答头改变。它是否是gate过度依赖的特征，本轮没有删列消融或真正上游OOF对照，不能作因果结论。JSON保存两门按真/假标签分层的gate概率和answermax分布，仅供解释，不据cal筛特征或再调参数。','',
        '所有962阈值候选（含identity）、8模型、OOF概率和最终双头输出均保留。未改模型/阈值、未用GPU、未读取sealed test。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('GATE_SHIFT_DIAGNOSIS_COMPLETE', {k:{g:r[g] for g in ('keep','rescue')} for k,r in rows.items()},flush=True)


if __name__=='__main__':run()
