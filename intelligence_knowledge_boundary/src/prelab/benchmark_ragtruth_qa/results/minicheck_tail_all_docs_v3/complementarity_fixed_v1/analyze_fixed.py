"""Fixed existing thresholds, calibration-only complementarity; no model fit."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
QA=OUT.parents[2]
KINDS=('Evident Baseless Info','Subtle Baseless Info','Evident Conflict','Subtle Conflict')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def metric(y,p):
    y=np.asarray(y,int);p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def overlap(y,a,b):
    y=np.asarray(y,int);a=np.asarray(a,bool);b=np.asarray(b,bool);ca=a==y;cb=b==y
    result={'n':len(y),'both_correct':int((ca&cb).sum()),'both_wrong':int((~ca&~cb).sum()),
            'tail2_only_correct':int((ca&~cb).sum()),'semantic_sequence_claim_only_correct':int((~ca&cb).sum())}
    assert sum(result[k] for k in ('both_correct','both_wrong','tail2_only_correct','semantic_sequence_claim_only_correct'))==len(y)
    return result

def recall_overlap(mask,a,b):
    a=a[mask];b=b[mask];n=len(a)
    return {'positive_windows':n,'both_detected':int((a&b).sum()),'tail2_only_detected':int((a&~b).sum()),
            'semantic_sequence_claim_only_detected':int((~a&b).sum()),'both_missed':int((~a&~b).sum()),
            'tail2_detected':int(a.sum()),'semantic_sequence_claim_detected':int(b.sum()),
            'tail2_recall':float(a.mean()) if n else None,'semantic_sequence_claim_recall':float(b.mean()) if n else None}

def harp_keys(x):
    if isinstance(x,dict):return {k.replace('semantic_sequence_claim','harp_claim'):harp_keys(v) for k,v in x.items()}
    return x

def run():
    assert not (OUT/'complete.json').exists(),'Preserve completed analysis'
    data=QA/'data';windows=rows(data/'windows_k4_calibration.jsonl');answers=rows(data/'answers_calibration.jsonl')
    tokens=rows(data/'tokens_calibration.jsonl');texts={r['response_id']:r['original_response'] for r in rows(data/'calibration.jsonl')}
    assert len(windows)==42241 and len(answers)==len(tokens)==159
    assert all(r['partition']=='calibration' for r in windows+answers+tokens)
    y=np.asarray([w['label'] for w in windows]);ay=np.asarray([a['label'] for a in answers])
    diagnostics=QA/'results/minicheck_tail_all_docs_v3/completed_diagnostics'
    dcomplete=read(diagnostics/'complete.json');assert sha(diagnostics/'DIAGNOSTICS.json')==dcomplete['diagnostics_sha256']
    prior=read(diagnostics/'DIAGNOSTICS.json')['models']
    tail=QA/'results/minicheck_tail_all_docs_v3/tail2';tc=read(tail/'complete.json');te=tc['selected']
    assert te['epoch']==2 and not tc['test_opened']
    tp=tail/f'epoch_{te["epoch"]:02d}_scores.npz';assert sha(tp)==te['artifacts_sha256']['_scores.npz']
    with np.load(tp,allow_pickle=False) as z:
        assert np.array_equal(z['cal_window_labels'],y) and np.array_equal(z['cal_answer_labels'],ay)
        ts=z['cal_window_scores'].copy();tas=z['cal_answer_scores'].copy()
    sf=QA/'results/claim_pooling_v1';se=read(sf/'summary.json')['selected']['minicheck_hidden64_risk_tcn_w32']
    sp=sf/(se['candidate']+'_scores.npz');assert sha(sp)==se['scores_sha256']
    with np.load(sp,allow_pickle=False) as z:
        assert len(z['window_scores'])==168123+42241 and len(z['answer_scores'])==793
        ss=z['window_scores'][168123:].copy();sas=z['answer_scores'][634:].copy()
    a=ts>=te['thresholds']['window']['threshold'];b=ss>=se['thresholds']['window']['threshold']
    aa=tas>=te['thresholds']['answer']['threshold'];ab=sas>=se['thresholds']['answer']['threshold']
    models={}
    for name,p,ap,entry in [('tail2',a,aa,te),('semantic_sequence_claim',b,ab,se)]:
        wm=metric(y,p);am=metric(ay,ap)
        assert all(v==prior[name]['window_metrics'][k] for k,v in wm.items())
        assert all(v==prior[name]['answer_metrics'][k] for k,v in am.items())
        models[name]={'window':wm,'answer':am,'fixed_thresholds':entry['thresholds']}
    # Rebuild type membership from original span characters and raw offsets,
    # rather than accepting the previous diagnostic type counts as input.
    typed={k:{} for k in KINDS}
    for t in tokens:
        rid=t['response_id'];text=texts[rid]
        for k in KINDS:typed[k][rid]=set()
        for label in t['original_labels']:
            assert text[label['start']:label['end']]==label['text']
            for i,(l,r) in enumerate(t['response_token_offsets']):
                if any(text[c].isalnum() for c in range(max(l,label['start']),min(r,label['end']))):
                    typed[label['label_type']][rid].add(i)
    masks={k:np.asarray([bool(set(w['token_indices'])&typed[k][w['response_id']]) for w in windows]) for k in KINDS}
    assert np.array_equal(np.logical_or.reduce(list(masks.values())),y==1)
    types={k:recall_overlap(mask,a,b) for k,mask in masks.items()}
    for k in KINDS:
        assert types[k]['positive_windows']==prior['tail2']['by_type'][k]['positive_windows']
        assert types[k]['tail2_detected']==prior['tail2']['by_type'][k]['detected_windows']
        assert types[k]['semantic_sequence_claim_detected']==prior['semantic_sequence_claim']['by_type'][k]['detected_windows']
    clean={r['response_id'] for r in answers if r['label']==0}
    clean_w=np.asarray([w['response_id'] in clean for w in windows])
    negative=y==0
    neg_overlap={'negative_windows':int(negative.sum()),'both_false_positive':int((negative&a&b).sum()),
        'tail2_only_false_positive':int((negative&a&~b).sum()),
        'semantic_sequence_claim_only_false_positive':int((negative&~a&b).sum()),
        'both_true_negative':int((negative&~a&~b).sum())}
    result={'status':'complete_fixed_threshold_description','calibration_answers':159,'calibration_windows':42241,
        'models':models,'all_windows_correctness_overlap':overlap(y,a,b),
        'positive_windows_recall_overlap':recall_overlap(y==1,a,b),'negative_windows_error_overlap':neg_overlap,
        'by_original_error_type':types,'all_answers_correctness_overlap':overlap(ay,aa,ab),
        'clean_answer_windows_correctness_overlap':overlap(y[clean_w],a[clean_w],b[clean_w]),
        'risk_answer_windows_correctness_overlap':overlap(y[~clean_w],a[~clean_w],b[~clean_w]),
        'type_positive_windows_can_overlap':True,'new_threshold_or_alpha_or_epoch_selection':False,
        'combined_model_or_oracle_performance_computed':False,'GPU_used':False,'official_test_opened':False,
        'scope':'Repeated QA calibration159 only. tail2 fit3680 vs semantic_sequence_claim fit634; these are not data/architecture-matched systems.',
        'interpretation':'Different detections demonstrate complementarity at these frozen cutoffs; different false positives coexist. Overlap alone does not establish that probability averaging will improve F1.',
        'source_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),tp,sp,tail/'complete.json',sf/'summary.json',
            diagnostics/'complete.json',diagnostics/'DIAGNOSTICS.json',data/'windows_k4_calibration.jsonl',
            data/'answers_calibration.jsonl',data/'tokens_calibration.jsonl',data/'calibration.jsonl']}}
    he=read(sf/'summary.json')['selected']['full_lb_harp64_tcn']
    hp=sf/(he['candidate']+'_scores.npz');assert sha(hp)==he['scores_sha256']
    with np.load(hp,allow_pickle=False) as z:
        assert len(z['window_scores'])==168123+42241 and len(z['answer_scores'])==793
        hs=z['window_scores'][168123:].copy();has=z['answer_scores'][634:].copy()
    h=hs>=he['thresholds']['window']['threshold'];ha=has>=he['thresholds']['answer']['threshold']
    hm=metric(y,h);ham=metric(ay,ha)
    assert all(v==he['metrics']['calibration']['windows'][k] for k,v in hm.items())
    assert all(v==he['metrics']['calibration']['answers'][k] for k,v in ham.items())
    hc={'peer':'harp_claim','window_metrics':hm,'answer_metrics':ham,'fixed_thresholds':he['thresholds'],
        'all_windows_correctness_overlap':harp_keys(overlap(y,a,h)),
        'positive_windows_recall_overlap':harp_keys(recall_overlap(y==1,a,h)),
        'by_original_error_type':{k:harp_keys(recall_overlap(mask,a,h)) for k,mask in masks.items()},
        'all_answers_correctness_overlap':harp_keys(overlap(ay,aa,ha)),
        'negative_windows_error_overlap':{'negative_windows':int(negative.sum()),
            'both_false_positive':int((negative&a&h).sum()),'tail2_only_false_positive':int((negative&a&~h).sum()),
            'harp_claim_only_false_positive':int((negative&~a&h).sum()),'both_true_negative':int((negative&~a&~h).sum())}}
    result['priority_HARP_vs_tail2']=hc
    result['source_sha256'][str(hp.resolve())]=sha(hp)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'FIXED_CALIBRATION_PREDICTIONS.npz',y=y,tail2_predictions=a,
                        semantic_sequence_claim_predictions=b,tail2_scores=ts,semantic_sequence_claim_scores=ss,
                        harp_claim_predictions=h,harp_claim_scores=hs,
                        **{'type_'+str(i):masks[k] for i,k in enumerate(KINDS)})
    save(OUT/'COMPLEMENTARITY.json',result)
    ov=result['all_windows_correctness_overlap'];pos=result['positive_windows_recall_overlap']
    hov=hc['all_windows_correctness_overlap'];hpos=hc['positive_windows_recall_overlap']
    report=['# 固定阈值的互补性','',
        '先列较强的 HARP 陈述传播基线与 tail2：',
        f"42,241 窗中，两者都对 {hov['both_correct']}，都错 {hov['both_wrong']}，仅 tail2 对 {hov['tail2_only_correct']}，仅 HARP 对 {hov['harp_claim_only_correct']}。",
        f"5,984 个风险窗口：共同检出 {hpos['both_detected']}，tail2 独有 {hpos['tail2_only_detected']}，HARP 独有 {hpos['harp_claim_only_detected']}，共同漏掉 {hpos['both_missed']}。",'',
        '| 原错误类型 | 风险窗 | 共同检出 | tail2独有 | HARP独有 | 共同漏检 |',
        '|---|---:|---:|---:|---:|---:|']
    for k,r in hc['by_original_error_type'].items():report.append(f"| {k} | {r['positive_windows']} | {r['both_detected']} | {r['tail2_only_detected']} | {r['harp_claim_only_detected']} | {r['both_missed']} |")
    hn=hc['negative_windows_error_overlap']
    report += ['',f"正常窗口的误报：共同 {hn['both_false_positive']}，仅 tail2 {hn['tail2_only_false_positive']}，仅 HARP {hn['harp_claim_only_false_positive']}。",'',
        '以下保留旧 semantic_sequence_claim 与 tail2 的对照：','',
        '只比较 tail2 原选中第2轮与旧 semantic_sequence_claim 各自原阈值；没有重新挑阈值、轮次或传播强度。',
        '',f"同一批 42,241 个校准窗口：两者都对 {ov['both_correct']}，都错 {ov['both_wrong']}，仅 tail2 对 {ov['tail2_only_correct']}，仅旧融合对 {ov['semantic_sequence_claim_only_correct']}。",
        '',f"5,984 个风险窗口：共同检出 {pos['both_detected']}，tail2 独有 {pos['tail2_only_detected']}，旧融合独有 {pos['semantic_sequence_claim_only_detected']}，共同漏掉 {pos['both_missed']}。",'',
        '| 原错误类型 | 风险窗 | 共同检出 | tail2独有 | 旧融合独有 | 共同漏检 | tail2召回 | 旧融合召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,r in types.items():report.append(f"| {k} | {r['positive_windows']} | {r['both_detected']} | {r['tail2_only_detected']} | {r['semantic_sequence_claim_only_detected']} | {r['both_missed']} | {r['tail2_recall']:.3f} | {r['semantic_sequence_claim_recall']:.3f} |")
    report+=['',f"正常窗口的误报：共同误报 {neg_overlap['both_false_positive']}，仅 tail2 误报 {neg_overlap['tail2_only_false_positive']}，仅旧融合误报 {neg_overlap['semantic_sequence_claim_only_false_positive']}。",
        '', '确有不同的检出，但也有不同的误报；不能仅凭互补计数保证概率平均会涨分。这里不计算用金标选预测的理想融合成绩。',
        '', '类型窗口可交叉，分类型数字不能直接相加。tail2 用3,680答训练，旧融合用634答；这是已反复使用的开发校准结果，不是同训练规模对照或独立测试结果。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save(OUT/'complete.json',{'status':'complete','files_sha256':{n:sha(OUT/n) for n in ('COMPLEMENTARITY.json','FIXED_CALIBRATION_PREDICTIONS.npz','REPORT.md')},'official_test_opened':False,'GPU_used':False})
    print(json.dumps({'HARP_comparison':hc,'semantic_comparison':{'all':ov,'positive':pos,'negative':neg_overlap,'types':types}},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
