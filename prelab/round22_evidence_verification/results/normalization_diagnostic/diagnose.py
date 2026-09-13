"""Frozen R21 development-score normalization; no estimator fitting or GPU.

Four existing probes, five predeclared temperatures. Calibration-only window
selection; original answer cutoffs remain untouched. All fold choices saved
before outer-label metric calculation. Old R16 validation/test is never parsed.
"""
from pathlib import Path
from collections import defaultdict
import importlib.util, json, pickle, hashlib, math
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
R21=OUT.parents[2]/'round21_semantic_internal_probe'
UTILITY=R21/'results/audit21.py'
spec=importlib.util.spec_from_file_location('independent21_norm22',UTILITY)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
read,lines,sha,close=a.read,a.lines,a.sha,a.close
p,u=a.p,a.u
BASES=('base','slots_base','r19_all','base_harp_delta')
TEMPERATURES=(.25,.5,1.,2.,4.)
VARIANTS=('original',)+tuple('T'+str(t).replace('.','p') for t in TEMPERATURES)+('selected',)
METHODS=tuple(base+'__'+v for base in BASES for v in VARIANTS)
p.METHODS=METHODS


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')


def savel(path,rows):
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def specification():
    return {'version':'r22-frozen-local-logit-normalization-v1','base_methods':list(BASES),'temperatures':list(TEMPERATURES),
      'formula':'p * exp((logit(s)-max_item_logit)/T), p=max original window score within the visible answer',
      'logit_numeric_clipping':{'lower':float(np.finfo(float).eps),'upper':1.-float(np.finfo(float).eps)},
      'windows':'all original output-defined raw four-BPE windows; no label filtering in normalization or answer max',
      'selection':'Each T chooses cal window risk-F1 cutoff; cutoff ties precision then higher cutoff; T ties F1, precision, smaller T',
      'answer_cutoff':'Original frozen cutoff unchanged; every answer maximum and decision must reproduce exactly',
      'scope':'Repeated exploratory development on R16 actual train301 questions/278 groups and old frozen fivefold scores; not new training or unseen test',
      'new_model_or_external_signal':False,'old_validation_or_test_parsed':False,'bootstrap':{'draws':2000,'seed':20260922}}


def normalize(windows,score,t):
    score=np.asarray(score,np.float64)
    assert score.shape==(len(windows),) and np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
    by_item=defaultdict(list)
    for i,w in enumerate(windows):
        assert len(w['item_ids'])==1
        by_item[w['item_ids'][0]].append(i)
    eps=np.finfo(float).eps;bounded=np.clip(score,eps,1-eps)
    logits=np.log(bounded)-np.log1p(-bounded);result=np.empty(len(score),np.float64)
    for ix in by_item.values():
        maximum=float(score[ix].max());local=logits[ix]
        result[ix]=maximum*np.exp((local-local.max())/t)
        assert result[ix].max()==maximum
        # Exact peaks survive, including original ties and probability endpoints.
        assert (result[ix][score[ix]==maximum]==maximum).all()
    return result


def self_test():
    ww=[{'item_ids':[k]} for k in ('A','A','A','B','B','C','C')]
    s=np.asarray([.1,.9,.9,0.,0.,.2,1.])
    for t in TEMPERATURES:
        got=normalize(ww,s,t)
        for group in ([0,1,2],[3,4],[5,6]):assert got[group].max()==s[group].max()
        assert got[0]<got[1]==got[2] and got[3]==got[4]==0
    x=normalize(ww,s,1.)
    assert abs(x[0]-(.1*(1-.9)/(1-.1)))<1e-14


def threshold_on_cal(pack,score):
    ix=[i for i,w in enumerate(pack['windows']) if w['main_eligible']]
    return u.threshold([pack['windows'][i]['gold'] for i in ix],score[ix])


def feature_scores(frozen,design):
    result={}
    for name,key in [('base','base'),('slots_base','slots'),('r19_all','r19_all')]:result[name]=a.a19.predict(frozen['models'][name],design[key])
    pc=frozen['projections']['delta']
    delta=((design['delta'].astype(np.float64)-pc['mean'])@pc['components'].T).astype(np.float32)
    matrix=np.column_stack((design['base'],design['harp'],delta))
    result['base_harp_delta']=a.a19.predict(frozen['models']['base_harp_delta'],matrix)
    return result


def count_independent(rows,name,unit):
    tp=fp=fn=tn=0
    for r in rows:
        if not r['main_eligible']:continue
        y=int(r['gold']);pred=bool(r['predictions'][name])
        if y==1:
            if pred:tp+=1
            else:fn+=1
        else:
            if pred:fp+=1
            else:tn+=1
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn),
      'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),unit:tp+fp+fn+tn}


def bootstrap(rows):
    groups=sorted({r['group_id'] for r in rows});assert len(groups)==278
    gi={g:i for i,g in enumerate(groups)};table=np.zeros((278,len(METHODS),3),np.int64)
    for r in rows:
        if not r['main_eligible']:continue
        y=r['gold']
        for j,m in enumerate(METHODS):
            z=r['predictions'][m];table[gi[r['group_id']],j]+=[int(y==1 and z),int(y==0 and z),int(y==1 and not z)]
    cfg=specification()['bootstrap'];indices=np.random.default_rng(cfg['seed']).integers(0,278,(cfg['draws'],278))
    sums=table[indices].sum(1);tp,fp,fn=sums[:,:,0],sums[:,:,1],sums[:,:,2]
    f1=2*tp/(2*tp+fp+fn)
    result={'groups':278,'draws':cfg['draws'],'seed':cfg['seed'],'method_f1':{},'selected_minus_original':{}}
    for j,m in enumerate(METHODS):result['method_f1'][m]=u.interval(f1[:,j])
    for b in BASES:result['selected_minus_original'][b]=u.interval(f1[:,METHODS.index(b+'__selected')]-f1[:,METHODS.index(b+'__original')])
    return result


def run():
    self_test();assert not (OUT/'complete.json').exists(),'Refuse overwrite of completed diagnostic'
    protocol=specification();pp=OUT/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:save(pp,protocol)
    a.sources()
    audit=read(R21/'results/INDEPENDENT_AUDIT21.json');assert audit['status']=='passed' and audit['complete_sha256']==sha(R21/'results/complete21.json')
    source={'complete21_sha256':sha(R21/'results/complete21.json'),'independent_audit21_sha256':sha(R21/'results/INDEPENDENT_AUDIT21.json'),
      'protocol_sha256':sha(pp),'script_sha256':sha(__file__),'utility_sha256':sha(UTILITY)}
    save(OUT/'source_snapshot.json',source)
    pack,_,_,_=a.a19.rebuild(features=False)
    with np.load(R21/'results/designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('base','slots','r19_all','harp','delta')}
    assignments=read(a.R19/'data/fold_assignment.json')['groups'];choice={};cache=[]
    # This phase only examines calibration labels, never outer metric results.
    for fold in range(5):
        frozen=pickle.loads((R21/'results'/f'fold_{fold}_frozen.pkl').read_bytes())
        cg=sorted(g for g,f in assignments.items() if f==(fold+1)%5);eg=sorted(g for g,f in assignments.items() if f==fold)
        assert frozen['calibration_groups']==cg and frozen['evaluation_groups']==eg
        ca,cx=p.subset(pack,cg);ev,ex=p.subset(pack,eg)
        original=feature_scores(frozen,d);selection={};transforms={}
        for name,v in original.items():
            old_rows=lines(R21/'results'/f'fold_{fold}_window_scores.jsonl')
            assert np.array_equal(v[ex],np.asarray([r['scores'][name] for r in old_rows]))
            assert threshold_on_cal(ca,v[cx])==frozen['thresholds'][name]['window']
            entries=[];transform={}
            for t in TEMPERATURES:
                values=normalize(pack['windows'],v,t);ts=threshold_on_cal(ca,values[cx])
                assert np.array_equal(u.item_max(pack,v),u.item_max(pack,values))
                key=(ts['validation_f1'],ts['validation_precision'],-t)
                entries.append({'temperature':t,'window_threshold':ts,'selection_key':list(key)})
                transform[t]=values
            j=max(range(len(entries)),key=lambda i:tuple(entries[i]['selection_key']))
            selection[name]={'selected_index':j,'temperature':entries[j]['temperature'],'window_threshold':entries[j]['window_threshold'],
              'original_window_threshold':frozen['thresholds'][name]['window'],'answer_threshold':frozen['thresholds'][name]['answer'],
              'all_temperature_calibration_candidates':entries,'answer_threshold_changed':False}
            transforms[name]=transform
        choice[str(fold)]={'calibration_groups':cg,'evaluation_groups':eg,'methods':selection}
        cache.append((ev,ex,original,transforms,frozen['thresholds']))
        print('NORMALIZATION_CAL_FROZEN_FOLD',fold,flush=True)
    save(OUT/'selection_freeze.json',{'source_snapshot_sha256':sha(OUT/'source_snapshot.json'),'status':'frozen_before_outer_metrics','folds':choice})
    freeze_hash=sha(OUT/'selection_freeze.json')
    # Evaluate all fixed candidates only after every calibration choice is saved.
    allw=[];alla=[];fold_metrics={};answer_checks={}
    for fold,(ev,ex,original,transforms,old_ts) in enumerate(cache):
        wr=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']]
        ar=[dict(r,scores={},predictions={},fold=fold) for r in ev['items']]
        detail={};answer_checks[str(fold)]={}
        for base in BASES:
            selected=choice[str(fold)]['methods'][base];records={
              'original':(original[base],selected['original_window_threshold']),
              'selected':(transforms[base][selected['temperature']],selected['window_threshold'])}
            for entry in selected['all_temperature_calibration_candidates']:
                label='T'+str(entry['temperature']).replace('.','p');records[label]=(transforms[base][entry['temperature']],entry['window_threshold'])
            original_av=u.item_max(ev,original[base][ex])
            for variant,(vv,wt) in records.items():
                name=base+'__'+variant;v=vv[ex];ts={'window':wt,'answer':selected['answer_threshold']}
                av=u.item_max(ev,v);assert np.array_equal(av,original_av)
                answer_prediction=av>=ts['answer']['threshold']
                assert np.array_equal(answer_prediction,original_av>=old_ts[base]['answer']['threshold'])
                for row,s in zip(wr,v):row['scores'][name]=float(s);row['predictions'][name]=bool(s>=wt['threshold'])
                for row,s,pred in zip(ar,av,answer_prediction):row['scores'][name]=float(s);row['predictions'][name]=bool(pred)
                detail[name]=u.measures(ev,v,ts)[0]
            answer_checks[str(fold)][base]={'all_7_variants_preserve_all_answer_maxima_exactly':True,'all_answer_predictions_exact':True}
        fold_metrics[str(fold)]=detail;allw.extend(wr);alla.extend(ar)
    assert len(allw)==12222 and len(alla)==602 and len({w['window_key'] for w in allw})==12222
    pooled=p.pooled(pack,allw,alla)
    old_summary=read(R21/'results/summary.json')['methods']
    for base in BASES:
        for unit in ('windows','answers','highlight_tokens'):u.assert_metrics(pooled[base+'__original'][unit],old_summary[base][unit],base+'.original.'+unit)
        for variant in VARIANTS:
            name=base+'__'+variant
            for rows,unit in [(allw,'windows'),(alla,'answers')]:u.assert_metrics(count_independent(rows,name,unit),pooled[name][unit],name+'.independent.'+unit)
            assert pooled[name]['answers']==pooled[base+'__original']['answers']
            assert pooled[name]['safe_refusal_false_positives']==pooled[base+'__original']['safe_refusal_false_positives']
            # Within-answer monotone rescaling cannot add new ranking information.
            close(pooled[name]['conditional_ranking']['mean_auroc'],pooled[base+'__original']['conditional_ranking']['mean_auroc'])
            close(pooled[name]['conditional_ranking']['mean_average_precision'],pooled[base+'__original']['conditional_ranking']['mean_average_precision'])
    summary={'scope':protocol['scope'],'coverage':u.coverage(pack),'selection_freeze_sha256':freeze_hash,'pooled':pooled,'folds':fold_metrics,
      'temperatures_by_fold':{b:[choice[str(f)]['methods'][b]['temperature'] for f in range(5)] for b in BASES},
      'paired_bootstrap':{'windows':bootstrap(allw),'answers':bootstrap(alla)},'answer_invariance':answer_checks,
      'no_refit':True,'new_verification_signals_used':False,'outer_candidate_results_used_to_choose_T':False}
    savel(OUT/'window_scores_oof.jsonl',allw);savel(OUT/'answer_scores_oof.jsonl',alla);save(OUT/'summary.json',summary)
    report=['这是对已经反复查看过的开发折分数作变换，未训练新模型，也未使用新增核查信号。温度仅由校准折选定；整答阈值沿用原值。\n',
      '| 冻结模型 | 原窗口F1 | 归一后窗口F1 | 差值95%配对区间 | 原/新整答F1 | 五折所选T |',
      '|---|---:|---:|---|---:|---|']
    for base in BASES:
        before=pooled[base+'__original'];after=pooled[base+'__selected'];ci=summary['paired_bootstrap']['windows']['selected_minus_original'][base]
        # Interval helper names are recorded verbatim as well as formatted below.
        report.append(f"| {base} | {before['windows']['f1']:.3f} | {after['windows']['f1']:.3f} | {json.dumps(ci)} | {after['answers']['f1']:.3f} | {summary['temperatures_by_fold'][base]} |")
    report+=['\n采用 `p × exp((logit(s)−本回答最大logit)/T)`，其中p为本回答原始最高窗口风险。全部602条回答在所有温度下最高分及整答判定精确不变，安全拒答误报数也不变。',
      '该变换保留回答内原有排序，只改变不同回答中的非最高窗口如何共用统一阈值；因此不能解释为获得了新的语义信息。最高窗口仍可能只是4BPE与风险片段重叠。',
      '主窗口分母9526，风险1063；整答分母598，风险151（包括117个安全拒答负例）。4个未决回答不补成负例；全部12222个候选窗口均参与各回答归一化，未用金标提前筛选。',
      '每个温度的完整校准表、固定外层结果和逐窗口分数均保留。主比较仅使用事先保存的校准选择；不根据外层最高F1另挑温度。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8')
    assert sha(OUT/'selection_freeze.json')==freeze_hash and source['complete21_sha256']==sha(R21/'results/complete21.json')
    assert source['script_sha256']==sha(__file__) and source['protocol_sha256']==sha(pp)
    a.sources()
    names=('protocol.json','source_snapshot.json','selection_freeze.json','summary.json','window_scores_oof.jsonl','answer_scores_oof.jsonl','REPORT.md')
    save(OUT/'complete.json',{'status':'passed','files_sha256':{n:sha(OUT/n) for n in names},'script_sha256':sha(__file__),
      'independent_integer_counts_verified':True,'answer_scores_and_predictions_exact_all_variants':True,'refitted':False,'original_validation_or_test_parsed':False})
    print('NORMALIZATION_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
