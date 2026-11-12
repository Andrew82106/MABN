"""Independent read-only R22 result/threshold/index audit. No model fitting."""
from pathlib import Path
import json,hashlib,pickle,math
from collections import Counter,defaultdict
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';PRE=ROOT.parent;OLD=PRE/'round21_semantic_internal_probe/results'
BASES=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned')
def read(p):return json.loads(Path(p).read_text('utf-8'))
def readl(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def exact(a,b):return np.array_equal(np.asarray(a),np.asarray(b))
def threshold(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);assert len(y)==len(s) and set(y)=={0,1} and np.isfinite(s).all()
    # Independent distinct-score exhaustive candidate search via sorted counts.
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
    last=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(yy)[last];count=last+1;positive=int(y.sum())
    candidates=[(0.,0.,float(np.nextafter(ss[0],np.inf)))]
    candidates += [(2*int(t)/(int(n)+positive),int(t)/int(n),float(v)) for t,n,v in zip(tp,count,ss[last])]
    candidates.append((2*int(tp[-1])/(len(y)+positive),int(tp[-1])/len(y),float(np.nextafter(ss[-1],-np.inf))))
    f,p,t=max(candidates)
    return {'threshold':t,'validation_f1':f,'validation_precision':p,'tokens':len(y),'scorable_tokens':len(y),'risk_tokens':positive}
def counts(y,p):
    y=np.asarray(y,int);p=np.asarray(p,bool);assert len(y)==len(p)
    tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
def manual_lr(obj,x):
    # Match the documented two in-place float32 scaler operations, not a new fit.
    z=x.copy();z-=obj['scaler'].mean_;z/=obj['scaler'].scale_
    return expit((z.astype(np.float32)@obj['model'].coef_.T+obj['model'].intercept_).ravel())

def audit():
    config=read(ROOT/'protocol22.json');summary=read(OUT/'summary22.json');complete=read(OUT/'complete22.json');freeze=read(OUT/'fit_freeze22.json')
    file_hashes={}
    for manifest in [complete,freeze]:
        for name,h in manifest['files_sha256'].items():assert sha(OUT/name)==h;file_hashes[name]=h
    snap=read(OUT/'source_snapshot22.json')
    for name,h in snap['files_sha256'].items():assert sha(name)==h
    assert read(OLD/'INDEPENDENT_AUDIT21.json')['status']=='passed'
    for name,h in read(OLD/'complete21.json')['files_sha256'].items():assert sha(OLD/name)==h
    times={n:read(OUT/n)['utc'] for n in ['fit_started22.json','fit_freeze22.json','test_started22.json','complete22.json']}
    assert list(times.values())==sorted(times.values())
    items=readl(OUT/'answer_index22.jsonl');windows=readl(OUT/'candidate_windows22.jsonl')
    assert len(items)==602 and len(windows)==12222
    iid={a['item_id']:i for i,a in enumerate(items)};wid={w['window_key']:i for i,w in enumerate(windows)}
    assert len(iid)==602 and len(wid)==12222 and len({a['row_id'] for a in items})==602
    train={}
    # Inputs contain no generated answer labels; only actual split/IDs are retained.
    for r in readl(PRE/'round16_dataset_expansion/data/inputs.jsonl'):
        if r['split']=='train':train[r['row_id']]={k:r[k] for k in ('row_id','question_id','group_id','split')}
    assert set(train)=={a['row_id'] for a in items}
    for a in items:
        assert all(a[k]==train[a['row_id']][k] for k in ('row_id','question_id','group_id','split'))
    assert len({a['question_id'] for a in items})==301 and len({a['group_id'] for a in items})==278
    assert Counter(a['main_eligible'] for a in items)=={True:598,False:4}
    assert sum(a['reviewed_safe_refusal'] for a in items)==117
    assert all(a['gold']==0 and a['main_eligible'] for a in items if a['reviewed_safe_refusal'])
    assert all(not a['main_eligible'] for a in items if a['localization_status']=='unresolved')
    with np.load(OUT/'verifier_designs22.npz',allow_pickle=False) as z:new={k:z[k].copy() for k in z.files}
    wa=np.asarray([iid[w['item_ids'][0]] for w in windows],np.int64)
    assert exact(wa,new['window_answer_index']) and (np.bincount(wa,minlength=602)>0).all()
    for w,j in zip(windows,wa):
        a=items[j];assert w['row_id']==a['row_id'] and w['group_id']==a['group_id'] and w['split']=='train'
    assert new['hidden'].shape==(602,3584) and new['extra'].shape==(602,2)
    expected_extra=np.column_stack(((new['ab_logits'][:,1].astype(float)-new['ab_logits'][:,0].astype(float)).astype(np.float32),new['ab_full_vocab_probabilities'].sum(1)))
    assert exact(expected_extra,new['extra'])
    direct_error=float(np.max(np.abs(expit(new['ab_logits'][:,1].astype(float)-new['ab_logits'][:,0].astype(float))-new['direct'])))
    assert direct_error<2e-7
    # Bind the 602 design rows to the per-answer extraction artifacts and original generation hashes.
    fm=read(ROOT/'data/feature_manifest.json');plans=read(ROOT/'data/plans.json');generation_hash_bound=0
    assert set(fm['records'])==set(plans)==set(train)
    for i,a in enumerate(items):
        rid=a['row_id'];entry=fm['records'][rid];p=ROOT/entry['npz'];s=ROOT/entry['json'];side=read(s)
        assert sha(p)==entry['npz_sha256']==side['npz_sha256'] and sha(s)==entry['json_sha256']
        assert side['source_generation_sha256']==a['annotation']['source_generation_sha256']==plans[rid]['source_generation_sha256']
        assert side['plan_sha256']==digest(plans[rid]) and side['labels_read'] is False and side['output_regenerated'] is False
        with np.load(p,allow_pickle=False) as z:
            assert exact(new['hidden'][i],z['verifier_hidden']) and exact(new['ab_logits'][i],z['ab_logits'])
            assert exact(new['ab_full_vocab_probabilities'][i],z['ab_full_vocab_probabilities']) and new['direct'][i]==float(z['ab_probabilities'][1])
        generation_hash_bound+=1
    assignment=read(PRE/'round19_three_signal_probe/data/fold_assignment.json')['groups']
    assert set(assignment)=={a['group_id'] for a in items}
    wg=np.asarray([w['group_id'] for w in windows]);ag=np.asarray([a['group_id'] for a in items])
    we=np.asarray([w['main_eligible'] for w in windows]);ae=np.asarray([a['main_eligible'] for a in items])
    wy=np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows]);ay=np.asarray([a['gold'] if a['main_eligible'] else -1 for a in items])
    groups_windows=[np.flatnonzero(wa==i) for i in range(602)]
    def max_answer(v):return np.asarray([v[ix].max() for ix in groups_windows])
    def cutoff(v,cg):
        wix=np.flatnonzero(np.isin(wg,cg)&we);aix=np.flatnonzero(np.isin(ag,cg)&ae)
        return {'window':threshold(wy[wix],v[wix]),'answer':threshold(ay[aix],max_answer(v)[aix])}
    with np.load(OLD/'designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ['base','slots','r19_all','harp','delta','ecs','pks']}
    folds=[];oof_w_seen=[];oof_a_seen=[];computed_metrics={};old_diff={name:0. for name in BASES}
    expected_oof_windows=[];expected_oof_answers=[]
    for fold in range(5):
        f=pickle.loads((OUT/f'fold_{fold}_frozen22.pkl').read_bytes())
        with np.load(OUT/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:sc={k:z[k].copy() for k in z.files}
        eg=sorted(g for g,k in assignment.items() if k==fold);cg=sorted(g for g,k in assignment.items() if k==(fold+1)%5);fg=sorted(set(assignment)-set(eg)-set(cg))
        assert f['fit_groups']==fg and f['calibration_groups']==cg and f['evaluation_groups']==eg
        assert not (set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        ix=np.flatnonzero(np.isin(ag,fg)&ae)
        assert exact(ix,f['fit_answer_indices']) and exact(ix,f['projection']['fit_ix'])
        old=pickle.loads((OLD/f'fold_{fold}_frozen.pkl').read_bytes())
        reproduced={name:manual_lr(old['models'][name],d[key]) for name,key in [('base','base'),('slots_base','slots'),('r19_all','r19_all')]}
        pc=old['projections']['delta'];projected=((d['delta'].astype(float)-pc['mean'])@pc['components'].T).astype(np.float32)
        reproduced['base_harp_delta']=manual_lr(old['models']['base_harp_delta'],np.column_stack((d['base'],d['harp'],projected)))
        reproduced['lookback_tuned']=manual_lr(old['models']['lookback_tuned']['candidate'],d['base'][:,:784])
        c=old['models']['redeep_tuned']['candidate']
        pk=d['pks'][:,c['layers']].astype(float).sum(1);ec=d['ecs'][:,c['heads']].astype(float).sum(1)
        pp=(np.column_stack((pk,ec))-c['minimum'])/c['scale'];reproduced['redeep_tuned']=pp[:,0]-c['beta']*pp[:,1]
        for name in BASES:
            error=float(np.max(np.abs(reproduced[name]-sc[name])));old_diff[name]=max(old_diff[name],error)
            assert error<2e-14,(fold,name,error)
            assert f['thresholds'][name]==old['thresholds'][name]==cutoff(sc[name],cg)
        eps=np.finfo(float).eps;b=np.clip(sc['slots_base'],eps,1-eps);logit=np.log(b)-np.log1p(-b)
        rel={}
        for t in config['T']:
            v=np.empty(len(windows))
            for inds in groups_windows:v[inds]=np.exp((logit[inds]-logit[inds].max())/t)
            assert exact(max_answer(v),np.ones(602));rel[t]=v
        assert exact(sc['direct_answer'],new['direct'])
        probs=[sc[f'probe_C{j}_answer'] for j in range(3)]
        assert exact(sc['verifier_direct_broadcast'],new['direct'][wa])
        assert f['thresholds']['verifier_direct_broadcast']==cutoff(new['direct'][wa],cg)
        tables={};broadcast=[]
        for j,p in enumerate(probs):
            c=config['C'][j];ts=cutoff(p[wa],cg)
            broadcast.append({'candidate_index':j,'C':c,'thresholds':ts,'selection_key':[ts['answer']['validation_f1'],ts['answer']['validation_precision'],-c]})
        assert broadcast==f['calibration_candidates']['learned_broadcast'];j=max(range(3),key=lambda k:broadcast[k]['selection_key'])
        assert f['selection']['verifier_learned_broadcast']==broadcast[j]
        assert f['thresholds']['verifier_learned_broadcast']==broadcast[j]['thresholds']
        assert exact(sc['verifier_learned_broadcast'],probs[j][wa])
        selected={}
        for method,sources in [('verifier_direct_slots',[(None,0.,new['direct'])]),('verifier_learned_slots',[(j,config['C'][j],p) for j,p in enumerate(probs)])]:
            table=[];values=[]
            for j,c,p in sources:
                for t in config['T']:
                    v=p[wa]*rel[t];assert exact(max_answer(v),p)
                    ts=cutoff(v,cg);tw,ta=ts['window'],ts['answer']
                    key=[min(tw['validation_f1'],ta['validation_f1']),tw['validation_f1'],tw['validation_precision'],-c,-t]
                    table.append({'candidate_index':j,'C':c,'T':t,'thresholds':ts,'selection_key':key});values.append(v)
            assert table==f['calibration_candidates'][method]
            best=max(range(len(table)),key=lambda k:table[k]['selection_key'])
            chosen=dict(table[best],table_index=best);assert chosen==f['selection'][method]
            assert exact(values[best],sc[method]) and f['thresholds'][method]==chosen['thresholds']
            selected[method]={'C':chosen['C'],'T':chosen['T'],'candidate_index':chosen['candidate_index']}
        evw=np.flatnonzero(np.isin(wg,eg));eva=np.flatnonzero(np.isin(ag,eg));oof_w_seen.extend(evw.tolist());oof_a_seen.extend(eva.tolist())
        savedw=readl(OUT/f'fold_{fold}_window_scores22.jsonl');saveda=readl(OUT/f'fold_{fold}_answer_scores22.jsonl')
        assert [r['window_key'] for r in savedw]==[windows[i]['window_key'] for i in evw]
        assert [r['item_id'] for r in saveda]==[items[i]['item_id'] for i in eva]
        for name in config['methods']:
            av=max_answer(sc[name]);wt=f['thresholds'][name]['window']['threshold'];at=f['thresholds'][name]['answer']['threshold']
            for r,i in zip(savedw,evw):assert r['fold']==fold and r['scores'][name]==sc[name][i] and r['predictions'][name]==bool(sc[name][i]>=wt)
            for r,i in zip(saveda,eva):assert r['fold']==fold and r['scores'][name]==av[i] and r['predictions'][name]==bool(av[i]>=at)
        priorw=readl(OLD/f'fold_{fold}_window_scores.jsonl');priora=readl(OLD/f'fold_{fold}_answer_scores.jsonl')
        for name in BASES:
            for current,prior,key in [(savedw,priorw,'window_key'),(saveda,priora,'item_id')]:
                assert [r[key] for r in current]==[r[key] for r in prior]
                assert [r['scores'][name] for r in current]==[r['scores'][name] for r in prior]
                assert [r['predictions'][name] for r in current]==[r['predictions'][name] for r in prior]
        expected_oof_windows.extend(savedw);expected_oof_answers.extend(saveda)
        folds.append({'fold':fold,'fit_groups':len(fg),'calibration_groups':len(cg),'evaluation_groups':len(eg),
                      'fit_eligible_answers':len(ix),'evaluation_all_answers':len(eva),'evaluation_candidate_windows':len(evw),
                      'all_calibration_thresholds_and_candidate_tables_exact':True,'selected':selected})
        print('AUDIT22_FOLD_PASSED',fold,flush=True)
    assert sorted(oof_w_seen)==list(range(12222)) and sorted(oof_a_seen)==list(range(602))
    assert expected_oof_windows==readl(OUT/'window_scores_oof22.jsonl') and expected_oof_answers==readl(OUT/'answer_scores_oof22.jsonl')
    ew=[r for r in expected_oof_windows if r['main_eligible']];ea=[r for r in expected_oof_answers if r['main_eligible']]
    assert len(ew)==9526 and sum(r['gold'] for r in ew)==1063 and len(ea)==598 and sum(r['gold'] for r in ea)==151
    for name in config['methods']:
        result={}
        for unit,rows in [('windows',ew),('answers',ea)]:
            cc=counts([r['gold'] for r in rows],[r['predictions'][name] for r in rows])
            assert all(cc[k]==summary['methods'][name][unit][k] for k in cc)
            result[unit]=cc
        computed_metrics[name]=result
    old_summary=read(OLD/'summary.json')['methods']
    for name in BASES:assert summary['methods'][name]==old_summary[name]
    result={'status':'passed','scope':'R22 result/index/calibration/OOF/old-baseline independent read-only audit; coefficient/PCA/scaler audit is a separate companion',
        'root':str(ROOT.resolve()),'complete22_sha256':sha(OUT/'complete22.json'),'summary22_sha256':sha(OUT/'summary22.json'),
        'fit_freeze_sha256':sha(OUT/'fit_freeze22.json'),'run22_sha256':sha(ROOT/'src/run22.py'),
        'cohort':{'answers':602,'questions':301,'groups':278,'eligible_answers':598,'risk_answers':151,'safe_refusals':117,'unresolved_excluded':4,
                  'candidate_windows':12222,'eligible_windows':9526,'risk_windows':1063,'generation_hash_bound_design_rows':generation_hash_bound},
        'folds':folds,'old_baseline_manual_max_abs_error':old_diff,'old_six_baseline_oof_scores_predictions_and_summary_exact':True,
        'calibration_selection_reconstructed_without_outer_labels':True,'every_candidate_calibration_table_exact':True,
        'direct_logit_sigmoid_max_abs_error':direct_error,'oof_every_answer_and_window_once':True,
        'all_reported_answer_window_confusion_counts_exact':True,'times':times,'all_folds_frozen_before_outer_scoring':True,
        'metrics':computed_metrics,'no_fit_or_tuning_or_gpu':True,'original_r16_validation_test_answers_or_labels_read':False,
        'limitations':['The same R16 train folds have been repeatedly used for development; this is exploratory OOF evidence, not a fresh final test.',
                      'The verifier consumes an additional same-Qwen evidence-check forward pass and the completed answer; it is not information available during original generation.',
                      'The learned answer score improves answer detection, but the predeclared joint localization score is worse than old slots on this OOF set.',
                      'Risk scores from class-weighted classifiers are not demonstrated calibrated probabilities; no claims of calibrated hallucination rates.']}
    save(OUT/'RESULTS_AUDIT22.json',result)
    print(json.dumps({'status':'passed','old_baseline_max_error':old_diff,'oof':result['cohort']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4):audit()
