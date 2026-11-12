"""Read-only R23b provenance/OOF/safe-refusal/old-baseline audit; no fitting."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import pickle
import time
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';PRE=ROOT.parent
R22=PRE/'round22_evidence_verification/results'
R24=PRE/'round24_sequence_persistence/results'
BASES=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned',
       'verifier_learned_broadcast','verifier_learned_slots','slots_base_smooth')
NEW=('local_direct','local_probe','local_mean_fusion','local_slots_fusion','local_slots_fusion_smooth')
METHODS=BASES+NEW


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text('utf-8').splitlines() if s]
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def threshold(y,s):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);assert set(y)=={0,1} and np.isfinite(s).all()
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    last=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.r_[0,np.cumsum(yy)[last]];n=np.r_[0,last+1]
    cuts=np.r_[np.nextafter(ss[0],np.inf),ss[last]]
    f1=2*tp/(n+y.sum());precision=np.divide(tp,n,out=np.zeros(len(n)),where=n>0)
    j=max(range(len(n)),key=lambda i:(f1[i],precision[i],cuts[i]))
    return {'threshold':float(cuts[j]),'validation_f1':float(f1[j]),'validation_precision':float(precision[j]),
            'tokens':len(y),'scorable_tokens':len(y),'risk_tokens':int(y.sum())}


def counts(records,method,unit):
    rr=[r for r in records if r['main_eligible']]
    y=np.asarray([r['gold'] for r in rr],bool);p=np.asarray([r['predictions'][method] for r in rr],bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,
        'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'alert_rate':float(p.mean()),'risk_rate':float(y.mean()),unit:len(y),'risk_'+unit:int(y.sum())}


def smooth_logodds(scores,stay,temp):
    if stay==.5 and temp==1.:return scores.copy()
    p=np.clip(scores,1e-12,1-1e-12);e=(np.log(p)-np.log1p(-p))/temp
    lt,ld=np.log(stay),np.log(1-stay)
    f=np.empty(len(e));b=np.empty(len(e));f[0]=e[0];b[-1]=0.
    def transition(z):return np.logaddexp(ld,lt+z)-np.logaddexp(lt,ld+z)
    for i in range(1,len(e)):f[i]=e[i]+transition(f[i-1])
    for i in range(len(e)-2,-1,-1):b[i]=transition(e[i+1]+b[i+1])
    return expit(f+b)


def main():
    started=time.perf_counter();frozen=read(OUT/'fit_freeze.json');complete=read(OUT/'complete.json')
    for manifest in (frozen,complete):
        for name,h in manifest['files_sha256'].items():assert sha(OUT/name)==h,name
    assert read(OUT/'test_started.json')['freeze_sha256']==sha(OUT/'fit_freeze.json')
    assert frozen['snapshot']['code_sha256']==sha(ROOT/'src/run23.py')
    assert frozen['snapshot']['protocol_sha256']==sha(ROOT/'protocol.json')
    assert frozen['snapshot']['feature_manifest_sha256']==sha(ROOT/'data/feature_manifest.json')
    assert frozen['snapshot']['r22_complete_sha256']==sha(R22/'complete22.json')
    assert frozen['snapshot']['r24_complete_sha256']==sha(R24/'complete.json')
    assert read(OUT/'fit_started.json')['utc']<frozen['utc']<read(OUT/'test_started.json')['utc']<complete['utc']
    for folder,name in [(R22,'complete22.json'),(R24,'complete.json')]:
        for n,h in read(folder/name)['files_sha256'].items():assert sha(folder/n)==h
    windows=rows(OUT/'candidate_windows.jsonl');oofw=rows(OUT/'window_scores_oof.jsonl');oofa=rows(OUT/'answer_scores_oof.jsonl')
    assert windows==rows(R22/'candidate_windows22.jsonl')
    assert len(windows)==len(oofw)==12222 and len(oofa)==602
    wm={w['window_key']:w for w in oofw};am={a['item_id']:a for a in oofa}
    assert len(wm)==len(windows) and len(am)==602
    olda={a['item_id']:a for a in rows(R22/'answer_scores_oof22.jsonl')}
    # Compare the unmodified annotations/source identities, not detector scores.
    for a in oofa:
        prior=olda[a['item_id']]
        assert {k:v for k,v in a.items() if k not in ('scores','predictions')}=={k:v for k,v in prior.items() if k not in ('scores','predictions')}
    for w in windows:
        assert {k:wm[w['window_key']][k] for k in w}==w
        assert w['split']=='train' and len(w['item_ids'])==1
    groups=read(PRE/'round18_input_information_diagnostics/data/fold_assignment.json')['groups']
    assert len(groups)==278 and set(groups)=={a['group_id'] for a in oofa}
    assert all(a['fold']==groups[a['group_id']] for a in oofa)
    assert all(w['fold']==groups[w['group_id']] for w in oofw)
    by_item=defaultdict(list)
    for i,w in enumerate(windows):by_item[w['item_ids'][0]].append(i)
    seqs=[np.asarray(sorted(ix,key=lambda j:windows[j]['raw_token_indices'][0])) for ix in by_item.values()]
    gap_count=sum(int(np.count_nonzero(np.diff([windows[j]['raw_token_indices'][0] for j in ix])!=1)) for ix in seqs)
    safe=[a for a in oofa if a['reviewed_safe_refusal'] and a['main_eligible']]
    assert len(safe)==117
    assert all(by_item[a['item_id']] for a in safe)
    assert all(not windows[j]['main_eligible'] for a in safe for j in by_item[a['item_id']])

    fm=read(ROOT/'data/feature_manifest.json');sig=read(ROOT/'data/signature.json');plans=read(ROOT/'data/plans.json')
    assert fm['complete'] and fm['completed_count']==fm['expected_count']==602 and fm['windows_completed']==12222
    assert not fm['labels_used'] and not fm['original_validation_or_test_parsed']
    assert fm['batch_size']==1 and not fm['use_cache'] and not fm['padding']
    assert sig['code_sha256']==sha(ROOT/'src/extract23.py') and digest(sig)==fm['signature_sha256']
    gen_manifest=read(PRE/'round16_dataset_expansion/data/generation_manifest.json')
    lookup={};feature_arrays={};feature_hashes={}
    for rid,rec in fm['records'].items():
        path=ROOT/rec['npz'];sidepath=ROOT/rec['json'];side=read(sidepath)
        assert sha(path)==rec['npz_sha256']==side['npz_sha256']
        assert sha(sidepath)==rec['json_sha256']
        gp=PRE/'round16_dataset_expansion/data/generation_records'/f'{rid}.json';gen=read(gp)
        assert gen['split']=='train' and sha(gp)==side['source_generation_sha256']==gen_manifest['record_sha256'][rid]
        assert plans[rid]['source_generation_sha256']==sha(gp)
        assert plans[rid]['input_row_sha256']==gen['input_row_sha256']
        assert side['row_id']==rid and side['plan_sha256']==digest(plans[rid])
        assert side['signature_sha256']==fm['signature_sha256'] and not side['labels_used'] and not side['output_regenerated']
        assert side['batch_size']==1 and not side['use_cache'] and not side['padding']
        assert am[gen['items'][0]['item_id']]['text']==gen['items'][0]['text']
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        assert a['verifier_hidden'].shape==(side['windows'],3584)
        assert all(np.isfinite(x).all() for x in a.values())
        feature_arrays[rid]=a
        for j,w in enumerate(plans[rid]['windows']):
            assert side['window_keys'][j]==w['window_key'] and w['row_id']==rid
            assert a['window_start'][j]==w['start'] and a['window_end'][j]==w['end']
            assert gen['response'][w['start']:w['end']]==w['text']
            assert w['window_key'] not in lookup
            lookup[w['window_key']]=(rid,j)
        feature_hashes[str(path.resolve())]=sha(path);feature_hashes[str(sidepath.resolve())]=sha(sidepath)
    assert feature_hashes==read(OUT/'feature_files.json')
    with np.load(OUT/'designs.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    for i,w in enumerate(windows):
        rid,j=lookup[w['window_key']];a=feature_arrays[rid];assert rid==w['row_id']
        assert np.array_equal(d['hidden'][i],a['verifier_hidden'][j])
        extra=np.asarray([float(a['ab_logits'][j,1])-float(a['ab_logits'][j,0]),float(a['ab_full_vocab_probabilities'][j].sum())],np.float32)
        assert np.array_equal(d['extra'][i],extra) and d['direct'][i]==a['ab_probabilities'][j,1]
    with np.load(PRE/'round21_semantic_internal_probe/results/designs.npz',allow_pickle=False) as z:
        assert np.array_equal(d['base'],z['base']) and np.array_equal(d['slots'],z['slots'])
    del feature_arrays

    assignments=[];old_exact=0;safe_answer_checks=0;smoothing=[];smooth_error=0.
    for fold in range(5):
        obj=pickle.loads((OUT/f'fold_{fold}_frozen.pkl').read_bytes())
        prior=pickle.loads((R22/f'fold_{fold}_frozen22.pkl').read_bytes())
        gg=obj['groups'];expected={'evaluation_groups':sorted(g for g,f in groups.items() if f==fold),
            'calibration_groups':sorted(g for g,f in groups.items() if f==(fold+1)%5)}
        expected['fit_groups']=sorted(set(groups)-set(expected['evaluation_groups'])-set(expected['calibration_groups']))
        assert gg==expected and gg=={k:prior[k] for k in gg}
        assert not any(set(gg[a])&set(gg[b]) for a,b in [('fit_groups','calibration_groups'),('fit_groups','evaluation_groups'),('calibration_groups','evaluation_groups')])
        assignments.append({k:len(v) for k,v in gg.items()})
        with np.load(OUT/f'fold_{fold}_scores.npz',allow_pickle=False) as z:values={m:z[m] for m in METHODS}
        with np.load(R22/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:
            for m in BASES[:-1]:
                assert np.array_equal(values[m],z[m]) and obj['thresholds'][m]==prior['thresholds'][m];old_exact+=len(values[m])
        smprior=read(R24/f'fold_{fold}_calibration.json')
        with np.load(R24/f'fold_{fold}_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(values['slots_base_smooth'],z['slots_base_smooth'])
            assert obj['thresholds']['slots_base_smooth']==smprior['thresholds']['slots_base_smooth'];old_exact+=12222
        assert np.array_equal(values['local_direct'],d['direct'])
        wi=[i for i,w in enumerate(windows) if w['group_id'] in gg['calibration_groups'] and w['main_eligible']]
        ca=[a for a in oofa if a['group_id'] in gg['calibration_groups'] and a['main_eligible']]
        def limits(v):
            return {'window':threshold([windows[i]['gold'] for i in wi],v[wi]),
                    'answer':threshold([a['gold'] for a in ca],[max(v[by_item[a['item_id']]]) for a in ca])}
        assert limits(values['local_direct'])==obj['thresholds']['local_direct']
        table=[]
        for entry in obj['smoothing_candidates']:
            stay,temp=entry['stay'],entry['temperature'];vv=np.empty(12222)
            for ix in seqs:vv[ix]=smooth_logodds(values['local_slots_fusion'][ix],stay,temp)
            ts=limits(vv)
            for unit in ('window','answer'):
                assert {k:v for k,v in ts[unit].items() if k!='threshold'}=={k:v for k,v in entry['thresholds'][unit].items() if k!='threshold'}
                assert abs(ts[unit]['threshold']-entry['thresholds'][unit]['threshold'])<3e-14
            w,a=ts['window'],ts['answer']
            key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],int(stay==.5 and temp==1),-stay,-abs(np.log(temp))]
            assert key==entry['key'];table.append(key)
            if stay==.5 and temp==1.:assert ts==obj['thresholds']['local_slots_fusion']
            if len(table)-1==obj['smoothing_selected']:
                err=float(np.max(np.abs(vv-values['local_slots_fusion_smooth'])))
                assert err<3e-14;smooth_error=max(smooth_error,err)
        assert max(range(12),key=lambda j:table[j])==obj['smoothing_selected']
        smoothing.append({'fold':fold,'selected_index':obj['smoothing_selected'],'candidates':12,'selection_exact':True})
        for i,w in enumerate(windows):
            if w['group_id'] not in gg['evaluation_groups']:continue
            row=wm[w['window_key']]
            for method in METHODS:
                value=float(values[method][i])
                assert row['scores'][method]==value and row['predictions'][method]==bool(value>=obj['thresholds'][method]['window']['threshold'])
        for a in oofa:
            if a['group_id'] not in gg['evaluation_groups']:continue
            ix=by_item[a['item_id']]
            for method in METHODS:
                expected_value=float(max(values[method][ix])) if ix else None
                assert a['scores'][method]==expected_value
                expected_pred=bool(expected_value>=obj['thresholds'][method]['answer']['threshold']) if expected_value is not None else None
                assert a['predictions'][method]==expected_pred
                if a['reviewed_safe_refusal'] and a['main_eligible']:
                    assert expected_value is not None;safe_answer_checks+=1
    summary=read(OUT/'summary.json');old_summary=read(R22/'summary22.json')['methods'];old_smooth=read(R24/'summary.json')['methods']
    for m in METHODS:
        assert counts(oofw,m,'windows')==summary['methods'][m]['windows']
        assert counts(oofa,m,'answers')==summary['methods'][m]['answers']
        assert summary['methods'][m]['safe_refusals']==117
        assert summary['methods'][m]['safe_refusal_false_positives']==sum(a['predictions'][m] for a in safe)
    for m in BASES[:-1]:assert summary['methods'][m]==old_summary[m]
    assert summary['methods']['slots_base_smooth']==old_smooth['slots_base_smooth']
    assert safe_answer_checks==117*14
    result={'status':'passed','scope':'Only frozen R16 actual-train602 answers/301questions/278event groups. No original heldout or RAGTruth labels read; no GPU/refitting.',
        'feature_rows':602,'all_feature_rows_batch1_no_cache_no_padding':True,'feature_window_bindings':12222,
        'all_source_generation_hashes_actual_train':True,'source_annotation_geometry_unchanged_from_R22':True,
        'all_five_fold_groups_exact_disjoint':True,'fold_group_counts':assignments,'all_OOF_windows_once':12222,'all_OOF_answers_once':602,
        'main_window_denominator':9526,'main_risk_windows':1063,'main_answer_denominator':598,'main_risk_answers':151,
        'safe_refusal_answers':117,'safe_refusal_method_max_checks':safe_answer_checks,
        'safe_refusal_policy':'All candidate windows enter each answer max even when excluded from localization. Only then main_eligible answers are scored; all117 safe refusals remain negative answers. Unlike QA, their windows are not added as negative localization examples.',
        'all14_methods_OOF_scores_and_predictions_exact':True,'all14_main_confusion_metrics_exact':True,
        'old9_baseline_full_fold_values_exact':old_exact,'old9_complete_summary_dicts_exact':True,
        'local_direct_cal_thresholds_exact':True,'smoothing':smoothing,'smoothing_independent_logodds_max_abs':smooth_error,
        'smoothing_raw_start_gap_count':gap_count,'smoothing_geometry':'Original R24 declared answer/item sequence over all candidate windows; no cross-answer or cross-fold propagation.',
        'fit_freeze_precedes_test_recorded':True,'files_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'protocol.json',ROOT/'src/run23.py',
            ROOT/'data/feature_manifest.json',OUT/'fit_freeze.json',OUT/'test_started.json',OUT/'complete.json',OUT/'summary.json']},
        'selected_metrics':{m:{'window_f1':summary['methods'][m]['windows']['f1'],'answer_f1':summary['methods'][m]['answers']['f1']} for m in METHODS},
        'limits':['This audit checks frozen artifact consistency; timestamps are not an external cryptographic proof of historical ordering.',
            'No bootstrap draws or new-method highlight/conditional-ranking metrics were independently recomputed; main window/answer confusion and OOF scores were.',
            'PCA/scaler/45LR numerical audit is separately delegated; no new fit was run.',
            'Repeated development on assistant-annotated R16 is not fresh external testing or proof on human-annotated QA.'],
        'seconds':time.perf_counter()-started}
    (OUT/'RESULTS_AUDIT23B.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['status','feature_rows','feature_window_bindings','safe_refusal_method_max_checks','old9_baseline_full_fold_values_exact','smoothing_independent_logodds_max_abs','smoothing_raw_start_gap_count','seconds']}))


if __name__=='__main__':main()
