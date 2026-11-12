"""Independent CPU-only audit; never imports the training/metric implementation."""
from pathlib import Path
import json, hashlib, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
QA=BASE.parents[1]
REPORT=HERE/'INDEPENDENT_RESULT_AUDIT.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
    return h.hexdigest()
def equal(actual,expected):
    if isinstance(expected,dict):
        for k,v in expected.items(): equal(actual[k],v)
    elif isinstance(expected,float): assert abs(float(actual)-expected)<=1e-12,(actual,expected)
    else: assert actual==expected,(actual,expected)
def metrics(y,s,t):
    pred=s>=t; yy=y.astype(bool)
    tp=int((pred&yy).sum());fp=int((pred&~yy).sum());fn=int((~pred&yy).sum());tn=int((~pred&~yy).sum())
    return dict(n=len(y),positive=int(yy.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))
def threshold(y,s):
    # Ascending unique-score histogram; does not use trainer's descending sort routine.
    values,inv=np.unique(s.astype(np.float64),return_inverse=True)
    n=np.bincount(inv,minlength=len(values));pos=np.bincount(inv,weights=y,minlength=len(values)).astype(np.int64)
    above=np.cumsum(n[::-1])[::-1];tp=np.cumsum(pos[::-1])[::-1];total=int(y.sum())
    cuts=np.r_[values,np.nextafter(values[-1],np.inf)]
    predicted=np.r_[above,0];true=np.r_[tp,0]
    f1=2*true/(predicted+total)
    precision=np.divide(true,predicted,out=np.zeros(len(cuts)),where=predicted>0)
    best=max(range(len(cuts)),key=lambda k:(f1[k],precision[k],cuts[k]))
    return dict(threshold=float(cuts[best]),f1=float(f1[best]),precision=float(precision[best]),rows=len(y),positive=total)

def run():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4);started=time.perf_counter()
    source_checks=[]
    for manifest in ('protocol_freeze.json','preparation_complete.json'):
        for p,h in read(BASE/manifest)['files_sha256'].items():
            assert sha(p)==h,p;source_checks.append(p)
    export=read(QA/'fit_expansion/data/export_freeze.json')
    gold_manifest=read(QA/'data/gold_manifest.json')
    answers=[];tokens=[];metadata_paths=[]
    for part,directory,n in [('fit',QA/'fit_expansion/data',3680),('calibration',QA/'data',159)]:
        ap=directory/f'answers_{part}.jsonl';tp=directory/f'tokens_{part}.jsonl'
        for p in (ap,tp):
            expected=(export['output_files_sha256'][str(p.resolve())] if part=='fit' else
                next(r['sha256'] for r in gold_manifest['outputs'] if (QA/r['path']).resolve()==p.resolve()))
            assert sha(p)==expected;metadata_paths.append(p)
        aa=lines(ap);tt=lines(tp);assert len(aa)==len(tt)==n
        answers.extend(aa);tokens.extend(tt)
    assert len({a['response_id'] for a in answers})==3839
    groups=[{a['group_id'] for a in answers[:3680]},{a['group_id'] for a in answers[3680:]}]
    assert len(groups[0])==615 and len(groups[1])==154 and groups[0].isdisjoint(groups[1])
    geometries=[];risk_counts=[]
    for a,t in zip(answers,tokens):
        assert a['response_id']==t['response_id'] and a['partition']==t['partition']
        text=a['original_response'];assert text==t['original_response']
        assert hashlib.sha256(text.encode()).hexdigest()==a['answer_sha256']==t['answer_sha256']
        marked=np.zeros(len(text),bool)
        for s in a['original_labels']:
            assert text[s['start']:s['end']]==s['text']
            marked[s['start']:s['end']]=True
        offsets=t['response_token_offsets'];lex=[];risk=[];nonspace=[]
        for start,end in offsets:
            positions=[j for j in range(start,end) if text[j].isalnum()]
            lex.append(bool(positions));risk.append(any(marked[j] for j in positions))
            nonspace.append(any(not c.isspace() for c in text[start:end]))
        lex=np.asarray(lex,bool);risk=np.asarray(risk,np.int8)
        assert np.array_equal(lex,t['lexical_mask']) and np.array_equal(risk,t['risk_mask'])
        assert a['label']==int(bool(a['original_labels']))==t['answer_risk']
        n=len(offsets);assert n==t['token_count']
        ix=np.arange(max(n-3,1))[:,None]+np.arange(min(n,4))[None,:]
        use=lex[ix];keep=use.any(axis=1);ix=ix[keep];use=use[keep]
        wy=risk[ix].any(axis=1).astype(np.int8)
        assert len(wy)==a['eligible_window_count'] and int(wy.sum())==a['positive_window_count']
        geometries.append((ix,use,wy,np.asarray(nonspace,bool)))
        risk_counts.append(int(risk.sum()))
    print('AUDIT_GOLD_COMPLETE',len(answers),flush=True)
    index=read(BASE/'mapped_index.json')['answers'];assert len(index)==3839
    matrix=np.load(BASE/'mapped_final_hidden.npy',mmap_mode='r');assert matrix.shape==(906338,1024) and matrix.dtype==np.float32
    cursor=0
    for a,t,row,g in zip(answers,tokens,index,geometries):
        assert row['response_id']==a['response_id'] and row['partition']==a['partition']
        assert np.array_equal(row['nonspace'],g[3]) and 1<=len(row['doc_ranges'])<=2
        for lo,hi in row['doc_ranges']:
            assert lo==cursor and hi-lo==t['token_count'];cursor=hi
    assert cursor==len(matrix)
    complete=read(HERE/'complete.json');assert complete['status']=='complete_development_only'
    records=[];saved_predictions={}
    for epoch in range(4):
        stem=f'epoch_{epoch:02d}';entry=read(HERE/(stem+'.json'))
        for ext,h in entry['artifacts_sha256'].items(): assert sha(HERE/(stem+ext))==h
        with np.load(HERE/(stem+'_token_predictions.npz')) as z:
            assert set(z.files)=={a['response_id'] for a in answers}
            probabilities={k:z[k].copy() for k in z.files}
        if epoch==complete['selected']['epoch']: saved_predictions=probabilities
        with np.load(HERE/(stem+'_scores.npz')) as z: original={k:z[k].copy() for k in z.files}
        pieces={}
        for part,indices in [('fit',range(3680)),('cal',range(3680,3839))]:
            ws=[];wy=[];ss=[];yy=[];ends=[0]
            for i in indices:
                a=answers[i];ix,mask,risk,_=geometries[i];p=probabilities[a['response_id']]
                assert p.shape==(tokens[i]['token_count'],) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                one=np.where(mask,p[ix],-np.inf).max(axis=1)
                ws.extend(one);wy.extend(risk);ss.append(one.max());yy.append(a['label']);ends.append(len(ws))
            pieces[part]={'window_scores':np.asarray(ws,dtype=np.float64),'window_labels':np.asarray(wy),
                'answer_scores':np.asarray(ss,dtype=np.float64),'answer_labels':np.asarray(yy),'answer_window_offsets':np.asarray(ends)}
            for k,v in pieces[part].items(): assert np.array_equal(v,original[part+'_'+k]),(epoch,part,k)
        thresholds={g:threshold(pieces['cal'][g+'_labels'],pieces['cal'][g+'_scores']) for g in ('window','answer')}
        equal(thresholds,entry['thresholds'])
        checked={}
        for part,name in [('fit','fit_at_cal_thresholds'),('cal','calibration')]:
            mm={g+'s':metrics(pieces[part][g+'_labels'],pieces[part][g+'_scores'],thresholds[g]['threshold']) for g in ('window','answer')}
            equal(mm,entry[name]);checked[part]=mm
        key=[min(thresholds['window']['f1'],thresholds['answer']['f1']),thresholds['window']['f1'],thresholds['window']['precision'],-epoch]
        equal(key,entry['selection_key']);equal(entry,complete['all_epochs'][epoch])
        records.append({'epoch':epoch,'thresholds':thresholds,'metrics':checked,'selection_key':key})
        print('AUDIT_EPOCH',epoch,'COUNTS_THRESHOLDS_EXACT',flush=True)
    selected=max(records[1:],key=lambda r:r['selection_key'])
    assert complete['selected']['epoch']==selected['epoch']==1
    state=torch.load(HERE/'epoch_01.pt',map_location='cpu',weights_only=False)
    weight=state['model_state_dict']['weight'];bias=state['model_state_dict']['bias']
    assert weight.shape==(1,1024) and bias.shape==(1,) and weight.dtype==bias.dtype==torch.float32
    with np.load(BASE/'training_weights.npz') as z: ww={k:z[k].copy() for k in z.files}
    maxdiff=0.;equal_answers=0;fit_bce=0.;row_count=0
    with torch.inference_mode():
        for i,(a,row,geo) in enumerate(zip(answers,index,geometries)):
            values=[]
            for lo,hi in row['doc_ranges']:
                x=torch.from_numpy(np.array(matrix[lo:hi],copy=True));assert torch.isfinite(x).all()
                value=F.linear(x,weight,bias).ravel();value[~torch.from_numpy(geo[3])]=0.
                values.append(value);row_count+=len(x)
            # Explicit per-position minimum across all available documents.
            logit=values[0]
            for value in values[1:]:logit=torch.minimum(logit,value)
            p=torch.sigmoid(logit).numpy();reference=saved_predictions[a['response_id']]
            difference=float(np.max(np.abs(p-reference)));maxdiff=max(maxdiff,difference)
            assert np.array_equal(p,reference),(a['response_id'],difference)
            equal_answers+=1
            if i<3680:
                lo,hi=ww['bounds'][i];yy=torch.as_tensor(ww['y'][lo:hi],dtype=torch.float32)
                loss=F.binary_cross_entropy_with_logits(logit,yy,reduction='none')*torch.as_tensor(ww['loss'][lo:hi],dtype=torch.float32)
                fit_bce+=float(loss.double().sum())
    assert row_count==906338 and equal_answers==3839
    equal(fit_bce/int(ww['target_mass']),read(HERE/'epoch_01.json')['fit_weighted_bce'])
    assert not torch.cuda.is_initialized()
    report={'status':'passed','scope':'Independent human-span/offset gold, raw4 geometry, all4 stored epochs, selected CPU linear+all-doc-min replay; no training/GPU/test',
        'answers':3839,'fit_answers':3680,'calibration_answers':159,'groups':{'fit':615,'calibration':154},
        'raw_tokens':sum(t['token_count'] for t in tokens),'risk_lexical_tokens':sum(risk_counts),
        'counts':{'fit_windows':653979,'fit_positive_windows':58433,'calibration_windows':42241,'calibration_positive_windows':5984},
        'epochs':records,'selected_epoch':1,'epoch0_excluded_from_selection':True,
        'selected_cpu_replay':{'mapped_rows':row_count,'answers_exact':equal_answers,'maximum_probability_difference':maxdiff,
            'all_documents_min_after_mapping':True,'fit_weighted_bce_exact':True},
        'hash_checks':{'frozen_protocol_and_preparation_artifacts':len(source_checks),'metadata_paths':[str(p) for p in metadata_paths],
            'complete_sha256':sha(HERE/'complete.json'),'selected_checkpoint_sha256':sha(HERE/'epoch_01.pt')},
        'cache_audit_scope':'Mapped matrix and complete cache source manifest hashes verified; immutable lower22 arrays not all reread during this bounded result audit. Original full cache geometry/numeric gates are preserved.',
        'interpretation':'This is a same-budget extra semantic-checker control, not evidence it beats all existing baselines. Calibration remains development; no new final-test claim.',
        'seconds':time.perf_counter()-started,'training_record_audit':'INDEPENDENT_TRAINING_RECORD_AUDIT.json (separately produced)',
        'GPU_used':False,'test_opened':False,'retrained':False}
    snapshot=read(BASE/'source_snapshot.json')['files_sha256']
    manifests=[QA/'fit_expansion/minicheck/encoder22_feature_manifest.json',
        QA/'fit_expansion/minicheck_backfill_v2/encoder22_feature_manifest.json',
        QA/'fit_expansion/minicheck/claim_feature_manifest.json',
        QA/'semantic_baseline/cuda_variant/claim_feature_manifest.json',
        QA/'fit_expansion/minicheck_unselected_v2/feature_manifest.json']
    for p in manifests: assert sha(p)==snapshot[str(p)] and read(p)['status']=='complete'
    report['hash_checks']['actual_cache_manifest_links']={str(p):sha(p) for p in manifests}
    training_report=HERE/'INDEPENDENT_TRAINING_RECORD_AUDIT.json'
    if training_report.exists():
        assert read(training_report)['status']=='passed'
        report['training_record_audit_sha256']=sha(training_report)
        report['combined_training_and_result_audit_status']='passed'
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FROZEN0_INDEPENDENT_RESULT_AUDIT_PASSED',selected['metrics']['cal']['windows']['f1'],selected['metrics']['cal']['answers']['f1'],flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):run()
    except Exception:
        REPORT.with_name('INDEPENDENT_RESULT_AUDIT_FAILURE.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()},indent=2)+'\n')
        raise
