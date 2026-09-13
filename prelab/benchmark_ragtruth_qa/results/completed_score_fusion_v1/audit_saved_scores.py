"""Independent saved-score arithmetic/count audit; no fitting or model forward."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
QA=OUT.parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metric(y,s,t):
    positive=y==1;pred=s>=t
    tp=int((positive&pred).sum());fp=int((~positive&pred).sum());fn=int((positive&~pred).sum());tn=int((~positive&~pred).sum())
    return {'n':len(y),'positive':int(positive.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def best_threshold(y,s):
    # Ascending prefixes + searchsorted, independent of producer descending
    # last-index implementation. This verifies its frozen choice; no change saved.
    order=np.argsort(s,kind='stable');ss=s[order];yy=y[order]
    ts=np.r_[np.unique(ss),np.nextafter(ss[-1],np.inf)]
    first=np.searchsorted(ss,ts,side='left');prefix=np.r_[0,np.cumsum(yy)]
    tp=yy.sum()-prefix[first];fp=len(yy)-first-tp;fn=yy.sum()-tp
    denominator=2*tp+fp+fn
    f1=np.divide(2*tp,denominator,out=np.zeros(len(ts)),where=denominator>0)
    precision=np.divide(tp,tp+fp,out=np.zeros(len(ts)),where=tp+fp>0)
    i=int(np.lexsort((ts,precision,f1))[-1])
    return {'threshold':float(ts[i]),'f1':float(f1[i]),'precision':float(precision[i]),'rows':len(y),'positive':int(y.sum())}

def run():
    complete=read(OUT/'complete.json');assert sha(OUT/'summary.json')==complete['summary_sha256']
    summary=read(OUT/'summary.json');data=QA/'data'
    w=rows(data/'windows_k4_fit.jsonl')+rows(data/'windows_k4_calibration.jsonl')
    a=rows(data/'answers_fit.jsonl')+rows(data/'answers_calibration.jsonl')
    t=rows(data/'tokens_fit.jsonl')+rows(data/'tokens_calibration.jsonl')
    assert len(w)==210364 and len(a)==793
    assert all(x['partition']=='fit' for x in w[:168123]+a[:634])
    assert all(x['partition']=='calibration' for x in w[168123:]+a[634:])
    wy=np.asarray([x['label'] for x in w]);ay=np.asarray([x['label'] for x in a]);aid={x['response_id']:i for i,x in enumerate(a)}
    owner=np.asarray([aid[x['response_id']] for x in w]);tok={x['response_id']:x for x in t}
    taildir=QA/'results/minicheck_tail_all_docs_v3/tail2';tc=read(taildir/'complete.json');te=tc['selected']
    tokenfile=taildir/f'epoch_{te["epoch"]:02d}_token_predictions.npz'
    assert sha(tokenfile)==te['artifacts_sha256']['_token_predictions.npz']
    with np.load(tokenfile,allow_pickle=False) as z:
        native={x['response_id']:z[x['response_id']].copy() for x in a}
    expected_tail=np.asarray([max(float(native[x['response_id']][j]) for j in x['token_indices'] if tok[x['response_id']]['lexical_mask'][j]) for x in w])
    peers={'lookback':('lookback_regularization_v2','lb_prefix_pre_header'),
           'harp_claim':('claim_pooling_v1','full_lb_harp64_tcn'),
           'semantic_claim':('claim_pooling_v1','minicheck_hidden64_risk_tcn_w32')}
    checks=[];selected={};sources={str(tokenfile.resolve()):sha(tokenfile)}
    for family,(folder,method) in peers.items():
        directory=QA/'results'/folder;e=read(directory/'summary.json')['selected'][method]
        peerfile=directory/(e['candidate']+'_scores.npz');assert sha(peerfile)==e['scores_sha256']
        sources[str(peerfile.resolve())]=sha(peerfile)
        with np.load(peerfile,allow_pickle=False) as z:peer=z['window_scores' if 'window_scores' in z else 'scores'].astype(np.float64)
        candidates=summary['all_candidates'][family];assert [c['tail_weight'] for c in candidates]==[0.,.25,.5,.75,1.]
        expected_keys=[]
        for c in candidates:
            alpha=c['tail_weight'];path=OUT/(c['candidate']+'_scores.npz');assert sha(path)==c['scores_sha256']
            with np.load(path,allow_pickle=False) as z:wv=z['window_scores'].copy();av=z['answer_scores'].copy()
            assert np.array_equal(wv,(1-alpha)*peer+alpha*expected_tail)
            maxima=np.full(len(a),-np.inf);np.maximum.at(maxima,owner,wv)
            assert np.array_equal(av,maxima),'Every answer max must include all original eligible windows'
            assert np.isfinite(wv).all() and np.all((wv>=0)&(wv<=1))
            count_tables={}
            for part,wl,wr,al,ar in [('fit',0,168123,0,634),('calibration',168123,210364,634,793)]:
                count_tables[part]={}
                for unit,y,s,th in [('windows',wy[wl:wr],wv[wl:wr],c['thresholds']['window']['threshold']),
                                    ('answers',ay[al:ar],av[al:ar],c['thresholds']['answer']['threshold'])]:
                    m=metric(y,s,th)
                    assert all(value==c['metrics'][part][unit][key] for key,value in m.items()),(c['candidate'],part,unit)
                    count_tables[part][unit]=m
            thresholds={'window':best_threshold(wy[168123:],wv[168123:]),'answer':best_threshold(ay[634:],av[634:])}
            assert thresholds==c['thresholds']
            key=[min(thresholds['window']['f1'],thresholds['answer']['f1']),thresholds['window']['f1'],thresholds['window']['precision'],-alpha]
            assert key==c['selection_key'];expected_keys.append(key)
            if alpha==0:assert np.array_equal(wv,peer) and thresholds==e['thresholds']
            if alpha==1:assert np.array_equal(wv,expected_tail) and thresholds==te['thresholds']
            checks.append({'candidate':c['candidate'],'tail_weight':alpha,'formula_exact':True,'all793_answer_maxima_exact':True,
                'threshold_choice_independently_verified':True,'counts':count_tables,'alpha_endpoint_exact':alpha in (0,1)})
        best=candidates[max(range(len(candidates)),key=lambda i:expected_keys[i])]
        assert best==summary['selected'][family]
        selected[family]={'candidate':best['candidate'],'alpha':best['tail_weight'],
            'window_f1':best['metrics']['calibration']['windows']['f1'],'answer_f1':best['metrics']['calibration']['answers']['f1']}
    assert len(checks)==15
    fix=OUT/'PRE_RUN_ALIGNMENT_FIX.json'
    result={'status':'passed','scope':'All15 saved candidates arithmetic, original gold denominators, TP/FP/FN/TN/P/R/F1 for fit634 subset and cal159, independent cal threshold/candidate-choice verification; no neural replay.',
        'fit_windows':168123,'calibration_windows':42241,'fit_answers':634,'calibration_answers':159,
        'native_tail_endpoint_rebuilt_with_lexical_mask':True,'candidate_checks':checks,'selected':selected,
        'same_five_alpha_budget_for_each_of_three_peers':True,'actual_source_counts':'All peers fit634; same tail2 fit3680 for every combination.',
        'source_sha256':sources,'summary_sha256':sha(OUT/'summary.json'),'script_sha256':sha(__file__),
        'pre_run_alignment_fix_present':fix.exists(),'pre_run_alignment_fix_sha256':sha(fix) if fix.exists() else None,
        'AUROC_AP_recomputed':False,'model_fitting':False,'GPU_used':False,'official_test_opened':False,
        'limits':'These are repeated calibration selections. This audit validates arithmetic, not future generalization or matched training data across model families.'}
    (OUT/'INDEPENDENT_SAVED_SCORE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','candidates':len(checks),'selected':selected},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
