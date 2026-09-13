"""Independent audit of actual labels, fitted data roles, decisions and F1."""
import json,pickle,hashlib
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve
from common4 import ROOT,PRE,readl,save,sha
from fit import aggregate,answer_scores,unit_probs

def main():
    rows=readl(ROOT/'data/rows.jsonl');rr={r['id']:r for r in rows};anns={r['id']:r for r in readl(ROOT/'data/annotations.jsonl')};manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert len(rows)==len(rr)==684 and len({r['group'] for r in rows})==684
    assert Counter(r['split'] for r in rows)=={'train':464,'val':120,'test':100}
    assert sha(ROOT/'data/rows.jsonl')==manifest['rows_sha256'] and sha(ROOT/'data/annotations.jsonl')==manifest['annotations_sha256']
    old=readl(PRE/'round2/data/news_large/labeled.jsonl')+readl(PRE/'round2/data/news_confirmation/rows.jsonl')+[r for r in readl(PRE/'round3/data/rows.jsonl') if r['split']=='test'];oldgroups={r['group'] for r in old}
    assert not oldgroups&{r['group'] for r in rows if r['split']=='test'}
    raw={r['id']:r for r in readl(PRE/'data/raw/response.jsonl')};sources={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
    for r in rows:
        a=raw[r['id']];s=sources[r['source_id']];assert r['response']==a['response'] and r['evidence']==s['source_info'] and r['label']==int(bool(a['labels']))
        assert hashlib.sha256(' '.join(r['evidence'].split()).encode()).hexdigest()==r['group']
        assert anns[r['id']]['spans']==a['labels'];assert all(p['label_type']=='Evident Conflict' and not p.get('implicit_true') and not p.get('due_to_null') for p in a['labels'])
        for p in a['labels']:assert r['response'][p['start']:p['end']].strip()==p['text'].strip()
    query_manifest=json.loads((ROOT/'data/query_manifest.json').read_text());trids={r['id'] for r in rows if r['split']=='train'};covered=[]
    for fold in query_manifest['source_OOF_folds']:
        a=set(fold['fit_ids']);b=set(fold['predicted_ids']);assert not a&b and a|b==trids
        assert not {rr[i]['group'] for i in a}&{rr[i]['group'] for i in b};covered+=list(b)
    assert len(covered)==len(set(covered))==len(trids)
    b=pickle.loads((ROOT/'data/unit_inputs.pkl').read_bytes());qs=b['queries'];assert qs==readl(ROOT/'data/queries.jsonl');train=np.array([q['split']=='train' for q in qs]);mean=np.zeros(14336,np.float64)
    assert b['pca'].n_samples_==int(train.sum()) and b['pca'].n_components_==16
    records=readl(ROOT/'data/readouts/records.jsonl');recs={r['query_id']:r for r in records};assert len(recs)==len(records)==len(qs)
    for j,q in enumerate(qs):
        r=rr[q['id']];lo,hi=q['target_span'];a,c=q['sentence_span'];assert 0<=a<=lo<hi<=c<=len(r['response'])
        assert r['response'][lo:hi]==q['target_text'] and r['response'][a:c]==q['statement'] and r['evidence']==q['evidence']
        assert not any(k in q for k in ['label','gold','expected','answer','spans'])
        truth=int(any(lo<s['end'] and hi>s['start'] for s in anns[r['id']]['spans']));assert truth==b['y'][j]
        with np.load(ROOT/f'data/readouts/{q["query_id"]}.npz') as z:
            assert z['hidden'].shape==(4,3584) and z['probs'].shape==(3,) and all(np.isfinite(z[k]).all() for k in z.files)
            assert abs(z['probs'].sum()-1)<1e-6 and z['probs'].min()>=0
            if train[j]:mean+=z['hidden'].ravel().astype(np.float64)
            rec=recs[q['query_id']]
            if rec['origin'].startswith('round3'):
                with np.load(PRE/f'round3/data/supplement/features/direct_{rec["original_query_id"]}.npz') as f:assert np.array_equal(f['prompt'],z['hidden']) and np.array_equal(f['verdict_probs'],z['probs'])
    difference=float(np.linalg.norm(mean/train.sum()-b['pca'].mean_)/(np.linalg.norm(b['pca'].mean_)+1e-9));assert difference<1e-4,difference
    nets=pickle.loads((ROOT/'results/checkpoints/unit_models.pkl').read_bytes());assert len(nets)==24
    va=b['base']['split_indices']['val'];yv=np.array([b['base']['rows'][i]['label'] for i in va]);prob={}
    for mid,cp in nets.items():
        assert cp['model'][0].n_samples_seen_==train.sum();assert np.allclose(cp['model'][0].mean_,b['x'][cp['feature']][train].mean(0,dtype=np.float64),rtol=1e-5,atol=1e-5)
        prob[mid]=cp['model'].predict_proba(b['x'][cp['feature']])[:,1]
    for mid in ['raw_B','raw_B_over_AB']:prob[mid]=unit_probs(b,nets,mid)
    trials=readl(ROOT/'results/validation_trials.jsonl');assert len(trials)==630;maxdiff=0
    for c in trials:
        if c['model_id'] is None:s=np.array([aggregate(b['base'][c['scope']][i],c['aggregate']) for i in va])
        else:s,_=answer_scores(b,prob[c['model_id']],'val',c['scope'],c['aggregate'])
        pred=s>=c['threshold'];f=f1_score(yv,pred);p=precision_score(yv,pred,zero_division=0);maxdiff=max(maxdiff,abs(f-c['val_metrics']['f1']));assert abs(f-c['val_metrics']['f1'])<1e-12 and abs(p-c['val_metrics']['precision'])<1e-12
        pp,rec,thresholds=precision_recall_curve(yv,s);ff=2*pp[:-1]*rec[:-1]/np.maximum(1e-15,pp[:-1]+rec[:-1]);j=max(range(len(ff)),key=lambda j:(round(ff[j],12),round(pp[j],12),thresholds[j]));assert abs(ff[j]-f)<1e-12 and abs(pp[j]-p)<1e-12
    key=lambda c:(c['val_metrics']['f1'],c['val_metrics']['precision'],-c['mean_calls'],-c['candidate_index']);selection=json.loads((ROOT/'results/selection.json').read_text());assert selection['selected']==max(trials,key=key)
    for fam,c in selection['family_winners'].items():assert c==max([x for x in trials if x['family']==fam],key=key)
    predictions=readl(ROOT/'results/test_predictions.jsonl');assert {r['id'] for r in predictions}=={r['id'] for r in rows if r['split']=='test'}
    metrics=json.loads((ROOT/'results/metrics.json').read_text());yy=np.array([rr[p['id']]['label'] for p in predictions]);assert yy.sum()==40
    for m in metrics:
        if m['name']=='always_error':p=np.ones(len(yy),bool)
        else:
            p=np.array([r['methods'][m['name']]['pred'] for r in predictions]);s=np.array([r['methods'][m['name']]['score'] for r in predictions]);assert np.array_equal(s>=m['config']['threshold'],p)
        assert abs(m['f1']-f1_score(yy,p))<1e-12 and abs(m['precision']-precision_score(yy,p,zero_division=0))<1e-12 and abs(m['recall']-recall_score(yy,p))<1e-12
        assert m['tp']==int((p&(yy==1)).sum()) and m['fp']==int((p&(yy==0)).sum()) and m['fn']==int((~p&(yy==1)).sum()) and m['tn']==int((~p&(yy==0)).sum())
    ev=json.loads((ROOT/'results/evaluation_manifest.json').read_text());assert ev['selection_sha256']==sha(ROOT/'results/selection.json')
    selected=next(m for m in metrics if m['name']=='selected');assert ev['selected_primary_target_met']==(selected['f1']>.7)
    save(ROOT/'results/audit.json',{'passed':True,'n_test':100,'all_prior584_sources_excluded':True,'original_human_labels_verified':True,'source_OOF_verified':True,
        'query_spans_and_labels_verified':len(qs),'training_only_transform_verified':True,'pca_mean_relative_error':difference,'validation_candidates_independently_checked':len(trials),
        'F1_max_difference':maxdiff,'selected_test_F1':selected['f1'],'target_met':selected['f1']>.7,
        'code_sha256':{p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))}})
    print('INDEPENDENT AUDIT PASSED',selected['f1'],flush=True)

if __name__=='__main__':main()
