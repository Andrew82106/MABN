import json,pickle,hashlib
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,precision_recall_curve
from common5 import R5,R4,PRE,readl,save,sha
from readouts import prompt
from run_confirmation import features
from analyze_oracle import load_inputs,score

def main():
    protocol=json.loads((R5/'protocol.json').read_text());manifest=json.loads((R5/'results/analysis_manifest.json').read_text());assert sha(R5/'protocol.json')==manifest['frozen_protocol_sha256']
    for p,h in protocol['frozen_dependencies'].items():assert sha(R4/p)==h
    for p,h in protocol['data_sha256'].items():assert sha(R5/'data'/p)==h
    rows=readl(R5/'data/rows.jsonl');rr={r['id']:r for r in rows};ann={a['id']:a['spans'] for a in readl(R5/'data/annotations.jsonl')};assert len(rr)==len({r['group'] for r in rows})==170
    raw={r['id']:r for r in readl(PRE/'data/raw/response.jsonl')};src={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
    for r in rows:
        original=raw[r['id']];assert r['response']==original['response'] and r['evidence']==src[r['source_id']]['source_info'] and ann[r['id']]==original['labels'] and r['label']==int(bool(ann[r['id']]))
        assert r['group']==hashlib.sha256(' '.join(r['evidence'].split()).encode()).hexdigest()
        for a in ann[r['id']]:assert a['label_type']=='Evident Conflict' and not a.get('implicit_true') and not a.get('due_to_null') and r['response'][a['start']:a['end']].strip()==a['text'].strip()
    b=pickle.loads((R4/'data/unit_inputs.pkl').read_bytes());fit_groups={r['group'] for r in b['base']['rows'] if r['split']=='train'};assert not fit_groups&{r['group'] for r in rows}
    qs,index,x,nets=load_inputs();recs=readl(R5/'data/readouts/records.jsonl');rec={r['query_id']:r for r in recs};assert len(rec)==len(recs)==len(qs)
    allowed={'query_id','id','split','kind','sentence_span','target_span','statement','target_text','evidence','base_features'}
    copied=0
    for q in qs:
        assert set(q)==allowed and q['kind'] in ['sentence','fact'];r=rr[q['id']];lo,hi=q['target_span'];a,c=q['sentence_span']
        assert 0<=a<=lo<hi<=c<=len(r['response']) and r['response'][lo:hi]==q['target_text'] and r['response'][a:c]==q['statement'] and r['evidence']==q['evidence']
        record=rec[q['query_id']];assert record['prompt_sha256']==hashlib.sha256(prompt(q).encode()).hexdigest()
        with np.load(R5/f'data/readouts/{q["query_id"]}.npz') as z:
            assert z['hidden'].shape==(4,3584) and z['probs'].shape==(3,) and np.isfinite(z['hidden']).all() and np.isfinite(z['probs']).all() and abs(z['probs'].sum()-1)<1e-6
            if record['origin']=='exact prior prompt cache':
                with np.load(record['path']) as old:assert np.array_equal(z['hidden'],old['hidden']) and np.array_equal(z['probs'],old['probs']);copied+=1
    check=[i for i,q in enumerate(b['queries']) if q['split']=='val'][:21];xx=features([b['queries'][i] for i in check],R4/'data/readouts',b['pca'])
    for k in xx:assert np.allclose(xx[k],b['x'][k][check],rtol=1e-4,atol=1e-4)
    cases=readl(R5/'data/cases.jsonl');assert Counter(c['split'] for c in cases)=={'val':90,'diagnostic':40};pairs={}
    for c in cases:
        r=rr[c['id']];q=qs[index[c['span_query']]];assert q['id']==r['id'] and r['label']==c['label'];pairs.setdefault(c['pair_id'],[]).append(c)
        if c['label']:
            first=min(ann[c['id']],key=lambda a:(a['start'],a['end']));assert q['target_span']==[first['start'],first['end']]
        else:assert ann[c['id']]==[]
    assert len({c['id'] for c in cases})==len(cases)
    for pair in pairs.values():assert len(pair)==2 and {c['label'] for c in pair}=={0,1} and len({rr[c['id']]['original_model'] for c in pair})==1 and len({c['split'] for c in pair})==1
    thresholds=json.loads((R5/'results/unit_thresholds.json').read_text());assert sha(R5/'results/unit_thresholds.json')==manifest['unit_thresholds_sha256'];val=[c for c in cases if c['split']=='val'];test=[c for c in cases if c['split']=='diagnostic'];yv=np.array([c['label'] for c in val]);yt=np.array([c['label'] for c in test]);preds=readl(R5/'results/unit_predictions.jsonl');assert [c['case_id'] for c in preds]==[c['case_id'] for c in test]
    ms=json.loads((R5/'results/unit_metrics.json').read_text())
    for c,m in zip(thresholds['thresholds'],ms):
        assert c['mode']==m['mode'] and c['method']==m['method']
        ix=[index[a[c['mode']+'_query']] for a in val];s=score(qs,x,nets,c['method'],ix);p=s>=c['threshold'];pp,recall,_=precision_recall_curve(yv,s);ff=2*pp*recall/np.maximum(1e-15,pp+recall)
        assert abs(f1_score(yv,p)-max(ff))<1e-12 and abs(f1_score(yv,p)-c['validation']['f1'])<1e-12
        ix=[index[a[c['mode']+'_query']] for a in test];s=score(qs,x,nets,c['method'],ix);p=s>=c['threshold'];name=c['mode']+'_'+c['method']
        assert np.allclose(s,[a['methods'][name]['score'] for a in preds],atol=1e-12) and np.array_equal(p,[a['methods'][name]['pred'] for a in preds])
        assert abs(f1_score(yt,p)-m['f1'])<1e-12 and abs(precision_score(yt,p)-m['precision'])<1e-12 and abs(recall_score(yt,p)-m['recall'])<1e-12 and abs(roc_auc_score(yt,s)-m['auc'])<1e-12
    policies=readl(R5/'data/policies.jsonl');diagnostic=[p for p in policies if p['split']=='diagnostic'];pr=score(qs,x,nets,'frozen_internal_probe',list(range(len(qs))));pipeline=readl(R5/'results/pipeline_predictions.jsonl');yy=np.array([p['label'] for p in pipeline]);assert yy.sum()==20 and len(yy)==50
    assert [p['id'] for p in pipeline]==[p['id'] for p in diagnostic]
    for p in policies:
        assert len(p['automatic'])==len(p['oracle_sentence'])==len(p['oracle_span'])
        if not p['label']:assert p['automatic']==p['oracle_sentence']==p['oracle_span']
        else:
            a=min(ann[p['id']],key=lambda a:(a['start'],a['end']));span=[a['start'],a['end']]
            assert any(qs[index[j]]['kind']=='sentence' and qs[index[j]]['target_span'][0]<=span[0] and qs[index[j]]['target_span'][1]>=span[1] for j in p['oracle_sentence'])
            assert any(qs[index[j]]['kind']=='fact' and qs[index[j]]['target_span']==span for j in p['oracle_span'])
    for m in json.loads((R5/'results/pipeline_metrics.json').read_text()):
        mode=m['mode'];s=np.array([1-np.prod(1-np.clip(pr[[index[j] for j in p[mode]]],0,1-1e-8)) for p in diagnostic]);pp=s>=m['threshold']
        assert np.allclose(s,[p['methods'][mode]['score'] for p in pipeline],atol=1e-12) and np.array_equal(pp,[p['methods'][mode]['pred'] for p in pipeline])
        assert abs(f1_score(yy,pp)-m['f1'])<1e-12 and abs(precision_score(yy,pp)-m['precision'])<1e-12 and abs(recall_score(yy,pp)-m['recall'])<1e-12
        assert m['tp']==int((pp&(yy==1)).sum()) and m['fp']==int((pp&(yy==0)).sum())
    replay=json.loads((R5/'results/readout_manifest.json').read_text())['replay_checks'];assert replay and all(r['hook_logits_unchanged'] and r['hidden_difference']==0 and r['ABC_difference']==0 for r in replay)
    save(R5/'results/audit.json',{'passed':True,'diagnostic_not_automatic_test':True,'original_human_labels_and_sources_verified':True,'no_train_source_overlap':True,'matched_units':len(cases),'same_generator_unique_source_pairs':len(pairs),'queries_with_exact_text_alignment':len(qs),'cached_arrays_verified':copied,'no_verdict_labels_in_model_inputs':True,'prior_transform_feature_assembly_verified':21,'all_unit_and_pipeline_metrics_recomputed':True,'equal_call_budgets_and_unchanged_clean_inputs':True,'code_sha256':{p.name:sha(p) for p in sorted((R5/'src').glob('*.py'))}})
    print('DIAGNOSTIC AUDIT PASSED',flush=True)

if __name__=='__main__':main()

