"""Frozen-size comparison on expanded evidence QA; no backbone fitting or test tuning."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'round10_dual_granularity'
NEW = ROOT.parent/'round16_dataset_expansion'
spec = importlib.util.spec_from_file_location('r17_reused_r13', ROOT.parent/'round13_generalization_diagnostics/src/run13.py')
r13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r13)
r10, np = r13.r10, r13.np
FEATURES = ('lb', 'nll', 'lb_nll')
METHODS = tuple(c+'_'+f for c in ('legacy','expanded') for f in FEATURES)
save, savel, sha = r10.save, r10.savel, r10.sha


def read(p): return json.loads(p.read_text('utf-8'))
def readl(p): return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s.strip()]


def verify_frozen_sources():
    frozen = read(NEW/'data/annotation_freeze.json')
    assert frozen['answers']==800 and not frozen['human_gold']
    for name, digest in frozen['files_sha256'].items(): assert sha(NEW/name)==digest, name
    assert sha(NEW/'data/input_freeze.json')==frozen['input_freeze_sha256']
    for name,digest in read(NEW/'data/input_freeze.json')['files_sha256'].items(): assert sha(NEW/name)==digest,name
    for name,digest in read(NEW/'data/legacy_files_snapshot.json')['files_sha256'].items(): assert sha(ROOT.parent/name)==digest,name
    assert sha(NEW/'data/generation_manifest.json')==frozen['generation_manifest_sha256']
    manifest=read(ROOT/'data/feature_manifest.json'); assert manifest['complete'] and manifest['completed_count']==800
    assert set(manifest['records'])=={r['row_id'] for r in readl(NEW/'data/inputs.jsonl')}
    for path,digest in manifest['extraction_signature']['code_sha256'].items():assert sha(Path(path))==digest,path
    return {'annotation_freeze_sha256':sha(NEW/'data/annotation_freeze.json'),
            'input_freeze_sha256':sha(NEW/'data/input_freeze.json'),
            'generation_manifest_sha256':sha(NEW/'data/generation_manifest.json'),
            'feature_manifest_sha256':sha(ROOT/'data/feature_manifest.json'),
            'legacy_snapshot_sha256':sha(NEW/'data/legacy_files_snapshot.json')}


def snapshot():
    return {'sources':verify_frozen_sources(),
            'code':{str(p.resolve()):sha(p) for p in [ROOT/'PLAN.md',ROOT/'protocol.json',Path(__file__),
                    ROOT.parent/'round13_generalization_diagnostics/src/run13.py',
                    ROOT.parent/'round12_conditional_fusion/src/run12.py',
                    ROOT.parent/'round11_logprob/src/run11.py',
                    OLD/'src/evaluate10.py',ROOT.parent/'round8_token_localization/src/evaluate8.py']}}


def metadata(root):
    """Multiple questions may share one event group; pairs are checked by question."""
    rows=readl(root/'data/inputs.jsonl'); records={};items=[];groups={};pairs=defaultdict(list)
    for row in rows:
        rid=row['row_id'];assert rid not in records
        assert groups.setdefault(row['group_id'],row['split'])==row['split']
        pairs[row['question_id']].append(row['condition'])
        p=root/'data/generation_records'/(rid+'.json');g=read(p)
        for k in ('row_id','question_id','split','condition'): assert g[k]==row[k]
        assert len(g['items'])==1
        item=dict(g['items'][0],**{k:row[k] for k in ('row_id','question_id','group_id','split','condition','category')})
        items.append(item); records[rid]=(g,sha(p))
    assert all(sorted(p)==['complete','partial'] for p in pairs.values())
    return rows,items,records


class Bank:
    def __init__(self, old_records, new_records):
        self.records={**old_records,**new_records};self.old_ids=set(old_records);self.cache={};self.files={}
        self.manifest=read(ROOT/'data/feature_manifest.json')

    def row(self,rid):
        if rid in self.cache:return self.cache[rid]
        g,digest=self.records[rid];directory=OLD if rid in self.old_ids else ROOT
        p=directory/'data/features'/(rid+'.npz');side=p.with_suffix('.json');meta=read(side)
        assert meta['source_generation_sha256']==digest
        assert meta['arrays_sha256']==sha(p)
        if rid not in self.old_ids:
            declared=self.manifest['records'][rid]
            assert p.resolve()==(ROOT/declared['npz']).resolve() and side.resolve()==(ROOT/declared['json']).resolve()
            assert sha(p)==declared['npz_sha256'] and sha(side)==declared['json_sha256']
            assert meta['extraction_signature_sha256']==self.manifest['extraction_signature_sha256']
        with np.load(p,allow_pickle=False) as arr:
            assert arr['token_ids'].tolist()==g['response_token_ids']
            offsets=arr['response_token_offsets'] if 'response_token_offsets' in arr else np.column_stack((arr['token_start'],arr['token_end']))
            assert offsets.tolist()==g['response_token_offsets']
            lb=arr['lb'].copy() if 'lb' in arr else arr['lookback_features'].copy()
            nll=arr['nll'].copy() if 'nll' in arr else arr['token_nll'].copy()
        n=len(g['response_token_ids'])
        assert lb.shape==(n,784) and nll.shape==(n,) and lb.dtype==nll.dtype==np.float32
        assert np.isfinite(lb).all() and np.isfinite(nll).all() and (nll>=0).all()
        self.cache[rid]=np.concatenate((lb,nll[:,None]),axis=1)
        for f in (p,side):self.files[str(f.resolve())]=sha(f)
        return self.cache[rid]


def cohort(root,split,meta,bank):
    items,tokens,regions,coverage=r10.cohort(root,split,meta)
    records=meta[2];by_token={t['token_key']:t for t in tokens};windows=[];matrix=[]
    for item in items:
        g=records[item['row_id']][0];offsets=np.asarray(g['response_token_offsets'])
        # Scope comes from the frozen output parser, never risk annotations.
        left,right=(item['start'],item['end']) if item['parse_ok'] else (0,len(g['response']))
        indices=np.flatnonzero((offsets[:,1]>left)&(offsets[:,0]<right))
        if not len(indices):continue
        assert np.all(np.diff(indices)==1)
        features=bank.row(item['row_id'])
        resolved=item['asserted_eligible'] and item['localization_status']=='resolved'
        for start in range(max(1,len(indices)-3)):
            raw=indices[start:start+4]
            tt=[by_token[item['row_id']+f'__token{int(j)}'] for j in raw]
            lexical=[t for t in tt if t['lexical']]
            if not lexical:continue
            if resolved:assert all(t['main_eligible'] for t in lexical)
            a,b=max(left,int(offsets[raw[0],0])),min(right,int(offsets[raw[-1],1]))
            windows.append({'window_key':item['item_id']+f'__w4__start{int(raw[0])}',
                **{k:item[k] for k in ('row_id','question_id','group_id','split','condition','category')},
                'item_ids':[item['item_id']],'width':4,'actual_width':len(raw),'short_window':len(raw)<4,
                'raw_token_indices':raw.tolist(),'token_keys':[t['token_key'] for t in lexical],
                'start':a,'end':b,'text':g['response'][a:b],
                'main_eligible':resolved,'gold':int(any(t['gold']==1 for t in lexical)) if resolved else None})
            matrix.append(features[raw].mean(axis=0))
    assert len({w['window_key'] for w in windows})==len(windows)
    if root==NEW:
        gold={w['window_key']:w for w in readl(NEW/'data'/f'windows_k4_{split}.jsonl')}
        eligible=[w for w in windows if w['main_eligible']]
        assert set(gold)=={w['window_key'] for w in eligible}
        for w in eligible:
            assert all(w[k]==gold[w['window_key']][k] for k in ('gold','start','end','raw_token_indices','token_keys','text'))
    coverage.update(questions=len({i['question_id'] for i in items}),
                    candidate_windows=len(windows),eligible_windows=sum(w['main_eligible'] for w in windows),
                    risk_windows=sum(w['gold']==1 for w in windows))
    return {'items':items,'tokens':tokens,'regions':regions,'windows':windows,
            'matrix':np.asarray(matrix,np.float32),'coverage':coverage}


def combine(a,b):
    result={k:a[k]+b[k] for k in ('items','tokens','regions','windows')}
    result['matrix']=np.concatenate((a['matrix'],b['matrix']))
    result['coverage']={k:a['coverage'][k]+b['coverage'][k] for k in ('questions','planned_answers','groups','eligible_tokens','eligible_windows','risk_windows')}
    return result


def columns(feature):
    return np.arange(784) if feature=='lb' else np.asarray([784]) if feature=='nll' else np.arange(785)


def answer_scores(pack,values):
    by_item=defaultdict(list)
    for w,v in zip(pack['windows'],values):by_item[w['item_ids'][0]].append(float(v))
    out=np.asarray([max(by_item[i['item_id']]) if by_item[i['item_id']] else np.nan for i in pack['items']])
    # Missing scores are a delivery failure, never silently deleted from a known-label denominator.
    assert all(np.isfinite(v) for i,v in zip(pack['items'],out) if i['main_eligible'])
    return out


def count(rows,values,threshold,unit):
    ix=[j for j,r in enumerate(rows) if r['main_eligible']]
    out=r13.count([rows[j]['gold'] for j in ix],np.asarray(values)[ix],threshold)
    out[unit]=out.pop('tokens');out['risk_'+unit]=out.pop('risk_tokens')
    return out


def metrics(pack,values,thresholds):
    av=answer_scores(pack,values)
    win=count(pack['windows'],values,thresholds['window']['threshold'],'windows')
    answer=count(pack['items'],av,thresholds['answer']['threshold'],'answers')
    selected=[w for w,v in zip(pack['windows'],values) if v>=thresholds['window']['threshold']]
    covered={k for w in selected for k in w['token_keys']}
    native=[t for t in pack['tokens'] if t['main_eligible']]
    yp=[k['token_key'] in covered for k in native]
    high=r13.count([t['gold'] for t in native],np.asarray(yp,float),predictions=yp)
    # Binary union has no continuous risk ranking interpretation.
    high.pop('auroc');high.pop('average_precision')
    high['marked_token_fraction']=sum(yp)/len(yp) if yp else None
    high['risk_regions']=len(pack['regions'])
    high['risk_regions_any_hit']=sum(bool(set(r['token_keys'])&covered) for r in pack['regions'])
    high['risk_regions_fully_hit']=sum(bool(r['token_keys']) and set(r['token_keys'])<=covered for r in pack['regions'])
    return {'windows':win,'answers':answer,'highlight_tokens':high}


def predictions(pack,models):
    wr=[dict(w,scores={},predictions={}) for w in pack['windows']]
    ar=[{k:i[k] for k in ('item_id','row_id','question_id','group_id','split','condition','category','text','start','end','gold','main_eligible','reviewed_safe_refusal')} for i in pack['items']]
    for a in ar:a.update(scores={},predictions={})
    summary={}
    for name,model in models.items():
        value=r13.probability(model,pack['matrix'][:,model['columns']]);av=answer_scores(pack,value)
        ts=model['thresholds'];summary[name]=metrics(pack,value,ts)
        for w,v in zip(wr,value):w['scores'][name]=float(v);w['predictions'][name]=bool(v>=ts['window']['threshold'])
        for a,v in zip(ar,av):a['scores'][name]=float(v) if np.isfinite(v) else None;a['predictions'][name]=bool(v>=ts['answer']['threshold']) if np.isfinite(v) else None
    return wr,ar,summary


def fit():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'fit_freeze17.json').exists(),'Completed fit is frozen'
    config=read(ROOT/'protocol.json');assert config['C']==.01 and config['target_loss_mass']==3854 and config['width']==4
    snap=snapshot();save(out/'source_snapshot17.json',snap)
    save(out/'fit_started17.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot17.json'),'test_labels_used':False})
    om,nm=metadata(OLD),metadata(NEW);bank=Bank(om[2],nm[2])
    legacy=cohort(OLD,'train',om,bank);new=cohort(NEW,'train',nm,bank);validation=cohort(NEW,'validation',nm,bank)
    expanded=combine(legacy,new)
    assert legacy['coverage']['questions']==120 and expanded['coverage']['questions']==421
    assert legacy['coverage']['eligible_tokens']==config['target_loss_mass']
    assert not {i['group_id'] for i in expanded['items']} & {i['group_id'] for i in validation['items']}
    models={};fit_metrics={};start=time.perf_counter()
    wi=np.flatnonzero([w['main_eligible'] for w in validation['windows']])
    ai=np.flatnonzero([i['main_eligible'] for i in validation['items']])
    for cohort_name,pack in [('legacy',legacy),('expanded',expanded)]:
        ix=np.flatnonzero([w['main_eligible'] for w in pack['windows']]);rows=[pack['windows'][j] for j in ix]
        for feature in FEATURES:
            name=cohort_name+'_'+feature;cols=columns(feature)
            model=r13.fit_model(pack['matrix'][ix][:,cols],rows,'lb',config['C'],config['target_loss_mass'],{})
            vv=r13.probability(model,validation['matrix'][:,cols]);av=answer_scores(validation,vv)
            thresholds={'window':r10.threshold_search([validation['windows'][j]['gold'] for j in wi],vv[wi]),
                        'answer':r10.threshold_search([validation['items'][j]['gold'] for j in ai],av[ai])}
            model.update(columns=cols,thresholds=thresholds,signal=feature,cohort=cohort_name,
                         fit_window_keys=[r['window_key'] for r in rows],
                         fit_groups=sorted({r['group_id'] for r in rows}))
            models[name]=model
            fit_metrics[name]=metrics(pack,r13.probability(model,pack['matrix'][:,cols]),thresholds)
            print('FIT',name,'windows',len(ix),'thresholds',json.dumps(thresholds),flush=True)
    vw,va,vm=predictions(validation,models)
    # Training common-set diagnostics are explicitly training-internal.
    _,_,legacy_common=predictions(legacy,models)
    assert snapshot()==snap
    for p,digest in bank.files.items():assert sha(Path(p))==digest
    (out/'frozen_models.pkl').write_bytes(pickle.dumps(models,protocol=5))
    save(out/'feature_files_fit.json',bank.files)
    save(out/'fit_metrics.json',{'own_training':fit_metrics,'common_legacy_training':legacy_common,
          'coverage':{'legacy_train':legacy['coverage'],'new_train':new['coverage'],'expanded_train':expanded['coverage']}})
    save(out/'validation_metrics.json',{'coverage':validation['coverage'],'methods':vm})
    savel(out/'window_scores_validation.jsonl',vw);savel(out/'answer_scores_validation.jsonl',va)
    files=['source_snapshot17.json','frozen_models.pkl','feature_files_fit.json','fit_metrics.json','validation_metrics.json','window_scores_validation.jsonl','answer_scores_validation.jsonl']
    save(out/'fit_freeze17.json',{'utc':r10.utc(),'methods':list(METHODS),'primary':'expanded_lb','test_labels_used':False,
         'fit_and_validation_seconds':time.perf_counter()-start,'files_sha256':{p:sha(out/p) for p in files},
         'thresholds':{k:m['thresholds'] for k,m in models.items()}})
    print('ROUND17_FIT_FROZEN',flush=True)


def bootstrap(rows,config):
    groups=sorted({r['group_id'] for r in rows});lookup={g:j for j,g in enumerate(groups)}
    c=np.zeros((len(groups),len(METHODS),3),np.int64)
    for row in rows:
        if not row['main_eligible']:continue
        for j,name in enumerate(METHODS):
            p,y=row['predictions'][name],row['gold'];assert p is not None
            c[lookup[row['group_id']],j]+=[int(p and y),int(p and not y),int(not p and y)]
    sample=np.random.default_rng(config['seed']).integers(0,len(groups),(config['draws'],len(groups)))
    weights=np.stack([np.bincount(s,minlength=len(groups)) for s in sample])
    total=np.einsum('bg,gmc->bmc',weights,c,optimize=True);tp,fp,fn=(total[:,:,j] for j in range(3))
    denom=2*tp+fp+fn;f1=np.divide(2*tp,denom,out=np.full(tp.shape,np.nan),where=denom>0)
    return {'groups':len(groups),**config,'fixed_models_and_thresholds':True,
       'f1':{name:r10.ci(f1[:,j]) for j,name in enumerate(METHODS)},
       'expanded_minus_legacy':{f:r10.ci(f1[:,METHODS.index('expanded_'+f)]-f1[:,METHODS.index('legacy_'+f)]) for f in FEATURES}}


def test():
    out=ROOT/'results';assert not (out/'test_complete17.json').exists(),'Final test is immutable; no retuning'
    frozen=read(out/'fit_freeze17.json');snap=read(out/'source_snapshot17.json')
    assert snapshot()==snap
    for p,digest in frozen['files_sha256'].items():assert sha(out/p)==digest,p
    for p,digest in read(out/'feature_files_fit.json').items():assert sha(Path(p))==digest,p
    save(out/'test_started17.json',{'utc':r10.utc(),'fit_freeze_sha256':sha(out/'fit_freeze17.json'),'retuning_allowed':False})
    om,nm=metadata(OLD),metadata(NEW);bank=Bank(om[2],nm[2]);pack=cohort(NEW,'test',nm,bank)
    assert pack['coverage']['questions']==50 and pack['coverage']['groups']==49
    models=pickle.loads((out/'frozen_models.pkl').read_bytes())
    assert list(models)==list(METHODS)
    for model in models.values():assert not set(model['fit_groups']) & {i['group_id'] for i in pack['items']}
    wr,ar,summary=predictions(pack,models)
    conf=read(ROOT/'protocol.json')['bootstrap']
    result={'coverage':pack['coverage'],'primary_method':'expanded_lb','methods':summary,
            'paired_bootstrap':{'windows':bootstrap(wr,conf),'answers':bootstrap(ar,conf)},
            'unit_note':'answer=one generated answer under one evidence condition; pair retained in group bootstrap',
            'validation_only_thresholds':frozen['thresholds'],'human_gold':False}
    result['categories']={}
    for category in sorted({r['category'] for r in ar}):
        cw=[r for r in wr if r['category']==category];ca=[r for r in ar if r['category']==category]
        result['categories'][category]={name:{
            'windows':count(cw,[r['scores'][name] for r in cw],models[name]['thresholds']['window']['threshold'],'windows'),
            'answers':count(ca,[r['scores'][name] for r in ca],models[name]['thresholds']['answer']['threshold'],'answers')}
            for name in METHODS}
    assert snapshot()==snap
    savel(out/'window_scores_test.jsonl',wr);savel(out/'answer_scores_test.jsonl',ar)
    save(out/'feature_files_test.json',bank.files);save(out/'metrics_test.json',result)
    files=['window_scores_test.jsonl','answer_scores_test.jsonl','feature_files_test.json','metrics_test.json']
    save(out/'test_complete17.json',{'utc':r10.utc(),'fit_freeze_sha256':sha(out/'fit_freeze17.json'),
          'files_sha256':{p:sha(out/p) for p in files},'retuning_allowed':False})
    print('ROUND17_TEST_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','test']);args=p.parse_args()
    with r10.threadpool_limits(limits=4):{'fit':fit,'test':test}[args.stage]()
