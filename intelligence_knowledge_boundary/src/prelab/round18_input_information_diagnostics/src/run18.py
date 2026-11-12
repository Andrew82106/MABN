"""Bounded input/mapping diagnostics on R16 training groups only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT.parent/'round16_dataset_expansion'
R17=ROOT.parent/'round17_expanded_retraining'
spec=importlib.util.spec_from_file_location('r18_reused_r17',R17/'src/run17.py')
r17=importlib.util.module_from_spec(spec);spec.loader.exec_module(r17)
r10,r13,np=r17.r10,r17.r13,r17.np
read,readl,save,savel,sha=r17.read,r17.readl,r17.save,r17.savel,r17.sha
sys.path.insert(0,str(ROOT/'src'))
METHODS=('mean_lr','slots_lr','shuffled_slots_lr','mean_mlp','mean_surface','mean_alignment','mean_both','mean_hidden32')


def source_snapshot():
    af=read(NEW/'data/annotation_freeze.json')
    assert sha(NEW/'data/input_freeze.json')==af['input_freeze_sha256']
    # Frozen trees are hashed, not parsed as validation or test labels.
    for name,digest in af['files_sha256'].items():assert sha(NEW/name)==digest,name
    for name,digest in read(NEW/'data/input_freeze.json')['files_sha256'].items():assert sha(NEW/name)==digest,name
    for name,digest in read(NEW/'data/legacy_files_snapshot.json')['files_sha256'].items():assert sha(ROOT.parent/name)==digest,name
    fm=read(ROOT/'data/feature_manifest.json')
    assert fm['complete'] and fm['completed_count']==602
    ids={r['row_id'] for r in readl(NEW/'data/inputs.jsonl') if r['split']=='train'}
    assert set(fm['records'])==ids
    for path,digest in fm['extraction_signature']['code_sha256'].items():assert sha(Path(path))==digest,path
    files=[ROOT/'PLAN.md',ROOT/'protocol.json',ROOT/'data/fold_assignment.json',Path(__file__),ROOT/'src/models18.py',
           R17/'src/run17.py',ROOT.parent/'round13_generalization_diagnostics/src/run13.py',
           ROOT.parent/'round12_conditional_fusion/src/run12.py',ROOT.parent/'round11_logprob/src/run11.py',
           ROOT.parent/'round10_dual_granularity/src/evaluate10.py',ROOT.parent/'round8_token_localization/src/evaluate8.py']
    return {'annotation_freeze_sha256':sha(NEW/'data/annotation_freeze.json'),
            'feature_manifest_sha256':sha(ROOT/'data/feature_manifest.json'),
            'R17_fit_freeze_sha256':sha(R17/'results/fit_freeze17.json'),
            'R17_test_complete_sha256':sha(R17/'results/test_complete17.json'),
            'code_sha256':{str(p.resolve()):sha(p) for p in files}}


def train_metadata():
    rows=[r for r in readl(NEW/'data/inputs.jsonl') if r['split']=='train']
    assert len(rows)==602 and len({r['question_id'] for r in rows})==301
    items=[];records={};pairs=defaultdict(list)
    source=read(NEW/'data/generation_manifest.json')
    for row in rows:
        rid=row['row_id'];p=NEW/'data/generation_records'/(rid+'.json');g=read(p)
        assert sha(p)==source['record_sha256'][rid]
        assert g['split']=='train' and len(g['items'])==1
        item=dict(g['items'][0],**{k:row[k] for k in ('row_id','question_id','group_id','split','condition','category')})
        items.append(item);records[rid]=(g,sha(p));pairs[row['question_id']].append(row['condition'])
    assert all(sorted(p)==['complete','partial'] for p in pairs.values())
    return rows,items,records


class Bank:
    def __init__(self,records):
        self.records=records;self.manifest=read(ROOT/'data/feature_manifest.json');self.cache={};self.files={}
    def arrays(self,rid):
        if rid in self.cache:return self.cache[rid]
        g,checksum=self.records[rid];entry=self.manifest['records'][rid]
        p=ROOT/entry['npz'];side=ROOT/entry['json'];meta=read(side)
        assert sha(p)==entry['npz_sha256']==meta['arrays_sha256'] and sha(side)==entry['json_sha256']
        assert meta['source_generation_sha256']==checksum
        assert meta['extraction_signature_sha256']==self.manifest['extraction_signature_sha256']
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert a['token_ids'].tolist()==g['response_token_ids']
        assert a['response_token_offsets'].tolist()==g['response_token_offsets']
        n=len(g['response_token_ids'])
        for key,width in [('lb',784),('new_features',16),('surface_features',8),('hidden_28',3584)]:
            assert a[key].shape==(n,width) and a[key].dtype==np.float32 and np.isfinite(a[key]).all()
        assert a['nll'].shape==(n,) and a['nll'].dtype==np.float32 and np.isfinite(a['nll']).all()
        self.cache[rid]=a
        for f in (p,side):self.files[str(f.resolve())]=sha(f)
        return a
    def row(self,rid):
        a=self.arrays(rid);return np.concatenate((a['lb'],a['nll'][:,None]),axis=1)


def prepare():
    meta=train_metadata();bank=Bank(meta[2]);pack=r17.cohort(NEW,'train',meta,bank)
    assert pack['coverage']['groups']==278 and pack['coverage']['eligible_windows']==9526
    designs={'base':pack['matrix'],'surface':[],'alignment':[],'hidden':[],'slots':[],'shuffled_slots':[]}
    for w in pack['windows']:
        a=bank.arrays(w['row_id']);raw=w['raw_token_indices'];n=len(raw)
        for out,key in [('surface','surface_features'),('alignment','new_features'),('hidden','hidden_28')]:
            designs[out].append(a[key][raw].mean(axis=0))
        tokens=np.concatenate((a['lb'][raw],a['nll'][raw,None]),axis=1)
        if n<4:tokens=np.concatenate((tokens,np.repeat(tokens[-1:],4-n,axis=0)))
        mask=np.asarray([1]*n+[0]*(4-n),np.float32)
        designs['slots'].append(np.concatenate((tokens.reshape(-1),mask)))
        seed=int.from_bytes(hashlib.sha256(w['window_key'].encode()).digest()[:8],'little')
        perm=np.random.default_rng(seed).permutation(4)
        designs['shuffled_slots'].append(np.concatenate((tokens[perm].reshape(-1),mask[perm])))
    for key in designs:designs[key]=np.asarray(designs[key],np.float32);assert np.isfinite(designs[key]).all()
    assert all(x.shape[0]==len(pack['windows']) for x in designs.values())
    assert designs['slots'].shape[1]==designs['shuffled_slots'].shape[1]==3144
    # Four-slot means contain the original signal for all full windows.
    full=np.asarray([w['actual_width']==4 for w in pack['windows']])
    assert np.array_equal(designs['slots'][full,:3140].reshape(-1,4,785).mean(axis=1),designs['base'][full])
    return pack,designs,bank.files


def subset(pack,groups):
    groups=set(groups)
    indices=np.flatnonzero([w['group_id'] in groups for w in pack['windows']])
    result={k:[r for r in pack[k] if r['group_id'] in groups] for k in ('items','tokens','windows')}
    selected_rows={r['row_id'] for r in result['items']}
    result['regions']=[r for r in pack['regions'] if r['row_id'] in selected_rows]
    return result,indices


def scored_records(pack,values,thresholds,name):
    av=r17.answer_scores(pack,values)
    windows=[dict(w,scores={name:float(s)},predictions={name:bool(s>=thresholds['window']['threshold'])}) for w,s in zip(pack['windows'],values)]
    answers=[]
    for a,s in zip(pack['items'],av):
        row={k:a[k] for k in ('item_id','row_id','question_id','group_id','condition','category','text','gold','main_eligible','reviewed_safe_refusal')}
        row.update(scores={name:float(s) if np.isfinite(s) else None},predictions={name:bool(s>=thresholds['answer']['threshold']) if np.isfinite(s) else None})
        answers.append(row)
    return windows,answers


def pooled(windows,answers,pack):
    result={};native=[t for t in pack['tokens'] if t['main_eligible']]
    by_answer=defaultdict(list)
    for w in windows:
        if w['main_eligible']:by_answer[w['item_ids'][0]].append(w)
    for name in METHODS:
        wr=[w for w in windows if w['main_eligible']];ar=[a for a in answers if a['main_eligible']]
        def discrete(rows,unit):
            m=r13.count([r['gold'] for r in rows],[float(r['predictions'][name]) for r in rows],predictions=[r['predictions'][name] for r in rows])
            m[unit]=m.pop('tokens');m['risk_'+unit]=m.pop('risk_tokens');m.pop('auroc');m.pop('average_precision');return m
        covered={k for w in windows if w['predictions'][name] for k in w['token_keys']}
        pp=[t['token_key'] in covered for t in native]
        hm=r13.count([t['gold'] for t in native],np.asarray(pp,float),predictions=pp)
        hm.pop('auroc');hm.pop('average_precision');hm['marked_token_fraction']=sum(pp)/len(pp)
        hm['risk_regions']=len(pack['regions']);hm['risk_regions_any_hit']=sum(bool(set(r['token_keys'])&covered) for r in pack['regions'])
        hm['risk_regions_fully_hit']=sum(bool(r['token_keys']) and set(r['token_keys'])<=covered for r in pack['regions'])
        ranking=[]
        for iid,rows in by_answer.items():
            y=np.asarray([r['gold'] for r in rows]);s=np.asarray([r['scores'][name] for r in rows])
            if len(set(y))!=2:continue
            tied=np.flatnonzero(s==s.max())
            ranking.append({'item_id':iid,'auroc':float(r13.roc_auc_score(y,s)),
                            'average_precision':float(r13.average_precision_score(y,s)),
                            'all_peak_windows_risky':bool((y[tied]==1).all())})
        safe=[a for a in ar if a['reviewed_safe_refusal']]
        result[name]={'windows':discrete(wr,'windows'),'answers':discrete(ar,'answers'),
            'highlight_tokens':hm,'conditional_ranking':{'answers_with_both_labels':len(ranking),
            'peak_hit_answers':sum(r['all_peak_windows_risky'] for r in ranking),
            'mean_auroc':float(np.mean([r['auroc'] for r in ranking])),
            'mean_average_precision':float(np.mean([r['average_precision'] for r in ranking])),'details':ranking},
            'safe_refusal_false_positives':sum(a['predictions'][name] for a in safe),'safe_refusals':len(safe)}
    return result


def bootstrap(rows,config,contrasts):
    groups=sorted({r['group_id'] for r in rows});idx={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),len(METHODS),3),np.int64)
    for row in rows:
        if not row['main_eligible']:continue
        y=row['gold']
        for j,name in enumerate(METHODS):
            p=row['predictions'][name];assert p is not None
            counts[idx[row['group_id']],j]+=[int(p and y),int(p and not y),int(not p and y)]
    sample=np.random.default_rng(config['seed']).integers(0,len(groups),(config['draws'],len(groups)))
    weights=np.stack([np.bincount(s,minlength=len(groups)) for s in sample])
    total=np.einsum('bg,gmc->bmc',weights,counts,optimize=True);tp,fp,fn=(total[:,:,k] for k in range(3))
    d=2*tp+fp+fn;f=np.divide(2*tp,d,out=np.full(tp.shape,np.nan),where=d>0)
    return {**config,'groups':len(groups),'f1':{n:r10.ci(f[:,j]) for j,n in enumerate(METHODS)},
            'contrasts':{k:{'methods':names,**r10.ci(f[:,METHODS.index(names[0])]-f[:,METHODS.index(names[1])])} for k,names in contrasts.items()},
            'limit':'Fixed fitted folds and thresholds, no refitting uncertainty or multiple-comparison correction.'}


def run():
    import models18
    out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'complete18.json').exists(),'Completed diagnostic immutable'
    config=read(ROOT/'protocol.json');assert tuple(config['methods'])==METHODS
    snap=source_snapshot();save(out/'source_snapshot18.json',snap)
    save(out/'started18.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot18.json'),'original_validation_or_test_used':False})
    pack,designs,feature_files=prepare();assignment=read(ROOT/'data/fold_assignment.json')['groups']
    assert set(assignment)=={i['group_id'] for i in pack['items']}
    savel(out/'candidate_windows.jsonl',pack['windows']);save(out/'feature_files.json',feature_files)
    gid=np.asarray([w['group_id'] for w in pack['windows']]);eligible=np.asarray([w['main_eligible'] for w in pack['windows']])
    all_windows=[];all_answers=[];fold_results={};fold_models={};assigned=defaultdict(int);started=time.perf_counter()
    for fold in range(5):
        fold_file=out/f'fold_{fold}_frozen.pkl';detail_file=out/f'fold_{fold}_metrics.json'
        assert not fold_file.exists(),'Do not silently replace a fitted fold'
        eg=sorted(g for g,f in assignment.items() if f==fold);cg=sorted(g for g,f in assignment.items() if f==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg));assert not(set(eg)&set(cg))
        tr,tx=subset(pack,fg);ca,cx=subset(pack,cg);ev,ex=subset(pack,eg)
        fit_ix=np.flatnonzero(np.isin(gid,fg)&eligible);fit_rows=[pack['windows'][i] for i in fit_ix]
        assert len(fg)+len(cg)+len(eg)==278
        local_models={};detail={};cw=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']]
        aa=None
        for method in METHODS:
            model=models18.fit_model(method,designs,fit_ix,fit_rows,config,fold)
            value=models18.predict(model,designs);assert value.shape==(len(pack['windows']),) and np.isfinite(value).all()
            cav=value[cx];av=r17.answer_scores(ca,cav)
            wi=np.flatnonzero([w['main_eligible'] for w in ca['windows']]);ai=np.flatnonzero([a['main_eligible'] for a in ca['items']])
            ts={'window':r10.threshold_search([ca['windows'][i]['gold'] for i in wi],cav[wi]),
                'answer':r10.threshold_search([ca['items'][i]['gold'] for i in ai],av[ai])}
            model['thresholds']=ts;model['fit_groups']=fg;model['calibration_groups']=cg;model['evaluation_groups']=eg
            local_models[method]=model
            detail[method]={stage:r17.metrics(p,value[ix],ts) for stage,p,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]}
            detail[method]['thresholds']=ts
            ww,aaa=scored_records(ev,value[ex],ts,method)
            for target,row in zip(cw,ww):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            if aa is None:aa=[dict(a,scores={},predictions={},fold=fold) for a in aaa]
            for target,row in zip(aa,aaa):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            members=models18.predict_members(model,designs)
            if members is not None:
                seed_summary=[]
                for j,mv in enumerate(members):
                    mav=r17.answer_scores(ca,mv[cx]);mts={'window':r10.threshold_search([ca['windows'][i]['gold'] for i in wi],mv[cx][wi]),
                     'answer':r10.threshold_search([ca['items'][i]['gold'] for i in ai],mav[ai])}
                    seed_summary.append({'member_index':j,'thresholds':mts,'evaluation':r17.metrics(ev,mv[ex],mts),
                                         'fit':r17.metrics(tr,mv[tx],mts)})
                detail[method]['individual_seeds_descriptive_not_selected']=seed_summary
            print('FOLD',fold+1,'METHOD',method,'FIT_F1',detail[method]['fit']['windows']['f1'],
                  'HELD_F1',detail[method]['evaluation']['windows']['f1'],flush=True)
        fold_models[fold]={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg,'models':local_models}
        fold_file.write_bytes(pickle.dumps(fold_models[fold],protocol=5));save(detail_file,detail)
        savel(out/f'fold_{fold}_window_scores.jsonl',cw);savel(out/f'fold_{fold}_answer_scores.jsonl',aa)
        fold_results[str(fold)]=detail;all_windows.extend(cw);all_answers.extend(aa)
        for g in eg:assigned[g]+=1
        assert source_snapshot()==snap
    assert set(assigned)==set(assignment) and all(v==1 for v in assigned.values())
    assert len(all_windows)==len(pack['windows']) and len(all_answers)==602
    result={'scope':config['scope'],'original_validation_or_test_used':False,'coverage':pack['coverage'],
            'folds':fold_results,'methods':pooled(all_windows,all_answers,pack),
            'paired_bootstrap':{'windows':bootstrap(all_windows,config['bootstrap'],config['primary_contrasts']),
                                'answers':bootstrap(all_answers,config['bootstrap'],config['primary_contrasts'])},
            'fold_mean':{m:{stage:{metric:float(np.mean([fold_results[str(f)][m][stage]['windows'][metric] for f in range(5)]))
              for metric in ('f1','auroc','average_precision')} for stage in ('fit','calibration','evaluation')} for m in METHODS},
            'seconds':time.perf_counter()-started,'limits':config['limits']}
    assert source_snapshot()==snap
    savel(out/'window_scores_oof.jsonl',all_windows);savel(out/'answer_scores_oof.jsonl',all_answers);save(out/'summary.json',result)
    files=['source_snapshot18.json','feature_files.json','candidate_windows.jsonl','window_scores_oof.jsonl','answer_scores_oof.jsonl','summary.json']
    files += [f'fold_{f}_{suffix}' for f in range(5) for suffix in ('frozen.pkl','metrics.json','window_scores.jsonl','answer_scores.jsonl')]
    save(out/'complete18.json',{'utc':r10.utc(),'original_validation_or_test_used':False,'methods':list(METHODS),
         'files_sha256':{name:sha(out/name) for name in files}})
    print('ROUND18_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run']);args=p.parse_args()
    with r10.threadpool_limits(limits=4):run()
