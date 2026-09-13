"""Fixed six-way three-signal comparison on exactly the R18 training-source folds."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
R18=ROOT.parent/'round18_input_information_diagnostics'
spec=importlib.util.spec_from_file_location('r19_reuses_r18',R18/'src/run18.py')
r18=importlib.util.module_from_spec(spec);spec.loader.exec_module(r18)
r17,r10,r13,np=r18.r17,r18.r10,r18.r13,r18.np
read,save,savel,sha=r18.read,r18.save,r18.savel,r18.sha
_original_readl=r18.readl
_split_field=re.compile(r'(?<!\\)"split"\s*:\s*"([^"\\]+)"')


def readl(path):
    """Prefilter the single mixed input file before parsing its held-out bodies."""
    path=Path(path)
    if path.resolve() != (r18.NEW/'data/inputs.jsonl').resolve():return _original_readl(path)
    rows=[]
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            matches=list(_split_field.finditer(line));assert len(matches)==1
            if matches[0].group(1)=='train':rows.append(json.loads(line))
    assert len(rows)==602 and all(row['split']=='train' for row in rows)
    return rows


# R18 metadata/snapshot helpers resolve their readl through this private module.
# Other JSONL readers and all legacy source files are unchanged.
r18.readl=readl
METHODS=('base','base_mmd','base_ecs','base_pks','base_all','signals_only')
# In-process registry for unchanged metric helpers; no R18 file is changed.
r18.METHODS=METHODS
sys.path.insert(0,str(ROOT/'src'))


def snapshot():
    previous=read(R18/'results/complete18.json')
    for name,digest in previous['files_sha256'].items():assert sha(R18/'results'/name)==digest,name
    fm=read(ROOT/'data/feature_manifest.json')
    assert fm['complete'] and fm['completed_count']==602
    ids={row['row_id'] for row in readl(r18.NEW/'data/inputs.jsonl') if row['split']=='train'}
    assert set(fm['records'])==ids
    for path,digest in fm['extraction_signature']['code_sha256'].items():assert sha(Path(path))==digest,path
    assert sha(ROOT/'data/fold_assignment.json')==sha(R18/'data/fold_assignment.json')
    files=[ROOT/'PLAN.md',ROOT/'protocol.json',ROOT/'data/fold_assignment.json',Path(__file__),ROOT/'src/models19.py']
    return {'r18_source_snapshot':r18.source_snapshot(),'r18_complete_sha256':sha(R18/'results/complete18.json'),
            'feature_manifest_sha256':sha(ROOT/'data/feature_manifest.json'),
            'code_sha256':{str(p.resolve()):sha(p) for p in files}}


class SignalBank:
    def __init__(self,records):
        self.records=records;self.manifest=read(ROOT/'data/feature_manifest.json');self.cache={};self.files={}
    def row(self,rid):
        if rid in self.cache:return self.cache[rid]
        g,digest=self.records[rid];entry=self.manifest['records'][rid]
        p=ROOT/entry['npz'];side=ROOT/entry['json'];meta=read(side)
        assert sha(p)==entry['npz_sha256']==meta['arrays_sha256']
        assert sha(side)==entry['json_sha256'] and meta['source_generation_sha256']==digest
        assert meta['extraction_signature_sha256']==self.manifest['extraction_signature_sha256']
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert a['token_ids'].tolist()==g['response_token_ids']
        assert a['response_token_offsets'].tolist()==g['response_token_offsets']
        n=len(g['response_token_ids'])
        for key,width in [('token_lumina_mmd',2),('token_redeep_ecs',784),('token_redeep_pks',28)]:
            assert a[key].shape==(n,width) and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),(rid,key)
        # First source aggregation for each token, then the caller averages BPE windows.
        mmd=np.column_stack((a['token_lumina_mmd'].mean(axis=1),a['token_lumina_mmd'].max(axis=1)))
        ecs=a['token_redeep_ecs'].reshape(n,4,7,28).mean(axis=(2,3))
        pks=a['token_redeep_pks'].reshape(n,4,7).mean(axis=2)
        value={'mmd':mmd,'ecs':ecs,'pks':pks}
        for f in (p,side):self.files[str(f.resolve())]=sha(f)
        self.cache[rid]=value
        return value


def prepare():
    meta=r18.train_metadata();basebank=r18.Bank(meta[2]);signals=SignalBank(meta[2])
    pack=r17.cohort(r18.NEW,'train',meta,basebank)
    assert pack['coverage']['groups']==278 and pack['coverage']['eligible_windows']==9526
    assert pack['windows']==readl(R18/'results/candidate_windows.jsonl')
    designs={'base':pack['matrix'],'mmd':[],'ecs':[],'pks':[]}
    for window in pack['windows']:
        a=signals.row(window['row_id']);ix=window['raw_token_indices']
        for key in ('mmd','ecs','pks'):designs[key].append(a[key][ix].mean(axis=0))
    for key in designs:
        designs[key]=np.asarray(designs[key],np.float32)
        assert len(designs[key])==len(pack['windows']) and np.isfinite(designs[key]).all()
    lengths={rid:len(g['response_token_ids']) for rid,(g,_) in meta[2].items()}
    return pack,designs,{'r18':basebank.files,'r19':signals.files},lengths


def baseline_check(model,values,thresholds,designs,fold):
    old=pickle.loads((R18/'results'/f'fold_{fold}_frozen.pkl').read_bytes())['models']['mean_lr']
    for key in ('fit_ix','fit_y','base_weights','loss_weights','class_factors'):
        assert np.array_equal(model[key],old[key]),('baseline',fold,key)
    assert model['fit_window_keys']==old['fit_window_keys']
    for key in ('mean_','scale_','var_','n_samples_seen_'):
        assert np.array_equal(getattr(model['scaler'],key),getattr(old['scaler'],key)),('scaler',fold,key)
    for key in ('coef_','intercept_','classes_','n_iter_'):
        assert np.array_equal(getattr(model['model'],key),getattr(old['model'],key)),('LR',fold,key)
    old_values=old['model'].predict_proba(old['scaler'].transform(designs['base']).astype(np.float32))[:,1]
    assert np.array_equal(values,old_values) and thresholds==old['thresholds']
    return {'passed':True,'all_candidate_scores_max_absolute_difference':float(np.max(np.abs(values-old_values))),
            'coefficients_scaler_weights_indices_thresholds_exact':True,
            'r18_fold_sha256':sha(R18/'results'/f'fold_{fold}_frozen.pkl')}


def strata(windows,answers,lengths):
    def length_bin(row):
        n=lengths[row['row_id']];return '<=16' if n<=16 else '17..32' if n<=32 else '>32'
    output={}
    for field,fn in [('category',lambda row:row['category']),('condition',lambda row:row['condition']),('answer_length',length_bin)]:
        groups={fn(row) for row in answers};output[field]={}
        for group in sorted(groups):
            output[field][group]={}
            for name in METHODS:
                output[field][group][name]={}
                for unit,all_rows in [('windows',windows),('answers',answers)]:
                    rows=[row for row in all_rows if row['main_eligible'] and fn(row)==group]
                    score=[row['scores'][name] for row in rows];pred=[row['predictions'][name] for row in rows]
                    m=r13.count([row['gold'] for row in rows],score,predictions=pred)
                    m[unit]=m.pop('tokens');m['risk_'+unit]=m.pop('risk_tokens')
                    m.pop('auroc');m.pop('average_precision') # No mixed-fold continuous rankings.
                    output[field][group][name][unit]=m
    return output


def run():
    import models19
    out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'started19.json').exists(),'Do not silently repeat an already started formal fit'
    config=read(ROOT/'protocol.json');assert tuple(config['methods'])==METHODS
    snap=snapshot();pack,designs,feature_files,lengths=prepare()
    assignment=read(ROOT/'data/fold_assignment.json')['groups']
    assert set(assignment)=={item['group_id'] for item in pack['items']}
    save(out/'source_snapshot19.json',snap)
    save(out/'started19.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot19.json'),'original_validation_or_test_used':False})
    savel(out/'candidate_windows.jsonl',pack['windows']);save(out/'feature_files.json',feature_files)
    save(out/'response_lengths.json',lengths)
    gid=np.asarray([window['group_id'] for window in pack['windows']]);eligible=np.asarray([w['main_eligible'] for w in pack['windows']])
    all_windows=[];all_answers=[];fold_results={};baseline_checks={};assigned=defaultdict(int);started=time.perf_counter()
    for fold in range(5):
        eg=sorted(g for g,f in assignment.items() if f==fold);cg=sorted(g for g,f in assignment.items() if f==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg));assert not(set(eg)&set(cg))
        tr,tx=r18.subset(pack,fg);ca,cx=r18.subset(pack,cg);ev,ex=r18.subset(pack,eg)
        fit_ix=np.flatnonzero(np.isin(gid,fg)&eligible);fit_rows=[pack['windows'][i] for i in fit_ix]
        wi=np.flatnonzero([w['main_eligible'] for w in ca['windows']]);ai=np.flatnonzero([a['main_eligible'] for a in ca['items']])
        models={};detail={};cw=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']];aa=None
        for name in METHODS:
            model=models19.fit_model(name,designs,fit_ix,fit_rows,config,fold)
            values=models19.predict(model,designs)
            assert values.shape==(len(pack['windows']),) and np.isfinite(values).all()
            av=r17.answer_scores(ca,values[cx])
            ts={'window':r10.threshold_search([ca['windows'][i]['gold'] for i in wi],values[cx][wi]),
                'answer':r10.threshold_search([ca['items'][i]['gold'] for i in ai],av[ai])}
            if name=='base':baseline_checks[str(fold)]=baseline_check(model,values,ts,designs,fold)
            model.update(thresholds=ts,fit_groups=fg,calibration_groups=cg,evaluation_groups=eg)
            models[name]=model
            detail[name]={stage:r17.metrics(p,values[ix],ts) for stage,p,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]}
            detail[name]['thresholds']=ts
            ww,ar=r18.scored_records(ev,values[ex],ts,name)
            for target,row in zip(cw,ww):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            if aa is None:aa=[dict(row,scores={},predictions={},fold=fold) for row in ar]
            for target,row in zip(aa,ar):target['scores'].update(row['scores']);target['predictions'].update(row['predictions'])
            print('FOLD',fold+1,'METHOD',name,'FIT_F1',detail[name]['fit']['windows']['f1'],
                  'HELD_F1',detail[name]['evaluation']['windows']['f1'],flush=True)
        frozen={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg,'models':models}
        (out/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(frozen,protocol=5))
        save(out/f'fold_{fold}_metrics.json',detail);savel(out/f'fold_{fold}_window_scores.jsonl',cw);savel(out/f'fold_{fold}_answer_scores.jsonl',aa)
        fold_results[str(fold)]=detail;all_windows.extend(cw);all_answers.extend(aa)
        for g in eg:assigned[g]+=1
        assert snapshot()==snap
    assert set(assigned)==set(assignment) and all(value==1 for value in assigned.values())
    assert len(all_windows)==len(pack['windows']) and len(all_answers)==602
    summary={'scope':config['scope'],'original_validation_or_test_used':False,'primary_method':config['primary_method'],
        'primary_contrast':config['primary_contrast'],'coverage':pack['coverage'],'folds':fold_results,
        'methods':r18.pooled(all_windows,all_answers,pack),
        'paired_bootstrap':{'windows':r18.bootstrap(all_windows,config['bootstrap'],config['primary_contrasts']),
                            'answers':r18.bootstrap(all_answers,config['bootstrap'],config['primary_contrasts'])},
        'fold_mean':{name:{stage:{metric:float(np.mean([fold_results[str(f)][name][stage]['windows'][metric] for f in range(5)]))
          for metric in ('f1','auroc','average_precision')} for stage in ('fit','calibration','evaluation')} for name in METHODS},
        'strata':strata(all_windows,all_answers,lengths),'baseline_exact_reproduction':baseline_checks,
        'seconds':time.perf_counter()-started,'limits':config['limits']}
    assert snapshot()==snap
    # A final content check binds the arrays actually used to the saved file list.
    for files in feature_files.values():
        for path,digest in files.items():assert sha(Path(path))==digest,path
    savel(out/'window_scores_oof.jsonl',all_windows);savel(out/'answer_scores_oof.jsonl',all_answers)
    save(out/'summary.json',summary);save(out/'baseline_reproduction.json',baseline_checks)
    files=['source_snapshot19.json','feature_files.json','response_lengths.json','candidate_windows.jsonl',
           'window_scores_oof.jsonl','answer_scores_oof.jsonl','summary.json','baseline_reproduction.json']
    files += [f'fold_{f}_{suffix}' for f in range(5) for suffix in ('frozen.pkl','metrics.json','window_scores.jsonl','answer_scores.jsonl')]
    save(out/'complete19.json',{'utc':r10.utc(),'original_validation_or_test_used':False,'methods':list(METHODS),
         'files_sha256':{name:sha(out/name) for name in files}})
    print('ROUND19_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','prepare']);args=parser.parse_args()
    with r10.threadpool_limits(limits=4):
        if args.stage=='run':run()
        else:
            snapshot();p,d,f,lengths=prepare();print('PREPARE19_OK',p['coverage'],{k:v.shape for k,v in d.items()},flush=True)
