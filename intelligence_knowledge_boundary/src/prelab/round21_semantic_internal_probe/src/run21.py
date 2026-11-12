"""Semantic-vector probe development on frozen R16 train event folds."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r21_reuses_r20',ROOT.parent/'round20_localization_optimization/src/run20.py')
r20=importlib.util.module_from_spec(spec);spec.loader.exec_module(r20)
r19,r18,r17,r13,r10=r20.r19,r20.r18,r20.r17,r20.r13,r20.r10
read,readl,save,savel,sha=r20.read,r20.readl,r20.save,r20.savel,r20.sha
sys.path.insert(0,str(ROOT/'src'))
NEW=('harp256','hidden256','delta256','base_harp','base_delta','base_harp_delta')
METHODS=('base','slots_base','r19_all','lookback_tuned','redeep_tuned')+NEW
r18.METHODS=METHODS


def protocol():
    return {'version':'r21-vector-v1','methods':list(METHODS),'primary_method':'base_harp_delta',
      'scope':'R16 actual train301 questions/278 groups/602 fixed Qwen answers; exploratory same-group fivefold development',
      'C':[.001,.01,.1],'seed':20260919,'loss_mass':3854,
      'baseline_grids':{'lookback_C':[.001,.01,.1],'redeep_heads':[1,4,16],'redeep_layers':[4,14,28],'redeep_beta':[.2,.6,1.]},
      'pca':{'components':256,'n_iter':3,'seed':20260919,'seed_rule':'20260919 + zero_based_outer_fold','whiten':False,'weights':'fit-only label-independent group balanced'},
      'selection':'max min(cal_window_F1,cal_answer_F1); then window F1, window precision, lower complexity/C',
      'bootstrap':{'draws':2000,'seed':20260919},
      'contrasts':{'primary_vs_slots':['base_harp_delta','slots_base'],'primary_vs_lookback':['base_harp_delta','lookback_tuned'],
          'primary_vs_redeep':['base_harp_delta','redeep_tuned'],'primary_vs_r19':['base_harp_delta','r19_all'],
          'harp_vs_hidden':['harp256','hidden256'],'primary_vs_base':['base_harp_delta','base']},
      'heldout_original_used':False,'gold_changed':False,'granularity':'same original four raw BPE windows and answer=max windows',
      'cortex_exact_reproduction':False,'harp_original_weak_supervision_reproduced':False,
      'lumina_full_reproduced':False}


def prepare():
    fm=read(ROOT/'data/feature_manifest.json');assert fm['complete'] and fm['completed_count']==602
    for p,h in fm['extraction_signature']['code_sha256'].items():assert sha(Path(p))==h
    pack,old,files,lengths=r19.prepare()
    meta=r18.train_metadata();bank=r18.Bank(meta[2]);cache={};feature_files={}
    outputs={k:[] for k in ('harp','delta','hidden','ecs','pks','slots')}
    oldfm=read(r19.ROOT/'data/feature_manifest.json')
    for w in pack['windows']:
        rid=w['row_id'];raw=w['raw_token_indices']
        if rid not in cache:
            rec=fm['records'][rid];p=ROOT/rec['npz'];side=ROOT/rec['json'];m=read(side)
            assert sha(p)==rec['npz_sha256']==m['arrays_sha256'] and sha(side)==rec['json_sha256']
            assert m['source_generation_sha256']==meta[2][rid][1]
            with np.load(p,allow_pickle=False) as z:
                a={k:z[k].copy() for k in ('harp_256','hidden_delta_28','token_ids','response_token_offsets')}
            g=meta[2][rid][0];assert a['token_ids'].tolist()==g['response_token_ids']
            assert a['response_token_offsets'].tolist()==g['response_token_offsets']
            o=oldfm['records'][rid];op=r19.ROOT/o['npz'];assert sha(op)==o['npz_sha256']
            with np.load(op,allow_pickle=False) as z:
                a['ecs']=z['token_redeep_ecs'].copy();a['pks']=z['token_redeep_pks'].copy()
            cache[rid]=a
            for f in (p,side):feature_files[str(f.resolve())]=sha(f)
        a=cache[rid];b=bank.arrays(rid)
        for target,key in [('harp','harp_256'),('delta','hidden_delta_28'),('ecs','ecs'),('pks','pks')]:outputs[target].append(a[key][raw].mean(axis=0))
        outputs['hidden'].append(b['hidden_28'][raw].mean(axis=0))
        tokens=np.concatenate((b['lb'][raw],b['nll'][raw,None]),axis=1);n=len(tokens)
        if n<4:tokens=np.concatenate((tokens,np.repeat(tokens[-1:],4-n,axis=0)))
        outputs['slots'].append(np.concatenate((tokens.reshape(-1),np.asarray([1]*n+[0]*(4-n),np.float32))))
    d={k:np.asarray(v,np.float32) for k,v in outputs.items()};d['base']=old['base']
    d['r19_all']=np.concatenate([old[k] for k in ('base','mmd','ecs','pks')],axis=1)
    for k,width in [('harp',256),('delta',3584),('hidden',3584),('ecs',784),('pks',28),('slots',3144),('base',785),('r19_all',795)]:
        assert d[k].shape==(len(pack['windows']),width) and np.isfinite(d[k]).all()
    files['r21']=feature_files;files['r18_extra']=bank.files
    return pack,d,files,lengths


def projection(x,ix,rows,fold):
    b=r10.base_weights(rows,'token');w=np.asarray(b,np.float64);w/=w.sum()
    xf=x[ix].astype(np.float64);center=w@xf;xc=xf-center
    _,singular,components=randomized_svd(xc*np.sqrt(w[:,None]),n_components=256,n_iter=3,random_state=20260919+fold,flip_sign=True)
    total=float(np.einsum('ij,i,ij->',xc,w,xc))
    z=((x.astype(np.float64)-center)@components.T).astype(np.float32)
    return {'mean':center,'components':components,'singular_values':singular,'explained_variance_ratio_sum':float(np.sum(singular**2)/total),
        'fit_ix':np.asarray(ix),'fit_keys':[r['window_key'] for r in rows],'base_weights':b,'seed':20260919+fold,'n_iter':3},z


def thresholds(pack,values):
    wi=np.flatnonzero([w['main_eligible'] for w in pack['windows']]);ai=np.flatnonzero([a['main_eligible'] for a in pack['items']])
    av=r17.answer_scores(pack,values)
    return {'window':r10.threshold_search([pack['windows'][i]['gold'] for i in wi],values[wi]),
      'answer':r10.threshold_search([pack['items'][i]['gold'] for i in ai],av[ai])}


def candidate_key(ts,complexity):
    w,a=ts['window'],ts['answer']
    return (min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],-float(complexity))


def fit_lr_grid(x,pack,fit_ix,cal,cx,config):
    entries=[];objects=[]
    for c in config['C']:
        obj,v=r20.fit(x,pack['windows'],fit_ix,c,config)
        ts=thresholds(cal,v[cx]);entries.append({'C':c,'thresholds':ts,'selection_key':candidate_key(ts,c)})
        objects.append((obj,v,ts))
    chosen=max(range(len(entries)),key=lambda i:entries[i]['selection_key'])
    selected,v,ts=objects[chosen];obj=dict(selected)
    obj['all_lr_candidates']=[candidate for candidate,_,_ in objects]
    obj['calibration_candidates']=entries;obj['selected_candidate']=chosen
    return obj,v,ts


def snapshot():
    fm=read(ROOT/'data/feature_manifest.json');assert fm['complete']
    return {'r19_source':r19.snapshot(),'feature_manifest_sha256':sha(ROOT/'data/feature_manifest.json'),
      'code_sha256':{str(p.resolve()):sha(p) for p in [ROOT/'PLAN.md',ROOT/'protocol.json',Path(__file__),ROOT/'src/baselines21.py']}}


def run():
    import baselines21
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'started21.json').exists(),'Inspect existing run; do not overwrite or silently restart'
    config=read(ROOT/'protocol.json');assert config==protocol()
    snap=snapshot();pack,d,files,lengths=prepare()
    save(out/'source_snapshot21.json',snap);save(out/'feature_files.json',files)
    save(out/'started21.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot21.json')})
    savel(out/'candidate_windows.jsonl',pack['windows'])
    np.savez_compressed(out/'designs.npz',**d)
    assignment=read(r19.ROOT/'data/fold_assignment.json')['groups']
    gid=np.asarray([w['group_id'] for w in pack['windows']]);eligible=np.asarray([w['main_eligible'] for w in pack['windows']])
    all_windows=[];all_answers=[];fold_results={};started=time.perf_counter()
    for fold in range(5):
        eg=sorted(g for g,f in assignment.items() if f==fold);cg=sorted(g for g,f in assignment.items() if f==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg))
        tr,tx=r18.subset(pack,fg);ca,cx=r18.subset(pack,cg);ev,ex=r18.subset(pack,eg)
        fi=np.flatnonzero(np.isin(gid,fg)&eligible);fr=[pack['windows'][i] for i in fi]
        models={};values={};ts_all={};projections={}
        models['base'],values['base']=r20.frozen_baseline(pack,fold)
        models['slots_base'],values['slots_base']=r20.frozen_baseline(pack,fold,d['slots'])
        old=pickle.loads((r19.ROOT/'results'/f'fold_{fold}_frozen.pkl').read_bytes())['models']['base_all']
        models['r19_all']=old;values['r19_all']=old['model'].predict_proba(old['scaler'].transform(d['r19_all']).astype(np.float32))[:,1]
        for name in ('base','slots_base','r19_all'):
            ts_all[name]=thresholds(ca,values[name][cx]);assert ts_all[name]==models[name]['thresholds']
        baseline_features={'lb':d['base'][:,:784],'ecs':d['ecs'],'pks':d['pks']}
        candidate_sets={
            'lookback_tuned':baselines21.fit_lookback_candidates(baseline_features['lb'][fi],fr,{'seed':config['seed'],'loss_mass':config['loss_mass']}),
            'redeep_tuned':baselines21.fit_redeep_candidates(d['ecs'][fi],d['pks'][fi],fr,{'seed':config['seed'],'loss_mass':config['loss_mass']})}
        for name,candidates in candidate_sets.items():
            entries=[];candidate_values=[]
            for j,candidate in enumerate(candidates):
                v=np.asarray(baselines21.predict(candidate,baseline_features),np.float64)
                assert v.shape==(len(pack['windows']),) and np.isfinite(v).all()
                ts=thresholds(ca,v[cx]);complexity=candidate['complexity']
                selection=candidate_key(ts,0)[:3]+tuple(-float(x) for x in complexity)
                entries.append({'candidate_index':j,'candidate_id':candidate['candidate_id'],'complexity':complexity,'thresholds':ts,'selection_key':selection})
                candidate_values.append(v)
            j=max(range(len(entries)),key=lambda j:entries[j]['selection_key'])
            models[name]={'candidate':candidates[j],'all_candidates':candidates,'selected_candidate':j,'calibration_candidates':entries}
            values[name]=candidate_values[j];ts_all[name]=entries[j]['thresholds']
            print('FOLD',fold+1,'SELECTED_BASELINE',name,j,flush=True)
        projected={}
        for key in ('hidden','delta'):
            projections[key],projected[key]=projection(d[key],fi,fr,fold)
            print('FOLD',fold+1,'PROJECTED',key,'VAR',projections[key]['explained_variance_ratio_sum'],flush=True)
        new_designs={'harp256':d['harp'],'hidden256':projected['hidden'],'delta256':projected['delta'],
            'base_harp':np.concatenate((d['base'],d['harp']),axis=1),
            'base_delta':np.concatenate((d['base'],projected['delta']),axis=1),
            'base_harp_delta':np.concatenate((d['base'],d['harp'],projected['delta']),axis=1)}
        for name,x in new_designs.items():
            models[name],values[name],ts_all[name]=fit_lr_grid(x,pack,fi,ca,cx,config)
            models[name]['design']=name
            print('FOLD',fold+1,'SELECTED_NEW',name,'C',models[name]['C'],flush=True)
        fw=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']];fa=None;detail={}
        for name in METHODS:
            v=values[name];ts=ts_all[name]
            detail[name]={stage:r17.metrics(p,v[ix],ts) for stage,p,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]}
            detail[name]['thresholds']=ts
            ww,aa=r18.scored_records(ev,v[ex],ts,name)
            for a,b in zip(fw,ww):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
            if fa is None:fa=[dict(a,scores={},predictions={},fold=fold) for a in aa]
            for a,b in zip(fa,aa):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
            print('FOLD',fold+1,name,'WINDOW',detail[name]['evaluation']['windows']['f1'],'ANSWER',detail[name]['evaluation']['answers']['f1'],flush=True)
        frozen={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg,'models':models,'projections':projections,'thresholds':ts_all}
        (out/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(frozen,protocol=5))
        save(out/f'fold_{fold}_metrics.json',detail);savel(out/f'fold_{fold}_window_scores.jsonl',fw);savel(out/f'fold_{fold}_answer_scores.jsonl',fa)
        fold_results[str(fold)]=detail;all_windows.extend(fw);all_answers.extend(fa)
    assert len(all_windows)==len(pack['windows']) and len(all_answers)==602
    assert len({w['window_key'] for w in all_windows})==len(all_windows)
    r19.METHODS=METHODS
    summary={'scope':config['scope'],'coverage':pack['coverage'],'primary_method':config['primary_method'],'folds':fold_results,
      'methods':r18.pooled(all_windows,all_answers,pack),
      'fold_mean':{m:{stage:{metric:float(np.mean([fold_results[str(f)][m][stage]['windows'][metric] for f in range(5)])) for metric in ('f1','auroc','average_precision')} for stage in ('fit','calibration','evaluation')} for m in METHODS},
      'paired_bootstrap':{unit:r18.bootstrap(rr,config['bootstrap'],config['contrasts']) for unit,rr in [('windows',all_windows),('answers',all_answers)]},
      'strata':r19.strata(all_windows,all_answers,lengths),'seconds':time.perf_counter()-started,'original_validation_or_test_used':False}
    old19=read(r19.ROOT/'results/summary.json')['methods'];old18=read(r18.ROOT/'results/summary.json')['methods']
    assert summary['methods']['base']==old19['base'] and summary['methods']['slots_base']==old18['slots_lr'] and summary['methods']['r19_all']==old19['base_all']
    assert snapshot()==snap
    for group in files.values():
        for p,digest in group.items():assert sha(Path(p))==digest
    savel(out/'window_scores_oof.jsonl',all_windows);savel(out/'answer_scores_oof.jsonl',all_answers);save(out/'summary.json',summary)
    names=['source_snapshot21.json','feature_files.json','designs.npz','candidate_windows.jsonl','window_scores_oof.jsonl','answer_scores_oof.jsonl','summary.json']
    names += [f'fold_{f}_{s}' for f in range(5) for s in ('frozen.pkl','metrics.json','window_scores.jsonl','answer_scores.jsonl')]
    save(out/'complete21.json',{'utc':r10.utc(),'files_sha256':{n:sha(out/n) for n in names},'original_validation_or_test_used':False})
    print('ROUND21_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','run']);args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='initialize':
            path=ROOT/'protocol.json';assert not path.exists();save(path,protocol());print(path)
        elif args.stage=='prepare':
            p,d,_,_=prepare();print(p['coverage'],{k:v.shape for k,v in d.items()})
        else:run()
