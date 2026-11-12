"""Fixed structural iteration; frozen old data and event folds, no new LLM pass."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
R19=ROOT.parent/'round19_three_signal_probe'
spec=importlib.util.spec_from_file_location('r20_reuses_r19',R19/'src/run19.py')
r19=importlib.util.module_from_spec(spec);spec.loader.exec_module(r19)
r18,r17,r13,r10=r19.r18,r19.r17,r19.r13,r19.r10
read,readl,save,savel,sha=r19.read,r19.readl,r19.save,r19.savel,r19.sha
METHODS=('base','slots_base','token_lr_strong','token_lr','token_tree','window_tree','gated_unconditional','gated_conditional')
r18.METHODS=METHODS


def protocol():
    return {'version':'r20-structure-v1','methods':list(METHODS),'primary_method':'gated_conditional',
        'target':['answer_f1_about_0.75','four_bpe_window_f1_about_0.75'],
        'scope':'Exploratory R16 actual train301 questions/278 groups/602 fixed answers; original heldout excluded',
        'seed':20260919,'loss_mass':3854,'lr_cs':{'token_lr_strong':.001,'token_lr':.01,'conditional':.001},
        'tree':{'learning_rate':.05,'max_iter':100,'max_leaf_nodes':7,'max_depth':3,
                'min_samples_leaf':20,'l2_regularization':10.,'early_stopping':False,'random_state':20260919},
        'bootstrap':{'draws':2000,'seed':20260919},
        'contrasts':{**{m+'_vs_base':[m,'base'] for m in METHODS if m!='base'},
                     'gated_conditional_vs_slots':['gated_conditional','slots_base']},
        'full_response_used_by_gate':True,'feature_widths':{'token_full':819,'token_reduced':147,'window_tree':441,'answer_tree':441},
        'selection':'All fixed candidates reported; calibration only selects separate window and answer thresholds',
        'no_original_validation_or_test':True,'original_labels_changed':False}


def prepare():
    meta=r18.train_metadata();bank=r18.Bank(meta[2]);signals=r19.SignalBank(meta[2])
    pack=r17.cohort(r18.NEW,'train',meta,bank)
    assert pack['windows']==readl(R19/'results/candidate_windows.jsonl')
    full=[];small=[]
    for t in pack['tokens']:
        rid,j=t['row_id'],t['token_index'];a=bank.arrays(rid);s=signals.row(rid)
        extras=np.concatenate((a['nll'][j:j+1],s['mmd'][j],s['ecs'][j],s['pks'][j],a['new_features'][j],a['surface_features'][j]))
        full.append(np.concatenate((a['lb'][j],extras)))
        small.append(np.concatenate((a['lb'][j].reshape(4,7,28).mean(axis=1).reshape(-1),extras)))
    full=np.asarray(full,np.float32);small=np.asarray(small,np.float32)
    lookup={t['token_key']:i for i,t in enumerate(pack['tokens'])}
    by_item=defaultdict(list)
    for i,t in enumerate(pack['tokens']):
        # Visible lexical/output-parser membership only; no risk span or stance.
        if t['lexical'] and len(t['item_ids'])==1:by_item[t['item_ids'][0]].append(i)
    win_ix=[];window=[];slots=[]
    def stats(x):return np.concatenate((x.mean(axis=0),x.max(axis=0),x.std(axis=0))).astype(np.float32)
    for w in pack['windows']:
        ix=np.asarray([lookup[k] for k in w['token_keys']],np.int64);assert len(ix)
        win_ix.append(ix)
        raw=[lookup[w['row_id']+f'__token{j}'] for j in w['raw_token_indices']]
        window.append(stats(small[raw]))
        tokens=full[raw,:785];n=len(tokens)
        if n<4:tokens=np.concatenate((tokens,np.repeat(tokens[-1:],4-n,axis=0)))
        slots.append(np.concatenate((tokens.reshape(-1),np.asarray([1]*n+[0]*(4-n),np.float32))))
    answer=[]
    for a in pack['items']:
        ix=by_item[a['item_id']]
        if not ix:
            # Failed parser: use visible lexical tokens within whole response.
            ix=[i for i,t in enumerate(pack['tokens']) if t['row_id']==a['row_id'] and t['lexical']]
        assert ix,a['item_id'];answer.append(stats(small[ix]))
    answer=np.asarray(answer,np.float32);window=np.asarray(window,np.float32);slots=np.asarray(slots,np.float32)
    assert full.shape==(len(pack['tokens']),819) and small.shape==(len(pack['tokens']),147)
    assert window.shape==(len(pack['windows']),441) and answer.shape==(602,441)
    assert all(np.isfinite(x).all() for x in (full,small,window,answer))
    ai={a['item_id']:i for i,a in enumerate(pack['items'])}
    return pack,{'full':full,'small':small,'window':window,'answer':answer,'slots':slots},win_ix,np.asarray([ai[w['item_ids'][0]] for w in pack['windows']]),{'r18':bank.files,'r19':signals.files}


def fit(x,rows,ix,c=None,config=None):
    config=config or protocol();ix=np.asarray(ix,int);rr=[rows[i] for i in ix]
    # Same hierarchy works for answer rows after explicitly adding membership.
    rr=[dict(r,item_ids=[r['item_id']]) if 'item_ids' not in r else r for r in rr]
    assert all(r['main_eligible'] for r in rr)
    y=np.asarray([r['gold'] for r in rr],int);assert set(y)=={0,1}
    b,w,f=r13.weights(rr,config['loss_mass'])
    sc=StandardScaler().fit(x[ix],sample_weight=b) if c is not None else None
    z=sc.transform(x[ix]).astype(np.float32) if sc is not None else x[ix]
    model=(LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=config['seed']) if c is not None
           else HistGradientBoostingClassifier(**config['tree']))
    model.fit(z,y,sample_weight=w)
    if c is not None:assert max(model.n_iter_)<2000
    value=model.predict_proba(sc.transform(x).astype(np.float32) if sc is not None else x)[:,1]
    assert np.isfinite(value).all()
    frozen={'model':model,'scaler':sc,'fit_ix':ix,'fit_y':y,'base_weights':b,'loss_weights':w,'class_factors':f,
        'fit_keys':[r.get('token_key',r.get('window_key',r.get('item_id'))) for r in rr],
        'fit_groups':sorted({r['group_id'] for r in rr}),'width':x.shape[1],'C':c,'loss_mass':config['loss_mass']}
    return frozen,value


def replay(model,x):
    sc=model['scaler'];z=sc.transform(x).astype(np.float32) if sc is not None else x
    return model['model'].predict_proba(z)[:,1]


def frozen_baseline(pack,fold,slots=None):
    directory=R19 if slots is None else r18.ROOT
    key='base' if slots is None else 'slots_lr'
    old=pickle.loads((directory/'results'/f'fold_{fold}_frozen.pkl').read_bytes())['models'][key]
    raw=pack['matrix'] if slots is None else slots
    value=old['model'].predict_proba(old['scaler'].transform(raw).astype(np.float32))[:,1]
    return old,value


def snapshot():
    prior=read(R19/'results/complete19.json')
    for name,digest in prior['files_sha256'].items():assert sha(R19/'results'/name)==digest
    return {'r19_complete_sha256':sha(R19/'results/complete19.json'),'source':r19.snapshot(),
        'local_sha256':{str(p.resolve()):sha(p) for p in [ROOT/'PLAN.md',ROOT/'protocol.json',Path(__file__)]}}


def run():
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'started20.json').exists(),'Refusing to overwrite prior attempt; inspect state first'
    config=read(ROOT/'protocol.json');assert config==protocol()
    snap=snapshot();pack,d,win_ix,win_answer,files=prepare()
    save(out/'source_snapshot20.json',snap);save(out/'feature_files.json',files)
    save(out/'started20.json',{'utc':r10.utc(),'source_sha256':sha(out/'source_snapshot20.json')})
    np.savez_compressed(out/'designs.npz',**d,win_answer=win_answer)
    savel(out/'token_index.jsonl',[{k:t[k] for k in ['token_key','row_id','group_id','gold','main_eligible']} for t in pack['tokens']])
    savel(out/'candidate_windows.jsonl',pack['windows'])
    assignment=read(R19/'data/fold_assignment.json')['groups'];assert set(assignment)=={a['group_id'] for a in pack['items']}
    tg=np.asarray([t['group_id'] for t in pack['tokens']]);wg=np.asarray([w['group_id'] for w in pack['windows']]);ag=np.asarray([a['group_id'] for a in pack['items']])
    te=np.asarray([t['main_eligible'] for t in pack['tokens']]);we=np.asarray([w['main_eligible'] for w in pack['windows']]);ae=np.asarray([a['main_eligible'] for a in pack['items']])
    risk_items={a['item_id'] for a in pack['items'] if a['main_eligible'] and a['gold']==1}
    conditional=np.asarray([len(t['item_ids'])==1 and t['item_ids'][0] in risk_items for t in pack['tokens']])
    windows=[];answers=[];details={};started=time.perf_counter()
    pool=lambda v:np.asarray([max(v[ix]) for ix in win_ix])
    for fold in range(5):
        eg=sorted(g for g,f in assignment.items() if f==fold);cg=sorted(g for g,f in assignment.items() if f==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg))
        ti=np.flatnonzero(np.isin(tg,fg)&te);wi=np.flatnonzero(np.isin(wg,fg)&we);ai=np.flatnonzero(np.isin(ag,fg)&ae)
        ci=np.flatnonzero(np.isin(tg,fg)&te&conditional)
        tr,tx=r18.subset(pack,fg);ca,cx=r18.subset(pack,cg);ev,ex=r18.subset(pack,eg)
        models={};values={};parts={};part_scores={}
        models['base'],values['base']=frozen_baseline(pack,fold)
        models['slots_base'],values['slots_base']=frozen_baseline(pack,fold,d['slots'])
        for name,key,rows,ix,c in [('token_lr_strong','full',pack['tokens'],ti,.001),('token_lr','full',pack['tokens'],ti,.01),
                ('token_tree','small',pack['tokens'],ti,None),('window_tree','window',pack['windows'],wi,None),
                ('answer_gate','answer',pack['items'],ai,None),('conditional','full',pack['tokens'],ci,.001)]:
            m,v=fit(d[key],rows,ix,c,config);m.update(design=key)
            parts[name]=m;part_scores[name]=v
            if name in ('token_lr_strong','token_lr','token_tree'):values[name]=pool(v)
            elif name=='window_tree':values[name]=v
            print('FOLD',fold+1,'COMPONENT',name,'FIT',len(ix),flush=True)
        gate=part_scores['answer_gate'][win_answer]
        values['gated_unconditional']=gate*values['token_lr_strong']
        values['gated_conditional']=gate*pool(part_scores['conditional'])
        fold_detail={};fw=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']];fa=None;thresholds={}
        for name in METHODS:
            v=values[name];assert len(v)==len(pack['windows']) and np.isfinite(v).all()
            cwi=np.flatnonzero([w['main_eligible'] for w in ca['windows']]);cai=np.flatnonzero([a['main_eligible'] for a in ca['items']])
            av=r17.answer_scores(ca,v[cx]);ts={'window':r10.threshold_search([ca['windows'][i]['gold'] for i in cwi],v[cx][cwi]),
                'answer':r10.threshold_search([ca['items'][i]['gold'] for i in cai],av[cai])}
            if name in ('base','slots_base'):assert ts==models[name]['thresholds']
            thresholds[name]=ts
            fold_detail[name]={stage:r17.metrics(p,v[ix],ts) for stage,p,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]}
            fold_detail[name]['thresholds']=ts
            ww,aa=r18.scored_records(ev,v[ex],ts,name)
            for a,b in zip(fw,ww):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
            if fa is None:fa=[dict(a,scores={},predictions={},fold=fold) for a in aa]
            for a,b in zip(fa,aa):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
            print('FOLD',fold+1,name,'WINDOW',fold_detail[name]['evaluation']['windows']['f1'],'ANSWER',fold_detail[name]['evaluation']['answers']['f1'],flush=True)
        frozen={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg,'parts':parts,'baseline':models['base'],'slots_baseline':models['slots_base'],'thresholds':thresholds}
        (out/f'fold_{fold}_frozen.pkl').write_bytes(pickle.dumps(frozen,protocol=5))
        save(out/f'fold_{fold}_metrics.json',fold_detail);savel(out/f'fold_{fold}_window_scores.jsonl',fw);savel(out/f'fold_{fold}_answer_scores.jsonl',fa)
        details[str(fold)]=fold_detail;windows.extend(fw);answers.extend(fa)
    assert len(windows)==len(pack['windows']) and len(answers)==602
    assert len({w['window_key'] for w in windows})==len(windows)
    lengths={rid:len(g['response_token_ids']) for rid,(g,_) in r18.train_metadata()[2].items()}
    # Reused registry is private to this imported module, no old file changes.
    r19.METHODS=METHODS
    summary={'scope':config['scope'],'coverage':pack['coverage'],'primary_method':config['primary_method'],
        'methods':r18.pooled(windows,answers,pack),'folds':details,
        'fold_mean':{m:{stage:{metric:float(np.mean([details[str(f)][m][stage]['windows'][metric] for f in range(5)])) for metric in ('f1','auroc','average_precision')} for stage in ('fit','calibration','evaluation')} for m in METHODS},
        'paired_bootstrap':{unit:r18.bootstrap(rr,config['bootstrap'],config['contrasts']) for unit,rr in [('windows',windows),('answers',answers)]},
        'strata':r19.strata(windows,answers,lengths),'seconds':time.perf_counter()-started,'original_validation_or_test_used':False}
    old=read(R19/'results/summary.json')['methods']['base'];assert summary['methods']['base']==old
    assert summary['methods']['slots_base']==read(r18.ROOT/'results/summary.json')['methods']['slots_lr']
    assert snapshot()==snap
    for group in files.values():
        for p,digest in group.items():assert sha(Path(p))==digest
    savel(out/'window_scores_oof.jsonl',windows);savel(out/'answer_scores_oof.jsonl',answers);save(out/'summary.json',summary)
    names=['source_snapshot20.json','feature_files.json','designs.npz','token_index.jsonl','candidate_windows.jsonl','window_scores_oof.jsonl','answer_scores_oof.jsonl','summary.json']
    names += [f'fold_{f}_{s}' for f in range(5) for s in ('frozen.pkl','metrics.json','window_scores.jsonl','answer_scores.jsonl')]
    save(out/'complete20.json',{'utc':r10.utc(),'files_sha256':{n:sha(out/n) for n in names},'original_validation_or_test_used':False})
    print('ROUND20_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','run']);args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='initialize':
            path=ROOT/'protocol.json';assert not path.exists();save(path,protocol());print(path)
        elif args.stage=='prepare':
            pack,d,_,_,_=prepare();print(pack['coverage'],{k:v.shape for k,v in d.items()})
        else:run()
