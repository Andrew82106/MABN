"""Frozen fivefold development: one extra same-Qwen evidence verification.

CPU only. Fits answer probes, preserves output-defined original BPE windows,
and freezes all fold choices before outer metrics. No original held-out use.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import importlib.util, pickle, sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
R21=ROOT.parent/'round21_semantic_internal_probe'
spec=importlib.util.spec_from_file_location('r22_reuses21',R21/'src/run21.py')
r21=importlib.util.module_from_spec(spec);spec.loader.exec_module(r21)
r19,r18,r17,r13,r10=r21.r19,r21.r18,r21.r17,r21.r13,r21.r10
read,readl,save,savel,sha=r21.read,r21.readl,r21.save,r21.savel,r21.sha
sys.path.insert(0,str(R21/'src'))
import baselines21

BASES=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned')
NEW=('verifier_direct_broadcast','verifier_learned_broadcast','verifier_direct_slots','verifier_learned_slots')
METHODS=BASES+NEW
r18.METHODS=METHODS;r19.METHODS=METHODS


def protocol():
    return {'version':'r22-answer-verification-v1','methods':list(METHODS),'primary_method':'verifier_learned_slots',
      'scope':'Exploratory repeated CV on R16 actual train301 questions/278 groups/602 frozen answers; original validation/test excluded',
      'seed':20260922,'C':[.001,.01,.1],'T':[.25,.5,1.,2.,4.],'loss_mass':3854.,
      'probe':'Answer-level LR on fit-only PCA64 verifier_hidden plus B-A logit and full-vocabulary A+B probability mass',
      'pca':{'components':64,'n_iter':3,'seed_rule':'20260922 + zero_based_outer_fold','whiten':False,'weights':'fit-only answer base weights normalized to sum1'},
      'weights':'Each fit event group equal, condition equal, answer equal before fit-only class factors; re-equalize group loss and normalize total mass3854',
      'lr':{'solver':'liblinear','penalty':'l2','max_iter':2000,'scaler':'fit-only base-weighted StandardScaler','cpu_threads':4},
      'folds':'Fixed R18/R19 assignment; outer=f, calibration=(f+1)%5, remaining three folds fit',
      'answer_labels':'Resolved asserted risk; individually reviewed safe refusals risk0; unresolved excluded without zero imputation',
      'window_labels':'Unchanged 4 raw BPE, punctuation counts length, OR of labeled lexical risk; only resolved assertions evaluated',
      'direct_score':'Saved softmax over A/B, B probability; model verdict is never used as gold',
      'relative_slots':'exp((clip-logit(s)-max_answer_clip-logit(s))/T), all visible output windows, epsilon=float64 machine epsilon',
      'joint_score':'Verifier answer risk * relative_slots; exact max over answer windows equals verifier risk',
      'broadcast':'Verifier answer risk copied to every answer window; explicit post-answer localization control',
      'individual_threshold_rule':'Each calibration window and answer threshold maximizes risk F1; tie precision then higher threshold',
      'learned_broadcast_selection':'Choose C by calibration answer F1, answer precision, smaller C; independently freeze window threshold of selected C',
      'joint_selection':'Choose T or C/T by min(cal window F1, cal answer F1); tie window F1, window precision, smaller C, smaller T',
      'fit_counts':{'new_lr':15,'pca':5,'learned_joint_candidates':75,'direct_joint_candidates':25},
      'all_candidate_tables_saved':True,'no_fit_plus_cal_refit':True,
      'bootstrap':{'draws':2000,'seed':20260922},
      'contrasts':{'primary_vs_slots':['verifier_learned_slots','slots_base'],
        'primary_vs_lookback':['verifier_learned_slots','lookback_tuned'],'primary_vs_redeep':['verifier_learned_slots','redeep_tuned'],
        'primary_vs_r19':['verifier_learned_slots','r19_all'],'primary_vs_r21':['verifier_learned_slots','base_harp_delta'],
        'primary_vs_direct_joint':['verifier_learned_slots','verifier_direct_slots'],
        'primary_vs_learned_broadcast':['verifier_learned_slots','verifier_learned_broadcast']},
      'extra_same_qwen_verification':True,'original_generation_state':False,'original_answer_changed':False,
      'risk_scores_proven_calibrated':False,'original_validation_or_test_used':False,'gold_changed':False}


def snapshot():
    assert r21.snapshot()==read(R21/'results/source_snapshot21.json')
    complete=read(R21/'results/complete21.json')
    for name,h in complete['files_sha256'].items():assert sha(R21/'results'/name)==h
    audit=read(R21/'results/INDEPENDENT_AUDIT21.json');assert audit['status']=='passed' and audit['complete_sha256']==sha(R21/'results/complete21.json')
    manifest=read(ROOT/'data/feature_manifest.json');sig=read(ROOT/'data/signature.json')
    assert manifest['complete'] and manifest['completed_count']==manifest['expected_count']==602
    assert sig['code_sha256']==sha(ROOT/'src/extract22.py') and manifest['signature_sha256']==r10.digest(sig)
    assert manifest['labels_read'] is False and manifest['original_validation_or_test_parsed'] is False
    return {'prior_r21':r21.snapshot(),'prior_complete21_sha256':sha(R21/'results/complete21.json'),
      'prior_audit21_sha256':sha(R21/'results/INDEPENDENT_AUDIT21.json'),
      'files_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'protocol22.json',ROOT/'EXPERIMENT22.md',ROOT/'data/feature_manifest.json',ROOT/'data/signature.json',ROOT/'data/plans.json',ROOT/'src/extract22.py']}}


def prepare():
    meta=r18.train_metadata();bank=r18.Bank(meta[2]);pack=r17.cohort(r18.NEW,'train',meta,bank)
    assert pack['windows']==readl(R21/'results/candidate_windows.jsonl')
    assert pack['coverage']['eligible_windows']==9526 and pack['coverage']['item_evaluable']==598
    with np.load(R21/'results/designs.npz',allow_pickle=False) as z:
        designs={k:z[k].copy() for k in ('base','slots','r19_all','harp','delta','ecs','pks')}
    assert np.array_equal(designs['base'],pack['matrix'])
    fm=read(ROOT/'data/feature_manifest.json');sig=read(ROOT/'data/signature.json');plans=read(ROOT/'data/plans.json')
    assert fm['complete'] and set(fm['records'])==set(plans)==set(meta[2])
    hidden=[];extra=[];direct=[];logits=[];mass=[];files={}
    for item in pack['items']:
        rid=item['row_id'];entry=fm['records'][rid];path=ROOT/entry['npz'];side=ROOT/entry['json'];m=read(side)
        assert sha(path)==entry['npz_sha256']==m['npz_sha256'] and sha(side)==entry['json_sha256']
        assert m['source_generation_sha256']==meta[2][rid][1] and m['signature_sha256']==fm['signature_sha256']
        assert m['plan_sha256']==r10.digest(plans[rid]) and plans[rid]['source_generation_sha256']==meta[2][rid][1]
        assert m['labels_read'] is False and m['output_regenerated'] is False
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)=={'verifier_hidden','ab_logits','ab_probabilities','ab_full_vocab_probabilities'}
            h=z['verifier_hidden'].copy();ab=z['ab_logits'].copy();pair=z['ab_probabilities'].copy();full=z['ab_full_vocab_probabilities'].copy()
        assert h.shape==(3584,) and all(x.shape==(2,) for x in (ab,pair,full))
        assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in (h,ab,pair,full))
        assert ((pair>=0)&(pair<=1)).all() and abs(float(pair.sum())-1)<1e-6
        assert ((full>=0)&(full<=1)).all() and float(full.sum())<=1+1e-6
        gap=float(ab[1])-float(ab[0]);pred=1/(1+np.exp(-gap))
        assert abs(pred-float(pair[1]))<2e-7 and m['risk_score']==float(pair[1])
        value=float(full.sum());assert m['pair_vocab_mass']==value
        hidden.append(h);extra.append([gap,value]);direct.append(float(pair[1]));logits.append(ab);mass.append(full)
        for f in (path,side):files[str(f.resolve())]=sha(f)
    arrays={'hidden':np.asarray(hidden,np.float32),'extra':np.asarray(extra,np.float32),'direct':np.asarray(direct,np.float64),
      'ab_logits':np.asarray(logits,np.float32),'ab_full_vocab_probabilities':np.asarray(mass,np.float32)}
    assert arrays['hidden'].shape==(602,3584) and arrays['extra'].shape==(602,2)
    ai={a['item_id']:i for i,a in enumerate(pack['items'])}
    wanswer=np.asarray([ai[w['item_ids'][0]] for w in pack['windows']],np.int64)
    lengths={rid:len(g['response_token_ids']) for rid,(g,_) in meta[2].items()}
    return pack,designs,arrays,wanswer,{'new_verifier':files,'r18':bank.files},lengths


def prior_scores(fold,d):
    old=pickle.loads((R21/'results'/f'fold_{fold}_frozen.pkl').read_bytes());result={}
    def predict(obj,x):return obj['model'].predict_proba(obj['scaler'].transform(x).astype(np.float32))[:,1]
    for name,key in [('base','base'),('slots_base','slots'),('r19_all','r19_all')]:result[name]=predict(old['models'][name],d[key])
    pc=old['projections']['delta'];delta=((d['delta'].astype(np.float64)-pc['mean'])@pc['components'].T).astype(np.float32)
    result['base_harp_delta']=predict(old['models']['base_harp_delta'],np.column_stack((d['base'],d['harp'],delta)))
    feature={'lb':d['base'][:,:784],'ecs':d['ecs'],'pks':d['pks']}
    for name in ('lookback_tuned','redeep_tuned'):result[name]=baselines21.predict(old['models'][name]['candidate'],feature)
    return result,{m:old['thresholds'][m] for m in BASES}


def relative_slots(windows,score,t):
    by_answer=defaultdict(list);score=np.asarray(score,np.float64)
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    for i,w in enumerate(windows):by_answer[w['item_ids'][0]].append(i)
    eps=np.finfo(float).eps;b=np.clip(score,eps,1-eps);logit=np.log(b)-np.log1p(-b);out=np.empty(len(score))
    for ix in by_answer.values():
        v=logit[ix];out[ix]=np.exp((v-v.max())/t);assert out[ix].max()==1.
    return out


def fit_projection(hidden,ix,items,fold):
    rows=[items[i] for i in ix];b=r10.base_weights(rows,'item');w=b.astype(np.float64,copy=True);w/=w.sum()
    raw=hidden[ix].astype(np.float64);center=w@raw
    _,sv,components=randomized_svd((raw-center)*np.sqrt(w[:,None]),n_components=64,n_iter=3,random_state=20260922+fold,flip_sign=True)
    z=((hidden.astype(np.float64)-center)@components.T).astype(np.float32)
    total=float(np.einsum('ij,i,ij->',raw-center,w,raw-center))
    obj={'mean':center,'components':components,'singular_values':sv,'fit_ix':ix,'fit_keys':[r['item_id'] for r in rows],
      'fit_groups':sorted({r['group_id'] for r in rows}),'base_weights':b,'pca_weights':w,'seed':20260922+fold,'n_iter':3,
      'explained_variance_ratio_sum':float(np.sum(sv**2)/total),'whiten':False}
    return obj,z


def fit_probes(x,ix,items,config):
    rows=[items[i] for i in ix];assert all(r['main_eligible'] for r in rows)
    y=np.asarray([r['gold'] for r in rows],int);b=r10.base_weights(rows,'item');w,f=r10.loss_weights(rows,y,b);w*=config['loss_mass']/w.sum()
    assert set(y)=={0,1};sc=StandardScaler().fit(x[ix],sample_weight=b);zz=sc.transform(x).astype(np.float32)
    objects=[];scores=[]
    for c in config['C']:
        model=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=config['seed'])
        model.fit(zz[ix],y,sample_weight=w);assert model.n_iter_.max()<2000
        objects.append({'model':model,'scaler':sc,'C':c,'fit_ix':ix,'fit_keys':[r['item_id'] for r in rows],'fit_y':y,
          'base_weights':b,'loss_weights':w,'class_factors':f,'fit_groups':sorted({r['group_id'] for r in rows}),'width':66,'target_loss_mass':config['loss_mass']})
        scores.append(model.predict_proba(zz)[:,1])
    return objects,scores


def thresholds(cal,v):return r21.thresholds(cal,v)


def key(ts,c=0.,t=0.):
    w,a=ts['window'],ts['answer'];return (min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],-c,-t)


def synthetic_checks():
    windows=[{'item_ids':[x]} for x in ('A','A','B','B','C')];s=np.asarray([.1,.9,0.,0.,1.])
    membership=np.asarray([0,0,1,1,2]);p=np.asarray([.45,0.,1.])
    for t in protocol()['T']:
        local=relative_slots(windows,s,t);v=p[membership]*local
        assert np.array_equal(np.asarray([max(v[membership==i]) for i in range(3)]),p)
    assert key({'window':{'validation_f1':.8,'validation_precision':.8},'answer':{'validation_f1':.5}})<key({'window':{'validation_f1':.7,'validation_precision':.7},'answer':{'validation_f1':.7}})


def fit():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'fit_started22.json').exists(),'Refuse silent rerun of started fit'
    config=read(ROOT/'protocol22.json');assert config==protocol();synthetic_checks()
    snap=snapshot();pack,d,new,wanswer,files,lengths=prepare()
    save(out/'source_snapshot22.json',snap);save(out/'feature_files22.json',files);save(out/'lengths22.json',lengths)
    save(out/'fit_started22.json',{'utc':r10.utc(),'source_snapshot_sha256':sha(out/'source_snapshot22.json')})
    savel(out/'candidate_windows22.jsonl',pack['windows']);savel(out/'answer_index22.jsonl',pack['items'])
    np.savez_compressed(out/'verifier_designs22.npz',**new,window_answer_index=wanswer)
    assignment=read(r19.ROOT/'data/fold_assignment.json')['groups'];ag=np.asarray([a['group_id'] for a in pack['items']]);ae=np.asarray([a['main_eligible'] for a in pack['items']])
    started=time.perf_counter();artifact_names=[]
    for fold in range(5):
        eg=sorted(g for g,f in assignment.items() if f==fold);cg=sorted(g for g,f in assignment.items() if f==(fold+1)%5);fg=sorted(set(assignment)-set(eg)-set(cg))
        assert not(set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        ca,cx=r18.subset(pack,cg);ix=np.flatnonzero(np.isin(ag,fg)&ae)
        pc,z=fit_projection(new['hidden'],ix,pack['items'],fold);x=np.column_stack((z,new['extra'])).astype(np.float32)
        candidates,probs=fit_probes(x,ix,pack['items'],config)
        old,oldts=prior_scores(fold,d);rel={t:relative_slots(pack['windows'],old['slots_base'],t) for t in config['T']}
        for name in BASES:assert thresholds(ca,old[name][cx])==oldts[name]
        all_values=dict(old);all_thresholds=dict(oldts);selection={};tables={}
        direct=new['direct'];dv=direct[wanswer]
        all_values['verifier_direct_broadcast']=dv;all_thresholds['verifier_direct_broadcast']=thresholds(ca,dv[cx])
        tables['learned_broadcast']=[];broadcast_values=[]
        for j,(candidate,p) in enumerate(zip(candidates,probs)):
            v=p[wanswer];ts=thresholds(ca,v[cx]);a=ts['answer']
            entry={'candidate_index':j,'C':candidate['C'],'thresholds':ts,'selection_key':[a['validation_f1'],a['validation_precision'],-candidate['C']]}
            tables['learned_broadcast'].append(entry);broadcast_values.append(v)
        j=max(range(3),key=lambda j:tables['learned_broadcast'][j]['selection_key']);selection['verifier_learned_broadcast']=tables['learned_broadcast'][j]
        all_values['verifier_learned_broadcast']=broadcast_values[j];all_thresholds['verifier_learned_broadcast']=selection['verifier_learned_broadcast']['thresholds']
        for method,sources in [('verifier_direct_slots',[(None,0.,direct)]),('verifier_learned_slots',[(j,c['C'],p) for j,(c,p) in enumerate(zip(candidates,probs))])]:
            table=[];vv=[]
            for j,c,p in sources:
                for t in config['T']:
                    v=p[wanswer]*rel[t];assert np.array_equal(r17.answer_scores(pack,v),p)
                    ts=thresholds(ca,v[cx]);table.append({'candidate_index':j,'C':c,'T':t,'thresholds':ts,'selection_key':list(key(ts,c,t))});vv.append(v)
            best=max(range(len(table)),key=lambda j:table[j]['selection_key']);selection[method]=dict(table[best],table_index=best)
            all_values[method]=vv[best];all_thresholds[method]=table[best]['thresholds'];tables[method]=table
        assert set(all_values)==set(METHODS)
        # No outer metric is computed before every fold artifact is frozen.
        frozen={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg,'projection':pc,'all_lr_candidates':candidates,
          'selection':selection,'calibration_candidates':tables,'thresholds':all_thresholds,'fit_answer_indices':ix,
          'old_scores_and_thresholds_from_r21':True,'extra_same_qwen_verification':True}
        fn=f'fold_{fold}_frozen22.pkl';(out/fn).write_bytes(pickle.dumps(frozen,protocol=5))
        sn=f'fold_{fold}_scores22.npz';np.savez_compressed(out/sn,**all_values,
          **{f'probe_C{j}_answer':p for j,p in enumerate(probs)},direct_answer=direct,
          projected_verifier=z,probe_design=x)
        cn=f'fold_{fold}_calibration22.json';save(out/cn,{'selection':selection,'candidates':tables,'thresholds':all_thresholds})
        artifact_names.extend((fn,sn,cn));print('R22_FIT_CAL_FROZEN',fold,flush=True)
    assert snapshot()==snap
    for values in files.values():
        for f,h in values.items():assert sha(Path(f))==h
    artifact_names+=['source_snapshot22.json','feature_files22.json','lengths22.json','candidate_windows22.jsonl','answer_index22.jsonl','verifier_designs22.npz']
    save(out/'fit_freeze22.json',{'utc':r10.utc(),'fit_seconds':time.perf_counter()-started,'new_lr_fits':15,'status':'frozen_before_any_outer_metrics',
      'files_sha256':{n:sha(out/n) for n in artifact_names},'source_snapshot_sha256':sha(out/'source_snapshot22.json'),'original_validation_or_test_used':False})
    print('R22_FIT_COMPLETE',flush=True)


def test():
    out=ROOT/'results';assert (out/'fit_freeze22.json').exists() and not (out/'test_started22.json').exists()
    frozen_all=read(out/'fit_freeze22.json');config=read(ROOT/'protocol22.json');assert config==protocol()
    for name,h in frozen_all['files_sha256'].items():assert sha(out/name)==h
    snap=read(out/'source_snapshot22.json');assert snapshot()==snap
    pack,_,new,wanswer,files,lengths=prepare()
    save(out/'test_started22.json',{'utc':r10.utc(),'fit_freeze_sha256':sha(out/'fit_freeze22.json')})
    allw=[];alla=[];details={};candidate_metrics={};started=time.perf_counter()
    for fold in range(5):
        f=pickle.loads((out/f'fold_{fold}_frozen22.pkl').read_bytes())
        tr,tx=r18.subset(pack,f['fit_groups']);ca,cx=r18.subset(pack,f['calibration_groups']);ev,ex=r18.subset(pack,f['evaluation_groups'])
        with np.load(out/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        wr=[dict(w,scores={},predictions={},fold=fold) for w in ev['windows']];ar=None;detail={}
        for name in METHODS:
            v=values[name];ts=f['thresholds'][name]
            detail[name]={stage:r17.metrics(pp,v[ix],ts) for stage,pp,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]};detail[name]['thresholds']=ts
            ww,aa=r18.scored_records(ev,v[ex],ts,name)
            for a,b in zip(wr,ww):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
            if ar is None:ar=[dict(r,scores={},predictions={},fold=fold) for r in aa]
            for a,b in zip(ar,aa):a['scores'].update(b['scores']);a['predictions'].update(b['predictions'])
        oldw=readl(R21/'results'/f'fold_{fold}_window_scores.jsonl');olda=readl(R21/'results'/f'fold_{fold}_answer_scores.jsonl')
        for name in BASES:
            for current,prior in [(wr,oldw),(ar,olda)]:
                assert [r['scores'][name] for r in current]==[r['scores'][name] for r in prior]
                assert [r['predictions'][name] for r in current]==[r['predictions'][name] for r in prior]
        # Fixed outer diagnostics for every predeclared candidate, never reselect.
        rel={t:relative_slots(pack['windows'],values['slots_base'],t) for t in config['T']};candidate_metrics[str(fold)]={}
        for family,table in f['calibration_candidates'].items():
            rows=[]
            for entry in table:
                j=entry['candidate_index'];p=values['direct_answer'] if j is None else values[f'probe_C{j}_answer']
                v=p[wanswer] if family=='learned_broadcast' else p[wanswer]*rel[entry['T']]
                rows.append({'candidate':entry,'evaluation':r17.metrics(ev,v[ex],entry['thresholds'])})
            candidate_metrics[str(fold)][family]=rows
        details[str(fold)]=detail;allw.extend(wr);alla.extend(ar)
        save(out/f'fold_{fold}_metrics22.json',detail);savel(out/f'fold_{fold}_window_scores22.jsonl',wr);savel(out/f'fold_{fold}_answer_scores22.jsonl',ar)
        print('R22_EVALUATED',fold,flush=True)
    assert len(allw)==12222 and len(alla)==602 and len({w['window_key'] for w in allw})==12222
    summary={'scope':config['scope'],'coverage':pack['coverage'],'primary_method':config['primary_method'],'methods':r18.pooled(allw,alla,pack),
      'folds':details,'all_candidates_outer_diagnostics_not_selection':candidate_metrics,
      'fold_mean':{m:{stage:{k:float(np.mean([details[str(f)][m][stage]['windows'][k] for f in range(5)])) for k in ('f1','auroc','average_precision')} for stage in ('fit','calibration','evaluation')} for m in METHODS},
      'paired_bootstrap':{unit:r18.bootstrap(rr,config['bootstrap'],config['contrasts']) for unit,rr in [('windows',allw),('answers',alla)]},
      'strata':r19.strata(allw,alla,lengths),'evaluation_seconds':time.perf_counter()-started,'fit_seconds':frozen_all['fit_seconds'],
      'new_lr_fits':15,'extra_same_qwen_verification':True,'original_generation_state':False,'original_validation_or_test_used':False}
    old=read(R21/'results/summary.json')['methods']
    for name in BASES:assert summary['methods'][name]==old[name]
    assert snapshot()==snap
    for values in files.values():
        for path,h in values.items():assert sha(Path(path))==h
    savel(out/'window_scores_oof22.jsonl',allw);savel(out/'answer_scores_oof22.jsonl',alla);save(out/'summary22.json',summary)
    names=['summary22.json','window_scores_oof22.jsonl','answer_scores_oof22.jsonl','fit_freeze22.json']
    names+=[f'fold_{f}_{suffix}22.json'+('l' if 'scores' in suffix else '') for f in range(5) for suffix in ('metrics','window_scores','answer_scores')]
    save(out/'complete22.json',{'utc':r10.utc(),'files_sha256':{n:sha(out/n) for n in names},'original_validation_or_test_used':False,'model_or_threshold_retuned_on_outer':False})
    print('R22_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','self-test','prepare','fit','test']);args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='initialize':
            path=ROOT/'protocol22.json';assert not path.exists();save(path,protocol());print('R22_PROTOCOL_FROZEN')
        elif args.stage=='self-test':synthetic_checks();print('R22_SYNTHETIC_PASSED_NO_FIT')
        elif args.stage=='prepare':
            snapshot();pp,_,vv,_,_,_=prepare();print(pp['coverage'],{k:v.shape for k,v in vv.items()})
        elif args.stage=='fit':fit()
        else:test()
