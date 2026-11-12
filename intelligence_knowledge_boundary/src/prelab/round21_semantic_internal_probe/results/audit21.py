"""Independent R21 frozen-artifact replay. No classifier/PCA fit or GPU calls.

Only actual R16 train rows are parsed. Reuses independent audit geometry/count
utilities, never the production baseline prediction or evaluator functions.
"""
from pathlib import Path
from collections import defaultdict
import importlib.util, pickle, json
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
UTILITY=ROOT.parent/'round20_localization_optimization/results/audit20.py'
spec=importlib.util.spec_from_file_location('independent20_for21',UTILITY)
a20=importlib.util.module_from_spec(spec);spec.loader.exec_module(a20)
a19,p,u=a20.a19,a20.p,a20.u
read,lines,sha,close=a20.read,a20.lines,a20.sha,a20.close
R19,R18,NEW=a19.ROOT,a19.R18,a19.NEW
METHODS=('base','slots_base','r19_all','lookback_tuned','redeep_tuned','harp256','hidden256','delta256','base_harp','base_delta','base_harp_delta')
p.METHODS=METHODS;a19.METHODS=METHODS


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')


def sources():
    a19.verify_sources()
    out=ROOT/'results';complete=read(out/'complete21.json');snap=read(out/'source_snapshot21.json')
    assert complete['original_validation_or_test_used'] is False
    u.verify_tree(out,complete['files_sha256']);u.verify_tree(Path('.'),snap['code_sha256'])
    assert snap['r19_source']==read(R19/'results/source_snapshot19.json')
    assert sha(ROOT/'data/feature_manifest.json')==snap['feature_manifest_sha256']
    assert read(out/'started21.json')['source_snapshot_sha256']==sha(out/'source_snapshot21.json')
    fm=read(ROOT/'data/feature_manifest.json')
    assert fm['complete'] and fm['completed_count']==602 and u.digest(fm['extraction_signature'])==fm['extraction_signature_sha256']
    u.verify_tree(Path('.'),fm['extraction_signature']['code_sha256'])
    return complete,snap


def build():
    pack,_,_,lengths=a19.rebuild(features=False)
    manifests={name:read(base/'data/feature_manifest.json') for name,base in [('r18',R18),('r19',R19),('r21',ROOT)]}
    files={name:{} for name in manifests};arrays={};design={k:[] for k in ('harp','hidden','delta','ecs','pks','slots','base','r19_all')}
    basis_path=ROOT/'data/harp_basis.npz';basis_meta=read(basis_path.with_suffix('.json'))
    assert sha(basis_path)==basis_meta['arrays_sha256'] and basis_meta['static_model_parameter_basis'] and not basis_meta['fitted_on_labels_or_examples']
    with np.load(basis_path,allow_pickle=False) as z:basis=z['components'].copy()
    assert basis.shape==(256,3584);close(basis@basis.T,np.eye(256),atol=2e-8)
    rows=a19.train_rows(NEW/'data/inputs.jsonl')
    by_row=defaultdict(list)
    for w in pack['windows']:by_row[w['row_id']].append(w)
    token_count=0;harp_max_delta=0.
    for row in rows:
        rid=row['row_id'];gp=NEW/'data/generation_records'/(rid+'.json');g=read(gp)
        old=a19.load_arrays(R18,manifests['r18'],rid,g,gp,{'lb':784,'nll':None,'hidden_28':3584},files['r18'])
        other=a19.load_arrays(R19,manifests['r19'],rid,g,gp,{'token_lumina_mmd':2,'token_redeep_ecs':784,'token_redeep_pks':28},files['r19'])
        new=a19.load_arrays(ROOT,manifests['r21'],rid,g,gp,{'harp_256':256,'hidden_delta_28':3584,'hidden_noctx_28':3584},files['r21'])
        side=read(ROOT/manifests['r21']['records'][rid]['json']);plan=side['intervention_plan']
        assert plan['actual_split']=='train' and plan['original_prompt']==row['prompt']
        marker='\n\nSearch results:\n';assert plan['noctx_prompt']==row['prompt'].split(marker)[0]+marker
        assert plan['response_ids_sha256']==u.digest(g['response_token_ids']) and plan['original_prefix_token_ids']==g['input_token_ids']
        assert np.array_equal(new['hidden_delta_28'],old['hidden_28']-new['hidden_noctx_28'])
        # The basis is model-only; independently apply its stored matrix.
        hh=(old['hidden_28'].astype(np.float64)@basis.T).astype(np.float32)
        hd=float(np.max(np.abs(hh-new['harp_256'])));harp_max_delta=max(harp_max_delta,hd)
        close(hh,new['harp_256'],atol=2e-7,rtol=2e-7)
        signals=a19.token_signals(other);base=np.column_stack((old['lb'],old['nll']))
        token_count+=len(g['response_token_ids'])
        for w in by_row[rid]:
            ix=w['raw_token_indices'];count=len(ix)
            for key,arr in [('harp',new['harp_256']),('hidden',old['hidden_28']),('delta',new['hidden_delta_28']),('ecs',other['token_redeep_ecs']),('pks',other['token_redeep_pks'])]:design[key].append(arr[ix].mean(0))
            design['base'].append(base[ix].mean(0))
            design['r19_all'].append(np.concatenate((base[ix].mean(0),*[signals[k][ix].mean(0) for k in ('mmd','ecs','pks')])))
            slots=base[ix]
            if count<4:slots=np.concatenate((slots,np.repeat(slots[-1:],4-count,axis=0)))
            design['slots'].append(np.concatenate((slots.ravel(),np.asarray([1]*count+[0]*(4-count),np.float32))))
    d={k:np.asarray(v,np.float32) for k,v in design.items()}
    with np.load(ROOT/'results/designs.npz',allow_pickle=False) as z:
        assert set(z.files)==set(d)
        for k,v in d.items():assert np.array_equal(v,z[k]),('design',k,float(np.max(np.abs(v-z[k]))))
    files['r18_extra']=files['r18'].copy()
    assert files==read(ROOT/'results/feature_files.json')
    assert pack['windows']==lines(ROOT/'results/candidate_windows.jsonl')
    return pack,d,files,lengths,{'records':len(rows),'tokens':token_count,'all_window_designs_exact':True,'hidden_delta_exact':True,'harp_projection_max_float32_difference':harp_max_delta,'basis_arrays_sha256':sha(basis_path)}


def weighted_auc_pairwise(x,y,weight):
    """Weighted Mann-Whitney arithmetic with half credit to equal-value pairs."""
    out=[];y=np.asarray(y,int);weight=np.asarray(weight,float)
    for j in range(x.shape[1]):
        order=np.argsort(x[:,j],kind='stable');v=x[order,j];yy=y[order];w=weight[order]
        starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
        pos=np.add.reduceat(w*yy,starts);neg=np.add.reduceat(w*(1-yy),starts)
        prior_neg=np.r_[0.,np.cumsum(neg)[:-1]]
        out.append(float(np.dot(pos,prior_neg+.5*neg)/(pos.sum()*neg.sum())))
    return np.asarray(out)


def inspect_common(candidate,rows):
    y=np.asarray([r['gold'] for r in rows],int);b,w,f=u.weights(rows,3854.)
    assert candidate['fit_keys']==[r['window_key'] for r in rows]
    assert candidate['fit_groups']==sorted({r['group_id'] for r in rows}) and np.array_equal(candidate['fit_labels'],y)
    assert candidate['eval_labels_seen'] is False
    for key,actual in [('base_weights',b),('loss_weights',w),('class_factors',f)]:close(candidate[key],actual)
    close(candidate['loss_mass'],3854.)
    assert candidate['config']['seed']==20260919 and candidate['config']['loss_mass']==3854
    return y,b,w,f


def inspect_lb(candidate,x,ix,rows,config):
    inspect_common(candidate,rows)
    # Older helper stores nominal mass; baseline records the floating-point sum.
    # inspect_common already checked the actual mass to numeric precision.
    proxy=dict(candidate,fit_ix=ix,fit_y=candidate['fit_labels'],width=784,loss_mass=candidate['config']['loss_mass'])
    return a20.inspect(proxy,x,[dict(r) for r in ALL_WINDOWS],ix,config,candidate['C'])


def redeep_scores(candidates,d,ix,rows):
    assert len(candidates)==27
    y,b,_,_=inspect_common(candidates[0],rows)
    ea=weighted_auc_pairwise(d['ecs'][ix],1-y,b);pa=weighted_auc_pairwise(d['pks'][ix],y,b)
    first=candidates[0];close(ea,first['ecs_fit_auc'],atol=1e-12);close(pa,first['pks_fit_auc'],atol=1e-12)
    # Numerical integration and pairwise sums can differ at machine epsilon.
    # Enforce the frozen code's exact sort on its independently verified AUCs.
    er=np.lexsort((np.arange(784),-first['ecs_fit_auc']));pr=np.lexsort((np.arange(28),-first['pks_fit_auc']))
    assert np.array_equal(er,first['ecs_order']) and np.array_equal(pr,first['pks_order'])
    grid={(h,l,beta) for h in (1,4,16) for l in (4,14,28) for beta in (.2,.6,1.)}
    assert {(c['Kh'],c['Kl'],c['beta']) for c in candidates}==grid
    assert [c['complexity'] for c in candidates]==sorted(c['complexity'] for c in candidates)
    scores=[]
    for c in candidates:
        inspect_common(c,rows)
        for k in ('ecs_fit_auc','pks_fit_auc','ecs_order','pks_order'):assert np.array_equal(c[k],first[k])
        h,l=c['Kh'],c['Kl'];assert np.array_equal(c['heads'],er[:h]) and np.array_equal(c['layers'],pr[:l])
        assert c['complexity']==[h+l,h,l,c['beta']] and c['alpha']==1 and c['minmax_clipped'] is False
        assert c['selected_head_coordinates']==[{'layer_1based':int(j//28+1),'head_0based':int(j%28)} for j in er[:h]]
        assert c['selected_layer_numbers_1based']==(pr[:l]+1).tolist()
        e=d['ecs'][:,er[:h]].astype(np.float64).sum(1);pks=d['pks'][:,pr[:l]].astype(np.float64).sum(1)
        v=np.column_stack((pks,e));low=v[ix].min(0);high=v[ix].max(0);scale=high-low;scale[scale==0]=1
        for k,vv in [('minimum',low),('maximum',high),('scale',scale)]:assert np.array_equal(c[k],vv)
        norm=(v-low)/scale;scores.append(norm[:,0]-c['beta']*norm[:,1])
    return scores,{'candidates':27,'full_heads':784,'full_layers':28,'fit_only_weighted_auc_direction_verified':True,'maximum_auc_arithmetic_difference':float(max(np.max(np.abs(ea-first['ecs_fit_auc'])),np.max(np.abs(pa-first['pks_fit_auc'])))),'minmax_fit_only_no_eval_clipping':True}


def calibrate(pack,v):
    av=u.item_max(pack,v);ts={}
    for kind,rr,ss in [('window',pack['windows'],v),('answer',pack['items'],av)]:
        ii=[j for j,r in enumerate(rr) if r['main_eligible']]
        ts[kind]=u.threshold([rr[j]['gold'] for j in ii],ss[ii])
    return ts


def selection_key(ts,complexity):
    w,a=ts['window'],ts['answer']
    return (min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],*[-float(c) for c in complexity])


def select_check(wrapper,candidates,scores,cal,cx,lr=False):
    entries=wrapper['calibration_candidates'];assert len(entries)==len(candidates)
    keys=[]
    for j,(c,v,entry) in enumerate(zip(candidates,scores,entries)):
        ts=calibrate(cal,v[cx]);complexity=[c['C']] if lr else c['complexity']
        for k in ts:u.assert_metrics(ts[k],entry['thresholds'][k],f'candidate{j}.{k}')
        keys.append(selection_key(ts,complexity));close(keys[-1],entry['selection_key'])
        if lr:assert entry['C']==c['C']
        else:assert entry['candidate_index']==j and entry['candidate_id']==c['candidate_id'] and entry['complexity']==complexity
    chosen=max(range(len(keys)),key=lambda j:keys[j]);assert chosen==wrapper['selected_candidate']
    selected=wrapper if lr else wrapper['candidate'];actual=candidates[chosen]
    if lr or actual['kind']=='lookback_lr':
        for k in ('coef_','intercept_','classes_'):assert np.array_equal(getattr(selected['model'],k),getattr(actual['model'],k))
        for k in ('mean_','var_','scale_'):assert np.array_equal(getattr(selected['scaler'],k),getattr(actual['scaler'],k))
    else:assert selected['candidate_id']==actual['candidate_id']
    return scores[chosen],entries[chosen]['thresholds'],{'selected_candidate':chosen,'candidate_count':len(candidates),'all_calibration_thresholds_and_min_dual_F1_selection_verified':True}


def project(saved,x,ix,rows,fold):
    assert np.array_equal(saved['fit_ix'],ix) and saved['fit_keys']==[r['window_key'] for r in rows]
    b,_,_=u.weights(rows,3854.);close(saved['base_weights'],b/b.sum())
    assert saved['seed']==20260919+fold and saved['n_iter']==3
    w=b/b.sum();raw=x[ix].astype(np.float64);center=w@raw;close(saved['mean'],center,atol=2e-10)
    c=saved['components'];assert c.shape==(256,3584);close(c@c.T,np.eye(256),atol=2e-8)
    singular=saved['singular_values'];assert (singular>=0).all() and np.all(np.diff(singular)<=0)
    total=float(np.einsum('ij,i,ij->',raw-center,w,raw-center));close(np.sum(singular**2)/total,saved['explained_variance_ratio_sum'])
    z=((x.astype(np.float64)-center)@c.T).astype(np.float32)
    projected=(raw-center)@c.T;observed=np.sum(projected**2*w[:,None],axis=0)
    # Randomized SVD is approximate; it cannot claim exact covariance eigenvalues.
    rel=float(np.max(np.abs(observed-singular**2)/np.maximum(observed,1e-30)))
    assert rel<.15,('PCA projected energy incompatible',fold,rel)
    assert float(observed.sum())<=total*(1+1e-10)
    return z,{'fit_groups_only':True,'dimensions':256,'orthonormal':True,'saved_trace_ratio_verified':True,'randomized_projection_energy_max_relative_difference':rel,'stored_base_weights_are_sum1_pca_weights_not_mean1_lr_weights':True,'refitted':False}


def run():
    global ALL_WINDOWS
    out=ROOT/'results';assert (out/'complete21.json').exists()
    sources();config=read(ROOT/'protocol.json');assert tuple(config['methods'])==METHODS
    pack,d,files,lengths,feature_check=build();ALL_WINDOWS=pack['windows']
    summary=read(out/'summary.json');u.assert_metrics(u.coverage(pack),summary['coverage'],'coverage')
    assignments=read(R19/'data/fold_assignment.json')['groups']
    wg=np.asarray([r['group_id'] for r in pack['windows']]);eligible=np.asarray([r['main_eligible'] for r in pack['windows']])
    audit={'status':'passed','refitted':False,'production_prediction_or_evaluator_imported':False,'original_validation_or_test_parsed':False,'features':feature_check,'folds':{}}
    allw=[];alla=[];seen=[]
    for fold in range(5):
        frozen=pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes());detail=read(out/f'fold_{fold}_metrics.json')
        assert detail==summary['folds'][str(fold)]
        eg=sorted(g for g,f in assignments.items() if f==fold);cg=sorted(g for g,f in assignments.items() if f==(fold+1)%5);fg=sorted(set(assignments)-set(eg)-set(cg))
        assert not(set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        for k,group in [('fit_groups',fg),('calibration_groups',cg),('evaluation_groups',eg)]:assert frozen[k]==group
        tr,tx=p.subset(pack,fg);ca,cx=p.subset(pack,cg);ev,ex=p.subset(pack,eg)
        ix=np.flatnonzero(np.isin(wg,fg)&eligible);rows=[pack['windows'][i] for i in ix]
        models=frozen['models'];values={};checks={};projections={}
        wr=lines(out/f'fold_{fold}_window_scores.jsonl');ar=lines(out/f'fold_{fold}_answer_scores.jsonl')
        assert all(r['fold']==fold for r in wr+ar)
        for name,base,oldkey,key in [('base',R19,'base','base'),('slots_base',R18,'slots_lr','slots'),('r19_all',R19,'base_all','r19_all')]:
            old=pickle.loads((base/'results'/f'fold_{fold}_frozen.pkl').read_bytes())['models'][oldkey];current=models[name]
            for k in ('coef_','intercept_','classes_','n_iter_'):assert np.array_equal(getattr(current['model'],k),getattr(old['model'],k))
            for k in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(current['scaler'],k),getattr(old['scaler'],k))
            for k in ('fit_ix','fit_y','base_weights','loss_weights','class_factors'):assert np.array_equal(current[k],old[k])
            values[name]=a19.predict(current,d[key]);assert np.array_equal(values[name],a19.predict(old,d[key]))
            assert frozen['thresholds'][name]==old['thresholds'];checks[name]={'exact_prior_model_and_thresholds':True}
            for rr,kind in [(wr,'window'),(ar,'answer')]:
                prior=lines(base/'results'/f'fold_{fold}_{kind}_scores.jsonl')
                assert [r['scores'][name] for r in rr]==[r['scores'][oldkey] for r in prior]
                assert [r['predictions'][name] for r in rr]==[r['predictions'][oldkey] for r in prior]
        wrapper=models['lookback_tuned'];candidates=wrapper['all_candidates'];assert [c['C'] for c in candidates]==[.001,.01,.1]
        scores=[]
        for c in candidates:inspect_lb(c,d['base'][:,:784],ix,rows,config);scores.append(a19.predict(c,d['base'][:,:784]))
        values['lookback_tuned'],ts,checks['lookback_tuned']=select_check(wrapper,candidates,scores,ca,cx)
        assert ts==frozen['thresholds']['lookback_tuned']
        wrapper=models['redeep_tuned'];candidates=wrapper['all_candidates'];scores,checks['redeep_formula']=redeep_scores(candidates,d,ix,rows)
        values['redeep_tuned'],ts,checks['redeep_tuned']=select_check(wrapper,candidates,scores,ca,cx)
        assert ts==frozen['thresholds']['redeep_tuned']
        projected={}
        for k in ('hidden','delta'):projected[k],projections[k]=project(frozen['projections'][k],d[k],ix,rows,fold)
        new={'harp256':d['harp'],'hidden256':projected['hidden'],'delta256':projected['delta'],'base_harp':np.column_stack((d['base'],d['harp'])),'base_delta':np.column_stack((d['base'],projected['delta'])),'base_harp_delta':np.column_stack((d['base'],d['harp'],projected['delta']))}
        for name,x in new.items():
            wrapper=models[name];candidates=wrapper['all_lr_candidates'];assert [c['C'] for c in candidates]==[.001,.01,.1]
            scores=[]
            for c in candidates:a20.inspect(c,x,pack['windows'],ix,config,c['C']);scores.append(a19.predict(c,x))
            values[name],ts,checks[name]=select_check(wrapper,candidates,scores,ca,cx,lr=True)
            assert ts==frozen['thresholds'][name]
        differences={}
        for name in METHODS:
            v=values[name];ts=frozen['thresholds'][name];assert ts==detail[name]['thresholds']
            independent=calibrate(ca,v[cx])
            for k in ts:u.assert_metrics(independent[k],ts[k],f'{fold}.{name}.{k}.threshold')
            for stage,pp,jj in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]:
                expected=u.measures(pp,v[jj],ts)[0]
                for unit in expected:u.assert_metrics(expected[unit],detail[name][stage][unit],f'{fold}.{name}.{stage}.{unit}')
            differences[name]=p.verify_saved(ev,v[ex],ts,wr,ar,name)
            assert np.array_equal(v[ex],np.asarray([w['scores'][name] for w in wr])),(fold,name,differences[name])
        audit['folds'][str(fold)]={'models':checks,'projections':projections,'maximum_saved_score_difference':differences,'fit_windows':len(ix),'fit_groups':len(fg),'calibration_groups':len(cg),'evaluation_groups':len(eg)}
        allw.extend(wr);alla.extend(ar);seen.extend(eg)
        print('AUDITED21_FOLD',fold,flush=True)
    assert len(seen)==len(set(seen))==278
    assert allw==lines(out/'window_scores_oof.jsonl') and alla==lines(out/'answer_scores_oof.jsonl')
    rebuilt=p.pooled(pack,allw,alla)
    for name in METHODS:
        actual=summary['methods'][name]
        for unit in ('windows','answers','highlight_tokens'):u.assert_metrics(rebuilt[name][unit],actual[unit],name+'.'+unit)
        for key in ('safe_refusals','safe_refusal_false_positives'):assert rebuilt[name][key]==actual[key]
        ex,ac=rebuilt[name]['conditional_ranking'],actual['conditional_ranking']
        for key in ('answers_with_both_labels','peak_hit_answers','mean_auroc','mean_average_precision'):close(ex[key],ac[key])
        assert {r['item_id']:r for r in ex['details']}=={r['item_id']:r for r in ac['details']}
        for stage in ('fit','calibration','evaluation'):
            for metric in ('f1','auroc','average_precision'):close(np.mean([summary['folds'][str(f)][name][stage]['windows'][metric] for f in range(5)]),summary['fold_mean'][name][stage][metric])
    audit['pooled']=rebuilt
    audit['paired_bootstrap']={unit:p.bootstrap(rr,config['bootstrap'],config['contrasts'],summary['paired_bootstrap'][unit]) for unit,rr in [('windows',allw),('answers',alla)]}
    audit['strata']=a19.strata(allw,alla,lengths,summary['strata'])
    sources()
    for hashes in files.values():u.verify_tree(Path('.'),hashes)
    audit.update(coverage=u.coverage(pack),auditor_sha256=sha(__file__),complete_sha256=sha(out/'complete21.json'),
      fitted_lr_candidates=105,redeep_formula_candidates=135,reported_configurations=11,
      limitations=['Repeated training-source CV is exploratory; no untouched confirmation set.',
      'ReDeEP is full-head signal/local Qwen JSD-window adaptation, not exact released Llama implementation.',
      'LUMINA has no full IPR cache and is not represented by MMD alone.',
      'Window or peak overlap is not complete semantic or exact-token localization.',
      'Randomized PCA projection is checked against saved fit statistics/energy, not re-estimated.'])
    save(out/'INDEPENDENT_AUDIT21.json',audit)
    print('INDEPENDENT_AUDIT21_PASSED',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
