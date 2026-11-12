"""Independent post-freeze R20 replay; no runtime evaluator or fitter imported.

Raw tree nodes are walked independently. Token/window/answer aggregation and
counts use earlier independent audit utilities. Only R16 train examples parsed.
"""
from pathlib import Path
import importlib.util, pickle, json
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
R19=ROOT.parent/'round19_three_signal_probe'
spec=importlib.util.spec_from_file_location('independent19',R19/'src/audit19.py')
a19=importlib.util.module_from_spec(spec);spec.loader.exec_module(a19)
p=a19.prior;u=a19.u
read,lines,sha,close=a19.read,a19.lines,a19.sha,a19.close
METHODS=('base','slots_base','token_lr_strong','token_lr','token_tree','window_tree','gated_unconditional','gated_conditional')
p.METHODS=METHODS;a19.METHODS=METHODS

def save(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')

def build():
    pack,_,_,lengths=a19.rebuild(features=False)
    items={a['row_id']:a for a in pack['items']}
    for token in pack['tokens']:
        item=items[token['row_id']]
        chars={j for j in range(token['start'],token['end']) if token['text'][j-token['start']].isalnum()}
        owner=item['start'] is not None and any(item['start']<=j<item['end'] for j in chars)
        token['item_ids']=[item['item_id']] if owner else []
    manifests={key:read(base/'data/feature_manifest.json') for key,base in [('r18',a19.R18),('r19',R19)]}
    files={'r18':{},'r19':{}};arrays={};full=[];small=[]
    for row in a19.train_rows(a19.NEW/'data/inputs.jsonl'):
        rid=row['row_id'];gp=a19.NEW/'data/generation_records'/(rid+'.json');g=read(gp)
        old=a19.load_arrays(a19.R18,manifests['r18'],rid,g,gp,{'lb':784,'nll':None,'new_features':16,'surface_features':8},files['r18'])
        new=a19.load_arrays(R19,manifests['r19'],rid,g,gp,{'token_lumina_mmd':2,'token_redeep_ecs':784,'token_redeep_pks':28},files['r19'])
        signals=a19.token_signals(new)
        extras=np.column_stack((old['nll'],signals['mmd'],signals['ecs'],signals['pks'],old['new_features'],old['surface_features']))
        for j in range(len(g['response_token_ids'])):
            full.append(np.concatenate((old['lb'][j],extras[j])))
            layer=old['lb'][j].reshape(28,28)
            reduced=np.concatenate([layer[b*7:(b+1)*7].mean(0) for b in range(4)])
            small.append(np.concatenate((reduced,extras[j])))
    full=np.asarray(full,np.float32);small=np.asarray(small,np.float32)
    ti={t['token_key']:i for i,t in enumerate(pack['tokens'])}
    def stats(x):return np.concatenate([x.mean(0),x.max(0),x.std(0)]).astype(np.float32)
    win_ix=[];window=[];slots=[];base=[]
    for w in pack['windows']:
        win_ix.append(np.asarray([ti[k] for k in w['token_keys']],int))
        raw=[ti[w['row_id']+f'__token{j}'] for j in w['raw_token_indices']]
        window.append(stats(small[raw]));original=full[raw,:785];n=len(raw)
        base.append(original.mean(0))
        if n<4:original=np.concatenate([original,np.repeat(original[-1:],4-n,axis=0)])
        slots.append(np.concatenate([original.reshape(-1),np.asarray([1]*n+[0]*(4-n),np.float32)]))
    answers=[]
    for a in pack['items']:
        ix=[i for i,t in enumerate(pack['tokens']) if t['lexical'] and t['item_ids']==[a['item_id']]]
        if not ix:ix=[i for i,t in enumerate(pack['tokens']) if t['row_id']==a['row_id'] and t['lexical']]
        assert ix;answers.append(stats(small[ix]))
    ai={a['item_id']:i for i,a in enumerate(pack['items'])}
    wanswer=np.asarray([ai[w['item_ids'][0]] for w in pack['windows']])
    designs={'full':full,'small':small,'window':np.asarray(window,np.float32),'answer':np.asarray(answers,np.float32),'slots':np.asarray(slots,np.float32)}
    with np.load(ROOT/'results/designs.npz',allow_pickle=False) as z:
        assert set(z.files)==set(designs)|{'win_answer'}
        for key,value in designs.items():assert np.array_equal(value,z[key]),('design',key,float(np.max(np.abs(value-z[key]))))
        assert np.array_equal(wanswer,z['win_answer'])
    exported=lines(ROOT/'results/token_index.jsonl')
    assert exported==[{k:t[k] for k in ('token_key','row_id','group_id','gold','main_eligible')} for t in pack['tokens']]
    assert pack['windows']==lines(ROOT/'results/candidate_windows.jsonl')
    assert files==read(ROOT/'results/feature_files.json')
    return pack,designs,win_ix,wanswer,np.asarray(base,np.float32),files,lengths

def predict(model,x):
    if model['scaler'] is not None:
        z=x.copy();z-=model['scaler'].mean_;z/=model['scaler'].scale_
        return expit(z.astype(np.float32)@model['model'].coef_[0]+model['model'].intercept_[0])
    # Independent numeric traversal, not HistGradientBoosting.predict_proba.
    c=model['model'];logits=np.full(len(x),float(c._baseline_prediction[0,0]))
    for trees in c._predictors:
        assert len(trees)==1
        nodes=trees[0].nodes;assert not nodes['is_categorical'].any()
        at=np.zeros(len(x),int);active=np.ones(len(x),bool)
        for depth in range(5):
            ii=np.flatnonzero(active);nn=at[ii];leaf=nodes['is_leaf'][nn].astype(bool)
            logits[ii[leaf]]+=nodes['value'][nn[leaf]];active[ii[leaf]]=False
            ii,nn=ii[~leaf],nn[~leaf]
            if not len(ii):break
            column=x[ii,nodes['feature_idx'][nn]]
            left=np.where(np.isnan(column),nodes['missing_go_to_left'][nn],column<=nodes['num_threshold'][nn])
            at[ii]=np.where(left,nodes['left'][nn],nodes['right'][nn])
        assert not active.any()
    return expit(logits)

def inspect(model,x,rows,ix,config,c):
    assert np.array_equal(model['fit_ix'],ix)
    rr=[rows[j] for j in ix]
    rr=[dict(r,item_ids=[r['item_id']]) if 'item_ids' not in r else r for r in rr]
    assert all(r['main_eligible'] for r in rr)
    y=np.asarray([r['gold'] for r in rr]);assert set(y)=={0,1} and np.array_equal(y,model['fit_y'])
    keys=[r.get('token_key',r.get('window_key',r.get('item_id'))) for r in rr]
    assert keys==model['fit_keys'] and model['fit_groups']==sorted({r['group_id'] for r in rr})
    b,loss,factors=u.weights(rr,3854)
    close(b,model['base_weights']);close(loss,model['loss_weights']);close(factors,model['class_factors'])
    close(loss.sum(),3854);assert model['loss_mass']==3854 and model['C']==c and model['width']==x.shape[1]
    classifier=model['model']
    if c is None:
        assert model['scaler'] is None
        for key,value in config['tree'].items():assert classifier.get_params()[key]==value
        assert classifier.n_iter_==100 and not classifier.do_early_stopping_
        # Constant initial logit uses only the weighted fitted class rate.
        rate=np.average(y,weights=loss)
        close(classifier._baseline_prediction[0,0],np.log(rate/(1-rate)))
        assert all(int(tree[0].nodes['count'][0])==len(ix) for tree in classifier._predictors)
    else:
        assert classifier.C==c and classifier.random_state==config['seed'] and classifier.solver=='liblinear'
        assert classifier.max_iter==2000 and max(classifier.n_iter_)<2000
        raw=x[ix];weight=b.astype(raw.dtype).astype(float)
        mean=np.average(raw.astype(float),axis=0,weights=weight)
        var=np.average((raw.astype(float)-mean)**2,axis=0,weights=weight)
        close(mean,model['scaler'].mean_,atol=2e-8);close(var,model['scaler'].var_,atol=2e-8)
        eps=np.finfo(float).eps;constant=var<=weight.sum()*eps*var+(weight.sum()*mean*eps)**2
        scale=np.sqrt(var);scale[constant]=1
        close(scale,model['scaler'].scale_,atol=2e-8);close(weight.sum(),model['scaler'].n_samples_seen_)
    assert classifier.classes_.tolist()==[0,1]
    return {'fit_rows':len(ix),'fit_groups':len(model['fit_groups']),'width':x.shape[1],
            'fit_only_indices_weights_preprocessing_verified':True,'C':c,'loss_mass':float(loss.sum())}

def sources():
    a19.verify_sources()
    out=ROOT/'results';complete=read(out/'complete20.json');snap=read(out/'source_snapshot20.json')
    assert complete['original_validation_or_test_used'] is False
    u.verify_tree(out,complete['files_sha256']);u.verify_tree(Path('.'),snap['local_sha256'])
    assert snap['source']==read(R19/'results/source_snapshot19.json')
    assert snap['r19_complete_sha256']==sha(R19/'results/complete19.json')
    return complete,snap

def run():
    assert (ROOT/'results/complete20.json').exists()
    out=ROOT/'results';complete,snap=sources();config=read(ROOT/'protocol.json')
    assert tuple(config['methods'])==METHODS and config['primary_method']=='gated_conditional'
    pack,d,win_ix,wanswer,base,files,lengths=build()
    summary=read(out/'summary.json');u.assert_metrics(u.coverage(pack),summary['coverage'],'coverage')
    assignment=read(R19/'data/fold_assignment.json')['groups']
    tg=np.asarray([r['group_id'] for r in pack['tokens']]);wg=np.asarray([r['group_id'] for r in pack['windows']]);ag=np.asarray([r['group_id'] for r in pack['items']])
    te=np.asarray([r['main_eligible'] for r in pack['tokens']]);we=np.asarray([r['main_eligible'] for r in pack['windows']]);ae=np.asarray([r['main_eligible'] for r in pack['items']])
    audit={'status':'passed','new_components':30,'reported_configurations':8,'folds':{},'original_validation_or_test_parsed':False,'refitted':False}
    allw=[];alla=[];observed=[]
    for fold in range(5):
        f=pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes());detail=read(out/f'fold_{fold}_metrics.json')
        assert detail==summary['folds'][str(fold)]
        eg=sorted(g for g,i in assignment.items() if i==fold);cg=sorted(g for g,i in assignment.items() if i==(fold+1)%5)
        fg=sorted(set(assignment)-set(eg)-set(cg))
        assert not(set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        for k,v in [('fit_groups',fg),('calibration_groups',cg),('evaluation_groups',eg)]:assert f[k]==v
        ti=np.flatnonzero(np.isin(tg,fg)&te);wi=np.flatnonzero(np.isin(wg,fg)&we);ai=np.flatnonzero(np.isin(ag,fg)&ae)
        # Construct eligible risk answers from FIT rows only, unlike a global mask.
        risky_fit={a['item_id'] for a in pack['items'] if a['group_id'] in fg and a['main_eligible'] and a['gold']==1}
        ci=np.asarray([j for j,t in enumerate(pack['tokens']) if t['main_eligible'] and t['item_ids'] and t['item_ids'][0] in risky_fit])
        tr,tx=p.subset(pack,fg);ca,cx=p.subset(pack,cg);ev,ex=p.subset(pack,eg)
        wr=lines(out/f'fold_{fold}_window_scores.jsonl');ar=lines(out/f'fold_{fold}_answer_scores.jsonl')
        assert all(r['fold']==fold for r in wr+ar)
        components={};checks={}
        for name,key,rows,ix,c in [('token_lr_strong','full',pack['tokens'],ti,.001),('token_lr','full',pack['tokens'],ti,.01),
            ('token_tree','small',pack['tokens'],ti,None),('window_tree','window',pack['windows'],wi,None),
            ('answer_gate','answer',pack['items'],ai,None),('conditional','full',pack['tokens'],ci,.001)]:
            m=f['parts'][name];assert m['design']==key
            checks[name]=inspect(m,d[key],rows,ix,config,c)
            values=predict(m,d[key]);assert np.isfinite(values).all()
            # Cross-check our independent arithmetic against the saved estimator.
            z=m['scaler'].transform(d[key]).astype(np.float32) if m['scaler'] is not None else d[key]
            native=m['model'].predict_proba(z)[:,1]
            assert np.array_equal(values,native),(fold,name,float(np.max(np.abs(values-native))))
            components[name]=values
        values={};basecheck={}
        for name,key,directory,oldkey,x in [('base','baseline',R19,'base',base),('slots_base','slots_baseline',a19.R18,'slots_lr',d['slots'])]:
            old=pickle.loads((directory/'results'/f'fold_{fold}_frozen.pkl').read_bytes())['models'][oldkey]
            m=f[key]
            for field in ('coef_','intercept_','classes_','n_iter_'):assert np.array_equal(getattr(m['model'],field),getattr(old['model'],field))
            for field in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(m['scaler'],field),getattr(old['scaler'],field))
            for field in ('fit_ix','fit_y','base_weights','loss_weights','class_factors'):assert np.array_equal(m[field],old[field])
            values[name]=predict(m,x);assert np.array_equal(values[name],predict(old,x))
            assert f['thresholds'][name]==old['thresholds'];basecheck[name]={'exact':True,'prior_fold_sha256':sha(directory/'results'/f'fold_{fold}_frozen.pkl')}
        for name in ('token_lr_strong','token_lr','token_tree'):values[name]=np.asarray([components[name][ix].max() for ix in win_ix])
        values['window_tree']=components['window_tree']
        gate=components['answer_gate'][wanswer]
        values['gated_unconditional']=values['token_lr_strong']*gate
        values['gated_conditional']=np.asarray([components['conditional'][ix].max() for ix in win_ix])*gate
        score_errors={}
        for name in METHODS:
            v=values[name];ts=f['thresholds'][name];assert ts==detail[name]['thresholds']
            for unit,rr,ss in [('window',ca['windows'],v[cx]),('answer',ca['items'],u.item_max(ca,v[cx]))]:
                ii=[j for j,r in enumerate(rr) if r['main_eligible']]
                u.assert_metrics(u.threshold([rr[j]['gold'] for j in ii],ss[ii]),ts[unit],f'{fold}.{name}.{unit}.cal threshold')
            for stage,pp,ix in [('fit',tr,tx),('calibration',ca,cx),('evaluation',ev,ex)]:
                expected=u.measures(pp,v[ix],ts)[0]
                for unit in expected:u.assert_metrics(expected[unit],detail[name][stage][unit],f'{fold}.{name}.{stage}.{unit}')
            score_errors[name]=p.verify_saved(ev,v[ex],ts,wr,ar,name)
            assert np.array_equal(v[ex],np.asarray([w['scores'][name] for w in wr]))
        for name,directory,oldname in [('base',R19,'base'),('slots_base',a19.R18,'slots_lr')]:
            for current,kind,idkey in [(wr,'window','window_key'),(ar,'answer','item_id')]:
                old=lines(directory/'results'/f'fold_{fold}_{kind}_scores.jsonl')
                assert [r[idkey] for r in current]==[r[idkey] for r in old]
                assert [r['scores'][name] for r in current]==[r['scores'][oldname] for r in old]
                assert [r['predictions'][name] for r in current]==[r['predictions'][oldname] for r in old]
        audit['folds'][str(fold)]={'components':checks,'baseline_exact_reproduction':basecheck,'maximum_saved_probability_difference':score_errors,
          'conditional_training_only_risky_fit_answers':len(risky_fit),'conditional_eval_all_output_tokens':len(pack['tokens'])}
        allw.extend(wr);alla.extend(ar);observed.extend(eg)
        print('AUDITED20_FOLD',fold,flush=True)
    assert len(observed)==len(set(observed))==278
    assert allw==lines(out/'window_scores_oof.jsonl') and alla==lines(out/'answer_scores_oof.jsonl')
    rebuilt=p.pooled(pack,allw,alla)
    for name in METHODS:
        actual=summary['methods'][name]
        for unit in ('windows','answers','highlight_tokens'):u.assert_metrics(rebuilt[name][unit],actual[unit],name+'.'+unit)
        for key in ('safe_refusals','safe_refusal_false_positives'):assert rebuilt[name][key]==actual[key]
        expected,got=rebuilt[name]['conditional_ranking'],actual['conditional_ranking']
        for key in ('answers_with_both_labels','peak_hit_answers','mean_auroc','mean_average_precision'):close(expected[key],got[key])
        assert {r['item_id']:r for r in expected['details']}=={r['item_id']:r for r in got['details']}
        for stage in ('fit','calibration','evaluation'):
            for metric in ('f1','auroc','average_precision'):
                close(np.mean([summary['folds'][str(f)][name][stage]['windows'][metric] for f in range(5)]),summary['fold_mean'][name][stage][metric])
    audit['pooled']=rebuilt
    audit['paired_bootstrap']={unit:p.bootstrap(rr,config['bootstrap'],config['contrasts'],summary['paired_bootstrap'][unit])
                             for unit,rr in [('windows',allw),('answers',alla)]}
    audit['strata']=a19.strata(allw,alla,lengths,summary['strata'])
    sources()
    for hashes in files.values():u.verify_tree(Path('.'),hashes)
    audit.update(auditor_sha256=sha(__file__),complete_sha256=sha(out/'complete20.json'),
       independent_utilities_sha256={str(R19/'src/audit19.py'):sha(R19/'src/audit19.py'),str(a19.UTILITY):sha(a19.UTILITY),str(p.UTILITY):sha(p.UTILITY)},
       limitations=['Class-weighted gate/local products are combination scores, not demonstrated calibrated probabilities.',
         'New structures also add old alignment/text features; improvement cannot be attributed to structure alone.',
         'Full-response gate is post-generation; conditional gold is used only inside fit.',
         'Repeated training-source exploration, assistant gold, fixed-model bootstrap; no fresh confirmatory test.'])
    save(out/'INDEPENDENT_AUDIT20.json',audit);print('INDEPENDENT_AUDIT20_PASSED',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
