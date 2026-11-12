"""Bounded CPU baseline builders with separate fit/calibration/inference APIs.

No files, evaluation labels, GPU, automatic experiment or source mutations.
Inputs are already aggregated ORIGINAL four-BPE windows. Scores are risks, not
necessarily probabilities. See BASELINES.md for official sources/adaptations.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import hashlib
import json
from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION='r21-local-baselines-v1'
LB_CS=(.001,.01,.1)
REDEEP_HEADS=(1,4,16)
REDEEP_LAYERS=(4,14,28)
REDEEP_BETAS=(.2,.6,1.)
DEFAULT_CONFIG={'seed':20260913,'loss_mass':3854.,'max_iter':2000,'threads':4}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def matrix(value,width,name):
    x=np.asarray(value)
    if x.ndim!=2 or x.shape[1]!=width or not np.isfinite(x).all():
        raise ValueError(f'{name}: expected finite N x {width}')
    return x


def item_id(row):
    if 'item_ids' in row:
        if len(row['item_ids'])!=1:raise ValueError('Exactly one item per baseline window required')
        return row['item_ids'][0]
    return row['item_id']


def key(row):return row.get('window_key',row.get('token_key',row.get('item_id')))


def fit_weights(rows,loss_mass=3854.):
    """Equal group / condition / answer / window base weight; fit class weights."""
    if not rows or not all(r['main_eligible'] and r['gold'] in (0,1) for r in rows):
        raise ValueError('Fit accepts resolved labeled rows only')
    if len({key(r) for r in rows})!=len(rows):raise ValueError('Duplicate fit keys')
    y=np.asarray([r['gold'] for r in rows],int)
    if set(y)!={0,1}:raise ValueError('Both fit classes required')
    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for j,row in enumerate(rows):grouped[row['group_id']][row['condition']][item_id(row)].append(j)
    base=np.empty(len(rows),float)
    for conditions in grouped.values():
        for answers in conditions.values():
            for indices in answers.values():base[indices]=1/(len(conditions)*len(answers)*len(indices))
    base/=base.mean()
    factors=base.sum()/(2*np.asarray([base[y==c].sum() for c in (0,1)]))
    loss=base*factors[y]
    for conditions in grouped.values():
        indices=[j for answers in conditions.values() for indices in answers.values() for j in indices]
        loss[indices]*=(len(rows)/len(grouped))/loss[indices].sum()
    loss*=loss_mass/loss.sum()
    return base,loss,factors


def _fit_record(rows,base,loss,factors,config):
    return {'version':VERSION,'fit_keys':[key(r) for r in rows],
       'fit_groups':sorted({r['group_id'] for r in rows}),
       'fit_labels':np.asarray([r['gold'] for r in rows],int),
       'base_weights':base,'loss_weights':loss,'class_factors':factors,
       'fit_rows_sha256':digest([{k:r[k] for k in ('group_id','condition','gold')}|{'key':key(r),'item_id':item_id(r)} for r in rows]),
       'config':config.copy(),'loss_mass':float(loss.sum()),'eval_labels_seen':False}


def fit_lookback_candidates(fit_lb,fit_rows,config=None):
    """Fit 3 pure-LB classifiers; no NLL or extra signal is accepted here."""
    cfg=DEFAULT_CONFIG|dict(config or {})
    x=matrix(fit_lb,784,'fit_lb')
    if len(x)!=len(fit_rows):raise ValueError('Feature/fit row count mismatch')
    base,loss,factors=fit_weights(fit_rows,cfg['loss_mass'])
    candidates=[]
    with threadpool_limits(limits=cfg['threads']):
        scaler=StandardScaler().fit(x,sample_weight=base)
        z=scaler.transform(x).astype(np.float32)
        for c in LB_CS:
            model=LogisticRegression(C=c,penalty='l2',solver='liblinear',max_iter=cfg['max_iter'],random_state=cfg['seed'])
            model.fit(z,np.asarray([r['gold'] for r in fit_rows]),sample_weight=loss)
            if model.n_iter_.max()>=cfg['max_iter']:raise RuntimeError('Lookback LR hit iteration limit')
            record=_fit_record(fit_rows,base,loss,factors,cfg)
            record.update(family='lookback_local',candidate_id=f'lookback_C{c:g}',kind='lookback_lr',C=c,
                          complexity=[c],scaler=copy.deepcopy(scaler),model=model,
                          adaptation='Qwen post-read pure-LB784 mean4, fit-group weights and scaler; not released Llama classifier')
            candidates.append(record)
    return candidates


def auc_ranking(x,y,weights):
    """Fit-only signed AUC, matching official ReDeEP ranking direction.

    Constant columns score 0.5; exact ties resolve by original column index.
    Group base weights (not class factors) prevent long answers dominating rank.
    """
    y=np.asarray(y,int);weights=np.asarray(weights,float)
    if len(x)!=len(y) or set(y)!={0,1}:raise ValueError('AUC requires aligned two-class fit labels')
    auc=np.asarray([.5 if np.ptp(x[:,j])==0 else roc_auc_score(y,x[:,j],sample_weight=weights) for j in range(x.shape[1])])
    order=np.lexsort((np.arange(x.shape[1]),-auc))
    return order,auc


def fit_redeep_candidates(fit_ecs,fit_pks,fit_rows,config=None):
    """Select from all 784 ECS heads / 28 PKS layers on FIT only; 27 formulas."""
    cfg=DEFAULT_CONFIG|dict(config or {})
    ecs,pks=matrix(fit_ecs,784,'fit_ecs'),matrix(fit_pks,28,'fit_pks')
    if len(ecs)!=len(pks) or len(ecs)!=len(fit_rows):raise ValueError('Feature/fit row count mismatch')
    base,loss,factors=fit_weights(fit_rows,cfg['loss_mass'])
    y=np.asarray([r['gold'] for r in fit_rows])
    er,ea=auc_ranking(ecs,1-y,base);pr,pa=auc_ranking(pks,y,base)
    common=_fit_record(fit_rows,base,loss,factors,cfg)
    common.update(family='redeep_local',kind='redeep_formula',ecs_order=er,pks_order=pr,
       ecs_fit_auc=ea,pks_fit_auc=pa,ranking='fit-only group-base-weighted AUC: ECS vs 1-y, PKS vs y',
       adaptation='All Qwen ECS heads / standard-JSD PKS, four-BPE window supervision; fit-only ranking/scaling')
    candidates=[]
    for kh in REDEEP_HEADS:
        for kl in REDEEP_LAYERS:
            heads,layers=er[:kh].copy(),pr[:kl].copy()
            # Explicit float64 sums, same order at fit and inference.
            e=ecs[:,heads].astype(float).sum(1);p=pks[:,layers].astype(float).sum(1)
            minimum=np.asarray([p.min(),e.min()]);maximum=np.asarray([p.max(),e.max()])
            scale=maximum-minimum;scale[scale==0]=1.
            for beta in REDEEP_BETAS:
                candidate=copy.deepcopy(common)
                candidate.update(candidate_id=f'redeep_h{kh}_l{kl}_b{beta:g}',heads=heads.copy(),layers=layers.copy(),
                   Kh=kh,Kl=kl,beta=beta,alpha=1.,complexity=[kh+kl,kh,kl,beta],
                   minimum=minimum.copy(),maximum=maximum.copy(),scale=scale.copy(),
                   selected_head_coordinates=[{'layer_1based':int(h//28+1),'head_0based':int(h%28)} for h in heads],
                   selected_layer_numbers_1based=(layers+1).tolist(),minmax_clipped=False)
                candidates.append(candidate)
    # The runner resolves otherwise identical calibration results by first index.
    # Keep this equivalent to our explicit smaller-complexity tie rule.
    return sorted(candidates,key=lambda candidate:tuple(candidate['complexity']))


def predict(candidate,features:Mapping):
    """Inference has no labels/groups/condition inputs. Return all supplied rows."""
    if candidate['kind']=='lookback_lr':
        x=matrix(features['lb'],784,'lb')
        with threadpool_limits(limits=candidate['config']['threads']):
            score=candidate['model'].predict_proba(candidate['scaler'].transform(x).astype(np.float32))[:,1]
    elif candidate['kind']=='redeep_formula':
        ecs,pks=matrix(features['ecs'],784,'ecs'),matrix(features['pks'],28,'pks')
        if len(ecs)!=len(pks):raise ValueError('ECS/PKS row mismatch')
        e=ecs[:,candidate['heads']].astype(float).sum(1)
        p=pks[:,candidate['layers']].astype(float).sum(1)
        scaled=(np.column_stack((p,e))-candidate['minimum'])/candidate['scale']
        score=scaled[:,0]-candidate['beta']*scaled[:,1]
    else:raise ValueError('Unknown frozen baseline kind')
    score=np.asarray(score,float)
    if score.ndim!=1 or not np.isfinite(score).all():raise ValueError('Nonfinite baseline scores')
    return score


def best_threshold(rows,score):
    """Risk F1, then precision, then higher threshold; no missing-score dropping."""
    score=np.asarray(score,float)
    if score.shape!=(len(rows),) or not np.isfinite(score).all():raise ValueError('Every calibration row requires a finite score')
    ix=np.asarray([j for j,r in enumerate(rows) if r['main_eligible']],int)
    if not len(ix):raise ValueError('No eligible calibration labels')
    y=np.asarray([rows[j]['gold'] for j in ix],int);s=score[ix]
    if set(y)!={0,1}:raise ValueError('Both calibration classes required')
    order=np.argsort(-s,kind='stable');s,y=s[order],y[order]
    end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.r_[0,np.cumsum(y)[end]];n=np.r_[0,end+1]
    thresholds=np.r_[np.nextafter(s.max(),np.inf),s[end]]
    f=2*tp/(n+y.sum());precision=np.divide(tp,n,out=np.zeros(len(n),float),where=n>0)
    k=max(range(len(n)),key=lambda j:(f[j],precision[j],thresholds[j]))
    return {'threshold':float(thresholds[k]),'validation_f1':float(f[k]),'validation_precision':float(precision[k]),
            'scorable_rows':len(ix),'total_rows':len(rows),'risk_rows':int(y.sum())}


def answer_max(cal_windows,score,cal_answers):
    """All candidate windows, including safe refusals; no gold prefilter."""
    if len(cal_windows)!=len(score):raise ValueError('Window score mismatch')
    scores=defaultdict(list)
    for r,v in zip(cal_windows,score):scores[item_id(r)].append(float(v))
    value=[]
    for r in cal_answers:
        if r['item_id'] not in scores:raise ValueError('Missing windows for calibration answer: '+r['item_id'])
        value.append(max(scores[r['item_id']]))
    return np.asarray(value)


def _select_scores(candidates,scores,cal_windows,cal_answers):
    if len(candidates)!=len(scores) or not candidates:raise ValueError('Empty/mismatched candidates')
    cal_groups={r['group_id'] for r in cal_windows}|{r['group_id'] for r in cal_answers}
    table=[]
    for candidate,score in zip(candidates,scores):
        if cal_groups&set(candidate['fit_groups']):raise ValueError('Fit/calibration group overlap')
        ts={'window':best_threshold(cal_windows,score),
            'answer':best_threshold(cal_answers,answer_max(cal_windows,score,cal_answers))}
        wf,af=ts['window']['validation_f1'],ts['answer']['validation_f1']
        objective=[min(wf,af),wf,ts['window']['validation_precision']]
        table.append({'candidate_id':candidate['candidate_id'],'thresholds':ts,'selection_objective':objective,
                      'complexity':candidate['complexity'],'calibration_groups':sorted(cal_groups)})
    selected=max(range(len(table)),key=lambda j:(*table[j]['selection_objective'],*[-float(v) for v in table[j]['complexity']]))
    frozen=copy.deepcopy(candidates[selected])
    frozen.update(thresholds=table[selected]['thresholds'],calibration_groups=sorted(cal_groups),
       selection_rule='max min(window_F1,answer_F1), then window_F1, window_precision, smaller complexity/C; individual cutoff ties choose higher threshold',
       calibration_table=table,selected_index=selected,selection_frozen=True,
       calibration_rows_sha256=digest({'windows':[{'key':key(r),'group_id':r['group_id'],'gold':r['gold'],'eligible':r['main_eligible']} for r in cal_windows],
                                      'answers':[{'key':r['item_id'],'group_id':r['group_id'],'gold':r['gold'],'eligible':r['main_eligible']} for r in cal_answers]}))
    return frozen


def select_calibration(candidates,cal_features,cal_windows,cal_answers):
    """Freeze candidate+two cutoffs before the caller supplies evaluation data."""
    return _select_scores(candidates,[predict(c,cal_features) for c in candidates],cal_windows,cal_answers)


def lumina_score(ipr,mmd,*,source_aggregation=None):
    """Exact official default combination when BOTH compatible signals exist.

    Does not invent missing IPR. R19's two single-source interventions additionally
    require a predeclared source aggregation and remain a local adaptation.
    """
    if ipr is None:raise ValueError('LUMINA requires IPR; R19 caches contain MMD only. New IPR extraction is required.')
    ipr,mmd=np.asarray(ipr,float),np.asarray(mmd,float)
    if mmd.ndim==2:
        if source_aggregation=='mean':mmd=mmd.mean(1)
        elif source_aggregation=='max':mmd=mmd.max(1)
        else:raise ValueError('Multiple MMD interventions need a protocol-frozen aggregation')
    if ipr.ndim!=1 or ipr.shape!=mmd.shape or not np.isfinite(ipr).all() or not np.isfinite(mmd).all():
        raise ValueError('Aligned finite IPR/MMD vectors required')
    return .5*ipr-.5*mmd


def self_test():
    """Only analytic/synthetic checks, no LogisticRegression.fit or real data."""
    rows=[]
    for group,y in [('fit0',0),('fit1',1)]:
        for j in range(4):rows.append(dict(window_key=f'{group}{j}',group_id=group,condition='complete',item_ids=[group],main_eligible=True,gold=y))
    b,loss,f=fit_weights(rows);assert np.isclose(loss.sum(),3854)
    x=np.column_stack([np.r_[np.arange(4),np.arange(4)+10],np.ones(8),-np.r_[np.arange(4),np.arange(4)+10]])
    order,auc=auc_ranking(x,np.asarray([r['gold'] for r in rows]),b)
    assert order.tolist()==[0,1,2] and np.allclose(auc,[1,.5,0])
    windows=[];answers=[];s1=[];s2=[]
    for iid,golds,p1,p2 in [('A',[1]*8+[0]*3,[.9]*8+[.1]*3,[.7]*8+[.8]*3),('B',[1],[.2],[.7]),('C',[0],[.5],[.1])]:
        answers.append(dict(item_id=iid,group_id='cal',main_eligible=True,gold=int(any(golds))))
        for j,(y,v1,v2) in enumerate(zip(golds,p1,p2)):
            windows.append(dict(window_key=iid+str(j),item_ids=[iid],group_id='cal',main_eligible=True,gold=y));s1.append(v1);s2.append(v2)
    candidates=[dict(candidate_id=f'candidate{i}',fit_groups=['fit0'],complexity=[i]) for i in range(2)]
    selected=_select_scores(candidates,[s1,s2],windows,answers)
    assert selected['selected_index']==1 # Candidate0 has better window F1, worse dual objective.
    try:_select_scores(candidates,[s1,s2],[dict(w,group_id='fit0') for w in windows],answers)
    except ValueError:pass
    else:raise AssertionError('Group leak not rejected')
    try:lumina_score(None,np.ones(3))
    except ValueError:pass
    else:raise AssertionError('Missing IPR must not be fabricated')
    assert np.allclose(lumina_score([.4,.8],[[.1,.3],[.2,.4]],source_aggregation='mean'),[.1,.25])
    print('BASELINES21_SYNTHETIC_PASSED; no model fitted, no real data opened.')


if __name__=='__main__':self_test()
