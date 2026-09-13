"""Frozen-coefficient mass-feature sensitivity; no fitting or threshold changes."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parent
readl=lambda p:[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
items=readl(ROOT/'answer_index22.jsonl')
windows=readl(ROOT/'candidate_windows22.jsonl')
z=np.load(ROOT/'verifier_designs22.npz');mass=z['extra'][:,1];membership=z['window_answer_index']
def stats(v):
    v=np.asarray(v,np.float64)
    return {'n':len(v),'min':float(v.min()),'median':float(np.median(v)),
        'max':float(v.max()),'mean':float(v.mean()),'std':float(v.std()),
        'quantiles':dict(zip(['p01','p05','p25','p75','p95','p99'],map(float,np.quantile(v,[.01,.05,.25,.75,.95,.99]))))}
def metric(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {'n':len(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
def mass_groups(mask):
    v=mass[np.asarray(mask,bool)];u,c=np.unique(v,return_counts=True)
    return dict(stats(v),exact_one=int((v==1).sum()),above_one=int((v>1).sum()),
        within_one_float32_epsilon_of_one=int((np.abs(v.astype(np.float64)-1)<=np.finfo(np.float32).eps).sum()),
        unique=[{'value':float(a),'count':int(b)} for a,b in zip(u,c)])
categories=sorted({r['category'] for r in items})
out={'scope':'Read-only frozen coefficient sensitivity. No refit, new thresholds, C selection or GPU.',
    'source_sha256':{n:sha(ROOT/n) for n in ['verifier_designs22.npz','answer_index22.jsonl','candidate_windows22.jsonl','summary22.json','complete22.json']},
    'mass_definition':'float32 sum of two full-vocabulary probabilities for A and B',
    'global':mass_groups(np.ones(len(items),bool)),
    'by_gold':{str(g):mass_groups([r['main_eligible'] and r['gold']==g for r in items]) for g in [0,1]},
    'by_category':{g:mass_groups([r['category']==g for r in items]) for g in categories},
    'by_safe_refusal':{str(g):mass_groups([r['reviewed_safe_refusal']==g for r in items]) for g in [False,True]},
    'folds':[]}
modes=['original','zero_centered_mass_contribution','raw_mass_set_to_one','rounding_band_mass_set_to_one']
methods=['verifier_learned_broadcast','verifier_learned_slots']
oof={m:{mode:{'answers':[],'windows':[]} for mode in modes} for m in methods}
flip_ids={m:{mode:[] for mode in modes[1:]} for m in methods}
gid=np.array([r['group_id'] for r in items]);wgid=np.array([r['group_id'] for r in windows])
for f in range(5):
    frozen=pickle.loads((ROOT/f'fold_{f}_frozen22.pkl').read_bytes())
    scores=np.load(ROOT/f'fold_{f}_scores22.npz');x=scores['probe_design'];ev=np.flatnonzero(np.isin(gid,frozen['evaluation_groups']));wev=np.flatnonzero(np.isin(wgid,frozen['evaluation_groups']))
    fr={'fold':f,'candidate_coefficients':[],'selected_contributions':{}}
    probs={};contrib={}
    for j,c in enumerate(frozen['all_lr_candidates']):
        sc,model=c['scaler'],c['model'];scaled=x.copy();scaled-=sc.mean_;scaled/=sc.scale_
        logits=scaled@model.coef_[0]+model.intercept_[0]
        p=expit(logits);assert np.allclose(p,scores[f'probe_C{j}_answer'],rtol=0,atol=3e-16)
        contribution=scaled[:,-1]*model.coef_[0,-1]
        probs[j]={'original':p,'zero_centered_mass_contribution':expit(logits-contribution)}
        for mode in ['raw_mass_set_to_one','rounding_band_mass_set_to_one']:
            altered=x.copy()
            if mode=='raw_mass_set_to_one':altered[:,-1]=1
            else:altered[np.abs(mass.astype(np.float64)-1)<=np.finfo(np.float32).eps,-1]=1
            altered-=sc.mean_;altered/=sc.scale_
            probs[j][mode]=expit(altered@model.coef_[0]+model.intercept_[0])
        contrib[j]=contribution
        fr['candidate_coefficients'].append({'C':c['C'],'mass_fit_mean':float(sc.mean_[-1]),'mass_fit_variance':float(sc.var_[-1]),
            'mass_scale':float(sc.scale_[-1]),'mass_standardized_coefficient':float(model.coef_[0,-1]),
            'logit_margin_coefficient':float(model.coef_[0,-2]),'outer_absolute_mass_logit_contribution':stats(np.abs(contribution[ev])),
            'outer_absolute_other_logit_contribution_sum':stats(np.abs(scaled[ev,:-1]*model.coef_[0,:-1]).sum(axis=1))})
    for method in methods:
        sel=frozen['selection'][method];j=sel['candidate_index'];original=probs[j]['original'];thresholds=frozen['thresholds'][method]
        local=np.ones(len(windows))
        if method.endswith('_slots'):
            raw=np.asarray(scores['slots_base'],np.float64);eps=np.finfo(np.float64).eps;q=np.clip(raw,eps,1-eps);logits=np.log(q)-np.log1p(-q)
            for i in range(len(items)):
                ii=np.flatnonzero(membership==i);local[ii]=np.exp((logits[ii]-logits[ii].max())/sel['T'])
        original_w=original[membership]*local
        assert np.allclose(original_w,scores[method],rtol=0,atol=3e-16)
        original_pred=original>=thresholds['answer']['threshold']
        max_ix=int(ev[np.argmax(np.abs(contrib[j][ev]))])
        fr['selected_contributions'][method]={'C':sel['C'],'T':sel.get('T'),'mass_logit_abs':stats(np.abs(contrib[j][ev])),
          'largest_contribution_item':{'item_id':items[max_ix]['item_id'],'mass':float(mass[max_ix]),'contribution':float(contrib[j][max_ix]),
            'original_score':float(original[max_ix]),'without_centered_mass_score':float(probs[j]['zero_centered_mass_contribution'][max_ix]),
            'threshold':thresholds['answer']['threshold'],'gold':items[max_ix]['gold']}}
        for mode in modes:
            av=probs[j][mode];wv=av[membership]*local
            for i in ev:
                r=items[i]
                if not r['main_eligible']:continue
                pred=bool(av[i]>=thresholds['answer']['threshold'])
                oof[method][mode]['answers'].append({'id':r['item_id'],'gold':r['gold'],'pred':pred,'score':float(av[i])})
                if mode!='original' and pred!=original_pred[i]:flip_ids[method][mode].append({'item_id':r['item_id'],'gold':r['gold'],'original_prediction':bool(original_pred[i]),'diagnostic_prediction':pred})
            for i in wev:
                r=windows[i]
                if r['main_eligible']:oof[method][mode]['windows'].append({'id':r['window_key'],'gold':r['gold'],'pred':bool(wv[i]>=thresholds['window']['threshold']),'score':float(wv[i])})
    out['folds'].append(fr)
out['frozen_coefficient_sensitivity']={}
for method in methods:
    out['frozen_coefficient_sensitivity'][method]={}
    for mode in modes:
        q={}
        for unit in ['answers','windows']:
            rows=oof[method][mode][unit];base=oof[method]['original'][unit]
            q[unit]=metric([r['gold'] for r in rows],[r['pred'] for r in rows])
            q[unit]['prediction_flips_vs_original']=sum(r['pred']!=b['pred'] for r,b in zip(rows,base))
            q[unit]['score_change_abs']=stats([abs(r['score']-b['score']) for r,b in zip(rows,base)])
        if mode!='original':q['answer_prediction_flip_details']=flip_ids[method][mode]
        out['frozen_coefficient_sensitivity'][method][mode]=q
out['interpretation_limits']=[
    'Zeroing an existing centered feature contribution is diagnostic only, not a valid retrained ablation or new result.',
    'Raw mass values well below one include actual remaining vocabulary mass, not merely one-ULP summation noise; saturated values can nevertheless be quantized.',
    'Small training variance can amplify a held-out low-mass outlier, so stability should not be inferred from overall F1 alone.',
    'All frozen models, thresholds, label definitions and source result files remain unchanged.'
]
out['script_sha256']=sha(Path(__file__))
(ROOT/'MASS_FEATURE_DIAGNOSTIC22.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'global':{k:v for k,v in out['global'].items() if k!='unique'},'sensitivity':{m:{mode:{u:dict((k,v) for k,v in x[u].items() if k!='score_change_abs') for u in ['answers','windows']} for mode,x in mm.items()} for m,mm in out['frozen_coefficient_sensitivity'].items()}},ensure_ascii=False,indent=2))
