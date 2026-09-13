"""Posthoc frozen-OOF diagnosis; no fitting, threshold changes or old holdout parsing."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib, json, re, math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT.parent
R16=PRE/'round16_dataset_expansion'
R18=PRE/'round18_input_information_diagnostics'
R19=PRE/'round19_three_signal_probe'
OUT=ROOT/'research'

def read(p): return json.loads(p.read_text('utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def train_rows():
    selected=[]
    for s in (R16/'data/inputs.jsonl').open(encoding='utf-8'):
        m=re.search(r'(?<!\\)"split"\s*:\s*"([^"\\]+)"',s)
        if m and m[1]=='train': selected.append(json.loads(s))
    assert len(selected)==602
    return selected
def count(rr,method,prediction=None):
    assert all(r['gold'] in (0,1) for r in rr)
    y=np.asarray([r['gold'] for r in rr],int)
    p=np.asarray([r['predictions'][method] for r in rr] if prediction is None else prediction,bool)
    tp=int((p&(y==1)).sum()); fp=int((p&(y==0)).sum()); fn=int((~p&(y==1)).sum()); tn=int((~p&(y==0)).sum())
    return dict(n=len(rr),risk=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
         precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
         f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
def best_threshold(rr,method):
    """Oracle: evaluates exposed outer labels, never saved as model parameters."""
    score=np.asarray([r['scores'][method] for r in rr]); y=np.asarray([r['gold'] for r in rr],int)
    order=np.argsort(-score,kind='stable'); ss=score[order]; yy=y[order]
    end=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(yy)[end]; selected=end+1
    f=np.divide(2*tp,selected+y.sum(),out=np.zeros(len(end),float),where=selected+y.sum()>0)
    precision=tp/selected
    j=max(range(len(end)),key=lambda k:(f[k],precision[k],ss[end[k]]))
    threshold=float(ss[end[j]])
    return dict(threshold=threshold,metrics=count(rr,method,score>=threshold)),score>=threshold

def joint_oracle(rr,method,group_field):
    """Exact pooled-F1 optimum among one threshold per specified known group.

    Fractional-programming search over all finite threshold candidates plus
    predict-none; original predictions/thresholds remain unchanged.
    """
    buckets=defaultdict(list)
    for j,r in enumerate(rr):buckets[r[group_field]].append(j)
    options={}; gold=sum(r['gold'] for r in rr)
    for group,ii in buckets.items():
        s=np.asarray([rr[j]['scores'][method] for j in ii]); y=np.asarray([rr[j]['gold'] for j in ii])
        order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]
        end=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
        tp=np.r_[0,np.cumsum(yy)[end]];n=np.r_[0,end+1]
        t=np.r_[math.nextafter(float(s.max()),math.inf),ss[end]]
        options[group]=(ii,tp,n,t)
    value=0.; selected={}
    for iteration in range(100):
        total_tp=total_p=0
        for group,(ii,tp,n,t) in options.items():
            objective=2*tp-value*n
            k=max(range(len(t)),key=lambda j:(objective[j],t[j]))
            selected[group]=k;total_tp+=int(tp[k]);total_p+=int(n[k])
        next_value=2*total_tp/(total_p+gold)
        if abs(next_value-value)<1e-14:break
        value=next_value
    else:raise AssertionError('Oracle fractional search failed to converge')
    prediction=[False]*len(rr);thresholds={}
    for group,(ii,tp,n,t) in options.items():
        threshold=float(t[selected[group]]);thresholds[str(group)]=threshold
        for j in ii:prediction[j]=rr[j]['scores'][method]>=threshold
    measures=count(rr,method,prediction)
    assert abs(measures['f1']-next_value)<1e-12
    return dict(metrics=measures,thresholds=thresholds,iterations=iteration+1,
           restriction='Exact pooled optimum only within separate constant thresholds per '+group_field,
           not_deployable=True)
def rank(rr,method):
    y=[r['gold'] for r in rr]; s=np.asarray([r['scores'][method] for r in rr])
    if len(set(y))!=2:return None
    return dict(auroc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for base, filename in [(R18,'complete18.json'),(R19,'complete19.json')]:
        for name,expected in read(base/'results'/filename)['files_sha256'].items():
            assert sha(base/'results'/name)==expected,('Changed frozen experiment',base.name,name)
    inputs={r['row_id']:r for r in train_rows()}
    annotations={a['row_id']:a for a in lines(R16/'data/annotations_train.jsonl')}
    wr=lines(R19/'results/window_scores_oof.jsonl'); ar=lines(R19/'results/answer_scores_oof.jsonl')
    summary=read(R19/'results/summary.json')
    r18w={r['window_key']:r for r in lines(R18/'results/window_scores_oof.jsonl')}
    r18a={r['item_id']:r for r in lines(R18/'results/answer_scores_oof.jsonl')}
    methods=['base','base_mmd','base_ecs','base_pks','base_all','signals_only','r18_slots','r18_mlp','r18_hidden32']
    for r in wr+ar:
        old=r18w[r['window_key']] if 'window_key' in r else r18a[r['item_id']]
        for new,key in [('r18_slots','slots_lr'),('r18_mlp','mean_mlp'),('r18_hidden32','mean_hidden32')]:
            r['scores'][new]=old['scores'][key];r['predictions'][new]=old['predictions'][key]
    ww=[w for w in wr if w['main_eligible']]; aa=[a for a in ar if a['main_eligible']]
    byrow=defaultdict(list)
    for w in ww:byrow[w['row_id']].append(w)
    byanswer={a['row_id']:a for a in ar}
    gens={rid:read(R16/'data/generation_records'/(rid+'.json')) for rid in inputs}
    risk_tokens={}
    for rid,a in annotations.items():
        bad={k for s in a['risk_spans'] for k in range(s['start'],s['end']) if gens[rid]['response'][k].isalnum()}
        risk_tokens[rid]={j for j,(l,r) in enumerate(gens[rid]['response_token_offsets']) if any(k in bad for k in range(l,r))}
    for w in ww:
        rid=w['row_id'];goldidx=risk_tokens[rid]
        w['answer_risk']=annotations[rid]['original_risk']
        w['answer_length']=len(gens[rid]['response_token_ids'])
        w['length_bin']='<=16' if w['answer_length']<=16 else '17..32' if w['answer_length']<=32 else '>32'
        w['evidence_relation']=annotations[rid]['evidence_relation']
        lo,hi=min(w['raw_token_indices']),max(w['raw_token_indices'])
        w['risk_token_overlap']=len(set(w['raw_token_indices'])&goldidx)
        w['position']='supported_answer' if not goldidx else 'before_first_risk' if hi<min(goldidx) else 'after_last_risk' if lo>max(goldidx) else 'between_or_overlap_risk'
        w['nearest_risk_token_gap']=min(abs(t-s) for t in w['raw_token_indices'] for s in goldidx) if goldidx else None
    result={'scope':'Posthoc R18-informed/R19-informed training-source OOF diagnosis; no original validation/test parsed; no fitting or deployed threshold changes',
       'denominators':dict(questions=301,groups=278,answers=len(aa),risk_answers=sum(a['gold'] for a in aa),windows=len(ww),risk_windows=sum(w['gold'] for w in ww)),
       'methods':{},'strata':{},'oracle':{},'data_quality':{},'source_hashes':{}}
    for m in methods:
        local=[]; positive=[]; fn=[]
        for a in aa:
            rr=byrow[a['row_id']]
            if a['gold']!=1 or not rr:continue
            s=np.asarray([r['scores'][m] for r in rr]); top=[r for r,v in zip(rr,s) if v==max(s)]
            record=dict(row_id=a['row_id'],peak_all_risk=all(r['gold']==1 for r in top),answer_alert=a['predictions'][m],any_window_alert=any(r['predictions'][m] for r in rr))
            positive.append(record)
            if not a['predictions'][m]:fn.append(record)
            ranking=rank(rr,m)
            if ranking:local.append(ranking)
        fp=[w for w in ww if w['gold']==0 and w['predictions'][m]]
        lost=[w for w in ww if w['gold']==1 and not w['predictions'][m]]
        result['methods'][m]={'windows':count(ww,m),'answers':count(aa,m),
          'safe_refusal_fp':sum(a['predictions'][m] for a in aa if a['reviewed_safe_refusal']),
          'risk_answer_peak_all_risk':sum(p['peak_all_risk'] for p in positive),'risk_answers':len(positive),
          'answer_fn_peak_all_risk':sum(p['peak_all_risk'] for p in fn),'answer_fn':len(fn),
          'answer_fn_with_window_alert':sum(p['any_window_alert'] for p in fn),
          'mixed_answers':len(local),'within_answer_mean_auroc':float(np.mean([p['auroc'] for p in local])),
          'within_answer_mean_ap':float(np.mean([p['ap'] for p in local])),
          'fp_by_answer_gold':dict(Counter(w['answer_risk'] for w in fp)),
          'fp_by_position':dict(Counter(w['position'] for w in fp)),
          'fp_in_risky_answers_with_gap1':sum(w['answer_risk']==1 and w['nearest_risk_token_gap']==1 for w in fp),
          'fp_in_risky_answers_with_gap2to4':sum(w['answer_risk']==1 and w['nearest_risk_token_gap'] in (2,3,4) for w in fp),
          'fn_by_positive_overlap_tokens':dict(Counter(w['risk_token_overlap'] for w in lost)),
          'positive_by_overlap_tokens':dict(Counter(w['risk_token_overlap'] for w in ww if w['gold']))}
        # Per-fold outer-label oracle cutoffs: optimistic diagnosis, not deployable.
        oracle={}; predictions=[]; refs=[]
        for f in range(5):
            rr=[r for r in ww if r['fold']==f]; info,p=best_threshold(rr,m)
            oracle[str(f)]=info; refs.extend(rr);predictions.extend(p.tolist())
        # Remove all false alerts in truly risk-free answers using gold: how much
        # could perfect answer triage help? This is a label-informed upper bound.
        triage=[w['predictions'][m] and w['answer_risk']==1 for w in ww]
        # Correct only immediate non-risk boundary windows; never deploy this.
        boundary=[w['predictions'][m] and not(w['gold']==0 and w['nearest_risk_token_gap']==1) for w in ww]
        # P/R upper envelope may select a separate cutoff for each known answer.
        peranswer_pred=[]; peranswer_rows=[]
        for rr in byrow.values():
            if not any(r['gold'] for r in rr):p=np.zeros(len(rr),bool)
            else:_,p=best_threshold(rr,m)
            peranswer_rows.extend(rr);peranswer_pred.extend(p.tolist())
        result['oracle'][m]={'warning':'Uses exposed gold; diagnostic only, cannot deploy or claim as achieved performance',
            'per_fold_outer_best_threshold':oracle,'per_fold_oracle_pooled':count(refs,m,predictions),
            'joint_optimal_five_fold_cutoffs':joint_oracle(ww,m,'fold'),
            'perfect_answer_triage_keep_frozen_window_alerts':count(ww,m,triage),
            'fix_only_immediate_boundary_false_positives':count(ww,m,boundary),
            'gold_per_answer_separate_cutoffs_plus_perfect_triage':count(peranswer_rows,m,peranswer_pred),
            'joint_optimal_per_answer_cutoffs':joint_oracle(ww,m,'row_id')}
    for field in ('category','condition','length_bin','evidence_relation','risk_token_overlap'):
        result['strata'][field]={}
        for value in sorted({w[field] for w in ww},key=str):
            rr=[w for w in ww if w[field]==value]
            result['strata'][field][str(value)]={m:count(rr,m) for m in methods}
    result['answer_strata']={}
    for field in ('category','condition','evidence_relation','length_bin'):
        for a in aa:
            a['evidence_relation']=annotations[a['row_id']]['evidence_relation']
            n=len(gens[a['row_id']]['response_token_ids']); a['length_bin']='<=16' if n<=16 else '17..32' if n<=32 else '>32'
        result['answer_strata'][field]={str(v):{m:count([a for a in aa if a[field]==v],m) for m in methods}
                                     for v in sorted({a[field] for a in aa},key=str)}
    result['transitions']={}
    for m in methods[1:]:
        changes=[w for w in ww if w['predictions'][m]!=w['predictions']['base']]
        result['transitions'][m]={'total':len(changes),'counts':dict(Counter(
           ('TP' if w['gold'] else 'FP')+'->'+('FN' if w['gold'] else 'TN') if w['predictions']['base'] else
           ('FN' if w['gold'] else 'TN')+'->'+('TP' if w['gold'] else 'FP') for w in changes)),
           'by_category':{c:dict(Counter(('gain_tp' if w['gold'] else 'add_fp') if w['predictions'][m] else ('lose_tp' if w['gold'] else 'remove_fp')
                         for w in changes if w['category']==c)) for c in sorted({w['category'] for w in ww})}}
    result['data_quality']['label_counts']=dict(Counter(a['evidence_relation'] for a in annotations.values()))
    result['data_quality']['stance_counts']=dict(Counter(a['original_stance'] for a in annotations.values()))
    result['data_quality']['risk_spans_empty_mismatch']=[rid for rid,a in annotations.items() if a['localization_status']=='resolved' and bool(a['risk_spans'])!=(a['original_risk']==1)]
    result['data_quality']['adjudicated_count']=sum(a.get('adjudicated',False) for a in annotations.values())
    result['data_quality']['reference_correctness_field_present']=sum('reference_correctness' in a for a in annotations.values())
    result['data_quality']['human_gold_true']=sum(a.get('human_gold',False) for a in annotations.values())
    # Short deterministic reading set: 5 high-confidence FNs, 5 high-confidence
    # FPs, plus 10 risk-bearing answers with best/worst within-answer ranking.
    selections=[]
    for typ in ('fn','fp'):
        pool=[w for w in ww if w['gold']==1 and not w['predictions']['base_all']] if typ=='fn' else [w for w in ww if w['gold']==0 and w['predictions']['base_all']]
        pool.sort(key=lambda w:(w['scores']['base_all'] if typ=='fn' else -w['scores']['base_all'],w['window_key']))
        seen=set()
        for w in pool:
            if w['row_id'] in seen:continue
            selections.append((w['row_id'],typ,w['window_key']));seen.add(w['row_id'])
            if len(seen)==5:break
    ranked=[]
    for rid,rr in byrow.items():
        ranking=rank(rr,'base_all')
        if ranking:ranked.append((ranking['auroc'],rid))
    ranked.sort()
    for _,rid in ranked[:5]+ranked[-5:]:selections.append((rid,'ranking_audit',None))
    # Include contrasted unsupported/correctness edge candidates without labeling
    # by word matching: search only supplies a reading list.
    for rid,a in annotations.items():
        if a['original_risk']==1 and a['evidence_relation']=='contradicted':selections.append((rid,'contradiction_audit',None))
    packets=[]; seen=set()
    for rid,reason,key in selections:
        if rid in seen:continue
        seen.add(rid); a=annotations[rid]
        packets.append({'row_id':rid,'selection':reason,'target_window':key,
          'question':inputs[rid]['questions'][0],'passages':inputs[rid]['passages'],
          'response':gens[rid]['response'],'annotation':a,
          'answer_prediction':byanswer[rid],
          'windows':[{'key':w['window_key'],'text':w['text'],'gold':w['gold'],'score':w['scores']['base_all'],'pred':w['predictions']['base_all']} for w in byrow[rid]]})
    result['data_quality']['reading_packet_count']=len(packets)
    for p in [R16/'data/annotation_freeze.json',R18/'results/complete18.json',R19/'results/complete19.json',
              R19/'results/window_scores_oof.jsonl',R19/'results/answer_scores_oof.jsonl',R16/'data/annotations_train.jsonl']:
        result['source_hashes'][str(p.resolve())]=sha(p)
    save(OUT/'ERROR_DIAGNOSIS.json',result)
    save(OUT/'diagnostic_reading_packets.json',packets)
    for path,expected in result['source_hashes'].items():assert sha(Path(path))==expected
    print('DIAGNOSIS_READY',len(packets),'manual reading packets')

if __name__=='__main__':run()
