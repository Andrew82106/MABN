"""Gold locations are supplied only to the diagnostic selector, never as verdicts."""
import json,pickle,hashlib,re
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from common5 import R5,R4,PRE,readl,writel,save,sha,DEPENDENCIES
from make_queries import sentences,factual_span,values

def word_spans(text):return [m.span() for m in re.finditer(r"\b\w+(?:[-']\w+)*\b",text)]

def context(row,span):
    blocks=[s for s in sentences(row['response']) if s[0]<span[1] and s[1]>span[0]]
    return [min(s[0] for s in blocks),max(s[1] for s in blocks)]

def negative_match(pos,clean):
    text=clean['response'];wanted=max(1,len(word_spans(pos['target_text'])));options=[]
    rel=(pos['target_span'][0]-pos['sentence_span'][0])/max(1,pos['sentence_span'][1]-pos['sentence_span'][0])
    for sentence in sentences(text):
        a,b=sentence;words=word_spans(text[a:b]);n=min(wanted,len(words))
        if n==0:continue
        for k in range(len(words)-n+1):
            lo=a+words[k][0];hi=a+words[k+n-1][1]
            cost=abs(np.log((b-a+1)/(len(pos['statement'])+1)))+abs(np.log((hi-lo+1)/(len(pos['target_text'])+1)))+.1*abs((lo-a)/max(1,b-a)-rel)
            options.append((float(cost),sentence,[lo,hi]))
    return min(options,key=lambda x:(x[0],x[1][0],x[2][0]))

def main():
    for p in ['data','data/readouts','results','logs']:(R5/p).mkdir(parents=True,exist_ok=True)
    if (R5/'protocol.json').exists():print('ORACLE DIAGNOSTIC FROZEN');return
    rows=[{**r,'diagnostic_split':'val'} for r in readl(R4/'data/rows.jsonl') if r['split']=='val']+[{**r,'diagnostic_split':'diagnostic'} for r in readl(R4/'confirmation/data/rows.jsonl')]
    rr={r['id']:r for r in rows};assert len(rr)==170
    anns={a['id']:a['spans'] for p in [R4/'data/annotations.jsonl',R4/'confirmation/data/annotations.jsonl'] for a in readl(p) if a['id'] in rr}
    bundle=pickle.loads((R4/'data/unit_inputs.pkl').read_bytes());bb=bundle['base'];off={};scores={}
    for i,r in enumerate(bb['rows']):
        if r['id'] in rr:off[r['id']]=bb['offsets'][i];scores[r['id']]=bb['base_scores'][i]
    for p in readl(R4/'confirmation/results/predictions.jsonl'):
        off[p['id']]=np.asarray(p['offsets']);scores[p['id']]=np.asarray(p['methods']['base_scores']['token_risks'],np.float32)
    allold=readl(R4/'data/queries.jsonl')+readl(R4/'confirmation/data/queries.jsonl');old={r['id']:[q for q in allold if q['id']==r['id'] and q['kind'] in ['sentence','fact']] for r in rows}
    queries={};policies=[];cases=[]
    def add(r,kind,sentence,target):
        qid=hashlib.sha256(f'{r["id"]}|{kind}|{sentence}|{target}'.encode()).hexdigest()[:20]
        q={'query_id':qid,'id':r['id'],'split':r['diagnostic_split'],'kind':kind,'sentence_span':sentence,'target_span':target,
           'statement':r['response'][sentence[0]:sentence[1]],'target_text':r['response'][target[0]:target[1]],'evidence':r['evidence'],'base_features':values(off[r['id']],scores[r['id']],target)}
        if qid in queries:assert queries[qid]==q
        queries[qid]=q;return qid
    def builds(r,blocks,special=None):
        ids=[]
        for block in blocks:
            target,_=factual_span(r['response'],block,off[r['id']],scores[r['id']])
            if special is not None and block==special[0]:target=special[1]
            ids.extend([add(r,'sentence',block,block),add(r,'fact',block,target)])
        return ids
    for r in rows:
        automatic=[add(r,q['kind'],q['sentence_span'],q['target_span']) for q in old[r['id']]]
        blocks=[q['target_span'] for q in old[r['id']] if q['kind']=='sentence'];assert builds(r,blocks)==automatic
        if r['label']:
            g=min(anns[r['id']],key=lambda a:(a['start'],a['end']));span=[g['start'],g['end']];ctx=context(r,span)
            assisted=[list(s) for s in blocks]
            if ctx not in assisted:assisted[-1]=ctx
            selected=builds(r,assisted);targeted=builds(r,assisted,(ctx,span))
        else:selected=automatic;targeted=automatic
        assert len(automatic)==len(selected)==len(targeted)
        policies.append({'id':r['id'],'split':r['diagnostic_split'],'label':r['label'],'automatic':automatic,'oracle_sentence':selected,'oracle_span':targeted})
    # Matched correct spans come from completely clean answers, without using risk scores.
    for split in ['val','diagnostic']:
        for gen in sorted({r['original_model'] for r in rows if r['diagnostic_split']==split}):
            positives=sorted([r for r in rows if r['diagnostic_split']==split and r['original_model']==gen and r['label']],key=lambda r:int(r['id']))
            negatives=sorted([r for r in rows if r['diagnostic_split']==split and r['original_model']==gen and not r['label']],key=lambda r:int(r['id']))
            if not positives:continue
            pq=[]
            for r in positives:
                g=min(anns[r['id']],key=lambda a:(a['start'],a['end']));span=[g['start'],g['end']];ctx=context(r,span);pq.append(queries[add(r,'fact',ctx,span)])
            options=[[negative_match(q,n) for n in negatives] for q in pq];cost=np.array([[x[0] for x in z] for z in options]);aa,bb=linear_sum_assignment(cost);assert len(aa)==len(positives)
            for i,j in zip(aa,bb):
                pos=positives[i];neg=negatives[j];pair_id=f'{split}_{pos["id"]}_{neg["id"]}';_,nc,nt=options[i][j]
                for r,ctx,target,label in [(pos,pq[i]['sentence_span'],pq[i]['target_span'],1),(neg,nc,nt,0)]:
                    cases.append({'case_id':pair_id+'_'+str(label),'pair_id':pair_id,'id':r['id'],'split':split,'label':label,
                        'sentence_query':add(r,'sentence',ctx,ctx),'span_query':add(r,'fact',ctx,target),'matching_cost':float(cost[i,j])})
    qs=sorted(queries.values(),key=lambda q:q['query_id']);writel(R5/'data/rows.jsonl',rows);writel(R5/'data/annotations.jsonl',[{'id':r['id'],'spans':anns[r['id']]} for r in rows]);writel(R5/'data/queries.jsonl',qs);writel(R5/'data/policies.jsonl',policies);writel(R5/'data/cases.jsonl',cases)
    save(R5/'protocol.json',{'purpose':'Diagnostic only: remove automatic location selection and ask whether frozen readouts distinguish annotated conflict spans from length-matched clean spans. This is not automatic localization or a new independent test.',
        'model':'Same local frozen Qwen2.5-7B NF4; no external LLM, no free generation','unit_probe':'abc_base_hidden16_balanced_0.01, existing round4 weights and PCA, no retraining',
        'data':'Previously viewed round4 validation120 and confirmation50; these are reused solely for diagnosis. Each matched pair uses one earliest annotated error and one clean answer from the same original generator; each clean source used once, matched by text length and within-sentence position only, without risk scores.',
        'matched_cases':dict(Counter(c['split'] for c in cases)),'queries':len(qs),
        'unit_analysis':'For sentence and exact-span units, compare original attention top10mean, direct B probability, and frozen internal-state probe. AUROC and paired ranking are threshold independent. Unit F1 thresholds fitted only on matched validation units and frozen before diagnostic scoring. On balanced units all-positive F1=2/3; unit F1 is not token or answer F1.',
        'pipeline_analysis':'Keep old answer classifier, threshold and number of checks fixed. In each erroneous answer, supply the context of its earliest annotated error by replacing the lowest-ranked selected sentence only if missing; then additionally replace that sentence factual target with the full annotated error span. Clean answers remain exactly identical. This uses label-derived location assistance, not a deployable detector.',
        'metrics_policy':'Report all controls, precision/recall/confusions, bootstrap intervals, paired ranking and length-only score AUCs. No score-based example selection, no test tuning, no claim to exceed automatic-detection target.',
        'frozen_dependencies':{p:sha(R4/p) for p in DEPENDENCIES},'data_sha256':{p.name:sha(p) for p in (R5/'data').glob('*.jsonl')},'generated_tokens':0})
    print('PREPARED',len(qs),'queries',dict(Counter(c['split'] for c in cases)),flush=True)

if __name__=='__main__':main()

