"""Freeze a second, untouched sample after the first round4 results are known."""
import json,random,hashlib
from collections import Counter
import networkx as nx
from transformers import AutoTokenizer
from common4 import ROOT,PRE,readl,writel,save,sha

DEST=ROOT/'confirmation'

def main():
    for p in ['data','data/features','data/readouts','results','logs']:(DEST/p).mkdir(parents=True,exist_ok=True)
    if (DEST/'protocol.json').exists():print('CONFIRMATION ALREADY FROZEN');return
    prior=readl(ROOT/'data/rows.jsonl');used={r['group'] for r in prior};assert len(used)==684
    raw=readl(PRE/'data/raw/response.jsonl');sources={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
    tok=AutoTokenizer.from_pretrained(PRE/'models/Qwen2.5-7B-Instruct-bnb-4bit',local_files_only=True);eligible=[]
    for r in raw:
        s=sources[r['source_id']]
        if s['task_type']!='Summary' or r['quality']!='good':continue
        spans=r['labels'];group=hashlib.sha256(' '.join(s['source_info'].split()).encode()).hexdigest()
        if group in used or any(a['label_type']!='Evident Conflict' or a.get('implicit_true') or a.get('due_to_null') for a in spans):continue
        if any(r['response'][a['start']:a['end']].strip()!=a['text'].strip() for a in spans):continue
        prefix=tok.apply_chat_template([{'role':'user','content':s['prompt']}],tokenize=False,add_generation_prompt=True)
        enc=tok(prefix+r['response'],add_special_tokens=False,return_offsets_mapping=True);n=sum(b>len(prefix) for a,b in enc['offset_mapping'])
        if len(enc['input_ids'])>3072 or not 16<=n<=384:continue
        eligible.append({'id':r['id'],'source_id':r['source_id'],'group':group,'original_model':r['model'],'label':int(bool(spans)),
            'prompt':s['prompt'],'evidence':s['source_info'],'response':r['response'],'split':'confirmation','official_split':r['split'],
            'feature_path':str((DEST/f'data/features/{r["id"]}.npz').resolve())})
    print('ELIGIBLE',dict(Counter(r['original_model']+'_'+str(r['label']) for r in eligible)),flush=True)
    print('UNIQUE POSITIVE SOURCES',len({r['group'] for r in eligible if r['label']}),flush=True)
    random.Random(20260914).shuffle(eligible)
    available=Counter(r['original_model'] for r in eligible if r['label'])
    models=sorted(m for m,n in available.items() if n>=2)
    # Fix total20errors/30clean; choose feasible generator quotas as evenly as possible.
    # Each generator keeps exactly40% error, preventing generator/label confounding.
    def allocations_of(total,rest):
        if len(rest)==1:
            if 2<=total<=available[rest[0]] and total%2==0:yield (total,)
            return
        for n in range(2,min(available[rest[0]],total-2*(len(rest)-1))+1,2):
            for tail in allocations_of(total-n,rest[1:]):yield (n,)+tail
    allocations=sorted(allocations_of(20,models),key=lambda x:(sum((v-20/len(models))**2 for v in x),x))
    fresh=None
    for errors in allocations:
        quotas={m:(e*3//2,e) for m,e in zip(models,errors)};G=nx.DiGraph();candidates={}
        for m,counts in quotas.items():
            for label,n in enumerate(counts):G.add_edge('START',(m,label),capacity=n)
        for r in eligible:
            if r['original_model'] not in models:continue
            k=(r['original_model'],r['label']);g='source_'+r['group'];G.add_edge(k,g,capacity=1);G.add_edge(g,'END',capacity=1);candidates.setdefault((k,g),r)
        n,flow=nx.maximum_flow(G,'START','END')
        if n==50:
            fresh=[candidates[(k,g)] for k in [(m,y) for m in models for y in [0,1]] for g,v in flow[k].items() if v];break
    assert fresh is not None,'No feasible50-row sample; do not alter model or inspect scores'
    fresh=sorted(fresh,key=lambda r:int(r['id']));assert len(fresh)==len({r['group'] for r in fresh})==50 and sum(r['label'] for r in fresh)==20
    assert not used&{r['group'] for r in fresh};original={r['id']:r for r in raw}
    writel(DEST/'data/rows.jsonl',fresh);writel(DEST/'data/annotations.jsonl',[{'id':r['id'],'spans':original[r['id']]['labels']} for r in fresh])
    selection=json.loads((ROOT/'results/selection.json').read_text());ms={r['name']:r for r in json.loads((ROOT/'results/metrics.json').read_text())}
    frozen={n:selection['family_winners'][n] for n in ['sentences_facts','sentences3','base_scores']}
    save(DEST/'protocol.json',{'reason':'Round4 first-test-selected primary failed: F1 0.645. Two prespecified alternatives exceeded0.70. They are now selected using that viewed test as development and must be confirmed on new sources.',
        'primary':'sentences_facts: learned probe with internal states, six checks maximum','controls':['sentences3: direct B self-check score, three checks maximum','base_scores: original attention probe'],
        'n':50,'errors':20,'clean':30,'seed':20260914,'quotas':quotas,'all_prior684_sources_excluded':True,
        'sampling_amendment_before_any_confirmation_scores':'Only16 eligible positive Llama source groups remained, insufficient for50rows20positive. All generators together had22eligible positive sources, also insufficient for a proposed100rows40positive. Freeze50new mixed-generator sources20positive30clean, maintaining40% error within each represented generator. Both infeasible preparation attempts ended before any rows or protocol were frozen, any model was applied, or any confirmation scores were inspected. This confirms mixed-generator news, not the initial Llama-only distribution.',
        'frozen_methods':frozen,'models_retrained':False,'thresholds_recalibrated':False,'new_confirmation_scores_seen':False,
        'earlier_first_test_F1':{n:ms[n]['f1'] for n in ['selected','sentences_facts','sentences3']},
        'rows_sha256':sha(DEST/'data/rows.jsonl'),'annotations_sha256':sha(DEST/'data/annotations.jsonl'),
        'frozen_dependencies':{p:sha(ROOT/p) for p in ['results/selection.json','results/checkpoints/base.pkl','results/checkpoints/unit_models.pkl','data/unit_inputs.pkl']},
        'success_rule':'Primary answer-level error-class F1 >0.70. No reselection or threshold changes after this confirmation. Report all attempts and controls, token localization separately.'})
    print('CONFIRMATION FROZEN',quotas,flush=True)

if __name__=='__main__':main()
