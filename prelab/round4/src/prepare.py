import random,hashlib
from collections import Counter
import networkx as nx
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer
from common4 import ROOT,PRE,readl,writel,save,sha

def main():
    for p in ['data','data/features','results','results/checkpoints','logs']: (ROOT/p).mkdir(parents=True,exist_ok=True)
    if (ROOT/'data/manifest.json').exists():print('DATA FROZEN');return
    old=[]
    for folder,file in [('news_large','labeled.jsonl'),('news_confirmation','rows.jsonl')]:
        for r in readl(PRE/f'round2/data/{folder}/{file}'):
            old.append({**r,'feature_path':str((PRE/f'round2/data/{folder}/features/{r["id"]}.npz').resolve())})
    old += [r for r in readl(PRE/'round3/data/rows.jsonl') if r['split']=='test']
    assert len(old)==584 and len({r['group'] for r in old})==584
    used={r['group'] for r in old};strata=[r['original_model']+str(r['label']) for r in old]
    ti,vi=next(StratifiedShuffleSplit(n_splits=1,test_size=120,random_state=20260911).split(old,strata))
    train=set(ti.tolist());rows=[]
    for i,r in enumerate(old):rows.append({**r,'split':'train' if i in train else 'val','data_role':'previously-viewed development'})
    original={r['id']:r for r in readl(PRE/'data/raw/response.jsonl')};sources={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
    tok=AutoTokenizer.from_pretrained(PRE/'models/Qwen2.5-7B-Instruct-bnb-4bit',local_files_only=True)
    eligible=[]
    for r in original.values():
        s=sources[r['source_id']]
        if not r['model'].startswith('llama-2-') or s['task_type']!='Summary' or r['quality']!='good':continue
        spans=r['labels'];group=hashlib.sha256(' '.join(s['source_info'].split()).encode()).hexdigest()
        if group in used or any(a['label_type']!='Evident Conflict' or a.get('implicit_true') or a.get('due_to_null') for a in spans):continue
        if any(r['response'][a['start']:a['end']].strip()!=a['text'].strip() for a in spans):continue
        prefix=tok.apply_chat_template([{'role':'user','content':s['prompt']}],tokenize=False,add_generation_prompt=True)
        enc=tok(prefix+r['response'],add_special_tokens=False,return_offsets_mapping=True);n=sum(b>len(prefix) for a,b in enc['offset_mapping'])
        if len(enc['input_ids'])>3072 or not 16<=n<=384:continue
        eligible.append({'id':r['id'],'source_id':r['source_id'],'group':group,'original_model':r['model'],'label':int(bool(spans)),
            'prompt':s['prompt'],'evidence':s['source_info'],'response':r['response'],'split':'test','data_role':'new-source heldout',
            'official_split':r['split'],'feature_path':str((ROOT/f'data/features/{r["id"]}.npz').resolve())})
    rng=random.Random(20260911);rng.shuffle(eligible);G=nx.DiGraph();candidates={}
    quotas={'llama-2-7b-chat':(21,14),'llama-2-13b-chat':(21,14),'llama-2-70b-chat':(18,12)}
    for model,qq in quotas.items():
        for label,n in enumerate(qq):G.add_edge('START',(model,label),capacity=n)
    for r in eligible:
        k=(r['original_model'],r['label']);g='source_'+r['group'];G.add_edge(k,g,capacity=1);G.add_edge(g,'END',capacity=1);candidates.setdefault((k,g),r)
    total,flow=nx.maximum_flow(G,'START','END');assert total==100,total
    fresh=[candidates[(k,g)] for k in [(m,y) for m in quotas for y in [0,1]] for g,n in flow[k].items() if n]
    rows += sorted(fresh,key=lambda r:int(r['id']));assert len({r['group'] for r in rows})==684
    anns=[{'id':r['id'],'spans':original[r['id']]['labels']} for r in rows]
    writel(ROOT/'data/rows.jsonl',rows);writel(ROOT/'data/annotations.jsonl',anns)
    save(ROOT/'data/manifest.json',{'rows':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
        'class_counts':dict(Counter(r['split']+'_'+str(r['label']) for r in rows)),
        'fresh_quotas':quotas,'new_test_groups_excluded_from_all_prior584':True,
        'development_reuses_earlier_tests':True,'rows_sha256':sha(ROOT/'data/rows.jsonl'),'annotations_sha256':sha(ROOT/'data/annotations.jsonl')})
    print('DATA COMPLETE',Counter(r['split'] for r in rows),flush=True)

if __name__=='__main__':main()
