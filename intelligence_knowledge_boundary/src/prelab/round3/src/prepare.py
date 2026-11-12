import random,json,hashlib
from collections import Counter
from transformers import AutoTokenizer
from shared import ROOT,PRE,OLD,readl,writel,sha

def main():
    dest=ROOT/'data'; dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists(): print('ALREADY FROZEN'); return
    old=readl(OLD/'data/news_large/labeled.jsonl'); prior=readl(OLD/'data/news_confirmation/rows.jsonl')
    used={r['group'] for r in old+prior}
    anns={r['id']:r for r in readl(OLD/'data/news_large/local_annotations.jsonl')}
    rows=[]
    for r in old:
        if r['split'] not in ['train','val']:continue
        rows.append({**r,'feature_path':str((OLD/f'data/news_large/features/{r["id"]}.npz').resolve())})
    sources={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
    tok=AutoTokenizer.from_pretrained(PRE/'models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    eligible={}
    for r in readl(PRE/'data/raw/response.jsonl'):
        s=sources[r['source_id']]
        if r['model']!='llama-2-7b-chat' or s['task_type']!='Summary' or r['quality']!='good':continue
        group=hashlib.sha256(' '.join(s['source_info'].split()).encode()).hexdigest()
        if group in used:continue
        spans=r['labels']
        if any(a['label_type']!='Evident Conflict' or a.get('implicit_true') or a.get('due_to_null') for a in spans):continue
        if any(r['response'][a['start']:a['end']].strip()!=a['text'].strip() for a in spans):continue
        prefix=tok.apply_chat_template([{'role':'user','content':s['prompt']}],tokenize=False,add_generation_prompt=True)
        enc=tok(prefix+r['response'],add_special_tokens=False,return_offsets_mapping=True)
        n=sum(b>len(prefix) for a,b in enc['offset_mapping'])
        if len(enc['input_ids'])>3072 or not 16<=n<=384:continue
        eligible[group]={'id':r['id'],'source_id':r['source_id'],'group':group,'split':'test','official_split':r['split'],
          'original_model':r['model'],'prompt':s['prompt'],'evidence':s['source_info'],'response':r['response'],
          'label':int(bool(spans)),'spans':spans,'feature_path':str((dest/f'features/{r["id"]}.npz').resolve())}
    counts=Counter(r['label'] for r in eligible.values()); rng=random.Random(20260910); selected=[]
    for label in [0,1]:
        rr=sorted([r for r in eligible.values() if r['label']==label],key=lambda r:int(r['id']));rng.shuffle(rr)
        assert len(rr)>=40,(label,len(rr));selected+=rr[:40]
    for r in sorted(selected,key=lambda r:int(r['id'])):
        anns[r['id']]={'id':r['id'],'spans':r.pop('spans'),'ignore':[],'answers':[]};rows.append(r)
    writel(dest/'rows.jsonl',rows);writel(dest/'annotations.jsonl',[anns[r['id']] for r in rows])
    manifest={'n':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
       'fresh_test_eligible':dict(counts),'test_n':80,'test_clean':40,'test_error':40,
       'fresh_source_disjoint_from_all_round2_news':True,'scope':'new-source cross-generator transfer; original training and validation unchanged',
       'rows_sha256':sha(dest/'rows.jsonl'),'annotations_sha256':sha(dest/'annotations.jsonl')}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
