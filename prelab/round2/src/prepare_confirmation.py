"""Freeze unused, source-disjoint human-labeled news before candidate fitting."""
import json,random,hashlib
from collections import Counter
from transformers import AutoTokenizer
from common import ROOT,PRELAB,readl,writel,sha

def main():
    dest=ROOT/'data/news_confirmation'; dest.mkdir(exist_ok=True)
    if (dest/'rows.jsonl').exists(): print('ALREADY FROZEN'); return
    used={r['group'] for s in ['train','val','test'] for r in readl(PRELAB/f'data/processed/{s}.jsonl')}
    sources={r['source_id']:r for r in readl(PRELAB/'data/raw/source_info.jsonl')}
    tok=AutoTokenizer.from_pretrained(PRELAB/'models/Qwen2.5-0.5B-Instruct',local_files_only=True); eligible={}
    for r in readl(PRELAB/'data/raw/response.jsonl'):
        s=sources[r['source_id']]
        if s['task_type']!='Summary' or r['model']!='mistral-7B-instruct' or r['quality']!='good': continue
        group=hashlib.sha256(' '.join(s['source_info'].split()).encode()).hexdigest()
        if group in used: continue
        spans=r['labels']
        if any(a['label_type']!='Evident Conflict' or a.get('implicit_true') or a.get('due_to_null') for a in spans): continue
        if any(r['response'][a['start']:a['end']].strip()!=a['text'].strip() for a in spans): continue
        prefix=tok.apply_chat_template([{'role':'user','content':s['prompt']}],tokenize=False,add_generation_prompt=True)
        enc=tok(prefix+r['response'],add_special_tokens=False,return_offsets_mapping=True)
        n=sum(b>len(prefix) for a,b in enc['offset_mapping'])
        if len(enc['input_ids'])>3072 or not 16<=n<=384: continue
        eligible[group]={'id':r['id'],'source_id':r['source_id'],'group':group,'official_split':r['split'],'split':'test',
          'original_model':r['model'],'prompt':s['prompt'],'evidence':s['source_info'],'response':r['response'],
          'label':int(bool(spans)),'spans':spans,'n_tokens':n}
    rng=random.Random(20260912); selected=[]
    for label,limit in [(0,80),(1,30)]:
        rr=sorted([r for r in eligible.values() if r['label']==label],key=lambda r:int(r['id'])); rng.shuffle(rr); selected+=rr[:limit]
    selected=sorted(selected,key=lambda r:int(r['id'])); count=Counter(r['label'] for r in selected)
    assert count[0]>=20 and count[1]>=10,count
    writel(dest/'rows.jsonl',[{k:v for k,v in r.items() if k!='spans'} for r in selected])
    writel(dest/'annotations.jsonl',[{'id':r['id'],'spans':r['spans']} for r in selected])
    m={'seed':20260912,'n':len(selected),'counts':dict(count),'eligible_counts':dict(Counter(r['label'] for r in eligible.values())),
       'source_disjoint_from_all_previous_news':True,'source':'Previously unused official-training RAGTruth sources; newly held out for this study, not the official test set',
       'purpose':'Confirmation of frozen probes and one bounded alarm-aware supervised control; never used for fitting or parameter selection',
       'rows_sha256':sha(dest/'rows.jsonl'),'annotations_sha256':sha(dest/'annotations.jsonl')}
    (dest/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8'); print(json.dumps(m,indent=2))

if __name__=='__main__': main()
