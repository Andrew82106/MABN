"""Freeze splits before model outputs; independent subjects for multi-fact RAG."""
import hashlib
import json
import random
from collections import Counter
import requests
import pyarrow.parquet as pq
from common import ROOT,PRELAB,norm,writel,sha

def trivia():
    files={'train':('455223e4a3ba6977611914d96b1b812fe19a6cd4ea3aef4df381f093b297670c',55391498),
           'validation':('48a5005c0eb4f8a4ae5b9868644297fd5bf1e694aeb3dc9c8ab958cac0d5b201',7335321)}
    raw={}
    for split,(digest,size) in files.items():
        path=ROOT/f'data/trivia_{split}.parquet'
        for attempt in range(6):
            if path.exists() and path.stat().st_size==size and sha(path)==digest: break
            offset=path.stat().st_size if path.exists() and path.stat().st_size<size else 0
            url=f'https://hf-mirror.com/datasets/mandarjoshi/trivia_qa/resolve/main/rc.nocontext/{split}-00000-of-00001.parquet?offset={offset}'
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=90) as r:
                    r.raise_for_status()
                    if offset: assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with path.open('ab' if offset else 'wb') as f:
                        for b in r.iter_content(1024*1024): f.write(b)
            except requests.RequestException as e:
                print('Data download resume',split,attempt,type(e).__name__,flush=True)
        assert sha(path)==digest
        raw[split]=pq.read_table(path).to_pylist()
    rng=random.Random(20260910); out=[]; seen=set()
    for source,targets in [('validation',[('val',300),('test',400)]),('train',[('train',2000)])]:
        data=raw[source]; rng.shuffle(data)
        # The HF version can contain repeated questions across evidence instances.
        unique=[]
        for r in data:
            key=norm(r['question'])
            if key in seen: continue
            seen.add(key); unique.append(r)
        cursor=0
        for split,n in targets:
            for r in unique[cursor:cursor+n]:
                a=r['answer']; aliases=list(dict.fromkeys([a['value']]+a['aliases']+a.get('normalized_aliases',[])))
                out.append({'id':'trivia_'+r['question_id'],'group':norm(r['question']),'split':split,
                            'question':r['question'],'aliases':aliases,'prompt':'Answer the following question with only a short answer, without explanation.\nQuestion: '+r['question']+'\nAnswer:'})
            cursor+=n
    assert len(out)==2700
    writel(ROOT/'data/trivia_questions.jsonl',out)
    print('Trivia',Counter(r['split'] for r in out),flush=True)

def squad():
    rng=random.Random(20260911)
    raw=json.loads((ROOT/'data/squad_train.json').read_text(encoding='utf8'))['data']
    dev=json.loads((ROOT/'data/squad_dev.json').read_text(encoding='utf8'))['data']
    rng.shuffle(raw); val_titles={x['title'] for x in raw[:70]}
    dev_titles={x['title'] for x in dev}
    candidates={'train':[],'val':[],'test':[]}
    for official,data in [('train',raw),('test',dev)]:
        for article in data:
            title=article['title']
            if official=='train' and title in dev_titles: continue
            split='test' if official=='test' else ('val' if title in val_titles else 'train')
            for pi,p in enumerate(article['paragraphs']):
                if not 400<=len(p['context'])<=2500: continue
                qa=[]; answers=set()
                for q in p['qas']:
                    aliases=list(dict.fromkeys(a['text'] for a in q['answers']))
                    if not aliases or len(aliases[0].split())>8 or norm(aliases[0]) in answers: continue
                    if any(w in q['question'].lower() for w in ['why ','explain','summarize']): continue
                    qa.append({'qid':q['id'],'question':q['question'],'aliases':aliases})
                    answers.add(norm(aliases[0]))
                if len(qa)<3: continue
                rng.shuffle(qa); qa=qa[:3]
                questions='\n'.join(f'{i+1}. {q["question"]}' for i,q in enumerate(qa))
                instruction='Answer the three questions using only the evidence. Return a JSON array of exactly three short answer strings in question order. Copy answer words from the evidence; use "UNKNOWN" if the evidence does not answer a question. Do not explain.'
                prompt=instruction+'\nEvidence:\n'+p['context']+'\nQuestions:\n'+questions+'\nAnswers:'
                no_evidence=instruction+'\nEvidence:\n[Evidence not supplied]\nQuestions:\n'+questions+'\nAnswers:'
                candidates[split].append({'id':'squad_'+qa[0]['qid'],'group':title,'split':split,'evidence':p['context'],
                     'qas':qa,'prompt':prompt,'no_evidence_prompt':no_evidence,'source':'SQuAD v1.1 Wikipedia evidence'})
    out=[]
    for split,n in [('train',700),('val',150),('test',200)]:
        rng.shuffle(candidates[split]); assert len(candidates[split])>=n
        out.extend(candidates[split][:n])
    writel(ROOT/'data/rag_questions.jsonl',out)
    print('Multi-fact RAG',Counter(r['split'] for r in out),'titles',Counter({s:len({r['group'] for r in out if r['split']==s}) for s in candidates}),flush=True)

if __name__=='__main__':
    trivia(); squad()
    files=[ROOT/'data/trivia_questions.jsonl',ROOT/'data/rag_questions.jsonl']
    (ROOT/'results/data_manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in files},
       'design':'2700 own-generation closed-book QA; 1050 three-fact RAG outputs; fixed source-group splits before generation',
       'automatic_label_limit':'Alias matching for QA; exact extractive match / zero-overlap candidate errors / ambiguous exclusion for RAG. Review sampled outputs; not independently human-annotated factuality.'},indent=2),encoding='utf8')
