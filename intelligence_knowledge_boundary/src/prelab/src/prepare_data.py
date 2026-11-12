"""Freeze a single-generator news benchmark, with spans separated from training."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
SEED=20260909
def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    sources={x['source_id']:x for x in read_jsonl(ROOT/'data/raw/source_info.jsonl')}
    rows=[]; omitted=Counter()
    for r in read_jsonl(ROOT/'data/raw/response.jsonl'):
        s=sources[r['source_id']]
        if s['task_type']!='Summary' or r['model']!='mistral-7B-instruct': continue
        if r['quality']!='good': omitted['quality']+=1; continue
        labs=r['labels']
        if any(x['label_type']!='Evident Conflict' or x.get('implicit_true',False) or x.get('due_to_null',False) for x in labs):
            omitted['not_pure_explicit_conflict']+=1; continue
        # Only trust exact released offsets; do not silently repair annotation spans.
        if any(r['response'][x['start']:x['end']].strip()!=x['text'].strip() for x in labs):
            omitted['annotation_offset_mismatch']+=1; continue
        prompt=tok.apply_chat_template([{'role':'user','content':s['prompt']}],tokenize=False,add_generation_prompt=True)
        enc=tok(prompt+r['response'],return_offsets_mapping=True,add_special_tokens=False)
        answer_tokens=sum(b>len(prompt) for a,b in enc['offset_mapping'])
        if len(enc['input_ids'])>3072 or not 16<=answer_tokens<=384:
            omitted['length']+=1; continue
        group=hashlib.sha256(' '.join(s['source_info'].split()).encode()).hexdigest()
        rows.append({'id':r['id'],'source_id':r['source_id'],'group':group,'official_split':r['split'],
                     'source':s['source'],'original_model':r['model'],'prompt':s['prompt'],
                     'evidence':s['source_info'],'response':r['response'],'label':int(bool(labs)),
                     'spans':labs,'n_tokens':answer_tokens,'total_tokens':len(enc['input_ids'])})
    test_groups={x['group'] for x in rows if x['official_split']=='test'}
    test=[dict(x,split='test') for x in rows if x['official_split']=='test']
    candidates=[x for x in rows if x['official_split']=='train' and x['group'] not in test_groups]
    # Deduplicate article groups before sampling (one response from one generator).
    candidates=list({x['group']:x for x in candidates}.values())
    test=list({x['group']:x for x in test}.values())
    rng=random.Random(SEED); chosen=[]
    for label in [0,1]:
        items=sorted([x for x in candidates if x['label']==label],key=lambda x:x['id'])
        rng.shuffle(items)
        nv=min(40 if label==0 else 25,max(5,len(items)//5))
        nt=min(140 if label==0 else 100,len(items)-nv)
        chosen.extend(dict(x,split='val') for x in items[:nv])
        chosen.extend(dict(x,split='train') for x in items[nv:nv+nt])
    chosen+=test
    for split in ['train','val','test']:
        data=sorted([x for x in chosen if x['split']==split],key=lambda x:int(x['id']))
        write_jsonl(ROOT/f'data/processed/{split}.jsonl',[{k:v for k,v in x.items() if k!='spans'} for x in data])
        # Separate files: probe training code never loads any span annotations.
        write_jsonl(ROOT/f'data/annotations/{split}_spans.jsonl',[{'id':x['id'],'spans':x['spans']} for x in data])
    counts={split:dict(Counter(x['label'] for x in chosen if x['split']==split)) for split in ['train','val','test']}
    manifest={'seed':SEED,'task':'News summarization: explicit evidence conflict vs no annotated hallucination',
              'generator':'mistral-7B-instruct','target_model':'Qwen2.5-0.5B-Instruct','mode':'teacher-forced replay',
              'counts':counts,'excluded':dict(omitted),'total':len(chosen),
              'raw_sha256':{p.name:sha(p) for p in (ROOT/'data/raw').glob('*.jsonl')},
              'split_sha256':{p.name:sha(p) for p in (ROOT/'data/processed').glob('*.jsonl')},
              'span_supervision':'None in fitting or model/layer/threshold selection; heldout spans only used for final localization evaluation',
              'scope_limit':'Source-relative conflict labels do not independently establish real-world falsity. English news, fixed retrieved evidence, no live search agent.'}
    (ROOT/'results/dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
