import json
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=[json.loads(l) for l in (ROOT/'data/raw/response.jsonl').read_text(encoding='utf8').splitlines()]
s={x['source_id']:x for x in map(json.loads,(ROOT/'data/raw/source_info.jsonl').read_text(encoding='utf8').splitlines())}
c=Counter(); groups=defaultdict(set); examples=[]
for x in r:
    if x['quality']!='good': continue
    labs=x['labels']
    eligible=not labs or all(a['label_type']=='Evident Conflict' and not a.get('implicit_true',False) for a in labs)
    if eligible:
        k=(s[x['source_id']]['task_type'],x['split'],bool(labs))
        c[k]+=1; groups[(k[0],k[1],x['source_id'])].add(bool(labs))
        if labs and len(examples)<3: examples.append(x)
print(c)
print('Paired sources',Counter((k[0],k[1]) for k,v in groups.items() if len(v)==2))
print(json.dumps(examples,ensure_ascii=False,indent=2)[:5000])
print('Summary by generator', Counter((x['model'],x['split'],bool(x['labels'])) for x in r if s[x['source_id']]['task_type']=='Summary' and x['quality']=='good' and (not x['labels'] or all(a['label_type']=='Evident Conflict' and not a.get('implicit_true',False) for a in x['labels']))))
