"""Source-only packets and storage of explicitly chosen quantity questions."""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT/'data/curation'
sys.path.insert(0, str(ROOT.parent/'round9_evidence_binding/src'))
from curation_io9 import segments

def read(path):
    return [json.loads(x) for x in path.read_text('utf-8-sig').splitlines() if x]

def packet(start, count):
    pool = {x['candidate_id']:x for x in read(CUR/'available_source_pool.jsonl')}
    for index, a in enumerate(read(CUR/'allocation_data_build.jsonl')[start:start+count],start):
        row=pool[a['candidate_id']]
        print('ALLOCATION',index,row['candidate_id'],row['source_title'])
        print('QUESTION',row['original_question']);print('ANSWER',row['original_answer_quote'])
        for n,s in enumerate(segments(row['source_content'])):
            print(n,s['text'])

def add(specs, rejections=()):
    pool={x['candidate_id']:x for x in read(CUR/'available_source_pool.jsonl')}
    path=CUR/'quantity_candidates.jsonl'
    old=read(path) if path.exists() else []
    seen={x['candidate_id'] for x in old}
    allowed={x['candidate_id'] for x in read(CUR/'allocation_data_build.jsonl')}
    for cid,subject,question,answer,c,e,p,note in specs:
        assert cid in allowed and cid not in seen,cid
        source=pool[cid];text=source['source_content'];ss=segments(text)
        quotes=[];intervals=[]
        for selected in [c,e,p]:
            if type(selected) is int:
                quote=ss[selected]['text']
            else:
                quote=selected
            assert quote==quote.strip() and text.count(quote)==1,(cid,quote)
            start=text.index(quote);intervals.append((start,start+len(quote)));quotes.append(quote)
        intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),cid
        old.append(dict(candidate_id=cid,category='quantity',subject=subject,question=question,
                        reference_answer=answer,common_quote=quotes[0],evidence_quote=quotes[1],
                        partial_quote=quotes[2],review_note=note,status='accepted'))
        seen.add(cid)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in old),'utf-8')
    rejpath=CUR/'quantity_rejections.jsonl';rej=read(rejpath) if rejpath.exists() else []
    for cid,reason in rejections:
        assert cid not in seen and cid not in {x['candidate_id'] for x in rej}
        rej.append({'candidate_id':cid,'reason':reason})
    rejpath.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rej),'utf-8')
    inv=json.loads((ROOT.parent/'round10_dual_granularity/data/curation/legacy_isolation_inventory.json').read_text('utf-8'))
    norm=lambda s:re.sub(r'[^\w]+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
    visible=inv['normalized_old_visible_passages']+[norm(p['title']+' '+p['text']) for x in read(ROOT.parent/'round10_dual_granularity/data/test_inputs.jsonl') for p in x['passages']]
    hits=[]
    for x in old[-len(specs):] if specs else []:
        matches=[v for v in visible if ' '+norm(x['subject'])+' ' in ' '+v+' ']
        if matches:hits.append({'candidate_id':x['candidate_id'],'subject':x['subject'],'matches':matches})
    print(json.dumps({'accepted':len(old),'rejected':len(rej),'literal_old_subject_hits_review_required':hits},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=5)
    a=p.parse_args();packet(a.start,a.count)
