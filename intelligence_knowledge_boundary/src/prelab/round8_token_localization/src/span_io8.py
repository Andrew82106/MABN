"""Source-only packets and exact span serialization; never load probe scores."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
R7=ROOT.parent/'round7_evidence_grounding'


def readl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet(split,start,count):
    annotations=readl(R7/'data'/f'annotations_{split}.jsonl')
    groups=list(dict.fromkeys(x['group_id'] for x in annotations))
    chosen=groups[start:start+count]
    inputs={x['row_id']:x for x in readl(R7/'data/inputs.jsonl')}
    for index,g in enumerate(chosen,start):
        print('GROUP',index,g)
        rows=[x for x in annotations if x['group_id']==g]
        for rid in dict.fromkeys(x['row_id'] for x in rows):
            src=inputs[rid]
            print('INPUT',json.dumps({'row_id':rid,'questions':src['questions'],'passages':src['passages']},ensure_ascii=False))
            for item in rows:
                if item['row_id']==rid:
                    print('ANSWER',json.dumps({k:item.get(k) for k in ('item_id','text','start','end','risk','stance','evidence_relation','reference_correctness','rationale','reference_evidence')},ensure_ascii=False))


def locate(text,quote,occurrence=None):
    assert isinstance(quote,str) and quote.strip() and quote==quote.strip()
    positions=[]; start=0
    while True:
        at=text.find(quote,start)
        if at<0:break
        positions.append(at);start=at+1
    assert positions, 'Quoted span is absent: '+quote
    if occurrence is None:
        assert len(positions)==1, 'Repeated quote requires explicit occurrence: '+quote
        return positions[0]
    assert type(occurrence) is int and 0<=occurrence<len(positions)
    return positions[occurrence]


def serialize(split,decisions,output,annotator):
    path=Path(decisions)
    doc=json.loads(path.read_text(encoding='utf-8-sig'))
    entries=doc['decisions'] if isinstance(doc,dict) else doc
    lookup={r['item_id']:r for r in entries}
    assert len(lookup)==len(entries)
    original=readl(R7/'data'/f'annotations_{split}.jsonl')
    expected={r['item_id'] for r in original if r['risk']==1}
    assert set(lookup)==expected, ('Risk decisions must be complete',expected-set(lookup),set(lookup)-expected)
    records=[]
    for old in original:
        gen_path=R7/'data/generation_records'/(old['row_id']+'.json')
        assert sha(gen_path)==old['source_generation_sha256']
        gen=json.loads(gen_path.read_text(encoding='utf-8'))
        assert gen['response'][old['start']:old['end']]==old['text']
        record={k:old[k] for k in ('item_id','row_id','question_id','group_id','split','condition','text','start','end','source_generation_sha256')}
        record.update(original_risk=old['risk'],original_stance=old['stance'],
                      original_evidence_relation=old['evidence_relation'],
                      localization_status='resolved' if old['risk'] in (0,1) else 'excluded',
                      claim_scope=[{'start':old['start'],'end':old['end'],'text':old['text']}],
                      risk_spans=[],annotator=annotator,token_scores_viewed=False,
                      human_gold=False,source_annotation_sha256=sha(R7/'data'/f'annotations_{split}.jsonl'))
        if old['risk']==1:
            d=lookup[old['item_id']]
            assert d.get('text',old['text'])==old['text']
            record['localization_status']=d.get('localization_status','resolved')
            assert record['localization_status'] in ('resolved','unresolved')
            record['rationale']=d.get('rationale','')
            record['decision_file_sha256']=sha(path)
            for index,s in enumerate(d.get('spans',[])):
                pos=locate(old['text'],s['quote'],s.get('occurrence'))
                a,b=old['start']+pos,old['start']+pos+len(s['quote'])
                assert gen['response'][a:b]==s['quote']
                assert s.get('rationale',d.get('rationale','')).strip()
                record['risk_spans'].append({'start':a,'end':b,'text':s['quote'],
                    'claim_id':s.get('claim_id',old['item_id']+f'__claim_{index+1}'),
                    'rationale':s.get('rationale',d.get('rationale','')),
                    'evidence_refs':s.get('evidence_refs',d.get('evidence_refs',[]))})
            if record['localization_status']=='resolved':
                assert record['risk_spans'],'A risk item requires explicitly selected risk spans'
            else:
                assert record['rationale'].strip()
            ordered=sorted(record['risk_spans'],key=lambda s:(s['start'],s['end']))
            assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])), 'Overlapping manual spans'
            record['risk_spans']=ordered
        else:
            record['rationale']='Reuse the frozen Round7 source-reviewed item label; no positive localization span.'
        records.append(record)
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps({'output':str(target),'items':len(records),'risk_items_reviewed':len(entries),
                      'spans':sum(len(r['risk_spans']) for r in records),'sha256':sha(target)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('packet');q.add_argument('--split',required=True);q.add_argument('--start',type=int,default=0);q.add_argument('--count',type=int,default=5)
    q=sub.add_parser('serialize');q.add_argument('--split',required=True);q.add_argument('--decisions',required=True);q.add_argument('--output',required=True);q.add_argument('--annotator',required=True)
    a=p.parse_args()
    if a.command=='packet':packet(a.split,a.start,a.count)
    else:serialize(a.split,a.decisions,a.output,a.annotator)
