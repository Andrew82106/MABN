"""Persist explicitly authored external-span decisions without reading scores."""
import json
from span_io8 import ROOT,R7,readl,locate


def add(decisions):
    target=ROOT/'data/decisions/external_spans.json'
    existing=json.loads(target.read_text(encoding='utf-8'))['decisions'] if target.exists() else []
    ids={r['item_id'] for r in existing}
    gold={r['item_id']:r for r in readl(R7/'data/annotations_external_test.jsonl')}
    for item_id,spans,reason in decisions:
        assert item_id not in ids and gold[item_id]['risk']==1
        entry={'item_id':item_id,'text':gold[item_id]['text'],'rationale':reason,
               'evidence_refs':gold[item_id]['visible_source_titles'],
               'spans':[{'quote':s,'rationale':reason} for s in spans]}
        for s in spans:locate(entry['text'],s)
        existing.append(entry);ids.add(item_id)
    target.write_text(json.dumps({'annotator':'/root','token_scores_viewed':False,
                                  'decisions':existing},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('EXPLICIT_EXTERNAL_DECISIONS',len(existing))
