"""Read-only FIT scope inventory. No changed features, scores, or labels."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HEADER=re.compile(r'^\s*Passage\s+([123])\s*:\s*$', re.I)
BULLET=re.compile(r'^\s*[*+\-]\s+\S')


def run():
    rows={r['response_id']:r for r in (json.loads(l) for l in (ROOT/'data/fit.jsonl').open(encoding='utf-8'))}
    assert len(rows)==634 and all(r['partition']=='fit' for r in rows.values())
    by=defaultdict(list); path=ROOT/'results/citation_alignment_v1/claims.jsonl'
    for line in path.open(encoding='utf-8'):
        # Mixed file: inspect identifier first; never parse calibration payloads.
        match=re.search(r'"response_id"\s*:\s*"(\d+)"',line)
        if match and match.group(1) in rows:
            c=json.loads(line); assert c['partition']=='fit'; by[c['response_id']].append(c)
    inherited={}; headers=[]; intervals=[]; records=[]
    for rid,row in rows.items():
        text=row['original_response']; active=None; cursor=0
        for line in text.splitlines(keepends=True):
            end=cursor+len(line.rstrip('\r\n')); m=HEADER.fullmatch(line.rstrip('\r\n'))
            if m:
                active={'source':int(m.group(1)),'start':cursor,'end':end,'text':text[cursor:end]}
                headers.append({'response_id':rid,**active})
            elif not line.strip():
                pass
            elif active and BULLET.match(line):
                block=dict(response_id=rid,start=cursor,end=end,header=active.copy())
                intervals.append(block)
                for c in by[rid]:
                    if cursor<=c['start'] and c['end']<=end and c['parser']['status']=='none':
                        key=(rid,c['claim_index']); assert key not in inherited
                        inherited[key]=active['source']
                        records.append(dict(response_id=rid,claim_index=c['claim_index'],start=c['start'],end=c['end'],
                            text=c['text'],old_parser=c['parser'],old_features=c['features'],header=active.copy(),
                            proposed_inherited_source=active['source']))
            else:
                active=None
            cursor+=len(line)
    touched_tokens=defaultdict(set)
    for line in (ROOT/'data/tokens_fit.jsonl').open(encoding='utf-8'):
        t=json.loads(line);rid=t['response_id'];claims=by[rid]
        for i,(a,b) in enumerate(t['response_token_offsets']):
            chars=[j for j in range(a,b) if t['original_response'][j].isalnum()]
            if not chars: continue
            overlaps=[sum(c['start']<=j<c['end'] for j in chars) for c in claims]
            chosen=max(range(len(claims)),key=lambda j:(overlaps[j],-j));assert overlaps[chosen]>0
            if (rid,claims[chosen]['claim_index']) in inherited: touched_tokens[rid].add(i)
    wc=Counter(); affected_windows=[]
    for line in (ROOT/'data/windows_k4_fit.jsonl').open(encoding='utf-8'):
        w=json.loads(line)
        if touched_tokens[w['response_id']].intersection(w['lexical_token_indices']):
            wc[w['response_id']]+=1;affected_windows.append(w['window_id'])
    cases=json.loads((OUT/'CASES.json').read_text(encoding='utf-8'))['cases']; relevant=[]
    for case in cases:
        rid=case['response_id'];s=case['original_gold']
        overlapping=[c for c in by[rid] if c['start']<s['end'] and s['start']<c['end']]
        relevant.append({'case_id':case['case_id'],'overlapping_claims':overlapping,
                         'proposed_inherited_claims':[r for r in records if r['response_id']==rid and r['start']<s['end'] and s['start']<r['end']]})
    selected_ids={c['response_id'] for c in cases}
    output={'status':'read_only_inventory_complete','scope':'original FIT634 only; no calibration text parsed',
        'rule':'Standalone whole-line Passage 1/2/3: only. Empty lines preserve. Only immediately subsequent *,+,- bullet lines inherit; plain text or any other heading closes. No inheritance across a wrapped plain line. Claim must be fully inside bullet line with old parser status none. Explicit or unknown references unchanged.',
        'counts':{'fit_answers':634,'fit_claims':sum(map(len,by.values())),
            'standalone_headers':len(headers),'answers_with_standalone_headers':len({h['response_id'] for h in headers}),
            'bullet_lines_in_header_scope':len(intervals),'newly_attributed_claims':len(records),
            'answers_with_newly_attributed_claims':len({r['response_id'] for r in records}),
            'affected_lexical_tokens':sum(map(len,touched_tokens.values())),'affected_original_windows':len(affected_windows)},
        'new_attributions':records,'affected_window_ids':affected_windows,
        'selected_case_actual_claims':relevant,
        'selected_answer_all_actual_claims':{rid:by[rid] for rid in sorted(selected_ids,key=int)},
        'old_sources_unchanged':True,'feature_exported':False,'GPU_used':False,'trained':False,'labels_used_for_scope':False,
        'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/fit.jsonl',path,ROOT/'src/build_citation_alignment.py',ROOT/'src/build_cited_source_semantic.py']}}
    (OUT/'SCOPE_CHECK.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output['counts']))


if __name__=='__main__':run()
