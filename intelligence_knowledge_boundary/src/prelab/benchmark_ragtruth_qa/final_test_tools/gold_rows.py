"""Count-independent frozen gold geometry. Pure rows/plans; no dataset I/O."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import build_gold as old

def map_row(row,plan):
    assert row['quality']=='good'
    original,covered=old.check_fixed_layout(row,plan)
    text,labels=row['original_response'],row['labels'];offsets=original['response_token_offsets'];raw=original['response_token_offsets_raw']
    n=len(offsets);assert n==len(original['answer_token_ids'])>0
    lc,rc,lex,risk=old.map_characters(text,labels,offsets)
    oraclelex,oraclerisk=old.independent_label_check(text,labels,offsets,lex,risk)
    ident={k:row[k] for k in ('response_id','source_id','group_id','partition')};ident['answer_id']=row['response_id']
    flags=[];span_tokens=[];edges=[]
    for i,lab in enumerate(labels):
        a,b=lab['start'],lab['end'];ref={**ident,'span_index':i,'original_start':a,'original_end':b,'original_text':lab['text']}
        if a==b:flags.append('zero_length_span');edges.append({**ref,'kind':'zero_length_spans'})
        if not any(lc[a:b]):flags.append('span_without_isalnum');edges.append({**ref,'kind':'spans_without_isalnum'})
        assert all(covered[j] or text[j].isspace() for j in range(a,b)),'Missing span coverage; stop without relabeling'
        indices=[j for j,(left,right) in enumerate(offsets) if any(text[k].isalnum() for k in range(max(left,a),min(right,b)))]
        span_tokens.append({'span_index':i,'original_start':a,'original_end':b,'risk_token_indices':indices})
    if labels and not any(risk):
        flags.append('answer_risk_without_risk_token');edges.append({**ident,'kind':'risk_answers_without_risk_tokens','original_labels':labels})
    token={**ident,'original_response':text,'answer_sha256':row['answer_sha256'],'original_labels':labels,
        'token_count':n,'token_ids':original['answer_token_ids'],'answer_token_positions':original['answer_token_positions'],
        'response_token_offsets_raw':raw,'response_token_offsets':offsets,'lexical_mask':lex,'risk_mask':risk,
        'risk_character_spans':old.character_spans(text,rc),'token_risk_character_spans':[old.character_spans(text,rc,a,b) if risk[j] else [] for j,(a,b) in enumerate(offsets)],
        'span_token_mapping':span_tokens,'answer_risk':int(bool(labels)),'first_answer_token_preserved':True,'edge_case_flags':sorted(set(flags))}
    windows=[];excluded=[]
    for start,end in old.windows_for_count(n):
        indices=list(range(start,end));li=[j for j in indices if lex[j]];ri=[j for j in indices if risk[j]]
        intervals=old.merge_intervals(offsets[start:end]);left=min(a for a,_ in intervals);right=max(b for _,b in intervals)
        wr=old.merge_intervals([(s['start'],s['end']) for j in indices for s in token['token_risk_character_spans'][j]])
        assert bool(li)==any(oraclelex[start:end])
        direct=int(any(text[k].isalnum() and any(s['start']<=k<s['end'] for s in labels) for j in indices for k in range(*offsets[j])))
        assert int(bool(ri))==direct
        w={**ident,'window_id':f"{row['response_id']}__k4_{start:05d}",'k':4,'stride':1,'token_start':start,'token_end':end,'token_indices':indices,
            'answer_token_positions':original['answer_token_positions'][start:end],'token_ids':original['answer_token_ids'][start:end],
            'character_intervals':intervals,'char_start':left,'char_end':right,'bounding_text':text[left:right],
            'lexical_token_indices':li,'risk_token_indices':ri,'risk_character_spans':[{'start':a,'end':b,'text':text[a:b]} for a,b in wr],
            'eligible':bool(li),'label':int(bool(ri)) if li else None}
        if li:windows.append(w)
        else:w['exclusion_reason']='no_lexical_token';excluded.append(w)
    if not windows:flags.append('no_eligible_window');edges.append({**ident,'kind':'answers_without_eligible_windows'})
    answer={**ident,'original_response':text,'answer_sha256':row['answer_sha256'],'original_labels':labels,'quality':'good','eligible':True,
        'label':int(bool(labels)),'token_count':n,'lexical_token_count':sum(lex),'risk_token_count':sum(risk),'official_span_count':len(labels),
        'candidate_window_count':len(old.windows_for_count(n)),'eligible_window_count':len(windows),'positive_window_count':sum(w['label'] for w in windows),
        'excluded_window_count':len(excluded),'edge_case_flags':sorted(set(flags))}
    return token,windows,excluded,answer,edges
