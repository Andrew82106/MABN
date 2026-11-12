"""Fixed100 inspection of official synthetic TRAINING ONLY; no model dependency."""
from pathlib import Path
import json,re,random,hashlib
from collections import Counter

ROOT=Path(__file__).resolve().parent
TYPES={'entity','relation','invented','subjective','unverifiable','contradictory'}
TAGS=TYPES|{'mark','delete'}
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def prompt_parts(prompt):
    marker=list(re.finditer(r'\nPlease identify[^\n]*\n(?:Text: |\[Text\] )',prompt))
    if len(marker)!=1:raise ValueError(f'prompt boundary count={len(marker)}')
    m=marker[0];context=prompt[:m.start()];answer=prompt[m.end():]
    if answer.endswith('\n[Edited] '):answer=answer[:-len('\n[Edited] ')]
    refs=list(re.finditer(r'(?:^|\n)Reference \[(\d+)\]: ',context))
    if not refs:raise ValueError('No reference blocks')
    assert [int(x.group(1)) for x in refs]==list(range(1,len(refs)+1))
    texts=[context[x.end():refs[j+1].start() if j+1<len(refs) else len(context)].rstrip('\n') for j,x in enumerate(refs)]
    return texts,answer

def parse_markup(text):
    stack=[];corrupt=[];edited=[];position=0;spans=[];nodes=[];unknown=[]
    def plain(s):
        nonlocal position
        names=[n['tag'] for n in stack]
        if 'mark' not in names:corrupt.append(s);position+=len(s)
        if 'delete' not in names:edited.append(s)
    end=0
    for match in re.finditer(r'<(/?)([A-Za-z_]+)>',text):
        plain(text[end:match.start()]);end=match.end();closing,name=match.groups()
        if name not in TAGS:unknown.append(name);plain(match.group());continue
        if not closing:stack.append({'tag':name,'start':position,'deletions':[]});continue
        if not stack or stack[-1]['tag']!=name:raise ValueError(f'unbalanced close {name} at {match.start()}')
        node=stack.pop();node['end']=position
        if name=='delete':
            for parent in stack:
                if parent['tag'] in TYPES:parent['deletions'].append((node['start'],position))
        if name in TYPES:
            intervals=node['deletions'] or [(node['start'],position)]
            spans.extend({'type':name,'start':a,'end':b,'explicit_delete':bool(node['deletions'])} for a,b in intervals if b>a)
            nodes.append({'type':name,'start':node['start'],'end':position,'has_explicit_deletion':bool(node['deletions'])})
    plain(text[end:])
    if stack:raise ValueError('unclosed tags: '+','.join(n['tag'] for n in stack))
    if unknown:raise ValueError('unknown markup tags: '+','.join(sorted(set(unknown))))
    corrupt=''.join(corrupt);edited=''.join(edited)
    for s in spans:s['text']=corrupt[s['start']:s['end']]
    return corrupt,edited,spans,nodes

def inspect():
    protocol=load(ROOT/'INSPECTION_PROTOCOL.json');data=load(ROOT/'training.json')
    assert isinstance(data,list) and len(data)>=100
    selected=sorted(random.Random(protocol['sample_seed']).sample(range(len(data)),protocol['sample_count']))
    assert selected==load(ROOT/'SAMPLE_INDICES.json')['indices']
    schemas=Counter(tuple(sorted(x)) for x in data)
    # Full-file census is structural only, not a full alignment/semantic-quality audit.
    full_prompt=Counter(sha(x['prompt']) for x in data)
    reference_sets=Counter();reference_blocks=Counter();reference_failures=0
    for row in data:
        try:
            references,_=prompt_parts(row['prompt']);hashes=sorted(sha(x) for x in references)
            reference_sets[tuple(hashes)]+=1;reference_blocks.update(hashes)
        except (AssertionError,ValueError):reference_failures+=1
    summary={'population':len(data),'raw_bytes':(ROOT/'training.json').stat().st_size,
        'schema_counts':{'|'.join(k):v for k,v in schemas.items()},
        'unique_full_prompts':len(full_prompt),'extra_exact_prompt_duplicates':sum(v-1 for v in full_prompt.values()),
        'explicit_source_subject_question_clean_answer_fields_present':any(set(x)-{'prompt','completion'} for x in data),
        'sample_size':len(selected),'seed':protocol['sample_seed'],'row_records':[]}
    summary['full_file_reference_structure_only']={
        'parsed_prompts':len(data)-reference_failures,'unparsed_prompts':reference_failures,
        'unique_unordered_exact_reference_sets':len(reference_sets),
        'extra_rows_sharing_exact_reference_set':sum(v-1 for v in reference_sets.values()),
        'largest_exact_reference_set_cluster':max(reference_sets.values()),
        'unique_reference_block_hashes':len(reference_blocks),
        'blocks_shared_between_rows':sum(v>1 for v in reference_blocks.values()),
        'limitation':'Shared blocks can be distractors; exact reference sets are a grouping proxy, not original article IDs or proof of source independence.'}
    rows=[];counts=Counter();types=Counter();refcounts=Counter();no_delete=Counter();lengths=[]
    for i in selected:
        row=data[i];r={'raw_index':i,'prompt_sha256':sha(row['prompt']),'completion_sha256':sha(row['completion']),
            'label_provenance':'synthetic, no human verification implied'}
        try:
            refs,answer=prompt_parts(row['prompt']);refcounts[len(refs)]+=1
            r.update(evidence_blocks=refs,corrupted_answer=answer,reference_text_sha256=[sha(s) for s in refs])
            lengths.append({'reference_characters':sum(map(len,refs)),'answer_characters':len(answer)})
            reconstructed,edited,spans,nodes=parse_markup(row['completion'])
            counts['balanced_known_markup']+=1
            exact=reconstructed==answer;counts['exact_corrupted_answer_reconstruction']+=int(exact)
            r.update(markup_balanced=True,exact_answer_roundtrip=exact,
                reconstructed_corrupted_answer=reconstructed,edited_candidate_not_clean_gold=edited,
                synthetic_error_spans=spans,typed_regions=nodes)
            if not exact:
                at=next((j for j,(a,b) in enumerate(zip(answer,reconstructed)) if a!=b),min(len(answer),len(reconstructed)))
                r['mismatch']={'first_character':at,'prompt_context':answer[max(0,at-30):at+60],
                    'markup_context':reconstructed[max(0,at-30):at+60],
                    'whitespace_normalized_equal_diagnostic_only':' '.join(answer.split())==' '.join(reconstructed.split())}
            if exact:
                assert all(answer[s['start']:s['end']]==s['text'] and s['end']>s['start'] for s in spans)
                counts['exact_span_alignment_rows']+=1
                counts['rows_with_nonempty_risk_spans']+=bool(spans)
                types.update(s['type'] for s in spans)
            else:counts['failed_exact_alignment']+=1
            no_delete.update(n['type'] for n in nodes if not n['has_explicit_deletion'])
            counts['rows_retaining_typed_region_without_deletion_in_edited_candidate']+=any(not n['has_explicit_deletion'] for n in nodes)
            r['suitable_for_direct_character_supervision_without_text_repair']=bool(exact)
        except (AssertionError,ValueError,KeyError) as exc:
            counts['parse_or_schema_failure']+=1;r.update(error=str(exc),suitable_for_direct_character_supervision_without_text_repair=False)
        rows.append(r)
    summary.update(sample_checks=dict(counts),sample_reference_counts=dict(refcounts),
        exact_aligned_sample_span_types=dict(types),sample_regions_without_explicit_deletion=dict(no_delete),
        sample_lengths={'mean_reference_characters':sum(x['reference_characters'] for x in lengths)/len(lengths),
            'mean_answer_characters':sum(x['answer_characters'] for x in lengths)/len(lengths)},
        findings=['Original pre-corruption diversified passage and article IDs are not separate released fields.',
            'Edited candidate is only a markup projection, not known original or certified error-free text.',
            'Fixed100 checks do not establish full30k alignment, semantic correctness, or source independence.'],
        existing_fit_cal_modified=False,GPU_used=False,trained=False,evaluation_data_opened=False)
    summary.pop('row_records')
    with (ROOT/'SAMPLE100_INSPECTION.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (ROOT/'SCHEMA_ALIGNMENT_REPORT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':inspect()
