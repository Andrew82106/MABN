"""Exact character census of fixed official FAVA synthetic TRAINING only."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import re
import time

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'fava_training'
RAW=SOURCE/'training.json'
EXPECTED_SHA='5f6422b13b48b3f6fdff7bab4698b3702c90154c10cad519047f58600e6536a7'
FACTUAL={'entity','relation','invented','contradictory'}
OTHER={'subjective','unverifiable'}
spec=importlib.util.spec_from_file_location('fixed_fava_parser',SOURCE/'inspect_training.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def emit(stream,row):stream.write(json.dumps(row,ensure_ascii=False)+'\n')
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(x):return old.sha(json.dumps(x,ensure_ascii=False,separators=(',',':')))


def protocol():
    return {'version':'fava-full-training-exact-character-census-v1',
        'repository':'fava-uw/fava-data','revision':'f4e40415d525b18bcb49ba241f26fd8e12eeb606',
        'population':30073,'input_bytes':137320668,'input_sha256':EXPECTED_SHA,
        'scope':'Existing official training.json only, all rows in original order. No download, model, GPU, training, current fit/cal mutation or human evaluation/test reads.',
        'parser':'Reuse unchanged inspect_training.py prompt_parts and parse_markup. No whitespace normalization, fuzzy matching, tag repair, answer rewriting or new parser heuristics for acceptance.',
        'acceptance':'Prompt and known balanced markup parse; reconstructed corrupted answer equals original prompt answer character-for-character; every nonempty span bounds/text matches the answer. Positions are Python Unicode codepoint half-open offsets, not tokenizer offsets.',
        'factual_candidate_types':sorted(FACTUAL),'separately_reported_other_types':sorted(OTHER),
        'semantic_limit':'All original labels stay synthetic and retain their original six types. Exact alignment does not verify facts or annotation completeness. These four factual candidates are not automatically converted to public QA human classes.',
        'negative_limit':'No-span rows are only unmarked, not certified negatives. The edited projection is never exported as a clean training example or treated as pre-corruption original truth. Unmarked answer positions are not certified true tokens.',
        'failure_retention':'Every failed record retains raw index, original prompt/completion, raw type/tag counts, stage/category/reason and source hashes when available. Mismatch may report whitespace-normalized equality for diagnosis only, never acceptance.',
        'grouping':'Per-reference SHA256 in original order; unordered exact reference-multiset hash and member lists; separate shared-reference-block members. Preserve duplicates in the multiset. No split creation or connected-component merging. These are conservative grouping proxies; shared retrieval distractors do not identify original article IDs.',
        'outputs':['rows.jsonl','failures.jsonl','reference_groups.jsonl','shared_reference_blocks.jsonl','summary.json','SAMPLE100_AGREEMENT.json','complete.json'],
        'crosscheck':'Verify all100 previously sampled outcomes and exact aligned spans against unchanged SAMPLE100_INSPECTION.jsonl.',
        'existing_fit_cal_modified':False,'GPU_used':False,'trained':False,'evaluation_data_opened':False}


def sources():
    paths=[Path(__file__),RAW,SOURCE/'inspect_training.py',SOURCE/'MANIFEST.json',
        SOURCE/'DOWNLOAD_PROVENANCE.json',SOURCE/'SAMPLE100_INSPECTION.jsonl',SOURCE/'SAMPLE_INDICES.json']
    return {str(p.resolve()):file_sha(p) for p in paths}


def design():
    assert not (OUT/'protocol.json').exists()
    assert RAW.stat().st_size==137320668 and file_sha(RAW)==EXPECTED_SHA
    assert old.TYPES==FACTUAL|OTHER
    original=read(SOURCE/'MANIFEST.json')
    for n in ('inspect_training.py','SAMPLE100_INSPECTION.jsonl','SAMPLE_INDICES.json'):
        assert file_sha(SOURCE/n)==original['files_sha256'][n]
    save(OUT/'protocol.json',protocol())
    save(OUT/'design_freeze.json',{'protocol_sha256':file_sha(OUT/'protocol.json'),
        'source_sha256':sources(),'full_census_started':False,'trained':False,'GPU_used':False})
    print('FAVA_FULL_CENSUS_PROTOCOL_FROZEN',flush=True)


def check():
    frozen=read(OUT/'design_freeze.json')
    assert read(OUT/'protocol.json')==protocol()
    assert frozen['protocol_sha256']==file_sha(OUT/'protocol.json') and frozen['source_sha256']==sources()
    return frozen


def intervals_length(spans):
    pairs=sorted((s['start'],s['end']) for s in spans)
    end=-1;total=0
    for a,b in pairs:
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def failure_category(stage,exc):
    s=str(exc)
    if stage=='schema':return 'schema_or_field_type'
    if stage=='prompt':
        if 'prompt boundary count=' in s:return 'prompt_boundary_count'
        if 'No reference blocks' in s:return 'missing_reference_blocks'
        if isinstance(exc,AssertionError):return 'nonsequential_reference_numbering'
        return 'other_prompt_parse'
    if stage=='markup':
        if s.startswith('unbalanced close '):return 'unbalanced_closing_tag'
        if s.startswith('unclosed tags:'):return 'unclosed_tags'
        if s.startswith('unknown markup tags:'):return 'unknown_markup_tags'
        return 'other_markup_parse'
    return 'span_bounds_or_text_validation'


def run():
    frozen=check();assert not (OUT/'started.json').exists()
    save(OUT/'started.json',{'time':time.time(),'design_freeze_sha256':file_sha(OUT/'design_freeze.json')})
    tick=time.perf_counter();data=read(RAW);assert isinstance(data,list) and len(data)==30073
    sample={r['raw_index']:r for r in map(json.loads,(SOURCE/'SAMPLE100_INSPECTION.jsonl').read_text(encoding='utf-8').splitlines())}
    group_rows=defaultdict(list);group_hashes={};block_members=defaultdict(list)
    counters=Counter();failures=Counter();schemas=Counter();ref_counts=Counter();raw_open=Counter();raw_close=Counter()
    span_types=Counter();node_types=Counter();typed_without_delete=Counter();row_types=Counter();failure_raw_types=Counter()
    samples_checked=[];all_prompts=Counter();all_completions=Counter();mismatch_kinds=Counter();characters=Counter()
    with (OUT/'rows.jsonl').open('w',encoding='utf-8') as rows_file,(OUT/'failures.jsonl').open('w',encoding='utf-8') as failure_file:
        for i,row in enumerate(data):
            r={'raw_index':i,'synthetic_not_human_gold':True,'label_semantics_verified':False,
               'exact_character_alignment':False,'edited_projection_is_clean_gold':False,
               'unmarked_text_is_verified_negative':False}
            stage='schema';failed=False
            if isinstance(row,dict):
                schemas['|'.join(sorted(row))]+=1
                for field in ('prompt','completion'):
                    if isinstance(row.get(field),str):r[field+'_sha256']=old.sha(row[field])
                completion=row.get('completion','')
                if isinstance(completion,str):
                    opens=Counter();closes=Counter()
                    for slash,name in re.findall(r'<(/?)([A-Za-z_]+)>',completion):
                        (closes if slash else opens)[name]+=1
                    r['raw_markup_openings']=dict(opens);r['raw_markup_closings']=dict(closes)
                    raw_open.update(opens);raw_close.update(closes)
            else:schemas[type(row).__name__]+=1
            try:
                assert isinstance(row,dict) and set(row)=={'prompt','completion'}
                assert isinstance(row['prompt'],str) and isinstance(row['completion'],str)
                all_prompts[r['prompt_sha256']]+=1;all_completions[r['completion_sha256']]+=1
                stage='prompt';refs,answer=old.prompt_parts(row['prompt'])
                hashes=[old.sha(text) for text in refs];multiset=sorted(hashes);gid=digest(multiset)
                group_rows[gid].append(i);group_hashes[gid]=multiset
                for ordinal,h in enumerate(hashes):block_members[h].append([i,ordinal+1])
                r.update(reference_text_sha256=hashes,reference_set_group_sha256=gid,
                    answer_sha256=old.sha(answer),answer_characters=len(answer),reference_characters=sum(map(len,refs)))
                ref_counts[len(refs)]+=1;counters['prompt_parsed']+=1
                stage='markup';reconstructed,edited,spans,nodes=old.parse_markup(row['completion'])
                counters['balanced_known_markup']+=1
                r.update(reconstructed_answer_sha256=old.sha(reconstructed),
                    edited_projection_sha256=old.sha(edited),edited_projection_characters=len(edited),
                    typed_regions=nodes,parsed_spans_before_acceptance=spans)
                for n in nodes:
                    node_types[n['type']]+=1
                    if not n['has_explicit_deletion']:typed_without_delete[n['type']]+=1
                if any(not n['has_explicit_deletion'] for n in nodes):counters['rows_typed_region_without_explicit_deletion']+=1
                if reconstructed!=answer:
                    failed=True;r['failure_stage']='roundtrip';r['failure_category']='answer_character_mismatch'
                    first=next((j for j,(a,b) in enumerate(zip(answer,reconstructed)) if a!=b),min(len(answer),len(reconstructed)))
                    normalized=' '.join(answer.split())==' '.join(reconstructed.split())
                    r['failure_reason']='Reconstructed corrupted answer is not character-identical to prompt answer.'
                    r['mismatch']={'first_character':first,'answer_characters':len(answer),'reconstructed_characters':len(reconstructed),
                        'prompt_context':answer[max(0,first-30):first+60],
                        'reconstructed_context':reconstructed[max(0,first-30):first+60],
                        'whitespace_normalized_equal_diagnostic_only':normalized}
                    mismatch_kinds['whitespace_normalized_equal' if normalized else 'other_content_difference']+=1
                else:
                    stage='span_validation'
                    assert all(0<=s['start']<s['end']<=len(answer) and answer[s['start']:s['end']]==s['text'] and s['type'] in old.TYPES for s in spans)
                    r['exact_character_alignment']=True;r['aligned_synthetic_spans']=spans
                    factual=[s for s in spans if s['type'] in FACTUAL];other=[s for s in spans if s['type'] in OTHER]
                    r.update(factual_candidate_span_count=len(factual),other_type_span_count=len(other),
                        exact_factual_candidate_available=bool(factual),nonempty_span_count=len(spans))
                    r['alignment_class']=('factual_and_other' if factual and other else 'factual_only' if factual else 'other_only' if other else 'no_nonempty_spans')
                    counters['exact_aligned_rows']+=1;counters[r['alignment_class']]+=1
                    span_types.update(s['type'] for s in spans);row_types.update(set(s['type'] for s in spans))
                    all_union=intervals_length(spans);fact_union=intervals_length(factual);other_union=intervals_length(other)
                    r.update(all_span_union_characters=all_union,factual_union_characters=fact_union,
                        other_union_characters=other_union,between_type_group_overlap_characters=fact_union+other_union-all_union)
                    characters.update(answer=len(answer),all_risk=all_union,factual=fact_union,other=other_union,
                        between_groups_overlap=fact_union+other_union-all_union)
            except (AssertionError,ValueError,KeyError,TypeError) as exc:
                failed=True;r.update(failure_stage=stage,failure_category=failure_category(stage,exc),
                                     failure_reason=str(exc) or type(exc).__name__)
            if failed:
                counters['failed_rows']+=1;failures[r['failure_category']]+=1
                failure_raw_types.update({k:v for k,v in r.get('raw_markup_openings',{}).items() if k in old.TYPES})
                # Full originals are retained only in the failure ledger; the
                # immutable raw training array remains the source for every row.
                emit(failure_file,{**r,'original_record':row})
            if i in sample:
                prior=sample[i];assert r['prompt_sha256']==prior['prompt_sha256'] and r['completion_sha256']==prior['completion_sha256']
                assert r['exact_character_alignment']==prior['suitable_for_direct_character_supervision_without_text_repair']
                if r['exact_character_alignment']:
                    assert r['aligned_synthetic_spans']==prior['synthetic_error_spans']
                else:assert r['failure_reason']==prior['error']
                samples_checked.append(i)
            emit(rows_file,r)
            if (i+1)%5000==0:print('FAVA_FULL_ALIGNMENT_PROGRESS',i+1,len(data),flush=True)
    assert counters['exact_aligned_rows']+counters['failed_rows']==len(data)
    with (OUT/'reference_groups.jsonl').open('w',encoding='utf-8') as f:
        for gid,members in sorted(group_rows.items()):
            emit(f,{'reference_set_group_sha256':gid,'reference_hash_multiset':group_hashes[gid],
                    'raw_indices':members,'group_proxy_not_article_id':True})
    shared=0
    with (OUT/'shared_reference_blocks.jsonl').open('w',encoding='utf-8') as f:
        for h,members in sorted(block_members.items()):
            if len({a for a,b in members})>1:
                shared+=1;emit(f,{'reference_text_sha256':h,'members_raw_index_and_reference_ordinal':members,
                    'may_be_shared_distractor':True})
    assert samples_checked==sorted(sample) and len(samples_checked)==100
    save(OUT/'SAMPLE100_AGREEMENT.json',{'passed':True,'records':100,'raw_indices':samples_checked,
        'acceptance_and_exact_spans_unchanged':True,'old_file_sha256':file_sha(SOURCE/'SAMPLE100_INSPECTION.jsonl')})
    summary={'population':len(data),'input_sha256':EXPECTED_SHA,'schema_counts':dict(schemas),
        'counts':dict(counters),'failure_categories':dict(failures),'mismatch_diagnostics':dict(mismatch_kinds),
        'raw_markup_openings':dict(raw_open),'raw_markup_closings':dict(raw_close),
        'exact_aligned_span_types':dict(span_types),'exact_aligned_rows_by_type_nonexclusive':dict(row_types),
        'all_parsed_typed_regions_by_type':dict(node_types),
        'all_parsed_regions_without_explicit_deletion':dict(typed_without_delete),
        'failed_rows_raw_type_openings':dict(failure_raw_types),'reference_counts':dict(ref_counts),
        'aligned_character_counts':dict(characters),
        'factual_candidate_rows':counters['factual_only']+counters['factual_and_other'],
        'factual_candidate_spans':sum(span_types[t] for t in FACTUAL),
        'separate_other_type_spans':sum(span_types[t] for t in OTHER),
        'grouping':{'unique_prompts':len(all_prompts),'unique_completions':len(all_completions),
            'unique_unordered_reference_sets':len(group_rows),
            'extra_rows_sharing_exact_reference_set':sum(len(v)-1 for v in group_rows.values()),
            'largest_exact_set_group':max(map(len,group_rows.values())),
            'unique_reference_block_hashes':len(block_members),'reference_blocks_shared_across_rows':shared,
            'original_article_ids_recovered':False,'source_independence_proven':False,'new_split_created':False},
        'seconds':time.perf_counter()-tick,'sample100_agreement':True,
        'negative_examples_certified':0,'edited_candidates_promoted_to_negative':0,
        'synthetic_not_human_gold':True,'existing_fit_cal_modified':False,'GPU_used':False,'trained':False,'evaluation_data_opened':False}
    save(OUT/'summary.json',summary)
    assert frozen==check()
    names=protocol()['outputs'][:-1]
    save(OUT/'complete.json',{'status':'full_training_character_census_complete_not_admitted_to_training',
        'population':len(data),'files_sha256':{n:file_sha(OUT/n) for n in names},
        'design_freeze_sha256':file_sha(OUT/'design_freeze.json'),'GPU_used':False,'trained':False,'evaluation_data_opened':False})
    print('FAVA_FULL_CENSUS_COMPLETE',json.dumps({'counts':dict(counters),'failures':dict(failures),
        'factual_candidate_rows':summary['factual_candidate_rows'],'factual_candidate_spans':summary['factual_candidate_spans']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('design','check','run'));args=p.parse_args()
    try:globals()[args.stage]()
    except BaseException as exc:
        save(OUT/f'FAILURE_{args.stage}_{time.time_ns()}.json',{'error':repr(exc),'GPU_used':False,'trained':False})
        raise
