"""Export all fixed FAVA single-place silver repair pairs. CPU data preparation only."""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, json, re, time

OUT=Path(__file__).resolve().parent; QA=OUT.parent
UP=QA/'auxiliary_fava_v2'; OLD=QA/'research/fact_pair_data_feasibility_v1'
COUNTS=QA/'research/fava_local_relative_repair_v1'
spec=importlib.util.spec_from_file_location('existing_fava_local_markup',OLD/'audit_fava_pairs.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
TAG=re.compile(r'<(/?)([A-Za-z_]+)>')
INPUT_KEYS={'retrieved_passages','question','response'}

def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def read(p):return json.loads(Path(p).read_text('utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines()]
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(name,xs):
    with (OUT/name).open('w',encoding='utf-8',newline='\n') as f:
        for x in xs:f.write(json.dumps(x,ensure_ascii=False)+'\n')

def input_paths():
    return [Path(__file__),a.RAW,a.CAND,Path(a.inspect.__file__),OLD/'audit_fava_pairs.py',
            UP/'complete.json',UP/'manifest.json',UP/'DATA_PROTOCOL.json',UP/'SOURCE_ISOLATION_REPORT.json',
            UP/'candidate_material_group_index.jsonl',UP/'fixed_candidate_index.jsonl',
            COUNTS/'PROTOCOL.json',COUNTS/'COUNTS.json',COUNTS/'ROW_COUNT_INDEX.jsonl',COUNTS/'complete.json']

def sources():
    manifest=read(UP/'manifest.json');complete=read(UP/'complete.json')
    assert complete['manifest_sha256']==sha(UP/'manifest.json') and complete['answers']==manifest['answers']==7482
    for name in ('candidate_fit.jsonl','DATA_PROTOCOL.json','SOURCE_ISOLATION_REPORT.json',
                 'candidate_material_group_index.jsonl','fixed_candidate_index.jsonl'):
        assert sha(UP/name)==manifest['artifacts_sha256'][name],name
    assert sha(a.RAW)==manifest['sources_sha256'][str(a.RAW.resolve())]
    old=read(COUNTS/'COUNTS.json');assert old['counts']['local_replacements']==10040
    assert old['by_type']['entity']['replacements']==5335 and old['by_type']['relation']['replacements']==4705
    return {str(p.relative_to(QA)):sha(p) for p in input_paths()}

def protocol():
    return {
        'version':'fava-local-silver-repair-candidates-v1','fixed_count':10040,'original_answers':5275,'material_groups':5238,
        'selection':'Exactly preceding10040 top-level entity/relation nodes, one mark+one delete, no descendant type, unequal nonempty sides each1..3 whitespace words. No selection by model score, QA outcome or semantic judgment.',
        'construction':'Use original erroneous answer y and markup-coordinate [s,e); produce only y[:s]+repair+y[e:]. Other marked or unmarked errors stay verbatim. Check prefix/suffix and exact reverse. Never use the whole edited projection.',
        'model_inputs':'Two separate rows per pair. Each model_inputs has ONLY retrieved_passages (unchanged original numbered reference text), question empty, response ONE version. No target, other version, markup, completion, original checking instructions, gold answer or generator metadata in model_inputs.',
        'targets':'Independent Python Unicode-codepoint half-open char ranges in each response. Original author-marked error and author-suggested preferred repair are silver targets, not certified truth. Outside both local ranges unknown; no whole-answer labels or full repaired token labels.',
        'provenance':'Original source revision/raw_index/response_id/source_id/group_id/prompt+completion SHA; typed node preorder ordinal and raw completion markup bounds+SHA. No first-str.find lookup.',
        'eligibility':'Retain every10040 pair. Structural unusable flags: all evidence bodies empty/whitespace, all evidence bodies without alnum, either target without alnum, empty answer. Partial empty reference sections are warnings only when other evidence exists. Unicode str.isalnum used for pre-tokenization structural checks, not a tokenizer eligibility guarantee.',
        'isolation':'Inherit frozen auxiliary_fava_v2 exact nonempty SHA/20-word and material-component quarantine. Bind complete/manifest/source-isolation report and per-row group membership. No renewed QA material scan and no QA gold/cal/test reads; inherited limitations remain.',
        'weights_suggestion':'Retain original group IDs. Equal participating material groups, then original answers within group, then pairs within original answer: 1/(G*A_g*P_a). Both variants are one pair unit. This is a candidate base-weight suggestion, not final loss/class calibration. If structural pairs later excluded, recompute participating fit-only denominators explicitly.',
        'training_comparison_plan':'Same local pair data for local silver BCE versus local silver BCE+relative ranking; no whole corrected-answer negative. Fix optimizer/budget/material split before future training. This preparation trains neither.',
        'operations':{'CPU_only':True,'tokenize':False,'training':False,'GPU':False,'QA_gold_cal_test':False,'old_data_modified':False},
        'sources_sha256':sources()}

def design():
    p=protocol();assert not (OUT/'protocol.json').exists(),'Do not overwrite a frozen design'
    save('protocol.json',p);save('design_complete.json',{'protocol_sha256':sha(OUT/'protocol.json'),'status':'prepared_design_not_exported'})
    print('FAVA_LOCAL_PAIRS_DESIGN_COMPLETE',flush=True)

def node_spans(completion):
    stack=[];result=[]
    for m in TAG.finditer(completion):
        closing,tag=m.groups();assert tag in a.inspect.TAGS
        if not closing:stack.append((tag,m.start()))
        else:
            old,start=stack.pop();assert old==tag
            if tag in a.inspect.TYPES:result.append({'type':tag,'start':start,'end':m.end()})
    assert not stack
    return sorted(result,key=lambda x:x['start'])

def eligible_nodes(completion):
    tree=a.tree(completion);allnodes=a.collect(tree);positions=node_spans(completion)
    assert len(allnodes)==len(positions)
    selected=[]
    for ordinal,(n,p) in enumerate(zip(allnodes,positions)):
        assert n['type']==p['type']
        if n['nested_typed'] or n['type'] not in {'entity','relation'}:continue
        if not n['bad'] or not n['repair'] or n['bad']==n['repair']:continue
        if n['mark_count']!=1 or n['delete_count']!=1 or n['typed_descendants']!=0:continue
        if not (1<=len(n['bad'].split())<=3 and 1<=len(n['repair'].split())<=3):continue
        selected.append((ordinal,n,p))
    return tree,selected

def eligibility(refs,original,repair,bad):
    reasons=[];warnings=[]
    empties=[j+1 for j,s in enumerate(refs) if not s.strip()]
    if empties:warnings.append('one_or_more_empty_reference_bodies')
    if len(empties)==len(refs):reasons.append('all_reference_bodies_empty_or_whitespace')
    if not any(c.isalnum() for s in refs for c in s):reasons.append('all_reference_bodies_without_alnum')
    if not any(c.isalnum() for c in bad):reasons.append('original_target_without_alnum')
    if not any(c.isalnum() for c in repair):reasons.append('preferred_target_without_alnum')
    if not original.strip():reasons.append('original_response_empty_or_whitespace')
    return {'structural_eligible':not reasons,'reasons':reasons,'warnings':warnings,'empty_reference_ordinals':empties,
            'not_semantic_truth_verification':True,'not_tokenizer_coverage_verification':True}

def export():
    tick=time.perf_counter();assert read(OUT/'protocol.json')==protocol()
    assert not (OUT/'export_started.json').exists(),'Do not overwrite an interrupted/finished export'
    save('export_started.json',{'protocol_sha256':sha(OUT/'protocol.json')})
    upstream_candidate_hash=sha(a.CAND);upstream_manifest_hash=sha(UP/'manifest.json')
    upstream_isolation_hash=sha(UP/'SOURCE_ISOLATION_REPORT.json')
    raw=read(a.RAW);candidates=lines(a.CAND);assert len(raw)==30073 and len(candidates)==7482
    original_index={r['raw_index']:r for r in lines(UP/'fixed_candidate_index.jsonl')}
    groups={r['group_id']:r for r in lines(UP/'candidate_material_group_index.jsonl')}
    expected={r['raw_index']:r for r in lines(COUNTS/'ROW_COUNT_INDEX.jsonl')}
    pairs=[];inputs=[];provenance=[];pairs_by_answer=collections.defaultdict(list);answers_by_group=collections.defaultdict(set)
    checks=collections.Counter();types=collections.Counter();structural_reasons=collections.Counter();warnings=collections.Counter()
    empty_ids=set();changed_char=collections.Counter();node_refs=set()
    for r in candidates:
        ri=r['raw_index'];d=raw[ri];fixed=original_index[ri];group=groups[r['group_id']]
        assert not fixed['quarantined'] and not group['quarantined']
        for key in ('response_id','source_id','group_id','completion_sha256','original_prompt_sha256'):assert fixed[key]==r[key]
        assert ri in group['raw_indices'] and r['response_id'] in group['response_ids']
        assert digest(d['prompt'])==r['original_prompt_sha256'] and digest(d['completion'])==r['completion_sha256']
        refs,answer=a.inspect.prompt_parts(d['prompt'])
        assert len(refs)==5 and answer==r['original_response'] and digest(answer)==r['answer_sha256']
        assert [digest(s) for s in refs]==r['reference_text_sha256']
        lo,hi=r['evidence_original_prompt_range'];assert d['prompt'][lo:hi]==r['retrieved_passages']
        tree,nodes=eligible_nodes(d['completion']);assert a.project(tree,'corrupt')==answer
        if not nodes:assert ri not in expected;continue
        actual=collections.Counter(n['type'] for _,n,_ in nodes)
        assert actual['entity']==expected[ri]['entity_replacements'] and actual['relation']==expected[ri]['relation_replacements']
        assert expected[ri]['group_id']==r['group_id'] and expected[ri]['response_id']==r['response_id']
        for ordinal,n,p in nodes:
            s,e=n['start'],n['end'];repair=n['repair'];bad=n['bad'];assert answer[s:e]==bad
            patched=answer[:s]+repair+answer[e:];pe=s+len(repair)
            assert patched[:s]==answer[:s] and patched[pe:]==answer[e:] and patched[s:pe]==repair
            assert patched[:s]+bad+patched[pe:]==answer
            pid=f"{r['response_id']}__typed_{ordinal:03d}";oi=pid+'__original';pi=pid+'__preferred'
            assert (ri,ordinal) not in node_refs;node_refs.add((ri,ordinal))
            versions=[]
            for iid,text in ((oi,answer),(pi,patched)):
                mi={'retrieved_passages':r['retrieved_passages'],'question':'','response':text}
                assert set(mi)==INPUT_KEYS and all(isinstance(v,str) for v in mi.values())
                # Reserved FAVA annotation tags may not enter either text input.
                assert not any(m.group(2) in a.inspect.TAGS for value in mi.values() for m in TAG.finditer(value))
                inputs.append({'input_id':iid,'model_inputs':mi});versions.append(digest(canonical(mi)))
            status=eligibility(refs,answer,repair,bad)
            if status['empty_reference_ordinals']:empty_ids.add(ri)
            structural_reasons.update(status['reasons']);warnings.update(status['warnings'])
            pairs.append({'pair_id':pid,'source_response_id':r['response_id'],'raw_index':ri,'source_id':r['source_id'],'group_id':r['group_id'],
                'partition':'auxiliary_candidate_fit','official_split':'train','type':n['type'],'synthetic':True,'human_gold':False,
                'original':{'input_id':oi,'response_sha256':digest(answer),'model_inputs_sha256':versions[0],
                            'target':{'start':s,'end':e,'text':bad,'meaning':'author_marked_error_silver'}},
                'preferred':{'input_id':pi,'response_sha256':digest(patched),'model_inputs_sha256':versions[1],
                             'target':{'start':s,'end':pe,'text':repair,'meaning':'author_suggested_noisy_local_repair'}},
                'outside_targets':'unknown_no_new_supervision','preference_is_factual_certificate':False,
                'checks':{'original_slice_exact':True,'preferred_slice_exact':True,'outside_prefix_suffix_exact':True,'reverse_exact':True},
                'eligibility':status,'provenance_id':pid})
            provenance.append({'provenance_id':pid,'pair_id':pid,'raw_index':ri,'source_dataset':r['source_dataset'],'source_revision':r['source_revision'],
                'source_response_id':r['response_id'],'source_id':r['source_id'],'group_id':r['group_id'],
                'original_prompt_sha256':r['original_prompt_sha256'],'completion_sha256':r['completion_sha256'],
                'typed_node_preorder_index':ordinal,'completion_node_char_span':[p['start'],p['end']],
                'completion_node_sha256':digest(d['completion'][p['start']:p['end']]),'author_type':n['type'],
                'reference_text_sha256':r['reference_text_sha256'],'reference_empty_ordinals':status['empty_reference_ordinals'],
                'upstream_candidate_sha256':upstream_candidate_hash,'upstream_manifest_sha256':upstream_manifest_hash,
                'source_isolation':'inherited_from_frozen_auxiliary_fava_v2_not_rescanned',
                'source_isolation_report_sha256':upstream_isolation_hash,
                'whole_answer_correctness_label_created':False})
            pairs_by_answer[r['response_id']].append(pid);answers_by_group[r['group_id']].add(r['response_id'])
            checks['exact_reversible_pairs']+=1;types[n['type']]+=1
            changed_char[f"{len(bad)}->{len(repair)}"]+=1
    assert len(pairs)==10040 and len(inputs)==20080 and len(pairs_by_answer)==5275 and len(answers_by_group)==5238
    assert types=={'entity':5335,'relation':4705}
    assert len({x['input_id'] for x in inputs})==20080
    group_mass=collections.Counter();answer_mass=collections.Counter()
    for p in pairs:
        ag=len(answers_by_group[p['group_id']]);pa=len(pairs_by_answer[p['source_response_id']]);den=len(answers_by_group)*ag*pa
        p['suggested_base_weight']={'numerator':1,'denominator':den,'value':1/den,
                                    'participating_groups':5238,'original_answers_in_group':ag,'pairs_in_original_answer':pa,
                                    'scope':'all_candidate_pairs_not_final_training_loss'}
        group_mass[p['group_id']]+=1/den;answer_mass[p['source_response_id']]+=1/den
    assert abs(sum(group_mass.values())-1)<1e-10 and max(abs(x-1/5238) for x in group_mass.values())<1e-15
    answer_rows=[{'source_response_id':rid,'raw_index':int(rid.removeprefix('fava_train_')),'pair_ids':ids,'pairs':len(ids),
                  'suggested_base_mass':answer_mass[rid]} for rid,ids in pairs_by_answer.items()]
    group_rows=[{'group_id':gid,'source_response_ids':sorted(rids),'original_answers':len(rids),
                 'pairs':sum(len(pairs_by_answer[r]) for r in rids),'suggested_base_mass':group_mass[gid],
                 'inherited_conservative_material_component':True,'entity_or_event_independence_not_claimed':True} for gid,rids in sorted(answers_by_group.items())]
    for name,xs in [('candidate_pairs.jsonl',pairs),('model_inputs.jsonl',inputs),('provenance.jsonl',provenance),
                    ('original_answer_index.jsonl',answer_rows),('material_group_index.jsonl',group_rows)]:rows(name,xs)
    quality={'pairs':len(pairs),'input_rows':len(inputs),'original_answers':len(pairs_by_answer),'material_groups':len(answers_by_group),
        'types':dict(types),'exact_checks':dict(checks),'structurally_eligible':sum(p['eligibility']['structural_eligible'] for p in pairs),
        'structurally_ineligible':sum(not p['eligibility']['structural_eligible'] for p in pairs),
        'eligibility_reasons':dict(structural_reasons),'warnings':dict(warnings),'original_raw_ids_with_empty_reference_section':sorted(empty_ids),
        'max_pairs_in_original_answer':max(map(len,pairs_by_answer.values())),
        'max_participating_original_answers_in_group':max(map(len,answers_by_group.values())),
        'max_pairs_in_group':max(g['pairs'] for g in group_rows),'suggested_base_total_mass':sum(group_mass.values()),
        'all_rows_retained':True,'source_isolation_inherited_not_rescanned':True,'new_whole_answer_labels':False,
        'GPU_used':False,'tokenizer_loaded':False,'trained':False,'QA_gold_cal_test_read':False,'seconds':time.perf_counter()-tick}
    save('QUALITY_REPORT.json',quality)
    assert sources()==read(OUT/'protocol.json')['sources_sha256']
    names=['candidate_pairs.jsonl','model_inputs.jsonl','provenance.jsonl','original_answer_index.jsonl','material_group_index.jsonl','QUALITY_REPORT.json','protocol.json','design_complete.json','export_started.json']
    manifest={'status':'all_local_silver_candidates_exported_pending_independent_quality_check',
        **{k:quality[k] for k in ('pairs','input_rows','original_answers','material_groups','types','structurally_eligible','structurally_ineligible')},
        'artifacts_sha256':{n:sha(OUT/n) for n in names},'protocol_sha256':sha(OUT/'protocol.json'),
        'source_isolation_inherited':True,'source_isolation_rescanned':False,'upstream_manifest_sha256':sha(UP/'manifest.json'),
        'upstream_complete_sha256':sha(UP/'complete.json'),'whole_answer_gold_created':False,'outside_targets_unknown':True,
        'GPU_used':False,'tokenized':False,'trained':False,'QA_gold_cal_test_read':False,'old_files_changed':False}
    save('manifest.json',manifest);save('export_complete.json',{'manifest_sha256':sha(OUT/'manifest.json'),'status':'export_complete_not_tokenized_or_trained'})
    print(json.dumps(quality,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['design','export']);args=parser.parse_args()
    globals()[args.stage]()
