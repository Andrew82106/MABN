"""Read-only feasibility census. Outputs diagnostics, not a paired training dataset."""
from pathlib import Path
import hashlib,importlib.util,json,re
from collections import Counter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RAW=ROOT/'additional_data/fava_training/training.json'
CAND=ROOT/'auxiliary_fava_v2/candidate_fit.jsonl'
spec=importlib.util.spec_from_file_location('fixed_fava_inspection',ROOT/'additional_data/fava_training/inspect_training.py')
inspect=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspect)

def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def norm(s):return ' '.join(s.split())

def tree(text):
    root={'tag':'root','children':[]};stack=[root];end=0
    for m in re.finditer(r'<(/?)([A-Za-z_]+)>',text):
        stack[-1]['children'].append(text[end:m.start()]);closing,tag=m.groups()
        assert tag in inspect.TAGS
        if closing:assert stack[-1]['tag']==tag;stack.pop()
        else:
            node={'tag':tag,'children':[]};stack[-1]['children'].append(node);stack.append(node)
        end=m.end()
    stack[-1]['children'].append(text[end:]);assert len(stack)==1
    return root

def project(node,side):
    if isinstance(node,str):return node
    if node['tag']==('mark' if side=='corrupt' else 'delete'):return ''
    return ''.join(project(x,side) for x in node['children'])

def collect(node,types_inside=False,offset=0):
    results=[]
    if isinstance(node,str):return results
    if node['tag'] in inspect.TYPES:
        bad=project(node,'corrupt');good=project(node,'edited')
        def tags(n):
            if isinstance(n,str):return []
            return [n['tag']]+[t for child in n['children'] for t in tags(child)]
        alltags=tags(node)
        results.append({'type':node['tag'],'start':offset,'end':offset+len(bad),
                        'bad':bad,'repair':good,'mark_count':alltags.count('mark'),'delete_count':alltags.count('delete'),
                        'nested_typed':types_inside,'typed_descendants':sum(t in inspect.TYPES for t in alltags)-1})
        types_inside=True
    for child in node['children']:
        results.extend(collect(child,types_inside,offset));offset+=len(project(child,'corrupt'))
    return results

def run():
    protocol={'scope':'All and only7482 frozen candidate rows. Read official30073 training.json and code; no cal/test/model scores, no training/GPU.',
      'questions':['Does an explicit diversified/original answer field survive release?','Are marked corruption coordinates deterministic?','Do edits supply replacements or only deletion/bare tags?','Is whole repaired text literally contained in a reference?'],
      'semantics':'Edited projection is not automatically an original uncorrupted answer or a factuality certificate. String inclusion is reported as a narrow textual check, absence is not proof of contradiction. No output is a new label/training pair.',
      'uniqueness':'Coordinates follow markup traversal, never a first str.find. Count node nesting/number of edit tags separately; no claim of uniquely minimal Levenshtein edit or original-history identity.',
      'sources_sha256':{'raw_training':file_sha(RAW),'fixed_candidate_file':file_sha(CAND),'existing_parser':file_sha(Path(inspect.__file__))}}
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text('utf-8'))==protocol
    else:write(OUT/'PROTOCOL.json',protocol)
    data=json.loads(RAW.read_text('utf-8'));candidates=[json.loads(s) for s in CAND.read_text('utf-8').splitlines()]
    assert len(data)==30073 and len(candidates)==7482
    schema=Counter(tuple(sorted(r)) for r in data);counts=Counter();node_counts=Counter();by_type={};records=[];examples=[]
    for r in candidates:
        d=data[r['raw_index']];refs,answer=inspect.prompt_parts(d['prompt'])
        corrupt,edited,spans,_=inspect.parse_markup(d['completion'])
        assert answer==corrupt==r['original_response']
        assert r['original_prompt_sha256']==sha(d['prompt']) and r['completion_sha256']==sha(d['completion'])
        tr=tree(d['completion']);assert project(tr,'corrupt')==corrupt and project(tr,'edited')==edited
        raw_nodes=collect(tr)
        counts['empty_typed_tags']+=sum(not n['bad'] and not n['repair'] for n in raw_nodes)
        nodes=[n for n in raw_nodes if n['bad'] or n['repair']]
        assert all(corrupt[n['start']:n['end']]==n['bad'] for n in nodes)
        assert {n['type'] for n in nodes}<={'entity','relation','contradictory','invented'}
        top=[n for n in nodes if not n['nested_typed']]
        patch=[];cursor=0
        for n in top:
            assert cursor<=n['start']
            patch.extend([corrupt[cursor:n['start']],n['repair']]);cursor=n['end']
        patch.append(corrupt[cursor:])
        confined=''.join(patch)==edited
        changed=[n for n in top if n['bad']!=n['repair']]
        unedited=[n for n in top if n['bad']==n['repair']]
        replacements=[n for n in top if n['bad'] and n['repair'] and n['bad']!=n['repair']]
        deletions=[n for n in top if n['bad'] and not n['repair']]
        additions=[n for n in top if not n['bad'] and n['repair']]
        simple=[n for n in replacements if n['mark_count']==n['delete_count']==1 and n['typed_descendants']==0]
        local=[n for n in simple if n['type'] in {'entity','relation'} and 1<=len(n['bad'].split())<=3 and 1<=len(n['repair'].split())<=3]
        exact=[i+1 for i,s in enumerate(refs) if norm(edited) and edited in s]
        normalized=[i+1 for i,s in enumerate(refs) if norm(edited) and norm(edited) in norm(s)]
        counts['rows']+=1;counts['markup_bad_projection_exact']+=1
        for k,v in {'has_changed_projection':edited!=corrupt,'edited_projection_empty_or_whitespace':not norm(edited),'has_replacement':bool(replacements),'has_deletion':bool(deletions),'has_bare_or_noop_error':bool(unedited),
                    'all_top_errors_change':not unedited,'one_changed_type_node':len(changed)==1,'exactly_one_type_node':len(top)==1,
                    'single_type_replacement':len(top)==1 and bool(simple),'has_simple_mark_delete_pair':bool(simple),
                    'has_entity_relation_1to3word_pair':bool(local),
                    'all_projection_changes_confined_to_typed_nodes':confined,
                    'single_entity_relation_1to3word_pair_only':len(top)==1 and bool(local) and confined,
                    'has_nested_type':any(n['nested_typed'] for n in nodes),'whole_repair_exact_ref_substring':bool(exact),
                    'whole_repair_whitespace_normalized_ref_substring':bool(normalized),
                    'has_preserved_original_field':any(k not in {'prompt','completion'} for k in d)}.items():counts[k]+=int(v)
        for n in nodes:
            typ=n['type'];by_type.setdefault(typ,Counter());by_type[typ]['nodes']+=1
            if n['bad']==n['repair']:cat='no_change_bare_or_noop'
            elif not n['repair']:cat='deletion_only'
            elif not n['bad']:cat='insertion_only'
            else:cat='replacement'
            node_counts[cat]+=1;by_type[typ][cat]+=1
        node_counts['simple_one_mark_one_delete_replacements']+=len(simple)
        node_counts['entity_relation_1to3word_replacements']+=len(local)
        rec={'response_id':r['response_id'],'raw_index':r['raw_index'],'group_id':r['group_id'],
            'original_answer_field_present':False,'corrupt_projection_exact':True,'edited_projection_sha256':sha(edited),
            'typed_nodes':len(nodes),'top_typed_nodes':len(top),'changed_top_nodes':len(changed),'replacements':len(replacements),
            'deletions':len(deletions),'insertions':len(additions),'bare_or_noop_top_nodes':len(unedited),
            'simple_mark_delete_pairs':len(simple),'entity_relation_1to3word_pairs':len(local),
            'all_projection_changes_confined_to_typed_nodes':confined,
            'whole_repair_exact_ref_indices':exact,'whole_repair_whitespace_norm_ref_indices':normalized,
            'independent_original_identity_verified':False,'whole_answer_support_human_verified':False,
            'no_new_training_labels':True}
        records.append(rec)
        if r['raw_index'] in {6,12,24}:
            examples.append({'raw_index':r['raw_index'],'response_id':r['response_id'],'references':refs,'released_corrupt_answer':corrupt,
                             'edited_projection_not_certified_original':edited,'typed_nodes':nodes})
    (OUT/'PAIRABILITY_INDEX.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),'utf-8')
    write(OUT/'TRAIN_ONLY_EXAMPLES.json',examples)
    summary={'status':'complete_readonly_feasibility_census','raw_schemas':{','.join(k):v for k,v in schema.items()},'counts':dict(counts),
      'node_counts':dict(node_counts),'by_type':{k:dict(v) for k,v in by_type.items()},
      'original_uncorrupted_answer_identity_verified_rows':0,'whole_answer_semantic_support_verified_rows':0,
      'zero_meaning':'0 have a retained independently authenticated original field or an all-facts support certificate in this release; not a claim all repairs are false.',
      'protocol_sha256':file_sha(OUT/'PROTOCOL.json'),'artifacts_sha256':{n:file_sha(OUT/n) for n in ['PAIRABILITY_INDEX.jsonl','TRAIN_ONLY_EXAMPLES.json']},
      'trained':False,'GPU_used':False,'official_test_read':False,'calibration_read':False,'formal_labels_or_pairs_created':False}
    write(OUT/'FAVA_PAIR_COUNTS.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':run()
