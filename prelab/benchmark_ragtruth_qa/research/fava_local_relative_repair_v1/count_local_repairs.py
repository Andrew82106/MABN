"""Train-only counts and reversible patch checks; does not export new training pairs."""
from pathlib import Path
import collections, hashlib, importlib.util, json

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
OLD = QA / 'research/fact_pair_data_feasibility_v1'
spec = importlib.util.spec_from_file_location('fava_existing_pair_audit', OLD / 'audit_fava_pairs.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    protocol = {
        'scope': 'All7482 fixed FAVA auxiliary candidates; only original train fields and existing parser.',
        'selection': 'Reuse previous top-level nonnested entity/relation nodes; exactly1 mark and1 delete; each projected side1..3 whitespace-delimited words; nonempty unequal sides; no descendant type.',
        'pair_check': 'Runtime only: replace one markup-coordinate slice in original corrupt answer, verify exact prefix/suffix and exact reverse patch. Never apply all edited projection changes.',
        'output': 'Counts and per-original-row count diagnostics only. No paired texts, new truth labels, tokenization, models, cal/test access.',
        'inputs_sha256': {str(p.relative_to(QA)):sha(p) for p in [a.RAW,a.CAND,OLD/'audit_fava_pairs.py',OLD/'FAVA_PAIR_COUNTS.json']}}
    save('PROTOCOL.json', protocol)
    raw=json.loads(a.RAW.read_text('utf-8'))
    rows=[json.loads(x) for x in a.CAND.read_text('utf-8').splitlines()]
    assert len(rows)==7482
    counts=collections.Counter(); bytype=collections.Counter()
    type_rows=collections.defaultdict(set); type_groups=collections.defaultdict(set)
    all_groups=set(); diagnostics=[]; other_faults=collections.Counter(); lengths=collections.Counter()
    for r in rows:
        original=r['original_response']; completion=raw[r['raw_index']]['completion']
        assert a.sha(completion)==r['completion_sha256']
        tree=a.tree(completion)
        assert a.project(tree,'corrupt')==original
        nodes=[n for n in a.collect(tree) if (n['bad'] or n['repair']) and not n['nested_typed']]
        eligible=[n for n in nodes if n['type'] in {'entity','relation'} and n['bad'] and n['repair']
                  and n['bad']!=n['repair'] and n['mark_count']==n['delete_count']==1 and n['typed_descendants']==0
                  and 1<=len(n['bad'].split())<=3 and 1<=len(n['repair'].split())<=3]
        if not eligible: continue
        counts['original_answers_with_local_replacements']+=1; all_groups.add(r['group_id'])
        rc=collections.Counter()
        for n in eligible:
            start,end=n['start'],n['end']; bad,repair=n['bad'],n['repair']
            assert original[start:end]==bad
            patched=original[:start]+repair+original[end:]
            new_end=start+len(repair)
            assert patched[:start]==original[:start] and patched[new_end:]==original[end:]
            assert patched[start:new_end]==repair
            assert patched[:start]+bad+patched[new_end:]==original
            counts['local_replacements']+=1; counts['exact_roundtrip']+=1
            bytype[n['type']]+=1; rc[n['type']]+=1
            type_rows[n['type']].add(r['raw_index']); type_groups[n['type']].add(r['group_id'])
            lengths[f"{len(bad.split())}->{len(repair.split())}"]+=1
            others=[x for x in nodes if x is not n and x['bad']]
            other_faults['with_other_nonempty_typed_nodes' if others else 'without_other_nonempty_typed_nodes']+=1
        diagnostics.append({'raw_index':r['raw_index'],'response_id':r['response_id'],'group_id':r['group_id'],
                            'entity_replacements':rc['entity'],'relation_replacements':rc['relation']})
    assert counts['local_replacements']==10040 and counts['original_answers_with_local_replacements']==5275
    (OUT/'ROW_COUNT_INDEX.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in diagnostics),encoding='utf-8')
    report={'status':'complete_train_only_counts_not_new_training_data','counts':dict(counts),
            'groups_with_local_replacements':len(all_groups),
            'by_type':{t:{'replacements':bytype[t],'original_answers':len(type_rows[t]),'existing_material_groups':len(type_groups[t])} for t in sorted(bytype)},
            'answers_having_both_types':len(type_rows['entity'] & type_rows['relation']),
            'per_patch_other_tagged_faults':dict(other_faults),'word_length_pairs':dict(sorted(lengths.items())),
            'interpretation':'Repair text is author-released synthetic edit preference, not independently certified truth. Unchanged text may still be wrong; no whole-answer-negative label is inferred.',
            'protocol_sha256':sha(OUT/'PROTOCOL.json'),'row_index_sha256':sha(OUT/'ROW_COUNT_INDEX.jsonl'),
            'GPU_used':False,'cal_test_read':False,'new_training_pairs_or_labels_exported':False}
    save('COUNTS.json',report); print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
