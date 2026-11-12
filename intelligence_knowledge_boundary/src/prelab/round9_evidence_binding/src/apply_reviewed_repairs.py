"""Apply a root-reviewed localized proposal without reselecting or reshuffling.

Default writes a preview into data/repair_preview; --apply promotes that same
deterministic data after parent source review. Never generates model responses.
"""
import argparse
import collections
import copy
import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('build_inputs9.py'))
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    proposal_path=P/'data/curation/donor_repair_proposals_v2.json';prop=B.read_json(proposal_path)
    assert B.sha(P/'data/paired_inputs.jsonl')==prop['snapshot_inputs_sha256'],'Original snapshot changed'
    assert B.sha(P/'data/donor_review_packets.jsonl')==prop['snapshot_packets_sha256'],'Original packets changed'
    pool={r['candidate_id']:r for r in B.read_rows(P/'data/curation/source_pool.jsonl')}
    old_inputs=B.read_rows(P/'data/paired_inputs.jsonl');old_refs=B.read_rows(P/'data/paired_references.jsonl')
    old_packets=B.read_rows(P/'data/donor_review_packets.jsonl')
    selected=B.read_json(P/'data/curation/selected_target_candidates.json')
    assignments=B.read_json(P/'data/curation/donor_assignments_proposed.json')
    by_target={r['candidate_id']:r for r in selected}
    candidates={r['candidate_id']:r for r in B.load_candidates(pool)}
    inputs_by_id={r['row_id']:r for r in old_inputs};ref_by_id={r['question_id']:r for r in old_refs}
    changed_inputs={};changed_refs={};changed_targets={};changed_packets={};change_log=[]
    for change in prop['proposals']:
        old_id=change['old_question_id'];new_id=change['new_question_id']
        original=inputs_by_id[old_id+'__complete'];original_packet=old_packets[change['snapshot_row_index']]
        assert original_packet['question_id']==old_id
        target=copy.deepcopy(by_target[old_id] if new_id==old_id else candidates[new_id])
        target.update({'common_quote':change['target_common'],'evidence_quote':change['target_evidence'],
                       'partial_quote':change['target_partial'],'question':change['target_question']})
        target['assembly_repair_notes']=change['reasons']
        target['rationale']=target['rationale']+' Source-review assembly note: '+'; '.join(change['reasons'])
        donor={'donor_id':change['proposed_donor_id'],'quote':change['proposed_donor_quote'],
               'ranking_score':'explicit_source_review_repair','span':change['proposed_donor_span'],
               'review_status':'source_repair_pending_root_final_check'}
        rows,refs=B.build_pairs(pool,[target],{new_id:donor},{new_id:original['split']})
        target_position=next(i for i,p in enumerate(original['passages']) if p['title']==pool[old_id]['source_title'])
        for row in rows:
            current_target=next(p for p in row['passages'] if p['title']==pool[new_id]['source_title'])
            current_donor=next(p for p in row['passages'] if p['title']==pool[donor['donor_id']]['source_title'])
            row['passages']=[current_target,current_donor] if target_position==0 else [current_donor,current_target]
            row['prompt']=B.TEMPLATE.format(questions='1. '+target['question'],search_results='\n\n'.join(
                f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(row['passages'])))
            changed_inputs[old_id+'__'+row['condition']]=row
        ref=refs[0];ref['source_review_repair_notes']=change['reasons'];changed_refs[old_id]=ref
        changed_targets[old_id]=target
        source=pool[new_id];other=pool[donor['donor_id']]
        packet=copy.deepcopy(original_packet)
        packet.update({'question_id':new_id,'question':target['question'],'subjects':target['subjects'],
            'reference_answer':target['reference_answer'],'target_title':source['source_title'],
            'common_quote':target['common_quote'],'evidence_quote':target['evidence_quote'],
            'partial_quote':target['partial_quote'],'target_source_hash':source['source_content_sha256'],
            'donor_id':donor['donor_id'],'donor_title':other['source_title'],'donor_quote':donor['quote'],
            'donor_source_url':other['source_url'],'donor_revision_id':other['revision_id'],
            'donor_source_hash':other['source_content_sha256'],
            'donor_original_question':other['original_question'],'donor_original_answer_quote':other['original_answer_quote'],
            'alternatives':[],'review_status':'localized_repair_pending_root_final_check',
            'source_review_repair_notes':change['reasons'],'original_snapshot_row_index':change['snapshot_row_index']})
        changed_packets[old_id]=packet
        assignments.pop(old_id);assignments[new_id]=donor
        change_log.append({'original_packet_index':change['snapshot_row_index'],'old_question_id':old_id,
            'new_question_id':new_id,'split_preserved':original['split'],'target_passage_position_preserved':target_position,
            'old_donor_id':original_packet['donor_id'],'new_donor_id':donor['donor_id'],'reasons':change['reasons']})
    inputs=[changed_inputs.get(r['row_id'],r) for r in old_inputs]
    refs=[changed_refs.get(r['question_id'],r) for r in old_refs]
    packets=[changed_packets.get(r['question_id'],r) for r in old_packets]
    selected=[changed_targets.get(r['candidate_id'],r) for r in selected]
    old_subjects,old_material=B.old_material()
    conflicts,lengths=B.checks(selected,inputs,assignments,old_material)
    new_old_mentions=[]
    for row in inputs:
        if row['condition']!='complete':continue
        text=B.norm(' '.join(p['title']+' '+p['text'] for p in row['passages']))
        hits=[s for s in old_subjects if B.has_phrase(text,s)]
        if hits:new_old_mentions.append({'question_id':row['question_id'],'old_subjects':hits})
    assert not conflicts,conflicts
    assert not new_old_mentions,new_old_mentions
    assert all(r['relative_gap']<=.2 for r in lengths),[r for r in lengths if r['relative_gap']>.2]
    assert len({pool[c]['source_content_sha256'] for c in assignments}|{pool[d['donor_id']]['source_content_sha256'] for d in assignments.values()})==400
    # All three target excerpts remain disjoint, exact and uniquely located.
    for target in selected:
        spans=sorted([B.source_span(pool[target['candidate_id']],target[k]) for k in ['common_quote','evidence_quote','partial_quote']],key=lambda s:s['start'])
        assert all(a['end']<=b['start'] for a,b in zip(spans,spans[1:])),target['candidate_id']
    unchanged=[r for r in old_inputs if r['question_id'] not in changed_targets]
    final_by_id={r['row_id']:r for r in inputs}
    assert all(r==final_by_id[r['row_id']] for r in unchanged)
    folder=P/('data' if args.apply else 'data/repair_preview')
    if args.apply:
        backup=P/'data/curation/snapshots/pre_repair_source_review'
        backup.mkdir(parents=True,exist_ok=True)
        names=['paired_inputs.jsonl','paired_references.jsonl','donor_review_packets.jsonl','assembly_review_manifest.json',
               'curation/selected_target_candidates.json','curation/donor_assignments_proposed.json','curation/pair_lengths.json']
        for name in names:
            source=P/'data'/name;dest=backup/name
            assert not dest.exists(),f'Repair snapshot exists: {dest}'
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    B.write_rows(folder/'paired_inputs.jsonl',inputs);B.write_rows(folder/'paired_references.jsonl',refs)
    B.write_rows(folder/'donor_review_packets.jsonl',packets)
    B.write_json(folder/'curation/selected_target_candidates.json',selected)
    B.write_json(folder/'curation/donor_assignments_proposed.json',assignments)
    B.write_json(folder/'curation/pair_lengths.json',lengths);B.write_json(folder/'curation/pair_conflicts.json',conflicts)
    manifest={'status':'repaired_pairs_pending_root_final_source_freeze','groups':200,'rows':400,'seed':B.SEED,
        'counts':dict(collections.Counter(f"{r['split']}:{r['category']}" for r in refs)),
        'proposal_file_sha256':B.sha(proposal_path),'original_inputs_sha256':prop['snapshot_inputs_sha256'],
        'original_packets_sha256':prop['snapshot_packets_sha256'],'changed_groups':len(changed_targets),
        'unchanged_groups':len(unchanged)//2,'unchanged_rows_exactly_preserved':len(unchanged),
        'unique_target_sources':200,'unique_donor_sources':200,'unique_source_fulltext_hashes':400,
        'literal_cross_group_or_old_target_conflicts':len(conflicts)+len(new_old_mentions),
        'length_gap_max':max(r['relative_gap'] for r in lengths),'length_gap_over_20pct':0,
        'source_review_files':prop['source_review_files'],'changes':change_log,
        'restrictions':'Source-only curation; no generated model responses or detection scores read; not a data freeze.',
        'files':{str(f.relative_to(folder)):B.sha(f) for f in [folder/'paired_inputs.jsonl',folder/'paired_references.jsonl',folder/'donor_review_packets.jsonl',folder/'curation/selected_target_candidates.json',folder/'curation/donor_assignments_proposed.json',folder/'curation/pair_lengths.json',folder/'curation/pair_conflicts.json']}}
    B.write_json(folder/'assembly_review_manifest.json',manifest)
    print({'destination':str(folder),'changed_groups':len(changed_targets),'unchanged_rows':len(unchanged),
        'max_length_gap':manifest['length_gap_max'],'inputs_sha256':manifest['files']['paired_inputs.jsonl']})

if __name__=='__main__':main()
