"""Output-ledger and original-character audit; no reparsing/repair/model."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import census as run

OUT=Path(__file__).resolve().parent


def main():
    run.check();done=run.read(OUT/'complete.json')
    for name,h in done['files_sha256'].items():assert run.file_sha(OUT/name)==h
    raw=run.read(run.RAW);rows=[json.loads(s) for s in (OUT/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    failed=[json.loads(s) for s in (OUT/'failures.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(raw)==30073 and [r['raw_index'] for r in rows]==list(range(30073))
    counts=Counter();types=Counter();failure_counts=Counter();blocks=Counter();block_rows=defaultdict(set)
    groups=defaultdict(list)
    for i,(original,r) in enumerate(zip(raw,rows)):
        assert run.old.sha(original['prompt'])==r['prompt_sha256']
        assert run.old.sha(original['completion'])==r['completion_sha256']
        refs,answer=run.old.prompt_parts(original['prompt'])
        hashes=[run.old.sha(s) for s in refs]
        assert hashes==r['reference_text_sha256'] and run.old.sha(answer)==r['answer_sha256']
        gid=run.digest(sorted(hashes));assert gid==r['reference_set_group_sha256'];groups[gid].append(i)
        for h in hashes:blocks[h]+=1;block_rows[h].add(i)
        assert r['edited_projection_is_clean_gold'] is False and r['unmarked_text_is_verified_negative'] is False
        if r['exact_character_alignment']:
            spans=r['aligned_synthetic_spans'];counts[r['alignment_class']]+=1
            assert all(answer[s['start']:s['end']]==s['text'] and 0<=s['start']<s['end']<=len(answer) for s in spans)
            types.update(s['type'] for s in spans)
            assert r['factual_candidate_span_count']==sum(s['type'] in run.FACTUAL for s in spans)
            assert r['other_type_span_count']==sum(s['type'] in run.OTHER for s in spans)
            assert r['exact_factual_candidate_available']==bool(r['factual_candidate_span_count'])
        else:failure_counts[r['failure_category']]+=1
    assert [r['raw_index'] for r in failed]==[r['raw_index'] for r in rows if not r['exact_character_alignment']]
    for f in failed:
        assert f['original_record']==raw[f['raw_index']]
        assert {k:v for k,v in f.items() if k!='original_record'}==rows[f['raw_index']]
    group_file=[json.loads(s) for s in (OUT/'reference_groups.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(group_file)==len(groups)
    for g in group_file:
        assert g['raw_indices']==groups[g['reference_set_group_sha256']]
        assert run.digest(g['reference_hash_multiset'])==g['reference_set_group_sha256']
    shared=[json.loads(s) for s in (OUT/'shared_reference_blocks.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['reference_text_sha256'] for r in shared}=={h for h,v in block_rows.items() if len(v)>1}
    for r in shared:
        for i,ordinal in r['members_raw_index_and_reference_ordinal']:
            assert rows[i]['reference_text_sha256'][ordinal-1]==r['reference_text_sha256']
    summary=run.read(OUT/'summary.json')
    assert dict(types)==summary['exact_aligned_span_types'] and dict(failure_counts)==summary['failure_categories']
    assert all(v==summary['counts'][k] for k,v in counts.items())
    assert len(failed)==summary['counts']['failed_rows'] and sum(counts.values())==summary['counts']['exact_aligned_rows']
    clarification={'hashes_with_multiple_total_occurrences':sum(v>1 for v in blocks.values()),
        'hashes_shared_across_distinct_rows':sum(len(v)>1 for v in block_rows.values()),
        'repeated_only_within_one_row_hashes':sum(blocks[h]>1 and len(v)==1 for h,v in block_rows.items()),
        'explanation':'Old inspect_training structural counter used total occurrences>1, including repetition within one row. The new shared_reference_blocks ledger requires at least2 distinct raw rows. No discrepancy in input; these count different events.',
        'not_original_article_identity':True}
    old=run.read(run.SOURCE/'SCHEMA_ALIGNMENT_REPORT.json')['full_file_reference_structure_only']
    assert clarification['hashes_with_multiple_total_occurrences']==old['blocks_shared_between_rows']
    assert clarification['hashes_shared_across_distinct_rows']==summary['grouping']['reference_blocks_shared_across_rows']
    run.save(OUT/'GROUPING_CLARIFICATION.json',clarification)
    run.save(OUT/'AUDIT.json',{'passed':True,'all30073_original_hashes_and_reference_groups_checked':True,
        'all29315_accepted_span_coordinates_rechecked_against_original_answer':True,
        'all758_failures_full_original_records_retained_exact':True,'type_counts_and_failure_counts_reconstructed':True,
        'negative_or_clean_edit_promotion':False,'summary_sha256':run.file_sha(OUT/'summary.json'),
        'complete_sha256':run.file_sha(OUT/'complete.json'),'audit_code_sha256':run.file_sha(Path(__file__)),
        'trained':False,'GPU_used':False,'evaluation_data_opened':False})
    print('FAVA_FULL_CENSUS_OUTPUT_AUDIT_PASSED',json.dumps(clarification),flush=True)


if __name__=='__main__':main()
