"""Source-only preliminary pool. No generations, detector scores or final sampling."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ROUND = ROOT / 'prelab/round9_evidence_binding'
OLD = ROOT / 'prelab'

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

old_titles, old_subjects = set(), set()
old_files = []
for round_name in ['round6_evidence_grounding', 'round7_evidence_grounding']:
    for name in ['inputs.jsonl', 'dev_inputs.jsonl']:
        path = OLD / round_name / 'data' / name
        if path.exists():
            old_files.append(path)
            for row in rows(path):
                old_titles.update(p['title'].casefold() for p in row['passages'])
                old_subjects.update(s.casefold() for s in row.get('subjects', []))
preview = OLD / 'round6_evidence_grounding/planning/hotpotqa_preview.json'
old_files.append(preview)
for wrapped in json.loads(preview.read_text(encoding='utf-8'))['rows']:
    row = wrapped.get('row', wrapped)
    old_titles.update(t.casefold() for t in row.get('context', {}).get('title', []))

source_paths = {
    'train': ROUND / 'data/raw/ragognize/train_sources.jsonl',
    'test': OLD / 'round7_evidence_grounding/data/external_ragognize/raw/test_sources.jsonl',
}
pool, excluded, counts = [], [], {}
for split, path in source_paths.items():
    by_id = collections.defaultdict(list)
    raw = rows(path)
    for row in raw:
        by_id[row['user_prompt_index']].append(row)
    local_pool = []
    for qid, pair in by_id.items():
        complete = [r for r in pair if r['answerable']]
        partial = [r for r in pair if not r['answerable']]
        if not complete or not partial:
            excluded.append({'source_split': split, 'source_id': qid, 'reason': 'no_official_pair'})
            continue
        overlap = sorted({p['title'] for r in pair for p in r['documents'] if p['title'].casefold() in old_titles})
        row = complete[0]
        details = row['details']['user_prompt']['details']
        article = details['suitable_article']
        if overlap or article['title'].casefold() in old_subjects:
            excluded.append({'source_split': split, 'source_id': qid, 'reason': 'old_title_or_subject_exact', 'overlap_titles': overlap})
            continue
        item = {
            'candidate_id': f'ragognize_{split}_{qid:04d}',
            'status': 'unreviewed_source_pool_not_selected',
            'source_split': split,
            'source_id': qid,
            'source_file': str(path.relative_to(ROOT)),
            'original_question': row['user_prompt'],
            'official_information_type': row['information_type'],
            'official_category': row['category'],
            'original_answer_quote': details['answer_quote_from_passage'],
            'source_title': article['title'],
            'source_url': article['url'],
            'revision_id': article['revision_id'],
            'retrieval_date_utc': article['retrieval_date_utc'],
            'source_content': article['content'],
            'source_content_sha256': hashlib.sha256(article['content'].encode('utf-8')).hexdigest(),
            'official_complete_document_titles': [p['title'] for p in row['documents']],
            'requires_review': ['target_alias_and_event_isolation', 'question_answer_leakage', 'attribute_type', 'all_duplicate_and_inferable_evidence', 'retained_subject_context', 'reference_consistency', 'natural_other_subject_distractor'],
        }
        pool.append(item)
        local_pool.append(item)
    counts[split] = {'raw_rows': len(raw), 'raw_question_ids': len(by_id), 'preliminary_pairs': len(local_pool), 'by_original_type': dict(collections.Counter(r['official_information_type'] for r in local_pool))}

folder = ROUND / 'data/curation'
folder.mkdir(parents=True, exist_ok=True)
for name, values in [('source_pool.jsonl', pool), ('source_pool_exclusions.jsonl', excluded)]:
    (folder / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8')
manifest = {
    'status': 'feasibility_only_not_selected_or_semantically_qualified',
    'scope': 'Official train and test source-only projections, no official validation split; Round8 reuses Round7 sources.',
    'isolation_level': 'Exact casefold titles across every official pair document plus exact target-title against old subject names, including Round6 preview. Alias, co-reference, event and within-new-pool connected-group review still required.',
    'old_title_count': len(old_titles), 'old_subject_count': len(old_subjects),
    'counts': counts, 'total_preliminary_pairs': len(pool),
    'unique_target_titles': len({r['source_title'] for r in pool}),
    'by_original_type': dict(collections.Counter(r['official_information_type'] for r in pool)),
    'non_draft_by_original_type': dict(collections.Counter(r['official_information_type'] for r in pool if not r['source_title'].startswith('Draft:'))),
    'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_paths.values()},
    'old_input_hashes': {str(p.relative_to(ROOT)): sha(p) for p in old_files},
    'pool_sha256': sha(folder / 'source_pool.jsonl'),
    'exclusions_sha256': sha(folder / 'source_pool_exclusions.jsonl'),
}
(folder / 'source_pool_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'counts': counts, 'total': len(pool), 'pool_sha256': manifest['pool_sha256']}, ensure_ascii=False))
