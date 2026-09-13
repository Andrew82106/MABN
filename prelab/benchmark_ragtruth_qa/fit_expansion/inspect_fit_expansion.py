"""Inventory only released answers whose source is in the CURRENT eligible fit set.

No feature extraction, training, resplitting, calibration/test JSON decoding, or
label editing. The raw mixed-partition file is streamed for whitelist filtering.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RAW = ROOT.parent / 'data/raw/response.jsonl'
SID = re.compile(r'"source_id"\s*:\s*"([^"\\]+)"')
TARGET = 'llama-2-7b-chat'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def readl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def writej(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def writel(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), 'utf-8')


def aggregate(rows):
    spans = [s for r in rows for s in r['labels']]
    by_type = Counter(s['label_type'] for s in spans)
    return {
        'answers': len(rows),
        'sources': len({r['source_id'] for r in rows}),
        'groups': len({r['group_id'] for r in rows}),
        'risk_answers': sum(bool(r['labels']) for r in rows),
        'no_annotated_risk_answers': sum(not r['labels'] for r in rows),
        'official_spans': len(spans),
        'official_span_types': dict(sorted(by_type.items())),
        'answers_by_span_type_nonexclusive': dict(sorted(Counter(
            t for r in rows for t in {s['label_type'] for s in r['labels']}).items())),
        'flag_true_spans': {k: sum(s.get(k) is True for s in spans)
                            for k in ('implicit_true', 'due_to_null')},
        'answer_characters': sum(len(r['original_response']) for r in rows),
        'whitespace_separated_words_descriptive_only': sum(len(r['original_response'].split()) for r in rows),
        'span_integrity_issue_answers': sum(bool(r['span_integrity_issues']) for r in rows),
        'empty_answers': sum(not r['original_response'].strip() for r in rows),
    }


def main():
    paths = [ROOT/'data/answers_fit.jsonl', ROOT/'data/fit.jsonl',
             ROOT/'data/development_manifest.json', ROOT/'data/gold_manifest.json', RAW]
    source_hashes = {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    gold = readl(ROOT/'data/answers_fit.jsonl')
    fit = readl(ROOT/'data/fit.jsonl')
    assert len(gold) == len(fit) == 634
    assert all(r['partition'] == 'fit' and r['eligible'] and r['quality'] == 'good' for r in gold)
    allowed = {r['source_id']: r for r in gold}
    by_source = {r['source_id']: r for r in fit}
    assert len(allowed) == 634 and set(allowed) == set(by_source)
    assert len({r['group_id'] for r in gold}) == 615
    for sid, r in allowed.items():
        old = by_source[sid]
        assert old['partition'] == 'fit' and old['official_split'] == 'train'
        assert old['model'] == TARGET and old['response_id'] == r['response_id']
        assert old['group_id'] == r['group_id'] and old['answer_sha256'] == r['answer_sha256']
        assert old['labels'] == r['original_labels']

    admitted = []
    seen = set()
    line_count = 0
    with RAW.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line_count += 1
            ids = SID.findall(line)
            if not any(s in allowed for s in ids):
                continue  # No JSON decoding or inspection of non-whitelisted content.
            assert len(ids) == 1, ('ambiguous_source_key', lineno)
            r = json.loads(line)
            sid = r['source_id']
            assert sid in allowed and r['split'] == 'train'
            key = (sid, r['model'])
            assert key not in seen, ('duplicate_source_model', key)
            seen.add(key)
            old = by_source[sid]
            assert isinstance(r['response'], str) and isinstance(r['labels'], list)
            issues = []
            for i, s in enumerate(r['labels']):
                if not (isinstance(s['start'], int) and isinstance(s['end'], int)
                        and 0 <= s['start'] < s['end'] <= len(r['response'])):
                    issues.append({'label_index': i, 'reason': 'invalid_character_interval'})
                elif r['response'][s['start']:s['end']] != s['text']:
                    issues.append({'label_index': i, 'reason': 'text_offset_mismatch'})
            row = {
                'response_id': r['id'], 'source_id': sid, 'group_id': old['group_id'],
                'partition': 'fit', 'official_split': r['split'], 'model': r['model'],
                'temperature': r['temperature'], 'quality': r['quality'],
                'original_response': r['response'], 'labels': r['labels'],
                'answer_sha256': hashlib.sha256(r['response'].encode()).hexdigest(),
                'existing_fit_response_id': old['response_id'],
                'existing_fit_prompt_sha256': old['prompt_sha256'],
                'raw_response_line_1based': lineno,
                'raw_response_line_sha256': hashlib.sha256(line.encode()).hexdigest(),
                'official_quality_eligible': r['quality'] == 'good',
                'span_integrity_issues': issues,
                'annotation_origin': 'RAGTruth released human annotation, unchanged',
                'native_llama_generation_trace': False,
            }
            if r['model'] == TARGET:
                assert r['id'] == old['response_id'] and r['response'] == old['original_response']
                assert r['labels'] == old['labels'] and r['quality'] == 'good'
            admitted.append(row)
    models = sorted({r['model'] for r in admitted})
    assert len(models) == 6 and TARGET in models
    assert len(admitted) == 6 * len(allowed)
    assert all((sid, model) in seen for sid in allowed for model in models)
    additional = sorted([r for r in admitted if r['model'] != TARGET],
                        key=lambda r: (r['source_id'], r['model'], r['response_id']))
    good = [r for r in additional if r['official_quality_eligible']]
    nongood = [r for r in additional if not r['official_quality_eligible']]
    quality_table = {m: dict(Counter(r['quality'] for r in additional if r['model'] == m))
                     for m in models if m != TARGET}
    exact_answer_groups = defaultdict(list)
    for r in admitted:
        if r['quality'] == 'good':
            exact_answer_groups[(r['source_id'], r['answer_sha256'])].append(r['response_id'])
    exact_dupes = [{'source_id': sid, 'answer_sha256': ah, 'response_ids': ids}
                   for (sid, ah), ids in exact_answer_groups.items() if len(ids) > 1]
    rows_by_id = {r['response_id']: r for r in admitted}
    duplicate_label_disagreements = []
    for group in exact_dupes:
        signatures = [tuple((s['start'], s['end'], s['label_type'], s.get('implicit_true'),
                             s.get('due_to_null')) for s in rows_by_id[rid]['labels'])
                      for rid in group['response_ids']]
        if len(set(signatures)) > 1:
            duplicate_label_disagreements.append(group)
    combined = [r for r in admitted if r['quality'] == 'good']
    per_source = Counter(r['source_id'] for r in good)
    stats = {
        'status': 'fit_only_feasibility_inventory',
        'existing': aggregate([r for r in admitted if r['model'] == TARGET]),
        'additional_good': aggregate(good),
        'combined_good_if_all_added': aggregate(combined),
        'additional_good_by_model': {m: aggregate([r for r in good if r['model'] == m])
                                    for m in models if m != TARGET},
        'quality_counts_by_other_model': quality_table,
        'quality_excluded_other_answers': len(nongood),
        'quality_excluded_counts': dict(Counter(r['quality'] for r in nongood)),
        'new_sources': 0, 'new_groups': 0,
        'source_count_by_available_good_additions': dict(sorted(Counter(
            per_source[s] for s in allowed).items())),
        'same_source_exact_answer_duplicate_groups': exact_dupes,
        'same_source_exact_answer_duplicate_excess': sum(len(x['response_ids']) - 1 for x in exact_dupes),
        'same_source_exact_answer_duplicate_label_disagreements': duplicate_label_disagreements,
        'windows_not_counted': 'No tokenization/features performed; word count is not a BPE/window count.',
        'label_definitions': 'Risk means at least one released hallucination span; no spans means no human-annotated risk, not a new world-truth certification. Types overlap at answer level.',
    }
    writel(OUT/'additional_good_candidates.jsonl', good)
    writel(OUT/'additional_quality_excluded.jsonl', nongood)
    writel(OUT/'fit_source_whitelist.jsonl', [
        {'source_id': s, 'group_id': allowed[s]['group_id'], 'partition': 'fit',
         'existing_response_id': allowed[s]['response_id']}
        for s in sorted(allowed)])
    writej(OUT/'statistics.json', stats)
    outputs = [OUT/n for n in ('additional_good_candidates.jsonl', 'additional_quality_excluded.jsonl',
                              'fit_source_whitelist.jsonl', 'statistics.json')]
    assert source_hashes == {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    writej(OUT/'manifest.json', {
        'status': 'fit_only_candidates_inventoried_not_added_to_training',
        'scope': 'Only the 634 current eligible fit source IDs and their original 615 groups; five other released generators.',
        'access': {'raw_mixed_file_streamed': True, 'raw_lines_streamed': line_count,
                   'raw_json_rows_decoded': len(admitted), 'decoded_official_splits': ['train'],
                   'decoded_source_ids_exactly_fit_whitelist': True,
                   'calibration_content_or_labels_opened': False, 'official_test_content_or_labels_opened': False,
                   'nonwhitelisted_raw_rows_json_decoded': 0},
        'existing_fit_answers': len(gold), 'fit_sources': len(allowed), 'fit_groups': 615,
        'other_model_rows': len(additional), 'good_additional_answers': len(good),
        'quality_excluded_other_answers': len(nongood),
        'other_models': [m for m in models if m != TARGET],
        'span_integrity_issue_good_answers': sum(bool(r['span_integrity_issues']) for r in good),
        'no_relabeling_or_resplitting': True, 'no_feature_extraction_or_training': True,
        'original_llama_native_trace_claimed': False,
        'source_files_sha256_before_and_after_equal': source_hashes,
        'files_sha256': {p.name: sha(p) for p in outputs},
        'audit_code_sha256': sha(Path(__file__)),
    })
    print(json.dumps({k: stats[k] for k in ('existing', 'additional_good', 'combined_good_if_all_added',
                                          'additional_good_by_model', 'quality_excluded_counts',
                                          'same_source_exact_answer_duplicate_excess')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
