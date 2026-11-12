"""Verify every actual-output annotation and freeze its pre-fit provenance."""
from collections import Counter
from datetime import datetime, timezone
import json
from evaluate7 import ROOT, metadata, load_labels, sha, save


def main():
    target = ROOT/'data/annotation_freeze.json'
    assert not target.exists(), 'Annotation freeze already exists'
    assert not (ROOT/'results/freeze.json').exists(), 'Must precede detector fitting'
    _, _, items = metadata(ROOT)
    splits = ['train', 'validation', 'test', 'external_test']
    labels = load_labels(ROOT, items, splits)
    expected = {'train':720, 'validation':240, 'test':240, 'external_test':100}
    counts = {}
    for split in splits:
        rows = [r for r in labels.values() if r['split'] == split]
        assert len(rows) == expected[split]
        assert all(r['detector_scores_used'] is False for r in rows)
        assert all(r.get('rationale', '').strip() for r in rows)
        counts[split] = {'items': len(rows), 'groups': len({r['group_id'] for r in rows}),
                        'risk': dict(Counter(str(r['risk']) for r in rows)),
                        'stance': dict(Counter(r['stance'] for r in rows)),
                        'relation': dict(Counter(r['evidence_relation'] for r in rows)),
                        'sha256': sha(ROOT/'data'/f'annotations_{split}.jsonl')}
    required = ['train_annotation_merge.json', 'validation_annotation_manifest.json',
                'test_annotation_manifest.json', 'external_annotation_manifest.json',
                'annotation_reviews/train_middle30_second_review.json',
                'annotation_reviews/train_sample24_second_review.json',
                'annotation_reviews/validation_second_review.json',
                'annotation_reviews/test_root_independent_review.json',
                'external_annotation_second_review.json']
    sample = json.loads((ROOT/'data/annotation_reviews/train_sample24_second_review.json').read_text(encoding='utf-8'))
    # The sampling review must be explicitly completed; source annotation hashes
    # are checked by the merge/adjudication and the review's own provenance.
    assert sample['status'] == 'complete'
    assert sample['summary']['groups_reviewed'] == 24 and len(sample['items']) == 144
    assert sample['summary']['suggested_core_label_changes'] == 0
    assert not sample['detector_scores_or_features_viewed']
    assert all(sha(ROOT/rel) == h for rel, h in sample['file_hashes_after'].items())
    reviewed_train = 30+24
    manifest = {'status': 'all_1300_annotations_frozen_before_formal_fit',
                'utc': datetime.now(timezone.utc).isoformat(), 'splits': counts,
                'exact_generation_text_spans_identity_and_sha256_checked': True,
                'detector_scores_used_for_annotation': False, 'human_gold': False,
                'annotation_process': 'Assistant source-based primary labeling plus assistant '
                    'second review. Main test, external and validation entirely second-reviewed; '
                    '54/120 training groups second-reviewed (30 assigned middle groups plus '
                    '24 seeded random groups from the other 90). Not human annotation or a '
                    'claim of fully blinded independent human reliability.',
                'train_second_review_groups': reviewed_train,
                'primary_root_train_review_note': 'train_root_independent_review.json contains '
                    'root primary judgments for indices 60:90 despite its historical filename.',
                'annotation_guide_sha256': sha(ROOT/'ANNOTATION_GUIDE.md'),
                'input_sha256': sha(ROOT/'data/inputs.jsonl'),
                'generated_sha256': sha(ROOT/'data/generated.jsonl'),
                'reference_sha256': sha(ROOT/'data/references.jsonl'),
                'review_files_sha256': {rel:sha(ROOT/'data'/rel) for rel in required}}
    save(target, manifest)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
