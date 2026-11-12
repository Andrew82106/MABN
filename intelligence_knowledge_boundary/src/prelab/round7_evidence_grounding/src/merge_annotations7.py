"""Merge the three disjoint train annotation assignments and verify actual outputs."""
from datetime import datetime, timezone
from pathlib import Path
import json
from collections import Counter

from evaluate7 import ROOT, readl, sha, savel, save, metadata, load_labels

CHUNKS = ('annotations_train_first60.jsonl', 'annotations_train_middle30.jsonl',
          'annotations_train_final30.jsonl')


def main():
    assert not (ROOT/'results/freeze.json').exists(), 'Training already frozen'
    _, _, items = metadata(ROOT)
    expected = [i for i in items if i['split'] == 'train']
    merged, ownership = {}, {}
    for name in CHUNKS:
        rows = readl(ROOT/'data'/name)
        for row in rows:
            assert row['split'] == 'train'
            assert row['item_id'] not in merged, 'Assignments overlap: '+row['item_id']
            assert row['detector_scores_used'] is False
            merged[row['item_id']] = row
        ownership[name] = {'items': len(rows), 'groups': len({r['group_id'] for r in rows}),
                           'sha256': sha(ROOT/'data'/name)}
    assert set(merged) == {r['item_id'] for r in expected}, 'Train annotation coverage incomplete'
    assert len(expected) == 720
    ordered = [merged[r['item_id']] for r in expected]
    savel(ROOT/'data/annotations_train.jsonl', ordered)
    load_labels(ROOT, expected, ['train'])
    manifest = {'status': 'complete_train_annotations_merged_and_aligned',
                'utc': datetime.now(timezone.utc).isoformat(), 'items': len(ordered),
                'groups': len({r['group_id'] for r in ordered}), 'chunks': ownership,
                'annotation_sha256': sha(ROOT/'data/annotations_train.jsonl'),
                'evidence_relation': dict(Counter(r['evidence_relation'] for r in ordered)),
                'stance': dict(Counter(r['stance'] for r in ordered)),
                'risk_items': sum(r['risk'] == 1 for r in ordered),
                'false_refusal_rationale_items': [r['item_id'] for r in ordered if r.get('false_refusal_rationale')],
                'all_text_span_identity_generation_hash_checks_passed': True,
                'human_gold': False, 'detector_scores_used': False}
    save(ROOT/'data/train_annotation_merge.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
