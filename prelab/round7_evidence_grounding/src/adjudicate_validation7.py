"""Freeze validation annotations after source-based second review, before fitting."""
from datetime import datetime, timezone
import json
from annotation_io7 import ROOT, readl, sha


def main():
    path = ROOT/'data/annotations_validation.jsonl'
    review_path = ROOT/'data/annotation_reviews/validation_second_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    rows = readl(path)
    lookup = {r['item_id']: r for r in review['decisions']}
    assert len(rows) == len(lookup) == 240
    assert review['independent_review_complete'] and not review['detector_scores_read']
    assert sha(path) == review['first_review_comparison']['annotation_snapshot_sha256']
    fields = ('stance', 'evidence_relation', 'reference_correctness')
    disputed = 'hotpot_5ab2653e554299340b5254b1__partial__2'
    assert {r['item_id'] for r in rows if any(r[k] != lookup[r['item_id']][k] for k in fields)} == {disputed}
    archive = ROOT/'data/annotation_reviews/validation_initial.jsonl'
    assert not archive.exists(), 'Already adjudicated'
    assert not (ROOT/'results/freeze.json').exists()
    archive.write_bytes(path.read_bytes())
    decision = {
        'item_id': disputed, 'final': 'Retain asserted/unresolved; risk remains null.',
        'reason': 'Cedric naturally could refer to the requested musician in the music context, '
                  'but the visible passages do not establish the full-name mapping. Retain the '
                  'existing ambiguous-coreference rule instead of treating exact name mismatch '
                  'as absence or importing unseen biographical knowledge as certain evidence.',
        'alternative': review['first_review_comparison']['differences'][0],
    }
    for row in rows:
        other = lookup[row['item_id']]
        assert all(row[k] == other[k] for k in ('text', 'start', 'end', 'source_generation_sha256'))
        if row['item_id'] == disputed:
            row['adjudication'] = decision
        row['second_review'] = {'file': str(review_path.relative_to(ROOT)).replace('\\', '/'),
                                'sha256': sha(review_path), 'human_gold': False,
                                'initial_label_agreement': row['item_id'] != disputed}
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    manifest = {'status': 'frozen_assistant_labels_after_two_reviews',
                'utc': datetime.now(timezone.utc).isoformat(), 'items': 240, 'groups': 40,
                'agreements': 239, 'disputes': [decision],
                'secondary_flag_adjudication': 'Keep initial multi_claim granularity: nested city/state '
                    'locations count as one location fact; restatement of a known year within a '
                    'refusal is not a separately answered comparison. No main-label effect.',
                'initial_sha256': sha(archive), 'review_sha256': sha(review_path),
                'final_sha256': sha(path), 'human_gold': False, 'detector_scores_used': False}
    (ROOT/'data/validation_annotation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('status','items','groups','agreements','final_sha256')}))


if __name__ == '__main__':
    main()
