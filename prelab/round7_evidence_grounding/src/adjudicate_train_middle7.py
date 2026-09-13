"""Preserve the primary labels and adjudicate two explicit semantic ambiguities."""
from datetime import datetime, timezone
import json
from annotation_io7 import ROOT, readl, sha


def main():
    path = ROOT/'data/annotations_train_middle30.jsonl'
    review_path = ROOT/'data/annotation_reviews/train_middle30_second_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    rows = readl(path)
    lookup = {r['item_id']: r for g in review['groups'] for r in g['item_reviews']}
    assert len(rows) == len(lookup) == 180
    assert review['complete'] and not review['detector_scores_used']
    disputes = {
        'hotpot_5a7a30565542996a35c17128__partial__1',
        'hotpot_5ae2ad255542996483e64a3b__partial__1',
    }
    assert {i for i, r in lookup.items() if not r['agrees_with_initial_semantic_labels']} == disputes
    archive = ROOT/'data/annotation_reviews/train_middle30_initial.jsonl'
    assert not archive.exists(), 'Already adjudicated'
    assert not (ROOT/'results/freeze.json').exists()
    archive.write_bytes(path.read_bytes())
    decisions = []
    for row in rows:
        other = lookup[row['item_id']]
        assert row['text'] == other['text']
        assert row['source_generation_sha256'] == other['source_generation_sha256']
        if row['item_id'] in disputes:
            before = {k: row[k] for k in ('stance', 'evidence_relation', 'reference_correctness', 'risk')}
            row.update(stance='asserted', evidence_relation='unresolved',
                       reference_correctness='unresolved', risk=None)
            decision = {'item_id': row['item_id'], 'initial': before,
                        'final': {k: row[k] for k in before},
                        'reason': 'The unusual negative wording permits both missing-information and '
                                  'world-fact readings. Neither is uniquely established. Apply the '
                                  'existing unresolved rule, consistently with the held-out Sidney '
                                  'Morgan and Interconnect examples, before fitting any detector.'}
            row['adjudication'] = decision
            row['rationale'] = other['rationale']
            decisions.append(decision)
        row['second_review'] = {'file': str(review_path.relative_to(ROOT)).replace('\\', '/'),
                                'sha256': sha(review_path), 'human_gold': False,
                                'initial_label_agreement': row['item_id'] not in disputes}
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    manifest = {'status': 'train_middle30_adjudicated_before_fit',
                'utc': datetime.now(timezone.utc).isoformat(), 'items': 180,
                'agreements': 178, 'disputes': decisions,
                'initial_sha256': sha(archive), 'review_sha256': sha(review_path),
                'final_sha256': sha(path), 'human_gold': False, 'detector_scores_used': False}
    (ROOT/'data/annotation_reviews/train_middle30_adjudication.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
