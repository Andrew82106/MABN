"""Freeze the complete blind two-assistant review of the main test labels."""
from datetime import datetime, timezone
import json
from pathlib import Path
from annotation_io7 import ROOT, readl, sha

DISPUTES = {
 'hotpot_5ae2a0f9554299495565daec__partial__1',
 'hotpot_5ab8a96f55429919ba4e234d__partial__1',
 'hotpot_5ab8a96f55429919ba4e234d__partial__3',
}


def main():
    path = ROOT/'data/annotations_test.jsonl'
    secondary = ROOT/'data/annotation_reviews/test_root_independent_review.json'
    review = json.loads(secondary.read_text(encoding='utf-8'))
    initial = readl(path)
    assert len(initial) == len(review['items']) == 240
    lookup = {r['item_id']: r for r in review['items']}
    fields = ('stance', 'evidence_relation', 'reference_correctness')
    actual = {r['item_id'] for r in initial if any(r[k] != lookup[r['item_id']][k] for k in fields)}
    assert actual == DISPUTES
    archive = ROOT/'data/annotation_reviews/test_initial.jsonl'
    assert not archive.exists(), 'Main test labels already adjudicated; preserve the frozen run'
    archive.write_bytes(path.read_bytes())
    disagreements = []
    for row in initial:
        other = lookup[row['item_id']]
        assert all(row[k] == other[k] for k in ('text', 'start', 'end', 'source_generation_sha256'))
        if row['item_id'] in DISPUTES:
            assert row['risk'] == 1 and row['evidence_relation'] == 'contradicted'
            row['false_refusal_rationale'] = True
            decision = {'item_id': row['item_id'], 'first': {k: row[k] for k in fields},
                'second': {k: other[k] for k in fields},
                'first_rationale': row['rationale'], 'second_rationale': other['rationale'],
                'final': 'Retain first asserted/contradicted label.',
                'reason': 'PLAN includes any additional false factual assertion. The answer explicitly '
                          'denies the source mentioning a subject whose source title and biography are '
                          'visible. This differs from merely saying the requested birth year is missing. '
                          'The same rule is applied across train/validation/test before fitting or scoring.'}
            disagreements.append(decision)
            row['adjudication'] = decision
        row['second_review'] = {'file': 'data/annotation_reviews/test_root_independent_review.json',
                                 'sha256': sha(secondary), 'human_gold': False,
                                 'independent_label_agreement': row['item_id'] not in DISPUTES}
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in initial), encoding='utf-8')
    manifest = {'status': 'frozen_assistant_labels_after_two_reviews', 'utc': datetime.now(timezone.utc).isoformat(),
        'items': 240, 'groups': 40, 'independent_agreements': 237, 'semantic_disputes': disagreements,
        'initial_annotation_sha256': sha(archive), 'second_review_sha256': sha(secondary),
        'final_annotation_sha256': sha(path), 'detector_scores_used': False, 'human_gold': False,
        'false_refusal_rationale_items': sorted(DISPUTES),
        'risk_items': sum(r['risk'] == 1 for r in initial),
        'supported_items': sum(r['risk'] == 0 for r in initial),
        'abstained': sum(r['stance'] == 'abstained' for r in initial),
        'unresolved': [r['item_id'] for r in initial if r['evidence_relation'] == 'unresolved']}
    (ROOT/'data/test_annotation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('status', 'items', 'groups', 'independent_agreements',
                                             'risk_items', 'supported_items', 'abstained', 'unresolved')}))


if __name__ == '__main__':
    main()
