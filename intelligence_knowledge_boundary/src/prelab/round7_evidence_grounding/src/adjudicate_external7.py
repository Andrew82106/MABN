"""Apply the documented blind second review; preserve initial opinions."""
from datetime import datetime, timezone
import json
from pathlib import Path
from run7 import sha, save, savel, readl

ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = 'f13a8b780d883ec82cf916ad2ec82549ad9eabd5d986dd195532a094da1bf4cd'


def main():
    path = ROOT/'data/annotations_external_test.jsonl'
    review_path = ROOT/'data/external_annotation_second_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    assert review['coverage']['answers_reviewed'] == 100
    assert review['primary_label_agreement_count'] == 99
    assert not review['blinding']['detector_scores_visible']
    assert sha(path) == INITIAL_SHA, 'Apply once to the preserved initial annotations'
    archive = ROOT/'data/annotation_reviews/external_initial.jsonl'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        assert sha(archive) == INITIAL_SHA
    else:
        archive.write_bytes(path.read_bytes())
    rows = readl(path)
    for row in rows:
        row['second_review'] = {'review_file': 'data/external_annotation_second_review.json',
                                'review_sha256': sha(review_path), 'human_gold': False}
        if row['item_id'] == 'ragognize_test_1858__complete__1':
            assert row['text'] == 'The spouse of Clarence Sexton died in 2024.'
            row['multi_claim'] = False
        if row['item_id'] == 'ragognize_test_1557__partial__1':
            assert row['risk'] is None and row['evidence_relation'] == 'unresolved'
            row['rationale'] = ('The grammar can deny that the Interconnect plan provides functionality, '
                'which full reference evidence refutes; however, based on the given information can '
                'also introduce an awkward source-insufficiency response. Independent assistant '
                'review favored the first reading; final adjudication preserves unresolved because '
                'negation scope remains ambiguous. No detector score was consulted.')
            row['adjudication'] = {'decision': 'retain_unresolved', 'alternative_risk': 1,
                                  'alternative_reference_correctness': 'incorrect',
                                  'reviewer_disagreement_preserved': True}
    savel(path, rows)
    manifest_path = ROOT/'data/external_annotation_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(status='frozen_assistant_annotation_after_second_review',
                    frozen_utc=datetime.now(timezone.utc).isoformat(),
                    annotations_sha256=sha(path), initial_annotation_sha256=INITIAL_SHA,
                    initial_annotation_archive='data/annotation_reviews/external_initial.jsonl',
                    second_review_sha256=sha(review_path),
                    second_review_primary_agreements=99, semantic_disputes=1,
                    retained_unresolved=['ragognize_test_1557__partial__1'],
                    metadata_corrections=['ragognize_test_1858__complete__1 multi_claim=false'],
                    detector_scores_used=False, independently_human_reviewed=False)
    save(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'items', 'risk_items', 'supported_items',
                                             'abstentions', 'unresolved', 'annotations_sha256')}))


if __name__ == '__main__':
    main()
