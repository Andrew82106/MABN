"""Unblind a completed review for agreement diagnostics, never relabel training."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r13_label_compare', ROOT/'src/run13.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def main():
    review_path = ROOT/'results/blind_label_review.jsonl'
    reviews = r.r10.readl(review_path)
    samples = {x['id']: x for x in r.r10.readl(ROOT/'data/blind_label_sample.jsonl')}
    keys = {x['id']: x for x in r.r10.readl(ROOT/'data/label_sample_key.jsonl')}
    assert len(reviews) == 30 and {x['id'] for x in reviews} == set(samples) == set(keys)
    meta = r.r10.metadata(r.SOURCE); _, tokens, _, _ = r.r10.cohort(r.SOURCE, 'train', meta)
    by_item = {}
    for token in tokens:
        if token['main_eligible']:
            assert len(token['item_ids']) == 1
            by_item.setdefault(token['item_ids'][0], []).append(token)
    result = []; totals = {'tokens_compared': 0, 'token_labels_changed': 0, 'old_risk_tokens': 0,
                           'review_risk_tokens': 0, 'risk_token_overlap': 0}
    for review in reviews:
        sid = review['id']; sample = samples[sid]; key = keys[sid]; answer = sample['answer']
        assert review['answer'] == answer
        assert review['sample_file_sha256'] == r.r10.sha(ROOT/'data/blind_label_sample.jsonl')
        assert review['old_gold_or_prediction_scores_viewed'] is False
        assert review['status'] in ('supported', 'unsupported')
        for spans in [key['old_spans'], review['spans'], *review['alternative_spans']]:
            for span in spans:
                assert answer[span['start']:span['end']] == span['text']
        positions = lambda spans: {j for span in spans for j in range(span['start'], span['end']) if answer[j].isalnum()}
        old_chars, new_chars = positions(key['old_spans']), positions(review['spans'])
        changed = []; old_count = new_count = overlap = 0
        for token in by_item[key['item_id']]:
            old = bool(set(token['characters']) & old_chars); new = bool(set(token['characters']) & new_chars)
            assert int(old) == token['gold']
            old_count += old; new_count += new; overlap += old and new
            if old != new:
                changed.append({'token_key': token['token_key'], 'text': token['text'], 'old': int(old), 'review': int(new)})
        span_pairs = lambda spans: sorted((s['start'], s['end']) for s in spans)
        record = {'id': sid, 'item_id': key['item_id'], 'category': key['category'],
                  'old_gold': key['old_gold'], 'review_gold': int(review['status'] == 'unsupported'),
                  'semantic_agreement': key['old_gold'] == int(review['status'] == 'unsupported'),
                  'span_boundaries_identical': span_pairs(key['old_spans']) == span_pairs(review['spans']),
                  'old_spans': key['old_spans'], 'review_spans': review['spans'],
                  'token_label_changes': changed, 'old_risk_tokens': old_count,
                  'review_risk_tokens': new_count, 'risk_token_overlap': overlap}
        result.append(record)
        totals['tokens_compared'] += len(by_item[key['item_id']]); totals['token_labels_changed'] += len(changed)
        totals['old_risk_tokens'] += old_count; totals['review_risk_tokens'] += new_count; totals['risk_token_overlap'] += overlap
    shared = [x for x in result if x['old_gold'] == x['review_gold'] == 1]
    output = {'schema': 'round13-label-agreement-v1', 'sample_size': 30,
              'review_file_sha256': r.r10.sha(review_path), 'semantic_agreements': sum(x['semantic_agreement'] for x in result),
              'both_judge_risk': len(shared), 'boundary_disagreements_among_shared_risk': sum(not x['span_boundaries_identical'] for x in shared),
              **totals, 'records': result,
              'interpretation': 'Independent assistant agreement on a stratified sample, not human gold accuracy. Boundary disagreements do not establish errors. Old labels/scores remain unchanged.',
              'semantic_disagreement_note': 'review_20: original label includes an implicit Nebula-client relation from the question; blind reviewer judges the explicit AI GUIDES sentence alone. Needs a shared policy, not an automatic correction.'}
    r.r10.save(ROOT/'results/label_agreement.json', output)
    print(json.dumps({k:v for k,v in output.items() if k != 'records'},ensure_ascii=False))


if __name__ == '__main__':
    main()
