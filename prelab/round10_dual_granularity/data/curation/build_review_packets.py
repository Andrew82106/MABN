"""Source-only review packets. Never reads generations or detector scores."""
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

CUR = Path(__file__).resolve().parent
CATEGORIES = ['time', 'quantity', 'location', 'relation', 'action']
TEMPLATE = 'Please answer the following questions using these search results. Write one short sentence for each numbered item.\n\nQuestions:\n{questions}\n\nSearch results:\n{search_results}'
SEED = 20260911

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def norm(s):
    return re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', s).casefold()).strip()

def main():
    pool = {r['candidate_id']: r for r in map(json.loads, (CUR/'remaining_source_pool.jsonl').read_text(encoding='utf-8').splitlines())}
    inventory = read(CUR/'legacy_isolation_inventory.json')
    old_text = inventory['normalized_old_visible_passages']
    targets = []
    for category in CATEGORIES:
        data = read(CUR/f'{category}_candidates.json')
        group = data.get('candidates', []) if isinstance(data, dict) else data
        targets.extend(group[:12])
    assert len(targets) == 60
    assert len({t['candidate_id'] for t in targets}) == 60
    target_first = set(random.Random(SEED).sample([t['candidate_id'] for t in targets], 30))
    target_ids = {t['candidate_id'] for t in targets}
    donors = {}
    for path in sorted(CUR.glob('donors_*.json')):
        data = read(path)
        group = data.get('donors', []) if isinstance(data, dict) else data
        for d in group:
            assert d['target_candidate_id'] not in donors, d['target_candidate_id']
            donors[d['target_candidate_id']] = d
    assert not target_ids.intersection(d['donor_candidate_id'] for d in donors.values())
    assert len({d['donor_candidate_id'] for d in donors.values()}) == len(donors)
    packets = []
    source_hits = []
    for idx, target in enumerate(targets):
        cid = target['candidate_id']
        if cid not in donors:
            continue
        donor = donors[cid]
        source = pool[cid]
        donor_source = pool[donor['donor_candidate_id']]
        dq = donor['donor_quote']
        assert donor_source['source_content'].count(dq) == 1
        for key in ['common_quote', 'evidence_quote', 'partial_quote']:
            assert source['source_content'].count(target[key]) == 1, (cid, key)
        rows = []
        for condition, key in [('complete', 'evidence_quote'), ('partial', 'partial_quote')]:
            quotes = sorted([target['common_quote'], target[key]], key=source['source_content'].index)
            passages = [{'title': source['source_title'], 'text': ' '.join(quotes)},
                        {'title': donor_source['source_title'], 'text': dq}]
            if cid not in target_first:
                passages.reverse()
            prompt = TEMPLATE.format(questions='1. '+target['question'], search_results='\n\n'.join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)))
            rows.append({'row_id': cid+'__'+condition, 'question_id': cid, 'group_id': cid,
                         'split': 'test', 'condition': condition, 'questions': [target['question']],
                         'subjects': target['subjects'], 'passages': passages, 'prompt': prompt,
                         'system': 'You are a helpful assistant.', 'dataset': 'RAGognize_fresh_heldout_round10',
                         'expected_items': 1, 'category': target['category']})
        hits = []
        for role, terms in [('target', target['subjects']), ('donor', donor.get('donor_subjects', []))]:
            for term in terms:
                nt = norm(term)
                matches = [s for s in old_text if ' '+nt+' ' in ' '+s+' ']
                if matches:
                    hits.append({'role': role, 'term': term, 'old_visible_matches': matches})
        source_hits.extend({'question_id': cid, **h} for h in hits)
        lengths = [sum(len(p['text']) for p in row['passages']) for row in rows]
        words = [sum(len(p['text'].split()) for p in row['passages']) for row in rows]
        packets.append({'group_index': idx, 'question_id': cid, 'category': target['category'],
                        'question': target['question'], 'reference_answer': target['reference_answer'],
                        'target': target, 'donor': donor, 'paired_inputs': rows,
                        'material_char_lengths': lengths, 'material_word_lengths': words,
                        'relative_char_gap': abs(lengths[0]-lengths[1])/max(lengths),
                        'relative_word_gap': abs(words[0]-words[1])/max(words),
                        'old_visible_subject_hits': hits,
                        'status': 'paired_source_review_pending_no_model_generation'})
    for category in CATEGORIES:
        subset = [p for p in packets if p['category'] == category]
        if subset:
            (CUR/f'paired_review_{category}.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in subset), encoding='utf-8')
    (CUR/'donor_subject_overlap_review.json').write_text(json.dumps(source_hits, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'groups': len(packets), 'per_category': {c: sum(p['category']==c for p in packets) for c in CATEGORIES},
                      'length_over_20pct': [{k:p[k] for k in ['question_id','relative_char_gap','relative_word_gap']} for p in packets if max(p['relative_char_gap'],p['relative_word_gap']) > .20],
                      'subject_hits': [{k:h[k] for k in ['question_id','role','term']} for h in source_hits]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
