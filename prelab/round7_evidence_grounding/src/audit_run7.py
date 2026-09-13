"""Validate saved GPU artifacts against exact prompts/outputs, without labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import model7
from run7 import ROOT, readl, sha, save, stage_signature, generation_signature


ITEM_FEATURES = {
    'features': {**{f'hidden_{l}': (3584,) for l in (7, 14, 21, 28)},
                 'mean_nll': (), 'mean_entropy': (), 'surface': (11,)},
    'attention': {'lookback_features': (784,), 'redeep_ecs': (784,), 'redeep_pks': (28,)},
    'lumina': {'lumina_mmd': (), 'lumina_ipr': (), 'lumina_score': ()},
}


def audit(data_file, output):
    rows = readl(ROOT/data_file)
    assert len({r['row_id'] for r in rows}) == len(rows)
    gen_signature = model7.digest(generation_signature())
    signatures = {s: model7.digest(stage_signature(s)) for s in (*ITEM_FEATURES, 'baselines')}
    counts, nonfinite, parse_failures, source_max_tokens = Counter(), [], [], 0
    pairs = {}
    for row in rows:
        rid = row['row_id']
        gp = ROOT/'data/generation_records'/(rid+'.json')
        generated = json.loads(gp.read_text(encoding='utf-8'))
        assert all(generated[k] == row[k] for k in ('row_id', 'question_id', 'split', 'condition', 'dataset'))
        assert generated['prompt_hash'] == model7.prompt_hash(row)
        assert generated['generation_signature_hash'] == gen_signature
        assert generated['input_tokens'] == len(generated['input_token_ids']) <= 3072
        source_max_tokens = max(source_max_tokens, generated['input_tokens'])
        offsets = np.array(generated['response_token_offsets'], dtype=int)
        assert offsets.shape == (len(generated['response_token_ids']), 2)
        assert (offsets[:, 0] <= offsets[:, 1]).all()
        assert (offsets >= 0).all() and (offsets <= len(generated['response'])).all()
        assert (np.diff(offsets[:, 0]) >= 0).all()
        items = generated['items']
        assert len(items) == row['expected_items']
        ids = [i['item_id'] for i in items]
        for item in items:
            if item['parse_ok']:
                assert generated['response'][item['start']:item['end']] == item['text']
            else:
                parse_failures.append(item['item_id'])
        for stage in (*ITEM_FEATURES, 'baselines'):
            path = ROOT/'data'/stage/(rid+'.json')
            record = json.loads(path.read_text(encoding='utf-8'))
            assert record['row_id'] == rid and record['prompt_hash'] == generated['prompt_hash']
            assert record['generation_signature_hash'] == gen_signature
            assert record['stage_signature_hash'] == signatures[stage]
            assert record['source_generation_sha256'] == sha(gp)
            assert [i['item_id'] for i in record['items']] == ids
            counts[stage] += 1
            if stage == 'baselines':
                continue
            npz = path.with_suffix('.npz')
            assert record['arrays_sha256'] == sha(npz)
            with np.load(npz, allow_pickle=False) as arrays:
                assert arrays['item_ids'].tolist() == ids
                assert np.array_equal(arrays['response_token_offsets'], offsets)
                for key, shape in ITEM_FEATURES[stage].items():
                    arr = arrays[key]
                    assert arr.shape == (len(ids), *shape), (rid, key, arr.shape)
                    for index, item in enumerate(items):
                        if item['parse_ok'] and not np.isfinite(arr[index]).all():
                            nonfinite.append({'item_id': item['item_id'], 'stage': stage, 'feature': key})
            for slot in record['items']:
                if not items[ids.index(slot['item_id'])]['parse_ok']:
                    continue
                original = items[ids.index(slot['item_id'])]
                overlap = [j for j, (a, b) in enumerate(offsets) if b > original['start'] and a < original['end']]
                assert slot['response_token_indices'] == overlap
                if stage == 'features':
                    j = slot['last_content_response_token_index']
                    assert offsets[j, 0] <= original['last_content_character'] < offsets[j, 1]
                    assert slot['last_content_absolute_token_index'] == generated['input_tokens'] + j
                elif stage == 'attention':
                    assert slot['last_absolute_query_position'] == generated['input_tokens'] + max(overlap)
                else:
                    assert slot['score_available_after_response_token_index'] == max(overlap)
            if stage == 'attention':
                assert all(0 <= i < generated['input_tokens'] for i in record['context_token_indices'])
            if stage == 'lumina':
                assert record['same_response_ids_both_passes']
                assert record['response_tokens'] == len(offsets)
                prior = pairs.setdefault(row['question_id'], record['random_context_sha256'])
                assert prior == record['random_context_sha256'], 'Paired conditions changed random sources'
        counts['generation'] += 1
        counts['items'] += len(items)
        counts['truncated_responses'] += int(generated['truncated'])
    result = {'status': 'passed' if not nonfinite else 'requires_feature_review',
              'data_file': data_file, 'data_sha256': sha(ROOT/data_file), 'counts': dict(counts),
              'max_input_tokens': source_max_tokens, 'parse_failures': parse_failures,
              'nonfinite_valid_item_features': nonfinite, 'labels_read': False,
              'detector_performance_computed': False,
              'checks': ['Exact input/output identities and source hashes', 'Frozen extractor code signatures',
                         'All requested methods and items present', 'Saved array hashes, shapes and finite item values',
                         'UTF-8 response character spans and identical token offsets across methods',
                         'All aggregated token positions within the declared item',
                         'Hidden-state last-content position and attention/LUMINA availability timing',
                         'LUMINA same original answer IDs and paired random context']}
    save(ROOT/output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert not nonfinite, 'Nonfinite features require review'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='data/inputs.jsonl')
    parser.add_argument('--output', default='data/runtime_audit.json')
    arguments = parser.parse_args()
    audit(arguments.data_file, arguments.output)
