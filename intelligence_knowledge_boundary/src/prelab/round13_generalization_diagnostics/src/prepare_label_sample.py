"""Prepare a blinded diagnostic sample from training labels; no model scores."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'round12_conditional_fusion'
spec = importlib.util.spec_from_file_location('r13_sample_r12', PREVIOUS/'src/run12.py')
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)


def main():
    meta = r12.r10.metadata(r12.SOURCE)
    items, _, _, _ = r12.r10.cohort(r12.SOURCE, 'train', meta)
    by_input = {x['row_id']: x for x in meta[0]}
    selected, used = [], set()
    rng = np.random.default_rng(20260913)
    for category in sorted({i['category'] for i in items}):
        for risk in [1, 0]:
            candidates = sorted([i for i in items if i['category'] == category and i['asserted_eligible']
                                 and i['localization_status'] == 'resolved' and i['gold'] == risk
                                 and i['group_id'] not in used], key=lambda i: i['item_id'])
            assert len(candidates) >= 3
            picked = []
            for j in rng.permutation(len(candidates)):
                item = candidates[int(j)]
                if item['group_id'] in used:
                    continue
                picked.append(item); used.add(item['group_id'])
                if len(picked) == 3:
                    break
            assert len(picked) == 3
            selected.extend(picked)
    rng.shuffle(selected)
    blind, key = [], []
    for j, item in enumerate(selected, 1):
        sid = f'review_{j:02d}'; row = by_input[item['row_id']]; generation = meta[2][item['row_id']][0]
        blind.append({'id': sid, 'question': row['questions'][0], 'passages': row['passages'],
                      'answer': generation['response'], 'span_coordinate': 'Unicode character offsets in this exact answer string, end exclusive'})
        key.append({'id': sid, 'item_id': item['item_id'], 'row_id': item['row_id'], 'group_id': item['group_id'],
                    'category': item['category'], 'old_gold': item['gold'], 'old_spans': item['annotation']['risk_spans']})
    assert len(blind) == len(used) == 30
    (ROOT/'data').mkdir(parents=True, exist_ok=True); (ROOT/'results').mkdir(exist_ok=True)
    r12.r10.savel(ROOT/'data/blind_label_sample.jsonl', blind)
    r12.r10.savel(ROOT/'data/label_sample_key.jsonl', key)
    print('30 unique training groups; no score/old label/condition in blind file.')


if __name__ == '__main__':
    main()
