"""Label-independent source-group folds for honest branch stacking."""
from pathlib import Path
import hashlib
import json
from collections import Counter

HERE = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (HERE / 'data/split_assignments.jsonl').read_text(encoding='utf-8').splitlines() if s]
fit = [r for r in rows if r['split'] == 'fit']
groups = sorted({r['group_id'] for r in fit}, key=lambda g: hashlib.sha256(('20260928-oof|' + g).encode()).hexdigest())
folds = {g: i % 3 for i, g in enumerate(groups)}
out = HERE / 'fusion/folds'
out.mkdir(parents=True, exist_ok=True)
manifest = {'seed': '20260928-oof', 'folds': 3, 'uses_labels': False, 'groups': len(groups), 'answers': len(fit), 'members': []}
for k in range(3):
    train = sorted(r['response_id'] for r in fit if folds[r['group_id']] != k)
    held = sorted(r['response_id'] for r in fit if folds[r['group_id']] == k)
    assert set(train).isdisjoint(held) and len(train) + len(held) == len(fit)
    for name, ids in [('fit', train), ('held', held)]:
        path = out / f'fold_{k}_{name}_ids.json'
        payload = json.dumps(ids, ensure_ascii=False, indent=2) + '\n'
        if path.exists():
            assert path.read_text(encoding='utf-8') == payload
        else:
            path.write_text(payload, encoding='utf-8')
    manifest['members'].append({'fold': k, 'fit_answers': len(train), 'held_answers': len(held),
        'held_groups': sum(v == k for v in folds.values())})
manifest['group_to_fold'] = folds
(out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in manifest.items() if k != 'group_to_fold'}, ensure_ascii=False))
