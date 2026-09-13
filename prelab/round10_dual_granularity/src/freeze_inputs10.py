"""Freeze independently approved fresh test inputs before their first generation."""
import json
from pathlib import Path

import run10
import evaluate10

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = ROOT/'data/freeze.json'
    assert not lock.exists(), 'Do not replace input freeze'
    dev = run10.readl(ROOT/'data/dev_inputs.jsonl')
    test = run10.readl(ROOT/'data/test_inputs.jsonl')
    assert len(dev) == 320 and len(test) == 120
    assert all(r['split'] in ('train', 'validation') for r in dev)
    assert all(r['split'] == 'test' for r in test)
    approval = json.loads((ROOT/'data/reviews/final_source_approval.json').read_text('utf-8'))
    assert approval['status'] == 'approved' and approval['approved_test_groups'] == 60
    assert approval['test_inputs_sha256'] == run10.sha(ROOT/'data/test_inputs.jsonl')
    assert approval['source_reviewed'] and not approval['new_model_outputs_used']
    rows = dev+test
    assert len({r['row_id'] for r in rows}) == len(rows)
    assert not ({r['group_id'] for r in dev} & {r['group_id'] for r in test})
    for row in rows:
        run10.run9.safe_identity(row); run10.run9.validate_visible_strings(row)
        assert row['expected_items'] == len(row['questions']) == 1
    assert len({r['group_id'] for r in test}) == 60
    for rid in [r['row_id'] for r in test]:
        assert not (ROOT/'data/generation_records'/(rid+'.json')).exists(), 'Test already generated'
    run10.savel(ROOT/'data/inputs.jsonl', rows)
    names = {'data/inputs.jsonl', 'data/dev_inputs.jsonl', 'data/test_inputs.jsonl',
             'data/reuse_manifest.json', 'protocol.json', 'PLAN.md', 'ANNOTATION_GUIDE.md',
             'FEATURES.md', 'EVALUATION.md'}
    for folder in ('src', 'data/curation', 'data/reviews'):
        names.update(f.relative_to(ROOT).as_posix() for f in (ROOT/folder).rglob('*')
                     if f.is_file() and f.suffix in ('.py', '.json', '.jsonl', '.md'))
    for optional in ('data/test_references.jsonl', 'data/references.jsonl'):
        if (ROOT/optional).exists(): names.add(optional)
    freeze = {'schema': 'round10-input-freeze-v1', 'status': 'frozen',
              'created_before_test_generation': True, 'expected_rows': len(rows),
              'new_test_groups': 60, 'source_approval_sha256': run10.sha(ROOT/'data/reviews/final_source_approval.json'),
              'files_sha256': {name: run10.sha(ROOT/name) for name in sorted(names)},
              'external_source_sha256': {name: run10.sha(path) for name, path in evaluate10.EXTERNAL_SOURCES.items()}}
    run10.save(lock, freeze)
    run10.verify_inputs(rows)
    run10.pack(rows)
    print('FROZEN_FRESH_TEST_INPUTS',len(test),flush=True)


if __name__ == '__main__': main()
