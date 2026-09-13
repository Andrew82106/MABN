"""Auditable assertion-only repair; never alter a score, model or threshold."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'results'
    assert not any((out/n).exists() for n in ('metrics_main.json', 'metrics_external.json', 'test_complete.json'))
    freeze_path, start_path = out/'freeze.json', out/'test_started.json'
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    started = json.loads(start_path.read_text(encoding='utf-8'))
    source = ROOT/'src/evaluate7.py'
    assert frozen['evaluator_sha256'] == sha(source)
    assert started['freeze_sha256'] == sha(freeze_path)
    assert sha(out/'frozen_models.pkl') == frozen['frozen_model_sha256']
    assert sha(out/'selection.json') == frozen['selection_sha256']
    for rel, h in frozen['data_sha256'].items():
        assert sha(ROOT/rel) == h
    for split, h in {**frozen['development_annotation_sha256'], **started['held_out_annotation_sha256']}.items():
        assert sha(ROOT/'data'/f'annotations_{split}.jsonl') == h
    archive = out/'numerical_fix_20260910'
    assert not archive.exists(), 'Repair already applied'
    archive.mkdir()
    for path in (source, freeze_path, start_path):
        (archive/path.name).write_bytes(path.read_bytes())
    before = source.read_bytes()
    old = b'assert v is None or finite(v) and 0 <= v <= 1'
    new = b'assert v is None or finite(v) and 0 <= v <= 1 + 1e-6  # float32 probability-sum rounding'
    assert before.count(old) == 1
    source.write_bytes(before.replace(old, new))
    record = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Two external direct-check scores sum float32 softmax entries to '
                  '1.0000000298023224 and 1.0000000074505806. The original strict '
                  'unit-interval assertion rejected ordinary rounding before results were saved.',
        'affected_item_ids': ['ragognize_test_1258__partial__1', 'ragognize_test_1633__partial__1'],
        'scope': 'Exactly one assertion upper bound changed from 1 to 1+1e-6. '
                 'Lower bound, finite checks, all raw scores, classifier code, '
                 'weights, selected parameters and thresholds are unchanged.',
        'evaluator_before_sha256': sha(archive/'evaluate7.py'),
        'evaluator_after_sha256': sha(source),
        'original_freeze_sha256': sha(archive/'freeze.json'),
        'original_test_started_sha256': sha(archive/'test_started.json'),
        'frozen_model_sha256': frozen['frozen_model_sha256'],
        'selection_sha256': frozen['selection_sha256'],
        'test_metrics_previously_saved_or_inspected': False,
        'training_repeated': False, 'thresholds_or_labels_changed': False,
    }
    record_path = archive/'repair.json'
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    frozen['evaluator_sha256'] = sha(source)
    frozen['numerical_validation_repair'] = {
        'file': 'results/numerical_fix_20260910/repair.json', 'sha256': sha(record_path),
        'original_freeze_sha256': record['original_freeze_sha256'],
        'no_retraining_retuning_or_score_changes': True,
    }
    freeze_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    started['freeze_sha256'] = sha(freeze_path)
    started['numerical_validation_repair_sha256'] = sha(record_path)
    started['original_attempt_manifest_sha256'] = record['original_test_started_sha256']
    start_path.write_text(json.dumps(started, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
