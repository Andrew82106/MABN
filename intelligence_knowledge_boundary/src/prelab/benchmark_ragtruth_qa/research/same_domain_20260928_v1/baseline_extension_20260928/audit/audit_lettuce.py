"""CPU-only independent audit of LettuceDetect's actual prepared labels/mapping."""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
sys.path.insert(0, str(OUT))
from evaluate_extensions import sha, load, save, indexed


def main():
    sem = OUT.parent / 'semantic'
    prepared = sem / 'lettuce_v1_run'
    receipt = load(prepared / 'PREPARED.json')
    for path, expected in receipt['sources'].items():
        assert sha(path) == expected
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(receipt['protocol']['base'], local_files_only=True)
    total = native_labels = whitespace_tokens = 0
    training_check = None
    if (prepared / 'TRAIN_COMPLETE.json').exists():
        trained = load(prepared / 'TRAIN_COMPLETE.json')
        selected = load(prepared / 'SELECTED.json')
        history = load(prepared / 'HISTORY.json')
        assert trained['epochs'] == 6 and [r['epoch'] for r in history] == list(range(1, 7))
        best = max(history, key=lambda r: (r['evaluation']['selection_key'], -r['epoch']))
        assert best == selected and trained['selected_epoch'] == selected['epoch']
        assert sha(prepared / 'best.pt') == trained['checkpoint_sha256']
        assert (sha(prepared / 'validation_predictions.jsonl') == trained['validation_predictions_sha256'] ==
                sha(prepared / f'validation_epoch{selected["epoch"]}.jsonl'))
        training_check = {'epochs': 6, 'selected_epoch': selected['epoch'],
            'selection_recomputed_from_validation_history': True,
            'checkpoint_sha256_verified': trained['checkpoint_sha256'],
            'validation_replay_identical_to_selected_epoch': True}
    split_checks, prediction_checks = {}, {}
    for part in ['fit', 'validation', 'test']:
        records = indexed(prepared / f'{part}_prepared.jsonl')
        inputs = indexed(RUN / 'data' / part / 'inputs.jsonl')
        geometry = indexed(RUN / 'data' / part / 'geometry.jsonl')
        gold = indexed(RUN / 'data' / part / 'answers.jsonl') if part != 'test' else {}
        assert records.keys() == inputs.keys() == geometry.keys()
        for rid, r in records.items():
            inp, geo = inputs[rid], geometry[rid]
            text = inp['original_response']
            enc = tokenizer(inp['released_prompt'], text, return_offsets_mapping=True, truncation=False)
            assert enc['input_ids'] == r['input_ids']
            seq_ids = enc.sequence_ids()
            expected_offsets = [list(o) if seq_ids[i] == 1 else [-1, -1] for i, o in enumerate(enc['offset_mapping'])]
            assert expected_offsets == r['native_offsets']
            assert enc['offset_mapping'][r['answer_start']][0] == 0
            assert seq_ids[r['answer_start']] == 1
            if part != 'test':
                spans = [s for s in gold[rid]['released_hallucinations'] if s.get('valid', True)]
                expected = []
                for i, (a, b) in enumerate(expected_offsets):
                    expected.append(-100 if i < r['answer_start'] else
                                    int(a >= 0 and b > a and any(b > s['start'] and a < s['end'] for s in spans)))
                assert expected == r['labels'], (rid, 'native train/validation span labels')
                native_labels += sum(x != -100 for x in expected)
            else:
                assert 'labels' not in r
            ii, jj, weights = r['mapping']
            n = len(geo['lexical_mask'])
            assert r['original_token_count'] == n
            assert len(ii) == len(jj) == len(weights)
            assert all(0 <= i < n and r['answer_start'] <= j < len(r['input_ids'])
                       and expected_offsets[j][0] >= 0 and w > 0
                       for i, j, w in zip(ii, jj, weights))
            totals = np.bincount(ii, weights=weights, minlength=n)
            assert np.allclose(totals, 1., atol=1e-12, rtol=0), (rid, 'zero-filled or nonnormalized mapping')
            actual_by_token = defaultdict(dict)
            for i, j, w in zip(ii, jj, weights):
                assert j not in actual_by_token[i]
                actual_by_token[i][j] = w
            # Independently reconstruct lexical/punctuation character weights.
            owners = [[] for _ in text]
            for j, (a, b) in enumerate(expected_offsets):
                if a >= 0 and b > a:
                    for k in range(a, b):
                        owners[k].append(j)
            for i, (a, b) in enumerate(geo['response_token_offsets']):
                chars = [k for k in range(a, b) if not text[k].isspace()]
                if not chars:
                    whitespace_tokens += 1
                    # These tokens must still have native-derived mass; the
                    # adapter may use a fixed nearest-token rule for whitespace
                    # omitted by ModernBERT offsets.
                    continue
                expected_weights = defaultdict(float)
                for k in chars:
                    assert owners[k]
                    for j in owners[k]:
                        expected_weights[j] += 1 / (len(chars) * len(owners[k]))
                assert set(expected_weights) == set(actual_by_token[i])
                assert all(abs(w - actual_by_token[i][j]) <= 1e-12 for j, w in expected_weights.items())
            total += 1
        split_checks[part] = {'rows': len(records), 'prepared_sha256': sha(prepared / f'{part}_prepared.jsonl')}
        prediction_path = prepared / f'{part}_predictions.jsonl'
        if part in ['validation', 'test'] and prediction_path.exists():
            predictions = indexed(prediction_path)
            assert predictions.keys() == records.keys()
            for rid, pred in predictions.items():
                r = records[rid]
                assert pred['response_sha256'] == r['response_sha256']
                assert pred['native_answer_offsets'] == r['native_offsets'][r['answer_start']:]
                native = np.zeros(len(r['input_ids']), dtype=np.float64)
                native[r['answer_start']:] = pred['native_token_scores']
                ii, jj, ww = r['mapping']
                recomputed = np.bincount(ii, weights=native[jj] * ww, minlength=r['original_token_count'])
                assert np.allclose(recomputed, pred['token_scores'], rtol=0, atol=1e-12), (rid, 'actual score projection')
            prediction_checks[part] = {'rows': len(predictions), 'sha256': sha(prediction_path),
                                      'actual_projection_recomputed_from_saved_native_scores': True}
    assert total == 1842
    report = {'status': 'PASS', 'all_answers_independently_retokenized': total,
        'native_train_and_validation_labels_checked': native_labels,
        'all_native_labels_match_original_valid_character_spans': True,
        'all_inputs_untruncated': True, 'no_labels_in_test_prepared': True,
        'all_target_token_mapping_weights_sum_to_one': True,
        'nonwhitespace_mapping_independently_reconstructed': True,
        'whitespace_target_tokens_with_native_derived_nonzero_mass': whitespace_tokens,
        'splits': split_checks, 'prepared_receipt_sha256': sha(prepared / 'PREPARED.json'),
        'completed_prediction_checks': prediction_checks,
        'training_check': training_check,
        'code_sha256': {str(sem / n): sha(sem / n) for n in ['lettuce_v1.py', 'common.py']},
        'GPU_used': False, 'stage': 'input/label/mapping plus available actual native-score projections'}
    save(OUT / 'LETTUCE_INPUT_AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
