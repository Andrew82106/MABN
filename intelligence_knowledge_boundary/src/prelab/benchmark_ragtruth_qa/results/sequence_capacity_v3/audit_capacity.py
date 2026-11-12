"""Read-only CPU arithmetic/checkpoint audit, without fitting or test access."""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_sequence_capacity as cap
spec = importlib.util.spec_from_file_location('old_sequence_audit_helpers', cap.OLD / 'audit_sequence.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def main():
    start = time.perf_counter(); torch.set_num_threads(4); assert not torch.cuda.is_initialized()
    complete = cap.read(OUT / 'complete.json')
    for name, expected in complete['files_sha256'].items(): assert cap.sha(OUT / name) == expected
    meta, index, snapshot = cap.source(); assert snapshot == cap.read(OUT / 'source_snapshot.json')
    summary = cap.read(OUT / 'summary.json'); diagnostics = cap.read(OUT / 'DIAGNOSTICS.json')
    old_summary = cap.read(cap.SOURCE / 'summary.json')
    with np.load(cap.OLD / 'fit_token_weights.npz', allow_pickle=False) as z: weights = {k: z[k].copy() for k in z.files}
    normalized = np.load(cap.DESIGN, mmap_mode='r')
    lexical = np.concatenate([np.asarray(t['lexical_mask'], bool) for t in meta['tokens']])
    yt = np.concatenate([np.asarray(t['risk_mask'], int) for t in meta['tokens']])
    starts = {a['response_id']: a['left'] for a in index}
    assert [a['response_id'] for a in index] == [a['response_id'] for a in meta['tokens']] == [a['response_id'] for a in meta['answers']]
    ix = np.asarray([[starts[w['response_id']] + k for k in w['token_indices']] for w in meta['windows']])
    wlex = lexical[ix]; assert wlex.any(1).all()
    yw = np.asarray([w['label'] for w in meta['windows']]); ya = np.asarray([a['label'] for a in meta['answers']])
    assert list(meta['answer_windows']) == [a['response_id'] for a in meta['answers']]
    answer_windows = list(meta['answer_windows'].values()); methods = []
    for method, width in cap.METHODS.items():
        choices = []
        for epoch in range(1, 31):
            report = cap.read(OUT / method / f'epoch_{epoch:03d}.json')
            with np.load(OUT / method / f'epoch_{epoch:03d}_scores.npz', allow_pickle=False) as z:
                p, logits, window, answer = (z[k] for k in ('token_scores', 'token_logits', 'window_scores', 'answer_scores'))
            assert np.array_equal(torch.sigmoid(torch.from_numpy(logits)).numpy(), p)
            expected = np.where(wlex, p[ix], -np.inf).max(1).astype(np.float64)
            assert np.array_equal(expected, window)
            assert np.array_equal(np.asarray([expected[aix].max() for aix in answer_windows]), answer)
            tw = helper.threshold(yw[168123:], window[168123:]); ta = helper.threshold(ya[634:], answer[634:])
            for unit, result in (('window', tw), ('answer', ta)):
                saved = report['thresholds'][unit]; assert result == (saved['f1'], saved['precision'], saved['threshold'])
            for part, wix, aix in (('fit', slice(0, 168123), slice(0, 634)), ('calibration', slice(168123, None), slice(634, None))):
                for unit, y, score, threshold in (('windows', yw[wix], window[wix], tw[2]), ('answers', ya[aix], answer[aix], ta[2])):
                    for key, value in helper.counts(y, score, threshold).items(): assert value == report['metrics'][part][unit][key]
            fl = logits[:170361].astype(np.float64)
            bce = float(weights['loss'] @ (np.logaddexp(0, fl) - yt[:170361] * fl) / 139518)
            assert bce == report['full_fit_weighted_BCE']
            choices.append(([min(tw[0], ta[0]), tw[0], tw[1], -epoch], epoch))
        epoch = max(choices, key=lambda x: x[0])[1]; assert epoch == summary['selected'][method]['epoch']
        model = cap.new_model(width); checkpoint = torch.load(OUT / method / f'epoch_{epoch:03d}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['protocol_sha256'] == cap.sha(OUT / 'protocol.json')
        model.load_state_dict(checkpoint['model_state_dict']); logits, p = cap.full.predict(model, normalized, index)
        with np.load(OUT / method / f'epoch_{epoch:03d}_scores.npz', allow_pickle=False) as z:
            assert np.array_equal(logits, z['token_logits']) and np.array_equal(p, z['token_scores'])
        methods.append({'method': method, 'width': width, 'parameters': sum(p.numel() for p in model.parameters()),
                        'selected_epoch': epoch, 'all30_epoch_aggregation_threshold_metrics_BCE_exact': True,
                        'selected_full_token_checkpoint_predictions_exact': True})
    # Reconstruct highlighted raw positions globally, independently of set-based reporting.
    spans_by_type = {}
    for kind in ('Evident Conflict', 'Subtle Conflict'):
        spans = []; risk = np.zeros(213159, bool); empty = 0
        for tokens in meta['tokens']:
            if tokens['partition'] != 'calibration': continue
            for mapping in tokens['span_token_mapping']:
                if tokens['original_labels'][mapping['span_index']]['label_type'] != kind: continue
                pos = np.asarray(mapping['risk_token_indices'], int) + starts[tokens['response_id']]
                if not len(pos): empty += 1; continue
                risk[pos] = True; spans.append(pos)
        spans_by_type[kind] = (spans, empty, risk[ix].any(1))
    for method, typed in diagnostics['calibration_conflict'].items():
        folder = OUT if method in cap.METHODS else cap.SOURCE
        selected = (summary if folder == OUT else old_summary)['selected'][method]
        path = folder / method / f"epoch_{selected['epoch']:03d}_scores.npz"
        assert cap.sha(path) == selected['scores_sha256']
        with np.load(path, allow_pickle=False) as z: score = z['window_scores']
        alerts = score >= selected['thresholds']['window']['threshold']; highlighted = np.zeros(213159, bool)
        highlighted[ix[alerts].ravel()] = True
        for kind, (spans, empty, positive) in spans_by_type.items():
            expected = {'original_spans': len(spans) + empty, 'localizable_spans': len(spans), 'unlocalizable_spans': empty,
                        'span_any_hit': sum(bool(highlighted[pos].any()) for pos in spans),
                        'span_full_hit': sum(bool(highlighted[pos].all()) for pos in spans),
                        'positive_windows': int(positive.sum()), 'hit_positive_windows': int((positive & alerts).sum())}
            for key, value in expected.items(): assert typed[kind][key] == value
    assert cap.source()[-1] == snapshot and not torch.cuda.is_initialized()
    result = {'status': 'passed', 'code_sha256': cap.sha(Path(__file__)), 'complete_sha256': cap.sha(OUT / 'complete.json'),
              'all_source_and_output_hashes_verified': True, 'exact_frozen_width64_input_reused': True,
              'scaler_PCA_weights_shuffle_not_refit_or_changed': True, 'methods': methods,
              'all_three_widths_conflict_counts_independently_exact': True, 'frozen_width64_not_retrained_or_changed': True,
              'test_opened': False, 'gpu_used': False, 'seconds': time.perf_counter() - start,
              'limitation': 'Implementation self-audit; repeated calibration selection and single-seed limits remain.'}
    cap.save(OUT / 'AUDIT.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
