"""Read-only CPU audit of fixed full-head TCN scores and type recalls."""
from pathlib import Path
import importlib.util
import json
import pickle
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_sequence_full as full
spec = importlib.util.spec_from_file_location('old_sequence_audit_helpers', full.OLD / 'audit_sequence.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def main():
    start = time.perf_counter(); assert not torch.cuda.is_initialized(); torch.set_num_threads(4)
    complete = full.read(OUT / 'complete.json'); prep = full.read(OUT / 'preparation_manifest.json')
    for manifest in (complete, prep):
        for name, expected in manifest['files_sha256'].items(): assert full.sha(OUT / name) == expected
    meta, fm, hm, index, _, source = full.source(); assert source == full.read(OUT / 'source_snapshot.json')
    summary = full.read(OUT / 'summary.json'); diagnostics = full.read(OUT / 'DIAGNOSTICS.json')
    with np.load(full.OLD / 'fit_token_weights.npz', allow_pickle=False) as z: weights = {key: z[key].copy() for key in z.files}
    common = np.load(OUT / 'matrices/common1025.npy', mmap_mode='r'); harp = np.load(OUT / 'matrices/harp64.npy', mmap_mode='r')
    previous = np.load(full.OLD / 'raw_token_features.npy', mmap_mode='r')
    assert common.shape == (213159, 1025) and harp.shape == (213159, 64)
    assert prep['weights_sha256'] == full.sha(full.OLD / 'fit_token_weights.npz')
    assert prep['shuffle_sha256'] == full.sha(full.OLD / 'epoch_answer_order.npy')
    feature_records = {r['response_id']: r for r in fm['records']}
    for entry in index:
        rid = entry['response_id']; left, right = entry['left'], entry['right']
        with np.load(ROOT / 'data/features' / (rid + '.npz'), allow_pickle=False) as z:
            assert np.array_equal(common[left:right, :1024], z['lb'])
            assert np.array_equal(common[left:right, 1024], z['nll'])
        side = full.read(ROOT / 'data/harp_features' / (rid + '.json'))
        assert side['source_npz_sha256'] == feature_records[rid]['npz_sha256']
        with np.load(ROOT / 'data/harp_features' / (rid + '.npz'), allow_pickle=False) as z:
            assert np.array_equal(harp[left:right], z['harp'][:, :64])
    lexical = np.concatenate([np.asarray(t['lexical_mask'], bool) for t in meta['tokens']])
    y_token = np.concatenate([np.asarray(t['risk_mask'], int) for t in meta['tokens']])
    starts = {entry['response_id']: entry['left'] for entry in index}
    window_index = np.asarray([[starts[w['response_id']] + k for k in w['token_indices']] for w in meta['windows']])
    window_lexical = lexical[window_index]; assert window_lexical.any(1).all()
    yw = np.asarray([w['label'] for w in meta['windows']]); ya = np.asarray([a['label'] for a in meta['answers']])
    answer_windows = list(meta['answer_windows'].values()); methods = []
    for method in full.METHODS:
        aux = previous[:, 129:193] if method == full.METHODS[0] else harp
        scaler = pickle.loads((OUT / (method + '_scaler.pkl')).read_bytes())
        normalized = np.load(OUT / 'matrices' / (method + '.npy'), mmap_mode='r')
        mu = np.zeros(1089); var = np.zeros(1089); mass = 0.
        for left in range(0, 170361, 16384):
            right = min(left + 16384, 170361); x = np.column_stack((common[left:right], aux[left:right])).astype(np.float64)
            w = weights['base'][left:right].astype(np.float32).astype(np.float64)
            old = float(np.float32(mass)); new = float(w.sum()); batch_mean = w @ x / new; centered = x - batch_mean
            batch_var = np.einsum('i,ij,ij->j', w, centered, centered) / new
            total = old + new; difference = mu - batch_mean
            var = (old * var + new * batch_var + difference**2 * old * new / total) / total
            mu = (old * mu + new * batch_mean) / total; mass = total
        assert np.allclose(mu, scaler.mean_, atol=2e-10, rtol=2e-10)
        assert np.allclose(var, scaler.var_, atol=2e-10, rtol=2e-10)
        for left in range(0, 213159, 16384):
            right = min(left + 16384, 213159); x = np.column_stack((common[left:right], aux[left:right]))
            assert np.array_equal(scaler.transform(x).astype(np.float32), normalized[left:right])
        choices = []
        for epoch in range(1, 31):
            report = full.read(OUT / method / f'epoch_{epoch:03d}.json')
            with np.load(OUT / method / f'epoch_{epoch:03d}_scores.npz', allow_pickle=False) as z:
                p, logits, window, answer = (z[k] for k in ('token_scores', 'token_logits', 'window_scores', 'answer_scores'))
            assert np.array_equal(torch.sigmoid(torch.from_numpy(logits)).numpy(), p)
            expected = np.where(window_lexical, p[window_index], -np.inf).max(1).astype(np.float64)
            assert np.array_equal(expected, window)
            assert np.array_equal(np.asarray([expected[ix].max() for ix in answer_windows]), answer)
            tw = helper.threshold(yw[168123:], window[168123:]); ta = helper.threshold(ya[634:], answer[634:])
            for level, result in (('window', tw), ('answer', ta)):
                saved = report['thresholds'][level]; assert result == (saved['f1'], saved['precision'], saved['threshold'])
            for part, wix, aix in (('fit', slice(0, 168123), slice(0, 634)), ('calibration', slice(168123, None), slice(634, None))):
                for unit, y, s, threshold in (('windows', yw[wix], window[wix], tw[2]), ('answers', ya[aix], answer[aix], ta[2])):
                    counts = helper.counts(y, s, threshold)
                    for key, value in counts.items(): assert value == report['metrics'][part][unit][key]
            fit_logits = logits[:170361].astype(np.float64)
            bce = float(weights['loss'] @ (np.logaddexp(0, fit_logits) - y_token[:170361] * fit_logits) / 139518)
            assert bce == report['full_fit_weighted_BCE']
            choices.append(([min(tw[0], ta[0]), tw[0], tw[1], -epoch], epoch))
        epoch = max(choices, key=lambda x: x[0])[1]; assert epoch == summary['selected'][method]['epoch']
        model = full.new_model(); checkpoint = torch.load(OUT / method / f'epoch_{epoch:03d}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict']); logits, probability = full.predict(model, normalized, index)
        with np.load(OUT / method / f'epoch_{epoch:03d}_scores.npz', allow_pickle=False) as z:
            assert np.array_equal(logits, z['token_logits']) and np.array_equal(probability, z['token_scores'])
        methods.append({'method': method, 'selected_epoch': epoch, 'all_30_epoch_thresholds_metrics_loss_exact': True,
                        'full_selected_checkpoint_prediction_reload_exact': True,
                        'scaler_mean_error': float(np.abs(mu - scaler.mean_).max()), 'scaler_variance_error': float(np.abs(var - scaler.var_).max())})
    # A direct global token mask independently checks all reported type recalls.
    span_maps = {}
    for part in ('fit', 'calibration'):
        for kind in full.protocol()['diagnostics']['types']:
            spans = []; risk_mask = np.zeros(213159, bool); empty = 0
            for tokens in meta['tokens']:
                if tokens['partition'] != part: continue
                for mapping in tokens['span_token_mapping']:
                    label = tokens['original_labels'][mapping['span_index']]
                    if label['label_type'] != kind: continue
                    ix = np.asarray(mapping['risk_token_indices'], int) + starts[tokens['response_id']]
                    if not len(ix): empty += 1; continue
                    risk_mask[ix] = True; spans.append(ix)
            span_maps[(part, kind)] = (spans, empty, risk_mask[window_index].any(1))
    for method, typed in diagnostics['by_error_type'].items():
        folder = OUT if method in full.METHODS else full.OLD
        selected = (summary if folder == OUT else full.read(folder / 'summary.json'))['selected'][method]
        with np.load(folder / method / f"epoch_{selected['epoch']:03d}_scores.npz", allow_pickle=False) as z: score = z['window_scores']
        alerts = score >= selected['thresholds']['window']['threshold']; highlighted = np.zeros(213159, bool)
        highlighted[window_index[alerts].ravel()] = True
        for (part, kind), (spans, empty, positive) in span_maps.items():
            saved = typed[part][kind]
            expected = {'original_spans': len(spans) + empty, 'localizable_spans': len(spans), 'unlocalizable_spans': empty,
                        'spans_any_hit': sum(bool(highlighted[ix].any()) for ix in spans),
                        'spans_fully_hit': sum(bool(highlighted[ix].all()) for ix in spans),
                        'positive_windows': int(positive.sum()), 'hit_positive_windows': int((positive & alerts).sum())}
            for key, value in expected.items(): assert saved[key] == value
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed', 'code_sha256': full.sha(Path(__file__)), 'complete_sha256': full.sha(OUT / 'complete.json'),
              'all_source_and_output_hashes_verified': True, 'all793_commonLB_NLL_and_bottom64_inputs_exact': True,
              'PCA_reused_without_refit': True, 'same_frozen_weights_and_shuffle': True, 'methods': methods,
              'all_four_methods_type_window_and_span_recall_counts_independently_exact': True,
              'no_frozen_training_files_changed': True, 'test_opened': False, 'gpu_used': False,
              'seconds': time.perf_counter() - start, 'limitation': 'Implementation self-audit; calibration and single-seed limits remain.'}
    full.save(OUT / 'AUDIT.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
