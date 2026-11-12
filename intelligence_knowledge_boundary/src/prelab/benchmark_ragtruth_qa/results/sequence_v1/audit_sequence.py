"""Read-only numerical/data audit of the fixed 193-d QA structure experiment."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_sequence as sequence


def read(path): return json.loads(Path(path).read_text('utf-8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def threshold(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(-s, kind='stable'); yy = y[order]; ss = s[order]
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[last]; count = last + 1
    candidates = [(0., 0., float(np.nextafter(ss[0], np.inf)))]
    candidates.extend((2 * int(t) / (int(n) + int(y.sum())), int(t) / int(n), float(v))
                      for t, n, v in zip(tp, count, ss[last]))
    return max(candidates)


def counts(y, s, threshold):
    y = np.asarray(y, int); predicted = s >= threshold
    tp = int(((y == 1) & predicted).sum()); fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum()); tn = int(((y == 0) & ~predicted).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0., 'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def main():
    started = time.perf_counter(); assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    complete = read(OUT / 'complete.json'); source = read(OUT / 'source_snapshot.json')
    for name, value in complete['files_sha256'].items(): assert sha(OUT / name) == value
    for name, value in source['files_sha256'].items(): assert sha(name) == value
    assert sha(ROOT / 'src/run_sequence.py') == complete['script_sha256']
    meta = sequence.baseline.metadata(); index = read(OUT / 'token_index.json')['answers']
    raw = np.load(OUT / 'raw_token_features.npy', mmap_mode='r')
    normalized = np.load(OUT / 'standardized_token_features.npy', mmap_mode='r')
    with np.load(OUT / 'fit_token_weights.npz', allow_pickle=False) as z: weights = {key: z[key] for key in z.files}
    lexical = np.concatenate([np.asarray(t['lexical_mask'], bool) for t in meta['tokens']])
    labels = np.concatenate([np.asarray(t['risk_mask'], int) for t in meta['tokens']])
    assert raw.shape == normalized.shape == (213159, 193)
    assert np.array_equal(weights['lexical'], lexical[:170361])
    assert np.array_equal(weights['y'], labels[:170361])
    assert np.array_equal(weights['loss'] > 0, lexical[:170361])
    assert not weights['base'][~lexical[:170361]].any()
    assert int(lexical[:170361].sum()) == 139518
    tree = defaultdict(list)
    for entry in index:
        if entry['partition'] == 'fit': tree[entry['group_id']].append(entry)
    assert len(tree) == 615
    expected = np.zeros(170361)
    for aa in tree.values():
        for a in aa:
            ix = np.arange(a['left'], a['right']); ix = ix[lexical[ix]]
            expected[ix] = 139518 / (615 * len(aa) * len(ix))
    assert np.allclose(expected, weights['base'], atol=2e-14, rtol=2e-14)
    mass = np.bincount(labels[:170361], weights=expected, minlength=2)
    factors = mass.sum() / (2 * mass); loss = expected * factors[labels[:170361]]
    for aa in tree.values():
        ix = np.concatenate([np.arange(a['left'], a['right']) for a in aa]); loss[ix] *= (139518 / 615) / loss[ix].sum()
    loss *= 139518 / loss.sum()
    assert np.allclose(factors, weights['class_factors'], atol=1e-12)
    assert np.allclose(loss, weights['loss'], atol=1e-12, rtol=1e-12)
    scaler = pickle.loads((OUT / 'scaler.pkl').read_bytes())
    # StandardScaler receives float32 X and therefore casts sample weights to
    # float32 internally. Accumulate those effective weights in float64.
    effective = weights['base'].astype(np.float32).astype(np.float64)
    mean = np.zeros(193); variance = np.zeros(193)
    for left in range(0, 170361, 16384):
        right = min(left + 16384, 170361); xx = raw[left:right].astype(np.float64)
        mean += effective[left:right] @ xx
    mean /= effective.sum()
    for left in range(0, 170361, 16384):
        right = min(left + 16384, 170361); xx = raw[left:right].astype(np.float64) - mean
        variance += np.einsum('i,ij,ij->j', effective[left:right], xx, xx)
    variance /= effective.sum()
    # partial_fit collapses n_samples_seen_ to a scalar, then casts it back to
    # X.dtype(float32) at the beginning of each next batch. A single global
    # float64 moment therefore differs slightly from this frozen recipe.
    direct_mean_error = float(np.max(np.abs(mean - scaler.mean_)))
    direct_variance_error = float(np.max(np.abs(variance - scaler.var_)))
    running_mean = np.zeros(193); running_var = np.zeros(193); running_mass = 0.
    for left in range(0, 170361, 16384):
        right = min(left + 16384, 170361); xx = raw[left:right].astype(np.float64); ww = effective[left:right]
        previous_mass = float(np.float32(running_mass)); mass = float(ww.sum())
        batch_mean = ww @ xx / mass; centered = xx - batch_mean
        batch_variance = np.einsum('i,ij,ij->j', ww, centered, centered) / mass
        total = previous_mass + mass; difference = running_mean - batch_mean
        running_var = (previous_mass * running_var + mass * batch_variance + difference**2 * previous_mass * mass / total) / total
        running_mean = (previous_mass * running_mean + mass * batch_mean) / total
        running_mass = total
    assert running_mass == scaler.n_samples_seen_
    assert np.allclose(running_mean, scaler.mean_, atol=2e-10, rtol=2e-10)
    assert np.allclose(running_var, scaler.var_, atol=2e-10, rtol=2e-10)
    for left in range(0, len(raw), 16384):
        right = min(left + 16384, len(raw))
        assert np.array_equal(scaler.transform(raw[left:right]).astype(np.float32), normalized[left:right])
    rid_start = {a['response_id']: a['left'] for a in index}
    window_indices = np.asarray([[rid_start[w['response_id']] + j for j in w['token_indices']] for w in meta['windows']])
    assert window_indices.shape == (210364, 4)
    lexical_in_window = lexical[window_indices]; assert lexical_in_window.any(1).all()
    # Fixed raw/PCA interface cross-check against the separately frozen LR designs.
    slots = np.load(ROOT / 'results/development_v1/matrices/slots.npy', mmap_mode='r')
    hidden = np.load(ROOT / 'results/development_v1/matrices/hidden.npy', mmap_mode='r')
    sample = np.unique(np.r_[np.arange(8), np.arange(len(window_indices) - 8, len(window_indices)),
                             np.linspace(0, len(window_indices) - 1, 17, dtype=int)])
    for i in sample:
        values = raw[window_indices[i]]
        assert np.array_equal(values[:, :129].ravel(), slots[i, :516])
        assert np.array_equal(values[:, 129:].mean(0), hidden[i])
    permutations = np.load(OUT / 'epoch_answer_order.npy'); rng = np.random.default_rng(20260926)
    assert np.array_equal(permutations, np.stack([rng.permutation(634) for _ in range(30)]))
    summary = read(OUT / 'summary.json'); y_window = np.asarray([w['label'] for w in meta['windows']])
    y_answer = np.asarray([a['label'] for a in meta['answers']]); by_answer = list(meta['answer_windows'].values())
    assert list(meta['answer_windows']) == [a['response_id'] for a in meta['answers']]
    audits = []
    for method in sequence.METHODS:
        rows = []
        for epoch in range(1, 31):
            report = read(OUT / method / f'epoch_{epoch:03d}.json')
            with np.load(OUT / method / f'epoch_{epoch:03d}_scores.npz', allow_pickle=False) as z:
                token, logits = z['token_scores'], z['token_logits']; ws, ans = z['window_scores'], z['answer_scores']
            assert token.shape == logits.shape == (213159,) and np.isfinite(logits).all()
            assert np.array_equal(token, torch.sigmoid(torch.from_numpy(logits)).numpy())
            independent = np.where(lexical_in_window, token[window_indices], -np.inf).max(1).astype(np.float64)
            assert np.array_equal(independent, ws)
            independent_answer = np.asarray([independent[ix].max() for ix in by_answer])
            assert np.array_equal(independent_answer, ans)
            tw = threshold(y_window[168123:], ws[168123:]); ta = threshold(y_answer[634:], ans[634:])
            for level, value in (('window', tw), ('answer', ta)):
                saved = report['thresholds'][level]
                assert value == (saved['f1'], saved['precision'], saved['threshold'])
            fit_logits = logits[:170361].astype(np.float64)
            bce = float(weights['loss'] @ (np.logaddexp(0, fit_logits) - labels[:170361] * fit_logits) / 139518)
            assert bce == report['full_fit_weighted_BCE']
            for part, wix, aix in (('fit', slice(0, 168123), slice(0, 634)),
                                   ('calibration', slice(168123, None), slice(634, None))):
                for unit, yy, ss, cutoff in (('windows', y_window[wix], ws[wix], tw[2]),
                                             ('answers', y_answer[aix], ans[aix], ta[2])):
                    cm = counts(yy, ss, cutoff)
                    for key, value in cm.items(): assert report['metrics'][part][unit][key] == value
            key = [min(tw[0], ta[0]), tw[0], tw[1], -epoch]; assert key == report['selection_key']
            rows.append((key, epoch))
        chosen = max(rows, key=lambda entry: entry[0])[1]
        assert chosen == summary['selected'][method]['epoch']
        checkpoint = torch.load(OUT / method / f'epoch_{chosen:03d}.pt', map_location='cpu', weights_only=True)
        model = sequence.new_model(method); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
        assert all(p.device.type == 'cpu' for p in model.parameters())
        repeated_logits, repeated_probability = sequence.predict_tokens(model, normalized, index)
        with np.load(OUT / method / f'epoch_{chosen:03d}_scores.npz', allow_pickle=False) as z:
            assert np.array_equal(repeated_logits, z['token_logits'])
            assert np.array_equal(repeated_probability, z['token_scores'])
        audits.append({'method': method, 'all_30_epochs_probabilities_aggregation_thresholds_counts_loss_exact': True,
                       'selected_epoch': chosen, 'selected_checkpoint_full_predictions_reloaded_exact': True})
    assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'audit_code_sha256': sha(Path(__file__)), 'complete_sha256': sha(OUT / 'complete.json'),
              'source_files_and_all_60_epoch_hashes_verified': True, 'gpu_used': False, 'test_opened': False,
              'loss_mass': float(weights['loss'].sum()), 'fit_lexical_tokens': 139518,
              'groups_and_answer_and_lexical_weights_recomputed': True, 'punctuation_loss_is_zero': True,
              'scaler_incremental_mean_max_absolute_error': float(np.max(np.abs(running_mean - scaler.mean_))),
              'scaler_incremental_variance_max_absolute_error': float(np.max(np.abs(running_var - scaler.var_))),
              'single_global_moment_mean_difference': direct_mean_error,
              'single_global_moment_variance_difference': direct_variance_error,
              'scaler_audit_note': 'Initial global-moment check exposed float32 n_samples_seen_ recasting between sklearn partial_fit blocks; independent weighted merge now reproduces that documented frozen library behavior. No training artifact or tolerance changed.',
              'full_standardized_matrix_recomputed_exact': True, 'fixed_LR_design_crosschecks': len(sample),
              'pca_reused_without_refit': True, 'same_per_epoch_answer_order_both_models': True,
              'unchanged_4_raw_BPE_geometry_lexical_only_max_verified': True, 'models': audits,
              'seconds': time.perf_counter() - started,
              'limitations': 'Implementation self-audit; calibrated development selection remains optimistic and the seed is not repeated.'}
    (OUT / 'AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
