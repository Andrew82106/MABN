"""Matched CPU TCNs: full LB heads with PCA64 versus HARP bottom64."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F

import run_sequence as v1

ROOT, OLD = v1.ROOT, v1.OUT
OUT = ROOT / 'results/sequence_full_v2'
METHODS = ('full_lb_pca64_tcn', 'full_lb_harp64_tcn')
SEED, EPOCHS, BATCH, THREADS, WIDTH = 20260926, 30, 8, 4, 1089
sha, read, save, lines, digest = v1.sha, v1.read, v1.save, v1.lines, v1.digest


def protocol():
    return {'version': 'qa-full-head-matched-tcn-v2', 'scope': 'Only frozen fit634/calibration159; no test',
        'methods': list(METHODS), 'input_dimensions': WIDTH,
        'input_A': 'Each raw token: all1024 layer-major/head-minor LB heads + NLL1 + exact reused fit-only hiddenPCA64',
        'input_B': 'Same all1024 LB + NLL1 + first64 columns of frozen HARP256, ascending smallest head eigenvalues',
        'harp': 'Reuse frozen data/harp_features and data/harp_basis; no basis fit, no reprojection during formal preparation',
        'pca': 'Reuse exact sequence_v1 raw_token_features columns129:193, bound to original development_v1 fit-only PCA64',
        'network': '1089->64 GELU, two residual ONE-Conv1d64->64 blocks kernel3 dilation1/2 GELU dropout.2, ->1',
        'padding': 'Same zero state masking after projection/each block as sequence_v1; each answer separate',
        'receptive_field_raw_tokens': 7, 'offline_future_tokens': True,
        'seed': SEED, 'epochs_per_model': EPOCHS, 'batch_answers': BATCH, 'cpu_threads': THREADS,
        'initialization': 'Reset same seed before each same-shape network, so all initial parameter tensors exactly match',
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01, 'all_parameters': True},
        'weights': 'Reuse exact sequence_v1 fit_token_weights: group/answer/lexical equal base, class factors then group rebalance, loss mass139518',
        'scaler': 'Separate fit-only lexical-base-weighted StandardScalers, same partial_fit blocks16384 and float32 transform; shared1025 dimensions must match exactly',
        'shuffle': 'Reuse identical sequence_v1 epoch_answer_order.npy, not a new random schedule',
        'loss': 'Same sum(w*BCE_logits)*634/(actual_batch_answers*139518); punctuation/padding weight0',
        'aggregation': 'Same unchanged eligible4 raw BPE windows; max only over output-defined lexical tokens, then all-window max for answer',
        'selection': 'Same cal max min(windowF1,answerF1), then windowF1/precision, earlier epoch; separate global thresholds',
        'early_stop': False, 'new_seeds_or_hyperparameters': False,
        'diagnostics': {'types': ['Evident Conflict', 'Subtle Conflict', 'Evident Baseless Info', 'Subtle Baseless Info'],
            'window_recall': 'Unique positive windows intersecting lexical risk tokens of each original type; overlap between types allowed',
            'span_recall': 'Per original span with lexical risk tokens: any token covered / every token covered by alerted windows',
            'empty_lexical_spans': 'Report original count and unlocalizable count, never change official answer labels',
            'thresholds': 'Use same selected global window threshold; never tune by error type',
            'case_selection': 'For cal Conflict spans: first3 newly hit and first3 lost B versus A, response_id then span_index order'},
        'comparisons': 'Retain selected sequence_v1 MLP/193dimTCN; frozen LR and HARP LR candidates reported without refit',
        'limitations': ['Cal selects30 checkpoints and thresholds; not fresh-test performance.',
            'A/B input widths and parameter counts match. Comparing A to193dimTCN also increases projection parameters, so it is not a pure information intervention.',
            'A/B are different64dim hidden projections. A poor result cannot establish absence of information in full hidden states.',
            'One fixed seed only; no significance or unseen-generalization claim.']}


class FullTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(WIDTH, 64)
        self.convolutions = nn.ModuleList([nn.Conv1d(64, 64, 3, padding=d, dilation=d) for d in (1, 2)])
        self.dropout = nn.Dropout(.2); self.output = nn.Linear(64, 1)

    forward = v1.TokenTCN.forward


def new_model():
    torch.manual_seed(SEED)
    return FullTCN().cpu()


def source():
    assert not torch.cuda.is_initialized()
    meta = v1.baseline.metadata(); fm, _ = v1.baseline.feature_snapshot()
    old_complete = read(OLD / 'complete.json')
    old_source = read(OLD / 'source_snapshot.json')
    for name, expected in old_source['files_sha256'].items(): assert sha(name) == expected
    for name in ('fit_token_weights.npz', 'epoch_answer_order.npy', 'token_index.json', 'raw_token_features.npy'):
        assert sha(OLD / name) == old_complete['files_sha256'][name]
    hm = read(ROOT / 'data/harp_manifest.json'); hs = read(ROOT / 'data/harp_signature.json')
    bm = read(ROOT / 'data/harp_basis.json'); basis_path = ROOT / 'data/harp_basis.npz'
    assert hm['complete'] and hm['record_count'] == 793 and hm['token_count'] == 213159
    assert not hm['labels_or_test_read'] and not hm['cuda_initialized']
    assert digest(hs) == hm['signature_sha256'] == bm['signature_sha256']
    assert hs['source_feature_manifest_sha256'] == sha(ROOT / 'data/feature_manifest.json')
    assert sha(basis_path) == hm['basis_sha256'] == bm['npz_sha256'] and bm['check']['passed']
    with np.load(basis_path, allow_pickle=False) as z:
        components = z['components'].copy(); eigenvalues = z['eigenvalues'].copy()
    assert components.shape == (256, 4096) and components.dtype == np.float64
    assert eigenvalues.shape == (256,) and np.all(np.diff(eigenvalues) >= 0)
    assert np.max(np.abs(components[:64] @ components[:64].T - np.eye(64))) < 1e-8
    index = read(OLD / 'token_index.json')['answers']
    assert [x['response_id'] for x in index] == [x['response_id'] for x in hm['records']] == [x['response_id'] for x in fm['records']]
    paths = [Path(__file__), ROOT / 'src/run_sequence.py', ROOT / 'src/run_development.py',
             ROOT / 'data/gold_manifest.json', ROOT / 'data/feature_manifest.json', ROOT / 'data/feature_signature.json',
             ROOT / 'data/harp_manifest.json', ROOT / 'data/harp_signature.json', basis_path, ROOT / 'data/harp_basis.json',
             ROOT / 'results/development_v1/hidden_pca.pkl', OLD / 'complete.json', OLD / 'preparation_manifest.json',
             OLD / 'fit_token_weights.npz', OLD / 'epoch_answer_order.npy', OLD / 'token_index.json', OLD / 'raw_token_features.npy']
    snap = {'files_sha256': {str(p.resolve()): sha(p) for p in paths}, 'harp_signature_sha256': hm['signature_sha256'],
            'basis_sha256': hm['basis_sha256'], 'test_opened': False, 'basis_recomputed': False, 'pca_refitted': False}
    return meta, fm, hm, index, components, snap


def preflight():
    meta, fm, hm, index, components, snap = source()
    plans = {p['response_id']: p for p in lines(ROOT / 'data/feature_preparation/plans.jsonl')}
    samples = []
    for entry in (index[0], index[-1]):
        rid = entry['response_id']; arrays = v1.baseline.load_features(rid, meta, fm, plans)
        rec = next(x for x in hm['records'] if x['response_id'] == rid)
        path = ROOT / 'data/harp_features' / (rid + '.npz'); assert sha(path) == rec['npz_sha256']
        with np.load(path, allow_pickle=False) as z:
            harp = z['harp'].copy()
            for name in ('token_ids', 'response_token_offsets', 'answer_token_positions'):
                assert np.array_equal(z[name], arrays[name])
        expected = (arrays['hidden_last'].astype(np.float64) @ components.T).astype(np.float32)
        assert np.array_equal(expected, harp)
        samples.append({'response_id': rid, 'full256_projection_exact': True, 'coordinate_match': True,
                        'bottom64_shape': [entry['token_count'], 64]})
    model = new_model().eval(); second = new_model().eval()
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in model.state_dict().items())
    gen = torch.Generator().manual_seed(SEED + 1); x = torch.randn(2, 16, WIDTH, generator=gen)
    mask = torch.ones(2, 16); mask[0, 9:] = 0
    with torch.no_grad():
        original = model(x, mask); changed = x.clone(); changed[0, 9:] = 1000
        assert torch.allclose(model(changed, mask)[0, :9], original[0, :9], atol=2e-6, rtol=1e-6)
        assert torch.allclose(model(x[:1, :9], torch.ones(1, 9))[0], original[0, :9], atol=2e-6, rtol=1e-6)
        changed = x.clone(); changed[1] = 1000
        assert torch.equal(model(changed, mask)[0], original[0])
        changed = x.clone(); changed[1, :4] += 100
        assert torch.equal(model(changed, mask)[1, 8], original[1, 8])
    result = {'passed': True, 'source_snapshot': snap, 'samples': samples,
              'A_B_initial_parameters_exact': True, 'parameters_each': sum(p.numel() for p in model.parameters()),
              'padding_no_cross_answer_and_RF7_checks_passed': True, 'cuda_initialized': torch.cuda.is_initialized()}
    save(OUT / 'PREFLIGHT.json', result); print('QA_FULL_PREFLIGHT_PASSED', result['parameters_each'], flush=True)


def freeze():
    assert not (OUT / 'protocol.json').exists()
    pre = read(OUT / 'PREFLIGHT.json'); assert pre['passed']
    snap = source()[-1]; assert snap == pre['source_snapshot']
    save(OUT / 'protocol.json', protocol()); save(OUT / 'source_snapshot.json', snap)
    save(OUT / 'freeze.json', {'utc': time.time(), 'protocol_sha256': sha(OUT / 'protocol.json'),
                             'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'), 'test_opened': False})
    print('QA_FULL_PROTOCOL_FROZEN', flush=True)


def prepare(meta, fm, hm, index):
    n = 213159; directory = OUT / 'matrices'; directory.mkdir(parents=True, exist_ok=True)
    common = np.lib.format.open_memmap(directory / 'common1025.npy', mode='w+', dtype=np.float32, shape=(n, 1025))
    harp64 = np.lib.format.open_memmap(directory / 'harp64.npy', mode='w+', dtype=np.float32, shape=(n, 64))
    previous = np.load(OLD / 'raw_token_features.npy', mmap_mode='r')
    plans = {p['response_id']: p for p in lines(ROOT / 'data/feature_preparation/plans.jsonl')}
    records = {r['response_id']: r for r in hm['records']}
    for j, entry in enumerate(index):
        rid = entry['response_id']; a = v1.baseline.load_features(rid, meta, fm, plans); left, right = entry['left'], entry['right']
        common[left:right] = np.column_stack((a['lb'], a['nll']))
        assert np.array_equal(previous[left:right, :128], a['lb'].reshape(-1, 4, 8, 32).mean(2).reshape(-1, 128))
        assert np.array_equal(previous[left:right, 128], a['nll'])
        path = ROOT / 'data/harp_features' / (rid + '.npz'); side = path.with_suffix('.json'); rec = records[rid]
        metadata = read(side)
        assert sha(path) == rec['npz_sha256'] == metadata['npz_sha256'] and sha(side) == rec['metadata_sha256']
        assert metadata['basis_sha256'] == hm['basis_sha256'] and metadata['signature_sha256'] == hm['signature_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert z['harp'].shape == (right - left, 256) and z['harp'].dtype == np.float32
            for key in ('token_ids', 'response_token_offsets', 'answer_token_positions'): assert np.array_equal(z[key], a[key])
            harp64[left:right] = z['harp'][:, :64]
        if (j + 1) % 100 == 0: print('QA_FULL_DESIGN', j + 1, 793, flush=True)
    common.flush(); harp64.flush(); assert np.isfinite(common).all() and np.isfinite(harp64).all()
    with np.load(OLD / 'fit_token_weights.npz', allow_pickle=False) as z: weights = {key: z[key].copy() for key in z.files}
    order = np.load(OLD / 'epoch_answer_order.npy'); assert order.shape == (30, 634)
    designs = {}; scalers = {}
    for method in METHODS:
        auxiliary = previous[:, 129:193] if method == METHODS[0] else harp64
        scaler = StandardScaler()
        for left in range(0, 170361, 16384):
            right = min(left + 16384, 170361)
            raw = np.column_stack((common[left:right], auxiliary[left:right]))
            scaler.partial_fit(raw, sample_weight=weights['base'][left:right])
        (OUT / (method + '_scaler.pkl')).write_bytes(pickle.dumps(scaler, protocol=5)); scalers[method] = scaler
        dest = np.lib.format.open_memmap(directory / (method + '.npy'), mode='w+', dtype=np.float32, shape=(n, WIDTH))
        for left in range(0, n, 16384):
            right = min(left + 16384, n); raw = np.column_stack((common[left:right], auxiliary[left:right]))
            dest[left:right] = scaler.transform(raw).astype(np.float32)
        dest.flush(); assert np.isfinite(dest).all(); designs[method] = dest
    for key in ('mean_', 'var_', 'scale_'):
        assert np.array_equal(getattr(scalers[METHODS[0]], key)[:1025], getattr(scalers[METHODS[1]], key)[:1025])
    assert np.array_equal(designs[METHODS[0]][:, :1025], designs[METHODS[1]][:, :1025])
    paths = [directory / 'common1025.npy', directory / 'harp64.npy']
    paths.extend(directory / (m + '.npy') for m in METHODS); paths.extend(OUT / (m + '_scaler.pkl') for m in METHODS)
    save(OUT / 'preparation_manifest.json', {'complete': True, 'input_shape_each': [n, WIDTH],
        'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'),
        'files_sha256': {str(p.relative_to(OUT)): sha(p) for p in paths},
        'same_standardized_common1025_exact': True, 'weights_sha256': sha(OLD / 'fit_token_weights.npz'),
        'shuffle_sha256': sha(OLD / 'epoch_answer_order.npy'), 'test_opened': False, 'gpu_used': False})
    return designs, weights, order


def batch(chosen, x, index, weights=None):
    length = max(index[int(i)]['token_count'] for i in chosen)
    xb = torch.zeros(len(chosen), length, WIDTH); mask = torch.zeros(len(chosen), length)
    yb = torch.zeros(len(chosen), length); wb = torch.zeros(len(chosen), length)
    for j, i in enumerate(chosen):
        entry = index[int(i)]; left, right = entry['left'], entry['right']; n = right - left
        xb[j, :n] = torch.from_numpy(np.array(x[left:right], copy=True)); mask[j, :n] = 1
        if weights is not None:
            assert entry['partition'] == 'fit' and right <= 170361
            yb[j, :n] = torch.from_numpy(weights['y'][left:right].astype(np.float32))
            wb[j, :n] = torch.from_numpy(weights['loss'][left:right].astype(np.float32))
    return xb, mask, yb, wb


@torch.no_grad()
def predict(model, x, index):
    model.eval(); logits = np.empty(len(x), np.float32)
    for left in range(0, len(index), BATCH):
        chosen = list(range(left, min(left + BATCH, len(index)))); xx, mask, _, _ = batch(chosen, x, index)
        values = model(xx, mask).numpy()
        for j, i in enumerate(chosen):
            entry = index[i]; logits[entry['left']:entry['right']] = values[j, :entry['token_count']]
    assert np.isfinite(logits).all()
    return logits, torch.sigmoid(torch.from_numpy(logits)).numpy()


def train():
    assert read(OUT / 'protocol.json') == protocol() and not (OUT / 'started.json').exists()
    meta, fm, hm, index, _, snap = source(); assert snap == read(OUT / 'source_snapshot.json')
    save(OUT / 'started.json', {'utc': time.time(), 'freeze_sha256': sha(OUT / 'freeze.json'), 'test_opened': False})
    started = time.perf_counter(); designs, weights, order = prepare(meta, fm, hm, index)
    preparation_seconds = time.perf_counter() - started; selected = {}; histories = {}; files = []
    for method in METHODS:
        model = new_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        directory = OUT / method; directory.mkdir(parents=True, exist_ok=True); history = []; best = None
        for epoch in range(EPOCHS):
            begin = time.perf_counter(); model.train()
            for left in range(0, 634, BATCH):
                chosen = order[epoch, left:left + BATCH]; xx, mask, yy, ww = batch(chosen, designs[method], index, weights)
                optimizer.zero_grad(set_to_none=True)
                objective = (F.binary_cross_entropy_with_logits(model(xx, mask), yy, reduction='none') * ww).sum() * (634 / (len(chosen) * 139518))
                assert torch.isfinite(objective); objective.backward(); optimizer.step()
            fitting = time.perf_counter() - begin; logits, probs = predict(model, designs[method], index)
            arrays, thresholds, metrics, fit_loss = v1.evaluate(meta, index, logits, probs, weights)
            arrays.update(token_scores=probs, token_logits=logits)
            name = f'epoch_{epoch + 1:03d}'; npz = directory / (name + '_scores.npz'); np.savez_compressed(npz, **arrays)
            state = directory / (name + '.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'method': method, 'epoch': epoch + 1, 'seed': SEED, 'device': 'cpu',
                        'protocol_sha256': sha(OUT / 'protocol.json'), 'preparation_manifest_sha256': sha(OUT / 'preparation_manifest.json'),
                        'torch_rng_state': torch.get_rng_state()}, state)
            w, a = thresholds['window'], thresholds['answer']; key = [min(w['f1'], a['f1']), w['f1'], w['precision'], -(epoch + 1)]
            entry = {'method': method, 'epoch': epoch + 1, 'parameters': sum(p.numel() for p in model.parameters()),
                     'full_fit_weighted_BCE': fit_loss, 'thresholds': thresholds, 'selection_key': key, 'metrics': metrics,
                     'epoch_training_seconds': fitting, 'epoch_total_seconds': time.perf_counter() - begin,
                     'model_sha256': sha(state), 'scores_sha256': sha(npz), 'test_opened': False}
            report = directory / (name + '.json'); save(report, entry); history.append(entry)
            files.extend(str(p.relative_to(OUT)) for p in (npz, state, report))
            if best is None or key > best: best = key; selected[method] = entry
            save(OUT / 'progress.json', {'method': method, 'epoch': epoch + 1, 'selected': selected, 'elapsed_seconds': time.perf_counter() - started})
            print('QA_FULL_EPOCH', method, epoch + 1, 'bce', round(fit_loss, 6), 'cal_w', round(w['f1'], 6),
                  'cal_a', round(a['f1'], 6), 'seconds', round(entry['epoch_total_seconds'], 2), flush=True)
        histories[method] = history; del model, optimizer; gc.collect()
    assert snap == source()[-1] and not torch.cuda.is_initialized()
    result = {'scope': protocol()['scope'], 'selected': selected, 'all_epochs': histories,
              'preparation_seconds': preparation_seconds, 'wall_seconds': time.perf_counter() - started,
              'models': 2, 'epochs': 60, 'seed': SEED, 'input_dimensions': WIDTH,
              'calibration_results_selection_optimistic': True, 'test_opened': False, 'gpu_used': False,
              'limitations': protocol()['limitations']}
    save(OUT / 'summary.json', result); files.extend(['summary.json', 'protocol.json', 'source_snapshot.json', 'freeze.json', 'PREFLIGHT.json', 'preparation_manifest.json'])
    save(OUT / 'complete.json', {'status': 'complete_development_only', 'files_sha256': {name: sha(OUT / name) for name in files},
                                'script_sha256': sha(Path(__file__)), 'test_opened': False, 'gpu_used': False})
    print('QA_FULL_COMPLETE', round(result['wall_seconds'], 2), flush=True)


def diagnostics():
    meta = v1.baseline.metadata(); new = read(OUT / 'summary.json'); previous = read(OLD / 'summary.json')
    selected = {}; score_arrays = {}; answer_arrays = {}; directories = {}
    for folder, result in ((OLD, previous), (OUT, new)):
        for method, entry in result['selected'].items():
            selected[method] = entry; directories[method] = folder
            path = folder / method / f"epoch_{entry['epoch']:03d}_scores.npz"; assert sha(path) == entry['scores_sha256']
            with np.load(path, allow_pickle=False) as z:
                score_arrays[method] = z['window_scores'].copy(); answer_arrays[method] = z['answer_scores'].copy()
    rows = []; typed = {}; span_status = {}
    for method, entry in selected.items():
        score = score_arrays[method]; threshold = entry['thresholds']['window']['threshold']; alert = score >= threshold
        cover = defaultdict(set)
        for w, flagged in zip(meta['windows'], alert):
            if flagged: cover[w['response_id']].update(w['token_indices'])
        details = {}; statuses = {}
        for part in ('fit', 'calibration'):
            details[part] = {}
            for kind in protocol()['diagnostics']['types']:
                spans = []; risk_by_answer = defaultdict(set); unlocalizable = 0
                for tokens in meta['tokens']:
                    if tokens['partition'] != part: continue
                    for mapping in tokens['span_token_mapping']:
                        label = tokens['original_labels'][mapping['span_index']]
                        if label['label_type'] != kind: continue
                        ix = set(mapping['risk_token_indices']); rid = tokens['response_id']
                        if not ix: unlocalizable += 1; continue
                        risk_by_answer[rid].update(ix); any_hit = bool(ix & cover[rid]); full_hit = ix <= cover[rid]
                        spans.append((rid, mapping['span_index'], any_hit, full_hit))
                        if part == 'calibration' and 'Conflict' in kind:
                            statuses[(rid, mapping['span_index'])] = {'response_id': rid, 'span_index': mapping['span_index'],
                                'label_type': kind, 'text': label['text'], 'any_hit': any_hit, 'full_hit': full_hit}
                positive = [i for i, w in enumerate(meta['windows']) if w['partition'] == part and
                            bool(set(w['token_indices']) & risk_by_answer[w['response_id']])]
                details[part][kind] = {'original_spans': len(spans) + unlocalizable, 'localizable_spans': len(spans),
                    'unlocalizable_spans': unlocalizable, 'spans_any_hit': sum(s[2] for s in spans),
                    'spans_fully_hit': sum(s[3] for s in spans), 'span_any_hit_recall': sum(s[2] for s in spans) / len(spans) if spans else None,
                    'span_full_hit_recall': sum(s[3] for s in spans) / len(spans) if spans else None,
                    'positive_windows': len(positive), 'hit_positive_windows': int(alert[positive].sum()),
                    'positive_window_recall': float(alert[positive].mean()) if positive else None}
        typed[method] = details; span_status[method] = statuses
        rows.append({'method': method, 'epoch': entry['epoch'], 'parameters': entry['parameters'], 'metrics': entry['metrics'],
                     'fit_minus_cal_window_F1': entry['metrics']['fit']['windows']['f1'] - entry['metrics']['calibration']['windows']['f1']})
    a, b = (span_status[m] for m in METHODS)
    assert set(a) == set(b)
    gained = [b[k] for k in sorted(a) if not a[k]['any_hit'] and b[k]['any_hit']]
    lost = [b[k] for k in sorted(a) if a[k]['any_hit'] and not b[k]['any_hit']]
    result = {'same_geometry_and_labels': True, 'global_thresholds_no_type_tuning': True, 'rows': rows,
              'by_error_type': typed, 'conflict_B_vs_A': {'newly_any_hit_count': len(gained), 'lost_any_hit_count': len(lost),
                 'first3_gained': gained[:3], 'first3_lost': lost[:3]}, 'test_opened': False,
              'span_coverage_means_union_of_alerted_4raw_windows': True,
              'positive_windows_per_type_can_overlap_other_types': True,
              'no_claim_of_test_performance': True}
    save(OUT / 'DIAGNOSTICS.json', result)
    comparison = [{'candidate': row['method'], 'kind': 'sequence', 'metrics': row['metrics']} for row in rows]
    baseline_hashes = {}
    for folder in ('development_v1', 'harp_development_v1'):
        directory = ROOT / 'results' / folder; baseline_complete = read(directory / 'complete.json')
        assert sha(directory / 'summary.json') == baseline_complete['files_sha256']['summary.json']
        baseline_hashes[folder] = sha(directory / 'complete.json')
        for family in read(directory / 'summary.json')['all_candidates'].values():
            for candidate in family:
                comparison.append({'candidate': candidate['candidate'], 'kind': 'frozen_LR', 'metrics': candidate['metrics']})
    for name, value in (('all_risk', 1.), ('no_risk', 0.)):
        scores = np.full(len(meta['windows']), value)
        comparison.append({'candidate': name, 'kind': 'constant', 'metrics': v1.baseline.metrics(meta, scores,
            {'window': {'threshold': .5}, 'answer': {'threshold': .5}})})
    for entry in comparison:
        for part, nw, na in (('fit', 168123, 634), ('calibration', 42241, 159)):
            assert entry['metrics'][part]['windows']['n'] == nw and entry['metrics'][part]['answers']['n'] == na
    save(OUT / 'BASELINE_COMPARISON.json', {'entries': comparison, 'same_unchanged_gold_and_denominators': True,
        'baseline_complete_hashes': baseline_hashes, 'sequence_complete_sha256': sha(OUT / 'complete.json'), 'test_opened': False})
    report = ['两项新TCN均为1089维输入、94529参数、RF7、同seed/权重/30轮；下列是校准选轮与选阈值后的开发成绩。', '',
        '| 方法 | 轮次 | fit窗F1 | cal窗F1 | fit整答F1 | cal整答F1 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']; report.append(f"| {row['method']} | {row['epoch']} | {m['fit']['windows']['f1']:.3f} | {m['calibration']['windows']['f1']:.3f} | {m['fit']['answers']['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    report += ['', '冲突类使用同一个全局窗口阈值；原span被已报警4BPE窗覆盖任一风险token算至少命中，覆盖全部风险token算完整命中。', '',
               '| 方法 | 冲突类 | 阳性窗召回 | 原span至少命中 | 原span完整命中 |', '|---|---|---:|---:|---:|']
    for method, parts in typed.items():
        for kind in ('Evident Conflict', 'Subtle Conflict'):
            d = parts['calibration'][kind]
            report.append(f"| {method} | {kind} | {d['hit_positive_windows']}/{d['positive_windows']} | {d['spans_any_hit']}/{d['localizable_spans']} | {d['spans_fully_hit']}/{d['localizable_spans']} |")
    report += ['', '校准集Subtle Conflict只有5个原span，细分类结果样本很少。原标签与4BPE几何未改，未按错误类型另调阈值。',
               'A/B输入维数与参数数相同；与旧193维TCN比较时投影参数也增加，不能把差异全部归因为保留信息。PCA64/HARP64都不是完整隐藏状态。',
               '旧MLP和193维TCN原结果均保留，未重训；测试未打开。全部60轮checkpoint与预测已保存。']
    for unit, label in (('windows', '窗口'), ('answers', '整答')):
        best = max((entry for entry in comparison if entry['kind'] == 'frozen_LR'), key=lambda entry: entry['metrics']['calibration'][unit]['f1'])
        report.append(f"现有LR候选中校准{label}最高：{best['candidate']}，F1={best['metrics']['calibration'][unit]['f1']:.3f}。全部候选及常数基线见BASELINE_COMPARISON.json。")
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', 'utf-8')
    print('QA_FULL_DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['preflight', 'freeze', 'train', 'diagnostics']); args = parser.parse_args()
    torch.set_num_threads(THREADS); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=THREADS):
        {'preflight': preflight, 'freeze': freeze, 'train': train, 'diagnostics': diagnostics}[args.stage]()
