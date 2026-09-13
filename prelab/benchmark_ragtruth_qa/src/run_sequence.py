"""CPU-only fixed token MLP / offline seven-token TCN development experiment."""
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

import run_development as baseline

ROOT = baseline.ROOT
OUT = ROOT / 'results/sequence_v1'
PCA_PATH = ROOT / 'results/development_v1/hidden_pca.pkl'
SEED, THREADS, EPOCHS, ANSWER_BATCH = 20260926, 4, 30, 8
METHODS = ('token_mlp', 'token_tcn')
sha, read, lines, save, digest = baseline.sha, baseline.read, baseline.lines, baseline.save, baseline.digest


def protocol():
    return {'version': 'qa-token-sequence-v1', 'scope': 'Only frozen fit634/calibration159, never official test',
        'methods': list(METHODS), 'seed': SEED, 'epochs_per_model': EPOCHS, 'batch_answers': ANSWER_BATCH,
        'cpu_threads': THREADS, 'input_dimensions': 193,
        'input': 'Each raw token: four contiguous 8-layer bands retaining32 heads =128 LB; NLL1; reused fit-only hidden PCA64',
        'pca': 'Reuse development_v1/hidden_pca.pkl exactly; no additional PCA fit or whitening',
        'scaler': 'One common fit-only StandardScaler, lexical label-independent base weights, partial_fit fixed16384 blocks; transform float32',
        'token_geometry': 'All raw answer tokens remain in order, including punctuation; no cross-answer propagation or truncation',
        'labels': 'Frozen human lexical risk_mask. Fit139518 lexical tokens; punctuation/context tokens loss weight0',
        'window_score': 'Max probability only over output-defined lexical positions inside each unchanged eligible4-raw-BPE window',
        'answer_score': 'Max over every eligible window, official original-span answer label unchanged',
        'base_weight': 'Source-connected group equal -> answer equal -> lexical token equal; mean1 on139518 fit lexical tokens',
        'loss_weight': 'Fit-only weighted binary class factors, re-equalize group loss, total139518; both models identical',
        'minibatch_objective': 'sum(loss_weight * BCE_logits) * 634 / (actual_batch_answers * 139518); padding/punctuation weight0',
        'shuffle': 'Fixed numpy default_rng(seed) answer permutations for every epoch, shared by both models',
        'mlp': '193->64 GELU dropout.2 ->64 GELU dropout.2 ->1, pointwise token head',
        'tcn': '193->64 GELU; two residual blocks each ONE Conv1d64->64 kernel3 dilation1/2, GELU/dropout.2; ->1',
        'tcn_padding': 'Symmetric zero padding; zero padded states after projection and every residual block',
        'receptive_field_raw_tokens': 7, 'future_tokens_used_by_tcn': True, 'online_detector': False,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01, 'all_parameters_including_bias': True},
        'selection': 'Cal max min(windowF1,answerF1), then windowF1, window precision, earlier epoch; separate cal thresholds',
        'threshold': 'Same fixed baseline F1, then precision, then higher cutoff; every eligible cal row retained',
        'early_stopping': False, 'additional_seed_selection': False, 'test_opened': False,
        'save': 'Every epoch model+optimizer state, full fit/cal raw-token/window/answer probabilities, thresholds and metrics',
        'comparisons': 'All12 frozen LR candidates and all-risk/no-risk baselines on same fit/cal labels and geometry',
        'interpretation': ['Cal selects checkpoints and thresholds, so its reported performance is selection optimistic.',
                           'TCN has more parameters than MLP; this comparison does not isolate context width alone.',
                           'Fixed PCA64 and layer averaging can discard useful directions; failure cannot establish absent internal information.']}


class TokenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(193, 64), nn.GELU(), nn.Dropout(.2),
                                    nn.Linear(64, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, x, valid):
        return self.layers(x).squeeze(-1) * valid


class TokenTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(193, 64)
        self.convolutions = nn.ModuleList([nn.Conv1d(64, 64, 3, padding=d, dilation=d) for d in (1, 2)])
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(64, 1)

    def forward(self, x, valid):
        mask = valid[:, None, :]
        h = F.gelu(self.projection(x)).transpose(1, 2) * mask
        for conv in self.convolutions:
            h = (h + self.dropout(F.gelu(conv(h)))) * mask
        return self.output(h.transpose(1, 2)).squeeze(-1) * valid


def new_model(method):
    torch.manual_seed(SEED)
    return (TokenMLP() if method == 'token_mlp' else TokenTCN()).cpu()


def snapshot():
    meta = baseline.metadata(); feature_manifest, _ = baseline.feature_snapshot()
    matrix = read(ROOT / 'results/development_v1/matrix_manifest.json')
    assert sha(PCA_PATH) == matrix['pca_sha256']
    pca = pickle.loads(PCA_PATH.read_bytes())
    assert pca['fit_answers'] == 634 and pca['fit_groups'] == 615 and not pca['whiten']
    assert pca['components'].shape == (64, 4096) and pca['mean'].shape == (4096,)
    fit_ids = {a['response_id'] for a in meta['answers'] if a['partition'] == 'fit'}
    assert {s['response_id'] for s in pca['sample']} == fit_ids
    paths = [Path(__file__), ROOT / 'src/run_development.py', PCA_PATH,
             ROOT / 'data/gold_manifest.json', ROOT / 'data/feature_manifest.json',
             ROOT / 'data/feature_signature.json', ROOT / 'data/feature_preparation/plans.jsonl',
             ROOT / 'results/development_v1/matrix_manifest.json']
    source = {'files_sha256': {str(path.resolve()): sha(path) for path in paths},
              'pca_sha256': sha(PCA_PATH), 'feature_signature_sha256': feature_manifest['signature_sha256'],
              'test_content_read': False, 'pca_refitted': False}
    return meta, feature_manifest, pca, source


def cpu_checks_and_benchmark():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(THREADS)
    reports = []
    for method in METHODS:
        model = new_model(method).eval()
        generator = torch.Generator().manual_seed(SEED + 1)
        x = torch.randn(2, 16, 193, generator=generator)
        valid = torch.ones(2, 16); valid[0, 9:] = 0
        with torch.no_grad():
            original = model(x, valid)
            changed = x.clone(); changed[0, 9:] = 1000
            assert torch.allclose(model(changed, valid)[0, :9], original[0, :9], atol=2e-6, rtol=1e-6)
            single = model(x[:1, :9], torch.ones(1, 9))
            assert torch.allclose(single[0], original[0, :9], atol=2e-6, rtol=1e-6)
            changed = x.clone(); changed[1] = 1000
            assert torch.equal(model(changed, valid)[0], original[0])
            changed = x.clone(); changed[1, :4] += 100
            unchanged_at = 8
            assert torch.equal(model(changed, valid)[1, unchanged_at], original[1, unchanged_at])
        # Synthetic data only: no official labels, features, or warm start.
        model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        x = torch.randn(8, 448, 193, generator=generator); mask = torch.ones(8, 448)
        y = torch.randint(0, 2, (8, 448), generator=generator).float()
        optimizer.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(model(x, mask), y).backward(); optimizer.step()
        times = []
        for _ in range(8):
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x, mask), y).backward(); optimizer.step()
            times.append(time.perf_counter() - started)
        model.eval(); started = time.perf_counter()
        with torch.no_grad():
            for _ in range(8): model(x, mask)
        forward = (time.perf_counter() - started) / 8
        epoch = float(np.median(times)) * 80 + forward * 100
        reports.append({'method': method, 'parameters': sum(p.numel() for p in model.parameters()),
                        'median_train_batch_seconds': float(np.median(times)), 'forward_batch_seconds': forward,
                        'estimated_30_epoch_model_seconds': epoch * EPOCHS,
                        'padding_and_cross_answer_checks_passed': True, 'far_token_no_influence_passed': True})
    result = {'passed': True, 'synthetic_only': True, 'benchmark_batch': [8, 448, 193],
              'threads': THREADS, 'models': reports, 'gpu_initialized': torch.cuda.is_initialized(),
              'estimated_total_60_epoch_model_minutes': sum(r['estimated_30_epoch_model_seconds'] for r in reports) / 60,
              'exclusions': ['one-time raw feature/PCA projection and scaler', 'all-epoch disk writes and metrics', 'CPU contention changes'],
              'formal_training_warm_started_from_benchmark': False}
    save(OUT / 'CPU_PREFLIGHT.json', result); print(json.dumps(result), flush=True)


def freeze():
    assert not (OUT / 'protocol.json').exists() and read(OUT / 'CPU_PREFLIGHT.json')['passed']
    meta, _, _, source = snapshot()
    assert sum(sum(t['lexical_mask']) for t in meta['tokens'] if t['partition'] == 'fit') == 139518
    save(OUT / 'protocol.json', protocol()); save(OUT / 'source_snapshot.json', source)
    save(OUT / 'freeze.json', {'utc': time.time(), 'protocol_sha256': sha(OUT / 'protocol.json'),
                             'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'),
                             'preflight_sha256': sha(OUT / 'CPU_PREFLIGHT.json'), 'test_opened': False})
    print('QA_SEQUENCE_FROZEN', flush=True)


def make_weights(meta, index):
    fit_count = sum(t['token_count'] for t in meta['tokens'] if t['partition'] == 'fit')
    lexical = np.concatenate([np.asarray(t['lexical_mask'], bool) for t in meta['tokens'] if t['partition'] == 'fit'])
    y = np.concatenate([np.asarray(t['risk_mask'], np.int64) for t in meta['tokens'] if t['partition'] == 'fit'])
    assert fit_count == 170361 and int(lexical.sum()) == 139518
    tree = defaultdict(list)
    for entry in index:
        if entry['partition'] == 'fit': tree[entry['group_id']].append(entry)
    b = np.zeros(fit_count, np.float64)
    for group_answers in tree.values():
        for answer in group_answers:
            ix = np.arange(answer['left'], answer['right']); selected = ix[lexical[ix]]
            assert len(selected) > 0
            b[selected] = 1 / (len(group_answers) * len(selected))
    b *= 139518 / b.sum()
    mass = np.bincount(y, weights=b, minlength=2); factors = mass.sum() / (2 * mass)
    loss = b * factors[y]
    for group_answers in tree.values():
        ix = np.concatenate([np.arange(a['left'], a['right']) for a in group_answers])
        loss[ix] *= (139518 / len(tree)) / loss[ix].sum()
    loss *= 139518 / loss.sum()
    assert np.array_equal(loss > 0, lexical) and not np.any(y[~lexical])
    group_mass = [sum(loss[a['left']:a['right']].sum() for a in aa) for aa in tree.values()]
    assert np.allclose(group_mass, 139518 / 615, atol=1e-10)
    return {'base': b, 'loss': loss, 'class_factors': factors, 'y': y, 'lexical': lexical}


def prepare(meta, manifest, pca):
    source = read(OUT / 'source_snapshot.json')
    plans = {p['response_id']: p for p in lines(ROOT / 'data/feature_preparation/plans.jsonl')}
    index = []; cursor = 0
    for t in meta['tokens']:
        index.append({key: t[key] for key in ('response_id', 'answer_id', 'source_id', 'group_id', 'partition')} |
                     {'left': cursor, 'right': cursor + t['token_count'], 'token_count': t['token_count']})
        cursor += t['token_count']
    assert cursor == 213159
    raw = np.lib.format.open_memmap(OUT / 'raw_token_features.npy', mode='w+', dtype=np.float32, shape=(cursor, 193))
    for j, entry in enumerate(index):
        a = baseline.load_features(entry['response_id'], meta, manifest, plans)
        band = a['lb'].reshape(-1, 4, 8, 32).mean(2).reshape(-1, 128)
        hidden = ((a['hidden_last'].astype(np.float64) - pca['mean']) @ pca['components'].T).astype(np.float32)
        raw[entry['left']:entry['right']] = np.column_stack((band, a['nll'], hidden))
        if (j + 1) % 100 == 0: print('QA_SEQUENCE_DESIGN', j + 1, 793, flush=True)
    raw.flush(); assert np.isfinite(raw).all()
    weights = make_weights(meta, index); np.savez_compressed(OUT / 'fit_token_weights.npz', **weights)
    save(OUT / 'token_index.json', {'answers': index, 'all_raw_tokens': cursor,
                                  'fit_raw_tokens': 170361, 'fit_lexical_tokens': 139518})
    scaler = StandardScaler()
    for left in range(0, 170361, 16384):
        right = min(left + 16384, 170361)
        scaler.partial_fit(raw[left:right], sample_weight=weights['base'][left:right])
    (OUT / 'scaler.pkl').write_bytes(pickle.dumps(scaler, protocol=5))
    normalized = np.lib.format.open_memmap(OUT / 'standardized_token_features.npy', mode='w+', dtype=np.float32, shape=raw.shape)
    for left in range(0, cursor, 16384):
        right = min(left + 16384, cursor)
        normalized[left:right] = scaler.transform(raw[left:right]).astype(np.float32)
    normalized.flush(); assert np.isfinite(normalized).all()
    permutations = np.stack([rng.permutation(634) for rng in [np.random.default_rng(SEED)] for _ in range(EPOCHS)])
    np.save(OUT / 'epoch_answer_order.npy', permutations, allow_pickle=False)
    paths = ('raw_token_features.npy', 'standardized_token_features.npy', 'fit_token_weights.npz',
             'token_index.json', 'scaler.pkl', 'epoch_answer_order.npy')
    save(OUT / 'preparation_manifest.json', {'complete': True, 'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'),
        'files_sha256': {name: sha(OUT / name) for name in paths}, 'pca_reused_sha256': source['pca_sha256'],
        'input_shape': list(raw.shape), 'same_design_and_scaler_for_both_models': True,
        'loss_mass': float(weights['loss'].sum()), 'fit_lexical_tokens': 139518, 'gpu_used': False})
    return normalized, index, weights, permutations


def batch_data(chosen, x, index, weights=None):
    length = max(index[i]['token_count'] for i in chosen)
    xb = torch.zeros(len(chosen), length, 193); valid = torch.zeros(len(chosen), length)
    yb = torch.zeros(len(chosen), length); wb = torch.zeros(len(chosen), length)
    for position, i in enumerate(chosen):
        entry = index[int(i)]; left, right = entry['left'], entry['right']; n = right - left
        xb[position, :n] = torch.from_numpy(np.array(x[left:right], copy=True)); valid[position, :n] = 1
        if weights is not None:
            assert entry['partition'] == 'fit' and right <= 170361
            yb[position, :n] = torch.from_numpy(weights['y'][left:right].astype(np.float32))
            wb[position, :n] = torch.from_numpy(weights['loss'][left:right].astype(np.float32))
    return xb, valid, yb, wb


@torch.no_grad()
def predict_tokens(model, x, index):
    model.eval(); logits = np.empty(len(x), np.float32)
    for left in range(0, len(index), ANSWER_BATCH):
        chosen = list(range(left, min(left + ANSWER_BATCH, len(index))))
        xb, valid, _, _ = batch_data(chosen, x, index)
        out = model(xb, valid).numpy()
        for j, i in enumerate(chosen):
            entry = index[i]; logits[entry['left']:entry['right']] = out[j, :entry['token_count']]
    assert np.isfinite(logits).all()
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return logits, probabilities


def aggregate(meta, index, probabilities):
    starts = {a['response_id']: a['left'] for a in index}
    values = np.empty(len(meta['windows']), np.float64)
    for j, window in enumerate(meta['windows']):
        rid = window['response_id']; tokens = meta['by_response'][rid]['tokens']
        positions = [k for k in window['token_indices'] if tokens['lexical_mask'][k]]
        assert positions  # Existing eligible raw window; no new filtering.
        values[j] = probabilities[np.asarray(positions) + starts[rid]].max()
    return values, baseline.answer_scores(meta, values)


def evaluate(meta, index, logits, probabilities, weights):
    window, answer = aggregate(meta, index, probabilities)
    lo, hi = meta['bounds']['calibration']; ci = [i for i, a in enumerate(meta['answers']) if a['partition'] == 'calibration']
    thresholds = {'window': baseline.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], window[lo:hi]),
                  'answer': baseline.choose_threshold([meta['answers'][i]['label'] for i in ci], answer[ci])}
    fit_logits = logits[:170361].astype(np.float64)
    bce = np.logaddexp(0, fit_logits) - weights['y'] * fit_logits
    fit_loss = float(weights['loss'] @ bce / 139518)
    metrics = baseline.metrics(meta, window, thresholds)
    return {'window_scores': window, 'answer_scores': answer}, thresholds, metrics, fit_loss


def train():
    assert not torch.cuda.is_initialized(); torch.set_num_threads(THREADS)
    cfg = read(OUT / 'protocol.json'); assert cfg == protocol()
    assert not (OUT / 'started.json').exists(), 'No silent restart of a formal run'
    meta, manifest, pca, source = snapshot(); assert source == read(OUT / 'source_snapshot.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'freeze.json')['protocol_sha256']
    save(OUT / 'started.json', {'utc': time.time(), 'freeze_sha256': sha(OUT / 'freeze.json'), 'test_opened': False})
    started = time.perf_counter(); x, index, weights, permutations = prepare(meta, manifest, pca)
    preparation_seconds = time.perf_counter() - started
    selected = {}; histories = {}; files = []
    for method in METHODS:
        model = new_model(method); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        directory = OUT / method; directory.mkdir(parents=True, exist_ok=True)
        history = []; best_key = None
        for epoch in range(EPOCHS):
            epoch_started = time.perf_counter(); model.train(); order = permutations[epoch]
            for left in range(0, len(order), ANSWER_BATCH):
                chosen = order[left:left + ANSWER_BATCH]
                xb, valid, yb, wb = batch_data(chosen, x, index, weights)
                optimizer.zero_grad(set_to_none=True)
                losses = F.binary_cross_entropy_with_logits(model(xb, valid), yb, reduction='none')
                objective = (losses * wb).sum() * (634 / (len(chosen) * 139518))
                assert torch.isfinite(objective)
                objective.backward(); optimizer.step()
            fit_seconds = time.perf_counter() - epoch_started
            logits, probabilities = predict_tokens(model, x, index)
            arrays, thresholds, metrics, fit_loss = evaluate(meta, index, logits, probabilities, weights)
            arrays.update(token_scores=probabilities, token_logits=logits)
            filename = f'epoch_{epoch + 1:03d}'
            scorepath = directory / (filename + '_scores.npz'); np.savez_compressed(scorepath, **arrays)
            statepath = directory / (filename + '.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1, 'method': method, 'seed': SEED,
                        'protocol_sha256': sha(OUT / 'protocol.json'), 'preparation_manifest_sha256': sha(OUT / 'preparation_manifest.json'),
                        'torch_rng_state': torch.get_rng_state(), 'device': 'cpu'}, statepath)
            w, a = thresholds['window'], thresholds['answer']
            key = [min(w['f1'], a['f1']), w['f1'], w['precision'], -(epoch + 1)]
            entry = {'method': method, 'epoch': epoch + 1, 'parameters': sum(p.numel() for p in model.parameters()),
                     'full_fit_weighted_BCE': fit_loss, 'thresholds': thresholds, 'selection_key': key,
                     'metrics': metrics, 'epoch_training_seconds': fit_seconds,
                     'epoch_total_seconds': time.perf_counter() - epoch_started,
                     'model_sha256': sha(statepath), 'scores_sha256': sha(scorepath),
                     'test_opened': False, 'calibration_selected': True}
            reportpath = directory / (filename + '.json'); save(reportpath, entry)
            history.append(entry); files.extend([str(p.relative_to(OUT)) for p in (scorepath, statepath, reportpath)])
            if best_key is None or key > best_key:
                best_key = key; selected[method] = entry
            save(OUT / 'progress.json', {'method': method, 'completed_epochs': epoch + 1, 'selected': selected,
                                        'elapsed_seconds': time.perf_counter() - started, 'test_opened': False})
            print('QA_SEQUENCE_EPOCH', method, epoch + 1, 'bce', round(fit_loss, 6), 'cal_w', round(w['f1'], 6),
                  'cal_a', round(a['f1'], 6), 'seconds', round(entry['epoch_total_seconds'], 2), flush=True)
        histories[method] = history; del model, optimizer; gc.collect()
    assert source == snapshot()[3] and not torch.cuda.is_initialized()
    result = {'scope': cfg['scope'], 'selected': selected, 'all_epochs': histories,
              'preparation_seconds': preparation_seconds, 'wall_seconds': time.perf_counter() - started,
              'models': 2, 'epochs': 60, 'seed': SEED, 'input_features': 193,
              'fit_answers': 634, 'fit_lexical_tokens': 139518, 'calibration_answers': 159,
              'fit_windows': 168123, 'calibration_windows': 42241,
              'calibration_results_selection_optimistic': True, 'test_opened': False,
              'future_tokens_used_by_tcn': True, 'interpretation': cfg['interpretation']}
    save(OUT / 'summary.json', result); files.append('summary.json')
    files.extend(['protocol.json', 'source_snapshot.json', 'freeze.json', 'CPU_PREFLIGHT.json', 'preparation_manifest.json',
                  'raw_token_features.npy', 'standardized_token_features.npy', 'fit_token_weights.npz',
                  'token_index.json', 'scaler.pkl', 'epoch_answer_order.npy'])
    save(OUT / 'complete.json', {'status': 'complete_development_only', 'files_sha256': {name: sha(OUT / name) for name in files},
                                'script_sha256': sha(Path(__file__)), 'test_opened': False, 'gpu_used': False})
    print('QA_SEQUENCE_COMPLETE', round(result['wall_seconds'], 2), flush=True)


def compare():
    result = read(OUT / 'summary.json'); base = read(ROOT / 'results/development_v1/summary.json')
    completion = read(ROOT / 'results/development_v1/complete.json')
    assert completion['official_test_opened'] is False
    assert sha(ROOT / 'results/development_v1/summary.json') == completion['files_sha256']['summary.json']
    meta = baseline.metadata(); entries = []
    for family, candidates in base['all_candidates'].items():
        for candidate in candidates:
            entries.append({'candidate': candidate['candidate'], 'kind': 'frozen_LR', 'metrics': candidate['metrics']})
    for method, selected in result['selected'].items():
        entries.append({'candidate': method, 'kind': 'new_sequence', 'selected_epoch': selected['epoch'], 'metrics': selected['metrics']})
    for name, score in (('all_risk', 1.), ('no_risk', 0.)):
        values = np.full(len(meta['windows']), score, np.float64)
        entries.append({'candidate': name, 'kind': 'constant', 'metrics': baseline.metrics(meta, values,
                        {'window': {'threshold': .5}, 'answer': {'threshold': .5}})})
    for entry in entries:
        for part, nw, na in (('fit', 168123, 634), ('calibration', 42241, 159)):
            assert entry['metrics'][part]['windows']['n'] == nw and entry['metrics'][part]['answers']['n'] == na
    comparison = {'same_unchanged_gold_and_denominators': True, 'entries': entries,
                  'baseline_complete_sha256': sha(ROOT / 'results/development_v1/complete.json'),
                  'sequence_complete_sha256': sha(OUT / 'complete.json'), 'test_opened': False,
                  'calibration_performance_is_selection_optimistic': True}
    save(OUT / 'COMPARISON.json', comparison)
    report = ['固定193维输入的逐token MLP与离线7-token TCN已各训练30轮；同一seed与权重。以下是校准选轮/阈值后的开发成绩，不是最终测试。', '',
              '| 方法 | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |', '|---|---:|---:|---:|---:|']
    for entry in entries:
        m = entry['metrics']; report.append(f"| {entry['candidate']} | {m['fit']['windows']['f1']:.3f} | {m['calibration']['windows']['f1']:.3f} | {m['fit']['answers']['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    report += ['', '标点保留为上下文并占4 raw BPE窗口位置；不参加loss，也不让其未监督概率进入窗口max。两网络完全同口径。',
               'TCN每个残差块仅一个卷积，dilation1/2，总感受野7 raw tokens，使用后续token；参数数更多，不能单独归因为上下文作用。',
               'PCA64与四段层平均是固定的信息压缩，不能用这轮结果推断原隐藏状态没有检测信息。全部60轮模型、概率、阈值与曲线已保留。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', 'utf-8')
    print('QA_SEQUENCE_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['preflight', 'freeze', 'train', 'compare'])
    args = parser.parse_args()
    torch.set_num_threads(THREADS); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=THREADS):
        {'preflight': cpu_checks_and_benchmark, 'freeze': freeze, 'train': train, 'compare': compare}[args.stage]()
