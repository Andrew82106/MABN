"""Two fixed CPU capacity controls on frozen full-LB+PCA64 token features."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F

import run_sequence_full as full

ROOT = full.ROOT; OLD = full.OLD; SOURCE = full.OUT
OUT = ROOT / 'results/sequence_capacity_v3'
METHODS = {'full_lb_pca64_tcn_w16': 16, 'full_lb_pca64_tcn_w32': 32}
SEED, EPOCHS, BATCH, THREADS = 20260926, 30, 8, 4
sha, read, save = full.sha, full.read, full.save
DESIGN = SOURCE / 'matrices/full_lb_pca64_tcn.npy'


def protocol():
    return {'version': 'qa-fixed-full-input-capacity-v3', 'scope': 'fit634/calibration159 only; official test never opened',
        'methods_widths': METHODS, 'input': 'Exact frozen standardized sequence_full_v2 A: LB1024+NLL1+PCA64,1089 dimensions',
        'reused_input_sha256': sha(DESIGN), 'scaler_or_PCA_refit': False,
        'network': '1089->width GELU; two single residual Conv1d kernel3 dilation1/2, GELU dropout.2; width->1',
        'receptive_field_raw_tokens': 7, 'offline_future_tokens': True,
        'seed': SEED, 'epochs_each': EPOCHS, 'batch_answers': BATCH, 'cpu_threads': THREADS,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .01, 'all_parameters': True},
        'weights_and_shuffle': 'Reuse sequence_v1 exact fit_token_weights and epoch_answer_order, target loss139518',
        'loss_aggregation_selection': 'Unchanged full_v2/sequence_v1 rules: lexical loss, lexical-only max inside same4raw windows, all-window answer max; cal maxmin F1 then windowF1/precision/earlier epoch',
        'early_stop': False, 'additional_seeds': False,
        'reference': 'Frozen full_v2 A width64 selected checkpoint, not retrained; old MLP/193dimTCN retained',
        'diagnostics': 'Report fit/cal gap, selected and final fit BCE/calF1, same global-threshold type recalls, with no type-specific tuning',
        'limitations': ['Only two prespecified widths and one seed; cal checkpoint selection remains optimistic.',
            'Capacity changes parameter count and initialization/optimization; it does not identify a unique cause of errors.',
            'All three widths use the same compressed PCA64; failure does not establish missing information in the full hidden state.']}


class CapacityTCN(nn.Module):
    def __init__(self, width):
        super().__init__(); self.projection = nn.Linear(1089, width)
        self.convolutions = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in (1, 2)])
        self.dropout = nn.Dropout(.2); self.output = nn.Linear(width, 1)

    forward = full.v1.TokenTCN.forward


def new_model(width):
    torch.manual_seed(SEED)
    return CapacityTCN(width).cpu()


def source():
    assert not torch.cuda.is_initialized()
    completion = read(SOURCE / 'complete.json'); prep = read(SOURCE / 'preparation_manifest.json')
    assert read(SOURCE / 'AUDIT.json')['status'] == 'passed'
    assert sha(DESIGN) == prep['files_sha256'][str(DESIGN.relative_to(SOURCE))]
    assert sha(SOURCE / 'summary.json') == completion['files_sha256']['summary.json']
    assert sha(OLD / 'fit_token_weights.npz') == prep['weights_sha256']
    assert sha(OLD / 'epoch_answer_order.npy') == prep['shuffle_sha256']
    meta = full.v1.baseline.metadata(); index = read(OLD / 'token_index.json')['answers']
    assert len(index) == 793 and sum(a['token_count'] for a in index) == 213159
    paths = [Path(__file__), ROOT / 'src/run_sequence_full.py', ROOT / 'src/run_sequence.py', ROOT / 'src/run_development.py',
             SOURCE / 'complete.json', SOURCE / 'preparation_manifest.json', SOURCE / 'summary.json', SOURCE / 'AUDIT.json',
             SOURCE / 'full_lb_pca64_tcn_scaler.pkl', DESIGN,
             OLD / 'fit_token_weights.npz', OLD / 'epoch_answer_order.npy', OLD / 'token_index.json', ROOT / 'data/gold_manifest.json']
    snap = {'files_sha256': {str(path.resolve()): sha(path) for path in paths},
            'test_opened': False, 'GPU_used': False, 'same_input_as_frozen_width64': True}
    return meta, index, snap


def preflight():
    meta, index, snap = source(); reports = []
    generator = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(2, 16, 1089, generator=generator); mask = torch.ones(2, 16); mask[0, 9:] = 0
    for method, width in METHODS.items():
        model = new_model(width).eval()
        with torch.no_grad():
            original = model(x, mask); changed = x.clone(); changed[0, 9:] = 1000
            assert torch.allclose(model(changed, mask)[0, :9], original[0, :9], atol=2e-6, rtol=1e-6)
            assert torch.allclose(model(x[:1, :9], torch.ones(1, 9))[0], original[0, :9], atol=2e-6, rtol=1e-6)
            changed = x.clone(); changed[1] = 1000; assert torch.equal(model(changed, mask)[0], original[0])
            changed = x.clone(); changed[1, :4] += 100; assert torch.equal(model(changed, mask)[1, 8], original[1, 8])
        reports.append({'method': method, 'width': width, 'parameters': sum(p.numel() for p in model.parameters()),
                        'padding_answer_isolation_and_RF7_checks_passed': True})
    save(OUT / 'PREFLIGHT.json', {'passed': True, 'models': reports, 'source_snapshot': snap, 'cuda_initialized': torch.cuda.is_initialized()})
    print('QA_CAPACITY_PREFLIGHT_PASSED', reports, flush=True)


def freeze():
    assert not (OUT / 'protocol.json').exists(); pre = read(OUT / 'PREFLIGHT.json'); assert pre['passed']
    snap = source()[-1]; assert snap == pre['source_snapshot']
    save(OUT / 'protocol.json', protocol()); save(OUT / 'source_snapshot.json', snap)
    save(OUT / 'freeze.json', {'utc': time.time(), 'protocol_sha256': sha(OUT / 'protocol.json'),
                             'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'), 'test_opened': False})
    print('QA_CAPACITY_FROZEN', flush=True)


def train():
    assert read(OUT / 'protocol.json') == protocol() and not (OUT / 'started.json').exists()
    meta, index, snap = source(); assert snap == read(OUT / 'source_snapshot.json')
    save(OUT / 'started.json', {'utc': time.time(), 'freeze_sha256': sha(OUT / 'freeze.json'), 'test_opened': False})
    x = np.load(DESIGN, mmap_mode='r'); assert x.shape == (213159, 1089)
    with np.load(OLD / 'fit_token_weights.npz', allow_pickle=False) as z: weights = {k: z[k].copy() for k in z.files}
    order = np.load(OLD / 'epoch_answer_order.npy'); assert order.shape == (30, 634)
    started = time.perf_counter(); files = []; selected = {}; histories = {}
    for method, width in METHODS.items():
        model = new_model(width); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        directory = OUT / method; directory.mkdir(parents=True, exist_ok=True); history = []; best = None
        for epoch in range(EPOCHS):
            begin = time.perf_counter(); model.train()
            for left in range(0, 634, BATCH):
                chosen = order[epoch, left:left + BATCH]; xx, mask, yy, ww = full.batch(chosen, x, index, weights)
                optimizer.zero_grad(set_to_none=True)
                objective = (F.binary_cross_entropy_with_logits(model(xx, mask), yy, reduction='none') * ww).sum() * (634 / (len(chosen) * 139518))
                assert torch.isfinite(objective); objective.backward(); optimizer.step()
            fitting = time.perf_counter() - begin; logits, probabilities = full.predict(model, x, index)
            arrays, thresholds, metrics, loss = full.v1.evaluate(meta, index, logits, probabilities, weights)
            arrays.update(token_scores=probabilities, token_logits=logits)
            name = f'epoch_{epoch + 1:03d}'; scorepath = directory / (name + '_scores.npz'); np.savez_compressed(scorepath, **arrays)
            modelpath = directory / (name + '.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'width': width, 'method': method, 'epoch': epoch + 1, 'seed': SEED, 'device': 'cpu',
                        'protocol_sha256': sha(OUT / 'protocol.json'), 'source_snapshot_sha256': sha(OUT / 'source_snapshot.json'),
                        'torch_rng_state': torch.get_rng_state()}, modelpath)
            w, a = thresholds['window'], thresholds['answer']; key = [min(w['f1'], a['f1']), w['f1'], w['precision'], -(epoch + 1)]
            entry = {'method': method, 'width': width, 'epoch': epoch + 1, 'parameters': sum(p.numel() for p in model.parameters()),
                     'thresholds': thresholds, 'selection_key': key, 'metrics': metrics, 'full_fit_weighted_BCE': loss,
                     'epoch_training_seconds': fitting, 'epoch_total_seconds': time.perf_counter() - begin,
                     'scores_sha256': sha(scorepath), 'model_sha256': sha(modelpath), 'test_opened': False}
            reportpath = directory / (name + '.json'); save(reportpath, entry); history.append(entry)
            files.extend(str(p.relative_to(OUT)) for p in (scorepath, modelpath, reportpath))
            if best is None or key > best: best = key; selected[method] = entry
            save(OUT / 'progress.json', {'method': method, 'epoch': epoch + 1, 'selected': selected, 'seconds': time.perf_counter() - started})
            print('QA_CAPACITY_EPOCH', width, epoch + 1, 'bce', round(loss, 6), 'cal_w', round(w['f1'], 6), 'cal_a', round(a['f1'], 6), flush=True)
        histories[method] = history; del model, optimizer; gc.collect()
    assert source()[-1] == snap and not torch.cuda.is_initialized()
    summary = {'selected': selected, 'all_epochs': histories, 'wall_seconds': time.perf_counter() - started, 'seed': SEED,
               'models': 2, 'epochs': 60, 'test_opened': False, 'gpu_used': False,
               'frozen_width64_retrained': False, 'calibration_results_selection_optimistic': True}
    save(OUT / 'summary.json', summary); files.extend(['summary.json', 'PREFLIGHT.json', 'protocol.json', 'source_snapshot.json', 'freeze.json'])
    save(OUT / 'complete.json', {'status': 'complete_development_only', 'files_sha256': {name: sha(OUT / name) for name in files},
                                'script_sha256': sha(Path(__file__)), 'test_opened': False, 'gpu_used': False})
    print('QA_CAPACITY_COMPLETE', round(summary['wall_seconds'], 2), flush=True)


def report():
    meta = full.v1.baseline.metadata(); current = read(OUT / 'summary.json'); old = read(SOURCE / 'summary.json')
    entries = dict(current['selected']); entries['full_lb_pca64_tcn'] = old['selected']['full_lb_pca64_tcn']
    rows = []; diagnostics = {}
    for method, entry in entries.items():
        directory = OUT if method in METHODS else SOURCE
        path = directory / method / f"epoch_{entry['epoch']:03d}_scores.npz"; assert sha(path) == entry['scores_sha256']
        with np.load(path, allow_pickle=False) as z: window_scores = z['window_scores'].copy()
        alerts = window_scores >= entry['thresholds']['window']['threshold']; covered = {}
        for w, flag in zip(meta['windows'], alerts):
            if flag: covered.setdefault(w['response_id'], set()).update(w['token_indices'])
        typed = {}
        for kind in ('Evident Conflict', 'Subtle Conflict'):
            spans = []; risk = {}; empty = 0
            for tokens in meta['tokens']:
                if tokens['partition'] != 'calibration': continue
                for mapping in tokens['span_token_mapping']:
                    label = tokens['original_labels'][mapping['span_index']]
                    if label['label_type'] != kind: continue
                    rid = tokens['response_id']; ix = set(mapping['risk_token_indices'])
                    if not ix: empty += 1; continue
                    risk.setdefault(rid, set()).update(ix)
                    spans.append((bool(ix & covered.get(rid, set())), ix <= covered.get(rid, set())))
            pos = [j for j, w in enumerate(meta['windows']) if w['partition'] == 'calibration' and bool(set(w['token_indices']) & risk.get(w['response_id'], set()))]
            typed[kind] = {'original_spans': len(spans) + empty, 'localizable_spans': len(spans), 'unlocalizable_spans': empty,
                'span_any_hit': sum(s[0] for s in spans), 'span_full_hit': sum(s[1] for s in spans),
                'positive_windows': len(pos), 'hit_positive_windows': int(alerts[pos].sum()),
                'positive_window_recall': float(alerts[pos].mean()) if pos else None}
        diagnostics[method] = typed
        history = current['all_epochs'][method] if method in METHODS else old['all_epochs'][method]
        rows.append({'method': method, 'width': entry.get('width', 64), 'epoch': entry['epoch'], 'parameters': entry['parameters'],
                     'metrics': entry['metrics'], 'selected_fit_BCE': entry['full_fit_weighted_BCE'],
                     'final_fit_BCE': history[-1]['full_fit_weighted_BCE'],
                     'final_cal_window_F1': history[-1]['metrics']['calibration']['windows']['f1'],
                     'fit_minus_cal_window_F1': entry['metrics']['fit']['windows']['f1'] - entry['metrics']['calibration']['windows']['f1']})
    comparison = read(SOURCE / 'BASELINE_COMPARISON.json')['entries']
    comparison += [{'candidate': method, 'kind': 'new_capacity', 'metrics': entry['metrics']} for method, entry in current['selected'].items()]
    save(OUT / 'DIAGNOSTICS.json', {'rows': rows, 'calibration_conflict': diagnostics, 'same_global_threshold_no_type_tuning': True,
                                  'width64_reused_without_refit': True, 'test_opened': False})
    save(OUT / 'BASELINE_COMPARISON.json', {'entries': comparison, 'same_geometry_and_labels': True, 'test_opened': False,
                                          'full_v2_complete_sha256': sha(SOURCE / 'complete.json'), 'capacity_complete_sha256': sha(OUT / 'complete.json')})
    text = ['相同1089维输入与RF7，只有TCN宽度改变；宽64原冻结结果直接复用。全部是校准选轮/选阈值后的开发成绩。', '',
            '| 宽度 | 参数 | 轮次 | fit窗F1 | cal窗F1 | cal整答F1 | 最后轮cal窗F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda row: row['width']):
        m = row['metrics']; text.append(f"| {row['width']} | {row['parameters']} | {row['epoch']} | {m['fit']['windows']['f1']:.3f} | {m['calibration']['windows']['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} | {row['final_cal_window_F1']:.3f} |")
    text += ['', '| 宽度 | 冲突类 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |', '|---|---|---:|---:|---:|']
    for row in sorted(rows, key=lambda row: row['width']):
        for kind, d in diagnostics[row['method']].items():
            text.append(f"| {row['width']} | {kind} | {d['hit_positive_windows']}/{d['positive_windows']} | {d['span_any_hit']}/{d['localizable_spans']} | {d['span_full_hit']}/{d['localizable_spans']} |")
    text += ['', '权重、特征、PCA、scaler、shuffle、优化器、30轮及全局阈值规则均沿用；未打开test。',
             'Subtle Conflict只有5个校准span。单seed和校准反复选型不能确认泛化或唯一误差原因；未根据部分结果增加宽度/轮数。']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n', 'utf-8')
    print('QA_CAPACITY_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['preflight', 'freeze', 'train', 'report']); args = parser.parse_args()
    torch.set_num_threads(THREADS); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=THREADS):
        {'preflight': preflight, 'freeze': freeze, 'train': train, 'report': report}[args.stage]()
