"""CPU-only residual-head training on a frozen, selected semantic encoder cache."""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
import run_development as q
import run_full_context_fusion_v2 as fusion
import cache_frozen_context_fusion as cache

OUT = q.ROOT / 'results/frozen_context_generation_fusion_v1'
NFIT, NDEV, EPOCHS, MASS = 3680, 3839, 6, 560300
MODES = ('fusion', 'semantic_only')


def design():
    OUT.mkdir(parents=True, exist_ok=True)
    p = {'version': 'frozen-semantic-generation-head-v1',
        'scope': '3680fit+159calibration, original full inputs/labels/4rawBPE windows. Official test remains sealed.',
        'architecture': 'Fixed upstream selected semantic z plus generation_residual_fusion_v2 gated residual: mapped semantic768, LB1024, NLL and LB mean/2*population std. Train52944 head parameters only; no backbone loaded or updated by this entry.',
        'modes': {'fusion': 'Original aligned LB/NLL.', 'semantic_only': 'Identical head but LB/NLL and absolute LB summaries zero.'},
        'optimizer': {'name': 'AdamW', 'lr': 1e-4, 'weight_decay': .01, 'clip_norm': 1., 'precision': 'FP32 CPU', 'threads': 4},
        'training': {'epochs_per_mode': EPOCHS, 'effective_answers': 8, 'microbatch_answers': 1,
            'head_seed': fusion.FUSION_SEED, 'target_mass': MASS,
            'weights_orders': 'Read exact committed full_context_generation_fusion_v2 weights and all6 orders without changes.',
            'loss': 'sum(tokenBCE*original_loss_weights)*3680/(actual_batch_answers*560300), accumulate before clipping/update; original zero lexical-mask weights retained.'},
        'selection': 'Save epochs0..6. Separate cal-only window/answer thresholds with original F1/precision/higher-threshold ties. Select only epochs1..6 by min(windowF1,answerF1), windowF1, windowPrecision, earlier epoch. No upstream reselection.',
        'numerics': 'Always actual CPU sigmoid(z+residual), including epoch0. Zero residual must preserve logit exactly; report CPU-versus-GPU probability differences and original-threshold counts, without replacing probabilities or modifying gradients.',
        'limits': 'Two-stage extra semantic detector plus original-generator white-box signals; not a pure generation probe. Upstream selected using the same calibration, not cross-fitted. Frozen training-state cache contains in-sample upstream fit states. No claim of independent calibration or same compute as QA-only training.',
        'artifacts': 'All7 head+optimizer/RNG checkpoints, token probabilities, two-granularity scores, thresholds, metrics, epochs/updates/times for each mode. No implicit overwrite/resume.'}
    fusion.freeze(OUT / 'protocol.json', p)
    print('FROZEN_HEAD_DESIGN_READY', flush=True)


def cpu_check():
    design(); torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    before = torch.get_rng_state().clone(); model = fusion.new_fusion('cpu')
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 52944
    h = torch.zeros(4, 768); z = torch.tensor([-.5, 0., .5, 1.])
    lb = torch.linspace(.1, .9, 4096).reshape(4, 1024); nll = torch.ones(4, 1)
    output = model(h, z, lb, nll); assert torch.equal(output, z)
    # Exactly the production weighted BCE normalization on a tiny fixed example.
    loss = (F.binary_cross_entropy_with_logits(output, torch.tensor([1., 1., 0., 0.]), reduction='none') * torch.ones(4)).sum() * NFIT / (8 * MASS)
    loss.backward()
    assert (model.residual.weight.grad[0, -2:].abs() > 0).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    lo, nn = fusion.feature_row(np.ones((4, 1025), np.float32), np.array([[0, 4]]), 0, 'semantic_only')
    parts = model.components(h, z, torch.from_numpy(lo), torch.from_numpy(nn)[:, None])
    assert torch.count_nonzero(parts['lookback_stats']) == 0 and torch.equal(parts['logit'], z)
    assert not h.requires_grad and not z.requires_grad and not torch.cuda.is_initialized()
    fusion.freeze(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'initial_logit_exact': True,
        'finite_backward_and_added_weights_receive_gradient': True, 'semantic_only_zero_generation': True,
        'trainable_parameters': 52944, 'backbone_loaded': False, 'GPU_used': False,
        'code_sha256': q.sha(Path(__file__)), 'architecture_sha256': q.sha(Path(fusion.architecture.__file__))})
    print('FROZEN_HEAD_CPU_CHECK_PASSED', flush=True)


def source_hashes():
    paths = [Path(__file__), Path(cache.__file__), Path(fusion.__file__), Path(fusion.architecture.__file__),
        Path(fusion.evaluation.__file__), Path(fusion.mapping.__file__), Path(q.__file__),
        cache.OUT / 'complete.json', OUT / 'protocol.json', OUT / 'CPU_SELFCHECK.json',
        fusion.OUT / 'preparation_complete.json']
    paths += [fusion.OUT / n for n in ('inputs.jsonl', 'training_weights.npz', 'answer_orders.npy', 'generation.npy', 'generation_bounds.npy')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def prepare():
    design()
    assert not torch.cuda.is_initialized()
    assert q.read(OUT / 'CPU_SELFCHECK.json')['passed']
    done = cache.check_complete()
    if done is None:
        return None
    fusion.check_prepared()
    bounds = np.load(fusion.OUT / 'generation_bounds.npy')
    assert np.array_equal(bounds, np.load(cache.OUT / 'bounds.npy'))
    with np.load(fusion.OUT / 'training_weights.npz') as w:
        assert int(w['target_mass']) == MASS and np.array_equal(w['bounds'], bounds[:NFIT])
    assert np.load(cache.OUT / 'hidden.npy', mmap_mode='r').shape == (708506, 768)
    assert np.load(fusion.OUT / 'answer_orders.npy').shape == (6, NFIT)
    report = {'status': 'prepared_not_trained', 'source_sha256': source_hashes(),
        'selected_upstream_epoch': done['selected_epoch'], 'trainable_head_parameters': 52944,
        'GPU_used': False, 'official_test_opened': False}
    fusion.freeze(OUT / 'preparation_complete.json', report)
    print('FROZEN_HEAD_PREPARED', flush=True)
    return report


def train(mode):
    prepared = prepare()
    if prepared is None:
        return
    torch.set_num_threads(4); assert not torch.cuda.is_initialized()
    directory = OUT / mode; directory.mkdir(exist_ok=True)
    assert not (directory / 'started.json').exists(), 'No overwrite or implicit resume'
    q.save(directory / 'started.json', {'time': time.time(), 'mode': mode,
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    answers, tokens, _ = fusion.mapping.metadata()
    rows = q.lines(fusion.OUT / 'inputs.jsonl')
    assert len(rows) == len(answers) == NDEV
    assert [r['response_id'] for r in rows] == [a['response_id'] for a in answers]
    signal = np.load(fusion.OUT / 'generation.npy', mmap_mode='r')
    hidden = np.load(cache.OUT / 'hidden.npy', mmap_mode='r')
    baseline = np.load(cache.OUT / 'baseline_logit.npy', mmap_mode='r')
    gpu_probability = np.load(cache.OUT / 'baseline_gpu_probability.npy', mmap_mode='r')
    bounds = np.load(cache.OUT / 'bounds.npy')
    order = np.load(fusion.OUT / 'answer_orders.npy')
    with np.load(fusion.OUT / 'training_weights.npz') as z:
        weights = {k: z[k].copy() for k in z.files}
    upstream = q.read(cache.OUT / 'REPLAY.json')
    model = fusion.new_fusion('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    def forward(i):
        lo, hi = bounds[i]
        h = torch.from_numpy(np.array(hidden[lo:hi], copy=True))
        original_z = torch.from_numpy(np.array(baseline[lo:hi], copy=True))
        lb, nll = fusion.feature_row(signal, bounds, i, mode)
        result = model(h, original_z, torch.from_numpy(lb), torch.from_numpy(nll)[:, None])
        return result, original_z
    history = []; best = None; start = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        tick = time.perf_counter(); online = 0.
        if epoch:
            model.train()
            for first in range(0, NFIT, 8):
                chosen = order[epoch - 1, first:first + 8]; optimizer.zero_grad(set_to_none=True)
                for i in chosen:
                    score, _ = forward(i); lo, hi = weights['bounds'][i]
                    y = torch.from_numpy(weights['y'][lo:hi]).float()
                    w = torch.from_numpy(weights['loss'][lo:hi]).float()
                    objective = (F.binary_cross_entropy_with_logits(score, y, reduction='none') * w).sum() * NFIT / (len(chosen) * MASS)
                    assert torch.isfinite(objective); objective.backward(); online += float(objective.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(norm)
                optimizer.step()
                if (first + len(chosen)) % 400 == 0:
                    print('FROZEN_HEAD_TRAIN', mode, epoch, first + len(chosen), round(time.perf_counter() - tick, 1), flush=True)
        model.eval(); probabilities = {}; bce = 0.; numerical = []
        with torch.no_grad():
            for i, row in enumerate(rows):
                z, original_z = forward(i); assert torch.isfinite(z).all()
                p = torch.sigmoid(z).numpy(); probabilities[row['response_id']] = p
                if epoch == 0:
                    assert torch.equal(z, original_z), 'Zero residual must preserve the cached logit exactly'
                    lo, hi = bounds[i]
                    numerical.append(dict(response_id=row['response_id'], **cache.difference(p, gpu_probability[lo:hi])))
                if i < NFIT:
                    lo, hi = weights['bounds'][i]
                    y = torch.from_numpy(weights['y'][lo:hi]).float(); w = torch.from_numpy(weights['loss'][lo:hi]).float()
                    bce += float((F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).double().sum())
        fit = fusion.evaluation.score_geometry(answers, tokens, probabilities, range(NFIT))
        cal = fusion.evaluation.score_geometry(answers, tokens, probabilities, range(NFIT, NDEV))
        assert len(fit['window_scores']) == 653979 and len(cal['window_scores']) == 42241
        if epoch == 0:
            old_threshold = upstream['original_thresholds']
            q.save(directory / 'ZERO_RESIDUAL_REPLAY.json', {'logit_exact': True,
                'cpu_sigmoid_vs_original_gpu': numerical,
                'max_probability_difference': max(n['max_absolute'] for n in numerical),
                'nonexact_tokens': sum(n['nonexact'] for n in numerical),
                'metrics_at_original_thresholds': {'fit': fusion.evaluation.metrics_for(fit, old_threshold), 'calibration': fusion.evaluation.metrics_for(cal, old_threshold)},
                'original_gpu_metrics': upstream['upstream_original_metrics'],
                'original_thresholds_unchanged': old_threshold, 'probability_replacement': False})
        thresholds = {'window': q.choose_threshold(cal['window_labels'], cal['window_scores']),
                      'answer': q.choose_threshold(cal['answer_labels'], cal['answer_scores'])}
        key = [min(thresholds['window']['f1'], thresholds['answer']['f1']), thresholds['window']['f1'], thresholds['window']['precision'], -epoch]
        stem = f'epoch_{epoch:02d}'
        torch.save({'fusion_state_dict': fusion.evaluation.cpu_state(model.state_dict()),
            'optimizer_state_dict': fusion.evaluation.cpu_state(optimizer.state_dict()),
            'torch_rng_state': torch.get_rng_state(), 'mode': mode, 'epoch': epoch,
            'preparation_sha256': q.sha(OUT / 'preparation_complete.json')}, directory / (stem + '.pt'))
        np.savez_compressed(directory / (stem + '_token_predictions.npz'), **probabilities)
        np.savez_compressed(directory / (stem + '_scores.npz'), **{'fit_' + k: v for k, v in fit.items()}, **{'cal_' + k: v for k, v in cal.items()})
        entry = {'epoch': epoch, 'mode': mode, 'thresholds': thresholds, 'selection_key': key,
            'fit_at_cal_thresholds': fusion.evaluation.metrics_for(fit, thresholds), 'calibration': fusion.evaluation.metrics_for(cal, thresholds),
            'fit_weighted_bce': bce / MASS, 'online_objective_sum': online,
            'optimizer_steps': 460 if epoch else 0, 'cumulative_optimizer_steps': epoch * 460,
            'seconds': time.perf_counter() - tick,
            'artifacts_sha256': {ext: q.sha(directory / (stem + ext)) for ext in ('.pt', '_token_predictions.npz', '_scores.npz')},
            'GPU_used': False, 'official_test_opened': False}
        q.save(directory / (stem + '.json'), entry); history.append(entry)
        if epoch and (best is None or key > best['selection_key']):
            best = entry
        print('FROZEN_HEAD_EPOCH_COMPLETE', mode, epoch, entry['calibration']['windows']['f1'], entry['calibration']['answers']['f1'], flush=True)
    assert prepared['source_sha256'] == source_hashes()
    assert not torch.cuda.is_initialized()
    q.save(directory / 'complete.json', {'status': 'complete_development_only', 'mode': mode, 'selected': best,
        'all_epochs': history, 'seconds': time.perf_counter() - start, 'trainable_parameters': 52944,
        'backbone_updated': False, 'GPU_used': False, 'official_test_opened': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('design', 'cpu-check', 'prepare', 'train'))
    parser.add_argument('--mode', choices=MODES, default='fusion'); args = parser.parse_args()
    try:
        if args.action == 'train': train(args.mode)
        else: {'design': design, 'cpu-check': cpu_check, 'prepare': prepare}[args.action]()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        q.save(OUT / f'FAILURE_{args.action}_{time.time_ns()}.json', {'mode': args.mode, 'error': repr(exc), 'GPU_used': False, 'official_test_opened': False})
        raise
