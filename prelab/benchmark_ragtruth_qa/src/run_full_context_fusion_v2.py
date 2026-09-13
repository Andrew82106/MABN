"""Independent v2 fusion: add absolute LB summaries; reuse all prepared v1 data.

CPU design/checks/prepare only unless GPU is separately scheduled. prepare
requires committed v1 preparation and never re-extracts any source cache.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
import run_development as q
import run_full_context_encoder_v2 as base
import tail_finetune as mapping
import tail_finetune_all_docs as evaluation
import modernbert_generation_bridge as bridge
import generation_residual_fusion_v2 as architecture
import run_full_context_fusion as original

ROOT = q.ROOT
OUT = ROOT / 'results/full_context_generation_fusion_v2'
MODES = ('fusion', 'semantic_only')
FUSION_SEED = 20261006
REUSED_FILES = ('inputs.jsonl', 'training_weights.npz', 'answer_orders.npy',
                'generation.npy', 'generation_bounds.npy', 'generation_manifest.json')


def protocol():
    return {'version': 'qa-full-context-generation-residual-v2-absolute-lb',
        'scope': 'Same frozen3680 QA fit and159 calibration, complete input, human labels and4rawBPE metrics as full_context_encoder_v2. No auxiliary9678 and no official test.',
        'controls': {'fusion': 'Actual aligned legacy LB1024 and NLL1.',
            'semantic_only': 'Identical residual architecture and budget, but LB and NLL inputs are identically zero, including both added LB summaries. Extra learned semantic capacity control.'},
        'signals': 'Original793 and new3046 uniform Llama-2-7B-chat NF4 teacher-forced replay, source-only post-read legacy LB, selected token NLL from previous position. Same exact original full-string BPE IDs and offsets, no new tokenization.',
        'representation': 'One unchanged ModernBERT token-classifier forward. Map actual two-logit difference and final768 encoder state independently onto original raw BPE using existing nonwhitespace character map. FP32 mapping and fusion.',
        'residual': 'Preserve semantic LayerNorm768→32 SiLU, Lookback LayerNorm1024→24 SiLU and log1p(NLL)→8 SiLU. Concatenate semantic32+generation32+baseline_logit1+LBmean1+2*LB_population_std1; sigmoid67→1 gate times67→1 residual. Both head biases unchanged in size; only4 extra weight parameters. Residual weight/bias initialized exactly zero.',
        'absolute_summaries': 'Compute mean and population std over all1024 LB heads at the same original token. LB in[0,1] implies mean and2*std in[0,1]. No fitted scaler, no clipping, and no LayerNorm on these two values.',
        'reuse': 'Copy inputs.jsonl, generation.npy, generation_bounds.npy, generation_manifest.json, training_weights.npz and answer_orders.npy from completed full_context_generation_fusion_v1; verify exact byte hashes. No source feature re-extraction or new tokenization. Preserve v1 and currently running plain semantic v2.',
        'initialization': {'backbone_seed': base.SEED, 'fusion_seed': FUSION_SEED,
            'rng': 'Construct fusion on CPU under fork_rng(devices=[]), seed only CPU default_generator, then move to device. Preserve all backbone/dropout CPU and CUDA RNG states. No reset during epochs.'},
        'training': {'epochs_per_mode': 6, 'backbone_lr': 1e-5, 'fusion_lr': 1e-4,
            'optimizer': 'AdamW', 'weight_decay': .01, 'clip_combined_gradient_norm': 1.,
            'effective_batch_answers': 8, 'microbatch_answers': 1,
            'precision': 'Unchanged v2 BF16 model forward/non-reentrant recompute, FP32 parameters/gradients/optimizer, hidden/logit mapping, residual head and BCE; TF32 off.',
            'weights_orders': 'Byte-identical v2 training_weights.npz and all6 answer_orders.npy; no learned feature scaler.',
            'loss': 'Original token risk BCE*loss weights; each answer sum scaled3680/(actual batch answers*560300); punctuation and whitespace keep original zero loss.'},
        'selection': 'Unchanged v2: every epoch full fit/cal; separate cal-only F1 thresholds with precision/higher-cutoff tie; choose epochs1..6 by min(windowF1,answerF1),windowF1,windowPrecision,earlier epoch. Epoch0 not selectable.',
        'artifacts': 'All7 per-mode model+fusion+optimizer/RNG checkpoints, raw token predictions, window/answer scores, thresholds, metrics, time and memory. No overwrite/resume.',
        'execution': 'CPU preparation first. Actual GPU smoke uses fixed longest QA input, zero synthetic targets and no real labels. Root must separately schedule either mode; never interrupt scheduled v2 baseline.',
        'limits': ['This v2 is an untrained replacement candidate. Synthetic information-loss evidence does not establish a main F1 bottleneck or promise improvement; original fit mean-only AUROC is about0.522.',
            'Adds a fully trained semantic checker, not a pure original-generator white-box probe.',
            'ModernBERT sees the complete answer including future tokens: offline detection, not online.',
            'Added3046 are texts from other generators replayed in one Llama checkpoint, not those generators native trajectories.',
            'Comparing fusion with semantic_only isolates availability of generation signals conditional on this common architecture; comparison to plain v2 also changes capacity.',
            'Threshold/epoch selection reuses development calibration; official held-out test remains sealed.']}


def freeze(path, value):
    if path.exists():
        assert q.read(path) == value, ('Frozen content changed', str(path))
    else:
        q.save(path, value)


def design():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', protocol())
    print('FUSION_DESIGN_FROZEN', flush=True)


def new_fusion(device):
    # torch.manual_seed would also reseed CUDA; seed only this CPU generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FUSION_SEED)
        model = architecture.GenerationResidualFusion()
    return model.to(device)


def source_files():
    paths = [Path(__file__), Path(bridge.__file__), Path(architecture.__file__), Path(base.__file__),
        Path(original.__file__), Path(original.architecture.__file__),
        original.OUT / 'preparation_complete.json', original.OUT / 'protocol.json',
        original.OUT / 'source_snapshot.json',
        base.OUT / 'preparation_complete.json', base.OUT / 'protocol.json', base.OUT / 'inputs.jsonl',
        base.OUT / 'training_weights.npz', base.OUT / 'answer_orders.npy',
        OUT / 'protocol.json'] + [original.OUT / n for n in REUSED_FILES]
    return {str(p.resolve()): q.sha(p) for p in paths}


def prepare():
    design()
    if not (original.OUT / 'preparation_complete.json').exists():
        print('WAIT_COMPLETE_V1_PREPARATION_NO_FIT', flush=True)
        return
    assert not torch.cuda.is_initialized()
    old = original.check_prepared()
    assert not (OUT / 'prepare_started.json').exists(), 'No implicit overwrite/resume'
    assert not any((OUT / n).exists() for n in REUSED_FILES)
    snapshot = source_files()
    q.save(OUT / 'prepare_started.json', {'time': time.time(), 'source_sha256': snapshot})
    reused = {}
    for name in REUSED_FILES:
        digest = old['files_sha256'][name]
        assert snapshot[str((original.OUT / name).resolve())] == digest
        shutil.copyfile(original.OUT / name, OUT / name)
        assert q.sha(OUT / name) == digest, name
        reused[name] = {'source': str((original.OUT / name).resolve()), 'sha256': digest,
                        'bytes': (OUT / name).stat().st_size}
        print('FUSION_V2_REUSED', name, flush=True)
    rows = q.lines(OUT / 'inputs.jsonl')
    signal = np.load(OUT / 'generation.npy', mmap_mode='r')
    bounds = np.load(OUT / 'generation_bounds.npy')
    orders = np.load(OUT / 'answer_orders.npy')
    assert len(rows) == 3839 and signal.shape == (708506, 1025) and signal.dtype == np.float32
    assert bounds.shape == (3839, 2) and bounds[0, 0] == 0 and bounds[-1, 1] == len(signal)
    assert np.array_equal(bounds[:-1, 1], bounds[1:, 0])
    assert all(int(hi - lo) == row['raw_token_count'] for row, (lo, hi) in zip(rows, bounds))
    assert orders.shape == (6, 3680)
    assert all(np.array_equal(np.sort(order), np.arange(3680)) for order in orders)
    with np.load(OUT / 'training_weights.npz') as weights:
        assert np.array_equal(weights['bounds'], bounds[:3680])
    manifest = q.read(OUT / 'generation_manifest.json')
    assert manifest['answers'] == 3839 and manifest['original_raw_tokens'] == len(signal)
    q.save(OUT / 'reuse_identity.json', {'source_preparation_sha256': q.sha(original.OUT / 'preparation_complete.json'),
        'files': reused, 'answers': 3839, 'raw_tokens': 708506, 'exact_bytes': True,
        'source_feature_reextraction': False, 'retokenization': False, 'GPU_used': False})
    del signal
    assert snapshot == source_files()
    q.save(OUT / 'source_snapshot.json', {'files_sha256': snapshot})
    names = REUSED_FILES + ('protocol.json', 'source_snapshot.json', 'reuse_identity.json')
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_trained', 'answers': 3839,
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'GPU_used': False, 'official_test_opened': False})
    assert not torch.cuda.is_initialized()
    print('FUSION_V2_ALL3839_REUSED_PREPARED', flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['answers'] == 3839 and q.read(OUT / 'protocol.json') == protocol()
    for name, digest in p['files_sha256'].items(): assert q.sha(OUT / name) == digest, name
    assert q.read(OUT / 'source_snapshot.json')['files_sha256'] == source_files()
    original.check_prepared()
    return p


def feature_row(signal, bounds, i, mode):
    lo, hi = bounds[i]
    if mode == 'semantic_only':
        return np.zeros((hi - lo, 1024), np.float32), np.zeros(hi - lo, np.float32)
    row = signal[lo:hi]
    return np.array(row[:, :1024], copy=True), np.array(row[:, 1024], copy=True)


def cpu_check():
    design()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    from transformers import ModernBertConfig, ModernBertForTokenClassification
    torch.manual_seed(base.SEED)
    config = ModernBertConfig(vocab_size=32, hidden_size=768, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=12, max_position_embeddings=64,
        local_attention=16, reference_compile=False, num_labels=2, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    model = ModernBertForTokenClassification(config).train()
    state = torch.get_rng_state().clone(); fusion = new_fusion('cpu')
    assert torch.equal(state, torch.get_rng_state())
    row = {'input_ids': [1, 5, 6, 7, 8, 9, 2], 'raw_token_count': 4,
        'mapping': [[0, 0, 1, 2, 3], [2, 3, 3, 4, 5], [.5, .5, 1., 1., 1.]]}
    lb = np.full((4, 1024), .4, np.float32); nll = np.ones(4, np.float32)
    state = torch.get_rng_state().clone(); expected = base.logits(model, row, 'cpu')
    torch.set_rng_state(state)
    actual, baseline_original = bridge.forward_fused(model, fusion, row, lb, nll, 'cpu')
    assert torch.equal(expected, baseline_original) and torch.equal(actual, expected)
    F.binary_cross_entropy_with_logits(actual, torch.tensor([0., 1., 0., 1.])).backward()
    assert fusion.residual.weight.grad.abs().sum() > 0
    assert model.model.embeddings.tok_embeddings.weight.grad.abs().sum() > 0
    assert torch.isfinite(fusion.residual.weight.grad).all()
    assert fusion.residual.weight.grad[0, -2].abs() > 0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FUSION_SEED)
        old_head = original.architecture.GenerationResidualFusion()
    old_parameters = sum(p.numel() for p in old_head.parameters())
    new_parameters = sum(p.numel() for p in fusion.parameters())
    assert new_parameters - old_parameters == 4
    x = torch.linspace(.125, .625, 1024)
    synthetic_lb = torch.stack((x, x + .25, torch.full_like(x, .25), torch.full_like(x, .75)))
    h = torch.zeros(4, 768); z = torch.zeros(4); n = torch.ones(4, 1)
    with torch.no_grad():
        parts = fusion.components(h, z, synthetic_lb, n)
        expected_stats = torch.stack((synthetic_lb.mean(-1), 2 * synthetic_lb.std(-1, unbiased=False)), -1)
        assert torch.equal(parts['lookback_stats'], expected_stats)
        assert ((expected_stats >= 0) & (expected_stats <= 1)).all()
        assert torch.equal(parts['logit'], z)
        assert abs(float(expected_stats[1, 0] - expected_stats[0, 0]) - .25) < 1e-7
        assert torch.equal(expected_stats[0, 1], expected_stats[1, 1])
        ln_branch_max_difference = float((parts['generation'][0] - parts['generation'][1]).abs().max())
        # Fixed synthetic readout, not optimization: show the preserved mean
        # is now accessible to the head even when the original LN is invariant.
        accessible = copy.deepcopy(fusion)
        accessible.gate.weight.zero_(); accessible.gate.bias.zero_()
        accessible.residual.weight.zero_(); accessible.residual.bias.zero_()
        accessible.residual.weight[0, -2] = 1
        changed = accessible(h, z, synthetic_lb, n)
        assert torch.allclose(changed, .5 * expected_stats[:, 0], atol=0, rtol=0)
        assert abs(float(changed[1] - changed[0]) - .125) < 1e-7
        control_lb, control_nll = feature_row(np.ones((4, 1025), np.float32), np.array([[0, 4]]), 0, 'semantic_only')
        control = fusion.components(h, z, torch.from_numpy(control_lb), torch.from_numpy(control_nll)[:, None])
        assert torch.count_nonzero(control['lookback_stats']) == 0
        assert not np.any(control_lb) and not np.any(control_nll)
    # Also check a nonconstant dispersion feature receives a finite gradient.
    fusion.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(fusion(h, z, synthetic_lb, n), torch.zeros(4)).backward()
    assert torch.isfinite(fusion.residual.weight.grad).all()
    assert (fusion.residual.weight.grad[0, -2:].abs() > 0).all()
    assert not torch.cuda.is_initialized()
    freeze(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'initial_fusion_preserves_cpu_rng': True,
        'dropout_path_baseline_and_zero_residual_exact': True, 'real_tiny_modernbert_backward': True,
        'old_parameter_count': old_parameters, 'new_parameter_count': new_parameters,
        'added_weight_parameters': new_parameters - old_parameters,
        'synthetic_stats': expected_stats.tolist(), 'shifted_ln_branch_max_difference': ln_branch_max_difference,
        'mean_shift_fixed_readout_logit_difference': float(changed[1] - changed[0]),
        'fixed_readout_is_not_trained': True, 'both_added_residual_weight_gradients_nonzero_finite': True,
        'semantic_only_zero_summaries': True, 'GPU_used': False, 'actual_training_performed': False,
        'source_sha256': q.sha(Path(__file__)), 'architecture_sha256': q.sha(Path(architecture.__file__)),
        'bridge_sha256': q.sha(Path(bridge.__file__))})
    print('FUSION_V2_CPU_CHECK_PASSED', flush=True)


def gpu_smoke():
    check_prepared()
    assert q.read(OUT / 'CPU_SELFCHECK.json')['passed']
    base.configure_gpu()
    rows = q.lines(OUT / 'inputs.jsonl')
    i = max(range(len(rows)), key=lambda j: (len(rows[j]['input_ids']), -j))
    row = rows[i]
    signal = np.load(OUT / 'generation.npy', mmap_mode='r')
    bounds = np.load(OUT / 'generation_bounds.npy')
    lb, nll = feature_row(signal, bounds, i, 'fusion')
    model = base.load_model().cuda()
    rng = torch.cuda.get_rng_state().clone(); cpu_rng = torch.get_rng_state().clone()
    fusion = new_fusion('cuda')
    assert torch.equal(rng, torch.cuda.get_rng_state()) and torch.equal(cpu_rng, torch.get_rng_state())
    model.eval(); fusion.eval()
    with torch.no_grad():
        expected = base.logits(model, row, 'cuda')
        fused, original = bridge.forward_fused(model, fusion, row, lb, nll, 'cuda')
        repeated, _ = bridge.forward_fused(model, fusion, row, lb, nll, 'cuda')
    assert torch.equal(fused, original) and torch.equal(original, expected)
    assert torch.equal(fused, repeated)
    model.train(); fusion.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 1e-5},
        {'params': fusion.parameters(), 'lr': 1e-4}], weight_decay=.01)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    fused, _ = bridge.forward_fused(model, fusion, row, lb, nll, 'cuda')
    objective = F.binary_cross_entropy_with_logits(fused, torch.zeros_like(fused))
    objective.backward()
    assert fusion.residual.weight.grad.abs().sum() > 0
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.); assert torch.isfinite(norm)
    optimizer.step(); base.check_fp32_state(model, optimizer)
    assert all(p.dtype == torch.float32 and (p.grad is None or torch.isfinite(p.grad).all()) for p in fusion.parameters())
    torch.cuda.synchronize()
    report = {'passed': True, 'response_id': row['response_id'], 'input_tokens': len(row['input_ids']),
        'synthetic_target_only': True, 'zero_residual_same_forward_exact': True, 'repeat_exact': True,
        'cpu_and_cuda_rng_unchanged_by_fusion_initialization': True, 'fp32_optimizer_and_parameters': True,
        'step_seconds': time.perf_counter() - start, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json'), 'official_test_opened': False}
    q.save(OUT / 'GPU_SELFCHECK.json', report)
    del model, fusion, optimizer; torch.cuda.empty_cache()
    print('FUSION_GPU_SMOKE_PASSED', flush=True)


def train(mode):
    check_prepared()
    gate = q.read(OUT / 'GPU_SELFCHECK.json')
    assert gate['passed'] and gate['preparation_sha256'] == q.sha(OUT / 'preparation_complete.json')
    assert q.read(OUT / 'CPU_SELFCHECK.json')['passed']
    directory = OUT / mode; directory.mkdir(exist_ok=True)
    assert not (directory / 'started.json').exists(), 'No automatic overwrite/resume'
    base.configure_gpu()
    answers, tokens, _ = mapping.metadata()
    rows = q.lines(OUT / 'inputs.jsonl')
    with np.load(OUT / 'training_weights.npz') as z: weights = {k: z[k].copy() for k in z.files}
    mass = int(weights['target_mass']); order = np.load(OUT / 'answer_orders.npy')
    signal = np.load(OUT / 'generation.npy', mmap_mode='r'); bounds = np.load(OUT / 'generation_bounds.npy')
    model = base.load_model().cuda(); fusion = new_fusion('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 1e-5},
        {'params': fusion.parameters(), 'lr': 1e-4}], weight_decay=.01)
    q.save(directory / 'started.json', {'time': time.time(), 'mode': mode,
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    history = []; best = None; start = time.perf_counter()
    for epoch in range(7):
        tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); online = 0.
        if epoch:
            model.train(); fusion.train()
            for first in range(0, base.NFIT, 8):
                chosen = order[epoch - 1, first:first + 8]; optimizer.zero_grad(set_to_none=True)
                for i in chosen:
                    lb, nll = feature_row(signal, bounds, i, mode)
                    z, _ = bridge.forward_fused(model, fusion, rows[i], lb, nll, 'cuda')
                    lo, hi = weights['bounds'][i]
                    y = torch.as_tensor(weights['y'][lo:hi], dtype=torch.float32, device='cuda')
                    w = torch.as_tensor(weights['loss'][lo:hi], dtype=torch.float32, device='cuda')
                    objective = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() * base.NFIT / (len(chosen) * mass)
                    assert torch.isfinite(objective); objective.backward(); online += float(objective.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.); assert torch.isfinite(norm)
                optimizer.step()
                if (first + len(chosen)) % 200 == 0: print('FUSION_TRAIN', mode, epoch, first + len(chosen), 3680, round(time.perf_counter() - tick, 1), flush=True)
        model.eval(); fusion.eval(); probabilities = {}; fit_bce = 0.
        with torch.no_grad():
            for i, row in enumerate(rows):
                lb, nll = feature_row(signal, bounds, i, mode)
                z, _ = bridge.forward_fused(model, fusion, row, lb, nll, 'cuda')
                assert torch.isfinite(z).all()
                probabilities[row['response_id']] = torch.sigmoid(z).cpu().numpy()
                if i < base.NFIT:
                    lo, hi = weights['bounds'][i]
                    y = torch.as_tensor(weights['y'][lo:hi], dtype=torch.float32, device='cuda')
                    w = torch.as_tensor(weights['loss'][lo:hi], dtype=torch.float32, device='cuda')
                    fit_bce += float((F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).double().sum())
                if (i + 1) % 400 == 0: print('FUSION_EVAL', mode, epoch, i + 1, 3839, flush=True)
        fit = evaluation.score_geometry(answers, tokens, probabilities, range(base.NFIT))
        cal = evaluation.score_geometry(answers, tokens, probabilities, range(base.NFIT, len(rows)))
        assert len(fit['window_scores']) == 653979 and len(cal['window_scores']) == 42241
        thresholds = {'window': q.choose_threshold(cal['window_labels'], cal['window_scores']),
            'answer': q.choose_threshold(cal['answer_labels'], cal['answer_scores'])}
        key = [min(thresholds['window']['f1'], thresholds['answer']['f1']), thresholds['window']['f1'], thresholds['window']['precision'], -epoch]
        stem = f'epoch_{epoch:02d}'
        state = {'model_state_dict': evaluation.cpu_state(model.state_dict()),
            'fusion_state_dict': evaluation.cpu_state(fusion.state_dict()),
            'optimizer_state_dict': evaluation.cpu_state(optimizer.state_dict()), 'epoch': epoch, 'mode': mode,
            'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state().cpu(),
            'preparation_sha256': q.sha(OUT / 'preparation_complete.json')}
        torch.save(state, directory / (stem + '.pt')); del state
        np.savez_compressed(directory / (stem + '_token_predictions.npz'), **probabilities)
        np.savez_compressed(directory / (stem + '_scores.npz'), **{'fit_' + k: v for k, v in fit.items()}, **{'cal_' + k: v for k, v in cal.items()})
        entry = {'epoch': epoch, 'mode': mode, 'thresholds': thresholds, 'selection_key': key,
            'fit_at_cal_thresholds': evaluation.metrics_for(fit, thresholds), 'calibration': evaluation.metrics_for(cal, thresholds),
            'fit_weighted_bce': fit_bce / mass, 'online_objective_sum': online, 'optimizer_steps': 460 if epoch else 0,
            'seconds': time.perf_counter() - tick, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'artifacts_sha256': {ext: q.sha(directory / (stem + ext)) for ext in ('.pt', '_token_predictions.npz', '_scores.npz')},
            'official_test_opened': False}
        q.save(directory / (stem + '.json'), entry); history.append(entry)
        if epoch and (best is None or key > best['selection_key']): best = entry
        print('FUSION_EPOCH_COMPLETE', mode, epoch, entry['calibration']['windows']['f1'], entry['calibration']['answers']['f1'], flush=True)
    q.save(directory / 'complete.json', {'status': 'complete_development_only', 'mode': mode, 'selected': best,
        'all_epochs': history, 'seconds': time.perf_counter() - start, 'official_test_opened': False})
    del model, fusion, optimizer; torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('design', 'cpu-check', 'prepare', 'gpu-smoke', 'train'))
    parser.add_argument('--mode', choices=MODES, default='fusion')
    args = parser.parse_args()
    try:
        if args.action == 'train': train(args.mode)
        else: {'design': design, 'cpu-check': cpu_check, 'prepare': prepare, 'gpu-smoke': gpu_smoke}[args.action]()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        q.save(OUT / f'FAILURE_{args.action}_{time.time_ns()}.json', {'action': args.action, 'mode': args.mode,
            'error_type': type(exc).__name__, 'error': str(exc), 'official_test_opened': False})
        raise


if __name__ == '__main__':
    main()
