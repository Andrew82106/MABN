"""QA-only ModernBERT plus aligned Llama residuals; separate frozen controls.

CPU design/checks are safe before the new replay is complete. prepare requires
all 3046 new caches. gpu-smoke/train require separate root GPU scheduling.
"""
from pathlib import Path
import argparse
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
import generation_residual_fusion as architecture

ROOT = q.ROOT
OUT = ROOT / 'results/full_context_generation_fusion_v1'
NEW = ROOT / 'fit_expansion/llama_features_v3'
MODES = ('fusion', 'semantic_only')
FUSION_SEED = 20261006


def protocol():
    return {'version': 'qa-full-context-generation-residual-v1',
        'scope': 'Same frozen3680 QA fit and159 calibration, complete input, human labels and4rawBPE metrics as full_context_encoder_v2. No auxiliary9678 and no official test.',
        'controls': {'fusion': 'Actual aligned legacy LB1024 and NLL1.',
            'semantic_only': 'Identical residual architecture and budget, but LB and NLL inputs are identically zero. Extra learned semantic capacity control.'},
        'signals': 'Original793 and new3046 uniform Llama-2-7B-chat NF4 teacher-forced replay, source-only post-read legacy LB, selected token NLL from previous position. Same exact original full-string BPE IDs and offsets, no new tokenization.',
        'representation': 'One unchanged ModernBERT token-classifier forward. Map actual two-logit difference and final768 encoder state independently onto original raw BPE using existing nonwhitespace character map. FP32 mapping and fusion.',
        'residual': 'Semantic LayerNorm768→32 SiLU, Lookback LayerNorm1024→24 SiLU, log1p(NLL)→8 SiLU; concatenate semantic32+generation32+baseline_logit1, sigmoid65→1 gate times65→1 residual. Residual weight/bias initialized exactly zero.',
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
        'limits': ['Adds a fully trained semantic checker, not a pure original-generator white-box probe.',
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


def feature_path(rid, old_ids):
    return (ROOT / 'data/features' if rid in old_ids else NEW / 'features') / (rid + '.npz')


def validate_feature(path, token, manifest_record, old):
    meta = q.read(path.with_suffix('.json'))
    assert meta['complete'] and meta['response_id'] == token['response_id']
    expected = meta['npz_sha256'] if old else meta['arrays_sha256']
    assert q.sha(path) == expected == manifest_record['npz_sha256']
    with np.load(path, allow_pickle=False) as z:
        for key, goldkey in [('token_ids', 'token_ids'), ('answer_token_positions', 'answer_token_positions'),
                             ('response_token_offsets', 'response_token_offsets'), ('response_token_offsets_raw', 'response_token_offsets_raw')]:
            assert np.array_equal(z[key], np.asarray(token[goldkey])), (token['response_id'], key)
        lb, nll = z['lb'], z['nll']
    count = token['token_count']
    assert lb.shape == (count, 1024) and nll.shape == (count,)
    assert lb.dtype == nll.dtype == np.float32
    assert np.isfinite(lb).all() and np.isfinite(nll).all()
    assert ((lb >= 0) & (lb <= 1)).all() and (nll >= 0).all()
    return lb, nll, {'response_id': token['response_id'], 'path': str(path.resolve()),
        'npz_sha256': expected, 'sidecar_sha256': q.sha(path.with_suffix('.json')), 'raw_tokens': count}


def source_files():
    paths = [Path(__file__), Path(bridge.__file__), Path(architecture.__file__), Path(base.__file__),
        base.OUT / 'preparation_complete.json', base.OUT / 'protocol.json', base.OUT / 'inputs.jsonl',
        base.OUT / 'training_weights.npz', base.OUT / 'answer_orders.npy',
        ROOT / 'data/feature_manifest.json', NEW / 'feature_manifest.json',
        ROOT / 'data/feature_signature.json', NEW / 'signature.json', OUT / 'protocol.json']
    return {str(p.resolve()): q.sha(p) for p in paths}


def prepare():
    design()
    if not (NEW / 'feature_manifest.json').exists():
        print('WAIT_FULL3046_FEATURE_MANIFEST_NO_FIT', flush=True)
        return
    new = q.read(NEW / 'feature_manifest.json')
    assert new['complete'] and new['completed_count'] == 3046
    assert not new['labels_used'] and not new['test_read']
    base.check_prepared()
    assert not (OUT / 'prepare_started.json').exists(), 'No implicit overwrite/resume'
    old = q.read(ROOT / 'data/feature_manifest.json')
    assert old['complete'] and old['completed_records'] == 793
    records = {r['response_id']: r for r in old['records']}
    old_ids = set(records)
    assert not old_ids & {r['response_id'] for r in new['records']}
    records.update({r['response_id']: r for r in new['records']})
    answers, tokens, _ = mapping.metadata()
    assert len(records) == len(answers) == 3839
    assert set(records) == {a['response_id'] for a in answers}
    snapshot = source_files()
    q.save(OUT / 'prepare_started.json', {'time': time.time(), 'source_sha256': snapshot})
    for name in ('inputs.jsonl', 'training_weights.npz', 'answer_orders.npy'):
        shutil.copyfile(base.OUT / name, OUT / name)
        assert q.sha(base.OUT / name) == q.sha(OUT / name)
    total = sum(t['token_count'] for t in tokens)
    assert total == 708506
    signal = np.lib.format.open_memmap(OUT / 'generation.npy', mode='w+', dtype=np.float32, shape=(total, 1025))
    bounds, audit, cursor = [], [], 0
    for index, (answer, token) in enumerate(zip(answers, tokens)):
        rid = answer['response_id']
        assert token['response_id'] == rid
        lb, nll, check = validate_feature(feature_path(rid, old_ids), token, records[rid], rid in old_ids)
        end = cursor + len(nll)
        signal[cursor:end, :1024] = lb
        signal[cursor:end, 1024] = nll
        bounds.append([cursor, end]); audit.append(check); cursor = end
        if (index + 1) % 200 == 0: print('FUSION_PREPARE', index + 1, 3839, flush=True)
    signal.flush(); del signal
    np.save(OUT / 'generation_bounds.npy', np.asarray(bounds, np.int64))
    q.save(OUT / 'generation_manifest.json', {'records': audit, 'answers': 3839, 'original_raw_tokens': total,
        'generation_signals_only_no_gold_inputs': True, 'exact_raw_ids_and_offsets': True, 'official_test_opened': False})
    assert snapshot == source_files()
    q.save(OUT / 'source_snapshot.json', {'files_sha256': snapshot})
    names = ('inputs.jsonl', 'training_weights.npz', 'answer_orders.npy', 'generation.npy',
        'generation_bounds.npy', 'generation_manifest.json', 'protocol.json', 'source_snapshot.json')
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_trained', 'answers': 3839,
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'GPU_used': False, 'official_test_opened': False})
    print('FUSION_ALL3839_PREPARED', flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['answers'] == 3839 and q.read(OUT / 'protocol.json') == protocol()
    for name, digest in p['files_sha256'].items(): assert q.sha(OUT / name) == digest, name
    assert q.read(OUT / 'source_snapshot.json')['files_sha256'] == source_files()
    base.check_prepared()
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
    actual, original = bridge.forward_fused(model, fusion, row, lb, nll, 'cpu')
    assert torch.equal(expected, original) and torch.equal(actual, expected)
    F.binary_cross_entropy_with_logits(actual, torch.tensor([0., 1., 0., 1.])).backward()
    assert fusion.residual.weight.grad.abs().sum() > 0
    assert model.model.embeddings.tok_embeddings.weight.grad.abs().sum() > 0
    old = q.read(ROOT / 'data/feature_manifest.json')
    old_records = {r['response_id']: r for r in old['records']}
    _, tokens, _ = mapping.metadata()
    old_tokens = [t for t in tokens if t['response_id'] in old_records]
    chosen = [old_tokens[0], old_tokens[633], old_tokens[634], old_tokens[-1]]
    anchors = []
    for token in chosen:
        _, _, record = validate_feature(ROOT / 'data/features' / (token['response_id'] + '.npz'), token, old_records[token['response_id']], True)
        anchors.append(record)
    freeze(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'initial_fusion_preserves_cpu_rng': True,
        'dropout_path_baseline_and_zero_residual_exact': True, 'real_tiny_modernbert_backward': True,
        'old_fit_cal_coordinate_anchors': anchors, 'GPU_used': False, 'actual_training_performed': False,
        'source_sha256': q.sha(Path(__file__)), 'bridge_sha256': q.sha(Path(bridge.__file__))})
    print('FUSION_CPU_CHECK_PASSED', flush=True)


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
