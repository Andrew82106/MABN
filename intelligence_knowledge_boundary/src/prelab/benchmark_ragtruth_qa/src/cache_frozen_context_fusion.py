"""Cache one completed, selected semantic encoder; GPU only via explicit infer.

No fitting or test access. CPU prepare waits for all six upstream epochs.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import run_development as q
import run_full_context_encoder_v2 as base
import run_full_context_fusion_v2 as fusion
import modernbert_generation_bridge as bridge
import tail_finetune as mapping
import tail_finetune_all_docs as metrics

OUT = q.ROOT / 'results/full_context_frozen_cache_v1'
SOURCE = base.OUT / 'full_finetune'
PROBABILITY_ATOL = 2e-6


def design():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 'completed-full-context-frozen-cache-v1',
        'source': 'Only the final selected checkpoint after full_context_encoder_v2 completes epochs0..6. Never select an upstream epoch here.',
        'scope': 'Same3839 development answers:3680fit+159calibration. No official test.',
        'forward': 'One eval/no_grad forward_aligned per answer, original batch1 full input and exact character-to-original-BPE map. BF16 CUDA forward, TF32 disabled, FP32 hidden/logit mapping. Actual two-logit difference is mapped independently of hidden.',
        'arrays': {'hidden.npy': 'float32[708506,768]', 'baseline_logit.npy': 'float32[708506]',
            'baseline_gpu_probability.npy': 'float32[708506]', 'bounds.npy': 'int64[3839,2]'},
        'replay': 'Compare every original token GPU probability and both original4rawBPE/answer score vectors. Report exact differences and metrics at the original selected thresholds; do not change upstream selection.',
        'probability_numeric_gate': {'absolute_tolerance': PROBABILITY_ATOL, 'relative_tolerance': 0,
            'reason': 'Allow only small floating-point replay discrepancies, record all non-exact values and original-threshold count differences. Exceeding tolerance stops before cache complete.'},
        'cpu_followup': 'CPU sigmoid can differ from saved GPU probability; preserve actual CPU sigmoid, require zero-residual logit equality, record probability/count differences. No probability replacement or gradient workaround.',
        'no_backbone_training': True, 'upstream_calibration_reused_not_crossfit': True}
    fusion.freeze(OUT / 'protocol.json', protocol)
    print('FROZEN_CONTEXT_CACHE_DESIGN_READY', flush=True)


def selected():
    path = SOURCE / 'complete.json'
    if not path.exists():
        print('WAIT_FULL_SIX_EPOCH_COMPLETE_NO_MODEL_LOADED', flush=True)
        return None
    complete = q.read(path)
    assert not complete['official_test_opened']
    assert {e['epoch'] for e in complete['all_epochs']} == set(range(7))
    entry = complete['selected']; assert 1 <= entry['epoch'] <= 6
    stem = f"epoch_{entry['epoch']:02d}"
    paths = {ext: SOURCE / (stem + ext) for ext in ('.pt', '_token_predictions.npz', '_scores.npz')}
    for ext, p in paths.items():
        assert q.sha(p) == entry['artifacts_sha256'][ext]
    return entry, paths


def source_hashes(paths):
    files = [Path(__file__), Path(bridge.__file__), Path(base.__file__), Path(mapping.__file__),
             Path(metrics.__file__), Path(q.__file__), Path(fusion.__file__),
             SOURCE / 'complete.json', base.OUT / 'preparation_complete.json',
             fusion.OUT / 'preparation_complete.json', OUT / 'protocol.json']
    files += [fusion.OUT / n for n in ('inputs.jsonl', 'generation_bounds.npy')]
    files += list(paths.values())
    return {str(p.resolve()): q.sha(p) for p in files}


def prepare():
    design()
    chosen = selected()
    if chosen is None:
        return None
    assert not torch.cuda.is_initialized()
    fusion.check_prepared()
    entry, paths = chosen
    rows = q.lines(fusion.OUT / 'inputs.jsonl')
    bounds = np.load(fusion.OUT / 'generation_bounds.npy')
    assert len(rows) == 3839 and bounds.shape == (3839, 2) and bounds[-1, 1] == 708506
    report = {'status': 'prepared_not_extracted', 'selected_epoch': entry['epoch'],
              'source_sha256': source_hashes(paths), 'answers': 3839, 'raw_tokens': 708506,
              'GPU_used': False, 'official_test_opened': False}
    fusion.freeze(OUT / 'preparation_complete.json', report)
    print('FROZEN_CONTEXT_CACHE_PREPARED_NO_GPU', flush=True)
    return entry, paths, rows, bounds


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'n': a.size, 'nonexact': int(np.count_nonzero(a != b)),
            'max_absolute': float(d.max(initial=0)), 'mean_absolute': float(d.mean()) if a.size else 0.}


def infer():
    prepared = prepare()
    if prepared is None:
        return
    entry, paths, rows, bounds = prepared
    assert not (OUT / 'infer_started.json').exists(), 'No overwrite or implicit resume'
    q.save(OUT / 'infer_started.json', {'time': time.time(), 'selected_epoch': entry['epoch'],
           'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    base.configure_gpu()
    model = base.load_model()
    state = torch.load(paths['.pt'], map_location='cpu', weights_only=False, mmap=True)
    assert state['epoch'] == entry['epoch']
    model.load_state_dict(state['model_state_dict'], strict=True); del state
    model.requires_grad_(False).eval().cuda()
    assert not any(p.requires_grad for p in model.parameters())
    hcache = np.lib.format.open_memmap(OUT / 'hidden.npy', mode='w+', dtype=np.float32, shape=(708506, 768))
    zcache = np.lib.format.open_memmap(OUT / 'baseline_logit.npy', mode='w+', dtype=np.float32, shape=(708506,))
    pcache = np.lib.format.open_memmap(OUT / 'baseline_gpu_probability.npy', mode='w+', dtype=np.float32, shape=(708506,))
    reference = np.load(paths['_token_predictions.npz'], allow_pickle=False)
    assert set(reference.files) == {r['response_id'] for r in rows}
    row_differences = []; probabilities = {}
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.no_grad():
        for i, (row, (lo, hi)) in enumerate(zip(rows, bounds)):
            z, h = bridge.forward_aligned(model, row, 'cuda')
            p = torch.sigmoid(z).cpu().numpy()
            zcache[lo:hi] = z.cpu().numpy(); hcache[lo:hi] = h.cpu().numpy(); pcache[lo:hi] = p
            assert h.shape == (hi - lo, 768) and torch.isfinite(h).all() and torch.isfinite(z).all()
            original = reference[row['response_id']]
            row_differences.append(dict(response_id=row['response_id'], **difference(p, original)))
            probabilities[row['response_id']] = p
            if (i + 1) % 400 == 0:
                print('FROZEN_CONTEXT_CACHE', i + 1, 3839, round(time.perf_counter() - start, 1), flush=True)
    torch.cuda.synchronize(); seconds = time.perf_counter() - start
    reference.close(); hcache.flush(); zcache.flush(); pcache.flush()
    del hcache, zcache, pcache, z, h, model
    torch.cuda.empty_cache()
    print('FROZEN_CONTEXT_FORWARD_FINISHED_GPU_MODEL_RELEASED', seconds, flush=True)
    np.save(OUT / 'bounds.npy', bounds)
    answers, tokens, _ = mapping.metadata()
    fit = metrics.score_geometry(answers, tokens, probabilities, range(base.NFIT))
    cal = metrics.score_geometry(answers, tokens, probabilities, range(base.NFIT, len(rows)))
    comparisons = {}
    with np.load(paths['_scores.npz'], allow_pickle=False) as old:
        for part, result in (('fit', fit), ('cal', cal)):
            for key, value in result.items():
                if key.endswith('_labels'):
                    assert np.array_equal(value, old[part + '_' + key])
                else:
                    comparisons[part + '_' + key] = difference(value, old[part + '_' + key])
    counts = {'fit': metrics.metrics_for(fit, entry['thresholds']), 'calibration': metrics.metrics_for(cal, entry['thresholds'])}
    exact_counts = {'fit': counts['fit'] == entry['fit_at_cal_thresholds'],
                    'calibration': counts['calibration'] == entry['calibration']}
    peak = max(r['max_absolute'] for r in row_differences)
    report = {'selected_epoch': entry['epoch'], 'token_rows': row_differences,
        'token_max_absolute_difference': peak, 'token_nonexact': sum(r['nonexact'] for r in row_differences),
        'score_differences': comparisons, 'original_thresholds': entry['thresholds'],
        'metrics_at_original_thresholds': counts, 'upstream_original_metrics': {'fit': entry['fit_at_cal_thresholds'], 'calibration': entry['calibration']},
        'all_metrics_exact': exact_counts, 'seconds': seconds, 'backbone_parameters_updated': False,
        'official_test_opened': False}
    q.save(OUT / 'REPLAY.json', report)
    assert peak <= PROBABILITY_ATOL, 'Recorded probability replay exceeds frozen numerical tolerance'
    assert q.read(OUT / 'preparation_complete.json')['source_sha256'] == source_hashes(paths)
    names = ('hidden.npy', 'baseline_logit.npy', 'baseline_gpu_probability.npy', 'bounds.npy', 'REPLAY.json', 'protocol.json', 'preparation_complete.json')
    q.save(OUT / 'complete.json', {'status': 'complete', 'selected_epoch': entry['epoch'],
        'answers': 3839, 'raw_tokens': 708506, 'source_paths': {k: str(p) for k, p in paths.items()},
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'GPU_forward_seconds': seconds,
        'GPU_model_released': True, 'no_backbone_training': True, 'official_test_opened': False})
    print('FROZEN_CONTEXT_CACHE_COMPLETE', flush=True)


def check_complete():
    path = OUT / 'complete.json'
    if not path.exists():
        print('WAIT_FROZEN_CONTEXT_CACHE_COMPLETE', flush=True)
        return None
    done = q.read(path)
    assert done['answers'] == 3839 and done['raw_tokens'] == 708506 and not done['official_test_opened']
    for name, digest in done['files_sha256'].items():
        assert q.sha(OUT / name) == digest, name
    paths = {k: Path(v) for k, v in done['source_paths'].items()}
    assert q.read(OUT / 'preparation_complete.json')['source_sha256'] == source_hashes(paths)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('design', 'prepare', 'infer', 'check'))
    action = parser.parse_args().action
    try:
        {'design': design, 'prepare': prepare, 'infer': infer, 'check': check_complete}[action]()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        q.save(OUT / f'FAILURE_{action}_{time.time_ns()}.json', {'error': repr(exc), 'official_test_opened': False})
        raise
