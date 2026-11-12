"""Freeze and replay only the existing R16 train scene, on original Qwen BPEs."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT.parent / 'benchmark_ragtruth_qa'
sys.path.insert(0, str(QA / 'src'))
import run_development as q
import run_full_context_encoder_v2 as base
import modernbert_generation_bridge as bridge
import run_control_scene_transfer as transfer

OUT = ROOT / 'data'
SOURCE = QA / 'results/full_context_aux_transfer_v1/transfer'
SCENE = transfer.OUT
ATOL = 2e-6


def selected():
    done = q.read(SOURCE / 'complete.json')
    assert not done['official_test_opened']
    assert [e['epoch'] for e in done['all_QA_epochs']] == [1, 2, 3]
    entry = done['selected']
    assert entry['epoch'] == 2
    path = SOURCE / 'qa_epoch_02.pt'
    assert q.sha(path) == entry['artifacts_sha256']['.pt']
    return entry, path


def source_hashes():
    _, checkpoint = selected()
    names = [Path(__file__), Path(base.__file__), Path(bridge.__file__), Path(transfer.__file__),
        Path(bridge.mapping.__file__), Path(q.__file__), checkpoint, SOURCE / 'complete.json',
        SCENE / 'preparation_complete.json', SCENE / 'auxiliary_transfer/inference_complete.json',
        SCENE / 'auxiliary_transfer/token_predictions.npz']
    names += [SCENE / n for n in ('inputs.jsonl', 'windows.jsonl', 'answers.jsonl', 'folds.json')]
    return {str(p.resolve()): q.sha(p) for p in names}


def freeze(path, value):
    if path.exists():
        assert q.read(path) == value, str(path)
    else:
        q.save(path, value)


def prepare():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    transfer.check_prepared()
    entry, checkpoint = selected()
    old = q.read(SCENE / 'auxiliary_transfer/inference_complete.json')
    assert old['source_freeze']['checkpoint_sha256'] == q.sha(checkpoint)
    assert q.sha(SCENE / 'auxiliary_transfer/token_predictions.npz') == old['token_predictions_sha256']
    rows = q.lines(SCENE / 'inputs.jsonl')
    windows, answers = q.lines(SCENE / 'windows.jsonl'), q.lines(SCENE / 'answers.jsonl')
    assert len(rows) == len(answers) == 602 and len(windows) == 12222
    assert {r['response_id'] for r in rows} == {a['row_id'] for a in answers}
    sizes = np.asarray([r['raw_token_count'] for r in rows], np.int64)
    ends = np.cumsum(sizes); bounds = np.column_stack((ends - sizes, ends))
    assert ends[-1] == 14968
    row_index = {r['response_id']: i for i, r in enumerate(rows)}
    for row in rows:
        assert row['raw_token_count'] == len(row['raw_token_ids']) == len(row['raw_token_offsets'])
        ix, columns, weight = map(np.asarray, row['mapping'])
        assert len(ix) == len(columns) == len(weight)
        assert (ix >= 0).all() and (ix < row['raw_token_count']).all()
        assert (columns >= 0).all() and (columns < len(row['input_ids'])).all()
        assert (weight > 0).all() and np.isfinite(weight).all()
        mass = np.bincount(ix.astype(int), weights=weight, minlength=row['raw_token_count'])
        assert np.allclose(mass[mass > 0], 1., rtol=0, atol=1e-6)
    for w in windows:
        row = rows[row_index[w['row_id']]]
        assert w['lexical_token_indices'] and set(w['lexical_token_indices']) <= set(w['raw_token_indices'])
        assert all(0 <= j < row['raw_token_count'] for j in w['raw_token_indices'])
    # Test the existing mapping primitive on overlapping fractional positions.
    h = torch.tensor([[1., 3.], [5., 7.]])
    mapped = bridge.map_hidden(h, [[0, 0, 1], [0, 1, 1], [.5, .5, 1.]], 3)
    assert torch.equal(mapped, torch.tensor([[3., 5.], [5., 7.], [0., 0.]]))
    protocol = {'version': 'r28-semantic-scene-cache-v1', 'answers': 602, 'raw_Qwen_BPE': 14968,
        'checkpoint': 'Fixed auxiliary->QA2 selected solely by prior QA calibration; never reselect on R16.',
        'input': 'Existing unchanged control_scene_transfer_v1 input_ids/mapping/raw Qwen IDs and offsets; all only R16 original train.',
        'features': 'One unchanged BF16 ModernBERT eval forward per answer; actual classifier z1-z0 and last hidden mapped independently in FP32. Whole-answer offline evidence checker, not native Qwen states.',
        'cache': {'hidden.npy': [14968, 768], 'logit.npy': [14968], 'gpu_probability.npy': [14968], 'bounds.npy': [602, 2]},
        'numerics': {'old_probability_absolute_tolerance': ATOL, 'relative_tolerance': 0,
            'old_logit_available': False, 'old_logit_not_reconstructed_from_probability': True,
            'repeat_first_and_last_same_ids_logit_and_hidden_absolute_tolerance': ATOL,
            'all_nonexact_differences_and_original_window_answer_scores_recorded': True},
        'human_gold': False, 'original_validation_test_opened': False, 'new_training': False,
        'historical_limit': 'No old R16 validation/test reads in THIS run; those old splits were exposed in prior rounds. This is not a fresh test.'}
    freeze(OUT / 'protocol.json', protocol)
    if (OUT / 'bounds.npy').exists():
        assert np.array_equal(np.load(OUT / 'bounds.npy'), bounds)
    else:
        np.save(OUT / 'bounds.npy', bounds)
    check = {'passed': True, 'answers': 602, 'raw_tokens': 14968, 'map_indices_and_row_ids_verified': True,
        'mapped_nonempty_token_weights_sum_one': True, 'fractional_overlap_toy_passed': True,
        'no_model_loaded': True, 'GPU_used': False, 'source_sha256': source_hashes(),
        'protocol_sha256': q.sha(OUT / 'protocol.json'), 'bounds_sha256': q.sha(OUT / 'bounds.npy')}
    freeze(OUT / 'preparation_complete.json', check)
    print('R28_CPU_PREPARE_PASSED', len(rows), int(ends[-1]), flush=True)
    return rows, bounds


def difference(a, b):
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return {'count': x.size, 'nonexact': int(np.count_nonzero(x != y)),
        'max_abs': float(diff.max(initial=0)), 'mean_abs': float(diff.mean()) if diff.size else 0.}


def infer():
    rows, bounds = prepare()
    assert not (OUT / 'started.json').exists(), 'No overwrite or implicit resume'
    q.save(OUT / 'started.json', {'time': time.time(), 'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    entry, checkpoint = selected()
    base.configure_gpu()
    model = base.load_model()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(state['model_state_dict'], strict=True); del state
    model.requires_grad_(False).eval().cuda()
    hidden = np.lib.format.open_memmap(OUT / 'hidden.npy', mode='w+', dtype=np.float32, shape=(14968, 768))
    logits = np.lib.format.open_memmap(OUT / 'logit.npy', mode='w+', dtype=np.float32, shape=(14968,))
    probs = np.lib.format.open_memmap(OUT / 'gpu_probability.npy', mode='w+', dtype=np.float32, shape=(14968,))
    original = np.load(SCENE / 'auxiliary_transfer/token_predictions.npz', allow_pickle=False)
    assert set(original.files) == {r['response_id'] for r in rows}
    deltas, repeats, actual, old_prob = [], [], {}, {}
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.no_grad():
        for i, (row, (lo, hi)) in enumerate(zip(rows, bounds)):
            z, h = bridge.forward_aligned(model, row, 'cuda')
            assert h.shape == (hi - lo, 768) and torch.isfinite(h).all() and torch.isfinite(z).all()
            p = torch.sigmoid(z).cpu().numpy()
            hidden[lo:hi] = h.cpu().numpy(); logits[lo:hi] = z.cpu().numpy(); probs[lo:hi] = p
            actual[row['response_id']] = p; old_prob[row['response_id']] = original[row['response_id']]
            deltas.append(dict(response_id=row['response_id'], **difference(p, old_prob[row['response_id']])))
            if i in (0, len(rows)-1):
                zz, hh = bridge.forward_aligned(model, row, 'cuda')
                repeats.append({'response_id': row['response_id'], 'logit': difference(z.cpu().numpy(), zz.cpu().numpy()),
                    'hidden': difference(h.cpu().numpy(), hh.cpu().numpy())})
            if (i + 1) % 100 == 0:
                print('R28_CACHE', i + 1, len(rows), flush=True)
    torch.cuda.synchronize(); seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    original.close(); hidden.flush(); logits.flush(); probs.flush()
    del hidden, logits, probs, model, z, h, zz, hh
    torch.cuda.empty_cache()
    windows, answers = q.lines(SCENE / 'windows.jsonl'), q.lines(SCENE / 'answers.jsonl')
    neww, newa = transfer.aggregate(actual, windows, answers)
    oldw, olda = transfer.aggregate(old_prob, windows, answers)
    report = {'row_probability_differences': deltas, 'repeated_anchors': repeats,
        'window_score_difference': difference(neww, oldw), 'answer_score_difference': difference(newa, olda),
        'old_logit_available': False, 'original_logit_equality_claimed': False,
        'old_probability_max_abs': max(r['max_abs'] for r in deltas),
        'seconds': seconds, 'peak_cuda_allocated_bytes': peak, 'human_gold': False,
        'labels_used_for_extraction': False, 'original_validation_test_opened': False}
    q.save(OUT / 'REPLAY.json', report)
    assert report['old_probability_max_abs'] <= ATOL
    assert max(r[k]['max_abs'] for r in repeats for k in ('hidden', 'logit')) <= ATOL
    assert q.read(OUT / 'preparation_complete.json')['source_sha256'] == source_hashes()
    names = ('hidden.npy', 'logit.npy', 'gpu_probability.npy', 'bounds.npy', 'REPLAY.json', 'protocol.json', 'preparation_complete.json')
    q.save(OUT / 'complete.json', {'status': 'complete', 'answers': 602, 'raw_tokens': 14968,
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'seconds': seconds,
        'human_gold': False, 'GPU_model_released': True, 'original_validation_test_opened': False})
    print('R28_CACHE_COMPLETE_GPU_RELEASED', flush=True)


def check_complete():
    done = q.read(OUT / 'complete.json')
    assert done['status'] == 'complete' and done['raw_tokens'] == 14968
    for n, expected in done['files_sha256'].items():
        assert q.sha(OUT / n) == expected
    assert q.read(OUT / 'preparation_complete.json')['source_sha256'] == source_hashes()
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'infer', 'check'))
    stage = parser.parse_args().stage
    try:
        {'prepare': prepare, 'infer': infer, 'check': check_complete}[stage]()
    except BaseException as exc:
        q.save(OUT / f'FAILURE_{stage}_{time.time_ns()}.json', {'error': repr(exc), 'original_validation_test_opened': False})
        raise
