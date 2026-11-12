"""R31 large final-checkpoint cache. CPU prepare never reads live epochs.

The infer command is an explicit GPU operation; root must separately schedule it.
No original heldout data and no gold-dependent feature construction are used.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT.parent / 'benchmark_ragtruth_qa'
sys.path.insert(0, str(QA / 'src'))
import run_development as q
import modernbert_generation_bridge as bridge
import run_control_scene_transfer as transfer

# Private namespace: never call large.bind() or alter the old runner's globals.
BASE_PATH = QA / 'src/run_full_context_encoder_v2.py'
spec = importlib.util.spec_from_file_location('_r31_private_large_inference', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.MODEL = QA.parent / 'models/ModernBERT-large'
MODEL = base.MODEL
LARGE = QA / 'results/full_context_encoder_large_v1'
SOURCE = LARGE / 'full_finetune'
SCENE = transfer.OUT
OUT = ROOT / 'data'
ATOL = 2e-6


def freeze(path, value):
    if path.exists():
        assert q.read(path) == value, str(path)
    else:
        q.save(path, value)


def source_hashes():
    paths = [Path(__file__), BASE_PATH, Path(bridge.__file__), Path(bridge.mapping.__file__),
        Path(transfer.__file__), Path(q.__file__), QA / 'src/run_full_context_encoder_large.py',
        LARGE / 'preparation_complete.json', LARGE / 'protocol.json',
        SCENE / 'preparation_complete.json']
    paths += [SCENE / n for n in ('inputs.jsonl', 'windows.jsonl', 'answers.jsonl', 'folds.json')]
    paths += [MODEL / n for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                                  'special_tokens_map.json', 'download_manifest.json')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def protocol():
    return {'version': 'r31-large-cache-v1', 'answers': 602, 'raw_Qwen_BPE': 14968,
        'selection': 'Require completed upstream epochs0..6; use its single selected nonzero epoch unchanged. No R16 selection. No unfinished upstream scores read by prepare.',
        'input': 'Unchanged original R16-train control_scene_transfer input IDs, full visible evidence/question/answer, original Qwen BPE offsets and character map. ModernBERT base and large tokenizer files must be byte identical.',
        'features': 'One offline full-answer BF16/eval forward, FP32 parameters; map actual z1-z0 and final hidden1024 independently in FP32. Not native generator states.',
        'cache': {'hidden.npy': [14968, 1024], 'logit.npy': [14968],
                  'gpu_probability.npy': [14968], 'bounds.npy': [602, 2]},
        'numerics': {'absolute_tolerance': ATOL, 'relative_tolerance': 0,
            'anchors': 'First and last fixed scene rows: compare hook logits to original no-hook logits path; repeat hidden/logit. Additionally replay first original QA fit row against its selected saved raw-BPE probability.',
            'prior_large_R16_cache_available': False,
            'no_claim_of_equality_to_auxiliary_R28_cache': True},
        'GPU': 'Explicit infer only after root authorization and prior GPU process actual exit. Complete manifest alone is not GPU authorization.',
        'human_gold': False, 'original_validation_test_opened': False,
        'official_QA_test_opened': False, 'new_training': False}


def prepare():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    transfer.check_prepared()
    large_prepared = q.read(LARGE / 'preparation_complete.json')
    assert large_prepared['answers'] == 3839 and large_prepared['official_test_opened'] is False
    for name, expected in large_prepared['files_sha256'].items():
        assert q.sha(LARGE / name) == expected, name
    config = q.read(MODEL / 'config.json')
    assert config['hidden_size'] == 1024 and config['num_hidden_layers'] == 28
    for n in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        assert q.sha(MODEL / n) == q.sha(QA.parent / 'models/ModernBERT-base' / n), n
    rows = q.lines(SCENE / 'inputs.jsonl')
    windows, answers = q.lines(SCENE / 'windows.jsonl'), q.lines(SCENE / 'answers.jsonl')
    assert len(rows) == len(answers) == 602 and len(windows) == 12222
    assert {r['response_id'] for r in rows} == {a['row_id'] for a in answers}
    sizes = np.asarray([r['raw_token_count'] for r in rows], np.int64)
    ends = np.cumsum(sizes); bounds = np.column_stack((ends-sizes, ends))
    assert ends[-1] == 14968
    by_id = {r['response_id']: r for r in rows}
    for r in rows:
        assert r['raw_token_count'] == len(r['raw_token_ids']) == len(r['raw_token_offsets'])
        ix, cols, weight = map(np.asarray, r['mapping'])
        assert len(ix) == len(cols) == len(weight)
        assert (ix >= 0).all() and (ix < r['raw_token_count']).all()
        assert (cols >= 0).all() and (cols < len(r['input_ids'])).all()
        assert (weight > 0).all() and np.isfinite(weight).all()
        mass = np.bincount(ix.astype(int), weights=weight, minlength=r['raw_token_count'])
        assert np.allclose(mass[mass > 0], 1., rtol=0, atol=1e-6)
        assert len(r['input_ids']) <= config['max_position_embeddings']
    for w in windows:
        assert w['lexical_token_indices'] and set(w['lexical_token_indices']) <= set(w['raw_token_indices'])
        assert all(0 <= i < by_id[w['row_id']]['raw_token_count'] for i in w['raw_token_indices'])
    h = torch.tensor([[1., 3.], [5., 7.]])
    assert torch.equal(bridge.map_hidden(h, [[0, 0, 1], [0, 1, 1], [.5, .5, 1.]], 3),
                       torch.tensor([[3., 5.], [5., 7.], [0., 0.]]))
    freeze(OUT / 'protocol.json', protocol())
    if (OUT / 'bounds.npy').exists(): assert np.array_equal(np.load(OUT / 'bounds.npy'), bounds)
    else: np.save(OUT / 'bounds.npy', bounds)
    freeze(OUT / 'preparation_complete.json', {'passed': True, 'answers': 602, 'raw_tokens': 14968,
        'tokenizer_files_exact': True, 'coordinates_unchanged': True, 'fractional_mapping_toy_passed': True,
        'source_sha256': source_hashes(), 'protocol_sha256': q.sha(OUT / 'protocol.json'),
        'bounds_sha256': q.sha(OUT / 'bounds.npy'), 'large_live_epochs_or_predictions_read': False,
        'model_loaded': False, 'GPU_used': False})
    print('R31_CACHE_CPU_PREPARED_602_NO_MODEL_NO_EPOCH_READ', flush=True)
    return rows, bounds


def selected():
    done = q.read(SOURCE / 'complete.json')
    assert done['status'] == 'complete_development_only' and done['official_test_opened'] is False
    assert [e['epoch'] for e in done['all_epochs']] == list(range(7))
    entry = done['selected']
    assert entry == max(done['all_epochs'][1:], key=lambda e: e['selection_key'])
    checkpoint = SOURCE / f"epoch_{entry['epoch']:02d}.pt"
    predictions = SOURCE / f"epoch_{entry['epoch']:02d}_token_predictions.npz"
    assert q.sha(checkpoint) == entry['artifacts_sha256']['.pt']
    assert q.sha(predictions) == entry['artifacts_sha256']['_token_predictions.npz']
    return entry, checkpoint, predictions


def difference(a, b):
    x, y = np.asarray(a), np.asarray(b); assert x.shape == y.shape
    d = np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {'n': x.size, 'nonexact': int(np.count_nonzero(x != y)), 'max_abs': float(d.max(initial=0))}


def infer():
    # Gate before loading a checkpoint/model or initializing CUDA.
    if not (SOURCE / 'complete.json').exists():
        print('WAIT_LARGE_ALL_SIX_EPOCHS_NO_GPU', flush=True); return
    rows, bounds = prepare()
    entry, checkpoint, saved_path = selected()
    assert not (OUT / 'started.json').exists(), 'No overwrite/implicit resume'
    binding = {'complete_sha256': q.sha(SOURCE / 'complete.json'),
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': q.sha(checkpoint),
        'selected_probability_sha256': q.sha(saved_path), 'epoch': entry['epoch'],
        'selected_by_R16': False, 'preparation_sha256': q.sha(OUT / 'preparation_complete.json')}
    freeze(OUT / 'selected_binding.json', binding)
    q.save(OUT / 'started.json', {'time': time.time(), 'selected_binding': binding})
    base.configure_gpu(); model = base.load_model()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert state['epoch'] == entry['epoch']
    assert state['preparation_sha256'] == q.sha(LARGE / 'preparation_complete.json')
    model.load_state_dict(state['model_state_dict'], strict=True); del state
    assert model.config.hidden_size == 1024
    model.requires_grad_(False).eval().cuda()
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    repeats = []
    with torch.no_grad():
        # Fixed first QA training row, never selected by gold or model performance.
        with (LARGE / 'inputs.jsonl').open(encoding='utf-8') as stream:
            import json
            anchor = json.loads(next(stream))
        assert anchor['partition'] == 'fit'
        za, ha = bridge.forward_aligned(model, anchor, 'cuda')
        with np.load(saved_path, allow_pickle=False) as saved:
            qa_replay = difference(torch.sigmoid(za).cpu().numpy(), saved[anchor['response_id']])
        qa_replay['response_id'] = anchor['response_id']
        q.save(OUT / 'QA_ANCHOR.json', qa_replay)
        assert qa_replay['max_abs'] <= ATOL
        # Check no-hook classifier and repeat path before committing all602.
        for row in (rows[0], rows[-1]):
            z, h = bridge.forward_aligned(model, row, 'cuda')
            direct = base.logits(model, row, 'cuda')
            zz, hh = bridge.forward_aligned(model, row, 'cuda')
            rec = {'response_id': row['response_id'], 'direct_logit': difference(z.cpu().numpy(), direct.cpu().numpy()),
                   'repeat_logit': difference(z.cpu().numpy(), zz.cpu().numpy()),
                   'repeat_hidden': difference(h.cpu().numpy(), hh.cpu().numpy())}
            repeats.append(rec)
        q.save(OUT / 'SCENE_ANCHORS.json', repeats)
        assert max(r[k]['max_abs'] for r in repeats for k in ('direct_logit', 'repeat_logit', 'repeat_hidden')) <= ATOL
        hidden = np.lib.format.open_memmap(OUT / 'hidden.npy', mode='w+', dtype=np.float32, shape=(14968, 1024))
        logits = np.lib.format.open_memmap(OUT / 'logit.npy', mode='w+', dtype=np.float32, shape=(14968,))
        probs = np.lib.format.open_memmap(OUT / 'gpu_probability.npy', mode='w+', dtype=np.float32, shape=(14968,))
        for i, (row, (lo, hi)) in enumerate(zip(rows, bounds)):
            z, h = bridge.forward_aligned(model, row, 'cuda')
            assert h.shape == (hi-lo, 1024) and z.shape == (hi-lo,)
            assert torch.isfinite(h).all() and torch.isfinite(z).all()
            hidden[lo:hi] = h.cpu().numpy(); logits[lo:hi] = z.cpu().numpy()
            probs[lo:hi] = torch.sigmoid(z).cpu().numpy()
            if (i+1) % 100 == 0: print('R31_CACHE', i+1, 602, flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    hidden.flush(); logits.flush(); probs.flush()
    del hidden, logits, probs, model, z, h, za, ha, zz, hh, direct
    torch.cuda.empty_cache()
    assert q.read(OUT / 'preparation_complete.json')['source_sha256'] == source_hashes()
    assert binding['complete_sha256'] == q.sha(SOURCE / 'complete.json')
    assert binding['checkpoint_sha256'] == q.sha(checkpoint)
    names = ('hidden.npy', 'logit.npy', 'gpu_probability.npy', 'bounds.npy', 'protocol.json',
        'preparation_complete.json', 'selected_binding.json', 'QA_ANCHOR.json', 'SCENE_ANCHORS.json')
    q.save(OUT / 'complete.json', {'status': 'complete', 'answers': 602, 'raw_tokens': 14968,
        'files_sha256': {n: q.sha(OUT / n) for n in names}, 'seconds': elapsed,
        'peak_cuda_allocated_bytes': peak, 'human_gold': False, 'GPU_model_released': True,
        'labels_used_for_features': False, 'original_validation_test_opened': False})
    print('R31_LARGE_CACHE_COMPLETE_GPU_RELEASED', flush=True)


def check_complete():
    done = q.read(OUT / 'complete.json')
    assert done['status'] == 'complete' and done['raw_tokens'] == 14968 and done['answers'] == 602
    for n, expected in done['files_sha256'].items(): assert q.sha(OUT / n) == expected
    p = q.read(OUT / 'preparation_complete.json')
    assert p['source_sha256'] == source_hashes()
    _, checkpoint, saved = selected()
    b = q.read(OUT / 'selected_binding.json')
    assert b['checkpoint_sha256'] == q.sha(checkpoint) and b['selected_probability_sha256'] == q.sha(saved)
    assert b['complete_sha256'] == q.sha(SOURCE / 'complete.json')
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'infer', 'check'))
    stage = parser.parse_args().stage
    try: {'prepare': prepare, 'infer': infer, 'check': check_complete}[stage]()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        q.save(OUT / f'FAILURE_{stage}_{time.time_ns()}.json', {'error': repr(exc), 'original_validation_test_opened': False})
        raise
