"""Three material-group folds for honest fit-side Lookback/large stacking.

prepare/check/cpu-test do not fit real models or initialize CUDA. Formal fit-lb,
train-large and combine are separate, explicitly scheduled stages. No imported
module globals are changed. Only native634 windows train the two combiners.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import gc
from pathlib import Path
import pickle
import time
import traceback
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import ModernBertConfig, ModernBertForTokenClassification
import run_development as q
import run_full_context_encoder_v2 as encoder
import tail_finetune as mapping
import tail_finetune_all_docs as geometry

ROOT = q.ROOT
OUT = ROOT / 'results/group_crossfit_lb_large_v1'
LARGE = ROOT / 'results/full_context_encoder_large_v1'
MODEL = ROOT.parent / 'models/ModernBERT-large'
LB = ROOT / 'results/lookback_regularization_v2'
LB_NAME = 'lb_prefix_pre_header_C0.0001'
LB_MATRIX = ROOT / 'results/lookback_controls_v2/matrices/lb_prefix_pre_header.npy'
SPLIT_SEED = 20261012
TRAIN_SEED = 20261005
TOKEN_MASS = 560300
WINDOW_MASS = 168123
TREE = dict(max_iter=100, max_depth=2, max_leaf_nodes=4, min_samples_leaf=20,
            learning_rate=.05, l2_regularization=1., early_stopping=False,
            monotonic_cst=[1, 1], random_state=20261008)


def protocol():
    return dict(version='group-crossfit-lb-large-v1', folds=3, split_seed=SPLIT_SEED,
        split='Sort615 fit material-group IDs, fixed NumPy permutation, strided3 folds of205. All original and added generator answers share the group holdout.',
        scope='All3680 fit answers retained across folds; native634/168123 windows train combiners. Calibration159/42241 windows unchanged; official test never read.',
        large=dict(initialization='Generic answerdotai/ModernBERT-large, revision45bb4654a4d5aaff24dd11d4781fa46d39bf8c13; fresh same-seed token head per fold',
            epochs=3, seed=TRAIN_SEED, effective_batch=8, lr=1e-5, weight_decay=.01,
            clip=1., foreach=False, checkpointing='use_reentrant=False',
            precision='BF16 CUDA forward, FP32 logit difference/mapping/BCE/parameters/gradient/Adam states; TF32 disabled',
            order='First3 original global answer_orders filtered to fold-fit answers without reordering; same initial RNG seed each fold',
            weights='Within fold-fit: equal groups; original/added half base mass when both exist, then answer and lexical-token equal; fit-only binary balancing then equalize group loss. Fixed total560300.',
            objective='Each answer weighted token BCE sum × N_fold_fit/(actual_answers_in_batch ×560300). Accumulate8 answers, actual smaller final batch; clip1 then AdamW.',
            selection='Exactly epoch3; no calibration or holdout gold/score based checkpoint selection. Historical choice of3 inherited from previous calibration-selected large model is disclosed.',
            saved='Three online training logs and one final model/optimizer/CPU+CUDA RNG checkpoint per fold, plus all fold-held token probabilities; no repeated epoch weight copies.'),
        lookback=dict(method='lb_prefix_pre_header', C=.0001, solver='liblinear', seed=20260924,
            max_iter=2000, width=1024, scaler='Fold-fit native windows only, base-weighted StandardScaler partial_fit blocks16384; float32 transform',
            weights='Fold-fit native groups→answer→window equal base; fit-only class balancing then equal group loss; fixed total168123.',
            calibration='No fold calibration or C search. Predict only fold-held native windows.'),
        combiners=dict(inputs=['Lookback probability', 'large probability'], model=TREE,
            modes=['old_in_sample', 'group_oof'], fits=2,
            fit='Both use exactly same native634 labels, q.base_weights and168123 loss mass. Only the two fit-side score columns differ.',
            calibration='Reuse byte-bound old full-fit LB and selected large epoch3 scores; identical two columns in both modes. Separate F1-optimal window/answer thresholds only; no model fit or early stopping on calibration.',
            aggregation='Original four-raw-BPE lexical-max windows, then max all eligible windows per answer. Same candidate supplies both reported F1s.'),
        limits=['Corrects own-label exposure of fit-side upstream predictions, not all distribution shift: each OOF upstream sees410 rather than615 groups.',
            'Hyperparameters and epoch3 were chosen in earlier development. Repeated159 calibration is not an independent final test.',
            'Large is an additional semantic encoder; reconstructed Llama Lookback features are not original release traces. Offline detection, not a pure generator probe.',
            'Old controls remain frozen; the new in-sample matched two-input tree is not relabelled as the strongest prior method.'],
        official_test_opened=False)


def fold_token_weights(answers, tokens, original_ids, target_mass=TOKEN_MASS):
    """Explicit subset version of mapping.token_weights; no hardcoded3680."""
    tree = defaultdict(list)
    for i, a in enumerate(answers):
        assert a['partition'] == 'fit'
        tree[a['group_id']].append(i)
    strata = {g: {True: [], False: []} for g in tree}
    for g, ix in tree.items():
        for i in ix: strata[g][answers[i]['response_id'] in original_ids].append(i)
        assert strata[g][True]
    base, labels, bounds, cursor = [], [], [], 0
    for i, (a, t) in enumerate(zip(answers, tokens)):
        lex = np.asarray(t['lexical_mask'], bool); y = np.asarray(t['risk_mask'], int)
        assert lex.any() and not y[~lex].any()
        bins = strata[a['group_id']]; native = a['response_id'] in original_ids
        b = lex.astype(np.float64) * (.5 if bins[False] else 1.) / (len(bins[native]) * lex.sum())
        base.append(b); labels.append(y); bounds.append((cursor, cursor + len(y))); cursor += len(y)
    b = np.concatenate(base); y = np.concatenate(labels); b *= target_mass / b.sum()
    factors = target_mass / (2 * np.bincount(y, weights=b, minlength=2)); loss = b * factors[y]
    for ix in tree.values():
        pos = np.concatenate([np.arange(*bounds[i]) for i in ix])
        loss[pos] *= (target_mass / len(tree)) / loss[pos].sum()
    loss *= target_mass / loss.sum()
    return dict(base=b, loss=loss, y=y, bounds=np.asarray(bounds, np.int64), target_mass=target_mass,
        class_factors=factors, fit_answers=len(answers), groups=len(tree),
        original_membership=np.asarray([a['response_id'] in original_ids for a in answers], bool),
        groups_without_auxiliary=sum(not bins[False] for bins in strata.values()))


def fold_window_weights(rows):
    tree = defaultdict(lambda: defaultdict(list))
    for i, w in enumerate(rows): tree[w['group_id']][w['answer_id']].append(i)
    b = np.empty(len(rows), np.float64)
    for answers in tree.values():
        for ix in answers.values(): b[ix] = 1 / (len(answers) * len(ix))
    b /= b.mean(); b *= WINDOW_MASS / len(rows)
    y = np.asarray([w['label'] for w in rows], int)
    mass = np.bincount(y, weights=b, minlength=2); factors = mass.sum() / (2 * mass); loss = b * factors[y]
    for answers in tree.values():
        ix = [j for entries in answers.values() for j in entries]
        loss[ix] *= (WINDOW_MASS / len(tree)) / loss[ix].sum()
    loss *= WINDOW_MASS / loss.sum()
    return dict(base=b, loss=loss, y=y, class_factors=factors, target_mass=WINDOW_MASS)


def source_paths():
    paths = [Path(__file__), Path(q.__file__), Path(mapping.__file__), Path(encoder.__file__), Path(geometry.__file__),
        ROOT / 'src/run_full_context_encoder_large.py', ROOT / 'src/train_completed_score_combiner.py',
        LARGE / 'preparation_complete.json', LARGE / 'inputs.jsonl', LARGE / 'training_weights.npz', LARGE / 'answer_orders.npy',
        LARGE / 'GPU_SELFCHECK.json', LARGE / 'full_finetune/complete.json', LARGE / 'full_finetune/epoch_03.json',
        LARGE / 'full_finetune/epoch_03_token_predictions.npz', LARGE / 'full_finetune/epoch_03_scores.npz',
        LB / 'complete.json', LB / 'summary.json', LB / f'{LB_NAME}.pkl', LB / f'{LB_NAME}_scores.npz', LB_MATRIX,
        LB_MATRIX.parent.parent / 'matrix_manifest.json',
        ROOT / 'data/gold_manifest.json', ROOT / 'fit_expansion/data/export_freeze.json']
    for folder, names in [(ROOT / 'fit_expansion/data', ['answers_fit.jsonl', 'tokens_fit.jsonl']),
        (ROOT / 'data', ['answers_fit.jsonl', 'tokens_fit.jsonl', 'windows_k4_fit.jsonl',
                       'answers_calibration.jsonl', 'tokens_calibration.jsonl', 'windows_k4_calibration.jsonl']),
        (MODEL, ['download_manifest.json', 'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'])]:
        paths.extend(folder / n for n in names)
    return paths


def prepare():
    assert not torch.cuda.is_initialized() and not (OUT / 'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True); q.save(OUT / 'protocol.json', protocol())
    snap = {str(p.resolve()): q.sha(p) for p in source_paths()}
    download = q.read(MODEL / 'download_manifest.json')
    assert download['revision'] == '45bb4654a4d5aaff24dd11d4781fa46d39bf8c13' and download['status'] == 'complete'
    for r in download['files']:
        key = str((MODEL / r['filename']).resolve())
        if key in snap: assert snap[key] == r['actual_sha256']
    for n, h in q.read(LARGE / 'preparation_complete.json')['files_sha256'].items():
        key = str((LARGE / n).resolve())
        if key in snap: assert snap[key] == h
    old_artifacts = q.read(LARGE / 'full_finetune/epoch_03.json')['artifacts_sha256']
    for suffix in ['_token_predictions.npz', '_scores.npz']:
        assert snap[str((LARGE / ('full_finetune/epoch_03' + suffix)).resolve())] == old_artifacts[suffix]
    for n in [f'{LB_NAME}.pkl', f'{LB_NAME}_scores.npz']:
        assert snap[str((LB / n).resolve())] == q.read(LB / 'complete.json')['files_sha256'][n]
    assert snap[str(LB_MATRIX.resolve())] == q.read(LB_MATRIX.parent.parent / 'matrix_manifest.json')['files_sha256']['lb_prefix_pre_header']
    meta = q.metadata(); answers, tokens, _ = mapping.metadata()
    answers, tokens = answers[:3680], tokens[:3680]
    inputs = q.lines(LARGE / 'inputs.jsonl')[:3680]
    assert [a['response_id'] for a in answers] == [r['response_id'] for r in inputs]
    assert [a['response_id'] for a in answers[:634]] == [a['response_id'] for a in meta['answers'][:634]]
    original_ids = {a['response_id'] for a in answers[:634]}
    groups = sorted({a['group_id'] for a in answers}); assert len(groups) == 615
    assert set(groups) == {a['group_id'] for a in answers[:634]}
    assert not set(groups) & {a['group_id'] for a in meta['answers'][634:]}
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(groups).tolist()
    order = np.load(LARGE / 'answer_orders.npy')[:3]
    assert np.array_equal(order, np.stack([np.random.default_rng(TRAIN_SEED + e).permutation(3680) for e in range(3)]))
    whole_weights = fold_token_weights(answers, tokens, original_ids)
    with np.load(LARGE / 'training_weights.npz') as old:
        assert set(whole_weights) == set(old.files)
        assert all(np.array_equal(old[k], np.asarray(v)) for k, v in whole_weights.items())
    ww = fold_window_weights(meta['windows'][:WINDOW_MASS]); oldw = q.base_weights(meta)
    assert all(np.array_equal(ww[k], v) for k, v in zip(['base', 'loss', 'class_factors', 'y'], oldw))
    q.savel(OUT / 'fit_inputs.jsonl', inputs)
    q.save(OUT / 'fit_identity.json', [dict(response_id=a['response_id'], group_id=a['group_id'],
        original=a['response_id'] in original_ids, raw_tokens=t['token_count']) for a, t in zip(answers, tokens)])
    folds = []
    for f in range(3):
        hold = set(shuffled[f::3]); fit = set(groups) - hold; assert len(hold) == 205 and len(fit) == 410
        fi = [i for i, a in enumerate(answers) if a['group_id'] in fit]
        hi = [i for i, a in enumerate(answers) if a['group_id'] in hold]
        fw = [j for j, w in enumerate(meta['windows'][:WINDOW_MASS]) if w['group_id'] in fit]
        hw = [j for j, w in enumerate(meta['windows'][:WINDOW_MASS]) if w['group_id'] in hold]
        folder = OUT / f'fold_{f}'; folder.mkdir()
        np.savez_compressed(folder / 'token_weights.npz', **fold_token_weights([answers[i] for i in fi], [tokens[i] for i in fi], original_ids))
        np.savez_compressed(folder / 'window_weights.npz', **fold_window_weights([meta['windows'][j] for j in fw]))
        positions = {g: i for i, g in enumerate(fi)}
        global_order = np.stack([[int(j) for j in oo if int(j) in positions] for oo in order])
        local_order = np.array([[positions[int(j)] for j in oo] for oo in global_order], np.int64)
        assert all(sorted(oo.tolist()) == list(range(len(fi))) for oo in local_order)
        np.save(folder / 'global_orders.npy', global_order); np.save(folder / 'local_orders.npy', local_order)
        folds.append(dict(fold=f, fit_groups=sorted(fit), hold_groups=sorted(hold), fit_answer_indices=fi,
            hold_answer_indices=hi, fit_window_indices=fw, hold_window_indices=hw,
            native_fit_answers=sum(i < 634 for i in fi), native_hold_answers=sum(i < 634 for i in hi),
            optimizer_updates_per_epoch=(len(fi) + 7) // 8, final_batch_answers=len(fi) % 8 or 8))
    assert sorted(j for f in folds for j in f['hold_answer_indices']) == list(range(3680))
    assert sorted(j for f in folds for j in f['hold_window_indices']) == list(range(WINDOW_MASS))
    q.save(OUT / 'folds.json', folds)
    assert q.read(LARGE / 'full_finetune/complete.json')['selected']['epoch'] == 3
    old_lb = q.read(LB / 'summary.json')['selected']['lb_prefix_pre_header']; assert old_lb['candidate'] == LB_NAME
    with np.load(LB / f'{LB_NAME}_scores.npz') as z: lb = z['window_scores'].copy()
    with np.load(LARGE / 'full_finetune/epoch_03_token_predictions.npz') as z:
        probabilities = {a['response_id']: z[a['response_id']].copy() for a in meta['answers']}
    large = np.array([max(probabilities[w['response_id']][w['lexical_token_indices']]) for w in meta['windows']], np.float64)
    with np.load(LARGE / 'full_finetune/epoch_03_scores.npz') as z:
        assert np.array_equal(large[:WINDOW_MASS], z['fit_window_scores'][:WINDOW_MASS])
        assert np.array_equal(large[WINDOW_MASS:], z['cal_window_scores'])
    old_columns = np.column_stack((lb, large)); assert old_columns.shape == (210364, 2)
    np.save(OUT / 'old_input_scores.npy', old_columns)
    np.savez_compressed(OUT / 'combiner_weights.npz', **ww)
    q.save(OUT / 'source_snapshot.json', dict(files_sha256=snap, official_test_opened=False))
    artifacts = [p for p in OUT.rglob('*') if p.is_file()]
    report = dict(status='prepared_not_trained', fit_answers=3680, fit_groups=615, native_answers=634,
        native_fit_windows=168123, calibration_answers=159, calibration_windows=42241,
        fold_counts=[{k: (len(f[k]) if isinstance(f[k], list) else f[k]) for k in f} for f in folds],
        original_full_token_weights_exact=True, original_full_window_weights_exact=True,
        old_large_native_window_replay_exact=True, window_order_sha256=q.digest([w['window_id'] for w in meta['windows']]),
        files_sha256={str(p.relative_to(OUT)): q.sha(p) for p in artifacts}, GPU_used=False, real_fits=0, official_test_opened=False)
    q.save(OUT / 'preparation_complete.json', report)
    print('CROSSFIT_PREPARED', report['fold_counts'], flush=True)


def check():
    assert q.read(OUT / 'protocol.json') == protocol()
    done = q.read(OUT / 'preparation_complete.json')
    for n, h in done['files_sha256'].items(): assert q.sha(OUT / n) == h, n
    for p, h in q.read(OUT / 'source_snapshot.json')['files_sha256'].items(): assert q.sha(p) == h, p
    return done


def train_epoch(model, optimizer, rows, weights, order, device, epoch, *, progress=True):
    """Same loop for tiny CPU and scheduled GPU; N is explicit subset length."""
    n = len(rows); mass = float(weights['target_mass'])
    assert sorted(map(int, order)) == list(range(n))
    model.train(); online = 0.; batches = []; tick = time.perf_counter()
    for first in range(0, n, 8):
        chosen = order[first:first + 8]; optimizer.zero_grad(set_to_none=True)
        for i in chosen:
            z = encoder.logits(model, rows[int(i)], device)
            lo, hi = weights['bounds'][int(i)]
            y = torch.as_tensor(weights['y'][lo:hi], dtype=torch.float32, device=device)
            w = torch.as_tensor(weights['loss'][lo:hi], dtype=torch.float32, device=device)
            loss = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() * n / (len(chosen) * mass)
            assert torch.isfinite(loss); loss.backward(); online += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(norm)
        optimizer.step(); batches.append(len(chosen))
        if progress and (first + len(chosen)) % 200 == 0:
            print('CROSSFIT_TRAIN', epoch, first + len(chosen), n, round(time.perf_counter() - tick, 1), flush=True)
    encoder.check_fp32_state(model, optimizer)
    assert all(g['foreach'] is False for g in optimizer.param_groups)
    return dict(epoch=epoch, fit_answers=n, target_mass=mass, updates=len(batches), final_batch=batches[-1],
        online_objective_sum=online, seconds=time.perf_counter() - tick,
        online_is_fixed_final_model_bce=False)


def cpu_test():
    check(); assert not torch.cuda.is_initialized(); torch.set_num_threads(4)
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, pad_token_id=0, num_labels=2, local_attention=16, reference_compile=False)
    config._attn_implementation = 'sdpa'; torch.manual_seed(TRAIN_SEED)
    model = ModernBertForTokenClassification(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    row = dict(input_ids=[1, 3, 4, 5, 6, 7, 8, 2], raw_token_count=3,
        mapping=[[0, 0, 1, 2], [4, 5, 6, 7], [.5, .5, 1., 1.]])
    rows = [copy.deepcopy(row) for _ in range(10)]
    weights = dict(y=np.tile([1., 0., 1.], 10), loss=np.ones(30), bounds=np.arange(0, 33, 3)[:-1, None] + [0, 3], target_mass=30.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01, foreach=False)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    logs = [train_epoch(model, optimizer, rows, weights, np.arange(10), 'cpu', e, progress=False) for e in range(1, 4)]
    assert all(x['updates'] == 2 and x['final_batch'] == 2 for x in logs)
    assert {int(s['step']) for s in optimizer.state.values()} == {6}
    assert model.classifier.weight.grad.abs().sum() > 0 and not torch.equal(initial['classifier.weight'], model.classifier.weight)
    # At fixed logits, expected minibatch scaling averages to the full objective.
    values = np.linspace(.1, 1., 10)
    minibatch = [values[:8].sum() * 10 / (8 * 30), values[8:].sum() * 10 / (2 * 30)]
    assert np.isclose(np.average(minibatch, weights=[8, 2]), values.sum() / 30)
    folds = q.read(OUT / 'folds.json'); ids = q.read(OUT / 'fit_identity.json')
    for f in folds:
        assert not set(f['fit_groups']) & set(f['hold_groups'])
        assert all(ids[i]['group_id'] in f['fit_groups'] for i in f['fit_answer_indices'])
        assert all(ids[i]['group_id'] in f['hold_groups'] for i in f['hold_answer_indices'])
    # Weight construction has no access to excluded examples; changing them cannot affect its inputs.
    assert not torch.cuda.is_initialized()
    q.save(OUT / 'CPU_SELFCHECK.json', dict(passed=True, actual_tiny_ModernBERT_mapping_backward=True,
        optimizer_same_object_steps=[2, 4, 6], per_epoch_logs=logs, final_batch_scaling_checked=True,
        full_original_weights_exact=True, group_exclusion_all_generators=True,
        formal_real_fits=0, GPU_used=False, official_test_opened=False))
    print('CROSSFIT_CPU_TEST_PASSED', flush=True)


def fit_lb():
    check(); assert q.read(OUT / 'CPU_SELFCHECK.json')['passed']
    matrix = np.load(LB_MATRIX, mmap_mode='r'); assert matrix.shape == (210364, 1024)
    for f in q.read(OUT / 'folds.json'):
        folder = OUT / f"fold_{f['fold']}"; assert not (folder / 'lb_started.json').exists()
        q.save(folder / 'lb_started.json', dict(time=time.time(), preparation_sha256=q.sha(OUT / 'preparation_complete.json')))
        ix = np.asarray(f['fit_window_indices']); hold = np.asarray(f['hold_window_indices'])
        with np.load(folder / 'window_weights.npz') as z: w = {k: z[k].copy() for k in z.files}
        scaler = StandardScaler()
        for lo in range(0, len(ix), 16384):
            sl = slice(lo, lo + 16384); scaler.partial_fit(np.asarray(matrix[ix[sl]]), sample_weight=w['base'][sl])
        design = scaler.transform(np.asarray(matrix[ix])).astype(np.float32)
        model = LogisticRegression(C=.0001, solver='liblinear', penalty='l2', max_iter=2000, random_state=20260924)
        tick = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning); model.fit(design, w['y'], sample_weight=w['loss'])
        assert model.n_iter_.max() < 2000
        scores = model.predict_proba(scaler.transform(np.asarray(matrix[hold])).astype(np.float32))[:, 1]
        obj = dict(model=model, scaler=scaler, fit_indices=ix, hold_indices=hold, weights=w,
            fit_groups=f['fit_groups'], hold_groups=f['hold_groups'], C=.0001, preparation_sha256=q.sha(OUT / 'preparation_complete.json'))
        (folder / 'lb.pkl').write_bytes(pickle.dumps(obj, protocol=5))
        np.savez_compressed(folder / 'lb_hold_scores.npz', window_indices=hold, scores=scores)
        q.save(folder / 'lb_complete.json', dict(seconds=time.perf_counter() - tick, iterations=model.n_iter_.tolist(),
            files_sha256={n: q.sha(folder / n) for n in ['lb.pkl', 'lb_hold_scores.npz']}, official_test_opened=False))
        print('CROSSFIT_LB_COMPLETE', f['fold'], flush=True)


def train_large(fold):
    check(); assert q.read(OUT / 'CPU_SELFCHECK.json')['passed']
    assert q.read(LARGE / 'GPU_SELFCHECK.json')['passed']
    f = q.read(OUT / 'folds.json')[fold]; folder = OUT / f'fold_{fold}'
    assert not (folder / 'large_started.json').exists()
    all_rows = q.lines(OUT / 'fit_inputs.jsonl'); rows = [all_rows[i] for i in f['fit_answer_indices']]
    with np.load(folder / 'token_weights.npz') as z: weights = {k: z[k].copy() for k in z.files}
    order = np.load(folder / 'local_orders.npy'); encoder.configure_gpu(); torch.cuda.reset_peak_memory_stats()
    q.save(folder / 'large_started.json', dict(time=time.time(), fold=fold, preparation_sha256=q.sha(OUT / 'preparation_complete.json')))
    tick = time.perf_counter(); torch.manual_seed(TRAIN_SEED)
    model = ModernBertForTokenClassification.from_pretrained(MODEL, local_files_only=True, use_safetensors=True,
        num_labels=2, torch_dtype=torch.float32, attn_implementation='sdpa', reference_compile=False).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01, foreach=False)
    history = []
    for e in range(1, 4):
        log = train_epoch(model, optimizer, rows, weights, order[e - 1], 'cuda', e)
        history.append(log); q.save(folder / f'large_epoch_{e:02d}.json', log)
        print('CROSSFIT_LARGE_EPOCH', fold, e, round(log['seconds'], 1), flush=True)
    # Save final only, before any held-out inference. No held-out/calibration metric selects this checkpoint.
    state = dict(model=geometry.cpu_state(model.state_dict()), optimizer=geometry.cpu_state(optimizer.state_dict()),
        torch_rng=torch.get_rng_state(), cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()],
        epoch=3, fold=fold, fit_groups=f['fit_groups'], preparation_sha256=q.sha(OUT / 'preparation_complete.json'))
    torch.save(state, folder / 'large_final.pt'); del state
    model.eval(); probabilities = {}
    with torch.no_grad():
        for k, i in enumerate(f['hold_answer_indices']):
            row = all_rows[i]; probabilities[row['response_id']] = torch.sigmoid(encoder.logits(model, row, 'cuda')).cpu().numpy()
            if (k + 1) % 200 == 0: print('CROSSFIT_HOLD_PREDICT', fold, k + 1, len(f['hold_answer_indices']), flush=True)
    np.savez_compressed(folder / 'large_hold_token_probabilities.npz', **probabilities)
    q.save(folder / 'large_complete.json', dict(epoch=3, hold_answers=len(probabilities), history=history,
        seconds=time.perf_counter() - tick, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        no_hold_gold_or_calibration_used_for_training_or_epoch_selection=True,
        files_sha256={n: q.sha(folder / n) for n in ['large_final.pt', 'large_hold_token_probabilities.npz']}, official_test_opened=False))
    del optimizer, model; gc.collect(); torch.cuda.empty_cache()
    print('CROSSFIT_LARGE_COMPLETE', fold, flush=True)


def combine():
    check(); assert not (OUT / 'combine_started.json').exists()
    folds = q.read(OUT / 'folds.json')
    # All upstream folds must be complete before any calibration threshold is examined.
    for f in folds:
        folder = OUT / f"fold_{f['fold']}"
        for kind in ['lb', 'large']:
            for n, h in q.read(folder / f'{kind}_complete.json')['files_sha256'].items(): assert q.sha(folder / n) == h
    q.save(OUT / 'combine_started.json', dict(time=time.time(), fixed_fits=2))
    meta = q.metadata(); old = np.load(OUT / 'old_input_scores.npy'); oof = old.copy()
    oof[:WINDOW_MASS] = np.nan; coverage = np.zeros(WINDOW_MASS, int)
    for f in folds:
        folder = OUT / f"fold_{f['fold']}"; hold = np.asarray(f['hold_window_indices'])
        with np.load(folder / 'lb_hold_scores.npz') as z:
            assert np.array_equal(z['window_indices'], hold); oof[hold, 0] = z['scores']
        with np.load(folder / 'large_hold_token_probabilities.npz') as z:
            probabilities = {k: z[k].copy() for k in z.files}
        for j in hold:
            w = meta['windows'][j]; oof[j, 1] = max(probabilities[w['response_id']][w['lexical_token_indices']])
        coverage[hold] += 1
    assert (coverage == 1).all() and np.isfinite(oof).all()
    assert np.array_equal(oof[WINDOW_MASS:], old[WINDOW_MASS:])
    np.save(OUT / 'oof_input_scores.npy', oof)
    with np.load(OUT / 'combiner_weights.npz') as z: weights = z['loss'].copy(); y = z['y'].copy()
    methods = {}
    for name, x in [('old_in_sample', old), ('group_oof', oof)]:
        model = HistGradientBoostingClassifier(**TREE); start = time.perf_counter()
        model.fit(x[:WINDOW_MASS], y, sample_weight=weights)
        scores = model.predict_proba(x)[:, 1]; answer = q.answer_scores(meta, scores)
        thresholds = dict(window=q.choose_threshold([w['label'] for w in meta['windows'][WINDOW_MASS:]], scores[WINDOW_MASS:]),
            answer=q.choose_threshold([a['label'] for a in meta['answers'][634:]], answer[634:]))
        (OUT / f'{name}.pkl').write_bytes(pickle.dumps(model, protocol=5))
        np.savez_compressed(OUT / f'{name}_scores.npz', window_scores=scores, answer_scores=answer)
        entry = dict(method=name, metrics=q.metrics(meta, scores, thresholds), thresholds=thresholds,
            iterations=model.n_iter_, seconds=time.perf_counter() - start,
            files_sha256={n: q.sha(OUT / n) for n in [f'{name}.pkl', f'{name}_scores.npz']})
        methods[name] = entry; q.save(OUT / f'{name}.json', entry)
    q.save(OUT / 'summary.json', dict(methods=methods, same_calibration_input_columns_exact=True,
        native_oof_coverage_once=True, calibration_is_repeated_development=True, official_test_opened=False))
    q.save(OUT / 'complete.json', dict(summary_sha256=q.sha(OUT / 'summary.json'), upstream_LR_fits=3,
        upstream_large_fits=3, combiner_fits=2, official_test_opened=False))
    print('CROSSFIT_COMBINERS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'check', 'cpu-test', 'fit-lb', 'train-large', 'combine'])
    parser.add_argument('--fold', type=int, choices=[0, 1, 2]); args = parser.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        try:
            if args.stage == 'train-large':
                assert args.fold is not None; train_large(args.fold)
            else: {'prepare': prepare, 'check': check, 'cpu-test': cpu_test, 'fit-lb': fit_lb, 'combine': combine}[args.stage]()
        except Exception:
            if OUT.exists():
                q.save(OUT / f'FAILURE_{args.stage}_{args.fold}_{time.time_ns()}.json', dict(stage=args.stage, fold=args.fold, traceback=traceback.format_exc()))
            raise
