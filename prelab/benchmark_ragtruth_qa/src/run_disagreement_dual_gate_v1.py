"""Our two-head disagreement decoder. CPU prepare/check; train is explicit.

Only the new gates receive group-OOF fit predictions. The frozen upstream
scores were trained/selected previously and are NOT cross-fitted here.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import argparse
import pickle
import time
import traceback
import numpy as np
import sklearn
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_development as q

OUT = q.ROOT / 'results/disagreement_dual_gate_v1'
NAMES = ('current', 'semantic_claim', 'tail2', 'large')
NFIT = 168123
SEED = 20261014
SPLIT_SEED = 20261012
GRID = tuple(round(.05 + .03 * i, 2) for i in range(31))
PARAMS = dict(max_iter=100, learning_rate=.05, max_leaf_nodes=15, max_depth=3,
    min_samples_leaf=50, l2_regularization=1., random_state=SEED,
    early_stopping=False)
SOURCE = q.ROOT / 'results/large_fixed_convex_v1'
CURRENT = 'semantic_claim__old_tree__large_weight0.4'


def protocol():
    return dict(version='our-disagreement-dual-gate-v1', identity='Our method; never part of a paper baseline',
        scope=dict(fit_answers=634, cal_answers=159, fit_windows=168123, cal_windows=42241, fit_groups=615, cal_groups=154),
        sources=list(NAMES), current_candidate=CURRENT, model=PARAMS, CPU_threads=4,
        features=dict(width=40, logit_clip=1e-6,
            columns='4 logit(p),4 p-old_window_threshold,6 signed pairwise logit differences,6 equal old-threshold decisions,16 local probability max/mean (4 sources x radii1,3),4 source answermax probabilities',
            neighborhood='Symmetric clipped radius on each same-answer token_start-contiguous eligible-window chain. Break at missing starts; never bridge answers. Offline use of future neighbors is explicit.',
            identity='IDs locate boundaries/folds only, never numeric/text features. No gold, original text, label-derived boundaries, spans or risk types enter feature generation.'),
        gates=dict(keep='Only windows where frozen current >= its old window threshold. Target is original risk label.',
            rescue='Only current-negative windows with at least one of semantic_claim/tail2/large >= its own old window threshold. Same original risk target.',
            outside='Current-negative windows without any component alert stay negative.',
            weights='Slice byte-bound original native group->answer->window base_weights by this gate and fold-fit indices. Compute class factors mass/(2*class_mass) using ONLY those labels; preserve this subset original base mass. No original binary loss reuse, second balancing, or postbalance group renormalization. Gate restriction changes group exposure.',
            missing_class='Fail explicitly if a real fold/gate lacks either class; no constant model or changed structure fallback.'),
        folds=dict(count=3, seed=SPLIT_SEED,
            rule='Sort615 fit group IDs; NumPy default_rng permutation; strided3 folds of205 groups. The same fold assignments apply to both gates and all answers/windows in a group.',
            interpretation='Only the two gates are group-OOF. Frozen upstream fit scores and historical choices remain in-sample/cal-developed; this is not leakage-free end-to-end cross-fitting.'),
        selection=dict(grid=list(GRID), grid_count=961, include_identity=True,
            rule='Use the two gate OOF probabilities on fit only. Joint unweighted whole-fit-window F1, then precision, then smaller L1 threshold distance to0.5. Stable enumeration breaks exact remaining ties. Identity is first with neutral distance0 and exact original current decisions.',
            identity='No keep/rescue thresholds applied; preserve current window predictions exactly. This adds no learned fallback classifier.',
            no_cal_selection=True),
        training=dict(gate_models=8, OOF_models=6, final_models=2,
            order='Fit3 folds x2 gates and save all OOF scores/models/weights. Select and freeze thresholds on fit OOF. Then fit each gate on all its fit subset once; freeze both models before predicting/scoring cal.',
            all_fit_thresholds='Same OOF-selected thresholds apply to final fit and cal predictions; no final-fit threshold reselection.'),
        output=dict(window='Binary decision: keep/current-positive or rescue/eligible-negative threshold decisions. Gate probabilities separately saved; no fabricated unified calibrated window score.',
            answer='A separate unchanged head: retain exact original current answer_scores and its old answer threshold. It is intentionally NOT max over gated windows.',
            reporting='Whole fit OOF window metrics for threshold selection, final-model in-sample fit window metrics, and cal window metrics; answer metrics always original frozen current head. No window AUROC/AP invented from binary output.'),
        limitations=['Repeated calibration was used to develop/select the upstream models, current candidate and this hypothesis.',
            'Group OOF at gate level cannot remove own-label exposure in the upstream fit scores.',
            'The answer head and local decisions may disagree; this is one shared backbone with two output heads, not an all-window-max claim.',
            'This is offline extra-semantic-model plus generator-signal decoding, not a pure generator probe or baseline modification.'],
        official_test_opened=False, GPU_used=False)


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    return dict(n=len(y), positive=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def features(probability, thresholds, structure):
    """Consumes probabilities and nongold geometry only."""
    p = np.asarray(probability, np.float64)
    assert p.shape == (len(structure), 4) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    z = logit(np.clip(p, 1e-6, 1-1e-6)); t = np.asarray(thresholds)
    decisions = p >= t
    columns = [z, p-t]; names = [f'{n}_logit' for n in NAMES] + [f'{n}_threshold_margin' for n in NAMES]
    for a, b in combinations(range(4), 2):
        columns.append((z[:, a]-z[:, b])[:, None]); names.append(f'logit_diff_{NAMES[a]}_{NAMES[b]}')
    for a, b in combinations(range(4), 2):
        columns.append((decisions[:, a] == decisions[:, b])[:, None]); names.append(f'old_decision_agrees_{NAMES[a]}_{NAMES[b]}')
    chains = []; left = 0
    for i in range(1, len(structure)+1):
        if i == len(structure) or structure[i]['response_id'] != structure[i-1]['response_id'] or structure[i]['token_start'] != structure[i-1]['token_start']+1:
            chains.append((left, i)); left = i
    for radius in (1, 3):
        mx = np.empty_like(p); mean = np.empty_like(p)
        for lo, hi in chains:
            for i in range(lo, hi):
                block = p[max(lo, i-radius):min(hi, i+radius+1)]
                mx[i] = block.max(axis=0); mean[i] = block.mean(axis=0)
        columns.extend([mx, mean]); names += [f'{n}_chain_radius{radius}_max' for n in NAMES] + [f'{n}_chain_radius{radius}_mean' for n in NAMES]
    aw = defaultdict(list)
    for i, r in enumerate(structure): aw[r['response_id']].append(i)
    amax = np.empty_like(p)
    for ix in aw.values(): amax[ix] = p[ix].max(axis=0)
    columns.append(amax); names += [f'{n}_answer_max' for n in NAMES]
    x = np.column_stack(columns).astype(np.float32)
    assert x.shape == (len(p), 40) and len(names) == 40 and np.isfinite(x).all()
    keep = decisions[:, 0]; rescue = ~keep & decisions[:, 1:].any(axis=1)
    assert not (keep & rescue).any()
    return x, names, keep, rescue, chains


def gate_weights(base, y):
    b = np.asarray(base, np.float64); y = np.asarray(y, int)
    assert len(b) == len(y) and len(b) and (b > 0).all() and set(y) == {0, 1}
    mass = np.bincount(y, weights=b, minlength=2)
    factors = b.sum()/(2*mass); w = b*factors[y]
    assert np.isclose(w.sum(), b.sum()) and np.isclose(w[y == 0].sum(), w[y == 1].sum())
    return w, dict(rows=len(b), raw_class_rows=np.bincount(y, minlength=2).tolist(),
        base_class_mass=mass.tolist(), class_factors=factors.tolist(),
        loss_class_mass=np.bincount(y, weights=w, minlength=2).tolist(), target_mass=float(b.sum()))


def threshold_grid(y, current, rescue, pk, pr):
    """Disjoint gate masks make the whole-fit grid a sum of two fixed counts."""
    assert len(y) == len(current) == len(rescue) == len(pk) == len(pr)
    assert not (current & rescue).any()
    assert np.isfinite(pk[current]).all() and np.isfinite(pr[rescue]).all()
    identity = dict(mode='identity', keep_threshold=None, rescue_threshold=None,
        distance_to_half=0., metrics=metrics(y, current))
    entries = [identity]
    kstats = [metrics(y[current], pk[current] >= t) for t in GRID]
    rstats = [metrics(y[rescue], pr[rescue] >= t) for t in GRID]
    pos = int(np.asarray(y).sum())
    for i, tk in enumerate(GRID):
        for j, tr in enumerate(GRID):
            tp = kstats[i]['tp']+rstats[j]['tp']; fp = kstats[i]['fp']+rstats[j]['fp']; fn = pos-tp
            m = dict(n=len(y), positive=pos, tp=tp, fp=fp, fn=fn, tn=len(y)-pos-fp,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/pos if pos else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            entries.append(dict(mode='gates', keep_threshold=tk, rescue_threshold=tr,
                distance_to_half=abs(tk-.5)+abs(tr-.5), metrics=m))
    def key(e): return (e['metrics']['f1'], e['metrics']['precision'], -e['distance_to_half'])
    chosen = max(entries, key=key)
    assert chosen['metrics'] == metrics(y, decide(current, rescue, pk, pr, chosen))
    return chosen, entries


def decide(current, rescue, pk, pr, chosen):
    if chosen['mode'] == 'identity': return current.copy()
    return (current & (pk >= chosen['keep_threshold'])) | (rescue & (pr >= chosen['rescue_threshold']))


def load_sources(meta):
    paths = [Path(__file__), Path(q.__file__), SOURCE/'complete.json', SOURCE/'summary.json',
        q.DATA/'gold_manifest.json', q.OUT/'training_weights.npz', q.OUT/'complete.json']
    for part in q.PARTITIONS:
        paths += [q.DATA/f'{n}_{part}.jsonl' for n in ('answers', 'tokens', 'windows_k4')]
    complete = q.read(SOURCE/'complete.json'); summary = q.read(SOURCE/'summary.json')
    assert complete['summary_sha256'] == q.sha(SOURCE/'summary.json') and not complete['official_test_opened']
    e = summary['selected']['semantic_claim__old_tree']; assert e['candidate'] == CURRENT
    large_e = next(r for r in summary['all_candidates']['semantic_claim__old_tree'] if r['large_weight'] == 1.)
    cp = q.ROOT/'results/claim_pooling_v1'; paths += [cp/'complete.json', cp/'summary.json']
    semantic_e = q.read(cp/'summary.json')['selected']['minicheck_hidden64_risk_tcn_w32']
    tail = q.ROOT/'results/minicheck_tail_all_docs_v3/tail2'; paths += [tail/'complete.json']
    tail_done = q.read(tail/'complete.json'); assert tail_done['status'] == 'complete_development_only' and not tail_done['test_opened']
    tail_e = tail_done['selected']; assert tail_e['epoch'] == 2
    records = [(SOURCE, e), (cp, semantic_e), (tail, tail_e), (SOURCE, large_e)]
    probabilities = []; answer_probabilities = []; entries = []
    for name, (folder, item) in zip(NAMES, records):
        fn = 'epoch_02_scores.npz' if name == 'tail2' else item['candidate']+'_scores.npz'
        path = folder/fn; paths.append(path)
        expected_hash = item['artifacts_sha256']['_scores.npz'] if name == 'tail2' else item['scores_sha256']
        assert q.sha(path) == expected_hash
        with np.load(path, allow_pickle=False) as z:
            if name == 'tail2':
                assert np.array_equal(z['fit_window_labels'][:NFIT], [w['label'] for w in meta['windows'][:NFIT]])
                assert np.array_equal(z['cal_window_labels'], [w['label'] for w in meta['windows'][NFIT:]])
                p = np.r_[z['fit_window_scores'][:NFIT], z['cal_window_scores']]
                a = np.r_[z['fit_answer_scores'][:634], z['cal_answer_scores']]
            else: p, a = z['window_scores'].copy(), z['answer_scores'].copy()
        assert p.shape == (210364,) and a.shape == (793,)
        assert np.array_equal(q.answer_scores(meta, p), a)
        measured = q.metrics(meta, p, item['thresholds'])
        expected_cal = item['calibration'] if name == 'tail2' else item['metrics']['calibration']
        assert measured['calibration'] == expected_cal
        if name != 'tail2': assert measured == item['metrics']
        probabilities.append(p); answer_probabilities.append(a)
        entries.append(dict(name=name, path=str(path.resolve()), scores_sha256=expected_hash,
            thresholds=item['thresholds'], original_calibration=expected_cal,
            native_fit_metrics=measured['fit']))
    with np.load(q.OUT/'training_weights.npz') as z:
        base = z['base_weights'].copy(); old_y = z['y'].copy()
    orig = q.base_weights(meta)
    assert np.array_equal(base, orig[0]) and np.array_equal(old_y, orig[3])
    assert np.isclose(base.sum(), NFIT)
    assert q.sha(q.OUT/'training_weights.npz') == q.read(q.OUT/'complete.json')['files_sha256']['training_weights.npz']
    return np.column_stack(probabilities), np.column_stack(answer_probabilities), entries, base, {str(p.resolve()): q.sha(p) for p in paths}


def tiny():
    model = HistGradientBoostingClassifier(**PARAMS)
    assert model.get_params()['early_stopping'] is False and model.get_params()['random_state'] == SEED
    # A missing start and a new answer must each block neighborhood propagation.
    p = np.array([[.1]*4, [.2]*4, [.9]*4, [.8]*4, [.7]*4])
    structure = [dict(response_id=r, token_start=t) for r, t in [('a',0), ('a',1), ('a',5), ('b',0), ('b',1)]]
    x, names, keep, rescue, chains = features(p, [.5]*4, structure)
    assert chains == [(0, 2), (2, 3), (3, 5)]
    assert x[0, names.index('current_chain_radius3_max')] == np.float32(.2)
    assert x[0, names.index('current_answer_max')] == np.float32(.9)
    assert np.isfinite(features(np.array([[0, 1, 0, 1.]]), [.5]*4, [dict(response_id='x', token_start=0)])[0]).all()
    # Real production HGBs on deterministic synthetic data only, no real sample fit.
    rng = np.random.default_rng(SEED); n = 2400
    p = rng.uniform(.01, .99, (n, 4)); y = (p[:, 1]+p[:, 2]+rng.normal(0, .2, n) > 1).astype(int)
    s = [dict(response_id=f'tiny_{i//40}', token_start=i%40) for i in range(n)]
    x, _, keep, rescue, _ = features(p, [.5]*4, s)
    gate_probs = []; details = []
    with threadpool_limits(4):
        for mask in (keep, rescue):
            b = rng.uniform(.5, 1.5, mask.sum()); w, detail = gate_weights(b, y[mask])
            m = HistGradientBoostingClassifier(**PARAMS).fit(x[mask], y[mask], sample_weight=w)
            pred = np.full(n, np.nan); pred[mask] = m.predict_proba(x[mask])[:, 1]
            assert np.isfinite(pred[mask]).all() and m.n_iter_ == 100
            assert np.array_equal(pred[mask], pickle.loads(pickle.dumps(m)).predict_proba(x[mask])[:, 1])
            gate_probs.append(pred); details.append(detail)
    selected, grid = threshold_grid(y, keep, rescue, *gate_probs)
    assert len(grid) == 962 and selected['metrics']['f1'] >= grid[0]['metrics']['f1']
    for i in (1, 125, 480, 961):
        assert grid[i]['metrics'] == metrics(y, decide(keep, rescue, *gate_probs, grid[i]))
    assert np.array_equal(decide(keep, rescue, *gate_probs, grid[0]), keep)
    frozen_answer = rng.random(60); assert np.array_equal(frozen_answer.copy(), frozen_answer)
    return dict(status='passed', synthetic_rows=n, synthetic_HGB_fits=2,
        effective_parameters=model.get_params(), sklearn_version=sklearn.__version__,
        real_data_fits=0, finite_clipped_logits=True, chain_gap_and_answer_boundary=True,
        balanced_mass=details, saved_model_replay_exact=True, whole_window_grid_counts_exact=True,
        identity_exact=True, offline_feature_width=40)


def prepare():
    assert not (OUT/'preparation_complete.json').exists(), 'No overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = protocol()
    if (OUT/'protocol.json').exists(): assert q.read(OUT/'protocol.json') == cfg
    else: q.save(OUT/'protocol.json', cfg)
    start = time.perf_counter(); test = tiny(); q.save(OUT/'CPU_SELFCHECK.json', test)
    meta = q.metadata(); assert meta['bounds'] == {'fit': [0, 168123], 'calibration': [168123, 210364]}
    p, a, entries, base, bindings = load_sources(meta)
    structure = [{k: w[k] for k in ('response_id', 'token_start', 'token_end')} for w in meta['windows']]
    thresholds = [e['thresholds']['window']['threshold'] for e in entries]
    x, names, keep, rescue, chains = features(p, thresholds, structure)
    y = np.asarray([w['label'] for w in meta['windows']], np.int8)
    group_ids = np.asarray([w['group_id'] for w in meta['windows']])
    groups = sorted(set(group_ids[:NFIT])); assert len(groups) == 615 and not set(groups) & set(group_ids[NFIT:])
    perm = np.random.default_rng(SPLIT_SEED).permutation(groups).tolist()
    folds = []; coverage = np.zeros(NFIT, int)
    for fold in range(3):
        hold_groups = perm[fold::3]; assert len(hold_groups) == 205
        held = np.isin(group_ids[:NFIT], hold_groups); coverage += held
        fold_record = dict(fold=fold, held_groups=hold_groups, fit_groups=sorted(set(groups)-set(hold_groups)), gates={})
        for name, mask in [('keep', keep), ('rescue', rescue)]:
            ix = np.flatnonzero(mask[:NFIT] & ~held); hi = np.flatnonzero(mask[:NFIT] & held)
            w, stat = gate_weights(base[ix], y[ix])
            np.savez_compressed(OUT/f'fold_{fold}_{name}_indices_weights.npz', fit_indices=ix, held_indices=hi, sample_weights=w)
            fold_record['gates'][name] = dict(training=stat, held_rows=len(hi), held_groups_have_no_training_rows=True)
        folds.append(fold_record)
    assert np.array_equal(coverage, np.ones(NFIT, int))
    whole = {}
    for name, mask in [('keep', keep), ('rescue', rescue)]:
        ix = np.flatnonzero(mask[:NFIT]); w, stat = gate_weights(base[ix], y[ix]); whole[name] = stat
        np.savez_compressed(OUT/f'full_{name}_indices_weights.npz', fit_indices=ix, sample_weights=w)
    np.save(OUT/'window_features.npy', x)
    np.savez_compressed(OUT/'frozen_inputs.npz', window_probabilities=p, answer_probabilities=a,
        window_labels=y, answer_labels=np.asarray([a['label'] for a in meta['answers']], np.int8),
        current_positive=keep, rescue_eligible=rescue, native_base_weights=base)
    q.save(OUT/'feature_names.json', names); q.save(OUT/'folds.json', folds)
    q.save(OUT/'source_binding.json', dict(files_sha256=bindings, source_entries=entries))
    q.save(OUT/'geometry.json', dict(window_order_sha256=q.digest(meta['windows']),
        answer_ids=[a['response_id'] for a in meta['answers']], structure_sha256=q.digest(structure),
        group_order_sha256=q.digest(group_ids.tolist()), chains=len(chains), chain_rule=cfg['features']['neighborhood']))
    q.save(OUT/'CPU_DATA_CHECK.json', dict(status='passed', real_data_fits=0,
        shape=list(x.shape), fit_windows=NFIT, cal_windows=len(x)-NFIT,
        gate_full_fit_weight_statistics=whole,
        fit_gate_rows={'keep': int(keep[:NFIT].sum()), 'rescue': int(rescue[:NFIT].sum())},
        cal_gate_rows_no_new_fit={'keep': int(keep[NFIT:].sum()), 'rescue': int(rescue[NFIT:].sum())},
        frozen_source_predictions_answermax_and_counts_exact=True, original_native_base_weights_exact=True,
        feature_gold_free_whitelist=True, group_fold_isolation=True, seconds=time.perf_counter()-start,
        estimated_real_fit_minutes=[1, 5], estimated_peak_RAM_GiB=[.5, 1.5],
        disk_feature_bytes=x.nbytes, estimated_total_output_MiB=[45, 90]))
    files = [p for p in OUT.iterdir() if p.is_file() and p.name not in ('preparation_complete.json',) and not p.name.startswith('FAILURE')]
    q.save(OUT/'preparation_complete.json', dict(status='CPU_prepared_not_real_fit',
        files_sha256={p.name: q.sha(p) for p in files}, source_sha256=bindings,
        real_data_fits=0, GPU_used=False, official_test_opened=False))
    (OUT/'PLAN.md').write_text('# 双门控分歧解码器：待审核的自研方法\n\n'
        '仅CPU准备及两次合成小样本真实HGB训练通过；尚未拟合真实门控。protocol.json固定40维、2门、3组折OOF、31×31阈值加identity。\n\n'
        '窗口门控与整答原head分开：整答仍逐值使用current旧分数/阈值，不宣称它来自门控窗max。只有新门层OOF；上游fit分数仍非交叉拟合，旧cal开发暴露没有消除。\n\n'
        '原组→答→窗base质量在门/训练折内切片，再仅按该子集标签平衡两类、保持子集总质量；不重复套旧binary loss。门筛选和类平衡后组质量不再保证相等，定义与实际统计均保留。\n\n'
        '真实运行预计8个小树模型合计1–5分钟、峰内存0.5–1.5 GiB、输出45–90 MiB；这是估计。无GPU、基线修改或test。入口prepare/check仅准备，train需根代理另行授权。\n', encoding='utf-8')
    print('DUAL_GATE_CPU_PREPARED_NOT_REAL_FIT', q.read(OUT/'CPU_DATA_CHECK.json'), flush=True)


def check():
    done = q.read(OUT/'preparation_complete.json')
    assert done['status'] == 'CPU_prepared_not_real_fit' and not done['GPU_used']
    assert q.read(OUT/'protocol.json') == protocol()
    for name, h in done['files_sha256'].items(): assert q.sha(OUT/name) == h, name
    for name, h in done['source_sha256'].items(): assert q.sha(name) == h, name
    x = np.load(OUT/'window_features.npy', mmap_mode='r')
    assert x.shape == (210364, 40) and x.dtype == np.float32 and np.isfinite(x).all()
    assert q.read(OUT/'CPU_SELFCHECK.json')['status'] == 'passed'
    print('DUAL_GATE_CHECK_PASSED_NO_FIT', flush=True)


def train():
    check(); assert not (OUT/'started.json').exists(), 'Never silently resume/refit'
    started = time.perf_counter()
    q.save(OUT/'started.json', dict(start_time=time.time(), preparation_sha256=q.sha(OUT/'preparation_complete.json'), actual_real_fit_started=True))
    x = np.load(OUT/'window_features.npy', mmap_mode='r')
    with np.load(OUT/'frozen_inputs.npz') as z:
        y=z['window_labels'].copy(); ay=z['answer_labels'].copy(); current=z['current_positive'].copy(); rescue=z['rescue_eligible'].copy()
        old_answer=z['answer_probabilities'][:, 0].copy()
    pk = np.full(NFIT, np.nan); pr = np.full(NFIT, np.nan); model_files=[]; fit_logs=[]
    with threadpool_limits(4):
        for fold in range(3):
            for name, buffer in [('keep', pk), ('rescue', pr)]:
                path=OUT/f'fold_{fold}_{name}_indices_weights.npz'
                with np.load(path) as z: ix=z['fit_indices']; hi=z['held_indices']; weight=z['sample_weights']
                tick=time.perf_counter(); model=HistGradientBoostingClassifier(**PARAMS)
                model.fit(x[ix], y[ix], sample_weight=weight)
                assert model.n_iter_==100 and np.isnan(buffer[hi]).all()
                buffer[hi]=model.predict_proba(x[hi])[:, 1]
                path=OUT/f'fold_{fold}_{name}.pkl'; path.write_bytes(pickle.dumps(model)); model_files.append(path)
                fit_logs.append(dict(fold=fold, gate=name, rows=len(ix), held_rows=len(hi), iterations=model.n_iter_, seconds=time.perf_counter()-tick))
        assert np.isfinite(pk[current[:NFIT]]).all() and np.isnan(pk[~current[:NFIT]]).all()
        assert np.isfinite(pr[rescue[:NFIT]]).all() and np.isnan(pr[~rescue[:NFIT]]).all()
        chosen, table=threshold_grid(y[:NFIT], current[:NFIT], rescue[:NFIT], pk, pr)
        np.savez_compressed(OUT/'fit_OOF_gate_probabilities.npz', keep_probability=pk, rescue_probability=pr)
        q.save(OUT/'fit_only_threshold_grid.json', table)
        q.save(OUT/'fit_only_selection.json', dict(selected=chosen, threshold_candidates=len(table), selected_on='whole_fit_OOF_windows_only', no_cal_prediction_used=True))
        final_models={}
        for name in ('keep', 'rescue'):
            with np.load(OUT/f'full_{name}_indices_weights.npz') as z: ix=z['fit_indices']; weight=z['sample_weights']
            tick=time.perf_counter(); model=HistGradientBoostingClassifier(**PARAMS).fit(x[ix], y[ix], sample_weight=weight)
            assert model.n_iter_==100; final_models[name]=model
            path=OUT/f'full_{name}.pkl'; path.write_bytes(pickle.dumps(model)); model_files.append(path)
            fit_logs.append(dict(fold='full_fit', gate=name, rows=len(ix), iterations=model.n_iter_, seconds=time.perf_counter()-tick))
        q.save(OUT/'full_fit_frozen.json', dict(models_sha256={p.name:q.sha(p) for p in model_files},
            selection_sha256=q.sha(OUT/'fit_only_selection.json'), real_models_fitted=8, calibration_not_scored=True))
        final_pk=np.full(len(y), np.nan); final_pr=np.full(len(y), np.nan)
        final_pk[current]=final_models['keep'].predict_proba(x[current])[:, 1]
        final_pr[rescue]=final_models['rescue'].predict_proba(x[rescue])[:, 1]
    pred=decide(current, rescue, final_pk, final_pr, chosen)
    assert not pred[~current & ~rescue].any()
    entries=q.read(OUT/'source_binding.json')['source_entries']; ats=entries[0]['thresholds']['answer']['threshold']
    answer_metrics={'fit':q.count(ay[:634],old_answer[:634],ats),'calibration':q.count(ay[634:],old_answer[634:],ats)}
    assert answer_metrics['fit']==entries[0]['native_fit_metrics']['answers']
    assert answer_metrics['calibration']==entries[0]['original_calibration']['answers']
    np.savez_compressed(OUT/'final_two_head_predictions.npz', window_decisions=pred, keep_probability=final_pk,
        rescue_probability=final_pr, answer_scores=old_answer, answer_decisions=old_answer>=ats)
    result=dict(status='complete_development_only', selected=chosen, real_gate_fits=8,
        window_metrics={'fit_OOF_at_selected_thresholds':chosen['metrics'],
            'final_fit_in_sample':metrics(y[:NFIT],pred[:NFIT]),'calibration':metrics(y[NFIT:],pred[NFIT:])},
        answer_metrics_unchanged=answer_metrics, answer_threshold_unchanged=ats,
        source_current_window_metrics={'fit':entries[0]['native_fit_metrics']['windows'], 'calibration':entries[0]['original_calibration']['windows']},
        local_and_answer_heads_distinct=True, final_answer_is_NOT_gated_window_max=True,
        threshold_grid_count=962, fit_log=fit_logs, seconds=time.perf_counter()-started,
        no_cal_selection=True, only_gate_level_OOF=True, GPU_used=False, official_test_opened=False)
    q.save(OUT/'summary.json',result)
    q.save(OUT/'complete.json',dict(status='complete_development_only', summary_sha256=q.sha(OUT/'summary.json'),
        selected_sha256=q.sha(OUT/'fit_only_selection.json'), predictions_sha256=q.sha(OUT/'final_two_head_predictions.npz'),
        full_fit_freeze_sha256=q.sha(OUT/'full_fit_frozen.json'), real_fits=8, official_test_opened=False, GPU_used=False))
    print('DUAL_GATE_COMPLETE',result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','check','train'))
    args=parser.parse_args()
    try: globals()[args.command]()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        q.save(OUT/f'FAILURE_{args.command}_{time.time_ns()}.json',dict(command=args.command, traceback=traceback.format_exc(), no_automatic_structure_change=True))
        raise
