"""Give each completed peer the same fixed large-model signal and fit budget."""
from pathlib import Path
import argparse
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_development as q
import run_completed_score_fusion as inputs
import train_citation_alignment as linear
import train_completed_score_combiner as tree

OUT = q.ROOT / 'results/large_matched_combination_v1'
LARGE = q.ROOT / 'results/full_context_encoder_large_v1/full_finetune'


def protocol():
    return {
        'version': 'large-matched-combination-v1', 'peers': list(inputs.PEERS),
        'large_selection': 'Use only completed six-epoch run selected epoch1..6. Never select a different epoch for an individual combination.',
        'scope': 'Original634 fit answers/168123 windows,159 calibration answers/42241 windows; no new gold or official-test access.',
        'mapping': 'Unchanged original4rawBPE window, maximum over existing lexical token indices; answer is maximum over all original eligible windows.',
        'large_signal': 'One identical selected whole-input ModernBERT-large window probability for all3 peers. Its backbone trained3680 fit answers.',
        'linear': {'control': 'Reuse all frozen original2scores+8citation-feature LR and scalers, no refit.',
            'candidate': 'Append just large probability to original2scores+8features =>11columns.',
            'C': list(linear.CS), 'fits': 9,
            'scaler': 'Original fit-only weighted StandardScaler; original q.base_weights base and loss unchanged.',
            'hyperparameters': linear.protocol()['LR'],
            'selection': 'Original q.selection_key selects one C per peer; same candidate serves both units.'},
        'tree': {'control': 'Reuse each original two-score monotone tree, no refit.',
            'candidate': 'Append same large probability to original peer+tail2 probabilities =>3columns.',
            'fits': 3, 'hyperparameters': {**tree.protocol()['model'], 'monotonic_cst': [1, 1, 1]}},
        'calibration': 'Only original159 calibration answers choose two F1-optimal thresholds, original tie rules; no gradient or early stopping on calibration.',
        'budget': 'Exactly9LR and3fixed100-step trees. Same signal and fit budget for every peer. No extra C,depth or epoch search.',
        'limits': ['Additional offline semantic encoder; not a pure generator white-box probe.',
            'Base fit predictions are in-sample, not cross-fitted stacking.',
            'Calibration repeatedly developed; cannot claim independent-test improvement or SOTA.',
            'Must compare each single candidate against all stronger existing controls and standalone large; do not combine maxima from different candidates.'],
        'GPU_used': False, 'official_test_opened': False}


def bindings():
    files = [Path(__file__), Path(q.__file__), Path(inputs.__file__), Path(linear.__file__), Path(tree.__file__),
        linear.OUT/'complete.json', linear.OUT/'summary.json', tree.OUT/'complete.json', tree.OUT/'summary.json',
        inputs.OUT/'complete.json', inputs.OUT/'summary.json', linear.FEATURES/'preparation_complete.json',
        q.ROOT/'src/run_full_context_encoder_large.py', q.ROOT/'src/run_full_context_encoder_v2.py',
        LARGE.parent/'preparation_complete.json', LARGE.parent/'protocol.json']
    return {str(p.resolve()): q.sha(p) for p in files}


def load_original(meta):
    for directory in (inputs.OUT, linear.OUT, tree.OUT):
        assert q.sha(directory/'summary.json') == q.read(directory/'complete.json')['summary_sha256']
    feature_manifest = q.read(linear.FEATURES/'preparation_complete.json')
    for name, digest in feature_manifest['files_sha256'].items():
        assert q.sha(linear.FEATURES/name) == digest
    assert feature_manifest['window_order_sha256'] == q.digest([w['window_id'] for w in meta['windows']])
    features = np.load(linear.FEATURES/'window_features.npy')
    assert features.shape == (210364, 8) and np.isfinite(features).all()
    sources = q.read(inputs.OUT/'summary.json'); result = {}
    for peer in inputs.PEERS:
        components = []
        for alpha in (0., 1.):
            entry = next(e for e in sources['all_candidates'][peer] if e['tail_weight'] == alpha)
            path = inputs.OUT/(entry['candidate']+'_scores.npz')
            assert q.sha(path) == entry['scores_sha256']
            with np.load(path) as z: components.append(z['window_scores'].copy())
        result[peer] = np.column_stack(components)
    return features, result


def control_replay(features, originals):
    linear_summary = q.read(linear.OUT/'summary.json'); tree_summary = q.read(tree.OUT/'summary.json')
    report = []
    for peer, two in originals.items():
        family = peer+'__two_scores_and_citation'
        scaler_path = linear.OUT/(family+'_scaler.pkl')
        scaler = pickle.loads(scaler_path.read_bytes())
        x = scaler.transform(np.column_stack((two, features)))
        for entry in linear_summary['all_candidates'][family]:
            name = entry['candidate']; model_path = linear.OUT/(name+'.pkl'); score_path = linear.OUT/(name+'_scores.npz')
            assert q.sha(scaler_path) == entry['scaler_sha256']
            assert q.sha(model_path) == entry['model_sha256'] and q.sha(score_path) == entry['scores_sha256']
            model = pickle.loads(model_path.read_bytes())
            with np.load(score_path) as z: expected = z['window_scores'].copy()
            actual = model.predict_proba(x)[:, 1]
            assert np.array_equal(actual, expected)
            report.append({'candidate': name, 'max_abs_diff': 0., 'refitted': False})
        entry = tree_summary['methods'][peer]
        model_path = tree.OUT/(peer+'.pkl'); score_path = tree.OUT/(peer+'_scores.npz')
        assert q.sha(model_path) == entry['model_sha256'] and q.sha(score_path) == entry['scores_sha256']
        model = pickle.loads(model_path.read_bytes())
        with np.load(score_path) as z: expected = z['window_scores'].copy()
        assert np.array_equal(model.predict_proba(two)[:, 1], expected)
        report.append({'candidate': peer+'_monotone_tree', 'max_abs_diff': 0., 'refitted': False})
    assert len(report) == 12
    return report


def prepare():
    assert not (OUT/'preparation_complete.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT/'protocol.json', protocol())
    meta = q.metadata(); features, originals = load_original(meta)
    report = control_replay(features, originals)
    q.save(OUT/'CONTROL_REPLAY_CHECK.json', {'passed': True, 'controls': report, 'new_fits': 0})
    # Synthetic mapping checks use existing metadata, never a live/unfinished epoch.
    probabilities = {a['response_id']: np.linspace(0., 1., len(meta['by_response'][a['response_id']]['tokens']['lexical_mask']), dtype=np.float32)
        for a in meta['answers']}
    score = map_windows(meta, probabilities)
    assert len(score) == 210364 and np.isfinite(score).all()
    answer = q.answer_scores(meta, score)
    assert len(answer) == 793
    grouped = {}
    for w, value in zip(meta['windows'], score):
        grouped.setdefault(w['response_id'], []).append(value)
    for i, a in enumerate(meta['answers']):
        assert answer[i] == max(grouped[a['response_id']])
    q.save(OUT/'CPU_SELFCHECK.json', {'passed': True, 'all793_answer_max': True,
        'window_order_sha256': q.digest([w['window_id'] for w in meta['windows']]),
        'controls_replayed': 12, 'large_predictions_read': False, 'new_fits': 0, 'GPU_used': False})
    q.save(OUT/'preparation_complete.json', {'source_sha256': bindings(),
        'protocol_sha256': q.sha(OUT/'protocol.json'), 'CPU_check_sha256': q.sha(OUT/'CPU_SELFCHECK.json'),
        'control_check_sha256': q.sha(OUT/'CONTROL_REPLAY_CHECK.json'), 'ready_for_completed_large_only': True})
    print('LARGE_COMBINATION_PREPARED_12_CONTROLS_EXACT_NO_NEW_FIT', flush=True)


def map_windows(meta, probabilities):
    return np.asarray([max(float(probabilities[w['response_id']][j]) for j in w['token_indices']
        if meta['by_response'][w['response_id']]['tokens']['lexical_mask'][j]) for w in meta['windows']], np.float64)


def run():
    frozen = q.read(OUT/'preparation_complete.json')
    assert frozen['source_sha256'] == bindings()
    assert frozen['protocol_sha256'] == q.sha(OUT/'protocol.json') and q.read(OUT/'protocol.json') == protocol()
    assert frozen['CPU_check_sha256'] == q.sha(OUT/'CPU_SELFCHECK.json')
    assert frozen['control_check_sha256'] == q.sha(OUT/'CONTROL_REPLAY_CHECK.json')
    assert not (OUT/'started.json').exists()
    completed = q.read(LARGE/'complete.json')
    assert not completed['official_test_opened'] and [e['epoch'] for e in completed['all_epochs']] == list(range(7))
    chosen = max(completed['all_epochs'][1:], key=lambda e: e['selection_key'])
    assert chosen == completed['selected']
    path = LARGE/f"epoch_{chosen['epoch']:02d}_token_predictions.npz"
    assert q.sha(path) == chosen['artifacts_sha256']['_token_predictions.npz']
    meta = q.metadata(); features, originals = load_original(meta)
    with np.load(path) as z:
        assert len(z.files) == 3839
        probabilities = {a['response_id']: z[a['response_id']].copy() for a in meta['answers']}
    large = map_windows(meta, probabilities)
    assert q.metrics(meta, large, chosen['thresholds'])['calibration'] == chosen['calibration']
    base_weight, loss_weight, _, y = q.base_weights(meta)
    assert len(y) == 168123
    q.save(OUT/'started.json', {'time': time.time(), 'preparation_sha256': q.sha(OUT/'preparation_complete.json'),
        'large_complete_sha256': q.sha(LARGE/'complete.json'), 'large_tokens_sha256': q.sha(path), 'selected_epoch': chosen['epoch']})
    np.save(OUT/'large_window_probability.npy', large)
    lo, hi = meta['bounds']['calibration']; start = time.perf_counter(); candidates = {}; selected = {}

    def save_candidate(name, peer, mode, c, model, scaler, x):
        score = model.predict_proba(x)[:, 1]; answer = q.answer_scores(meta, score)
        thresholds = {'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], score[lo:hi]),
            'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], answer[634:])}
        entry = {'candidate': name, 'peer': peer, 'mode': mode, 'C': c,
            'input_width': x.shape[1], 'thresholds': thresholds, 'metrics': q.metrics(meta, score, thresholds),
            'selection_key': list(q.selection_key(thresholds, c or 0.)), 'large_epoch': chosen['epoch']}
        (OUT/(name+'.pkl')).write_bytes(pickle.dumps({'model': model, 'scaler': scaler}, protocol=5))
        np.savez_compressed(OUT/(name+'_scores.npz'), window_scores=score, answer_scores=answer)
        entry.update(model_sha256=q.sha(OUT/(name+'.pkl')), scores_sha256=q.sha(OUT/(name+'_scores.npz')))
        q.save(OUT/(name+'.json'), entry)
        print('LARGE_MATCHED_COMPLETE', name, entry['metrics']['calibration']['windows']['f1'], entry['metrics']['calibration']['answers']['f1'], flush=True)
        return entry

    for peer, two in originals.items():
        x = np.column_stack((two, features, large))
        assert x.shape == (210364, 11)
        scaler = StandardScaler().fit(x[:len(y)], sample_weight=base_weight); scaled = scaler.transform(x)
        entries = []
        for c in linear.CS:
            model = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20261010)
            model.fit(scaled[:len(y)], y, sample_weight=loss_weight)
            entries.append(save_candidate(f'{peer}__large_lr__C{c:g}', peer, 'large_lr', c, model, scaler, scaled))
        candidates[peer+'__large_lr'] = entries
        selected[peer+'__large_lr'] = max(entries, key=lambda e: e['selection_key'])
        hyper = {k: v for k, v in protocol()['tree']['hyperparameters'].items() if k != 'type'}
        model = HistGradientBoostingClassifier(**hyper); x = np.column_stack((two, large))
        model.fit(x[:len(y)], y, sample_weight=loss_weight)
        assert model.n_iter_ == 100 and model.n_features_in_ == 3
        entry = save_candidate(peer+'__large_tree', peer, 'large_tree', None, model, None, x)
        candidates[peer+'__large_tree'] = [entry]; selected[peer+'__large_tree'] = entry
    q.save(OUT/'summary.json', {'selected': selected, 'all_candidates': candidates,
        'large_selected': chosen, 'new_LR_fits': 9, 'new_tree_fits': 3,
        'control_refits': 0, 'seconds': time.perf_counter()-start, 'official_test_opened': False, 'GPU_used': False})
    report = ['# 各基线获得相同large信号', '', '| 对象 | 组合 | 定位F1 | 整答F1 |', '|---|---|---:|---:|']
    for entry in selected.values():
        m = entry['metrics']['calibration']
        report.append(f"| {entry['peer']} | {entry['mode']} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report += ['', '同一已固定large轮次，三对象各三档LR和一次固定单调树；原控制未重训。',
        '这里仍是反复开发的校准集、额外离线语义核查融合；原4BPE窗口/标签/完整回答max未改，官方测试未打开。',
        '须逐候选对照原强基线和large单独成绩，不能拼接两个最优指标。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    q.save(OUT/'complete.json', {'summary_sha256': q.sha(OUT/'summary.json'), 'new_fits': 12,
        'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run')); args = p.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[args.stage]()
