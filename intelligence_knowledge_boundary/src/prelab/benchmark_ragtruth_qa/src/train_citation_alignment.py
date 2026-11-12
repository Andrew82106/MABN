"""Matched small linear detectors with/without source-citation alignment."""
from pathlib import Path
import argparse
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_development as q
import run_completed_score_fusion as scores_source

FEATURES = q.ROOT / 'results/citation_alignment_v1'
OUT = q.ROOT / 'results/citation_alignment_lr_v1'
CS = (.001, .01, .1)
MODES = ('two_scores_only', 'two_scores_and_citation')


def protocol():
    return {'version': 'citation-alignment-matched-lr-v1', 'peers': list(scores_source.PEERS),
        'modes': list(MODES), 'C': list(CS), 'fits': 18,
        'training': 'Same original634 fit/168123 windows. Both modes use q.base_weights base and loss weights unchanged; no new labels.',
        'base_inputs': 'Frozen peer and tail2 probabilities for the exact original4BPE windows, no raw source ID or generator ID.',
        'additional_inputs': 'All predeclared CPU citation alignment features, no type label or human span metadata.',
        'standardization': 'StandardScaler fit-only, sample_weight=original base weights, same deterministic ordering.',
        'LR': {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'seed': 20261010,
               'sample_weight': 'Original source group/class loss, total168123.'},
        'calibration': 'Same159 answers/42241 windows; separate F1-optimal thresholds, precision/higher-threshold ties. Choose C per peer/mode using existing q.selection_key; no gradients or fit using cal.',
        'answer': 'Maximum over every original eligible window, original human answer label; no citation-only filtering.',
        'scope': 'Offline original native634 subset. Tail2 base model trained3680; peers trained634. Same base supervision for both feature modes.',
        'limits': ['Upstream scores on fit are in-sample, not cross-fitted stacking.',
            'Lexical evidence overlap is an inexpensive cue, not semantic proof or a new gold label.',
            'Features cannot check world truth beyond the supplied evidence.',
            'Already repeatedly selected development calibration; final test remains unopened.'],
        'GPU_used': False, 'official_test_opened': False}


def prepare():
    assert not (OUT / 'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    print('MATCHED_CITATION_LR_PROTOCOL_FROZEN', flush=True)


def run():
    cfg = q.read(OUT / 'protocol.json'); assert cfg == protocol()
    if not (FEATURES / 'preparation_complete.json').exists():
        print('WAIT_COMPLETE_CITATION_FEATURES_NO_FIT', flush=True)
        return
    assert not (OUT / 'started.json').exists()
    fm = q.read(FEATURES / 'preparation_complete.json')
    for name, sha in fm['files_sha256'].items(): assert q.sha(FEATURES / name) == sha, name
    meta = q.metadata()
    names = q.read(FEATURES / 'feature_names.json')
    new = np.load(FEATURES / 'window_features.npy')
    assert new.shape == (210364, len(names)) and np.isfinite(new).all()
    assert fm['window_order_sha256'] == q.digest([w['window_id'] for w in meta['windows']])
    base_weight, loss_weight, _, y = q.base_weights(meta)
    assert len(y) == 168123
    completed = q.read(scores_source.OUT / 'complete.json')
    assert q.sha(scores_source.OUT / 'summary.json') == completed['summary_sha256']
    sources = q.read(scores_source.OUT / 'summary.json')
    q.save(OUT / 'started.json', {'code_sha256': q.sha(Path(__file__)),
        'feature_preparation_sha256': q.sha(FEATURES / 'preparation_complete.json'),
        'base_scores_complete_sha256': q.sha(scores_source.OUT / 'complete.json'),
        'protocol_sha256': q.sha(OUT / 'protocol.json'), 'feature_names': names})
    all_candidates, selected = {}, {}; start = time.perf_counter()
    for peer in scores_source.PEERS:
        inputs = []
        for alpha in (0., 1.):
            e = next(e for e in sources['all_candidates'][peer] if e['tail_weight'] == alpha)
            p = scores_source.OUT / (e['candidate'] + '_scores.npz')
            assert q.sha(p) == e['scores_sha256']
            with np.load(p) as z: inputs.append(z['window_scores'].copy())
        original = np.column_stack(inputs)
        for mode in MODES:
            x = original if mode == 'two_scores_only' else np.column_stack((original, new))
            scaler = StandardScaler().fit(x[:len(y)], sample_weight=base_weight)
            scaled = scaler.transform(x)
            family = peer + '__' + mode
            (OUT / (family + '_scaler.pkl')).write_bytes(pickle.dumps(scaler, protocol=5))
            entries = []
            for c in CS:
                tick = time.perf_counter()
                model = LogisticRegression(C=c, solver='liblinear', penalty='l2', max_iter=2000, random_state=20261010)
                model.fit(scaled[:len(y)], y, sample_weight=loss_weight)
                score = model.predict_proba(scaled)[:, 1]
                aa = q.answer_scores(meta, score)
                lo, hi = meta['bounds']['calibration']
                ts = {'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], score[lo:hi]),
                      'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], aa[634:])}
                metrics = q.metrics(meta, score, ts)
                candidate = family + f'__C{c:g}'
                (OUT / (candidate + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
                np.savez_compressed(OUT / (candidate + '_scores.npz'), window_scores=score, answer_scores=aa)
                entry = {'candidate': candidate, 'peer': peer, 'mode': mode, 'C': c,
                    'thresholds': ts, 'metrics': metrics, 'selection_key': list(q.selection_key(ts, c)),
                    'input_width': x.shape[1], 'iterations': model.n_iter_.tolist(),
                    'seconds': time.perf_counter() - tick,
                    'model_sha256': q.sha(OUT / (candidate + '.pkl')),
                    'scaler_sha256': q.sha(OUT / (family + '_scaler.pkl')),
                    'scores_sha256': q.sha(OUT / (candidate + '_scores.npz'))}
                q.save(OUT / (candidate + '.json'), entry); entries.append(entry)
                print('CITATION_LR_COMPLETE', candidate, metrics['calibration']['windows']['f1'], metrics['calibration']['answers']['f1'], flush=True)
            all_candidates[family] = entries
            selected[family] = max(entries, key=lambda e: e['selection_key'])
    q.save(OUT / 'summary.json', {'selected': selected, 'all_candidates': all_candidates,
        'fits_completed': 18, 'seconds': time.perf_counter() - start,
        'official_test_opened': False, 'GPU_used': False})
    report = ['# 引用来源对应特征的匹配对照', '', '三个底层对象、两种输入，各相同三档C；原训练/校准窗口和标签全部保留。', '',
        '| 对象 | 输入 | C | 定位F1 | 整答F1 |', '|---|---|---:|---:|---:|']
    for e in selected.values():
        m = e['metrics']['calibration']
        report.append(f"| {e['peer']} | {e['mode']} | {e['C']:g} | {m['windows']['f1']:.4f} | {m['answers']['f1']:.4f} |")
    report += ['', '来源覆盖仅为词面特征，不据此自动重打标签；有无引用的全部回答都进入评测。底层训练分数不是交叉拟合，因此仍可能过拟合。',
        '本表仍是反复开发的159答，不能作为独立测试或SOTA结论。还须与此前更强的固定权重/单调树组合比较，不能只报弱两分数LR对照。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'), 'fits_completed': 18,
        'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[args.stage]()
