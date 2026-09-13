"""Fit one small monotone two-score combiner per peer; no GPU."""
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_development as q
import run_completed_score_fusion as source

OUT = q.ROOT / 'results/completed_score_combiner_v1'


def protocol():
    return {'version': 'monotone-two-score-combiner-v1', 'peers': list(source.PEERS),
        'inputs': ['existing peer four-BPE window probability', 'selected tail2 four-BPE window probability'],
        'training': 'Original native634 fit answers/168123 windows only; use unchanged q.base_weights loss and source groups.',
        'model': {'type': 'HistGradientBoostingClassifier', 'max_iter': 100, 'max_depth': 2,
                  'max_leaf_nodes': 4, 'min_samples_leaf': 20, 'learning_rate': .05,
                  'l2_regularization': 1., 'early_stopping': False, 'monotonic_cst': [1, 1],
                  'random_state': 20261008},
        'monotonicity': 'Increasing either input risk cannot reduce predicted risk.',
        'budget': 'Exactly one fixed100-step fit per peer, three fits total; no architecture/hyperparameter search.',
        'calibration': 'Original159 answers/42241 windows only set separate F1-optimal thresholds; no training gradients or early stopping.',
        'answer': 'Maximum over every original eligible window; unchanged human gold and4rawBPE geometry.',
        'limits': ['Upstream models also trained on the634 fit answers; these are in-sample base predictions, not cross-fitted stacking. Development generalization may fail.',
            'Tail base trained3680; each peer trained634. Same input supervision is available to all3 combiners.',
            'This is an offline supervised window-score combiner with extra semantic checking, not a new native attention mechanism.',
            'Calibration has been repeatedly inspected. Keep all failures and compare with stronger completed fixed-weight combinations.'],
        'official_test_opened': False, 'GPU_used': False}


def prepare():
    assert not (OUT / 'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    print('THREE_FIXED_MONOTONE_FITS_PREPARED', flush=True)


def run():
    cfg = q.read(OUT / 'protocol.json'); assert cfg == protocol()
    assert not (OUT / 'started.json').exists()
    done = q.read(source.OUT / 'complete.json')
    assert q.sha(source.OUT / 'summary.json') == done['summary_sha256']
    previous = q.read(source.OUT / 'summary.json')
    meta = q.metadata(); _, weights, _, y = q.base_weights(meta)
    assert len(y) == 168123 and abs(weights.sum() - 168123) < 1e-7
    q.save(OUT / 'started.json', {'code_sha256': q.sha(Path(__file__)),
        'source_complete_sha256': q.sha(source.OUT / 'complete.json'), 'protocol_sha256': q.sha(OUT / 'protocol.json')})
    models = {}; start = time.perf_counter()
    for peer in source.PEERS:
        entries = previous['all_candidates'][peer]
        components = []
        for alpha in (0., 1.):
            e = next(e for e in entries if e['tail_weight'] == alpha)
            path = source.OUT / (e['candidate'] + '_scores.npz')
            assert q.sha(path) == e['scores_sha256']
            with np.load(path) as z: components.append(z['window_scores'].copy())
        x = np.column_stack(components)
        hyper = {k: v for k, v in cfg['model'].items() if k != 'type'}
        model = HistGradientBoostingClassifier(**hyper)
        tick = time.perf_counter()
        model.fit(x[:len(y)], y, sample_weight=weights)
        assert model.n_iter_ == 100 and model.n_features_in_ == 2
        scores = model.predict_proba(x)[:, 1]
        answer = q.answer_scores(meta, scores)
        lo, hi = meta['bounds']['calibration']
        thresholds = {'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], scores[lo:hi]),
                      'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], answer[634:])}
        metrics = q.metrics(meta, scores, thresholds)
        (OUT / (peer + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
        np.savez_compressed(OUT / (peer + '_scores.npz'), window_scores=scores, answer_scores=answer)
        entry = {'peer': peer, 'thresholds': thresholds, 'metrics': metrics, 'iterations': model.n_iter_,
            'seconds': time.perf_counter() - tick, 'parameters': hyper,
            'model_sha256': q.sha(OUT / (peer + '.pkl')), 'scores_sha256': q.sha(OUT / (peer + '_scores.npz')),
            'same_peer_best_fixed_weight': previous['selected'][peer]['metrics']['calibration']}
        q.save(OUT / (peer + '.json'), entry); models[peer] = entry
        print('MONOTONE_COMBINER_COMPLETE', peer, metrics['calibration']['windows']['f1'], metrics['calibration']['answers']['f1'], flush=True)
    q.save(OUT / 'summary.json', {'methods': models, 'seconds': time.perf_counter() - start,
        'official_test_opened': False, 'GPU_used': False, 'fixed_fits': 3})
    report = ['# 训练两分数的非线性组合', '', '三个对象各做一次相同预算的固定训练；不按校准成绩调整树深或训练轮数。', '',
        '| 对象 | 训练定位F1 | 校准定位F1 | 校准整答F1 | 原固定权重组合定位F1 |', '|---|---:|---:|---:|---:|']
    for peer, e in models.items():
        m = e['metrics']
        report.append(f"| {peer} | {m['fit']['windows']['f1']:.4f} | {m['calibration']['windows']['f1']:.4f} | {m['calibration']['answers']['f1']:.4f} | {e['same_peer_best_fixed_weight']['windows']['f1']:.4f} |")
    report += ['', '输入只有两种已有窗口风险。约束任一风险升高不能降低组合风险。学习过程使用原634训练回答；校准159回答只选阈值。',
        '底层模型也见过这634答，因此不属于交叉拟合的堆叠训练；训练拟合更好不意味着会泛化。原人工标签、窗口、拒答和测试封存规则均未改变。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'), 'fixed_fits_completed': 3, 'official_test_opened': False})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[args.stage]()
