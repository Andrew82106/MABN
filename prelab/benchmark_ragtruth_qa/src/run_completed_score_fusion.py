"""Matched five-weight window score fusion for already completed detectors."""
from pathlib import Path
import time
import numpy as np
import run_development as q

OUT = q.ROOT / 'results/completed_score_fusion_v1'
ALPHAS = (0., .25, .5, .75, 1.)
PEERS = {
    'lookback': ('lookback_regularization_v2', 'lb_prefix_pre_header'),
    'harp_claim': ('claim_pooling_v1', 'full_lb_harp64_tcn'),
    'semantic_claim': ('claim_pooling_v1', 'minicheck_hidden64_risk_tcn_w32'),
}


def protocol():
    return {'version': 'completed-window-probability-fusion-v1', 'tail_weight': list(ALPHAS),
        'peers': {k: list(v) for k, v in PEERS.items()},
        'selection_before_fusion': 'Use each upstream already selected model/epoch and existing claim alpha; never reopen its epoch search.',
        'formula': 'window_probability=(1-alpha)*peer_probability+alpha*tail2_probability. This combines original four-BPE window scores, not raw token probabilities.',
        'tail': 'Completed all-doc MiniCheck tail2 selected epoch; trained on3680 QA answers.',
        'peers_training': 'All peers trained on634 original QA fit answers; tail adds the same3680 supervision to each combination.',
        'answer_score': 'Maximum over all original eligible windows, no gold filtering.',
        'scope': 'Original native634 fit subset and159 calibration. Fit is not full3680. No official test or new inference.',
        'threshold': 'Each candidate gets the same cal-only separate window/answer F1 thresholds; ties precision then higher cutoff.',
        'weight_selection': 'Same5 weights for every peer. Max min(windowF1,answerF1),windowF1,windowPrecision,then smaller alpha.',
        'limits': ['Uses repeatedly viewed development calibration; improvement does not prove held-out improvement.',
            'Endpoints include each original model. A selected calibration score cannot establish a guarantee on future data.',
            'Extra semantic checkers and whole-answer processing remain offline; not a pure generator probe.',
            'All3 peers receive the same tail signal and five-weight budget. No baseline is deliberately held at an inferior configuration.'],
        'official_test_opened': False, 'GPU_used': False}


def prepare():
    assert not (OUT / 'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    print('FIXED_FIVE_WEIGHT_COMPARISON_PREPARED', flush=True)


def run():
    assert q.read(OUT / 'protocol.json') == protocol()
    assert not (OUT / 'started.json').exists()
    meta = q.metadata()
    lo, hi = meta['bounds']['calibration']
    directory = q.ROOT / 'results/minicheck_tail_all_docs_v3/tail2'
    completed = q.read(directory / 'complete.json')
    assert not completed['test_opened']
    tail = completed['selected']
    path = directory / f"epoch_{tail['epoch']:02d}_token_predictions.npz"
    assert q.sha(path) == tail['artifacts_sha256']['_token_predictions.npz']
    with np.load(path) as z:
        assert len(z.files) == 3839
        # Reproduce the upstream lexical maximum; punctuation keeps the width
        # but does not enter the token-risk maximum. Read each answer once.
        probabilities = {a['response_id']: z[a['response_id']] for a in meta['answers']}
    tail_scores = np.asarray([
        max(float(probabilities[w['response_id']][j]) for j in w['token_indices']
            if meta['by_response'][w['response_id']]['tokens']['lexical_mask'][j])
        for w in meta['windows']], np.float64)
    assert q.metrics(meta, tail_scores, tail['thresholds'])['calibration'] == tail['calibration']
    loaded, bindings = {}, {'tail_complete_sha256': q.sha(directory / 'complete.json'),
        'tail_token_scores_sha256': q.sha(path), 'tail_selected_epoch': tail['epoch']}
    for name, (folder, method) in PEERS.items():
        directory = q.ROOT / 'results' / folder
        entry = q.read(directory / 'summary.json')['selected'][method]
        path = directory / (entry['candidate'] + '_scores.npz')
        assert q.sha(path) == entry['scores_sha256']
        with np.load(path) as z:
            key = 'window_scores' if 'window_scores' in z else 'scores'
            scores = z[key].astype(np.float64)
        assert len(scores) == 210364 and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        assert q.metrics(meta, scores, entry['thresholds']) == entry['metrics']
        loaded[name] = (entry, scores)
        bindings[name] = {'summary_sha256': q.sha(directory / 'summary.json'),
                          'scores_sha256': q.sha(path), 'candidate': entry['candidate']}
    q.save(OUT / 'started.json', {'source_bindings': bindings, 'code_sha256': q.sha(Path(__file__)),
        'protocol_sha256': q.sha(OUT / 'protocol.json'), 'time': time.time()})
    families, selected = {}, {}
    cal_y = [w['label'] for w in meta['windows'][lo:hi]]
    cal_ay = [a['label'] for a in meta['answers'][634:]]
    for name, (original, peer_scores) in loaded.items():
        candidates = []
        for alpha in ALPHAS:
            scores = (1 - alpha) * peer_scores + alpha * tail_scores
            answer = q.answer_scores(meta, scores)
            thresholds = {'window': q.choose_threshold(cal_y, scores[lo:hi]),
                          'answer': q.choose_threshold(cal_ay, answer[634:])}
            metrics = q.metrics(meta, scores, thresholds)
            if alpha == 0:
                assert np.array_equal(scores, peer_scores) and metrics == original['metrics']
                assert thresholds == original['thresholds']
            if alpha == 1:
                assert np.array_equal(scores, tail_scores) and metrics['calibration'] == tail['calibration']
                assert thresholds == tail['thresholds']
            candidate = f'{name}_tail{alpha:g}'
            np.savez_compressed(OUT / (candidate + '_scores.npz'), window_scores=scores, answer_scores=answer)
            entry = {'candidate': candidate, 'peer': name, 'tail_weight': alpha,
                'thresholds': thresholds, 'metrics': metrics, 'fit_scope': 'original_native634_subset',
                'selection_key': list(q.selection_key(thresholds, alpha)),
                'scores_sha256': q.sha(OUT / (candidate + '_scores.npz'))}
            q.save(OUT / (candidate + '.json'), entry)
            candidates.append(entry)
        families[name] = candidates
        selected[name] = max(candidates, key=lambda e: e['selection_key'])
    q.save(OUT / 'summary.json', {'selected': selected, 'all_candidates': families,
        'source_bindings': bindings, 'new_neural_training': False, 'GPU_used': False,
        'official_test_opened': False, 'development_selection_optimistic': True})
    report = ['# 相同预算的窗口分数组合', '', '每项基线都加入相同的已训练尾层核查信号，分别比较相同的五档权重。', '',
        '| 组合对象 | 尾层权重 | 定位F1 | 整答F1 |', '|---|---:|---:|---:|']
    for name, entries in families.items():
        for e in entries:
            m = e['metrics']['calibration']
            label = name + ('（选中）' if e == selected[name] else '')
            report.append(f"| {label} | {e['tail_weight']:g} | {m['windows']['f1']:.4f} | {m['answers']['f1']:.4f} |")
    report += ['', '这些是同一159答、42241原4BPE窗口的开发成绩。每答分数仍由全部窗口最大值获得，正常回答全部计入。',
        '只选择组合权重与阈值，原模型/轮次已先固定，未借此重新挑轮次。两端逐值复现原模型。',
        '各组合的尾层模型用3680答训练，另一模型用634答训练；额外监督量对三个组合一致。fit成绩仅对应原634子集。',
        '开发集已反复用于选型，不能将这次比较当作新测试或保证未来超过基线。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'), 'official_test_opened': False})
    print((OUT / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'run'])
    args = p.parse_args()
    {'prepare': prepare, 'run': run}[args.stage]()
