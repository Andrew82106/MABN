"""Condition likely-risk windows on answer evidence without uniform broadcasting.

Fixed gate, no new training or inference. Same six-alpha budget for every peer;
the completed uniform-broadcast control is reused exactly.
"""
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.special import expit, logit
import run_development as q
import run_answer_conditioned_scores as prior

OUT = q.ROOT / 'results/local_gated_answer_scores_v1'
POWER = 8


def protocol():
    return {
        'version': 'local-gated-answer-risk-v1', 'peers': list(prior.PEERS),
        'alpha': list(prior.ALPHAS), 'new_score_candidates': 18, 'new_fits': 0,
        'hypothesis': 'Uniform answer-level logit shifts can increase false alarms in otherwise-supported parts of risky answers. Restrict that correction using an already-estimated local-risk gate.',
        'gate': '(local_window_probability / max_local_probability_in_same_answer)**8. Fixed power8, no gate search. This uses predictions only, never gold or sentence selection.',
        'formula': 'sigmoid(logit(local_window)+alpha*gate*logit(shared_global_answer)). Shared global is the same completed Lookback+tail2 tree. Clip logits at1e-6; alpha0 exact old scores.',
        'control': 'Reuse all18 completed shared_lookback_tree_answer uniform-broadcast candidates at identical six alphas. Verify stored score/metrics/hash. No control refit.',
        'selection': 'Same original159cal thresholds and q.selection_key: min twoF1, windowF1, windowprecision, smalleralpha. Select one alpha for both levels.',
        'scope': 'All793 answers/210364 original4BPE windows, original634fit/159cal. Answer remains max over ALL final windows; no separate answer override or scoring-unit change.',
        'limits': ['The gate can amplify an incorrectly located local peak; global evidence does not prove that peak is the actual error.',
                   'This is offline extra-semantic-model score fusion, not a pure generator probe.',
                   'Repeatedly viewed calibration and in-sample upstream fitting remain optimistic development; no heldout guarantee.'],
        'official_test_opened': False, 'GPU_used': False,
    }


def source_files():
    paths = [Path(__file__), Path(prior.__file__), Path(q.__file__), q.DATA/'gold_manifest.json',
             prior.OUT/'summary.json', prior.OUT/'complete.json',
             prior.LOCAL/'summary.json', prior.GLOBAL/'summary.json']
    return {str(p.resolve()): q.sha(p) for p in paths}


def prepare():
    assert not (OUT/'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    p = np.asarray([.1, .3, .8], np.float64)
    gate = (p/p.max())**POWER
    assert gate[-1] == 1 and np.all(np.diff(gate) > 0) and gate[0] < .001
    assert np.array_equal(p.copy(), p)
    q.save(OUT/'protocol.json', protocol())
    q.save(OUT/'design_freeze.json', {'source_sha256': source_files(),
        'CPU_gate_monotonicity_and_bounds_passed': True, 'trained': False})
    print('LOCAL_GATE_DESIGN_FROZEN', flush=True)


def run():
    assert q.read(OUT/'protocol.json') == protocol()
    assert q.read(OUT/'design_freeze.json')['source_sha256'] == source_files()
    assert not (OUT/'started.json').exists()
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    ai = {a['response_id']: i for i, a in enumerate(meta['answers'])}
    wa = np.asarray([ai[w['response_id']] for w in meta['windows']])
    cal_y = [w['label'] for w in meta['windows'][lo:hi]]
    cal_ay = [a['label'] for a in meta['answers'][634:]]
    global_entry = q.read(prior.GLOBAL/'summary.json')['methods']['lookback']
    _, global_answer = prior.load_score(prior.GLOBAL, 'lookback', global_entry, meta)
    old = q.read(prior.OUT/'summary.json')
    local_entries = q.read(prior.LOCAL/'summary.json')['selected']
    q.save(OUT/'started.json', {'time': time.time(), 'design_sha256': q.sha(OUT/'design_freeze.json')})
    selected, families, controls = {}, {}, {}
    start = time.perf_counter()
    for peer in prior.PEERS:
        original_entry = local_entries[peer+'__two_scores_and_citation']
        local, own = prior.load_score(prior.LOCAL, original_entry['candidate'], original_entry, meta)
        assert own.min() > 0
        gate = (local/own[wa])**POWER
        assert np.isfinite(gate).all() and ((gate >= 0) & (gate <= 1)).all()
        assert np.array_equal(q.answer_scores(meta, gate), np.ones(793))
        uniform = old['all_candidates'][peer+'__shared_lookback_tree_answer']
        assert [e['alpha'] for e in uniform] == list(prior.ALPHAS)
        entries = []
        for alpha, old_entry in zip(prior.ALPHAS, uniform):
            old_scores, _ = prior.load_score(prior.OUT, old_entry['candidate'], old_entry, meta)
            global_z = logit(np.clip(global_answer[wa], 1e-6, 1-1e-6))
            local_z = logit(np.clip(local, 1e-6, 1-1e-6))
            uniform_replay = local.copy() if alpha == 0 else expit(local_z+alpha*global_z)
            assert np.array_equal(old_scores, uniform_replay)
            score = local.copy() if alpha == 0 else expit(local_z+alpha*gate*global_z)
            answer = q.answer_scores(meta, score)
            thresholds = {'window': q.choose_threshold(cal_y, score[lo:hi]),
                          'answer': q.choose_threshold(cal_ay, answer[634:])}
            metrics = q.metrics(meta, score, thresholds)
            if alpha == 0:
                assert np.array_equal(score, local) and thresholds == original_entry['thresholds']
                assert metrics == original_entry['metrics']
            name = f'{peer}__gate8__a{alpha:g}'
            path = OUT/(name+'_scores.npz')
            np.savez_compressed(path, window_scores=score, answer_scores=answer)
            e = {'candidate': name, 'peer': peer, 'alpha': alpha, 'power': POWER,
                 'thresholds': thresholds, 'metrics': metrics,
                 'selection_key': list(q.selection_key(thresholds, alpha)),
                 'scores_sha256': q.sha(path)}
            q.save(OUT/(name+'.json'), e); entries.append(e)
        families[peer] = entries
        selected[peer] = max(entries, key=lambda e: e['selection_key'])
        controls[peer] = max(uniform, key=lambda e: e['selection_key'])
        m = selected[peer]['metrics']['calibration']
        print('LOCAL_GATE_COMPLETE', peer, selected[peer]['alpha'], m['windows']['f1'], m['answers']['f1'], flush=True)
    q.save(OUT/'summary.json', {'selected': selected, 'uniform_controls': controls,
        'all_candidates': families, 'uniform_control_candidates': {
            p: old['all_candidates'][p+'__shared_lookback_tree_answer'] for p in prior.PEERS},
        'all18_uniform_scores_exact': True, 'seconds': time.perf_counter()-start,
        'new_fits': 0, 'official_test_opened': False, 'GPU_used': False})
    report = ['# 局部风险门控的整答校正', '',
        '固定8次幂门控，使已有局部低风险位置受到较小的整答分数改动。三对象均同六档权重；均匀广播控制原18份分数精确复现。', '',
        '| 对象 | 均匀控制：窗口/整答 | 门控alpha | 门控窗口F1 | 门控整答F1 |', '|---|---:|---:|---:|---:|']
    for peer, e in selected.items():
        c, m = controls[peer]['metrics']['calibration'], e['metrics']['calibration']
        report.append(f"| {peer} | {c['windows']['f1']:.6f}/{c['answers']['f1']:.6f} | {e['alpha']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report += ['', '所有原窗口保留，整答仍取最终窗口最大值；没有另接整答输出或改评测单位。门控依据预测，不能保证被增强的位置就是实际错误。',
        '无新训练或推理，仍含额外语义核查模型；原159答反复用于开发，官方测试未读，不是独立测试或稳定胜出证据。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    q.save(OUT/'complete.json', {'summary_sha256': q.sha(OUT/'summary.json'),
        'new_candidates': 18, 'control_candidates_exact': 18, 'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[p.parse_args().stage]()
