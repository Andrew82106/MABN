"""Fixed, matched answer-level conditioning of completed window detectors."""
from pathlib import Path
import argparse
import time
import numpy as np
from scipy.special import expit, logit
import run_development as q

OUT = q.ROOT / 'results/answer_conditioned_scores_v1'
LOCAL = q.ROOT / 'results/citation_alignment_lr_v1'
GLOBAL = q.ROOT / 'results/completed_score_combiner_v1'
PEERS = ('lookback', 'harp_claim', 'semantic_claim')
ALPHAS = (0., .2, .4, .6, .8, 1.)
MODES = ('own_answer_control', 'shared_lookback_tree_answer')


def protocol():
    return {
        'version': 'answer-conditioned-window-risk-v1',
        'local': 'Each already-selected two_scores_and_citation LR; do not reopen its C selection.',
        'shared_global': 'Already-completed Lookback+tail2 monotone tree answer-max score; currently strongest answer detector. Same score for all three peers.',
        'own_control': 'Use each local detector own answer-max score, to separate answer-wise rescaling from new global information.',
        'formula': 'sigmoid(logit(local_window)+alpha*logit(global_answer)). Broadcast only within that answer. Clip probabilities to [1e-6,1-1e-6] for logits; alpha=0 returns exact original scores.',
        'alpha': list(ALPHAS), 'peers': list(PEERS), 'modes': list(MODES),
        'selection': 'Same6 alpha per family, original cal-only separate F1 thresholds; q.selection_key selects one alpha for both levels.',
        'evaluation': 'Unchanged793 answers/210364 windows, original634fit/159cal; all original eligible4BPE windows. Answer score remains max final window score.',
        'training': 'No new training or model inference. Existing upstream heads are trained; not cross-fitted.',
        'limits': 'Repeatedly used calibration and prior best selection remain optimistic development. Added answer information cannot by itself identify the wrong words inside a risky answer.',
        'official_test_opened': False, 'GPU_used': False,
    }


def sources():
    return {str(p.resolve()): q.sha(p) for p in
            [Path(__file__), Path(q.__file__), LOCAL/'summary.json', LOCAL/'complete.json',
             GLOBAL/'summary.json', GLOBAL/'complete.json', q.DATA/'gold_manifest.json']}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    q.save(OUT/'protocol.json', protocol())
    q.save(OUT/'design_freeze.json', {'files_sha256': sources(), 'trained': False})
    print('ANSWER_CONDITIONED_DESIGN_FROZEN', flush=True)


def load_score(directory, name, entry, meta):
    path = directory/(name+'_scores.npz')
    assert q.sha(path) == entry['scores_sha256']
    with np.load(path, allow_pickle=False) as z:
        score, answer = z['window_scores'].copy(), z['answer_scores'].copy()
    assert score.shape == (210364,) and answer.shape == (793,)
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    assert np.array_equal(answer, q.answer_scores(meta, score))
    assert q.metrics(meta, score, entry['thresholds']) == entry['metrics']
    return score, answer


def run():
    assert q.read(OUT/'protocol.json') == protocol()
    assert q.read(OUT/'design_freeze.json')['files_sha256'] == sources()
    assert not (OUT/'started.json').exists()
    meta = q.metadata()
    lo, hi = meta['bounds']['calibration']
    local_entries = q.read(LOCAL/'summary.json')['selected']
    global_entry = q.read(GLOBAL/'summary.json')['methods']['lookback']
    _, global_answer = load_score(GLOBAL, 'lookback', global_entry, meta)
    loaded = {}
    for peer in PEERS:
        entry = local_entries[peer+'__two_scores_and_citation']
        loaded[peer] = (*load_score(LOCAL, entry['candidate'], entry, meta), entry)
    # Build from identifiers; answer/window ordering is never inferred from lengths.
    answer_index = {a['response_id']: i for i, a in enumerate(meta['answers'])}
    window_answer = np.asarray([answer_index[w['response_id']] for w in meta['windows']])
    assert np.bincount(window_answer, minlength=793).min() > 0
    q.save(OUT/'started.json', {'time': time.time(), 'design_sha256': q.sha(OUT/'design_freeze.json')})
    start = time.perf_counter()
    selected, families = {}, {}
    for peer, (original, own_answer, original_entry) in loaded.items():
        for mode in MODES:
            global_score = own_answer if mode == MODES[0] else global_answer
            broadcast = global_score[window_answer]
            entries = []
            for alpha in ALPHAS:
                score = original.copy() if alpha == 0 else expit(
                    logit(np.clip(original, 1e-6, 1-1e-6)) +
                    alpha*logit(np.clip(broadcast, 1e-6, 1-1e-6)))
                answer = q.answer_scores(meta, score)
                thresholds = {
                    'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], score[lo:hi]),
                    'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], answer[634:])}
                metrics = q.metrics(meta, score, thresholds)
                if alpha == 0:
                    assert np.array_equal(original, score)
                    assert thresholds == original_entry['thresholds'] and metrics == original_entry['metrics']
                candidate = f'{peer}__{mode}__a{alpha:g}'
                path = OUT/(candidate+'_scores.npz')
                np.savez_compressed(path, window_scores=score, answer_scores=answer)
                entry = {'candidate': candidate, 'peer': peer, 'mode': mode, 'alpha': alpha,
                         'thresholds': thresholds, 'metrics': metrics,
                         'selection_key': list(q.selection_key(thresholds, alpha)),
                         'scores_sha256': q.sha(path)}
                q.save(OUT/(candidate+'.json'), entry)
                entries.append(entry)
            family = peer+'__'+mode
            families[family] = entries
            selected[family] = max(entries, key=lambda e: e['selection_key'])
            m = selected[family]['metrics']['calibration']
            print('ANSWER_CONDITIONED_COMPLETE', family, selected[family]['alpha'],
                  m['windows']['f1'], m['answers']['f1'], flush=True)
    assert sum(map(len, families.values())) == 36
    q.save(OUT/'summary.json', {'selected': selected, 'all_candidates': families,
        'seconds': time.perf_counter()-start, 'new_fits': 0,
        'official_test_opened': False, 'GPU_used': False})
    report = ['# 整答风险对局部窗口的匹配校正', '',
        '三个已完成的局部组合都比较相同六档权重。控制只用自身整答分数，候选共同加入已完成Lookback树的整答分数。', '',
        '| 对象 | 整答来源 | alpha | 窗口F1 | 整答F1 |', '|---|---|---:|---:|---:|']
    for e in selected.values():
        m = e['metrics']['calibration']
        report.append(f"| {e['peer']} | {e['mode']} | {e['alpha']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    report += ['', '原159答校准集已反复使用，仅为开发结果；官方测试未读。最终整答仍取全部窗口最大值，未分开挑两套输出。',
        '没有新训练或推理；所有旧选中模型固定，上游训练内分数没有交叉拟合。各窗口同加整答logit不改变答内定位顺序。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    q.save(OUT/'complete.json', {'summary_sha256': q.sha(OUT/'summary.json'),
                               'candidates': 36, 'official_test_opened': False, 'GPU_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[p.parse_args().stage]()
