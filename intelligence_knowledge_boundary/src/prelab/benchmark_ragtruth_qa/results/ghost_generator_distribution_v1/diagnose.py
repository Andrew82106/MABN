"""Descriptive saved-score analysis; no fitting, threshold selection or test."""
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'fit_expansion'))
import run_development as q
import run_probe_expansion as expansion

LR = ROOT / 'results/ghost_matched_lr_v1'
TREE = ROOT / 'results/ghost_nll_nonlinear_v1'
MODELS = [('ghost_lr', LR, 'ghost_only__ghost_C0.0001'),
          ('nll_tree', TREE, 'nll_only'), ('ghost_tree', TREE, 'ghost_only'),
          ('nll_ghost_tree', TREE, 'nll_ghost')]


def run():
    assert not (OUT / 'complete.json').exists(), 'Keep completed diagnostics unchanged.'
    protocol = {'version': 'ghost-generator-distribution-v1',
        'methods': [m[0] for m in MODELS],
        'partitions': ['native634_fit', 'added3046_fit', 'cal159'],
        'added_generators': 'Exclusive canonical model field from frozen fit.jsonl; report multi-generator dedup provenance separately. Generator names are analysis metadata, not model features.',
        'metrics': 'Unweighted descriptive window and answer positive rates, AUROC/AP, confusion and F1 at each already-saved calibration threshold. Same all-eligible-window answermax.',
        'no_threshold_selection': True, 'no_bootstrap': True, 'new_fits': 0, 'GPU_used': False,
        'test_opened': False, 'limits': 'Native/added are training subsets of shared source groups, while calibration is held-out sources. Differences mix generator, length, prevalence and sample selection; no causal attribution or proof about overfitting.',
        'equivalent_result_search': {'searched': ['results/ghost_matched_diagnostics_v1/diagnose.py',
            'results/ghost_nll_nonlinear_v1/RUN_REPORT.md', 'results/ghost_matched_lr_v1/RUN_REPORT.md'],
            'finding': 'Existing diagnostics use combined fit/cal or calibration only; no equivalent native/added/generator breakdown.'}}
    if (OUT / 'protocol.json').exists():
        assert q.read(OUT / 'protocol.json') == protocol
    else:
        q.save(OUT / 'protocol.json', protocol)
    start = time.perf_counter()
    completions = {p: q.read(p / 'complete.json') for p in (LR, TREE)}
    assert all(c['status'] == 'complete' for c in completions.values())
    assert completions[LR]['new_fits'] == 5 and completions[TREE]['fixed_fits'] == 3
    assert not completions[LR]['test_opened'] and not completions[TREE]['official_test_opened']
    _, meta = expansion.metadata()
    assert len(meta['answers']) == 3839 and len(meta['windows']) == 696220
    assert meta['bounds'] == {'fit': [0, 653979], 'calibration': [653979, 696220]}
    fit_rows = q.lines(ROOT / 'fit_expansion/data/fit.jsonl')
    provenance = q.lines(ROOT / 'fit_expansion/data/answer_provenance.jsonl')
    assert len(fit_rows) == len(provenance) == 3680
    assert [r['response_id'] for r in fit_rows] == [a['response_id'] for a in meta['answers'][:3680]]
    prov = {r['canonical_response_id']: r for r in provenance}
    assert len(prov) == 3680
    for a, r in zip(meta['answers'][:3680], fit_rows):
        assert a['answer_sha256'] == r['answer_sha256']
        assert a['group_id'] == r['group_id'] and a['source_id'] == r['source_id']
        assert r['model'] in {p['model'] for p in prov[a['response_id']]['provenance']}
    assert all(r['model'] == 'llama-2-7b-chat' for r in fit_rows[:634])
    ai = {'native634_fit': np.arange(634), 'added3046_fit': np.arange(634, 3680),
          'cal159': np.arange(3680, 3839)}
    for name in sorted({r['model'] for r in fit_rows[634:]}):
        ai['added_model:' + name] = np.asarray([i for i in range(634, 3680) if fit_rows[i]['model'] == name])
    assert sum(len(v) for k, v in ai.items() if k.startswith('added_model:')) == 3046
    wy = np.asarray([w['label'] for w in meta['windows']], dtype=np.int8)
    ay = np.asarray([a['label'] for a in meta['answers']], dtype=np.int8)
    wi = {k: np.concatenate([np.asarray(meta['answer_windows'][meta['answers'][i]['response_id']], dtype=np.int64)
                            for i in indices]) for k, indices in ai.items()}
    assert np.array_equal(wi['native634_fit'], np.arange(168123))
    assert np.array_equal(wi['added3046_fit'], np.arange(168123, 653979))
    assert np.array_equal(wi['cal159'], np.arange(653979, 696220))
    cohorts = {}
    for name, indices in ai.items():
        answers = [meta['answers'][i] for i in indices]
        counts = [a['token_count'] for a in answers]
        cohorts[name] = {'answers': len(indices), 'groups': len({a['group_id'] for a in answers}),
            'windows': len(wi[name]), 'positive_windows': int(wy[wi[name]].sum()),
            'window_positive_rate': float(wy[wi[name]].mean()), 'positive_answers': int(ay[indices].sum()),
            'answer_positive_rate': float(ay[indices].mean()), 'raw_token_count_mean': float(np.mean(counts)),
            'raw_token_count_median': float(np.median(counts))}
    multi = [{'response_id': a['response_id'], 'canonical_model': fit_rows[i]['model'],
              'provenance_models': sorted({p['model'] for p in prov[a['response_id']]['provenance']})}
             for i, a in enumerate(meta['answers'][:3680])
             if len({p['model'] for p in prov[a['response_id']]['provenance']}) > 1]
    sources = {str(p.resolve()): q.sha(p) for p in [Path(__file__), Path(q.__file__), Path(expansion.__file__),
        OUT / 'protocol.json', ROOT / 'fit_expansion/data/fit.jsonl',
        ROOT / 'fit_expansion/data/answer_provenance.jsonl',
        *[ROOT / f'fit_expansion/data/{k}_fit.jsonl' for k in ('answers', 'tokens', 'windows_k4')],
        *[ROOT / f'data/{k}_calibration.jsonl' for k in ('answers', 'tokens', 'windows_k4')],
        LR / 'complete.json', TREE / 'complete.json']}
    results = {}
    for name, directory, stem in MODELS:
        files = [directory / (stem + '_result.json'), directory / (stem + '_scores.npz')]
        for p in files:
            assert q.sha(p) == completions[directory]['files_sha256'][p.name]
            sources[str(p.resolve())] = q.sha(p)
        entry = q.read(files[0]); ts = entry['thresholds']
        with np.load(files[1], allow_pickle=False) as z:
            ws, ans = z['window_scores'].copy(), z['answer_scores'].copy()
        assert ws.shape == (696220,) and ans.shape == (3839,) and np.isfinite(ws).all()
        assert np.array_equal(ans, q.answer_scores(meta, ws))
        assert q.metrics(meta, ws, ts) == entry['metrics']
        metrics = {}
        for part in ai:
            metrics[part] = {'windows': q.count(wy[wi[part]], ws[wi[part]], ts['window']['threshold']),
                             'answers': q.count(ay[ai[part]], ans[ai[part]], ts['answer']['threshold'])}
        # The disjoint training subsets must reproduce original aggregate counts.
        for scale in ('windows', 'answers'):
            for k in ('n', 'positive', 'tp', 'fp', 'fn', 'tn'):
                assert sum(metrics[s][scale][k] for s in ('native634_fit', 'added3046_fit')) == entry['metrics']['fit'][scale][k]
                assert sum(metrics[s][scale][k] for s in ai if s.startswith('added_model:')) == metrics['added3046_fit'][scale][k]
            assert metrics['cal159'][scale] == entry['metrics']['calibration'][scale]
        results[name] = {'thresholds_unchanged': ts, 'metrics': metrics,
                         'whole_fit_cal_metrics_exact': True, 'answermax_exact': True}
    data = {'status': 'complete', 'cohorts': cohorts, 'methods': results,
        'canonical_model_grouping': {'exclusive': True, 'multi_generator_duplicate_count': len(multi),
                                    'multi_generator_duplicates': multi},
        'window_order_sha256': q.digest(meta['windows']), 'source_sha256': sources,
        'seconds': time.perf_counter() - start, 'no_test': True, 'trained': False,
        'thresholds_reselected': False, 'GPU_used': False}
    q.save(OUT / 'DISTRIBUTION.json', data)
    lines = ['# GHOST 与 NLL：生成器分布诊断', '',
        '四个已完成模型、原阈值，未重训或重选。以下训练子集指标使用已经见过这些回答的模型；校准集也已反复用于开发。', '',
        '| 子集 | 回答 | 窗口 | 风险窗比例 | 平均原BPE长度 |', '|---|---:|---:|---:|---:|']
    for part in ('native634_fit', 'added3046_fit', 'cal159'):
        c = cohorts[part]
        lines.append(f"| {part} | {c['answers']} | {c['windows']} | {c['window_positive_rate']:.2%} | {c['raw_token_count_mean']:.1f} |")
    lines += ['', '| 模型 | 子集 | 窗口 AUROC | 窗口 AP | 窗口 F1 | TP/FP/FN | 整答 F1 |',
              '|---|---|---:|---:|---:|---|---:|']
    for name, e in results.items():
        for part in ('native634_fit', 'added3046_fit', 'cal159'):
            w, a = e['metrics'][part]['windows'], e['metrics'][part]['answers']
            lines.append(f"| {name} | {part} | {w['auroc']:.4f} | {w['average_precision']:.4f} | {w['f1']:.4f} | {w['tp']}/{w['fp']}/{w['fn']} | {a['f1']:.4f} |")
    lines += ['', '| 追加数据的原生成器 | 回答 | 风险窗比例 | GHOST树 AUROC/AP | NLL+GHOST树 AUROC/AP |',
              '|---|---:|---:|---|---|']
    for part, c in cohorts.items():
        if not part.startswith('added_model:'): continue
        g = results['ghost_tree']['metrics'][part]['windows']
        n = results['nll_ghost_tree']['metrics'][part]['windows']
        lines.append(f"| {part.split(':', 1)[1]} | {c['answers']} | {c['window_positive_rate']:.2%} | {g['auroc']:.4f}/{g['average_precision']:.4f} | {n['auroc']:.4f}/{n['average_precision']:.4f} |")
    lines += ['', '全部生成器下的四模型、两级计数/阈值和 AP/AUROC 均见 DISTRIBUTION.json。按导出时的 canonical model 排他分组；'
              f'有 {len(multi)} 条训练回答的去重来源含多个生成器，身份清单保留于 JSON，不重复计数。', '',
              '正例率、长度、回答生成器与来源划分共同变化；AP/F1尤其受正例率影响。全部追加回答都由同一个 Llama 重放，这些统计不能证明是重放造成差异，也不能仅由 fit F1 低于 cal F1 判定没有过拟合。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'status': 'complete', 'methods': 4, 'new_fits': 0,
        'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'DISTRIBUTION.json', 'REPORT.md')},
        'no_test': True, 'GPU_used': False})
    print('DISTRIBUTION_COMPLETE', data['seconds'], flush=True)
    print({k: {s: round(e['metrics'][s]['windows']['f1'], 6) for s in ('native634_fit', 'added3046_fit', 'cal159')}
           for k, e in results.items()}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
