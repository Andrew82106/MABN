"""Two fixed paired comparisons against the newly completed HARP controls."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_harp_group_intervals_v1'
HELPER = ROOT / 'results/completed_group_bootstrap_v1/bootstrap_fixed.py'
spec = importlib.util.spec_from_file_location('fixed_group_helpers', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
METHODS = {
    'current': ('large', 'semantic_claim__old_tree__large_weight0.4'),
    'harp_fava': ('fava', 'harp_claim__old_lr__fava_weight0.2'),
    'harp_nli': ('nli', 'harp_claim__old_lr__nli_weight0.2'),
}
REPEATS, SEED = 5000, 20261009


def run():
    assert not (OUT / 'started.json').exists(), 'Preserve any previous execution'
    OUT.mkdir(parents=True, exist_ok=True)
    records, files = {}, [Path(__file__), HELPER]
    for name, (trial, stem) in METHODS.items():
        directory = ROOT / f'results/{trial}_fixed_convex_v1'
        done = h.read(directory / 'complete.json')
        assert not done['official_test_opened']
        assert done['summary_sha256'] == h.sha(directory / 'summary.json')
        entry = h.read(directory / (stem + '.json'))
        scores = directory / (stem + '_scores.npz')
        assert entry['candidate'] == stem and entry['scores_sha256'] == h.sha(scores)
        records[name] = {'entry': entry, 'score_path': str(scores.resolve())}
        files += [directory / 'complete.json', directory / 'summary.json',
                  directory / (stem + '.json'), scores]
    answer_path = ROOT / 'data/answers_calibration.jsonl'
    window_path = ROOT / 'data/windows_k4_calibration.jsonl'
    files += [answer_path, window_path]
    frozen = {str(p.resolve()): h.sha(p) for p in files}
    protocol = {
        'cohort': 'Original cal159 answers / 154 material groups / 42241 four-raw-BPE windows.',
        'methods': METHODS, 'replicates': REPEATS, 'seed': SEED,
        'sampling': 'Sorted group IDs, PCG64 integers(0,154,(5000,154)); same draws for all methods and both units.',
        'statistic': 'Pooled micro F1, current minus each fixed HARP control; 2.5/97.5 linear percentiles.',
        'selection': 'No new model, epoch, alpha or threshold selection, including inside draws.',
        'limitation': 'Conditional diagnostic on repeatedly used calibration; omits prior selection uncertainty and is not independent significance or a guarantee of noninferiority.',
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False,
    }
    h.write(OUT / 'started.json', {'protocol': protocol, 'source_sha256': frozen})
    answers, windows = h.rows(answer_path), h.rows(window_path)
    assert len(answers) == 159 and len(windows) == 42241
    assert all(r['partition'] == 'calibration' and r['eligible'] for r in answers + windows)
    by_answer = {a['response_id']: i for i, a in enumerate(answers)}
    group_ids = sorted({a['group_id'] for a in answers})
    assert len(by_answer) == 159 and len(group_ids) == 154
    group_index = {g: i for i, g in enumerate(group_ids)}
    for w in windows:
        assert w['group_id'] == answers[by_answer[w['response_id']]]['group_id']
    records_by_unit = [windows, answers]
    gold = [np.asarray([r['label'] for r in rs], np.int8) for rs in records_by_unit]
    owners = [np.asarray([group_index[r['group_id']] for r in rs]) for rs in records_by_unit]
    assert [int(y.sum()) for y in gold] == [5984, 100]
    counts = np.zeros((len(METHODS), 2, 154, 3), np.int64)
    memberships = np.asarray([by_answer[w['response_id']] for w in windows])
    for mi, (name, rec) in enumerate(records.items()):
        entry = rec['entry']
        with np.load(rec['score_path'], allow_pickle=False) as z:
            assert z['window_scores'].shape == (210364,) and z['answer_scores'].shape == (793,)
            scores = [z['window_scores'][168123:].copy(), z['answer_scores'][634:].copy()]
        maxima = np.full(159, -np.inf)
        np.maximum.at(maxima, memberships, scores[0])
        assert np.array_equal(maxima, scores[1])
        for ui, unit in enumerate(('windows', 'answers')):
            threshold = entry['thresholds']['window' if ui == 0 else 'answer']['threshold']
            actual = h.metric(gold[ui], scores[ui], threshold)
            assert all(value == entry['metrics']['calibration'][unit][key] for key, value in actual.items())
            counts[mi, ui] = h.group_counts(gold[ui], scores[ui], threshold, owners[ui], 154)
            assert np.array_equal(counts[mi, ui].sum(0), [actual['tp'], actual['fp'], actual['fn']])
    rng = np.random.default_rng(SEED)
    assert type(rng.bit_generator).__name__ == 'PCG64'
    draws = rng.integers(0, 154, size=(REPEATS, 154))
    multiplicity = np.zeros((REPEATS, 154), np.int64)
    np.add.at(multiplicity, (np.repeat(np.arange(REPEATS), 154), draws.ravel()), 1)
    samples = np.einsum('rg,mugc->murc', multiplicity, counts)
    for row in (0, 1, REPEATS - 1):
        assert np.array_equal(samples[:, :, row], counts[:, :, draws[row], :].sum(2))
    bootstrap = h.f1(samples)
    points = h.f1(counts.sum(2))
    contrasts = {}
    for mi, name in enumerate(METHODS):
        if mi == 0:
            continue
        contrasts[name] = {}
        for ui, unit in enumerate(('windows', 'answers')):
            low, high = np.percentile(bootstrap[0, ui] - bootstrap[mi, ui], [2.5, 97.5], method='linear')
            contrasts[name][unit] = {'difference': float(points[0, ui] - points[mi, ui]),
                                     'percentile95': [float(low), float(high)],
                                     'contains_zero': bool(low <= 0 <= high)}
    np.savez_compressed(OUT / 'group_bootstrap.npz', groups=np.asarray(group_ids),
                        methods=np.asarray(list(METHODS)), counts=counts,
                        draws=draws.astype(np.int16), bootstrap_f1=bootstrap)
    result = {'protocol': protocol, 'methods': records, 'contrasts': contrasts,
              'zero_denominators': int((2 * samples[..., 0] + samples[..., 1] + samples[..., 2] == 0).sum()),
              'all_fixed_scores_counts_and_answer_maxima_exact': True,
              'draw_oracles_exact': True, 'official_test_opened': False, 'new_fits': 0}
    h.write(OUT / 'RESULT.json', result)
    lines = ['# 当前模型与新增强HARP对照', '',
             '同一校准集、固定模型和阈值，154材料组配对重采样5000次。以下是当前模型减去对照的F1差。', '',
             '| 对照 | 窗口差值 [95%条件区间] | 整答差值 [95%条件区间] |', '|---|---:|---:|']
    for name, values in contrasts.items():
        cells = []
        for unit in ('windows', 'answers'):
            d = values[unit]
            cells.append(f"{d['difference']:+.6f} [{d['percentile95'][0]:+.6f}, {d['percentile95'][1]:+.6f}]")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += ['', '区间没有计入此前反复选模型、轮次、融合权重和阈值的偏差；不能证明独立测试优势。跨零也不等于方法相同。未新增拟合或改变原评测口径。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert all(h.sha(Path(p)) == digest for p, digest in frozen.items())
    h.write(OUT / 'complete.json', {'result_sha256': h.sha(OUT / 'RESULT.json'),
                                   'arrays_sha256': h.sha(OUT / 'group_bootstrap.npz'),
                                   'report_sha256': h.sha(OUT / 'REPORT.md'),
                                   'official_test_opened': False, 'new_fits': 0})
    print(json.dumps(contrasts), flush=True)


if __name__ == '__main__':
    run()
