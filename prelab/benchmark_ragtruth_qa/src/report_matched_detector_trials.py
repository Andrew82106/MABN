"""Compare completed, already-selected matched trials; no fitting or thresholds."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/matched_detector_trials_v1'
TRIALS = ('large', 'nli', 'fava')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert not (OUT / 'REPORT.json').exists(), 'Preserve completed report'
    bindings, rows = {}, []
    for trial in TRIALS:
        folder = ROOT / f'results/{trial}_fixed_convex_v1'
        complete = read(folder / 'complete.json')
        summary = read(folder / 'summary.json')
        assert complete['summary_sha256'] == sha(folder / 'summary.json')
        assert complete['candidate_count'] == 36 and not complete['official_test_opened']
        bindings[str(folder / 'complete.json')] = sha(folder / 'complete.json')
        bindings[str(folder / 'summary.json')] = sha(folder / 'summary.json')
        assert len(summary['selected']) == 6
        for family, entry in summary['selected'].items():
            metric = entry['metrics']['calibration']
            window, answer = metric['windows'], metric['answers']
            assert (window['n'], window['positive']) == (42241, 5984)
            assert (answer['n'], answer['positive']) == (159, 100)
            for m in (window, answer):
                assert m['f1'] == 2 * m['tp'] / (2 * m['tp'] + m['fp'] + m['fn'])
            rows.append({'trial': trial, 'family': family, 'candidate': entry['candidate'],
                         'window_f1': window['f1'], 'answer_f1': answer['f1'],
                         'scores_sha256': entry['scores_sha256']})
    baselines = [r for r in rows if r['family'].startswith(('lookback__', 'harp_claim__'))]
    proposed = [r for r in rows if r['family'].startswith('semantic_claim__')]
    assert len(baselines) == 12 and len(proposed) == 6
    for row in proposed:
        row['below_baselines'] = [b['candidate'] for b in baselines
                                 if row['window_f1'] < b['window_f1']
                                 or row['answer_f1'] < b['answer_f1']]
        row['noninferior_point_estimates_in_both_metrics'] = not row['below_baselines']
    eligible = [r for r in proposed if r['noninferior_point_estimates_in_both_metrics']]
    selected = max(eligible, key=lambda r: (min(r['window_f1'], r['answer_f1']),
                                           r['window_f1'], r['answer_f1'])) if eligible else None
    report = {'rows': rows, 'proposed_not_below_each_matched_baseline': selected,
              'scope': '18 already-selected families from three completed36-candidate trials; no new alpha, threshold or fit. A point-estimate comparison on reused cal159, not statistical noninferiority or independent testing.',
              'baseline_identity': 'Lookback and HARP adaptations, each with equal access to tail2, citation features, the respective new semantic detector and six convex weights. Not full author-method replications.',
              'selection_rule': 'Among semantic_claim families, require both F1 point estimates at least every selected matched Lookback/HARP baseline, then maximize min(twoF1), windowF1, answerF1. If none passes, explicitly return null.',
              'source_sha256': bindings, 'new_fits': 0, 'official_test_opened': False,
              'stability_proven': False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 同预算三轮检测器对照', '',
             '| 新信号 | 既有组合 | 窗口F1 | 整答F1 |', '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['trial']} | {row['family']} | {row['window_f1']:.6f} | {row['answer_f1']:.6f} |")
    lines += ['', '每行两项来自同一候选，保留原4BPE窗口和整答max。只比较既有选型，不重新选阈值或拟合。', '']
    if selected:
        lines.append(f"在本表全部已选Lookback/HARP对照前，两项点估计均未落后的语义候选：{selected['candidate']}，{selected['window_f1']:.6f}/{selected['answer_f1']:.6f}。")
    else:
        lines.append('没有语义候选同时达到本表每个基线的两项点估计；不能宣称达成不落后要求。')
    lines += ['', '这是反复使用的159答校准集，不能据此宣称稳定优于基线、统计不劣或SOTA。独立测试仍封存。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'selected': selected, 'rows': len(rows)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
