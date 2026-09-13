"""All-span evaluation adaptation; max only over the detector's defined4-windows.

This rule was fixed after the v1 coverage failure. Preserve all native8 spans,
unchanged gold, and the original candidate's4-window threshold. No baseline,
prediction, or missing-score edits; no training or threshold selection.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import run_development as q
import score_current_on_lookback8_v1 as first

OUT = q.ROOT/'results/current_on_lookback8_v2'
OLD = first.OUT
CUTOFF = 0.657469850500832


def protocol():
    return {
        'version': 'current-frozen-candidate-on-native-lookback8-v2',
        'candidate': first.CANDIDATE,
        'posthoc_adaptation': 'Rule fixed after the frozen v1 coverage gate failed:80 original nonlexical4-windows were undefined, affecting384 native8 spans. This introduces post-hoc adaptation risk; not a new independent test.',
        'mapping': 'For each complete8rawBPE span, take max of all originally defined complete4BPE subwindows wholly inside it. Use five normally or the original three/four when nonlexical4 subwindows are undefined. Never impute scores or drop8 spans.',
        'geometry': 'All41685 calibration8 spans across159 answers, stride1; original response/token/character axis and gold risk OR unchanged. Every8 span must have at least one defined subwindow.',
        'primary': 'AUROC and average precision on all41685 native8 spans. F1 only at the frozen original4-window threshold0.657469850500832; it is an adapted8-span F1, not native4-window F1.',
        'sensitivity': 'Also report the41301 spans with all five subwindows using the same scores and threshold, only as a predeclared diagnostic; do not select between full and subset results.',
        'limits': 'The detector and its mixture were already selected on this calibration cohort. Shared8-span scale does not equalize training/inputs/compute. This adaptation belongs to our detector; Lookback scores and protocol remain unchanged.',
        'new_fits': 0, 'threshold_selection': False, 'GPU_used': False,
        'baseline_files_written': False, 'official_test_opened': False,
    }


def evaluate(y, scores):
    y = np.asarray(y, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    assert y.shape == scores.shape and np.isfinite(scores).all() and set(y.tolist()) == {0, 1}
    pred = scores >= CUTOFF
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return {'rows': len(y), 'positive': int(y.sum()), 'negative': int((y == 0).sum()),
            'auroc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'fixed_original4_threshold': CUTOFF, 'f1': 2*tp/(2*tp+fp+fn),
            'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': tp/(tp+fn) if tp+fn else 0.,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'threshold_reselected': False}


def cpu_check():
    scores = np.asarray([.1, .8, .3, .9, .2])
    assert max(scores[[0, 1, 2, 3, 4]]) == .9
    assert max(scores[[0, 1, 2, 4]]) == .8
    assert max(scores[[0, 2, 4]]) == .3
    for ix in ([0, 1, 2, 3, 4], [0, 1, 2, 4], [0, 2, 4]):
        assert scores[ix].max() == sorted(float(scores[i]) for i in ix)[-1]
    r = evaluate([0, 1, 0, 1], [CUTOFF-1e-6, CUTOFF, .9, .1])
    assert (r['tp'], r['fp'], r['fn'], r['tn']) == (1, 1, 1, 1)
    return {'passed': True, 'max_of_defined3_4_5_only': True,
            'fixed_threshold_inclusive_at_equal': True, 'real_scores_read': False}


def snapshot():
    paths = [Path(__file__), Path(first.__file__), Path(q.__file__),
             OLD/'protocol.json', OLD/'source_snapshot.json', OLD/'CPU_SELFCHECK.json',
             OLD/'GEOMETRY_CHECK.json', OLD/'FAILURE_REPORT.json',
             OLD/'geometry8.jsonl', OLD/'answer_checks.jsonl', OLD/'missing_constituents.jsonl',
             first.SOURCE/(first.CANDIDATE+'_scores.npz')]
    return {**first.bindings(), **{str(p.resolve()): q.sha(p) for p in paths}}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite a frozen run'
    q.save(OUT/'protocol.json', protocol()); q.save(OUT/'CPU_SELFCHECK.json', cpu_check())
    frozen = snapshot(); q.save(OUT/'source_snapshot.json', frozen)
    assert q.read(OLD/'source_snapshot.json') == first.bindings(), 'Frozen v1 source identity changed'
    failure = q.read(OLD/'FAILURE_REPORT.json')
    assert failure['status'] == 'stopped_missing_constituent_scores' and not failure['scores_loaded']
    assert failure['failed8_windows'] == 384 and failure['affected_answers'] == 33 and failure['missing_all_nonlexical']
    started = time.perf_counter()
    meta = q.metadata(); rows, missing, answer_checks = first.geometry(meta)
    assert rows == q.lines(OLD/'geometry8.jsonl')
    assert missing == q.lines(OLD/'missing_constituents.jsonl')
    assert answer_checks == q.lines(OLD/'answer_checks.jsonl')
    assert len(rows) == 41685 and len(answer_checks) == 159 and len(missing) == 384
    assert all(m['lexical_count'] == m['risk_count'] == 0 for r in missing for m in r['missing_complete4'])
    counts = np.asarray([r['constituent_count'] for r in rows], dtype=np.int8)
    assert all(len(r['source_window_indices']) == r['constituent_count'] >= 1 for r in rows)
    assert {int(c): int(np.sum(counts == c)) for c in np.unique(counts)} == {3: 16, 4: 368, 5: 41301}
    done = q.read(first.SOURCE/'complete.json'); summary = q.read(first.SOURCE/'summary.json')
    entry = q.read(first.SOURCE/(first.CANDIDATE+'.json'))
    assert q.sha(first.SOURCE/'summary.json') == done['summary_sha256'] and not done['official_test_opened']
    assert entry == summary['selected']['semantic_claim__old_tree']
    assert entry['candidate'] == first.CANDIDATE and entry['thresholds']['window']['threshold'] == CUTOFF
    path = first.SOURCE/(first.CANDIDATE+'_scores.npz')
    assert q.sha(path) == entry['scores_sha256']
    with np.load(path, allow_pickle=False) as z:
        original = z['window_scores'].copy(); original_answer = z['answer_scores'].copy()
    assert original.shape == (210364,) and original.dtype == np.float64 and np.isfinite(original).all()
    assert np.array_equal(q.answer_scores(meta, original), original_answer)
    labels = np.asarray([r['label'] for r in rows], dtype=np.int8)
    scores = np.asarray([max(original[r['source_window_indices']]) for r in rows], dtype=np.float64)
    # Independent scalar lookup checks geometry, unchanged label OR, and every max.
    for i, r in enumerate(rows):
        indices = r['source_window_indices']; t = meta['by_response'][r['response_id']]['tokens']
        assert r['label'] == int(any(t['risk_mask'][r['token_start']:r['token_end']]))
        assert r['label'] == max(meta['windows'][j]['label'] for j in indices)
        observed = []
        for j in indices:
            w = meta['windows'][j]
            assert w['response_id'] == r['response_id'] and w['partition'] == 'calibration'
            assert r['token_start'] <= w['token_start'] and w['token_end'] <= r['token_end']
            assert w['token_end'] - w['token_start'] == 4
            observed.append(float(original[j]))
        assert scores[i] == sorted(observed)[-1]
    complete_five = counts == 5
    metrics = {'candidate': first.CANDIDATE,
               'all_native8_primary': evaluate(labels, scores),
               'complete_five_only_sensitivity': evaluate(labels[complete_five], scores[complete_five]),
               'posthoc_adaptation': True, 'sensitivity_used_to_select': False,
               'native4_calibration_reference': entry['metrics']['calibration']['windows'],
               'new_fits': 0, 'GPU_used': False, 'official_test_opened': False}
    np.savez_compressed(OUT/'scores8.npz', scores=scores, labels=labels, constituent_count=counts,
                        answer_index=np.asarray([r['answer_index'] for r in rows], dtype=np.int32),
                        token_start=np.asarray([r['token_start'] for r in rows], dtype=np.int32),
                        token_end=np.asarray([r['token_end'] for r in rows], dtype=np.int32),
                        complete_five_sensitivity=complete_five)
    q.savel(OUT/'mapping8.jsonl', rows); q.savel(OUT/'answer_checks.jsonl', answer_checks)
    q.save(OUT/'metrics.json', metrics)
    q.save(OUT/'ARITHMETIC_CHECK.json', {'passed': True, 'all_rows': 41685, 'answers': 159,
            'minimum_defined_constituents': int(counts.min()), 'maximum_defined_constituents': int(counts.max()),
            'full_geometry_exact_to_v1': True, 'all_scalar_maxima_exact': True,
            'unchanged_labels': True, 'original_answermax_exact': True,
            'source_window_order_verified': True, 'missing_imputed': 0, 'spans_removed': 0,
            'threshold_reselected': False, 'baseline_files_written': False, 'official_test_opened': False})
    allm = metrics['all_native8_primary']; sub = metrics['complete_five_only_sensitivity']
    report = ['# 当前自研候选在原生8词元跨度上的评测适配v2', '',
              '**规则在v1覆盖门禁失败后确定，存在事后适配风险。** 当前候选此前已在同一校准集选择，结果不是独立测试。', '',
              '每个完整8词元窗只对模型原来有定义的完整4词元子窗取max；正常5个，缺纯nonlexical子窗时用已有3或4个。未补分数、删8窗或改gold。', '',
              '| 范围 | 8窗数 | AUROC | AP | 固定原4窗阈值下F1 | TP/FP/FN/TN |',
              '|---|---:|---:|---:|---:|---|']
    for name, m in [('全部原生8窗（主结果）', allm), ('完整5子窗（仅敏感性）', sub)]:
        report.append(f"| {name} | {m['rows']} | {m['auroc']:.9f} | {m['average_precision']:.9f} | {m['f1']:.9f} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |")
    report += ['', f'全部159答、41685窗保留；41301窗有5个子窗，368窗有4个，16窗有3个。各响应、token ID、预测位置、原始/裁剪字符坐标均与冻结输入相同。F1固定阈值为{CUTOFF}，不是模型原生4窗F1；没有重新选阈值。', '',
               '主结果始终是全量；敏感性子集不用于选优。没有改动Lookback或其他baseline目录/分数；本结果只为共享8词元跨度尺度，训练与输入条件仍不同。', '',
               '入口：src/score_current_on_lookback8_v2.py。映射逐窗见mapping8.jsonl；数值与身份核对见ARITHMETIC_CHECK.json。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    assert frozen == snapshot(), 'Read-only sources changed'
    files = ['protocol.json', 'source_snapshot.json', 'CPU_SELFCHECK.json', 'scores8.npz',
             'mapping8.jsonl', 'answer_checks.jsonl', 'metrics.json', 'ARITHMETIC_CHECK.json', 'REPORT.md']
    q.save(OUT/'complete.json', {'status': 'complete', 'answers': 159, 'full8_windows': 41685,
            'files_sha256': {f: q.sha(OUT/f) for f in files}, 'seconds': time.perf_counter()-started,
            'baseline_files_written': False, 'new_fits': 0, 'GPU_used': False,
            'official_test_opened': False, 'posthoc_adaptation': True})
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run'])
    parser.parse_args(); run()
