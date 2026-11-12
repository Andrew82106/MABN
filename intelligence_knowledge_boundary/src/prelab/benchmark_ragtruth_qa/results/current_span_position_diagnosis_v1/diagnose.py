"""Fixed saved-score span-position diagnosis. Gold is diagnostic only."""
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q

SOURCE = ROOT / 'results/large_fixed_convex_v1'
CANDIDATE = 'semantic_claim__old_tree__large_weight0.4'


def risk_runs(lexical, risk):
    """Consecutive positive lexical tokens; ignore punctuation, stop at clean lexical."""
    runs, current = [], []
    for j, (lex, pos) in enumerate(zip(lexical, risk)):
        assert not pos or lex
        if not lex:
            continue
        if pos:
            current.append(j)
        elif current:
            runs.append(current); current = []
    if current:
        runs.append(current)
    return runs


def touched_windows(indices, token_to_windows):
    return sorted({i for j in indices for i in token_to_windows[j]})


def protocol():
    return {'version': 'current-fixed-candidate-span-position-v1', 'candidate': CANDIDATE,
        'data': 'Original calibration159 answers,154 source groups,42241 windows; saved human labels and4rawBPE windows unchanged.',
        'first': 'For each original human span with lexical risk tokens, choose its earliest eligible window intersecting any of those risk_token_indices. Deduplicate these windows. It may start up to3 raw tokens before the span first risk token.',
        'later': 'All remaining gold-positive windows. If a window is first for one span and later for another, first takes priority. Count such overlaps explicitly.',
        'normal': 'Every gold-zero window. First/later/normal are mutually exclusive and exhaustive over original42241 windows.',
        'ranking': 'For first and later separately, their positive windows versus the SAME all-normal window set; AUROC/AP with the resulting subset prevalence disclosed. Normal-only AUROC/AP are undefined; report FPR/specificity.',
        'runs': 'Maximal risk runs along lexical-token order: skip nonlexical tokens including punctuation, but any gold-normal lexical token ends a run. Adjacent human spans may merge; these are not asserted to be single factual propositions.',
        'run_hit': 'A run is hit if any threshold-positive original4BPE window touches its risk tokens. Keep every run, including misses.',
        'first_hit_offset': 'Earliest alerted touching window token_start minus run first risk raw-token index (signed; negatives are window geometry, not advance warning). Also report first covered risk-token offset, which is nonnegative.',
        'whole_run': 'Report both fraction/all of touching risk windows alerted and fraction/all of risk lexical tokens covered by their union. These are distinct from exact semantic span detection.',
        'seeded_misses': 'Diagnostic-only count of missed positive windows touching a gold run that has at least one existing alert. This uses true run boundaries and is NOT a deployed extension rule.',
        'equivalent_search': {'examined': ['src/diagnose_selected_convex.py', 'src/diagnose_completed_tail.py',
            'results/claim_pooling_v1/span_extent_diagnostic.json', 'results/first_risk_token_matched_v1/protocol.json'],
            'finding': 'Existing results cover type recall, span union coverage or fit-only NLL; not this selected-candidate cal position/run breakdown.'},
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'test_opened': False,
        'limitations': 'Post-selection diagnostics on repeatedly used calibration. Gold positions/runs must never become deployment inputs; geometry and subset prevalence affect comparisons. No structure or extension parameter selected.'}


def tiny_check():
    assert risk_runs([1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]) == [[0, 2], [5]]
    windows = [set(range(i, i + 4)) for i in range(5)]
    ttw = defaultdict(list)
    for i, w in enumerate(windows):
        for token in w: ttw[token].append(i)
    first_a, first_b = touched_windows([2], ttw), touched_windows([5], ttw)
    assert first_a == [0, 1, 2] and first_b == [2, 3, 4]
    first = {first_a[0], first_b[0]}
    later = set(first_a[1:]) | set(first_b[1:])
    assert first & later == {2} and first == {0, 2}
    assert 0 - 2 == -2
    return {'status': 'passed', 'punctuation_bridge_normal_lexical_break': True,
            'overlapping_span_first_priority': True, 'signed_window_start_offset': True}


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a): return {'n': 0}
    return {'n': len(a), 'mean': float(a.mean()), 'min': float(a.min()),
            'q25': float(np.quantile(a, .25)), 'median': float(np.median(a)),
            'q75': float(np.quantile(a, .75)), 'max': float(a.max())}


def run():
    assert not (OUT / 'complete.json').exists()
    cfg = protocol()
    if (OUT / 'protocol.json').exists(): assert q.read(OUT / 'protocol.json') == cfg
    else: q.save(OUT / 'protocol.json', cfg)
    q.save(OUT / 'CPU_SELFCHECK.json', tiny_check())
    start = time.perf_counter()
    complete = q.read(SOURCE / 'complete.json')
    assert complete['candidate_count'] == 36 and not complete['official_test_opened']
    assert complete['summary_sha256'] == q.sha(SOURCE / 'summary.json')
    entry = q.read(SOURCE / 'summary.json')['selected']['semantic_claim__old_tree']
    assert entry['candidate'] == CANDIDATE and entry['large_weight'] == .4
    previous = ROOT / 'results/selected_convex_error_diagnosis_v1/summary.json'
    assert q.read(previous)['candidate'] == CANDIDATE
    path = SOURCE / (CANDIDATE + '_scores.npz')
    assert q.sha(path) == entry['scores_sha256']
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    windows = meta['windows'][lo:hi]; answers = meta['answers'][634:]; tokens = meta['tokens'][634:]
    assert len(windows) == 42241 and len(answers) == len(tokens) == 159
    assert len({a['group_id'] for a in answers}) == 154
    with np.load(path, allow_pickle=False) as z:
        full = z['window_scores']; answer_score = z['answer_scores'][634:]
        assert np.array_equal(q.answer_scores(meta, full)[634:], answer_score)
        scores = full[lo:hi].copy()
    gold = np.asarray([w['label'] for w in windows], bool)
    threshold = entry['thresholds']['window']['threshold']; pred = scores >= threshold
    counts = q.count(gold, scores, threshold)
    assert counts == entry['metrics']['calibration']['windows'] == q.read(previous)['metrics']['windows']
    assert q.count([a['label'] for a in answers], answer_score, entry['thresholds']['answer']['threshold']) == entry['metrics']['calibration']['answers']
    assert int(gold.sum()) == 5984 and counts['tp'] == 3839 and counts['fp'] == 1300
    ttw = defaultdict(lambda: defaultdict(list))
    for i, w in enumerate(windows):
        for token in w['token_indices']: ttw[w['response_id']][token].append(i)
    first_set, later_set, all_span_windows = set(), set(), set()
    span_rows, run_rows = [], []
    run_window_sets = []
    for t in tokens:
        rid = t['response_id']; token_lookup = ttw[rid]
        spans = t['span_token_mapping']
        for m in spans:
            ix = m['risk_token_indices']; touched = touched_windows(ix, token_lookup)
            item = {'response_id': rid, 'span_index': m['span_index'],
                    'char_start': m['original_start'], 'char_end': m['original_end'],
                    'label_type': t['original_labels'][m['span_index']]['label_type'],
                    'risk_raw_token_indices': ix, 'touching_windows': len(touched)}
            if touched:
                first = touched[0]; first_set.add(first); later_set.update(touched[1:]); all_span_windows.update(touched)
                item.update(first_window_id=windows[first]['window_id'], first_window_cal_index=first,
                    first_window_start_offset=windows[first]['token_start'] - ix[0],
                    first_window_detected=bool(pred[first]), first_window_score=float(scores[first]))
            span_rows.append(item)
        for run_index, ix in enumerate(risk_runs(t['lexical_mask'], t['risk_mask'])):
            touched = touched_windows(ix, token_lookup)
            assert touched and gold[touched].all()
            flagged = [j for j in touched if pred[j]]
            covered = {token for j in flagged for token in windows[j]['token_indices']} & set(ix)
            item = {'response_id': rid, 'run_index': run_index, 'risk_raw_token_indices': ix,
                'first_risk_token': ix[0], 'last_risk_token': ix[-1], 'risk_tokens': len(ix),
                'raw_width': ix[-1] - ix[0] + 1, 'touching_risk_windows': len(touched),
                'detected_windows': len(flagged), 'window_recall': len(flagged) / len(touched),
                'any_window_hit': bool(flagged), 'all_touching_windows_hit': len(flagged) == len(touched),
                'risk_token_union_covered': len(covered), 'risk_token_union_recall': len(covered) / len(ix),
                'all_risk_tokens_union_covered': len(covered) == len(ix),
                'first_alert_window_cal_index': flagged[0] if flagged else None,
                'first_alert_start_offset_raw': windows[flagged[0]]['token_start'] - ix[0] if flagged else None,
                'first_covered_risk_token_offset_raw': min(covered) - ix[0] if covered else None}
            run_rows.append(item); run_window_sets.append(set(touched))
    assert all_span_windows == set(np.flatnonzero(gold))
    first = np.asarray(sorted(first_set), dtype=np.int64)
    later = np.asarray(sorted(all_span_windows - first_set), dtype=np.int64)
    normal = np.flatnonzero(~gold)
    assert len(first) + len(later) + len(normal) == 42241 and not set(first) & set(later)
    position = {}
    for name, ix in [('first_touching_span_window', first), ('later_risk_window', later)]:
        ys = np.r_[np.ones(len(ix)), np.zeros(len(normal))]
        ss = np.r_[scores[ix], scores[normal]]
        position[name] = {'positive_windows': len(ix), 'shared_normal_windows': len(normal),
            'subset_positive_rate': len(ix) / len(ys), 'tp': int(pred[ix].sum()), 'fn': int((~pred[ix]).sum()),
            'recall_at_saved_threshold': float(pred[ix].mean()), 'auroc_vs_same_normal': float(roc_auc_score(ys, ss)),
            'ap_vs_same_normal': float(average_precision_score(ys, ss))}
    position['normal'] = {'windows': len(normal), 'fp': int(pred[normal].sum()), 'tn': int((~pred[normal]).sum()),
        'false_positive_rate': float(pred[normal].mean()), 'specificity': float((~pred[normal]).mean()),
        'auroc': None, 'ap': None, 'reason': 'No positive class within the normal-only group.'}
    assert position['first_touching_span_window']['tp'] + position['later_risk_window']['tp'] == 3839
    hit_runs = [r for r in run_rows if r['any_window_hit']]
    seeded_windows = set().union(*[w for r, w in zip(run_rows, run_window_sets) if r['any_window_hit']])
    missed = set(np.flatnonzero(gold & ~pred))
    run_summary = {'gold_runs': len(run_rows), 'hit_runs': len(hit_runs), 'missed_runs': len(run_rows) - len(hit_runs),
        'any_window_hit_rate': len(hit_runs) / len(run_rows),
        'all_touching_windows_hit_runs': sum(r['all_touching_windows_hit'] for r in run_rows),
        'all_risk_tokens_union_covered_runs': sum(r['all_risk_tokens_union_covered'] for r in run_rows),
        'mean_per_run_window_recall': float(np.mean([r['window_recall'] for r in run_rows])),
        'mean_per_run_risk_token_union_recall': float(np.mean([r['risk_token_union_recall'] for r in run_rows])),
        'risk_run_raw_width': distribution([r['raw_width'] for r in run_rows]),
        'first_alert_start_offset_raw_hit_runs_only': distribution([r['first_alert_start_offset_raw'] for r in hit_runs]),
        'first_covered_risk_offset_raw_hit_runs_only': distribution([r['first_covered_risk_token_offset_raw'] for r in hit_runs]),
        'hit_runs_alert_start_before_risk_start_due_to_geometry': sum(r['first_alert_start_offset_raw'] < 0 for r in hit_runs),
        'hit_runs_first_covered_risk_token_is_run_start': sum(r['first_covered_risk_token_offset_raw'] == 0 for r in hit_runs),
        'missed_positive_windows': len(missed), 'missed_windows_touching_a_hit_gold_run': len(missed & seeded_windows),
        'missed_windows_only_in_unhit_gold_runs': len(missed - seeded_windows),
        'per_run_window_counts_can_overlap': True, 'gold_seedability_is_not_deployable': True}
    data = {'status': 'complete', 'candidate': CANDIDATE, 'thresholds_unchanged': entry['thresholds'],
        'original_calibration_metrics_exact': entry['metrics']['calibration'], 'position_groups': position,
        'human_span_count': len(span_rows), 'spans_without_lexical_window': sum(not r['touching_windows'] for r in span_rows),
        'first_windows_deduplicated': len(first), 'first_and_later_role_overlap_windows_first_priority': len(first_set & later_set),
        'runs': run_summary, 'human_spans': span_rows, 'per_gold_run': run_rows,
        'window_order_sha256': q.digest(windows),
        'source_sha256': {str(p.resolve()): q.sha(p) for p in [Path(__file__), OUT / 'protocol.json', Path(q.__file__),
            SOURCE / 'complete.json', SOURCE / 'summary.json', path, previous,
            *[ROOT / f'data/{k}_calibration.jsonl' for k in ('answers', 'tokens', 'windows_k4')]]},
        'seconds': time.perf_counter() - start, 'new_fits': 0, 'new_thresholds': 0, 'no_test': True, 'GPU_used': False}
    q.save(OUT / 'DIAGNOSIS.json', data)
    a, b = position['first_touching_span_window'], position['later_risk_window']
    lines = ['# 当前候选：风险跨度位置诊断', '', f'固定 `{CANDIDATE}` 原窗口阈值 {threshold:.9f}；原cal159/42241窗，F1仍为0.690281。没有拟合或重选。', '',
        '| 位置 | 正窗 | 原阈值检出 | 召回 | 对相同正常窗AUROC | AP（子集正例率） |', '|---|---:|---:|---:|---:|---:|']
    for label, e in [('最早触及每个人工跨度的窗', a), ('其余风险窗', b)]:
        lines.append(f"| {label} | {e['positive_windows']} | {e['tp']} | {e['recall_at_saved_threshold']:.2%} | {e['auroc_vs_same_normal']:.4f} | {e['ap_vs_same_normal']:.4f}（{e['subset_positive_rate']:.2%}） |")
    lines += ['', f"正常窗 {len(normal)}，误报 {counts['fp']}（{position['normal']['false_positive_rate']:.2%}）。AUC/AP分别将首窗或后续窗与同一正常窗集合比较；AP的正例率不同，不能直接当作同一任务高低。",
        f"首窗采用原stride1重叠几何，可能比风险起点早最多3格；{len(first_set & later_set)}个窗同时扮演另一跨度的后续窗，固定首窗优先去重。正常词元截断run，标点不中断；这些连续段不等于单个事实。", '',
        f"共有 {len(run_rows)} 个gold风险run，{len(hit_runs)} 个至少命中一窗（{run_summary['any_window_hit_rate']:.2%}），{run_summary['missed_runs']} 个完全漏报；{run_summary['all_touching_windows_hit_runs']} 个所有触及窗均检出，{run_summary['all_risk_tokens_union_covered_runs']} 个风险词元被报警窗并集完整覆盖。",
        f"已命中run的首报警窗起点相对风险起点中位数 {run_summary['first_alert_start_offset_raw_hit_runs_only']['median']:g} raw BPE，首个实际覆盖的风险词元偏移中位数 {run_summary['first_covered_risk_offset_raw_hit_runs_only']['median']:g}；未命中run保留且不进入该条件中位数。负的窗起点偏移只是窗口几何，不是提前预警。", '',
        f"在原 {len(missed)} 个FN窗中，{len(missed & seeded_windows)} 个触及至少已有一次报警的真实run，另 {len(missed - seeded_windows)} 个只属于完全未报警run。前一部分提供检查跨度延展的具体空间，后一部分仍需要更好的风险识别；这里使用了真实边界，不能据此声称可部署收益或直接扩展预测。",
        '每个run的首命中位置、窗口召回和词元并集覆盖均在JSON。人工标签和全部失败保持；不据此新增特征或选择延展长度。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'status': 'complete', 'candidate': CANDIDATE,
        'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'CPU_SELFCHECK.json', 'DIAGNOSIS.json', 'REPORT.md')},
        'no_test': True, 'new_fits': 0, 'new_thresholds': 0})
    print('SPAN_POSITION_DIAGNOSIS_COMPLETE', data['seconds'], position, run_summary, flush=True)


if __name__ == '__main__':
    run()
