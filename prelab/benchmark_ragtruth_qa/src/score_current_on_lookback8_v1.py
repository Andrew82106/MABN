"""Read-only geometry-gated adaptation of one frozen candidate to 8-BPE spans.

No fitting, threshold selection, baseline writes, test reads, or score imputation.
Every 8-span must contain exactly five existing complete 4-windows; otherwise
write the full geometry failure inventory and exit before loading predictions.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import run_development as q

OUT = q.ROOT/'results/current_on_lookback8_v1'
SOURCE = q.ROOT/'results/large_fixed_convex_v1'
CANDIDATE = 'semantic_claim__old_tree__large_weight0.4'
EXPECTED_ANSWERS, EXPECTED_4, EXPECTED_8 = 159, 42241, 41685


def protocol():
    return {
        'version': 'current-frozen-candidate-on-native-lookback8-v1',
        'candidate': CANDIDATE,
        'scope': 'Only existing native QA calibration159, never official test.',
        'target_geometry': 'All full native8BPE spans: start0..N-8, stride1, including nonlexical spans. No padding, truncation, or filtering.',
        'target_label': 'OR of unchanged original risk_mask across all eight raw tokens; original lexical labeling policy retained.',
        'mapping': 'At 8-span start s, require the five original complete4BPE windows at starts s,s+1,s+2,s+3,s+4, each same answer and original token/offset axis; score=max(five existing scores).',
        'missing_policy': 'Any missing or shifted constituent anywhere stops the entire evaluation before prediction loading. No zero filling, subset scoring, shorter windows, or baseline changes.',
        'metrics': 'Calibration8-span AUROC/AP; optional F1 uses only the candidate already-frozen original4-window threshold. No threshold reselection.',
        'comparison_limit': 'Task-scale mapping of our own4-window detector, not a modification of Lookback. Shared8-span scale does not equalize model inputs, training, or repeated calibration selection.',
        'CPU_only': True, 'new_fits': 0, 'threshold_selection': False,
        'baseline_files_written': False, 'official_test_opened': False,
    }


def constituents(start):
    return [(j, j+4) for j in range(start, start+5)]


def cpu_check():
    for s in (0, 3, 27):
        c = constituents(s)
        assert len(c) == 5 and len(set(c)) == 5
        assert all(s <= a < b <= s+8 and b-a == 4 for a, b in c)
        assert set().union(*(set(range(a, b)) for a, b in c)) == set(range(s, s+8))
        assert max([.1, .2, .8, .3, .4]) == .8
        for positive in range(s, s+8):
            assert any(positive in range(a, b) for a, b in c)
    return {'passed': True, 'exact_five_complete_windows': True,
            'all_eight_token_positions_covered': True, 'real_scores_read': False}


def bindings():
    paths = [Path(__file__), Path(q.__file__), SOURCE/'complete.json',
             SOURCE/'summary.json', SOURCE/(CANDIDATE+'.json'),
             q.DATA/'gold_manifest.json', q.OUT/'score_index.json',
             q.DATA/'feature_preparation/plans.jsonl']
    for part in ('fit', 'calibration'):
        paths += [q.DATA/f'{name}_{part}.jsonl' for name in ('answers', 'tokens', 'windows_k4')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def geometry(meta):
    original_index = q.read(q.OUT/'score_index.json')
    assert [w['window_id'] for w in meta['windows']] == [w['window_id'] for w in original_index['windows']]
    plans = {p['response_id']: p for p in q.lines(q.DATA/'feature_preparation/plans.jsonl')}
    lo, hi = meta['bounds']['calibration']
    assert (lo, hi) == (168123, 210364) and hi-lo == EXPECTED_4
    answers = [a for a in meta['answers'] if a['partition'] == 'calibration']
    assert len(answers) == EXPECTED_ANSWERS
    rows, failures, answer_checks = [], [], []
    for ai, answer in enumerate(answers):
        rid = answer['response_id']; t = meta['by_response'][rid]['tokens']; p = plans[rid]
        assert p['partition'] == t['partition'] == answer['partition'] == 'calibration'
        assert p['answer_sha256'] == t['answer_sha256'] == answer['answer_sha256']
        assert p['original_response'] == t['original_response']
        n = t['token_count']; raw = p['original']
        for k in ('token_ids', 'answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw'):
            original_key = 'answer_token_ids' if k == 'token_ids' else k
            assert len(t[k]) == n and t[k] == raw[original_key], (rid, k)
        assert all((not r) or l for r, l in zip(t['risk_mask'], t['lexical_mask']))
        old = {}
        for ix in meta['answer_windows'][rid]:
            assert lo <= ix < hi
            w = meta['windows'][ix]; start, end = w['token_start'], w['token_end']
            assert w['response_id'] == rid and w['group_id'] == answer['group_id']
            assert w['token_indices'] == list(range(start, end))
            assert w['token_ids'] == t['token_ids'][start:end]
            assert w['answer_token_positions'] == t['answer_token_positions'][start:end]
            assert w['label'] == int(any(t['risk_mask'][start:end]))
            assert (start, end) not in old
            old[(start, end)] = ix
        begin = len(rows)
        for start in range(max(0, n-7)):
            expected = constituents(start)
            missing = [(a, b) for a, b in expected if (a, b) not in old]
            row = {'response_id': rid, 'answer_index': ai, 'group_id': answer['group_id'],
                   'token_start': start, 'token_end': start+8,
                   'label': int(any(t['risk_mask'][start:start+8])),
                   'source_window_indices': [old[x] for x in expected if x in old],
                   'constituent_count': 5-len(missing)}
            rows.append(row)
            if missing:
                failures.append({**row, 'missing_complete4': [
                    {'token_start': a, 'token_end': b,
                     'lexical_count': sum(t['lexical_mask'][a:b]),
                     'risk_count': sum(t['risk_mask'][a:b]),
                     'token_ids': t['token_ids'][a:b],
                     'response_token_offsets': t['response_token_offsets'][a:b],
                     'response_token_offsets_raw': t['response_token_offsets_raw'][a:b]}
                    for a, b in missing]})
            else:
                assert len(set(row['source_window_indices'])) == 5
                assert row['label'] == max(meta['windows'][j]['label'] for j in row['source_window_indices'])
        answer_checks.append({'response_id': rid, 'group_id': answer['group_id'], 'raw_tokens': n,
                              'full8_windows': len(rows)-begin, 'expected_full8': max(0, n-7),
                              'old4_windows': len(old), 'all_token_ids_positions_offsets_exact': True})
    assert len(rows) == EXPECTED_8 and sum(a['full8_windows'] for a in answer_checks) == EXPECTED_8
    return rows, failures, answer_checks


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Preserve the prior run and failure evidence'
    q.save(OUT/'protocol.json', protocol()); q.save(OUT/'CPU_SELFCHECK.json', cpu_check())
    frozen = bindings(); q.save(OUT/'source_snapshot.json', frozen)
    start = time.perf_counter()
    done = q.read(SOURCE/'complete.json'); summary = q.read(SOURCE/'summary.json')
    entry = q.read(SOURCE/(CANDIDATE+'.json'))
    assert q.sha(SOURCE/'summary.json') == done['summary_sha256'] and not done['official_test_opened']
    assert entry == summary['selected']['semantic_claim__old_tree'] and entry['candidate'] == CANDIDATE
    meta = q.metadata()
    rows, failures, answers = geometry(meta)
    q.savel(OUT/'geometry8.jsonl', rows); q.savel(OUT/'answer_checks.jsonl', answers)
    q.savel(OUT/'missing_constituents.jsonl', failures)
    check = {'passed': not failures, 'answers': len(answers), 'full8_windows': len(rows),
             'existing_calibration4_windows': EXPECTED_4, 'failed8_windows': len(failures),
             'affected_answers': len({r['response_id'] for r in failures}),
             'constituent_count_histogram': dict(sorted(Counter(r['constituent_count'] for r in rows).items())),
             'all_answer_token_axes_exact': True, 'scores_loaded': False,
             'missing_all_nonlexical': all(m['lexical_count'] == 0 for r in failures for m in r['missing_complete4']),
             'new_fits': 0, 'official_test_opened': False}
    q.save(OUT/'GEOMETRY_CHECK.json', check)
    if failures:
        q.save(OUT/'FAILURE_REPORT.json', {**check, 'status': 'stopped_missing_constituent_scores',
               'reason': 'Not every original8-span has five existing complete4-window scores. No subset evaluation or imputation performed.',
               'source_snapshot_sha256': q.sha(OUT/'source_snapshot.json'), 'seconds': time.perf_counter()-start})
        text = (f"# 8词元同尺度映射已停止\n\n全部{len(answers)}条校准回答、{len(rows)}个完整8词元窗口已核对。"
                f"其中{len(failures)}个8窗（{check['affected_answers']}条回答）缺少所需5个旧4窗之一或多个。"
                f"缺项是否全部原规则排除的非lexical4窗：{check['missing_all_nonlexical']}。\n\n"
                "按预先门禁停止；没有加载候选预测、计算AUROC/AP/F1、补零或删窗。"
                "原基线、候选、标签、阈值均未改。逐窗记录见missing_constituents.jsonl。\n")
        (OUT/'REPORT.md').write_text(text, encoding='utf-8')
        assert frozen == bindings()
        print(json.dumps(check, ensure_ascii=False), flush=True)
        return 2
    path = SOURCE/(CANDIDATE+'_scores.npz')
    assert q.sha(path) == entry['scores_sha256']
    with np.load(path, allow_pickle=False) as z: old_scores = z['window_scores'].copy()
    assert old_scores.shape == (210364,) and np.isfinite(old_scores).all()
    indices = np.asarray([r['source_window_indices'] for r in rows], dtype=np.int64)
    assert indices.shape == (EXPECTED_8, 5)
    scores = old_scores[indices].max(axis=1); y = np.asarray([r['label'] for r in rows])
    cutoff = float(entry['thresholds']['window']['threshold']); pred = scores >= cutoff
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    result = {'candidate': CANDIDATE, 'rows': len(y), 'answers': len(answers),
              'auroc': float(roc_auc_score(y, scores)), 'average_precision': float(average_precision_score(y, scores)),
              'f1_existing4_threshold': 2*tp/(2*tp+fp+fn), 'fixed_original4_threshold': cutoff,
              'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'threshold_reselected': False,
              'source_scores_sha256': q.sha(path), 'new_fits': 0, 'official_test_opened': False}
    np.savez_compressed(OUT/'scores8.npz', scores=scores, labels=y, source_window_indices=indices)
    q.save(OUT/'metrics.json', result)
    assert frozen == bindings()
    q.save(OUT/'complete.json', {'status': 'complete', 'metrics_sha256': q.sha(OUT/'metrics.json'),
                               'scores_sha256': q.sha(OUT/'scores8.npz'), 'seconds': time.perf_counter()-start,
                               'new_fits': 0, 'official_test_opened': False})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run'])
    parser.parse_args()
    raise SystemExit(run())
