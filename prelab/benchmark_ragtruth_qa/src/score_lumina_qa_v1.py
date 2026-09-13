"""Fixed LUMINA formula and two component readouts; CPU only, no fitting.

prepare/check do not read new features. run requires the full3839 cache and is
never called automatically. Paper answer-mean is separate from window/answermax.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_development as q
import run_lumina_qa_extraction_v1 as extract
import feature_qa as feature

ROOT = q.ROOT
sys.path.insert(0, str(ROOT / 'fit_expansion'))
import run_probe_expansion as expansion

OUT = ROOT / 'results/lumina_qa_scoring_v1'
PREP = ROOT / 'results/lumina_qa_preparation_v1'
FEATURES = ROOT / 'results/lumina_qa_features_v1'
METHODS = ('lumina', 'ipr', 'negative_mmd')
NFIT, NCAL, NTOTAL = 653979, 42241, 696220


def protocol():
    return {'version': 'fixed-lumina-public-QA-scoring-v1', 'primary': 'lumina',
            'methods': list(METHODS), 'classifier_fits': 0, 'coefficient_search': False,
            'token_scores': {'lumina': 'Use saved float32 column2: .5*IPR-.5*MMD_squared, unchanged.',
                             'ipr': 'Saved column0; higher means more risk.',
                             'negative_mmd': 'Minus saved column1; higher means more risk.'},
            'unused_columns': extract.NAMES[3:],
            'window': 'Original eligible4rawBPE token_indices, stride1. Float64 mean over actual raw tokens, including punctuation. Short answer uses its original short window. No lexical-only pooling, sigmoid, clipping or normalization.',
            'answermax': 'Maximum of ALL unchanged eligible window scores for that answer. Original official answer labels; do not derive answer labels from window OR or drop refusals.',
            'threshold': 'q.choose_threshold on original159 calibration only, separately for windows and answermax; >=, ties F1/precision/higher threshold. Apply both fixed thresholds to fit and calibration.',
            'cohort': {'fit_answers': 3680, 'calibration_answers': 159,
                       'fit_windows': NFIT, 'calibration_windows': NCAL, 'total_windows': NTOTAL,
                       'raw_answer_tokens': 708506},
            'paper_answer_mean': 'Separately average every raw answer token (including punctuation outside eligible windows), choose one cal answer threshold and report answer metrics only for each fixed score. Never pair this F1 with the answermax window result.',
            'reporting': 'All three readouts reported. No best-score selection; primary is always the fixed official formula. Report its window and answermax F1 together from that same formula; components remain ablations.',
            'gates': 'Before opening any new per-answer NPZ, require extractor complete3839/708506, all records validated, exact manifests/signature/plan binding. Any missing/invalid score blocks delivery; never reduce denominator.',
            'scope': 'Repeated public QA fit/cal development, not official-test performance. Input/model/precision/granularity adaptation to pinned LUMINA code is disclosed in upstream protocol.',
            'GPU_used': False, 'official_test_opened': False,
            'automatic_execution': False, 'fusion_grid': False}


def snapshot_paths():
    paths = [Path(__file__), Path(q.__file__), Path(expansion.__file__), Path(extract.__file__),
             ROOT / 'src/lumina_qa_signals.py', Path(feature.__file__),
             PREP / 'preparation_complete.json', PREP / 'CPU_OUTPUT_CHECK.json',
             PREP / 'protocol.json', PREP / 'feature_inputs.jsonl',
             ROOT / 'data/gold_manifest.json', ROOT / 'fit_expansion/data/export_freeze.json']
    for part in ('fit', 'calibration'):
        paths.extend(ROOT / f'data/{kind}_{part}.jsonl' for kind in ('answers', 'tokens', 'windows_k4'))
    paths.extend(ROOT / f'fit_expansion/data/{kind}_fit.jsonl' for kind in ('answers', 'tokens', 'windows_k4'))
    return paths


def require_complete(folder=FEATURES):
    """A manifest-only gate; no partial per-answer features are read here."""
    path = folder / 'features_complete.json'
    if not path.exists():
        raise FileNotFoundError(f'WAIT: full3839 LUMINA features_complete missing: {path}')
    done = q.read(path)
    assert done['status'] == 'complete' and done['records'] == 3839
    assert done['raw_answer_tokens'] == 708506 and done['all_records_validated']
    assert done['no_test'] and done['trained'] is False
    assert done['feature_names'] == extract.NAMES
    assert done['preparation_complete_sha256'] == q.sha(PREP / 'preparation_complete.json')
    for name, expected in done['files_sha256'].items():
        assert name in ('feature_manifest.json', 'signature.json', 'protocol.json', 'CPU_CHECK.json', 'GPU_SMOKE.json')
        assert q.sha(folder / name) == expected
    assert set(done['files_sha256']) == {'feature_manifest.json', 'signature.json', 'protocol.json', 'CPU_CHECK.json', 'GPU_SMOKE.json'}
    signature = q.read(folder / 'signature.json')
    assert q.digest(signature) == done['signature_sha256']
    manifest = q.read(folder / 'feature_manifest.json')
    assert manifest['status'] == 'complete' and manifest['records'] == len(manifest['entries']) == 3839
    assert manifest['raw_answer_tokens'] == 708506 and manifest['all_records_validated']
    assert manifest['signature_sha256'] == done['signature_sha256']
    assert manifest['feature_names'] == extract.NAMES
    assert not manifest['test_opened'] and not manifest['labels_used'] and not manifest['trained']
    return done, manifest


def token_columns(values):
    assert values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 7
    assert np.isfinite(values[:, :3]).all()
    assert np.array_equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
    # The other four probability columns never enter any score or threshold.
    return np.column_stack((values[:, 2], values[:, 0], -values[:, 1])).astype(np.float64)


def aggregate_answer(token_scores, windows):
    """No label argument: use raw geometry only, retain all supplied windows."""
    token_scores = np.asarray(token_scores, np.float64)
    assert token_scores.ndim == 2 and token_scores.shape[1] == 3 and len(token_scores) > 0
    assert np.isfinite(token_scores).all() and windows
    out = []
    for w in windows:
        indices = np.asarray(w['token_indices'], np.int64)
        assert 0 < len(indices) <= 4 and indices.tolist() == list(range(int(indices[0]), int(indices[-1]) + 1))
        assert indices[0] >= 0 and indices[-1] < len(token_scores)
        out.append(token_scores[indices].mean(axis=0, dtype=np.float64))
    means = np.asarray(out, np.float64)
    return means, means.max(axis=0), token_scores.mean(axis=0, dtype=np.float64)


def cpu_test():
    # Slot1 is punctuation; first example includes the final raw token in window2.
    s = np.repeat(np.asarray([0, 8, 0, 0, 4], np.float64)[:, None], 3, axis=1)
    w, a, p = aggregate_answer(s, [{'token_indices': [0, 1, 2, 3]}, {'token_indices': [1, 2, 3, 4]}])
    assert np.array_equal(w[:, 0], [2, 3]) and np.array_equal(a, [3, 3, 3])
    assert np.array_equal(p, [2.4, 2.4, 2.4])
    short = np.repeat(np.asarray([-2, -4], float)[:, None], 3, axis=1)
    sw, sa, sp = aggregate_answer(short, [{'token_indices': [0, 1]}])
    assert np.array_equal(sw, [[-3, -3, -3]]) and np.array_equal(sa, sp)
    # Raw trailing punctuation can fall outside every eligible window, but it
    # still contributes to the separately reported paper-style answer mean.
    six = np.repeat(np.asarray([0, 0, 0, 0, 6, 6], float)[:, None], 3, axis=1)
    _, xa, xp = aggregate_answer(six, [{'token_indices': [0, 1, 2, 3]}])
    assert np.array_equal(xa, [0, 0, 0]) and np.array_equal(xp, [2, 2, 2])
    f = np.zeros((2, 7), np.float32)
    f[:, 0] = [.4, .2]; f[:, 1] = [.2, .6]; f[:, 2] = .5 * f[:, 0] - .5 * f[:, 1]
    v = token_columns(f); f[:, 3:] = 9876
    assert np.array_equal(v, token_columns(f)) and np.array_equal(v[:, 2], -f[:, 1].astype(np.float64))
    y = np.asarray([0, 1, 1, 0, 1]); scores = np.asarray([-.3, -.2, -.2, .1, .4])
    threshold = q.choose_threshold(y, scores)
    brute = []
    for cut in [np.nextafter(scores.max(), np.inf), *np.unique(scores)]:
        m = q.count(y, scores, cut)
        brute.append((m['f1'], m['precision'], cut))
    assert (threshold['f1'], threshold['precision'], threshold['threshold']) == max(brute)
    with tempfile.TemporaryDirectory() as td:
        try:
            require_complete(Path(td))
            raise AssertionError('Partial cache accepted')
        except FileNotFoundError as e:
            assert 'WAIT:' in str(e)
    return {'status': 'passed', 'punctuation_retained': True, 'short_window_and_final_token': True,
            'answermax_not_paper_mean': True, 'paper_mean_includes_raw_tokens_outside_eligible_windows': True,
            'no_cross_answer_propagation': True, 'diagnostic_columns_ignored': True,
            'negative_scores_and_threshold_ties': True, 'missing_complete_blocks_before_NPZ': True,
            'new_scores_read': False, 'real_model_run': False, 'GPU_used': False, 'new_fits': 0}


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'preparation_complete.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'CPU_SELFCHECK.json', cpu_test())
    snap = {str(p.resolve()): q.sha(p) for p in snapshot_paths()}
    q.save(OUT / 'source_snapshot.json', {'files_sha256': snap, 'no_new_feature_file_read': True})
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_scored',
           'files_sha256': {n: q.sha(OUT / n) for n in ('protocol.json', 'CPU_SELFCHECK.json', 'source_snapshot.json')},
           'GPU_used': False, 'new_fits': 0, 'new_scores_read': False, 'official_test_opened': False})
    assert not torch.cuda.is_initialized()
    print('LUMINA_FIXED_SCORING_PREPARED_NOT_RUN', flush=True)


def check():
    assert not torch.cuda.is_initialized()
    p = q.read(OUT / 'preparation_complete.json')
    assert p['status'] == 'prepared_not_scored'
    for name, expected in p['files_sha256'].items():
        assert q.sha(OUT / name) == expected
    assert q.read(OUT / 'protocol.json') == protocol()
    for name, expected in q.read(OUT / 'source_snapshot.json')['files_sha256'].items():
        assert q.sha(name) == expected
    assert q.read(OUT / 'CPU_SELFCHECK.json')['status'] == 'passed'
    print('LUMINA_FIXED_SCORING_CHECKED_NO_NEW_SCORE_READ', flush=True)


def run():
    check()
    assert not (OUT / 'complete.json').exists(), 'Never overwrite completed scores'
    done, manifest = require_complete()  # Before metadata or any feature NPZ.
    rows = q.lines(PREP / 'feature_inputs.jsonl')
    _, meta = expansion.metadata()
    assert len(meta['answers']) == len(rows) == 3839 and len(meta['windows']) == NTOTAL
    assert meta['bounds'] == {'fit': [0, NFIT], 'calibration': [NFIT, NTOTAL]}
    assert [r['response_id'] for r in rows] == manifest['answer_order'] == [a['response_id'] for a in meta['answers']]
    assert q.digest(manifest['answer_order']) == q.read(PREP / 'preparation_complete.json')['response_order_sha256']
    started = time.perf_counter()
    windows = np.empty((NTOTAL, 3), np.float64)
    answermax = np.empty((3839, 3), np.float64)
    answermean = np.empty_like(answermax)
    raw_count = 0
    for i, (row, answer, tokens, entry) in enumerate(zip(rows, meta['answers'], meta['tokens'], manifest['entries'])):
        rid = row['response_id']; indices = meta['answer_windows'][rid]
        assert row['source_id'] == answer['source_id'] and row['group_id'] == answer['group_id']
        assert row['partition'] == answer['partition'] == ('fit' if i < 3680 else 'calibration')
        assert row['answer_sha256'] == answer['answer_sha256'] == tokens['answer_sha256']
        assert row['answer_token_ids'] == tokens['token_ids']
        assert row['response_token_offsets'] == tokens['response_token_offsets']
        assert row['response_token_offsets_raw'] == tokens['response_token_offsets_raw']
        assert row['original_answer_positions'] == tokens['answer_token_positions']
        assert entry['record_index'] == str(i) and entry['response_id'] == rid
        assert entry['record_sha256'] == q.digest(row)
        path = FEATURES / 'features' / f'{i:05d}.npz'
        assert entry['file'] == path.name and q.sha(path) == entry['npz_sha256']
        assert path.with_suffix('.json').exists() and q.read(path.with_suffix('.json')) == entry
        assert extract.validate_record(path.parent, i, row, done['signature_sha256'], False) == entry
        with np.load(path, allow_pickle=False) as z:
            ts = token_columns(z['lumina_features'])
        assert len(ts) == tokens['token_count']
        selected_windows = [meta['windows'][j] for j in indices]
        for w in selected_windows:
            assert w['eligible'] and w['response_id'] == rid
            assert len(w['token_indices']) == min(4, len(ts))
            assert any(tokens['lexical_mask'][j] for j in w['token_indices'])
            assert w['label'] == int(any(tokens['risk_mask'][j] for j in w['token_indices']))
        windows[indices], answermax[i], answermean[i] = aggregate_answer(ts, selected_windows)
        raw_count += len(ts)
    assert raw_count == 708506 and np.isfinite(windows).all()
    results, files = {}, []
    ay = np.asarray([a['label'] for a in meta['answers']])
    wy = np.asarray([w['label'] for w in meta['windows']])
    for j, name in enumerate(METHODS):
        ws, aa, pa = windows[:, j], answermax[:, j], answermean[:, j]
        assert np.array_equal(aa, q.answer_scores(meta, ws))
        thresholds = {'window': q.choose_threshold(wy[NFIT:], ws[NFIT:]),
                      'answer': q.choose_threshold(ay[3680:], aa[3680:])}
        paper_threshold = q.choose_threshold(ay[3680:], pa[3680:])
        file = OUT / f'{name}_scores.npz'
        np.savez_compressed(file, window_scores=ws, answermax_scores=aa, paper_answer_mean_scores=pa)
        result = {'method': name, 'primary': name == 'lumina', 'score_file': file.name,
                  'scores_sha256': q.sha(file), 'thresholds': thresholds,
                  'window_then_answermax_metrics': q.metrics(meta, ws, thresholds),
                  'paper_answer_mean': {'threshold': paper_threshold, 'metrics': {
                      'fit': q.count(ay[:3680], pa[:3680], paper_threshold['threshold']),
                      'calibration': q.count(ay[3680:], pa[3680:], paper_threshold['threshold'])}},
                  'new_fits': 0, 'no_new_coefficients': True}
        q.save(OUT / f'{name}_result.json', result)
        results[name] = result
        files.extend([file.name, f'{name}_result.json'])
    summary = {'primary': 'lumina', 'all_fixed_readouts': results, 'method_selection_performed': False,
               'feature_complete_sha256': q.sha(FEATURES / 'features_complete.json'),
               'window_order_sha256': q.digest([w['window_id'] for w in meta['windows']]),
               'answer_order_sha256': q.digest([a['response_id'] for a in meta['answers']]),
               'counts': {'fit_windows': NFIT, 'calibration_windows': NCAL, 'fit_answers': 3680,
                          'calibration_answers': 159, 'all_windows': NTOTAL, 'raw_answer_tokens': raw_count},
               'calibration_used_for_thresholds_not_independent_test': True,
               'seconds': time.perf_counter() - started, 'GPU_used': False, 'new_fits': 0,
               'official_test_opened': False}
    q.save(OUT / 'summary.json', summary)
    text = ['# LUMINA 固定公式公共 QA 计分', '',
            '所有数值为原159答校准表现；只选择阈值，没有训练分类器或选择混合系数。主公式固定，不从消融中挑最好者冒称LUMINA。', '',
            '| 固定分数 | 原4BPE窗口F1 | 同分数整答max F1 |', '|---|---:|---:|']
    for name, r in results.items():
        m = r['window_then_answermax_metrics']['calibration']
        text.append(f"| {name} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    text += ['', '论文式全答raw均值仅报告整答指标，不能和上表窗口F1拼成另一组双指标：', '',
             '| 固定分数 | 全答raw均值 F1 |', '|---|---:|']
    for name, r in results.items():
        text.append(f"| {name} | {r['paper_answer_mean']['metrics']['calibration']['f1']:.6f} |")
    text += ['', '窗口包含标点，整答标签与拒答处理保持原样。该重放使用统一Llama NF4，包含其他生成器原答；粒度/精度/输入模板适配见上游协议。官方test未打开。']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n', 'utf-8')
    files += ['summary.json', 'REPORT.md', 'protocol.json', 'CPU_SELFCHECK.json', 'preparation_complete.json', 'source_snapshot.json']
    q.save(OUT / 'complete.json', {'status': 'complete', 'new_fits': 0, 'GPU_used': False,
           'official_test_opened': False, 'feature_complete_sha256': summary['feature_complete_sha256'],
           'files_sha256': {name: q.sha(OUT / name) for name in files}})
    assert not torch.cuda.is_initialized()
    print((OUT / 'REPORT.md').read_text('utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check', 'run'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'check': check, 'run': run}[args.command]()
