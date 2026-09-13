"""Pinned LUMINA token score -> full-answer mean, with no fitted readout.

prepare/check never open feature records. score is an explicit complete-cache-only
command. verify independently recomputes means and rank/count metrics; it neither
fits nor chooses a deployed threshold. Existing localization scoring is untouched.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

import run_development as q
import run_lumina_qa_extraction_v1 as extraction

ROOT = q.ROOT
OUT = ROOT / 'results/lumina_official_answer_v1'
PREP = ROOT / 'results/lumina_qa_preparation_v1'
FEATURES = ROOT / 'results/lumina_qa_features_v1'
R7 = ROOT.parent / 'round7_evidence_grounding'
METHODS = ('lumina', 'ipr', 'negative_mmd')
NAMES = ['ipr', 'mmd', 'lumina', 'original_answer_probability',
         'original_max_probability', 'random_answer_probability', 'random_max_probability']
N, NFIT, NTOKENS = 3839, 3680, 708506
OFFICIAL_COMMIT = 'c43ff41d872b05f659dcb3ad3a6dd78226954319'
OFFICIAL_SHA = '1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f'


def protocol():
    return {
        'version': 'lumina-author-code-full-answer-mean-v1',
        'role': 'Appendix native-aggregation verification: author-code formula transfer and paper Eq2 full-answer mean; not the main unified-task comparison or original-paper numerical reproduction.',
        'unified_task_comparison': 'Existing score_lumina_qa_v1.py and lumina_qa_scoring_v1 remain the main fixed4-BPE/answermax task evaluation. External granularity mapping and uniform thresholds do not modify the frozen detector; this appendix neither replaces nor edits them.',
        'official_commit': OFFICIAL_COMMIT, 'official_code_sha256': OFFICIAL_SHA,
        'primary': 'lumina', 'ablations': ['ipr', 'negative_mmd'],
        'token_score': 'Saved float32 column2 = .5*IPR - .5*MMD_squared, unchanged. No new coefficients.',
        'aggregation': 'Float64 arithmetic mean over ALL original raw answer BPE, including punctuation, boundary-crossing first and final token. No lexical mask, windows, answermax, clipping or sigmoid.',
        'cohort': {'fit': 3680, 'calibration': 159, 'all_answers': N, 'all_raw_answer_tokens': NTOKENS},
        'labels': 'Unchanged official answer label (nonempty original annotation list), never reconstructed from token or window labels. All answers and refusals retained.',
        'main_metrics': ['AUROC', 'AUPRC_trapezoid', 'PCC'],
        'metric_definitions': {
            'AUROC': 'Unweighted roc_auc_score, risk-positive direction.',
            'AUPRC_trapezoid': 'Unweighted trapezoidal auc(recall, precision) from precision_recall_curve, tied scores grouped.',
            'AP': 'Also report noninterpolated average_precision_score explicitly; do not silently equate AP with trapezoidal PR-AUC.',
            'PCC': 'Pearson correlation between answer risk score and original binary answer label.',
            'F1Opt': 'Secondary Appendix E.1 evaluation statistic: maximum F1 over all achievable distinct-score cuts on the reported cohort. Return only the scalar optimum, never a threshold or a deployable policy.',
        },
        'selection': 'None: no trained readout, feature/weight/model selection, deployable threshold or cross-method winner. Report all three fixed scores.',
        'reporting': 'Calibration159 is the main development cohort; fit3680 is a separate descriptive cohort. No combined fit+cal result and no official-test claim.',
        'interface_adaptations': [
            'Original question, wrapper and all frozen answer IDs retained; two contexts share the same answer IDs.',
            'Whole fit-only donor material, shared across generators of a source, chosen without labels or scores.',
            'No author extra response-leading space, retokenization or12000-character truncation.',
        ],
        'numerical_disclosures': [
            'Llama2 NF4/BF16 replay with FP32 statistics is not historical native traces or fullprecision identity.',
            'Streaming all-layer IPR and algebraically equivalent unnormalized-top100 cosine MMD; memory reduction does not alter mathematical coefficients.',
            'Follow pinned code hidden_states[1:] including final layer and its repeated norm, rather than silently adopting paper Eq8 L-1.',
        ],
        'complete_gate': 'Require all3839/708506 and validate extractor signatures/hashes/axes before any score is computed. Missing records never reduce the denominator.',
        'commands': ['prepare', 'check', 'score', 'verify'], 'automatic_score': False,
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False,
    }


def source_paths():
    paths = [Path(__file__), Path(q.__file__), Path(extraction.__file__),
             ROOT / 'src/lumina_qa_signals.py', ROOT / 'src/feature_qa.py', ROOT / 'src/run_feature_qa.py',
             R7 / 'src/lumina7.py', R7 / 'references/lumina_official.py', R7 / 'references/lumina_source.json',
             PREP / 'protocol.json', PREP / 'feature_inputs.jsonl', PREP / 'preparation_complete.json',
             PREP / 'CPU_OUTPUT_CHECK.json', FEATURES / 'protocol.json', FEATURES / 'signature.json',
             ROOT / 'data/gold_manifest.json', ROOT / 'fit_expansion/data/export_freeze.json']
    for part in ('fit', 'calibration'):
        paths += [ROOT / f'data/answers_{part}.jsonl', ROOT / f'data/tokens_{part}.jsonl']
    paths += [ROOT / 'fit_expansion/data/answers_fit.jsonl', ROOT / 'fit_expansion/data/tokens_fit.jsonl']
    return paths


def require_complete(folder=FEATURES):
    path = folder / 'features_complete.json'
    if not path.exists():
        raise FileNotFoundError(f'WAIT: complete3839 LUMINA cache required: {path}')
    done = q.read(path)
    assert done['status'] == 'complete' and done['records'] == N
    assert done['raw_answer_tokens'] == NTOKENS and done['all_records_validated']
    assert done['feature_names'] == NAMES and done['no_test'] and done['trained'] is False
    assert done['preparation_complete_sha256'] == q.sha(PREP / 'preparation_complete.json')
    expected_files = {'feature_manifest.json', 'signature.json', 'protocol.json', 'CPU_CHECK.json', 'GPU_SMOKE.json'}
    assert set(done['files_sha256']) == expected_files
    for name, expected in done['files_sha256'].items():
        assert q.sha(folder / name) == expected
    signature = q.read(folder / 'signature.json')
    assert q.digest(signature) == done['signature_sha256']
    assert signature['official_commit'] == OFFICIAL_COMMIT and signature['lambda'] == .5
    assert signature['top_k'] == 100 and signature['feature_names'] == NAMES
    assert not signature['labels_used'] and not signature['trained'] and not signature['test_opened']
    for key in ('code_sha256', 'source_sha256'):
        for filename, expected in signature[key].items():
            assert q.sha(filename) == expected, filename
    manifest = q.read(folder / 'feature_manifest.json')
    assert manifest['status'] == 'complete' and manifest['records'] == len(manifest['entries']) == N
    assert manifest['raw_answer_tokens'] == NTOKENS and manifest['all_records_validated']
    assert manifest['signature_sha256'] == done['signature_sha256'] and manifest['feature_names'] == NAMES
    assert not manifest['labels_used'] and not manifest['trained'] and not manifest['test_opened']
    return done, manifest


def full_answer_mean(values):
    assert values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 7 and len(values)
    assert np.isfinite(values[:, :3]).all()
    assert np.array_equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
    return np.asarray([values[:, 2].mean(dtype=np.float64), values[:, 0].mean(dtype=np.float64),
                       -values[:, 1].mean(dtype=np.float64)], dtype=np.float64)


def metrics(labels, scores):
    y, s = np.asarray(labels, np.int64), np.asarray(scores, np.float64)
    assert y.shape == s.shape and y.ndim == 1 and len(y) and set(y) <= {0, 1} and np.isfinite(s).all()
    positive = int(y.sum())
    result = {'answers': len(y), 'positive_answers': positive, 'negative_answers': len(y) - positive}
    if 0 < positive < len(y):
        precision, recall, _ = precision_recall_curve(y, s)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=(precision + recall) > 0)
        result.update(AUROC=float(roc_auc_score(y, s)), AUPRC_trapezoid=float(auc(recall, precision)),
                      AP=float(average_precision_score(y, s)), F1Opt=float(f1.max()))
    else:
        result.update(AUROC=None, AUPRC_trapezoid=None, AP=None,
                      F1Opt=1.0 if positive == len(y) else 0.0)
    centered_y, centered_s = y - y.mean(), s - s.mean()
    denominator = np.linalg.norm(centered_y) * np.linalg.norm(centered_s)
    result['PCC'] = float(np.dot(centered_y, centered_s) / denominator) if denominator else None
    return result


def independent_metrics(labels, scores):
    """Rank sums, tied descending counts and scalar sums; no sklearn metrics."""
    y = list(map(int, labels)); s = list(map(float, scores)); n = len(y); p = sum(y); m = n - p
    result = {'answers': n, 'positive_answers': p, 'negative_answers': m}
    ascending = sorted(range(n), key=lambda i: s[i]); rank_sum = 0.; cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and s[ascending[end]] == s[ascending[cursor]]:
            end += 1
        rank_sum += ((cursor + 1 + end) / 2) * sum(y[ascending[i]] for i in range(cursor, end))
        cursor = end
    if p and m:
        order = sorted(range(n), key=lambda i: -s[i]); cursor = 0; tp = 0; prev_r = 0.; prev_p = 1.
        pr_area = ap = best_f1 = 0.
        while cursor < n:
            end = cursor + 1
            while end < n and s[order[end]] == s[order[cursor]]:
                end += 1
            tp += sum(y[order[i]] for i in range(cursor, end))
            recall = tp / p; precision = tp / end
            pr_area += (recall - prev_r) * (precision + prev_p) / 2
            ap += (recall - prev_r) * precision
            best_f1 = max(best_f1, 2 * tp / (end + p))
            prev_r, prev_p, cursor = recall, precision, end
        result.update(AUROC=(rank_sum - p * (p + 1) / 2) / (p * m),
                      AUPRC_trapezoid=pr_area, AP=ap, F1Opt=best_f1)
    else:
        result.update(AUROC=None, AUPRC_trapezoid=None, AP=None, F1Opt=1.0 if p == n else 0.0)
    ym = p / n; sm = math.fsum(s) / n
    numerator = math.fsum((a - ym) * (b - sm) for a, b in zip(y, s))
    denominator = math.sqrt(math.fsum((a - ym) ** 2 for a in y) * math.fsum((b - sm) ** 2 for b in s))
    result['PCC'] = numerator / denominator if denominator else None
    return result


def assert_metrics_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if left[key] is None or right[key] is None:
            assert left[key] is right[key], key
        else:
            assert abs(left[key] - right[key]) <= 1e-12, (key, left[key], right[key])


def cpu_test():
    values = np.zeros((6, 7), dtype=np.float32)
    values[:, 0] = [0, 8, 0, 0, 4, 6]
    values[:, 1] = [.5, 0, .25, 0, 0, 1]
    values[:, 2] = .5 * values[:, 0] - .5 * values[:, 1]
    expected = np.asarray([math.fsum(map(float, values[:, j])) / 6 for j in (2, 0, 1)])
    expected[2] *= -1
    assert np.array_equal(full_answer_mean(values), expected)
    changed = values.copy(); changed[:, 3:] = 9876
    assert np.array_equal(full_answer_mean(values), full_answer_mean(changed))
    assert np.array_equal(full_answer_mean(values[:1]), np.asarray([values[0, 2], values[0, 0], -values[0, 1]], float))
    for y, s in [([0, 1, 1, 0, 1], [-.3, -.2, -.2, .1, .4]),
                 ([0, 1, 0, 1], [1, 1, 1, 1]), ([0, 0], [-3, 4]), ([1, 1], [-2, -1])]:
        assert_metrics_equal(metrics(y, s), independent_metrics(y, s))
    with tempfile.TemporaryDirectory() as temp:
        try:
            require_complete(Path(temp))
            raise AssertionError('Incomplete cache was accepted')
        except FileNotFoundError as error:
            assert 'WAIT:' in str(error)
    return {'status': 'passed', 'all_raw_tokens_equal_weight': True, 'short_and_final_token': True,
            'diagnostic_columns_do_not_affect_scores': True, 'negative_scores_and_ties': True,
            'independent_rank_AP_PRarea_PCC_F1Opt': True, 'F1Opt_returns_no_threshold': True,
            'missing_complete_rejected_before_feature_read': True, 'new_features_read': False,
            'model_run': False, 'GPU_used': False, 'new_fits': 0, 'official_test_opened': False}


def prepare():
    assert not (OUT / 'preparation_started.json').exists(), 'Preparation is immutable; no overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'preparation_started.json', {'status': 'started_no_features_opened'})
    assert q.sha(R7 / 'references/lumina_official.py') == OFFICIAL_SHA
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'CPU_SELFCHECK.json', cpu_test())
    q.save(OUT / 'source_snapshot.json', {'files_sha256': {str(p.resolve()): q.sha(p) for p in source_paths()}})
    plan = ('# LUMINA 作者原生整答聚合：附录核验\n\n'
            'A层：本目录仅复核每答全部原始 token 的作者分数均值，固定LUMINA；IPR、负MMD为消融。'
            'B层：统一4-BPE/answermax及统一阈值仍由旧score_lumina_qa_v1.py负责，是主任务比较；旧文件不改。\n\n'
            '主指标 AUROC、梯形积分 AUPRC、PCC；另明确列 AP。F1Opt 仅为论文式评测统计，不保存阈值，不用于选择。'
            '没有4-BPE、整答最大值、训练、调权或部署阈值。cal159仍是已使用的开发集。\n\n'
            'prepare/check不读特征；score须3839答缓存完整。verify独立用逐列math.fsum与并列排名/计数重算。'
            '四命令均显式调用，prepare/check不自动启动score。旧计分和特征目录只读。\n')
    (OUT / 'PLAN.md').write_text(plan, encoding='utf-8')
    names = ['protocol.json', 'CPU_SELFCHECK.json', 'source_snapshot.json', 'PLAN.md']
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_not_scored',
           'files_sha256': {name: q.sha(OUT / name) for name in names},
           'new_features_read': False, 'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})
    print('LUMINA_OFFICIAL_ANSWER_PREPARED_NOT_SCORED', flush=True)


def check():
    prepared = q.read(OUT / 'preparation_complete.json')
    assert prepared['status'] == 'prepared_not_scored'
    for name, expected in prepared['files_sha256'].items():
        assert q.sha(OUT / name) == expected, name
    assert q.read(OUT / 'protocol.json') == protocol()
    for name, expected in q.read(OUT / 'source_snapshot.json')['files_sha256'].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / 'CPU_SELFCHECK.json')['status'] == 'passed'
    print('LUMINA_OFFICIAL_ANSWER_CHECKED_NO_FEATURE_READ', flush=True)


def answer_metadata():
    # No windows are loaded. The exact released answer label remains independent
    # of lexical-token evaluability, including punctuation-only annotations.
    aa = q.lines(ROOT / 'fit_expansion/data/answers_fit.jsonl')
    tt = q.lines(ROOT / 'fit_expansion/data/tokens_fit.jsonl')
    assert len(aa) == len(tt) == NFIT
    assert aa[:634] == q.lines(ROOT / 'data/answers_fit.jsonl')
    assert tt[:634] == q.lines(ROOT / 'data/tokens_fit.jsonl')
    aa += q.lines(ROOT / 'data/answers_calibration.jsonl')
    tt += q.lines(ROOT / 'data/tokens_calibration.jsonl')
    assert len(aa) == len(tt) == N
    assert len({a['response_id'] for a in aa}) == N
    for i, (a, t) in enumerate(zip(aa, tt)):
        assert a['partition'] == t['partition'] == ('fit' if i < NFIT else 'calibration')
        assert a['response_id'] == t['response_id'] and a['answer_sha256'] == t['answer_sha256']
        assert a['label'] == int(bool(a['original_labels'])) == t['answer_risk']
    groups = {p: {a['group_id'] for a in aa if a['partition'] == p} for p in ('fit', 'calibration')}
    assert len(groups['fit']) == 615 and len(groups['calibration']) == 154
    assert not groups['fit'] & groups['calibration']
    return aa, tt


def collect(independent=False):
    done, manifest = require_complete()  # Must precede annotation or per-answer feature reads.
    rows = q.lines(PREP / 'feature_inputs.jsonl'); answers, tokens = answer_metadata()
    order = [a['response_id'] for a in answers]
    assert len(rows) == N and [r['response_id'] for r in rows] == manifest['answer_order'] == order
    assert q.digest(order) == q.read(PREP / 'preparation_complete.json')['response_order_sha256']
    scores = np.empty((N, 3), np.float64); raw_count = 0
    for i, (r, a, t, entry) in enumerate(zip(rows, answers, tokens, manifest['entries'])):
        for key in ('response_id', 'source_id', 'group_id', 'partition', 'answer_sha256'):
            assert r[key] == a[key], (i, key)
        assert r['answer_token_ids'] == t['token_ids']
        assert r['response_token_offsets'] == t['response_token_offsets']
        assert r['response_token_offsets_raw'] == t['response_token_offsets_raw']
        assert r['original_answer_positions'] == t['answer_token_positions']
        assert entry['record_index'] == str(i) and entry['response_id'] == r['response_id']
        assert entry['record_sha256'] == q.digest(r)
        path = FEATURES / 'features' / f'{i:05d}.npz'
        assert entry['file'] == path.name and q.sha(path) == entry['npz_sha256']
        assert q.read(path.with_suffix('.json')) == entry
        assert extraction.validate_record(path.parent, i, r, done['signature_sha256'], False) == entry
        with np.load(path, allow_pickle=False) as data:
            values = data['lumina_features']
            assert values.shape == (t['token_count'], 7)
            if independent:
                assert np.array_equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
                scores[i] = [math.fsum(map(float, values[:, j])) / len(values) for j in (2, 0, 1)]
                scores[i, 2] *= -1
            else:
                scores[i] = full_answer_mean(values)
            raw_count += len(values)
    assert raw_count == NTOKENS and np.isfinite(scores).all()
    labels = np.asarray([a['label'] for a in answers], np.int64)
    return scores, labels, order, done


def summarize(scores, labels, metric_fn=metrics):
    return {name: {part: metric_fn(labels[left:right], scores[left:right, j])
                   for part, left, right in [('fit', 0, NFIT), ('calibration', NFIT, N)]}
            for j, name in enumerate(METHODS)}


def score():
    check()
    assert not (OUT / 'complete.json').exists(), 'Never overwrite complete scores'
    assert not (OUT / 'score_started.json').exists(), 'Interrupted run retained; no silent replacement'
    require_complete()
    q.save(OUT / 'score_started.json', {'status': 'started', 'preparation_sha256': q.sha(OUT / 'preparation_complete.json'),
           'feature_complete_sha256': q.sha(FEATURES / 'features_complete.json')})
    started = time.perf_counter()
    scores, labels, order, done = collect()
    path = OUT / 'answer_scores.npz'; pending = path.with_suffix('.npz.pending')
    assert not path.exists()
    with pending.open('wb') as handle:
        np.savez_compressed(handle, scores=scores, methods=np.asarray(METHODS), labels=labels,
                            response_ids=np.asarray(order), partitions=np.asarray(['fit'] * NFIT + ['calibration'] * (N - NFIT)))
    pending.replace(path)
    results = summarize(scores, labels)
    summary = {'status': 'complete_author_code_formula_transfer_development_only', 'primary': 'lumina',
               'results': results, 'answer_order_sha256': q.digest(order), 'raw_tokens': NTOKENS,
               'feature_signature_sha256': done['signature_sha256'],
               'feature_complete_sha256': q.sha(FEATURES / 'features_complete.json'),
               'answer_scores_sha256': q.sha(path), 'seconds': time.perf_counter() - started,
               'F1Opt_is_evaluation_statistic_not_deployment': True, 'threshold_saved': False,
               'model_selection': False, 'new_fits': 0, 'GPU_used': False, 'official_test_opened': False}
    q.save(OUT / 'summary.json', summary)
    report = ['# LUMINA 作者原生全答均值：附录核验', '',
              '这是A层作者原生聚合复核；B层统一任务主比较仍使用旧4-BPE/answermax入口，未被本附录替代。', '',
              '固定作者分数，每答全部原始token算术均值。主方法固定LUMINA，后两行仅消融。', '',
              '| cal159 | AUROC | AUPRC（梯形） | PCC | AP | F1Opt（评测统计） |',
              '|---|---:|---:|---:|---:|---:|']
    for name in METHODS:
        r = results[name]['calibration']
        row = [name] + [('N/A' if r[k] is None else f'{r[k]:.6f}')
                        for k in ('AUROC', 'AUPRC_trapezoid', 'PCC', 'AP', 'F1Opt')]
        report.append('| ' + ' | '.join(row) + ' |')
    report += ['', 'F1Opt仅复现论文附录评测项；没有保存部署阈值，也没有用它选择公式或模型。'
               '无4-BPE、answermax、分类器或调权。fit描述指标见summary。', '',
               '这是作者代码公式与论文整答聚合的公共QA迁移。NF4重放、模板和donor接口差异见protocol；'
               'cal159为开发集，官方test150未读。']
    pending_report = OUT / 'REPORT.md.pending'
    pending_report.write_text('\n'.join(report) + '\n', encoding='utf-8')
    pending_report.replace(OUT / 'REPORT.md')
    q.save(OUT / 'complete.json', {'status': 'complete', 'files_sha256': {
        name: q.sha(OUT / name) for name in ('answer_scores.npz', 'summary.json', 'REPORT.md', 'protocol.json',
                                          'preparation_complete.json', 'source_snapshot.json', 'CPU_SELFCHECK.json')},
        'feature_complete_sha256': summary['feature_complete_sha256'], 'new_fits': 0,
        'threshold_saved': False, 'GPU_used': False, 'official_test_opened': False})
    print('LUMINA_OFFICIAL_ANSWER_SCORED_ALL3839', flush=True)


def verify():
    check(); complete = q.read(OUT / 'complete.json')
    assert complete['status'] == 'complete' and complete['new_fits'] == 0 and complete['threshold_saved'] is False
    assert complete['feature_complete_sha256'] == q.sha(FEATURES / 'features_complete.json')
    for name, expected in complete['files_sha256'].items():
        assert q.sha(OUT / name) == expected, name
    with np.load(OUT / 'answer_scores.npz', allow_pickle=False) as data:
        saved = data['scores'].copy(); saved_y = data['labels'].copy(); saved_order = data['response_ids'].tolist()
        assert data['methods'].tolist() == list(METHODS)
        assert data['partitions'].tolist() == ['fit'] * NFIT + ['calibration'] * (N - NFIT)
    independent, labels, order, _ = collect(independent=True)
    assert order == saved_order and np.array_equal(labels, saved_y)
    assert saved.shape == independent.shape == (N, 3) and saved.dtype == np.float64
    error = float(np.max(np.abs(saved - independent)))
    assert np.allclose(saved, independent, rtol=1e-12, atol=1e-12)
    oracle = summarize(independent, labels, independent_metrics)
    summary = q.read(OUT / 'summary.json')
    assert summary['primary'] == 'lumina' and summary['answer_order_sha256'] == q.digest(order)
    assert summary['answer_scores_sha256'] == q.sha(OUT / 'answer_scores.npz')
    for name in METHODS:
        for part in ('fit', 'calibration'):
            assert_metrics_equal(summary['results'][name][part], oracle[name][part])
    audit = {'status': 'passed', 'complete_sha256': q.sha(OUT / 'complete.json'),
             'answers': N, 'raw_tokens': NTOKENS, 'all_answer_mean_max_abs_error': error,
             'independent_rank_and_count_metrics': True, 'all_three_fixed_scores_verified': True,
             'F1Opt_scalar_only': True, 'threshold_saved': False, 'new_fits': 0,
             'GPU_used': False, 'official_test_opened': False}
    audit_path = OUT / 'INDEPENDENT_VERIFY.json'
    if audit_path.exists():
        assert q.read(audit_path) == audit
    else:
        q.save(audit_path, audit)
    print('LUMINA_OFFICIAL_ANSWER_INDEPENDENT_VERIFY_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check', 'score', 'verify'))
    args = parser.parse_args()
    {'prepare': prepare, 'check': check, 'score': score, 'verify': verify}[args.command]()
