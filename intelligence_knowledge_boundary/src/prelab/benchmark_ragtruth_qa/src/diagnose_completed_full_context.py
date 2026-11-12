"""Fixed-score QA diagnostics, gated on all six full-context epochs completing.

No model loading, inference, fitting, threshold selection or official test access.
The caller must explicitly invoke run after the parent schedules this analysis.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import run_development as q
from diagnose_completed_tail import KINDS

OUT = q.ROOT / 'results/completed_full_context_diagnostics_v1'
FULL = q.ROOT / 'results/full_context_encoder_v2/full_finetune'
TAIL = q.ROOT / 'results/minicheck_tail_all_docs_v3/tail2'
BEST = q.ROOT / 'results/citation_alignment_lr_v1'
BEST_KEY = 'harp_claim__two_scores_and_citation'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'prepared_not_run', 'full_context_completion_required': str(FULL / 'complete.json'),
        'models': ['full_context_v2', 'tail2', 'harp_tail_citation'],
        'selection': 'Read each already completed selected entry; never select epochs or thresholds here.',
        'calibration': {'answers': 159, 'windows_4raw_bpe': 42241, 'risk_windows': 5984,
                        'risk_answers': 100, 'normal_answers': 59},
        'types': list(KINDS), 'type_metrics': 'Recall only; original type masks may overlap.',
        'comparisons': 'Full-context versus fixed HARP+tail+8-feature LR, plus full-context versus tail2.',
        'scope': 'Repeatedly used development calibration; unequal training pipelines disclosed. Complementary correct predictions do not guarantee a deployable fusion gain.',
        'no_new_training_inference_thresholds_or_test': True, 'code_sha256': q.sha(Path(__file__))}
    path = OUT / 'PLAN.json'
    if path.exists():
        assert q.read(path) == plan
    else:
        q.save(path, plan)
    print('DIAGNOSTIC_PREPARED_NO_SCORES_READ', flush=True)


def ready():
    missing = [str(p / 'complete.json') for p in (FULL, TAIL, BEST) if not (p / 'complete.json').exists()]
    print('WAIT_COMPLETE_NO_SCORES_READ' if missing else 'COMPLETION_FILES_PRESENT_NOT_RUN', missing, flush=True)
    return not missing


def agreement(y, a, b):
    """Counts at the two ORIGINAL thresholds; no combined decision rule."""
    y, a, b = np.asarray(y, bool), np.asarray(a, bool), np.asarray(b, bool)
    assert y.shape == a.shape == b.shape
    positive = {'n': int(y.sum()), 'both_detected': int((y & a & b).sum()),
                'only_a_detected': int((y & a & ~b).sum()),
                'only_b_detected': int((y & ~a & b).sum()), 'both_missed': int((y & ~a & ~b).sum())}
    negative = {'n': int((~y).sum()), 'both_false_positive': int((~y & a & b).sum()),
                'only_a_false_positive': int((~y & a & ~b).sum()),
                'only_b_false_positive': int((~y & ~a & b).sum()),
                'both_correct_negative': int((~y & ~a & ~b).sum())}
    assert sum(v for k, v in positive.items() if k != 'n') == positive['n']
    assert sum(v for k, v in negative.items() if k != 'n') == negative['n']
    return {'positive': positive, 'negative': negative,
            'all': {'both_correct': int(((a == y) & (b == y)).sum()),
                    'only_a_correct': int(((a == y) & (b != y)).sum()),
                    'only_b_correct': int(((a != y) & (b == y)).sum()),
                    'both_wrong': int(((a != y) & (b != y)).sum())}}


def self_check():
    # Each possible positive/negative pair decision occurs exactly once.
    y = [1] * 4 + [0] * 4
    a = [1, 1, 0, 0] * 2
    b = [1, 0, 1, 0] * 2
    r = agreement(y, a, b)
    assert all(v == 1 for k, v in r['positive'].items() if k != 'n')
    assert all(v == 1 for k, v in r['negative'].items() if k != 'n')
    assert all(v == 2 for v in r['all'].values())
    print('SYNTHETIC_OVERLAP_COUNTS_PASSED_NO_DATA_READ', flush=True)


def epoch_scores(directory, y, ay, *, require_six=False):
    complete_path = directory / 'complete.json'
    complete = q.read(complete_path)
    assert not complete.get('test_opened', complete.get('official_test_opened', False))
    if require_six:
        assert {e['epoch'] for e in complete['all_epochs']} == set(range(7))
        assert complete['selected']['epoch'] in range(1, 7)
    entry = complete['selected']
    path = directory / f"epoch_{entry['epoch']:02d}_scores.npz"
    assert q.sha(path) == entry['artifacts_sha256']['_scores.npz']
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z['cal_window_labels'], y)
        assert np.array_equal(z['cal_answer_labels'], ay)
        scores = z['cal_window_scores'].copy(), z['cal_answer_scores'].copy()
    provenance = {'complete_sha256': q.sha(complete_path), 'scores_sha256': q.sha(path),
                  'path': str(path), 'selected_epoch': entry['epoch']}
    return entry, scores, provenance


def run():
    # Check completion BEFORE metadata or any selected/partial-epoch score file.
    if not ready():
        return
    assert not (OUT / 'complete.json').exists(), 'Preserve completed diagnostics'
    prepare()
    meta = q.metadata()
    lo, hi = meta['bounds']['calibration']
    windows = meta['windows'][lo:hi]
    answers = [a for a in meta['answers'] if a['partition'] == 'calibration']
    y = np.asarray([w['label'] for w in windows], np.int8)
    ay = np.asarray([a['label'] for a in answers], np.int8)
    assert (len(y), int(y.sum()), len(ay), int(ay.sum())) == (42241, 5984, 159, 100)
    typed = {kind: defaultdict(set) for kind in KINDS}
    for token in meta['tokens']:
        if token['partition'] != 'calibration':
            continue
        for span in token['span_token_mapping']:
            kind = token['original_labels'][span['span_index']]['label_type']
            typed[kind][token['response_id']].update(span['risk_token_indices'])
    masks = {k: np.asarray([bool(set(w['token_indices']) & typed[k][w['response_id']]) for w in windows]) for k in KINDS}
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), y.astype(bool))
    clean = {a['response_id'] for a in answers if a['label'] == 0}
    clean_windows = np.asarray([w['response_id'] in clean for w in windows])
    assert not y[clean_windows].any()
    source = {
        'full_context_v2': epoch_scores(FULL, y, ay, require_six=True),
        'tail2': epoch_scores(TAIL, y, ay)}
    completion = q.read(BEST / 'complete.json')
    assert not completion['official_test_opened']
    assert q.sha(BEST / 'summary.json') == completion['summary_sha256']
    e = q.read(BEST / 'summary.json')['selected'][BEST_KEY]
    p = BEST / (e['candidate'] + '_scores.npz')
    assert q.sha(p) == e['scores_sha256']
    with np.load(p, allow_pickle=False) as z:
        assert len(z['window_scores']) == len(meta['windows'])
        assert np.array_equal(q.answer_scores(meta, z['window_scores']), z['answer_scores'])
        pair = z['window_scores'][lo:hi].copy(), z['answer_scores'][634:].copy()
    source['harp_tail_citation'] = (dict(e, calibration=e['metrics']['calibration']), pair,
        {'complete_sha256': q.sha(BEST / 'complete.json'), 'scores_sha256': q.sha(p),
         'path': str(p), 'candidate': e['candidate']})
    models, wp, ap = {}, {}, {}
    for name, (entry, (score, ascore), origin) in source.items():
        assert np.isfinite(score).all() and np.isfinite(ascore).all()
        thresholds = entry['thresholds']
        wm = q.count(y, score, thresholds['window']['threshold'])
        am = q.count(ay, ascore, thresholds['answer']['threshold'])
        assert wm == entry['calibration']['windows'] and am == entry['calibration']['answers'], name
        wp[name] = score >= thresholds['window']['threshold']
        ap[name] = ascore >= thresholds['answer']['threshold']
        pred = wp[name]
        clean_alert_ids = {w['response_id'] for w, flag in zip(windows, pred) if flag and w['response_id'] in clean}
        clean_fp = int((pred & (y == 0) & clean_windows).sum())
        internal_fp = int((pred & (y == 0) & ~clean_windows).sum())
        assert clean_fp + internal_fp == wm['fp']
        types = {k: {'positive_windows': int(mask.sum()), 'detected_windows': int(pred[mask].sum()),
                     'missed_windows': int((~pred[mask]).sum()), 'recall': float(pred[mask].mean())} for k, mask in masks.items()}
        models[name] = {'window_metrics': wm, 'answer_metrics': am, 'thresholds': thresholds,
            'by_type_recall_only': types,
            'false_positive_windows_in_clean_answers': clean_fp,
            'clean_answer_negative_windows': int(clean_windows.sum()),
            'false_positive_windows_in_risky_answers': internal_fp,
            'risky_answer_negative_windows': int(((y == 0) & ~clean_windows).sum()),
            'normal_answers_false_positive_at_answer_threshold': am['fp'],
            'normal_answer_false_positive_rate': am['fp'] / 59,
            'normal_answers_with_any_window_alert_at_window_threshold': len(clean_alert_ids),
            'source': origin}
    comparisons = {}
    for name in ('harp_tail_citation', 'tail2'):
        a, b = 'full_context_v2', name
        comparisons[f'{a}_versus_{b}'] = {'a': a, 'b': b,
            'windows': agreement(y, wp[a], wp[b]), 'answers': agreement(ay, ap[a], ap[b]),
            'by_type_positive_windows': {k: agreement(y[mask], wp[a][mask], wp[b][mask])['positive'] for k, mask in masks.items()}}
    report = {'models': models, 'comparisons': comparisons,
        'calibration': {'answers': 159, 'risk_answers': 100, 'normal_answers': 59,
                        'windows': 42241, 'risk_windows': 5984, 'normal_windows': 36257},
        'fit_sizes': {'full_context_v2': 3680, 'tail2': 3680,
                      'harp_tail_citation': 'HARP and final LR fit634; tail component fit3680. Unequal training pipelines.'},
        'window_definition': 'Original overlapping4rawBPE stride1; unchanged lexical risk mapping. Not exact word-level F1.',
        'type_masks_can_overlap': True, 'type_f1_not_defined_or_reported': True,
        'fixed_completed_models_and_original_thresholds': True,
        'no_fusion_rule_or_oracle_model_created': True, 'official_test_opened': False,
        'training_inference_or_GPU_used': False,
        'interpretation_limit': 'Repeated development data. Rescued misses show potential complementarity with its false-positive cost; neither they nor shared misses alone establish inadequate training data or a realizable improvement.'}
    q.save(OUT / 'DIAGNOSTICS.json', report)
    text = ['# 完整资料模型与已有组合的固定分数诊断', '',
        '六轮完成后读取已固定的选中模型；只用原校准159答/42241个重叠4BPE窗和各自原阈值，没有训练、推理、改阈值或读取测试。', '',
        '| 方法 | 窗口TP/FP/FN/TN | 窗口F1 | 整答TP/FP/FN/TN | 整答F1 | 正常回答内误报窗 | 风险回答内误报窗 |',
        '|---|---|---:|---|---:|---:|---:|']
    for name, d in models.items():
        w, a = d['window_metrics'], d['answer_metrics']
        wc = '/'.join(str(w[k]) for k in ('tp', 'fp', 'fn', 'tn'))
        ac = '/'.join(str(a[k]) for k in ('tp', 'fp', 'fn', 'tn'))
        text.append(f"| {name} | {wc} | {w['f1']:.4f} | {ac} | {a['f1']:.4f} | {d['false_positive_windows_in_clean_answers']} | {d['false_positive_windows_in_risky_answers']} |")
    text += ['', '| 人工风险类型 | 原风险窗数 | 完整资料召回 | tail2召回 | HARP组合召回 |', '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        values = [models[n]['by_type_recall_only'][kind] for n in models]
        text.append(f"| {kind} | {values[0]['positive_windows']} | " + ' | '.join(f"{v['detected_windows']}/{v['positive_windows']} ({v['recall']:.1%})" for v in values) + ' |')
    c = comparisons['full_context_v2_versus_harp_tail_citation']['windows']
    p, n = c['positive'], c['negative']
    text += ['', f"5984个风险窗：共同检出 {p['both_detected']}；仅完整资料检出 {p['only_a_detected']}；仅HARP组合检出 {p['only_b_detected']}；共同漏检 {p['both_missed']}。",
        f"36257个正常窗：共同误报 {n['both_false_positive']}；仅完整资料误报 {n['only_a_false_positive']}；仅HARP组合误报 {n['only_b_false_positive']}；共同正确 {n['both_correct_negative']}。", '',
        '四类只计算召回，类型之间可重叠；不把其他风险类型作为负类计算类型F1。正常回答整答误报和任一窗口报警分别记录，因原两个阈值不同而不可混用。',
        '完整资料与tail2均使用3680训练答；HARP及最终LR使用634答，其tail分支使用3680答。因此不能把本表视为同训练流程的纯结构排名。',
        '互补检出也伴随误报成本，不能直接当成可部署组合收益；共同漏检也不能单凭此证明缺训练样本。该校准集已反复开发，测试仍封存。']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'diagnostics_sha256': q.sha(OUT / 'DIAGNOSTICS.json'),
        'code_sha256': q.sha(Path(__file__)), 'official_test_opened': False, 'GPU_used': False})
    print((OUT / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check-ready', 'self-check', 'run'))
    action = parser.parse_args().action
    {'prepare': prepare, 'check-ready': ready, 'self-check': self_check, 'run': run}[action]()
