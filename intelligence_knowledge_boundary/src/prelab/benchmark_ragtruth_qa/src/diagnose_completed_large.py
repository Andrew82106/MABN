"""Fixed-score error comparison after the six-epoch large run and matched fusion.

No model fitting, inference, new threshold choice, or official-test access.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import run_development as q
from diagnose_completed_tail import KINDS
from diagnose_completed_full_context import agreement, epoch_scores

OUT = q.ROOT / 'results/completed_large_diagnostics_v1'
LARGE = q.ROOT / 'results/full_context_encoder_large_v1/full_finetune'
BASE = q.ROOT / 'results/full_context_encoder_v2/full_finetune'
COMBO = q.ROOT / 'results/large_matched_combination_v2'


def prepare():
    assert not (OUT / 'PLAN.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'PLAN.json', {
        'code_sha256': q.sha(Path(__file__)),
        'scope': 'Original159 calibration answers/42241 overlapping4rawBPE windows only.',
        'selection': 'Use already completed selected epochs and all six selected matched-combination families. No further winner selection or threshold fitting.',
        'analysis': 'Base versus large training curves; fixed-threshold confusion, original four-type recall, clean-answer versus risky-answer false positives, and large-versus-base error overlap.',
        'limits': 'Repeated development calibration; type masks overlap, so recall only. Different model dimensions and optimizer foreach behavior; no isolated causal size claim. No deployable oracle is constructed.',
        'official_test_opened': False, 'new_fits': 0, 'GPU_used': False})
    print('LARGE_DIAGNOSIS_PREPARED_NO_SCORES_READ', flush=True)


def run():
    assert q.read(OUT / 'PLAN.json')['code_sha256'] == q.sha(Path(__file__))
    assert not (OUT / 'complete.json').exists()
    for d in (LARGE, BASE, COMBO):
        assert (d / 'complete.json').exists(), f'Wait for completion: {d}'
    meta = q.metadata()
    lo, hi = meta['bounds']['calibration']
    windows = meta['windows'][lo:hi]
    answers = meta['answers'][634:]
    y = np.asarray([w['label'] for w in windows], np.int8)
    ay = np.asarray([a['label'] for a in answers], np.int8)
    assert (len(y), int(y.sum()), len(ay), int(ay.sum())) == (42241, 5984, 159, 100)
    typed = {k: defaultdict(set) for k in KINDS}
    for token in meta['tokens']:
        if token['partition'] != 'calibration':
            continue
        for span in token['span_token_mapping']:
            kind = token['original_labels'][span['span_index']]['label_type']
            typed[kind][token['response_id']].update(span['risk_token_indices'])
    masks = {k: np.asarray([bool(set(w['token_indices']) & typed[k][w['response_id']])
                           for w in windows]) for k in KINDS}
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), y.astype(bool))
    clean_ids = {a['response_id'] for a in answers if not a['label']}
    clean = np.asarray([w['response_id'] in clean_ids for w in windows])
    assert not y[clean].any()
    sources = {
        'generic_base': epoch_scores(BASE, y, ay, require_six=True),
        'generic_large': epoch_scores(LARGE, y, ay, require_six=True)}
    completed = q.read(COMBO / 'complete.json')
    assert not completed['official_test_opened']
    assert q.sha(COMBO / 'summary.json') == completed['summary_sha256']
    summary = q.read(COMBO / 'summary.json')
    assert len(summary['selected']) == 6
    for family, entry in summary['selected'].items():
        path = COMBO / (entry['candidate'] + '_scores.npz')
        assert q.sha(path) == entry['scores_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(q.answer_scores(meta, z['window_scores']), z['answer_scores'])
            scores = z['window_scores'][lo:hi].copy(), z['answer_scores'][634:].copy()
        sources[family] = (dict(entry, calibration=entry['metrics']['calibration']), scores,
                           {'scores_sha256': q.sha(path), 'candidate': entry['candidate']})
    results, predictions = {}, {}
    for name, (entry, (ws, ans), origin) in sources.items():
        thresholds = entry['thresholds']
        wm = q.count(y, ws, thresholds['window']['threshold'])
        am = q.count(ay, ans, thresholds['answer']['threshold'])
        assert wm == entry['calibration']['windows'] and am == entry['calibration']['answers']
        pred = ws >= thresholds['window']['threshold']
        predictions[name] = pred
        clean_fp = int((pred & ~y.astype(bool) & clean).sum())
        other_fp = int((pred & ~y.astype(bool) & ~clean).sum())
        assert clean_fp + other_fp == wm['fp']
        results[name] = {'windows': wm, 'answers': am, 'thresholds': thresholds,
            'types_recall_only': {k: {'positive': int(m.sum()), 'detected': int(pred[m].sum()),
                                    'recall': float(pred[m].mean())} for k, m in masks.items()},
            'false_positive_windows_clean_answers': clean_fp,
            'false_positive_windows_risky_answers': other_fp, 'source': origin}
    curves = {}
    for name, directory in [('generic_base', BASE), ('generic_large', LARGE)]:
        c = q.read(directory / 'complete.json')
        curves[name] = [{
            'epoch': e['epoch'], 'fit': e['fit_at_cal_thresholds'], 'calibration': e['calibration']}
            for e in c['all_epochs']]
    overlap = agreement(y, predictions['generic_large'], predictions['generic_base'])
    report = {'models': results, 'training_curves': curves, 'large_vs_base_overlap': overlap,
        'all_six_selected_families_reported': True, 'type_masks_can_overlap': True,
        'no_new_model_or_threshold_or_inference': True, 'official_test_opened': False,
        'bindings': {str(d / 'complete.json'): q.sha(d / 'complete.json') for d in (LARGE, BASE, COMBO)}}
    q.save(OUT / 'summary.json', report)
    lines = ['# 已完成large及公平组合错误诊断', '',
        '仅使用已选模型及其原阈值。反复开发校准159答；不训练、不调阈值、不读取官方测试。', '',
        '| 方法 | 定位F1 | 整答F1 | 正常答内误报窗 | 风险答内误报窗 |',
        '|---|---:|---:|---:|---:|']
    for name, e in results.items():
        lines.append(f"| {name} | {e['windows']['f1']:.6f} | {e['answers']['f1']:.6f} | {e['false_positive_windows_clean_answers']} | {e['false_positive_windows_risky_answers']} |")
    lines += ['', '| 类型 | 原风险窗 | base检出 | large检出 |', '|---|---:|---:|---:|']
    for kind in KINDS:
        a, b = [results[n]['types_recall_only'][kind] for n in ('generic_base', 'generic_large')]
        lines.append(f"| {kind} | {a['positive']} | {a['detected']} | {b['detected']} |")
    lines += ['', '类型可能重叠，只报告召回，不将其他风险类型充作负类。训练曲线和全部六种组合的类型计数保存在summary.json。',
        'base和large同时改变尺寸及AdamW foreach执行方式；组合还包含其他语义/生成信号，不能视为单一因素因果结论。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'),
                                 'official_test_opened': False, 'new_fits': 0, 'GPU_used': False})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
