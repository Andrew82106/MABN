"""Fixed-score type diagnosis; run only after all auxiliary-transfer QA epochs."""
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q
from diagnose_completed_tail import KINDS
from diagnose_completed_full_context import epoch_scores, FULL, TAIL


def main():
    directory = OUT / 'transfer'
    complete = q.read(directory / 'complete.json')
    assert not complete['official_test_opened']
    assert [e['epoch'] for e in complete['all_QA_epochs']] == [1, 2, 3]
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
    clean_ids = {a['response_id'] for a in answers if a['label'] == 0}
    clean = np.asarray([w['response_id'] in clean_ids for w in windows])
    source = {'QA_only_selected': epoch_scores(FULL, y, ay, require_six=True),
              'tail2_selected': epoch_scores(TAIL, y, ay)}
    for entry in complete['all_QA_epochs']:
        epoch = entry['epoch']
        path = directory / f'qa_epoch_{epoch:02d}_scores.npz'
        assert q.sha(path) == entry['artifacts_sha256']['_scores.npz']
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['cal_window_labels'], y)
            assert np.array_equal(z['cal_answer_labels'], ay)
            scores = z['cal_window_scores'].copy(), z['cal_answer_scores'].copy()
        source[f'aux_then_QA_{epoch}'] = (entry, scores, {'path': str(path), 'sha256': q.sha(path), 'epoch': epoch})
    result = {}
    for name, (entry, (scores, ascores), provenance) in source.items():
        wt, at = entry['thresholds']['window']['threshold'], entry['thresholds']['answer']['threshold']
        wm, am = q.count(y, scores, wt), q.count(ay, ascores, at)
        assert wm == entry['calibration']['windows'] and am == entry['calibration']['answers']
        pred = scores >= wt
        clean_fp = int((pred & (y == 0) & clean).sum())
        internal_fp = int((pred & (y == 0) & ~clean).sum())
        assert clean_fp + internal_fp == wm['fp']
        result[name] = {'window_threshold': wt, 'answer_threshold': at,
            'window_metrics': wm, 'answer_metrics': am,
            'by_type_recall': {k: {'positive_windows': int(mask.sum()), 'detected_windows': int(pred[mask].sum()),
                'missed_windows': int((~pred[mask]).sum()), 'recall': float(pred[mask].mean())} for k, mask in masks.items()},
            'false_positive_windows_in_clean_answers': clean_fp,
            'false_positive_windows_in_risky_answers': internal_fp, 'source': provenance}
    prior = q.read(ROOT / 'results/completed_full_context_diagnostics_v1/DIAGNOSTICS.json')['models']
    for name, old in [('QA_only_selected', 'full_context_v2'), ('tail2_selected', 'tail2')]:
        assert result[name]['by_type_recall'] == prior[old]['by_type_recall_only']
        assert result[name]['window_metrics'] == prior[old]['window_metrics']
    report = {'models': result, 'selected_auxiliary_QA_epoch': complete['selected']['epoch'],
        'original_mask_sizes': {k: int(m.sum()) for k, m in masks.items()},
        'type_masks_overlap': True, 'type_specific_false_positives_not_defined_for_binary_head': True,
        'scope': 'Four type recalls plus overall binary false positives at each already saved threshold. No new threshold, model selection, training, inference or official test.',
        'causal_limit': 'Each epoch has its own original calibration threshold. Recall changes include threshold/false-positive tradeoffs and do not alone prove forgetting. No auxiliary-final checkpoint was evaluated.',
        'GPU_used': False, 'official_test_opened': False, 'auxiliary_final_evaluated': False,
        'complete_sha256': q.sha(directory / 'complete.json'), 'code_sha256': q.sha(Path(__file__))}
    q.save(OUT / 'TYPE_DIAGNOSTICS.json', report)
    text = ['# 原人工风险类型诊断', '', '仅复用每轮保存分数与各自原阈值；类型可重叠。二元头不输出风险类型，因此类型只算召回，误报按统一二元风险标签计算。', '',
            '| 方法 | 窗口阈值 | 窗口 F1 | 总 FP | 正常回答内 FP | 风险回答内 FP |',
            '|---|---:|---:|---:|---:|---:|']
    for name, r in result.items():
        text.append(f"| {name} | {r['window_threshold']:.6f} | {r['window_metrics']['f1']:.6f} | {r['window_metrics']['fp']} | {r['false_positive_windows_in_clean_answers']} | {r['false_positive_windows_in_risky_answers']} |")
    text += ['', '| 风险类型 | 原窗口数 | QA-only选中 | tail2选中 | aux→QA1 | aux→QA2 | aux→QA3 |', '|---|---:|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        values = [r['by_type_recall'][kind] for r in result.values()]
        text.append(f"| {kind} | {values[0]['positive_windows']} | " + ' | '.join(f"{v['detected_windows']}/{v['positive_windows']} ({v['recall']:.1%})" for v in values) + ' |')
    text += ['', 'QA1/2/3各自阈值不同，召回变化同时受阈值与误报取舍影响，不能单凭下降断言能力遗忘。没有评估辅助训练结束时的checkpoint，所以也不能直接测量从辅助阶段到QA阶段的能力损失。',
             '全部来自已反复开发的159答校准集和原4BPE窗；没有改变金标、范围、阈值或模型选择，官方测试集封存。']
    (OUT / 'TYPE_DIAGNOSTICS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('AUX_TRANSFER_TYPE_DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
