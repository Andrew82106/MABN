"""Describe completed models on existing QA calibration; never tune a model."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import run_development as q

OUT = q.ROOT / 'results/minicheck_tail_all_docs_v3/completed_diagnostics'
KINDS = ('Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Conflict')


def run():
    assert not (OUT / 'complete.json').exists()
    meta = q.metadata()
    lo, hi = meta['bounds']['calibration']
    windows = meta['windows'][lo:hi]
    answers = [a for a in meta['answers'] if a['partition'] == 'calibration']
    y = np.asarray([w['label'] for w in windows])
    ay = np.asarray([a['label'] for a in answers])
    typed_tokens = {kind: defaultdict(set) for kind in KINDS}
    spans = {kind: [] for kind in KINDS}
    for t in meta['tokens']:
        if t['partition'] != 'calibration':
            continue
        for mapping in t['span_token_mapping']:
            kind = t['original_labels'][mapping['span_index']]['label_type']
            ix = set(mapping['risk_token_indices'])
            typed_tokens[kind][t['response_id']].update(ix)
            spans[kind].append((t['response_id'], ix))
    masks = {kind: np.asarray([bool(set(w['token_indices']) & typed_tokens[kind][w['response_id']]) for w in windows]) for kind in KINDS}
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    clean = {a['response_id'] for a in answers if a['label'] == 0}
    clean_windows = np.asarray([w['response_id'] in clean for w in windows])
    assert not y[clean_windows].any()
    sources = {}
    for mode in ('frozen0', 'tail2'):
        path = q.ROOT / 'results/minicheck_tail_all_docs_v3' / mode
        complete = q.read(path / 'complete.json')
        assert not complete['test_opened']
        e = complete['selected']
        scores = path / f"epoch_{e['epoch']:02d}_scores.npz"
        assert q.sha(scores) == e['artifacts_sha256']['_scores.npz']
        with np.load(scores) as z:
            assert np.array_equal(z['cal_window_labels'], y)
            assert np.array_equal(z['cal_answer_labels'], ay)
            sources[mode] = (e, z['cal_window_scores'].copy(), z['cal_answer_scores'].copy())
    for name, folder, method in (
        ('lookback_official_near', 'lookback_regularization_v2', 'lb_prefix_pre_header'),
        ('semantic_sequence_claim', 'claim_pooling_v1', 'minicheck_hidden64_risk_tcn_w32'),
    ):
        path = q.ROOT / 'results' / folder
        e = q.read(path / 'summary.json')['selected'][method]
        scores = path / (e['candidate'] + '_scores.npz')
        assert q.sha(scores) == e['scores_sha256']
        with np.load(scores) as z:
            key = 'window_scores' if 'window_scores' in z else 'scores'
            full = z[key]
            assert len(full) == len(meta['windows'])
            sources[name] = (dict(e, calibration=e['metrics']['calibration']), full[lo:hi], q.answer_scores(meta, full)[634:])
    result = {}
    for name, (entry, score, ascore) in sources.items():
        threshold = entry['thresholds']['window']['threshold']
        pred = score >= threshold
        wm = q.count(y, score, threshold)
        am = q.count(ay, ascore, entry['thresholds']['answer']['threshold'])
        assert wm == entry['calibration']['windows'] and am == entry['calibration']['answers']
        covered = defaultdict(set)
        for w, flagged in zip(windows, pred):
            if flagged:
                covered[w['response_id']].update(w['token_indices'])
        types = {}
        for kind, mask in masks.items():
            localizable = [(rid, ix) for rid, ix in spans[kind] if ix]
            types[kind] = {'positive_windows': int(mask.sum()), 'detected_windows': int(pred[mask].sum()),
                'recall': float(pred[mask].mean()), 'original_spans': len(spans[kind]),
                'localizable_spans': len(localizable),
                'spans_any_covered': sum(bool(ix & covered[rid]) for rid, ix in localizable),
                'spans_fully_covered': sum(ix <= covered[rid] for rid, ix in localizable)}
        fp_clean = int((pred & (y == 0) & clean_windows).sum())
        fp_inside = int((pred & (y == 0) & ~clean_windows).sum())
        assert fp_clean + fp_inside == wm['fp']
        result[name] = {'window_metrics': wm, 'answer_metrics': am, 'by_type': types,
            'false_positive_windows_in_clean_answers': fp_clean,
            'false_positive_windows_in_positive_answers': fp_inside,
            'original_selected_epoch': entry.get('epoch'), 'no_threshold_refit': True}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'models': result, 'calibration_answers': 159, 'calibration_windows': 42241,
        'positive_windows_per_type_can_overlap': True, 'span_hit_means_union_of_alerted_4raw_windows': True,
        'model_selection_performed_here': False, 'official_test_opened': False, 'GPU_used': False,
        'fit_sizes': {'frozen0': 3680, 'tail2': 3680, 'lookback_official_near': 634, 'semantic_sequence_claim': 634},
        'scope': 'Repeated QA development, fixed existing selected models/thresholds. Different fit sizes are disclosed, not a pure matched-architecture comparison.'}
    q.save(OUT / 'DIAGNOSTICS.json', report)
    text = ['# 已完成模型的错误分布', '', '只读原校准159答的已有选中模型和阈值，没有重新训练或选参。', '',
        '| 方法 | 窗口F1 | 整答F1 | 显性冲突检出/997窗 | 完全正常回答中的误报窗 | 风险回答内部的误报窗 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, d in result.items():
        text.append(f"| {name} | {d['window_metrics']['f1']:.4f} | {d['answer_metrics']['f1']:.4f} | {d['by_type']['Evident Conflict']['detected_windows']} | {d['false_positive_windows_in_clean_answers']} | {d['false_positive_windows_in_positive_answers']} |")
    text += ['', '前两项用3680答训练，后两项用634答；评测名单和标签相同，但不能把差异全部解释成模型结构效果。',
        '分类型只报告召回，不把其他风险类型当成负类来伪造各类F1。类型窗口可重叠；跨度命中是被高亮窗口覆盖，不代表精确识别整段错误。']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'code_sha256': q.sha(Path(__file__)),
        'diagnostics_sha256': q.sha(OUT / 'DIAGNOSTICS.json'), 'official_test_opened': False})
    print((OUT / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    run()
