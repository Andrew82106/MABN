"""Posthoc description of frozen R17 scores. Never fit or update a threshold/model."""
from pathlib import Path
from collections import defaultdict
import json
import pickle
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESULTS))
from audit17 import read, lines, sha, counts, threshold, verify_tree

METHODS = ('legacy_lb', 'expanded_lb', 'legacy_lb_nll', 'expanded_lb_nll')
PAIRS = {'lb': ('legacy_lb', 'expanded_lb'), 'lb_nll': ('legacy_lb_nll', 'expanded_lb_nll')}


def distribution(values):
    values = np.asarray(values, float)
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'quantile_probabilities': [0, .1, .25, .5, .75, .9, 1],
            'quantiles': np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(values) else None}


def measured(rows, method, unit, override=None):
    rows = [r for r in rows if r['main_eligible']]
    scores = np.asarray([r['scores'][method] for r in rows], float)
    pred = [r['predictions'][method] for r in rows] if override is None else scores >= override
    assert np.isfinite(scores).all() and all(p is not None for p in pred)
    return counts([r['gold'] for r in rows], pred, scores, unit)


def changes(rows, old, new, unit):
    before, after = measured(rows, old, unit), measured(rows, new, unit)
    return {'legacy': before, 'expanded': after,
            'expanded_minus_legacy': {k: after[k]-before[k] if after[k] is not None and before[k] is not None else None
                 for k in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'auroc', 'average_precision')}}


def score_summary(rows, method, unit, frozen_threshold, allow_test_oracle):
    rr = [r for r in rows if r['main_eligible']]
    y = np.asarray([r['gold'] for r in rr]); values = np.asarray([r['scores'][method] for r in rr])
    out = {'frozen_threshold': frozen_threshold, 'frozen_metrics': measured(rr, method, unit),
           'risk_scores': distribution(values[y == 1]), 'nonrisk_scores': distribution(values[y == 0])}
    if allow_test_oracle:
        best = threshold(y, values)
        oracle = measured(rr, method, unit, best['threshold'])
        out['posthoc_test_oracle_not_deployable'] = {
            'WARNING': 'Uses already exposed test labels to select a threshold. Diagnostic ceiling on this sample only; not validation, not a deployed result.',
            'threshold': best['threshold'], 'metrics': oracle,
            'optimistic_f1_headroom': oracle['f1']-out['frozen_metrics']['f1']}
    return out


def within_answer(rows):
    by_answer = defaultdict(list)
    for row in rows:
        if row['main_eligible']:
            by_answer[row['row_id']].append(row)
    out = {}
    for method in METHODS:
        per_answer = []
        for rid, rr in sorted(by_answer.items()):
            y, s = [r['gold'] for r in rr], [r['scores'][method] for r in rr]
            if set(y) != {0, 1}:
                continue
            maximum = max(s)
            peaks = [r for r in rr if r['scores'][method] == maximum]
            per_answer.append({'row_id': rid, 'group_id': rr[0]['group_id'], 'category': rr[0]['category'],
                'windows': len(rr), 'risk_windows': sum(y), 'auroc': float(roc_auc_score(y, s)),
                'average_precision': float(average_precision_score(y, s)),
                'maximum_score': maximum, 'maximum_ties': len(peaks),
                'every_tied_peak_is_risk': all(r['gold'] == 1 for r in peaks),
                'any_tied_peak_is_risk': any(r['gold'] == 1 for r in peaks),
                'peak_window_keys': [r['window_key'] for r in peaks]})
        out[method] = {'mixed_answers': len(per_answer),
                'mean_auroc': float(np.mean([r['auroc'] for r in per_answer])),
                'mean_average_precision': float(np.mean([r['average_precision'] for r in per_answer])),
                'answers_where_every_tied_peak_is_risk': sum(r['every_tied_peak_is_risk'] for r in per_answer),
                'answers_where_any_tied_peak_is_risk': sum(r['any_tied_peak_is_risk'] for r in per_answer),
                'interpretation': 'Conditional on already-known risky answers containing both window labels. Not an all-answer detector accuracy, not exact-token localization, and not an independently tested top-window strategy.',
                'per_answer': per_answer}
    return out


def per_answer_window_changes(windows, answers, old, new):
    by_answer = defaultdict(list)
    for w in windows:
        if w['main_eligible']:
            by_answer[w['row_id']].append(w)
    output = []
    for answer in answers:
        rr = by_answer[answer['row_id']]
        if not rr:
            continue
        lost = [r['window_key'] for r in rr if r['gold'] == 1 and r['predictions'][old] and not r['predictions'][new]]
        gained = [r['window_key'] for r in rr if r['gold'] == 1 and not r['predictions'][old] and r['predictions'][new]]
        removed_fp = [r['window_key'] for r in rr if r['gold'] == 0 and r['predictions'][old] and not r['predictions'][new]]
        added_fp = [r['window_key'] for r in rr if r['gold'] == 0 and not r['predictions'][old] and r['predictions'][new]]
        output.append({k: answer[k] for k in ('row_id', 'item_id', 'group_id', 'category', 'condition', 'text', 'gold')})
        output[-1].update(risk_windows=sum(r['gold'] for r in rr), lost_tp_window_keys=lost,
              gained_tp_window_keys=gained, removed_fp_window_keys=removed_fp, added_fp_window_keys=added_fp,
              metrics=changes(rr, old, new, 'windows'))
    return {'evaluated_answers': len(output), 'answers_losing_any_tp': sum(bool(r['lost_tp_window_keys']) for r in output),
            'answers_gaining_any_tp': sum(bool(r['gained_tp_window_keys']) for r in output),
            'lost_tp_windows': sum(len(r['lost_tp_window_keys']) for r in output),
            'gained_tp_windows': sum(len(r['gained_tp_window_keys']) for r in output), 'per_answer': output}


def run():
    assert read(RESULTS/'INDEPENDENT_AUDIT17.json')['status'] == 'passed'
    complete, freeze = read(RESULTS/'test_complete17.json'), read(RESULTS/'fit_freeze17.json')
    assert complete['fit_freeze_sha256'] == sha(RESULTS/'fit_freeze17.json')
    verify_tree(RESULTS, complete['files_sha256']); verify_tree(RESULTS, freeze['files_sha256'])
    protected = {str(p): sha(p) for p in RESULTS.iterdir() if p.is_file() and p.name not in ('REPORT.md',)}
    metric = read(RESULTS/'metrics_test.json'); val = read(RESULTS/'validation_metrics.json'); fit = read(RESULTS/'fit_metrics.json')
    models = pickle.loads((RESULTS/'frozen_models.pkl').read_bytes())
    scores = {split: {unit: lines(RESULTS/(prefix+'_scores_'+split+'.jsonl'))
                      for unit, prefix in [('windows', 'window'), ('answers', 'answer')]}
              for split in ('validation', 'test')}
    result = {'status': 'posthoc_statistical_diagnosis_complete',
        'scope': 'Descriptive analysis of already frozen scores; no fitting, feature extraction, deployment threshold updates, or new evaluation run.',
        'oracle_policy': 'Any test-oracle value is optimistically test-selected and explicitly non-deployable. It cannot replace the frozen result or establish a new achievement.',
        'predeclared_primary': 'expanded_lb versus legacy_lb',
        'comparison_note': 'Fusion is secondary and cannot replace the predeclared primary after viewing test.',
        'models': list(METHODS), 'source_hashes': protected,
        'train_validation_test': {n: {'own_training': fit['own_training'][n],
                                   'common_legacy_training': fit['common_legacy_training'][n],
                                   'validation': val['methods'][n], 'test': metric['methods'][n]} for n in METHODS},
        'coverage': {'training': fit['coverage'], 'validation': val['coverage'], 'test': metric['coverage']},
        'score_distributions_and_operating_points': {}, 'stratified': {},
        'within_answer_window_ranking': {}, 'per_answer_window_changes': {},
        'safe_refusals': {}, 'cross_threshold_diagnostic': {}, 'training_weight_composition': {},
        'previously_frozen_group_confidence_intervals': metric['paired_bootstrap']}
    for split, units in scores.items():
        result['score_distributions_and_operating_points'][split] = {}
        result['within_answer_window_ranking'][split] = within_answer(units['windows'])
        for unit, rows in units.items():
            target = 'window' if unit == 'windows' else 'answer'
            result['score_distributions_and_operating_points'][split][unit] = {
                n: score_summary(rows, n, unit, freeze['thresholds'][n][target]['threshold'], split == 'test') for n in METHODS}
        result['stratified'][split] = {}
        for unit, rows in units.items():
            result['stratified'][split][unit] = {}
            for field in ('category', 'condition'):
                result['stratified'][split][unit][field] = {
                    value: {pair: changes([r for r in rows if r[field] == value], *names, unit)
                            for pair, names in PAIRS.items()} for value in sorted({r[field] for r in rows})}
            result['stratified'][split][unit]['category_by_condition'] = {
                cat+' / '+cond: {pair: changes([r for r in rows if r['category'] == cat and r['condition'] == cond], *names, unit)
                                for pair, names in PAIRS.items()}
                for cat, cond in sorted({(r['category'], r['condition']) for r in rows
                                         if r['main_eligible']})}
        refuse = [r for r in units['answers'] if r['reviewed_safe_refusal']]
        result['safe_refusals'][split] = {'answers': len(refuse), 'never_used_as_window_negative_labels': True,
             'methods': {n: {'answer_threshold': freeze['thresholds'][n]['answer']['threshold'],
                             'false_alarms': sum(r['predictions'][n] for r in refuse),
                             'max_window_score_distribution': distribution([r['scores'][n] for r in refuse])} for n in METHODS},
             'per_answer': [{'row_id': r['row_id'], 'category': r['category'], 'condition': r['condition'],
                            'scores': {n: r['scores'][n] for n in METHODS},
                            'predictions': {n: r['predictions'][n] for n in METHODS}} for r in refuse]}
    for pair, (old, new) in PAIRS.items():
        result['per_answer_window_changes'][pair] = per_answer_window_changes(scores['test']['windows'], scores['test']['answers'], old, new)
        result['cross_threshold_diagnostic'][pair] = {
            'note': 'Posthoc score/threshold decomposition only; probabilities from separately fit heads need not share calibration.',
            'window_score_head_x_threshold_source': {
                model+' @ '+source: measured(scores['test']['windows'], model, 'windows',
                                             freeze['thresholds'][source]['window']['threshold'])
                for model in (old, new) for source in (old, new)}}
    for n in METHODS:
        m = models[n]; keys = m['fit_window_keys']; w = np.asarray(m['loss_weights'])
        is_new = np.asarray([k.startswith('r16_') for k in keys])
        result['training_weight_composition'][n] = {'total': float(w.sum()),
            'legacy_loss_mass': float(w[~is_new].sum()), 'new_loss_mass': float(w[is_new].sum()),
            'new_fraction': float(w[is_new].sum()/w.sum()), 'groups': len(m['fit_groups']),
            'class_factors_before_group_renormalization': np.asarray(m['class_factors']).tolist()}
    parent_peak_path = Path(__file__).parent/'within_answer_peak_audit.json'
    if parent_peak_path.exists():
        parent_peaks = read(parent_peak_path)
        for n in ('legacy_lb', 'expanded_lb'):
            own = result['within_answer_window_ranking']['test'][n]
            assert own['mixed_answers'] == parent_peaks[n]['answers_with_both_window_labels']
            assert own['answers_where_every_tied_peak_is_risk'] == parent_peaks[n]['peak_is_risk']
            assert np.isclose(own['mean_auroc'], parent_peaks[n]['mean_AUROC'])
            assert np.isclose(own['mean_average_precision'], parent_peaks[n]['mean_AP'])
        result['parent_peak_analysis_independently_verified_sha256'] = sha(parent_peak_path)
    result['interpretation_zh'] = [
        '主窗口F1从0.676降至0.629：少检出31个风险窗口，同时少31个误报窗口。不是全面退化，也不是取得主目标改善。',
        '窗口数值阈值0.708→0.703，几乎没变且略降低。交叉套用两阈值仍保留下降；不能说只是把阈值调高了。',
        '风险窗口分数在测试上的低端下降，尤其行为类净少23个TP、数量类净少9个TP；关系类净多5个TP。多个重叠窗口可来自同一答案，不能当作大量独立失败。',
        '主模型测试窗口AUROC0.932→0.946、AP0.666→0.677；26个混合答案内平均AUROC0.894→0.946。说明存在排序信息，而不同答案/资料构成下统一报警界限仍不稳定。',
        '在这26份已知含风险且有正负窗口的回答内，扩充LB最高分窗口26/26落在风险范围，旧LB为23/26（并列最高分也核查）。这只说明条件性的相对排序，不能称100%发现有问题的回答或100%逐词定位；无风险回答同样必有最高分窗口。',
        '仅为诊断而使用测试金标寻找最佳阈值，扩充LB窗口上限约0.669，仍低于旧模型相同诊断约0.691；因此不能把全部差距归于当前阈值，且这些上限不是可交付成绩。',
        '扩充训练总权重固定，新增来源占约69.85%损失权重，改变了学习分布。此事实不能单独证明分布变化、标签差异或特征不足哪一个造成失效。',
        '回答级安全拒答误报20/20→9/20，但风险回答TP23→18；回答F1仅小幅变化。安全拒答未作为定位训练负例，仅被验证回答阈值间接处理。',
        '验证F1有小幅上升，测试没有稳定提升，训练内依然明显乐观。配对组置信区间跨0，不能证明扩充必然伤害或已经解决过拟合。',
        '题型子集很小、事后分层未校正多重比较；地点只有3题。没有通过此分析改阈值、标签、模型或重新测试。']
    for path, before in protected.items():
        assert sha(path) == before, path
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({'status': result['status'], 'output': str(path),
          'primary_window_changes': {k: result['per_answer_window_changes']['lb'][k]
                for k in ('answers_losing_any_tp', 'answers_gaining_any_tp', 'lost_tp_windows', 'gained_tp_windows')}}, ensure_ascii=False))


if __name__ == '__main__':
    run()
