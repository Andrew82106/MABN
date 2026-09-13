"""Current-negative component agreement counts; no new detector or threshold."""
from pathlib import Path
import sys
from itertools import combinations
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q


def run():
    assert not (OUT / 'SUPPLEMENT_COMPLETE.json').exists()
    done = q.read(OUT / 'complete.json')
    for name, sha in done['files_sha256'].items():
        assert q.sha(OUT / name) == sha
    data = q.read(OUT / 'DIAGNOSIS.json')
    with np.load(OUT / 'aligned_calibration_scores.npz', allow_pickle=False) as z:
        y = z['labels'].astype(bool)
        preds = {str(n): p for n, p in zip(z['threshold_names'], z['predictions'])}
    negative = ~preds['current']
    assert negative.sum() == 37102 and (negative & y).sum() == 2145
    assert (negative & ~y).sum() == 34957
    names = ('semantic_claim', 'tail2', 'large')
    records = []
    for members in [*(tuple([n]) for n in names), *combinations(names, 2)]:
        selected = negative & np.logical_and.reduce([preds[n] for n in members])
        tp = int((selected & y).sum()); fp = int((selected & ~y).sum())
        if len(members) == 1:
            r = data['methods'][members[0]]
            assert tp == r['current_FN_recovered_windows'] and fp == r['new_FP_over_current']
        records.append({'members': list(members), 'rule': 'intersection of already-saved individual threshold alerts, restricted to current-negative windows',
            'selected_windows': int(selected.sum()), 'recoverable_TP': tp, 'new_FP': fp,
            'positive_fraction_of_selected_windows': tp / (tp + fp) if tp + fp else None,
            'fraction_of_current_FN': tp / 2145,
            'per_member_threshold_unchanged': {n: data['methods'][n]['thresholds_unchanged']['window']['threshold'] for n in members}})
    result = {'status': 'complete', 'condition': 'Current prediction negative, using its existing fixed window threshold',
        'denominators': {'windows': 37102, 'current_FN': 2145, 'current_TN': 34957,
            'positive_rate': 2145 / 37102}, 'records': records,
        'limits': 'Descriptive conditional counts only; no intersection detector is deployed or scored as a new candidate. No gate is trained, and no agreement combination is selected. Labels only count true and false alerts.',
        'sources_sha256': {n: q.sha(OUT / n) for n in ('DIAGNOSIS.json', 'aligned_calibration_scores.npz', 'complete.json', 'supplement.py')},
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'test_opened': False}
    q.save(OUT / 'COMPONENT_INTERSECTIONS.json', result)
    q.save(OUT / 'ROOT_REPORTED_DILATION_NOTE.json', {
        'status': 'reported_by_parent_not_independently_replayed',
        'source': 'Root message during this diagnosis; the command and predictions were not saved as a formal model artifact.',
        'reported_rule': 'Per-answer contiguous-chain max dilation, left/right0..8; fit selects the unchanged0/0 rule.',
        'reported_fit_selected': {'left': 0, 'right': 0, 'fit_F1': .808219},
        'reported_calibration_original_F1': .690281,
        'reported_best_nonidentity_one_step_calibration_F1': .688220,
        'use': 'Corroborating reported negative result for this tested dilation grid only. Not a saved model, not an independent audit, not evidence against every possible structured rule.',
        'rerun_here': False, 'included_in_main_score_inventory': False})
    q.save(OUT / 'EXECUTION_NOTE.json', {
        'first_diagnose_exit': 1, 'reason': 'The audit expected tail2 status complete, but the genuine terminal marker is complete_development_only.',
        'correction': 'Changed only this new diagnosis status assertion to the exact existing terminal value. No source score/model/threshold or label was modified.',
        'final_diagnose_exit': 0, 'diagnosis_seconds': data['seconds']})
    core = data['union_diagnostics']['current_component_scores']
    all_r = data['union_diagnostics']['all_roster_threshold_scores']
    lines = ['# 补充结论与当前漏报区的共同报警', '',
        '**现有信号并非已证明不足以到 .75；但已完成的组合没有实现这一目标。** 仅current已有的semantic_claim、tail2、large三个组成分数，就能在原阈值下补1026个FN、命中40/70个全漏run。凭gold排除新增误报且保留当前1300误报时，oracle F1为0.800889；真实直接OR却为0.640132，因为同时新增3051个误报。', '',
        '这把问题具体化为“能否辨别哪些补报警正确”，而不是单纯“是否出现过风险分数”。若当前1300误报不变，至少需净补532个FN才能到.75。若保留三个组成分数可补的全部1026个TP，则最多容许824个新增FP；实际有3051个，至少要排除其中2227个，同时保住这些TP。这只是条件算术，未证明可学习。', '',
        '| 当前未报警区内的固定原阈值条件 | 可补TP | 新增FP | 这些报警中真阳性比例 | 覆盖2145 FN |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {' ∩ '.join(r['members'])} | {r['recoverable_TP']} | {r['new_FP']} | {r['positive_fraction_of_selected_windows']:.2%} | {r['fraction_of_current_FN']:.2%} |")
    lines += ['', '条件区总37102窗，其中2145 FN、34957 TN，阳性率5.78%。两两交集只按已存阈值统计；未选交集作为新规则、未训练gate、未重选阈值。比条件区平均更高的阳性比例不等于足以达到目标的部署精度。', '',
        '全部信号联合可以碰到70/70个全漏run，但弱GHOST/NLL报警也带来大量误报：全体直接OR F1仅0.320367。gold只保留正确新增报警的0.880418不能当成绩；连current单独凭gold删掉所有误报也可到0.781635，说明宽松oracle本身不能决定下一模型可行性。',
        '108个原固定large/NLI/FAVA组合的窗口F1均未达到.75，最高仍为当前0.690281。这些旧网格没有在本次增加权重，但历史cal选型偏差仍在。现有证据支持“存在互补、现有汇总没可靠利用”，不足以证明必须新增证据对齐信号，也不足以保证再训练融合能成功。', '',
        '根代理另报告未保存的left/right0..8形态延展检查：fit选择仍0/0（fit F1 0.808219），cal原0.690281；最佳非恒等一格cal0.688220。这里仅记录其来源和边界，未独立重放，不当正式模型或把它并入分数清单。它说明该批已试的简单扩展没有提升，不能概括所有结构化方法。',
        '主统计、完整分母/哈希/17分数、108候选与170run逐项命中见DIAGNOSIS.json；本补充不修改主complete或任何既有模型结果。']
    assert core['new_true_positive_windows'] == 1026 and core['new_false_positive_windows'] == 3051
    assert all_r['current_missed_runs_recovered'] == 70
    (OUT / 'INTERPRETATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'SUPPLEMENT_COMPLETE.json', {'status': 'complete', 'files_sha256': {n: q.sha(OUT / n) for n in
        ('supplement.py', 'COMPONENT_INTERSECTIONS.json', 'ROOT_REPORTED_DILATION_NOTE.json', 'EXECUTION_NOTE.json', 'INTERPRETATION.md')},
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'test_opened': False})
    print('COMPONENT_INTERSECTIONS_COMPLETE', records, flush=True)


if __name__ == '__main__':
    run()
