"""Validation-only branch disagreements at each branch's frozen risk threshold.

This computes diagnostic counts, not an oracle or a new evaluated method.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
sys.path.insert(0, str(RUN))
import metrics

CELLS = ['both_correct', 'semantic_only_correct', 'whitebox_only_correct', 'both_wrong']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scores(path):
    rows = metrics.readl(path)
    assert len(rows) == len({r['response_id'] for r in rows})
    return {r['response_id']: np.asarray(r['token_scores'], np.float64) for r in rows}


def longest_run(indices, starts):
    chunks = []
    for index in indices:
        if not chunks or starts[index] != starts[chunks[-1][-1]] + 1:
            chunks.append([])
        chunks[-1].append(index)
    return max(chunks, key=lambda x: (len(x), -starts[x[0]])) if chunks else []


def main():
    semantic = RUN / 'model_design/modernbert_large_v1'
    whitebox = RUN / 'whitebox_model'
    files = {'semantic_completion': semantic / 'complete.json',
             'whitebox_completion': whitebox / 'COMPLETE.json',
             'semantic_predictions': semantic / 'best_validation_predictions.jsonl',
             'whitebox_predictions': whitebox / 'best_validation_token_predictions.jsonl',
             'answers': RUN / 'data/validation/answers.jsonl',
             'sequences': RUN / 'data/validation/sequences.jsonl', 'metrics': RUN / 'metrics.py'}
    sc, wc = read(files['semantic_completion']), read(files['whitebox_completion'])
    assert sc['best']['epoch'] == 5 and wc['best']['epoch'] == 15
    st, wt = sc['best']['thresholds'], wc['best']['evaluation']['thresholds']
    ss, ws = load_scores(files['semantic_predictions']), load_scores(files['whitebox_predictions'])
    answers = metrics.readl(files['answers'])
    sequences = {r['response_id']: r for r in metrics.readl(files['sequences'])}
    assert ss.keys() == ws.keys() == sequences.keys() == {r['response_id'] for r in answers}
    assert metrics.evaluate_partition('validation', ss)['thresholds'] == st
    assert metrics.evaluate_partition('validation', ws)['thresholds'] == wt
    selected = [r for r in answers if not r['unanswerability_hint_setting']]
    assert len(selected) == 135
    notes_path = OUT / 'CASE_NOTES.json'
    notes = read(notes_path) if notes_path.exists() else {'cases': []}
    assert {r['response_id'] for r in notes['cases']} <= {r['response_id'] for r in selected}
    counts = {truth: dict.fromkeys(CELLS, 0) for truth in ['positive', 'negative']}
    affected = {truth: {cell: set() for cell in CELLS} for truth in counts}
    candidates = []
    for answer in selected:
        rid = answer['response_id']
        seq = sequences[rid]
        lexical, risk = np.asarray(seq['lexical_mask'], bool), np.asarray(seq['risk_mask'], bool)
        assert ss[rid].shape == ws[rid].shape == lexical.shape == risk.shape
        starts = [i for i in range(max(1, len(risk) - 3)) if lexical[i:min(i + 4, len(risk))].any()]
        labels = np.array([(risk[i:i + 4] & lexical[i:i + 4]).any() for i in starts])
        sp, wp = metrics.pool_windows(ss[rid], lexical), metrics.pool_windows(ws[rid], lexical)
        assert len(labels) == len(sp) == len(wp)
        scorrect, wcorrect = (sp >= st['window']) == labels, (wp >= wt['window']) == labels
        decisions = {'both_correct': scorrect & wcorrect, 'semantic_only_correct': scorrect & ~wcorrect,
                     'whitebox_only_correct': ~scorrect & wcorrect, 'both_wrong': ~scorrect & ~wcorrect}
        for truth, positive in [('positive', True), ('negative', False)]:
            for cell, mask in decisions.items():
                indices = np.flatnonzero(mask & (labels == positive))
                counts[truth][cell] += len(indices)
                if len(indices):
                    affected[truth][cell].add(rid)
                if cell not in ['semantic_only_correct', 'whitebox_only_correct'] or not len(indices):
                    continue
                chunk = longest_run(indices, starts)
                lo, hi = starts[chunk[0]], min(starts[chunk[-1]] + 4, len(risk))
                begin, end = seq['response_token_offsets'][lo][0], seq['response_token_offsets'][hi - 1][1]
                text = answer['original_response']
                candidates.append({'truth': truth, 'cell': cell, 'response_id': rid, 'group_id': answer['group_id'],
                    'consecutive_windows': len(chunk), 'raw_token_range': [lo, hi], 'character_range': [begin, end],
                    'question': answer['question'], 'answer_region': text[begin:end],
                    'answer_surrounding': text[max(0, begin - 130):min(len(text), end + 130)],
                    'semantic_score_range': [float(sp[chunk].min()), float(sp[chunk].max())],
                    'whitebox_score_range': [float(wp[chunk].min()), float(wp[chunk].max())]})
    examples = []
    for truth in counts:
        for cell in ['semantic_only_correct', 'whitebox_only_correct']:
            seen = set()
            for row in sorted((x for x in candidates if x['truth'] == truth and x['cell'] == cell),
                              key=lambda x: (-x['consecutive_windows'], x['response_id'])):
                if row['group_id'] in seen:
                    continue
                examples.append(row); seen.add(row['group_id'])
                if len(seen) == 2:
                    break
    pos, neg = counts['positive'], counts['negative']
    semantic_fn = pos['whitebox_only_correct'] + pos['both_wrong']
    semantic_tn = neg['semantic_only_correct'] + neg['both_correct']
    result = {'scope': 'validation no_hint only; automatic labels; diagnostic, not an oracle method',
        'answers': len(selected), 'groups': len({r['group_id'] for r in selected}),
        'geometry': 'original Llama BPE, length4 stride1, omit nonlexical windows, lexical maximum',
        'semantic': {'model': 'Context-large', 'epoch': 5, 'thresholds': st},
        'whitebox': {'model': 'Whitebox', 'epoch': 15, 'thresholds': wt},
        'counts_by_true_window_label': counts,
        'answers_affected_by_cell': {truth: {cell: len(ids) for cell, ids in cells.items()}
                                     for truth, cells in affected.items()},
        'if_whitebox_risk_flags_were_added_to_semantic_flags': {
            'semantic_false_negative_windows': semantic_fn,
            'recovered_positive_windows': pos['whitebox_only_correct'],
            'fraction_of_semantic_misses_recovered': pos['whitebox_only_correct'] / semantic_fn,
            'semantic_true_negative_windows': semantic_tn,
            'new_false_positive_windows': neg['semantic_only_correct'],
            'fraction_of_semantic_true_negatives_flagged': neg['semantic_only_correct'] / semantic_tn,
            'new_false_positives_per_recovered_positive': neg['semantic_only_correct'] / pos['whitebox_only_correct'],
            'remaining_shared_misses': pos['both_wrong'],
            'interpretation': 'A diagnostic marginal count, not a selected fusion or oracle result'},
        'whitebox_can_also_reject_semantic_false_positives': neg['whitebox_only_correct'],
        'examples': examples, 'qualitative_case_notes': notes,
        'test_labels_read': False, 'training_or_candidate_changes': False,
        'source_sha256': {str(path): sha(path) for path in files.values()}}
    if notes_path.exists():
        result['source_sha256'][str(notes_path)] = sha(notes_path)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'DIAGNOSTIC.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 验证集两分支互补性', '',
        '仅使用无暗示验证集 135 条回答、67 个来源组；不增加候选，不涉及测试。', '',
        f"Context-large epoch5 窗口阈值 {st['window']:.9f}；白盒 epoch15 阈值 {wt['window']:.9f}。",
        '共同原始 BPE 长度4、步长1，忽略全标点窗口，取有效词元最大风险。', '',
        '| 发布窗口标签 | 双方都对 | 只有语义对 | 只有白盒对 | 双方都错 | 合计 |',
        '|---|---:|---:|---:|---:|---:|']
    for truth, name in [('positive', '风险正类'), ('negative', '风险负类')]:
        row = counts[truth]
        lines.append('| ' + name + ' | ' + ' | '.join(str(row[c]) for c in CELLS) + f' | {sum(row.values())} |')
    lines += ['', f"若仅把白盒的风险提示加到语义提示上，可补回 {pos['whitebox_only_correct']} / {semantic_fn} 个语义漏报窗口（{pos['whitebox_only_correct'] / semantic_fn:.1%}），同时增加 {neg['semantic_only_correct']} 个误报。",
        f"白盒也正确排除了 {neg['whitebox_only_correct']} 个语义误报窗口；双方仍共同漏掉 {pos['both_wrong']} 个正窗口。",
        '这些是已知验证标签下的差异计数，不代表存在能自动选择正确分支的模型，也不作为新的方法成绩。',
        '窗口高度重叠，不能把这些计数当作独立事件或独立样本。发布标签为自动标注，负窗口不等同于人工确认无误。', '',
        '少量连续差异区间见 DIAGNOSTIC.json：每格按最长连续区间选择两个不同来源组，仅作验证错例检查。']
    if notes['cases']:
        lines += ['', '## 少量验证错例', '']
        for case in notes['cases']:
            lines.append(f"- {case['response_id']}（{case['category']}）：{case['observation']}")
        lines += ['', notes['interpretation']]
    (OUT / 'DIAGNOSTIC.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['counts_by_true_window_label',
          'if_whitebox_risk_flags_were_added_to_semantic_flags', 'whitebox_can_also_reject_semantic_false_positives']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
