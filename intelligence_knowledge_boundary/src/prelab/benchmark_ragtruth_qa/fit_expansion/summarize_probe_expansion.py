"""Read-only development diagnostics for the completed data-size experiment."""
from pathlib import Path
import json
import sys
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
import run_development as q

OUT = HERE / 'probe_v1'


def main():
    summary = q.read(OUT / 'summary.json')
    meta = q.metadata()
    windows = meta['windows'][168123:]
    tokens = {t['response_id']: t for t in meta['tokens'][634:]}
    types = defaultdict(lambda: np.zeros(len(windows), bool))
    for i, window in enumerate(windows):
        t = tokens[window['response_id']]
        for span, mapping in zip(t['original_labels'], t['span_token_mapping']):
            if set(mapping['risk_token_indices']) & set(window['token_indices']):
                types[span['label_type']][i] = True
    y = np.array([w['label'] for w in windows], bool)
    assert np.array_equal(np.logical_or.reduce(list(types.values())), y)
    results = {}
    lines = [
        '数据扩充已完成第一次匹配训练对照，但没有带来定位收益。以下均为原 159 份开发校准回答；官方测试未打开。',
        '',
        '| 特征 | 训练回答 | 4词元窗口F1 | 整答F1 |',
        '|---|---:|---:|---:|',
    ]
    for name, selected in summary['selected'].items():
        with np.load(OUT / (selected['candidate'] + '_scores.npz')) as z:
            scores = z['window_scores'][-42241:]
        pred = scores >= selected['thresholds']['window']['threshold']
        metrics = selected['metrics']['calibration']
        results[name] = {
            'candidate': selected['candidate'],
            'calibration': metrics,
            'recall_by_original_human_type': {
                kind: {'positive_windows': int(mask.sum()), 'detected': int((pred & mask).sum()),
                       'recall': float(pred[mask].mean())}
                for kind, mask in sorted(types.items())
            },
            'FP': int((pred & ~y).sum()), 'FN': int((~pred & y).sum()),
        }
        label = '隐状态64维＋句风险' if name.endswith('_risk') else '隐状态64维'
        n = 634 if name.startswith('original') else 3680
        lines.append(f"| {label} | {n} | {metrics['windows']['f1']:.4f} | {metrics['answers']['f1']:.4f} |")
    lines += [
        '',
        '同一批 615 个训练来源组内增加其他生成器的人工标注回答：634→3680；并没有增加独立问题。原 159 份校准回答、原人工标签、原4个BPE滑动窗口均保持。',
        '',
        '两种规模均使用原634答拟合的PCA64，不重新学习投影。每种特征均比较相同三档 C；损失总质量同为168123，避免把样本数变化意外变成正则化变化。扩充训练中原回答与辅助回答各占一半基础权重，再做相同的类别平衡和来源组等权。9次新拟合、3次旧模型回放均已完成。',
        '',
        '纯隐状态定位下降0.0029，加入句风险后下降0.0021。这说明本轮“固定特征＋更多同源回答”的组合没有改善；不能据此断言数据数量无关，也不能将微小下降称为稳定退化。更换生成器风格和拟合标准化参数也同时发生。',
        '',
        '逐类召回见 data_size_diagnostic.json；类别窗口可以重叠，召回不等于该类F1。这里只作错误诊断，不改变任何标签或选择规则。',
        '',
        '这些输入来自额外的 MiniCheck 核查模型，原生成模型与核查编码器都保持冻结；此对照不属于原生成模型的纯白盒探针。扩充后的原生成状态基线仍须使用相同3680答训练，不能用不同训练规模宣称我们胜出。',
        '',
        '选型和阈值使用了同一校准集，结果存在选择乐观；不能代替封存测试。',
    ]
    q.save(OUT / 'data_size_diagnostic.json', {
        'results': results, 'type_windows_can_overlap': True,
        'diagnostic_only': True, 'official_test_opened': False,
        'summary_sha256': q.sha(OUT / 'summary.json'),
        'code_sha256': q.sha(__file__),
    })
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: {'window_f1': v['calibration']['windows']['f1'],
                         'answer_f1': v['calibration']['answers']['f1'],
                         'type_recall': v['recall_by_original_human_type']}
                      for k, v in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
