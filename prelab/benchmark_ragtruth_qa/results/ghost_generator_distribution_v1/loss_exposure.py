"""Root-requested supplement: existing loss mass, no new weighting or fit."""
from pathlib import Path
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q


def run():
    assert not (OUT / 'LOSS_EXPOSURE.json').exists()
    weight_path = ROOT / 'fit_expansion/llama_baselines_v1/training_weights.npz'
    source_weight = ROOT / 'fit_expansion/probe_v1/expanded3680_weights.npz'
    assert q.sha(weight_path) == q.sha(source_weight)
    with np.load(weight_path, allow_pickle=False) as z:
        base, loss, y = z['base'], z['loss'], z['y']
    assert len(y) == 653979 and abs(loss.sum() - 168123) < 1e-6
    fit_path = ROOT / 'fit_expansion/data/fit.jsonl'
    window_path = ROOT / 'fit_expansion/data/windows_k4_fit.jsonl'
    fit = q.lines(fit_path)
    rid_to_group = {r['response_id']: 'native634_fit' if i < 634 else 'added_model:' + r['model']
                    for i, r in enumerate(fit)}
    groups = []
    # Only fit-window identity and label fields; no response content is used.
    import json
    with window_path.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            w = json.loads(line)
            assert w['label'] == y[i]
            groups.append(rid_to_group[w['response_id']])
    assert len(groups) == len(y)
    groups = np.asarray(groups)
    indices = {'native634_fit': np.arange(168123), 'added3046_fit': np.arange(168123, 653979)}
    assert np.all(groups[:168123] == 'native634_fit') and np.all(groups[168123:] != 'native634_fit')
    indices.update({name: np.flatnonzero(groups == name) for name in sorted(set(groups)) if name.startswith('added_model:')})
    entries = {}
    for name, ix in indices.items():
        entries[name] = {'windows': len(ix), 'raw_window_share': len(ix) / len(y),
            'base_mass': float(base[ix].sum()), 'base_mass_share': float(base[ix].sum() / base.sum()),
            'loss_mass': float(loss[ix].sum()), 'loss_mass_share': float(loss[ix].sum() / loss.sum()),
            'positive_loss_mass': float(loss[ix][y[ix] == 1].sum()),
            'negative_loss_mass': float(loss[ix][y[ix] == 0].sum())}
    assert abs(sum(entries[k]['loss_mass'] for k in ('native634_fit', 'added3046_fit')) - loss.sum()) < 1e-6
    data = {'status': 'complete', 'total_loss_mass': float(loss.sum()), 'cohorts': entries,
        'source_sha256': {str(p.resolve()): q.sha(p) for p in (Path(__file__), weight_path, source_weight, fit_path, window_path)},
        'interpretation': 'Existing sample-weight coefficient mass, not actual gradient magnitude or causal influence. No recalculation or change of training weights.',
        'no_test': True, 'new_fits': 0, 'GPU_used': False}
    q.save(OUT / 'LOSS_EXPOSURE.json', data)
    lines = ['', '## 旧训练损失权重', '', '| 子集 | 原始窗口占比 | base质量占比 | loss质量占比 |', '|---|---:|---:|---:|']
    for name, e in entries.items():
        lines.append(f"| {name} | {e['raw_window_share']:.2%} | {e['base_mass_share']:.2%} | {e['loss_mass_share']:.2%} |")
    lines += ['', '全部 loss 总质量仍为168123。追加数据的窗口较多，不等于其训练权重按窗口数同等增大；以上仅是损失系数质量，不是实测梯度或因果影响。', '',
        '观察上，追加集的风险窗更少、答案更短，但 GHOST 树在追加集的 AUROC 高于原生训练集；因此较低的追加集 F1/AP 不能直接解释为“跨生成器重放学不会”。统一阈值、正例率、回答风格与来源差异均可能参与，当前统计无法分离其因果贡献。']
    report = OUT / 'REPORT.md'
    report.write_text(report.read_text(encoding='utf-8') + '\n'.join(lines) + '\n', encoding='utf-8')
    complete = q.read(OUT / 'complete.json')
    complete['files_sha256'].update({n: q.sha(OUT / n) for n in ('REPORT.md', 'LOSS_EXPOSURE.json', 'loss_exposure.py')})
    complete['loss_exposure_supplement'] = 'Existing-weight read-only breakdown requested after main diagnostic; no new fit or threshold.'
    q.save(OUT / 'complete.json', complete)
    print('LOSS_EXPOSURE_COMPLETE', {k: e['loss_mass_share'] for k, e in entries.items()}, flush=True)


if __name__ == '__main__':
    run()
