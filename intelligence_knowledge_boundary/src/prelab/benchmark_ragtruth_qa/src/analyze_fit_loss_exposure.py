"""Describe existing fit-only loss exposure without modifying weights or data."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fit_loss_exposure_v1'
TYPES = ('Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Conflict')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def quantiles(values):
    levels = (0, .1, .5, .9, .99, 1)
    return {str(q): float(np.quantile(values, q)) for q in levels}


def main():
    assert not OUT.exists(), 'Preserve prior analysis'
    OUT.mkdir()
    token_path = ROOT / 'fit_expansion/data/tokens_fit.jsonl'
    weight_path = ROOT / 'results/full_context_encoder_v2/training_weights.npz'
    sources = {str(p.resolve()): sha(p) for p in (Path(__file__), token_path, weight_path)}
    save(OUT / 'design.json', {'input_sha256': sources,
        'scope': 'Only3680fit raw BPE labels and already frozen full-context loss weights.',
        'comparison': 'Existing base/group-stratum weights, final binary-balanced group weights, and descriptive uniform lexical mass. No replacement weights exported.',
        'types': 'Independent four original span types, with exact union check. Type counts can overlap.',
        'new_training': False, 'GPU_used': False, 'calibration_or_test_read': False})
    with token_path.open(encoding='utf-8') as stream:
        tokens = [json.loads(line) for line in stream if line.strip()]
    with np.load(weight_path, allow_pickle=False) as archive:
        weights = {k: archive[k].copy() for k in archive.files}
    assert len(tokens) == 3680 and all(t['partition'] == 'fit' for t in tokens)
    labels, lexical, kinds, groups, original = [], [], [], [], []
    answer_rows = []
    for i, t in enumerate(tokens):
        n = t['token_count']; lo, hi = weights['bounds'][i]
        assert hi - lo == n
        y = np.asarray(t['risk_mask'], bool)
        lex = np.asarray(t['lexical_mask'], bool)
        types = np.zeros((n, 4), bool)
        assert len(t['original_labels']) == len(t['span_token_mapping'])
        for span, mapped in zip(t['original_labels'], t['span_token_mapping']):
            types[mapped['risk_token_indices'], TYPES.index(span['label_type'])] = True
        assert np.array_equal(types.any(1), y) and not y[~lex].any()
        assert np.array_equal(y, weights['y'][lo:hi])
        labels.append(y); lexical.append(lex); kinds.append(types)
        groups.extend([t['group_id']] * n)
        original.extend([bool(weights['original_membership'][i])] * n)
        answer_rows.append({'response_id': t['response_id'], 'group_id': t['group_id'],
            'original': bool(weights['original_membership'][i]), 'lexical_tokens': int(lex.sum()),
            'risk_tokens': int(y.sum()), 'base_mass': float(weights['base'][lo:hi].sum()),
            'final_loss_mass': float(weights['loss'][lo:hi].sum())})
    y, lex, typ = np.concatenate(labels), np.concatenate(lexical), np.concatenate(kinds)
    original = np.asarray(original, bool)
    assert len(y) == 665708 and lex.sum() == 560300 and y.sum() == 47398
    masks = {'all_lexical': lex, 'risk_union': y, 'clean_lexical': lex & ~y,
             'baseless_union': typ[:, :2].any(1), 'conflict_union': typ[:, 2:].any(1),
             'original_lexical': lex & original, 'expanded_lexical': lex & ~original}
    masks.update({kind: typ[:, i] for i, kind in enumerate(TYPES)})
    comparisons = {}
    for name, w in (('uniform_lexical_reference', lex.astype(float)), ('base_group_stratum', weights['base']),
                    ('actual_binary_balanced_group', weights['loss'])):
        assert w.shape == y.shape and np.isfinite(w).all() and (w >= 0).all()
        assert not w[~lex].any() and np.isclose(w.sum(), 560300)
        mass = float(w.sum()); positive_mass = float(w[y].sum())
        comparisons[name] = {'total_mass': mass, 'risk_mass': positive_mass,
            'risk_fraction': positive_mass / mass,
            'masks': {k: {'tokens': int(mask.sum()), 'mass': float(w[mask].sum()),
                'fraction_of_total_mass': float(w[mask].sum() / mass),
                'fraction_of_risk_mass': float(w[mask & y].sum() / positive_mass),
                'per_token_weight_quantiles': quantiles(w[mask])} for k, mask in masks.items()}}
    group_mass = defaultdict(float)
    for row in answer_rows:
        group_mass[row['group_id']] += row['final_loss_mass']
    assert len(group_mass) == 615
    assert np.allclose(list(group_mass.values()), 560300 / 615, atol=1e-8, rtol=0)
    result = {'comparisons': comparisons, 'binary_class_factors': weights['class_factors'].tolist(),
        'answer_loss_mass_quantiles': quantiles([r['final_loss_mass'] for r in answer_rows]),
        'group_final_mass_equal': True, 'groups': 615,
        'limits': 'Loss mass is a coefficient exposure, not observed loss, gradient influence, or causal evidence that reweighting helps. Uniform lexical reference preserves original BPE/mask and is not the original LettuceDetect encoder-token CE.',
        'new_training': False, 'GPU_used': False, 'calibration_or_test_read': False}
    save(OUT / 'summary.json', result)
    with (OUT / 'per_answer.jsonl').open('w', encoding='utf-8') as stream:
        for row in answer_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# Existing training-loss exposure', '',
             '| Weight system | Risk share of all mass | Conflict share of risk mass | Original634 share of mass |',
             '|---|---:|---:|---:|']
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['risk_fraction']:.6f} | {c['masks']['conflict_union']['fraction_of_risk_mass']:.6f} | {c['masks']['original_lexical']['fraction_of_total_mass']:.6f} |")
    lines += ['', result['limits'], '', 'Only original fit metadata and existing weights were read. No labels, loss weights, models, thresholds, calibration or test data were changed.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert sources == {path: sha(Path(path)) for path in sources}
    save(OUT / 'complete.json', {'status': 'complete_fit_only_diagnosis',
        'design_sha256': sha(OUT / 'design.json'), 'summary_sha256': sha(OUT / 'summary.json'),
        'new_training': False, 'GPU_used': False, 'calibration_or_test_read': False})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
