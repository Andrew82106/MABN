"""Read-only training-signal diagnosis. Never fit a model or read test predictions."""
import sys
import json
import pickle
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import run17 as r
np = r.np


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0}
    return {'n': len(a), 'mean': float(a.mean()), 'std': float(a.std()),
            **{k: float(v) for k, v in zip(('min', 'q25', 'median', 'q75', 'max'), np.quantile(a, [0, .25, .5, .75, 1]))}}


def train_meta(root):
    rows = [x for x in r.readl(root / 'data/inputs.jsonl') if x['split'] == 'train']
    items, records = [], {}
    for row in rows:
        path = root / 'data/generation_records' / (row['row_id'] + '.json')
        g = r.read(path)
        assert len(g['items']) == 1
        item = dict(g['items'][0], **{k: row[k] for k in ('row_id', 'question_id', 'group_id', 'split', 'condition', 'category')})
        items.append(item)
        records[row['row_id']] = g, r.sha(path)
    return rows, items, records


def composition(pack, records, category=None):
    items = [i for i in pack['items'] if category is None or i['category'] == category]
    ids = {i['item_id'] for i in items}
    windows = [w for w in pack['windows'] if w['main_eligible'] and w['item_ids'][0] in ids]
    tokens = [t for t in pack['tokens'] if t['main_eligible'] and any(i in ids for i in t['item_ids'])]
    resolved = [i for i in items if i['asserted_eligible'] and i['localization_status'] == 'resolved']
    risk_items = [i for i in resolved if i['gold'] == 1]
    lengths, scopes, spans, positions, end_positions, span_bpe, risk_fractions = [], [], [], [], [], [], []
    for i in items:
        g = records[i['row_id']][0]
        lengths.append(len(g['response_token_ids']))
        if i not in resolved:
            continue
        offsets = np.asarray(g['response_token_offsets'])
        scope = np.flatnonzero((offsets[:, 1] > i['start']) & (offsets[:, 0] < i['end']))
        scopes.append(len(scope))
        char_width = max(1, i['end'] - i['start'])
        if i['gold'] == 1:
            risk_fractions.append(sum(s['end'] - s['start'] for s in i['annotation']['risk_spans']) / char_width)
        for s in i['annotation']['risk_spans']:
            spans.append(s['end'] - s['start'])
            positions.append((s['start'] - i['start']) / char_width)
            end_positions.append((s['end'] - i['start']) / char_width)
            span_bpe.append(int(np.sum((offsets[:, 1] > s['start']) & (offsets[:, 0] < s['end']))))
    conditions = {}
    for condition in ('complete', 'partial'):
        selected = [i for i in items if i['condition'] == condition]
        conditions[condition] = {'answers': len(selected),
            'resolved_supported': sum(i['gold'] == 0 and i['asserted_eligible'] and i['localization_status'] == 'resolved' for i in selected),
            'resolved_risky': sum(i['gold'] == 1 and i['asserted_eligible'] and i['localization_status'] == 'resolved' for i in selected),
            'safe_refusals': sum(i['reviewed_safe_refusal'] for i in selected),
            'unresolved_or_excluded_other': sum(not i['main_eligible'] for i in selected)}
    return {'questions': len({i['question_id'] for i in items}), 'event_or_subject_groups': len({i['group_id'] for i in items}),
        'answers': len(items), 'resolved_answers': len(resolved), 'resolved_supported_answers': sum(i['gold'] == 0 for i in resolved),
        'resolved_risky_answers': len(risk_items), 'questions_with_resolved_risk': len({i['question_id'] for i in risk_items}),
        'groups_with_resolved_risk': len({i['group_id'] for i in risk_items}),
        'safe_refusals': sum(i['reviewed_safe_refusal'] for i in items), 'conditions': conditions,
        'evidence_relations': dict(Counter(i['annotation']['evidence_relation'] for i in items)),
        'eligible_windows': len(windows), 'risk_windows': sum(w['gold'] for w in windows),
        'risk_window_fraction': sum(w['gold'] for w in windows) / len(windows) if windows else None,
        'eligible_tokens': len(tokens), 'risk_tokens': sum(t['gold'] for t in tokens),
        'raw_response_bpe_length': summary(lengths), 'resolved_answer_scope_raw_bpe_length': summary(scopes),
        'risk_span_char_length': summary(spans), 'risk_span_raw_bpe_length': summary(span_bpe),
        'risk_span_fraction_at_most_four_bpe': float(np.mean(np.asarray(span_bpe) <= 4)) if span_bpe else None,
        'risk_span_relative_char_start': summary(positions), 'risk_span_relative_char_end': summary(end_positions),
        'risky_answer_annotated_char_fraction': summary(risk_fractions),
        'windows_per_resolved_answer': len(windows) / len(resolved) if resolved else None}


def moments(x, w):
    w = w / w.sum()
    mean = (x * w[:, None]).sum(0)
    var = ((x - mean) ** 2 * w[:, None]).sum(0)
    return mean, np.sqrt(var)


def main():
    out = ROOT / 'results/diagnosis'
    models_path = ROOT / 'results/frozen_models.pkl'
    model_hash = r.sha(models_path)
    models = pickle.loads(models_path.read_bytes())
    om, nm = train_meta(r.OLD), train_meta(r.NEW)
    bank = r.Bank(om[2], nm[2])
    old = r.cohort(r.OLD, 'train', om, bank)
    new = r.cohort(r.NEW, 'train', nm, bank)
    expanded = r.combine(old, new)
    records = {**om[2], **nm[2]}
    packs = {'legacy': old, 'new': new, 'expanded': expanded}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Training annotations, frozen training features/models, and already-frozen validation summary only. No model fit, no test predictions, no threshold search.',
        'frozen_model_sha256': model_hash, 'cohorts': {}, 'feature_distribution': {}, 'models': {}}
    matrices, labels, eligible_rows = {}, {}, {}
    for name, pack in packs.items():
        result['cohorts'][name] = composition(pack, records)
        result['cohorts'][name]['by_category'] = {c: composition(pack, records, c) for c in sorted({i['category'] for i in pack['items']})}
        ix = np.flatnonzero([w['main_eligible'] for w in pack['windows']])
        x = pack['matrix'][ix].astype(float)
        rows = [pack['windows'][j] for j in ix]
        y = np.asarray([w['gold'] for w in rows])
        b = r.r13.r10.base_weights(rows, 'token')
        mu, sd = moments(x, b)
        matrices[name], labels[name], eligible_rows[name] = x, y, rows
        result['feature_distribution'][name] = {
            'unit': 'Eligible overlapping four-raw-BPE window; Lookback and NLL averaged over all raw tokens including punctuation.',
            'base_weighted_mean': mu.tolist(), 'base_weighted_std': sd.tolist(),
            'lb_mean_across_784_dimensions': float(mu[:784].mean()),
            'lb_dimension_std_distribution': summary(sd[:784]),
            'lb_window_cells_at_most_0_01_fraction': float(np.mean(x[:, :784] <= .01)),
            'lb_window_cells_at_least_0_99_fraction': float(np.mean(x[:, :784] >= .99)),
            'lb_near_constant_dimensions_std_below_1e_6': int(np.sum(sd[:784] < 1e-6)),
            'nll_window_mean_distribution': summary(x[:, 784]),
            'nll_by_gold': {str(k): summary(x[y == k, 784]) for k in (0, 1)}}
    old_mean = np.asarray(result['feature_distribution']['legacy']['base_weighted_mean'])
    old_std = np.asarray(result['feature_distribution']['legacy']['base_weighted_std'])
    new_mean = np.asarray(result['feature_distribution']['new']['base_weighted_mean'])
    new_std = np.asarray(result['feature_distribution']['new']['base_weighted_std'])
    shifts = np.abs(new_mean - old_mean) / np.maximum(old_std, 1e-12)
    ratios = new_std / np.maximum(old_std, 1e-12)
    result['old_new_feature_shift'] = {'lb_absolute_mean_shift_in_legacy_std': summary(shifts[:784]),
        'lb_dimensions_mean_shift_over_1_legacy_std': int(np.sum(shifts[:784] > 1)),
        'lb_new_to_legacy_std_ratio': summary(ratios[:784]),
        'nll_mean_shift_in_legacy_std': float(shifts[784]), 'nll_std_ratio': float(ratios[784]),
        'interpretation_limit': 'Marginal distribution comparison alone neither proves semantic invariance nor rules out domain shift.'}
    for name, m in models.items():
        cohort = m['cohort']
        x, rows, y = matrices[cohort], eligible_rows[cohort], labels[cohort]
        assert [w['window_key'] for w in rows] == m['fit_window_keys']
        b, w = m['base_weights'], m['loss_weights']
        head = m['model']; coef = head.coef_[0]
        probs = r.r13.probability(m, x[:, m['columns']])
        group_mass, answer_mass, source_mass = defaultdict(float), defaultdict(float), defaultdict(float)
        for row, weight in zip(rows, w):
            group_mass[row['group_id']] += float(weight)
            answer_mass[row['item_ids'][0]] += float(weight)
            source_mass['legacy' if row['row_id'] in om[2] else 'new'] += float(weight)
        detail = {'features': len(coef), 'parameters_including_intercept': len(coef) + 1,
            'C': m['C'], 'penalty': head.penalty, 'solver': head.solver, 'iterations': head.n_iter_.tolist(),
            'base_weight_sum': float(b.sum()), 'loss_weight_sum': float(w.sum()),
            'loss_weight_by_source': dict(source_mass), 'loss_weight_by_label': np.bincount(y, weights=w).tolist(),
            'per_group_loss_mass': summary(list(group_mass.values())), 'per_answer_loss_mass': summary(list(answer_mass.values())),
            'coefficient_l2_norm_standardized': float(np.linalg.norm(coef)), 'coefficient_abs_max_standardized': float(np.max(np.abs(coef))),
            'intercept': float(head.intercept_[0]), 'scaler_min_scale': float(m['scaler'].scale_.min()),
            'scaler_max_scale': float(m['scaler'].scale_.max()), 'scaler_zero_variance_dimensions': int(np.sum(m['scaler'].var_ == 0)),
            'train_probability_by_gold': {str(k): summary(probs[y == k]) for k in (0, 1)},
            'train_probability_below_0_01_fraction': float(np.mean(probs < .01)),
            'train_probability_above_0_99_fraction': float(np.mean(probs > .99))}
        if m['signal'] == 'nll':
            detail['standardized_nll_coefficient'] = float(coef[0])
            detail['validation_selected_raw_nll_cutoff'] = {}
            for unit, d in m['thresholds'].items():
                t = d['threshold']; logit = math.log(t / (1 - t))
                detail['validation_selected_raw_nll_cutoff'][unit] = float(m['scaler'].mean_[0] + m['scaler'].scale_[0] * (logit - head.intercept_[0]) / coef[0])
        result['models'][name] = detail
    result['model_direction_changes'] = {}
    for f in r.FEATURES:
        a, b = models['legacy_' + f], models['expanded_' + f]
        va = a['model'].coef_[0]
        vb = b['model'].coef_[0] / b['scaler'].scale_ * a['scaler'].scale_
        result['model_direction_changes'][f] = {'cosine_in_legacy_standardized_coordinates': float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))}
    fit, val = r.read(ROOT / 'results/fit_metrics.json'), r.read(ROOT / 'results/validation_metrics.json')
    result['already_frozen_fit_validation_summary'] = {'coverage': val['coverage'], 'methods': {}}
    for name in models:
        result['already_frozen_fit_validation_summary']['methods'][name] = {
            which: {unit: {k: source[name][unit][k] for k in ('f1', 'auroc', 'average_precision', 'precision', 'recall')}
                    for unit in ('windows', 'answers')}
            for which, source in [('own_training', fit['own_training']), ('same_legacy_training', fit['common_legacy_training']), ('validation', val['methods'])]}
    result['limits'] = [
        '421 questions give 842 paired answers, not 421 positive hallucination examples. Safe refusals and unresolved answers are excluded from localization.',
        '13247 windows overlap with stride one; adjacent width-four windows share three raw BPE tokens. Answers, question pairs and event groups induce dependence.',
        '398 combined group IDs are not a certification that old and new events are globally independent; legacy grouping is preserved.',
        'Every frozen fit retains C=0.01 and loss mass 3854. More observations broaden support but do not increase total data-term mass relative to regularization.',
        'Lookback 784 consists of 28 layers times 28 attention-head source-versus-generated-prefix position ratios, followed by window averaging and linear logistic regression. It has no direct source-claim semantic comparison, but the ratios can indirectly encode useful model behavior.',
        'NLL measures conditional surprise of the generated token, not evidential support. A fluent unsupported answer can have low NLL; a rare supported name can have high NLL.',
        'High training ranking and a training-to-validation gap support limited generalization for these frozen features and model. They do not prove a universal inability to learn, uniquely identify overfitting as the cause, or show that merely increasing capacity will help.',
        'No new test prediction was read, no model was trained, and no threshold or hyperparameter was selected in this review.']
    assert r.sha(models_path) == model_hash
    result['training_feature_files_verified'] = len(bank.files)
    result['frozen_model_unchanged_after_review'] = True
    out.mkdir(exist_ok=True)
    (out / 'training_signal_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps({'cohorts': {k: {z: v[z] for z in ('questions','event_or_subject_groups','answers','resolved_answers','resolved_supported_answers','resolved_risky_answers','questions_with_resolved_risk','groups_with_resolved_risk','safe_refusals','eligible_windows','risk_windows','risk_window_fraction')} for k,v in result['cohorts'].items()},
        'shift': result['old_new_feature_shift'], 'model_direction_changes': result['model_direction_changes']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with r.r10.threadpool_limits(limits=4):
        main()
