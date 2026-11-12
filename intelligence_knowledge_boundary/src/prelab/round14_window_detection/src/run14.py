"""Window detection with same-window token-max controls; no test retuning."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'round13_generalization_diagnostics'
spec = importlib.util.spec_from_file_location('r14_reuses_r13', PREVIOUS/'src/run13.py')
r13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r13)
r10, np, SOURCE = r13.r10, r13.np, r13.SOURCE
WIDTHS = (1, 4, 8)
METHODS = ('window_lr', 'token_max')


def snapshot():
    prior = r13.snapshot()
    assert prior == json.loads((PREVIOUS/'results/source_snapshot13.json').read_text('utf-8'))
    protocol = json.loads((ROOT/'protocol.json').read_text('utf-8'))
    assert protocol['widths'] == list(WIDTHS) and protocol['C'] == .01 and protocol['classifier_seed'] == r13.SEED
    files = [ROOT/'PLAN.md', ROOT/'protocol.json', *sorted((ROOT/'src').glob('*.py'))]
    return {'prior': prior, 'prior_models_sha256': r10.sha(PREVIOUS/'results/fitted_folds.pkl'),
            'prior_completion_sha256': r10.sha(PREVIOUS/'results/complete13.json'),
            'local_files_sha256': {p.relative_to(ROOT).as_posix(): r10.sha(p) for p in files}}


def starts(n, width):
    return list(range(n-width+1)) if n >= width else [0]


def catalogs(items, tokens, records, bank):
    token_by_key = {t['token_key']: t for t in tokens}
    rows, matrices = {}, {}
    for width in WIDTHS:
        cc, xx = [], []
        for item in items:
            if not item['asserted_eligible'] or item['localization_status'] != 'resolved':
                continue
            g = records[item['row_id']][0]; offsets = np.asarray(g['response_token_offsets'])
            indices = np.flatnonzero((offsets[:, 1] > item['start']) & (offsets[:, 0] < item['end']))
            assert len(indices) and np.all(np.diff(indices) == 1)
            features = bank.row(item['row_id'])['lb']
            for start in starts(len(indices), width):
                raw = indices[start:start+width]
                tt = [token_by_key[item['row_id']+f'__token{int(j)}'] for j in raw]
                # Unknown lexical material never becomes a negative window label.
                assert not any(t['lexical'] and not t['main_eligible'] for t in tt)
                eligible = [t for t in tt if t['main_eligible']]
                if not eligible:
                    continue
                left = max(item['start'], int(offsets[raw[0], 0])); right = min(item['end'], int(offsets[raw[-1], 1]))
                cc.append({'window_key': item['item_id']+f'__w{width}__start{int(raw[0])}',
                           'width': width, 'actual_width': len(raw), 'short_window': len(raw) < width,
                           'row_id': item['row_id'], 'item_ids': [item['item_id']],
                           'group_id': item['group_id'], 'condition': item['condition'],
                           'category': item['category'], 'start': left, 'end': right,
                           'text': g['response'][left:right], 'raw_token_indices': raw.tolist(),
                           'token_keys': [t['token_key'] for t in eligible],
                           'gold': int(any(t['gold'] for t in eligible))})
                xx.append(features[raw].mean(axis=0))
        rows[width] = cc; matrices[width] = np.asarray(xx, np.float32)
        assert len({r['window_key'] for r in cc}) == len(cc)
    native = [t for t in tokens if t['main_eligible']]
    assert [r['token_keys'][0] for r in rows[1]] == [t['token_key'] for t in native]
    assert [r['gold'] for r in rows[1]] == [t['gold'] for t in native]
    assert np.array_equal(matrices[1], bank.token_matrix(native, 'lb'))
    return rows, matrices


def count(y, scores, threshold=None, predictions=None):
    value = r13.count(y, scores, threshold, predictions)
    value['windows'] = value.pop('tokens'); value['risk_windows'] = value.pop('risk_tokens')
    return value


def bootstrap(rows):
    groups = sorted({r['group_id'] for r in rows}); gi = {g: j for j, g in enumerate(groups)}
    counts = np.zeros((len(groups), 2, 3), np.int64)
    for row in rows:
        for j, method in enumerate(METHODS):
            p, y = row['predictions'][method], row['gold']
            counts[gi[row['group_id']], j] += [int(p and y), int(p and not y), int(not p and y)]
    samples = np.random.default_rng(20260914).integers(0, len(groups), (2000, len(groups)))
    weights = np.stack([np.bincount(s, minlength=len(groups)) for s in samples])
    total = np.einsum('bg,gmc->bmc', weights, counts, optimize=True)
    tp, fp, fn = (total[:, :, k] for k in range(3)); f1 = 2*tp/(2*tp+fp+fn)
    return {'groups': len(groups), 'draws': 2000, 'seed': 20260914,
            'fixed_models_and_thresholds': True,
            'methods': {name: r10.ci(f1[:, j]) for j, name in enumerate(METHODS)},
            'window_lr_minus_token_max_f1': r10.ci(f1[:, 0]-f1[:, 1])}


def highlights(rows, items, tokens, regions):
    by_item = {i['item_id']: i for i in items if i['asserted_eligible'] and i['localization_status'] == 'resolved'}
    native = [t for t in tokens if t['main_eligible']]
    token_lookup = {t['token_key']: t for t in native}
    region_by_item = defaultdict(list)
    for region in regions:
        region_by_item[region['item_id']].append(region)
    windows_by_item = defaultdict(list)
    for row in rows:
        windows_by_item[row['item_ids'][0]].append(row)
    results, records = {}, []
    for method in METHODS:
        all_covered = set(); normal_alerts = risky_alerts = region_hit = region_total = region_full = 0
        normal_total = sum(i['gold'] == 0 for i in by_item.values())
        risky_total = sum(i['gold'] == 1 for i in by_item.values())
        for iid, item in by_item.items():
            active = [w for w in windows_by_item[iid] if w['predictions'][method]]
            covered = {k for w in active for k in w['token_keys']}
            all_covered |= covered
            normal_alerts += int(item['gold'] == 0 and bool(active))
            risky_alerts += int(item['gold'] == 1 and bool(active))
            merges = r10.metric.merge_spans(active)
            for region in region_by_item[iid]:
                required = set(region['token_keys']); region_total += 1
                region_hit += bool(required & covered)
                region_full += bool(required) and required <= covered
            records.append({'width': rows[0]['width'], 'method': method, 'item_id': iid, 'row_id': item['row_id'],
                            'text': item['text'], 'item_gold': item['gold'], 'item_start': item['start'],
                            'merged_intervals': merges, 'covered_token_keys': sorted(covered),
                            'gold_spans': [{'start': rr['start'], 'end': rr['end'], 'text': rr['text']} for rr in region_by_item[iid]]})
        total_risk = sum(t['gold'] == 1 for t in native)
        covered_risk = sum(token_lookup[k]['gold'] == 1 for k in all_covered)
        results[method] = {'eligible_answers': len(by_item), 'eligible_tokens': len(native),
                           'highlighted_tokens': len(all_covered), 'highlighted_token_fraction': len(all_covered)/len(native),
                           'risk_tokens_covered': covered_risk, 'risk_token_coverage': covered_risk/total_risk,
                           'normal_answer_alerts': normal_alerts, 'normal_answers': normal_total,
                           'normal_answer_false_alarm_rate': normal_alerts/normal_total,
                           'risk_answer_alerts': risky_alerts, 'risk_answers': risky_total,
                           'risk_answer_any_alert_recall': risky_alerts/risky_total,
                           'gold_regions': region_total, 'gold_regions_hit_any_token': region_hit,
                           'gold_region_any_overlap_recall': region_hit/region_total,
                           'gold_regions_fully_covered': int(region_full),
                           'note': 'Coarse union coverage, not calibrated token probabilities or exact token localization F1. Excludes refusals/unresolved answers.'}
    return results, records


def run():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    assert not (out/'complete14.json').exists(), 'Window exploration already completed'
    snap = snapshot(); r10.save(out/'source_snapshot14.json', snap)
    r10.save(out/'started14.json', {'utc': r10.utc(), 'source_snapshot_sha256': r10.sha(out/'source_snapshot14.json'),
                                  'original_validation_or_test_labels_used': False})
    meta = r10.metadata(SOURCE); items, tokens, regions, coverage = r10.cohort(SOURCE, 'train', meta)
    bank = r13.r12.r11.Bank(SOURCE, meta[2]); cc, xx = catalogs(items, tokens, meta[2], bank)
    native = [t for t in tokens if t['main_eligible']]
    token_x = bank.token_matrix(native, 'lb')
    old_folds = pickle.loads((PREVIOUS/'results/fitted_folds.pkl').read_bytes())
    fold_details, all_models, predictions = {}, {}, {k: [] for k in WIDTHS}
    begin = time.perf_counter()
    for fold in range(5):
        prior = old_folds[fold]; saved = {}; detail = {}
        old_model = prior['models']['lb_c001']
        token_p = r13.probability(old_model, token_x)
        token_scores = dict(zip([t['token_key'] for t in native], token_p))
        target_mass = sum(r['group_id'] in prior['fit_groups'] for r in cc[1])
        for width in WIDTHS:
            rows, matrix = cc[width], xx[width]
            ix, ca, ev = [np.asarray([j for j, row in enumerate(rows) if row['group_id'] in prior[name]], int)
                          for name in ['fit_groups', 'calibration_groups', 'evaluation_groups']]
            fit_rows = [rows[j] for j in ix]
            model = r13.fit_model(matrix[ix], fit_rows, 'lb', .01, target_mass, {})
            direct = r13.probability(model, matrix)
            pool = np.asarray([max(token_scores[k] for k in row['token_keys']) for row in rows])
            y = np.asarray([row['gold'] for row in rows], int)
            scores = {'window_lr': direct, 'token_max': pool}
            selected = {name: r10.threshold_search(y[ca], value[ca]) for name, value in scores.items()}
            model.update(fit_indices=ix, calibration_indices=ca, evaluation_indices=ev, thresholds=selected)
            if width == 1:
                assert np.array_equal(ix, old_model['fit_indices'])
                assert np.allclose(direct, pool, rtol=0, atol=1e-10)
                assert selected['window_lr']['threshold'] == old_model['threshold'] == selected['token_max']['threshold']
            saved[width] = model
            detail[str(width)] = {name: {'threshold': selected[name],
                                  'fit': count(y[ix], value[ix], selected[name]['threshold']),
                                  'calibration': count(y[ca], value[ca], selected[name]['threshold']),
                                  'evaluation': count(y[ev], value[ev], selected[name]['threshold'])}
                                  for name, value in scores.items()}
            for j in ev:
                predictions[width].append({**rows[int(j)], 'fold': fold,
                     'scores': {name: float(value[j]) for name, value in scores.items()},
                     'predictions': {name: bool(value[j] >= selected[name]['threshold']) for name, value in scores.items()}})
        all_models[fold] = {**{k: prior[k] for k in ['fit_groups', 'calibration_groups', 'evaluation_groups']}, 'models': saved}
        fold_details[str(fold)] = detail
        print(f'FOLD {fold+1}/5 complete', flush=True)
    elapsed = time.perf_counter()-begin
    summary = {'schema': 'round14-window-detection-v1', 'scope': 'exploratory training-source grouped CV',
               'coverage': coverage, 'folds': fold_details, 'widths': {}, 'fit_and_score_seconds': elapsed,
               'do_not_compare_different_width_f1_as_exact_localization': True}
    all_highlights = []
    for width in WIDTHS:
        rows = predictions[width]
        assert len(rows) == len(cc[width]) == len({r['window_key'] for r in rows})
        y = np.asarray([r['gold'] for r in rows])
        coarse, hh = highlights(rows, items, tokens, regions); all_highlights.extend(hh)
        summary['widths'][str(width)] = {'windows': len(rows), 'positive_windows': int(y.sum()),
            'positive_window_rate': float(y.mean()), 'short_windows': sum(r['short_window'] for r in rows),
            'always_alert_f1': float(2*y.sum()/(len(y)+y.sum())),
            'methods': {name: count(y, [r['scores'][name] for r in rows], predictions=[r['predictions'][name] for r in rows]) for name in METHODS},
            'coarse_highlight': coarse, 'paired_bootstrap': bootstrap(rows),
            'fold_mean_auroc': {name: float(np.mean([fold_details[str(f)][str(width)][name]['evaluation']['auroc'] for f in range(5)])) for name in METHODS}}
        r10.savel(out/f'window_catalog_k{width}.jsonl', cc[width])
        r10.savel(out/f'window_scores_k{width}.jsonl', rows)
    assert snapshot() == snap
    (out/'fitted_folds.pkl').write_bytes(pickle.dumps(all_models, protocol=5))
    r10.savel(out/'highlight_regions.jsonl', all_highlights); r10.save(out/'summary.json', summary)
    files = ['source_snapshot14.json', 'fitted_folds.pkl', 'summary.json', 'highlight_regions.jsonl']
    files += [f'window_{kind}_k{k}.jsonl' for kind in ('catalog', 'scores') for k in WIDTHS]
    r10.save(out/'complete14.json', {'utc': r10.utc(), 'direct_fits': 15, 'reused_token_models': 5,
                                   'original_validation_or_test_labels_used': False,
                                   'files_sha256': {name: r10.sha(out/name) for name in files}})
    print('ROUND14_WINDOW_EXPLORATION_COMPLETE', flush=True)


def smoke():
    assert starts(10, 8) == [0, 1, 2] and starts(3, 8) == [0]
    toy = np.asarray([0, 0, 0, 1, 0, 0, 0, 0])
    assert int(toy.max()) == 1
    assert r10.metric.merge_spans([{'start': 0, 'end': 8}, {'start': 2, 'end': 10}]) == [[0, 10]]
    print('ROUND14_SMOKE_PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['smoke', 'run'])
    args = parser.parse_args()
    with r10.threadpool_limits(limits=4):
        {'smoke': smoke, 'run': run}[args.stage]()
