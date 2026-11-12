"""Independent CPU replay of the completed fixed large-combination comparison.

No fit calls, GPU, unfinished large predictions, or official-test data access.
Thresholds, counts, grouping, weights and selection are recomputed independently.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import pickle
import sys
import time
import traceback
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
import run_development as q

ROOT = q.ROOT
OUT = ROOT / 'results/large_matched_combination_v2'
LARGE = ROOT / 'results/full_context_encoder_large_v1/full_finetune'
OLD_LR = ROOT / 'results/citation_alignment_lr_v1'
OLD_TREE = ROOT / 'results/completed_score_combiner_v1'
INPUTS = ROOT / 'results/completed_score_fusion_v1'
FEATURES = ROOT / 'results/citation_alignment_v1'
PEERS = ('lookback', 'harp_claim', 'semantic_claim')
CS = (.001, .01, .1)
TRAINER = Path(__file__).with_name('run_large_matched_combination_v2.py')
FLOAT_ATOL = 1e-10  # Independent weighted-moment roundoff only; probabilities must be exact.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def checked_pickle(path, expected):
    assert sha(path) == expected, str(path)
    return pickle.loads(Path(path).read_bytes())


def checked_npz(path, expected):
    assert sha(path) == expected, str(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def best_threshold(labels, scores):
    """Aggregate equal scores in ascending order; independently scan every cutoff."""
    y, s = np.asarray(labels, np.int64), np.asarray(scores, np.float64)
    assert len(y) == len(s) and set(y.tolist()) == {0, 1} and np.isfinite(s).all()
    values, bins = np.unique(s, return_inverse=True)
    n = np.bincount(bins, minlength=len(values))
    positives = np.bincount(bins, weights=y, minlength=len(values)).astype(np.int64)
    predicted = np.cumsum(n[::-1])[::-1]
    tp = np.cumsum(positives[::-1])[::-1]
    f1 = 2 * tp / (predicted + int(y.sum()))
    precision = tp / predicted
    choices = [(float(f), float(p), float(t)) for f, p, t in zip(f1, precision, values)]
    choices.append((0., 0., float(np.nextafter(values[-1], np.inf))))
    f, p, t = max(choices)
    return {'threshold': t, 'f1': f, 'precision': p, 'rows': len(y), 'positive': int(y.sum())}


def counts(labels, scores, threshold):
    y, s = np.asarray(labels, np.int64), np.asarray(scores, np.float64)
    assert len(y) == len(s) and np.isfinite(s).all()
    # Confusion matrix via encoded pairs instead of the trainer's separate masks.
    c = np.bincount(2 * y + (s >= threshold).astype(np.int64), minlength=4)
    tn, fp, fn, tp = map(int, c)
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'auroc': float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, s)) if y.sum() else None}


def answer_max(meta, scores):
    best = {}
    for row, score in zip(meta['windows'], scores):
        rid = row['response_id']
        best[rid] = max(best.get(rid, -np.inf), float(score))
    assert len(best) == len(meta['answers'])
    return np.asarray([best[a['response_id']] for a in meta['answers']], np.float64)


def metric_set(meta, scores, thresholds):
    answer = answer_max(meta, scores)
    result = {}
    for part in ('fit', 'calibration'):
        wi = [i for i, w in enumerate(meta['windows']) if w['partition'] == part]
        ai = [i for i, a in enumerate(meta['answers']) if a['partition'] == part]
        result[part] = {'windows': counts([meta['windows'][i]['label'] for i in wi], scores[wi], thresholds['window']['threshold']),
                        'answers': counts([meta['answers'][i]['label'] for i in ai], answer[ai], thresholds['answer']['threshold'])}
    return result


def check_entry(meta, scores, saved_answer, entry, coefficient=None, check_key=True):
    assert scores.shape == (210364,) and np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    answer = answer_max(meta, scores)
    assert np.array_equal(answer, saved_answer), 'answer maximum changed'
    ts = {'window': best_threshold([w['label'] for w in meta['windows'][168123:]], scores[168123:]),
          'answer': best_threshold([a['label'] for a in meta['answers'][634:]], answer[634:])}
    assert ts == entry['thresholds'], (entry.get('candidate', entry.get('peer')), 'threshold mismatch')
    metrics = metric_set(meta, scores, ts)
    assert metrics == entry['metrics'], (entry.get('candidate', entry.get('peer')), 'metric mismatch')
    key = [min(ts['window']['f1'], ts['answer']['f1']), ts['window']['f1'], ts['window']['precision'], -float(coefficient or 0.)]
    if check_key:
        assert key == entry['selection_key']
    return metrics, key


def fit_weights(meta):
    fit = meta['windows'][:168123]
    groups = defaultdict(set)
    lengths = Counter()
    for w in fit:
        groups[w['group_id']].add(w['answer_id'])
        lengths[w['answer_id']] += 1
    assert len(groups) == 615 and len(lengths) == 634
    n = len(fit)
    b = np.asarray([n / (len(groups) * len(groups[w['group_id']]) * lengths[w['answer_id']]) for w in fit])
    y = np.asarray([w['label'] for w in fit], np.int64)
    factors = n / (2 * np.bincount(y, weights=b, minlength=2))
    loss = b * factors[y]
    group_indices = defaultdict(list)
    for i, w in enumerate(fit):
        group_indices[w['group_id']].append(i)
    for ix in group_indices.values():
        loss[ix] *= (n / len(groups)) / loss[ix].sum()
    loss *= n / loss.sum()
    # Independent closed form compared against frozen implementation output.
    rb, rl, rf, ry = q.base_weights(meta)
    assert np.array_equal(y, ry)
    for actual, expected in ((b, rb), (loss, rl), (factors, rf)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=FLOAT_ATOL)
    return b, loss, y


def scaler_check(scaler, x, weight):
    assert x.shape[0] == 210364 and int(scaler.n_features_in_) == x.shape[1]
    xf = x[:len(weight)].astype(np.float64)
    mean = np.average(xf, axis=0, weights=weight)
    variance = np.average((xf - mean) ** 2, axis=0, weights=weight)
    np.testing.assert_allclose(scaler.mean_, mean, rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_allclose(scaler.var_, variance, rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_allclose(scaler.n_samples_seen_, weight.sum(), rtol=0, atol=1e-7)
    assert np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
    return {'fit_only_weighted_mean_max_diff': float(np.max(np.abs(scaler.mean_ - mean))),
            'fit_only_weighted_variance_max_diff': float(np.max(np.abs(scaler.var_ - variance)))}


def linear_parameters(model, c, width):
    params = model.get_params()
    for k, v in {'C': c, 'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'random_state': 20261010}.items():
        assert params[k] == v
    assert model.n_features_in_ == width and np.array_equal(model.classes_, [0, 1])
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    assert np.asarray(model.n_iter_).max() < 2000


def tree_parameters(model, width):
    expected = {'max_iter': 100, 'max_depth': 2, 'max_leaf_nodes': 4, 'min_samples_leaf': 20,
                'learning_rate': .05, 'l2_regularization': 1., 'early_stopping': False,
                'monotonic_cst': [1] * width, 'random_state': 20261008}
    params = model.get_params()
    for key, value in expected.items():
        assert params[key] == value, key
    assert model.n_iter_ == 100 and model.n_features_in_ == width
    assert np.array_equal(model.classes_, [0, 1])


def cpu_selfcheck():
    # Ties, repeated probabilities, threshold endpoints and a non-F1-optimal higher precision cutoff.
    for y, s in [([0, 1, 1, 0, 1], [.2, .4, .4, .7, .9]), ([0, 1], [.5, .5]),
                 ([1, 0, 1, 0], [0., 0., 1., 1.])]:
        actual = best_threshold(y, s)
        candidates = [np.nextafter(max(s), np.inf), *set(s)]
        wanted = max((counts(y, s, t)['f1'], counts(y, s, t)['precision'], t) for t in candidates)
        assert (actual['f1'], actual['precision'], actual['threshold']) == wanted
        assert actual == q.choose_threshold(y, s)
    meta = {'answers': [{'response_id': 'a'}, {'response_id': 'b'}],
            'windows': [{'response_id': 'a'}, {'response_id': 'b'}, {'response_id': 'a'}]}
    assert answer_max(meta, np.asarray([.9, .2, .4])).tolist() == [.9, .2]
    save(OUT/'AUDIT_CPU_SELFCHECK.json', {'passed': True, 'synthetic_threshold_ties_and_endpoints': True,
         'noncontiguous_answer_max': True, 'large_predictions_read': False, 'models_loaded': False,
         'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})
    save(OUT/'AUDIT_DESIGN.json', {'auditor_sha256': sha(Path(__file__)), 'trainer_sha256': sha(TRAINER),
         'preparation_sha256': sha(OUT/'preparation_complete.json'), 'controls': 12, 'new_models': 12,
         'probability_replay': 'exact array equality', 'weighted_moment_atol': FLOAT_ATOL,
         'gate': 'Both large and combination complete.json must exist before loading predictions or models.',
         'scope': 'Original native634 fit/159 calibration only. This audit does not prove global process history or independence of repeatedly inspected calibration.'})
    print('LARGE_MATCHED_AUDIT_CPU_PREPARED_NO_PREDICTIONS_NO_FIT', flush=True)


def run():
    # This gate precedes all prediction/model reads, including old controls.
    if not (OUT/'complete.json').exists() or not (LARGE/'complete.json').exists():
        print('WAIT_COMPLETED_LARGE_AND_12_COMBINATIONS_NO_PREDICTIONS_READ', flush=True)
        return
    started = time.perf_counter()
    design = read(OUT/'AUDIT_DESIGN.json')
    assert design['auditor_sha256'] == sha(Path(__file__)) and design['trainer_sha256'] == sha(TRAINER)
    assert design['preparation_sha256'] == sha(OUT/'preparation_complete.json')
    reads = set()

    def file_access_guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if str(p).casefold().startswith(str(ROOT.parent).casefold()):
            low = str(p).replace('\\', '/').casefold()
            assert not any(token in low for token in ('official_test', '/test.jsonl', '_test.jsonl', '/heldout'))
            assert not low.endswith('/data/raw/response.jsonl')
            reads.add(str(p))

    sys.addaudithook(file_access_guard)
    frozen = read(OUT/'preparation_complete.json')
    for path, expected in frozen['source_sha256'].items():
        assert sha(path) == expected
    assert frozen['protocol_sha256'] == sha(OUT/'protocol.json')
    assert frozen['CPU_check_sha256'] == sha(OUT/'CPU_SELFCHECK.json')
    assert frozen['control_check_sha256'] == sha(OUT/'CONTROL_REPLAY_CHECK.json')
    done = read(OUT/'complete.json'); summary = read(OUT/'summary.json')
    assert sha(OUT/'summary.json') == done['summary_sha256'] and done['new_fits'] == 12
    assert not done['official_test_opened'] and not done['GPU_used']
    assert summary['new_LR_fits'] == 9 and summary['new_tree_fits'] == 3 and summary['control_refits'] == 0
    assert not summary['official_test_opened'] and not summary['GPU_used']
    lc = read(LARGE/'complete.json')
    assert not lc['official_test_opened'] and [e['epoch'] for e in lc['all_epochs']] == list(range(7))
    for e in lc['all_epochs'][1:]:
        m = e['calibration']
        assert e['selection_key'] == [min(m['windows']['f1'], m['answers']['f1']), m['windows']['f1'], m['windows']['precision'], -e['epoch']]
    chosen = max(lc['all_epochs'][1:], key=lambda e: e['selection_key'])
    assert chosen == lc['selected'] == summary['large_selected']
    start_record = read(OUT/'started.json')
    assert start_record['large_complete_sha256'] == sha(LARGE/'complete.json')
    assert start_record['preparation_sha256'] == sha(OUT/'preparation_complete.json')
    assert start_record['selected_epoch'] == chosen['epoch']
    meta = q.metadata()
    assert len(meta['answers']) == 793 and len(meta['windows']) == 210364
    assert meta['bounds'] == {'fit': [0, 168123], 'calibration': [168123, 210364]}
    b, loss, y = fit_weights(meta)
    large_path = LARGE/f"epoch_{chosen['epoch']:02d}_token_predictions.npz"
    assert sha(large_path) == start_record['large_tokens_sha256'] == chosen['artifacts_sha256']['_token_predictions.npz']
    with np.load(large_path, allow_pickle=False) as z:
        assert len(z.files) == 3839
        probabilities = {a['response_id']: z[a['response_id']].copy() for a in meta['answers']}
    large = []
    for w in meta['windows']:
        tokens = meta['by_response'][w['response_id']]['tokens']
        index = np.asarray(w['token_indices'], np.int64)
        mask = np.asarray(tokens['lexical_mask'], bool)[index]
        raw = probabilities[w['response_id']]
        assert len(raw) == len(tokens['lexical_mask']) and np.isfinite(raw).all()
        assert mask.any() and len(index) == min(4, len(raw))
        large.append(float(np.max(raw[index][mask])))
    large = np.asarray(large, np.float64)
    assert np.array_equal(large, np.load(OUT/'large_window_probability.npy', allow_pickle=False))
    assert metric_set(meta, large, chosen['thresholds'])['calibration'] == chosen['calibration']
    la = answer_max(meta, large)
    large_ts = {'window': best_threshold([w['label'] for w in meta['windows'][168123:]], large[168123:]),
                'answer': best_threshold([a['label'] for a in meta['answers'][634:]], la[634:])}
    assert large_ts == chosen['thresholds']
    fm = read(FEATURES/'preparation_complete.json')
    for name, expected in fm['files_sha256'].items():
        assert sha(FEATURES/name) == expected
    assert fm['window_order_sha256'] == q.digest([w['window_id'] for w in meta['windows']])
    features = np.load(FEATURES/'window_features.npy', allow_pickle=False)
    assert features.shape == (210364, 8) and np.isfinite(features).all()
    old_lr, old_tree, source = [read(directory/'summary.json') for directory in (OLD_LR, OLD_TREE, INPUTS)]
    for directory in (OLD_LR, OLD_TREE, INPUTS):
        assert sha(directory/'summary.json') == read(directory/'complete.json')['summary_sha256']
    controls, new_checks, comparisons = [], [], []
    expected_families = {peer+'__'+mode for peer in PEERS for mode in ('large_lr', 'large_tree')}
    assert set(summary['all_candidates']) == set(summary['selected']) == expected_families
    common_tail = None
    for peer in PEERS:
        two_columns = []
        for alpha in (0., 1.):
            e = next(e for e in source['all_candidates'][peer] if e['tail_weight'] == alpha)
            scores = checked_npz(INPUTS/(e['candidate']+'_scores.npz'), e['scores_sha256'])
            check_entry(meta, scores['window_scores'], scores['answer_scores'], e, alpha)
            two_columns.append(scores['window_scores'])
        two = np.column_stack(two_columns)
        if common_tail is not None:
            assert np.array_equal(two[:, 1], common_tail)
        common_tail = two[:, 1]
        family = peer+'__two_scores_and_citation'
        old_entries = old_lr['all_candidates'][family]
        assert [e['C'] for e in old_entries] == list(CS)
        scaler = checked_pickle(OLD_LR/(family+'_scaler.pkl'), old_entries[0]['scaler_sha256'])
        old_x = np.column_stack((two, features))
        old_scaler_check = scaler_check(scaler, old_x, b)
        for e in old_entries:
            assert e['peer'] == peer and e['input_width'] == 10
            assert sha(OLD_LR/(family+'_scaler.pkl')) == e['scaler_sha256']
            model = checked_pickle(OLD_LR/(e['candidate']+'.pkl'), e['model_sha256'])
            linear_parameters(model, e['C'], 10)
            stored = checked_npz(OLD_LR/(e['candidate']+'_scores.npz'), e['scores_sha256'])
            actual = model.predict_proba(scaler.transform(old_x))[:, 1]
            assert np.array_equal(actual, stored['window_scores'])
            metrics, _ = check_entry(meta, actual, stored['answer_scores'], e, e['C'])
            controls.append({'candidate': e['candidate'], 'model_replay_max_abs_diff': 0., 'refitted': False,
                             'calibration': metrics['calibration'], **old_scaler_check})
        assert max(old_entries, key=lambda e: e['selection_key']) == old_lr['selected'][family]
        old_tree_entry = old_tree['methods'][peer]
        model = checked_pickle(OLD_TREE/(peer+'.pkl'), old_tree_entry['model_sha256'])
        tree_parameters(model, 2)
        stored = checked_npz(OLD_TREE/(peer+'_scores.npz'), old_tree_entry['scores_sha256'])
        actual = model.predict_proba(two)[:, 1]
        assert np.array_equal(actual, stored['window_scores'])
        old_tree_metrics, _ = check_entry(meta, actual, stored['answer_scores'], old_tree_entry, check_key=False)
        controls.append({'candidate': peer+'_monotone_tree', 'model_replay_max_abs_diff': 0., 'refitted': False,
                         'calibration': old_tree_metrics['calibration']})
        for mode in ('large_lr', 'large_tree'):
            entries = summary['all_candidates'][peer+'__'+mode]
            assert [e['C'] for e in entries] == (list(CS) if mode == 'large_lr' else [None])
            for entry in entries:
                name = entry['candidate']
                assert read(OUT/(name+'.json')) == entry and entry['peer'] == peer and entry['mode'] == mode
                assert entry['large_epoch'] == chosen['epoch']
                pack = checked_pickle(OUT/(name+'.pkl'), entry['model_sha256'])
                assert set(pack) == {'model', 'scaler'}
                x = np.column_stack((old_x, large)) if mode == 'large_lr' else np.column_stack((two, large))
                extra = {}
                if mode == 'large_lr':
                    linear_parameters(pack['model'], entry['C'], 11)
                    extra = scaler_check(pack['scaler'], x, b)
                    np.testing.assert_allclose(pack['scaler'].mean_[:10], scaler.mean_, rtol=0, atol=FLOAT_ATOL)
                    np.testing.assert_allclose(pack['scaler'].var_[:10], scaler.var_, rtol=0, atol=FLOAT_ATOL)
                    x = pack['scaler'].transform(x)
                else:
                    assert pack['scaler'] is None
                    tree_parameters(pack['model'], 3)
                assert entry['input_width'] == x.shape[1]
                assert entry['iterations'] == np.asarray(pack['model'].n_iter_).tolist()
                actual = pack['model'].predict_proba(x)[:, 1]
                stored = checked_npz(OUT/(name+'_scores.npz'), entry['scores_sha256'])
                assert np.array_equal(actual, stored['window_scores'])
                metrics, _ = check_entry(meta, actual, stored['answer_scores'], entry, entry['C'])
                new_checks.append({'candidate': name, 'model_replay_max_abs_diff': 0., 'calibration': metrics['calibration'], **extra})
            selected = max(entries, key=lambda e: e['selection_key'])
            assert selected == summary['selected'][peer+'__'+mode]
            own_control = old_lr['selected'][family] if mode == 'large_lr' else old_tree_entry
            comparisons.append({'peer': peer, 'mode': mode, 'candidate': selected['candidate'],
                'candidate_metrics': selected['metrics']['calibration'],
                'matched_control': own_control.get('candidate', peer+'_monotone_tree'),
                'matched_control_metrics': own_control['metrics']['calibration'],
                'same_peer_selected_fixed_weight_metrics': source['selected'][peer]['metrics']['calibration'],
                'standalone_large_metrics': chosen['calibration'],
                'both_metrics_from_same_candidate': True})
    assert len(controls) == len(new_checks) == 12 and len(comparisons) == 6
    all_control_comparisons = []
    for new in new_checks:
        for old in controls:
            nw, na = new['calibration']['windows']['f1'], new['calibration']['answers']['f1']
            ow, oa = old['calibration']['windows']['f1'], old['calibration']['answers']['f1']
            all_control_comparisons.append({'candidate': new['candidate'], 'control': old['candidate'],
                'window_F1_difference': nw-ow, 'answer_F1_difference': na-oa,
                'candidate_at_least_control_on_both': nw>=ow and na>=oa})
    result = {'passed': True, 'new_models_replayed': new_checks, 'frozen_controls_replayed': controls,
        'every_new_candidate_vs_every_old_control': all_control_comparisons,
        'same_candidate_comparisons': comparisons, 'large_selected_epoch': chosen['epoch'],
        'all_model_and_window_probabilities_exact': True, 'all_answer_maxima_exact': True,
        'all_thresholds_and_counts_independently_recomputed': True,
        'all_C_and_epoch_selection_same_candidate_for_both_units': True,
        'fit_only_scaler_moments_checked': True, 'new_and_old_fit_weight_definition_same': True,
        'fit_rows': len(y), 'fit_group_count': 615, 'base_mass': float(b.sum()), 'loss_mass': float(loss.sum()),
        'summary_sha256': sha(OUT/'summary.json'), 'complete_sha256': sha(OUT/'complete.json'),
        'auditor_sha256': sha(Path(__file__)), 'read_files_within_prelab': sorted(reads),
        'official_test_opened_by_audit': False, 'test_read_claim_scope': 'Auditor file guard plus frozen source/manifest review; not a proof of all prior process history.',
        'new_fits': 0, 'GPU_used': False, 'seconds': time.perf_counter()-started,
        'limits': 'Repeated calibration and in-sample base fit predictions remain. Equal combiner fit budget does not equalize upstream model cost or establish future dominance.'}
    save(OUT/'INDEPENDENT_AUDIT.json', result)
    lines = ['# Independent large-combination audit', '',
             'All 12 new models and 12 frozen controls replay exactly. All thresholds, counts, answer maxima and single-candidate selection checks pass. No refitting or GPU was used.', '',
             '| Peer | Combination | Window F1 | Answer F1 | Matched control window F1 | Matched control answer F1 |',
             '|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        a, b2 = r['candidate_metrics'], r['matched_control_metrics']
        lines.append(f"| {r['peer']} | {r['mode']} | {a['windows']['f1']:.6f} | {a['answers']['f1']:.6f} | {b2['windows']['f1']:.6f} | {b2['answers']['f1']:.6f} |")
    lines += ['', 'Each row uses one candidate for both metrics. Standalone large and the same peer fixed-weight controls are retained in the JSON report. Upstream cost differs; fit scores are in-sample and calibration has been repeatedly inspected. This is not an independent test or a guarantee of future superiority.']
    (OUT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('LARGE_MATCHED_INDEPENDENT_AUDIT_PASSED_12_NEW_12_OLD_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('selfcheck', 'run')); args = p.parse_args()
    with threadpool_limits(limits=4):
        try:
            {'selfcheck': cpu_selfcheck, 'run': run}[args.stage]()
        except Exception:
            save(OUT/f'AUDIT_FAILURE_{time.time_ns()}.json', {'stage': args.stage, 'traceback': traceback.format_exc(), 'new_fits': 0, 'GPU_used': False})
            raise
