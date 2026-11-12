"""Independent post-run audit; no experiment training/prediction functions imported.

Only R16 original training examples are parsed. Reuses the independent R17
character/count utilities, not a production evaluator. No classifier/PCA refit.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import importlib.util
import json
import pickle
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT.parent/'round16_dataset_expansion'
UTILITY = ROOT.parent/'round17_expanded_retraining/results/audit17.py'
spec = importlib.util.spec_from_file_location('independent_r17_count_utilities', UTILITY)
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
read, lines, sha, close = u.read, u.lines, u.sha, u.close
METHODS = ('mean_lr', 'slots_lr', 'shuffled_slots_lr', 'mean_mlp',
           'mean_surface', 'mean_alignment', 'mean_both', 'mean_hidden32')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', 'utf-8')


def rebuild():
    """Reconstruct source window geometry and all eight feature designs directly."""
    rows = [r for r in lines(NEW/'data/inputs.jsonl') if r['split'] == 'train']
    annotations = lines(NEW/'data/annotations_train.jsonl')
    labels = {a['row_id']: a for a in annotations}
    assert len(labels) == len(annotations) == len(rows) == 602
    safe_path = NEW/'data/safe_refusals_train.json'
    assert sha(safe_path) == read(NEW/'data/question_label_policy.json')['reviewed_safe_refusal_files_sha256']['train']
    safe = set(read(safe_path)['safe_refusal_item_ids'])
    manifest = read(ROOT/'data/feature_manifest.json')
    assert set(manifest['records']) == {r['row_id'] for r in rows}
    pack = {k: [] for k in ('items', 'tokens', 'windows', 'regions')}
    design = {k: [] for k in ('base', 'surface', 'alignment', 'hidden', 'slots', 'shuffled_slots')}
    files, pairs = {}, defaultdict(list)
    for row in rows:
        rid = row['row_id']; pairs[row['question_id']].append(row['condition'])
        gp = NEW/'data/generation_records'/(rid+'.json')
        g, a = read(gp), labels[rid]
        assert g['split'] == 'train' and a['source_generation_sha256'] == sha(gp)
        assert not a.get('token_scores_viewed', False)
        item, tt, ww, rr, _ = u.label_geometry(row, g, a, safe)
        pack['items'].append(item); pack['tokens'].extend(tt); pack['windows'].extend(ww)
        for region in rr:
            region.update(row_id=rid, group_id=row['group_id'])
        pack['regions'].extend(rr)
        entry = manifest['records'][rid]
        path, side = ROOT/entry['npz'], ROOT/entry['json']; meta = read(side)
        assert sha(path) == entry['npz_sha256'] == meta['arrays_sha256']
        assert sha(side) == entry['json_sha256'] and meta['source_generation_sha256'] == sha(gp)
        assert meta['extraction_signature_sha256'] == manifest['extraction_signature_sha256']
        for p in (path, side): files[str(p.resolve())] = sha(p)
        with np.load(path, allow_pickle=False) as arr:
            assert arr['token_ids'].tolist() == g['response_token_ids']
            assert arr['response_token_offsets'].tolist() == g['response_token_offsets']
            n = len(g['response_token_ids'])
            for key, width in [('lb', 784), ('new_features', 16), ('surface_features', 8), ('hidden_28', 3584)]:
                assert arr[key].shape == (n, width) and arr[key].dtype == np.float32
                assert np.isfinite(arr[key]).all()
            assert arr['nll'].shape == (n,) and arr['nll'].dtype == np.float32
            assert np.isfinite(arr['nll']).all() and (arr['nll'] >= 0).all()
            assert (arr['lb'] >= 0).all() and (arr['lb'] <= 1).all()
            signal = np.column_stack((arr['lb'], arr['nll']))
            for w in ww:
                raw = w['raw_token_indices']; count = len(raw)
                design['base'].append(signal[raw].mean(0))
                for key, source in [('surface', 'surface_features'), ('alignment', 'new_features'), ('hidden', 'hidden_28')]:
                    design[key].append(arr[source][raw].mean(0))
                slots = signal[raw]
                if count < 4: slots = np.vstack((slots, np.repeat(slots[-1:], 4-count, axis=0)))
                mask = np.asarray([1]*count+[0]*(4-count), np.float32)
                design['slots'].append(np.concatenate((slots.ravel(), mask)))
                seed = int.from_bytes(hashlib.sha256(w['window_key'].encode()).digest()[:8], 'little')
                permutation = np.random.default_rng(seed).permutation(4)
                design['shuffled_slots'].append(np.concatenate((slots[permutation].ravel(), mask[permutation])))
    assert all(sorted(cc) == ['complete', 'partial'] for cc in pairs.values())
    pack['questions'] = len(pairs)
    design = {k: np.asarray(v, np.float32) for k, v in design.items()}
    assert (pack['questions'], len(pack['windows'])) == (301, 12222)
    assert sum(w['main_eligible'] for w in pack['windows']) == 9526
    return pack, design, files


def subset(pack, groups):
    groups = set(groups)
    ix = np.asarray([j for j, w in enumerate(pack['windows']) if w['group_id'] in groups], int)
    result = {k: [r for r in pack[k] if r['group_id'] in groups] for k in ('items', 'tokens', 'windows', 'regions')}
    return result, ix


def raw_matrix(model, design, indices=None):
    method = model['method']
    def x(k): return design[k] if indices is None else design[k][indices]
    if method == 'slots_lr': return x('slots')
    if method == 'shuffled_slots_lr': return x('shuffled_slots')
    blocks = [x('base')]
    if method in ('mean_surface', 'mean_both'): blocks.append(x('surface'))
    if method in ('mean_alignment', 'mean_both'): blocks.append(x('alignment'))
    if method == 'mean_hidden32':
        pc = model['projection']
        blocks.append((x('hidden').astype(np.float64)-pc['mean']) @ pc['components'].T)
    return blocks[0] if len(blocks) == 1 else np.column_stack(blocks)


def standardized(model, raw):
    z = raw.copy(); z -= model['scaler'].mean_; z /= model['scaler'].scale_
    return z.astype(np.float32)


def forward_mlp(state, z):
    """Direct saved-tensor arithmetic, no production module/network loading."""
    tx = torch.as_tensor(z, dtype=torch.float32)
    scores, logits = [], []
    with torch.no_grad():
        for block in tx.split(1024):
            h = torch.nn.functional.linear(block, state['network.0.weight'], state['network.0.bias']).relu()
            logit = torch.nn.functional.linear(h, state['network.3.weight'], state['network.3.bias']).squeeze(-1)
            scores.extend(logit.sigmoid().numpy().astype(float).tolist())
            logits.extend(logit.numpy().astype(float).tolist())
    return np.asarray(scores), np.asarray(logits)


def predict(model, design):
    z = standardized(model, raw_matrix(model, design))
    if model['kind'] == 'lr':
        c = model['model']; return expit(z @ c.coef_[0]+c.intercept_[0]), None, None
    output = [forward_mlp(m['state_dict'], z) for m in model['members']]
    member_scores = np.stack([r[0] for r in output])
    return member_scores.mean(0), member_scores, np.stack([r[1] for r in output])


def inspect_fit(model, design, fit_ix, fit_rows, config, fold):
    assert np.array_equal(model['fit_ix'], fit_ix)
    assert model['fit_window_keys'] == model['fit_keys'] == [r['window_key'] for r in fit_rows]
    y = np.asarray([r['gold'] for r in fit_rows], int)
    assert np.array_equal(model['fit_y'], y)
    b, loss, factors = u.weights(fit_rows, 3854)
    close(b, model['base_weights']); close(loss, model['loss_weights']); close(factors, model['class_factors'])
    close(loss.sum(), 3854.)
    assert model['C'] == .01 and model['target_loss_mass'] == 3854
    result = {'fit_windows': len(fit_rows), 'total_loss_mass': float(loss.sum()), 'input_width': model['input_width']}
    if model['method'] == 'mean_hidden32':
        pc = model['projection']; h = design['hidden'][fit_ix].astype(np.float64)
        normal = b/b.sum(); center = normal @ h
        close(center, pc['mean'], 'PCA fit-only mean', atol=1e-9)
        centered = h-center; components = pc['components']
        assert components.shape == (32, 3584) and pc['n_components'] == 32 and pc['n_iter'] == 5
        assert pc['seed'] == config['hidden_projection']['seed']+fold and pc['whiten'] is False
        close(components @ components.T, np.eye(32), 'PCA orthonormality', atol=1e-8)
        total = float(np.sum(centered**2*normal[:, None]))
        close(total, pc['total_variance'], 'PCA fit variance')
        projected = centered @ components.T
        directional = np.sum(projected**2*normal[:, None], axis=0)
        close(np.asarray(pc['singular_values'])**2, pc['explained_variance'], 'PCA singular/variance')
        close(np.asarray(pc['explained_variance'])/total, pc['explained_variance_ratio'], 'PCA ratio')
        # Approximate randomized SVD may omit some residual energy. Inspect rather
        # than silently refit to manufacture a bitwise-equal decomposition.
        relative = np.max(np.abs(directional-pc['explained_variance'])/np.maximum(directional, 1e-15))
        assert np.isfinite(relative) and relative < .05, ('Unexpected PCA projection energy', relative)
        result['pca'] = {'fit_mean_verified': True, 'orthonormality_verified': True,
                        'maximum_relative_projection_energy_residual': float(relative),
                        'second_svd_fit_performed': False, 'explained_variance_ratio': float(directional.sum()/total)}
    else:
        assert model['projection'] is None
    raw = raw_matrix(model, design, fit_ix)
    sb = b.astype(raw.dtype).astype(np.float64)
    mean = np.average(raw.astype(np.float64), axis=0, weights=sb)
    variance = np.average((raw.astype(np.float64)-mean)**2, axis=0, weights=sb)
    close(mean, model['scaler'].mean_, 'fit mean', atol=2e-8)
    close(variance, model['scaler'].var_, 'fit variance', atol=2e-8)
    eps = np.finfo(float).eps
    constant = variance <= sb.sum()*eps*variance + (sb.sum()*mean*eps)**2
    scale = np.sqrt(variance); scale[constant] = 1
    close(scale, model['scaler'].scale_, 'fit scale', atol=2e-8)
    if model['kind'] == 'lr':
        c = model['model']
        assert c.C == .01 and c.solver == 'liblinear' and c.penalty == 'l2'
        assert c.random_state == 20260913 and c.max_iter == 2000 and c.n_iter_.max() < 2000
        assert c.classes_.tolist() == [0, 1]
    else:
        assert model['mlp_config'] == config['mlp']
        assert [m['seed'] for m in model['members']] == config['mlp']['seeds']
        assert model['early_stopping'] is False and model['epoch_selection'] == 'fixed final epoch'
        close(model['weight_decay'], 1/(.01*3854))
        for member in model['members']:
            assert member['final_epoch'] == 80 and [h['epoch'] for h in member['history']] == list(range(1, 81))
            for history in member['history']:
                close(history['l2_penalty'], .5*model['weight_decay']*history['parameter_norm_squared_including_bias'])
                close(history['objective'], history['weighted_bce']+history['l2_penalty'])
    return result, loss


def verify_saved(pack, values, ts, wr, ar, name):
    assert [r['window_key'] for r in pack['windows']] == [r['window_key'] for r in wr]
    assert [r['item_id'] for r in pack['items']] == [r['item_id'] for r in ar]
    for wanted, got in zip(pack['windows'], wr):
        for key, val in wanted.items(): assert got[key] == val, (key, wanted['window_key'])
    maximum = u.item_max(pack, values)
    window_error = float(np.max(np.abs(values-np.asarray([w['scores'][name] for w in wr]))))
    assert window_error < 2e-7, (name, window_error)
    assert [bool(s >= ts['window']['threshold']) for s in values] == [w['predictions'][name] for w in wr]
    for wanted, got, value in zip(pack['items'], ar, maximum):
        for key in ('item_id', 'row_id', 'question_id', 'group_id', 'condition', 'category', 'text', 'gold', 'main_eligible', 'reviewed_safe_refusal'):
            assert wanted[key] == got[key], (key, wanted['item_id'])
        if np.isfinite(value):
            close(value, got['scores'][name], name+'.answer', atol=2e-7)
            assert bool(value >= ts['answer']['threshold']) == got['predictions'][name]
        else:
            assert not wanted['main_eligible'] and got['scores'][name] is None and got['predictions'][name] is None
    return window_error


def pooled(pack, windows, answers):
    result = {}
    for method in METHODS:
        wr = [w for w in windows if w['main_eligible']]; ar = [a for a in answers if a['main_eligible']]
        win = u.counts([w['gold'] for w in wr], [w['predictions'][method] for w in wr], unit='windows')
        ans = u.counts([a['gold'] for a in ar], [a['predictions'][method] for a in ar], unit='answers')
        covered = {k for w in windows if w['predictions'][method] for k in w['token_keys']}
        tt = [t for t in pack['tokens'] if t['main_eligible']]
        marked = [t['token_key'] in covered for t in tt]
        high = u.counts([t['gold'] for t in tt], marked)
        high.update(marked_token_fraction=sum(marked)/len(marked), risk_regions=len(pack['regions']),
               risk_regions_any_hit=sum(bool(set(r['token_keys']) & covered) for r in pack['regions']),
               risk_regions_fully_hit=sum(bool(r['token_keys']) and set(r['token_keys']) <= covered for r in pack['regions']))
        by_answer = defaultdict(list)
        for row in wr: by_answer[row['item_ids'][0]].append(row)
        local = []
        for iid, rr in by_answer.items():
            y = np.asarray([r['gold'] for r in rr]); s = np.asarray([r['scores'][method] for r in rr])
            if set(y) != {0, 1}: continue
            local.append(dict(item_id=iid, auroc=float(roc_auc_score(y, s)),
                 average_precision=float(average_precision_score(y, s)),
                 all_peak_windows_risky=bool((y[s == s.max()] == 1).all())))
        safe = [a for a in ar if a['reviewed_safe_refusal']]
        result[method] = {'windows': win, 'answers': ans, 'highlight_tokens': high,
             'conditional_ranking': {'answers_with_both_labels': len(local),
                 'peak_hit_answers': sum(r['all_peak_windows_risky'] for r in local),
                 'mean_auroc': float(np.mean([r['auroc'] for r in local])),
                 'mean_average_precision': float(np.mean([r['average_precision'] for r in local])), 'details': local},
             'safe_refusals': len(safe), 'safe_refusal_false_positives': sum(a['predictions'][method] for a in safe)}
    return result


def bootstrap(rows, config, contrasts, actual):
    groups = sorted({r['group_id'] for r in rows}); assert len(groups) == actual['groups'] == 278
    ix = {g: j for j, g in enumerate(groups)}
    table = np.zeros((len(groups), len(METHODS), 3), np.int64)
    for row in rows:
        if not row['main_eligible']: continue
        for j, name in enumerate(METHODS):
            y, p = row['gold'], row['predictions'][name]; assert p is not None
            table[ix[row['group_id']], j] += [int(y == 1 and p), int(y == 0 and p), int(y == 1 and not p)]
    sampled = np.random.default_rng(config['seed']).integers(0, len(groups), (config['draws'], len(groups)))
    total = table[sampled].sum(1)
    tp, fp, fn = total[:, :, 0], total[:, :, 1], total[:, :, 2]
    denom = 2*tp+fp+fn
    values = np.divide(2*tp, denom, out=np.full(tp.shape, np.nan), where=denom > 0)
    out = {'f1': {}, 'contrasts': {}}
    for j, name in enumerate(METHODS):
        ci = u.interval(values[:, j]); u.assert_metrics(ci, actual['f1'][name], name+'.CI'); out['f1'][name] = ci
    for label, names in contrasts.items():
        assert actual['contrasts'][label]['methods'] == names
        ci = u.interval(values[:, METHODS.index(names[0])]-values[:, METHODS.index(names[1])])
        u.assert_metrics(ci, actual['contrasts'][label], label+'.CI'); out['contrasts'][label] = ci
    return out


def run():
    out = ROOT/'results'
    assert (out/'complete18.json').exists(), 'Wait for completed frozen five-fold output; never restart training'
    complete, snap, config = read(out/'complete18.json'), read(out/'source_snapshot18.json'), read(ROOT/'protocol.json')
    assert complete['original_validation_or_test_used'] is False
    assert tuple(complete['methods']) == tuple(config['methods']) == METHODS
    u.verify_tree(out, complete['files_sha256']); u.verify_tree(Path('.'), snap['code_sha256'])
    af = read(NEW/'data/annotation_freeze.json'); u.verify_tree(NEW, af['files_sha256'])
    assert snap['annotation_freeze_sha256'] == sha(NEW/'data/annotation_freeze.json')
    u.verify_tree(ROOT.parent, read(NEW/'data/legacy_files_snapshot.json')['files_sha256'])
    fm = read(ROOT/'data/feature_manifest.json')
    assert fm['complete'] and fm['completed_count'] == 602 and snap['feature_manifest_sha256'] == sha(ROOT/'data/feature_manifest.json')
    u.verify_tree(Path('.'), fm['extraction_signature']['code_sha256'])
    assert u.digest(fm['extraction_signature']) == fm['extraction_signature_sha256']
    pack, designs, files = rebuild(); assert files == read(out/'feature_files.json')
    expected = {w['window_key']: w for w in pack['windows']}
    candidates = lines(out/'candidate_windows.jsonl')
    assert candidates == pack['windows']
    assignment = read(ROOT/'data/fold_assignment.json')['groups']
    assert set(assignment) == {a['group_id'] for a in pack['items']} and len(assignment) == 278
    summary = read(out/'summary.json'); u.assert_metrics(u.coverage(pack), summary['coverage'], 'coverage')
    audit = {'status': 'passed', 'folds': {}, 'pooled': {}, 'paired_bootstrap': {},
             'original_validation_or_test_parsed': False, 'classifier_or_pca_refitted': False}
    all_windows, all_answers, observed_groups = [], [], []
    gid = np.asarray([w['group_id'] for w in pack['windows']]); main = np.asarray([w['main_eligible'] for w in pack['windows']])
    for fold in range(5):
        stored = pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes())
        detail = read(out/f'fold_{fold}_metrics.json'); assert detail == summary['folds'][str(fold)]
        eg = sorted(g for g, f in assignment.items() if f == fold)
        cg = sorted(g for g, f in assignment.items() if f == (fold+1)%5)
        fg = sorted(set(assignment)-set(eg)-set(cg))
        for key, val in [('fit_groups', fg), ('calibration_groups', cg), ('evaluation_groups', eg)]: assert stored[key] == val
        assert not (set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        tr, tx = subset(pack, fg); ca, cx = subset(pack, cg); ev, ex = subset(pack, eg)
        fit_ix = np.flatnonzero(np.isin(gid, fg)&main); fit_rows = [pack['windows'][j] for j in fit_ix]
        wr, ar = lines(out/f'fold_{fold}_window_scores.jsonl'), lines(out/f'fold_{fold}_answer_scores.jsonl')
        assert all(r['fold'] == fold for r in wr+ar)
        assert tuple(stored['models']) == METHODS
        fold_audit = {}
        for name in METHODS:
            model = stored['models'][name]
            assert model['method'] == name and model['fold'] == fold
            assert model['fit_groups'] == fg and model['calibration_groups'] == cg and model['evaluation_groups'] == eg
            info, loss = inspect_fit(model, designs, fit_ix, fit_rows, config, fold)
            values, members, logits = predict(model, designs)
            close(values[fit_ix], model['final_fit_scores'], name+'.fit score', atol=2e-7)
            ts = model['thresholds']; assert ts == detail[name]['thresholds']
            av = u.item_max(ca, values[cx])
            for unit, rows, scores in [('window', ca['windows'], values[cx]), ('answer', ca['items'], av)]:
                ii = [j for j, r in enumerate(rows) if r['main_eligible']]
                best = u.threshold([rows[j]['gold'] for j in ii], scores[ii])
                u.assert_metrics(best, ts[unit], name+'.cal threshold.'+unit)
            for stage, subset_pack, indices in [('fit', tr, tx), ('calibration', ca, cx), ('evaluation', ev, ex)]:
                got = u.measures(subset_pack, values[indices], ts)[0]
                for unit in got: u.assert_metrics(got[unit], detail[name][stage][unit], f'{fold}.{name}.{stage}.{unit}')
            info['maximum_saved_probability_difference'] = verify_saved(ev, values[ex], ts, wr, ar, name)
            if members is not None:
                descriptions = detail[name]['individual_seeds_descriptive_not_selected']
                assert len(descriptions) == 3
                info['individual_fit_probability_maximum_differences'] = []
                for j, member in enumerate(model['members']):
                    error = float(np.max(np.abs(members[j, fit_ix]-member['final_fit_scores'])))
                    info['individual_fit_probability_maximum_differences'].append(error)
                    # Saved training-only batches and full-cohort replay can use
                    # different float32 GEMM layouts; preserve the measured error.
                    close(members[j, fit_ix], member['final_fit_scores'], name+'.member fit scores', atol=1e-6)
                    y = np.asarray(model['fit_y']); ll = logits[j, fit_ix]
                    bce = float(np.sum(loss*(np.logaddexp(0, ll)-y*ll))/3854)
                    norm = sum(float(v.double().square().sum()) for v in member['state_dict'].values())
                    close(norm, member['history'][-1]['parameter_norm_squared_including_bias'], 'MLP final norm')
                    close(bce, member['history'][-1]['weighted_bce'], 'MLP weighted final BCE', atol=2e-6)
                    assert descriptions[j]['member_index'] == j
                    mts = descriptions[j]['thresholds']; mav = u.item_max(ca, members[j, cx])
                    for unit, rows, scores in [('window', ca['windows'], members[j, cx]), ('answer', ca['items'], mav)]:
                        ii = [k for k, r in enumerate(rows) if r['main_eligible']]
                        best = u.threshold([rows[k]['gold'] for k in ii], scores[ii])
                        u.assert_metrics(best, mts[unit], name+'.member cal.'+unit)
                    for stage, pp, indices in [('fit', tr, tx), ('evaluation', ev, ex)]:
                        got = u.measures(pp, members[j, indices], mts)[0]
                        for unit in got: u.assert_metrics(got[unit], descriptions[j][stage][unit], name+'.member.'+stage+'.'+unit)
                info['three_member_predictions_and_final_objectives_verified'] = True
            fold_audit[name] = info
        audit['folds'][str(fold)] = fold_audit
        all_windows.extend(wr); all_answers.extend(ar); observed_groups.extend(eg)
        print('AUDITED_FOLD', fold, flush=True)
    assert len(observed_groups) == len(set(observed_groups)) == 278
    assert all_windows == lines(out/'window_scores_oof.jsonl') and all_answers == lines(out/'answer_scores_oof.jsonl')
    assert len({w['window_key'] for w in all_windows}) == len(expected) == len(all_windows)
    assert len({a['item_id'] for a in all_answers}) == len(all_answers) == 602
    rebuilt = pooled(pack, all_windows, all_answers)
    for name in METHODS:
        actual = summary['methods'][name]
        for unit in ('windows', 'answers', 'highlight_tokens'): u.assert_metrics(rebuilt[name][unit], actual[unit], name+'.pooled.'+unit)
        for key in ('safe_refusals', 'safe_refusal_false_positives'): assert rebuilt[name][key] == actual[key]
        rr = rebuilt[name]['conditional_ranking']; ar = actual['conditional_ranking']
        for key in ('answers_with_both_labels', 'peak_hit_answers', 'mean_auroc', 'mean_average_precision'): close(rr[key], ar[key], name+'.within.'+key)
        assert {r['item_id']: r for r in rr['details']} == {r['item_id']: r for r in ar['details']}
        for stage in ('fit', 'calibration', 'evaluation'):
            for key in ('f1', 'auroc', 'average_precision'):
                value = np.mean([summary['folds'][str(f)][name][stage]['windows'][key] for f in range(5)])
                close(value, summary['fold_mean'][name][stage][key], name+'.foldmean')
    audit['pooled'] = rebuilt
    for unit, rows in [('windows', all_windows), ('answers', all_answers)]:
        audit['paired_bootstrap'][unit] = bootstrap(rows, config['bootstrap'], config['primary_contrasts'], summary['paired_bootstrap'][unit])
    u.verify_tree(out, complete['files_sha256']); u.verify_tree(Path('.'), snap['code_sha256'])
    audit.update(auditor_sha256=sha(__file__), independent_utility_sha256=sha(UTILITY),
                 complete_sha256=sha(out/'complete18.json'),
                 limits=['No classifier or PCA refitting. PCA checks saved-basis train-only statistics and projection energy.',
                    'All labels remain assistant judgments; consistency is not semantic truth.',
                    'Conditional peak hits only concern known mixed-risk answers, not all-answer detection.',
                    'Group bootstrap conditions on these five fitted folds; overlapping folds and multiple comparisons remain limitations.'])
    write(out/'INDEPENDENT_AUDIT18.json', audit)
    print('INDEPENDENT_AUDIT18_PASSED', flush=True)


def self_test():
    row = dict(row_id='r', question_id='q', group_id='g', split='train', condition='complete', category='test')
    g = {'response': 'a, 12.', 'response_token_ids': [1, 2, 3, 4, 5, 6],
         'response_token_offsets': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
         'items': [dict(item_id='r__1', text='a, 12.', start=0, end=6, parse_ok=True)]}
    a = dict(g['items'][0], original_stance='asserted', original_risk=1, localization_status='resolved',
             risk_spans=[dict(start=3, end=5, text='12')])
    item, tt, ww, rr, _ = u.label_geometry(row, g, a, set())
    assert [t['gold'] for t in tt] == [0, None, None, 1, 1, None]
    assert len(ww) == 3 and all(w['gold'] == 1 for w in ww)
    for region in rr: region.update(group_id='g', row_id='r')
    pp, ix = subset(dict(items=[item], tokens=tt, windows=ww, regions=rr), {'g'})
    assert len(pp['regions']) == 1 and ix.tolist() == [0, 1, 2]
    state = {'network.0.weight': torch.tensor([[1., -1.]]), 'network.0.bias': torch.tensor([.5]),
             'network.3.weight': torch.tensor([[2.]]), 'network.3.bias': torch.tensor([-1.])}
    p, z = forward_mlp(state, np.asarray([[1., 2.], [3., 1.]], np.float32))
    close(z, [-1., 4.]); close(p, expit([-1., 4.]), atol=1e-7)
    print('SYNTHETIC_AUDIT18_PASSED; no real cohort or fitted model opened.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4): self_test() if args.self_test else run()
