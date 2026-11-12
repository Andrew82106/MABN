"""Source-group out-of-fold token fusion; no training on branch in-sample scores."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import readl, evaluate_partition, pool_windows
import evidence_features

FEATURES = ['semantic_logit', 'whitebox_logit', 'logit_product', 'logit_absolute_difference',
            'semantic_mean3', 'semantic_mean9', 'whitebox_mean3', 'whitebox_mean9',
            'nll', 'entropy', 'selected_vs_best_logp']
CS = [.001, .01, .1, 1.]
ALL_FEATURES = FEATURES + evidence_features.NAMES


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def predictions(path):
    records = readl(path)
    assert len(records) == len({r['response_id'] for r in records}), path
    return {r['response_id']: np.asarray(r['token_scores'], np.float64) for r in records}


def centered_mean(p, width):
    sums = np.r_[0., np.cumsum(p)]
    i = np.arange(len(p)); lo, hi = np.maximum(i - width // 2, 0), np.minimum(i + width // 2 + 1, len(p))
    return (sums[hi] - sums[lo]) / (hi - lo)


def feature_matrix(semantic, whitebox, nll, entropy, margin):
    a, b = np.asarray(semantic), np.asarray(whitebox)
    assert a.shape == b.shape and a.ndim == 1 and len(a)
    assert np.isfinite(a).all() and np.isfinite(b).all() and ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).all()
    aa, bb = np.clip(a, 1e-5, 1 - 1e-5), np.clip(b, 1e-5, 1 - 1e-5)
    la, lb = np.log(aa / (1 - aa)), np.log(bb / (1 - bb))
    x = np.column_stack([la, lb, la * lb, np.abs(la - lb), centered_mean(a, 3), centered_mean(a, 9),
                         centered_mean(b, 3), centered_mean(b, 9), nll, entropy, margin])
    assert x.shape == (len(a), len(FEATURES)) and np.isfinite(x).all()
    return x


def matrices(semantic, whitebox, granularity='token'):
    assert semantic.keys() == whitebox.keys()
    members = {r['response_id']: r['split'] for r in readl(HERE / 'data/split_assignments.jsonl')}
    requested = {members[rid] for rid in semantic}
    inputs = {r['response_id']: r for split in requested for r in readl(HERE / 'data' / split / 'inputs.jsonl')}
    output = {}
    for rid in sorted(semantic):
        with np.load(HERE / 'signals' / (rid + '.npz'), allow_pickle=False) as z:
            n = len(z['token_ids'])
            assert semantic[rid].shape == whitebox[rid].shape == (n,), rid
            assert len(z['response_token_offsets']) == len(z['lexical_mask']) == n, rid
            context = evidence_features.features(inputs[rid]['original_response'], inputs[rid]['retrieved_passages'],
                                                  z['response_token_offsets'])
            if granularity == 'window':
                ids = window_indices(z['lexical_mask'])
                sa = np.array([semantic[rid][v].max() for v in ids])
                sb = np.array([whitebox[rid][v].max() for v in ids])
                nll = np.array([z['nll'][v].mean() for v in ids])
                entropy = np.array([z['entropy'][v].mean() for v in ids])
                margin = np.array([z['selected_vs_best_logp'][v].min() for v in ids])
                context = np.stack([np.r_[context[v, :3].mean(0), context[v, 3:].max(0)] for v in ids])
                base = feature_matrix(sa, sb, nll, entropy, margin)
            else:
                base = feature_matrix(semantic[rid], whitebox[rid], z['nll'], z['entropy'], z['selected_vs_best_logp'])
            output[rid] = np.column_stack([base, context])
    return output


def window_indices(lexical):
    lexical = np.asarray(lexical, bool)
    n = len(lexical)
    return [np.arange(i, min(i + 4, n))[lexical[i:min(i + 4, n)]]
            for i in range(max(1, n - 3)) if lexical[i:min(i + 4, n)].any()]


def assemble_oof(config, family):
    scores = {}; receipts = []
    assert len(config[family]['folds']) == 3
    for fold, entry in enumerate(config[family]['folds']):
        held_path = HERE / f'fusion/folds/fold_{fold}_held_ids.json'
        fit_path = HERE / f'fusion/folds/fold_{fold}_fit_ids.json'
        held, fitted = set(json.loads(held_path.read_text())), set(json.loads(fit_path.read_text()))
        actual_fit_path = Path(entry['training_ids'])
        actual_fit = json.loads(actual_fit_path.read_text(encoding='utf-8'))
        if isinstance(actual_fit, dict): actual_fit = actual_fit.get('response_ids', actual_fit.get('fit_response_ids', actual_fit.get('fit_ids')))
        assert actual_fit is not None and set(actual_fit) == fitted, (family, fold, 'training IDs')
        part = predictions(entry['predictions'])
        assert set(part) == held and set(part).isdisjoint(fitted) and scores.keys().isdisjoint(part), (family, fold)
        manifest_path = Path(entry['manifest'])
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if family == 'semantic':
            assert manifest['held_groups_excluded_from_training'] and not manifest['held_labels_used']
            for key in ['fit_ids', 'held_ids', 'prepared', 'fit_inputs', 'held_inputs', 'checkpoint', 'predictions']:
                assert sha(manifest[key]['path']) == manifest[key]['sha256'], (family, fold, key)
            assert Path(manifest['predictions']['path']).resolve() == Path(entry['predictions']).resolve()
            assert set(json.loads(Path(manifest['fit_ids']['path']).read_text())) == fitted
        else:
            folder = manifest_path.parent
            assert manifest['training_member_count'] == 0 and manifest['all_predictions_out_of_fit'] and manifest['fixed_final']
            assert manifest['prediction_sha256'] == sha(entry['predictions'])
            assert manifest['fit_ids_file_sha256'] == sha(actual_fit_path)
            assert manifest['checkpoint_sha256'] == sha(folder / 'best.pt')
            assert manifest['training_run_sha256'] == sha(folder / 'RUN.json')
            assert set(json.loads((folder / 'RUN.json').read_text())['fit_ids']) == fitted
        scores.update(part)
        receipts.append({'fold': fold, 'predictions_sha256': sha(entry['predictions']),
                         'training_ids_sha256': sha(actual_fit_path), 'held_ids_sha256': sha(held_path),
                         'prediction_manifest_sha256': sha(manifest_path)})
    return scores, receipts


def write_predictions(path, scores, granularity='token'):
    with Path(path).open('w', encoding='utf-8') as f:
        for rid, p in sorted(scores.items()):
            f.write(json.dumps({'response_id': rid, granularity + '_scores': np.asarray(p).tolist()}, allow_nan=False) + '\n')


def train(args):
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'SELECTED.json').exists(), 'Use a fresh output directory'
    config = json.loads(args.config.read_text(encoding='utf-8'))
    sa, ra = assemble_oof(config, 'semantic')
    sb, rb = assemble_oof(config, 'whitebox')
    sequences = {r['response_id']: r for r in readl(HERE / 'data/fit/sequences.jsonl')}
    answers = readl(HERE / 'data/fit/answers.jsonl')
    assert sa.keys() == sb.keys() == sequences.keys()
    va, vb = predictions(config['semantic']['validation']), predictions(config['whitebox']['validation'])
    fit_features, val_features = matrices(sa, sb, args.granularity), matrices(va, vb, args.granularity)
    counts = Counter(a['group_id'] for a in answers)
    matrices_x, targets, weights = [], [], []
    for a in answers:
        r, rid = sequences[a['response_id']], a['response_id']
        lexical, y = np.asarray(r['lexical_mask'], bool), np.asarray(r['risk_mask'], bool)
        if args.granularity == 'window':
            y = np.array([y[v].any() for v in window_indices(lexical)])
            lexical = np.ones(len(y), bool)
        assert len(lexical) == len(fit_features[rid])
        matrices_x.append(fit_features[rid][lexical]); targets.extend(y[lexical])
        weights.extend(np.full(lexical.sum(), 1 / (counts[a['group_id']] * lexical.sum())))
    x, y, w = np.concatenate(matrices_x), np.asarray(targets), np.asarray(weights)
    assert abs(w.sum() - len(counts)) < 1e-8
    scaler = StandardScaler().fit(x, sample_weight=w)
    xx = scaler.transform(x); vv = {rid: scaler.transform(v) for rid, v in val_features.items()}
    candidates = []; best = None
    # Fixed average is included as a simple, genuinely label-free fusion control.
    average = {rid: (va[rid] + vb[rid]) * .5 for rid in va}
    if args.granularity == 'window':
        validation_geometry = {r['response_id']: r for r in readl(HERE / 'data/validation/geometry.jsonl')}
        average = {rid: (pool_windows(va[rid], validation_geometry[rid]['lexical_mask']) +
                         pool_windows(vb[rid], validation_geometry[rid]['lexical_mask'])) * .5 for rid in va}
    is_window = args.granularity == 'window'
    report = evaluate_partition('validation', average, scores_are_windows=is_window)
    candidates.append({'name': 'fixed_average', 'report': report})
    write_predictions(args.out / 'fixed_average_validation.jsonl', average, args.granularity)
    save(args.out / 'fixed_average.json', report)
    for width, family in [(len(FEATURES), 'interaction'), (len(ALL_FEATURES), 'evidence_interaction')]:
      for c in CS:
        for balance in [False, True]:
            fit_weights = w.copy()
            if balance:
                mass = [w[~y].sum(), w[y].sum()]
                assert min(mass) > 0
                fit_weights *= np.where(y, w.sum() / (2 * mass[1]), w.sum() / (2 * mass[0]))
            model = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=20260928)
            model.fit(xx[:, :width], y, sample_weight=fit_weights)
            if int(model.n_iter_.max()) >= model.max_iter:
                raise RuntimeError('Fusion logistic regression did not converge')
            scores = {rid: model.predict_proba(v[:, :width])[:, 1] for rid, v in vv.items()}
            report = evaluate_partition('validation', scores, scores_are_windows=is_window)
            name = f'{family}_C{c:g}_balanced{int(balance)}'
            entry = {'name': name, 'C': c, 'balanced': balance, 'feature_count': width, 'report': report}
            candidates.append(entry)
            save(args.out / (name + '.json'), entry)
            key = report['selection_key'] + [-width, -c, -int(balance)]
            if best is None or key > best['selection_key']:
                best = dict(entry, selection_key=key)
                with (args.out / 'best.pkl').open('wb') as f:
                    pickle.dump({'model': model, 'scaler': scaler, 'features': ALL_FEATURES, 'width': width,
                                 'granularity': args.granularity}, f)
                write_predictions(args.out / 'best_validation_predictions.jsonl', scores, args.granularity)
            print(json.dumps({'name': name, 'answer_f1': report['cohorts']['no_hint']['answer']['f1'],
                              'window_f1': report['cohorts']['no_hint']['window']['f1']}), flush=True)
    best.update({'thresholds': best['report']['thresholds'], 'feature_names': ALL_FEATURES[:best['feature_count']], 'oof_receipts': {'semantic': ra, 'whitebox': rb},
                 'config_sha256': sha(args.config), 'model_sha256': sha(args.out / 'best.pkl'),
                 'training_answers': len(answers), 'training_groups': len(counts),
                 'granularity': args.granularity, 'test_labels_read': False})
    save(args.out / 'SELECTED.json', best)
    save(args.out / 'HISTORY.json', candidates)


def predict(args):
    assert args.semantic and args.whitebox and args.destination
    a, b = predictions(args.semantic), predictions(args.whitebox)
    with (args.out / 'best.pkl').open('rb') as f: state = pickle.load(f)
    assert state['features'] == ALL_FEATURES
    x = matrices(a, b, state['granularity'])
    result = {rid: state['model'].predict_proba(state['scaler'].transform(v)[:, :state['width']])[:, 1] for rid, v in x.items()}
    write_predictions(args.destination, result, state['granularity'])
    save(args.destination.with_suffix('.receipt.json'), {'responses': len(result), 'model_sha256': sha(args.out / 'best.pkl'),
         'semantic_sha256': sha(args.semantic), 'whitebox_sha256': sha(args.whitebox),
         'predictions_sha256': sha(args.destination), 'granularity': state['granularity'], 'test_labels_read': False})


def selfcheck():
    x = np.array([.1, .2, .9, .5])
    assert np.allclose(centered_mean(x, 3), [.15, .4, 1.6 / 3, .7])
    assert np.allclose(centered_mean(x, 9), x.mean())
    f = feature_matrix(x, 1 - x, np.ones(4), np.ones(4), -np.ones(4))
    assert np.isfinite(f).all() and f.shape == (4, len(FEATURES))
    w = window_indices([0, 1, 0, 0, 1, 0])
    assert [v.tolist() for v in w] == [[1], [1, 4], [4]]
    print('FUSION_CPU_CHECK_PASSED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selfcheck', 'train', 'predict'])
    p.add_argument('--config', type=Path)
    p.add_argument('--out', type=Path, default=HERE / 'fusion/interaction_v1')
    p.add_argument('--semantic', type=Path); p.add_argument('--whitebox', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--granularity', choices=['token', 'window'], default='window')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        {'selfcheck': lambda a: selfcheck(), 'train': train, 'predict': predict}[args.stage](args)
