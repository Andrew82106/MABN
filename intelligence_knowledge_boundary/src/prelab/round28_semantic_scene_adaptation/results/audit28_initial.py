"""One CPU replay of frozen fits, calibration choices and OOF counts; no refit."""
from pathlib import Path
import sys
import pickle
import importlib.util
import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / 'src'))
import run28 as run
q = run.q
helper = run.cache.QA / 'results/minicheck_tail_all_docs_v3/postrun_audit.py'
spec = importlib.util.spec_from_file_location('independent_metrics', helper)
ind = importlib.util.module_from_spec(spec); spec.loader.exec_module(ind)


def main():
    done = q.read(OUT / 'complete.json'); assert done['status'] == 'complete_development_only'
    for n, value in done['files_sha256'].items():
        assert q.sha(OUT / n) == value
    frozen = q.read(OUT / 'fit_complete.json')
    for n, value in frozen['files_sha256'].items():
        assert q.sha(OUT / n) == value
    windows, answers, folds, aw = run.metadata()
    wy = np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows])
    ay = np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers])
    wp = {m: np.zeros(len(windows), bool) for m in run.METHODS}
    ap = {m: np.zeros(len(answers), bool) for m in run.METHODS}
    wc, ac = np.zeros(len(windows), int), np.zeros(len(answers), int)
    reports = []
    with np.load(OUT / 'designs.npz') as z:
        hidden, logit = z['hidden'].copy(), z['logit'].copy()
    with threadpool_limits(limits=4):
        for fold, groups in enumerate(folds):
            f = pickle.loads((OUT / f'fold_{fold}_frozen.pkl').read_bytes())
            assert f['groups'] == groups
            ix = np.asarray([i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['fit_groups']])
            pc, model = f['PCA'], f['model']
            assert np.array_equal(ix, model['fit_ix']) and np.array_equal(ix, pc['fit_ix'])
            assert np.array_equal(model['fit_y'], wy[ix])
            assert model['fit_keys'] == pc['fit_keys'] == [windows[i]['window_key'] for i in ix]
            assert not set(model['fit_groups']) & (set(groups['calibration_groups']) | set(groups['evaluation_groups']))
            mean = np.average(hidden[ix].astype(np.float64), axis=0, weights=pc['normalized_base_weights'])
            assert np.allclose(mean, pc['mean'], rtol=0, atol=1e-12)
            assert np.allclose(pc['components'] @ pc['components'].T, np.eye(32), rtol=0, atol=1e-12)
            projected = ((hidden.astype(np.float64)-pc['mean']) @ pc['components'].T).astype(np.float32)
            design = np.column_stack((projected, logit)).astype(np.float32)
            sm = np.average(design[ix].astype(np.float64), axis=0, weights=model['base_weights'])
            sv = np.average((design[ix]-sm)**2, axis=0, weights=model['base_weights'])
            assert np.allclose(sm, model['scaler'].mean_, rtol=0, atol=1e-10)
            assert np.allclose(sv, model['scaler'].var_, rtol=1e-9, atol=1e-10)
            mass = {}
            for i, weight in zip(ix, model['loss_weights']):
                gid = windows[i]['group_id']; mass[gid] = mass.get(gid, 0.) + float(weight)
            assert abs(sum(mass.values())-3854) < 1e-8
            assert max(abs(v-3854/len(mass)) for v in mass.values()) < 1e-8
            assert model['C'] == .01 and model['width'] == 33 and int(max(model['model'].n_iter_)) < 2000
            semantic = model['model'].predict_proba(model['scaler'].transform(design).astype(np.float32))[:, 1]
            ci = np.asarray([i for i, w in enumerate(windows) if w['main_eligible'] and w['group_id'] in groups['calibration_groups']])
            cai = np.asarray([i for i, a in enumerate(answers) if a['main_eligible'] and a['group_id'] in groups['calibration_groups']])
            ei = np.asarray([i for i, w in enumerate(windows) if w['group_id'] in groups['evaluation_groups']])
            eai = np.asarray([i for i, a in enumerate(answers) if a['group_id'] in groups['evaluation_groups']])
            wc[ei] += 1; ac[eai] += 1
            with np.load(OUT / f'fold_{fold}_scores.npz') as scores:
                delta = float(np.abs(semantic-scores['semantic_probe']).max())
                assert delta <= 1e-12
                for m in run.METHODS:
                    v = scores[m]; av = np.asarray([max(v[aw[a['item_id']]]) for a in answers])
                    ts = f['thresholds'][m]
                    assert ind.threshold_from_roc(wy[ci], v[ci]) == ts['window']['threshold']
                    assert ind.threshold_from_roc(ay[cai], av[cai]) == ts['answer']['threshold']
                    wp[m][ei] = v[ei] >= ts['window']['threshold']
                    ap[m][eai] = av[eai] >= ts['answer']['threshold']
                for prior, fused in zip(run.BASES, run.FUSED):
                    table = f['alpha_tables'][fused]; keys = []
                    for j, alpha in enumerate(run.ALPHA):
                        v = scores[fused+'_all_alpha'][j]
                        assert np.array_equal(v, run.combine(scores[prior], semantic, alpha))
                        av = np.asarray([max(v[aw[a['item_id']]]) for a in answers])
                        wt, at = ind.threshold_from_roc(wy[ci], v[ci]), ind.threshold_from_roc(ay[cai], av[cai])
                        assert wt == table[j]['thresholds']['window']['threshold']
                        assert at == table[j]['thresholds']['answer']['threshold']
                        wm, am = ind.metrics(wy[ci], v[ci], wt), ind.metrics(ay[cai], av[cai], at)
                        key = [min(wm['f1'], am['f1']), wm['f1'], wm['precision'], -alpha]
                        assert key == table[j]['key']; keys.append(key)
                    chosen = max(range(6), key=lambda j: keys[j])
                    assert f['selected_alpha'][fused] == table[chosen]
                    assert np.array_equal(scores[fused], scores[fused+'_all_alpha'][chosen])
            reports.append({'fold': fold, 'assigned_fit_groups': len(groups['fit_groups']), 'active_loss_groups': len(mass),
                'fit_windows': len(ix), 'semantic_probability_replay_max_abs': delta,
                'PCA32_explained_variance_ratio': pc['explained_variance_ratio_sum'],
                'LR_iterations': int(max(model['model'].n_iter_)),
                'selected_alpha': {k: v['alpha'] for k, v in f['selected_alpha'].items()}})
    assert (wc == 1).all() and (ac == 1).all()
    summary = q.read(OUT / 'summary.json')['methods']
    for m in run.METHODS:
        assert run.cache.transfer.scene.counts(wy[wy>=0], wp[m][wy>=0]) == summary[m]['windows']
        assert run.cache.transfer.scene.counts(ay[ay>=0], ap[m][ay>=0]) == summary[m]['answers']
    q.save(OUT / 'AUDIT.json', {'passed': True, 'folds': reports,
        'no_refit_no_GPU_no_new_threshold_or_model_selection': True, 'independent_ROC_threshold_reconstruction': True,
        'all60_alpha_candidates_and_single_alpha_selections_checked': True,
        'fit_only_PCA_scaler_labels_weights_and_model_replay_checked': True,
        'each_outer_window_and_answer_scored_once': True, 'original_validation_test_read_this_run': False,
        'human_gold': False, 'complete_sha256': q.sha(OUT / 'complete.json'), 'audit_code_sha256': q.sha(Path(__file__))})
    print('R28_INDEPENDENT_FROZEN_REPLAY_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
