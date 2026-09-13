"""Read-only replay of the three completed fold classifiers; never refit."""
from pathlib import Path
import sys
import pickle
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import run_group_crossfit_lb_large as run

with threadpool_limits(limits=4):
    run.check()
    matrix = np.load(run.LB_MATRIX, mmap_mode='r')
    folds = run.q.read(run.OUT / 'folds.json'); identity = run.q.read(run.OUT / 'fit_identity.json')
    windows = run.q.metadata()['windows'][:run.WINDOW_MASS]
    covered = np.zeros(len(windows), int); reports = []
    for f in folds:
        folder = run.OUT / f"fold_{f['fold']}"
        done = run.q.read(folder / 'lb_complete.json')
        for name, digest in done['files_sha256'].items(): assert run.q.sha(folder / name) == digest
        model = pickle.loads((folder / 'lb.pkl').read_bytes())
        fit = np.asarray(f['fit_window_indices']); hold = np.asarray(f['hold_window_indices'])
        assert np.array_equal(model['fit_indices'], fit) and np.array_equal(model['hold_indices'], hold)
        assert not set(f['fit_groups']) & set(f['hold_groups'])
        assert all(windows[j]['group_id'] in f['fit_groups'] for j in fit)
        assert all(windows[j]['group_id'] in f['hold_groups'] for j in hold)
        assert all(identity[j]['group_id'] in f['hold_groups'] for j in f['hold_answer_indices'])
        assert np.array_equal(model['weights']['y'], [windows[j]['label'] for j in fit])
        assert model['C'] == model['model'].C == .0001
        with np.load(folder / 'window_weights.npz') as z:
            assert set(z.files) == set(model['weights'])
            assert all(np.array_equal(z[k], model['weights'][k]) for k in z.files)
        with np.load(folder / 'lb_hold_scores.npz') as z:
            assert np.array_equal(hold, z['window_indices']); stored = z['scores'].copy()
        same_shape = model['scaler'].transform(np.asarray(matrix[hold])).astype(np.float32)
        whole_replay = model['model'].predict_proba(same_shape)[:, 1]
        assert np.array_equal(whole_replay, stored)
        errors = []; chunk_errors = []
        for lo in range(0, len(hold), 8192):
            ix = hold[lo:lo + 8192]; raw = np.asarray(matrix[ix])
            transformed = model['scaler'].transform(raw).astype(np.float32)
            manual = raw.copy(); manual -= model['scaler'].mean_; manual /= model['scaler'].scale_
            assert np.array_equal(transformed, manual)
            replay = model['model'].predict_proba(transformed)[:, 1]
            chunk_errors.append(float(np.max(np.abs(replay - stored[lo:lo + len(ix)]))))
            coefficient = expit(manual @ model['model'].coef_[0] + model['model'].intercept_[0])
            errors.append(float(np.max(np.abs(coefficient - replay))))
        assert max(errors) < 1e-12 and np.isfinite(stored).all()
        covered[hold] += 1
        reports.append(dict(fold=f['fold'], fit_windows=len(fit), hold_windows=len(hold),
            scaler_transform_exact=True, model_probability_replay_exact=True,
            replay_same_whole_fold_matrix_shape=True, alternate_8192_batch_max_difference=max(chunk_errors),
            max_manual_coefficient_error=max(errors), iterations=model['model'].n_iter_.tolist(),
            model_sha256=run.q.sha(folder / 'lb.pkl'), scores_sha256=run.q.sha(folder / 'lb_hold_scores.npz')))
    assert (covered == 1).all() and covered.sum() == 168123
    assert not run.torch.cuda.is_initialized()
    run.q.save(run.OUT / 'LB_REPLAY.json', dict(passed=True, models=reports,
        native_fit_windows_covered_exactly_once=168123, official_test_opened=False, GPU_used=False, additional_fits=0))
    print('THREE_LB_REPLAYS_EXACT', flush=True)
