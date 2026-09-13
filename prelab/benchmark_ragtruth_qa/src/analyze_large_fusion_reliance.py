"""Read-only diagnostics of already fitted fusion, not causal feature importance."""
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_large_matched_combination_v2 as r


def run():
    out = r.q.ROOT / 'results/large_fusion_reliance_v1'
    assert not (out / 'summary.json').exists()
    completed = r.q.read(r.OUT / 'complete.json')
    assert r.q.sha(r.OUT / 'summary.json') == completed['summary_sha256']
    summary = r.q.read(r.OUT / 'summary.json')
    models = []
    for family, entry in summary['selected'].items():
        path = r.OUT / (entry['candidate'] + '.pkl')
        assert r.q.sha(path) == entry['model_sha256']
        saved = pickle.loads(path.read_bytes())
        model = saved['model']
        record = {'family': family, 'model_sha256': r.q.sha(path),
                  'fit': entry['metrics']['fit'], 'calibration': entry['metrics']['calibration']}
        if saved['scaler'] is not None:
            record['standardized_coefficients'] = model.coef_.tolist()
            record['column_order'] = ['old_peer', 'tail2'] + [f'old_citation_{i}' for i in range(8)] + ['large']
        else:
            counts = np.zeros(3, dtype=int)
            for step in model._predictors:
                for tree in step:
                    nodes = tree.nodes
                    for column in nodes['feature_idx'][~nodes['is_leaf'].astype(bool)]:
                        counts[column] += 1
            record['split_counts_not_importance'] = dict(zip(['old_peer', 'tail2', 'large'], counts.tolist()))
        models.append(record)
    meta = r.q.metadata()
    large = np.load(r.OUT / 'large_window_probability.npy')
    assert len(large) == len(meta['windows']) == 210364
    distributions = {}
    for part, (lo, hi) in meta['bounds'].items():
        y = np.asarray([w['label'] for w in meta['windows'][lo:hi]], bool)
        p = large[lo:hi]
        distributions[part] = {'rows': len(y), 'positive': int(y.sum()),
            'large_auroc': float(roc_auc_score(y, p)),
            'by_label': {str(label): {'n': int((y == label).sum()),
                'mean_probability': float(p[y == label].mean()),
                'quantiles_10_50_90': np.quantile(p[y == label], [.1, .5, .9]).tolist()}
                for label in [0, 1]}}
    r.q.save(out / 'summary.json', {'models': models, 'large_probability_distribution': distributions,
        'source_sha256': r.q.sha(Path(__file__)), 'upstream_complete_sha256': r.q.sha(r.OUT / 'complete.json'),
        'no_new_fit_or_prediction_rule': True, 'official_test_opened': False, 'GPU_used': False,
        'limits': 'Coefficients are conditional on correlated standardized inputs, not causal importance. Tree split counts are descriptive, not permutation importance. Fit upstream predictions are in-sample; this association does not establish that cross-fitting would improve calibration F1.'})
    print('RELIANCE_DIAGNOSIS_SAVED', distributions, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
