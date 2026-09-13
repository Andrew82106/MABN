"""Fixed type-balanced positive-loss diagnostic; binary gold stays unchanged."""
from pathlib import Path
from collections import defaultdict
import argparse
import pickle
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
import run_development as q
import run_probe_expansion as base

OUT = HERE / 'type_balanced_v1'
TYPE = {'Evident Baseless Info': 0, 'Subtle Baseless Info': 0,
        'Evident Conflict': 1, 'Subtle Conflict': 1}


def protocol():
    return {'version': 'expanded-frozen-probe-type-balanced-loss-v1',
        'data': 'Same3680fit/159cal, source615/154, unchanged original human labels and4rawBPE windows; no test',
        'features': 'Exactly existing expanded frozen MiniCheck PCA64 and optional claim risk logit; old PCA and expanded fit-only scalers unchanged',
        'classes_for_weights_only': ['negative', 'baseless', 'conflict'],
        'target_base_class_mass': [.5, .25, .25],
        'mixed_positive_window': 'Half type mass to each when both baseless/conflict overlap actual lexical risk positions; one unchanged binary target1',
        'weights': 'Original expanded source/native-aux base weights; fit-only type mass inverse weighting, then reequalize each615sourcegroup; total loss mass168123. No type enters model input or cal selection.',
        'methods': ['hidden64', 'hidden64_risk'], 'C': [1e-5, 1e-4, .001], 'new_fits': 6,
        'classifier': 'Same liblinear L2/maxiter2000/seed20260924/4CPU threads',
        'selection': 'Same original q.selection_key and separate cal risk-F1 thresholds as binary-loss control',
        'comparison': 'Old6 expanded candidates retained as binary-balanced controls; change loss type weighting only, not data/scaler/PCA/model/metric',
        'scope_limit': 'Extra frozen semantic checker, not native generator whitebox. Development calibration repeatedly used; no independent test claim.',
        'code_sha256': q.sha(__file__), 'source_complete_sha256': q.sha(base.OUT/'complete.json')}


def prepare():
    assert not (OUT/'prepare_started.json').exists()
    _, meta = base.metadata()
    OUT.mkdir(exist_ok=True)
    q.save(OUT/'prepare_started.json', {'utc': time.time(), 'protocol': protocol()})
    for name, digest in q.read(base.OUT/'complete.json')['files_sha256'].items():
        assert q.sha(base.OUT/name) == digest, name
    with np.load(base.OUT/'expanded3680_weights.npz') as z:
        b, original_loss, y = z['base'].copy(), z['loss'].copy(), z['y'].copy()
    token_types = {}
    counts = defaultdict(int)
    for a, t in zip(meta['answers'][:3680], meta['tokens'][:3680]):
        positive = np.zeros((t['token_count'], 2), bool)
        for span, span_map in zip(t['original_labels'], t['span_token_mapping']):
            positive[span_map['risk_token_indices'], TYPE[span['label_type']]] = True
            counts[span['label_type']] += 1
        assert np.array_equal(positive.any(1), np.asarray(t['risk_mask'], bool))
        token_types[a['response_id']] = positive
    mix = np.zeros((653979, 3), np.float64)
    groups = defaultdict(list)
    for i, w in enumerate(meta['windows'][:653979]):
        types = token_types[w['response_id']][w['token_indices']].any(0)
        assert bool(types.any()) == bool(y[i]) == bool(w['label'])
        if types.any(): mix[i, 1:] = types / types.sum()
        else: mix[i, 0] = 1
        groups[w['group_id']].append(i)
    type_mass = (b[:, None]*mix).sum(0)
    factors = np.asarray([.5, .25, .25])*b.sum()/type_mass
    loss = b*(mix@factors)
    for ix in groups.values(): loss[ix] *= (168123/615)/loss[ix].sum()
    assert (loss > 0).all() and abs(loss.sum()-168123) < 1e-6
    assert all(abs(loss[ix].sum()-168123/615) < 1e-8 for ix in groups.values())
    np.savez_compressed(OUT/'weights.npz', base=b, loss=loss, y=y, mix=mix, type_factors=factors)
    q.save(OUT/'LABEL_WEIGHT_CHECK.json', {'passed': True, 'original_human_span_counts': dict(counts),
        'unchanged_binary_labels': True, 'fit_only_type_mass': type_mass.tolist(), 'type_factors': factors.tolist(),
        'original_loss_type_mass': (original_loss[:, None]*mix).sum(0).tolist(),
        'new_loss_type_mass_after_group_renormalization': (loss[:, None]*mix).sum(0).tolist(),
        'type_mass_after_group_renorm_need_not_equal_target': True, 'loss_total': float(loss.sum()),
        'test_opened': False})
    q.save(OUT/'protocol.json', protocol())
    q.save(OUT/'preparation_complete.json', {'weights_sha256': q.sha(OUT/'weights.npz'),
        'protocol_sha256': q.sha(OUT/'protocol.json'), 'original_preparation_sha256': q.sha(base.OUT/'preparation_complete.json'),
        'test_opened': False})
    print('TYPE_BALANCED_PREPARED', type_mass.tolist(), factors.tolist(), flush=True)


def fit():
    assert not (OUT/'fit_started.json').exists()
    assert q.read(OUT/'protocol.json') == protocol()
    prep = q.read(OUT/'preparation_complete.json')
    assert q.sha(OUT/'weights.npz') == prep['weights_sha256']
    assert q.sha(base.OUT/'preparation_complete.json') == prep['original_preparation_sha256']
    for n, h in q.read(base.OUT/'preparation_complete.json')['files_sha256'].items(): assert q.sha(base.OUT/n) == h
    _, meta = base.metadata()
    with np.load(OUT/'weights.npz') as z: loss, y = z['loss'].copy(), z['y'].copy()
    raw = np.load(base.OUT/'matrices/window65.npy', mmap_mode='r')
    q.save(OUT/'fit_started.json', {'utc': time.time(), 'preparation_sha256': q.sha(OUT/'preparation_complete.json')})
    results = {}
    for method in protocol()['methods']:
        width = 64 if method == 'hidden64' else 65
        template = pickle.loads((base.OUT/f'expanded3680_{method}_C1e-05.pkl').read_bytes())
        scaler = template['scaler']
        x = np.empty((len(raw), width), np.float32)
        for lo in range(0, len(x), q.BATCH):
            hi = min(lo+q.BATCH, len(x)); x[lo:hi] = scaler.transform(raw[lo:hi, :width]).astype(np.float32)
        entries = []
        for c in protocol()['C']:
            name = f'{method}_C{c:g}'
            tick = time.perf_counter()
            model = LogisticRegression(C=c, solver='liblinear', max_iter=2000, random_state=20260924).fit(x[:653979], y, sample_weight=loss)
            assert model.n_iter_.max() < 2000
            scores = model.predict_proba(x)[:, 1]
            answer_scores = q.answer_scores(meta, scores)
            ts = {'window': q.choose_threshold([w['label'] for w in meta['windows'][653979:]], scores[653979:]),
                  'answer': q.choose_threshold([a['label'] for a in meta['answers'][3680:]], answer_scores[3680:])}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps({'model': model, 'scaler': scaler, 'C': c, 'thresholds': ts,
                                                         'fit_only': True, 'weight_sha256': q.sha(OUT/'weights.npz')}, protocol=5))
            np.savez_compressed(OUT/(name+'_scores.npz'), window_scores=scores, answer_scores=answer_scores)
            entry = {'candidate': name, 'C': c, 'selection_key': list(q.selection_key(ts, c)), 'thresholds': ts,
                     'metrics': q.metrics(meta, scores, ts), 'seconds': time.perf_counter()-tick,
                     'files_sha256': {ext: q.sha(OUT/(name+ext)) for ext in ['.pkl', '_scores.npz']}}
            entries.append(entry); q.save(OUT/(name+'_result.json'), entry)
            print('TYPE_BALANCED_FIT', name, entry['metrics']['calibration']['windows']['f1'], entry['metrics']['calibration']['answers']['f1'], flush=True)
        results[method] = {'selected': max(entries, key=lambda r: r['selection_key']), 'all_candidates': entries}
    q.save(OUT/'summary.json', {'results': results, 'new_fits': 6, 'official_test_opened': False})
    files = ['summary.json', 'protocol.json', 'preparation_complete.json', 'weights.npz', 'LABEL_WEIGHT_CHECK.json']
    files += [n.name for n in OUT.glob('*_result.json')]+[n.name for n in OUT.glob('*.pkl')]+[n.name for n in OUT.glob('*_scores.npz')]
    q.save(OUT/'complete.json', {'files_sha256': {f: q.sha(OUT/f) for f in files}, 'official_test_opened': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'fit']); args = p.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'fit': fit}[args.stage]()
