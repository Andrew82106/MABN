"""Fixed selected-candidate type recall, without new tuning or model inference."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import run_development as q
from diagnose_completed_tail import KINDS


def run():
    source = q.ROOT / 'results/large_fixed_convex_v1'
    out = q.ROOT / 'results/selected_convex_error_diagnosis_v1'
    assert not (out / 'summary.json').exists()
    completed = q.read(source / 'complete.json')
    assert q.sha(source / 'summary.json') == completed['summary_sha256']
    assert (source / 'INDEPENDENT_AUDIT.json').exists()
    selected = max(q.read(source / 'summary.json')['selected'].values(), key=lambda e: e['selection_key'])
    path = source / (selected['candidate'] + '_scores.npz')
    assert q.sha(path) == selected['scores_sha256']
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    windows = meta['windows'][lo:hi]
    with np.load(path, allow_pickle=False) as z:
        assert q.metrics(meta, z['window_scores'], selected['thresholds']) == selected['metrics']
        assert np.array_equal(q.answer_scores(meta, z['window_scores']), z['answer_scores'])
        scores = z['window_scores'][lo:hi]
    y = np.asarray([w['label'] for w in windows], bool)
    pred = scores >= selected['thresholds']['window']['threshold']
    typed = {k: defaultdict(set) for k in KINDS}
    for token in meta['tokens']:
        if token['partition'] != 'calibration':
            continue
        for span in token['span_token_mapping']:
            kind = token['original_labels'][span['span_index']]['label_type']
            typed[kind][token['response_id']].update(span['risk_token_indices'])
    masks = {k: np.asarray([bool(set(w['token_indices']) & typed[k][w['response_id']]) for w in windows]) for k in KINDS}
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), y)
    types = {k: {'risk_windows': int(m.sum()), 'detected': int(pred[m].sum()),
                 'missed': int((~pred[m]).sum()), 'recall': float(pred[m].mean())} for k, m in masks.items()}
    clean_ids = {a['response_id'] for a in meta['answers'][634:] if a['label'] == 0}
    clean = np.asarray([w['response_id'] in clean_ids for w in windows])
    clean_fp = int((pred & ~y & clean).sum())
    other_fp = int((pred & ~y & ~clean).sum())
    assert clean_fp + other_fp == selected['metrics']['calibration']['windows']['fp']
    report = {'candidate': selected['candidate'], 'metrics': selected['metrics']['calibration'],
        'thresholds': selected['thresholds'], 'type_recall_only': types,
        'false_positive_windows_clean_answers': clean_fp, 'false_positive_windows_risky_answers': other_fp,
        'type_masks_can_overlap': True, 'no_model_fits_or_new_thresholds': True,
        'official_test_opened': False, 'GPU_used': False,
        'source_sha256': q.sha(Path(__file__)), 'scores_sha256': q.sha(path),
        'limits': 'Post-selection descriptive development analysis. No new rule or class-dependent threshold; type masks overlap, so do not sum their counts or treat other risk types as negatives.'}
    q.save(out / 'summary.json', report)
    print(selected['candidate'], types, 'clean_FP', clean_fp, 'risky_FP', other_fp, flush=True)


if __name__ == '__main__':
    run()
