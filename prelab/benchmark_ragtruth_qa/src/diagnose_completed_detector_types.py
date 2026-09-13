"""Fixed-threshold type recall for one fully completed detector comparison."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import run_development as q
from diagnose_completed_tail import KINDS


def run(target):
    source = q.ROOT / f'results/{target}_fixed_convex_v1'
    out = q.ROOT / f'results/{target}_fixed_type_diagnosis_v1'
    assert not out.exists(), 'Preserve existing diagnosis'
    complete = q.read(source / 'complete.json')
    assert q.sha(source / 'summary.json') == complete['summary_sha256']
    summary = q.read(source / 'summary.json')
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    assert (lo, hi) == (168123, 210364)
    windows = meta['windows'][lo:hi]
    y = np.asarray([w['label'] for w in windows], bool)
    typed = {k: defaultdict(set) for k in KINDS}
    for token in meta['tokens']:
        if token['partition'] != 'calibration':
            continue
        for span in token['span_token_mapping']:
            kind = token['original_labels'][span['span_index']]['label_type']
            typed[kind][token['response_id']].update(span['risk_token_indices'])
    masks = {k: np.asarray([bool(set(w['token_indices']) & typed[k][w['response_id']])
                            for w in windows]) for k in KINDS}
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), y)
    clean_ids = {a['response_id'] for a in meta['answers'][634:] if a['label'] == 0}
    clean = np.asarray([w['response_id'] in clean_ids for w in windows])
    entries = []
    s = summary['target_selected']
    entries.append((target + '_standalone', np.load(source / 'detector_window_probability.npy'),
                    s['thresholds'], s['calibration']))
    for family, entry in summary['selected'].items():
        path = source / (entry['candidate'] + '_scores.npz')
        assert q.sha(path) == entry['scores_sha256']
        with np.load(path, allow_pickle=False) as archive:
            scores = archive['window_scores'].copy()
            assert np.array_equal(q.answer_scores(meta, scores), archive['answer_scores'])
        entries.append((entry['candidate'], scores, entry['thresholds'], entry['metrics']['calibration']))
    results = {}
    for name, scores, thresholds, expected in entries:
        metrics = q.metrics(meta, scores, thresholds)['calibration']
        assert metrics == expected
        pred = scores[lo:hi] >= thresholds['window']['threshold']
        types = {k: {'risk_windows': int(mask.sum()), 'detected': int(pred[mask].sum()),
                     'recall': float(pred[mask].mean())} for k, mask in masks.items()}
        a, b = int((pred & ~y & clean).sum()), int((pred & ~y & ~clean).sum())
        assert a + b == metrics['windows']['fp']
        results[name] = {'metrics': metrics, 'thresholds': thresholds, 'type_recall_only': types,
                         'false_positive_windows_clean_answers': a, 'false_positive_windows_risky_answers': b}
    out.mkdir()
    q.save(out / 'summary.json', {'methods': results, 'target': target, 'source_sha256': q.sha(Path(__file__)),
        'comparison_summary_sha256': q.sha(source / 'summary.json'),
        'limits': 'Descriptive developed-calibration analysis at the existing thresholds. Types overlap; no separate per-type thresholds, new winner selection or other-risk-as-negative scoring.',
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'official_test_opened': False})
    lines = ['# Fixed thresholds: error type recall', '',
             '| Method | Window F1 | Answer F1 | Evident conflict detected /997 | Clean-answer FP windows |',
             '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {name} | {r['metrics']['windows']['f1']:.6f} | {r['metrics']['answers']['f1']:.6f} | {r['type_recall_only']['Evident Conflict']['detected']} | {r['false_positive_windows_clean_answers']} |")
    lines += ['', 'All thresholds were already selected before this diagnosis. No new models or scores were fitted. Overlapping type counts are descriptive only; original labels and denominators remain unchanged.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=('nli', 'fava'))
    run(parser.parse_args().target)
