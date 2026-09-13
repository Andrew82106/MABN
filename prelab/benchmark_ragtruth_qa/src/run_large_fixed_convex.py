"""Fixed, no-refit convex combinations of six old controls and one large model."""
from pathlib import Path
import argparse
import time
import numpy as np
from threadpoolctl import threadpool_limits
import run_development as q

OUT = q.ROOT/'results/large_fixed_convex_v1'
OLD_LR = q.ROOT/'results/citation_alignment_lr_v1'
OLD_TREE = q.ROOT/'results/completed_score_combiner_v1'
LARGE = q.ROOT/'results/full_context_encoder_large_v1/full_finetune'
MATCHED = q.ROOT/'results/large_matched_combination_v2'
PEERS = ('lookback', 'harp_claim', 'semantic_claim')
ALPHAS = (0., .2, .4, .6, .8, 1.)


def protocol():
    return {'version': 'six-frozen-controls-plus-large-convex-v1',
        'bases': 'The three already-selected original two-scores+eight-citation-feature LRs and three original two-score monotone trees. These are the old models without large input, not the new large-trained combiners.',
        'peers': list(PEERS), 'large_weights': list(ALPHAS), 'candidate_count': 36,
        'large': 'Exactly one globally selected epoch from the complete six-epoch generic ModernBERT-large run. Use its already audited original-native window probability; no per-base epoch selection.',
        'formula': 'Each original eligible window: score=(1-alpha)*frozen_base_window_probability+alpha*frozen_large_window_probability. Probability-space convex combination, not logits or token-level fusion.',
        'answer': 'Maximum of the combined score over every original eligible window. Do not combine already-aggregated answer scores.',
        'scope': 'Original634 native fit answers/168123 windows and159 calibration answers/42241 windows. Original human labels and4rawBPE stride1 lexical geometry unchanged; no filtering or new annotations.',
        'selection': 'For each of the same36 candidates, original calibration-only two F1-optimal thresholds with precision then higher-cutoff ties. Select one alpha per base by max min(two F1), windowF1, window precision, then smaller alpha. The same candidate supplies both metrics.',
        'endpoints': 'alpha0 must reproduce each original control score, answermax, thresholds and metrics exactly; alpha1 must reproduce the one standalone large score and its calibration metrics/thresholds.',
        'training': 'None. Freeze original controls, their C choices, large epoch and all model parameters. No scaler or model refits, no early stopping or new optimizer steps.',
        'artifacts': 'Retain all36 scores, thresholds, fit/calibration counts, six selected candidates, original controls and standalone large comparisons.',
        'limits': ['Repeated development calibration; no independent-test claim.',
                   'All upstream models and prior selections remain as completed; their supervision and compute differ.',
                   'A calibration-selected alpha0 fallback cannot guarantee no degradation on future data.',
                   'Extra whole-input semantic checker and offline answer processing; not a pure generator-state probe.'],
        'GPU_used': False, 'official_test_opened': False, 'new_fits': 0}


def base_entries():
    lr = q.read(OLD_LR/'summary.json'); tree = q.read(OLD_TREE/'summary.json')
    entries = []
    for peer in PEERS:
        family = peer+'__two_scores_and_citation'
        a = lr['selected'][family]
        assert a == max(lr['all_candidates'][family], key=lambda e: e['selection_key'])
        entries.append((peer+'__old_lr', peer, 'old_lr', a, OLD_LR/(a['candidate']+'_scores.npz')))
        a = tree['methods'][peer]
        entries.append((peer+'__old_tree', peer, 'old_tree', a, OLD_TREE/(peer+'_scores.npz')))
    return entries


def bindings():
    files = [Path(__file__), Path(q.__file__), OLD_LR/'complete.json', OLD_LR/'summary.json',
             OLD_TREE/'complete.json', OLD_TREE/'summary.json', LARGE/'complete.json',
             MATCHED/'complete.json', MATCHED/'summary.json', MATCHED/'started.json',
             MATCHED/'large_window_probability.npy', MATCHED/'INDEPENDENT_AUDIT.json',
             MATCHED/'AUDIT_DESIGN.json', q.ROOT/'src/audit_large_matched_combination.py']
    files.extend(path for _, _, _, _, path in base_entries())
    return {str(p.resolve()): q.sha(p) for p in files}


def readiness():
    for d in (OLD_LR, OLD_TREE, LARGE, MATCHED):
        assert (d/'complete.json').exists(), str(d)
    for d in (OLD_LR, OLD_TREE, MATCHED):
        done = q.read(d/'complete.json')
        assert not done['official_test_opened']
        assert q.sha(d/'summary.json') == done['summary_sha256']
    audit = q.read(MATCHED/'INDEPENDENT_AUDIT.json')
    assert audit['passed'] and audit['new_fits'] == 0 and not audit['GPU_used']
    assert audit['summary_sha256'] == q.sha(MATCHED/'summary.json')
    assert audit['complete_sha256'] == q.sha(MATCHED/'complete.json')
    large_done = q.read(LARGE/'complete.json')
    assert not large_done['official_test_opened']
    assert [e['epoch'] for e in large_done['all_epochs']] == list(range(7))
    selected = max(large_done['all_epochs'][1:], key=lambda e: e['selection_key'])
    assert selected == large_done['selected'] == q.read(MATCHED/'summary.json')['large_selected']
    assert q.read(MATCHED/'started.json')['large_complete_sha256'] == q.sha(LARGE/'complete.json')
    return selected


def prepare():
    assert not (OUT/'preparation_complete.json').exists(), 'Preserve existing preparation'
    OUT.mkdir(parents=True, exist_ok=True)
    selected = readiness()
    # Synthetic endpoints and an example proving max(blend) differs from blend(max).
    a = np.asarray([.9, .1]); b = np.asarray([.2, .8])
    assert np.array_equal((1-0.)*a+0.*b, a)
    assert np.array_equal((1-1.)*a+1.*b, b)
    assert max(.5*a+.5*b) != .5*max(a)+.5*max(b)
    q.save(OUT/'protocol.json', protocol())
    q.save(OUT/'CPU_SELFCHECK.json', {'passed': True, 'synthetic_endpoints_exact': True,
        'window_blend_before_answer_max': True, 'real_predictions_loaded': False,
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})
    q.save(OUT/'preparation_complete.json', {'status': 'prepared_not_run',
        'source_sha256': bindings(), 'protocol_sha256': q.sha(OUT/'protocol.json'),
        'CPU_check_sha256': q.sha(OUT/'CPU_SELFCHECK.json'), 'large_selected_epoch': selected['epoch'],
        'frozen_bases': [{'family': name, 'peer': peer, 'mode': mode,
                          'original_candidate': e.get('candidate', peer), 'C': e.get('C'),
                          'scores_sha256': e['scores_sha256']}
                         for name, peer, mode, e, _ in base_entries()],
        'candidate_count': 36, 'new_fits': 0, 'GPU_used': False})
    print('SIX_OLD_CONTROLS_CONVEX_36_PREPARED_NOT_RUN', flush=True)


def run():
    frozen = q.read(OUT/'preparation_complete.json')
    assert frozen['source_sha256'] == bindings()
    assert q.read(OUT/'protocol.json') == protocol() and q.sha(OUT/'protocol.json') == frozen['protocol_sha256']
    assert q.sha(OUT/'CPU_SELFCHECK.json') == frozen['CPU_check_sha256']
    assert not (OUT/'started.json').exists(), 'No overwrite or automatic resume'
    large_entry = readiness()
    assert large_entry['epoch'] == frozen['large_selected_epoch']
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    assert (lo, hi) == (168123, 210364) and len(meta['answers']) == 793
    large = np.load(MATCHED/'large_window_probability.npy', allow_pickle=False)
    assert large.shape == (210364,) and np.isfinite(large).all() and np.all((large >= 0) & (large <= 1))
    assert q.metrics(meta, large, large_entry['thresholds'])['calibration'] == large_entry['calibration']
    loaded = []
    for name, peer, mode, entry, path in base_entries():
        assert q.sha(path) == entry['scores_sha256']
        with np.load(path, allow_pickle=False) as z:
            score, answer = z['window_scores'].copy(), z['answer_scores'].copy()
        assert score.shape == (210364,) and np.isfinite(score).all()
        assert np.array_equal(q.answer_scores(meta, score), answer)
        assert q.metrics(meta, score, entry['thresholds']) == entry['metrics']
        loaded.append((name, peer, mode, entry, score))
    q.save(OUT/'started.json', {'time': time.time(), 'preparation_sha256': q.sha(OUT/'preparation_complete.json'),
        'large_epoch': large_entry['epoch'], 'new_fits': 0, 'GPU_used': False})
    started = time.perf_counter(); all_candidates, selected = {}, {}; endpoints = []
    cal_y = [w['label'] for w in meta['windows'][lo:hi]]
    cal_answer_y = [a['label'] for a in meta['answers'][634:]]
    for family, peer, mode, control, base_score in loaded:
        candidates = []
        for alpha in ALPHAS:
            scores = (1-alpha)*base_score+alpha*large
            answer = q.answer_scores(meta, scores)
            thresholds = {'window': q.choose_threshold(cal_y, scores[lo:hi]),
                          'answer': q.choose_threshold(cal_answer_y, answer[634:])}
            metrics = q.metrics(meta, scores, thresholds)
            if alpha == 0.:
                assert np.array_equal(scores, base_score)
                assert thresholds == control['thresholds'] and metrics == control['metrics']
                endpoints.append({'base': family, 'endpoint': 'alpha0', 'probabilities_and_metrics_exact': True})
            if alpha == 1.:
                assert np.array_equal(scores, large)
                assert thresholds == large_entry['thresholds'] and metrics['calibration'] == large_entry['calibration']
                endpoints.append({'base': family, 'endpoint': 'alpha1', 'probabilities_and_calibration_exact': True})
            name = family+f'__large_weight{alpha:g}'
            np.savez_compressed(OUT/(name+'_scores.npz'), window_scores=scores, answer_scores=answer)
            entry = {'candidate': name, 'peer': peer, 'base_mode': mode, 'large_weight': alpha,
                     'large_epoch': large_entry['epoch'], 'thresholds': thresholds, 'metrics': metrics,
                     'selection_key': list(q.selection_key(thresholds, alpha)),
                     'scores_sha256': q.sha(OUT/(name+'_scores.npz')), 'new_fits': 0}
            q.save(OUT/(name+'.json'), entry); candidates.append(entry)
        all_candidates[family] = candidates
        selected[family] = max(candidates, key=lambda e: e['selection_key'])
        e = selected[family]; m = e['metrics']['calibration']
        print('LARGE_FIXED_CONVEX_FAMILY', family, e['large_weight'], m['windows']['f1'], m['answers']['f1'], flush=True)
    assert sum(map(len, all_candidates.values())) == 36 and len(selected) == 6 and len(endpoints) == 12
    assert bindings() == frozen['source_sha256']
    q.save(OUT/'summary.json', {'all_candidates': all_candidates, 'selected': selected,
        'controls': {name: entry for name, _, _, entry, _ in loaded}, 'large_selected': large_entry,
        'endpoint_checks': endpoints, 'seconds': time.perf_counter()-started,
        'candidate_count': 36, 'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})
    lines = ['# Frozen controls + fixed large probabilities', '',
             '| Base | Large weight | Window F1 | Answer F1 |', '|---|---:|---:|---:|']
    for family, entry in selected.items():
        m = entry['metrics']['calibration']
        lines.append(f"| {family} | {entry['large_weight']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    lines += ['', 'Every row uses one candidate for both metrics. All 36 candidates and the exact old-control/standalone-large endpoints are retained. No model was refitted. This is repeatedly developed calibration; selecting alpha0 cannot guarantee future nondegradation.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    q.save(OUT/'complete.json', {'summary_sha256': q.sha(OUT/'summary.json'), 'candidate_count': 36,
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run')); args = p.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[args.stage]()
