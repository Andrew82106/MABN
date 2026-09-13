"""Separate, fixed36-candidate convex comparisons for completed NLI and FAVA runs.

Preparation binds static inputs only; predictions require the chosen upstream
training run to finish. No GPU/model fit or cumulative fusion is performed here.
"""
from pathlib import Path
import argparse
import time
import numpy as np
from threadpoolctl import threadpool_limits
import run_development as q
import run_large_fixed_convex as original_controls

ALPHAS = (0., .2, .4, .6, .8, 1.)
CONFIG = {
    'nli': {'directory': 'full_context_nli_initialization_v2', 'run': 'full_finetune',
            'script': 'run_full_context_nli_initialization_v2.py', 'epochs_key': 'all_epochs',
            'all_epochs': list(range(7)), 'eligible_epochs': list(range(1, 7)), 'file_prefix': 'epoch_',
            'limits': 'Exploratory NLI initialization with limited upstream provenance; no claim of proven pretraining independence.'},
    'fava': {'directory': 'full_context_fava_transfer_v1', 'run': 'transfer',
             'script': 'run_fava_transfer.py', 'epochs_key': 'all_QA_epochs',
             'all_epochs': [1, 2, 3], 'eligible_epochs': [1, 2, 3], 'file_prefix': 'qa_epoch_',
             'limits': 'Synthetic silver FAVA auxiliary supervision, then original human QA training; auxiliary final checkpoint never enters model selection.'}}


def paths(target):
    c = CONFIG[target]
    return (q.ROOT/f'results/{target}_fixed_convex_v1',
            q.ROOT/'results'/c['directory'], q.ROOT/'results'/c['directory']/c['run'])


def protocol(target):
    return {'version': target+'-six-original-controls-convex-v1', 'target': target,
        'bases': 'Exactly the original3 selected two-score+8citation LRs and original3 two-score monotone trees used by large_fixed_convex_v1. Never use the new large convex combinations or large-trained combiners as bases.',
        'detector': 'One already selected epoch from the complete upstream run; selection belongs to that run, not individual fusion families. FAVA permits only its3 QA epochs, never the auxiliary checkpoint.',
        'weights': list(ALPHAS), 'candidate_count': 36,
        'formula': 'For each original eligible window: (1-alpha)*original_control_probability + alpha*target_detector_probability. Then recompute answermax over all original eligible windows.',
        'scope': 'Original634 native fit answers/168123 windows and159 calibration answers/42241 windows; human labels, raw4BPE stride1 lexical windows and refusals unchanged.',
        'calibration': 'Original two F1-optimal thresholds, ties by precision then higher threshold. One alpha per family selected by max min(two F1), windowF1, window precision, then smaller alpha. Both metrics come from the same candidate.',
        'endpoints': 'alpha0 exactly replays original controls, answers, thresholds and both partitions metrics; alpha1 exactly replays the single target detector and its calibration thresholds/metrics.',
        'training': 'No model/scaler fitting or optimizer steps. Do not change original base C choices or upstream epoch choices. Prepare reads static completed preparation and old control metadata only; run requires upstream completion first.',
        'artifacts': 'Independent target output, preparation/protocol freeze, selected target binding, target window cache, all36 scores/thresholds/counts and6 selected candidates.',
        'limits': [CONFIG[target]['limits'], 'Extra offline semantic detector, not a pure original-generator white-box probe.',
                   'Repeated calibration; alpha0 fallback does not guarantee nondegradation on unseen data.',
                   'Upstream training cost and supervision differ. NLI and FAVA are independent comparisons, not cumulative large→NLI→FAVA fusions.'],
        'GPU_used': False, 'new_fits': 0, 'official_test_opened': False}


def static_bindings(target):
    out, prepared, _ = paths(target)
    files = [Path(__file__), Path(q.__file__), Path(original_controls.__file__),
             q.ROOT/'src'/CONFIG[target]['script'], prepared/'protocol.json', prepared/'preparation_complete.json',
             original_controls.OLD_LR/'complete.json', original_controls.OLD_LR/'summary.json',
             original_controls.OLD_TREE/'complete.json', original_controls.OLD_TREE/'summary.json']
    files.extend(path for _, _, _, _, path in original_controls.base_entries())
    return {str(p.resolve()): q.sha(p) for p in files}


def old_controls_ready():
    for directory in (original_controls.OLD_LR, original_controls.OLD_TREE):
        completed = q.read(directory/'complete.json')
        assert not completed['official_test_opened']
        assert q.sha(directory/'summary.json') == completed['summary_sha256']


def prepare(target):
    out, prepared, _ = paths(target)
    assert not (out/'preparation_complete.json').exists(), 'Preserve target preparation'
    out.mkdir(parents=True, exist_ok=True)
    old_controls_ready()
    assert (prepared/'preparation_complete.json').exists()
    original_bases = original_controls.base_entries()
    assert len(original_bases) == 6
    a, b = np.asarray([.9, .1]), np.asarray([.2, .8])
    assert np.array_equal((1-0.)*a+0.*b, a) and np.array_equal((1-1.)*a+1.*b, b)
    assert max(.5*a+.5*b) != .5*max(a)+.5*max(b)
    q.save(out/'protocol.json', protocol(target))
    q.save(out/'CPU_SELFCHECK.json', {'passed': True, 'synthetic_endpoints_exact': True,
        'blend_before_answer_max': True, 'target_predictions_or_training_results_read': False,
        'real_predictions_loaded': False, 'GPU_used': False, 'new_fits': 0})
    q.save(out/'preparation_complete.json', {'status': 'prepared_not_run', 'target': target,
        'source_sha256': static_bindings(target), 'protocol_sha256': q.sha(out/'protocol.json'),
        'CPU_check_sha256': q.sha(out/'CPU_SELFCHECK.json'),
        'original_bases': [{'family': name, 'peer': peer, 'mode': mode, 'candidate': entry.get('candidate', peer),
                            'C': entry.get('C'), 'scores_sha256': entry['scores_sha256']}
                           for name, peer, mode, entry, _ in original_bases],
        'upstream_complete_required_only_at_run': True, 'candidate_count': 36,
        'GPU_used': False, 'new_fits': 0, 'official_test_opened': False})
    print('TARGET_CONVEX_PREPARED_NOT_RUN', target, 36, flush=True)


def completed_detector(target):
    _, _, directory = paths(target)
    assert (directory/'complete.json').exists(), f'Wait for complete upstream {target}'
    complete = q.read(directory/'complete.json')
    assert not complete['official_test_opened']
    cfg = CONFIG[target]; epochs = complete[cfg['epochs_key']]
    assert [e['epoch'] for e in epochs] == cfg['all_epochs']
    eligible = [e for e in epochs if e['epoch'] in cfg['eligible_epochs']]
    assert len(eligible) == len(cfg['eligible_epochs'])
    selected = max(eligible, key=lambda e: e['selection_key'])
    assert selected == complete['selected']
    path = directory/f"{cfg['file_prefix']}{selected['epoch']:02d}_token_predictions.npz"
    assert q.sha(path) == selected['artifacts_sha256']['_token_predictions.npz']
    return selected, path, q.sha(directory/'complete.json')


def run(target):
    out, _, _ = paths(target)
    frozen = q.read(out/'preparation_complete.json')
    assert frozen['target'] == target and frozen['source_sha256'] == static_bindings(target)
    assert q.read(out/'protocol.json') == protocol(target)
    assert frozen['protocol_sha256'] == q.sha(out/'protocol.json')
    assert frozen['CPU_check_sha256'] == q.sha(out/'CPU_SELFCHECK.json')
    assert not (out/'started.json').exists(), 'No overwrite/resume'
    # Completion precedes any target token-prediction loading.
    target_entry, target_path, target_complete_hash = completed_detector(target)
    meta = q.metadata(); lo, hi = meta['bounds']['calibration']
    assert (lo, hi) == (168123, 210364) and len(meta['answers']) == 793
    with np.load(target_path, allow_pickle=False) as z:
        assert len(z.files) == 3839
        probabilities = {a['response_id']: z[a['response_id']].copy() for a in meta['answers']}
    target_scores = np.asarray([max(float(probabilities[w['response_id']][j]) for j in w['token_indices']
        if meta['by_response'][w['response_id']]['tokens']['lexical_mask'][j]) for w in meta['windows']], np.float64)
    assert target_scores.shape == (210364,) and np.isfinite(target_scores).all()
    assert np.all((target_scores >= 0) & (target_scores <= 1))
    assert q.metrics(meta, target_scores, target_entry['thresholds'])['calibration'] == target_entry['calibration']
    controls = []
    for family, peer, mode, entry, path in original_controls.base_entries():
        assert q.sha(path) == entry['scores_sha256']
        with np.load(path, allow_pickle=False) as z:
            score, answer = z['window_scores'].copy(), z['answer_scores'].copy()
        assert np.array_equal(q.answer_scores(meta, score), answer)
        assert q.metrics(meta, score, entry['thresholds']) == entry['metrics']
        controls.append((family, peer, mode, entry, score))
    q.save(out/'started.json', {'time': time.time(), 'target': target,
        'preparation_sha256': q.sha(out/'preparation_complete.json'), 'upstream_complete_sha256': target_complete_hash,
        'upstream_token_predictions_sha256': q.sha(target_path), 'selected_epoch': target_entry['epoch'],
        'GPU_used': False, 'new_fits': 0})
    np.save(out/'detector_window_probability.npy', target_scores)
    started = time.perf_counter(); all_candidates, selected, endpoints = {}, {}, []
    wy = [w['label'] for w in meta['windows'][lo:hi]]; ay = [a['label'] for a in meta['answers'][634:]]
    for family, peer, mode, control, original_score in controls:
        entries = []
        for alpha in ALPHAS:
            scores = (1-alpha)*original_score+alpha*target_scores
            answer = q.answer_scores(meta, scores)
            thresholds = {'window': q.choose_threshold(wy, scores[lo:hi]), 'answer': q.choose_threshold(ay, answer[634:])}
            metrics = q.metrics(meta, scores, thresholds)
            if alpha == 0.:
                assert np.array_equal(scores, original_score)
                assert metrics == control['metrics'] and thresholds == control['thresholds']
                endpoints.append({'family': family, 'alpha': alpha, 'exact': True})
            elif alpha == 1.:
                assert np.array_equal(scores, target_scores)
                assert metrics['calibration'] == target_entry['calibration'] and thresholds == target_entry['thresholds']
                endpoints.append({'family': family, 'alpha': alpha, 'exact': True})
            name = family+f'__{target}_weight{alpha:g}'
            np.savez_compressed(out/(name+'_scores.npz'), window_scores=scores, answer_scores=answer)
            entry = {'candidate': name, 'peer': peer, 'base_mode': mode, 'target': target, 'target_weight': alpha,
                'target_epoch': target_entry['epoch'], 'thresholds': thresholds, 'metrics': metrics,
                'selection_key': list(q.selection_key(thresholds, alpha)), 'scores_sha256': q.sha(out/(name+'_scores.npz'))}
            q.save(out/(name+'.json'), entry); entries.append(entry)
        all_candidates[family] = entries
        selected[family] = max(entries, key=lambda e: e['selection_key'])
        e = selected[family]; m = e['metrics']['calibration']
        print('TARGET_CONVEX_FAMILY_COMPLETE', target, family, e['target_weight'], m['windows']['f1'], m['answers']['f1'], flush=True)
    assert sum(map(len, all_candidates.values())) == 36 and len(selected) == 6 and len(endpoints) == 12
    assert frozen['source_sha256'] == static_bindings(target)
    q.save(out/'summary.json', {'target': target, 'target_selected': target_entry, 'all_candidates': all_candidates,
        'selected': selected, 'controls': {family: entry for family, _, _, entry, _ in controls},
        'endpoint_checks': endpoints, 'candidate_count': 36, 'new_fits': 0, 'GPU_used': False,
        'official_test_opened': False, 'seconds': time.perf_counter()-started})
    lines = [f'# {target} independent fixed convex comparison', '', '| Original base | Target weight | Window F1 | Answer F1 |', '|---|---:|---:|---:|']
    for family, e in selected.items():
        m = e['metrics']['calibration']
        lines.append(f"| {family} | {e['target_weight']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    lines += ['', 'All36 candidates and12 exact endpoints are retained. The six bases are the original old controls; no large-combination output is fed into this run. Both metrics in each row belong to one candidate. Repeated calibration, not independent confirmation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    q.save(out/'complete.json', {'summary_sha256': q.sha(out/'summary.json'), 'target': target,
        'candidate_count': 36, 'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('target', choices=tuple(CONFIG)); args = p.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'run': run}[args.stage](args.target)
