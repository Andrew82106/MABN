"""Read-only CPU audit of every completed fixed convex combination."""
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import run_development as q
import audit_large_matched_combination as independent

ROOT = q.ROOT
OUT = ROOT/'results/large_fixed_convex_v1'
MATCHED = ROOT/'results/large_matched_combination_v2'
ALPHAS = (0., .2, .4, .6, .8, 1.)
PEERS = ('lookback', 'harp_claim', 'semantic_claim')


def score_archive(path, expected):
    assert q.sha(path) == expected
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def external_baselines():
    """Existing selected development reports only; no new candidate/threshold search."""
    files = [
        'results/lookback_regularization_v2/summary.json', 'results/claim_pooling_v1/summary.json',
        'semantic_baseline/cuda_variant/results/summary.json',
        'fit_expansion/llama_baselines_v1/summary.json', 'fit_expansion/harp_tcn_v1/summary.json',
        'results/full_context_encoder_v2/full_finetune/complete.json',
        'results/full_context_aux_transfer_v1/transfer/complete.json',
        'results/frozen_context_generation_fusion_v1/summary.json',
        'results/minicheck_tail_all_docs_v3/frozen0/complete.json',
        'results/minicheck_tail_all_docs_v3/tail2/complete.json',
        'results/citation_alignment_lr_v1/summary.json',
        'results/cited_source_semantic_lr_v1/summary.json',
        'results/citation_heading_scope_v1/summary.json',
        'results/citation_heading_semantic_lr_v1/summary.json',
        'results/completed_score_fusion_v1/summary.json',
        'results/completed_score_combiner_v1/summary.json']
    roster, unavailable = {}, []

    def visit(node, name, source):
        if not isinstance(node, dict):
            return
        m = node.get('metrics', {}).get('calibration') or node.get('calibration')
        if isinstance(m, dict) and {'windows', 'answers'} <= set(m):
            if m['windows'].get('n') == 42241 and m['answers'].get('n') == 159:
                roster[name] = {'metrics': m, 'source': source, 'candidate': node.get('candidate'),
                                'reused_completed_report_not_refitted': True}
                return
        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, name+'::'+key, source)

    for relative in files:
        path = ROOT/relative
        if not path.exists():
            unavailable.append({'path': relative, 'reason': 'No file at this catalogued path; no score invented.'})
            continue
        obj = q.read(path)
        if path.name == 'summary.json' and (path.parent/'complete.json').exists():
            complete = q.read(path.parent/'complete.json')
            if 'summary_sha256' in complete:
                assert complete['summary_sha256'] == q.sha(path)
        assert not obj.get('official_test_opened', obj.get('test_opened', False))
        source = {'path': str(path.resolve()), 'sha256': q.sha(path)}
        chosen = obj.get('selected', obj.get('methods', obj.get('models')))
        count = len(roster)
        if chosen is not None:
            visit(chosen, relative, source)
        if len(roster) == count:
            unavailable.append({'path': relative, 'reason': 'No selected metrics with exact42241/159 denominator in this schema; excluded from comparison, not counted as a weaker score.'})
    return roster, unavailable


def run():
    assert (OUT/'complete.json').exists(), 'Wait for all36 combinations'
    assert not (OUT/'INDEPENDENT_AUDIT.json').exists(), 'Preserve completed audit'
    started = time.perf_counter()
    done = q.read(OUT/'complete.json'); summary = q.read(OUT/'summary.json')
    assert done['summary_sha256'] == q.sha(OUT/'summary.json')
    assert done['candidate_count'] == summary['candidate_count'] == 36
    assert not done['official_test_opened'] and not done['GPU_used'] and done['new_fits'] == 0
    frozen = q.read(OUT/'preparation_complete.json')
    for path, expected in frozen['source_sha256'].items():
        assert q.sha(Path(path)) == expected
    assert frozen['protocol_sha256'] == q.sha(OUT/'protocol.json')
    assert frozen['CPU_check_sha256'] == q.sha(OUT/'CPU_SELFCHECK.json')
    assert q.read(OUT/'started.json')['preparation_sha256'] == q.sha(OUT/'preparation_complete.json')
    meta = q.metadata()
    assert meta['bounds'] == {'fit': [0, 168123], 'calibration': [168123, 210364]}
    families = {peer+'__'+mode for peer in PEERS for mode in ('old_lr', 'old_tree')}
    assert set(summary['all_candidates']) == set(summary['selected']) == set(summary['controls']) == families
    large = np.load(MATCHED/'large_window_probability.npy', allow_pickle=False)
    large_entry = summary['large_selected']
    assert large_entry == q.read(MATCHED/'summary.json')['large_selected']
    assert independent.metric_set(meta, large, large_entry['thresholds'])['calibration'] == large_entry['calibration']
    large_answer = independent.answer_max(meta, large)
    large_ts = {'window': independent.best_threshold([w['label'] for w in meta['windows'][168123:]], large[168123:]),
                'answer': independent.best_threshold([a['label'] for a in meta['answers'][634:]], large_answer[634:])}
    assert large_ts == large_entry['thresholds']
    entries, endpoints, metrics_by_candidate = [], [], {}
    bases = {}
    for family in sorted(families):
        peer, mode = family.split('__')
        control = summary['controls'][family]
        if mode == 'old_lr':
            directory = ROOT/'results/citation_alignment_lr_v1'
            original = q.read(directory/'summary.json')['selected'][peer+'__two_scores_and_citation']
            path = directory/(original['candidate']+'_scores.npz')
        else:
            directory = ROOT/'results/completed_score_combiner_v1'
            original = q.read(directory/'summary.json')['methods'][peer]
            path = directory/(peer+'_scores.npz')
        assert control == original
        old = score_archive(path, original['scores_sha256'])
        independent.check_entry(meta, old['window_scores'], old['answer_scores'], original,
                                original.get('C'), check_key=mode == 'old_lr')
        bases[family] = old['window_scores']
        candidates = summary['all_candidates'][family]
        assert [e['large_weight'] for e in candidates] == list(ALPHAS)
        for entry in candidates:
            name, alpha = entry['candidate'], entry['large_weight']
            assert q.read(OUT/(name+'.json')) == entry
            assert entry['peer'] == peer and entry['base_mode'] == mode and entry['large_epoch'] == large_entry['epoch']
            stored = score_archive(OUT/(name+'_scores.npz'), entry['scores_sha256'])
            expected = np.add(np.multiply(old['window_scores'], 1-alpha), np.multiply(large, alpha))
            assert np.array_equal(stored['window_scores'], expected), name
            metrics, key = independent.check_entry(meta, expected, stored['answer_scores'], entry, alpha)
            metrics_by_candidate[name] = metrics['calibration']
            entries.append({'candidate': name, 'large_weight': alpha, 'score_formula_max_abs_diff': 0.,
                            'answer_max_exact': True, 'thresholds_counts_exact': True,
                            'calibration': metrics['calibration'], 'selection_key': key})
            if alpha == 0.:
                assert np.array_equal(expected, old['window_scores'])
                assert np.array_equal(stored['answer_scores'], old['answer_scores'])
                assert metrics == original['metrics'] and entry['thresholds'] == original['thresholds']
                endpoints.append({'family': family, 'endpoint': 'alpha0', 'exact': True})
            elif alpha == 1.:
                assert np.array_equal(expected, large) and np.array_equal(stored['answer_scores'], large_answer)
                assert metrics['calibration'] == large_entry['calibration'] and entry['thresholds'] == large_entry['thresholds']
                endpoints.append({'family': family, 'endpoint': 'alpha1', 'exact': True})
        selected = max(candidates, key=lambda e: e['selection_key'])
        assert selected == summary['selected'][family]
    assert len(entries) == 36 and len(endpoints) == 12
    selected_entries = list(summary['selected'].values())
    selected_key = max(tuple(e['selection_key']) for e in selected_entries)
    winners = [e for e in selected_entries if tuple(e['selection_key']) == selected_key]
    assert len(winners) == 1, 'Report ties rather than invent a family tie-break'
    winner = winners[0]
    assert winner == max([e for es in summary['all_candidates'].values() for e in es], key=lambda e: e['selection_key'])
    baseline_roster = {'standalone_large': {'metrics': large_entry['calibration'], 'candidate': f"large_epoch{large_entry['epoch']}"}}
    for family, control in summary['controls'].items():
        baseline_roster[family] = {'metrics': control['metrics']['calibration'], 'candidate': control.get('candidate', family)}
    prior_audit = q.read(MATCHED/'INDEPENDENT_AUDIT.json')
    assert prior_audit['passed'] and prior_audit['summary_sha256'] == q.sha(MATCHED/'summary.json')
    for kind in ('new_models_replayed', 'frozen_controls_replayed'):
        for e in prior_audit[kind]:
            baseline_roster['prior_audited::'+e['candidate']] = {'metrics': e['calibration'], 'candidate': e['candidate']}
    other, unavailable = external_baselines(); baseline_roster.update(other)
    differences = []
    for entry in entries:
        m = entry['calibration']
        for baseline, b in baseline_roster.items():
            bm = b['metrics']
            differences.append({'candidate': entry['candidate'], 'baseline': baseline,
                'window_F1_difference': m['windows']['f1']-bm['windows']['f1'],
                'answer_F1_difference': m['answers']['f1']-bm['answers']['f1'],
                'candidate_at_least_baseline_on_both': m['windows']['f1']>=bm['windows']['f1'] and m['answers']['f1']>=bm['answers']['f1']})
    winner_diffs = [d for d in differences if d['candidate'] == winner['candidate']]
    result = {'passed': True, 'all_36_candidates': entries, 'all_12_endpoints': endpoints,
        'all_formula_probabilities_and_answer_max_exact': True, 'all_thresholds_counts_and_selections_independently_recomputed': True,
        'cross_six_selection_key': list(selected_key), 'cross_six_tie_count': len(winners),
        'cross_six_winner': winner, 'winner_is_semantic_tree': winner['peer']=='semantic_claim' and winner['base_mode']=='old_tree',
        'both_metrics_from_one_candidate': True, 'baseline_roster': baseline_roster,
        'all_candidate_baseline_differences': differences, 'winner_baseline_differences': winner_diffs,
        'winner_noninferior_both_among_catalogued_completed_reports': all(d['candidate_at_least_baseline_on_both'] for d in winner_diffs),
        'unavailable_baseline_report_schemas': unavailable,
        'calibration_answer_count': 159, 'calibration_window_count': 42241,
        'summary_sha256': q.sha(OUT/'summary.json'), 'complete_sha256': q.sha(OUT/'complete.json'),
        'auditor_sha256': q.sha(Path(__file__)), 'independent_metrics_helper_sha256': q.sha(Path(independent.__file__)),
        'official_test_opened': False, 'new_fits': 0, 'GPU_used': False, 'seconds': time.perf_counter()-started,
        'limits': 'Ranking across repeatedly developed calibration candidates is not independent confirmation. Baseline catalog scope and any unavailable schemas are explicit; future performance is not guaranteed.'}
    q.save(OUT/'INDEPENDENT_AUDIT.json', result)
    m = winner['metrics']['calibration']
    lines = ['# Independent fixed-convex audit', '',
        'All 36 saved formulas, answer maxima, thresholds, counts and choices pass; all 12 endpoints replay exactly. No refitting or GPU.', '',
        f"The unique cross-family winner under the existing selection key is **{winner['candidate']}**: window F1 **{m['windows']['f1']:.9f}**, answer F1 **{m['answers']['f1']:.9f}**. Both come from this one candidate.", '',
        '| Fixed base | Selected large weight | Window F1 | Answer F1 |', '|---|---:|---:|---:|']
    for family, e in summary['selected'].items():
        em = e['metrics']['calibration']
        lines.append(f"| {family} | {e['large_weight']:g} | {em['windows']['f1']:.6f} | {em['answers']['f1']:.6f} |")
    lines += ['', f"The JSON retains all 36×{len(baseline_roster)} candidate/baseline differences and the complete baseline roster. Any unavailable schemas are listed explicitly. This is repeatedly inspected calibration, not a new test; the result does not guarantee future nondegradation."]
    (OUT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('FIXED_CONVEX_AUDIT_PASSED', winner['candidate'], m['windows']['f1'], m['answers']['f1'],
          'baseline_reports', len(baseline_roster), 'unavailable', len(unavailable), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('run',)); p.parse_args()
    with threadpool_limits(limits=4):
        try:
            run()
        except Exception:
            q.save(OUT/f'AUDIT_FAILURE_{time.time_ns()}.json', {'traceback': traceback.format_exc(), 'new_fits': 0, 'GPU_used': False})
            raise
