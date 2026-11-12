"""Resolve two known report layouts; leave the completed36-candidate audit unchanged."""
from pathlib import Path
import run_development as q

OUT = q.ROOT/'results/large_fixed_convex_v1'


def run():
    audit = q.read(OUT/'INDEPENDENT_AUDIT.json'); assert audit['passed']
    original_hash = q.sha(OUT/'INDEPENDENT_AUDIT.json')
    extra = {}
    p = q.ROOT/'semantic_baseline/cuda_variant/results/summary.json'
    for name, entry in q.read(p)['reports'].items():
        m = entry['metrics']['calibration']
        assert m['windows']['n'] == 42241 and m['answers']['n'] == 159
        extra['MiniCheck_reports::'+name] = {'metrics': m, 'source': str(p.resolve()), 'sha256': q.sha(p)}
    for mode in ('fusion', 'semantic_only'):
        p = q.ROOT/f'results/frozen_context_generation_fusion_v1/{mode}/complete.json'
        completed = q.read(p); assert not completed['official_test_opened']
        entry = completed['selected']; m = entry['calibration']
        assert m['windows']['n'] == 42241 and m['answers']['n'] == 159
        extra['frozen_context_head::'+mode] = {'metrics': m, 'source': str(p.resolve()), 'sha256': q.sha(p)}
    comparisons = []
    for candidate in audit['all_36_candidates']:
        m = candidate['calibration']
        for name, baseline in extra.items():
            b = baseline['metrics']
            comparisons.append({'candidate': candidate['candidate'], 'baseline': name,
                'window_F1_difference': m['windows']['f1']-b['windows']['f1'],
                'answer_F1_difference': m['answers']['f1']-b['answers']['f1']})
    winner = audit['cross_six_winner']['candidate']
    wd = [r for r in comparisons if r['candidate'] == winner]
    assert original_hash == q.sha(OUT/'INDEPENDENT_AUDIT.json')
    q.save(OUT/'BASELINE_ROSTER_SUPPLEMENT.json', {'passed': True,
        'original_audit_unchanged_sha256': original_hash, 'resolved_report_entries': extra,
        'all_36_additional_differences': comparisons, 'winner_differences': wd,
        'winner_noninferior_both_for_these_reports': all(d['window_F1_difference']>=0 and d['answer_F1_difference']>=0 for d in wd),
        'same_159_calibration_only': True, 'new_fit_or_threshold_or_prediction': False,
        'official_test_opened': False, 'GPU_used': False,
        'scope': 'Two already identified report layouts only, not an exhaustive search or universal-best claim.'})
    print('TWO_REPORT_LAYOUTS_RESOLVED', len(extra), 'NO_SCORES_OR_SELECTION_CHANGED', flush=True)


if __name__ == '__main__':
    run()
