"""Read completed fixed scores only. Gold-gated unions are diagnostic oracles."""
from pathlib import Path
from collections import defaultdict
import sys
import time
import math
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q

CURRENT = 'semantic_claim__old_tree__large_weight0.4'
SOURCES = {}


def bind(path):
    path = Path(path)
    key = str(path.resolve())
    if key not in SOURCES:
        SOURCES[key] = q.sha(path)
    return SOURCES[key]


def read(path):
    bind(path)
    return q.read(path)


def directory(relative):
    folder = ROOT / relative
    done = read(folder / 'complete.json')
    assert not done.get('official_test_opened', done.get('test_opened', False))
    assert done.get('status', 'complete').startswith('complete')
    summary = read(folder / 'summary.json')
    expected = done.get('summary_sha256', done.get('files_sha256', {}).get('summary.json'))
    if expected:
        assert bind(folder / 'summary.json') == expected
    return folder, summary, done


def entry(name, folder, e, score_name=None, kind='full', description='', dependency=(), role='signal'):
    path = folder / (score_name or (e['candidate'] + '_scores.npz'))
    hashed = bind(path)
    if 'scores_sha256' in e:
        assert hashed == e['scores_sha256']
    return {'name': name, 'path': str(path.resolve()), 'scores_sha256': hashed,
        'kind': kind, 'thresholds': e['thresholds'],
        'expected_calibration': e.get('calibration', e.get('metrics', {}).get('calibration')),
        'description': description, 'dependencies': list(dependency), 'role': role}


def inventory():
    rows = []
    fixed_candidates = []
    convex = {}
    for target in ('large', 'nli', 'fava'):
        folder, summary, done = directory(f'results/{target}_fixed_convex_v1')
        assert done['candidate_count'] == 36
        convex[target] = (folder, summary)
        for family, candidates in summary['all_candidates'].items():
            assert len(candidates) == 6
            for e in candidates:
                fixed_candidates.append(entry(e['candidate'], folder, e,
                    description='Previously frozen probability-space convex candidate; no new weight or threshold here.',
                    dependency=(family, target), role='existing_fixed_candidate'))
                fixed_candidates[-1]['previously_selected_for_family'] = e == summary['selected'][family]
                fixed_candidates[-1]['original_weight'] = e.get('large_weight', e.get('target_weight'))
        family = 'semantic_claim__old_tree'
        for e in summary['all_candidates'][family]:
            if e.get('large_weight', e.get('target_weight')) == 1:
                target_e = summary['large_selected' if target == 'large' else 'target_selected']
                assert e['thresholds'] == target_e['thresholds']
                assert e['metrics']['calibration'] == target_e['calibration']
                rows.append(entry(target, folder, e,
                    description=f'One already-selected full-context {target} detector, endpoint1 reproduces its saved standalone score.',
                    dependency=('qa_human_supervision', 'extra_semantic_encoder')))
    folder, summary = convex['large']
    e = summary['selected']['semantic_claim__old_tree']
    assert e['candidate'] == CURRENT
    rows.insert(0, entry('current', folder, e,
        description='Current self-developed tree(semantic_claim,tail2) plus0.4 large. Not an independent source.',
        dependency=('semantic_claim', 'tail2', 'large'), role='current_reference'))

    folder = ROOT / 'results/minicheck_tail_all_docs_v3/tail2'
    done = read(folder / 'complete.json')
    assert done['status'] == 'complete_development_only' and not done['test_opened']
    e = done['selected']; assert e['epoch'] == 2
    score_name = f"epoch_{e['epoch']:02d}_scores.npz"
    assert bind(folder / score_name) == e['artifacts_sha256']['_scores.npz']
    rows.append(entry('tail2', folder, e, score_name, 'cal',
        'MiniCheck last-two-layer supervised all-doc adaptation, fixed selected epoch2.',
        ('minicheck_pretraining', 'qa_human_supervision')))

    folder, summary, done = directory('results/claim_pooling_v1')
    for name, method in [('semantic_claim', 'minicheck_hidden64_risk_tcn_w32'), ('harp_claim', 'full_lb_harp64_tcn')]:
        rows.append(entry(name, folder, summary['selected'][method],
            description='Existing self-developed automatic-claim propagation; derived from its raw sequence score, not another independent measurement.',
            dependency=(name.replace('_claim', '_sequence'), 'automatic_claims')))
    for name, folder_name, method in [('semantic_sequence', 'semantic_sequence_v1', 'minicheck_hidden64_risk_tcn_w32'),
                                     ('harp_sequence', 'sequence_full_v2', 'full_lb_harp64_tcn')]:
        folder, summary, done = directory('results/' + folder_name)
        e = summary['selected'][method]
        rows.append(entry(name, folder, e, method + f"/epoch_{e['epoch']:03d}_scores.npz",
            description='Previously selected local sequence adaptation, not a paper-faithful baseline.',
            dependency=('lookback', 'nll', 'harp') if name == 'harp_sequence' else ('minicheck_pretraining', 'generation_whitebox', 'qa_human_supervision')))
    folder, summary, done = directory('results/lookback_regularization_v2')
    rows.append(entry('lookback_local', folder, summary['selected']['lb_prefix_pre_header'],
        description='Existing4BPE standardized/weighted LR local adaptation; official8BPE Lookback is excluded from this axis.',
        dependency=('generation_attention', 'qa_human_supervision')))
    folder, summary, done = directory('semantic_baseline/cuda_variant/results')
    for key in ('minicheck_calibrated', 'minicheck_fixed'):
        path = folder / (key + '_scores.npz')
        assert bind(path) == done['files_sha256'][path.name]
        rows.append(entry(key, folder, summary['reports'][key], path.name,
            description='Identical frozen MiniCheck1-support score; only its already-saved threshold differs. No independent second signal.',
            dependency=('minicheck_pretraining', 'automatic_claims', 'document_chunks')))
    folder, summary, done = directory('results/ghost_matched_lr_v1')
    e = summary['selected']['ghost_only']
    rows.append(entry('ghost_lr', folder, e, description='Local4BPE GHOST4 LR; official answerRF has no window score and is excluded.',
        dependency=('generation_internal_geometry', 'qa_human_supervision')))
    folder, summary, done = directory('results/ghost_nll_nonlinear_v1')
    for name in ('nll_only', 'ghost_only', 'nll_ghost'):
        e = summary['methods'][name]
        rows.append(entry(name + '_tree', folder, e, name + '_scores.npz',
            description='Already completed fixed shallow-tree local adaptation. NLL is negative log likelihood, not entropy.',
            dependency=tuple(k for k in ('nll', 'ghost') if k in name)))
    # Raw NLL is an existing scalar feature. No saved standalone raw threshold exists;
    # report ranking only and exclude it from threshold unions rather than invent a cutoff.
    geometry = read(folder / 'geometry.json')
    rows.append({'name': 'raw_nll_ranking_only', 'path': str((folder / 'window_features.npy').resolve()),
        'scores_sha256': bind(folder / 'window_features.npy'), 'kind': 'raw_nll',
        'description': 'Column0 of the completed frozen5-column NLL+GHOST matrix; higher NLL fixed as riskier. No existing raw-score threshold.',
        'dependencies': ['nll_only_tree'], 'role': 'ranking_only', 'thresholds': None, 'expected_calibration': None})
    assert geometry['answer_ids'][-159:] == [a['response_id'] for a in q.lines(ROOT / 'data/answers_calibration.jsonl')]
    return rows, fixed_candidates


def boolean_metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
        'tn': int((~y & ~p).sum()), 'precision': tp / (tp + fp) if tp + fp else 0.,
        'recall': tp / (tp + fn), 'f1': 2 * tp / (2 * tp + fp + fn)}


def union_diagnostics(y, current, candidates):
    raw_union = np.logical_or.reduce([current, *candidates])
    # Only add gold-positive alerts, preserving all current true and false alerts.
    rescue_oracle = current | (y & raw_union)
    perfect_precision_oracle = y & raw_union
    return {'fixed_threshold_OR_not_oracle': boolean_metrics(y, raw_union),
        'gold_gated_rescue_oracle_current_FP_retained': boolean_metrics(y, rescue_oracle),
        'gold_gated_zero_FP_union_oracle': boolean_metrics(y, perfect_precision_oracle),
        'new_true_positive_windows': int((y & ~current & raw_union).sum()),
        'new_false_positive_windows': int((~y & ~current & raw_union).sum()),
        'missed_positive_windows': int((y & ~raw_union).sum())}


def tiny():
    y = np.array([1, 1, 0, 0], bool); a = np.array([1, 0, 1, 0], bool); b = np.array([0, 1, 0, 1], bool)
    r = union_diagnostics(y, a, [b])
    assert r['fixed_threshold_OR_not_oracle']['fp'] == 2
    assert r['gold_gated_rescue_oracle_current_FP_retained']['fp'] == 1
    assert r['gold_gated_rescue_oracle_current_FP_retained']['f1'] == .8
    assert r['gold_gated_zero_FP_union_oracle']['fp'] == 0 and r['new_true_positive_windows'] == 1
    return {'status': 'passed', 'OR_keeps_new_FP': True, 'rescue_oracle_keeps_current_FP': True,
        'zero_FP_oracle_explicit': True, 'no_fit_or_threshold_search': True}


def run():
    assert not (OUT / 'complete.json').exists(), 'Preserve completed diagnosis'
    start = time.perf_counter()
    bind(Path(__file__)); bind(Path(q.__file__))
    rows, fixed = inventory()
    cfg = {'version': 'current-fixed-signal-complementarity-v1',
        'scope': 'Only the same existing cal159/154 material groups/42241 lexical-eligible4rawBPE windows.',
        'source_policy': 'Completed saved predictions and their existing thresholds only. Never refit, retune, derive a new threshold, or use gold to construct actual scores.',
        'signal_roster': rows, 'existing_fixed_candidates': fixed,
        'fixed_candidate_policy': 'Report all108 previously frozen candidates of large/NLI/FAVA36 each. Weights were prespecified but old family/epoch/threshold selections used calibration. No new winner or weight is selected here; threshold-performance maxima are only descriptive.',
        'oracle': 'Report unfiltered OR with every new FP, then gold-positive-only rescue keeping original current FP, then a looser zero-FP gold-gated union. These are bounds on these fixed-threshold alert sets, not bounds on all information in the real-valued signals or deployed models.',
        'runs': 'Reuse exact170 gold lexical runs from current_span_position_diagnosis_v1; punctuation skipped, clean lexical ends run.70 current-all-missed runs fixed before examining alternatives. Run hit means any original4BPE alert touches its risk tokens, not complete semantic localization.',
        'excluded': {'lookback_official_span_v1': '8BPE source axis differs; no invented4BPE conversion.',
            'ghost_official_answer_rf_v1': 'Answer-only RF: no native4BPE scores; no broadcast.',
            'LUMINA': 'Not yet completed for this fixed inventory; do not read partial extraction.',
            'raw_NLL_threshold': 'No saved raw threshold; ranking only, not in union.'},
        'equivalent_results': 'Existing fixed complementarity compares only HARP/semantic with tail2, not current FN/run across this inventory. Reuse saved current runs and completed scores.',
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'official_test_opened': False,
        'limits': 'Repeated development calibration, selected model/signal multiplicity, dependent measurements, thresholds with unequal precision, and oracle access to gold preclude independent significance or achievable-F1 claims.'}
    if (OUT / 'protocol.json').exists():
        assert q.read(OUT / 'protocol.json') == cfg
    else:
        q.save(OUT / 'protocol.json', cfg)
    bind(OUT / 'protocol.json'); q.save(OUT / 'CPU_SELFCHECK.json', tiny())
    answers = q.lines(ROOT / 'data/answers_calibration.jsonl')
    tokens = q.lines(ROOT / 'data/tokens_calibration.jsonl')
    windows = q.lines(ROOT / 'data/windows_k4_calibration.jsonl')
    for name in ('answers', 'tokens', 'windows_k4'):
        bind(ROOT / f'data/{name}_calibration.jsonl')
    assert len(answers) == len(tokens) == 159 and len(windows) == 42241
    assert len({a['group_id'] for a in answers}) == 154
    assert [a['response_id'] for a in answers] == [t['response_id'] for t in tokens]
    aw = defaultdict(list); ttw = defaultdict(lambda: defaultdict(list))
    for i, w in enumerate(windows):
        aw[w['response_id']].append(i)
        for token in w['token_indices']: ttw[w['response_id']][token].append(i)
    answer_ix = [aw[a['response_id']] for a in answers]
    y = np.array([w['label'] for w in windows], bool); ay = np.array([a['label'] for a in answers], bool)
    assert int(y.sum()) == 5984 and int(ay.sum()) == 100
    previous = read(ROOT / 'results/current_span_position_diagnosis_v1/DIAGNOSIS.json')
    assert previous['candidate'] == CURRENT and previous['window_order_sha256'] == q.digest(windows)
    run_rows = previous['per_gold_run']; assert len(run_rows) == 170
    run_windows = [sorted({i for t in r['risk_raw_token_indices'] for i in ttw[r['response_id']][t]}) for r in run_rows]
    assert all(ix and y[ix].all() for ix in run_windows)

    def load(r):
        if r['kind'] == 'raw_nll':
            matrix = np.load(r['path'], mmap_mode='r', allow_pickle=False)
            assert matrix.shape == (696220, 5)
            s = np.asarray(matrix[-42241:, 0]).copy()
            return s, np.array([s[ix].max() for ix in answer_ix])
        with np.load(r['path'], allow_pickle=False) as z:
            if r['kind'] == 'cal':
                assert np.array_equal(z['cal_window_labels'], y) and np.array_equal(z['cal_answer_labels'], ay)
                assert np.array_equal(z['cal_answer_window_offsets'], np.r_[0, np.cumsum([len(ix) for ix in answer_ix])])
                s, a = z['cal_window_scores'].copy(), z['cal_answer_scores'].copy()
            else:
                assert len(z['window_scores']) in (210364, 696220)
                assert len(z['answer_scores']) in (793, 3839)
                s, a = z['window_scores'][-42241:].copy(), z['answer_scores'][-159:].copy()
        assert s.shape == (42241,) and np.isfinite(s).all()
        assert np.array_equal(np.array([s[ix].max() for ix in answer_ix]), a), r['name']
        ts = r['thresholds']
        wm = q.count(y, s, ts['window']['threshold']); am = q.count(ay, a, ts['answer']['threshold'])
        assert wm == r['expected_calibration']['windows'], (r['name'], 'window_metrics')
        assert am == r['expected_calibration']['answers'], (r['name'], 'answer_metrics')
        return s, a

    scores = {}; preds = {}; results = {}
    for r in rows:
        s, a = load(r); name = r['name']; scores[name] = s
        if r['thresholds'] is None:
            results[name] = {'auroc': float(q.roc_auc_score(y, s)), 'average_precision': float(q.average_precision_score(y, s)),
                'threshold': None, 'fixed_threshold_counts': None, 'reason': 'No saved threshold: ranking only.'}
            continue
        p = s >= r['thresholds']['window']['threshold']; preds[name] = p
        results[name] = {'window_metrics': q.count(y, s, r['thresholds']['window']['threshold']),
            'answer_metrics': q.count(ay, a, r['thresholds']['answer']['threshold']),
            'thresholds_unchanged': r['thresholds'], 'score_sha256': r['scores_sha256']}
    current = preds['current']
    assert int((y & ~current).sum()) == 2145 and int((~y & current).sum()) == 1300
    missed_run_ix = [i for i, ix in enumerate(run_windows) if not current[ix].any()]
    assert len(missed_run_ix) == 70
    assert all(bool(current[ix].any()) == r['any_window_hit'] for r, ix in zip(run_rows, run_windows))
    for name, p in preds.items():
        hits = [i for i in missed_run_ix if p[run_windows[i]].any()]
        results[name].update(current_FN_recovered_windows=int((y & ~current & p).sum()),
            new_FP_over_current=int((~y & ~current & p).sum()),
            current_TP_lost_if_replaced=int((y & current & ~p).sum()),
            current_FP_rejected_if_replaced=int((~y & current & ~p).sum()),
            current_all_missed_runs_hit=len(hits), current_all_missed_run_indices_hit=hits,
            all_gold_runs_hit=sum(bool(p[ix].any()) for ix in run_windows),
            pairwise_with_current=union_diagnostics(y, current, [p]))

    groups = {'current_only': [], 'current_component_scores': ['semantic_claim', 'tail2', 'large'],
        'extra_semantic_scores': ['semantic_sequence', 'semantic_claim', 'tail2', 'large', 'nli', 'fava', 'minicheck_calibrated', 'minicheck_fixed'],
        'generation_only_local_scores': ['lookback_local', 'harp_sequence', 'harp_claim', 'ghost_lr', 'nll_only_tree', 'ghost_only_tree', 'nll_ghost_tree'],
        'all_roster_threshold_scores': [n for n in preds if n != 'current']}
    unions = {}
    for group, names in groups.items():
        r = union_diagnostics(y, current, [preds[n] for n in names])
        union = np.logical_or.reduce([current] + [preds[n] for n in names])
        r['members_besides_current'] = names
        r['current_missed_runs_recovered'] = sum(bool(union[run_windows[i]].any()) for i in missed_run_ix)
        r['gold_runs_still_all_missed'] = [i for i, ix in enumerate(run_windows) if not union[ix].any()]
        unions[group] = r
    # Every candidate below already exists on disk. Do not make additional mixtures.
    fixed_rows = []
    for r in fixed:
        s, a = load(r); p = s >= r['thresholds']['window']['threshold']
        fixed_rows.append({'candidate': r['name'], 'original_weight': r['original_weight'],
            'previously_selected_for_family': r['previously_selected_for_family'],
            'window_metrics': r['expected_calibration']['windows'],
            'answer_metrics': r['expected_calibration']['answers'],
            'current_FN_recovered': int((y & ~current & p).sum()),
            'new_FP_over_current': int((~y & ~current & p).sum()),
            'current_missed_runs_recovered': sum(bool(p[run_windows[i]].any()) for i in missed_run_ix)})
    threshold_hashes = {n: q.digest(r['thresholds']) for n, r in zip([r['name'] for r in rows], rows)}
    required_tp = math.ceil(.75 * (int(y.sum()) + 1300) / (2 - .75))
    identical = []
    for i, name in enumerate(scores):
        for other in list(scores)[:i]:
            if np.array_equal(scores[name], scores[other]): identical.append([other, name])
    data = {'status': 'complete', 'candidate': CURRENT,
        'denominators': {'answers': 159, 'material_groups': 154, 'windows': 42241, 'positive_windows': 5984,
            'normal_windows': 36257, 'current_FN': 2145, 'gold_runs': 170, 'current_all_missed_runs': 70},
        'methods': results, 'union_diagnostics': unions, 'existing_fixed_candidates': fixed_rows,
        'existing_candidate_table': {'count': len(fixed_rows),
            'window_F1_at_least_075_count': sum(r['window_metrics']['f1'] >= .75 for r in fixed_rows),
            'window_F1_max_descriptive_not_new_selection': max(r['window_metrics']['f1'] for r in fixed_rows)},
        'required_additional_TP_for075_if_current1300FP_unchanged': required_tp - int((y & current).sum()),
        'identical_score_vectors': identical, 'threshold_sha256': threshold_hashes,
        'cal_window_order_sha256': q.digest(windows), 'cal_answer_order_sha256': q.digest([a['response_id'] for a in answers]),
        'per_gold_run': [dict(r, diagnosis_index=i, current_missed=i in missed_run_ix,
            touching_cal_window_indices=ix, hits_by_signal={n: int(p[ix].sum()) for n, p in preds.items()})
            for i, (r, ix) in enumerate(zip(run_rows, run_windows))],
        'source_sha256': SOURCES, 'new_fits': 0, 'new_thresholds': 0, 'test_opened': False,
        'GPU_used': False, 'seconds': time.perf_counter() - start}
    np.savez_compressed(OUT / 'aligned_calibration_scores.npz',
        names=np.asarray(list(scores)), scores=np.stack(list(scores.values())), labels=y,
        threshold_names=np.asarray(list(preds)), predictions=np.stack(list(preds.values())),
        current_FN=y & ~current, current_missed_run_indices=np.asarray(missed_run_ix))
    q.save(OUT / 'DIAGNOSIS.json', data)
    lines = ['# 当前候选：既有信号互补性上限诊断', '',
        '仅原cal159答、154资料组、42241个4-BPE窗；5984风险窗、36257正常窗。全部已有模型、权重、阈值保持，没有新增拟合或测试。各项是独立保存的分数，**不是统计独立的证据**。', '',
        '| 已存分数 | AUROC | AP | 原阈值F1 | 补中当前2145 FN | 新增FP | 命中原70全漏run |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        if 'window_metrics' not in r:
            lines.append(f"| {name} | {r['auroc']:.4f} | {r['average_precision']:.4f} | 无旧阈值 | N/A | N/A | N/A |")
            continue
        w = r['window_metrics']
        lines.append(f"| {name} | {w['auroc']:.4f} | {w['average_precision']:.4f} | {w['f1']:.4f} | {r['current_FN_recovered_windows']} | {r['new_FP_over_current']} | {r['current_all_missed_runs_hit']} |")
    lines += ['', '| 原阈值集合（均含current） | 新TP | 新FP | 实际OR F1 | gold只补真阳性、保留原FP的oracle F1 | gold再去全FP的oracle F1 | 补中70漏run |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in unions.items():
        lines.append(f"| {name} | {r['new_true_positive_windows']} | {r['new_false_positive_windows']} | {r['fixed_threshold_OR_not_oracle']['f1']:.4f} | {r['gold_gated_rescue_oracle_current_FP_retained']['f1']:.4f} | {r['gold_gated_zero_FP_union_oracle']['f1']:.4f} | {r['current_missed_runs_recovered']} |")
    lines += ['', f"若保留current原1300误报不变，到窗口F1 .75至少还需正确补中{data['required_additional_TP_for075_if_current1300FP_unchanged']}个FN。两种oracle都知道gold：前者能排除其他信号新增误报，后者还凭gold删除当前误报；均不是模型成绩或可部署融合。它们只是这些固定阈值报警集合的条件上限，不能证明连续分数信息的真正极限。", '',
        f"检查已冻结的large/NLI/FAVA共108个凸组合：达到窗口.75的为{data['existing_candidate_table']['window_F1_at_least_075_count']}个，表内最高{data['existing_candidate_table']['window_F1_max_descriptive_not_new_selection']:.6f}。所有候选的权重、阈值、补漏/新增误报已逐条保留；本轮没有选择新赢家。这些权重网格虽预先固定，旧epoch、族内权重与阈值已经用cal选择，不能称无cal选型的独立验证。", '',
        'current本身已经含semantic_claim、tail2和large；semantic_claim由同一semantic_sequence传播得到，HARP_claim由HARP_sequence传播得到。MiniCheck两个条目分数逐值相同，仅旧阈值不同；NLI/FAVA/large共享检测结构/QA监督，GHOST和NLL来自同一次Llama重放。因此不要把列数当作独立信息数量。',
        '正式Lookback是8-BPE，正式GHOST RF只有整答；本表不改变它们或广播成4-BPE。这里的Lookback/HARP/GHOST窗口模型均明确是历史本地适配。原始NLL没有现成独立阈值，只报既定高NLL方向的排序。',
        'run是沿词面token连续的gold风险段，标点不截断、正常词面会截断，可能合并多个原span；命中只表示一个4窗碰到风险token，不代表完整事实定位。全量170run的各信号命中数与70个全漏段均保留。',
        '结论须区分“某信号曾报警”与“能够可靠识别哪次报警正确”：高oracle可说明现有报警集合含补漏空间，不能证明不需新证据；低实际OR也不能证明必须换输入。没有新增证据对齐信号的训练实验，不能从本诊断作必要性因果结论。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    q.save(OUT / 'complete.json', {'status': 'complete', 'files_sha256': {n: q.sha(OUT / n) for n in
        ('diagnose.py', 'protocol.json', 'CPU_SELFCHECK.json', 'DIAGNOSIS.json', 'aligned_calibration_scores.npz', 'REPORT.md')},
        'new_fits': 0, 'new_thresholds': 0, 'GPU_used': False, 'official_test_opened': False})
    print('COMPLEMENTARITY_COMPLETE', data['seconds'], data['existing_candidate_table'], unions, flush=True)


if __name__ == '__main__':
    run()
