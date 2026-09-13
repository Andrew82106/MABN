"""Fixed-threshold calibration diagnostics; run only after all five GHOST LRs finish."""
from pathlib import Path
from collections import defaultdict
import argparse
import importlib.util
import json
import tempfile
import numpy as np

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
NEW = QA / 'results/ghost_matched_lr_v1'
OLD = QA / 'fit_expansion/llama_baselines_v1'
HELPER = QA / 'results/completed_group_bootstrap_v1/bootstrap_fixed.py'
spec = importlib.util.spec_from_file_location('_ghost_fixed_bootstrap_helpers', HELPER)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)  # Definitions only; never invokes its previous run().
read, rows, sha, write = b.read, b.rows, b.sha, b.write
FAMILIES = ('prefix_pre_header', 'legacy_lb_nll', 'prefix_post_header', 'harp64_legacy_lb_nll')
KINDS = ('Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Conflict')
METHODS = tuple(x + suffix for x in FAMILIES for suffix in ('__old', '__ghost')) + ('ghost_only',)
SEED, REPEATS = 20261010, 5000
NFIT, NTOTAL, NANSWER_FIT, NANSWERS = 653979, 696220, 3680, 3839


def protocol():
    return {'scope': 'Original exposed calibration159 answers,154 material groups,42241 raw4BPE windows only.',
        'families': list(FAMILIES), 'new_C': .0001, 'old_C': .0001,
        'contrasts': ['new GHOST appended minus corresponding original same-C control; four pairs'],
        'ghost_only': 'Describe existing metrics/type recall only; not an increment over a raw original family.',
        'selection': 'None: keep each saved candidate and its original separate window/answer thresholds; >= is positive.',
        'gate': 'No actual scores or new summaries read before new complete.json confirms all5 fits. Hash-bind all5 saved candidates and the four original same-C controls.',
        'sampling': 'Sorted material group_id; NumPy default_rng PCG64 seed20261010,5000 samples of154 groups with replacement; identical draws for all methods and both units.',
        'statistic': 'Micro F1 from grouped TP/FP/FN; new-minus-old paired delta;2.5/97.5 linear percentiles; zero denominator F1=0.',
        'types': 'Reuse original span_token_mapping risk_token_indices and label_type from calibration only. A window is in a type if its unchanged token_indices overlap that type. Denominators4086/814/997/109; overlaps allowed, OR must reproduce5984 binary-positive windows. Report recall only.',
        'checks': 'Original159/154/42241/100 positive answers/5984 positive windows; same window order, labels, group, each saved answer score equals all eligible-window max; saved counts/F1 match.',
        'limitations': 'Conditional interval on repeatedly-developed calibration; upstream threshold/model/feature selection bias is not included. Not independent significance or sealed-test evidence. No type-specific threshold or candidate selection.',
        'new_fits': 0, 'model_forward': False, 'GPU_used': False, 'test_opened': False}


def require_all_five(directory):
    path = Path(directory) / 'complete.json'
    if not path.exists():
        raise RuntimeError('WAIT: all five GHOST LR fits must complete; no partial results were read')
    complete = read(path)
    assert complete['status'] == 'complete' and complete['new_fits'] == 5
    assert complete['old_refits'] == 0 and complete['test_opened'] is False
    for method in (*FAMILIES, 'ghost_only'):
        stem = f'{method}__ghost_C0.0001'
        assert all(stem + suffix in complete['files_sha256'] for suffix in ('.pkl', '_scores.npz', '_result.json'))
    assert 'summary.json' in complete['files_sha256']
    return complete


def sampled_counts(counts, repeats=REPEATS, seed=SEED):
    ng = counts.shape[2]
    rng = np.random.default_rng(seed)
    assert type(rng.bit_generator).__name__ == 'PCG64'
    draws = rng.integers(0, ng, size=(repeats, ng))
    multiplicity = np.zeros((repeats, ng), np.int64)
    np.add.at(multiplicity, (np.repeat(np.arange(repeats), ng), draws.ravel()), 1)
    sample = np.einsum('rg,mugc->murc', multiplicity, counts)
    for j in (0, 1, repeats - 1):
        assert np.array_equal(sample[:, :, j, :], counts[:, :, draws[j], :].sum(axis=2))
    return draws, sample


def type_masks(tokens, windows):
    typed = {kind: defaultdict(set) for kind in KINDS}
    for t in tokens:
        union = set()
        for entry in t['span_token_mapping']:
            kind = t['original_labels'][entry['span_index']]['label_type']
            assert kind in typed
            ix = set(entry['risk_token_indices'])
            assert all(0 <= j < len(t['risk_mask']) and t['risk_mask'][j] for j in ix)
            typed[kind][t['response_id']].update(ix)
            union.update(ix)
        assert union == set(np.flatnonzero(t['risk_mask']).tolist())
    return {kind: np.asarray([bool(set(w['token_indices']) & typed[kind][w['response_id']])
                             for w in windows]) for kind in KINDS}


def self_test():
    # Same group can own multiple answers and overlapping windows; threshold is fixed.
    y = np.array([1, 0, 1, 0, 1, 0])
    owner = np.array([0, 0, 1, 1, 2, 2])
    score = np.array([.5, .6, .7, .1, .2, .3])
    c = b.group_counts(y, score, .5, owner, 3)
    assert c.tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
    m = b.metric(y, score, .5)
    assert [m[k] for k in ('tp', 'fp', 'fn', 'tn')] == [2, 1, 1, 2]
    counts = np.stack([np.stack([c, c]), np.stack([c, c])])
    draws, sample = sampled_counts(counts, 25)
    for j in range(25):
        ix = np.concatenate([np.flatnonzero(owner == group) for group in draws[j]])
        direct = b.metric(y[ix], score[ix], .5)
        assert np.array_equal(sample[0, 0, j], [direct[k] for k in ('tp', 'fp', 'fn')])
    assert np.array_equal(b.f1(sample)[0] - b.f1(sample)[1], np.zeros((2, 25)))
    assert b.f1([0, 0, 0]) == 0
    tokens = [{'response_id': 'a', 'risk_mask': [0, 1, 1, 0],
               'original_labels': [{'label_type': KINDS[0]}, {'label_type': KINDS[2]}],
               'span_token_mapping': [{'span_index': 0, 'risk_token_indices': [1, 2]},
                                      {'span_index': 1, 'risk_token_indices': [2]}]}]
    windows = [{'response_id': 'a', 'token_indices': [0, 1]},
               {'response_id': 'a', 'token_indices': [1, 2, 3]}]
    masks = type_masks(tokens, windows)
    assert masks[KINDS[0]].tolist() == [True, True] and masks[KINDS[2]].tolist() == [False, True]
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        try:
            require_all_five(tmp)
            raise AssertionError('Missing complete gate did not stop')
        except RuntimeError:
            pass
        write(Path(tmp) / 'complete.json', {'status': 'complete', 'new_fits': 4})
        try:
            require_all_five(tmp)
            raise RuntimeError('Partial five-fit gate did not stop')
        except AssertionError:
            pass
    report = {'status': 'passed', 'group_resampling_equals_repeated_raw_rows': True,
        'same_group_draws_both_units_and_methods': True, 'threshold_tie_ge_checked': True,
        'identity_delta_zero': True, 'zero_denominator_checked': True,
        'overlapping_type_masks_checked': True, 'missing_and_partial_complete_gates_checked': True,
        'actual_GHOST_features_or_scores_read': False, 'model_fit': False, 'GPU_used': False}
    write(OUT / 'CPU_SELFCHECK.json', report)
    return report


def prepare():
    assert not (OUT / 'design_freeze.json').exists(), 'Existing diagnostic design is immutable'
    self_test()
    write(OUT / 'protocol.json', protocol())
    files = [Path(__file__), HELPER, QA / 'src/diagnose_completed_tail.py',
             QA / 'src/run_ghost_matched_lr_v1.py', NEW / 'protocol.json',
             OLD / 'complete.json', OLD / 'summary.json', OUT / 'protocol.json', OUT / 'CPU_SELFCHECK.json']
    files += [QA / f'data/{name}_calibration.jsonl' for name in ('answers', 'tokens', 'windows_k4')]
    # Only static code, old completed artifacts and frozen labels are hashed here.
    # No new complete/summary/scores/features are opened by preparation.
    write(OUT / 'design_freeze.json', {'status': 'prepared_not_run',
        'files_sha256': {str(p.resolve()): sha(p) for p in files},
        'actual_GHOST_features_or_scores_read': False, 'new_fits': 0, 'GPU_used': False, 'test_opened': False})
    (OUT / 'PREPARATION.md').write_text(
        '已冻结四项新GHOST减原同C对照；GHOST-only仅描述。固定原阈值，原cal159答/154组/42241窗，5000次资料组配对重采样，seed20261010。四类型只报固定分母召回，允许重叠。\n\n'
        'CPU小算例通过：分组计数等于原行重复、同分阈值取≥、自身差值0、类型重叠、缺少/不足五fit完整门禁。尚未读活跃新特征或任何新拟合结果，未运行真实诊断。\n\n'
        '运行入口：diagnose.py run。只有上游完整五fit的complete.json通过才继续；不自动等待、不训练。区间未计入既有cal选型偏差，不是独立显著性证明。\n', encoding='utf-8')
    print('GHOST_DIAGNOSTICS_PREPARED_NO_NEW_RESULTS_READ', flush=True)


def run():
    complete = require_all_five(NEW)  # FIRST access to any actual new-fit artifact.
    assert not (OUT / 'complete.json').exists(), 'Do not overwrite completed diagnosis'
    frozen = read(OUT / 'design_freeze.json')
    assert read(OUT / 'protocol.json') == protocol()
    for p, digest in frozen['files_sha256'].items():
        assert sha(p) == digest, p
    consumed = {}

    def bound(path, digest=None):
        got = sha(path)
        if digest is not None:
            assert got == digest, str(path)
        consumed[str(Path(path).resolve())] = got
        return path

    bound(NEW / 'complete.json')
    ns = read(bound(NEW / 'summary.json', complete['files_sha256']['summary.json']))
    source = read(bound(NEW / 'source_snapshot.json', complete['files_sha256']['source_snapshot.json']))
    for name in ('answers', 'tokens', 'windows_k4'):
        path = QA / f'data/{name}_calibration.jsonl'
        bound(path, source['files_sha256'][str(path.resolve())])
    oc = read(bound(OLD / 'complete.json'))
    os = read(bound(OLD / 'summary.json', oc['files_sha256']['summary.json']))
    assert ns['new_fits'] == 5 and ns['old_refits'] == 0 and ns['test_opened'] is False
    assert os['fit_count'] == oc['fit_count'] == 20 and oc['official_test_opened'] is False
    assert set(ns['selected']) == set(ns['all_candidates']) == set((*FAMILIES, 'ghost_only'))
    answers = rows(QA / 'data/answers_calibration.jsonl')
    windows = rows(QA / 'data/windows_k4_calibration.jsonl')
    tokens = rows(QA / 'data/tokens_calibration.jsonl')
    assert len(answers) == len(tokens) == 159 and len(windows) == 42241
    assert all(x['partition'] == 'calibration' and x['eligible'] and x['label'] in (0, 1) for x in answers + windows)
    ai = {a['response_id']: j for j, a in enumerate(answers)}
    assert len(ai) == 159 and [t['response_id'] for t in tokens] == list(ai)
    for a, t in zip(answers, tokens):
        assert t['partition'] == 'calibration' and t['group_id'] == a['group_id']
        assert t['answer_sha256'] == a['answer_sha256']
    groups = sorted({a['group_id'] for a in answers}); assert len(groups) == 154
    gi = {g: j for j, g in enumerate(groups)}
    for w in windows:
        assert w['group_id'] == answers[ai[w['response_id']]]['group_id']
        token = tokens[ai[w['response_id']]]
        ix = w['token_indices']
        assert ix == list(range(w['token_start'], w['token_end']))
        assert len(ix) == min(4, len(token['token_ids'])) and any(token['lexical_mask'][j] for j in ix)
        assert w['label'] == int(any(token['risk_mask'][j] for j in ix))
    assert len({(w['response_id'], w['token_start'], w['token_end']) for w in windows}) == 42241
    owners = {'windows': np.array([gi[w['group_id']] for w in windows]),
              'answers': np.array([gi[a['group_id']] for a in answers])}
    gold = {'windows': np.array([w['label'] for w in windows]), 'answers': np.array([a['label'] for a in answers])}
    assert gold['windows'].sum() == 5984 and gold['answers'].sum() == 100
    wa = np.array([ai[w['response_id']] for w in windows])
    masks = type_masks(tokens, windows)
    assert [int(masks[k].sum()) for k in KINDS] == [4086, 814, 997, 109]
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), gold['windows'].astype(bool))
    entries = {}
    for method in (*FAMILIES, 'ghost_only'):
        e = ns['selected'][method]
        assert len(ns['all_candidates'][method]) == 1 and ns['all_candidates'][method][0] == e
        assert e['C'] == .0001 and e['candidate'] == f'{method}__ghost_C0.0001'
        entries['ghost_only' if method == 'ghost_only' else method + '__ghost'] = (e, NEW, complete)
        if method != 'ghost_only':
            old = os['selected'][method]
            assert old['C'] == .0001 and old['candidate'] == f'{method}_C0.0001'
            assert ns['old_controls'][method]['selected'] == old
            entries[method + '__old'] = (old, OLD, oc)
    counts = np.zeros((9, 2, 154, 3), np.int64)
    descriptions, predictions = {}, {}
    for mi, name in enumerate(METHODS):
        e, directory, done = entries[name]; stem = e['candidate']
        for suffix in ('.pkl', '_scores.npz', '_result.json'):
            bound(directory / (stem + suffix), done['files_sha256'][stem + suffix])
        assert read(directory / (stem + '_result.json')) == e
        assert sha(directory / (stem + '.pkl')) == e['model_sha256']
        assert sha(directory / (stem + '_scores.npz')) == e['scores_sha256']
        with np.load(directory / (stem + '_scores.npz'), allow_pickle=False) as z:
            assert z['window_scores'].shape == (NTOTAL,) and z['answer_scores'].shape == (NANSWERS,)
            scores = {'windows': z['window_scores'][NFIT:].copy(), 'answers': z['answer_scores'][NANSWER_FIT:].copy()}
        maximum = np.full(159, -np.inf); np.maximum.at(maximum, wa, scores['windows'])
        assert np.array_equal(maximum, scores['answers']), (name, 'answermax')
        description = {'candidate': stem, 'C': e['C'], 'thresholds': e['thresholds'], 'metrics': {}, 'type_recall': {}}
        for ui, unit in enumerate(('windows', 'answers')):
            s = scores[unit]; assert s.shape == gold[unit].shape and np.isfinite(s).all()
            t = e['thresholds']['window' if ui == 0 else 'answer']['threshold']
            m = b.metric(gold[unit], s, t)
            assert all(v == e['metrics']['calibration'][unit][k] for k, v in m.items()), (name, unit)
            counts[mi, ui] = b.group_counts(gold[unit], s, t, owners[unit], 154)
            assert np.array_equal(counts[mi, ui].sum(0), [m[k] for k in ('tp', 'fp', 'fn')])
            description['metrics'][unit] = m
        pred = scores['windows'] >= e['thresholds']['window']['threshold']; predictions[name] = pred
        for kind, mask in masks.items():
            description['type_recall'][kind] = {'denominator': int(mask.sum()), 'detected': int(pred[mask].sum()), 'recall': float(pred[mask].mean())}
        descriptions[name] = description
    draws, sample = sampled_counts(counts)
    boot = b.f1(sample); point = b.f1(counts.sum(axis=2)); contrasts = {}
    for family in FAMILIES:
        left, right = family + '__ghost', family + '__old'
        li, ri = METHODS.index(left), METHODS.index(right)
        value = {'left': left, 'right': right, 'by_type': {}}
        for ui, unit in enumerate(('windows', 'answers')):
            delta = boot[li, ui] - boot[ri, ui]; lo, hi = np.percentile(delta, [2.5, 97.5], method='linear')
            value[unit] = {'point_F1_difference': float(point[li, ui] - point[ri, ui]),
                'percentile95': [float(lo), float(hi)], 'contains_zero': bool(lo <= 0 <= hi),
                'count_difference': {k: descriptions[left]['metrics'][unit][k] - descriptions[right]['metrics'][unit][k]
                                     for k in ('tp', 'fp', 'fn', 'tn')}}
        for kind, mask in masks.items():
            pn, po = predictions[left], predictions[right]
            value['by_type'][kind] = {'denominator': int(mask.sum()),
                'recall_difference': float(pn[mask].mean() - po[mask].mean()),
                'newly_detected': int(np.count_nonzero(mask & pn & ~po)),
                'lost_detection': int(np.count_nonzero(mask & ~pn & po)),
                'both_missed': int(np.count_nonzero(mask & ~pn & ~po))}
        contrasts[family] = value
    np.savez_compressed(OUT / 'paired_group_counts.npz', group_ids=np.array(groups), methods=np.array(METHODS),
        units=np.array(['windows', 'answers']), count_order=np.array(['tp', 'fp', 'fn']),
        group_counts=counts, draws=draws.astype(np.int16), bootstrap_F1=boot)
    report = {'status': 'passed', 'fixed_methods': descriptions, 'contrasts': contrasts,
        'denominators': {'answers': 159, 'groups': 154, 'windows': 42241, 'positive_windows': 5984,
                         'type_windows': {k: int(v.sum()) for k, v in masks.items()}},
        'zero_F1_denominators': int(np.count_nonzero(2 * sample[..., 0] + sample[..., 1] + sample[..., 2] == 0)),
        'files_sha256': consumed, 'design_freeze_sha256': sha(OUT / 'design_freeze.json'),
        'bootstrap_sha256': sha(OUT / 'paired_group_counts.npz'), 'protocol': protocol(),
        'model_reselected': False, 'threshold_reselected': False, 'new_fits': 0, 'GPU_used': False, 'test_opened': False}
    write(OUT / 'DIAGNOSTICS.json', report)
    text = ['# GHOST固定阈值诊断', '', '仅已反复开发的cal159答/154资料组/42241窗；5000次成组配对重采样，seed20261010。原阈值不重选。', '',
            '| 新GHOST减原同C=1e-4 | 窗口ΔF1 [95%条件区间] | 整答ΔF1 [95%条件区间] |', '|---|---:|---:|']
    for family, value in contrasts.items():
        cells = []
        for unit in ('windows', 'answers'):
            x = value[unit]; lo, hi = x['percentile95']
            cells.append(f"{x['point_F1_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        text.append('| ' + family + ' | ' + ' | '.join(cells) + ' |')
    g = descriptions['ghost_only']['metrics']
    text += ['', f"GHOST-only仅描述：窗口F1 {g['windows']['f1']:.6f}，整答F1 {g['answers']['f1']:.6f}，不作为原家族增益。", '',
             '全部原计数、TP/FP/FN变化与四类型召回/新增/丢失检出保存在DIAGNOSTICS.json。类型窗口可重叠，只报召回，不造类型F1。', '',
             '区间未计入既有模型/阈值/特征选择偏差，不是独立显著性证明；跨零也不证明等效。没有重新拟合、推理、选阈值或读取测试。']
    (OUT / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    write(OUT / 'complete.json', {'status': 'complete_conditional_calibration_diagnostic',
        'diagnostics_sha256': sha(OUT / 'DIAGNOSTICS.json'), 'report_sha256': sha(OUT / 'REPORT.md'),
        'new_fits': 0, 'GPU_used': False, 'test_opened': False})
    print(json.dumps({'status': 'passed', 'contrasts': contrasts}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'self-test', 'run'))
    args = parser.parse_args()
    {'prepare': prepare, 'self-test': self_test, 'run': run}[args.stage]()
