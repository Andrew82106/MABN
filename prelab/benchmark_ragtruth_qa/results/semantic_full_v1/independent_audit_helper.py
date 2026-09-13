"""Bounded independent CPU arithmetic audit; no project imports or fitting."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import pickle
import sys
import time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
OLD = QA/'results/semantic_hidden_v1'
BASE = QA/'results/development_v1'
NFIT, NTOTAL, BATCH = 168123, 210364, 16384
CS = [1e-5, 1e-4, .001, .01, .1]
HASHES = {}


def sha(path):
    p = Path(path).resolve()
    assert p.is_relative_to(QA)
    assert not any(s in str(p.relative_to(QA)).lower() for s in ('sealed', 'withheld', 'test'))
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    value = h.hexdigest(); HASHES[str(p.relative_to(QA))] = value
    return value


def read(p):
    sha(p)
    return json.loads(Path(p).read_text('utf-8'))


class State:
    def __setstate__(self, s): self.__dict__.update(s)


class InertPickle(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('sklearn.'): return State
        if module.startswith('numpy') or module in ('builtins', 'collections'):
            return super().find_class(module, name)
        raise ValueError((module, name))


def unpickle(p):
    sha(p)
    with Path(p).open('rb') as f: return InertPickle(f).load()


def near(a, b, tolerance=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    assert error <= tolerance, (error, tolerance)
    return error


def choose(y, s):
    # Enumerate ascending score buckets, using counts rather than runner sorting.
    unique, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    positives = np.bincount(inv, weights=y, minlength=len(unique)).astype(np.int64)
    tp = np.r_[np.cumsum(positives[::-1])[::-1], 0]
    n = np.r_[np.cumsum(counts[::-1])[::-1], 0]
    cutoffs = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    f1 = 2*tp/(n+int(y.sum()))
    precision = np.divide(tp, n, out=np.zeros(len(n)), where=n != 0)
    k = max(range(len(cutoffs)), key=lambda i: (f1[i], precision[i], cutoffs[i]))
    return dict(threshold=float(cutoffs[k]), f1=float(f1[k]), precision=float(precision[k]),
                rows=len(y), positive=int(y.sum()))


def metrics(y, s, threshold):
    pred = s >= threshold
    tp = int(y[pred].sum()); fp = int(pred.sum())-tp
    fn = int(y.sum())-tp; tn = len(y)-tp-fp-fn
    unique, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(unique)); neg = counts-pos
    auc = np.sum(pos*(np.cumsum(neg)-.5*neg))/(y.sum()*(len(y)-y.sum()))
    ap = np.sum(pos[::-1]/y.sum()*np.cumsum(pos[::-1])/np.cumsum(counts[::-1]))
    return dict(n=len(y), positive=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn), auroc=float(auc), average_precision=float(ap))


def metric_error(a, b):
    assert a.keys() == b.keys()
    return near([a[k] for k in a], [b[k] for k in a])


def moments(raw, base):
    # Independent corrected two-pass batch variance plus weighted Chan merge.
    # Match float32 weight and previous-count rounding of installed sklearn1.6.1.
    count = 0.; mean = np.zeros(raw.shape[1]); var = mean.copy()
    for left in range(0, NFIT, BATCH):
        right = min(left+BATCH, NFIT)
        x = raw[left:right].astype(np.float64)
        w = base[left:right].astype(np.float32).astype(np.float64)
        mass = w.sum(); center = w@x/mass
        x -= center
        correction = w@x
        x *= x
        m2 = w@x-correction*correction/mass
        count = float(np.float32(count)); total = count+mass
        merged = (count*mean+mass*center)/total
        var = (count*var+m2+(center-mean)**2*count*mass/total)/total
        mean, count = merged, total
    eps = np.finfo(float).eps
    constant = var <= count*eps*var+(count*mean*eps)**2
    scale = np.sqrt(var); scale[constant] = 1
    return mean, var, scale, count


def main():
    clock = time.perf_counter()
    sha(Path(__file__))
    complete = read(OUT/'complete.json'); protocol = read(OUT/'protocol.json')
    summary = read(OUT/'summary.json'); started = read(OUT/'started.json')
    assert not complete['official_test_opened']
    for name, expected in complete['files_sha256'].items(): assert sha(OUT/name) == expected
    assert started['protocol_sha256'] == sha(OUT/'protocol.json')
    assert datetime.fromisoformat(started['utc']) < datetime.fromisoformat(complete['utc'])
    assert protocol['script_sha256'] == sha(QA/'src/run_semantic_full.py')
    assert protocol['base_code_sha256'] == sha(QA/'src/run_development.py')
    assert protocol['source_complete_sha256'] == sha(OLD/'complete.json')
    assert protocol['source_preparation_sha256'] == sha(OLD/'preparation_complete.json')
    assert protocol['C'] == CS and protocol['new_fits'] == 7
    prep = read(OLD/'preparation_complete.json')
    for name, expected in prep['files_sha256'].items(): assert sha(OLD/name) == expected
    matrix = read(OUT/'matrix_manifest.json')
    assert matrix['file_sha256'] == sha(OUT/'window_hidden1024.npy')
    assert matrix['source_preparation_sha256'] == sha(OLD/'preparation_complete.json')
    assert not matrix['labels_used_to_construct_features']
    print('AUDIT_SOURCE_HASHES_PASSED', flush=True)

    ix = read(BASE/'score_index.json'); windows, answers = ix['windows'], ix['answers']
    assert len(windows) == NTOTAL and len(answers) == 793
    assert len({w['window_id'] for w in windows}) == NTOTAL
    assert len({a['answer_id'] for a in answers}) == 793
    y = np.asarray([w['label'] for w in windows], int)
    ya = np.asarray([a['label'] for a in answers], int)
    byanswer = defaultdict(list)
    for j, w in enumerate(windows): byanswer[w['answer_id']].append(j)
    ai = [np.asarray(byanswer[a['answer_id']], int) for a in answers]
    assert np.array_equal(np.concatenate(ai), np.arange(NTOTAL))
    def answer_max(scores): return np.asarray([scores[ids].max() for ids in ai])
    bounds = {'fit': (0, NFIT, 0, 634), 'calibration': (NFIT, NTOTAL, 634, 793)}
    denominators = {}
    for part, (lo, hi, al, ar) in bounds.items():
        assert all(w['partition'] == part for w in windows[lo:hi])
        assert all(a['partition'] == part for a in answers[al:ar])
        denominators[part] = dict(windows=hi-lo, positive_windows=int(y[lo:hi].sum()),
            answers=ar-al, positive_answers=int(ya[al:ar].sum()),
            groups=len({a['group_id'] for a in answers[al:ar]}))
    assert not ({a['group_id'] for a in answers[:634]} & {a['group_id'] for a in answers[634:]})
    assert denominators == {'fit': dict(windows=168123, positive_windows=21477, answers=634,
        positive_answers=328, groups=615), 'calibration': dict(windows=42241, positive_windows=5984,
        answers=159, positive_answers=100, groups=154)}

    token_index = read(OLD/'token_index.json')['answers']; ti = {a['response_id']: a for a in token_index}
    assert len(ti) == 793
    left = 0
    for j, (a, r) in enumerate(zip(answers, token_index)):
        assert a['answer_id'] == r['response_id'] and a['partition'] == r['partition']
        assert a['group_id'] == r['group_id'] and r['left'] == left
        assert r['right']-r['left'] == r['token_count']; left = r['right']
        if j == 633: assert left == 170361
    assert left == 213159
    # Read only explicitly allowed fit/cal gold exports, checking the saved row order.
    wi = np.empty((NTOTAL, 4), np.int64); lengths = np.empty(NTOTAL, np.int8)
    j = 0; nonlex = 0
    for part in ('fit', 'calibration'):
        p = QA/'data'/f'windows_k4_{part}.jsonl'; sha(p)
        with p.open(encoding='utf-8') as f:
            for line in f:
                w = json.loads(line); saved = windows[j]; r = ti[w['response_id']]
                for k in ('window_id', 'answer_id', 'group_id', 'partition', 'label'): assert w[k] == saved[k]
                assert w['eligible'] and w['lexical_token_indices'] and w['k'] == 4 and w['stride'] == 1
                ids = w['token_indices']; n = r['token_count']; start = w['token_start']
                assert ids == list(range(start, min(start+4, n)))
                assert len(ids) == min(4, n) and 0 <= start < max(1, n-3)
                assert w['window_id'] == f"{w['response_id']}__k4_{start:05d}"
                assert r['partition'] == part and r['group_id'] == w['group_id']
                wi[j, :len(ids)] = np.asarray(ids)+r['left']; lengths[j] = len(ids)
                nonlex += len(ids)-len(w['lexical_token_indices']); j += 1
    assert j == NTOTAL
    for part, (lo, hi, al, ar) in bounds.items():
        p = QA/'data'/f'answers_{part}.jsonl'; sha(p)
        with p.open(encoding='utf-8') as f: goldanswers = [json.loads(line) for line in f]
        assert len(goldanswers) == ar-al
        for a, s in zip(goldanswers, answers[al:ar]):
            for k in ('answer_id', 'group_id', 'partition', 'label'): assert a[k] == s[k]
            assert a['eligible'] and a['eligible_window_count'] == len(byanswer[a['answer_id']])
            assert a['label'] == int(bool(a['original_labels']))

    source = np.load(OLD/'matrices/mapped_hidden1024.npy', mmap_mode='r', allow_pickle=False)
    full = np.load(OUT/'window_hidden1024.npy', mmap_mode='r', allow_pickle=False)
    token64 = np.load(OLD/'matrices/token_hidden64.npy', mmap_mode='r', allow_pickle=False)
    raw64 = np.load(OLD/'matrices/window_hidden64.npy', mmap_mode='r', allow_pickle=False)
    assert source.shape == (213159, 1024) and full.shape == (NTOTAL, 1024)
    assert token64.shape == (213159, 64) and raw64.shape == (NTOTAL, 64)
    assert source.dtype == full.dtype == token64.dtype == raw64.dtype == np.float32
    mapping_errors = {}
    for name, tokens, raw in [('full1024', source, full), ('reused64', token64, raw64)]:
        worst = 0.; exact = True
        for l in range(0, NTOTAL, 512):
            r = min(l+512, NTOTAL)
            for n in np.unique(lengths[l:r]):
                rows = np.flatnonzero(lengths[l:r] == n)+l
                expected = tokens[wi[rows, :n]].mean(axis=1)
                actual = np.asarray(raw[rows]); exact &= np.array_equal(expected, actual)
                worst = max(worst, near(expected, actual, 2e-6))
        assert exact
        mapping_errors[name] = dict(max_abs=worst, all_windows_exact=bool(exact))
    print('AUDIT_ALL_210364_WINDOW_MEANS_EXACT', flush=True)

    # Reuse contract: frozen PCA metadata and source hashes only, no PCA refitting.
    pca = unpickle(OLD/'hidden_pca.pkl')
    assert pca['components'].shape == (64, 1024) and pca['mean'].shape == (1024,)
    assert pca['fit_answers'] == 634 and pca['fit_groups'] == 615 and pca['sample_count'] == 20288
    assert len(pca['sample']) == len(pca['sample_weights']) == 20288
    assert all(ti[r['response_id']]['partition'] == 'fit' and
               ti[r['response_id']]['group_id'] == r['group_id'] and
               0 <= r['token_index'] < ti[r['response_id']]['token_count'] for r in pca['sample'])
    sha(BASE/'training_weights.npz')
    with np.load(BASE/'training_weights.npz', allow_pickle=False) as z:
        base = z['base_weights']; loss = z['loss_weights']; fit_y = z['y']; factors = z['class_factors']
    assert base.shape == loss.shape == fit_y.shape == (NFIT,)
    assert np.array_equal(fit_y, y[:NFIT])
    counts = Counter(w['answer_id'] for w in windows[:NFIT]); groups = defaultdict(set)
    group_ix = defaultdict(list)
    for j, w in enumerate(windows[:NFIT]):
        groups[w['group_id']].add(w['answer_id']); group_ix[w['group_id']].append(j)
    expected_base = np.asarray([NFIT/(615*len(groups[w['group_id']])*counts[w['answer_id']]) for w in windows[:NFIT]])
    base_error = near(expected_base, base)
    factors_expected = base.sum()/(2*np.bincount(fit_y, weights=base, minlength=2))
    factor_error = near(factors_expected, factors)
    expected_loss = base*factors_expected[fit_y]
    for ids in group_ix.values(): expected_loss[ids] *= (NFIT/615)/expected_loss[ids].sum()
    loss_error = near(expected_loss, loss)
    weights_report = dict(base_max_abs=base_error, class_factors_max_abs=factor_error,
                         loss_max_abs=loss_error, base_mass=float(base.sum()), loss_mass=float(loss.sum()))

    scaler_report = {}; candidates = []; selected = {}; replay_report = {}
    overall_score_error = overall_metric_error = 0.
    old64 = unpickle(OLD/'minicheck_hidden64_C0.001.pkl')['scaler']
    for method, raw in [('minicheck_hidden64', raw64), ('minicheck_hidden1024', full)]:
        objects = [unpickle(OUT/f'{method}_C{c:g}.pkl') for c in CS]; sc = objects[0]['scaler']
        if method == 'minicheck_hidden64':
            for field in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
                assert np.array_equal(getattr(sc, field), getattr(old64, field))
            scaler_report[method] = dict(frozen_scaler_reused_exact=True)
        else:
            mean, var, scale, count = moments(raw, base)
            scaler_report[method] = dict(mean_max_abs=near(mean, sc.mean_, 2e-11),
                variance_max_abs=near(var, sc.var_, 2e-10), scale_max_abs=near(scale, sc.scale_, 2e-11),
                count_max_abs=near(count, sc.n_samples_seen_), effective_count=float(count),
                fit_rows=NFIT, blocks=(NFIT+BATCH-1)//BATCH,
                convention='float32 weights and previous-count rounding; corrected two-pass batch variance plus weighted Chan merge')
        scores = np.empty((5, NTOTAL), np.float64)
        for obj, c in zip(objects, CS):
            assert obj['method'] == method and obj['C'] == c and obj['width'] == raw.shape[1]
            assert obj['fit_only'] and obj['fit_rows'] == NFIT and obj['fit_groups'] == 615
            assert obj['weights_sha256'] == sha(BASE/'training_weights.npz')
            assert obj['protocol_sha256'] == sha(OUT/'protocol.json')
            for field in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
                assert np.array_equal(getattr(obj['scaler'], field), getattr(sc, field))
            model = obj['model']
            assert model.C == c and model.solver == 'liblinear' and model.penalty == 'l2'
            assert model.random_state == 20260924 and model.max_iter == 2000 and max(model.n_iter_) < 2000
            assert np.array_equal(model.classes_, [0, 1]) and model.coef_.shape == (1, raw.shape[1])
        for left in range(0, NTOTAL, BATCH):
            right = min(left+BATCH, NTOTAL)
            x = np.array(raw[left:right], dtype=np.float32, copy=True)
            x -= sc.mean_; x /= sc.scale_
            for k, obj in enumerate(objects):
                model = obj['model']; scores[k, left:right] = expit((x@model.coef_.T+model.intercept_).ravel())
        family = []
        for k, (obj, c) in enumerate(zip(objects, CS)):
            name = f'{method}_C{c:g}'; result = read(OUT/(name+'_result.json'))
            assert result == summary['all_candidates'][method][k]
            assert result['model_sha256'] == sha(OUT/(name+'.pkl'))
            assert result['scores_sha256'] == sha(OUT/(name+'_scores.npz'))
            with np.load(OUT/(name+'_scores.npz'), allow_pickle=False) as z:
                actual, aa = z['window_scores'], z['answer_scores']
            assert actual.shape == (NTOTAL,) and aa.shape == (793,)
            score_error = near(scores[k], actual); overall_score_error = max(overall_score_error, score_error)
            expected_aa = answer_max(actual); assert np.array_equal(expected_aa, aa)
            ts = dict(window=choose(y[NFIT:], actual[NFIT:]), answer=choose(ya[634:], aa[634:]))
            assert ts == result['thresholds'] == obj['thresholds']
            w, a = ts['window'], ts['answer']; key = [min(w['f1'], a['f1']), w['f1'], w['precision'], -c]
            assert key == result['selection_key'] == list(obj['selection_key'])
            computed = {}
            for part, (lo, hi, al, ar) in bounds.items():
                computed[part] = dict(windows=metrics(y[lo:hi], actual[lo:hi], w['threshold']),
                                      answers=metrics(ya[al:ar], aa[al:ar], a['threshold']))
                for level in ('windows', 'answers'):
                    overall_metric_error = max(overall_metric_error,
                        metric_error(computed[part][level], result['metrics'][part][level]))
            reuse = method == 'minicheck_hidden64' and c in [.001, .01, .1]
            assert obj['source'] == result['source'] == ('replayed_frozen_model' if reuse else 'new_fit')
            if reuse:
                old = unpickle(OLD/(name+'.pkl'))
                for field in ('coef_', 'intercept_', 'classes_', 'n_iter_'):
                    assert np.array_equal(getattr(obj['model'], field), getattr(old['model'], field))
                assert old['thresholds'] == ts
                sha(OLD/(name+'_scores.npz'))
                with np.load(OLD/(name+'_scores.npz'), allow_pickle=False) as z:
                    old_error = near(actual, z['window_scores'])
                    unequal = int(np.sum(actual != z['window_scores']))
                    assert np.array_equal(aa, z['answer_scores'])
                assert summary['replays_max_abs'][name] == old_error
                replay_report[name] = dict(coefficients_exact=True, window_scores_max_abs=old_error,
                    window_scores_unequal_rows=unequal, answer_scores_exact=True, thresholds_exact=True)
            entry = dict(candidate=name, C=c, source=result['source'], score_replay_max_abs=score_error,
                answer_max_exact=True, thresholds_exact=True, thresholds=ts, selection_key=key, metrics=computed)
            candidates.append(entry); family.append(entry)
        chosen = max(family, key=lambda e: e['selection_key'])
        assert chosen['candidate'] == summary['selected'][method]['candidate']
        assert summary['selected'][method] == summary['all_candidates'][method][CS.index(chosen['C'])]
        selected[method] = chosen
        print('AUDIT_FAMILY_PASSED', method, chosen['C'], flush=True)
    assert Counter(e['source'] for e in candidates) == {'new_fit': 7, 'replayed_frozen_model': 3}
    # Confirm all read artifacts remain unchanged; never modify frozen files.
    before = dict(HASHES)
    for relative, expected in before.items(): assert sha(QA/relative) == expected
    report = dict(status='passed', audit_utc=datetime.now(timezone.utc).isoformat(),
        scope='Only new semantic_full implementation and authorized fit/cal frozen sources; no project imports, fitting, GPU, or official test.',
        scope_limits=['Existing MiniCheck inference and original character-to-state construction not rerun.',
                      'Frozen PCA reused and fit sample membership checked; no SVD/refitting or old full-chain re-audit.',
                      'Coefficient score replay checks saved artifacts, not an independent retraining.',
                      'Calibration-selected development comparison; no fresh-test inference or paired confidence intervals.'],
        source_hashes=HASHES, complete_file_hashes_verified=len(complete['files_sha256']),
        denominators=denominators, token_mapping=dict(raw_tokens=213159, raw_fit_tokens=170361,
            raw_calibration_tokens=42798, eligible_windows=NTOTAL, window_lengths=Counter(map(int, lengths)),
            nonlexical_raw_token_occurrences_preserved=nonlex, results=mapping_errors),
        old_pca_reuse=dict(pca_sha256=sha(OLD/'hidden_pca.pkl'), fit_only_samples=20288,
            fit_answers=634, fit_groups=615, variance_ratio=pca['explained_variance_ratio_sum']),
        weights=weights_report, scalers=scaler_report, selected=selected, candidates=candidates,
        old_three_replays=replay_report, checks=dict(candidate_coefficient_replays=10, calibration_thresholds_exact=20,
            candidate_selection_keys_exact=10, family_C_choices_exact=2, fit_cal_metric_blocks=40,
            saved_answer_max_arrays_exact=10, score_replay_max_abs=overall_score_error,
            metric_max_abs=overall_metric_error, new_fit_count=7, reused_fit_count=3),
        calibration_selection_optimistic=True, training_performed=False, GPU_used=False, official_test_opened=False,
        seconds=time.perf_counter()-clock)
    target = OUT/'INDEPENDENT_AUDIT_SEMANTIC_FULL.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['审计结果：passed。', '',
        '10 个冻结 LR 候选均完成系数回放；20 个校准阈值、10 个选择键与两族 C 选择一致。40 组 fit/cal 指标及全部整答最大值通过。', '',
        '213159 个已对齐词元到 210364 个窗口的均值完全一致。1024 维 scaler 独立按 fit 加权统计复核；64 维原 scaler/PCA 复用与 3 个旧模型回放通过。', '',
        '| 输入 | 所选 C | cal 窗口 F1 | cal 整答 F1 |', '|---|---:|---:|---:|']
    for method, e in selected.items():
        m=e['metrics']['calibration'];lines.append(f"| {method} | {e['C']:g} | {m['windows']['f1']:.6f} | {m['answers']['f1']:.6f} |")
    lines += ['', '完整 1024 维在这次校准选择中整答 F1 较高、窗口 F1 略低；不能据此声称两层同时改善。', '',
        '范围：CPU 分块复算；未训练、未用 GPU、未读 official test。原 MiniCheck 推理与既有 PCA 完整链不重审；结果仍是校准选择后的开发比较。', '',
        f"JSON SHA256：{hashlib.sha256(target.read_bytes()).hexdigest()}"]
    (OUT/'INDEPENDENT_AUDIT_SEMANTIC_FULL.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'passed','checks':report['checks'],'scalers':scaler_report,
        'selected':{k:{'C':e['C'],'cal_window_f1':e['metrics']['calibration']['windows']['f1'],
            'cal_answer_f1':e['metrics']['calibration']['answers']['f1']} for k,e in selected.items()},
        'json_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'md_sha256':hashlib.sha256((OUT/'INDEPENDENT_AUDIT_SEMANTIC_FULL.md').read_bytes()).hexdigest()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
