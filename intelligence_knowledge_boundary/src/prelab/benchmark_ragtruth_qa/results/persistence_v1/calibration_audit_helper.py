"""Independent CPU audit of frozen fit/cal scores; imports no project modules."""
from pathlib import Path
from collections import defaultdict
import hashlib
import itertools
import json
import sys
import numpy as np
from scipy.special import expit

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RESULTS = ROOT / 'results'
HASHES = {}

def sha(p):
    p = Path(p).resolve()
    assert p.is_relative_to(ROOT)
    assert not any(x in str(p.relative_to(ROOT)).lower() for x in ('sealed', 'withheld', 'test'))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    HASHES[str(p.relative_to(ROOT))] = h
    return h

def read(p):
    sha(p)
    return json.loads(Path(p).read_text('utf-8'))

def load_scores(p):
    sha(p)
    with np.load(p, allow_pickle=False) as z:
        w, a = z['window_scores'].astype(np.float64), z['answer_scores'].astype(np.float64)
    assert w.shape == (210364,) and a.shape == (793,)
    assert np.isfinite(w).all() and np.isfinite(a).all()
    assert np.all((w >= 0) & (w <= 1))
    return w, a

def threshold(y, s):
    # Ascending score buckets; count every possible >= decision plus predict-none.
    unique, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    positives = np.bincount(inv, weights=y, minlength=len(unique)).astype(np.int64)
    tp = np.r_[np.cumsum(positives[::-1])[::-1], 0]
    n = np.r_[np.cumsum(counts[::-1])[::-1], 0]
    ts = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    f1 = 2 * tp / (n + int(y.sum()))
    precision = np.divide(tp, n, out=np.zeros(len(n)), where=n != 0)
    k = max(range(len(ts)), key=lambda i: (f1[i], precision[i], ts[i]))
    return dict(threshold=float(ts[k]), f1=float(f1[k]), precision=float(precision[k]),
                rows=len(y), positive=int(y.sum()))

def metrics(y, s, t):
    pred = s >= t
    tp = int(y[pred].sum()); fp = int(pred.sum()) - tp
    fn = int(y.sum()) - tp; tn = len(y) - tp - fp - fn
    u, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(u))
    neg = counts - pos
    auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (y.sum() * (len(y) - y.sum()))
    pdesc = pos[::-1]; ndesc = counts[::-1]
    ap = np.sum(pdesc / y.sum() * np.cumsum(pdesc) / np.cumsum(ndesc))
    return dict(n=len(y), positive=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn), auroc=float(auc), average_precision=float(ap))

def posterior(raw, chains, stay, temp):
    if stay == .5 and temp == 1.:
        return raw.copy()
    q = np.clip(raw, 1e-12, 1-1e-12)
    emission = expit((np.log(q)-np.log1p(-q))/temp)
    out = np.empty_like(raw)
    switch = 1-stay
    for ids in chains:
        p = emission[ids]; n = len(ids)
        f0 = np.empty(n); f1 = np.empty(n)
        f0[0] = 1-p[0]; f1[0] = p[0]
        for j in range(1, n):
            z0 = (1-p[j])*(stay*f0[j-1]+switch*f1[j-1])
            z1 = p[j]*(switch*f0[j-1]+stay*f1[j-1])
            den = z0+z1; f0[j] = z0/den; f1[j] = z1/den
        b0 = b1 = 1.
        out[ids[-1]] = f1[-1]/(f0[-1]+f1[-1])
        for j in range(n-2, -1, -1):
            z0 = stay*(1-p[j+1])*b0 + switch*p[j+1]*b1
            z1 = switch*(1-p[j+1])*b0 + stay*p[j+1]*b1
            den = z0+z1; b0 = z0/den; b1 = z1/den
            out[ids[j]] = f1[j]*b1/(f0[j]*b0+f1[j]*b1)
    return out

def close_dict(a, b, tol=1e-12):
    assert a.keys() == b.keys()
    return max(abs(float(a[k])-float(b[k])) for k in a)

def main():
    protocol = read(OUT/'protocol.json'); started = read(OUT/'started.json')
    complete = read(OUT/'complete.json'); summary = read(OUT/'summary.json')
    assert complete['status'] == 'complete' and not complete['test_opened']
    for n, h in complete['files_sha256'].items(): assert sha(OUT/n) == h
    assert sha(ROOT/'src/run_persistence.py') == started['code_sha256']
    assert sha(OUT/'protocol.json') == started['protocol_sha256']
    assert protocol['stay'] == [.5,.8,.95,.99] and protocol['temperature'] == [.5,1.,2.]
    idx = read(RESULTS/'development_v1/score_index.json')
    windows, answers = idx['windows'], idx['answers']
    assert len(windows) == 210364 and len(answers) == 793
    assert len({w['window_id'] for w in windows}) == len(windows)
    assert len({a['answer_id'] for a in answers}) == len(answers)
    byanswer = defaultdict(list)
    for j,w in enumerate(windows): byanswer[w['answer_id']].append(j)
    ax = [np.asarray(byanswer[a['answer_id']], int) for a in answers]
    assert all(len(x) for x in ax)
    assert np.array_equal(np.concatenate(ax), np.arange(len(windows)))
    y = np.asarray([w['label'] for w in windows], int)
    ya = np.asarray([a['label'] for a in answers], int)
    assert set(y) == set(ya) == {0,1}
    bounds = {'fit':(0,168123,0,634), 'calibration':(168123,210364,634,793)}
    denominators = {}
    for part,(l,r,al,ar) in bounds.items():
        assert all(w['partition'] == part for w in windows[l:r])
        assert all(a['partition'] == part for a in answers[al:ar])
        for a,ix in zip(answers[al:ar],ax[al:ar]):
            assert all(windows[j]['group_id'] == a['group_id'] for j in ix)
        denominators[part] = dict(windows=r-l, positive_windows=int(y[l:r].sum()),
            answers=ar-al, positive_answers=int(ya[al:ar].sum()),
            groups=len({a['group_id'] for a in answers[al:ar]}))
    assert not ({a['group_id'] for a in answers[:634]} & {a['group_id'] for a in answers[634:]})
    chains = []
    for a,ids in zip(answers[634:],ax[634:]):
        starts = []
        for j in ids:
            prefix, suffix = windows[j]['window_id'].rsplit('__k4_',1)
            assert prefix == a['answer_id']
            starts.append(int(suffix))
        starts = np.asarray(starts)
        assert np.all(np.diff(starts)>0)
        chains.extend(np.split(ids-168123,np.flatnonzero(np.diff(starts)!=1)+1))
    assert len(chains) == started['calibration_chains'] == 235
    assert np.array_equal(np.sort(np.concatenate(chains)),np.arange(42241))
    cax = [ix-168123 for ix in ax[634:]]
    def amax(w): return np.asarray([w[ix].max() for ix in ax])
    def camax(w): return np.asarray([w[ix].max() for ix in cax])
    source = {}
    for folder,binding in started['source'].items():
        src = RESULTS/folder
        assert sha(src/'complete.json') == binding['complete_sha256']
        done = read(src/'complete.json')
        assert sha(src/'summary.json') == binding['summary_sha256'] == done['files_sha256']['summary.json']
        oldsummary = read(src/'summary.json')
        if folder == 'development_v1':
            assert sha(src/'score_index.json') == done['files_sha256']['score_index.json']
        for name,entry in oldsummary['selected'].items():
            fname = f"{name}/epoch_{entry['epoch']:03d}_scores.npz" if folder=='sequence_v1' else entry['candidate']+'_scores.npz'
            assert sha(src/fname) == binding['scores'][fname]
            frozen_names = {k.replace('\\','/'): v for k,v in done['files_sha256'].items()}
            assert frozen_names[fname] == binding['scores'][fname]
            source[name] = (entry,load_scores(src/fname))
    assert set(source) == set(summary['selected']) == set(summary['all_candidates']) and len(source) == 9
    rows = []; max_metric_error = 0.; max_threshold_error = 0.; max_selected_error = 0.
    candidate_count = 0; threshold_count = 0; metric_count = 0
    for name,(old,(raw,rawa)) in source.items():
        sel = summary['selected'][name]; table = summary['all_candidates'][name]
        assert len(table) == 12 and sel['raw_metrics'] == old['metrics']
        assert np.array_equal(amax(raw),rawa)
        saved,saveda = load_scores(OUT/(name+'_scores.npz'))
        assert np.array_equal(amax(saved),saveda)
        reconstructed = []
        for k,(stay,temp) in enumerate(itertools.product(protocol['stay'],protocol['temperature'])):
            t = table[k]; assert (t['stay'],t['temperature']) == (stay,temp)
            cs = posterior(raw[168123:],chains,stay,temp)
            ts = {'window':threshold(y[168123:],cs), 'answer':threshold(ya[634:],camax(cs))}
            for level in ts:
                delta = close_dict(ts[level],t['thresholds'][level]); max_threshold_error=max(max_threshold_error,delta)
                assert delta <= 1e-12, (name,stay,temp,level,ts[level],t['thresholds'][level])
                threshold_count += 1
            w,a = ts['window'],ts['answer']
            key = [min(w['f1'],a['f1']),w['f1'],w['precision'],int(stay==.5 and temp==1.),-stay,-abs(float(np.log(temp)))]
            assert np.array_equal(key,t['key']), (name,k,key,t['key'])
            reconstructed.append(key); candidate_count += 1
            if stay == .5 and temp == 1.:
                assert np.array_equal(cs,raw[168123:])
                assert ts == t['thresholds'] == old['thresholds']
            if k == sel['selected_index']:
                err = float(np.max(np.abs(cs-saved[168123:])))
                max_selected_error=max(max_selected_error,err); assert err <= 1e-12
        best = max(range(12),key=lambda k: reconstructed[k])
        assert best == sel['selected_index']
        for k,v in table[best].items(): assert sel[k] == v
        for scores,ascores,ts,expected in [(raw,rawa,old['thresholds'],old['metrics']),
                                          (saved,saveda,sel['thresholds'],sel['metrics'])]:
            for part,(l,r,al,ar) in bounds.items():
                for level,yy,ss in [('windows',y[l:r],scores[l:r]),('answers',ya[al:ar],ascores[al:ar])]:
                    got = metrics(yy,ss,ts['window' if level=='windows' else 'answer']['threshold'])
                    err = close_dict(got,expected[part][level]); max_metric_error=max(max_metric_error,err)
                    assert err <= 1e-12,(name,part,level,err)
                    metric_count += 1
        rows.append(dict(method=name,selected_stay=sel['stay'],selected_temperature=sel['temperature'],
            selected_index=best,identity_thresholds_exact=True,
            cal_window_f1_before=old['metrics']['calibration']['windows']['f1'],
            cal_window_f1_after=sel['metrics']['calibration']['windows']['f1'],
            cal_answer_f1_before=old['metrics']['calibration']['answers']['f1'],
            cal_answer_f1_after=sel['metrics']['calibration']['answers']['f1']))
        print(json.dumps(rows[-1]),flush=True)
    assert summary['calibration_results_are_selection_optimistic'] and summary['old_identity_thresholds_exact']
    report=dict(passed=True,blockers=[],scope='Frozen fit/cal only; no training/model/project imports or test reads',
        method='Independent ascending-score bucket thresholds, scalar binary forward/backward probability recurrence, rank metrics',
        candidate_count=candidate_count,threshold_count=threshold_count,selected_families=9,
        identity_threshold_count_exact=18,metric_blocks=metric_count,max_metric_abs_error=max_metric_error,
        max_candidate_threshold_abs_error=max_threshold_error,max_selected_cal_score_abs_error=max_selected_error,
        answermax_exact_for_raw_and_selected_scores=True,denominators=denominators,
        calibration_chains=len(chains),selection_keys_and_selected_indices_exact=True,
        fit_scores_not_used_in_candidate_selection=True,calibration_results_are_selection_optimistic=True,
        rows=rows,bindings=HASHES,helper_sha256=sha(Path(__file__)),
        limitations=['Raw-start chains recovered from frozen window_id suffix; independent full gold geometry audit delegated to parent.',
                     'Hash consistency verifies present artifacts, not external historical freeze chronology.',
                     'All before/after F1 are calibration-selected development figures, not held-out test gains.'])
    target=OUT/'CALIBRATION_AUDIT_PERSISTENCE_QA.json'
    assert not target.exists()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'report':str(target),'sha256':sha(target),'candidate_count':candidate_count,
        'threshold_count':threshold_count,'metric_blocks':metric_count,'max_metric_error':max_metric_error,
        'max_threshold_error':max_threshold_error,'max_selected_score_error':max_selected_error,
        'denominators':denominators},ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
