"""Direct baseless/conflict LR outputs with exact deterministic binary controls."""
from pathlib import Path
import argparse
import pickle
import sys
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'src'))
import run_development as q
import run_probe_expansion as base
OUT = HERE/'direct_type_heads_v1'
METHODS = ('hidden64', 'hidden64_risk')
CS = (1e-5, 1e-4, .001)
TYPE = {'Evident Baseless Info': 0, 'Subtle Baseless Info': 0,
        'Evident Conflict': 1, 'Subtle Conflict': 1}
NFIT, NALL, AFIT, AALL = 653979, 696220, 3680, 3839
SEED = 20260924


def old_name(method, c): return f'expanded3680_{method}_C{c:g}'


def protocol():
    return {'version': 'expanded-direct-two-type-LR-v1', 'methods': list(METHODS), 'C': list(CS),
        'fit_answers': AFIT, 'calibration_answers': 159, 'fit_windows': NFIT, 'calibration_windows': 42241,
        'fit_groups': 615, 'calibration_groups': 154, 'new_LR_fits': 12, 'reused_binary_models': 6,
        'input': 'Exactly probe_v1 frozen window65.npy: original fit634 PCA64; hidden64 first64 or hidden64_risk all65. No encoder/PCA/scaler fitting.',
        'targets': TYPE,
        'type_window': 'Each type is OR of original lexical span-risk token positions within the original4rawBPE window. Both types may be1; OR exactly equals unchanged binary label. Only fit type labels constructed.',
        'candidate': 'Two independent liblinear LR heads with the SAME candidate C. One predicts baseless, the other conflict. No3x3 per-head C grid.',
        'weights': 'Reuse original expanded3680_weights.npz loss unchanged for BOTH heads, mass168123 each. Original native/added strata, source groups and binary class factors unchanged. No type weighting, no new class factors.',
        'objective': 'Independent same-C regularized binary objectives (equivalently their equally weighted average, including both regularizers). Do not halve sample_weight, which would change effective C.',
        'control': 'For each method/C, reuse one original binary LR as TWO identical outputs; max(p,p)=p exactly. No refit and no independent capacity or new information. This is a deterministic binary control, NOT a strong capacity-matched learned ensemble.',
        'inference': 'Fixed max of the two type probabilities at every window, then original answermax. No gold/type/source/model identity fed into features. The max is a risk score, not a calibrated union probability.',
        'selection': 'Original q.choose_threshold for each of window/answer on the same fixed159cal. q.selection_key across same3Cs per input; one C shared by heads and both metrics. Retain all candidates and control choice.',
        'classifier': {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 2000, 'seed': SEED, 'CPU_threads': 4,
            'nonconvergence': 'ConvergenceWarning is an error; preserve failure, no extra iterations or grid.'},
        'limits': 'Same exposed calibration, no independent test. Frozen MiniCheck semantic-checker state, not native generator whitebox. Direct heads add genuinely different decision boundaries versus deterministic duplicated binary control; not a pure effective-capacity isolation.',
        'trained': False, 'GPU_used': False, 'official_test_opened': False}


def source_hashes():
    paths = [Path(__file__), Path(base.__file__), Path(q.__file__), OUT/'protocol.json',
        base.OUT/'complete.json', base.OUT/'summary.json', base.OUT/'protocol.json',
        base.OUT/'preparation_complete.json', base.OUT/'expanded3680_weights.npz',
        base.OUT/'matrices/window65.npy', base.OLD/'hidden_pca.pkl',
        HERE/'data/export_freeze.json', HERE/'data/answers_fit.jsonl', HERE/'data/tokens_fit.jsonl',
        HERE/'data/windows_k4_fit.jsonl', q.ROOT/'data/answers_calibration.jsonl',
        q.ROOT/'data/tokens_calibration.jsonl', q.ROOT/'data/windows_k4_calibration.jsonl']
    paths += [base.OUT/(old_name(m,c)+suffix) for m in METHODS for c in CS
              for suffix in ('.pkl', '_scores.npz', '_result.json')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def type_tokens(tok):
    text = tok['original_response']; n = tok['token_count']
    offsets = np.asarray(tok['response_token_offsets'], int)
    prefix = np.concatenate(([0], np.cumsum([c.isalnum() for c in text], dtype=np.int64)))
    y = np.zeros((n,2), bool)
    assert len(tok['original_labels']) == len(tok['span_token_mapping'])
    for span, mapping in zip(tok['original_labels'], tok['span_token_mapping']):
        assert span['label_type'] in TYPE
        start, end = span['start'], span['end']
        assert text[start:end] == span['text']
        lo = np.maximum(offsets[:,0], start); hi = np.minimum(offsets[:,1], end)
        active = lo < hi; hit = np.zeros(n, bool)
        hit[active] = prefix[hi[active]] > prefix[lo[active]]
        assert np.flatnonzero(hit).tolist() == mapping['risk_token_indices']
        y[hit,TYPE[span['label_type']]] = True
    assert np.array_equal(y.any(1), np.asarray(tok['risk_mask'], bool))
    assert not y[~np.asarray(tok['lexical_mask'], bool)].any()
    return y


def standardize(raw, scaler, width):
    x = np.empty((len(raw), width), np.float32)
    for lo in range(0, len(x), q.BATCH):
        hi = min(lo+q.BATCH, len(x))
        x[lo:hi] = scaler.transform(raw[lo:hi,:width]).astype(np.float32)
    return x


def thresholds(meta, score):
    answer = q.answer_scores(meta, score)
    return {'window': q.choose_threshold([w['label'] for w in meta['windows'][NFIT:]], score[NFIT:]),
            'answer': q.choose_threshold([a['label'] for a in meta['answers'][AFIT:]], answer[AFIT:])}


def controls(method, x, scaler, meta):
    entries, report = [], []
    weight_sha = q.sha(base.OUT/'expanded3680_weights.npz')
    for c in CS:
        name = old_name(method,c)
        obj = pickle.loads((base.OUT/(name+'.pkl')).read_bytes())
        entry = q.read(base.OUT/(name+'_result.json'))
        assert obj['regime'] == 'expanded3680' and obj['method'] == method
        assert obj['fit_only'] and obj['fit_rows'] == NFIT and obj['weight_sha256'] == weight_sha
        assert obj['C'] == obj['model'].C == c and obj['model'].random_state == SEED
        assert obj['model'].solver == 'liblinear' and obj['model'].max_iter == 2000
        for attr in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
            assert np.array_equal(getattr(obj['scaler'],attr),getattr(scaler,attr)), attr
        assert entry['model_sha256'] == q.sha(base.OUT/(name+'.pkl'))
        assert entry['scores_sha256'] == q.sha(base.OUT/(name+'_scores.npz'))
        score = obj['model'].predict_proba(x)[:,1]
        with np.load(base.OUT/(name+'_scores.npz'), allow_pickle=False) as data:
            assert np.array_equal(score, data['window_scores'])
            assert np.array_equal(q.answer_scores(meta,score), data['answer_scores'])
        assert np.array_equal(np.maximum(score,score),score)
        assert thresholds(meta,score) == entry['thresholds'] == obj['thresholds']
        assert q.metrics(meta,score,entry['thresholds']) == entry['metrics']
        entries.append(entry)
        report.append({'candidate':name, 'all696220_window_scores_exact':True,
            'all3839_answer_scores_exact':True, 'same_C_input_weight_scaler':True,
            'two_identical_outputs_max_exact':True, 'thresholds_and_metrics_exact':True})
    old_selected = q.read(base.OUT/'summary.json')['selected']['expanded3680_'+method]
    assert max(entries,key=lambda e:e['selection_key']) == old_selected
    return {'all_candidates':entries,'selected':old_selected}, report


def prepare():
    assert not (OUT/'prepare_started.json').exists(), 'Preserve preparation/failure'
    OUT.mkdir(parents=True,exist_ok=True)
    q.save(OUT/'protocol.json',protocol())
    snapshot = source_hashes()
    q.save(OUT/'prepare_started.json', {'source_sha256': snapshot, 'real_fits':0})
    for name, digest in q.read(base.OUT/'complete.json')['files_sha256'].items():
        assert q.sha(base.OUT/name) == digest
    assert q.read(base.OUT/'preparation_complete.json')['files_sha256']['matrices/window65.npy'] == q.sha(base.OUT/'matrices/window65.npy')
    _, meta = base.metadata()
    assert len(meta['windows']) == NALL and len(meta['answers']) == AALL
    assert meta['bounds'] == {'fit':[0,NFIT], 'calibration':[NFIT,NALL]}
    with np.load(base.OUT/'expanded3680_weights.npz',allow_pickle=False) as data:
        weights = {k:data[k].copy() for k in data.files}
    actual = base.expanded_weights(meta)
    for name,value in zip(('base','loss','y','class_factors'),actual):
        assert np.array_equal(weights[name],value), name
    token_types = {t['response_id']:type_tokens(t) for t in meta['tokens'][:AFIT]}
    targets = np.asarray([token_types[w['response_id']][w['token_indices']].any(0)
                          for w in meta['windows'][:NFIT]], np.int8)
    assert targets.shape == (NFIT,2)
    assert np.array_equal(targets.any(1),weights['y'].astype(bool))
    assert np.array_equal(weights['y'],[w['label'] for w in meta['windows'][:NFIT]])
    np.savez_compressed(OUT/'type_targets_fit.npz', targets=targets, original_binary=weights['y'])
    raw = np.load(base.OUT/'matrices/window65.npy',mmap_mode='r')
    assert raw.shape == (NALL,65) and raw.dtype == np.float32
    fixed, check = {}, {}
    for method in METHODS:
        width = 64 if method == 'hidden64' else 65
        obj = pickle.loads((base.OUT/(old_name(method,CS[0])+'.pkl')).read_bytes())
        x = standardize(raw,obj['scaler'],width)
        fixed[method],check[method] = controls(method,x,obj['scaler'],meta)
        del x
        print('DIRECT_TYPE_CONTROL_EXACT',method,'3binarymodels',flush=True)
    bit = targets[:,0]+2*targets[:,1]
    counts = np.bincount(bit,minlength=4).tolist()
    q.save(OUT/'CONTROLS.json',fixed)
    q.save(OUT/'CPU_SELFCHECK.json',{'passed':True,'controls':check,
        'fit_window_partition_clean_baseless_conflict_both':counts,
        'token_type_OR_exact':True,'window_type_OR_exact':True,'original_weight_reconstruction_exact':True,
        'source_groups_fit_cal_disjoint':True,'old_binary_loss_mass':float(weights['loss'].sum()),
        'fit_only_scaler_reused':True,'control_heads_deterministically_identical':True,
        'real_LR_fits':0,'GPU_used':False,'official_test_opened':False})
    assert snapshot == source_hashes()
    names = ['CONTROLS.json','CPU_SELFCHECK.json','type_targets_fit.npz']
    q.save(OUT/'preparation_complete.json',{'status':'prepared_not_fitted',
        'source_sha256':snapshot,'files_sha256':{n:q.sha(OUT/n) for n in names},
        'new_LR_fits':0,'future_fixed_new_fits':12,'control_refits':0,'official_test_opened':False})
    print('DIRECT_TYPE_HEADS_PREPARED_NO_FIT',counts,flush=True)


def check():
    p = q.read(OUT/'preparation_complete.json')
    assert p['source_sha256'] == source_hashes() and q.read(OUT/'protocol.json') == protocol()
    for name,digest in p['files_sha256'].items(): assert q.sha(OUT/name) == digest
    return p


def fit():
    prepared = check()
    assert not (OUT/'fit_started.json').exists(), 'Preserve prior fit/failure'
    _,meta = base.metadata()
    with np.load(OUT/'type_targets_fit.npz',allow_pickle=False) as data:
        targets = data['targets'].copy()
    with np.load(base.OUT/'expanded3680_weights.npz',allow_pickle=False) as data:
        loss = data['loss'].copy(); binary = data['y'].copy()
    assert np.array_equal(targets.any(1),binary.astype(bool))
    raw = np.load(base.OUT/'matrices/window65.npy',mmap_mode='r')
    q.save(OUT/'fit_started.json',{'time':time.time(),'preparation_sha256':q.sha(OUT/'preparation_complete.json')})
    all_candidates,selected = {},{}; new_fits = 0; files=[]; started=time.perf_counter()
    for method in METHODS:
        width = 64 if method == 'hidden64' else 65
        scaler = pickle.loads((base.OUT/(old_name(method,CS[0])+'.pkl')).read_bytes())['scaler']
        x = standardize(raw,scaler,width); entries=[]
        for c in CS:
            name=f'{method}_direct_types_C{c:g}'; tick=time.perf_counter(); heads=[]; head_values=[]
            for kind in range(2):
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=SEED)
                    model.fit(x[:NFIT],targets[:,kind],sample_weight=loss)
                assert model.n_iter_.max()<2000
                heads.append(model);head_values.append(model.predict_proba(x)[:,1]);new_fits+=1
            head_values=np.asarray(head_values).T
            score=head_values.max(1);answer=q.answer_scores(meta,score)
            ts=thresholds(meta,score);metrics=q.metrics(meta,score,ts)
            obj={'heads':heads,'scaler':scaler,'C':c,'method':method,'fit_rows':NFIT,
                'head_order':['baseless','conflict'],'fixed_aggregation':'max',
                'original_binary_weight_sha256':q.sha(base.OUT/'expanded3680_weights.npz'),
                'targets_sha256':q.sha(OUT/'type_targets_fit.npz'),'thresholds':ts}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5))
            np.savez_compressed(OUT/(name+'_scores.npz'),head_scores=head_values,window_scores=score,answer_scores=answer)
            entry={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),
                'metrics':metrics,'seconds':time.perf_counter()-tick,'n_iter':[h.n_iter_.tolist() for h in heads],
                'files_sha256':{ext:q.sha(OUT/(name+ext)) for ext in ('.pkl','_scores.npz')}}
            q.save(OUT/(name+'_result.json'),entry);entries.append(entry)
            files.extend(name+ext for ext in ('.pkl','_scores.npz','_result.json'))
            print('DIRECT_TYPE_CANDIDATE',name,metrics['calibration']['windows']['f1'],metrics['calibration']['answers']['f1'],flush=True)
        all_candidates[method]=entries;selected[method]=max(entries,key=lambda e:e['selection_key'])
        del x
    assert new_fits==12 and prepared==check()
    q.save(OUT/'summary.json',{'all_candidates':all_candidates,'selected':selected,
        'deterministic_binary_controls':q.read(OUT/'CONTROLS.json'),'new_LR_fits':new_fits,'control_refits':0,
        'seconds':time.perf_counter()-started,'official_test_opened':False,'extra_semantic_checker':True,
        'limits':protocol()['limits']})
    files += ['summary.json','CONTROLS.json','protocol.json','preparation_complete.json','CPU_SELFCHECK.json','type_targets_fit.npz']
    q.save(OUT/'complete.json',{'status':'complete_development_only',
        'files_sha256':{n:q.sha(OUT/n) for n in files},'new_LR_fits':12,'control_refits':0,'GPU_used':False,'official_test_opened':False})
    print('DIRECT_TYPE_HEADS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','check','fit'))
    stage=parser.parse_args().stage
    with threadpool_limits(limits=4):
        try:{'prepare':prepare,'check':check,'fit':fit}[stage]()
        except BaseException as exc:
            OUT.mkdir(parents=True,exist_ok=True)
            q.save(OUT/f'FAILURE_{stage}_{time.time_ns()}.json',{'error':repr(exc),'GPU_used':False,'official_test_opened':False})
            raise
