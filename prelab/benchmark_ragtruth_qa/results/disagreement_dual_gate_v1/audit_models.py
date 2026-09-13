"""Independently replay eight frozen trees and all962 fit-only decisions."""
from pathlib import Path
import sys
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_development as q


def count(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p));tn=int(np.sum(~y&~p))
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def run():
    tick=time.perf_counter();cfg=q.read(OUT/'protocol.json');done=q.read(OUT/'complete.json')
    assert q.sha(OUT/'summary.json')==done['summary_sha256']
    assert q.sha(OUT/'fit_only_selection.json')==done['selected_sha256']
    assert q.sha(OUT/'final_two_head_predictions.npz')==done['predictions_sha256']
    assert q.sha(OUT/'full_fit_frozen.json')==done['full_fit_freeze_sha256']
    summary=q.read(OUT/'summary.json');freeze=q.read(OUT/'full_fit_frozen.json')
    assert freeze['real_models_fitted']==8 and freeze['calibration_not_scored']
    x=np.load(OUT/'window_features.npy',mmap_mode='r')
    with np.load(OUT/'frozen_inputs.npz') as z:
        y=z['window_labels'].copy();ay=z['answer_labels'].copy();current=z['current_positive'].copy();rescue=z['rescue_eligible'].copy()
        original_answer=z['answer_probabilities'][:,0].copy()
    with np.load(OUT/'fit_OOF_gate_probabilities.npz') as z: old_oof=[z['keep_probability'].copy(),z['rescue_probability'].copy()]
    with np.load(OUT/'final_two_head_predictions.npz') as z: final={k:z[k].copy() for k in z.files}
    buffers=[np.full(168123,np.nan),np.full(168123,np.nan)];cover=[np.zeros(168123,int),np.zeros(168123,int)]
    records=[];full={}
    with threadpool_limits(4):
        for fold in range(3):
            for j,name in enumerate(('keep','rescue')):
                path=OUT/f'fold_{fold}_{name}.pkl';assert q.sha(path)==freeze['models_sha256'][path.name]
                model=pickle.loads(path.read_bytes());assert np.array_equal(model.classes_,[0,1])
                for k,v in cfg['model'].items():assert model.get_params()[k]==v
                assert model.n_iter_==100
                with np.load(OUT/f'fold_{fold}_{name}_indices_weights.npz') as z: hi=z['held_indices']
                pred=model.predict_proba(x[hi])[:,1]
                assert np.array_equal(pred,old_oof[j][hi])
                buffers[j][hi]=pred;cover[j][hi]+=1
                records.append(dict(fold=fold,gate=name,rows=len(hi),probability_exact=True))
        for j,(name,mask) in enumerate(zip(('keep','rescue'),(current,rescue))):
            assert np.array_equal(cover[j],mask[:168123].astype(int))
            assert np.array_equal(buffers[j],old_oof[j],equal_nan=True)
            path=OUT/f'full_{name}.pkl';assert q.sha(path)==freeze['models_sha256'][path.name]
            model=pickle.loads(path.read_bytes());assert model.n_iter_==100
            for k,v in cfg['model'].items():assert model.get_params()[k]==v
            pred=np.full(len(y),np.nan);pred[mask]=model.predict_proba(x[mask])[:,1]
            assert np.array_equal(pred,final[name+'_probability'],equal_nan=True)
            full[name]=pred;records.append(dict(fold='allfit',gate=name,rows=int(mask.sum()),probability_exact=True))
    table=q.read(OUT/'fit_only_threshold_grid.json');grid=cfg['selection']['grid'];assert len(table)==962 and len(grid)==31
    assert table[0]['mode']=='identity' and table[0]['metrics']==count(y[:168123],current[:168123])
    scores=[(table[0]['metrics']['f1'],table[0]['metrics']['precision'],0.)]
    for index,(tk,tr) in enumerate(( (tk,tr) for tk in grid for tr in grid),1):
        p=(current[:168123]&(buffers[0]>=tk)) | (rescue[:168123]&(buffers[1]>=tr))
        measured=count(y[:168123],p);e=table[index]
        assert e['mode']=='gates' and e['keep_threshold']==tk and e['rescue_threshold']==tr
        assert measured==e['metrics'] and e['distance_to_half']==abs(tk-.5)+abs(tr-.5)
        scores.append((measured['f1'],measured['precision'],-abs(tk-.5)-abs(tr-.5)))
    selected=table[max(range(962),key=lambda j:scores[j])]
    assert selected==summary['selected']==q.read(OUT/'fit_only_selection.json')['selected']
    if selected['mode']=='identity': p=current.copy()
    else:p=(current&(full['keep']>=selected['keep_threshold'])) | (rescue&(full['rescue']>=selected['rescue_threshold']))
    assert np.array_equal(p,final['window_decisions'])
    assert count(y[:168123],p[:168123])==summary['window_metrics']['final_fit_in_sample']
    assert count(y[168123:],p[168123:])==summary['window_metrics']['calibration']
    threshold=summary['answer_threshold_unchanged']
    assert np.array_equal(original_answer,final['answer_scores'])
    assert np.array_equal(original_answer>=threshold,final['answer_decisions'])
    for name,s in [('fit',slice(0,634)),('calibration',slice(634,None))]:
        assert q.count(ay[s],original_answer[s],threshold)==summary['answer_metrics_unchanged'][name]
    result=dict(status='passed',models_replayed=8,all_probability_values_exact=True,
        model_replays=records,all_962_OOF_counts_and_selection_exact=True,identity_preserved=True,
        full_fit_and_calibration_window_counts_exact=True,original_answer_head_probabilities_threshold_counts_exact=True,
        selected=selected,calibration_window_metrics=summary['window_metrics']['calibration'],
        calibration_answer_metrics=summary['answer_metrics_unchanged']['calibration'],
        no_fit=True,GPU_used=False,test_opened=False,seconds=time.perf_counter()-tick,
        audit_source_sha256=q.sha(Path(__file__)),upstream_complete_sha256=q.sha(OUT/'complete.json'))
    q.save(OUT/'INDEPENDENT_MODEL_REPLAY.json',result)
    print('DUAL_GATE_ALL8_MODELS_962_SELECTION_REPLAY_PASSED',result['seconds'],flush=True)


if __name__=='__main__':run()
