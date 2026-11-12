"""Read-only saved-score replay/count check; no production scorer imports."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
LOCAL=ROOT/'results/citation_alignment_lr_v1'
GLOBAL=ROOT/'results/completed_score_combiner_v1'
PRIOR=ROOT/'results/answer_conditioned_scores_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def count(y,score,threshold):
    prediction=score>=threshold
    tp=int(np.count_nonzero(prediction[y==1]));fn=int(np.count_nonzero(~prediction[y==1]))
    fp=int(np.count_nonzero(prediction[y==0]));tn=int(np.count_nonzero(~prediction[y==0]))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'auroc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score))}


def best_threshold(y,score):
    values,inverse=np.unique(score,return_inverse=True)
    n=np.bincount(inverse);positive=np.bincount(inverse,weights=y)
    tp=np.r_[np.cumsum(positive[::-1])[::-1],0.]
    total=np.r_[np.cumsum(n[::-1])[::-1],0]
    thresholds=np.r_[values,np.nextafter(values[-1],np.inf)]
    f1=2*tp/(total+y.sum());precision=np.divide(tp,total,out=np.zeros(len(total)),where=total>0)
    j=np.lexsort((thresholds,precision,f1))[-1]
    return {'threshold':float(thresholds[j]),'f1':float(f1[j]),'precision':float(precision[j]),
            'rows':len(y),'positive':int(y.sum())}


def main():
    summary=read(OUT/'summary.json');assert sha(OUT/'summary.json')==read(OUT/'complete.json')['summary_sha256']
    freeze=read(OUT/'design_freeze.json')
    for path,h in freeze['source_sha256'].items():assert sha(Path(path))==h,path
    assert read(OUT/'protocol.json')['alpha']==[0.,.2,.4,.6,.8,1.]
    windows=[];answers=[]
    for part in ('fit','calibration'):
        for base,target in [('windows_k4',windows),('answers',answers)]:
            target.extend(json.loads(l) for l in (ROOT/'data'/f'{base}_{part}.jsonl').open(encoding='utf-8'))
    assert len(windows)==210364 and len(answers)==793
    by={a['response_id']:[] for a in answers}
    for i,w in enumerate(windows):by[w['response_id']].append(i)
    iy=[np.array(by[a['response_id']],int) for a in answers]
    assert sum(map(len,iy))==len(windows) and min(map(len,iy))>0
    wy=np.array([w['label'] for w in windows]);ay=np.array([a['label'] for a in answers])
    assert wy[:168123].sum()==21477 and wy[168123:].sum()==5984
    assert ay[:634].sum()==328 and ay[634:].sum()==100

    def metrics(score,answer,entry):
        assert np.array_equal(answer,np.array([score[ix].max() for ix in iy]))
        actual={}
        for part,wsl,asl in [('fit',slice(0,168123),slice(0,634)),('calibration',slice(168123,None),slice(634,None))]:
            actual[part]={'windows':count(wy[wsl],score[wsl],entry['thresholds']['window']['threshold']),
                          'answers':count(ay[asl],answer[asl],entry['thresholds']['answer']['threshold'])}
        assert actual==entry['metrics'],entry.get('candidate')
        ts={'window':best_threshold(wy[168123:],score[168123:]),'answer':best_threshold(ay[634:],answer[634:])}
        assert ts==entry['thresholds'],entry.get('candidate')
        return actual

    def load(directory,name,entry):
        p=directory/(name+'_scores.npz');assert sha(p)==entry['scores_sha256'],p
        with np.load(p) as z:s=z['window_scores'].copy();a=z['answer_scores'].copy()
        assert s.shape==(210364,) and a.shape==(793,) and np.isfinite(s).all()
        assert ((s>=0)&(s<=1)).all()
        metrics(s,a,entry)
        return s,a

    ge=read(GLOBAL/'summary.json')['methods']['lookback'];_,global_answer=load(GLOBAL,'lookback',ge)
    old=read(PRIOR/'summary.json');local_entries=read(LOCAL/'summary.json')['selected']
    checked=[];peers={}
    for peer in ('lookback','harp_claim','semantic_claim'):
        le=local_entries[peer+'__two_scores_and_citation'];local,own=load(LOCAL,le['candidate'],le)
        gate=np.empty(len(windows));gz=np.empty(len(windows))
        for j,ix in enumerate(iy):
            maximum=local[ix].max();assert maximum==own[j] and maximum>0
            gate[ix]=np.power(local[ix]/maximum,8)
            gz[ix]=logit(np.clip(global_answer[j],1e-6,1-1e-6))
            assert gate[ix].max()==1
        assert ((gate>=0)&(gate<=1)).all()
        control=old['all_candidates'][peer+'__shared_lookback_tree_answer'];new=summary['all_candidates'][peer]
        assert [e['alpha'] for e in control]==[e['alpha'] for e in new]==[0.,.2,.4,.6,.8,1.]
        assert control==summary['uniform_control_candidates'][peer]
        local_z=logit(np.clip(local,1e-6,1-1e-6))
        for oe,e in zip(control,new):
            alpha=e['alpha'];assert e['power']==8
            old_score,_=load(PRIOR,oe['candidate'],oe)
            uniform=local.copy() if alpha==0 else expit(local_z+alpha*gz)
            assert np.array_equal(uniform,old_score)
            score,answer=load(OUT,e['candidate'],e)
            replay=local.copy() if alpha==0 else expit(local_z+alpha*gate*gz)
            assert np.array_equal(replay,score),e['candidate']
            if alpha==0:assert np.array_equal(score,local) and e['thresholds']==le['thresholds'] and e['metrics']==le['metrics']
            key=[min(e['thresholds']['window']['f1'],e['thresholds']['answer']['f1']),
                 e['thresholds']['window']['f1'],e['thresholds']['window']['precision'],-alpha]
            assert key==e['selection_key']
            checked.append({'candidate':e['candidate'],'new_score_maxdiff':0.,'uniform_score_maxdiff':0.,
                'all_window_scores_checked':len(score),'answer_max_all_windows_exact':True,
                'fit_cal_metrics_and_thresholds_exact':True})
        choose=lambda es:max(es,key=lambda e:e['selection_key'])
        assert choose(new)==summary['selected'][peer] and choose(control)==summary['uniform_controls'][peer]
        peers[peer]={'selected_alpha':choose(new)['alpha'],'calibration':choose(new)['metrics']['calibration'],
                     'gate_min':float(gate.min()),'gate_max':float(gate.max()),'gate_peak_is_prediction_only':True}
    assert len(checked)==18 and all(p['selected_alpha']==0 for p in peers.values())
    result={'status':'passed','new_candidates_checked':18,'uniform_controls_checked':18,
        'all_scores_replayed_exact':True,'checks':checked,'peers':peers,
        'static_review':'Gate reads local predictions and own answer-max only. Gold appears only in metric/threshold selection, not gate or peak construction. Final answer=max over all final window predictions, not shared global answer override.',
        'new_training':False,'new_search':False,'GPU_used':False,'official_test_opened':False,
        'complete_sha256':sha(OUT/'complete.json'),'source_sha256':sha(ROOT/'src/run_local_gated_answer_scores.py')}
    (OUT/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','new':18,'uniform':18,'selected_alpha':{p:r['selected_alpha'] for p,r in peers.items()}}))


if __name__=='__main__':main()
