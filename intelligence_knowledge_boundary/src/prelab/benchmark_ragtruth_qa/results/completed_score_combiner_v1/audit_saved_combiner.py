"""Bounded replay of three frozen CPU trees and saved-score count checks."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
QA=OUT.parents[1]
FUSION=QA/'results/completed_score_fusion_v1'
sys.path.insert(0,str(FUSION))
from audit_saved_scores import metric,best_threshold

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]

def run():
    summary=read(OUT/'summary.json');complete=read(OUT/'complete.json')
    assert sha(OUT/'summary.json')==complete['summary_sha256'] and complete['fixed_fits_completed']==3
    previous=read(FUSION/'summary.json');assert sha(FUSION/'summary.json')==read(FUSION/'complete.json')['summary_sha256']
    protocol=read(OUT/'protocol.json');w=rows(QA/'data/windows_k4_fit.jsonl')+rows(QA/'data/windows_k4_calibration.jsonl')
    a=rows(QA/'data/answers_fit.jsonl')+rows(QA/'data/answers_calibration.jsonl')
    wy=np.asarray([x['label'] for x in w]);ay=np.asarray([x['label'] for x in a]);aid={x['response_id']:i for i,x in enumerate(a)}
    owner=np.asarray([aid[x['response_id']] for x in w]);assert len(wy)==210364 and len(ay)==793
    checks={}
    for peer,entry in summary['methods'].items():
        features=[]
        for alpha in (0.,1.):
            endpoint=next(x for x in previous['all_candidates'][peer] if x['tail_weight']==alpha)
            path=FUSION/(endpoint['candidate']+'_scores.npz');assert sha(path)==endpoint['scores_sha256']
            with np.load(path,allow_pickle=False) as z:features.append(z['window_scores'].copy())
        x=np.column_stack(features);assert x.shape==(210364,2)
        path=OUT/(peer+'.pkl');assert sha(path)==entry['model_sha256']
        model=pickle.loads(path.read_bytes())
        assert model.n_features_in_==2 and model.n_iter_==100 and model.classes_.tolist()==[0,1]
        hyper={k:v for k,v in protocol['model'].items() if k!='type'}
        assert hyper==entry['parameters']
        for key,value in hyper.items():assert model.get_params()[key]==value
        assert model.early_stopping is False and model.monotonic_cst==[1,1]
        assert len(model._predictors)==100
        actual=model.predict_proba(x)[:,1]
        path=OUT/(peer+'_scores.npz');assert sha(path)==entry['scores_sha256']
        with np.load(path,allow_pickle=False) as z:wv=z['window_scores'].copy();av=z['answer_scores'].copy()
        assert np.array_equal(actual,wv),'Saved model must exactly reproduce all saved probabilities'
        maxima=np.full(793,-np.inf);np.maximum.at(maxima,owner,wv)
        assert np.array_equal(maxima,av)
        counts={}
        for partition,wl,wr,al,ar in [('fit',0,168123,0,634),('calibration',168123,210364,634,793)]:
            counts[partition]={}
            for unit,y,s,th in [('windows',wy[wl:wr],wv[wl:wr],entry['thresholds']['window']['threshold']),
                                ('answers',ay[al:ar],av[al:ar],entry['thresholds']['answer']['threshold'])]:
                result=metric(y,s,th)
                assert all(value==entry['metrics'][partition][unit][key] for key,value in result.items())
                counts[partition][unit]=result
        ts={'window':best_threshold(wy[168123:],wv[168123:]),'answer':best_threshold(ay[634:],av[634:])}
        assert ts==entry['thresholds']
        fixed=previous['selected'][peer]['metrics']['calibration'];assert fixed==entry['same_peer_best_fixed_weight']
        delta={unit:counts['calibration'][unit]['f1']-fixed[unit]['f1'] for unit in ('windows','answers')}
        checks[peer]={'all210364_saved_model_probabilities_exact':True,'all793_answer_maxima_exact':True,
            'fixed_hyperparameters_and100_iterations_verified':True,'cal_thresholds_independently_verified':True,
            'counts':counts,'delta_vs_best_same_peer_fixed_weight_calibration_F1':delta,
            'improves_both_F1':all(v>0 for v in delta.values()),'model_sha256':entry['model_sha256']}
    assert set(checks)=={'lookback','harp_claim','semantic_claim'}
    result={'status':'passed','checks':checks,'scope':'Three frozen saved CPU trees replayed, direct counts/P/R/F1 and calibration thresholds only; no re-fit or neural forward.',
        'fit_scope':'Original native634 subset, not full3680','calibration_answers':159,'calibration_windows':42241,
        'upstream_fit_scores_are_not_cross_fitted':True,'no_method_improves_both_F1':not any(v['improves_both_F1'] for v in checks.values()),
        'GPU_used':False,'new_fitting':False,'official_test_opened':False,'AUROC_AP_recomputed':False,
        'summary_sha256':sha(OUT/'summary.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'script_sha256':sha(__file__),'independent_count_utility_sha256':sha(FUSION/'audit_saved_scores.py')}
    (OUT/'INDEPENDENT_REPLAY_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','methods':{k:{'window_f1':v['counts']['calibration']['windows']['f1'],
        'answer_f1':v['counts']['calibration']['answers']['f1'],'delta':v['delta_vs_best_same_peer_fixed_weight_calibration_F1']} for k,v in checks.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
