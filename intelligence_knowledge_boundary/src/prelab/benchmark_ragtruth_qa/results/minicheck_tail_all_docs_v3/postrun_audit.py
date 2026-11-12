"""CPU-only audit of saved scores; never loads or forwards a neural model."""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_development as q


def threshold_from_roc(y,s):
    """Independent sklearn cumulative confusion counts, original fixed tie rule."""
    y=np.asarray(y,int);s=np.asarray(s,float);positive=int(y.sum());negative=len(y)-positive
    fp_rate,tp_rate,thresholds=roc_curve(y,s,drop_intermediate=False)
    tp=np.rint(tp_rate*positive).astype(np.int64);fp=np.rint(fp_rate*negative).astype(np.int64)
    f1=2*tp/(tp+fp+positive);precision=np.divide(tp,tp+fp,out=np.zeros(len(tp),float),where=tp+fp>0)
    thresholds=np.where(np.isinf(thresholds),np.nextafter(s.max(),np.inf),thresholds)
    best=max(range(len(tp)),key=lambda i:(float(f1[i]),float(precision[i]),float(thresholds[i])))
    return float(thresholds[best])


def metrics(y,s,t):
    y=np.asarray(y,int);s=np.asarray(s,float);tn,fp,fn,tp=map(int,confusion_matrix(y,s>=t,labels=[0,1]).ravel())
    return {'n':len(y),'positive':int(y.sum()),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'auroc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s))}


def same_metrics(actual,recorded):
    assert set(actual)==set(recorded)
    for key,value in actual.items():
        if key in ('n','positive','tp','fp','fn','tn'):assert value==recorded[key],key
        else:assert abs(value-recorded[key])<=1e-12,(key,value,recorded[key])


def main():
    assert (OUT/'tail2/complete.json').exists(),'Wait for all fixed training before auditing'
    canonical_w=np.asarray([w['label'] for w in q.lines(ROOT/'data/windows_k4_calibration.jsonl')])
    canonical_a=np.asarray([a['label'] for a in q.lines(ROOT/'data/answers_calibration.jsonl')])
    assert len(canonical_w)==42241 and len(canonical_a)==159
    all_records={}
    for condition in ('frozen0','tail2'):
        directory=OUT/condition;complete=q.read(directory/'complete.json');records=[]
        assert complete['status']=='complete_development_only' and complete['test_opened'] is False
        for epoch in range(4):
            prefix=f'epoch_{epoch:02d}';record=q.read(directory/(prefix+'.json'))
            assert record['epoch']==epoch and record['optimizer_steps']==(460 if epoch else 0)
            for suffix,expected in record['artifacts_sha256'].items():assert q.sha(directory/(prefix+suffix))==expected
            assert np.isfinite(record['fit_weighted_bce']) and record['fit_weighted_bce']>0
            assert np.isfinite(record['online_minibatch_objective_sum'])
            with np.load(directory/(prefix+'_scores.npz'),allow_pickle=False) as z:
                assert np.array_equal(z['cal_window_labels'],canonical_w)
                assert np.array_equal(z['cal_answer_labels'],canonical_a)
                cal={};fit={}
                for unit,plural in (('window','windows'),('answer','answers')):
                    y=z[f'cal_{unit}_labels'];s=z[f'cal_{unit}_scores'];cut=threshold_from_roc(y,s)
                    assert cut==record['thresholds'][unit]['threshold']
                    cal[plural]=metrics(y,s,cut);same_metrics(cal[plural],record['calibration'][plural])
                    fit[plural]=metrics(z[f'fit_{unit}_labels'],z[f'fit_{unit}_scores'],cut)
                    same_metrics(fit[plural],record['fit_at_cal_thresholds'][plural])
                key=[min(cal['windows']['f1'],cal['answers']['f1']),cal['windows']['f1'],cal['windows']['precision'],-epoch]
                assert key==record['selection_key']
            records.append(record)
        assert records==complete['all_epochs']
        best=max(records[1:],key=lambda r:r['selection_key'])
        assert complete['selected']==best
        all_records[condition]={'selected_epoch':best['epoch'],'selected_calibration':best['calibration'],
            'selected_fit_at_cal_thresholds':best['fit_at_cal_thresholds'],
            'fit_weighted_bce_by_epoch':[r['fit_weighted_bce'] for r in records],
            'window_f1_by_epoch':[r['calibration']['windows']['f1'] for r in records],
            'answer_f1_by_epoch':[r['calibration']['answers']['f1'] for r in records],
            'epoch_seconds':[r['seconds'] for r in records],
            'peak_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in records),
            'training_loop_seconds':complete['seconds'],'complete_sha256':q.sha(directory/'complete.json')}
    q.save(OUT/'POSTRUN_SCORE_AUDIT.json',{'passed':True,'audit':'Saved-score CPU audit, independent sklearn ROC threshold counts',
        'all_epochs':4,'conditions':all_records,'calibration_window_labels_equal_original_human_export':True,
        'thresholds_metrics_selection_and_checkpoint_hashes_verified':True,'neural_model_loaded':False,'GPU_used':False,
        'official_test_opened':False,'auditor_sha256':q.sha(Path(__file__))})
    print('POSTRUN_SAVED_SCORE_AUDIT_PASSED',flush=True)


if __name__=='__main__':main()
