"""Frozen-score combination of global and focused evidence verification."""
from pathlib import Path
import argparse,importlib.util,time,pickle
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r26_r23',ROOT.parent/'round23b_local_evidence_probe/src/run23.py')
r23=importlib.util.module_from_spec(spec);spec.loader.exec_module(r23)
r18,r17,r10=r23.r18,r23.r17,r23.r10
read,save,sha,savel=r23.read,r23.save,r23.sha,r23.savel
BASES=r23.METHODS
LOCAL=('slots_base_smooth','local_mean_fusion','local_slots_fusion_smooth')
NEW=tuple(m+'_global' for m in LOCAL);METHODS=BASES+NEW;r18.METHODS=METHODS


def protocol():
    return {'version':'r26-global-local-logodds-v1','scope':'Repeated R16 actual-train development, not fresh test',
      'global':'R22 frozen learned answer verifier broadcast','local':list(LOCAL),'methods':list(METHODS),
      'alpha':[0.,.125,.25,.5,1.,2.],'formula':'sigmoid(logit(local)+alpha*logit(global)); clip1e-12; alpha0 exact identity',
      'selection':'Same six alpha candidates per local stream; cal max min(windowF1,answerF1), then windowF1, precision, smaller alpha',
      'threshold':'Original separate cal thresholds, all-candidate-window max for answer including safe refusals',
      'new_model_training':False,'new_extraction':False,'all5fold_choices_frozen_before_outer':True,
      'original_labels_geometry_or_outputs_changed':False,'human_gold':False,'original_validation_or_test_used':False,
      'interpretation':'Combines two learned risks; scores are not calibrated truth probabilities; no resubstitution fitting.'}


def combine(local,global_score,alpha):
    if alpha==0:return local.copy()
    p=np.clip(local,1e-12,1-1e-12);g=np.clip(global_score,1e-12,1-1e-12)
    return expit(np.log(p)-np.log1p(-p)+alpha*(np.log(g)-np.log1p(-g)))


def prepare():
    complete=read(r23.ROOT/'results/complete.json')
    for n,h in complete['files_sha256'].items():assert sha(r23.ROOT/'results'/n)==h
    meta=r18.train_metadata();pack=r17.cohort(r18.NEW,'train',meta,r18.Bank(meta[2]))
    assert pack['windows']==r23.readl(r23.ROOT/'results/candidate_windows.jsonl')
    return pack


def calibrate():
    out=ROOT/'results';assert not (out/'started.json').exists() and read(ROOT/'protocol.json')==protocol()
    pack=prepare();save(out/'started.json',{'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json'),
      'source_complete_sha256':sha(r23.ROOT/'results/complete.json')});start=time.perf_counter();files=[]
    for fold in range(5):
        old=pickle.loads((r23.ROOT/'results'/f'fold_{fold}_frozen.pkl').read_bytes())
        cal,cx=r18.subset(pack,old['groups']['calibration_groups'])
        with np.load(r23.ROOT/'results'/f'fold_{fold}_scores.npz',allow_pickle=False) as z:v={m:z[m].copy() for m in BASES}
        ts={m:old['thresholds'][m] for m in BASES};tables={};choices={}
        for local,new in zip(LOCAL,NEW):
            table=[];scores=[]
            for alpha in protocol()['alpha']:
                score=combine(v[local],v['verifier_learned_broadcast'],alpha);t=r23.r22.thresholds(cal,score[cx]);w,a=t['window'],t['answer']
                key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],-alpha]
                table.append({'alpha':alpha,'thresholds':t,'key':key});scores.append(score)
                if alpha==0:assert np.array_equal(score,v[local]) and t==ts[local]
            j=max(range(len(table)),key=lambda i:table[i]['key']);choices[new]=table[j];tables[new]=table
            v[new]=scores[j];ts[new]=table[j]['thresholds']
        obj={'groups':old['groups'],'thresholds':ts,'tables':tables,'choices':choices}
        save(out/f'fold_{fold}_calibration.json',obj);np.savez_compressed(out/f'fold_{fold}_scores.npz',**v)
        files.extend([f'fold_{fold}_calibration.json',f'fold_{fold}_scores.npz'])
        print('R26_CALIBRATED',fold,{k:x['alpha'] for k,x in choices.items()},flush=True)
    save(out/'calibration_freeze.json',{'seconds':time.perf_counter()-start,'source_sha256':sha(r23.ROOT/'results/complete.json'),
      'code_sha256':sha(Path(__file__)),'files_sha256':{n:sha(out/n) for n in files}})


def test():
    out=ROOT/'results';frozen=read(out/'calibration_freeze.json');assert not (out/'test_started.json').exists()
    assert sha(Path(__file__))==frozen['code_sha256']
    for n,h in frozen['files_sha256'].items():assert sha(out/n)==h
    pack=prepare();save(out/'test_started.json',{'freeze_sha256':sha(out/'calibration_freeze.json')});allw=[];alla=[]
    for fold in range(5):
        f=read(out/f'fold_{fold}_calibration.json');ev,ix=r18.subset(pack,f['groups']['evaluation_groups'])
        with np.load(out/f'fold_{fold}_scores.npz',allow_pickle=False) as z:values={m:z[m].copy() for m in METHODS}
        wr=ar=None
        for m in METHODS:
            w,a=r18.scored_records(ev,values[m][ix],f['thresholds'][m],m)
            if wr is None:wr=[dict(x,scores={},predictions={},fold=fold) for x in w];ar=[dict(x,scores={},predictions={},fold=fold) for x in a]
            for x,y in zip(wr,w):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
            for x,y in zip(ar,a):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
        allw.extend(wr);alla.extend(ar)
    result=r18.pooled(allw,alla,pack);old=read(r23.ROOT/'results/summary.json')['methods']
    for m in BASES:assert result[m]==old[m]
    contrast={new+'_vs_local':[new,local] for local,new in zip(LOCAL,NEW)}
    summary={'methods':result,'old_baselines_exact':True,'scope':protocol()['scope'],
      'paired_bootstrap':{unit:r18.bootstrap(rows,{'draws':2000,'seed':20260926},contrast) for unit,rows in [('windows',allw),('answers',alla)]}}
    savel(out/'window_scores_oof.jsonl',allw);savel(out/'answer_scores_oof.jsonl',alla);save(out/'summary.json',summary)
    save(out/'complete.json',{'files_sha256':{n:sha(out/n) for n in ['calibration_freeze.json','summary.json','window_scores_oof.jsonl','answer_scores_oof.jsonl']}})
    for m in NEW:print(m,result[m]['windows']['f1'],result[m]['answers']['f1'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('initialize','calibrate','test'));a=p.parse_args()
    if a.stage=='initialize':
        assert not (ROOT/'protocol.json').exists();x=np.array([.1,.5,.9]);assert np.array_equal(combine(x,x,0),x)
        assert np.allclose(combine(x,np.full(3,.5),1),x,rtol=0,atol=2e-16)
        save(ROOT/'protocol.json',protocol())
    elif a.stage=='calibrate':calibrate()
    else:test()
