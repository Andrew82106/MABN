"""Calibrated forward/backward persistence on frozen development scores.

Independent small postprocessing study. The same budget is applied to strong
baselines and our current candidates; no pretrained model or new LR fit.
"""
from pathlib import Path
from collections import defaultdict
import argparse, importlib.util, itertools, pickle, time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r24_source22',ROOT.parent/'round22_evidence_verification/src/run22.py')
r22=importlib.util.module_from_spec(spec);spec.loader.exec_module(r22)
r18,r17,r10=r22.r18,r22.r17,r22.r10
read,readl,save,savel,sha=r22.read,r22.readl,r22.save,r22.savel,r22.sha
BASES=('slots_base','base_harp_delta','verifier_learned_slots','lookback_tuned')
METHODS=BASES+tuple(m+'_smooth' for m in BASES)
r18.METHODS=METHODS

def protocol():
    return {'version':'r24-window-persistence-v1','source':'https://arxiv.org/html/2606.31033v1#S3.SS4',
      'adaptation':'Forward/backward label-persistence applied to overlapping 4-raw-BPE window scores, not the full CORTEX method',
      'scope':'Repeated R16 actual-train event fivefold development; no fresh test claim',
      'base_methods':list(BASES),'methods':list(METHODS),'stay':[.5,.8,.95,.99],'temperature':[.5,1.,2.],
      'primary':'verifier_learned_slots_smooth','uniform_initial':True,'epsilon':1e-12,
      'sequence':'Every original answer item, sorted by raw-window start, all output-defined windows regardless of gold eligibility',
      'emissions':'p=sigmoid(logit(clipped saved probability)/temperature); [1-p,p]',
      'transition':'same label = stay; different label = 1-stay',
      'selection':'Calibration max min(window F1,answer F1), then window F1, precision, prefer raw identity, lower stay, lower abs(log temperature)',
      'threshold':'Calibration separate scales; F1 then precision then higher cutoff; answer=max all window posteriors',
      'future_windows_used':True,'online_detector':False,'gold_changed':False,'original_validation_or_test_used':False,
      'additional_fit':False,'all_calibration_choices_frozen_before_outer':True}

def smooth(s,stay,temp):
    s=np.asarray(s,np.float64)
    if stay==.5 and temp==1.:return s.copy()
    q=np.clip(s,1e-12,1-1e-12);p=expit((np.log(q)-np.log1p(-q))/temp)
    e=np.column_stack((1-p,p));n=len(s);f=np.empty((n,2));b=np.empty((n,2))
    tr=np.array([[stay,1-stay],[1-stay,stay]])
    f[0]=e[0]*.5;f[0]/=f[0].sum()
    for i in range(1,n):f[i]=e[i]*(f[i-1]@tr);f[i]/=f[i].sum()
    b[-1]=1
    for i in range(n-2,-1,-1):b[i]=tr@(e[i+1]*b[i+1]);b[i]/=b[i].sum()
    marginal=f*b;return marginal[:,1]/marginal.sum(1)

def geometry(windows):
    groups=defaultdict(list)
    for i,w in enumerate(windows):groups[w['item_ids'][0]].append(i)
    return [np.asarray(sorted(ix,key=lambda i:windows[i]['raw_token_indices'][0]),int) for ix in groups.values()]

def whole(scores,seqs,stay,temp):
    out=np.empty(len(scores),np.float64)
    for ix in seqs:out[ix]=smooth(scores[ix],stay,temp)
    return out

def selfcheck():
    s=np.array([.13,.72,.61,.22])
    for stay,temp in itertools.product(protocol()['stay'],protocol()['temperature']):
        got=smooth(s,stay,temp);p=expit(np.log(s/(1-s))/temp);total=0.;m=np.zeros(len(s))
        for seq in itertools.product([0,1],repeat=len(s)):
            value=.5
            for i,v in enumerate(seq):
                value*=p[i] if v else 1-p[i]
                if i:value*=stay if v==seq[i-1] else 1-stay
            total+=value;m+=value*np.asarray(seq)
        assert np.allclose(got,m/total,atol=2e-15)
    assert np.array_equal(smooth(s,.5,1),s)
    assert np.allclose(smooth(s[::-1],.95,1)[::-1],smooth(s,.95,1))
    print('R24_EXHAUSTIVE_16_SEQUENCES_PASSED',flush=True)

def prepare():
    # All source score artifacts were frozen and evaluated in R22 already.
    complete=read(r22.ROOT/'results/complete22.json')
    for n,h in complete['files_sha256'].items():assert sha(r22.ROOT/'results'/n)==h
    meta=r18.train_metadata();pack=r17.cohort(r18.NEW,'train',meta,r18.Bank(meta[2]))
    assert pack['windows']==readl(r22.ROOT/'results/candidate_windows22.jsonl')
    return pack,geometry(pack['windows'])

def fit():
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True);assert not (out/'calibration_started.json').exists()
    cfg=read(ROOT/'protocol.json');assert cfg==protocol();selfcheck();pack,seqs=prepare()
    snap={'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json'),
          'source_complete_sha256':sha(r22.ROOT/'results/complete22.json')}
    save(out/'calibration_started.json',{'utc':r10.utc(),'snapshot':snap});start=time.perf_counter()
    files=[]
    for fold in range(5):
        f=pickle.loads((r22.ROOT/'results'/f'fold_{fold}_frozen22.pkl').read_bytes())
        cal,cx=r18.subset(pack,f['calibration_groups']);values={};thresholds={};choices={};tables={}
        with np.load(r22.ROOT/'results'/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:
            for m in BASES:values[m]=z[m].copy();thresholds[m]=f['thresholds'][m]
        for m in BASES:
            table=[];candidate_scores=[]
            for stay,temp in itertools.product(cfg['stay'],cfg['temperature']):
                v=whole(values[m],seqs,stay,temp);ts=r22.thresholds(cal,v[cx]);w,a=ts['window'],ts['answer']
                key=[min(w['validation_f1'],a['validation_f1']),w['validation_f1'],w['validation_precision'],int(stay==.5 and temp==1),-stay,-abs(np.log(temp))]
                table.append({'stay':stay,'temperature':temp,'thresholds':ts,'key':key});candidate_scores.append(v)
            ix=max(range(len(table)),key=lambda i:table[i]['key']);new=m+'_smooth'
            values[new]=candidate_scores[ix];thresholds[new]=table[ix]['thresholds'];choices[new]=table[ix];tables[new]=table
        obj={'groups':{k:f[k] for k in ['fit_groups','calibration_groups','evaluation_groups']},'thresholds':thresholds,'choices':choices,'tables':tables}
        save(out/f'fold_{fold}_calibration.json',obj);np.savez_compressed(out/f'fold_{fold}_scores.npz',**values)
        files.extend([f'fold_{fold}_calibration.json',f'fold_{fold}_scores.npz']);print('R24_FROZEN',fold,flush=True)
    save(out/'calibration_freeze.json',{'snapshot':snap,'utc':r10.utc(),'seconds':time.perf_counter()-start,
          'files_sha256':{n:sha(out/n) for n in files}})

def test():
    out=ROOT/'results';freeze=read(out/'calibration_freeze.json');assert not (out/'test_started.json').exists()
    for n,h in freeze['files_sha256'].items():assert sha(out/n)==h
    assert freeze['snapshot']['code_sha256']==sha(Path(__file__))
    pack,_=prepare();save(out/'test_started.json',{'utc':r10.utc(),'freeze_sha256':sha(out/'calibration_freeze.json')})
    allw=[];alla=[]
    for fold in range(5):
        f=read(out/f'fold_{fold}_calibration.json');ev,ex=r18.subset(pack,f['groups']['evaluation_groups'])
        with np.load(out/f'fold_{fold}_scores.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        wr=None;ar=None
        for m in METHODS:
            w,a=r18.scored_records(ev,values[m][ex],f['thresholds'][m],m)
            if wr is None:wr=[dict(x,scores={},predictions={},fold=fold) for x in w];ar=[dict(x,scores={},predictions={},fold=fold) for x in a]
            for x,y in zip(wr,w):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
            for x,y in zip(ar,a):x['scores'].update(y['scores']);x['predictions'].update(y['predictions'])
        allw.extend(wr);alla.extend(ar)
    result=r18.pooled(allw,alla,pack);old=read(r22.ROOT/'results/summary22.json')['methods']
    for m in BASES:assert result[m]==old[m]
    savel(out/'window_scores_oof.jsonl',allw);savel(out/'answer_scores_oof.jsonl',alla)
    contrasts={m+'_smooth_vs_raw':[m+'_smooth',m] for m in BASES}
    summary={'scope':protocol()['scope'],'methods':result,'paired_bootstrap':{unit:r18.bootstrap(rows,{'draws':2000,'seed':20260924},contrasts) for unit,rows in [('windows',allw),('answers',alla)]},
             'future_windows_used':True,'source_labels_changed':False,'old_baselines_exact':True}
    save(out/'summary.json',summary)
    save(out/'complete.json',{'utc':r10.utc(),'source_complete_sha256':sha(r22.ROOT/'results/complete22.json'),
         'files_sha256':{n:sha(out/n) for n in ['calibration_freeze.json','summary.json','window_scores_oof.jsonl','answer_scores_oof.jsonl']}})
    for m,v in result.items():print(m,round(v['windows']['f1'],6),round(v['answers']['f1'],6),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','fit','test']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':
            assert not (ROOT/'protocol.json').exists();save(ROOT/'protocol.json',protocol());selfcheck()
        elif a.stage=='fit':fit()
        else:test()
