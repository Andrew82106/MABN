"""Equal-budget persistence calibration for frozen QA LR and sequence probes."""
from pathlib import Path
import argparse
import itertools
import time
import numpy as np
from scipy.special import expit
import run_development as base

ROOT=base.ROOT
OUT=ROOT/'results/persistence_v1'
SOURCES=('development_v1','harp_development_v1','sequence_v1')


def protocol():
    return {'version':'qa-window-persistence-v1','scope':'Existing fit634/cal159 only; no official test',
      'source_folders':list(SOURCES),'source_selection':'Already frozen selected C/epoch per family; never refit',
      'stay':[.5,.8,.95,.99],'temperature':[.5,1.,2.], 'uniform_initial':True,'epsilon':1e-12,
      'emission':'sigmoid(logit(clipped original risk)/temperature)',
      'geometry':'Unchanged eligible 4 raw BPE windows; separate chains for each answer and each gap in raw start positions',
      'selection':'Cal max min(windowF1,answerF1), windowF1, precision, exact identity, lower stay, temperature nearest1',
      'threshold':'Separate cal F1 then precision then higher threshold for windows and answers',
      'answer_score':'Maximum over ALL unchanged eligible windows; no gold-dependent filtering',
      'budget':'Same12 candidates per selected LR/MLP/TCN, with exact identity included',
      'new_training':False,'future_windows_used':True,'online_detector':False,
      'reference':'https://arxiv.org/html/2606.31033v1','full_CORTEX_reproduction':False,
      'interpretation':'Calibration selects parameters; reported F1 is optimistic development performance, not held-out test.'}


def smooth(s,stay,temp):
    s=np.asarray(s,np.float64)
    if stay==.5 and temp==1.:return s.copy()
    q=np.clip(s,1e-12,1-1e-12);p=expit((np.log(q)-np.log1p(-q))/temp)
    e=np.column_stack((1-p,p));f=np.empty_like(e);b=np.empty_like(e)
    tr=np.array([[stay,1-stay],[1-stay,stay]])
    f[0]=e[0]*.5;f[0]/=f[0].sum()
    for i in range(1,len(s)):
        f[i]=e[i]*(f[i-1]@tr);f[i]/=f[i].sum()
    b[-1]=1
    for i in range(len(s)-2,-1,-1):
        b[i]=tr@(e[i+1]*b[i+1]);b[i]/=b[i].sum()
    m=f*b
    return m[:,1]/m.sum(1)


def chains(meta):
    result=[]
    for a in meta['answers']:
        ids=np.asarray(meta['answer_windows'][a['response_id']],int)
        starts=np.asarray([meta['windows'][i]['token_start'] for i in ids])
        assert np.all(np.diff(starts)>0)
        result.extend(np.split(ids,np.flatnonzero(np.diff(starts)!=1)+1))
    assert np.array_equal(np.sort(np.concatenate(result)),np.arange(len(meta['windows'])))
    return result


def transform(s,seqs,stay,temp):
    out=s.copy()
    for ix in seqs:out[ix]=smooth(s[ix],stay,temp)
    return out


def selfcheck():
    s=np.asarray([.13,.72,.61,.22])
    for stay,temp in itertools.product(protocol()['stay'],protocol()['temperature']):
        p=expit(np.log(s/(1-s))/temp);total=0.;num=np.zeros(len(s))
        for path in itertools.product((0,1),repeat=len(s)):
            w=.5
            for j,v in enumerate(path):
                w*=p[j] if v else 1-p[j]
                if j:w*=stay if v==path[j-1] else 1-stay
            num+=w*np.asarray(path);total+=w
        assert np.allclose(smooth(s,stay,temp),num/total,atol=2e-15,rtol=0)
    assert np.array_equal(smooth(s,.5,1),s)
    joined=np.r_[s,1-s];seqs=[np.arange(4),np.arange(4,8)]
    got=transform(joined,seqs,.95,1)
    assert np.array_equal(got[:4],smooth(s,.95,1))
    assert np.array_equal(got[4:],smooth(1-s,.95,1))


def sources(meta):
    result={};snapshot={}
    for folder in SOURCES:
        source=ROOT/'results'/folder
        complete=base.read(source/'complete.json')
        for n,h in complete['files_sha256'].items():assert base.sha(source/n)==h
        summary=base.read(source/'summary.json')
        snapshot[folder]={'complete_sha256':base.sha(source/'complete.json'),'summary_sha256':base.sha(source/'summary.json')}
        for name,entry in summary['selected'].items():
            filename=(f"{name}/epoch_{entry['epoch']:03d}_scores.npz" if folder=='sequence_v1' else entry['candidate']+'_scores.npz')
            with np.load(source/filename,allow_pickle=False) as z:
                v=z['window_scores'].astype(np.float64);assert np.array_equal(base.answer_scores(meta,v),z['answer_scores'])
            assert name not in result and len(v)==210364
            result[name]=(v,entry)
            snapshot[folder].setdefault('scores',{})[filename]=base.sha(source/filename)
    return result,snapshot


def run():
    assert not (OUT/'started.json').exists()
    assert base.read(OUT/'protocol.json')==protocol()
    selfcheck();meta=base.metadata();seqs=chains(meta);data,snap=sources(meta)
    cal_start=meta['bounds']['calibration'][0];calseqs=[ix for ix in seqs if ix[0]>=cal_start]
    assert all(ix[-1]<cal_start for ix in seqs if ix[0]<cal_start)
    base.save(OUT/'started.json',{'source':snap,'code_sha256':base.sha(Path(__file__)),
      'protocol_sha256':base.sha(OUT/'protocol.json'),'gold_sha256':base.sha(ROOT/'data/gold_manifest.json'),
      'chains':len(seqs),'calibration_chains':len(calseqs),'test_opened':False})
    y=np.asarray([w['label'] for w in meta['windows'][cal_start:]])
    ya=np.asarray([a['label'] for a in meta['answers'] if a['partition']=='calibration'])
    selected={};tables={};start=time.perf_counter()
    for name,(raw,old) in data.items():
        table=[]
        for stay,temp in itertools.product(protocol()['stay'],protocol()['temperature']):
            v=transform(raw,calseqs,stay,temp)
            ts={'window':base.choose_threshold(y,v[cal_start:]),'answer':base.choose_threshold(ya,base.answer_scores(meta,v)[634:])}
            w,a=ts['window'],ts['answer']
            key=[min(w['f1'],a['f1']),w['f1'],w['precision'],int(stay==.5 and temp==1),-stay,-abs(np.log(temp))]
            table.append({'stay':stay,'temperature':temp,'thresholds':ts,'key':key})
            if stay==.5 and temp==1:assert ts==old['thresholds']
        j=max(range(len(table)),key=lambda j:table[j]['key']);best=table[j]
        v=transform(raw,seqs,best['stay'],best['temperature'])
        metrics=base.metrics(meta,v,best['thresholds'])
        np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=v,answer_scores=base.answer_scores(meta,v))
        selected[name]={'selected_index':j,**best,'metrics':metrics,'raw_metrics':old['metrics']};tables[name]=table
        base.save(OUT/'progress.json',{'completed':list(selected),'expected':len(data)})
        print('QA_PERSISTENCE',name,metrics['calibration']['windows']['f1'],metrics['calibration']['answers']['f1'],flush=True)
    base.save(OUT/'summary.json',{'selected':selected,'all_candidates':tables,'seconds':time.perf_counter()-start,
      'calibration_results_are_selection_optimistic':True,'test_opened':False,'old_identity_thresholds_exact':True})
    files=['protocol.json','started.json','summary.json']+[n+'_scores.npz' for n in selected]
    base.save(OUT/'complete.json',{'status':'complete','files_sha256':{n:base.sha(OUT/n) for n in files},'test_opened':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('initialize','run'));a=p.parse_args()
    if a.stage=='initialize':
        assert not (OUT/'protocol.json').exists();selfcheck();base.save(OUT/'protocol.json',protocol())
    else:run()
