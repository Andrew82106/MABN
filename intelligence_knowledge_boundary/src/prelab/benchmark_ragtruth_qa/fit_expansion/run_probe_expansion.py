"""Controlled data-size comparison with the frozen original MiniCheck PCA64.

The original generator and MiniCheck are frozen. All selected-checkpoint,
annotation and calibration geometry stays fixed. No GPU is used here.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,gc,pickle,sys,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
import run_development as q
import run_semantic_hidden as h
OUT=HERE/'probe_v1';SEM=HERE/'minicheck';OLD=h.OUT

def protocol():
    return {'version':'qa-expanded-frozen-state-probe-v1','regimes':['original634','expanded3680'],
      'methods':['hidden64','hidden64_risk'],'C':[.00001,.0001,.001],'new_fits':9,'reused_fits':3,
      'projection':'Reuse exactly original fit634-only PCA64, never refit on auxiliary or calibration; isolate added training examples',
      'features':'Frozen MiniCheck mapped original raw BPE state mean64 per4raw window, optionally official claim risk logit maximum over lexical BPEs',
      'mapping':'Same run_semantic_hidden.align: character-overlap hidden averaging; all nonwhitespace covered; original Llama token geometry retained',
      'base_weights':'Expanded: each source-connected group equal; native634 and auxiliary3046 each half within group, then answer/window uniform in each stratum. If one stratum absent give existing stratum all mass. Original634 reuse exact old weights.',
      'loss_weights':'Same fit-only binary class factors then re-equalize group loss, total mass168123 for BOTH training sizes; no inadvertent C change from expanded window count',
      'standardization':'Fit-only base-weighted StandardScaler, blocks16384, float32 transformed features; common scaler across C within regime/method',
      'classifier':'liblinear L2, max_iter2000, seed20260924,4 CPU threads',
      'selection':'Calibration-only separate window/answer F1 thresholds and same q.selection_key across three Cs; test unopened',
      'evaluation':'Unchanged159 calibration answers/42241 windows; answer max all eligible windows including refusals',
      'scope':'All original634 plus3046 unique published human fit answers; same615 source groups, no extra independent questions',
      'generator_id':'Only native/aux loss stratum and provenance, never a predictor input',
      'extra_checker':True,'encoder_finetuned':False,'native_generator_whitebox':False,
      'script_sha256':q.sha(__file__),'mapping_code_sha256':q.sha(h.__file__),'base_code_sha256':q.sha(q.__file__),
      'export_freeze_sha256':q.sha(HERE/'data/export_freeze.json'),'old_pca_sha256':q.sha(OLD/'hidden_pca.pkl'),
      'old_preparation_sha256':q.sha(OLD/'preparation_complete.json'),'old_full_complete_sha256':q.sha(ROOT/'results/semantic_full_v1/complete.json')}

def metadata():
    original=q.metadata();aa=q.lines(HERE/'data/answers_fit.jsonl');tt=q.lines(HERE/'data/tokens_fit.jsonl');ww=q.lines(HERE/'data/windows_k4_fit.jsonl')
    assert len(aa)==len(tt)==3680 and len(ww)==653979
    assert aa[:634]==original['answers'][:634] and tt[:634]==original['tokens'][:634] and ww[:168123]==original['windows'][:168123]
    aa+=original['answers'][634:];tt+=original['tokens'][634:];ww+=original['windows'][168123:]
    by={a['response_id']:{'answer':a,'tokens':t} for a,t in zip(aa,tt)};aw=defaultdict(list)
    for j,w in enumerate(ww):aw[w['response_id']].append(j)
    assert len(by)==3839 and all(aw[a['response_id']] for a in aa)
    groups={p:{a['group_id'] for a in aa if a['partition']==p} for p in q.PARTITIONS}
    assert len(groups['fit'])==615 and len(groups['calibration'])==154 and not groups['fit']&groups['calibration']
    return original,{'answers':aa,'tokens':tt,'windows':ww,'by_response':by,'answer_windows':dict(aw),
      'bounds':{'fit':[0,653979],'calibration':[653979,len(ww)]},'gold':{'counts':{}}}

def source_snapshot():
    freeze=q.read(HERE/'data/export_freeze.json')
    for section in ('source_files_sha256','output_files_sha256'):
        for path,expected in freeze[section].items():assert q.sha(path)==expected,path
    for name,expected in q.read(OLD/'preparation_complete.json')['files_sha256'].items():assert q.sha(OLD/name)==expected
    oldfull=ROOT/'results/semantic_full_v1'
    for name,expected in q.read(oldfull/'complete.json')['files_sha256'].items():assert q.sha(oldfull/name)==expected
    done=q.read(SEM/'inference_complete.json');fm=q.read(SEM/'claim_feature_manifest.json')
    assert done['status']==fm['status']=='complete' and done['rows']==fm['answers']==3046 and not done['test_opened']
    assert q.sha(SEM/'claim_feature_manifest.json')==done['claim_feature_manifest_sha256']
    paths=[SEM/'inference_complete.json',SEM/'claim_feature_manifest.json',SEM/'new_plans.jsonl',HERE/'data/export_freeze.json',OLD/'preparation_complete.json',OLD/'hidden_pca.pkl']
    return {'files_sha256':{str(p.resolve()):q.sha(p) for p in paths},'new_features_complete':True,'official_test_opened':False}

def prepare():
    assert q.read(OUT/'protocol.json')==protocol();assert not (OUT/'preparation_started.json').exists()
    snap=source_snapshot();q.save(OUT/'preparation_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_snapshot':snap})
    old,meta=metadata();pca=pickle.loads((OLD/'hidden_pca.pkl').read_bytes())
    oldidx={a['response_id']:a for a in q.read(OLD/'token_index.json')['answers']}
    oldh=np.load(OLD/'matrices/token_hidden64.npy',mmap_mode='r');oldr=np.load(OLD/'matrices/token_risk.npy',mmap_mode='r')
    plans={p['response_id']:p for p in q.lines(SEM/'new_plans.jsonl')};manifest={r['response_id']:r for r in q.read(SEM/'claim_feature_manifest.json')['records']}
    texts={r['response_id']:r['original_response'] for r in q.lines(HERE/'data/fit.jsonl')}
    matrix=OUT/'matrices';matrix.mkdir(exist_ok=True);total=sum(t['token_count'] for t in meta['tokens']);assert total==708506
    token=np.lib.format.open_memmap(matrix/'token_hidden64.npy',mode='w+',dtype=np.float32,shape=(total,64));risk=np.zeros(total,np.float64)
    index=[];cursor=0
    for a,t in zip(meta['answers'],meta['tokens']):
        rid=a['response_id'];n=t['token_count']
        if rid in oldidx:
            ix=oldidx[rid];assert n==ix['right']-ix['left'];token[cursor:cursor+n]=oldh[ix['left']:ix['right']];risk[cursor:cursor+n]=oldr[ix['left']:ix['right']]
        else:
            path=SEM/'claim_features'/(rid+'.npz');rec=manifest[rid];assert q.sha(path)==rec['npz_sha256'] and q.sha(path.with_suffix('.json'))==rec['metadata_sha256']
            with np.load(path,allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
            mapped,r,lex,check=h.align(texts[rid],t['response_token_offsets'],arr,plans[rid]['claims']);assert lex.tolist()==t['lexical_mask']
            token[cursor:cursor+n]=((mapped.astype(np.float64)-pca['mean'])@pca['components'].T).astype(np.float32);risk[cursor:cursor+n]=r
        index.append({'response_id':rid,'partition':a['partition'],'group_id':a['group_id'],'left':cursor,'right':cursor+n});cursor+=n
    token.flush();np.save(matrix/'token_risk.npy',risk);q.save(OUT/'token_index.json',{'answers':index});starts={a['response_id']:a['left'] for a in index}
    win=np.lib.format.open_memmap(matrix/'window65.npy',mode='w+',dtype=np.float32,shape=(len(meta['windows']),65))
    for j,w in enumerate(meta['windows']):
        ix=np.asarray(w['token_indices'])+starts[w['response_id']];lex=np.asarray(meta['by_response'][w['response_id']]['tokens']['lexical_mask'],bool)[w['token_indices']]
        win[j,:64]=token[ix].mean(0);win[j,64]=h.logit(np.asarray([risk[ix][lex].max()]))[0]
    win.flush();origix=np.r_[np.arange(168123),np.arange(653979,len(win))]
    reference=np.load(OLD/'matrices/window_hidden64.npy',mmap_mode='r');assert np.array_equal(win[origix,:64],reference)
    oldlogit=np.load(OLD/'matrices/window_risk_logit.npy',mmap_mode='r');assert np.array_equal(win[origix,64],oldlogit)
    assert snap==source_snapshot();files=['token_index.json','matrices/token_hidden64.npy','matrices/token_risk.npy','matrices/window65.npy']
    q.save(OUT/'preparation_complete.json',{'source_snapshot':snap,'files_sha256':{n:q.sha(OUT/n) for n in files},'original_features_exact':True,'total_tokens':total,'official_test_opened':False})
    print('EXPANDED_PROBE_FEATURES_READY',len(win),total,flush=True)

def expanded_weights(meta):
    rows=meta['windows'][:653979];native={a['response_id'] for a in meta['answers'][:634]};tree=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for j,w in enumerate(rows):tree[w['group_id']][int(w['response_id'] in native)][w['response_id']].append(j)
    b=np.zeros(len(rows),np.float64)
    for strata in tree.values():
        for answers in strata.values():
            for ix in answers.values():b[ix]=1/(len(strata)*len(answers)*len(ix))
    b*=168123/b.sum();y=np.asarray([w['label'] for w in rows],int);mass=np.bincount(y,weights=b,minlength=2);factors=mass.sum()/(2*mass);loss=b*factors[y]
    for strata in tree.values():
        ix=[j for answers in strata.values() for values in answers.values() for j in values];loss[ix]*=(168123/len(tree))/loss[ix].sum()
    assert abs(loss.sum()-168123)<1e-6
    return b,loss,y,factors

def preflight():
    old,meta=metadata();b,loss,y,f=expanded_weights(meta)
    native={a['response_id'] for a in meta['answers'][:634]};totals=defaultdict(float);native_mass=0.
    for i,w in enumerate(meta['windows'][:653979]):
        totals[w['group_id']]+=b[i]
        if w['response_id'] in native:native_mass+=b[i]
    assert max(abs(v-168123/615) for v in totals.values())<1e-8
    assert abs(native_mass/b.sum()-.5)<1e-10
    assert (b>0).all() and (loss>0).all() and set(y)=={0,1}
    return {'passed':True,'fit_answers':3680,'fit_groups':615,'fit_windows':len(y),'fit_positive_windows':int(y.sum()),
        'native_base_mass_fraction':native_mass/b.sum(),'base_mass':float(b.sum()),'loss_mass':float(loss.sum()),
        'original_fit_and_calibration_geometry_preserved':True,'feature_files_opened':False,'GPU_used':False,'official_test_opened':False}

def fit():
    assert q.read(OUT/'protocol.json')==protocol();assert not (OUT/'fit_started.json').exists()
    prep=q.read(OUT/'preparation_complete.json')
    for n,s in prep['files_sha256'].items():assert q.sha(OUT/n)==s
    assert prep['source_snapshot']==source_snapshot();q.save(OUT/'fit_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'preparation_sha256':q.sha(OUT/'preparation_complete.json')})
    old,meta=metadata();raw=np.load(OUT/'matrices/window65.npy',mmap_mode='r');oldix=np.r_[np.arange(168123),np.arange(653979,len(raw))]
    all_entries={};selected={};files=[]
    for regime in protocol()['regimes']:
        mm=old if regime=='original634' else meta;nr=mm['bounds']['fit'][1]
        if regime=='original634':
            with np.load(q.OUT/'training_weights.npz') as z:b=z['base_weights'].copy();loss=z['loss_weights'].copy();y=z['y'].copy();f=z['class_factors'].copy()
            xx=raw[oldix]
        else:b,loss,y,f=expanded_weights(meta);xx=raw
        np.savez_compressed(OUT/(regime+'_weights.npz'),base=b,loss=loss,y=y,class_factors=f);files.append(regime+'_weights.npz')
        for method in protocol()['methods']:
            width=64 if method=='hidden64' else 65
            if regime=='original634' and method=='hidden64':sc=pickle.loads((ROOT/'results/semantic_full_v1/minicheck_hidden64_C1e-05.pkl').read_bytes())['scaler']
            else:
                sc=StandardScaler()
                for l in range(0,nr,q.BATCH):r=min(l+q.BATCH,nr);sc.partial_fit(xx[l:r,:width],sample_weight=b[l:r])
            x=np.empty((len(xx),width),np.float32)
            for l in range(0,len(x),q.BATCH):r=min(l+q.BATCH,len(x));x[l:r]=sc.transform(xx[l:r,:width]).astype(np.float32)
            entries=[];lo,hi=mm['bounds']['calibration'];ci=[i for i,a in enumerate(mm['answers']) if a['partition']=='calibration']
            for c in protocol()['C']:
                name=f'{regime}_{method}_C{c:g}';start=time.perf_counter();reuse=regime=='original634' and method=='hidden64'
                if reuse:clf=pickle.loads((ROOT/f'results/semantic_full_v1/minicheck_hidden64_C{c:g}.pkl').read_bytes())['model']
                else:clf=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=20260924).fit(x[:nr],y,sample_weight=loss)
                assert clf.n_iter_.max()<2000;s=clf.predict_proba(x)[:,1];ans=q.answer_scores(mm,s)
                ts={'window':q.choose_threshold([w['label'] for w in mm['windows'][lo:hi]],s[lo:hi]),'answer':q.choose_threshold([mm['answers'][j]['label'] for j in ci],ans[ci])}
                if reuse:
                    with np.load(ROOT/f'results/semantic_full_v1/minicheck_hidden64_C{c:g}_scores.npz') as z:assert np.max(np.abs(s-z['window_scores']))<1e-12
                obj={'model':clf,'scaler':sc,'C':c,'thresholds':ts,'regime':regime,'method':method,'fit_rows':nr,'fit_only':True,'weight_sha256':q.sha(OUT/(regime+'_weights.npz'))}
                (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=s,answer_scores=ans)
                e={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),'metrics':q.metrics(mm,s,ts),'seconds':time.perf_counter()-start,'reused':reuse,
                   'scores_sha256':q.sha(OUT/(name+'_scores.npz')),'model_sha256':q.sha(OUT/(name+'.pkl'))}
                q.save(OUT/(name+'_result.json'),e);entries.append(e);files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json']);print('EXPANDED_PROBE_FIT',name,round(e['seconds'],1),flush=True)
            key=regime+'_'+method;all_entries[key]=entries;selected[key]=max(entries,key=lambda e:e['selection_key']);del x;gc.collect()
    q.save(OUT/'summary.json',{'selected':selected,'all_candidates':all_entries,'new_fits':9,'old_replays':3,'official_test_opened':False,'calibration_selection_optimistic':True})
    files+=['summary.json','protocol.json','preparation_complete.json','fit_started.json'];q.save(OUT/'complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'files_sha256':{n:q.sha(OUT/n) for n in files},'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','fit']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':
            assert not (OUT/'protocol.json').exists();q.save(OUT/'PREFLIGHT.json',preflight());q.save(OUT/'protocol.json',protocol());print('EXPANDED_PROBE_PROTOCOL_FROZEN_NO_TRAINING')
        elif a.stage=='prepare':prepare()
        else:fit()
