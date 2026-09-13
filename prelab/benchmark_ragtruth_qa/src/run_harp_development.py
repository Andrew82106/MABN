"""Fixed QA HARP-inspired projection controls, with identical LR budget."""
from pathlib import Path
import importlib.util,argparse,pickle,time,gc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/harp_development_v1'
spec=importlib.util.spec_from_file_location('qa_frozen_base',ROOT/'src/run_development.py')
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
METHODS=('harp256','harp64_lookback_nll','harp256_lookback_nll');WIDTHS=(256,1089,1281);BATCH=16384

def protocol():
    return {'version':'qa-harp-projection-controls-v1','methods':list(METHODS),'widths':dict(zip(METHODS,WIDTHS)),
      'C':[.001,.01,.1],'seed':20260927,'fit_windows':168123,'calibration_windows':42241,
      'source':'https://arxiv.org/html/2509.11536v2','adaptation':'Only bottom singular subspace projection; local supervised LR, not full original HARP training',
      'rank':'Bottom64 provides same-width alternative to fit-PCA64; bottom256 also tested without/with baseline LB+NLL',
      'data_and_weights':'Exact same all-window QA fit/cal and frozen base/class weights as development_v1; all human spans unchanged',
      'standardization':'Fit-only weighted StandardScaler partial_fit, same batch16384',
      'selection':'Cal max min(windowF1,answerF1), then windowF1, window precision, lowerC; separate cal thresholds',
      'fit_counts':9,'test_opened':False,'candidate_budget_same_as_baseline':True}

def run():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'started.json').exists()
    cfg=qa.read(ROOT/'harp_development_protocol.json');assert cfg==protocol()
    meta=qa.metadata();hm=qa.read(ROOT/'data/harp_manifest.json');assert hm['complete'] and hm['record_count']==793 and hm['token_count']==213159
    prior=qa.read(qa.OUT/'complete.json')
    for n,h in prior['files_sha256'].items():assert qa.sha(qa.OUT/n)==h
    snapshot={'code_sha256':qa.sha(Path(__file__)),'protocol_sha256':qa.sha(ROOT/'harp_development_protocol.json'),
              'harp_manifest_sha256':qa.sha(ROOT/'data/harp_manifest.json'),'prior_complete_sha256':qa.sha(qa.OUT/'complete.json')}
    qa.save(OUT/'started.json',snapshot);started=time.perf_counter();n=len(meta['windows'])
    hwin=np.lib.format.open_memmap(OUT/'harp_windows.npy',mode='w+',dtype=np.float32,shape=(n,256))
    for rec in hm['records']:
        rid=rec['response_id'];p=ROOT/'data/harp_features'/(rid+'.npz');assert qa.sha(p)==rec['npz_sha256']
        with np.load(p,allow_pickle=False) as z:
            h=z['harp'];t=meta['by_response'][rid]['tokens'];assert h.shape==(t['token_count'],256)
            assert z['token_ids'].tolist()==t['token_ids'] and z['response_token_offsets'].tolist()==t['response_token_offsets']
            for j in meta['answer_windows'][rid]:hwin[j]=h[meta['windows'][j]['token_indices']].mean(0)
    hwin.flush();base=np.load(qa.OUT/'matrices/base.npy',mmap_mode='r')
    with np.load(qa.OUT/'training_weights.npz') as z:b=z['base_weights'];loss=z['loss_weights'];y=z['y']
    assert len(y)==168123 and np.array_equal(y,np.asarray([w['label'] for w in meta['windows'][:168123]]))
    ci=[j for j,a in enumerate(meta['answers']) if a['partition']=='calibration'];lo,hi=meta['bounds']['calibration']
    families={};selected={};files=[]
    for method,width in zip(METHODS,WIDTHS):
        def raw(left,right):
            if method=='harp256':return np.asarray(hwin[left:right])
            return np.column_stack((base[left:right],hwin[left:right,:64 if method=='harp64_lookback_nll' else 256]))
        sc=StandardScaler()
        for left in range(0,168123,BATCH):right=min(left+BATCH,168123);sc.partial_fit(raw(left,right),sample_weight=b[left:right])
        fitpath=OUT/(method+'_fit.npy');zfit=np.lib.format.open_memmap(fitpath,mode='w+',dtype=np.float32,shape=(168123,width))
        for left in range(0,168123,BATCH):right=min(left+BATCH,168123);zfit[left:right]=sc.transform(raw(left,right)).astype(np.float32)
        zfit.flush();entries=[]
        for c in cfg['C']:
            start=time.perf_counter();model=LogisticRegression(C=c,solver='liblinear',max_iter=2000,penalty='l2',random_state=cfg['seed'])
            model.fit(zfit,y,sample_weight=loss);assert model.n_iter_.max()<2000
            scores=np.empty(n,np.float64)
            for left in range(0,n,BATCH):right=min(left+BATCH,n);scores[left:right]=model.predict_proba(sc.transform(raw(left,right)).astype(np.float32))[:,1]
            answer=qa.answer_scores(meta,scores)
            ts={'window':qa.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],scores[lo:hi]),'answer':qa.choose_threshold([meta['answers'][j]['label'] for j in ci],answer[ci])}
            key=qa.selection_key(ts,c);name=f'{method}_C{c:g}'
            obj={'model':model,'scaler':sc,'C':c,'method':method,'thresholds':ts,'width':width,'fit_only':True,'selection_key':key,'snapshot':snapshot}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=scores,answer_scores=answer)
            entry={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(key),'metrics':qa.metrics(meta,scores,ts),'seconds':time.perf_counter()-start,'iterations':model.n_iter_.tolist()}
            qa.save(OUT/(name+'_result.json'),entry);files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json']);entries.append(entry)
            print('QA_HARP_FIT',name,'seconds',round(entry['seconds'],1),flush=True)
        families[method]=entries;selected[method]=max(entries,key=lambda x:x['selection_key']);del zfit;gc.collect()
    summary={'selected':selected,'all_candidates':families,'test_opened':False,'calibration_results_are_selection_optimistic':True,'elapsed_seconds':time.perf_counter()-started,
             'baseline_selected':qa.read(qa.OUT/'summary.json')['selected'],'no_label_or_metric_changes':True}
    qa.save(OUT/'summary.json',summary);files+=['summary.json','started.json','harp_windows.npy']
    qa.save(OUT/'complete.json',{'status':'complete_development_only','snapshot':snapshot,'files_sha256':{n:qa.sha(OUT/n) for n in files},'test_opened':False})
    for m,e in selected.items():print(m,e['metrics']['calibration']['windows']['f1'],e['metrics']['calibration']['answers']['f1'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','fit']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':assert not (ROOT/'harp_development_protocol.json').exists();qa.save(ROOT/'harp_development_protocol.json',protocol());print('QA_HARP_PROTOCOL_FROZEN')
        else:run()
