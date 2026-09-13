"""Matched full-state versus PCA64 MiniCheck probes; frozen checker, CPU only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,pickle,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run_development as q

OLD=q.ROOT/'results/semantic_hidden_v1';OUT=q.ROOT/'results/semantic_full_v1'
METHODS=('minicheck_hidden64','minicheck_hidden1024')

def protocol():
    return {'version':'qa-minicheck-full-state-v1','methods':list(METHODS),'C':[.00001,.0001,.001,.01,.1],
      'hypothesis':'PCA retains most variance but can remove low-variance error information; compare full1024 to identical frozen PCA64 with matched five-C search.',
      'new_fits':7,'reused_fits':'Three existing PCA64 Cs .001/.01/.1 replayed, not refitted',
      'state':'Same frozen MiniCheck selected-document final hidden state, already character-aligned to original Llama raw tokens; punctuation retained, whitespace zero',
      'features':'Mean states over the same 4 actual raw BPE tokens for all210364 eligible windows; no labels used in construction',
      'fit':'Same168123 fit rows, frozen base/loss weights. Fit-only weighted StandardScaler in16384rowblocks, float32 transform.',
      'model':'liblinear L2, max_iter2000, seed20260924, four CPU threads',
      'selection':'Same calibration window and answer thresholds and q.selection_key, separate per family across five Cs',
      'answer_score':'Max over all eligible windows, including safe refusals',
      'scope':'Published human QA fit634/calibration159, selection optimistic, official test unopened',
      'extra_model':True,'generator_native_signal':False,'checker_finetuned':False,'new_GPU_inference':False,
      'script_sha256':q.sha(__file__),'base_code_sha256':q.sha(q.__file__),
      'source_complete_sha256':q.sha(OLD/'complete.json'),'source_preparation_sha256':q.sha(OLD/'preparation_complete.json')}

def run():
    cfg=q.read(OUT/'protocol.json');assert cfg==protocol();assert not (OUT/'started.json').exists()
    q.save(OUT/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':q.sha(OUT/'protocol.json')})
    start=time.perf_counter();meta=q.metadata();prep=q.read(OLD/'preparation_complete.json')
    for n,h in prep['files_sha256'].items():assert q.sha(OLD/n)==h
    source=np.load(OLD/'matrices/mapped_hidden1024.npy',mmap_mode='r')
    index=q.read(OLD/'token_index.json')['answers'];idx={a['response_id']:a for a in index}
    assert len(idx)==793 and source.shape==(213159,1024)
    full=np.lib.format.open_memmap(OUT/'window_hidden1024.npy',mode='w+',dtype=np.float32,shape=(210364,1024))
    for j,w in enumerate(meta['windows']):
        r=idx[w['response_id']];assert r['partition']==w['partition'] and r['group_id']==w['group_id']
        ix=np.asarray(w['token_indices'])+r['left'];assert ix[-1]<r['right']
        full[j]=source[ix].mean(0)
    full.flush();q.save(OUT/'matrix_manifest.json',{'file_sha256':q.sha(OUT/'window_hidden1024.npy'),
        'source_preparation_sha256':q.sha(OLD/'preparation_complete.json'),'rows':210364,'width':1024,'labels_used_to_construct_features':False})
    arrays={'minicheck_hidden64':np.load(OLD/'matrices/window_hidden64.npy',mmap_mode='r'),'minicheck_hidden1024':full}
    with np.load(q.OUT/'training_weights.npz',allow_pickle=False) as z:
        b=z['base_weights'].copy();loss=z['loss_weights'].copy();y=z['y'].copy()
    assert np.array_equal(y,[w['label'] for w in meta['windows'][:168123]])
    ci=[j for j,a in enumerate(meta['answers']) if a['partition']=='calibration'];lo,hi=meta['bounds']['calibration']
    selected={};all_entries={};files=[];replays={}
    for method in METHODS:
        raw=arrays[method]
        if method=='minicheck_hidden64':sc=pickle.loads((OLD/'minicheck_hidden64_C0.001.pkl').read_bytes())['scaler']
        else:
            sc=StandardScaler()
            for left in range(0,168123,q.BATCH):
                right=min(left+q.BATCH,168123);sc.partial_fit(raw[left:right],sample_weight=b[left:right])
        x=np.empty(raw.shape,np.float32)
        for left in range(0,len(x),q.BATCH):
            right=min(left+q.BATCH,len(x));x[left:right]=sc.transform(raw[left:right]).astype(np.float32)
        entries=[]
        for c in cfg['C']:
            begin=time.perf_counter();name=f'{method}_C{c:g}';reuse=method=='minicheck_hidden64' and c in [.001,.01,.1]
            if reuse:
                obj=pickle.loads((OLD/(name+'.pkl')).read_bytes());clf=obj['model']
                for k in ('mean_','scale_','var_'):assert np.array_equal(getattr(sc,k),getattr(obj['scaler'],k))
            else:
                clf=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=20260924)
                clf.fit(x[:168123],y,sample_weight=loss);assert clf.n_iter_.max()<2000
            s=clf.predict_proba(x)[:,1];aa=q.answer_scores(meta,s)
            ts={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],s[lo:hi]),
                'answer':q.choose_threshold([meta['answers'][j]['label'] for j in ci],aa[ci])}
            if reuse:
                with np.load(OLD/(name+'_scores.npz'),allow_pickle=False) as z:err=float(np.max(np.abs(s-z['window_scores'])))
                assert err<1e-12;replays[name]=err
                assert ts==obj['thresholds']
            obj={'model':clf,'scaler':sc,'method':method,'C':c,'thresholds':ts,'width':raw.shape[1],
                 'selection_key':q.selection_key(ts,c),'fit_only':True,'fit_rows':168123,'fit_groups':615,
                 'weights_sha256':q.sha(q.OUT/'training_weights.npz'),'protocol_sha256':q.sha(OUT/'protocol.json'),
                 'source':'replayed_frozen_model' if reuse else 'new_fit'}
            (OUT/(name+'.pkl')).write_bytes(pickle.dumps(obj,protocol=5));np.savez_compressed(OUT/(name+'_scores.npz'),window_scores=s,answer_scores=aa)
            e={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(q.selection_key(ts,c)),'metrics':q.metrics(meta,s,ts),
               'source':obj['source'],'seconds':time.perf_counter()-begin,'iterations':clf.n_iter_.tolist(),
               'model_sha256':q.sha(OUT/(name+'.pkl')),'scores_sha256':q.sha(OUT/(name+'_scores.npz'))}
            q.save(OUT/(name+'_result.json'),e);entries.append(e);files.extend([name+'.pkl',name+'_scores.npz',name+'_result.json'])
            print('MINICHECK_FULL_CANDIDATE',name,obj['source'],round(e['seconds'],1),flush=True)
        all_entries[method]=entries;selected[method]=max(entries,key=lambda e:e['selection_key']);del x
    assert cfg==protocol()
    q.save(OUT/'summary.json',{'selected':selected,'all_candidates':all_entries,'replays_max_abs':replays,
          'new_fits':7,'seconds':time.perf_counter()-start,'official_test_opened':False,'calibration_selection_optimistic':True})
    report=['冻结MiniCheck完整隐状态与PCA64对照；只报开发校准表现。','',
          '| 输入 | C | 4词元窗口F1 | 整答F1 |','|---|---:|---:|---:|']
    for name,e in selected.items():
        m=e['metrics']['calibration'];report.append(f"| {name} | {e['C']:g} | {m['windows']['f1']:.4f} | {m['answers']['f1']:.4f} |")
    report+=['','两族同样比较5档C，共7次新拟合、3次冻结模型回放。保留原人工标签、权重、划分和4词元尺度。',
             '这是额外核查模型的隐状态探针；原生成模型未提供新增信号，核查编码器也未微调。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8');files+=['protocol.json','started.json','matrix_manifest.json','summary.json','REPORT.md']
    q.save(OUT/'complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'files_sha256':{n:q.sha(OUT/n) for n in files},'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','run']);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='initialize':assert not (OUT/'protocol.json').exists();q.save(OUT/'protocol.json',protocol())
        else:run()
