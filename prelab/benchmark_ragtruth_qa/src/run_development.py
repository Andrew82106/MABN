"""Full-window RAGTruth QA fit/calibration development, never official test.

Four fixed feature families and three LR C values. Disk-backed matrices avoid
retaining copies of large designs. No GPU or model generation is performed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib, json, pickle, time, gc
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';OUT=ROOT/'results/development_v1'
PARTITIONS=('fit','calibration')
METHODS=('lookback_mean','lookback_nll','layerband_slots','hidden64_lookback_nll')
WIDTHS={'lookback_mean':1024,'lookback_nll':1025,'layerband_slots':520,'hidden64_lookback_nll':1089}
BATCH=16384


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.pending');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8');temp.replace(p)
def savel(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def protocol():
    return {'version':'ragtruth-qa-development-v1','scope':'Only official-train exported fit634/calibration159; test and withheld content never opened',
      'methods':list(METHODS),'widths':WIDTHS,'C':[.001,.01,.1],'seed':20260924,
      'model_replay':'Published llama-2-7b-chat answers, reconstructed fixed Llama2-7B NF4 forward pass; not original release token trajectory',
      'window':'All frozen eligible 4 raw BPE windows stride1; punctuation counts width; no gold-dependent feature selection',
      'fit_windows':168123,'calibration_windows':42241,'fit_answers':634,'calibration_answers':159,
      'safe_refusals':'No additional refusal classifier or exclusions; every quality-good answer and its eligible windows uses unchanged human labels',
      'answer_label':'Official nonempty original span list; never reconstructed from token/window OR',
      'answer_score':'Maximum over every eligible window for that answer; no gold-positive prefilter',
      'layerband_slots':'For each of four ordered raw tokens: mean over each contiguous 8-layer band, retain all32 heads (128 features), append NLL; append four padding mask bits =>520',
      'short_window':'Repeat last raw token to length4; mask true positions1 and padded positions0',
      'pca':{'components':64,'n_iter':3,'seed':20260924,'whiten':False,'fit_sample':'Each fit answer: floor(linspace(0,N-1,min(N,32))) raw BPE indices, no label use',
        'weight':'Each fit source-connected group equal, answer equal within group, selected token equal within answer; normalized sum1',
        'application':'Project ALL raw tokens with stored mean/components in float64 then castfloat32; window mean over four actual tokens; no training-window subsample'},
      'base_weight':'Source-connected group -> answer -> eligible window equal mass, normalized mean1',
      'loss_weight':'Fit-only class factors from base weights; re-equalize group loss; total mass168123 for every family/C',
      'lr':{'solver':'liblinear','penalty':'l2','max_iter':2000,'cpu_threads':4},
      'standardization':'StandardScaler fit-only weighted partial_fit in fixed16384-row blocks; float32 transform; same scaler reused across family Cs',
      'threshold':'Each candidate separately maximizes cal window or cal answer riskF1; ties precision then higher threshold; all/none endpoints included',
      'selection':'Max min(cal windowF1,cal answerF1), then windowF1, window precision, smaller C',
      'reporting':'All12 candidates fit/cal metrics plus four calibration-selected heads; no final/test performance claim',
      'no_calibration_refit':True,'missing_predictions':'Delivery failure; never remove scorable rows to improve counts',
      'resource_estimate':{'cpu_minutes_before_execution':[10,30],'peak_ram_gib_estimate':[5,9],'disk_backed_feature_bytes_approx':1350000000}}


def verify_gold():
    g=read(DATA/'gold_manifest.json');assert g['complete'] and g['official_test_opened'] is False and g['selfcheck_passed']
    assert digest(g['signature'])==g['signature_sha256']
    for key,path in g['source_paths'].items():assert sha(path)==g['signature']['files_sha256'][key]
    for rec in g['outputs']:assert sha(ROOT/rec['path'])==rec['sha256']
    assert g['counts']['fit']['eligible_windows']==168123 and g['counts']['calibration']['eligible_windows']==42241
    return g


def metadata():
    gold=verify_gold();answers=[];tokens=[];windows=[];bounds={};by_response={};cursor=0
    for part in PARTITIONS:
        aa=lines(DATA/f'answers_{part}.jsonl');tt=lines(DATA/f'tokens_{part}.jsonl');ww=lines(DATA/f'windows_k4_{part}.jsonl')
        assert len(aa)==len(tt)==gold['counts'][part]['answers'] and len(ww)==gold['counts'][part]['eligible_windows']
        assert [a['response_id'] for a in aa]==[t['response_id'] for t in tt]
        assert all(r['partition']==part for r in aa+tt+ww)
        for a,t in zip(aa,tt):
            assert a['eligible'] and a['quality']=='good' and a['label']==int(len(a['original_labels'])>0)
            assert t['answer_risk']==a['label'] and t['answer_sha256']==a['answer_sha256']
            by_response[a['response_id']]={'answer':a,'tokens':t}
        bounds[part]=[cursor,cursor+len(ww)];cursor+=len(ww);answers.extend(aa);tokens.extend(tt);windows.extend(ww)
    assert len({a['answer_id'] for a in answers})==793
    groups={part:{a['group_id'] for a in answers if a['partition']==part} for part in PARTITIONS}
    assert len(groups['fit'])==615 and len(groups['calibration'])==154 and not groups['fit']&groups['calibration']
    aw=defaultdict(list)
    for j,w in enumerate(windows):
        aw[w['response_id']].append(j);t=by_response[w['response_id']]['tokens'];ix=w['token_indices']
        assert w['eligible'] and len(ix)==min(4,t['token_count']) and ix==list(range(w['token_start'],w['token_end']))
        assert any(t['lexical_mask'][k] for k in ix) and w['label']==int(any(t['risk_mask'][k] for k in ix))
    assert all(aw[a['response_id']] for a in answers),'An answer without eligible windows must not disappear'
    return {'answers':answers,'tokens':tokens,'windows':windows,'bounds':bounds,'by_response':by_response,'answer_windows':dict(aw),'gold':gold}


def feature_snapshot(require_complete=True):
    fm=read(DATA/'feature_manifest.json')
    if require_complete:assert fm['complete'] and fm['completed_records']==fm['planned_records']==793
    sig=read(DATA/'feature_signature.json');assert digest(sig)==fm['signature_sha256']
    for key,name in [('runner_sha256','run_feature_qa_all.py'),('feature_code_sha256','feature_qa.py'),('loader_code_sha256','run_feature_qa.py')]:assert sha(ROOT/'src'/name)==sig[key]
    assert sig['test_or_withheld_content_read'] is False and fm['test_or_withheld_content_read'] is False
    assert sig['annotation_values_accessed'] is False and fm['annotation_values_accessed'] is False
    assert sha(DATA/'feature_preparation/plans.jsonl')==sig['plans_sha256']
    return fm,sig


def source_snapshot():
    verify_gold();fm,sig=feature_snapshot()
    paths=[Path(__file__),ROOT/'DEVELOPMENT_PROTOCOL.md',ROOT/'development_protocol.json',DATA/'gold_manifest.json',DATA/'feature_manifest.json',DATA/'feature_signature.json',DATA/'feature_preparation/plans.jsonl']
    return {'files_sha256':{str(p.resolve()):sha(p) for p in paths},'feature_signature_sha256':digest(sig),'official_test_opened':False}


def load_features(rid,meta,manifest,plans):
    records={r['response_id']:r for r in manifest['records']} if isinstance(manifest['records'],list) else manifest['records']
    rec=records[rid];path=DATA/'features'/(rid+'.npz');side=path.with_suffix('.json');s=read(side)
    assert s['complete'] and s['signature_sha256']==manifest['signature_sha256']
    assert sha(path)==rec['npz_sha256']==s['npz_sha256'] and sha(side)==rec['metadata_sha256']
    plan=plans[rid];assert digest(plan)==rec['plan_sha256']==s['plan_sha256']
    t=meta['by_response'][rid]['tokens'];assert plan['partition']==t['partition']
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in ('lb','nll','hidden_last','token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw','token_start','token_end')}
    n=t['token_count'];assert a['lb'].shape==(n,1024) and a['nll'].shape==(n,) and a['hidden_last'].shape==(n,4096)
    for k in ('lb','nll','hidden_last'):assert a[k].dtype==np.float32 and np.isfinite(a[k]).all()
    assert ((a['lb']>=0)&(a['lb']<=1)).all() and (a['nll']>=0).all()
    for k in ('token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw'):assert a[k].tolist()==t[k]
    assert np.array_equal(a['token_start'],a['response_token_offsets'][:,0]) and np.array_equal(a['token_end'],a['response_token_offsets'][:,1])
    return a


def sample_positions(n):return np.linspace(0,n-1,min(n,32),dtype=np.int64)


def fit_pca(meta,fm,plans):
    fit_answers=[a for a in meta['answers'] if a['partition']=='fit'];counts=defaultdict(int)
    for a in fit_answers:counts[a['group_id']]+=1
    sample=[];weights=[];identities=[]
    for answer in fit_answers:
        rid=answer['response_id'];data=load_features(rid,meta,fm,plans);ix=sample_positions(answer['token_count'])
        sample.append(data['hidden_last'][ix]);weights.extend([1/(counts[answer['group_id']]*len(ix))]*len(ix))
        identities.extend([{'response_id':rid,'group_id':answer['group_id'],'token_index':int(j)} for j in ix])
    raw=np.concatenate(sample).astype(np.float64);del sample;w=np.asarray(weights,np.float64);w/=w.sum();mean=w@raw
    xc=raw-mean
    _,s,components=randomized_svd(xc*np.sqrt(w[:,None]),n_components=64,n_iter=3,random_state=20260924,flip_sign=True)
    trace=float(np.einsum('ij,i,ij->',xc,w,xc));assert components.shape==(64,4096)
    return {'mean':mean,'components':components,'singular_values':s,'explained_variance_ratio_sum':float(np.sum(s*s)/trace),
      'sample':identities,'sample_weights':w,'sample_count':len(w),'fit_answers':634,'fit_groups':615,'seed':20260924,'n_iter':3,'whiten':False}


def build_matrices(meta):
    directory=OUT/'matrices';directory.mkdir(parents=True,exist_ok=True)
    fm,_=feature_snapshot();plans={p['response_id']:p for p in lines(DATA/'feature_preparation/plans.jsonl')}
    assert len(plans)==793 and {a['response_id'] for a in meta['answers']}==set(plans)
    pca=fit_pca(meta,fm,plans);(OUT/'hidden_pca.pkl').write_bytes(pickle.dumps(pca,protocol=5))
    n=len(meta['windows']);specs={'base':1025,'slots':520,'hidden':64}
    arrays={k:np.lib.format.open_memmap(directory/(k+'.npy'),mode='w+',dtype=np.float32,shape=(n,d)) for k,d in specs.items()}
    for i,answer in enumerate(meta['answers']):
        rid=answer['response_id'];a=load_features(rid,meta,fm,plans);ii=meta['answer_windows'][rid]
        projected=((a['hidden_last'].astype(np.float64)-pca['mean'])@pca['components'].T).astype(np.float32)
        bands=a['lb'].reshape(-1,4,8,32).mean(axis=2).reshape(-1,128)
        slots=np.column_stack((bands,a['nll']))
        for j in ii:
            w=meta['windows'][j];ix=w['token_indices'];count=len(ix)
            arrays['base'][j,:1024]=a['lb'][ix].mean(0);arrays['base'][j,1024]=a['nll'][ix].mean()
            arrays['hidden'][j]=projected[ix].mean(0)
            v=slots[ix]
            if count<4:v=np.concatenate((v,np.repeat(v[-1:],4-count,axis=0)))
            arrays['slots'][j]=np.concatenate((v.ravel(),np.asarray([1]*count+[0]*(4-count),np.float32)))
        if (i+1)%100==0:print('QA_DESIGN_ROWS',i+1,793,flush=True)
    for arr in arrays.values():arr.flush()
    manifest={'source_snapshot_sha256':sha(OUT/'source_snapshot.json'),'rows':n,'fit_rows':168123,'calibration_rows':42241,
      'files_sha256':{k:sha(directory/(k+'.npy')) for k in arrays},'pca_sha256':sha(OUT/'hidden_pca.pkl'),
      'pca_sample_count':pca['sample_count'],'training_windows_subsampled':False,'widths':specs}
    save(OUT/'matrix_manifest.json',manifest)
    return arrays


def base_weights(meta):
    rows=meta['windows'][:168123];tree=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):tree[r['group_id']][r['answer_id']].append(i)
    b=np.empty(len(rows),np.float64)
    for answers in tree.values():
        for ix in answers.values():b[ix]=1/(len(answers)*len(ix))
    b/=b.mean();y=np.asarray([r['label'] for r in rows],int);mass=np.bincount(y,weights=b,minlength=2)
    factors=mass.sum()/(2*mass);loss=b*factors[y]
    for answers in tree.values():
        ix=[j for entries in answers.values() for j in entries];loss[ix]*=(len(rows)/len(tree))/loss[ix].sum()
    loss*=168123/loss.sum()
    return b,loss,factors,y


def raw(arrays,method,left,right):
    if method=='lookback_mean':return np.asarray(arrays['base'][left:right,:1024])
    if method=='lookback_nll':return np.asarray(arrays['base'][left:right])
    if method=='layerband_slots':return np.asarray(arrays['slots'][left:right])
    return np.column_stack((arrays['base'][left:right],arrays['hidden'][left:right]))


def choose_threshold(y,s):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);assert set(y)=={0,1} and np.isfinite(s).all()
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];last=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    tp=np.r_[0,np.cumsum(yy)[last]];n=np.r_[0,last+1];threshold=np.r_[np.nextafter(ss[0],np.inf),ss[last]]
    f1=2*tp/(n+y.sum());precision=np.divide(tp,n,out=np.zeros(len(n),float),where=n>0)
    best=max(range(len(n)),key=lambda j:(f1[j],precision[j],threshold[j]))
    return {'threshold':float(threshold[best]),'f1':float(f1[best]),'precision':float(precision[best]),'rows':len(y),'positive':int(y.sum())}


def count(y,s,t):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);assert len(y)==len(s) and np.isfinite(s).all()
    p=s>=t;tp=int(np.count_nonzero((y==1)&p));fp=int(np.count_nonzero((y==0)&p));fn=int(np.count_nonzero((y==1)&~p));tn=int(np.count_nonzero((y==0)&~p))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
      'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'auroc':float(roc_auc_score(y,s)) if len(set(y))==2 else None,
      'average_precision':float(average_precision_score(y,s)) if y.sum() else None}


def answer_scores(meta,s):return np.asarray([max(s[meta['answer_windows'][a['response_id']]]) for a in meta['answers']],np.float64)


def selection_key(ts,c):return (min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c)


def metrics(meta,s,thresholds):
    aa=answer_scores(meta,s);result={}
    for part in PARTITIONS:
        left,right=meta['bounds'][part];ix=[j for j,a in enumerate(meta['answers']) if a['partition']==part]
        result[part]={'windows':count([w['label'] for w in meta['windows'][left:right]],s[left:right],thresholds['window']['threshold']),
          'answers':count([meta['answers'][j]['label'] for j in ix],aa[ix],thresholds['answer']['threshold'])}
    return result


def synthetic_tests():
    y=np.asarray([0,1,1,0,1]);s=np.asarray([.2,.4,.4,.7,.9]);best=choose_threshold(y,s)
    options=[np.nextafter(s.max(),np.inf),*np.unique(s)];keys=[]
    for t in options:
        m=count(y,s,t);keys.append((m['f1'],m['precision'],t))
    assert (best['f1'],best['precision'],best['threshold'])==max(keys)
    assert sample_positions(7).tolist()==list(range(7)) and len(sample_positions(300))==32 and sample_positions(300)[-1]==299
    # A non-lexical BPE still consumes the second raw slot; no hidden compression.
    x=np.arange(4*32*32,dtype=np.float32).reshape(4,32,32)
    b=x.reshape(4,4,8,32).mean(axis=2)
    assert np.array_equal(b[:,0],x[:,:8].mean(axis=1)) and np.array_equal(b[:,3],x[:,24:].mean(axis=1))
    print('QA_DEVELOPMENT_SYNTHETIC_PASSED_NO_FIT',flush=True)


def fit():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'started.json').exists(),'Do not silently restart formal development'
    cfg=read(ROOT/'development_protocol.json');assert cfg==protocol();synthetic_tests()
    meta=metadata();snap=source_snapshot();save(OUT/'source_snapshot.json',snap)
    save(OUT/'started.json',{'source_snapshot_sha256':sha(OUT/'source_snapshot.json'),'official_test_opened':False})
    start=time.perf_counter();arrays=build_matrices(meta);b,loss,factors,y=base_weights(meta)
    np.savez_compressed(OUT/'training_weights.npz',base_weights=b,loss_weights=loss,class_factors=factors,y=y)
    save(OUT/'fit_keys.json',{'window_ids':[r['window_id'] for r in meta['windows'][:168123]],'group_ids':[r['group_id'] for r in meta['windows'][:168123]],
      'answer_ids':[r['answer_id'] for r in meta['windows'][:168123]]})
    selected={};families={};n=len(meta['windows']);candidate_names=[]
    for method in METHODS:
        sc=StandardScaler()
        for left in range(0,168123,BATCH):
            right=min(left+BATCH,168123);sc.partial_fit(raw(arrays,method,left,right),sample_weight=b[left:right])
        zpath=OUT/'matrices'/(method+'_fit_standardized.npy')
        zfit=np.lib.format.open_memmap(zpath,mode='w+',dtype=np.float32,shape=(168123,WIDTHS[method]))
        for left in range(0,168123,BATCH):
            right=min(left+BATCH,168123);zfit[left:right]=sc.transform(raw(arrays,method,left,right)).astype(np.float32)
        zfit.flush();entries=[]
        for c in cfg['C']:
            begin=time.perf_counter();classifier=LogisticRegression(C=c,solver='liblinear',penalty='l2',max_iter=2000,random_state=cfg['seed'])
            classifier.fit(zfit,y,sample_weight=loss);assert classifier.n_iter_.max()<2000
            scores=np.empty(n,np.float64)
            for left in range(0,n,BATCH):
                right=min(left+BATCH,n);zz=sc.transform(raw(arrays,method,left,right)).astype(np.float32);scores[left:right]=classifier.predict_proba(zz)[:,1]
            assert np.isfinite(scores).all();answer=answer_scores(meta,scores)
            ci=[j for j,a in enumerate(meta['answers']) if a['partition']=='calibration'];lo,hi=meta['bounds']['calibration']
            ts={'window':choose_threshold([w['label'] for w in meta['windows'][lo:hi]],scores[lo:hi]),
              'answer':choose_threshold([meta['answers'][j]['label'] for j in ci],answer[ci])}
            key=selection_key(ts,c);name=f'{method}_C{c:g}';objects={'model':classifier,'scaler':sc,'C':c,'method':method,'width':WIDTHS[method],
              'thresholds':ts,'selection_key':key,'fit_only':True,'fit_rows':168123,'fit_groups':615,'fit_answers':634,
              'weights_sha256':sha(OUT/'training_weights.npz'),'fit_keys_sha256':sha(OUT/'fit_keys.json'),'pca_sha256':sha(OUT/'hidden_pca.pkl') if method=='hidden64_lookback_nll' else None}
            path=OUT/(name+'.pkl');path.write_bytes(pickle.dumps(objects,protocol=5))
            scorepath=OUT/(name+'_scores.npz');np.savez_compressed(scorepath,window_scores=scores,answer_scores=answer)
            entry={'candidate':name,'C':c,'thresholds':ts,'selection_key':list(key),'metrics':metrics(meta,scores,ts),
              'fit_and_score_seconds':time.perf_counter()-begin,'iterations':classifier.n_iter_.tolist(),'model_sha256':sha(path),'scores_sha256':sha(scorepath)}
            entries.append(entry);candidate_names.extend((path.name,scorepath.name));save(OUT/(name+'_result.json'),entry);candidate_names.append(name+'_result.json')
            print('QA_CANDIDATE_FINISHED',name,'seconds',round(entry['fit_and_score_seconds'],1),flush=True)
        j=max(range(len(entries)),key=lambda i:entries[i]['selection_key']);selected[method]=entries[j];families[method]=entries
        del zfit;gc.collect()
    assert snap==source_snapshot()
    summary={'scope':cfg['scope'],'calibration_results_are_selection_optimistic':True,'test_opened':False,'fits':12,
      'coverage':meta['gold']['counts'],'all_candidates':families,'selected':selected,'elapsed_seconds':time.perf_counter()-start,
      'risk_definition':'Faithfulness to released passages, unchanged human spans, implicit_true/due_to_null retained',
      'difference_from_r16':'All quality-good refusal text windows evaluated; long released responses; Llama NF4 reconstructed states'}
    save(OUT/'summary.json',summary)
    # Every candidate and all labels/coordinates remain addressable for later audit.
    save(OUT/'score_index.json',{'windows':[{'window_id':w['window_id'],'answer_id':w['answer_id'],'group_id':w['group_id'],'partition':w['partition'],'label':w['label']} for w in meta['windows']],
      'answers':[{'answer_id':a['answer_id'],'group_id':a['group_id'],'partition':a['partition'],'label':a['label']} for a in meta['answers']]})
    report=['四组、12个候选均使用全部人工QA开发窗口训练。以下仅为用于选参的校准结果，不能当成最终测试成绩。','',
      '| 方法 | 所选C | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|']
    for name,e in selected.items():
        m=e['metrics'];report.append(f"| {name} | {e['C']} | {m['fit']['windows']['f1']:.3f} | {m['calibration']['windows']['f1']:.3f} | {m['fit']['answers']['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    report+=['','634 fit回答/168123窗口，159 calibration回答/42241窗口；人工good答案及拒答文字均保留。测试未打开，未在fit+cal上重拟合。',
      '顺序方法是4个连续层带的适配，不是R18原全头顺序特征的完全重现。PCA只用每个fit回答最多32个确定性raw token拟合，但投影及探针训练覆盖全部窗口。',
      'Llama2 NF4特征是对公开原文的固定teacher-forced重建，不是公开数据最初生成时记录的内部状态。全部12个候选及精确预测已保存。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8')
    names=candidate_names+['source_snapshot.json','training_weights.npz','fit_keys.json','hidden_pca.pkl','matrix_manifest.json','summary.json','score_index.json','REPORT.md']
    save(OUT/'complete.json',{'status':'complete_development_only','files_sha256':{name:sha(OUT/name) for name in names},
      'source_snapshot_sha256':sha(OUT/'source_snapshot.json'),'script_sha256':sha(__file__),'official_test_opened':False,'final_test_claim':False})
    print('QA_DEVELOPMENT_COMPLETE',flush=True)


def wait_and_fit():
    # Short polls keep the orchestration interruptible; no original test paths.
    while True:
        path=DATA/'feature_manifest.json'
        if path.exists():
            m=read(path)
            if m.get('complete'):break
            print('QA_WAIT_FEATURES',m.get('completed_records'),m.get('planned_records'),flush=True)
        else:print('QA_WAIT_FEATURE_MANIFEST',flush=True)
        time.sleep(25)
    fit()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','self-test','prepare','fit','wait-and-fit']);args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='initialize':
            path=ROOT/'development_protocol.json';assert not path.exists();save(path,protocol());print('QA_DEVELOPMENT_PROTOCOL_FROZEN')
        elif args.stage=='self-test':synthetic_tests()
        elif args.stage=='prepare':
            m=metadata();print({k:v for k,v in m['gold']['counts'].items()});feature_snapshot();print('QA_DEVELOPMENT_INPUTS_READY')
        elif args.stage=='fit':fit()
        else:wait_and_fit()
