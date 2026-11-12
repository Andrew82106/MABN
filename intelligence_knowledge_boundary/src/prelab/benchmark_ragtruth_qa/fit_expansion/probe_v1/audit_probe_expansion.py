"""Independent read-only CPU audit; no model/scaler/PCA fitting or test access."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import hashlib, json, pickle, sys, time
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent; EXP=OUT.parent; QA=EXP.parent
sys.path.insert(0,str(QA/'src'))
import run_development as q
OLD=QA/'results/semantic_hidden_v1'; FULL=QA/'results/semantic_full_v1'
BATCH=16384; CS=(1e-5,1e-4,.001); hashes={}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    hashes[str(p.resolve())]=h.hexdigest();return h.hexdigest()
def near(a,b,tol=1e-10):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape,(a.shape,b.shape)
    d=float(np.max(np.abs(a-b),initial=0));assert d<=tol,d;return d
def choose(y,s):
    y=np.asarray(y,int);s=np.asarray(s,np.float64)
    order=np.argsort(s,kind='stable');ss=s[order];yy=y[order]
    vals,first=np.unique(ss,return_index=True)
    before=np.r_[0,np.cumsum(yy)[:-1]][first]
    tp=np.r_[int(y.sum())-before,0];n=np.r_[len(y)-first,0]
    thresholds=np.r_[vals,np.nextafter(vals[-1],np.inf)]
    f=2*tp/(n+int(y.sum()));p=np.divide(tp,n,out=np.zeros(len(n)),where=n!=0)
    best=max(range(len(n)),key=lambda i:(f[i],p[i],thresholds[i]))
    return dict(threshold=float(thresholds[best]),f1=float(f[best]),precision=float(p[best]),rows=len(y),positive=int(y.sum()))
def count(y,s,t):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);pred=s>=t
    tp=int(y[pred].sum());fp=int(pred.sum())-tp;fn=int(y.sum())-tp;tn=len(y)-tp-fp-fn
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        auroc=float(roc_auc_score(y,s)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,s)) if y.sum() else None)
def compare_dict(a,b):
    assert a.keys()==b.keys();err=0.
    for k,v in a.items():
        if isinstance(v,dict):err=max(err,compare_dict(v,b[k]))
        elif isinstance(v,float):err=max(err,near(v,b[k],1e-12))
        else:assert v==b[k],(k,v,b[k])
    return err
def moments(raw,base,nfit):
    count_=0.;mean=np.zeros(raw.shape[1]);var=mean.copy()
    for l in range(0,nfit,BATCH):
        r=min(l+BATCH,nfit);x=raw[l:r].astype(np.float64)
        w=base[l:r].astype(np.float32).astype(np.float64);mass=w.sum();center=w@x/mass
        x-=center;correction=w@x;x*=x;m2=w@x-correction*correction/mass
        count_=float(np.float32(count_));total=count_+mass
        merged=(count_*mean+mass*center)/total
        var=(count_*var+m2+(center-mean)**2*count_*mass/total)/total
        mean,count_=merged,total
    eps=np.finfo(float).eps;constant=var<=count_*eps*var+(count_*mean*eps)**2
    scale=np.sqrt(var);scale[constant]=1
    return mean,var,scale,count_
def key(ts,c):return [min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c]

def main():
    start=time.perf_counter();sha(__file__)
    done=read(OUT/'complete.json');prep=read(OUT/'preparation_complete.json');cfg=read(OUT/'protocol.json')
    summary=read(OUT/'summary.json');assert not done['official_test_opened']
    for name,h in done['files_sha256'].items():assert sha(OUT/name)==h,name
    for name,h in prep['files_sha256'].items():assert sha(OUT/name)==h,name
    for path,h in prep['source_snapshot']['files_sha256'].items():assert sha(path)==h,path
    export=read(EXP/'data/export_freeze.json')
    for section in ('source_files_sha256','output_files_sha256'):
        for path,h in export[section].items():assert sha(path)==h,path
    assert cfg['script_sha256']==sha(EXP/'run_probe_expansion.py')
    assert cfg['mapping_code_sha256']==sha(QA/'src/run_semantic_hidden.py')
    assert cfg['base_code_sha256']==sha(QA/'src/run_development.py')
    assert cfg['old_pca_sha256']==sha(OLD/'hidden_pca.pkl')
    assert read(OUT/'fit_started.json')['preparation_sha256']==sha(OUT/'preparation_complete.json')
    assert read(OUT/'preparation_started.json')['source_snapshot']==prep['source_snapshot']
    assert datetime.fromisoformat(read(OUT/'preparation_started.json')['utc'])<datetime.fromisoformat(read(OUT/'fit_started.json')['utc'])<datetime.fromisoformat(done['utc'])

    original=q.metadata();aa=lines(EXP/'data/answers_fit.jsonl');tt=lines(EXP/'data/tokens_fit.jsonl');ww=lines(EXP/'data/windows_k4_fit.jsonl')
    assert len(aa)==len(tt)==3680 and len(ww)==653979
    assert aa[:634]==original['answers'][:634] and tt[:634]==original['tokens'][:634] and ww[:168123]==original['windows'][:168123]
    aa+=original['answers'][634:];tt+=original['tokens'][634:];ww+=original['windows'][168123:]
    assert len(aa)==3839 and len(ww)==696220
    fitgroups={a['group_id'] for a in aa[:3680]};calgroups={a['group_id'] for a in aa[3680:]}
    assert len(fitgroups)==615 and len(calgroups)==154 and not fitgroups&calgroups
    assert fitgroups=={a['group_id'] for a in original['answers'][:634]}
    assert len({a['response_id'] for a in aa})==3839
    aw=defaultdict(list)
    for j,w in enumerate(ww):aw[w['response_id']].append(j)
    index=read(OUT/'token_index.json')['answers'];assert len(index)==3839
    th=np.load(OUT/'matrices/token_hidden64.npy',mmap_mode='r');tr=np.load(OUT/'matrices/token_risk.npy',mmap_mode='r')
    raw=np.load(OUT/'matrices/window65.npy',mmap_mode='r');assert th.shape==(708506,64) and tr.shape==(708506,) and raw.shape==(696220,65)
    oh=np.load(OLD/'matrices/token_hidden64.npy',mmap_mode='r');orr=np.load(OLD/'matrices/token_risk.npy',mmap_mode='r')
    oldindex={a['response_id']:a for a in read(OLD/'token_index.json')['answers']}
    old_tokens=0;checkedwindows=0;cursor=0
    for a,t,ix in zip(aa,tt,index):
        rid=a['response_id'];assert ix['response_id']==rid and ix['partition']==a['partition'] and ix['group_id']==a['group_id']
        assert ix['left']==cursor and ix['right']-cursor==t['token_count'];cursor=ix['right']
        if rid in oldindex:
            old=oldindex[rid];assert np.array_equal(th[ix['left']:ix['right']],oh[old['left']:old['right']])
            assert np.array_equal(tr[ix['left']:ix['right']],orr[old['left']:old['right']]);old_tokens+=t['token_count']
        wi=np.asarray(aw[rid],int);tokenindices=[ww[j]['token_indices'] for j in wi]
        assert len({len(v) for v in tokenindices})==1
        ii=np.asarray(tokenindices,int);lex=np.asarray(t['lexical_mask'],bool)[ii];assert lex.any(1).all()
        assert np.all((ii>=0)&(ii<t['token_count']))
        assert np.array_equal(th[ix['left']+ii].mean(1),raw[wi,:64])
        risks=tr[ix['left']+ii].copy();risks[~lex]=-np.inf;p=np.clip(risks.max(1),1e-6,1-1e-6)
        assert np.array_equal(np.log(p/(1-p)).astype(np.float32),raw[wi,64]);checkedwindows+=len(wi)
    assert cursor==708506 and old_tokens==213159 and checkedwindows==696220
    oldix=np.r_[np.arange(168123),np.arange(653979,696220)]
    assert np.array_equal(raw[oldix,:64],np.load(OLD/'matrices/window_hidden64.npy',mmap_mode='r'))
    assert np.array_equal(raw[oldix,64],np.load(OLD/'matrices/window_risk_logit.npy',mmap_mode='r'))
    print('EXPANSION_AUDIT_GEOMETRY_OLD793_ALL_WINDOWS_PASS',flush=True)

    pca=pickle.loads((OLD/'hidden_pca.pkl').read_bytes())
    native={a['response_id'] for a in aa[:634]};assert {s['response_id'] for s in pca['sample']}==native
    assert not {s['group_id'] for s in pca['sample']}&calgroups and pca['fit_answers']==634
    weightreports={};regimedata={}
    for regime,nfit,nans in [('original634',168123,634),('expanded3680',653979,3680)]:
        arows=original['answers'] if regime=='original634' else aa
        wrows=original['windows'] if regime=='original634' else ww
        x=raw[oldix] if regime=='original634' else raw
        with np.load(OUT/(regime+'_weights.npz'),allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
        b,loss,y,f=[weights[k] for k in ('base','loss','y','class_factors')]
        assert np.array_equal(y,[w['label'] for w in wrows[:nfit]]) and (b>0).all() and (loss>0).all()
        assert abs(b.sum()-168123)<1e-6 and abs(loss.sum()-168123)<1e-6
        if regime=='original634':
            with np.load(QA/'results/development_v1/training_weights.npz') as z:
                for k,oldkey in [('base','base_weights'),('loss','loss_weights'),('y','y'),('class_factors','class_factors')]:assert np.array_equal(weights[k],z[oldkey])
        grouped=defaultdict(list);byanswer=Counter(w['response_id'] for w in wrows[:nfit]);strata=defaultdict(set);astrata=defaultdict(set)
        for j,w in enumerate(wrows[:nfit]):
            rid=w['response_id'];g=w['group_id'];st=int(rid in native) if regime=='expanded3680' else 0
            grouped[g].append(j);strata[g].add(st);astrata[g,st].add(rid)
        expected=np.asarray([168123/(615*len(strata[w['group_id']])*len(astrata[w['group_id'],int(w['response_id'] in native) if regime=='expanded3680' else 0])*byanswer[w['response_id']]) for w in wrows[:nfit]])
        baseerr=near(expected,b,1e-9)
        mass=np.asarray([b[y==c].sum() for c in (0,1)]);factorerr=near(mass.sum()/(2*mass),f,1e-10)
        wantloss=b*f[y];groupbase=0.;grouploss=0.
        for ix in grouped.values():
            wantloss[ix]*=(168123/615)/wantloss[ix].sum()
            groupbase=max(groupbase,abs(b[ix].sum()-168123/615));grouploss=max(grouploss,abs(loss[ix].sum()-168123/615))
        if regime=='original634':wantloss*=168123/wantloss.sum()
        losserr=near(wantloss,loss,1e-9)
        nativefraction=float(sum(b[j] for j,w in enumerate(wrows[:nfit]) if w['response_id'] in native)/b.sum())
        if regime=='expanded3680':assert abs(nativefraction-.5)<1e-10 and all(len(v)==2 for v in strata.values())
        weightreports[regime]=dict(fit_answers=nans,fit_windows=nfit,positive_windows=int(y.sum()),source_groups=len(grouped),base_mass=float(b.sum()),loss_mass=float(loss.sum()),native_base_fraction=nativefraction,base_max_abs=baseerr,loss_max_abs=losserr,class_factor_max_abs=factorerr,group_base_max_abs=groupbase,group_loss_max_abs=grouploss)
        wa=defaultdict(list)
        for j,w in enumerate(wrows):wa[w['response_id']].append(j)
        regimedata[regime]=(arows,wrows,x,weights,wa,nfit,nans)
    print('EXPANSION_AUDIT_WEIGHT_MASS_PASS',flush=True)

    candidates=[];scalers={};selected={};maxscore=maxmetric=0.;scorevalues=answerchecks=0;oldchecks=[]
    for family,entries in summary['all_candidates'].items():
        regime,method=family.split('_',1);arows,wrows,x,weights,wa,nfit,nans=regimedata[regime]
        width=64 if method=='hidden64' else 65;xr=x[:,:width]
        objs=[pickle.loads((OUT/f'{family}_C{c:g}.pkl').read_bytes()) for c in CS];sc=objs[0]['scaler']
        if family=='original634_hidden64':
            oldsc=pickle.loads((FULL/'minicheck_hidden64_C1e-05.pkl').read_bytes())['scaler']
            for attr in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(sc,attr),getattr(oldsc,attr))
            scalers[family]={'frozen_original_exact':True}
        else:
            mean,var,scale,seen=moments(xr,weights['base'],nfit)
            scalers[family]={'mean_max_abs':near(mean,sc.mean_,2e-11),'var_max_abs':near(var,sc.var_,2e-10),'scale_max_abs':near(scale,sc.scale_,2e-11),'count_max_abs':near(seen,sc.n_samples_seen_,1e-9),'fit_rows':nfit,'calibration_rows_used':0}
        for obj,c in zip(objs,CS):
            assert obj['regime']==regime and obj['method']==method and obj['C']==c and obj['fit_only'] and obj['fit_rows']==nfit
            assert obj['weight_sha256']==sha(OUT/(regime+'_weights.npz'))
            for attr in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(sc,attr),getattr(obj['scaler'],attr))
            clf=obj['model'];assert clf.C==c and clf.solver=='liblinear' and clf.penalty=='l2' and clf.max_iter==2000 and clf.random_state==20260924
            assert max(clf.n_iter_)<2000 and clf.coef_.shape==(1,width) and np.array_equal(clf.classes_,[0,1])
        replay=np.empty((3,len(x)),np.float64)
        for l in range(0,len(x),BATCH):
            r=min(l+BATCH,len(x));z=np.array(xr[l:r],np.float32,copy=True);z-=sc.mean_;z/=sc.scale_
            for k,obj in enumerate(objs):replay[k,l:r]=expit((z@obj['model'].coef_.T+obj['model'].intercept_).ravel())
        ys=np.asarray([w['label'] for w in wrows]);ya=np.asarray([a['label'] for a in arows])
        for k,(c,obj,entry) in enumerate(zip(CS,objs,entries)):
            name=f'{family}_C{c:g}';assert entry==read(OUT/(name+'_result.json')) and entry['C']==c
            assert entry['model_sha256']==sha(OUT/(name+'.pkl')) and entry['scores_sha256']==sha(OUT/(name+'_scores.npz'))
            with np.load(OUT/(name+'_scores.npz')) as z:s=z['window_scores'];ans=z['answer_scores']
            scoreerr=near(replay[k],s,1e-12);maxscore=max(maxscore,scoreerr);scorevalues+=len(s)
            expectedans=np.asarray([np.max(s[wa[a['response_id']]]) for a in arows]);assert np.array_equal(ans,expectedans);answerchecks+=len(ans)
            ts={'window':choose(ys[nfit:],s[nfit:]),'answer':choose(ya[nans:],ans[nans:])}
            assert ts==entry['thresholds']==obj['thresholds'];assert key(ts,c)==entry['selection_key']
            mm={}
            for part,(lo,hi,al,ar) in {'fit':(0,nfit,0,nans),'calibration':(nfit,len(s),nans,len(ans))}.items():
                mm[part]={'windows':count(ys[lo:hi],s[lo:hi],ts['window']['threshold']),'answers':count(ya[al:ar],ans[al:ar],ts['answer']['threshold'])}
            metricerr=compare_dict(mm,entry['metrics']);maxmetric=max(maxmetric,metricerr)
            if family=='original634_hidden64':
                oldobj=pickle.loads((FULL/f'minicheck_hidden64_C{c:g}.pkl').read_bytes());oldentry=read(FULL/f'minicheck_hidden64_C{c:g}_result.json')
                for attr in ('coef_','intercept_','classes_','n_iter_'):assert np.array_equal(getattr(obj['model'],attr),getattr(oldobj['model'],attr))
                with np.load(FULL/f'minicheck_hidden64_C{c:g}_scores.npz') as z:od=near(s,z['window_scores'],1e-12);ad=near(ans,z['answer_scores'],1e-12)
                assert ts==oldentry['thresholds'];compare_dict(mm,oldentry['metrics'])
                oldchecks.append({'C':c,'coefficients_scaler_exact':True,'window_scores_max_abs':od,'answer_scores_max_abs':ad,'thresholds_exact':True})
            candidates.append({'candidate':name,'window_count':len(s),'answer_count':len(ans),'score_max_abs':scoreerr,'thresholds_exact':True,'metrics_max_abs':metricerr,'reused':entry['reused']})
        chosen=max(entries,key=lambda e:key(e['thresholds'],e['C']));assert chosen==summary['selected'][family]
        selected[family]={'C':chosen['C'],'calibration':chosen['metrics']['calibration']}
        print('EXPANSION_AUDIT_FAMILY_PASS',family,flush=True)
    assert sum(not e['reused'] for e in candidates)==9 and sum(e['reused'] for e in candidates)==3
    report={'status':'passed','scope':'Read-only CPU audit; no model/scaler/PCA fitting, GPU use or official-test answers/labels.',
        'counts':{'fit_answers_original':634,'fit_answers_expanded':3680,'aux_answers':3046,'fit_source_groups':615,'calibration_answers':159,'calibration_groups':154,'fit_windows_original':168123,'fit_windows_expanded':653979,'calibration_windows':42241,'raw_tokens_original_exact':old_tokens,'all_raw_tokens':708506,'all_windows_aggregation_exact':checkedwindows},
        'old793_token_and_window_features_exact':True,'same_fit_and_calibration_gold_geometry':True,'fit_calibration_groups_disjoint':True,'original_PCA634_frozen_no_refit':True,
        'weights':weightreports,'scalers':scalers,'candidate_checks':candidates,'old_replays':oldchecks,'coefficient_values_checked':scorevalues,'answer_max_checks':answerchecks,'coefficient_max_abs':maxscore,'metric_max_abs':maxmetric,'thresholds_exact_count':24,'selected_C_exact_count':4,'selected':selected,
        'limitations':['The3046 extra published responses reuse615 existing source-connected fit groups; no new independent questions are added.','MiniCheck is an additional frozen checker; these hidden states are not white-box states of each original answer generator.','Only already-open development/calibration results are examined. Repeated calibration selection is optimistic.','This helper checks every final window and old793 token reuse; a separate mapping audit covers additional3046 claim-cache identity/alignment/projection.','Recorded timestamps/hash chain establish internal consistency, not externally attested historical execution.'],
        'files_sha256':hashes,'seconds':time.perf_counter()-start}
    dest=OUT/'INDEPENDENT_NUMERIC_AUDIT.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['扩充训练独立数值核查通过；未重训、使用GPU或读取测试集。','',
        '原793份回答的全部213159词元特征及原窗口逐值相同；扩充后全部696220窗口的聚合也逐值相同。',
        '原634与扩充3680份回答共用615个训练来源组，159份校准回答来自另外154组。新增的是回答，未增加独立问题。',
        '两种规模的损失总质量均为168123；扩充基础权重在每组内原生/辅助回答各占一半。新增9个拟合和3个旧模型回放均通过，24个阈值和4个C选择一致。','',
        '| 训练规模与输入 | C | 校准窗口F1 | 校准整答F1 |','|---|---:|---:|---:|']
    for fam,v in selected.items():md.append(f"| {fam} | {v['C']:g} | {v['calibration']['windows']['f1']:.6f} | {v['calibration']['answers']['f1']:.6f} |")
    md+=['','这些是额外MiniCheck模型的状态探针，且仍是反复使用的开发校准成绩。扩充数据没有改善当前窗口定位表现。']
    (OUT/'INDEPENDENT_NUMERIC_AUDIT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('EXPANSION_NUMERIC_AUDIT_PASS',sha(dest),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
