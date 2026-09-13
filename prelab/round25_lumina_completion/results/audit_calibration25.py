"""Independent R25 frozen baseline/OOF/calibration audit. No fit/GPU/heldout.

The parent separately audits raw-token formulas and source extraction. Here we
check frozen formula arrays, cal/freeze/outer bindings, legacy score replay and
OOF score/decision/count delivery. Production modules are never imported.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,pickle,traceback
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;PRELAB=ROOT.parent
R21=PRELAB/'round21_semantic_internal_probe/results'
R22=PRELAB/'round22_evidence_verification/results'
R24=PRELAB/'round24_sequence_persistence/results'
REPORT=OUT/'CALIBRATION_AUDIT25.json'
OLD=('base','slots_base','r19_all','base_harp_delta','lookback_tuned','redeep_tuned','slots_base_smooth')
FIXED=('lumina_single_mean_lambda05','lumina_joint_body_lambda05')
CAL='lumina_joint_body_lambda_cal';METHODS=OLD+FIXED+(CAL,)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
class Frozen:
    def __setstate__(self,s):self.__dict__.update(s)
class Reader(pickle.Unpickler):
    def find_class(self,m,n):
        if m.startswith('sklearn.'):return Frozen
        if m.startswith('numpy') or m in ('builtins','collections'):return super().find_class(m,n)
        raise ValueError((m,n))
def unpickle(p):
    with Path(p).open('rb') as f:return Reader(f).load()
def near(a,b,tag,atol=3e-12):
    a,b=np.asarray(a),np.asarray(b);error=float(np.max(np.abs(a.astype(float)-b.astype(float))))
    assert a.shape==b.shape and np.allclose(a,b,rtol=0,atol=atol),(tag,error)
    return error
def sigmoid(z):
    r=np.empty(z.shape,np.float64);keep=z>=0;r[keep]=1/(1+np.exp(-z[keep]));e=np.exp(z[~keep]);r[~keep]=e/(1+e);return r
def lr(obj,x):
    z=x.copy();z-=obj['scaler'].mean_;z/=obj['scaler'].scale_
    return sigmoid((z.astype(np.float32)@obj['model'].coef_.T+obj['model'].intercept_).ravel())


def binding(report):
    complete=read(OUT/'complete25.json');frozen=read(OUT/'calibration_freeze25.json');start=read(OUT/'calibration_started25.json');test=read(OUT/'test_started25.json')
    checks={}
    for m in (complete,frozen):
        for n,h in m['files_sha256'].items():
            p=(OUT/n).resolve();assert p.is_relative_to(OUT) and sha(p)==h;checks[n]=h
    assert test['calibration_freeze_sha256']==sha(OUT/'calibration_freeze25.json') and test['retuning_allowed'] is False
    assert start['source_snapshot_sha256']==sha(OUT/'source_snapshot25.json')
    assert all(f'fold_{f}_{name}25.{ext}' in frozen['files_sha256'] for f in range(5) for name,ext in [('calibration','json'),('scores','npz')])
    times=[datetime.fromisoformat(m['utc']) for m in (start,frozen,test,complete)]
    assert times==sorted(times) and times[1]<times[2]
    assert frozen['all_calibration_choices_frozen_before_outer'] and complete['new_models_fitted']==frozen['new_models_fitted']==0
    cfg=read(ROOT/'scoring_protocol.json');source=read(OUT/'source_snapshot25.json')['files_sha256']
    for p in (ROOT/'src/run25.py',ROOT/'scoring_protocol.json'):
        assert source[str(p.resolve())]==sha(p)
    assert cfg['fixed_lambda']==.5 and cfg['secondary_lambda_grid']==[.25,.5,.75] and cfg['methods']==list(METHODS)
    assert cfg['new_lr_fits']==cfg['new_pca_fits']==0 and cfg['original_validation_test_used'] is False
    report['freeze_binding']={'all5_fold_tables_and_arrays_in_freeze':True,'freeze_hash_bound_before_outer':True,
        'recorded_timestamps_ordered':True,'calibration_started_utc':start['utc'],'calibration_frozen_utc':frozen['utc'],
        'outer_started_utc':test['utc'],'outer_completed_utc':complete['utc'],'files_sha256':checks,
        'limit':'Code and artifact links plus recorded timestamps support order; no claim about unrecorded external execution.'}
    return cfg,source


def smooth_logodds(s,seqs,stay,temp):
    if stay==.5 and temp==1:return s.copy()
    clipped=np.clip(s,1e-12,1-1e-12);emission=(np.log(clipped)-np.log1p(-clipped))/temp
    ls,ld=np.log(stay),np.log1p(-stay)
    def transition(x):return np.logaddexp(ls+x,ld)-np.logaddexp(ld+x,ls)
    output=np.empty(len(s))
    for ix in seqs:
        e=emission[ix];f=np.empty(len(ix));b=np.zeros(len(ix));f[0]=e[0]
        for i in range(1,len(ix)):f[i]=e[i]+transition(f[i-1])
        for i in range(len(ix)-2,-1,-1):b[i]=transition(e[i+1]+b[i+1])
        output[ix]=sigmoid(f+b)
    return output


def baseline(fold,d,windows,source):
    prior=unpickle(R21/f'fold_{fold}_frozen.pkl');r22=unpickle(R22/f'fold_{fold}_frozen22.pkl')
    v={name:lr(prior['models'][name],d[key]) for name,key in [('base','base'),('slots_base','slots'),('r19_all','r19_all')]}
    p=prior['projections']['delta'];delta=((d['delta'].astype(float)-p['mean'])@p['components'].T).astype(np.float32)
    v['base_harp_delta']=lr(prior['models']['base_harp_delta'],np.concatenate((d['base'],d['harp'],delta),axis=1))
    v['lookback_tuned']=lr(prior['models']['lookback_tuned']['candidate'],d['base'][:,:784])
    p=prior['models']['redeep_tuned']['candidate']
    pk=d['pks'][:,p['layers']].astype(float).sum(axis=1);ec=d['ecs'][:,p['heads']].astype(float).sum(axis=1)
    vals=(np.stack((pk,ec),axis=1)-p['minimum'])/p['scale'];v['redeep_tuned']=vals[:,0]-p['beta']*vals[:,1]
    with np.load(R22/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:old={n:z[n].copy() for n in OLD[:-1]}
    errors={n:near(v[n],old[n],('baseline replay',fold,n)) for n in OLD[:-1]}
    sm=read(R24/f'fold_{fold}_calibration.json');choice=sm['choices']['slots_base_smooth']
    assert sm['groups']=={k:r22[k] for k in ('fit_groups','calibration_groups','evaluation_groups')}
    byitem=defaultdict(list)
    for j,w in enumerate(windows):byitem[w['item_ids'][0]].append(j)
    seqs=[np.asarray(sorted(ix,key=lambda j:windows[j]['raw_token_indices'][0]),int) for ix in byitem.values()]
    v['slots_base_smooth']=smooth_logodds(v['slots_base'],seqs,choice['stay'],choice['temperature'])
    with np.load(R24/f'fold_{fold}_scores.npz',allow_pickle=False) as z:old['slots_base_smooth']=z['slots_base_smooth'].copy()
    errors['slots_base_smooth']=near(v['slots_base_smooth'],old['slots_base_smooth'],('smooth replay',fold))
    ts={n:r22['thresholds'][n] for n in OLD[:-1]};ts['slots_base_smooth']=sm['thresholds']['slots_base_smooth']
    for n in OLD[:-1]:assert ts[n]==prior['thresholds'][n]
    for path in (R21/f'fold_{fold}_frozen.pkl',R22/f'fold_{fold}_frozen22.pkl',R22/f'fold_{fold}_scores22.npz',R24/f'fold_{fold}_calibration.json',R24/f'fold_{fold}_scores.npz'):
        if str(path.resolve()) in source:assert sha(path)==source[str(path.resolve())]
    return old,ts,errors


def counts(rows,method,unit):
    rr=[r for r in rows if r['main_eligible']];y=np.asarray([r['gold'] for r in rr],int);p=np.asarray([r['predictions'][method] for r in rr],bool)
    tp=int(np.sum((y==1)&p));fp=int(np.sum((y==0)&p));fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p))
    return {unit:len(rr),'risk_'+unit:int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if tp+fn else None,'alert_rate':float(p.mean()),'risk_rate':float(y.mean())}


def audit(report):
    cfg,source=binding(report)
    windows=readl(OUT/'candidate_windows25.jsonl');answers=readl(OUT/'answer_index25.jsonl')
    assert windows==readl(R22/'candidate_windows22.jsonl') and answers==readl(R22/'answer_index22.jsonl')
    assert len(windows)==12222 and len(answers)==602 and all(r['split']=='train' for r in windows+answers)
    assert len({a['question_id'] for a in answers})==301 and len({a['group_id'] for a in answers})==278
    assert sum(w['main_eligible'] for w in windows)==9526 and sum(w['gold']==1 for w in windows)==1063
    assert sum(a['main_eligible'] for a in answers)==598 and sum(a['gold']==1 for a in answers)==151
    wlookup={w['window_key']:j for j,w in enumerate(windows)};alookup={a['item_id']:j for j,a in enumerate(answers)}
    assert len(wlookup)==len(windows) and len(alookup)==len(answers)
    byanswer=defaultdict(list)
    for j,w in enumerate(windows):byanswer[w['item_ids'][0]].append(j)
    assert set(byanswer)==set(alookup)
    assignment=read(PRELAB/'round19_three_signal_probe/data/fold_assignment.json')['groups']
    assert set(assignment)=={a['group_id'] for a in answers}
    with np.load(R21/'designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('base','slots','r19_all','harp','delta','ecs','pks')}
    with np.load(OUT/'formula_window_scores25.npz',allow_pickle=False) as z:formula={k:z[k].copy() for k in z.files}
    wf=readl(OUT/'window_scores_oof25.jsonl');af=readl(OUT/'answer_scores_oof25.jsonl');summary=read(OUT/'summary25.json')
    assert len(wf)==12222 and len(af)==602
    assert len({w['window_key'] for w in wf})==12222 and len({a['item_id'] for a in af})==602
    expected_w=[];expected_a=[];folds=[];replays={}
    for fold in range(5):
        f=read(OUT/f'fold_{fold}_calibration25.json')
        eg=sorted(g for g,v in assignment.items() if v==fold);cg=sorted(g for g,v in assignment.items() if v==(fold+1)%5);fg=sorted(set(assignment)-set(eg)-set(cg))
        assert f['groups']=={'fit_groups':fg,'calibration_groups':cg,'evaluation_groups':eg}
        with np.load(OUT/f'fold_{fold}_scores25.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        assert set(values)==set(METHODS) and all(s.shape==(12222,) and np.isfinite(s).all() for s in values.values())
        old,ts,errors=baseline(fold,d,windows,source)
        for n in OLD:
            assert np.array_equal(values[n],old[n]),('old arrays drift',fold,n)
            assert f['thresholds'][n]==ts[n]
        assert np.array_equal(values[FIXED[0]],formula['single_lambda05']) and np.array_equal(values[FIXED[1]],formula['joint_lambda05'])
        key={.25:'joint_lambda025',.5:'joint_lambda05',.75:'joint_lambda075'}[f['secondary_selected_lambda']]
        assert np.array_equal(values[CAL],formula[key])
        wi=[j for j,w in enumerate(windows) if w['group_id'] in eg];ai=[j for j,a in enumerate(answers) if a['group_id'] in eg]
        expected_w.extend(windows[j]['window_key'] for j in wi);expected_a.extend(answers[j]['item_id'] for j in ai)
        foldw=[w for w in wf if w['fold']==fold];folda=[a for a in af if a['fold']==fold]
        assert [w['window_key'] for w in foldw]==[windows[j]['window_key'] for j in wi]
        assert [a['item_id'] for a in folda]==[answers[j]['item_id'] for j in ai]
        for w,j in zip(foldw,wi):
            assert all(w[k]==v for k,v in windows[j].items())
            assert set(w['scores'])==set(w['predictions'])==set(METHODS)
            for n in METHODS:
                assert w['scores'][n]==float(values[n][j])
                assert w['predictions'][n]==bool(values[n][j]>=f['thresholds'][n]['window']['threshold'])
        for a,j in zip(folda,ai):
            assert all(a[k]==answers[j][k] for k in ('item_id','row_id','question_id','group_id','condition','gold','main_eligible','reviewed_safe_refusal'))
            assert set(a['scores'])==set(a['predictions'])==set(METHODS)
            for n in METHODS:
                score=float(values[n][byanswer[a['item_id']]].max())
                assert a['scores'][n]==score and a['predictions'][n]==bool(score>=f['thresholds'][n]['answer']['threshold'])
        folds.append({'fold':fold,'secondary_lambda':f['secondary_selected_lambda'],'outer_candidate_windows':len(wi),
            'outer_answers':len(ai),'all7_old_score_arrays_exact':True,'all7_old_thresholds_exact':True,
            'independent_old_replay_max_abs':errors,'OOF_scores_and_frozen_threshold_predictions_exact':True})
        replays[fold]=values;report['folds']=folds;save(report)
        print('R25_FOLD_BASELINE_OOF_PASSED',fold,flush=True)
    assert [w['window_key'] for w in wf]==expected_w and [a['item_id'] for a in af]==expected_a
    oldsummary=read(R22/'summary22.json')['methods'];smoothsummary=read(R24/'summary.json')['methods']
    for n in OLD[:-1]:assert summary['methods'][n]==oldsummary[n]
    assert summary['methods']['slots_base_smooth']==smoothsummary['slots_base_smooth']
    pooled={}
    for n in METHODS:
        pooled[n]={}
        for rr,unit in ((wf,'windows'),(af,'answers')):
            observed=counts(rr,n,unit);reference=summary['methods'][n][unit]
            assert observed.keys()==reference.keys()
            for k,v in observed.items():
                if v is None:assert reference[k] is None
                elif isinstance(v,int):assert reference[k]==v
                else:near(v,reference[k],('pooled counts',n,unit,k),atol=2e-15)
            pooled[n][unit]=observed
        safe=[a for a in af if a['main_eligible'] and a['reviewed_safe_refusal']]
        assert len(safe)==summary['methods'][n]['safe_refusals']==117 and all(a['gold']==0 for a in safe)
        assert sum(a['predictions'][n] for a in safe)==summary['methods'][n]['safe_refusal_false_positives']
    report.update(OOF_windows=12222,OOF_answers=602,evaluable_windows=9526,risk_windows=1063,
        evaluable_answers=598,risk_answers=151,safe_refusals=117,
        OOF_each_row_once_in_exact_frozen_outer_group=True,all7_old_pooled_summaries_exact=True,
        all20_main_pooled_metric_blocks_recomputed=True,pooled_counts=pooled,
        old_saved_score_arrays_compared=7*5*12222,
        old_manual_LR_ReDeEP_or_HMM_scores_replayed=7*5*12222,
        max_old_manual_replay_abs=max(v for f in folds for v in f['independent_old_replay_max_abs'].values()),
        primary_stays_fixed_lambda05=True,secondary_lambdas=[f['secondary_lambda'] for f in folds])
    peer=OUT/'LAMBDA_CALIBRATION_AUDIT25.json'
    if peer.exists():
        p=read(peer);assert p.get('passed') is True or p.get('status')=='passed'
        report['independent_lambda_calibration_audit']={'path':str(peer),'sha256':sha(peer),'passed':True}
    else:report['independent_lambda_calibration_audit']={'pending':True}
    report['audit_script_sha256']=sha(__file__)
    lines=['# R25 独立校准、旧基线与OOF审计','',
        '通过，无阻断。7条旧基线的全量保存分数、阈值及完整汇总与旧结果精确一致；另以冻结系数、ReDeEP公式和独立log-odds平滑递推回放。','',
        '全部5折校准表和score数组绑定于同一freeze；outer入口绑定freeze哈希，记录时间为先校准冻结再outer。OOF12222窗/602答各恰好一次，计分分母9526窗/598答；117条安全拒答保留负整答。','',
        '| 方法 | OOF定位F1 | OOF整答F1 |','|---|---:|---:|']
    for n in FIXED+(CAL,):lines.append(f"| {n} | {pooled[n]['windows']['f1']:.6f} | {pooled[n]['answers']['f1']:.6f} |")
    lines+=['','5折次要λ均为0.75；主方法仍固定λ=0.5，未因结果替换主方法。所有分数保留有符号原值，不裁剪为概率。',
        '未fit、使用GPU或读取原heldout/QA test。原词元公式、4BPE映射及提取源绑定由父代理另审；本报告不扩展为新测试有效性或未记录执行历史的证明。']
    if peer.exists():lines+=['',f'独立λ/阈值子审SHA256：{sha(peer)}。']
    (OUT/'CALIBRATION_AUDIT25.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    report={'status':'running','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
        'no_fit_or_GPU':True,'original_heldout_or_QA_test_read':False,'frozen_files_modified':False,
        'limits':['Raw-token formula/4BPE extraction-source audit belongs to parent; not repeated here.',
                  'New-method highlight-token/ranking/bootstrap details are outside this OOF main-count audit.',
                  'Repeated development OOF is not a fresh final test; secondary lambda selection does not replace fixed primary.']}
    try:
        with threadpool_limits(limits=4):audit(report)
        report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('R25_CALIBRATION_AUDIT_PASSED',sha(REPORT),flush=True)
