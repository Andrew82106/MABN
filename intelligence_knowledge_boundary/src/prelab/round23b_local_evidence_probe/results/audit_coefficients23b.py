"""Independent R23b frozen PCA/LR numerical audit; no fit/GPU/heldout.

Only actual-train candidate windows, OOF answer metadata, design matrices and
frozen models. sklearn pickle objects are inert holders; no production imports.
All45 coefficient arrays replay, but only15 selected full scores were stored.
The other30 candidates are checked against calibration artifacts by a peer.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import hashlib,json,pickle,traceback
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PRELAB=ROOT.parent
REPORT=OUT/'COEFFICIENT_AUDIT23B.json'
METHODS=('local_probe','local_mean_fusion','local_slots_fusion')
WIDTHS=(66,851,3210)
CS=(.001,.01,.1)


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
    def find_class(self,module,name):
        if module.startswith('sklearn.'):return Frozen
        if module.startswith('numpy') or module in ('builtins','collections'):return super().find_class(module,name)
        raise ValueError((module,name))
def unpickle(p):
    with Path(p).open('rb') as f:return Reader(f).load()
def near(a,b,tag,rtol=1e-10,atol=1e-11):
    a,b=np.asarray(a),np.asarray(b)
    error=float(np.max(np.abs(a.astype(float)-b.astype(float))))
    assert a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol),(tag,error)
    return error
def sigmoid(z):
    result=np.empty(z.shape,np.float64);keep=z>=0
    result[keep]=1/(1+np.exp(-z[keep]));e=np.exp(z[~keep]);result[~keep]=e/(1+e)
    return result


def binding(report):
    complete=read(OUT/'complete.json');freeze=read(OUT/'fit_freeze.json');checks={}
    for manifest in (complete,freeze):
        for n,h in manifest['files_sha256'].items():
            p=(OUT/n).resolve();assert p.is_relative_to(OUT)
            assert sha(p)==h;checks[n]=h
    cfg=read(ROOT/'protocol.json')
    assert cfg['C']==list(CS) and cfg['loss_mass']==3854 and cfg['new_lr_fits']==45 and cfg['new_pca_fits']==5
    assert cfg['seed']==20260923 and cfg['original_validation_or_test_used'] is False
    assert sha(ROOT/'protocol.json')==freeze['snapshot']['protocol_sha256']
    assert sha(ROOT/'src/run23.py')==freeze['snapshot']['code_sha256']
    assert read(OUT/'fit_started.json')['snapshot']==freeze['snapshot']
    assert read(OUT/'test_started.json')['freeze_sha256']==sha(OUT/'fit_freeze.json')
    report['frozen_files_sha256']=checks
    report['source_sha256']={str(p):sha(p) for p in (ROOT/'protocol.json',ROOT/'src/run23.py',
        PRELAB/'round21_semantic_internal_probe/src/run21.py',PRELAB/'round20_localization_optimization/src/run20.py',
        PRELAB/'round13_generalization_diagnostics/src/run13.py',PRELAB/'round10_dual_granularity/src/evaluate10.py')}
    return cfg


def metadata():
    windows=readl(OUT/'candidate_windows.jsonl');answers=readl(OUT/'answer_scores_oof.jsonl')
    assert len(windows)==12222 and len(answers)==602
    assert all(w['split']=='train' and len(w['item_ids'])==1 for w in windows)
    assert len({w['window_key'] for w in windows})==12222
    byitem={a['item_id']:a for a in answers};assert len(byitem)==602
    assert len({a['question_id'] for a in answers})==301 and len({a['group_id'] for a in answers})==278
    for w in windows:
        a=byitem[w['item_ids'][0]]
        assert all(w[k]==a[k] for k in ('row_id','question_id','group_id','condition'))
    # Named R22 actual-train metadata proves item labels/eligibility and groups.
    prior=readl(PRELAB/'round22_evidence_verification/results/answer_index22.jsonl')
    assert len(prior)==602 and all(a['split']=='train' for a in prior)
    assert set(byitem)=={a['item_id'] for a in prior}
    for a in prior:
        assert all(a[k]==byitem[a['item_id']][k] for k in ('row_id','question_id','group_id','condition','gold','main_eligible'))
    return windows,answers


def independent_weights(rows):
    n=len(rows);groups=defaultdict(list);conditions=defaultdict(set);items=defaultdict(set);item_counts=Counter()
    for i,r in enumerate(rows):
        g,c,a=r['group_id'],r['condition'],r['item_ids'][0]
        groups[g].append(i);conditions[g].add(c);items[g,c].add(a);item_counts[g,c,a]+=1
    G=len(groups)
    b=np.asarray([(n/G)/(len(conditions[r['group_id']])*len(items[r['group_id'],r['condition']])*item_counts[r['group_id'],r['condition'],r['item_ids'][0]]) for r in rows])
    y=np.asarray([r['gold'] for r in rows],int)
    masses=np.asarray([b[y==k].sum() for k in (0,1)])
    factors=b.sum()/(2*masses);u=b*factors[y];loss=np.empty(n)
    for ix in groups.values():loss[ix]=(3854/G)*u[ix]/u[ix].sum()
    return b,y,factors,loss,groups


def scaler_moments(design,ix,b):
    x=design[ix].astype(np.float64);w=b.astype(np.float32).astype(np.float64);n=w.sum()
    mean=np.sum(x*w[:,None],axis=0)/n
    centered=x-mean
    var=np.sum(centered*centered*w[:,None],axis=0)/n
    eps=np.finfo(np.float64).eps
    constant=var<=n*eps*var+(n*mean*eps)**2
    scale=np.sqrt(var);scale[constant]=1.
    return mean,var,scale,n,int(constant.sum())


def audit(report):
    cfg=binding(report);windows,answers=metadata()
    groups={a['group_id'] for a in answers};elig=np.asarray([w['main_eligible'] for w in windows],bool)
    frozen=[unpickle(OUT/f'fold_{i}_frozen.pkl') for i in range(5)]
    outers=[set(f['groups']['evaluation_groups']) for f in frozen]
    assert set.union(*outers)==groups and sum(map(len,outers))==len(groups)
    with np.load(OUT/'designs.npz',allow_pickle=False) as z:designs={k:z[k].copy() for k in ('hidden','extra','base','slots')}
    assert {k:a.shape for k,a in designs.items()}=={'hidden':(12222,3584),'extra':(12222,2),'base':(12222,785),'slots':(12222,3144)}
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in designs.values())
    h=designs['hidden'].astype(np.float64);appearances=np.zeros(len(windows),int);reports=[]
    for fold,f in enumerate(frozen):
        fg=set(f['groups']['fit_groups']);cg=set(f['groups']['calibration_groups']);eg=outers[fold]
        prior=unpickle(PRELAB/f'round22_evidence_verification/results/fold_{fold}_frozen22.pkl')
        assert all(f['groups'][k]==prior[k] for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert cg==outers[(fold+1)%5] and fg==groups-cg-eg
        assert not(fg&cg or fg&eg or cg&eg)
        ix=np.flatnonzero([w['main_eligible'] and w['group_id'] in fg for w in windows]);appearances[ix]+=1
        rows=[windows[i] for i in ix];keys=[w['window_key'] for w in rows];fitgroups=sorted({w['group_id'] for w in rows})
        b,y,factors,loss,gi=independent_weights(rows)
        pc=f['projection'];assert np.array_equal(pc['fit_ix'],ix) and pc['fit_keys']==keys and pc['fit_groups']==fitgroups
        berror=near(pc['base_weights'],b,'PCA base')
        pw=b/b.sum();pwerror=near(pc['pca_weights'],pw,'PCA weights',atol=1e-15)
        mean=np.sum(h[ix]*pw[:,None],axis=0)
        meanerror=near(pc['mean'],mean,'fit only PCA mean',atol=2e-11)
        components=pc['components'];sv=pc['singular_values']
        assert components.shape==(64,3584) and sv.shape==(64,) and np.isfinite(components).all() and np.isfinite(sv).all()
        assert pc['seed']==20260923+fold and pc['whiten'] is False
        ortho=near(components@components.T,np.eye(64),'PCA orthonormal',atol=2e-10)
        # Frozen basis only: no new PCA/SVD/eigendecomposition.
        projected=((h-pc['mean'])@components.T).astype(np.float32)
        with np.load(OUT/f'fold_{fold}_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(projected,z['projected_hidden']),(fold,'PCA forward')
            saved={m:z[m].copy() for m in METHODS}
        local=np.concatenate((projected,designs['extra']),axis=1)
        family=[local,np.concatenate((designs['base'],local),axis=1),np.concatenate((designs['slots'],local),axis=1)]
        fr={'fold':fold,'fit_windows':len(ix),'fit_answers':len({r['item_ids'][0] for r in rows}),
            'fit_groups':len(gi),'fit_class_counts':np.bincount(y,minlength=2).tolist(),
            'fit_ix_keys_groups_exact':True,'PCA_weight_max_abs':berror,'PCA_normalized_weight_max_abs':pwerror,
            'PCA_fit_mean_max_abs':meanerror,'PCA_orthonormal_max_abs':ortho,'PCA_all12222_forward_exact':True,'families':{}}
        for name,design,width in zip(METHODS,family,WIDTHS):
            assert design.shape==(12222,width) and design.dtype==np.float32
            bundle=f['models'][name];candidates=bundle['all_lr_candidates'];assert len(candidates)==3
            expected_mean,expected_var,expected_scale,nseen,constants=scaler_moments(design,ix,b)
            candidate_reports=[]
            for j,obj in enumerate(candidates):
                assert np.array_equal(obj['fit_ix'],ix) and obj['fit_keys']==keys and obj['fit_groups']==fitgroups
                assert np.array_equal(obj['fit_y'],y) and obj['width']==width and obj['C']==CS[j] and obj['loss_mass']==3854
                be=near(obj['base_weights'],b,'base weights');fe=near(obj['class_factors'],factors,'class factors');le=near(obj['loss_weights'],loss,'loss weights')
                lm=float(obj['loss_weights'].sum());near(lm,3854.,'loss mass',atol=1e-10)
                group_error=max(abs(obj['loss_weights'][ii].sum()-3854/len(gi)) for ii in gi.values());assert group_error<1e-10
                sc,model=obj['scaler'],obj['model'];assert sc.n_features_in_==width and sc.with_mean and sc.with_std
                se={'mean':near(sc.mean_,expected_mean,'scaler fit mean',atol=2e-11),
                    'variance':near(sc.var_,expected_var,'scaler fit variance',atol=2e-10),
                    'scale':near(sc.scale_,expected_scale,'scaler scale',atol=2e-10),
                    'n_samples_seen':near(sc.n_samples_seen_,nseen,'scaler weight mass',atol=1e-10)}
                assert model.coef_.shape==(1,width) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
                assert model.C==CS[j] and model.solver=='liblinear' and model.penalty=='l2' and model.class_weight is None
                assert model.random_state==20260923 and model.max_iter==2000 and model.n_iter_.max()<2000
                scaled=design.copy();scaled-=sc.mean_;scaled/=sc.scale_
                scores=sigmoid(scaled@model.coef_[0]+model.intercept_[0])
                assert scores.shape==(12222,) and np.isfinite(scores).all()
                chosen=j==bundle['selected_candidate'];savederror=None
                if chosen:
                    assert bundle['C']==obj['C']
                    for attr in ('coef_','intercept_'):assert np.array_equal(getattr(bundle['model'],attr),getattr(model,attr))
                    for attr in ('mean_','var_','scale_'):assert np.array_equal(getattr(bundle['scaler'],attr),getattr(sc,attr))
                    savederror=near(scores,saved[name],'selected full score',rtol=0,atol=2e-12)
                candidate_reports.append({'C':obj['C'],'fit_indices_keys_groups_y_exact':True,'full12222_coef_replay_finite':True,
                    'base_weight_max_abs':be,'class_factor_max_abs':fe,'loss_weight_max_abs':le,'loss_total_mass':lm,
                    'per_group_equal_loss_max_abs':float(group_error),'scaler_moment_max_abs':se,
                    'constant_columns':constants,'scaler_effective_weights':'base float32 before float64 moments',
                    'iterations':model.n_iter_.tolist(),'selected':chosen,'selected_saved_score_max_abs':savederror})
            fr['families'][name]={'width':width,'selected_candidate':int(bundle['selected_candidate']),
                'selected_C':bundle['C'],'candidates':candidate_reports}
        reports.append(fr);report['folds']=reports;save(report)
        print('R23B_FOLD_COEFFICIENT_AUDIT_PASSED',fold,flush=True)
    assert np.all(appearances[elig]==3) and np.all(appearances[~elig]==0)
    cs=[c for f in reports for m in f['families'].values() for c in m['candidates']]
    report.update(pca_checked=5,LR_candidates_checked=45,all_candidate_coefficients_replayed_windows=45*12222,
        selected_saved_scores_compared=15*12222,max_selected_saved_probability_abs=max(c['selected_saved_score_max_abs'] for c in cs if c['selected']),
        max_PCA_fit_mean_abs=max(f['PCA_fit_mean_max_abs'] for f in reports),
        max_scaler_mean_abs=max(c['scaler_moment_max_abs']['mean'] for c in cs),
        max_scaler_var_abs=max(c['scaler_moment_max_abs']['variance'] for c in cs),
        max_loss_weight_abs=max(c['loss_weight_max_abs'] for c in cs),
        eligible_windows=int(elig.sum()),ineligible_windows=int((~elig).sum()),all_eligible_fit_appearances=3,all_ineligible_fit_appearances=0)
    peer=OUT/'CALIBRATION_AUDIT23B.json'
    if peer.exists():
        p=read(peer);assert p.get('passed') is True or p.get('status')=='passed'
        report['independent_calibration_audit']={'path':str(peer),'sha256':sha(peer),'passed':True}
    else:report['independent_calibration_audit']={'pending':True}
    report['audit_script_sha256']=sha(__file__)
    lines=['# R23b 独立系数审计','',
        '通过，无阻断。仅使用R16实际train既有数据；未fit、重做PCA/SVD、使用GPU或读取原validation/test。','',
        '5个PCA的fit范围、加权均值与12222行固定基投影一致；45个LR的fit索引/标签/权重/scaler及系数回放通过。9526个可评窗口各入fit3次，2696个非主评窗口未入fit。','',
        f"15个选中模型的183330个保存分数最大回放误差：{report['max_selected_saved_probability_abs']:.3g}。其余30个模型没有保存全量分数，因此以系数回放和独立校准阈值核验为证，不声称逐元素对照不存在的原数组。",'',
        '| fold | local_probe C | local_mean_fusion C | local_slots_fusion C | fit窗口 |','|---:|---:|---:|---:|---:|']
    for fr in reports:
        values=[fr['families'][m]['selected_C'] for m in METHODS]
        lines.append(f"| {fr['fold']} | {values[0]} | {values[1]} | {values[2]} | {fr['fit_windows']} |")
    if peer.exists():
        lines+=['','独立校准子审确认90个阈值字典、45个选择key及15个C选择精确一致。整答先对全部候选窗口取最大值，再筛整答资格；117条安全拒答的非主评窗口仍参与整答评分。',
                f"校准报告SHA256：{sha(peer)}。"]
    lines+=['','这些是已有训练数据上的分组交叉验证结果；本审计不重审父代理负责的旧基线、平滑和汇总，也不证明未记录的外部执行历史。']
    (OUT/'COEFFICIENT_AUDIT23B.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    report={'status':'running','reviewer':'/root/data_build/extract_review','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'R23b R16 actual-train602 frozen5fold PCA/45LR only','no_fit_or_SVD':True,'no_GPU':True,
        'original_R16_validation_or_test_opened':False,'production_modules_imported':False,'old_files_modified':False,
        'limits':['Frozen PCA fit provenance, weighted means and exact fixed-basis forward checked; no PCA or SVD refit.',
                  'All45 coefficient arrays reconstruct full scores, but only15 selected full arrays were saved; other30 are checked against calibration thresholds in the independent peer audit.',
                  'Candidate answer score must max ALL candidate windows including safe-refusal non-main windows, then filter answer eligibility; QA rules are not used.',
                  'Hash and numerical checks do not independently prove unrecorded external execution history.']}
    try:
        with threadpool_limits(limits=4):audit(report)
        report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('R23B_COEFFICIENT_AUDIT_PASSED',sha(REPORT),flush=True)
