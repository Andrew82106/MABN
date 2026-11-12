"""Independent R27 additions only: fixed projections, mapping and30LR replay.

No production imports, new fit/SVD/GPU, original heldout or QA test. Previous
auditor functions supply inert pickle readers and independent weight/moment
formulas; the17 old methods are not reaudited here.
"""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,hashlib,traceback
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;PRELAB=ROOT.parent
LOCAL=PRELAB/'round23b_local_evidence_probe/results'
GLOBAL=PRELAB/'round22_evidence_verification/results'
HELPER=LOCAL/'audit_coefficients23b.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='2ef9157785dfbd15265ce06d51491d5859cac2c6f316e178ebee8d982e357085'
spec=importlib.util.spec_from_file_location('independent_r23b_formulas',HELPER)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
read,readl,sha,unpickle,near=q.read,q.readl,q.sha,q.unpickle,q.near
REPORT=OUT/'INDEPENDENT_AUDIT27.json'
METHODS=('joint_mean','joint_slots');WIDTHS=(915,3274);CS=(.001,.01,.1)

def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def binding(report):
    freeze,complete=read(OUT/'fit_freeze.json'),read(OUT/'complete.json')
    assert read(OUT/'started.json')==freeze['snapshot']
    assert read(OUT/'test_started.json')['freeze_sha256']==sha(OUT/'fit_freeze.json')
    checks={}
    for m in (freeze,complete):
        for n,h in m['files_sha256'].items():
            p=(OUT/n).resolve();assert p.is_relative_to(OUT) and sha(p)==h;checks[n]=h
    mapping={'code_sha256':ROOT/'src/run27.py','protocol_sha256':ROOT/'protocol.json',
        'r26_complete_sha256':PRELAB/'round26_global_local_fusion/results/complete.json',
        'r23_fit_sha256':LOCAL/'fit_freeze.json','r22_design_sha256':GLOBAL/'verifier_designs22.npz'}
    for k,p in mapping.items():assert sha(p)==freeze['snapshot'][k]
    cfg=read(ROOT/'protocol.json');assert cfg['C']==list(CS) and cfg['loss_mass']==3854 and cfg['seed']==20260927
    assert cfg['new_extraction'] is False and cfg['new_labels'] is False
    report['source_sha256']={str(p):sha(p) for p in (*mapping.values(),HELPER,ROOT/'src/run27.py')}
    report['freeze_provenance']={'all5fold10files_bound':True,'test_started_binds_exact_freeze':True,
        'UTC_timestamps_recorded':False,'files_sha256':checks,
        'limit':'Hash chain and entrypoint support freeze-before-outer; artifacts omit UTC timestamps, so no independent proof of external execution order.'}
    return cfg


def projection(pc,raw,ix,rows,seed,tag):
    assert np.array_equal(pc['fit_ix'],ix)
    keys=[r.get('window_key',r.get('item_id')) for r in rows]
    assert pc['fit_keys']==keys and pc['fit_groups']==sorted({r['group_id'] for r in rows})
    normalized=[dict(r,item_ids=[r['item_id']]) if 'item_ids' not in r else r for r in rows]
    b,_,_,_,_=q.independent_weights(normalized)
    be=near(b,pc['base_weights'],tag+' base weights')
    w=b/b.sum();we=near(w,pc['pca_weights'],tag+' PCA weights',atol=1e-15)
    mean=np.sum(raw[ix].astype(float)*w[:,None],axis=0)
    me=near(mean,pc['mean'],tag+' mean',atol=2e-11)
    assert pc['seed']==seed and pc['whiten'] is False and pc['components'].shape==(64,3584)
    projected=((raw.astype(float)-pc['mean'])@pc['components'].T).astype(np.float32)
    return projected,{'fit_indices_keys_groups_exact':True,'fit_rows':len(ix),'fit_groups':len(pc['fit_groups']),
        'base_weight_max_abs':be,'PCA_weight_max_abs':we,'fit_mean_max_abs':me,'no_refit':True}


def audit(report):
    binding(report)
    windows=readl(LOCAL/'candidate_windows.jsonl');answers=readl(GLOBAL/'answer_index22.jsonl')
    assert len(windows)==12222 and len(answers)==602 and all(r['split']=='train' for r in windows+answers)
    assert len({a['item_id'] for a in answers})==602 and len({a['group_id'] for a in answers})==278
    with np.load(LOCAL/'designs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('base','slots','hidden','extra')}
    with np.load(GLOBAL/'verifier_designs22.npz',allow_pickle=False) as z:g={k:z[k].copy() for k in ('hidden','extra','window_answer_index')}
    ai=g['window_answer_index'];assert ai.shape==(12222,) and np.issubdtype(ai.dtype,np.integer) and ai.min()>=0 and ai.max()<602
    assert all(answers[int(j)]['item_id']==w['item_ids'][0] and answers[int(j)]['row_id']==w['row_id'] for w,j in zip(windows,ai))
    groups={a['group_id'] for a in answers};appearance=np.zeros(12222,int);folds=[]
    wgid=np.asarray([w['group_id'] for w in windows]);wel=np.asarray([w['main_eligible'] for w in windows],bool)
    agid=np.asarray([a['group_id'] for a in answers]);ael=np.asarray([a['main_eligible'] for a in answers],bool)
    localfreeze=read(LOCAL/'fit_freeze.json')
    globalfreeze=read(GLOBAL/'fit_freeze22.json')
    for fold in range(5):
        obj=unpickle(OUT/f'fold_{fold}_frozen.pkl');loc=unpickle(LOCAL/f'fold_{fold}_frozen.pkl');glob=unpickle(GLOBAL/f'fold_{fold}_frozen22.pkl')
        assert sha(LOCAL/f'fold_{fold}_frozen.pkl')==obj['local_projection_source_sha256']==localfreeze['files_sha256'][f'fold_{fold}_frozen.pkl']
        assert sha(GLOBAL/f'fold_{fold}_frozen22.pkl')==obj['global_projection_source_sha256']==globalfreeze['files_sha256'][f'fold_{fold}_frozen22.pkl']
        assert obj['groups']==loc['groups'] and all(obj['groups'][k]==glob[k] for k in obj['groups'])
        fg,cg,eg=(set(obj['groups'][k]) for k in ('fit_groups','calibration_groups','evaluation_groups'))
        assert not(fg&cg or fg&eg or cg&eg) and fg|cg|eg==groups
        wi=np.flatnonzero(np.isin(wgid,list(fg))&wel);xi=np.flatnonzero(np.isin(agid,list(fg))&ael);appearance[wi]+=1
        assert np.array_equal(glob['fit_answer_indices'],xi)
        lp,lpinfo=projection(loc['projection'],d['hidden'],wi,[windows[i] for i in wi],20260923+fold,'local')
        gp,gpinfo=projection(glob['projection'],g['hidden'],xi,[answers[i] for i in xi],20260922+fold,'global')
        assert set(loc['projection']['fit_groups'])<=fg and set(glob['projection']['fit_groups'])<=fg
        local=np.concatenate((lp,d['extra'][:,:1]),axis=1)
        global_answer=np.concatenate((gp,g['extra'][:,:1]),axis=1);global_window=global_answer[ai]
        with np.load(LOCAL/f'fold_{fold}_scores.npz',allow_pickle=False) as z:assert np.array_equal(lp,z['projected_hidden'])
        with np.load(GLOBAL/f'fold_{fold}_scores22.npz',allow_pickle=False) as z:assert np.array_equal(global_answer,z['probe_design'][:,:65])
        with np.load(OUT/f'fold_{fold}_scores.npz',allow_pickle=False) as z:
            assert np.array_equal(local,z['local65']) and np.array_equal(global_window,z['global65'])
            saved={m:z[m].copy() for m in METHODS}
        assert local.shape==global_window.shape==(12222,65)
        rows=[windows[i] for i in wi];keys=[r['window_key'] for r in rows];fitgroups=sorted({r['group_id'] for r in rows})
        b,y,factors,loss,group_rows=q.independent_weights(rows)
        fr={'fold':fold,'local_projection':lpinfo,'global_projection':gpinfo,'global_answer_to_window_all12222_identity_exact':True,
            'local65_and_global65_fixed_forward_and_saved_arrays_exact':True,'both_full_vocab_mass_columns_omitted':True,
            'fit_windows':len(wi),'fit_answers':len({r['item_ids'][0] for r in rows}),'models':{}}
        for name,base,width in zip(METHODS,('base','slots'),WIDTHS):
            design=np.concatenate((d[base],local,global_window),axis=1).astype(np.float32)
            assert design.shape==(12222,width)
            bundle=obj['models'][name];candidates=bundle['all_lr_candidates'];assert len(candidates)==3
            mean,var,scale,nseen,constants=q.scaler_moments(design,wi,b);cr=[]
            for j,c in enumerate(candidates):
                assert np.array_equal(c['fit_ix'],wi) and c['fit_keys']==keys and c['fit_groups']==fitgroups and np.array_equal(c['fit_y'],y)
                assert c['C']==CS[j] and c['width']==width and c['loss_mass']==3854
                be=near(c['base_weights'],b,'LR weights');fe=near(c['class_factors'],factors,'LR class factors');le=near(c['loss_weights'],loss,'LR loss')
                near(c['loss_weights'].sum(),3854.,'total loss')
                ge=max(abs(c['loss_weights'][ii].sum()-3854/len(group_rows)) for ii in group_rows.values());assert ge<1e-10
                sc,model=c['scaler'],c['model']
                se={'mean':near(sc.mean_,mean,'scaler mean',atol=2e-11),'var':near(sc.var_,var,'scaler var',atol=2e-10),
                    'scale':near(sc.scale_,scale,'scaler scale',atol=2e-10),'weighted_n_seen':near(sc.n_samples_seen_,nseen,'weight count')}
                assert model.C==CS[j] and model.random_state==20260927 and model.solver=='liblinear' and model.penalty=='l2' and model.class_weight is None
                assert model.max_iter==2000 and model.n_iter_.max()<2000 and np.array_equal(model.classes_,[0,1])
                assert model.coef_.shape==(1,width) and model.intercept_.shape==(1,)
                z=design.copy();z-=sc.mean_;z/=sc.scale_
                scores=q.sigmoid(z@model.coef_[0]+model.intercept_[0]);assert np.isfinite(scores).all() and scores.shape==(12222,)
                selected=j==bundle['selected_candidate'];error=None
                if selected:
                    assert bundle['C']==c['C'] and np.array_equal(bundle['model'].coef_,model.coef_) and np.array_equal(bundle['model'].intercept_,model.intercept_)
                    error=near(scores,saved[name],'selected stored probabilities',rtol=0,atol=2e-12)
                cr.append({'C':c['C'],'fit_indices_y_keys_groups_exact':True,'base_weight_max_abs':be,'class_factor_max_abs':fe,
                    'loss_weight_max_abs':le,'group_loss_max_abs':float(ge),'scaler_max_abs':se,'constant_columns':constants,
                    'full12222_coefficients_replayed':True,'selected':selected,'saved_selected_probability_max_abs':error})
            fr['models'][name]={'width':width,'selected_C':bundle['C'],'selected_candidate':int(bundle['selected_candidate']),'candidates':cr}
        folds.append(fr);report['folds']=folds;save(report);print('R27_MAPPING_COEFFICIENTS_PASSED',fold,flush=True)
    assert np.all(appearance[wel]==3) and np.all(appearance[~wel]==0)
    cs=[c for f in folds for m in f['models'].values() for c in m['candidates']]
    report.update(status='passed_with_provenance_limitations',numerical_status='passed',blockers=[],
        local_global_projection_pairs_checked=5,new_LR_checked=30,all_candidate_window_scores_replayed=30*12222,
        selected_stored_window_scores_compared=10*12222,max_selected_score_abs=max(c['saved_selected_probability_max_abs'] for c in cs if c['selected']),
        max_scaler_mean_abs=max(c['scaler_max_abs']['mean'] for c in cs),max_scaler_var_abs=max(c['scaler_max_abs']['var'] for c in cs),
        max_loss_weight_abs=max(c['loss_weight_max_abs'] for c in cs),all_eligible_windows_in_fit3times=True,
        all_ineligible_windows_absent_from_fit=True,evaluable_windows=int(wel.sum()),candidate_windows=12222,answers=602)
    peer=OUT/'CALIBRATION_AUDIT27.json'
    if peer.exists():
        p=read(peer);assert p.get('passed') is True or p.get('status') in ('passed','passed_with_provenance_limitations')
        report['calibration_and_smoothing_subaudit']={'path':str(peer),'sha256':sha(peer),'passed':True}
    else:report['calibration_and_smoothing_subaudit']={'pending':True}
    summary=read(OUT/'summary.json')['methods']
    report['new_OOF_F1']={n:{'window_F1':summary[n]['windows']['f1'],'answer_F1':summary[n]['answers']['f1']} for n in METHODS+tuple(m+'_smooth' for m in METHODS)}
    report['audit_script_sha256']=sha(__file__)
    lines=['# R27 新增实现独立审计','',
        '数值核验通过，无数值阻断；来源与执行顺序限制保留。5折local65/global65映射及冻结投影fit范围正确，30个LR的fit索引、权重、scaler及系数回放通过。',
        '局部65列和整答65列均为PCA64加A/B logit差；两路的A+B词表概率质量均未进入新设计。global逐窗广播逐项对应原answer_id。','',
        '| 新方法 | OOF定位F1 | OOF整答F1 |','|---|---:|---:|']
    for n,v in report['new_OOF_F1'].items():lines.append(f"| {n} | {v['window_F1']:.6f} | {v['answer_F1']:.6f} |")
    lines+=['','只审新增实现，未重复17条旧方法的来源全审；未fit/PCA重拟合/GPU或读取原heldout/QA test。',
        '原运行prepare继承R26的先解析混合划分输入再筛train路径；只能确认这些保存投影/拟合/评分行属于实际train，不能声称原运行从未解析heldout输入文本。',
        'freeze与outer哈希链完整，但缺UTC时间记录，不能独立证明外部历史顺序。数据是反复查看的辅助标注开发集，不是新最终测试；本审计未重跑bootstrap或辅助高亮/排序。']
    if peer.exists():lines+=['',f'独立校准/平滑报告SHA256：{sha(peer)}。']
    (OUT/'INDEPENDENT_AUDIT27.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    report={'status':'running','reviewer':'/root/data_build/extract_review','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'R27新增映射/冻结投影/30LR与校准平滑；不重复17旧来源全审',
        'no_fit_SVD_or_GPU':True,'audit_read_original_heldout_or_QA_test':False,'production_modules_imported':False,'frozen_files_modified':False,
        'provenance_limitations':['Original production prepare calls R26 prepare, whose train_metadata parses mixed-split input before selecting actual train. This audit reads only frozen actual-train artifacts.',
            'No UTC timestamps in original started/freeze/test records; code and hashes support ordering but cannot independently prove external execution history.',
            'Repeatedly inspected assistant-labeled development OOF is not fresh final-test or human-QA evidence.'],
        'numeric_limitations':['No new PCA/SVD or LR fits; fixed bases verified through exact source hashes, fit indices, weighted means and full forward mapping.',
            'Only10 selected raw LR score arrays exist; other20 coefficient objects replay and are checked against cal tables in peer audit.',
            'Old17 source pipelines and new bootstrap/auxiliary ranking/highlighting are outside this bounded implementation audit.']}
    try:
        with threadpool_limits(limits=4):audit(report)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('R27_INDEPENDENT_IMPLEMENTATION_AUDIT_PASSED',sha(REPORT),flush=True)
