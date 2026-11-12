"""Completed heading-source18LR audit; inert coefficients, no fit/GPU/test."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,gc,traceback
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SEM=OUT.parent/'citation_heading_semantic_v1';SCOPE=OUT.parent/'citation_heading_scope_v1'
LEX=OUT.parent/'citation_alignment_v1';LEGACY=OUT.parent/'cited_source_semantic_v1'
UP=OUT.parent/'completed_score_fusion_v1';DEV=OUT.parent/'development_v1'
PEERS=('lookback','harp_claim','semantic_claim');MODES=('any_source_control','inherited_source_gap');CS=(.001,.01,.1)
NFIT,NTOTAL=168123,210364
REPORT=OUT/'INDEPENDENT_AUDIT.json';HASHES={}
def sha(p):
    p=Path(p).resolve();p.relative_to(ROOT)
    assert not any(x.lower() in ('raw','final_test','sealed','withheld') for x in p.parts)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    HASHES[str(p)]=h.hexdigest();return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(p,h):assert sha(p)==h,str(p)
def audit():
    assert (OUT/'complete.json').exists(),'No partial results'
    done=read(OUT/'complete.json');summary=read(OUT/'summary.json');cfg=read(OUT/'protocol.json');freeze=read(OUT/'design_freeze.json')
    assert done['status']=='complete' and done['fits_completed']==summary['fits_completed']==18
    assert done['reference_refits']==summary['reference_refits']==0 and not done['official_test_opened']
    verify(OUT/'summary.json',done['summary_sha256']);verify(OUT/'protocol.json',freeze['protocol_sha256'])
    for p,h in freeze['source_sha256'].items():verify(p,h)
    verify(OUT/'CPU_SELFCHECK.json',freeze['cpu_selfcheck_sha256']);verify(OUT/'PREVIOUS_SCOPE_REFERENCES.json',freeze['previous_references_sha256'])
    started=read(OUT/'started.json');verify(OUT/'design_freeze.json',started['design_freeze_sha256']);verify(SEM/'features_complete.json',started['feature_complete_sha256'])
    proto=cfg['frozen_producer_training_protocol'];assert proto==read(SEM/'training_protocol.json')
    assert proto['peers']==list(PEERS) and proto['modes']==list(MODES) and proto['C']==list(CS) and proto['planned_fits']==18
    assert cfg['input_widths']==dict(zip(MODES,(14,15)))
    H=OUT.parent/'lookback_controls_v2/audit_lookback_controls_v2.py'
    verify(H,'dc5f32fed2666604907e2474fdb6a9bec46b30f4953b1109b43112c6494414eb')
    spec=importlib.util.spec_from_file_location('prior_independent_audit',H);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    q=a.q;near=a.near
    answers,windows,indices=q.metadata();bw,weight_report=q.audit_weights(windows)
    y=np.array([w['label'] for w in windows]);ay=np.array([r['label'] for r in answers])
    assert (len(answers),len(windows),int(y[:NFIT].sum()),int(y[NFIT:].sum()),int(ay[:634].sum()),int(ay[634:].sum()))==(793,210364,21477,5984,328,100)
    # Actual frozen source branch; do not substitute the earlier6252 support.
    sf=read(SEM/'preparation_freeze.json')
    for n,h in sf['files_sha256'].items():verify(SEM/n,h)
    for p,h in sf['source_sha256'].items():verify(p,h)
    assert sf['no_test'] and not sf['trained']
    for folder,complete in [(LEX,'complete.json'),(SCOPE,'features_complete.json'),(SEM,'features_complete.json')]:
        f=read(folder/complete)
        for n,h in f['files_sha256'].items():verify(folder/n,h)
        assert read(folder/'geometry.json')['window_order_sha256']==digest([w['window_id'] for w in windows])
    inf=read(SEM/'inference_complete.json');agreement=read(SEM/'numeric_agreement.json')
    assert inf['status']=='complete' and (inf['answers'],inf['pairs'])==(37,1161) and inf['no_test'] and not inf['trained']
    verify(SEM/'preparation_freeze.json',inf['preparation_freeze_sha256']);verify(SEM/'numeric_agreement.json',inf['numeric_agreement_sha256'])
    assert agreement['response_id']=='16023' and agreement['passed'] and agreement['logit_max_abs_diff']<=2e-4 and agreement['support_max_abs_diff']<=2e-5
    verify(ROOT/'semantic_baseline/cuda_variant/scores/16023.json',agreement['reference_sha256'])
    plans={p['response_id']:p for p in lines(SEM/'pair_plans.jsonl')};assert len(plans)==37
    claim_scores={};softmax_error=0.;pair_count=0
    for rec in inf['records']:
        path=SEM/rec['path'];verify(path,rec['sha256']);score=read(path);plan=plans[rec['response_id']]
        assert score['response_id']==rec['response_id'] and score['partition']==plan['partition'] and plan['partition'] in ('fit','calibration')
        assert score['pair_plan_sha256']==digest(plan) and score['preparation_freeze_sha256']==sha(SEM/'preparation_freeze.json')
        assert score['source_ids']==[1,2,3] and score['claim_indices']==[c['claim_index'] for c in plan['claims']]
        expected_pairs=[{'claim_index':c['claim_index'],'global_claim_index':c['global_claim_index'],**p} for c in plan['claims'] for p in c['pairs']]
        assert score['pairs']==expected_pairs and all([p['source_id'] for p in c['pairs']]==[1,2,3] for c in plan['claims'])
        z=np.array(score['logits']);s=np.array(score['support_by_claim_source']);assert z.shape==(rec['pairs'],2) and s.shape==(len(plan['claims']),3)
        assert np.isfinite(z).all() and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        softmax_error=max(softmax_error,near(q.sigmoid(z[:,1]-z[:,0]),s.ravel(),'pair softmax',rtol=0,atol=2e-7))
        for ci,support in zip(score['claim_indices'],s):
            key=(rec['response_id'],ci);assert key not in claim_scores;claim_scores[key]=support
        pair_count+=rec['pairs']
    assert pair_count==1161 and len(claim_scores)==387
    old_scored={(p['response_id'],c['claim_index']) for p in lines(LEGACY/'pair_plans.jsonl') for c in p['claims']}
    assert len(old_scored)==2084 and not old_scored.intersection(claim_scores)
    claims=lines(SEM/'claims.jsonl');assert len(claims)==8852;values=np.zeros((8852,2),np.float64)
    for i,c in enumerate(claims):
        assert c['global_claim_index']==i;key=(c['response_id'],c['claim_index']);sid=c['inherited_source_id']
        assert (sid is not None)==(key in claim_scores)
        if sid is not None:
            assert sid in (1,2,3);s=claim_scores[key];values[i]=[max(s),max(s)-s[sid-1]]
    with np.load(SEM/'window_geometry.npz',allow_pickle=False) as z:
        ix=z['window_claim_indices'].copy();assert ix.shape==(NTOTAL,4)
        assert z['window_ids'].tolist()==[w['window_id'] for w in windows]
        assert z['response_ids'].tolist()==[w['response_id'] for w in windows]
    assert (ix>=-1).all() and (ix<len(values)).all() and (ix>=0).any(1).all()
    recomputed=np.array([values[j[j>=0]].mean(0) for j in ix],np.float32)
    lex=np.load(LEX/'window_features.npy',allow_pickle=False);scope=np.load(SCOPE/'window_features.npy',allow_pickle=False);semantic=np.load(SEM/'window_features.npy',allow_pickle=False)
    assert [x.shape for x in (lex,scope,semantic)]==[(NTOTAL,8),(NTOTAL,3),(NTOTAL,2)]
    assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in (lex,scope,semantic))
    assert np.array_equal(recomputed,semantic) and not semantic[scope[:,0]==0].any()
    assert read(SEM/'feature_names.json')==cfg['new_input_names']==['inherited_any_source_support_mean','inherited_source_support_gap_mean']
    verify(LEX/'window_features.npy',read(SCOPE/'geometry.json')['old8_features_sha256'])
    verify(SCOPE/'window_features.npy',read(SEM/'geometry.json')['old_scope_features_sha256'])
    # Historical13-column comparisons are bound, not used as matched controls.
    refs=read(OUT/'PREVIOUS_SCOPE_REFERENCES.json');assert refs==summary['previous_scope_references']
    historical=read(SCOPE/'summary.json');verify(SCOPE/'complete.json',refs['complete_sha256'])
    verify(SCOPE/'summary.json',read(SCOPE/'complete.json')['summary_sha256'])
    for peer in PEERS:
        assert refs['peers'][peer]['selected']==historical['selected'][peer]
        assert refs['peers'][peer]['all_candidates']==historical['all_candidates'][peer]
        for e in refs['peers'][peer]['all_candidates']:
            family=e['candidate'].split('__C')[0]
            for file,key in [(e['candidate']+'.pkl','model_sha256'),(e['candidate']+'_scores.npz','scores_sha256'),(family+'_scaler.pkl','scaler_sha256')]:verify(SCOPE/file,e[key])
    upstream=read(UP/'summary.json');verify(UP/'summary.json',read(UP/'complete.json')['summary_sha256'])
    family_reports={};metric_error=0.;coef_error=0.;thresholds=blocks=0
    for peer in PEERS:
        pair=[]
        for alpha in (0.,1.):
            e=next(e for e in upstream['all_candidates'][peer] if e['tail_weight']==alpha)
            p=UP/(e['candidate']+'_scores.npz');verify(p,e['scores_sha256'])
            with np.load(p,allow_pickle=False) as z:s=z['window_scores'].copy()
            assert s.shape==(NTOTAL,) and np.isfinite(s).all();pair.append(s)
        common=np.column_stack((*pair,lex,scope,semantic[:,0]));treatment=np.column_stack((common,semantic[:,1]))
        assert common.dtype==treatment.dtype==np.float64 and np.array_equal(treatment[:,:14],common)
        for mode,x in zip(MODES,(common,treatment)):
            width=x.shape[1];family=peer+'__'+mode;scaler_path=OUT/(family+'_scaler.pkl');sc=q.unpickle(scaler_path)
            fit=x[:NFIT];mass=bw.sum();mu=bw@fit/mass;center=fit-mu;correction=bw@center
            var=(bw@(center*center)-correction*correction/mass)/mass;eps=np.finfo(float).eps
            scale=np.sqrt(var);scale[var<=mass*eps*var+(mass*mu*eps)**2]=1.
            scaler_checks={'mean_max_abs':near(mu,sc.mean_,family+' mean',rtol=0,atol=2e-10),'var_max_abs':near(var,sc.var_,family+' variance',rtol=0,atol=2e-10),
                'scale_max_abs':near(scale,sc.scale_,family+' scale',rtol=0,atol=2e-10),'weighted_count_max_abs':near(mass,sc.n_samples_seen_,family+' mass',rtol=0,atol=1e-7)}
            assert sc.n_features_in_==width and sc.with_mean and sc.with_std
            scaled=(x-sc.mean_)/sc.scale_;entries=[];checks=[]
            for c in CS:
                name=family+f'__C{c:g}';e=read(OUT/(name+'.json'));model=q.unpickle(OUT/(name+'.pkl'))
                assert (e['peer'],e['mode'],e['C'],e['input_width'])==(peer,mode,c,width)
                verify(OUT/(name+'.pkl'),e['model_sha256']);verify(scaler_path,e['scaler_sha256']);verify(OUT/(name+'_scores.npz'),e['scores_sha256'])
                assert model.coef_.shape==(1,width) and model.intercept_.shape==(1,) and np.array_equal(model.classes_,[0,1])
                assert model.C==c and model.solver=='liblinear' and model.penalty=='l2' and model.max_iter==2000 and model.class_weight is None and model.random_state==20261010
                assert model.n_iter_.max()<2000 and model.n_iter_.tolist()==e['iterations']
                with np.load(OUT/(name+'_scores.npz'),allow_pickle=False) as z:s=z['window_scores'].copy();answer=z['answer_scores'].copy()
                assert s.shape==(NTOTAL,) and answer.shape==(793,) and np.isfinite(s).all()
                replay=q.sigmoid((scaled@model.coef_.T+model.intercept_).ravel());ce=near(replay,s,name+' coefficient',rtol=0,atol=2e-12);coef_error=max(coef_error,ce)
                expected_answer=np.array([s[indices[aa['answer_id']]].max() for aa in answers]);assert np.array_equal(answer,expected_answer)
                ts={'window':a.choose(y[NFIT:],s[NFIT:]),'answer':a.choose(ay[634:],answer[634:])};assert ts==e['thresholds'];thresholds+=2
                key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-c];assert key==e['selection_key']
                for part,wl,wh,al,ah in [('fit',0,NFIT,0,634),('calibration',NFIT,NTOTAL,634,793)]:
                    for unit,yy,ss,t in [('windows',y[wl:wh],s[wl:wh],ts['window']['threshold']),('answers',ay[al:ah],answer[al:ah],ts['answer']['threshold'])]:
                        got=a.metric(yy,ss,t);old=e['metrics'][part][unit];assert got.keys()==old.keys()
                        for k,v in got.items():
                            if isinstance(v,int):assert v==old[k]
                            else:metric_error=max(metric_error,near(v,old[k],(name,part,unit,k),rtol=0,atol=1e-12))
                        blocks+=1
                checks.append({'C':c,'coefficient_max_abs':ce,'stored_answermax_exact':True});entries.append(e)
            assert entries==summary['all_candidates'][family]
            selected=max(entries,key=lambda e:e['selection_key']);assert selected==summary['selected'][family]
            family_reports[family]={'width':width,'fit_only_scaler':scaler_checks,'candidates':checks,'selected_C':selected['C'],
                'cal_window_F1':selected['metrics']['calibration']['windows']['f1'],'cal_answer_F1':selected['metrics']['calibration']['answers']['f1']}
            print('HEADING_SOURCE_FAMILY_PASSED',family,flush=True)
            del scaled;gc.collect()
    assert thresholds==36 and blocks==72
    deltas={p:{'window_F1':family_reports[p+'__'+MODES[1]]['cal_window_F1']-family_reports[p+'__'+MODES[0]]['cal_window_F1'],
               'answer_F1':family_reports[p+'__'+MODES[1]]['cal_answer_F1']-family_reports[p+'__'+MODES[0]]['cal_answer_F1']} for p in PEERS}
    return {'status':'passed_with_provenance_limits','utc':datetime.now(timezone.utc).isoformat(),'blockers':[],
      'new_source_branch':{'answers':37,'claims':387,'pairs':1161,'disjoint_from_old2084_claims6252_pairs':True,'pair_softmax_max_abs':softmax_error,
        'all210364_new_window_feature_rows_exact':True,'all_common14_prefix_of_gap15_exact':True,'anchor_response_id':'16023','anchor_record':agreement,
        'anchor_was_hash_bound_not_reexecuted_by_auditor':True},
      'weights':weight_report,'families':family_reports,'matched_gap_minus_common_cal_F1':deltas,
      'coefficient_candidates':18,'scalers_checked':6,'max_coefficient_abs':coef_error,'thresholds_exact':thresholds,'selection_keys_exact':18,'selected_Cs_exact':6,
      'metric_blocks_checked':blocks,'max_metric_abs':metric_error,'window_scores_replayed':18*NTOTAL,'answermax_checks':18*793,
      'denominators':{'fit':{'answers':634,'positive_answers':328,'groups':615,'windows':NFIT,'positive_windows':21477},'calibration':{'answers':159,'positive_answers':100,'groups':154,'windows':42241,'positive_windows':5984}},
      'complete_sha256':sha(OUT/'complete.json'),'script_sha256':sha(__file__),'verified_files_sha256':HASHES,
      'no_fit_no_GPU':True,'production_modules_imported':False,'official_test_read':False,'frozen_outputs_modified':False,
      'limits':['Read-only replay of frozen matrices/coefficient artifacts cannot independently attest training history; fit-only ranges and weights also checked in hashed source.',
        'Original16023 GPU arithmetic comparison is validated through its frozen report/reference binding; no GPU arithmetic was rerun.',
        'Upstream fit scores are not cross-fitted and calibration has been repeatedly inspected; tiny selected deltas are development comparisons, not independent-test improvement.']}

if __name__=='__main__':
    assert not REPORT.exists(),'Do not overwrite a finalized independent report'
    try:
        with threadpool_limits(limits=4):report=audit()
    except Exception:
        (OUT/'INDEPENDENT_AUDIT.failure.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()},indent=2)+'\n',encoding='utf-8');raise
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['18个标题来源LR独立数值审计通过。六个fit-only scaler、18个系数、36阈值和6个C选择均核验；所有窗口和整答max分母保持。','',
        '新387 claims×3 sources=1161对与旧6252对无重叠；重算全部210364行新语义列精确。两组共享14列，gap组只多第15列；原16023数值anchor仅核绑定记录，未重跑GPU。','',
        '| peer | common窗口F1 | gap窗口F1 | 差值 | common整答F1 | gap整答F1 |','|---|---:|---:|---:|---:|---:|']
    for p in PEERS:
        c=report['families'][p+'__'+MODES[0]];g=report['families'][p+'__'+MODES[1]]
        md.append(f"| {p} | {c['cal_window_F1']:.10f} | {g['cal_window_F1']:.10f} | {g['cal_window_F1']-c['cal_window_F1']:+.10f} | {c['cal_answer_F1']:.10f} | {g['cal_answer_F1']:.10f} |")
    md+=['','fit634答/168123窗，cal159答/42241窗。以上仅为重复使用校准集的开发比较；HARP差值很小，不能解释为可靠泛化提升。未训练、GPU或读取官方test。']
    REPORT.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('HEADING_SOURCE_INDEPENDENT_AUDIT_PASSED',sha(REPORT),flush=True)
