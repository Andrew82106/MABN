"""Audit new auxiliary supervision, keeping original risk metrics and frozen baseline."""
from pathlib import Path
from collections import defaultdict
import importlib.util
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import run_semantic_multitask as run
qa=run.qa
spec=importlib.util.spec_from_file_location('fixed_threshold_helpers',run.OLD/'audit_sequence.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def audit():
    clock=time.perf_counter();torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    meta,index,snapshot=run.source();assert snapshot==qa.read(OUT/'source_snapshot.json')
    for name,expected in qa.read(OUT/'complete.json')['files_sha256'].items():assert qa.sha(OUT/name)==expected
    with np.load(run.OLD/'fit_token_weights.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    with np.load(OUT/'auxiliary_labels_weights.npz',allow_pickle=False) as z:ay,aw,factors=z['y'],z['fit_loss_weights'],z['class_factors']
    lexical=np.concatenate([np.asarray(t['lexical_mask'],bool) for t in meta['tokens']]);yt=np.concatenate([np.asarray(t['risk_mask'],int) for t in meta['tokens']])
    assert np.array_equal(ay.any(1),yt.astype(bool)) and not ay[~lexical].any() and not aw[~lexical[:170361]].any()
    groups=defaultdict(list)
    for a in index[:634]:groups[a['group_id']].extend(range(a['left'],a['right']))
    base=old['base'];reconstructed=np.zeros_like(aw)
    for k in range(2):
        # Independently sum positive/negative base mass, then group-normalize.
        p=ay[:170361,k].astype(bool);m=np.asarray([base[~p].sum(),base[p].sum()]);f=m.sum()/(2*m)
        assert np.allclose(f,factors[k],atol=1e-10,rtol=1e-10)
        raw=base*np.where(p,f[1],f[0])
        for ix in groups.values():reconstructed[ix,k]=raw[ix]/raw[ix].sum()*(139518/615)
        reconstructed[:,k]*=139518/reconstructed[:,k].sum()
        assert np.allclose(reconstructed[:,k],aw[:,k],atol=1e-10,rtol=1e-10)
        assert np.allclose([aw[ix,k].sum() for ix in groups.values()],139518/615,atol=1e-10,rtol=1e-10)
    # Main weights and standardized inputs are reused without any mutation or refitting.
    x=np.load(run.BASE/'standardized_token_features.npy',mmap_mode='r');starts={a['response_id']:a['left'] for a in index}
    wi=np.asarray([[starts[w['response_id']]+k for k in w['token_indices']] for w in meta['windows']]);wl=lexical[wi]
    yw=np.asarray([w['label'] for w in meta['windows']]);ya=np.asarray([a['label'] for a in meta['answers']]);summary=qa.read(OUT/'summary.json');choices=[]
    for epoch in range(1,31):
        entry=qa.read(OUT/run.METHOD/f'epoch_{epoch:03d}.json')
        with np.load(OUT/run.METHOD/f'epoch_{epoch:03d}_scores.npz',allow_pickle=False) as z:
            l,p,a,s,aux=(z[k] for k in ('token_logits','token_scores','answer_scores','window_scores','auxiliary_token_logits'))
        assert np.array_equal(torch.sigmoid(torch.from_numpy(l)).numpy(),p)
        assert np.array_equal(np.where(wl,p[wi],-np.inf).max(1).astype(np.float64),s)
        assert np.array_equal(np.asarray([s[ix].max() for ix in meta['answer_windows'].values()]),a)
        tw=helper.threshold(yw[168123:],s[168123:]);ta=helper.threshold(ya[634:],a[634:])
        for level,result in (('window',tw),('answer',ta)):
            v=entry['thresholds'][level];assert result==(v['f1'],v['precision'],v['threshold'])
        for part,wix,aix in (('fit',slice(0,168123),slice(0,634)),('calibration',slice(168123,None),slice(634,None))):
            for unit,y,score,cutoff in (('windows',yw[wix],s[wix],tw[2]),('answers',ya[aix],a[aix],ta[2])):
                for key,value in helper.counts(y,score,cutoff).items():assert value==entry['metrics'][part][unit][key]
        fl=l[:170361].astype(np.float64);risk_bce=float(old['loss']@(np.logaddexp(0,fl)-yt[:170361]*fl)/139518)
        aux_bce=[]
        for k in range(2):
            al=aux[:170361,k].astype(np.float64);aux_bce.append(float(aw[:,k]@(np.logaddexp(0,al)-ay[:170361,k]*al)/139518))
        assert risk_bce==entry['full_fit_weighted_BCE'] and np.allclose(aux_bce,entry['full_fit_auxiliary_BCE'],atol=1e-12,rtol=1e-12)
        assert abs(risk_bce+.25*np.mean(aux_bce)-entry['full_fit_combined_loss'])<1e-12
        choices.append(([min(tw[0],ta[0]),tw[0],tw[1],-epoch],epoch))
    epoch=max(choices)[1];assert epoch==summary['selected'][run.METHOD]['epoch']
    model=run.new_model();checkpoint=torch.load(OUT/run.METHOD/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']);l,p,aux=run.predict(model,x,index)
    with np.load(OUT/run.METHOD/f'epoch_{epoch:03d}_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(l,z['token_logits']) and np.array_equal(p,z['token_scores']) and np.array_equal(aux,z['auxiliary_token_logits'])
    assert run.source()[-1]==snapshot and not torch.cuda.is_initialized()
    result={'status':'passed','code_sha256':qa.sha(Path(__file__)),'complete_sha256':qa.sha(OUT/'complete.json'),
        'all_frozen_input_and_output_hashes_verified':True,'binary_risk_labels_and_main_weights_unchanged':True,
        'auxiliary_fit_only_factors_and615_group_weights_independently_checked':True,'nonlexical_and_padding_excluded_from_losses':True,
        'all30_epoch_main_only_aggregation_threshold_counts_exact':True,'all30_primary_auxiliary_combined_losses_checked':True,
        'selected_epoch':epoch,'full213159_main_and_auxiliary_checkpoint_predictions_exact':True,
        'no_PCA_scaler_or_baseline_refit':True,'test_opened':False,'GPU_used':False,'seconds':time.perf_counter()-clock}
    qa.save(OUT/'AUDIT.json',result);print('SEMANTIC_MULTITASK_AUDIT_PASSED',round(result['seconds'],2),flush=True)


def report():
    meta=qa.metadata();summary=qa.read(OUT/'summary.json');entry=summary['selected'][run.METHOD];history=summary['all_epochs'][run.METHOD]
    prior=qa.read(run.BASE/'summary.json');original=prior['selected'][run.baseline.METHOD];oldhistory=prior['all_epochs'][run.baseline.METHOD]
    with np.load(OUT/run.METHOD/f"epoch_{entry['epoch']:03d}_scores.npz",allow_pickle=False) as z:s=z['window_scores']
    alert=s>=entry['thresholds']['window']['threshold'];covered={}
    for w,flag in zip(meta['windows'],alert):
        if flag:covered.setdefault(w['response_id'],set()).update(w['token_indices'])
    typed={}
    for kind in run.TYPE:
        spans=[];risk={};empty=0
        for t in meta['tokens']:
            if t['partition']!='calibration':continue
            rid=t['response_id']
            for mapping in t['span_token_mapping']:
                if t['original_labels'][mapping['span_index']]['label_type']!=kind:continue
                pos=set(mapping['risk_token_indices'])
                if not pos:empty+=1;continue
                risk.setdefault(rid,set()).update(pos);spans.append((bool(pos&covered.get(rid,set())),pos<=covered.get(rid,set())))
        ix=[i for i,w in enumerate(meta['windows']) if w['partition']=='calibration' and bool(set(w['token_indices'])&risk.get(w['response_id'],set()))]
        typed[kind]={'original_spans':len(spans)+empty,'localizable_spans':len(spans),'unlocalizable_spans':empty,
            'span_any_hit':sum(a for a,b in spans),'span_full_hit':sum(b for a,b in spans),'positive_windows':len(ix),
            'hit_positive_windows':int(alert[ix].sum()),'positive_window_recall':float(alert[ix].mean())}
    baseline_diag=qa.read(run.BASE/'DIAGNOSTICS.json')['calibration_error_types'];baseline_diag[run.METHOD]=typed
    comparison=qa.read(run.BASE/'BASELINE_COMPARISON.json')['entries'];comparison.append({'candidate':run.METHOD,'kind':'extra_checker_auxiliary_type_sequence','metrics':entry['metrics']})
    qa.save(OUT/'BASELINE_COMPARISON.json',{'entries':comparison,'same_geometry_labels_and_main_inference':True,'test_opened':False})
    qa.save(OUT/'DIAGNOSTICS.json',{'calibration_error_types':baseline_diag,'global_total_risk_threshold_only':True,
        'fit_aux_counts':qa.read(OUT/'LABEL_REVIEW.json')['counts'],'all30_curve':history,'test_opened':False})
    lines=['仅新增两个辅助类型头：训练区分无依据/与资料冲突；所有推理和评测仍只用原总风险头。额外MiniCheck与原1090输入未变。','',
        '| 模型 | 选中轮次 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,e in ((run.baseline.METHOD,original),(run.METHOD,entry)):
        m=e['metrics'];w=m['calibration']['windows'];lines.append(f"| {name} | {e['epoch']} | {m['fit']['windows']['f1']:.3f} | {w['precision']:.3f} | {w['recall']:.3f} | {w['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    lines+=['','| 模型 | 类型 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |','|---|---|---:|---:|---:|']
    for name in (run.baseline.METHOD,run.METHOD):
        for kind,d in baseline_diag[name].items():
            lines.append(f"| {name} | {kind} | {d['hit_positive_windows']}/{d['positive_windows']} | {d['span_any_hit']}/{d['localizable_spans']} | {d['span_full_hit']}/{d['localizable_spans']} |")
    lines+=['','| 模型 | 所选轮risk BCE | 最后轮risk BCE | 最后轮cal窗F1 |','|---|---:|---:|---:|']
    for name,e,hist in ((run.baseline.METHOD,original,oldhistory),(run.METHOD,entry,history)):
        lines.append(f"| {name} | {e['full_fit_weighted_BCE']:.4f} | {hist[-1]['full_fit_weighted_BCE']:.4f} | {hist[-1]['metrics']['calibration']['windows']['f1']:.3f} |")
    lines+=['',f"辅助头所选轮fit BCE（无依据、冲突）={entry['full_fit_auxiliary_BCE']}；最后轮={history[-1]['full_fit_auxiliary_BCE']}。",'',
        '主干/risk初始化与原模型逐参数及训练模式dropout输出一致，辅助RNG隔离；同30次shuffle、同优化器、同风险权重，辅助系数0.25固定，未按类型选择checkpoint。所有旧基线保留。',
        '原人工risk标签与4原始BPE窗口完全不变。两个辅助标签可重叠，OR与risk逐词元一致；类权重只用fit统计，非lexical loss为0。',
        '这是单seed、cal选轮/选阈值后的开发结果，test未打开。辅助监督也改变优化和正则效应，不能单凭这次结果证明唯一错误原因。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8');print('SEMANTIC_MULTITASK_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):audit();report()
