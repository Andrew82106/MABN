"""Audit only the new semantic TCN; no refitting or repeated upstream extraction."""
from pathlib import Path
import importlib.util
import pickle
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import run_semantic_sequence as run
qa=run.qa
spec=importlib.util.spec_from_file_location('independent_threshold_helpers',run.OLD/'audit_sequence.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def audit():
    clock=time.perf_counter();torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    meta,index,snapshot=run.source();assert snapshot==qa.read(OUT/'source_snapshot.json')
    for manifest in ('complete.json','preparation_complete.json'):
        for name,expected in qa.read(OUT/manifest)['files_sha256'].items():assert qa.sha(OUT/name)==expected
    assert qa.read(run.SEM/'AUDIT.json')['status']=='passed'
    common=np.load(run.FULL/'matrices/common1025.npy',mmap_mode='r')
    h=np.load(run.SEM/'matrices/token_hidden64.npy',mmap_mode='r');r=np.load(run.SEM/'matrices/token_risk_logit.npy',mmap_mode='r')
    normalized=np.load(OUT/'standardized_token_features.npy',mmap_mode='r');scaler=pickle.loads((OUT/'scaler.pkl').read_bytes())
    with np.load(run.OLD/'fit_token_weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    mu=np.zeros(1090);var=np.zeros(1090);mass=0.
    for left in range(0,170361,16384):
        right=min(left+16384,170361);x=np.column_stack((common[left:right],h[left:right],r[left:right,None])).astype(np.float64)
        w=weights['base'][left:right].astype(np.float32).astype(np.float64);old=float(np.float32(mass));new=float(w.sum())
        m=w@x/new;xc=x-m;v=np.einsum('i,ij,ij->j',w,xc,xc)/new;total=old+new;d=mu-m
        var=(old*var+new*v+d*d*old*new/total)/total;mu=(old*mu+new*m)/total;mass=total
    assert np.allclose(mu,scaler.mean_,atol=2e-10,rtol=2e-10) and np.allclose(var,scaler.var_,atol=2e-10,rtol=2e-10)
    for left in range(0,len(normalized),16384):
        right=min(left+16384,len(normalized));x=np.column_stack((common[left:right],h[left:right],r[left:right,None]))
        assert np.array_equal(scaler.transform(x).astype(np.float32),normalized[left:right])
    starts={a['response_id']:a['left'] for a in index}
    lexical=np.concatenate([np.asarray(t['lexical_mask'],bool) for t in meta['tokens']])
    yt=np.concatenate([np.asarray(t['risk_mask'],int) for t in meta['tokens']])
    ix=np.asarray([[starts[w['response_id']]+k for k in w['token_indices']] for w in meta['windows']]);lex=lexical[ix]
    yw=np.asarray([w['label'] for w in meta['windows']]);ya=np.asarray([a['label'] for a in meta['answers']])
    assert list(meta['answer_windows'])==[a['response_id'] for a in meta['answers']]
    summary=qa.read(OUT/'summary.json');choices=[]
    for epoch in range(1,31):
        entry=qa.read(OUT/run.METHOD/f'epoch_{epoch:03d}.json')
        with np.load(OUT/run.METHOD/f'epoch_{epoch:03d}_scores.npz',allow_pickle=False) as z:
            p,l,s,a=(z[k] for k in ('token_scores','token_logits','window_scores','answer_scores'))
        assert np.array_equal(torch.sigmoid(torch.from_numpy(l)).numpy(),p)
        assert np.array_equal(np.where(lex,p[ix],-np.inf).max(1).astype(np.float64),s)
        assert np.array_equal(np.asarray([s[inds].max() for inds in meta['answer_windows'].values()]),a)
        tw=helper.threshold(yw[168123:],s[168123:]);ta=helper.threshold(ya[634:],a[634:])
        for level,result in (('window',tw),('answer',ta)):
            saved=entry['thresholds'][level];assert result==(saved['f1'],saved['precision'],saved['threshold'])
        for part,wi,ai in (('fit',slice(0,168123),slice(0,634)),('calibration',slice(168123,None),slice(634,None))):
            for unit,y,score,cutoff in (('windows',yw[wi],s[wi],tw[2]),('answers',ya[ai],a[ai],ta[2])):
                for key,value in helper.counts(y,score,cutoff).items():assert value==entry['metrics'][part][unit][key]
        fl=l[:170361].astype(np.float64);bce=float(weights['loss']@(np.logaddexp(0,fl)-yt[:170361]*fl)/139518)
        assert bce==entry['full_fit_weighted_BCE'];choices.append(([min(tw[0],ta[0]),tw[0],tw[1],-epoch],epoch))
    epoch=max(choices)[1];assert epoch==summary['selected'][run.METHOD]['epoch']
    model=run.new_model();checkpoint=torch.load(OUT/run.METHOD/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']);logits,p=run.predict(model,normalized,index)
    with np.load(OUT/run.METHOD/f'epoch_{epoch:03d}_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(logits,z['token_logits']) and np.array_equal(p,z['token_scores'])
    assert not torch.cuda.is_initialized()
    result={'status':'passed','code_sha256':qa.sha(Path(__file__)),'complete_sha256':qa.sha(OUT/'complete.json'),
        'source_hashes_and_exact1090_input_order_verified':True,'fit_only_scaler_moments_checked':True,
        'full213159_normalized_token_input_exact':True,'all30_epoch_aggregation_threshold_counts_BCE_exact':True,
        'selected_epoch':epoch,'selected_full213159token_checkpoint_reload_exact':True,
        'old_PCA_mapping_and_model_extraction_not_repeated':True,'test_opened':False,'GPU_used':False,'seconds':time.perf_counter()-clock}
    qa.save(OUT/'AUDIT.json',result);print('SEMANTIC_SEQUENCE_AUDIT_PASSED',round(result['seconds'],2),flush=True)


def report():
    meta=qa.metadata();summary=qa.read(OUT/'summary.json');entry=summary['selected'][run.METHOD]
    with np.load(OUT/run.METHOD/f"epoch_{entry['epoch']:03d}_scores.npz",allow_pickle=False) as z:scores=z['window_scores']
    alerts=scores>=entry['thresholds']['window']['threshold'];covered={}
    for w,alert in zip(meta['windows'],alerts):
        if alert:covered.setdefault(w['response_id'],set()).update(w['token_indices'])
    typed={}
    for kind in ('Evident Conflict','Subtle Conflict','Evident Baseless Info','Subtle Baseless Info'):
        spans=[];risk={};empty=0
        for tokens in meta['tokens']:
            if tokens['partition']!='calibration':continue
            rid=tokens['response_id']
            for mapping in tokens['span_token_mapping']:
                if tokens['original_labels'][mapping['span_index']]['label_type']!=kind:continue
                pos=set(mapping['risk_token_indices'])
                if not pos:empty+=1;continue
                risk.setdefault(rid,set()).update(pos);spans.append((bool(pos&covered.get(rid,set())),pos<=covered.get(rid,set())))
        positive=[j for j,w in enumerate(meta['windows']) if w['partition']=='calibration' and bool(set(w['token_indices'])&risk.get(w['response_id'],set()))]
        typed[kind]={'original_spans':len(spans)+empty,'localizable_spans':len(spans),'unlocalizable_spans':empty,
            'span_any_hit':sum(a for a,b in spans),'span_full_hit':sum(b for a,b in spans),'positive_windows':len(positive),
            'hit_positive_windows':int(alerts[positive].sum()),'positive_window_recall':float(alerts[positive].mean())}
    comparison=qa.read(run.SEM/'BASELINE_COMPARISON.json')['entries']
    comparison.append({'candidate':run.METHOD,'kind':'extra_checker_sequence','metrics':entry['metrics']})
    qa.save(OUT/'BASELINE_COMPARISON.json',{'entries':comparison,'same_geometry_and_labels':True,'test_opened':False})
    old_diag=qa.read(run.SEM/'DIAGNOSTICS.json')['calibration_error_types'];old_diag[run.METHOD]=typed
    history=summary['all_epochs'][run.METHOD]
    qa.save(OUT/'DIAGNOSTICS.json',{'calibration_error_types':old_diag,'global_threshold_no_type_tuning':True,
        'selected_fit_minus_cal_window_F1':entry['metrics']['fit']['windows']['f1']-entry['metrics']['calibration']['windows']['f1'],
        'selected_fit_BCE':entry['full_fit_weighted_BCE'],'last_fit_BCE':history[-1]['full_fit_weighted_BCE'],
        'last_cal_window_F1':history[-1]['metrics']['calibration']['windows']['f1'],'test_opened':False})
    rows=[(run.METHOD,entry['metrics'])]
    for _,e in qa.read(run.SEM/'summary.json')['selected'].items():rows.append((e['candidate'],e['metrics']))
    for name in ('minicheck_calibrated','lookback_minicheck_fusion'):
        e=qa.read(run.semantic.SEM/'results/summary.json')['reports'][name];rows.append((name,e['metrics']))
    for name,e in qa.read(run.FULL/'summary.json')['selected'].items():rows.append((name,e['metrics']))
    lines=['固定1090维、宽32序列融合结果。MiniCheck额外核查模型读取整句和资料，此方法是离线融合，全部为cal选轮/选阈值后的开发成绩。','',
        '| 方法 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|']
    for name,m in rows:
        w=m['calibration']['windows'];lines.append(f"| {name} | {m['fit']['windows']['f1']:.3f} | {w['precision']:.3f} | {w['recall']:.3f} | {w['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    lines+=['',f"选择第{entry['epoch']}轮；共固定30轮。该轮fit加权BCE={entry['full_fit_weighted_BCE']:.4f}，第30轮={history[-1]['full_fit_weighted_BCE']:.4f}；第30轮cal窗口F1={history[-1]['metrics']['calibration']['windows']['f1']:.3f}。",'',
        '| 类型 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |','|---|---:|---:|---:|']
    for kind,d in typed.items():lines.append(f"| {kind} | {d['hit_positive_windows']}/{d['positive_windows']} | {d['span_any_hit']}/{d['localizable_spans']} | {d['span_full_hit']}/{d['localizable_spans']} |")
    lines+=['','原始标签、4原始BPE窗口及全部评测分母未变。PCA未重拟合，模型与scaler仅用fit，单seed，无test访问；所有旧基线和9LR保存在BASELINE_COMPARISON.json。',
        '窗口大量重叠，不能当作独立数据；全量统计提升仍须在未打开的测试集确认。Subtle Conflict仅5个cal span。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8');print('SEMANTIC_SEQUENCE_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):audit();report()
