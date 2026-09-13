"""Independent type-loss audit; no fitting, GPU or test access."""
from pathlib import Path
from collections import defaultdict,Counter
import importlib.util,pickle,sys,json,time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;EXP=OUT.parent;sys.path.insert(0,str(EXP))
import run_probe_expansion as base
q=base.q
spec=importlib.util.spec_from_file_location('independent_expansion_helpers',base.OUT/'audit_probe_expansion.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
TYPE={'Evident Baseless Info':0,'Subtle Baseless Info':0,'Evident Conflict':1,'Subtle Conflict':1}

def main():
    tick=time.perf_counter();done=q.read(OUT/'complete.json');cfg=q.read(OUT/'protocol.json');summary=q.read(OUT/'summary.json')
    for n,h in done['files_sha256'].items():assert q.sha(OUT/n)==h,n
    for n,h in q.read(base.OUT/'complete.json')['files_sha256'].items():assert q.sha(base.OUT/n)==h,n
    assert cfg['code_sha256']==q.sha(EXP/'run_type_balanced_probe.py') and cfg['source_complete_sha256']==q.sha(base.OUT/'complete.json')
    prep=q.read(OUT/'preparation_complete.json');assert prep['protocol_sha256']==q.sha(OUT/'protocol.json') and prep['weights_sha256']==q.sha(OUT/'weights.npz')
    assert q.read(OUT/'fit_started.json')['preparation_sha256']==q.sha(OUT/'preparation_complete.json')
    original,meta=base.metadata();aa,tt,ww=meta['answers'],meta['tokens'],meta['windows']
    with np.load(OUT/'weights.npz') as z:w={k:z[k].copy() for k in z.files}
    with np.load(base.OUT/'expanded3680_weights.npz') as z:oldw={k:z[k].copy() for k in z.files}
    assert np.array_equal(w['base'],oldw['base']) and np.array_equal(w['y'],oldw['y'])
    assert np.array_equal(w['y'],[r['label'] for r in ww[:653979]])
    token_types={};counts=Counter();rawcount=0
    for a,t in zip(aa[:3680],tt[:3680]):
        text=t['original_response'];chars=np.zeros((len(text),2),bool);alnum=np.fromiter((c.isalnum() for c in text),bool)
        assert len(t['original_labels'])==len(t['span_token_mapping'])
        for span in t['original_labels']:
            s,e=span['start'],span['end'];assert text[s:e]==span['text']
            chars[s:e,TYPE[span['label_type']]]|=alnum[s:e];counts[span['label_type']]+=1
        found=np.asarray([chars[int(l):int(r)].any(0) for l,r in t['response_token_offsets']],bool)
        assert np.array_equal(found.any(1),t['risk_mask'])
        assert not np.any(found[~np.asarray(t['lexical_mask'],bool)])
        token_types[a['response_id']]=found;rawcount+=len(found)
    mix=np.zeros((653979,3),np.float64);groups=defaultdict(list)
    for i,row in enumerate(ww[:653979]):
        types=np.flatnonzero(token_types[row['response_id']][row['token_indices']].any(0))
        if len(types):mix[i,types+1]=1/len(types)
        else:mix[i,0]=1
        groups[row['group_id']].append(i)
    assert np.array_equal(mix,w['mix']) and np.array_equal(mix[:,0]==0,w['y']==1)
    mass=np.sum(w['base'][:,None]*mix,axis=0);factors=np.asarray([.5,.25,.25])*w['base'].sum()/mass
    assert np.array_equal(factors,w['type_factors']);loss=w['base']*np.sum(mix*factors,axis=1)
    for ix in groups.values():loss[ix]*=(168123/615)/loss[ix].sum()
    losserr=audit.near(loss,w['loss'],1e-10);assert abs(loss.sum()-168123)<1e-6
    check=q.read(OUT/'LABEL_WEIGHT_CHECK.json');assert dict(counts)==check['original_human_span_counts']
    audit.near((loss[:,None]*mix).sum(0),check['new_loss_type_mass_after_group_renormalization'],1e-8)
    raw=np.load(base.OUT/'matrices/window65.npy',mmap_mode='r');assert raw.shape==(696220,65)
    y=np.asarray([r['label'] for r in ww]);ya=np.asarray([r['label'] for r in aa]);oldsummary=q.read(base.OUT/'summary.json')
    records=[];selected={};maxscore=maxmetric=0.;scorecount=answercount=0
    for method,family in summary['results'].items():
        width=64 if method=='hidden64' else 65
        template=pickle.loads((base.OUT/f'expanded3680_{method}_C1e-05.pkl').read_bytes());sc=template['scaler']
        for entry in family['all_candidates']:
            c=entry['C'];name=entry['candidate'];assert entry==q.read(OUT/(name+'_result.json'))
            for ext,h in entry['files_sha256'].items():assert q.sha(OUT/(name+ext))==h
            obj=pickle.loads((OUT/(name+'.pkl')).read_bytes());model=obj['model']
            for attr in ('mean_','var_','scale_','n_samples_seen_'):assert np.array_equal(getattr(sc,attr),getattr(obj['scaler'],attr))
            assert obj['C']==model.C==c and obj['fit_only'] and obj['weight_sha256']==q.sha(OUT/'weights.npz')
            assert model.solver=='liblinear' and model.penalty=='l2' and model.max_iter==2000 and model.random_state==20260924
            assert max(model.n_iter_)<2000 and model.coef_.shape==(1,width)
            replay=np.empty(len(raw),np.float64)
            for l in range(0,len(raw),16384):
                r=min(l+16384,len(raw));z=np.array(raw[l:r,:width],np.float32,copy=True);z-=sc.mean_;z/=sc.scale_
                replay[l:r]=expit((z@model.coef_.T+model.intercept_).ravel())
            with np.load(OUT/(name+'_scores.npz')) as z:s=z['window_scores'];ans=z['answer_scores']
            err=audit.near(replay,s,1e-12);maxscore=max(err,maxscore);scorecount+=len(s)
            assert np.array_equal(ans,np.asarray([s[meta['answer_windows'][a['response_id']]].max() for a in aa]));answercount+=len(ans)
            ts={'window':audit.choose(y[653979:],s[653979:]),'answer':audit.choose(ya[3680:],ans[3680:])}
            assert ts==entry['thresholds']==obj['thresholds'] and audit.key(ts,c)==entry['selection_key']
            metrics={}
            for part,(l,r,al,ar) in {'fit':(0,653979,0,3680),'calibration':(653979,len(s),3680,len(ans))}.items():
                metrics[part]={'windows':audit.count(y[l:r],s[l:r],ts['window']['threshold']),'answers':audit.count(ya[al:ar],ans[al:ar],ts['answer']['threshold'])}
            me=audit.compare_dict(metrics,entry['metrics']);maxmetric=max(maxmetric,me)
            records.append({'candidate':name,'scaler_same_as_binary_loss_exact':True,'score_max_abs':err,'thresholds_exact':True,'metrics_max_abs':me})
        best=max(family['all_candidates'],key=lambda x:audit.key(x['thresholds'],x['C']));assert best==family['selected']
        old=oldsummary['selected']['expanded3680_'+method]
        selected[method]={'new_C':best['C'],'new_calibration':best['metrics']['calibration'],'old_binary_C':old['C'],'old_binary_calibration':old['metrics']['calibration']}
    report={'status':'passed','binary_labels_and_original_base_weights_exact':True,'type_geometry_independently_rebuilt_from_alnum_char_overlap':True,'fit_raw_tokens_checked':rawcount,'fit_windows':653979,'calibration_windows':42241,'fit_answers':3680,'calibration_answers':159,'type_weighting_calibration_inputs':0,'types_are_not_predictor_features':True,'classes':['negative','baseless','conflict'],'mixed_windows':int(np.sum((mix[:,1]>0)&(mix[:,2]>0))),'type_mass_before':mass.tolist(),'type_factors':factors.tolist(),'loss_mass':float(loss.sum()),'type_mass_after_group_renormalization':(loss[:,None]*mix).sum(0).tolist(),'loss_max_abs':losserr,'group_loss_equal':True,'frozen_PCA_and_scalers_reused':True,'coefficient_values_checked':scorecount,'answer_max_checks':answercount,'score_max_abs':maxscore,'metric_max_abs':maxmetric,'thresholds_exact':12,'C_selections_exact':2,'candidate_checks':records,'selected':selected,'official_test_read':False,'reviewer_fits':0,'reviewer_GPU':False,'seconds':time.perf_counter()-tick,'production_complete_sha256':q.sha(OUT/'complete.json'),'source_complete_sha256':q.sha(base.OUT/'complete.json'),'helper_sha256':q.sha(__file__),'limitations':['Type50/25/25 target applies before restoring equal source-group loss; final class totals need not retain that ratio.','This is already-used development calibration, not independent test.','PCA/scaler/source mapping reuse is inherited from the completed full expansion audit; no additional model fine-tuning or GPU audit was run.']}
    q.save(OUT/'INDEPENDENT_AUDIT_TYPE_BALANCED.json',report)
    md=['类型重加权诊断独立审计通过，无阻断。','',f'直接从人工span与原答案字符重建类型，覆盖{rawcount}个训练词元及653979窗口，二分类标签完全不变。类型只参与fit损失权重；PCA、scaler、特征及校准规则保持不变。',
        '6模型、12阈值和2次选型回放通过；总loss仍为168123。组等权归一后，三类质量不必严格保持50/25/25。','', '| 输入 | 原窗口F1 | 类型重加权窗口F1 | 类型重加权整答F1 |','|---|---:|---:|---:|']
    for k,v in selected.items():md.append(f"| {k} | {v['old_binary_calibration']['windows']['f1']:.6f} | {v['new_calibration']['windows']['f1']:.6f} | {v['new_calibration']['answers']['f1']:.6f} |")
    md+=['','定位没有改善，保留为负结果；不替换原模型。审计没有拟合、GPU或测试读取。']
    (OUT/'INDEPENDENT_AUDIT_TYPE_BALANCED.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('TYPE_BALANCED_AUDIT_PASS',q.sha(OUT/'INDEPENDENT_AUDIT_TYPE_BALANCED.json'),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
