"""Read-only arithmetic audit and fixed comparisons for semantic hidden LR."""
from collections import Counter
import importlib.util
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_semantic_hidden as run
qa = run.qa
spec = importlib.util.spec_from_file_location('independent_counts', ROOT/'results/sequence_v1/audit_sequence.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def audit():
    started = time.perf_counter(); meta, plans, records, snapshot = run.source()
    assert snapshot == qa.read(OUT/'source_snapshot.json')
    for name in ('complete.json', 'preparation_complete.json'):
        for relative, expected in qa.read(OUT/name)['files_sha256'].items(): assert qa.sha(OUT/relative) == expected
    index = qa.read(OUT/'token_index.json')['answers']; starts = {a['response_id']:a['left'] for a in index}
    mapped = np.load(OUT/'matrices/mapped_hidden1024.npy',mmap_mode='r')
    risk = np.load(OUT/'matrices/token_risk.npy'); rlogit = np.load(OUT/'matrices/token_risk_logit.npy')
    assert np.array_equal(run.logit(risk).astype(np.float32),rlogit)
    mapping_error = 0.; lexical = np.zeros(len(risk),bool)
    # Independent sparse character-to-source-token weights, rather than materializing character state vectors.
    for answer in meta['answers']:
        rid = answer['response_id']; tokens = meta['by_response'][rid]['tokens']; text = answer['original_response']
        arr = run.load_claim(answer,plans[rid],records[rid]); owners=[[] for _ in text]; crisk=np.zeros(len(text),np.float64)
        for j,(a,b) in enumerate(zip(arr['token_start'],arr['token_end'])):
            for char in range(a,b):
                if not text[char].isspace(): owners[char].append(j)
        for claim,support in zip(plans[rid]['claims'],arr['selected_support_per_claim']):
            for char in range(claim['start'],claim['end']):
                if text[char].isalnum(): crisk[char]=max(crisk[char],1-float(support))
        assert all(owners[j] for j,ch in enumerate(text) if not ch.isspace())
        for token,(a,b) in enumerate(tokens['response_token_offsets']):
            pos=starts[rid]+token; chars=[c for c in range(a,b) if not text[c].isspace()]
            alnum=[c for c in range(a,b) if text[c].isalnum()]; lexical[pos]=bool(alnum)
            assert lexical[pos] == tokens['lexical_mask'][token]
            expected_risk=float(crisk[alnum].max()) if alnum else 0.; assert expected_risk == risk[pos]
            coefficient=Counter()
            for char in chars:
                for source in owners[char]: coefficient[source]+=1/(len(chars)*len(owners[char]))
            expected=np.zeros(1024,np.float64)
            for source,weight in coefficient.items(): expected+=arr['hidden_last'][source].astype(np.float64)*weight
            expected=expected.astype(np.float32); actual=np.asarray(mapped[pos])
            mapping_error=max(mapping_error,float(np.abs(expected-actual).max()))
            assert np.allclose(expected,actual,atol=2e-6,rtol=1e-6)
    pca=pickle.loads((OUT/'hidden_pca.pkl').read_bytes()); oldpca=pickle.loads((run.OLD/'hidden_pca.pkl').read_bytes())
    assert pca['sample']==oldpca['sample'] and np.array_equal(pca['sample_weights'],oldpca['sample_weights'])
    assert {a['partition'] for a in index[:634]}=={'fit'} and len({a['group_id'] for a in index[:634]})==615
    sample_index=np.asarray([starts[s['response_id']]+s['token_index'] for s in pca['sample']])
    sample=np.asarray(mapped[sample_index],np.float64); w=pca['sample_weights']; mean=w@sample
    assert np.array_equal(mean,pca['mean']); centered=sample-mean
    components=pca['components']; orth=float(np.abs(components@components.T-np.eye(64)).max()); assert orth<1e-10
    projection=centered@components.T; trace=float(np.einsum('ij,i,ij->',centered,w,centered))
    ratio=float(np.einsum('ij,i,ij->',projection,w,projection)/trace)
    svd_ratio=float(np.sum(pca['singular_values']**2)/trace)
    assert svd_ratio==pca['explained_variance_ratio_sum'] and 0 <= svd_ratio <= ratio+1e-10 <= 1+1e-10
    projected=np.load(OUT/'matrices/token_hidden64.npy',mmap_mode='r')
    for left in range(0,len(mapped),16384):
        right=min(left+16384,len(mapped))
        assert np.array_equal(((mapped[left:right].astype(np.float64)-mean)@components.T).astype(np.float32),projected[left:right])
    wi=np.asarray([[starts[win['response_id']]+k for k in win['token_indices']] for win in meta['windows']])
    window_hidden=np.load(OUT/'matrices/window_hidden64.npy'); window_logit=np.load(OUT/'matrices/window_risk_logit.npy')
    assert np.array_equal(projected[wi].mean(1),window_hidden)
    semantic_window=np.where(lexical[wi],risk[wi],-np.inf).max(1)
    assert np.array_equal(run.logit(semantic_window).astype(np.float32),window_logit)
    # Comparison with independently implemented official scalar mapping, when upstream baseline is complete.
    semantic_path=run.SEM/'results/minicheck_calibrated_scores.npz'
    assert semantic_path.exists()
    with np.load(semantic_path,allow_pickle=False) as z: assert np.array_equal(semantic_window,z['window_scores'])
    base=np.load(run.OLD/'matrices/base.npy',mmap_mode='r')
    with np.load(run.OLD/'training_weights.npz',allow_pickle=False) as z: weights={k:z[k].copy() for k in z.files}
    yw=np.asarray([win['label'] for win in meta['windows']]); ya=np.asarray([a['label'] for a in meta['answers']])
    summary=qa.read(OUT/'summary.json'); audits=[]
    for method,width in zip(run.METHODS,run.WIDTHS):
        def raw(left,right):
            if method==run.METHODS[0]: return window_hidden[left:right]
            pieces=[base[left:right],window_hidden[left:right]]
            if method==run.METHODS[2]: pieces.append(window_logit[left:right,None])
            return np.column_stack(pieces)
        scaler=pickle.loads((OUT/(summary['all_candidates'][method][0]['candidate']+'.pkl')).read_bytes())['scaler']
        mu=np.zeros(width); var=np.zeros(width); mass=0.
        for left in range(0,168123,16384):
            right=min(left+16384,168123); xx=raw(left,right).astype(np.float64)
            w=weights['base_weights'][left:right].astype(np.float32).astype(np.float64)
            old=float(np.float32(mass)); new=float(w.sum()); m=w@xx/new; center=xx-m
            v=np.einsum('i,ij,ij->j',w,center,center)/new; total=old+new; delta=mu-m
            var=(old*var+new*v+delta*delta*old*new/total)/total; mu=(old*mu+new*m)/total; mass=total
        assert np.allclose(mu,scaler.mean_,atol=2e-10,rtol=2e-10) and np.allclose(var,scaler.var_,atol=2e-10,rtol=2e-10)
        fit_design=np.load(OUT/(method+'_fit.npy'),mmap_mode='r'); assert fit_design.shape==(168123,width)
        for left in range(0,168123,16384):
            right=min(left+16384,168123)
            assert np.array_equal(scaler.transform(raw(left,right)).astype(np.float32),fit_design[left:right])
        choices=[]
        for entry in summary['all_candidates'][method]:
            name=entry['candidate']; obj=pickle.loads((OUT/(name+'.pkl')).read_bytes())
            assert np.array_equal(obj['scaler'].mean_,scaler.mean_) and np.array_equal(obj['scaler'].var_,scaler.var_)
            assert obj['weights_file_sha256']==qa.sha(run.OLD/'training_weights.npz') and obj['fit_keys_sha256']==qa.sha(run.OLD/'fit_keys.json')
            with np.load(OUT/(name+'_scores.npz'),allow_pickle=False) as z: scores,answers=z['window_scores'],z['answer_scores']
            for left in range(0,len(scores),16384):
                right=min(left+16384,len(scores)); prediction=obj['model'].predict_proba(scaler.transform(raw(left,right)).astype(np.float32))[:,1]
                assert np.array_equal(prediction,scores[left:right])
            assert np.array_equal(np.asarray([scores[meta['answer_windows'][a['response_id']]].max() for a in meta['answers']]),answers)
            tw=helper.threshold(yw[168123:],scores[168123:]); ta=helper.threshold(ya[634:],answers[634:])
            for level,result in (('window',tw),('answer',ta)):
                s=entry['thresholds'][level]; assert result==(s['f1'],s['precision'],s['threshold'])
            for part,wix,aix in (('fit',slice(0,168123),slice(0,634)),('calibration',slice(168123,None),slice(634,None))):
                for unit,y,s,cutoff in (('windows',yw[wix],scores[wix],tw[2]),('answers',ya[aix],answers[aix],ta[2])):
                    for key,value in helper.counts(y,s,cutoff).items(): assert value==entry['metrics'][part][unit][key]
            choices.append(([min(tw[0],ta[0]),tw[0],tw[1],-entry['C']],name))
            audits.append({'candidate':name,'full210364_predictions_reload_exact':True,'thresholds_counts_exact':True,'iterations':entry['iterations']})
        assert max(choices)[1]==summary['selected'][method]['candidate']
    assert run.source()[-1]==snapshot
    result={'status':'passed','code_sha256':qa.sha(Path(__file__)),'complete_sha256':qa.sha(OUT/'complete.json'),
        'all793_character_weighted_hidden_alignment_verified':True,'hidden_mapping_max_abs_difference':mapping_error,
        'independent_official_scalar_mapping_all210364_exact':True,'all213159_projection_values_exact':True,
        'fit_only_PCA_sample_and_weights_exact':True,'PCA_orthogonality_max_error':orth,'PCA_actual_projection_variance_ratio':ratio,
        'PCA_randomized_SVD_variance_ratio':svd_ratio,
        'fit_only_scaler_independent_weighted_moments_passed':True,'candidates':audits,
        'all_frozen_input_and_output_hashes_verified':True,'test_opened':False,'GPU_used':False,'no_refitting':True,
        'seconds':time.perf_counter()-started,'limitation':'Implementation self-audit; separate interface review provided by evaluation agent.'}
    qa.save(OUT/'AUDIT.json',result); print('SEMANTIC_HIDDEN_AUDIT_PASSED',round(result['seconds'],2),flush=True)


def report():
    meta=qa.metadata(); summary=qa.read(OUT/'summary.json'); scalar=qa.read(run.SEM/'results/summary.json')
    native=qa.read(ROOT/'results/sequence_capacity_v3/BASELINE_COMPARISON.json')['entries']
    comparison=list(native); rows=[]; specifications=[]
    for method,entry in summary['selected'].items():
        rows.append((entry['candidate'],entry['metrics']))
        specifications.append((entry['candidate'],OUT/(entry['candidate']+'_scores.npz'),entry['thresholds']['window']['threshold']))
    for method,entry in scalar['reports'].items():
        rows.append((method,entry['metrics']))
        specifications.append((method,run.SEM/'results'/(method+'_scores.npz'),entry['thresholds']['window']['threshold']))
        comparison.append({'candidate':method,'kind':'extra_checker_scalar','metrics':entry['metrics']})
    for entries in summary['all_candidates'].values():
        comparison.extend({'candidate':e['candidate'],'kind':'extra_checker_hidden','metrics':e['metrics']} for e in entries)
    native_full=qa.read(ROOT/'results/sequence_full_v2/summary.json')['selected']
    for name,e in native_full.items(): rows.append((name,e['metrics']))
    diagnostics={}
    for name,path,cutoff in specifications:
        with np.load(path,allow_pickle=False) as z: scores=z['window_scores']
        alerts=scores>=cutoff; covered={}
        for w,flag in zip(meta['windows'],alerts):
            if flag: covered.setdefault(w['response_id'],set()).update(w['token_indices'])
        typed={}
        for kind in ('Evident Conflict','Subtle Conflict','Evident Baseless Info','Subtle Baseless Info'):
            spans=[];risk={};empty=0
            for tokens in meta['tokens']:
                if tokens['partition']!='calibration': continue
                for mapping in tokens['span_token_mapping']:
                    if tokens['original_labels'][mapping['span_index']]['label_type']!=kind:continue
                    pos=set(mapping['risk_token_indices']);rid=tokens['response_id']
                    if not pos:empty+=1;continue
                    risk.setdefault(rid,set()).update(pos);spans.append((bool(pos&covered.get(rid,set())),pos<=covered.get(rid,set())))
            positive=[j for j,w in enumerate(meta['windows']) if w['partition']=='calibration' and bool(set(w['token_indices'])&risk.get(w['response_id'],set()))]
            typed[kind]={'original_spans':len(spans)+empty,'localizable_spans':len(spans),'unlocalizable_spans':empty,
                'span_any_hit':sum(a for a,b in spans),'span_full_hit':sum(b for a,b in spans),
                'positive_windows':len(positive),'hit_positive_windows':int(alerts[positive].sum()),'positive_window_recall':float(alerts[positive].mean())}
        diagnostics[name]=typed
    qa.save(OUT/'DIAGNOSTICS.json',{'calibration_error_types':diagnostics,'global_selected_thresholds_no_type_tuning':True,'test_opened':False})
    qa.save(OUT/'BASELINE_COMPARISON.json',{'entries':comparison,'same_geometry_and_labels':True,'test_opened':False,
        'semantic_scalar_complete_sha256':qa.sha(run.SEM/'results/complete.json'),'hidden_complete_sha256':qa.sha(OUT/'complete.json')})
    text=['固定MiniCheck内部状态融合，全部是fit634/cal159开发结果。额外核查模型已读完整陈述与资料，属于离线方法。','',
        '| 方法 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|']
    for name,m in rows:
        w=m['calibration']['windows'];a=m['calibration']['answers'];text.append(f"| {name} | {m['fit']['windows']['f1']:.3f} | {w['precision']:.3f} | {w['recall']:.3f} | {w['f1']:.3f} | {a['f1']:.3f} |")
    text+=['','| 方法 | 冲突类 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |','|---|---|---:|---:|---:|']
    for name,d in diagnostics.items():
        for kind in ('Evident Conflict','Subtle Conflict'):
            k=d[kind];text.append(f"| {name} | {kind} | {k['hit_positive_windows']}/{k['positive_windows']} | {k['span_any_hit']}/{k['localizable_spans']} | {k['span_full_hit']}/{k['localizable_spans']} |")
    text+=['','映射全量793答、213159个原始词元，671625个非空白字符无缺覆盖；原金标与窗口分母未变。PCA64只在原20288个fit抽样位置拟合，解释方差97.70%。',
        '三族各3个C均保存，阈值与C只按既定cal规则选择；此处的整体与分类型成绩均需在封存test上再确认。Subtle Conflict仅5个cal span。',
        '不能把新增MiniCheck状态称作纯Llama白盒；不能因PCA64有高解释方差就断言保留了全部有用判别信息。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n','utf-8')
    print('SEMANTIC_HIDDEN_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): audit(); report()
