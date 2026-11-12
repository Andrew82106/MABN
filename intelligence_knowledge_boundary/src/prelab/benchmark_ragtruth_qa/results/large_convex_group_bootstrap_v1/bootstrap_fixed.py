"""Three fixed paired contrasts on exposed QA calibration; no reselection."""
from pathlib import Path
from collections import Counter
import argparse
import importlib.util
import json
import numpy as np

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
PREVIOUS = QA / 'results/completed_group_bootstrap_v1/bootstrap_fixed.py'
spec = importlib.util.spec_from_file_location('_fixed_group_bootstrap_helpers', PREVIOUS)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
read,rows,sha,write = old.read,old.rows,old.sha,old.write
METHODS = ('semantic_tree_large04','lookback_lr_large04','lookback_tree_large06','harp_citation_lr')
SEED, REPEATS = 20261009, 5000
CONTRASTS = tuple((METHODS[0]+'_minus_'+m,METHODS[0],m) for m in METHODS[1:])
CONVEX = QA / 'results/large_fixed_convex_v1'
CITATION = QA / 'results/citation_alignment_lr_v1'


def entries():
    stems = ['semantic_claim__old_tree__large_weight0.4','lookback__old_lr__large_weight0.4',
             'lookback__old_tree__large_weight0.6']
    result = {m:{'entry':read(CONVEX/(stem+'.json')),'directory':str(CONVEX),'entry_path':str(CONVEX/(stem+'.json'))}
              for m,stem in zip(METHODS[:3],stems)}
    baseline = read(CITATION/'summary.json')['selected']['harp_claim__two_scores_and_citation']
    result['harp_citation_lr']={'entry':baseline,'directory':str(CITATION),'entry_path':str(CITATION/'summary.json')}
    for name,record in result.items():
        e=record['entry'];record['score_path']=str(Path(record['directory'])/(e['candidate']+'_scores.npz'))
    assert result[METHODS[0]]['entry']['large_weight']==.4
    assert result[METHODS[1]]['entry']['large_weight']==.4
    assert result[METHODS[2]]['entry']['large_weight']==.6
    return result


def prepare():
    assert not (OUT/'design_freeze.json').exists()
    fixed=entries()
    for directory in (CONVEX,CITATION):
        complete=read(directory/'complete.json')
        assert not complete['official_test_opened'] and sha(directory/'summary.json')==complete['summary_sha256']
    protocol={'scope':'Already-exposed original159 QA calibration answers/154 source groups/42241 windows only.',
        'primary':'semantic_claim__old_tree__large_weight0.4',
        'comparators':['lookback__old_lr__large_weight0.4','lookback__old_tree__large_weight0.6',
                       'Previously selected HARP+tail+eight-citation-feature LR, original frozen threshold'],
        'methods':list(METHODS),'contrasts':list(CONTRASTS),'replicates':REPEATS,'seed':SEED,
        'sampling':'NumPy default_rng PCG64; sorted original group_id; integers(0,154,size=(5000,154)); all methods, answers and overlapping windows use the same group draws.',
        'statistic':'Pooled micro F1 from grouped TP/FP/FN; paired primary-minus-comparator difference separately for window and answer. Each fixed candidate supplies both units.',
        'interval':'2.5/97.5 percentiles, NumPy linear interpolation; zero-denominator F1=0, count degeneracy.',
        'selection':'None. Keep every original C/model/epoch/convex weight and both thresholds fixed in every draw.',
        'limitations':'Conditional interval on repeatedly-developed calibration, excluding all upstream model/epoch/threshold/alpha selection bias. Not independent significance evidence; no unseen-test claim.',
        'new_fits':0,'GPU_used':False,'test_opened':False}
    write(OUT/'protocol.json',protocol)
    files=[Path(__file__),PREVIOUS,OUT/'protocol.json',QA/'data/answers_calibration.jsonl',
           QA/'data/windows_k4_calibration.jsonl',CONVEX/'complete.json',CONVEX/'summary.json',
           CITATION/'complete.json',CITATION/'summary.json']
    for record in fixed.values():
        p=Path(record['score_path']);assert sha(p)==record['entry']['scores_sha256']
        files.extend([p,Path(record['entry_path'])])
    write(OUT/'design_freeze.json',{'status':'fixed_before_bootstrap','methods':fixed,
        'files_sha256':{str(p.resolve()):sha(p) for p in files},'no_threshold_or_model_reselection':True})
    print('LARGE_CONVEX_PAIRED_BOOTSTRAP_FROZEN',flush=True)


def run():
    assert not (OUT/'complete.json').exists()
    frozen=read(OUT/'design_freeze.json');protocol=read(OUT/'protocol.json')
    for p,digest in frozen['files_sha256'].items():assert sha(p)==digest,p
    answers=rows(QA/'data/answers_calibration.jsonl');windows=rows(QA/'data/windows_k4_calibration.jsonl')
    assert len(answers)==159 and len(windows)==42241
    assert all(r['partition']=='calibration' and r['eligible'] and r['label'] in (0,1) for r in answers+windows)
    answer_ids={r['response_id']:i for i,r in enumerate(answers)};assert len(answer_ids)==159
    group_ids=sorted({r['group_id'] for r in answers});assert len(group_ids)==154
    gi={g:i for i,g in enumerate(group_ids)}
    for w in windows:assert w['group_id']==answers[answer_ids[w['response_id']]]['group_id']
    own={'windows':np.array([gi[r['group_id']] for r in windows]),'answers':np.array([gi[r['group_id']] for r in answers])}
    gold={'windows':np.array([r['label'] for r in windows]),'answers':np.array([r['label'] for r in answers])}
    assert gold['windows'].sum()==5984 and gold['answers'].sum()==100
    window_answer=np.array([answer_ids[r['response_id']] for r in windows])
    counts=np.zeros((4,2,154,3),dtype=np.int64);method_results={}
    for mi,name in enumerate(METHODS):
        record=frozen['methods'][name];e=record['entry']
        with np.load(record['score_path'],allow_pickle=False) as z:
            assert z['window_scores'].shape==(210364,) and z['answer_scores'].shape==(793,)
            scores={'windows':z['window_scores'][168123:].copy(),'answers':z['answer_scores'][634:].copy()}
        maxima=np.full(159,-np.inf);np.maximum.at(maxima,window_answer,scores['windows'])
        assert np.array_equal(maxima,scores['answers'])
        method_results[name]={'candidate':e['candidate'],'thresholds':e['thresholds'],'metrics':{}}
        for ui,unit in enumerate(('windows','answers')):
            s=scores[unit];assert np.isfinite(s).all() and s.shape==gold[unit].shape
            t=e['thresholds']['window' if unit=='windows' else 'answer']['threshold']
            metric=old.metric(gold[unit],s,t)
            assert all(v==e['metrics']['calibration'][unit][k] for k,v in metric.items()),(name,unit)
            counts[mi,ui]=old.group_counts(gold[unit],s,t,own[unit],154)
            assert np.array_equal(counts[mi,ui].sum(axis=0),[metric['tp'],metric['fp'],metric['fn']])
            method_results[name]['metrics'][unit]=metric
    rng=np.random.default_rng(SEED);assert type(rng.bit_generator).__name__=='PCG64'
    draws=rng.integers(0,154,size=(REPEATS,154))
    # Multiplicity matrix avoids allocating method x unit x draws x groups x counts.
    multiplicity=np.zeros((REPEATS,154),np.int64)
    np.add.at(multiplicity,(np.repeat(np.arange(REPEATS),154),draws.ravel()),1)
    sample=np.einsum('rg,mugc->murc',multiplicity,counts)
    assert sample.shape==(4,2,REPEATS,3)
    for j in (0,1,REPEATS-1):
        oracle=np.sum(counts[:,:,draws[j],:],axis=2)
        assert np.array_equal(oracle,sample[:,:,j,:])
    boot=old.f1(sample);point=old.f1(counts.sum(axis=2));contrasts={}
    for name,left,right in CONTRASTS:
        li,ri=METHODS.index(left),METHODS.index(right);contrasts[name]={'left':left,'right':right}
        for ui,unit in enumerate(('windows','answers')):
            delta=boot[li,ui]-boot[ri,ui]
            lo,hi=np.percentile(delta,[2.5,97.5],method='linear')
            contrasts[name][unit]={'point_f1_difference':float(point[li,ui]-point[ri,ui]),
                'percentile95':[float(lo),float(hi)],'contains_zero':bool(lo<=0<=hi)}
    np.savez_compressed(OUT/'group_bootstrap.npz',group_ids=np.array(group_ids),methods=np.array(METHODS),
        units=np.array(['windows','answers']),count_order=np.array(['tp','fp','fn']),group_counts=counts,
        draws=draws.astype(np.int16),bootstrap_f1=boot)
    result={'status':'passed','protocol':protocol,'fixed_methods':method_results,'contrasts':contrasts,
        'group_answer_size_histogram':dict(Counter(Counter(r['group_id'] for r in answers).values())),
        'zero_f1_denominators':int(np.count_nonzero(2*sample[...,0]+sample[...,1]+sample[...,2]==0)),
        'checks':{'score_hashes_and_saved_counts_exact':True,'all_answer_maxima_exact':True,
            'paired_draws_identical_for_all_methods_and_both_units':True,'three_draws_direct_repetition_verified':True,
            'new_fits':0,'GPU_used':False,'test_opened':False},
        'design_freeze_sha256':sha(OUT/'design_freeze.json'),'bootstrap_arrays_sha256':sha(OUT/'group_bootstrap.npz')}
    write(OUT/'PAIRED_GROUP_INTERVALS.json',result)
    labels={'lookback_lr_large04':'LB旧LR＋large 0.4','lookback_tree_large06':'LB旧树＋large 0.6','harp_citation_lr':'原HARP＋tail＋8列LR'}
    report=['# 固定large组合的资料组配对区间','','主对象固定为semantic旧树＋large 0.4（窗口0.690281、整答0.891089）。仅已暴露cal159答、154资料组；同组全部回答及4BPE重叠窗口一起有放回采样5000次，seed20261009。模型、混合权重及两级阈值不重选。','',
        '| 主对象减对照 | 窗口ΔF1 [95%区间] | 整答ΔF1 [95%区间] |','|---|---:|---:|']
    for key,_,right in CONTRASTS:
        cells=[]
        for unit in ('windows','answers'):
            x=contrasts[key][unit];lo,hi=x['percentile95']
            cells.append(f"{x['point_f1_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        report.append(f"| {labels[right]} | {cells[0]} | {cells[1]} |")
    report+=['','均为同一固定候选的双F1差。区间只描述既定选择在反复开发cal上的条件波动，没有计入原模型、轮次、阈值和组合选择的偏差，不能作为独立显著性或未见测试的证明。跨零不等于方法等效。',
        '','已独立核原混淆计数、F1及整答窗口最大值；未拟合、未推理、未打开测试。逐组计数和全部共同抽样索引已保存。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    for p,digest in frozen['files_sha256'].items():assert sha(p)==digest,p
    write(OUT/'complete.json',{'status':'complete_conditional_calibration_diagnostic',
        'result_sha256':sha(OUT/'PAIRED_GROUP_INTERVALS.json'),'report_sha256':sha(OUT/'REPORT.md'),
        'protocol_sha256':sha(OUT/'protocol.json'),'test_opened':False})
    print(json.dumps({'status':'passed','contrasts':contrasts},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));stage=p.parse_args().stage
    {'prepare':prepare,'run':run}[stage]()
