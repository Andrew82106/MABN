"""Independent synthetic check of token-native ReDeEP fitting; CPU only."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import run_whitebox as impl


def main():
    fixture=impl.HERE/'cpu_fit_fixture';fixture.mkdir(exist_ok=True)
    rng=np.random.default_rng(20260928)
    counts=[5,7];label=np.asarray([0,1,0,1,0,0,0,1,1,0,1,1])
    ecs=rng.normal(size=(12,32));pks=rng.normal(size=(12,32))
    ecs[:,9]=1-label+.02*rng.normal(size=12)
    pks[:,17]=label+.03*rng.normal(size=12)
    plans=[];sequences=[];begin=0
    for i,n in enumerate(counts):
        rid=f'fixture_{i}';ids=list(range(10,10+n))
        plans.append({'response_id':rid,'partition':'fit','answer_token_ids':ids})
        sequences.append({'response_id':rid,'response_token_ids':ids,'risk_mask':label[begin:begin+n].tolist(),
                          'lexical_mask':[0]*n})
        np.savez_compressed(fixture/(rid+'.npz'),ecs_paper=ecs[begin:begin+n],pks_paper=pks[begin:begin+n])
        begin+=n
    plans.append({'response_id':'forbidden_validation','partition':'validation'})
    reads=[];writes=[]
    def read_fit_only(path):
        reads.append(str(path));assert path==impl.RUN/'data/fit/sequences.jsonl'
        return sequences
    def cache(method,plan):
        assert method=='redeep' and plan['partition']=='fit'
        return fixture/(plan['response_id']+'.npz')
    with patch.object(impl,'readl',read_fit_only),patch.object(impl,'npz_path',cache), \
         patch.object(impl,'publish',lambda p,x:writes.append(x)),patch.object(impl,'sha',lambda p:'fixture-only'):
        actual=impl.fit_redeep(plans,{'candidate_heads':[[i,0] for i in range(32)]})
    er=np.asarray([pearsonr(ecs[:,i],1-label).statistic for i in range(32)])
    pr=np.asarray([pearsonr(pks[:,i],label).statistic for i in range(32)])
    hi=sorted(range(32),key=lambda i:(-er[i],i))[:1]
    li=sorted(range(32),key=lambda i:(-pr[i],i))[:10]
    assert actual['fit_tokens']==12 and actual['fit_answers']==2
    assert actual['selected_head_indices']==hi==[9] and actual['selected_layers']==li
    assert np.allclose(actual['head_correlations'],er,atol=1e-14,rtol=0)
    assert np.allclose(actual['layer_correlations'],pr,atol=1e-14,rtol=0)
    es,ps=ecs[:,hi].sum(1),pks[:,li].sum(1)
    expected=MinMaxScaler().fit_transform(ps[:,None]).ravel()-.2*MinMaxScaler().fit_transform(es[:,None]).ravel()
    score=impl.formula.stable_minmax_apply(ps,actual['pks_bounds'])-.2*impl.formula.stable_minmax_apply(es,actual['ecs_bounds'])
    assert np.allclose(score,expected,atol=1e-14,rtol=0)
    report={'passed':True,'actual_fit_function_called':True,'synthetic_tokens':12,
        'fit_token_population_not_windows':True,'nonlexical_tokens_not_dropped':True,
        'validation_plan_ignored':True,'only_fit_label_path_read':True,
        'independent_scipy_rank_agreement':True,'independent_sklearn_minmax_agreement':True,
        'max_score_error':float(abs(score-expected).max()),'real_data_labels_read':False,'GPU_used':False,
        'run_whitebox_sha256':impl.sha(Path(impl.__file__))}
    impl.publish(impl.HERE/'FIT_NATIVE_CPU_AUDIT.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
