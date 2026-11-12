"""Six predeclared LR heads on immutable R10 caches; exposed-test exploration only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import json
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'round10_dual_granularity'
spec = importlib.util.spec_from_file_location('round11_reused_round10',SOURCE/'src/evaluate10.py')
r10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r10)
np = r10.np
FEATURES=('nll','lb','lb_nll')
WIDTHS={'nll':1,'lb':784,'lb_nll':785}
STATUS='exploratory_retest_of_previously_exposed_round10_test'


@contextmanager
def method_scope():
    """Configure reused pure routines in memory only; never edit R10 source files."""
    names=('FEATURES','ITEM_METHODS','TOKEN_METHODS','BROADCAST','LOCALIZATION_METHODS')
    old={name:getattr(r10,name) for name in names}
    r10.FEATURES=FEATURES
    r10.ITEM_METHODS=tuple(f+'__item' for f in FEATURES)
    r10.TOKEN_METHODS=tuple(f+'__token' for f in FEATURES)
    r10.BROADCAST={};r10.LOCALIZATION_METHODS=r10.TOKEN_METHODS
    try: yield
    finally:
        for name,value in old.items():setattr(r10,name,value)


class Bank(r10.Bank):
    def row(self,rid):
        if rid in self.cache:return self.cache[rid]
        g,checksum=self.records[rid]
        path=self.root/'data/features'/(rid+'.npz')
        meta=json.loads(path.with_suffix('.json').read_text('utf-8'))
        assert meta['source_generation_sha256']==checksum
        assert meta['arrays_sha256']==r10.sha(path)
        n=len(g['response_token_ids']);offsets=np.asarray(g['response_token_offsets'])
        with np.load(path,allow_pickle=False) as arrays:
            assert arrays['token_ids'].tolist()==g['response_token_ids']
            assert np.array_equal(arrays['token_start'],offsets[:,0])
            assert np.array_equal(arrays['token_end'],offsets[:,1])
            lb=arrays['lookback_features'].copy();nll=arrays['token_nll'].copy()
        assert lb.shape==(n,784) and nll.shape==(n,)
        assert lb.dtype==nll.dtype==np.float32
        assert np.isfinite(lb).all() and np.isfinite(nll).all() and (nll>=0).all()
        nll=nll.reshape(-1,1)
        value={'lb':lb,'nll':nll,'lb_nll':np.concatenate((lb,nll),axis=1)}
        assert all(value[k].shape==(n,w) for k,w in WIDTHS.items())
        self.cache[rid]=value
        return value


def source_snapshot():
    """R10 recursively verifies its frozen graph; test gold is hashed, not parsed."""
    assert r10.FEATURES!=FEATURES, 'Snapshot must use the original R10 protocol'
    old=r10.source_hashes(SOURCE)
    files=[ROOT/'PLAN.md',ROOT/'protocol.json',*sorted((ROOT/'src').glob('*.py'))]
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))
    assert protocol['features']==WIDTHS and protocol['primary_method']=='lb_nll'
    assert protocol['C']==list(r10.LR_C) and protocol['seed']==r10.SEED
    return {'round10_frozen_graph':old,'local_files_sha256':{p.relative_to(ROOT).as_posix():r10.sha(p) for p in files},
            'comparison_status':STATUS}


def metrics(items,tokens,regions,thresholds):
    with method_scope():
        out=r10.evaluate(items,tokens,regions,thresholds,with_bootstrap=False)
    out['primary_method_fixed_before_test']='lb_nll'
    out.pop('whitebox_increment_baseline',None)
    out['comparison_status']=STATUS
    # Remove R10's unused 0.5-threshold confusion fields from ranking diagnostics.
    for value in out['localization_methods'].values():
        rank=value['within_answer_ranking']
        rank['per_answer']={rid:{k:m[k] for k in ('auroc','average_precision','ranking_tokens')}
                            for rid,m in rank['per_answer'].items()}
    return out


def bootstrap(items,tokens,thresholds):
    groups=sorted({i['group_id'] for i in items});gi={g:j for j,g in enumerate(groups)}
    draws=2000;seed=20260912
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(groups),(draws,len(groups)))
    weights=np.stack([np.bincount(s,minlength=len(groups)) for s in samples])
    out={'groups':len(groups),'draws':draws,'seed':seed,'unit':'question group, both conditions and all tokens together',
         'comparison_status':STATUS,'subsets':{}}
    for subset,rows,suffix in [('answer_items',items,'__item'),('all_resolved_items',tokens,'__token')]:
        names=[f+suffix for f in FEATURES];counts=np.zeros((len(groups),3,3),np.int64)
        for row in rows:
            if not row['main_eligible']:continue
            for j,name in enumerate(names):
                v=row['scores'][name];assert r10.metric.finite(v)
                pred=v>=thresholds[name]['threshold'];y=row['gold']
                counts[gi[row['group_id']],j]+=[int(pred and y==1),int(pred and y==0),int(not pred and y==1)]
        summed=np.einsum('bg,gmc->bmc',weights,counts,optimize=True)
        tp,fp,fn=(summed[:,:,k] for k in range(3))
        def divide(a,b,ok):return np.divide(a,b,out=np.full(a.shape,np.nan,dtype=float),where=ok)
        arrays={'precision':divide(tp,tp+fp,tp+fp>0),'recall':divide(tp,tp+fn,tp+fn>0),
                'f1':divide(2*tp,2*tp+fp+fn,tp+fn>0)}
        out['subsets'][subset]={'methods':{name:{k:r10.ci(a[:,j]) for k,a in arrays.items()} for j,name in enumerate(names)},
            'contrasts':{'lb_nll_minus_lb':{k:r10.ci(a[:,2]-a[:,1]) for k,a in arrays.items()}}}
    return out


def fit():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    assert not (out/'freeze11.json').exists(), 'Do not overwrite frozen fit'
    snapshot=source_snapshot();r10.save(out/'source_snapshot11.json',snapshot)
    r10.save(out/'fit_started11.json',{'utc':r10.utc(),'comparison_status':STATUS,'source_snapshot_sha256':r10.sha(out/'source_snapshot11.json')})
    meta=r10.metadata(SOURCE)
    ti,tt,_,tc=r10.cohort(SOURCE,'train',meta);vi,vt,vr,vc=r10.cohort(SOURCE,'validation',meta)
    bank=Bank(SOURCE,meta[2]);begin=time.perf_counter()
    with method_scope():
        models,thresholds,candidates,audit=r10.fit_models(bank,ti,tt,vi,vt)
        r10.predict(bank,vi,vt,models)
    seconds=time.perf_counter()-begin
    assert source_snapshot()==snapshot
    (out/'frozen_models.pkl').write_bytes(pickle.dumps(models,protocol=5))
    r10.save(out/'selection.json',candidates);r10.save(out/'training_weights.json',audit)
    r10.save(out/'validation_metrics.json',{'coverage':vc,**metrics(vi,vt,vr,thresholds)})
    r10.savel(out/'answer_scores_validation.jsonl',vi);r10.savel(out/'token_scores_validation.jsonl',vt)
    frozen={'schema':'round11-logprob-exploratory-v1','utc':r10.utc(),'comparison_status':STATUS,
            'test_labels_used_in_fit':False,'source_snapshot_sha256':r10.sha(out/'source_snapshot11.json'),
            'model_sha256':r10.sha(out/'frozen_models.pkl'),'selection_sha256':r10.sha(out/'selection.json'),
            'training_weights_sha256':r10.sha(out/'training_weights.json'),'thresholds':thresholds,
            'feature_widths':WIDTHS,'train_coverage':tc,'validation_coverage':vc,'fit_wall_seconds':seconds,
            'primary_method':'lb_nll','primary_baseline':'lb','C_candidates':r10.LR_C,'seed':r10.SEED,'cpu_threads':4}
    r10.save(out/'freeze11.json',frozen);print('ROUND11_FIT_FROZEN',flush=True)


def test():
    out=ROOT/'results';assert not (out/'test_complete11.json').exists(), 'Exploratory score already completed'
    frozen=json.loads((out/'freeze11.json').read_text('utf-8'))
    snapshot=json.loads((out/'source_snapshot11.json').read_text('utf-8'))
    assert r10.sha(out/'source_snapshot11.json')==frozen['source_snapshot_sha256']
    assert source_snapshot()==snapshot
    for name,key in [('frozen_models.pkl','model_sha256'),('selection.json','selection_sha256'),('training_weights.json','training_weights_sha256')]:
        assert r10.sha(out/name)==frozen[key]
    start={'freeze11_sha256':r10.sha(out/'freeze11.json'),'comparison_status':STATUS,'retuning_allowed':False}
    path=out/'test_started11.json'
    if path.exists():assert json.loads(path.read_text('utf-8'))==start
    else:r10.save(path,start)
    meta=r10.metadata(SOURCE);items,tokens,regions,coverage=r10.cohort(SOURCE,'test',meta)
    models=pickle.loads((out/'frozen_models.pkl').read_bytes());bank=Bank(SOURCE,meta[2]);begin=time.perf_counter()
    with method_scope():r10.predict(bank,items,tokens,models)
    seconds=time.perf_counter()-begin
    result={'schema':'round11-logprob-exploratory-v1','coverage':coverage,**start,
            'test_load_and_predict_seconds':seconds,**metrics(items,tokens,regions,frozen['thresholds']),
            'group_bootstrap':bootstrap(items,tokens,frozen['thresholds'])}
    assert source_snapshot()==snapshot
    r10.save(out/'metrics_test.json',result);r10.savel(out/'answer_scores_test.jsonl',items);r10.savel(out/'token_scores_test.jsonl',tokens)
    r10.save(out/'test_complete11.json',{**start,'utc':r10.utc(),
        'metrics_sha256':r10.sha(out/'metrics_test.json'),'answer_scores_sha256':r10.sha(out/'answer_scores_test.jsonl'),
        'token_scores_sha256':r10.sha(out/'token_scores_test.jsonl'),'models':6})
    print('ROUND11_EXPLORATORY_TEST_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','test']);args=p.parse_args()
    with r10.threadpool_limits(limits=4):{'fit':fit,'test':test}[args.stage]()
