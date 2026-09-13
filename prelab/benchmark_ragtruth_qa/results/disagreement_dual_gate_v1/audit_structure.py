"""Independent CPU geometry/weight/selection-boundary audit; no fitting."""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import sys
import time
import numpy as np
from scipy.special import logit

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_development as q


def run():
    tick = time.perf_counter(); done = q.read(OUT/'preparation_complete.json')
    for n,h in done['files_sha256'].items(): assert q.sha(OUT/n)==h,n
    for n,h in done['source_sha256'].items(): assert q.sha(n)==h,n
    complete=q.read(OUT/'complete.json'); assert complete['status']=='complete_development_only'
    assert q.sha(OUT/'summary.json')==complete['summary_sha256']
    cfg=q.read(OUT/'protocol.json'); sources=q.read(OUT/'source_binding.json')['source_entries']
    rows=[w for part in ('fit','calibration') for w in q.lines(ROOT/f'data/windows_k4_{part}.jsonl')]
    geometry=q.read(OUT/'geometry.json'); assert q.digest(rows)==geometry['window_order_sha256']
    nfit=168123
    with np.load(OUT/'frozen_inputs.npz') as z:
        p=z['window_probabilities'].copy(); base=z['native_base_weights'].copy(); y=z['window_labels'].copy()
        saved_keep=z['current_positive'].copy(); saved_rescue=z['rescue_eligible'].copy()
    assert np.array_equal(y,[w['label'] for w in rows]) and p.shape==(210364,4)
    t=np.array([e['thresholds']['window']['threshold'] for e in sources]); decisions=p>=t
    assert np.array_equal(decisions[:,0],saved_keep)
    assert np.array_equal(~decisions[:,0]&decisions[:,1:].any(axis=1),saved_rescue)
    # Reconstruct all feature columns independently, consuming no labels.
    z=logit(np.clip(p,1e-6,1-1e-6)); blocks=[z,p-t]
    blocks += [(z[:,i]-z[:,j])[:,None] for i,j in combinations(range(4),2)]
    blocks += [(decisions[:,i]==decisions[:,j])[:,None] for i,j in combinations(range(4),2)]
    chains=[]; first=0
    for i in range(1,len(rows)+1):
        if i==len(rows) or rows[i]['response_id']!=rows[i-1]['response_id'] or rows[i]['token_start']-rows[i-1]['token_start']!=1:
            chains.append((first,i)); first=i
    for radius in (1,3):
        maxima=np.empty_like(p); means=np.empty_like(p)
        for left,right in chains:
            for i in range(left,right):
                neighborhood=p[max(left,i-radius):min(right,i+radius+1)]
                maxima[i]=np.max(neighborhood,axis=0); means[i]=np.mean(neighborhood,axis=0)
        blocks += [maxima,means]
    aw=defaultdict(list)
    for i,w in enumerate(rows): aw[w['response_id']].append(i)
    am=np.empty_like(p)
    for indices in aw.values(): am[indices]=p[indices].max(axis=0)
    rebuilt=np.column_stack(blocks+[am]).astype(np.float32)
    stored=np.load(OUT/'window_features.npy',mmap_mode='r')
    assert np.array_equal(rebuilt,stored)
    # Original base equalizes groups, answers within group, then windows.
    tree=defaultdict(lambda:defaultdict(list))
    for i,w in enumerate(rows[:nfit]): tree[w['group_id']][w['answer_id']].append(i)
    expected=np.empty(nfit,np.float64)
    for answers in tree.values():
        for indices in answers.values(): expected[indices]=1/(len(answers)*len(indices))
    expected/=expected.mean(); assert np.array_equal(expected,base)
    groups=np.array([w['group_id'] for w in rows[:nfit]])
    shuffled=np.random.default_rng(cfg['folds']['seed']).permutation(sorted(tree)).tolist()
    folds=q.read(OUT/'folds.json'); audits=[]; coverage=np.zeros(nfit,int)
    for fold in range(3):
        held=np.isin(groups,shuffled[fold::3]); coverage+=held
        assert folds[fold]['held_groups']==shuffled[fold::3]
        assert not set(groups[held]) & set(groups[~held])
        for name,mask in [('keep',saved_keep),('rescue',saved_rescue)]:
            with np.load(OUT/f'fold_{fold}_{name}_indices_weights.npz') as z:
                ix=z['fit_indices']; hi=z['held_indices']; w=z['sample_weights']
            assert np.array_equal(ix,np.flatnonzero(mask[:nfit]&~held))
            assert np.array_equal(hi,np.flatnonzero(mask[:nfit]&held))
            mass=np.bincount(y[ix],weights=base[ix],minlength=2); factors=base[ix].sum()/(2*mass)
            assert np.array_equal(w,base[ix]*factors[y[ix]])
            assert np.isclose(w[y[ix]==0].sum(),w[y[ix]==1].sum())
            audits.append(dict(fold=fold,gate=name,fit_rows=len(ix),held_rows=len(hi),weights_exact=True))
    assert np.array_equal(coverage,np.ones(nfit,int))
    for name,mask in [('keep',saved_keep),('rescue',saved_rescue)]:
        with np.load(OUT/f'full_{name}_indices_weights.npz') as z: ix=z['fit_indices']; w=z['sample_weights']
        assert np.array_equal(ix,np.flatnonzero(mask[:nfit]))
        mass=np.bincount(y[ix],weights=base[ix],minlength=2)
        assert np.array_equal(w,base[ix]*(base[ix].sum()/(2*mass))[y[ix]])
    # The source/code inspection boundary is explicit; hashes bind what was run.
    result=dict(status='passed',all_210364x40_features_exact=True,original_native_base_weights_exact=True,
        group_OOF_gate_training_indices_and_weights_exact=True,folds=audits,
        no_label_consumed_by_feature_reconstruction=True,only_gate_level_OOF=True,
        limitations='Frozen upstream fit scores remain in-sample. This audit cannot turn gate-only OOF into end-to-end held-out evidence.',
        source_sha256={n:q.sha(OUT/n) for n in ('protocol.json','preparation_complete.json','complete.json','source_binding.json','audit_structure.py')},
        new_fits=0,GPU_used=False,official_test_opened=False,seconds=time.perf_counter()-tick)
    q.save(OUT/'INDEPENDENT_STRUCTURE_REPLAY.json',result)
    print('DUAL_GATE_STRUCTURE_REPLAY_PASSED',result['seconds'],flush=True)


if __name__=='__main__': run()
