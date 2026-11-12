"""Uncached, unpadded single-window verification after R23 cache check failed.

Same exact prompts; slower complete forwards avoid the observed NF4 batch/cache
dependence of internal states. Old failed extraction and diagnostics stay frozen.
"""
from pathlib import Path
import argparse,importlib.util,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'round23_local_evidence_probe'
spec=importlib.util.spec_from_file_location('r23_frozen_prompts',OLD/'src/extract23.py')
e23=importlib.util.module_from_spec(spec);spec.loader.exec_module(e23)
base=e23.base
VERSION='r23b-full-single-window-verification-v2'

def prepare():
    tok,rows,records,old_sig,plans=e23.prepare()
    sig={'version':VERSION,'code_sha256':base.sha(Path(__file__)),'source':base.signature(),
         'original_prompt_signature_sha256':base.model7.digest(old_sig),'original_prompt_code_sha256':base.sha(OLD/'src/extract23.py'),
         'original_plans_sha256':base.sha(OLD/'data/plans.json'),'old_cache_diagnostic_sha256':base.sha(OLD/'results/CACHE_NUMERICAL_DIAGNOSTIC.json'),
         'batch_size':1,'use_cache':False,'padding':False,'A_id':old_sig['A_id'],'B_id':old_sig['B_id'],
         'mode':'post-answer focused complete forward; same original prompts; no labels or original output changes'}
    for name,value in [('signature.json',sig),('plans.json',plans)]:
        path=ROOT/'data'/name
        if path.exists():assert base.read(path)==value,'Frozen single-forward extraction changed'
        else:base.save(path,value)
    return tok,rows,records,sig,plans

def manifest(rows,sig,plans,records):
    entries={}
    for row in rows:
        rid=row['row_id'];p=ROOT/'data/features'/(rid+'.npz');side=p.with_suffix('.json')
        if not side.exists():continue
        m=base.read(side);assert m['signature_sha256']==base.model7.digest(sig)
        assert m['source_generation_sha256']==records[rid][1] and m['plan_sha256']==base.model7.digest(plans[rid])
        assert m['window_keys']==[w['window_key'] for w in plans[rid]['windows']]
        assert m['npz_sha256']==base.sha(p)
        entries[rid]={'npz':str(p.relative_to(ROOT)),'json':str(side.relative_to(ROOT)),
                      'npz_sha256':base.sha(p),'json_sha256':base.sha(side),'windows':m['windows']}
    result={'version':VERSION,'complete':len(entries)==602,'completed_count':len(entries),'expected_count':602,
            'windows_completed':sum(v['windows'] for v in entries.values()),'records':entries,
            'signature_sha256':base.model7.digest(sig),'labels_used':False,'original_validation_or_test_parsed':False,
            'batch_size':1,'use_cache':False,'padding':False}
    base.save(ROOT/'data/feature_manifest.json',result);return result

def run(selfcheck_only=False):
    _,rows,records,sig,plans=prepare()
    if manifest(rows,sig,plans,records)['complete']:print('ALREADY_COMPLETE_NO_GPU');return
    _,model=base.model7.load_model();checks=[]
    for row in (rows[0],rows[-1]):
        plan=plans[row['row_id']];ws=plan['windows'];indices=sorted({0,len(ws)//2,len(ws)-1})
        for i in indices:
            ids=plan['prefix_ids']+ws[i]['suffix_ids']
            first=e23.uncached(model,ids,sig['A_id'],sig['B_id'],'cuda')
            repeat=e23.uncached(model,ids,sig['A_id'],sig['B_id'],'cuda')
            assert all(np.array_equal(first[k],repeat[k]) for k in first)
            checks.append({'row_id':row['row_id'],'window_key':ws[i]['window_key'],'all_arrays_repeat_exact':True,
                           'cached_or_padded_alternative_used':False})
    base.save(ROOT/'data/selfcheck.json',{'status':'passed','signature_sha256':base.model7.digest(sig),'gpu_checks':checks,
               'scope':'Single full-forward numerical repeat, not a claim that v1 cached states match'})
    print('R23B_FULL_FORWARD_SELFCHECK_PASSED',flush=True)
    if selfcheck_only:return
    started=time.perf_counter();completed_windows=0
    for j,row in enumerate(rows):
        rid=row['row_id'];p=ROOT/'data/features'/(rid+'.npz');side=p.with_suffix('.json')
        if side.exists():continue
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();plan=plans[rid];ws=plan['windows'];parts=[]
        for w in ws:
            ids=plan['prefix_ids']+w['suffix_ids']
            parts.append(e23.uncached(model,ids,sig['A_id'],sig['B_id'],'cuda'))
        arrays={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
        assert arrays['verifier_hidden'].shape==(len(ws),3584) and all(np.isfinite(x).all() for x in arrays.values())
        arrays['window_start']=np.asarray([w['start'] for w in ws],np.int32);arrays['window_end']=np.asarray([w['end'] for w in ws],np.int32)
        base.save_arrays(p,arrays);torch.cuda.synchronize()
        meta={'row_id':rid,'signature_sha256':base.model7.digest(sig),'plan_sha256':base.model7.digest(plan),
              'source_generation_sha256':records[rid][1],'npz_sha256':base.sha(p),'windows':len(ws),
              'window_keys':[w['window_key'] for w in ws],'seconds':time.perf_counter()-t,
              'full_forward_tokens':sum(w['input_tokens'] for w in ws),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'batch_size':1,'use_cache':False,'padding':False,'labels_used':False,'output_regenerated':False}
        base.save(side,meta);completed_windows+=len(ws)
        if (j+1)%25==0 or j==0:
            manifest(rows,sig,plans,records);print('EXTRACT23B',j+1,602,'WINDOWS',completed_windows,'SECONDS',round(time.perf_counter()-started,1),flush=True)
    fm=manifest(rows,sig,plans,records);assert fm['complete'] and fm['windows_completed']==12222
    base.save(ROOT/'data/completion.json',{'loop_seconds':time.perf_counter()-started,'signature_sha256':base.model7.digest(sig),
               'manifest_sha256':base.sha(ROOT/'data/feature_manifest.json'),'labels_used':False,'original_output_changed':False})
    print('EXTRACT23B_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selfcheck','run']);a=p.parse_args()
    if a.stage=='prepare':
        _,rows,_,_,plans=prepare();print('PREPARED',len(rows),'WINDOWS',sum(len(p['windows']) for p in plans.values()))
    else:run(a.stage=='selfcheck')
