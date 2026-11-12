"""Focused evidence-check states for every existing output-defined BPE window.

Offline auxiliary forwards; no labels, predictions, fitted parameters, or
held-out source contents are loaded by this extractor.
"""
from pathlib import Path
from collections import defaultdict
import argparse, importlib.util, json, time
import numpy as np
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r23_source22',ROOT.parent/'round22_evidence_verification/src/extract22.py')
e22=importlib.util.module_from_spec(spec);spec.loader.exec_module(e22)
base=e22.base
VERSION='r23-focused-4bpe-verification-v1'
BATCH=8
SYSTEM='You are a careful checker of whether a highlighted fragment is supported by supplied documents.'
INSTRUCTION=(
 'Check only the focused fragment of the answer, using the supplied search results as evidence. '
 'Read the whole answer to understand the fragment, its references and relationships. '
 'Decide whether the factual information expressed by this fragment is supported in that context. '
 'Use only the supplied results, not facts from memory. '
 'Do not flag the fragment merely because a different part of the answer is unsupported. '
 'A refusal, grammatical connector, or fragment with no factual content is not an unsupported factual claim. '
 'A concrete unsupported guess is still unsupported even if hedged.\n'
 'Reply with exactly one letter:\n'
 'A: This fragment expresses no unsupported factual information.\n'
 'B: This fragment expresses information that contradicts the results or is not supported by them.\n\n')

def common_length(sequences):
    n=min(map(len,sequences))-1
    first=sequences[0]
    for i in range(n):
        if any(s[i]!=first[i] for s in sequences[1:]):return i
    return n

def prepare():
    rows,records,_=e22.e21.context()
    tok=e22.e21.AutoTokenizer.from_pretrained(base.model7.MODEL,local_files_only=True)
    coords=base.read(ROOT/'data/visible_windows.json')
    assert set(coords)==set(records) and sum(map(len,coords.values()))==12222
    sig={'version':VERSION,'code_sha256':base.sha(Path(__file__)), 'base':base.signature(),
         'visible_windows_sha256':base.sha(ROOT/'data/visible_windows.json'),
         'system':SYSTEM,'instruction':INSTRUCTION,'batch_size':BATCH,
         'A_id':tok.encode('A',add_special_tokens=False)[0], 'B_id':tok.encode('B',add_special_tokens=False)[0],
         'cache':'Exact longest shared token prefix; independent DynamicCache branch per batch; right-pad masked suffix only',
         'labels_used':False,'mode':'post-answer focused auxiliary verification'}
    assert all(len(tok.encode(x,add_special_tokens=False))==1 for x in ['A','B'])
    plans={}
    for row in rows:
        rid=row['row_id'];g,h=records[rid]
        q=row['questions'][0];q=q['question'] if isinstance(q,dict) else q
        docs='\n\n'.join(f"Source {i+1}: {p['title']}\n{p['text']}" for i,p in enumerate(row['passages']))
        common=INSTRUCTION+'Question:\n'+q+'\n\nSearch results:\n'+docs+'\n\nWhole answer:\n'+g['response']+'\n\nFocused fragment:\n'
        entries=[];sequences=[]
        for w in coords[rid]:
            assert g['response'][w['start']:w['end']]==w['text']
            focus=f"Characters {w['start']} to {w['end']} (end exclusive): "+json.dumps(w['text'],ensure_ascii=False)+'\n\nVerdict:'
            ids=base.model7.chat_ids(tok,common+focus,SYSTEM)
            assert len(ids)<=base.model7.CONFIG['max_input_tokens']
            entries.append(dict(w,input_tokens=len(ids)));sequences.append(ids)
        k=common_length(sequences);assert k>0
        for w,ids in zip(entries,sequences):w['suffix_ids']=ids[k:];assert w['suffix_ids']
        plans[rid]={'row_id':rid,'prefix_ids':sequences[0][:k],'windows':entries,
                    'source_generation_sha256':h,'input_row_sha256':base.model7.digest(row)}
    for name,value in [('signature.json',sig),('plans.json',plans)]:
        p=ROOT/'data'/name
        if p.exists():assert base.read(p)==value,'Frozen extraction changed'
        else:base.save(p,value)
    return tok,rows,records,sig,plans

@torch.inference_mode()
def prefix_cache(model,ids,device):
    z=torch.tensor([ids],device=device)
    p=model.model(input_ids=z,use_cache=True).past_key_values
    return tuple((k,v) for k,v in p.to_legacy_cache())

@torch.inference_mode()
def branches(model,cache,suffixes,a_id,b_id,pad_id,device):
    b=len(suffixes);m=max(map(len,suffixes));p=cache[0][0].shape[-2]
    ids=torch.full((b,m),pad_id,dtype=torch.long,device=device)
    mask=torch.zeros((b,p+m),dtype=torch.long,device=device);mask[:,:p]=1
    lengths=[]
    for i,s in enumerate(suffixes):
        ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,p:p+len(s)]=1;lengths.append(len(s))
    legacy=tuple((k.repeat(b,1,1,1),v.repeat(b,1,1,1)) for k,v in cache)
    branch=DynamicCache.from_legacy_cache(legacy)
    pos=torch.arange(p,p+m,device=device).unsqueeze(0).expand(b,-1)
    result=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,
                       past_key_values=branch,use_cache=True,output_hidden_states=False)
    h=result.last_hidden_state[torch.arange(b,device=device),torch.tensor(lengths,device=device)-1]
    logits=model.lm_head(h).float();ab=logits[:,[a_id,b_id]]
    return {'verifier_hidden':h.float().cpu().numpy(),'ab_logits':ab.cpu().numpy(),
            'ab_probabilities':ab.softmax(-1).cpu().numpy(),
            'ab_full_vocab_probabilities':logits.softmax(-1)[:,[a_id,b_id]].cpu().numpy()}

@torch.inference_mode()
def uncached(model,ids,a_id,b_id,device):
    z=torch.tensor([ids],device=device)
    h=model.model(input_ids=z,use_cache=False).last_hidden_state[:,-1]
    logits=model.lm_head(h).float();ab=logits[:,[a_id,b_id]]
    return {'verifier_hidden':h.float().cpu().numpy(),'ab_logits':ab.cpu().numpy(),
            'ab_probabilities':ab.softmax(-1).cpu().numpy(),
            'ab_full_vocab_probabilities':logits.softmax(-1)[:,[a_id,b_id]].cpu().numpy()}

def cpu_selfcheck():
    torch.manual_seed(23)
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,
                    attention_dropout=0.0)
    cfg._attn_implementation='sdpa';model=Qwen2ForCausalLM(cfg).eval()
    pre=[1,3,5,7,9,11,13];ss=[[17,19,23],[29],[31,37,41,43,47]]
    cache=prefix_cache(model,pre,'cpu');saved=tuple((k.clone(),v.clone()) for k,v in cache)
    batch=branches(model,cache,ss,2,4,0,'cpu');repeat=branches(model,cache,ss,2,4,0,'cpu')
    for k in batch:assert np.array_equal(batch[k],repeat[k])
    diffs={k:0. for k in batch}
    for i,s in enumerate(ss):
        full=uncached(model,pre+s,2,4,'cpu')
        for k in batch:
            diff=float(np.max(np.abs(batch[k][i]-full[k][0])));diffs[k]=max(diffs[k],diff)
            assert diff<2e-6,(k,diff)
    assert all(torch.equal(x,y) for kv,sv in zip(cache,saved) for x,y in zip(kv,sv))
    print('CPU_CACHE_BRANCH_CHECK',diffs,flush=True)
    return diffs

def manifest(rows,sig):
    records={}
    for row in rows:
        p=ROOT/'data/features'/(row['row_id']+'.npz');side=p.with_suffix('.json')
        if not side.exists():continue
        m=base.read(side);assert m['signature_sha256']==base.model7.digest(sig)
        assert m['npz_sha256']==base.sha(p)
        records[row['row_id']]={'npz':str(p.relative_to(ROOT)),'json':str(side.relative_to(ROOT)),
             'npz_sha256':base.sha(p),'json_sha256':base.sha(side),'windows':m['windows']}
    fm={'version':VERSION,'complete':len(records)==602,'completed_count':len(records),'expected_count':602,
        'windows_completed':sum(x['windows'] for x in records.values()),'records':records,
        'signature_sha256':base.model7.digest(sig),'labels_used':False,'original_validation_or_test_parsed':False}
    base.save(ROOT/'data/feature_manifest.json',fm);return fm

def run(selfcheck_only=False):
    cpu_diffs=cpu_selfcheck();tok,rows,records,sig,plans=prepare()
    if manifest(rows,sig)['complete']:print('ALREADY_COMPLETE');return
    _,model=base.model7.load_model();checked=[];start=time.perf_counter()
    for rid in (rows[0]['row_id'],rows[-1]['row_id']):
        plan=plans[rid];ws=plan['windows'];ix=sorted({0,len(ws)//2,len(ws)-1})
        cache=prefix_cache(model,plan['prefix_ids'],'cuda');clones=tuple((k.clone(),v.clone()) for k,v in cache)
        suffix=[ws[i]['suffix_ids'] for i in ix]
        got=branches(model,cache,suffix,sig['A_id'],sig['B_id'],tok.pad_token_id,'cuda')
        rep=branches(model,cache,suffix,sig['A_id'],sig['B_id'],tok.pad_token_id,'cuda')
        assert all(np.array_equal(got[k],rep[k]) for k in got)
        assert all(torch.equal(x,y) for kv,sv in zip(cache,clones) for x,y in zip(kv,sv))
        for j,i in enumerate(ix):
            ref=uncached(model,plan['prefix_ids']+ws[i]['suffix_ids'],sig['A_id'],sig['B_id'],'cuda')
            diff={k:float(np.abs(got[k][j]-ref[k][0]).max()) for k in got}
            mean_hidden=float(np.abs(got['verifier_hidden'][j]-ref['verifier_hidden'][0]).mean())
            # Quantized BF16 batching may round differently; store actual errors.
            assert mean_hidden<.03 and diff['verifier_hidden']<1.0 and diff['ab_logits']<.5 and diff['ab_probabilities']<.1,(rid,diff,mean_hidden)
            checked.append({'row_id':rid,'window_key':ws[i]['window_key'],'cache_vs_full_max_abs':diff,'mean_hidden_abs':mean_hidden,'repeat_exact':True,'parent_cache_unchanged':True})
        del cache,clones
    base.save(ROOT/'data/selfcheck.json',{'status':'passed','cpu_diffs':cpu_diffs,'gpu_checks':checked,
           'signature_sha256':base.model7.digest(sig),'seconds':time.perf_counter()-start})
    print('GPU_SELFCHECK_PASSED',flush=True)
    if selfcheck_only:return
    start=time.perf_counter()
    for j,row in enumerate(rows):
        rid=row['row_id'];p=ROOT/'data/features'/(rid+'.npz');side=p.with_suffix('.json')
        if side.exists():continue
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter();plan=plans[rid]
        cache=prefix_cache(model,plan['prefix_ids'],'cuda');parts=[];ws=plan['windows']
        for lo in range(0,len(ws),BATCH):
            parts.append(branches(model,cache,[w['suffix_ids'] for w in ws[lo:lo+BATCH]],sig['A_id'],sig['B_id'],tok.pad_token_id,'cuda'))
        arrays={k:np.concatenate([part[k] for part in parts],axis=0) for k in parts[0]}
        assert arrays['verifier_hidden'].shape==(len(ws),3584) and all(np.isfinite(x).all() for x in arrays.values())
        arrays['window_start']=np.asarray([w['start'] for w in ws],np.int32)
        arrays['window_end']=np.asarray([w['end'] for w in ws],np.int32)
        base.save_arrays(p,arrays);torch.cuda.synchronize()
        meta={'row_id':rid,'signature_sha256':base.model7.digest(sig),'plan_sha256':base.model7.digest(plan),
              'source_generation_sha256':records[rid][1],'npz_sha256':base.sha(p),'windows':len(ws),
              'window_keys':[w['window_key'] for w in ws],'seconds':time.perf_counter()-t,
              'prefix_tokens':len(plan['prefix_ids']),'suffix_tokens':sum(len(w['suffix_ids']) for w in ws),
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'labels_used':False,'output_regenerated':False}
        base.save(side,meta);del cache,parts,arrays
        if (j+1)%50==0:
            manifest(rows,sig);print('EXTRACT23',j+1,602,'SECONDS',round(time.perf_counter()-start,1),flush=True)
    fm=manifest(rows,sig);assert fm['complete'] and fm['windows_completed']==12222
    base.save(ROOT/'data/completion.json',{'loop_seconds':time.perf_counter()-start,'signature_sha256':base.model7.digest(sig),
              'manifest_sha256':base.sha(ROOT/'data/feature_manifest.json'),'labels_used':False,'original_output_changed':False})
    print('EXTRACT23_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu-selfcheck','selfcheck','run']);a=p.parse_args()
    if a.stage=='cpu-selfcheck':cpu_selfcheck()
    elif a.stage=='prepare':
        _,rows,_,_,plans=prepare();print('PREPARED',len(rows),'WINDOWS',sum(len(p['windows']) for p in plans.values()))
    else:run(a.stage=='selfcheck')
