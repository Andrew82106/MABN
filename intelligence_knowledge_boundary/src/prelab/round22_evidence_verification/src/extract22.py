"""One additional evidence-verification forward pass per original Qwen answer.

Consumes actual-train visible inputs and original generation only, never gold.
This is an offline auxiliary task, not original-generation confidence.
"""
import argparse
import importlib.util
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r22_source21',ROOT.parent/'round21_semantic_internal_probe/src/extract21.py')
e21=importlib.util.module_from_spec(spec);spec.loader.exec_module(e21)
base=e21.base
VERSION='r22-same-qwen-verification-ab-v1'
SYSTEM='You are a careful checker of whether an answer is supported by supplied documents.'
INSTRUCTION=(
 'Check whether the answer makes any factual claim that is not supported by the supplied search results. '
 'Use only these results, not facts from memory. A refusal or a statement that gives no factual answer is not an unsupported factual claim. '
 'A concrete unsupported guess is still unsupported even when it uses words such as "probably".\n'
 'Reply with exactly one letter:\n'
 'A: There is no unsupported factual claim in the answer.\n'
 'B: At least one factual claim contradicts the results or is not supported by them.\n\n')


def make_prompt(row,g):
    question=row['questions'][0]
    if isinstance(question,dict):
        # Input schema must be explicit rather than serialized as hidden metadata.
        question=question['question']
    assert isinstance(question,str)
    documents='\n\n'.join(f"Source {i+1}: {p['title']}\n{p['text']}" for i,p in enumerate(row['passages']))
    return INSTRUCTION+'Question:\n'+question+'\n\nSearch results:\n'+documents+'\n\nAnswer to check:\n'+g['response']+'\n\nVerdict:'


def prepare():
    rows,records,_=e21.context()
    tok=e21.AutoTokenizer.from_pretrained(base.model7.MODEL,local_files_only=True)
    a=tok.encode('A',add_special_tokens=False);b=tok.encode('B',add_special_tokens=False)
    assert len(a)==len(b)==1 and a!=b
    sig={'version':VERSION,'source':base.signature(),'code_sha256':base.sha(Path(__file__)),
         'system':SYSTEM,'instruction':INSTRUCTION,'A_id':a[0],'B_id':b[0],
         'mode':'additional offline verification task; original answer unchanged',
         'source_selector':'R16 actual split=train; all602, no label selection'}
    plans={}
    for row in rows:
        g,h=records[row['row_id']];prompt=make_prompt(row,g)
        ids=base.model7.chat_ids(tok,prompt,SYSTEM)
        assert len(ids)<=base.model7.CONFIG['max_input_tokens']
        plans[row['row_id']]={'row_id':row['row_id'],'system':SYSTEM,'prompt':prompt,'input_ids':ids,
           'source_generation_sha256':h,'input_row_sha256':base.model7.digest(row),'input_tokens':len(ids)}
    for name,value in [('signature.json',sig),('plans.json',plans)]:
        p=ROOT/'data'/name
        if p.exists():assert base.read(p)==value,'Frozen extraction configuration differs'
        else:base.save(p,value)
    return tok,rows,records,sig,plans


@torch.inference_mode()
def forward(model,ids,a_id,b_id):
    t=torch.tensor([ids],dtype=torch.long,device='cuda')
    h=model.model(input_ids=t,use_cache=False,output_hidden_states=False).last_hidden_state[:,-1]
    logits=model.lm_head(h).float()[0]
    ab=logits[[a_id,b_id]];pair=ab.softmax(dim=0);allp=logits.softmax(dim=0)
    return {'verifier_hidden':h[0].float().cpu().numpy(),
       'ab_logits':ab.cpu().numpy(),'ab_probabilities':pair.cpu().numpy(),
       'ab_full_vocab_probabilities':allp[[a_id,b_id]].cpu().numpy()}


def manifest(rows,sig):
    records={}
    for row in rows:
        rid=row['row_id'];p=ROOT/'data/features'/(rid+'.npz');side=p.with_suffix('.json')
        if not side.exists():continue
        meta=base.read(side);assert meta['signature_sha256']==base.model7.digest(sig)
        assert base.sha(p)==meta['npz_sha256']
        records[rid]={'npz':str(p.relative_to(ROOT)),'npz_sha256':meta['npz_sha256'],
            'json':str(side.relative_to(ROOT)),'json_sha256':base.sha(side)}
    result={'version':VERSION,'complete':len(records)==602,'completed_count':len(records),'expected_count':602,
        'records':records,'signature_sha256':base.model7.digest(sig),'source_generation_unchanged':True,
        'labels_read':False,'original_validation_or_test_parsed':False,'extra_forward_passes_per_answer':1}
    base.save(ROOT/'data/feature_manifest.json',result);return result


def run():
    tok,rows,records,sig,plans=prepare()
    if manifest(rows,sig)['complete']:
        print('ALREADY_COMPLETE_NO_GPU',flush=True);return
    _,model=base.model7.load_model()
    start=time.perf_counter();selfcheck=[]
    for j,row in enumerate(rows):
        rid=row['row_id'];p=ROOT/'data/features'/(rid+'.npz');side=p.with_suffix('.json')
        if side.exists():continue
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        arrays=forward(model,plans[rid]['input_ids'],sig['A_id'],sig['B_id'])
        assert arrays['verifier_hidden'].shape==(3584,) and all(np.isfinite(v).all() for v in arrays.values())
        if j in (0,len(rows)-1):
            repeat=forward(model,plans[rid]['input_ids'],sig['A_id'],sig['B_id'])
            assert all(np.array_equal(arrays[k],repeat[k]) for k in arrays)
            selfcheck.append({'row_id':rid,'repeat_max_abs':0.0})
        base.save_arrays(p,arrays)
        torch.cuda.synchronize()
        meta={'version':VERSION,'row_id':rid,'signature_sha256':base.model7.digest(sig),
            'plan_sha256':base.model7.digest(plans[rid]),'source_generation_sha256':records[rid][1],
            'npz_sha256':base.sha(p),'seconds':time.perf_counter()-t,'input_tokens':len(plans[rid]['input_ids']),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'label_tokens':['A=no unsupported factual claim','B=at least one unsupported factual claim'],
            'risk_score':float(arrays['ab_probabilities'][1]),
            'pair_vocab_mass':float(arrays['ab_full_vocab_probabilities'].sum()),
            'labels_read':False,'output_regenerated':False}
        base.save(side,meta)
        if (j+1)%50==0:
            manifest(rows,sig);print('EXTRACT22',j+1,602,'SECONDS',round(time.perf_counter()-start,1),flush=True)
    fm=manifest(rows,sig);assert fm['complete']
    assert base.sha(Path(__file__))==sig['code_sha256']
    base.save(ROOT/'data/extraction_checks.json',{'complete':True,'repeat_checks':selfcheck,'loop_seconds':time.perf_counter()-start,
        'signature_sha256':base.model7.digest(sig),'original_generation_changed':False,'labels_read':False})
    print('EXTRACT22_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    if args.stage=='prepare':
        _,rows,_,_,plans=prepare();print('PREPARED',len(rows),'input_tokens',min(p['input_tokens'] for p in plans.values()),max(p['input_tokens'] for p in plans.values()))
    else:run()
