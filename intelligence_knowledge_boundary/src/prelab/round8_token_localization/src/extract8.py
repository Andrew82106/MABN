"""Extract causal token states and reuse exact frozen native token features."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
R7=ROOT.parent/'round7_evidence_grounding'
sys.path.insert(0,str(R7/'src'))
import model7


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def save(p,j):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)


@torch.inference_mode()
def hidden_states(model,ids,positions):
    captured={}
    def hook(module,args,result):
        value=result[0] if isinstance(result,tuple) else result
        captured[21]=value[0,positions].detach().float().cpu().numpy()
    handle=model.model.layers[20].register_forward_hook(hook)
    try:
        output=model.model(input_ids=ids,use_cache=False,output_hidden_states=False)
        captured[28]=output.last_hidden_state[0,positions].detach().float().cpu().numpy()
    finally:
        handle.remove()
    return captured


def main():
    area=ROOT/'data/token_features';area.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((R7/'results/freeze.json').read_text(encoding='utf-8'))
    assert sha(R7/'data/inputs.jsonl')==freeze['data_sha256']['data/inputs.jsonl']
    rows=[r for r in readl(R7/'data/inputs.jsonl') if r['split'] in ('validation','test','external_test')]
    assert len(rows)==260
    signature={'extractor_sha256':sha(Path(__file__)),'source_model_code_sha256':sha(R7/'src/model7.py'),
               'r7_freeze_sha256':sha(R7/'results/freeze.json'),'layers':[21,28],
               'state_timing':'post-read current original generated token; layer21 before final norm, layer28 after final norm'}
    tok,model=model7.load_model()
    completed=[]
    for index,row in enumerate(rows):
        rid=row['row_id'];genpath=R7/'data/generation_records'/(rid+'.json')
        assert sha(genpath)==freeze['data_sha256'][str(genpath.relative_to(R7)).replace('\\','/')]
        gen=json.loads(genpath.read_text(encoding='utf-8'))
        source_arrays={stage:sha(R7/'data'/stage/(rid+'.npz')) for stage in ('features','attention','lumina')}
        assert all(h==freeze['data_sha256'][f'data/{stage}/{rid}.npz'] for stage,h in source_arrays.items())
        expected={**signature,'row_id':rid,'source_generation_sha256':sha(genpath),
                  'source_r7_arrays_sha256':source_arrays}
        target=area/(rid+'.json');npz=target.with_suffix('.npz')
        if target.exists():
            record=json.loads(target.read_text(encoding='utf-8'))
            assert all(record[k]==v for k,v in expected.items())
            assert record['arrays_sha256']==sha(npz)
            completed.append(record);print('CACHE',index+1,len(rows),rid,flush=True);continue
        prefix=model7.chat_ids(tok,row['prompt'],row['system']);assert prefix==gen['input_token_ids']
        answer=gen['response_token_ids'];offsets=model7.token_offsets(tok,answer,gen['response'])
        assert np.array_equal(offsets,gen['response_token_offsets'])
        ids=torch.tensor([prefix+answer],device='cuda')
        positions=list(range(len(prefix),len(prefix)+len(answer)))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        states=hidden_states(model,ids,positions)
        arrays={'token_indices':np.arange(len(answer),dtype=np.int32),'token_ids':np.asarray(answer,dtype=np.int64),
                'token_start':offsets[:,0],'token_end':offsets[:,1],'response_token_offsets':offsets,
                'hidden_21':states[21],'hidden_28':states[28]}
        align=[]
        with np.load(R7/'data/features'/(rid+'.npz'),allow_pickle=False) as old:
            for j,item in enumerate(gen['items']):
                assert str(old['item_ids'][j])==item['item_id']
                char=item['last_content_character']
                k=[n for n,(a,b) in enumerate(offsets) if a<=char<b][-1]
                diff={str(layer):float(np.max(np.abs(old[f'hidden_{layer}'][j]-states[layer][k]))) for layer in (21,28)}
                assert all(np.allclose(old[f'hidden_{l}'][j],states[l][k],rtol=2e-4,atol=2e-4) for l in (21,28)),diff
                align.append({'item_id':item['item_id'],'token_index':k,'layer_max_abs_difference':diff})
            arrays['mean_nll']=old['token_nll'].copy();arrays['mean_entropy']=old['token_entropy'].copy()
            assert np.array_equal(old['response_token_offsets'],offsets)
        with np.load(R7/'data/attention'/(rid+'.npz'),allow_pickle=False) as old:
            for key,source in [('lookback_features','token_lookback'),('redeep_ecs','token_redeep_ecs'),('redeep_pks','token_redeep_pks')]:
                arrays[key]=old[source].copy()
            assert np.array_equal(old['response_token_offsets'],offsets)
        with np.load(R7/'data/lumina'/(rid+'.npz'),allow_pickle=False) as old:
            arrays['lumina_score']=old['token_lumina_score'].copy()
            assert np.array_equal(old['response_token_offsets'],offsets)
        checks=[]
        # Bounded causal checks at beginning/middle/end on one response of each split.
        if not any(r['split']==row['split'] for r in completed):
            for k in sorted({0,len(answer)//2,len(answer)-1}):
                short=hidden_states(model,ids[:,:len(prefix)+k+1],[len(prefix)+k])
                cos={str(l):float(np.dot(states[l][k],short[l][0])/(np.linalg.norm(states[l][k])*np.linalg.norm(short[l][0])+1e-12)) for l in (21,28)}
                # Quantized matmuls can differ with sequence shape. Preserve the
                # truncated-prefix diagnostic, and independently test causality
                # by changing only future tokens at the identical input shape.
                perturbed=ids.clone()
                if len(prefix)+k+1<ids.shape[1]:
                    perturbed[:,len(prefix)+k+1:]=(perturbed[:,len(prefix)+k+1:]+1)%model.config.vocab_size
                alternate=hidden_states(model,perturbed,[len(prefix)+k])
                future_diff={str(l):float(np.max(np.abs(states[l][k]-alternate[l][0]))) for l in (21,28)}
                assert max(future_diff.values())==0.,future_diff
                checks.append({'token_index':k,'prefix_includes_current_token':True,'layer_cosine':cos,
                               'same_shape_future_perturbation_max_abs_difference':future_diff,
                               'prefix_shape_note':'NF4/bfloat16 numerical differences are retained; causal invariance checked with fixed shape.'})
        assert all(v.shape[0]==len(answer) and np.isfinite(v).all() for v in arrays.values())
        torch.cuda.synchronize()
        metadata={**expected,'split':row['split'],'question_id':row['question_id'],'items':gen['items'],
                  'response':gen['response'],'tokens':len(answer),'r7_last_item_alignment':align,
                  'causal_prefix_checks':checks,'seconds':time.perf_counter()-started,
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                  'labels_or_detector_scores_read':False}
        temp=npz.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temp.replace(npz);metadata['arrays_sha256']=sha(npz);save(target,metadata)
        completed.append(metadata)
        print('DONE',index+1,len(rows),rid,'seconds',round(metadata['seconds'],3),flush=True)
    save(area/'manifest.json',{'status':'complete','expected_rows':260,'completed_rows':len(completed),
          'items':sum(len(r['items']) for r in completed),'tokens':sum(r['tokens'] for r in completed),
          'seconds':sum(r['seconds'] for r in completed),'signature':signature,
          'record_files':{r['row_id']+'.json':sha(area/(r['row_id']+'.json')) for r in completed},
          'all_original_last_item_states_match':True,'no_future_answer_information':True})
    print('TOKEN_EXTRACTION_COMPLETE',flush=True)


if __name__=='__main__':main()
