"""7B implementation check on two old TRAIN rows; no new test data or labels."""
from pathlib import Path
import copy
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R7 = ROOT.parent/'round7_evidence_grounding'
sys.path.insert(0,str(R7/'src'))
import model7
import binding9


def readl(p):
    return [json.loads(x) for x in p.read_text('utf-8').splitlines() if x.strip()]


@torch.inference_mode()
def main():
    target = ROOT/'results/gpu_implementation_audit.json'
    assert not target.exists()
    rows = {r['row_id']:r for r in readl(R7/'data/inputs.jsonl')}
    old = [g for g in readl(R7/'data/generated.jsonl') if g['split']=='train' and len(g['response_token_ids'])>=12]
    old.sort(key=lambda g:len(g['input_token_ids'])+len(g['response_token_ids']))
    chosen=[old[0],old[-1]]
    tok,model=model7.load_model(); checks=[]
    for g in chosen:
        row=rows[g['row_id']]; vis=binding9.visible_only(row)
        torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        arrays,meta=binding9.extract_binding_features(tok,model,vis,g)
        p=len(g['input_token_ids']); n=len(g['response_token_ids']); pivot=n//2
        altered=copy.deepcopy(g)
        altered['response_token_ids']=g['response_token_ids'][:pivot+1]+[tok.encode(' X',add_special_tokens=False)[0]]*(n-pivot-1)
        altered['response']=tok.decode(altered['response_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        altered['response_token_offsets']=model7.token_offsets(tok,altered['response_token_ids'],altered['response']).tolist()
        later,_=binding9.extract_binding_features(tok,model,vis,altered)
        deltas={k:float(np.max(np.abs(arrays[k][:pivot+1]-later[k][:pivot+1]))) for k in ['hidden_21','hidden_28','lookback_features','binding_features','token_nll','token_entropy']}
        assert all(v <= 1e-5 for v in deltas.values()),deltas
        ids=torch.tensor([g['input_token_ids']+g['response_token_ids']],device='cuda')
        capture={}
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            capture['hidden_21']=h[0,p:].float().cpu().numpy()
        handle=model.model.layers[20].register_forward_hook(hook)
        try:
            output=model.model(input_ids=ids,use_cache=False,output_attentions=False,output_hidden_states=False)
        finally:
            handle.remove()
        capture['hidden_28']=output.last_hidden_state[0,p:].float().cpu().numpy()
        hidden={k:float(np.max(np.abs(arrays[k]-v))) for k,v in capture.items()}
        assert all(v <= 1e-5 for v in hidden.values())
        oldatt=np.load(R7/'data/attention'/(g['row_id']+'.npz'),allow_pickle=False)
        lb=float(np.max(np.abs(arrays['lookback_features']-oldatt['token_lookback'])))
        assert lb<=1e-5,lb
        checks.append({'old_training_row_id':g['row_id'],'input_tokens':p,'response_tokens':n,'future_pivot':pivot,
                       'same_shape_future_max_abs':deltas,'independent_hidden_max_abs':hidden,'old_global_lookback_max_abs':lb,
                       'core_seconds':meta['seconds'],'peak_allocated_gib':meta['peak_allocated_gib'],
                       'audit_seconds':time.perf_counter()-started})
        print('GPU_IMPLEMENTATION_CHECKED',g['row_id'],round(meta['seconds'],3),flush=True)
    target.write_text(json.dumps({'passed':True,'scope':'Shortest and longest old training rows, no detector labels/scores/new test data read','checks':checks,'model_config_sha256':model7.runtime_signature()['model_config_hash']},indent=2)+'\n','utf-8')


if __name__=='__main__':main()
