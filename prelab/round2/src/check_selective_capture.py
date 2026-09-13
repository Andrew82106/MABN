"""Verify memory-saving hooks equal standard Transformers hidden states."""
import json
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,AutoTokenizer
from common import ROOT,PRELAB
from engine import forward_features,text_prefix

@torch.inference_mode()
def main():
    torch.manual_seed(984); torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(PRELAB/'models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    cfg=Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2)
    cfg._attn_implementation='sdpa'; model=Qwen2ForCausalLM(cfg).half().cuda().eval()
    prompt='Name the capital of France.'; response='Paris.'; prefix=text_prefix(tok,prompt)
    enc=tok(prefix+response,return_offsets_mapping=True,add_special_tokens=False)
    ix=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix)]
    out=model.model(torch.tensor([enc['input_ids']],device='cuda'),use_cache=False,output_hidden_states=True)
    expected_before=torch.stack([out.hidden_states[l][0,np.array(ix)-1] for l in [1,2,3,4]],dim=1).cpu().numpy()
    expected_after=torch.stack([out.hidden_states[l][0,ix] for l in [1,2,3,4]],dim=1).cpu().numpy()
    actual=forward_features(tok,model,prompt,response)
    assert np.array_equal(expected_before,actual['before']) and np.array_equal(expected_after,actual['after'])
    result={'passed':True,'model':'random tiny Qwen2 with real tokenizer, same decoder implementation','before_after_equal_standard_output_hidden_states':True,'scope':'Hook layer indexing and final normalization, not large-model prediction quality'}
    (ROOT/'results/selective_capture_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(result)

if __name__=='__main__': main()
