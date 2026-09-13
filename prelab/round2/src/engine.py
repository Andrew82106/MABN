"""Frozen local models; causal token alignment and efficient lookback features."""
import math
import bisect
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import ROOT,PRELAB
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

MODELS={'small':PRELAB/'models/Qwen2.5-0.5B-Instruct','large':PRELAB/'models/Qwen2.5-7B-Instruct-bnb-4bit'}

def load_model(name):
    torch.set_num_threads(6)
    tok=AutoTokenizer.from_pretrained(MODELS[name],local_files_only=True)
    args={'local_files_only':True,'trust_remote_code':False,'attn_implementation':'sdpa',
          'torch_dtype':torch.bfloat16 if name=='large' else torch.float16}
    if name=='large': args['device_map']={'':'cuda:0'}
    model=AutoModelForCausalLM.from_pretrained(MODELS[name],**args)
    if name=='small': model=model.cuda()
    model.eval()
    print(f'MODEL LOADED {name}: layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}, GPU allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB',flush=True)
    return tok,model

def text_prefix(tok,prompt):
    return tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)

class LookbackCapture:
    """Compute only answer query rows, leaving actual SDPA output unchanged.

    Ratio of mean attention to prompt vs mean attention to prior generated tokens,
    matching Lookback-Lens extraction. First token has no generated prefix => ratio 1.
    """
    def __init__(self,positions,start):
        self.positions=positions; self.start=start; self.values=[]
    def __enter__(self):
        self.original=torch.nn.functional.scaled_dot_product_attention
        def wrapper(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,**kwargs):
            result=self.original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,**kwargs)
            assert q.shape[0]==1 and q.shape[1]==k.shape[1]
            ix=torch.tensor(self.positions,device=q.device)
            logits=(q[:,:,ix,:].float()@k.transpose(-1,-2).float())*(kwargs.get('scale') or 1/math.sqrt(q.shape[-1]))
            if attn_mask is not None:
                mask=attn_mask[...,ix,:]
                if mask.dtype==torch.bool: logits=logits.masked_fill(~mask,-float('inf'))
                else: logits=logits+mask
            if is_causal:
                logits=logits.masked_fill(torch.arange(k.shape[-2],device=q.device)[None,:]>ix[:,None],-float('inf'))
            a=logits.softmax(-1)[0]
            context=a[:,:,:self.start].sum(-1)/self.start
            generated=a[:,:,self.start:].sum(-1)/(ix-self.start+1).clamp(min=1)
            ratio=context/(context+generated+1e-12)
            self.values.append(ratio.T.half().cpu().numpy())
            return result
        torch.nn.functional.scaled_dot_product_attention=wrapper
        return self
    def __exit__(self,*args): torch.nn.functional.scaled_dot_product_attention=self.original

def original_token_offsets(tok,ids,text):
    decoder={v:k for k,v in bytes_to_unicode().items()}
    pieces=[bytes(decoder[c] for c in tok.convert_ids_to_tokens(int(i))) for i in ids]
    assert b''.join(pieces)==text.encode('utf8'),'Original token bytes do not match displayed response'
    boundaries=[0]
    for char in text: boundaries.append(boundaries[-1]+len(char.encode('utf8')))
    offsets=[]; cursor=0
    for piece in pieces:
        end=cursor+len(piece)
        offsets.append((bisect.bisect_right(boundaries,cursor)-1,bisect.bisect_left(boundaries,end)))
        cursor=end
    return offsets

@torch.inference_mode()
def forward_features(tok,model,prompt,response,lookback=True,response_token_ids=None):
    prefix=text_prefix(tok,prompt)
    if response_token_ids is None:
        enc=tok(prefix+response,add_special_tokens=False,return_offsets_mapping=True)
        positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix)]
        offsets=[(max(0,enc['offset_mapping'][i][0]-len(prefix)),enc['offset_mapping'][i][1]-len(prefix)) for i in positions]
    else:
        prefix_ids=tok(prefix,add_special_tokens=False)['input_ids']
        enc={'input_ids':prefix_ids+list(response_token_ids)}
        positions=list(range(len(prefix_ids),len(enc['input_ids'])))
        offsets=original_token_offsets(tok,response_token_ids,response)
    assert positions and positions[0]>0
    assert len(enc['input_ids'])<=4096,('context too long',len(enc['input_ids']))
    ix=torch.tensor(positions,device='cuda')
    ids=torch.tensor([enc['input_ids']],device='cuda')
    layers=[round(model.config.num_hidden_layers*f) for f in [.25,.5,.75,1.]]
    # Retain the four requested layers only. Keeping all 28 layers can make an
    # otherwise fitting 7B model spill into Windows shared GPU memory.
    retained={}; hooks=[]
    for layer in layers[:-1]:
        def save_layer(module,inputs,output,key=layer):
            retained[key]=output[0] if isinstance(output,tuple) else output
        hooks.append(model.model.layers[layer-1].register_forward_hook(save_layer))
    try:
        if lookback:
            with LookbackCapture((ix-1).tolist(),positions[0]) as captured:
                out=model.model(ids,use_cache=False,output_hidden_states=False)
            look=np.concatenate(captured.values,axis=1)
        else:
            out=model.model(ids,use_cache=False,output_hidden_states=False); look=None
    finally:
        for hook in hooks: hook.remove()
    retained[layers[-1]]=out.last_hidden_state
    before=torch.stack([retained[l][0,ix-1] for l in layers],dim=1).half().cpu().numpy()
    after=torch.stack([retained[l][0,ix] for l in layers],dim=1).half().cpu().numpy()
    nll=[]; entropy=[]; margins=[]
    for pos in ix.split(16):
        logits=model.lm_head(out.last_hidden_state[0,pos-1]).float()
        lp=logits.log_softmax(-1)
        nll.extend((-lp.gather(1,ids[0,pos,None])).flatten().cpu().tolist())
        entropy.extend((-(lp.exp()*lp).sum(-1)).cpu().tolist())
        top=logits.topk(2,dim=-1).values; margins.extend((top[:,0]-top[:,1]).cpu().tolist())
    result={'before':before,'after':after,'nll':np.array(nll,np.float32),'entropy':np.array(entropy,np.float32),
            'margin':np.array(margins,np.float32),'layers':np.array(layers),
            'offsets':np.array(offsets,np.int32),
            'token_ids':ids[0,ix].cpu().numpy()}
    if look is not None: result['lookback']=look
    return result

def extract(tok,model,row,contrast=False):
    f=forward_features(tok,model,row['prompt'],row['response'],response_token_ids=row.get('generation_token_ids'))
    if contrast:
        p=row.get('no_evidence_prompt')
        if p is None:
            assert row.get('evidence') and row['evidence'] in row['prompt']
            p=row['prompt'].replace(row['evidence'],'[Evidence not supplied]')
        g=forward_features(tok,model,p,row['response'],lookback=False,response_token_ids=row.get('generation_token_ids'))
        assert np.array_equal(f['token_ids'],g['token_ids']),'Context comparison token alignment changed'
        a=f['before'].astype(np.float32); b=g['before'].astype(np.float32)
        cosine=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-8)
        normdiff=np.linalg.norm(a-b,axis=-1)/(np.linalg.norm(a,axis=-1)+1e-8)
        f['support']=np.column_stack([f['nll'],g['nll'],f['nll']-g['nll'],f['entropy'],g['entropy'],f['margin'],cosine,normdiff]).astype(np.float32)
    return f

@torch.inference_mode()
def generate(tok,model,prompt,seed,max_tokens,return_trace=False):
    torch.manual_seed(seed)
    inputs=tok(text_prefix(tok,prompt),return_tensors='pt',add_special_tokens=False).to('cuda')
    output=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=True,temperature=.5,top_p=.95,
                          pad_token_id=tok.eos_token_id,use_cache=True)
    gen=output[0,inputs['input_ids'].shape[1]:]
    result=(tok.decode(gen,skip_special_tokens=True),len(gen)>=max_tokens)
    if return_trace: return (*result,[int(x) for x in gen.tolist() if x not in tok.all_special_ids])
    return result
