"""Official LettuceDetect v0.1.6 with resource-only training adaptation.

Uses the unmodified upstream RagTruthDataset, native two-logit CE and AdamW.
Eight-answer batches are split into microbatches, weighted by native valid token
count so accumulation reproduces the upstream batch CE. No local risk weighting.
"""
import argparse,contextlib,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification
import common as c
sys.path.insert(0,str(c.HERE/'official'/'LettuceDetect_v1'))
from lettucedetect.datasets.ragtruth import RagTruthDataset
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthSample
from lettucedetect.models.trainer import Trainer

OUT=c.HERE/'lettuce_v1_run'
BASE=c.PROJECT/'artifacts/models/initial/ModernBERT-large'
SEED=20260928

def prepare():
    assert not torch.cuda.is_initialized()
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    ss=c.read(c.HERE/'LETTUCE_V1_SOURCE_MANIFEST.json')
    for f,m in ss['files'].items():assert c.sha(c.HERE/'official/LettuceDetect_v1'/f)==m['sha256']
    summaries={}; bindings={}
    for split in ['fit','validation','test']:
        aa,gg=c.inputs(split)
        gold={r['response_id']:r for r in c.rows(c.DATA/split/'answers.jsonl')} if split!='test' else {}
        rr=[];shifted=0
        for a in aa:
            rid=a['response_id'];answer=a['original_response'];prompt=a['released_prompt']
            spans=[{'start':x['start'],'end':x['end'],'label':'hallucination'} for x in gold.get(rid,{}).get('released_hallucinations',[]) if x.get('valid',True)]
            sample=RagTruthSample(prompt,answer,spans,'train' if split=='fit' else 'dev','QA')
            # Preserve official sample encoding and label algorithm byte-for-byte.
            native=RagTruthDataset([sample],tok)[0]
            enc,_,offset,answer_start=RagTruthDataset.prepare_tokenized_input(tok,prompt,answer)
            untruncated=tok(prompt,answer,return_offsets_mapping=True,truncation=False)
            assert enc['input_ids'][0].tolist()==untruncated['input_ids'],'No truncation permitted'
            seq=untruncated.sequence_ids(); answer_offsets=[list(o) if seq[i]==1 else [-1,-1] for i,o in enumerate(offset.tolist())]
            assert seq[answer_start]==1 and len(enc['input_ids'][0])<=4096
            first_offset=offset[answer_start,0].item();shifted+=int(first_offset!=0)
            if split!='test':
                for j,(lo,hi) in enumerate(answer_offsets):
                    expected=-100 if j<answer_start else int(any(hi>s['start'] and lo<s['end'] for s in spans)) if hi>lo>=0 else 0
                    assert native['labels'][j].item()==expected, (rid,j,lo,hi,'native span mismatch')
            r={'response_id':rid,'input_ids':native['input_ids'].tolist(),'attention_mask':native['attention_mask'].tolist(),
               'answer_start':answer_start,'native_offsets':answer_offsets,'original_token_count':len(gg[rid]['lexical_mask']),
               'mapping':c.mapping(answer,gg[rid]['response_token_offsets'],answer_offsets),
               'response_sha256':__import__('hashlib').sha256(answer.encode()).hexdigest()}
            if split!='test':r['labels']=native['labels'].tolist()
            rr.append(r)
        c.write_rows(OUT/f'{split}_prepared.jsonl',rr)
        summaries[split]={'answers':len(rr),'max_input_tokens':max(len(r['input_ids']) for r in rr),'nonzero_first_answer_offset':shifted}
        bindings.update(c.source_hashes(split))
        if split!='test':bindings[str(c.DATA/split/'answers.jsonl')]=c.sha(c.DATA/split/'answers.jsonl')
    c.save(OUT/'PREPARED.json',{'summary':summaries,'sources':bindings,'generic_manifest_sha256':c.sha(BASE/'download_manifest.json'),
      'source_manifest_sha256':c.sha(c.HERE/'LETTUCE_V1_SOURCE_MANIFEST.json'),'test_labels_read':False,
      'protocol':{'method':'LettuceDetect v0.1.6 same-domain original-structure retraining','source_revision':ss['revision'],
       'base':str(BASE),'input':'official predict_prompt: released_prompt paired with original_response; no added abstention instruction',
       'labels':'unmodified upstream RagTruthDataset; prompt ignored, answer including trailing SEP classified; no lexical/group/answer reweighting',
       'optimizer':'official Trainer AdamW lr=1e-5/default weight_decay=.01; no scheduler or clipping',
       'epochs':6,'effective_batch':8,'microbatch':1,'seed':SEED,
       'precision':'FP32 parameters/gradients/Adam; BF16 CUDA forward; nonreentrant checkpointing',
       'selection':'shared validation no_hint joint key, earlier epoch ties; max/mean adapters selected on validation; test untouched'}})
    print('PREPARED',summaries,flush=True)

def load_model():
    torch.manual_seed(SEED)
    model=AutoModelForTokenClassification.from_pretrained(BASE,num_labels=2,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',reference_compile=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model

def autocast(device):return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()
def batch(record,device,labels=False):
    out={k:torch.tensor([record[k]],device=device) for k in ['input_ids','attention_mask']}
    if labels:out['labels']=torch.tensor([record['labels']],device=device)
    return out

@torch.no_grad()
def predict_records(model,rr,device='cuda'):
    model.eval();preds=[];tp=fp=fn=0
    for i,r in enumerate(rr):
        with autocast(device):p=torch.softmax(model(**batch(r,device)).logits.float(),-1)[0,:,1].cpu().numpy()
        if 'labels' in r:
            y=np.asarray(r['labels']);valid=y!=-100;pos=y==1;hat=p>=.5
            tp+=int((hat&pos&valid).sum());fp+=int((hat&~pos&valid).sum());fn+=int((~hat&pos&valid).sum())
        preds.append({'response_id':r['response_id'],'token_scores':c.mapped(p,r['mapping'],r['original_token_count']),
             'native_token_scores':p[r['answer_start']:].tolist(),'native_answer_offsets':r['native_offsets'][r['answer_start']:],
             'response_sha256':r['response_sha256']})
        if (i+1)%100==0:print('PREDICT',i+1,len(rr),flush=True)
    return preds,{'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/max(2*tp+fp+fn,1)}

def cpu_test():
    assert not torch.cuda.is_initialized()
    prepare()
    allrr=[r for sp in ['fit','validation','test'] for r in c.rows(OUT/f'{sp}_prepared.jsonl')]
    for r in allrr:
        p=c.mapped(np.ones(len(r['input_ids'])),r['mapping'],r['original_token_count'])
        assert all(0<=x<=1.000001 for x in p)
        assert len(p)==r['original_token_count']
    # Native-batch CE equals token-weighted microbatch CE, not answer-mean BCE.
    gen=torch.Generator().manual_seed(1);z=torch.randn(2,5,2,generator=gen,requires_grad=True)
    y=torch.tensor([[0,1,-100,1,0],[1,-100,-100,-100,0]])
    ce=torch.nn.functional.cross_entropy(z.reshape(-1,2),y.reshape(-1))
    micro=sum(torch.nn.functional.cross_entropy(z[i],y[i])*int((y[i]!=-100).sum()) for i in range(2))/int((y!=-100).sum())
    assert torch.allclose(ce,micro,atol=1e-7)
    g1=torch.autograd.grad(ce,z,retain_graph=True)[0];g2=torch.autograd.grad(micro,z)[0]
    assert torch.allclose(g1,g2,atol=1e-7)
    p1=torch.nn.Parameter(z.detach().clone());p2=torch.nn.Parameter(z.detach().clone())
    o1=torch.optim.AdamW([p1],lr=1e-5);o2=torch.optim.AdamW([p2],lr=1e-5)
    p1.grad=g1;p2.grad=g2;o1.step();o2.step();assert torch.allclose(p1,p2,atol=1e-7)
    c.save(OUT/'CPU_CHECK.json',{'passed':True,'native_dataset_exercised_all_answers':len(allrr),'microbatch_ce_equivalence':True,'microbatch_gradient_and_adamw_update_equivalence':True,'native_span_labels_equivalence':True,'GPU_used':False,'test_labels_read':False})
    print('CPU_CHECK_OK',flush=True)

def smoke():
    assert c.read(OUT/'CPU_CHECK.json')['passed'];torch.set_num_threads(4)
    model=load_model().cuda();model.train()
    # Instantiate the original Trainer, preserving its AdamW construction.
    trainer=Trainer(model,AutoTokenizer.from_pretrained(BASE,local_files_only=True),None,None,device=torch.device('cuda'))
    for g in trainer.optimizer.param_groups:g['foreach']=False
    r=max(c.rows(OUT/'fit_prepared.jsonl'),key=lambda r:len(r['input_ids']))
    start=time.time()
    with autocast('cuda'):loss=model(**batch(r,'cuda',True)).loss
    loss.backward();trainer.optimizer.step();torch.cuda.synchronize()
    assert torch.isfinite(loss)
    c.save(OUT/'GPU_SMOKE.json',{'passed':True,'tokens':len(r['input_ids']),'loss':float(loss),'elapsed_seconds':time.time()-start,
      'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'test_labels_read':False})
    print('GPU_SMOKE_OK',flush=True)

def atomic_checkpoint(path,state):
    temp=path.with_suffix('.pending');torch.save(state,temp);temp.replace(path)

def train():
    assert c.read(OUT/'GPU_SMOKE.json')['passed'];torch.set_num_threads(4)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    fit=c.rows(OUT/'fit_prepared.jsonl');val=c.rows(OUT/'validation_prepared.jsonl')
    model=load_model().cuda()
    trainer=Trainer(model,AutoTokenizer.from_pretrained(BASE,local_files_only=True),None,None,device=torch.device('cuda'))
    optimizer=trainer.optimizer
    for g in optimizer.param_groups:g['foreach']=False
    history=[];best=None;first_epoch=1
    latest=OUT/'resume.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];first_epoch=state['epoch']+1
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
    start=time.time()
    for epoch in range(first_epoch,7):
        model.train();order=np.random.default_rng(SEED+epoch).permutation(len(fit));loss_total=0.;seen=0
        for step in range(0,len(fit),8):
            chunk=[fit[i] for i in order[step:step+8]];denom=sum(sum(y!=-100 for y in r['labels']) for r in chunk)
            optimizer.zero_grad(set_to_none=True)
            for r in chunk:
                n=sum(y!=-100 for y in r['labels'])
                with autocast('cuda'):loss=model(**batch(r,'cuda',True)).loss
                assert torch.isfinite(loss)
                (loss*(n/denom)).backward();loss_total+=float(loss.detach())*n;seen+=n
            optimizer.step()
            if step%160==0:
                c.save(OUT/'STATUS.json',{'stage':'training','epoch':epoch,'answers':min(step+8,len(fit)),'total':len(fit),'elapsed_seconds':time.time()-start})
                print('TRAIN',epoch,step+8,len(fit),flush=True)
        preds,native=predict_records(model,val);report=c.val_reports(preds);key=report['evaluation']['selection_key']
        entry={'epoch':epoch,'train_native_ce':loss_total/seen,'native_validation_argmax':native,**report}
        history.append(entry);c.write_rows(OUT/f'validation_epoch{epoch}.jsonl',preds);c.save(OUT/'HISTORY.json',history)
        if best is None or key>best['evaluation']['selection_key']:
            best=entry;atomic_checkpoint(OUT/'best.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch})
            c.write_rows(OUT/'validation_predictions.jsonl',preds);c.save(OUT/'SELECTED.json',best)
        atomic_checkpoint(latest,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
            'epoch':epoch,'history':history,'best':best,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        print('EPOCH',epoch,'VAL',key,flush=True)
    c.save(OUT/'TRAIN_COMPLETE.json',{'epochs':6,'selected_epoch':best['epoch'],'checkpoint_sha256':c.sha(OUT/'best.pt'),
        'validation_predictions_sha256':c.sha(OUT/'validation_predictions.jsonl'),'test_labels_read':False,'elapsed_seconds':time.time()-start})

def predict(split):
    assert split in ['validation','test'];torch.set_num_threads(4)
    receipt=c.read(OUT/'TRAIN_COMPLETE.json');assert c.sha(OUT/'best.pt')==receipt['checkpoint_sha256']
    model=load_model();state=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);del state;model=model.cuda()
    pp,_=predict_records(model,c.rows(OUT/f'{split}_prepared.jsonl'))
    path=OUT/f'{split}_predictions.jsonl';c.write_rows(path,pp)
    c.save(OUT/f'{split}_PREDICTION_RECEIPT.json',{'method':'LettuceDetect-v1','split':split,'rows':len(pp),'checkpoint_sha256':receipt['checkpoint_sha256'],
      'predictions_sha256':c.sha(path),'input_sha256':c.sha(OUT/f'{split}_prepared.jsonl'),'source_input_sha256':c.source_hashes(split),
      'script_sha256':c.sha(__file__),'official_source_manifest_sha256':c.sha(c.HERE/'LETTUCE_V1_SOURCE_MANIFEST.json'),'test_labels_read':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu-test','smoke','train','predict']);p.add_argument('--split',choices=['validation','test'],default='test');a=p.parse_args()
    {'prepare':prepare,'cpu-test':cpu_test,'smoke':smoke,'train':train,'predict':lambda:predict(a.split)}[a.stage]()
if __name__=='__main__':main()
