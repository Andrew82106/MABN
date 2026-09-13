"""One prespecified CPU width32 TCN on generator signals and MiniCheck states."""
from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F
import run_semantic_hidden as semantic
import run_sequence as v1

qa=semantic.qa;ROOT=qa.ROOT;OLD=ROOT/'results/sequence_v1';SEM=semantic.OUT
OUT=ROOT/'results/semantic_sequence_v1';FULL=ROOT/'results/sequence_full_v2'
METHOD='minicheck_hidden64_risk_tcn_w32';WIDTH=1090;SEED=20260926;BATCH=8;EPOCHS=30


def protocol():
    return {'version':'qa-semantic-sequence-v1','scope':'Fixed fit634/cal159, no official test',
        'method':METHOD,'input':'All1024 original LB heads+NLL1+already-fit MiniCheck PCA64+per-token official claim risk logit1 =1090',
        'extra_checker':True,'checker_or_PCA_refit':False,
        'architecture':'1090->32 GELU, two single residual Conv1d32 kernel3 dilation1/2 with GELU/dropout.2,32->1',
        'parameters':41153,'fusion_receptive_field_raw_tokens':7,
        'offline':'MiniCheck states already see full claim and document; fusion RF7 does not limit total accessible future information to7 tokens.',
        'seed':SEED,'epochs':EPOCHS,'batch_answers':BATCH,'cpu_threads':4,
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'all_parameters':True},
        'scaler':'Only fit raw tokens, base group/answer/lexical weights, partial_fit16384 blocks, float32transform. Punctuation stays context, loss0.',
        'weights':'Reuse frozen sequence_v1 base/loss/classfactors/labels and30answer shuffles. Sum loss139518.',
        'batch_objective':'sum(weight*BCE)*634/(actual_batch_answer_count*139518)',
        'aggregation':'Token probability ->max lexical position in each unchanged4rawBPE window ->max all eligible windows for original answer.',
        'selection':'Each epoch uses separate cal F1 thresholds, ties precision/higher cutoff; checkpoint maxminF1, windowF1,windowprecision,earlier epoch.',
        'fixed_budget':'One network,one seed,30epochs,no early stop, no additional structure based on LR ranking.',
        'comparison':'Preserve9LR, official MiniCheck scalar/fusion, old native LR/TCN and constants; no sealed-test claim.'}


class SemanticTCN(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(WIDTH,32)
        self.convolutions=nn.ModuleList([nn.Conv1d(32,32,3,padding=d,dilation=d) for d in (1,2)])
        self.dropout=nn.Dropout(.2);self.output=nn.Linear(32,1)
    forward=v1.TokenTCN.forward


def new_model():
    torch.manual_seed(SEED);return SemanticTCN().cpu()


def batch(chosen,x,index,weights=None):
    length=max(index[int(i)]['token_count'] for i in chosen)
    xx=torch.zeros(len(chosen),length,WIDTH);mask=torch.zeros(len(chosen),length)
    yy=torch.zeros(len(chosen),length);ww=torch.zeros(len(chosen),length)
    for j,i in enumerate(chosen):
        entry=index[int(i)];left,right=entry['left'],entry['right'];n=right-left
        xx[j,:n]=torch.from_numpy(np.array(x[left:right],copy=True));mask[j,:n]=1
        if weights is not None:
            assert entry['partition']=='fit' and right<=170361
            yy[j,:n]=torch.from_numpy(weights['y'][left:right].astype(np.float32))
            ww[j,:n]=torch.from_numpy(weights['loss'][left:right].astype(np.float32))
    return xx,mask,yy,ww


@torch.no_grad()
def predict(model,x,index):
    model.eval();logits=np.empty(len(x),np.float32)
    for left in range(0,len(index),BATCH):
        chosen=list(range(left,min(left+BATCH,len(index))));xx,mask,_,_=batch(chosen,x,index)
        values=model(xx,mask).numpy()
        for j,i in enumerate(chosen):
            entry=index[i];logits[entry['left']:entry['right']]=values[j,:entry['token_count']]
    assert np.isfinite(logits).all();return logits,torch.sigmoid(torch.from_numpy(logits)).numpy()


def source():
    assert not torch.cuda.is_initialized();meta,_,_,ss=semantic.source()
    assert ss==qa.read(SEM/'source_snapshot.json') and qa.read(SEM/'complete.json')['status']=='complete_development_only'
    prep=qa.read(SEM/'preparation_complete.json');fullprep=qa.read(FULL/'preparation_manifest.json')
    files=[Path(__file__),ROOT/'src/run_sequence.py',ROOT/'src/run_development.py',ROOT/'src/run_semantic_hidden.py',
        SEM/'complete.json',SEM/'preparation_complete.json',SEM/'hidden_pca.pkl',SEM/'source_snapshot.json',SEM/'token_index.json',
        SEM/'matrices/token_hidden64.npy',SEM/'matrices/token_risk_logit.npy',FULL/'preparation_manifest.json',FULL/'matrices/common1025.npy',
        OLD/'complete.json',OLD/'fit_token_weights.npz',OLD/'epoch_answer_order.npy',OLD/'token_index.json']
    for name in ('token_index.json','hidden_pca.pkl','matrices/token_hidden64.npy','matrices/token_risk_logit.npy'):
        assert qa.sha(SEM/name)==prep['files_sha256'][str(Path(name))]
    assert qa.sha(FULL/'matrices/common1025.npy')==fullprep['files_sha256'][str(Path('matrices/common1025.npy'))]
    old=qa.read(OLD/'complete.json')
    for name in ('fit_token_weights.npz','epoch_answer_order.npy','token_index.json'):assert qa.sha(OLD/name)==old['files_sha256'][name]
    index=qa.read(OLD/'token_index.json')['answers'];si=qa.read(SEM/'token_index.json')['answers']
    for a,b in zip(index,si):
        for key in ('response_id','partition','left','right','token_count'):assert a[key]==b[key]
    assert len(index)==len(si)==793
    return meta,index,{'files_sha256':{str(p.resolve()):qa.sha(p) for p in files},'test_opened':False,'GPU_used':False}


def freeze():
    assert not (OUT/'protocol.json').exists();_,index,snap=source()
    model=new_model().eval();assert sum(p.numel() for p in model.parameters())==41153
    generator=torch.Generator().manual_seed(SEED+1);x=torch.randn(2,16,WIDTH,generator=generator);mask=torch.ones(2,16);mask[0,9:]=0
    with torch.no_grad():
        p=model(x,mask);changed=x.clone();changed[0,9:]=1000
        assert torch.allclose(model(changed,mask)[0,:9],p[0,:9],atol=2e-6,rtol=1e-6)
        assert torch.allclose(model(x[:1,:9],torch.ones(1,9))[0],p[0,:9],atol=2e-6,rtol=1e-6)
        changed=x.clone();changed[1]=1000;assert torch.equal(model(changed,mask)[0],p[0])
        changed=x.clone();changed[1,:4]+=100;assert torch.equal(model(changed,mask)[1,8],p[1,8])
    # Check exact1090 input packaging, including a short answer in a padded batch.
    synthetic=np.arange(7*WIDTH,dtype=np.float32).reshape(7,WIDTH)
    toy=[{'left':0,'right':3,'token_count':3},{'left':3,'right':7,'token_count':4}]
    xx,mm,_,_=batch([0,1],synthetic,toy);assert xx.shape==(2,4,WIDTH) and np.array_equal(xx[0,:3].numpy(),synthetic[:3])
    qa.save(OUT/'PREFLIGHT.json',{'passed':True,'padding_answer_isolation_RF7_packaging':True,'parameters':41153,'no_real_fitting':True})
    qa.save(OUT/'protocol.json',protocol());qa.save(OUT/'source_snapshot.json',snap)
    qa.save(OUT/'freeze.json',{'protocol_sha256':qa.sha(OUT/'protocol.json'),'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json'),'code_sha256':qa.sha(Path(__file__))})
    print('SEMANTIC_SEQUENCE_FROZEN',flush=True)


def prepare(index):
    common=np.load(FULL/'matrices/common1025.npy',mmap_mode='r');h=np.load(SEM/'matrices/token_hidden64.npy',mmap_mode='r')
    risk=np.load(SEM/'matrices/token_risk_logit.npy',mmap_mode='r')
    assert common.shape==(213159,1025) and h.shape==(213159,64) and risk.shape==(213159,)
    with np.load(OLD/'fit_token_weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    def raw(left,right):return np.column_stack((common[left:right],h[left:right],risk[left:right,None]))
    scaler=StandardScaler()
    for left in range(0,170361,16384):
        right=min(left+16384,170361);scaler.partial_fit(raw(left,right),sample_weight=weights['base'][left:right])
    x=np.lib.format.open_memmap(OUT/'standardized_token_features.npy',mode='w+',dtype=np.float32,shape=(213159,WIDTH))
    for left in range(0,len(x),16384):
        right=min(left+16384,len(x));x[left:right]=scaler.transform(raw(left,right)).astype(np.float32)
    x.flush();(OUT/'scaler.pkl').write_bytes(pickle.dumps(scaler,protocol=5));order=np.load(OLD/'epoch_answer_order.npy')
    assert order.shape==(30,634) and np.isclose(weights['loss'].sum(),139518)
    qa.save(OUT/'preparation_complete.json',{'files_sha256':{name:qa.sha(OUT/name) for name in ('standardized_token_features.npy','scaler.pkl')},
        'weights_sha256':qa.sha(OLD/'fit_token_weights.npz'),'shuffle_sha256':qa.sha(OLD/'epoch_answer_order.npy'),'test_opened':False})
    return x,weights,order


def train():
    assert qa.read(OUT/'protocol.json')==protocol() and not (OUT/'started.json').exists()
    assert qa.sha(Path(__file__))==qa.read(OUT/'freeze.json')['code_sha256']
    meta,index,snap=source();assert snap==qa.read(OUT/'source_snapshot.json')
    qa.save(OUT/'started.json',{'utc':time.time(),'freeze_sha256':qa.sha(OUT/'freeze.json')})
    clock=time.perf_counter();x,weights,order=prepare(index);preparation_seconds=time.perf_counter()-clock
    model=new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    directory=OUT/METHOD;directory.mkdir(exist_ok=True);history=[];files=[];best=None;selected=None
    for epoch in range(EPOCHS):
        start=time.perf_counter();model.train()
        for left in range(0,634,BATCH):
            chosen=order[epoch,left:left+BATCH];xx,mask,yy,ww=batch(chosen,x,index,weights)
            optimizer.zero_grad(set_to_none=True);loss=(F.binary_cross_entropy_with_logits(model(xx,mask),yy,reduction='none')*ww).sum()*(634/(len(chosen)*139518))
            assert torch.isfinite(loss);loss.backward();optimizer.step()
        training_seconds=time.perf_counter()-start;logits,p=predict(model,x,index)
        arrays,thresholds,metrics,bce=v1.evaluate(meta,index,logits,p,weights);arrays.update(token_logits=logits,token_scores=p)
        prefix=f'epoch_{epoch+1:03d}';np.savez_compressed(directory/(prefix+'_scores.npz'),**arrays)
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
            'epoch':epoch+1,'method':METHOD,'seed':SEED,'device':'cpu','protocol_sha256':qa.sha(OUT/'protocol.json'),
            'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json')},directory/(prefix+'.pt'))
        w,a=thresholds['window'],thresholds['answer'];key=[min(w['f1'],a['f1']),w['f1'],w['precision'],-(epoch+1)]
        entry={'method':METHOD,'epoch':epoch+1,'parameters':41153,'thresholds':thresholds,'selection_key':key,'metrics':metrics,
            'full_fit_weighted_BCE':bce,'epoch_training_seconds':training_seconds,'epoch_total_seconds':time.perf_counter()-start,
            'scores_sha256':qa.sha(directory/(prefix+'_scores.npz')),'model_sha256':qa.sha(directory/(prefix+'.pt')),'test_opened':False}
        qa.save(directory/(prefix+'.json'),entry);history.append(entry)
        files.extend(str((directory/(prefix+suffix)).relative_to(OUT)) for suffix in ('_scores.npz','.pt','.json'))
        if best is None or key>best:best=key;selected=entry
        qa.save(OUT/'progress.json',{'epoch':epoch+1,'selected':selected,'seconds':time.perf_counter()-clock})
        print('SEMANTIC_SEQUENCE_EPOCH',epoch+1,'BCE',round(bce,6),'cal_w',round(w['f1'],6),'cal_a',round(a['f1'],6),flush=True)
    assert source()[-1]==snap and not torch.cuda.is_initialized()
    qa.save(OUT/'summary.json',{'selected':{METHOD:selected},'all_epochs':{METHOD:history},'preparation_seconds':preparation_seconds,
        'wall_seconds':time.perf_counter()-clock,'test_opened':False,'GPU_used':False,'extra_checker':True,'calibration_selection_optimistic':True})
    files.extend(['summary.json','protocol.json','freeze.json','source_snapshot.json','preparation_complete.json','PREFLIGHT.json'])
    qa.save(OUT/'complete.json',{'status':'complete_development_only','files_sha256':{name:qa.sha(OUT/name) for name in files},'test_opened':False,'GPU_used':False})
    print('SEMANTIC_SEQUENCE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','train']);args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):{'freeze':freeze,'train':train}[args.stage]()
