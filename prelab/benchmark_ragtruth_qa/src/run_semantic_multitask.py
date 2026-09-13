"""One fixed auxiliary-type-supervision control; original binary risk stays primary."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F
import run_semantic_sequence as baseline

qa=baseline.qa;ROOT=qa.ROOT;BASE=baseline.OUT;OLD=baseline.OLD
OUT=ROOT/'results/semantic_multitask_v1';METHOD='semantic_tcn_aux_types_w32'
TYPE={'Evident Baseless Info':0,'Subtle Baseless Info':0,'Evident Conflict':1,'Subtle Conflict':1}
SEED=20260926;AUX_SEED=20261001;EPOCHS=30;BATCH=8;MASS=139518


def protocol():
    return {'version':'qa-semantic-auxiliary-types-v1','method':METHOD,'scope':'Frozen fit634/cal159 only; no official test',
        'input':'Exact standardized1090 token inputs of semantic_sequence_v1; no feature/PCA/scaler/checker refit',
        'architecture':'Same32width/RF7 trunk and binary risk head; extra Linear32->2 auxiliary independent BCE heads.',
        'parameters':41219,'main_seed':SEED,'aux_seed':AUX_SEED,'auxiliary_rng':'fork_rng CPU isolate, preserve post-main-init RNG state',
        'types':TYPE,'overlap':'Two binary auxiliary labels may both be1 on overlapping spans; their OR must equal unchanged raw lexical risk_mask.',
        'label_mapping':'Use frozen span_token_mapping, independently recompute each original span overlap containing at least one actual alphanumeric character; nonlexical zeros.',
        'loss':'original weighted risk BCE +0.25*mean(two auxiliary weighted BCEs); each term normalized by original target139518',
        'aux_weight':'Start with exact original base group/answer/lexical weights. For each auxiliary head, binary class factor from fit base masses only; reequalize source-connected group mass, normalize total139518. Nonlexical/padding loss0.',
        'inference':'Only original total-risk head is used in token scores, original4raw lexical-window max, answer max, thresholds and selection. Types never enter inputs or checkpoint selection.',
        'epochs':EPOCHS,'batch_answers':BATCH,'shuffle':'Same frozen30 answer permutations as original network',
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'all_parameters':True},'dropout':.2,'cpu_threads':4,
        'selection':'Exactly original cal overall-risk maxmin F1, then windowF1/windowprecision/earlier epoch; no type-aware tuning.',
        'budget':'One additional model, one seed, coefficient0.25 fixed, no early stopping or search',
        'comparison':'Frozen original semantic width32 model reused, not retrained; all previous baselines retained.',
        'offline_extra_checker':True,'limits':'Multitask gradients change regularization and optimization, so improvement alone does not establish a unique causal explanation. Single-seed cal-selected development only.'}


class MultiTaskTCN(nn.Module):
    def __init__(self):
        super().__init__();main=baseline.new_model()
        self.projection=main.projection;self.convolutions=main.convolutions;self.dropout=main.dropout;self.output=main.output
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(AUX_SEED);self.auxiliary=nn.Linear(32,2)
    def forward(self,x,valid):
        mask=valid[:,None,:];h=F.gelu(self.projection(x)).transpose(1,2)*mask
        for conv in self.convolutions:h=(h+self.dropout(F.gelu(conv(h))))*mask
        h=h.transpose(1,2)
        return self.output(h).squeeze(-1)*valid,self.auxiliary(h)*valid[:,:,None]


def new_model():return MultiTaskTCN().cpu()


def type_labels(tokens):
    n=tokens['token_count'];out=np.zeros((n,2),np.int64)
    assert len(tokens['original_labels'])==len(tokens['span_token_mapping'])
    for span,mapping in zip(tokens['original_labels'],tokens['span_token_mapping']):
        assert span['label_type'] in TYPE
        out[mapping['risk_token_indices'],TYPE[span['label_type']]]=1
    assert np.array_equal(out.any(1),np.asarray(tokens['risk_mask'],bool))
    assert not out[~np.asarray(tokens['lexical_mask'],bool)].any()
    return out


def independent_type_labels(text,tokens):
    y=np.zeros((tokens['token_count'],2),np.int64)
    for span in tokens['original_labels']:
        kind=TYPE[span['label_type']]
        for j,(a,b) in enumerate(tokens['response_token_offsets']):
            lo=max(a,span['start']);hi=min(b,span['end'])
            if any(c.isalnum() for c in text[lo:hi]) if lo<hi else False:y[j,kind]=1
    return y


def source():
    meta,index,snap=baseline.source();assert snap==qa.read(BASE/'source_snapshot.json')
    assert qa.read(BASE/'AUDIT.json')['status']=='passed' and qa.read(BASE/'complete.json')['status']=='complete_development_only'
    for name,expected in qa.read(BASE/'preparation_complete.json')['files_sha256'].items():assert qa.sha(BASE/name)==expected
    paths=[Path(__file__),ROOT/'src/run_semantic_sequence.py',BASE/'complete.json',BASE/'AUDIT.json',BASE/'preparation_complete.json',
        BASE/'source_snapshot.json',BASE/'standardized_token_features.npy',OLD/'fit_token_weights.npz',OLD/'epoch_answer_order.npy',OLD/'token_index.json',ROOT/'data/gold_manifest.json']
    return meta,index,{'files_sha256':{str(p.resolve()):qa.sha(p) for p in paths},'test_opened':False,'GPU_used':False}


def make_labels_weights(meta,index):
    ys=[];counts={}
    for answer,tokens in zip(meta['answers'],meta['tokens']):
        y=type_labels(tokens);assert np.array_equal(y,independent_type_labels(answer['original_response'],tokens));ys.append(y)
    y=np.concatenate(ys)
    with np.load(OLD/'fit_token_weights.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    base=old['base'];fit_y=y[:170361];weights=np.empty_like(fit_y,np.float64);factors=np.empty((2,2),np.float64)
    groups=defaultdict(list)
    for a in index[:634]:groups[a['group_id']].extend(range(a['left'],a['right']))
    assert len(groups)==615 and np.array_equal(fit_y.any(1),old['y'].astype(bool))
    for k in range(2):
        mass=np.bincount(fit_y[:,k],weights=base,minlength=2);assert (mass>0).all()
        factors[k]=mass.sum()/(2*mass);weights[:,k]=base*factors[k,fit_y[:,k]]
        for inds in groups.values():weights[inds,k]*=(MASS/len(groups))/weights[inds,k].sum()
        weights[:,k]*=MASS/weights[:,k].sum()
        assert np.isclose(weights[:,k].sum(),MASS) and not weights[~old['lexical'].astype(bool),k].any()
    for part,lo,hi in (('fit',0,170361),('calibration',170361,len(y))):
        v=y[lo:hi];counts[part]={'raw_tokens':len(v),'baseless_positive_tokens':int(v[:,0].sum()),'conflict_positive_tokens':int(v[:,1].sum()),
            'both_positive_tokens':int(v.all(1).sum()),'risk_positive_tokens':int(v.any(1).sum())}
    return y,weights,factors,counts


def freeze():
    assert not (OUT/'protocol.json').exists();meta,index,snap=source()
    y,weights,factors,counts=make_labels_weights(meta,index)
    # Explicit punctuation and overlapping-type example.
    toy={'token_count':3,'response_token_offsets':[[0,1],[1,2],[2,3]],'lexical_mask':[True,False,True],'risk_mask':[1,0,1],
        'original_labels':[{'start':0,'end':3,'label_type':'Evident Baseless Info'},{'start':2,'end':3,'label_type':'Subtle Conflict'}],
        'span_token_mapping':[{'risk_token_indices':[0,2]},{'risk_token_indices':[2]}]}
    assert type_labels(toy).tolist()==[[1,0],[0,0],[1,1]] and np.array_equal(type_labels(toy),independent_type_labels('A!B',toy))
    main=baseline.new_model();main_rng=torch.get_rng_state().clone();model=new_model();assert torch.equal(main_rng,torch.get_rng_state())
    for key,value in main.state_dict().items():assert torch.equal(value,model.state_dict()[key])
    assert sum(p.numel() for p in model.parameters())==41219
    x=torch.randn(2,16,1090,generator=torch.Generator().manual_seed(SEED+1));mask=torch.ones(2,16);mask[0,9:]=0
    main.eval();model.eval()
    with torch.no_grad():
        p=main(x,mask);risk,aux=model(x,mask);assert torch.equal(p,risk) and aux.shape==(2,16,2)
        altered=x.clone();altered[0,9:]=1000;assert torch.allclose(model(altered,mask)[0][0,:9],risk[0,:9],atol=2e-6,rtol=1e-6)
        altered=x.clone();altered[1]=1000;assert torch.equal(model(altered,mask)[0][0],risk[0])
    # Train-mode dropout RNG is also unaffected by introducing the deterministic auxiliary linear head.
    state=torch.get_rng_state().clone();main.train();model.train();a=main(x,mask);torch.set_rng_state(state);b=model(x,mask)[0];assert torch.equal(a,b)
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'auxiliary_labels_weights.npz',y=y,fit_loss_weights=weights,class_factors=factors)
    qa.save(OUT/'LABEL_REVIEW.json',{'status':'passed','all793_span_mapping_independently_recomputed':True,'risk_labels_unchanged':True,'counts':counts,
        'class_factors':factors.tolist(),'aux_loss_mass':weights.sum(0).tolist(),'source_groups':615,'test_opened':False})
    qa.save(OUT/'PREFLIGHT.json',{'passed':True,'main_parameters_and_eval_train_outputs_exact':True,'aux_rng_isolated':True,
        'overlap_and_punctuation_sample_passed':True,'parameters':41219,'GPU_used':False})
    qa.save(OUT/'protocol.json',protocol());qa.save(OUT/'source_snapshot.json',snap)
    qa.save(OUT/'freeze.json',{'code_sha256':qa.sha(Path(__file__)),'protocol_sha256':qa.sha(OUT/'protocol.json'),
        'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json'),'auxiliary_labels_weights_sha256':qa.sha(OUT/'auxiliary_labels_weights.npz')})
    print('SEMANTIC_MULTITASK_FROZEN',counts,flush=True)


def batch(chosen,x,index,weights,aux_y,aux_weights):
    xx,mask,yy,ww=baseline.batch(chosen,x,index,weights);ay=torch.zeros(*yy.shape,2);aw=torch.zeros_like(ay)
    for j,i in enumerate(chosen):
        a=index[int(i)];lo,hi=a['left'],a['right'];n=hi-lo;assert hi<=170361
        ay[j,:n]=torch.from_numpy(aux_y[lo:hi].astype(np.float32));aw[j,:n]=torch.from_numpy(aux_weights[lo:hi].astype(np.float32))
    return xx,mask,yy,ww,ay,aw


@torch.no_grad()
def predict(model,x,index):
    model.eval();logits=np.empty(len(x),np.float32);aux=np.empty((len(x),2),np.float32)
    for left in range(0,len(index),BATCH):
        chosen=list(range(left,min(left+BATCH,len(index))));xx,mask,_,_=baseline.batch(chosen,x,index)
        risk,other=model(xx,mask);risk=risk.numpy();other=other.numpy()
        for j,i in enumerate(chosen):
            a=index[i];lo,hi=a['left'],a['right'];logits[lo:hi]=risk[j,:hi-lo];aux[lo:hi]=other[j,:hi-lo]
    assert np.isfinite(logits).all() and np.isfinite(aux).all()
    return logits,torch.sigmoid(torch.from_numpy(logits)).numpy(),aux


def train():
    assert qa.read(OUT/'protocol.json')==protocol() and not (OUT/'started.json').exists();frozen=qa.read(OUT/'freeze.json')
    assert qa.sha(Path(__file__))==frozen['code_sha256'] and qa.sha(OUT/'auxiliary_labels_weights.npz')==frozen['auxiliary_labels_weights_sha256']
    meta,index,snap=source();assert snap==qa.read(OUT/'source_snapshot.json')
    qa.save(OUT/'started.json',{'utc':time.time(),'freeze_sha256':qa.sha(OUT/'freeze.json')});clock=time.perf_counter()
    x=np.load(BASE/'standardized_token_features.npy',mmap_mode='r');order=np.load(OLD/'epoch_answer_order.npy')
    with np.load(OLD/'fit_token_weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    with np.load(OUT/'auxiliary_labels_weights.npz',allow_pickle=False) as z:ay=z['y'];aw=z['fit_loss_weights']
    model=new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    directory=OUT/METHOD;directory.mkdir(exist_ok=True);history=[];selected=None;best=None;files=[]
    for epoch in range(EPOCHS):
        tick=time.perf_counter();model.train()
        for left in range(0,634,BATCH):
            chosen=order[epoch,left:left+BATCH];xx,mask,yy,ww,by,bw=batch(chosen,x,index,weights,ay,aw)
            optimizer.zero_grad(set_to_none=True);risk,aux=model(xx,mask)
            primary=(F.binary_cross_entropy_with_logits(risk,yy,reduction='none')*ww).sum()
            secondary=(F.binary_cross_entropy_with_logits(aux,by,reduction='none')*bw).sum((0,1)).mean()
            objective=(primary+.25*secondary)*(634/(len(chosen)*MASS));assert torch.isfinite(objective)
            objective.backward();optimizer.step()
        training_seconds=time.perf_counter()-tick;logits,p,aux=predict(model,x,index)
        arrays,thresholds,metrics,bce=baseline.v1.evaluate(meta,index,logits,p,weights)
        fl=aux[:170361].astype(np.float64);aux_bce=(aw*(np.logaddexp(0,fl)-ay[:170361]*fl)).sum(0)/MASS
        arrays.update(token_logits=logits,token_scores=p,auxiliary_token_logits=aux)
        prefix=f'epoch_{epoch+1:03d}';np.savez_compressed(directory/(prefix+'_scores.npz'),**arrays)
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
            'epoch':epoch+1,'seed':SEED,'aux_seed':AUX_SEED,'device':'cpu','protocol_sha256':qa.sha(OUT/'protocol.json'),
            'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json')},directory/(prefix+'.pt'))
        w,a=thresholds['window'],thresholds['answer'];key=[min(w['f1'],a['f1']),w['f1'],w['precision'],-(epoch+1)]
        entry={'method':METHOD,'epoch':epoch+1,'parameters':41219,'thresholds':thresholds,'selection_key':key,'metrics':metrics,
            'full_fit_weighted_BCE':bce,'full_fit_auxiliary_BCE':aux_bce.tolist(),'full_fit_combined_loss':float(bce+.25*aux_bce.mean()),
            'epoch_training_seconds':training_seconds,'epoch_total_seconds':time.perf_counter()-tick,
            'scores_sha256':qa.sha(directory/(prefix+'_scores.npz')),'model_sha256':qa.sha(directory/(prefix+'.pt')),'test_opened':False}
        qa.save(directory/(prefix+'.json'),entry);history.append(entry);files.extend(str((directory/(prefix+s)).relative_to(OUT)) for s in ('_scores.npz','.pt','.json'))
        if best is None or key>best:best=key;selected=entry
        qa.save(OUT/'progress.json',{'epoch':epoch+1,'selected':selected,'seconds':time.perf_counter()-clock})
        print('SEMANTIC_MULTITASK_EPOCH',epoch+1,'riskBCE',round(bce,5),'auxBCE',aux_bce.round(5).tolist(),'cal_w',round(w['f1'],6),'cal_a',round(a['f1'],6),flush=True)
    assert source()[-1]==snap and not torch.cuda.is_initialized()
    qa.save(OUT/'summary.json',{'selected':{METHOD:selected},'all_epochs':{METHOD:history},'wall_seconds':time.perf_counter()-clock,
        'test_opened':False,'GPU_used':False,'types_only_auxiliary_training_targets':True,'calibration_selection_optimistic':True})
    files.extend(['summary.json','protocol.json','freeze.json','source_snapshot.json','PREFLIGHT.json','LABEL_REVIEW.json','auxiliary_labels_weights.npz'])
    qa.save(OUT/'complete.json',{'status':'complete_development_only','files_sha256':{name:qa.sha(OUT/name) for name in files},'test_opened':False,'GPU_used':False})
    print('SEMANTIC_MULTITASK_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','train']);args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):{'freeze':freeze,'train':train}[args.stage]()
