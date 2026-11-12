"""One CPU-only direct-window TCN; frozen QA fit/calibration, never test.

Supervise unchanged 4rawBPE windows directly, keeping every eligible window.
Reuses exact old LR window features, fit-only scaler, PCA and window weights.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F

import run_development as old

ROOT = old.ROOT
OUT = ROOT / 'results/window_sequence_v1'
PRIOR = ROOT / 'results/development_v1'
TOKEN = ROOT / 'results/sequence_v1'
SEED, EPOCHS, THREADS, CHAIN_BATCH = 20260929, 30, 4, 8
WIDTH, HIDDEN, FIT_N, CAL_N, N = 1089, 32, 168123, 42241, 210364


def protocol():
    return {'version':'qa-direct-window-tcn-v1','scope':'Frozen official-train fit634/calibration159 only; no official test or withheld data',
        'method':'window_tcn','configurations':1,'seed':SEED,'epochs':EPOCHS,'CPU_threads':THREADS,'batch_chains':CHAIN_BATCH,
        'input':'Each unchanged eligible4rawBPE window: full1024-head LB mean + NLL mean + reused fit-PCA64 mean, exactly old LR1089 matrix',
        'scaler':'Reuse frozen hidden64_lookback_nll_C0.001.pkl fit-only scaler; all three original Cs share it. Recompute transformed old raw matrices and require old fit standardized matrix exact.',
        'PCA':'Reuse prior fixed fit-only PCA and cached64 window means; no new PCA or projection choices',
        'gold':'Original LR window risk label, including all four human span types; every eligible raw4BPE/stride1 window retained. No token-label aggregation at model output.',
        'weights':'Exactly prior LR base_weights/loss_weights/class_factors; independently recompute and require exact; fit loss mass168123',
        'chains':'Per answer split at any nonconsecutive raw token_start; no links across answers, raw-start gaps, fit/calibration or padding',
        'architecture':'Linear1089->32 GELU; two residual blocks each ONE Conv1d32->32 kernel3 dilation1 then2, symmetric padding, GELU/dropout.2; Linear32->1 per window',
        'padding':'Zero padded hidden states after projection and each residual block, mask output and zero loss for padding',
        'receptive_field_windows':7,'receptive_field_rawBPE_when_contiguous':10,'future_windows_used':True,'online_detector':False,
        'receptive_field_limit':'This is convolutional geometry, not total information in causal backbone states.',
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'all_parameters_including_bias':True},
        'shuffle':'Fixed numpy default_rng(seed), new permutation of all fit chains each epoch; every fit window exactly once each epoch',
        'minibatch_objective':'sum(original_loss_weight * BCE_logits) * number_of_fit_batches /168123. Denominator constant across batches; no per-batch weight-sum renormalization.',
        'window_score':'Direct sigmoid of each window output logit, no token max or postprocessing',
        'answer_score':'Maximum over ALL eligible original windows of each answer; original nonempty-span answer label',
        'selection':'Separate cal thresholds by F1, precision, higher cutoff; epoch by min(windowF1,answerF1), windowF1, window precision, earlier epoch',
        'early_stopping':False,'extra_seeds_or_architectures':False,'fit_calibration_refit':False,
        'save':'All30 CPU model+optimizer states, direct window logits/probabilities, answer scores, thresholds and metrics',
        'comparisons':'All12 frozen LR candidates and selected original token_tcn, with unchanged gold and denominators; no re-training those references',
        'causal_interpretation_limit':'Versus token_tcn this also changes193->1089 features,64->32 width,rawBPE RF7->10,weighting and direct supervision. It is a predefined practical alternative, not a clean one-factor estimate of the supervision mismatch.',
        'reported_performance':'Calibration chooses checkpoints and thresholds; its F1 is selection-optimistic, not independent test performance.'}


class WindowTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(WIDTH, HIDDEN)
        self.convolutions = nn.ModuleList([nn.Conv1d(HIDDEN,HIDDEN,3,padding=d,dilation=d) for d in (1,2)])
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(HIDDEN,1)

    def forward(self,x,valid):
        mask = valid[:,None,:]
        h = F.gelu(self.projection(x)).transpose(1,2) * mask
        for conv in self.convolutions:
            h = (h+self.dropout(F.gelu(conv(h)))) * mask
        return self.output(h.transpose(1,2)).squeeze(-1) * valid


def new_model():
    torch.manual_seed(SEED)
    return WindowTCN().cpu()


def chains(meta):
    seqs=[]
    for answer in meta['answers']:
        ix=np.asarray(meta['answer_windows'][answer['response_id']],dtype=np.int64)
        starts=np.asarray([meta['windows'][j]['token_start'] for j in ix])
        assert np.all(np.diff(starts)>0)
        for sub in np.split(ix,np.flatnonzero(np.diff(starts)!=1)+1):
            assert len(sub)>0 and np.array_equal(sub,np.arange(sub[0],sub[-1]+1))
            seqs.append({k:answer[k] for k in ('answer_id','response_id','source_id','group_id','partition')} |
                        {'left':int(sub[0]),'right':int(sub[-1])+1,'length':len(sub),
                         'raw_start_first':meta['windows'][sub[0]]['token_start'],
                         'raw_start_last':meta['windows'][sub[-1]]['token_start']})
    assert np.array_equal(np.concatenate([np.arange(s['left'],s['right']) for s in seqs]),np.arange(len(meta['windows'])))
    return seqs


def snapshot():
    meta=old.metadata(); complete=old.read(PRIOR/'complete.json'); matrix=old.read(PRIOR/'matrix_manifest.json')
    assert complete['status']=='complete_development_only' and not complete['official_test_opened']
    paths=[Path(__file__), ROOT/'src/run_development.py', ROOT/'data/gold_manifest.json', ROOT/'development_protocol.json',
           PRIOR/'complete.json',PRIOR/'matrix_manifest.json',PRIOR/'hidden_pca.pkl',PRIOR/'training_weights.npz',PRIOR/'fit_keys.json',
           PRIOR/'matrices/base.npy',PRIOR/'matrices/hidden.npy',PRIOR/'matrices/hidden64_lookback_nll_fit_standardized.npy',
           PRIOR/'hidden64_lookback_nll_C0.001.pkl',PRIOR/'summary.json',TOKEN/'complete.json',TOKEN/'summary.json']
    for key in ('base','hidden'):
        assert old.sha(PRIOR/f'matrices/{key}.npy')==matrix['files_sha256'][key]
    assert old.sha(PRIOR/'hidden_pca.pkl')==matrix['pca_sha256']
    for name in ('matrix_manifest.json','hidden_pca.pkl','training_weights.npz','fit_keys.json','hidden64_lookback_nll_C0.001.pkl','summary.json'):
        assert old.sha(PRIOR/name)==complete['files_sha256'][name]
    tc=old.read(TOKEN/'complete.json'); assert not tc['test_opened']
    assert old.sha(TOKEN/'summary.json')==tc['files_sha256']['summary.json']
    t=old.read(TOKEN/'summary.json')['selected']['token_tcn']; token_score=TOKEN/f"token_tcn/epoch_{t['epoch']:03d}_scores.npz"
    assert old.sha(token_score)==t['scores_sha256']; paths.append(token_score)
    seqs=chains(meta); fit_seqs=[s for s in seqs if s['partition']=='fit']
    assert len(seqs)==1187 and len(fit_seqs)==952 and len(seqs)-len(fit_seqs)==235
    assert all(s['right']<=FIT_N for s in fit_seqs)
    source={'files_sha256':{str(p.resolve()):old.sha(p) for p in paths},'chains':len(seqs),'fit_chains':len(fit_seqs),
            'calibration_chains':235,'raw_start_gaps':len(seqs)-len(meta['answers']),
            'test_read':False,'new_PCA':False,'new_scaler':False}
    return meta,seqs,source


def freeze():
    assert not torch.cuda.is_initialized()
    meta,seqs,source=snapshot()
    assert not (OUT/'protocol.json').exists(),'Do not replace frozen experiment'
    old.save(OUT/'protocol.json',protocol());old.save(OUT/'source_snapshot.json',source)
    old.save(OUT/'chain_index.json',{'chains':seqs,'window_count':N,'fit_window_count':FIT_N,'calibration_window_count':CAL_N})
    old.save(OUT/'freeze.json',{'utc':time.time(),'protocol_sha256':old.sha(OUT/'protocol.json'),
        'source_snapshot_sha256':old.sha(OUT/'source_snapshot.json'),'chain_index_sha256':old.sha(OUT/'chain_index.json'),'test_read':False})
    print('WINDOW_TCN_FROZEN',len(seqs),'chains',flush=True)


def preflight():
    assert not torch.cuda.is_initialized() and old.read(OUT/'protocol.json')==protocol()
    model=new_model().eval();g=torch.Generator().manual_seed(SEED+1)
    x=torch.randn(2,16,WIDTH,generator=g);valid=torch.ones(2,16);valid[0,9:]=0
    with torch.no_grad():
        out=model(x,valid)
        changed=x.clone();changed[0,9:]=1000
        assert torch.allclose(model(changed,valid)[0,:9],out[0,:9],atol=2e-6,rtol=1e-6)
        one=model(x[:1,:9],torch.ones(1,9))
        assert torch.allclose(one[0],out[0,:9],atol=2e-6,rtol=1e-6)
        changed=x.clone();changed[1]=1000
        assert torch.equal(model(changed,valid)[0],out[0])
        changed=x.clone();changed[1,:4]+=100
        assert torch.equal(model(changed,valid)[1,8],out[1,8])
    toy={'answers':[{'answer_id':'a','response_id':'a','source_id':'s','group_id':'g','partition':'fit'},
                    {'answer_id':'b','response_id':'b','source_id':'t','group_id':'h','partition':'calibration'}],
         'answer_windows':{'a':[0,1,2],'b':[3,4]},'windows':[{'token_start':i} for i in (0,1,4,0,1)]}
    assert [(s['left'],s['right']) for s in chains(toy)]==[(0,2),(2,3),(3,5)]
    # Only synthetic backward timing; no optimizer step and no warm-start state.
    model.train();xb=torch.randn(8,448,WIDTH,generator=g);mask=torch.ones(8,448);yb=torch.zeros(8,448)
    times=[]
    for _ in range(3):
        started=time.perf_counter();model.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(xb,mask),yb).backward();times.append(time.perf_counter()-started)
    result={'passed':True,'code_sha256':old.sha(__file__),'protocol_sha256':old.sha(OUT/'protocol.json'),
            'parameters':sum(p.numel() for p in model.parameters()),'padding_mask_checks':True,'no_cross_chain_propagation':True,
            'convolution_RF7_checked':True,'raw_start_gap_split_checked':True,'all_fit_chains':952,
            'synthetic_backward_batch_seconds_median':float(np.median(times)),'batch_shape':[8,448,WIDTH],
            'estimated30epoch_train_seconds_excluding_validation_io':float(np.median(times))*119*30,
            'no_optimizer_step_or_warmstart':True,'gpu_initialized':torch.cuda.is_initialized(),'test_read':False}
    old.save(OUT/'CPU_PREFLIGHT.json',result);print(json.dumps(result),flush=True)


def prepare(meta,seqs):
    obj=pickle.loads((PRIOR/'hidden64_lookback_nll_C0.001.pkl').read_bytes());scaler=obj['scaler']
    assert obj['fit_only'] and obj['width']==WIDTH and obj['fit_rows']==FIT_N and obj['fit_groups']==615
    base=np.load(PRIOR/'matrices/base.npy',mmap_mode='r');hidden=np.load(PRIOR/'matrices/hidden.npy',mmap_mode='r')
    assert base.shape==(N,1025) and hidden.shape==(N,64)
    oldz=np.load(PRIOR/'matrices/hidden64_lookback_nll_fit_standardized.npy',mmap_mode='r')
    x=np.lib.format.open_memmap(OUT/'standardized_window_features.npy',mode='w+',dtype=np.float32,shape=(N,WIDTH))
    for left in range(0,N,old.BATCH):
        right=min(left+old.BATCH,N)
        raw=np.column_stack((base[left:right],hidden[left:right]))
        x[left:right]=scaler.transform(raw).astype(np.float32)
        if left<FIT_N:
            boundary=min(right,FIT_N)
            assert np.array_equal(x[left:boundary],oldz[left:boundary]),'Old standardized fit design drift'
    x.flush();assert np.isfinite(x).all()
    b,loss,factors,y=old.base_weights(meta)
    with np.load(PRIOR/'training_weights.npz',allow_pickle=False) as z:
        assert all(np.array_equal(a,z[k]) for a,k in [(b,'base_weights'),(loss,'loss_weights'),(factors,'class_factors'),(y,'y')])
    np.savez_compressed(OUT/'training_weights.npz',base_weights=b,loss_weights=loss,class_factors=factors,y=y)
    fit_ids=np.asarray([i for i,s in enumerate(seqs) if s['partition']=='fit'])
    rng=np.random.default_rng(SEED);orders=np.stack([rng.permutation(fit_ids) for _ in range(EPOCHS)])
    assert len(fit_ids)==952 and math.ceil(len(fit_ids)/CHAIN_BATCH)==119
    for order in orders:
        present=np.concatenate([np.arange(seqs[i]['left'],seqs[i]['right']) for i in order])
        assert np.array_equal(np.sort(present),np.arange(FIT_N))
    np.save(OUT/'epoch_chain_order.npy',orders,allow_pickle=False)
    (OUT/'scaler.pkl').write_bytes(pickle.dumps(scaler,protocol=5))
    files=['standardized_window_features.npy','training_weights.npz','epoch_chain_order.npy','scaler.pkl','chain_index.json']
    old.save(OUT/'preparation_manifest.json',{'complete':True,'files_sha256':{n:old.sha(OUT/n) for n in files},
        'shape':[N,WIDTH],'fit_standardized_design_exact':True,'original_LR_weights_exact':True,
        'every_fit_window_once_per_epoch':True,'fit_batches_per_epoch':119,'minibatch_constant':119/FIT_N,
        'scaler_source_sha256':old.sha(PRIOR/'hidden64_lookback_nll_C0.001.pkl'),
        'PCA_source_sha256':old.sha(PRIOR/'hidden_pca.pkl'),'test_read':False,'gpu_used':False})
    return x,{'loss':loss,'y':y},orders


def batch(chosen,x,seqs,weights=None):
    length=max(seqs[int(i)]['length'] for i in chosen)
    xb=torch.zeros(len(chosen),length,WIDTH);valid=torch.zeros(len(chosen),length)
    yb=torch.zeros(len(chosen),length);wb=torch.zeros(len(chosen),length)
    for j,i in enumerate(chosen):
        s=seqs[int(i)];lo,hi=s['left'],s['right'];n=hi-lo
        xb[j,:n]=torch.from_numpy(np.array(x[lo:hi],copy=True));valid[j,:n]=1
        if weights is not None:
            assert s['partition']=='fit' and hi<=FIT_N
            yb[j,:n]=torch.from_numpy(weights['y'][lo:hi].astype(np.float32))
            wb[j,:n]=torch.from_numpy(weights['loss'][lo:hi].astype(np.float32))
    return xb,valid,yb,wb


@torch.no_grad()
def predict(model,x,seqs):
    model.eval();logits=np.empty(N,np.float32)
    for left in range(0,len(seqs),CHAIN_BATCH):
        chosen=list(range(left,min(left+CHAIN_BATCH,len(seqs))))
        xb,valid,_,_=batch(chosen,x,seqs);out=model(xb,valid).numpy()
        for j,i in enumerate(chosen):
            s=seqs[i];logits[s['left']:s['right']]=out[j,:s['length']]
    assert np.isfinite(logits).all()
    probabilities=torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float64)
    return logits,probabilities


def evaluate(meta,logits,scores,weights):
    answers=old.answer_scores(meta,scores)
    ts={'window':old.choose_threshold([w['label'] for w in meta['windows'][FIT_N:]],scores[FIT_N:]),
        'answer':old.choose_threshold([a['label'] for a in meta['answers'][634:]],answers[634:])}
    z=logits[:FIT_N].astype(np.float64);bce=np.logaddexp(0,z)-weights['y']*z
    return answers,ts,old.metrics(meta,scores,ts),float(weights['loss']@bce/FIT_N)


def train():
    assert not torch.cuda.is_initialized() and old.read(OUT/'protocol.json')==protocol()
    assert not (OUT/'started.json').exists(),'Do not silently restart formal training'
    meta,seqs,source=snapshot();assert source==old.read(OUT/'source_snapshot.json')
    frozen=old.read(OUT/'freeze.json');assert old.sha(OUT/'protocol.json')==frozen['protocol_sha256']
    assert old.sha(OUT/'chain_index.json')==frozen['chain_index_sha256']
    check=old.read(OUT/'CPU_PREFLIGHT.json');assert check['passed'] and check['code_sha256']==old.sha(__file__)
    old.save(OUT/'started.json',{'utc':time.time(),'freeze_sha256':old.sha(OUT/'freeze.json'),'test_read':False})
    started=time.perf_counter();x,weights,orders=prepare(meta,seqs);model=new_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    folder=OUT/'window_tcn';folder.mkdir(parents=True,exist_ok=True);history=[];selected=None;files=[]
    for epoch in range(EPOCHS):
        begin=time.perf_counter();model.train();order=orders[epoch]
        for left in range(0,len(order),CHAIN_BATCH):
            chosen=order[left:left+CHAIN_BATCH];xb,valid,yb,wb=batch(chosen,x,seqs,weights)
            optimizer.zero_grad(set_to_none=True)
            bce=F.binary_cross_entropy_with_logits(model(xb,valid),yb,reduction='none')
            objective=(bce*wb).sum()*(119/FIT_N)
            assert torch.isfinite(objective)
            objective.backward();optimizer.step()
        fit_seconds=time.perf_counter()-begin
        logits,scores=predict(model,x,seqs);answers,ts,metrics,loss=evaluate(meta,logits,scores,weights)
        name=f'epoch_{epoch+1:03d}';scorefile=folder/(name+'_scores.npz');statefile=folder/(name+'.pt')
        np.savez_compressed(scorefile,window_scores=scores,answer_scores=answers,window_logits=logits)
        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'epoch':epoch+1,'method':'window_tcn','seed':SEED,'protocol_sha256':old.sha(OUT/'protocol.json'),
            'preparation_manifest_sha256':old.sha(OUT/'preparation_manifest.json'),'torch_rng_state':torch.get_rng_state(),'device':'cpu'},statefile)
        w,a=ts['window'],ts['answer'];key=[min(w['f1'],a['f1']),w['f1'],w['precision'],-(epoch+1)]
        entry={'method':'window_tcn','epoch':epoch+1,'parameters':sum(p.numel() for p in model.parameters()),
            'thresholds':ts,'selection_key':key,'metrics':metrics,'full_fit_weighted_BCE':loss,
            'training_seconds':fit_seconds,'epoch_seconds':time.perf_counter()-begin,
            'model_sha256':old.sha(statefile),'scores_sha256':old.sha(scorefile),'test_read':False}
        old.save(folder/(name+'.json'),entry);history.append(entry)
        if selected is None or key>selected['selection_key']:selected=entry
        files.extend([str(p.relative_to(OUT)) for p in [scorefile,statefile,folder/(name+'.json')]])
        old.save(OUT/'progress.json',{'completed_epochs':epoch+1,'selected':selected,'seconds':time.perf_counter()-started,'test_read':False})
        print('WINDOW_TCN_EPOCH',epoch+1,'cal_w',round(w['f1'],6),'cal_a',round(a['f1'],6),'BCE',round(loss,6),'seconds',round(entry['epoch_seconds'],2),flush=True)
    assert source==snapshot()[2] and not torch.cuda.is_initialized()
    summary={'scope':protocol()['scope'],'selected':{'window_tcn':selected},'all_epochs':{'window_tcn':history},
        'new_models':1,'epochs':30,'input_width':WIDTH,'conv_width':HIDDEN,'seed':SEED,'coverage':meta['gold']['counts'],
        'window_supervision_direct':True,'token_max_used':False,'calibration_results_are_selection_optimistic':True,
        'test_opened':False,'gpu_used':False,'seconds':time.perf_counter()-started,'limitations':protocol()['causal_interpretation_limit']}
    old.save(OUT/'summary.json',summary)
    files += ['summary.json','protocol.json','freeze.json','source_snapshot.json','CPU_PREFLIGHT.json','chain_index.json',
              'standardized_window_features.npy','training_weights.npz','epoch_chain_order.npy','scaler.pkl','preparation_manifest.json']
    old.save(OUT/'complete.json',{'status':'complete_development_only','files_sha256':{n:old.sha(OUT/n) for n in files},
        'script_sha256':old.sha(__file__),'test_opened':False,'gpu_used':False})
    print('WINDOW_TCN_COMPLETE',round(summary['seconds'],2),flush=True)


def compare():
    meta=old.metadata();ours=old.read(OUT/'summary.json');prior=old.read(PRIOR/'summary.json');token=old.read(TOKEN/'summary.json')
    entries=[]
    for method,candidates in prior['all_candidates'].items():
        for e in candidates:entries.append({'candidate':e['candidate'],'kind':'frozen_LR','metrics':e['metrics']})
    for kind,e in [('frozen_token_tcn',token['selected']['token_tcn']),('direct_window_tcn',ours['selected']['window_tcn'])]:
        entries.append({'candidate':kind,'kind':kind,'epoch':e['epoch'],'metrics':e['metrics']})
    for e in entries:
        for part,nw,na in [('fit',FIT_N,634),('calibration',CAL_N,159)]:
            assert e['metrics'][part]['windows']['n']==nw and e['metrics'][part]['answers']['n']==na
    old.save(OUT/'COMPARISON.json',{'entries':entries,'same_gold_and_denominators':True,'test_opened':False,
        'old_complete_sha256':old.sha(PRIOR/'complete.json'),'token_complete_sha256':old.sha(TOKEN/'complete.json'),
        'window_complete_sha256':old.sha(OUT/'complete.json'),'calibration_selection_optimistic':True})
    report=['单配置窗口TCN已固定训练30轮；直接监督每个4原始词元窗口，不再由词元概率取max得到窗口分数。以下均为校准选参成绩，不是最终测试。','',
        '| 模型 | epoch/C | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|']
    chosen=[('同1089输入旧LR',prior['selected']['hidden64_lookback_nll']),('原tokenTCN',token['selected']['token_tcn']),('直接windowTCN',ours['selected']['window_tcn'])]
    for name,e in chosen:
        m=e['metrics'];report.append(f"| {name} | {e.get('epoch',e.get('C'))} | {m['fit']['windows']['f1']:.3f} | {m['calibration']['windows']['f1']:.3f} | {m['fit']['answers']['f1']:.3f} | {m['calibration']['answers']['f1']:.3f} |")
    report += ['', '634 fit回答/168123窗口，159 cal回答/42241窗口；119个训练批次、952条fit链，每一轮每个fit窗口恰好出现一次。',
        '输入是旧LR完整1089维窗口矩阵，PCA/scaler/标签/权重均复用并核对；没有新大模型生成或GPU调用。394处raw-start缺口断链，不跨答传播。',
        protocol()['causal_interpretation_limit'],
        '卷积使用前后窗口，是离线检测；RF7个窗口在连续处覆盖10个原始BPE，并非宣称模型内部信息只来自10个词元。全部30轮状态与概率均保留。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8');print('WINDOW_TCN_COMPARED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','preflight','train','compare']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(THREADS);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=THREADS):
        {'freeze':freeze,'preflight':preflight,'train':train,'compare':compare}[args.stage]()
