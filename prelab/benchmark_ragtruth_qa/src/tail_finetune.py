"""Reviewable late-MiniCheck fine-tuning skeleton and CPU-only replay tests.

No CUDA operation occurs when importing this file or running tiny-test/prepare.
Formal training additionally requires complete caches and a separate real-cache
GPU equivalence gate supplied after root schedules the GPU.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import RobertaConfig,RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer
from threadpoolctl import threadpool_limits
import run_development as qa

ROOT=qa.ROOT;EXP=ROOT/'fit_expansion';OUT=ROOT/'results/minicheck_tail_v1'
MODEL=ROOT/'semantic_baseline/model';SEED=20261004
CACHES=(EXP/'minicheck/encoder22_feature_manifest.json',EXP/'minicheck_backfill/encoder22_feature_manifest.json')


def protocol():
    return {'version':'qa-minicheck-last-two-token-v1','scope':'Combined3680 fit answers/615 original fit groups, original159 cal/154 disjoint groups; no test',
        'method':'Additional MiniCheck evidence encoder fine-tuning baseline, not original generator whitebox probing',
        'trainable':'Only original RoBERTa encoder.layer22 and23 (0-index) plus newly initialized Linear1024->1 token head',
        'frozen':'Cached float32 output of layer21, i.e. hidden_states[22], with complete valid doc+special+claim sequence. Front22 blocks and original Llama never loaded on GPU or updated.',
        'selection_of_documents':'Reuse each claim original maximum-support document, first tie, selected without gold; do not reselect after fine-tuning.',
        'token_loss':'New MiniCheck token logits are averaged per original nonwhitespace character then per original Llama raw BPE; BCE uses unchanged original raw lexical risk_mask. No new checker-token label transfer.',
        'unicode_overlap':'A character covered by multiple checker tokens distributes unit mass equally; missing nonwhitespace coverage stops. Pure-whitespace BPE logit0 and loss0.',
        'weights':'Within each fit source-group, original634 and auxiliary3046 each get half base mass when both exist; all mass to original if no auxiliary. Within each stratum answer->lexical token equal. Fit-only binary class factors, then reequalize group loss; total mass equals combined fit lexical token count.',
        'epochs':3,'seed':SEED,'effective_answer_batch':8,'answer_microbatch':1,'sequence_microbatch':2,
        'gradient_checkpointing':'Non-reentrant, preserving RNG, across both tail layers for each <=2-sequence microbatch; enables whole-answer mapped BCE without retaining large attention activations.',
        'optimizer':{'name':'AdamW','tail_lr':2e-5,'head_lr':1e-4,'weight_decay':.01,'gradient_clip_norm':1.,'scheduler':'none'},
        'precision':'float32 cache, parameters, optimizer, and computation; no autocast/TF32',
        'dropout':'Original checkpoint hidden and attention dropout0.1 in trainable tail during training; cached front was eval; eval tail/head deterministic.',
        'aggregation':'Sigmoid mapped original raw-BPE logits ->max lexical positions in unchanged4raw window ->max all eligible windows per answer. Original answer labels preserved.',
        'selection':'Per epoch separate original cal risk F1 thresholds, precision/higher-cutoff ties; checkpoint maxminF1, windowF1,windowprecision,earlier epoch. No type/generator-aware tuning.',
        'budget':'Two matched conditions: last2+head fine-tuned versus only same token head trained on frozen final-layer states. Each three epochs, samehead initialization/lr/weights/answer shuffle, no grid. Epoch0 diagnostic excluded from checkpoint selection.',
        'frozen_head_control':'Reuse original selected claim final states, character-map hidden1024 then same Linear head on CPU. Cached tail stays eval; fine-tuned tail uses original training dropout. This control holds data and head budget fixed; it measures the practical fine-tuning procedure including its dropout.',
        'generator_id':'Never input or selection. Only original-versus-auxiliary membership is used for the fixed half-mass loss split; no per-generator weighting.',
        'additional_runtime_gate':'CPU tiny replay/gradient checks and all-cache CPU geometry audit first; real pretrained-tail GPU replay equivalence must be separately passed before training.',
        'offline':True,'resource_estimate':'25,193,473 trainable parameters; FP32 weights+grad+Adam states about403MB, excluding activations/input. Two-sequence checkpointing aims below8GB but actual GPU peak/time must be measured, not claimed from CPU tests.'}


class TailTokenDetector(nn.Module):
    def __init__(self,config,layers=None):
        super().__init__();self.config=config
        self.layers=nn.ModuleList(layers if layers is not None else [RobertaLayer(config),RobertaLayer(config)])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED);self.head=nn.Linear(config.hidden_size,1)
            nn.init.normal_(self.head.weight,mean=0.,std=config.initializer_range);nn.init.zeros_(self.head.bias)
    def tail_hidden(self,hidden,mask):
        assert not hidden.requires_grad,'Cached front states must be detached'
        additive=(1-mask[:,None,None,:].to(hidden.dtype))*torch.finfo(hidden.dtype).min
        for layer in self.layers:hidden=layer(hidden,attention_mask=additive)[0]
        return hidden
    def forward(self,hidden,mask,checkpointed=False):
        if checkpointed and self.training:
            h=checkpoint(self.tail_hidden,hidden,mask,use_reentrant=False,preserve_rng_state=True)
        else:h=self.tail_hidden(hidden,mask)
        return self.head(h).squeeze(-1)


def load_pretrained_tail():
    config=RobertaConfig.from_json_file(str(MODEL/'config.json'));config._attn_implementation='eager'
    assert (config.num_hidden_layers,config.hidden_size,config.num_attention_heads)==(24,1024,16)
    model=TailTokenDetector(config)
    # Safe tensor-only deserialization; mmap keeps unrelated front weights off GPU.
    state=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    for j,layer in enumerate(model.layers):
        prefix=f'roberta.encoder.layer.{22+j}.';subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        assert subset;layer.load_state_dict(subset,strict=True)
    del state
    assert sum(p.numel() for p in model.parameters())==25193473
    return model


def character_map(text,raw_offsets,start,end):
    """Return sparse row/column/weight mapping; no labels are accepted."""
    owners=[[] for _ in text]
    for j,(a,b) in enumerate(zip(start,end)):
        if a<0 or b<0:assert a==b==-1;continue
        assert 0<=a<b<=len(text)
        for c in range(a,b):
            if not text[c].isspace():owners[c].append(j)
    assert all(owners[c] for c,ch in enumerate(text) if not ch.isspace()),'Missing full-answer nonwhitespace cache coverage'
    rows=[];cols=[];weights=[]
    for i,(a,b) in enumerate(raw_offsets):
        assert 0<=a<=b<=len(text);chars=[c for c in range(a,b) if not text[c].isspace()];counts=Counter()
        for c in chars:
            for j in owners[c]:counts[j]+=1/(len(chars)*len(owners[c]))
        if chars:assert abs(sum(counts.values())-1)<1e-12
        for j,w in sorted(counts.items()):rows.append(i);cols.append(j);weights.append(w)
    return np.asarray(rows,np.int64),np.asarray(cols,np.int64),np.asarray(weights,np.float32)


def mapped_logits(flat_logits,mapping,n_raw):
    rows,cols,weights=(torch.as_tensor(a,device=flat_logits.device) for a in mapping)
    out=flat_logits.new_zeros(n_raw)
    return out.index_add(0,rows,flat_logits[cols]*weights)


def answer_logits(model,arrays,mapping,n_raw,device,checkpointed):
    offsets=arrays['sequence_offsets'];parts=[]
    for first in range(0,len(offsets)-1,2):
        sequence=list(range(first,min(first+2,len(offsets)-1)));length=max(int(offsets[j+1]-offsets[j]) for j in sequence)
        hidden=torch.zeros(len(sequence),length,model.config.hidden_size,device=device,dtype=torch.float32)
        mask=torch.zeros(len(sequence),length,device=device,dtype=torch.long)
        for k,j in enumerate(sequence):
            a,b=offsets[j:j+2];n=int(b-a);hidden[k,:n]=torch.as_tensor(arrays['hidden22'][a:b],device=device);mask[k,:n]=1
        values=model(hidden,mask,checkpointed=checkpointed)
        parts.extend(values[k,:int(offsets[j+1]-offsets[j])] for k,j in enumerate(sequence))
    flat=torch.cat(parts);assert len(flat)==len(arrays['hidden22'])
    return mapped_logits(flat,mapping,n_raw)


def metadata():
    export=qa.read(EXP/'data/export_freeze.json');records=[];tokens=[];paths=[]
    for partition,directory in (('fit',EXP/'data'),('calibration',ROOT/'data')):
        ap=directory/f'answers_{partition}.jsonl';tp=directory/f'tokens_{partition}.jsonl';paths.extend([ap,tp])
        if partition=='fit':
            for p in (ap,tp):assert qa.sha(p)==export['output_files_sha256'][str(p.resolve())]
        aa=qa.lines(ap);tt=qa.lines(tp);assert [a['response_id'] for a in aa]==[t['response_id'] for t in tt]
        assert len(aa)==(3680 if partition=='fit' else 159)
        for a,t in zip(aa,tt):
            assert a['partition']==t['partition']==partition and a['answer_sha256']==t['answer_sha256']
            assert a['label']==int(bool(a['original_labels'])) and t['answer_risk']==a['label']
        records.extend(aa);tokens.extend(tt)
    groups={p:{a['group_id'] for a in records if a['partition']==p} for p in ('fit','calibration')}
    assert len(groups['fit'])==615 and len(groups['calibration'])==154 and not groups['fit']&groups['calibration']
    assert len({a['response_id'] for a in records})==3839
    # Only retained calibration paths are checked; no official-test files opened.
    gold=qa.read(ROOT/'data/gold_manifest.json')
    for p in paths[2:]:
        record=next(r for r in gold['outputs'] if (ROOT/r['path']).resolve()==p.resolve());assert qa.sha(p)==record['sha256']
    return records,tokens,paths


class CacheDataset:
    def __init__(self,answers,tokens,manifests=CACHES):
        self.answers=answers;self.tokens=tokens;self.cache={};self.manifests=tuple(Path(p) for p in manifests)
        for path in self.manifests:
            data=qa.read(path);assert data['status']=='complete' and not data.get('test_opened',False)
            for row in data['records']:
                rid=row['response_id'];assert rid not in self.cache,'Duplicate encoder22 cache identity'
                self.cache[rid]=(path.parent/'encoder22_features'/f'{rid}.npz',row)
        assert set(self.cache)=={a['response_id'] for a in answers},'Need all3680fit +159cal caches; no subset fallback'
    def load(self,i,verify_hash=True):
        answer=self.answers[i];t=self.tokens[i];rid=answer['response_id'];path,row=self.cache[rid];side=qa.read(path.with_suffix('.json'))
        if verify_hash:assert qa.sha(path)==row['npz_sha256']==side['npz_sha256'] and qa.sha(path.with_suffix('.json'))==row['metadata_sha256']
        assert side['response_id']==rid and side['partition']==answer['partition'] and side['original_answer_sha256']==answer['answer_sha256']
        assert side['hidden_dimension']==1024 and side['dtype']=='float32'
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        hidden=a['hidden22'];offset=a['sequence_offsets'];assert hidden.dtype==np.float32 and hidden.shape[1]==1024 and np.isfinite(hidden).all()
        assert offset[0]==0 and offset[-1]==len(hidden) and np.all((np.diff(offset)>0)&(np.diff(offset)<=512))
        assert a['input_ids'].shape==a['attention_mask'].shape==a['answer_token_start'].shape==a['answer_token_end'].shape==(len(hidden),)
        assert np.all(a['attention_mask']==1) and len(a['claim_index'])==len(a['document_index'])==len(offset)-1
        mapping=character_map(answer['original_response'],t['response_token_offsets'],a['answer_token_start'],a['answer_token_end'])
        mapped=set(mapping[0].tolist());assert all(j in mapped for j,islex in enumerate(t['lexical_mask']) if islex)
        return a,mapping


def token_weights(answers,tokens):
    fit=[i for i,a in enumerate(answers) if a['partition']=='fit'];tree=defaultdict(list)
    for i in fit:tree[answers[i]['group_id']].append(i)
    assert fit==list(range(3680))
    original_ids={a['response_id'] for a in qa.lines(ROOT/'data/answers_fit.jsonl')}
    assert len(original_ids)==634
    strata={group:{True:[],False:[]} for group in tree}
    for group,ids in tree.items():
        for i in ids:strata[group][answers[i]['response_id'] in original_ids].append(i)
        assert strata[group][True]
    base=[];label=[];bounds=[];cursor=0
    for i in fit:
        lex=np.asarray(tokens[i]['lexical_mask'],bool);y=np.asarray(tokens[i]['risk_mask'],int);assert lex.any() and not y[~lex].any()
        buckets=strata[answers[i]['group_id']];is_original=answers[i]['response_id'] in original_ids
        stratum_mass=.5 if buckets[False] else 1.
        b=lex.astype(np.float64)*stratum_mass/(len(buckets[is_original])*lex.sum());base.append(b);label.append(y);bounds.append((cursor,cursor+len(y)));cursor+=len(y)
    b=np.concatenate(base);y=np.concatenate(label);mass=sum(int(np.asarray(tokens[i]['lexical_mask']).sum()) for i in fit);b*=mass/b.sum()
    class_mass=np.bincount(y,weights=b,minlength=2);factors=mass/(2*class_mass);loss=b*factors[y]
    for ids in tree.values():
        inds=np.concatenate([np.arange(*bounds[i]) for i in ids]);loss[inds]*=(mass/len(tree))/loss[inds].sum()
    loss*=mass/loss.sum()
    return {'base':b,'loss':loss,'y':y,'bounds':bounds,'target_mass':mass,'class_factors':factors,'fit_answers':len(fit),'groups':len(tree),
        'original_membership':np.asarray([answers[i]['response_id'] in original_ids for i in fit],bool),
        'groups_without_auxiliary':sum(not bins[False] for bins in strata.values())}


def ensure_weights(answers,tokens):
    weights=token_weights(answers,tokens);path=OUT/'training_weights.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(weights)
            for key,value in weights.items():assert np.array_equal(z[key],np.asarray(value))
    else:np.savez_compressed(path,**weights)
    return weights


def data_check():
    answers,tokens,_=metadata();w=token_weights(answers,tokens);groups=defaultdict(list)
    for i,a in enumerate(answers[:3680]):groups[a['group_id']].append(i)
    entries=[];unit=w['target_mass']/615
    for group,ids in groups.items():
        old=sum(w['base'][slice(*w['bounds'][i])].sum() for i in ids if w['original_membership'][i])
        aux=sum(w['base'][slice(*w['bounds'][i])].sum() for i in ids if not w['original_membership'][i])
        assert np.isclose(old+aux,unit)
        assert (np.isclose(old,unit/2) and np.isclose(aux,unit/2)) if aux else np.isclose(old,unit)
        entries.append({'group_id':group,'original_base_mass':old,'auxiliary_base_mass':aux})
    qa.save(OUT/'DATA_WEIGHT_CHECK.json',{'passed':True,'fit_answers':3680,'original_answers':int(w['original_membership'].sum()),
        'auxiliary_answers':int((~w['original_membership']).sum()),'fit_groups':615,'calibration_answers':159,
        'fit_raw_tokens':len(w['y']),'target_lexical_mass':w['target_mass'],'groups_without_auxiliary':w['groups_without_auxiliary'],
        'class_factors':w['class_factors'].tolist(),'original_aux_mass_per_group':entries,'trained':False,'GPU_used':False})
    print('TAIL_DATA_WEIGHT_CHECK_PASSED',len(w['y']),w['target_mass'],flush=True)


def tiny_test():
    torch.set_num_threads(4);assert not torch.cuda.is_initialized();torch.manual_seed(SEED)
    config=RobertaConfig(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=24,num_attention_heads=4,
        max_position_embeddings=64,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,pad_token_id=1)
    config._attn_implementation='eager';full=RobertaModel(config,add_pooling_layer=False).eval()
    ids=torch.randint(3,97,(2,16));mask=torch.ones(2,16,dtype=torch.long);mask[1,11:]=0;ids[1,11:]=1
    for p in full.parameters():p.requires_grad_(False)
    with torch.no_grad():oracle=full(ids,attention_mask=mask,output_hidden_states=True);cache=oracle.hidden_states[22].detach()
    tail=TailTokenDetector(config,[copy.deepcopy(layer) for layer in full.encoder.layer[22:24]]).eval()
    for p in tail.parameters():p.requires_grad_(True)
    with torch.no_grad():replay=tail.tail_hidden(cache,mask)
    error=float((replay-oracle.last_hidden_state).abs().max());assert error<2e-6
    # Trimming cached padding and repadding must preserve valid positions.
    trimmed=cache.clone();trimmed[1,11:]=0
    with torch.no_grad():trim=tail.tail_hidden(trimmed,mask)
    assert torch.allclose(trim[mask.bool()],oracle.last_hidden_state[mask.bool()],atol=2e-6,rtol=1e-6)
    text='A中! B';mapping=character_map(text,[[0,2],[2,3],[3,4],[4,5]],np.asarray([-1,0,1,1,2,4]),np.asarray([-1,1,2,2,3,5]))
    log=torch.tensor([100.,2.,4.,8.,10.,12.],requires_grad=True);mapped=mapped_logits(log,mapping,4)
    assert torch.equal(mapped,torch.tensor([4.,10.,0.,12.]));mapped.sum().backward();assert log.grad[0]==0 and log.grad[2]==log.grad[3]
    # Cache does not require grad: non-reentrant checkpoint still trains both tail blocks and head.
    tail.train();before={k:v.detach().clone() for k,v in full.state_dict().items()};optimizer=torch.optim.AdamW(tail.parameters(),lr=1e-3)
    logits=tail(cache,mask,checkpointed=True);loss=F.binary_cross_entropy_with_logits(logits[mask.bool()],torch.ones_like(logits[mask.bool()]));loss.backward()
    norms=[float(sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None)) for layer in tail.layers]
    assert all(np.isfinite(v) and v>0 for v in norms) and tail.head.weight.grad is not None and cache.grad is None
    assert all(p.grad is None for p in full.parameters());optimizer.step()
    assert all(torch.equal(v,full.state_dict()[k]) for k,v in before.items())
    config_large=RobertaConfig.from_json_file(str(MODEL/'config.json'));config_large._attn_implementation='eager'
    with torch.device('meta'):count=sum(p.numel() for p in TailTokenDetector(config_large).parameters())
    assert count==25193473 and not torch.cuda.is_initialized()
    result={'passed':True,'tiny_model_layers':24,'cached_boundary':22,'replay_max_abs_error':error,
        'trimmed_padding_valid_positions_match':True,'unicode_mapping_and_gradient_test_passed':True,
        'nonreentrant_checkpoint_with_detached_cache_gradient_norms':norms,'front22_and_embeddings_unchanged_no_grad':True,
        'real_tail_parameter_count':count,'actual_pretrained_model_or_GPU_run':False,'trained_real_data':False}
    qa.save(OUT/'CPU_TINY_TEST.json',result);qa.save(OUT/'protocol.json',protocol())
    qa.save(OUT/'code_freeze.json',{'script_sha256':qa.sha(Path(__file__)),'protocol_sha256':qa.sha(OUT/'protocol.json'),'test_sha256':qa.sha(OUT/'CPU_TINY_TEST.json')})
    print('TAIL_CPU_TINY_PASSED',result,flush=True)


def prepare():
    assert qa.read(OUT/'CPU_TINY_TEST.json')['passed'] and qa.sha(Path(__file__))==qa.read(OUT/'code_freeze.json')['script_sha256']
    answers,tokens,paths=metadata();dataset=CacheDataset(answers,tokens);weight=token_weights(answers,tokens)
    cache_rows=[]
    for i in range(len(answers)):
        arrays,mapping=dataset.load(i);cache_rows.append({'response_id':answers[i]['response_id'],'valid_tokens':len(arrays['hidden22']),
            'sequences':len(arrays['sequence_offsets'])-1,'raw_tokens':tokens[i]['token_count'],'mapping_entries':len(mapping[0])})
        if (i+1)%200==0:print('TAIL_CACHE_CPU_REVIEW',i+1,len(answers),flush=True)
    ensure_weights(answers,tokens)
    sources=paths+list(dataset.manifests)+[MODEL/'config.json',MODEL/'pytorch_model.bin',Path(__file__),OUT/'protocol.json']
    snapshot={'files_sha256':{str(p.resolve()):qa.sha(p) for p in sources},'test_opened':False}
    qa.save(OUT/'source_snapshot.json',snapshot);qa.save(OUT/'cache_review.json',{'status':'passed','rows':cache_rows,'missing_coverage':0,'test_opened':False})
    qa.save(OUT/'preparation_complete.json',{'status':'complete','fit_answers':3680,'calibration_answers':159,'fit_groups':615,
        'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json'),'weights_sha256':qa.sha(OUT/'training_weights.npz'),
        'training_started':False,'GPU_used':False});print('TAIL_PREPARED_NO_GPU_TRAINING',flush=True)


@torch.no_grad()
def evaluate_calibration(model,dataset,device):
    model.eval();token_predictions={}
    for i in range(3680,3839):
        arrays,mapping=dataset.load(i,verify_hash=False);tokens=dataset.tokens[i];answer=dataset.answers[i]
        p=torch.sigmoid(answer_logits(model,arrays,mapping,tokens['token_count'],device,False)).cpu().numpy();token_predictions[answer['response_id']]=p
    return evaluate_probabilities(dataset.answers,dataset.tokens,token_predictions)


def evaluate_probabilities(answers,tokens_list,token_predictions):
    window_scores=[];window_labels=[];answer_scores=[];answer_labels=[]
    for i in range(3680,3839):
        tokens=tokens_list[i];answer=answers[i];p=token_predictions[answer['response_id']]
        lex=np.asarray(tokens['lexical_mask'],bool);risk=np.asarray(tokens['risk_mask'],int);n=len(p);scores=[]
        for start in range(max(1,n-3)):
            ix=np.arange(start,min(start+4,n))
            if not lex[ix].any():continue
            score=float(p[ix[lex[ix]]].max());scores.append(score);window_scores.append(score);window_labels.append(int(risk[ix].any()))
        answer_scores.append(max(scores));answer_labels.append(answer['label'])
    assert len(window_scores)==42241 and len(answer_scores)==159
    w=np.asarray(window_scores);a=np.asarray(answer_scores);tw=qa.choose_threshold(window_labels,w);ta=qa.choose_threshold(answer_labels,a)
    return {'window':tw,'answer':ta},{'windows':qa.count(window_labels,w,tw['threshold']),'answers':qa.count(answer_labels,a,ta['threshold'])},w,a,token_predictions


def frozen_prepare():
    """CPU-only preparation of the matched zero-unfrozen-layer head baseline."""
    from scipy.sparse import csr_matrix
    assert qa.read(OUT/'CPU_TINY_TEST.json')['passed'] and qa.sha(Path(__file__))==qa.read(OUT/'code_freeze.json')['script_sha256']
    answers,tokens,paths=metadata();ensure_weights(answers,tokens);directory=OUT/'frozen_head';directory.mkdir(exist_ok=True)
    manifests=[EXP/'minicheck/claim_feature_manifest.json',ROOT/'semantic_baseline/cuda_variant/claim_feature_manifest.json'];cache={}
    for path in manifests:
        manifest=qa.read(path);assert manifest['status']=='complete'
        for r in manifest['records']:
            rid=r['response_id'];assert rid not in cache;cache[rid]=(path.parent/'claim_features'/f'{rid}.npz',r)
    assert set(cache)=={a['response_id'] for a in answers},'Require complete new3046+old793 final claim features'
    total=sum(t['token_count'] for t in tokens);mapped=np.lib.format.open_memmap(directory/'mapped_hidden.npy',mode='w+',dtype=np.float32,shape=(total,1024))
    cursor=0;index=[]
    for answer,t in zip(answers,tokens):
        rid=answer['response_id'];path,record=cache[rid];side=qa.read(path.with_suffix('.json'))
        assert qa.sha(path)==record['npz_sha256']==side['npz_sha256'] and qa.sha(path.with_suffix('.json'))==record['metadata_sha256']
        assert side['partition']==answer['partition'] and side['original_answer_sha256']==answer['answer_sha256']
        with np.load(path,allow_pickle=False) as z:h=z['hidden_last'];start=z['token_start'];end=z['token_end']
        assert h.dtype==np.float32 and h.shape[1]==1024
        rows,cols,values=character_map(answer['original_response'],t['response_token_offsets'],start,end);n=t['token_count']
        matrix=csr_matrix((values,(rows,cols)),shape=(n,len(h)));mapped[cursor:cursor+n]=matrix@h
        index.append({'response_id':rid,'partition':answer['partition'],'left':cursor,'right':cursor+n});cursor+=n
    mapped.flush();qa.save(directory/'index.json',{'answers':index})
    sources=paths+manifests+[Path(__file__),OUT/'protocol.json',OUT/'training_weights.npz']
    qa.save(directory/'preparation_complete.json',{'status':'complete','source_files_sha256':{str(p.resolve()):qa.sha(p) for p in sources},
        'mapped_hidden_sha256':qa.sha(directory/'mapped_hidden.npy'),'index_sha256':qa.sha(directory/'index.json'),'trained':False,'GPU_used':False})
    print('FROZEN_HEAD_PREPARED_CPU_ONLY',total,flush=True)


def frozen_train():
    directory=OUT/'frozen_head';prep=qa.read(directory/'preparation_complete.json');assert prep['status']=='complete' and not (directory/'started.json').exists()
    for name,h in prep['source_files_sha256'].items():assert qa.sha(name)==h
    assert qa.sha(directory/'mapped_hidden.npy')==prep['mapped_hidden_sha256'] and qa.sha(directory/'index.json')==prep['index_sha256']
    answers,tokens,_=metadata();index=qa.read(directory/'index.json')['answers'];x=np.load(directory/'mapped_hidden.npy',mmap_mode='r')
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    config=RobertaConfig.from_json_file(str(MODEL/'config.json'))
    # Same initialization as TailTokenDetector.head without instantiating the tail.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);head=nn.Linear(1024,1);nn.init.normal_(head.weight,0.,config.initializer_range);nn.init.zeros_(head.bias)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);rng=np.random.default_rng(SEED);mass=int(weights['target_mass'])
    qa.save(directory/'started.json',{'utc':time.time(),'protocol_sha256':qa.sha(OUT/'protocol.json')});history=[];best=None
    for epoch in range(4):
        clock=time.perf_counter();training_loss=0.
        if epoch:
            order=rng.permutation(3680);head.train()
            for left in range(0,len(order),8):
                ids=order[left:left+8];optimizer.zero_grad(set_to_none=True)
                for i in ids:
                    a=index[i];xx=torch.from_numpy(np.array(x[a['left']:a['right']],copy=True));logits=head(xx).squeeze(-1)
                    lo,hi=weights['bounds'][i];y=torch.as_tensor(weights['y'][lo:hi],dtype=torch.float32);w=torch.as_tensor(weights['loss'][lo:hi],dtype=torch.float32)
                    loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()*(3680/(len(ids)*mass));loss.backward();training_loss+=float(loss.detach())
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        predictions={};head.eval()
        with torch.no_grad():
            for i in range(3680,3839):
                a=index[i];p=torch.sigmoid(head(torch.from_numpy(np.array(x[a['left']:a['right']],copy=True))).squeeze(-1)).numpy();predictions[a['response_id']]=p
        ts,metrics,w,a,tp=evaluate_probabilities(answers,tokens,predictions);key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-epoch]
        name=f'epoch_{epoch:02d}';torch.save({'head_state_dict':head.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch,'protocol_sha256':qa.sha(OUT/'protocol.json')},directory/(name+'.pt'))
        np.savez_compressed(directory/(name+'_cal_scores.npz'),window_scores=w,answer_scores=a,**{'token_'+rid:p for rid,p in tp.items()})
        entry={'epoch':epoch,'thresholds':ts,'metrics':metrics,'selection_key':key,'online_training_loss_sum':training_loss,'seconds':time.perf_counter()-clock,'test_opened':False}
        qa.save(directory/(name+'.json'),entry);history.append(entry)
        if epoch and (best is None or key>best['selection_key']):best=entry
    qa.save(directory/'complete.json',{'status':'complete_development_only','selected':best,'all_epochs':history,'test_opened':False,'GPU_used':False,'tail_trainable':False,'head_parameters':1025})
    print('FROZEN_HEAD_TRAINING_COMPLETE',flush=True)


def train():
    # Explicit real-cache equality/peak-memory gate is produced by a separately scheduled GPU check.
    assert qa.read(OUT/'preparation_complete.json')['status']=='complete'
    gate=qa.read(OUT/'GPU_SELFCHECK.json');assert gate['passed'] and gate['source_snapshot_sha256']==qa.sha(OUT/'source_snapshot.json')
    assert qa.sha(Path(__file__))==qa.read(OUT/'code_freeze.json')['script_sha256'] and qa.read(OUT/'protocol.json')==protocol()
    assert not (OUT/'started.json').exists()
    for name,h in qa.read(OUT/'source_snapshot.json')['files_sha256'].items():assert qa.sha(name)==h
    answers,tokens,_=metadata();dataset=CacheDataset(answers,tokens)
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    torch.manual_seed(SEED);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_pretrained_tail().to('cuda');optimizer=torch.optim.AdamW([{'params':model.layers.parameters(),'lr':2e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
    order=np.random.default_rng(SEED).permutation(3680);rng=np.random.default_rng(SEED);mass=int(weights['target_mass'])
    qa.save(OUT/'started.json',{'utc':time.time(),'protocol_sha256':qa.sha(OUT/'protocol.json')});history=[];best=None
    for epoch in range(4):
        training_loss=0.;start=time.perf_counter()
        if epoch:
            model.train();order=rng.permutation(3680)
            for left in range(0,len(order),8):
                ids=order[left:left+8];optimizer.zero_grad(set_to_none=True)
                for i in ids:
                    arrays,mapping=dataset.load(int(i),verify_hash=False);n=tokens[int(i)]['token_count'];logits=answer_logits(model,arrays,mapping,n,'cuda',True)
                    a,b=weights['bounds'][i];y=torch.as_tensor(weights['y'][a:b],device='cuda',dtype=torch.float32);w=torch.as_tensor(weights['loss'][a:b],device='cuda',dtype=torch.float32)
                    loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()*(3680/(len(ids)*mass));assert torch.isfinite(loss)
                    loss.backward();training_loss+=float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if (left+len(ids))%200==0:print('TAIL_TRAIN_PROGRESS',epoch,left+len(ids),3680,flush=True)
        ts,metrics,w,a,tp=evaluate_calibration(model,dataset,'cuda');key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-epoch]
        name=f'epoch_{epoch:02d}';torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch,
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state(),'protocol_sha256':qa.sha(OUT/'protocol.json')},OUT/(name+'.pt'))
        np.savez_compressed(OUT/(name+'_cal_scores.npz'),window_scores=w,answer_scores=a,**{'token_'+rid:p for rid,p in tp.items()})
        entry={'epoch':epoch,'thresholds':ts,'metrics':metrics,'selection_key':key,'online_training_loss_sum':training_loss,
            'seconds':time.perf_counter()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'test_opened':False}
        qa.save(OUT/(name+'.json'),entry);history.append(entry)
        if epoch and (best is None or key>best['selection_key']):best=entry
    qa.save(OUT/'complete.json',{'status':'complete_development_only','selected':best,'all_epochs':history,'test_opened':False,
        'extra_checker_finetuned':True,'original_generator_finetuned':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tiny-test','data-check','prepare','train','frozen-prepare','frozen-train']);args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):{'tiny-test':tiny_test,'data-check':data_check,'prepare':prepare,'train':train,'frozen-prepare':frozen_prepare,'frozen-train':frozen_train}[args.stage]()
