"""All-document token supervision: frozen MiniCheck head versus last-two-layer tuning.

Only gpu-smoke/train-tail touch CUDA. Protocol and CPU tests can be frozen before
cache completion; prepare requires every development answer and every doc/claim.
No per-document gold is manufactured. This is an extra verifier baseline.
"""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel
from threadpoolctl import threadpool_limits
import run_development as qa
import tail_finetune as old

ROOT=qa.ROOT; EXP=ROOT/'fit_expansion'; OUT=ROOT/'results/minicheck_tail_all_docs_v2'
OLD=ROOT/'semantic_baseline/cuda_variant'; NEW=EXP/'minicheck'; BACK=EXP/'minicheck_backfill'; EXTRA=EXP/'minicheck_unselected'
MODEL=old.MODEL; SEED=old.SEED; NFIT=3680; NDEV=3839
PLANS=(OLD/'plans.jsonl',NEW/'new_plans.jsonl')
MANIFESTS=(NEW/'encoder22_feature_manifest.json',BACK/'encoder22_feature_manifest.json',
           NEW/'claim_feature_manifest.json',OLD/'claim_feature_manifest.json',
           EXTRA/'feature_manifest.json')


def protocol():
    p=old.protocol();p.update({
        'version':'qa-minicheck-all-document-token-v2',
        'selection_of_documents':'Use complete Cartesian product of original fixed claims and document chunks. Original maximum-support choices identify cache locations only; no documents discarded.',
        'token_loss':'For each document, map its full-answer claim token logits to original raw BPE using nonwhitespace character overlap; total logit=min across real documents, then unchanged weighted original binary BCE. No per-doc label.',
        'aggregation':'Within-document character mean -> raw BPE logit; min logit over document_index sorted ascending (torch.min first tie); sigmoid -> max lexical positions per original4raw window -> max eligible windows per answer.',
        'supervision_interpretation':'All-context weak multiple-instance adaptation, not exact Luna per-window label-adjustment training; hard min expresses existential single-document support and has sparse winner gradients.',
        'frozen_head_control':'All-doc original final states mapped per document to raw BPE; same Linear1024->1 head, initialization/lr/weights/three answer orders on CPU; same min across documents. Cached tail eval; tuned tail uses checkpoint dropout.',
        'real_data_gate':'All3839 answers,3680fit/159cal; 615/154 disjoint groups. Every C*D pair exactly once in selected+unselected encoder22 AND final-state caches; same input token IDs and original coordinates 1:1.',
        'numeric_smoke':{'indices':'first/last fit, first/last cal, longest total valid input answer, first multi-document answer (deduplicated)',
            'replay_max_abs_tolerance':2e-4,'repeat_max_abs_tolerance':2e-6,
            'purpose':'Only same-doc hidden replay, finite nonzero detached-cache/checkpoint gradients, and actual memory/time. No threshold/architecture/budget changes from smoke; failures stop.',
            'target':'Synthetic all-zero risk for smoke gradient only, never a training label.'},
        'metrics':'Each epoch0..3 saves full deterministic fit and cal token/window/answer scores; fit metrics use that epoch cal thresholds. Full-fit weighted BCE divided by560300. Epoch0 diagnostics not selectable.',
        'artifact_policy':'Preserve all checkpoints with optimizer and RNG, answer orders, complete predictions, every epoch metrics, source and code hashes; no automatic retry/resume or subset training.',
        'cache_dtype':'float32 per sequence, complete valid doc+special+claim input; final-state cache may have only claim positions. No padding approximation to a real missing doc.',
        'cache_path_binding':'Physical original793 backfill directory may be specified before prepare; it cannot change expected original plans/answers/doc choices/IDs/coordinates/numeric tolerances. Actual directory and every source hash are frozen in prepare and must match for all later commands.',
        'all_doc_counts':{'answers':3839,'fit_answers':3680,'calibration_answers':159,'all_pairs':36777,'maximum_docs_in_current_frozen_plan':2},
        'runtime':'CPU4 threads for frozen head; CUDA only after explicit scheduling and passed gpu-smoke for tail2. Same3epochs; full-fit evaluation adds a deterministic pass per epoch.'})
    return p


def fixed_source_paths():
    return [Path(__file__),Path(old.__file__),Path(qa.__file__),*PLANS,
        EXP/'data/export_freeze.json',EXP/'data/answers_fit.jsonl',EXP/'data/tokens_fit.jsonl',
        ROOT/'data/gold_manifest.json',ROOT/'data/answers_fit.jsonl',ROOT/'data/answers_calibration.jsonl',ROOT/'data/tokens_calibration.jsonl',
        MODEL/'config.json',MODEL/'pytorch_model.bin',ROOT/'semantic_baseline/download_manifest.json']


def load_plans(answers):
    plans={};roots={}
    for path in PLANS:
        for p in qa.lines(path):
            rid=p['response_id'];assert rid not in plans;plans[rid]=p;roots[rid]=path.parent
    assert set(plans)=={a['response_id'] for a in answers}
    total=0
    for a in answers:
        p=plans[a['response_id']];assert p['partition']==a['partition'] and p['group_id']==a['group_id'] and p['answer_sha256']==a['answer_sha256']
        c=len(p['claims']);d=len(p['document_chunks']);assert c>0 and 1<=d<=2 and len(p['pair_lengths'])==c*d
        assert all(0<n<=512 for n in p['pair_lengths']);total+=c*d
        for claim in p['claims']:assert a['original_response'][claim['start']:claim['end']]==claim['text']
    assert total==36777
    return plans,roots


def freeze():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol_freeze.json').exists():check_protocol();print('PROTOCOL_ALREADY_FROZEN');return
    for name in ('CPU_ALLDOC_TEST.json','REAL_SELECTED_INTERFACE_CHECK.json','EVALUATION_GEOMETRY_CHECK.json'):
        assert qa.read(OUT/name)['passed']
    answers,tokens,_=old.metadata();plans,_=load_plans(answers);weights=old.token_weights(answers,tokens)
    assert len(weights['y'])==665708 and weights['target_mass']==560300
    np.savez_compressed(OUT/'training_weights.npz',**weights)
    rng=np.random.default_rng(SEED);np.save(OUT/'answer_orders.npy',np.stack([rng.permutation(NFIT) for _ in range(3)]))
    qa.save(OUT/'protocol.json',protocol())
    index=[{'response_id':a['response_id'],'partition':a['partition'],'group_id':a['group_id'],
        'answer_sha256':a['answer_sha256'],'raw_tokens':t['token_count'],'lexical_tokens':sum(t['lexical_mask']),
        'claims':len(plans[a['response_id']]['claims']),'documents':len(plans[a['response_id']]['document_chunks']),
        'original_plan_sha256':qa.digest(plans[a['response_id']])} for a,t in zip(answers,tokens)]
    qa.save(OUT/'dataset_index.json',{'answers':index,'test_opened':False})
    paths=fixed_source_paths()+[OUT/x for x in ('protocol.json','training_weights.npz','answer_orders.npy','dataset_index.json','CPU_ALLDOC_TEST.json',
        'REAL_SELECTED_INTERFACE_CHECK.json','EVALUATION_GEOMETRY_CHECK.json')]
    qa.save(OUT/'protocol_freeze.json',{'status':'frozen_waiting_complete_caches','files_sha256':{str(p.resolve()):qa.sha(p) for p in paths},
        'fit_answers':NFIT,'calibration_answers':159,'fit_groups':615,'calibration_groups':154,'fit_raw_tokens':len(weights['y']),
        'target_loss_mass':weights['target_mass'],'original_fit_answers':634,'auxiliary_fit_answers':3046,'test_opened':False,'GPU_used':False})
    print('ALLDOC_PROTOCOL_FROZEN_3680_FIT_159_CAL',flush=True)


def check_protocol():
    f=qa.read(OUT/'protocol_freeze.json');assert qa.read(OUT/'protocol.json')==protocol()
    for p,h in f['files_sha256'].items():assert qa.sha(p)==h,p
    return f


def _manifest(path):
    m=qa.read(path);assert m['status']=='complete' and not m.get('test_opened',False),path
    rows={r['response_id']:r for r in m['records']};assert len(rows)==len(m['records'])
    return rows


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


class AllDocDataset:
    """Each returned pair has exact original (claim_index,document_index).

    Geometry is checked without assigning checker-token/per-document labels.
    Arrays are read per answer; never hold the full40GB cache in RAM.
    """
    def __init__(self,answers,tokens):
        self.answers=answers;self.tokens=tokens;self.plans,self.roots=load_plans(answers)
        self.encoder={};self.final={};self.source_files=set(MANIFESTS);self.cache_signatures={}
        for directory in (OLD,NEW,BACK,EXTRA):
            for name in ('inference_complete.json','inference_source_snapshot.json'):
                path=directory/name;assert path.exists(),path;self.source_files.add(path)
            snap=qa.digest(qa.read(directory/'inference_source_snapshot.json'));self.cache_signatures[directory]=snap
        for path in MANIFESTS[:-1]:
            assert qa.read(path)['source_snapshot_sha256']==self.cache_signatures[path.parent]
            is_encoder=path.name=='encoder22_feature_manifest.json';target=self.encoder if is_encoder else self.final
            for rid,r in _manifest(path).items():
                assert rid in self.plans
                npz=path.parent/('encoder22_features' if is_encoder else 'claim_features')/f'{rid}.npz'
                target.setdefault(rid,[]).append((npz,r));self.source_files.update((npz,npz.with_suffix('.json')))
        expected=set(self.plans);assert set(self.encoder)==set(self.final)==expected,'All3839 selected identities required'
        self.expected_extra={rid for rid,p in self.plans.items() if len(p['document_chunks'])>1}
        extra=_manifest(EXTRA/'feature_manifest.json');assert set(extra)==expected
        assert qa.read(EXTRA/'feature_manifest.json')['source_snapshot_sha256']==self.cache_signatures[EXTRA]
        self.bound={p['response_id']:p for p in qa.lines(EXTRA/'bound_plans.jsonl')};assert set(self.bound)==expected
        self.source_files.add(EXTRA/'bound_plans.jsonl')
        for rid,record in extra.items():
            bound=self.bound[rid];plan=self.plans[rid]
            assert bound['original_plan_sha256']==qa.digest(plan) and bound['plan']==plan
            assert (record['pairs']>0)==(rid in self.expected_extra)
            assert record['pairs']==len(bound['unselected_pairs'])
            if not record['pairs']:continue
            assert record['bound_plan_sha256']==qa.digest(bound)
            files={str((EXTRA/k).resolve()):h for k,h in record['files_sha256'].items()}
            self.source_files.update(Path(k) for k in files)
            for subdir,target in (('encoder22_features',self.encoder),('claim_features',self.final)):
                npz=EXTRA/subdir/f'{rid}.npz'
                rr={'npz_sha256':files[str(npz.resolve())],'metadata_sha256':files[str(npz.with_suffix('.json').resolve())]}
                target[rid].append((npz,rr))
        self.scorepaths={rid:self.roots[rid]/'scores'/f'{rid}.json' for rid in expected}
        self.source_files.update(self.scorepaths.values())

    def side(self,path,record,answer,verify_hash):
        side=qa.read(path.with_suffix('.json'));rid=answer['response_id'];p=self.plans[rid]
        if verify_hash:
            assert qa.sha(path)==record['npz_sha256']==side['npz_sha256'],path
            assert qa.sha(path.with_suffix('.json'))==record['metadata_sha256'],path
        assert side['response_id']==rid and side['partition']==answer['partition']
        assert side['original_answer_sha256']==answer['answer_sha256']
        assert side['hidden_dimension']==1024 and side['dtype']=='float32'
        assert side['source_snapshot_sha256']==self.cache_signatures[path.parent.parent]
        assert side.get('original_plan_sha256',side.get('plan_sha256'))==qa.digest(p),path
        if path.parent.parent==EXTRA:
            assert side['bound_plan_sha256']==qa.digest(self.bound[rid])
            assert side['reference_score_sha256']==qa.sha(self.scorepaths[rid])
            assert Path(side['reference_score_path']).resolve()==self.scorepaths[rid].resolve()
            assert side['score_row_sha256']==qa.sha(EXTRA/'scores'/f'{rid}.json')
        else:assert side['score_row_sha256']==qa.sha(self.scorepaths[rid]),path
        return side

    def load(self,i,verify_hash=False,need_hidden=True):
        a=self.answers[i];t=self.tokens[i];rid=a['response_id'];p=self.plans[rid]
        C=len(p['claims']);D=len(p['document_chunks']);expected={(c,d) for c in range(C) for d in range(D)}
        score=qa.read(self.scorepaths[rid]);assert score['plan_sha256']==qa.digest(p)
        support=np.asarray(score['support_by_claim_document']);assert support.shape==(C,D) and np.isfinite(support).all()
        chosen=np.argmax(support,axis=1);enc={};final={}
        for path,record in self.encoder[rid]:
            self.side(path,record,a,verify_hash);z=_arrays(path);h=z['hidden22'];off=z['sequence_offsets']
            assert h.dtype==np.float32 and h.ndim==2 and h.shape[1]==1024 and np.isfinite(h).all()
            n=len(h);assert off[0]==0 and off[-1]==n and np.all((np.diff(off)>0)&(np.diff(off)<=512))
            assert all(z[k].shape==(n,) for k in ('input_ids','attention_mask','answer_token_start','answer_token_end'))
            assert np.all(z['attention_mask']==1) and len(z['claim_index'])==len(z['document_index'])==len(off)-1
            for j,(ci,di) in enumerate(zip(z['claim_index'],z['document_index'])):
                key=(int(ci),int(di));assert key in expected and key not in enc
                assert (key[1]==chosen[key[0]])==(path.parent.parent!=EXTRA)
                lo,hi=map(int,off[j:j+2]);assert hi-lo==p['pair_lengths'][key[0]*D+key[1]]
                pair={k:z[k][lo:hi] for k in ('input_ids','answer_token_start','answer_token_end')}
                pair['hidden22']=h[lo:hi] if need_hidden else None
                pair['claim_index'],pair['document_index']=key;enc[key]=pair
        for path,record in self.final[rid]:
            self.side(path,record,a,verify_hash);z=_arrays(path);h=z['hidden_last'];n=len(h)
            if path.parent.parent==EXTRA:
                off=z['sequence_offsets'];assert off[0]==0 and off[-1]==n and np.all(np.diff(off)>0)
                assert len(z['claim_index'])==len(z['document_index'])==len(off)-1
                for j,(ci,di) in enumerate(zip(z['claim_index'],z['document_index'])):
                    lo,hi=off[j:j+2];assert np.all(z['token_claim_index'][lo:hi]==ci) and np.all(z['token_document_index'][lo:hi]==di)
                z['claim_index']=z['token_claim_index'];z['document_index']=z['token_document_index']
            assert h.dtype==np.float32 and h.shape==(n,1024) and np.isfinite(h).all()
            assert all(z[k].shape==(n,) for k in ('token_ids','token_start','token_end','input_token_index','claim_index','document_index'))
            keys=list(zip(z['claim_index'].tolist(),z['document_index'].tolist()))
            for key in sorted(set(keys)):
                assert key in expected and key not in final
                assert (key[1]==chosen[key[0]])==(path.parent.parent!=EXTRA)
                ix=np.flatnonzero((z['claim_index']==key[0])&(z['document_index']==key[1]))
                assert np.all(np.diff(ix)==1)
                final[key]={k:z[k][ix] for k in ('hidden_last','token_ids','token_start','token_end','input_token_index')}
        assert set(enc)==set(final)==expected,(rid,'Missing or duplicate doc/claim pair')
        docs=[]
        for di in range(D):
            pairs=[];starts=[];ends=[];fs=[];fe=[];fh=[]
            for ci in range(C):
                e=enc[(ci,di)];f=final[(ci,di)];ix=f['input_token_index']
                assert np.all((ix>=0)&(ix<len(e['input_ids']))) and np.all(np.diff(ix)>0)
                assert np.array_equal(e['input_ids'][ix],f['token_ids'])
                assert np.array_equal(e['answer_token_start'][ix],f['token_start']) and np.array_equal(e['answer_token_end'][ix],f['token_end'])
                assert np.array_equal(np.flatnonzero(e['answer_token_start']>=0),ix),'Final cache must cover every claim-intersecting input token'
                claim=p['claims'][ci]
                assert np.all((f['token_start']>=claim['start'])&(f['token_end']<=claim['end']))
                e['reference_last']=f['hidden_last'];e['claim_input_index']=ix;pairs.append(e)
                starts.extend(e['answer_token_start']);ends.extend(e['answer_token_end']);fs.extend(f['token_start']);fe.extend(f['token_end']);fh.append(f['hidden_last'])
            mapping=old.character_map(a['original_response'],t['response_token_offsets'],np.asarray(starts),np.asarray(ends))
            final_map=old.character_map(a['original_response'],t['response_token_offsets'],np.asarray(fs),np.asarray(fe))
            mapped=set(mapping[0].tolist());assert all(j in mapped for j,l in enumerate(t['lexical_mask']) if l)
            docs.append({'document_index':di,'pairs':pairs,'mapping':mapping,'final_mapping':final_map,'final_hidden':np.concatenate(fh)})
        nonspace=np.asarray([any(not ch.isspace() for ch in a['original_response'][s:e]) for s,e in t['response_token_offsets']],bool)
        return {'response_id':rid,'docs':docs,'n_raw':t['token_count'],'nonspace':nonspace}


def min_risk(logits,valid=None):
    assert logits.ndim==2
    if valid is not None:logits=logits.masked_fill(~valid[:,None],torch.inf)
    result=logits.min(dim=0).values;assert torch.isfinite(result).all()
    return result


def tail_logits(model,bundle,device,checkpointed=False):
    values=[]
    for doc in bundle['docs']:
        pairs=doc['pairs'];parts=[]
        for first in range(0,len(pairs),2):
            block=pairs[first:first+2];length=max(len(p['input_ids']) for p in block)
            h=torch.zeros(len(block),length,model.config.hidden_size,dtype=torch.float32,device=device);mask=torch.zeros(len(block),length,dtype=torch.long,device=device)
            for j,pair in enumerate(block):
                n=len(pair['input_ids']);h[j,:n]=torch.as_tensor(pair['hidden22'],device=device);mask[j,:n]=1
            z=model(h,mask,checkpointed=checkpointed)
            parts.extend(z[j,:len(pair['input_ids'])] for j,pair in enumerate(block))
        values.append(old.mapped_logits(torch.cat(parts),doc['mapping'],bundle['n_raw']))
    return min_risk(torch.stack(values))


def make_head():
    config=RobertaConfig.from_json_file(str(MODEL/'config.json'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);head=nn.Linear(1024,1);nn.init.normal_(head.weight,0.,config.initializer_range);nn.init.zeros_(head.bias)
    return head


def prepare():
    """Full cache/hash/geometry audit and CPU-only per-doc final-state mapping."""
    check_protocol();assert not torch.cuda.is_initialized();assert not (OUT/'preparation_complete.json').exists()
    answers,tokens,_=old.metadata();ds=AllDocDataset(answers,tokens);plans=ds.plans
    total=sum(t['token_count']*len(plans[a['response_id']]['document_chunks']) for a,t in zip(answers,tokens))
    matrix=np.lib.format.open_memmap(OUT/'mapped_final_hidden.npy',mode='w+',dtype=np.float32,shape=(total,1024))
    cursor=0;index=[];pairs=0;checks=[];start=time.perf_counter()
    for i,a in enumerate(answers):
        b=ds.load(i,verify_hash=True,need_hidden=False);ranges=[]
        for d in b['docs']:
            n=b['n_raw'];r,c,w=d['final_mapping'];sparse=csr_matrix((w,(r,c)),shape=(n,len(d['final_hidden'])))
            matrix[cursor:cursor+n]=sparse@d['final_hidden'];ranges.append([cursor,cursor+n]);cursor+=n;pairs+=len(d['pairs'])
        index.append({'response_id':a['response_id'],'partition':a['partition'],'doc_ranges':ranges,'nonspace':b['nonspace'].tolist()})
        checks.append({'response_id':a['response_id'],'documents':len(ranges),'claims':len(ds.plans[a['response_id']]['claims']),
            'all_pairs_exact':True,'encoder_final_token_coordinates_exact':True,'all_nonwhitespace_covered_per_doc':True})
        if (i+1)%100==0:print('ALLDOC_PREPARE',i+1,NDEV,round(time.perf_counter()-start,1),flush=True)
    assert cursor==total and pairs==36777;matrix.flush();del matrix
    qa.save(OUT/'mapped_index.json',{'answers':index});qa.save(OUT/'CACHE_GEOMETRY_CHECK.json',{'passed':True,'records':checks,'pairs':pairs,'test_opened':False})
    source={str(p.resolve()):qa.sha(p) for p in sorted(ds.source_files,key=str)}
    qa.save(OUT/'source_snapshot.json',{'files_sha256':source,'protocol_freeze_sha256':qa.sha(OUT/'protocol_freeze.json'),
        'backfill_directory':str(BACK.resolve()),'test_opened':False})
    artifacts=['mapped_final_hidden.npy','mapped_index.json','CACHE_GEOMETRY_CHECK.json','source_snapshot.json']
    qa.save(OUT/'preparation_complete.json',{'status':'complete','answers':NDEV,'fit_answers':NFIT,'pairs':pairs,'mapped_rows':total,
        'files_sha256':{str((OUT/p).resolve()):qa.sha(OUT/p) for p in artifacts},'seconds':time.perf_counter()-start,'GPU_used':False,'test_opened':False})
    print('ALLDOC_PREPARATION_COMPLETE',flush=True)


def check_prepared():
    check_protocol();p=qa.read(OUT/'preparation_complete.json');assert p['status']=='complete'
    for path,h in p['files_sha256'].items():assert qa.sha(path)==h,path
    snap=qa.read(OUT/'source_snapshot.json');assert snap['protocol_freeze_sha256']==qa.sha(OUT/'protocol_freeze.json')
    assert snap['backfill_directory']==str(BACK.resolve()),'Use the backfill path already frozen by prepare'
    for path,h in snap['files_sha256'].items():assert qa.sha(path)==h,path
    return p


def weight_data():
    with np.load(OUT/'training_weights.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def score_geometry(answers,tokens,probabilities,indices):
    ws=[];wy=[];ans=[];ay=[];ends=[0]
    for i in indices:
        p=probabilities[answers[i]['response_id']];t=tokens[i];n=len(p);assert n==t['token_count'] and np.isfinite(p).all()
        lex=np.asarray(t['lexical_mask'],bool);risk=np.asarray(t['risk_mask'],int);one=[]
        for start in range(max(1,n-3)):
            ix=np.arange(start,min(start+4,n));ix=ix[lex[ix]]
            if not len(ix):continue
            v=float(p[ix].max());one.append(v);ws.append(v);wy.append(int(risk[ix].any()))
        assert one;ans.append(max(one));ay.append(answers[i]['label']);ends.append(len(ws))
    return {'window_scores':np.asarray(ws),'window_labels':np.asarray(wy),'answer_scores':np.asarray(ans),
        'answer_labels':np.asarray(ay),'answer_window_offsets':np.asarray(ends)}


def metrics_for(scores,thresholds):
    return {'windows':qa.count(scores['window_labels'],scores['window_scores'],thresholds['window']['threshold']),
        'answers':qa.count(scores['answer_labels'],scores['answer_scores'],thresholds['answer']['threshold'])}


def cpu_state(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return value


class FrozenFeatures:
    def __init__(self):
        self.x=np.load(OUT/'mapped_final_hidden.npy',mmap_mode='r');self.index=qa.read(OUT/'mapped_index.json')['answers']
    def logits(self,head,i):
        row=self.index[i];zs=[]
        for left,right in row['doc_ranges']:
            x=torch.from_numpy(np.array(self.x[left:right],copy=True));z=head(x).squeeze(-1)
            z=z.masked_fill(~torch.as_tensor(row['nonspace'],dtype=torch.bool),0.);zs.append(z)
        return min_risk(torch.stack(zs))


def train(which):
    check_prepared();directory=OUT/which;assert which in ('frozen0','tail2');directory.mkdir(exist_ok=True)
    assert not (directory/'started.json').exists(),'A started run is retained; no implicit overwrite/resume'
    answers,tokens,_=old.metadata();weights=weight_data();mass=int(weights['target_mass']);assert mass==560300
    orders=np.load(OUT/'answer_orders.npy');assert orders.shape==(3,NFIT)
    device='cpu' if which=='frozen0' else 'cuda';torch.manual_seed(SEED);torch.set_num_threads(4)
    if which=='tail2':
        gate=qa.read(OUT/'GPU_SELFCHECK.json');assert gate['passed'] and gate['source_snapshot_sha256']==qa.sha(OUT/'source_snapshot.json')
        assert gate['protocol_freeze_sha256']==qa.sha(OUT/'protocol_freeze.json')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=old.load_pretrained_tail().to(device);ds=AllDocDataset(answers,tokens)
        optimizer=torch.optim.AdamW([{'params':model.layers.parameters(),'lr':2e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
        def logits(i,training):return tail_logits(model,ds.load(int(i)),device,checkpointed=training)
    else:
        assert not torch.cuda.is_initialized();model=make_head();features=FrozenFeatures()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        def logits(i,training):return features.logits(model,int(i))
    qa.save(directory/'started.json',{'utc':time.time(),'protocol_freeze_sha256':qa.sha(OUT/'protocol_freeze.json'),
        'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json'),'device':device})
    history=[];best=None;all_start=time.perf_counter()
    for epoch in range(4):
        start=time.perf_counter();online=0.;nsteps=0
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        if epoch:
            model.train();order=orders[epoch-1]
            for first in range(0,NFIT,8):
                ids=order[first:first+8];optimizer.zero_grad(set_to_none=True)
                for i in ids:
                    z=logits(i,True);lo,hi=weights['bounds'][i];y=torch.as_tensor(weights['y'][lo:hi],device=device,dtype=torch.float32)
                    w=torch.as_tensor(weights['loss'][lo:hi],device=device,dtype=torch.float32)
                    objective=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).sum()*(NFIT/(len(ids)*mass))
                    assert torch.isfinite(objective);objective.backward();online+=float(objective.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step();nsteps+=1
                if (first+len(ids))%200==0:print('ALLDOC_TRAIN',which,epoch,first+len(ids),NFIT,round(time.perf_counter()-start,1),flush=True)
        probabilities={};fit_bce=0.;model.eval()
        with torch.no_grad():
            for i,a in enumerate(answers):
                z=logits(i,False);assert torch.isfinite(z).all();probabilities[a['response_id']]=torch.sigmoid(z).cpu().numpy()
                if i<NFIT:
                    lo,hi=weights['bounds'][i];y=torch.as_tensor(weights['y'][lo:hi],device=device,dtype=torch.float32);w=torch.as_tensor(weights['loss'][lo:hi],device=device,dtype=torch.float32)
                    fit_bce+=float((F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).double().sum())
                if (i+1)%400==0:print('ALLDOC_EVAL',which,epoch,i+1,NDEV,flush=True)
        cal=score_geometry(answers,tokens,probabilities,range(NFIT,NDEV));fit=score_geometry(answers,tokens,probabilities,range(NFIT))
        assert len(cal['window_scores'])==42241 and len(fit['window_scores'])==653979
        ts={'window':qa.choose_threshold(cal['window_labels'],cal['window_scores']),'answer':qa.choose_threshold(cal['answer_labels'],cal['answer_scores'])}
        fit_metrics=metrics_for(fit,ts);cal_metrics=metrics_for(cal,ts)
        key=[min(ts['window']['f1'],ts['answer']['f1']),ts['window']['f1'],ts['window']['precision'],-epoch]
        stem=f'epoch_{epoch:02d}'
        state={'model_state_dict':cpu_state(model.state_dict()),'optimizer_state_dict':cpu_state(optimizer.state_dict()),'epoch':epoch,
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state().cpu() if device=='cuda' else None,
            'protocol_freeze_sha256':qa.sha(OUT/'protocol_freeze.json'),'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json')}
        torch.save(state,directory/(stem+'.pt'))
        np.savez_compressed(directory/(stem+'_token_predictions.npz'),**probabilities)
        np.savez_compressed(directory/(stem+'_scores.npz'),**{'fit_'+k:v for k,v in fit.items()},**{'cal_'+k:v for k,v in cal.items()})
        entry={'epoch':epoch,'thresholds':ts,'calibration':cal_metrics,'fit_at_cal_thresholds':fit_metrics,
            'fit_weighted_bce':fit_bce/mass,'online_minibatch_objective_sum':online,'optimizer_steps':nsteps,'selection_key':key,
            'seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            'artifacts_sha256':{s:qa.sha(directory/(stem+s)) for s in ('.pt','_token_predictions.npz','_scores.npz')},'test_opened':False}
        qa.save(directory/(stem+'.json'),entry);history.append(entry)
        if epoch and (best is None or key>best['selection_key']):best=entry
        print('ALLDOC_EPOCH_COMPLETE',which,epoch,cal_metrics['windows']['f1'],cal_metrics['answers']['f1'],flush=True)
    qa.save(directory/'complete.json',{'status':'complete_development_only','selected':best,'all_epochs':history,
        'seconds':time.perf_counter()-all_start,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'GPU_used':device=='cuda','test_opened':False,'extra_checker_baseline':True,'original_generator_finetuned':False})
    del model,optimizer
    if device=='cuda':torch.cuda.empty_cache()
    print('ALLDOC_TRAIN_COMPLETE_GPU_RELEASED' if device=='cuda' else 'ALLDOC_FROZEN_TRAIN_COMPLETE_CPU',flush=True)


def gpu_smoke():
    """Run only when root schedules GPU; no model/parameter selection is performed."""
    check_prepared();assert not (OUT/'GPU_SELFCHECK.json').exists();answers,tokens,_=old.metadata();ds=AllDocDataset(answers,tokens)
    indices=sorted({0,NFIT-1,NFIT,NDEV-1,max(range(NDEV),key=lambda i:sum(ds.plans[answers[i]['response_id']]['pair_lengths'])),
        next(i for i,a in enumerate(answers) if len(ds.plans[a['response_id']]['document_chunks'])>1)})
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=old.load_pretrained_tail().to('cuda').eval();torch.cuda.reset_peak_memory_stats();records=[];start=time.perf_counter()
    with torch.no_grad():
        for i in indices:
            bundle=ds.load(i);error=0.;repeat=0.;compared=0
            for doc in bundle['docs']:
                for pair in doc['pairs']:
                    h=torch.as_tensor(pair['hidden22'][None],device='cuda');mask=torch.ones(h.shape[:2],device='cuda',dtype=torch.long)
                    one=model.tail_hidden(h,mask)[0];two=model.tail_hidden(h,mask)[0];ix=pair['claim_input_index']
                    difference=float((one[ix]-torch.as_tensor(pair['reference_last'],device='cuda')).abs().max())
                    rr=float((one-two).abs().max());error=max(error,difference);repeat=max(repeat,rr);compared+=len(ix)
            assert error<=2e-4 and repeat<=2e-6,(i,error,repeat)
            # Same actual per-doc geometry and new head; no model scores used to pick data.
            zz=tail_logits(model,bundle,'cuda');frozen=[]
            for doc in bundle['docs']:
                q=torch.as_tensor(doc['final_hidden'],device='cuda');frozen.append(old.mapped_logits(model.head(q).squeeze(-1),doc['final_mapping'],bundle['n_raw']))
            dz=float((zz-min_risk(torch.stack(frozen))).abs().max());assert dz<=2e-4,(i,dz)
            records.append({'index':i,'response_id':answers[i]['response_id'],'claim_positions_compared':compared,
                'same_doc_replay_max_abs':error,'repeat_max_abs':repeat,'mapped_logit_difference':dz})
    # Largest total-input answer is a fixed resource stress case, not selected by gold.
    i=max(indices,key=lambda j:sum(ds.plans[answers[j]['response_id']]['pair_lengths']));bundle=ds.load(i)
    model.train();optimizer=torch.optim.AdamW([{'params':model.layers.parameters(),'lr':2e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
    optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();z=tail_logits(model,bundle,'cuda',checkpointed=True)
    mask=torch.as_tensor(bundle['nonspace'],device='cuda');loss=F.binary_cross_entropy_with_logits(z[mask],torch.zeros_like(z[mask]));loss.backward()
    grads=[]
    for layer in model.layers:
        gg=[p.grad for p in layer.parameters()];assert all(g is not None and torch.isfinite(g).all() for g in gg)
        norm=float(torch.sqrt(sum(g.double().square().sum() for g in gg)));assert norm>0;grads.append(norm)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    report={'passed':True,'source_snapshot_sha256':qa.sha(OUT/'source_snapshot.json'),'protocol_freeze_sha256':qa.sha(OUT/'protocol_freeze.json'),
        'records':records,'gradient_answer_index':i,'layer_gradient_norms':grads,'detached_front_checkpoint_nonreentrant':True,
        'synthetic_smoke_target_only':True,'discarded_smoke_model':True,'largest_answer_step_seconds':time.perf_counter()-tick,
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'device':torch.cuda.get_device_name(),'torch':torch.__version__,'float32':True,'test_opened':False,'hyperparameters_changed':False}
    qa.save(OUT/'GPU_SELFCHECK.json',report);del model,optimizer;torch.cuda.empty_cache();print('ALLDOC_GPU_SMOKE_PASSED_RELEASED',flush=True)


def tiny_test():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4);torch.manual_seed(SEED)
    config=RobertaConfig(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=24,num_attention_heads=4,
        max_position_embeddings=64,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,pad_token_id=1)
    config._attn_implementation='eager';full=RobertaModel(config,add_pooling_layer=False).eval()
    for p in full.parameters():p.requires_grad_(False)
    ids=torch.randint(3,97,(4,9));mask=torch.ones_like(ids);mask[1,7:]=0;ids[1,7:]=1
    with torch.no_grad():truth=full(ids,attention_mask=mask,output_hidden_states=True);hidden=truth.hidden_states[22].detach()
    tail=old.TailTokenDetector(config,[copy.deepcopy(l) for l in full.encoder.layer[22:24]])
    for p in tail.parameters():p.requires_grad_(True)
    tail.eval()
    with torch.no_grad():replayed=tail.tail_hidden(hidden,mask)
    error=float((replayed-truth.last_hidden_state).abs().max());assert error<2e-6
    text='A中! B';raw=[[0,2],[2,3],[3,4],[4,5]];starts=np.asarray([-1,0,1,1,2,4]);ends=np.asarray([-1,1,2,2,3,5])
    mapping=old.character_map(text,raw,starts,ends);fake=torch.tensor([100.,2.,4.,8.,10.,12.],requires_grad=True)
    mapped=old.mapped_logits(fake,mapping,4);assert torch.equal(mapped,torch.tensor([4.,10.,0.,12.]));mapped.sum().backward()
    assert fake.grad[2]==fake.grad[3] and fake.grad[0]==0
    z=torch.tensor([[-2.,2.],[2.,-2.]],requires_grad=True);out=min_risk(z)
    assert torch.equal(out,torch.tensor([-2.,-2.])) and torch.equal(torch.sigmoid(out),torch.sigmoid(z).min(dim=0).values)
    F.binary_cross_entropy_with_logits(out,torch.zeros_like(out)).backward();assert z.grad[0,0]>0 and z.grad[1,1]>0 and z.grad[0,1]==z.grad[1,0]==0
    padded=torch.cat((z.detach(),torch.full((1,2),-99.)));assert torch.equal(min_risk(padded,torch.tensor([True,True,False])),out.detach())
    tie=torch.zeros(2,1,requires_grad=True);min_risk(tie).sum().backward();assert tie.grad.tolist()==[[1.],[0.]]
    # Frozen feature averaging commutes with the same linear head for mapped characters.
    hh=torch.randn(6,32);r,c,w=mapping;matrix=torch.sparse_coo_tensor(torch.tensor(np.stack((r,c))),torch.tensor(w),(4,6)).to_dense()
    direct=old.mapped_logits(tail.head(hh).squeeze(-1),mapping,4);projected=tail.head(matrix@hh).squeeze(-1);nonspace=torch.tensor([True,True,False,True])
    assert torch.allclose(direct[nonspace],projected[nonspace],atol=1e-7)
    # Exercise the actual two-doc/two-claim forward and a BPE crossing claims.
    raw2=[[0,2],[2,5]];bundle={'n_raw':2,'docs':[]}
    for di in range(2):
        pairs=[];ss=[];ee=[];reference=[]
        for ci in range(2):
            k=di*2+ci;n=int(mask[k].sum());s=np.full(n,-1,np.int64);e=s.copy()
            if ci==0:s[2:6]=[0,1,1,2];e[2:6]=[1,2,2,3]
            else:s[2]=4;e[2]=5
            pairs.append({'input_ids':ids[k,:n].numpy(),'hidden22':hidden[k,:n].numpy()})
            ss.extend(s);ee.extend(e);reference.append(truth.last_hidden_state[k,:n])
        m=old.character_map(text,raw2,np.asarray(ss),np.asarray(ee))
        bundle['docs'].append({'pairs':pairs,'mapping':m,'reference':torch.cat(reference)})
    actual=tail_logits(tail,bundle,'cpu')
    oracle=min_risk(torch.stack([old.mapped_logits(tail.head(d['reference']).squeeze(-1),d['mapping'],2) for d in bundle['docs']]))
    assert torch.allclose(actual,oracle,atol=2e-7)
    # Detached input with non-reentrant checkpoint must train both tail layers.
    before=[p.detach().clone() for p in full.parameters()];tail.train();detached=hidden.detach();assert not detached.requires_grad
    risk=tail_logits(tail,bundle,'cpu',checkpointed=True)
    loss=F.binary_cross_entropy_with_logits(risk,torch.zeros_like(risk));loss.backward();norms=[]
    for layer in tail.layers:
        g=[p.grad for p in layer.parameters()];assert all(v is not None and torch.isfinite(v).all() for v in g)
        norm=float(torch.sqrt(sum(v.double().square().sum() for v in g)));assert norm>0;norms.append(norm)
    opt=torch.optim.AdamW(tail.parameters(),lr=2e-5);opt.step()
    assert all(p.grad is None and torch.equal(p,b) for p,b in zip(full.parameters(),before))
    # Original4raw windows count punctuation geometry but use lexical maxima only.
    aa=[{'response_id':'toy','label':1}];tt=[{'token_count':5,'lexical_mask':[True,False,False,True,True],'risk_mask':[0,0,0,1,0]}]
    geo=score_geometry(aa,tt,{'toy':np.array([.1,.99,.99,.8,.2])},[0]);assert geo['window_scores'].tolist()==[.8,.8] and geo['window_labels'].tolist()==[1,1]
    assert not torch.cuda.is_initialized()
    qa.save(OUT/'CPU_ALLDOC_TEST.json',{'passed':True,'tiny24layer_replay_max_abs':error,'tail_gradient_norms':norms,
        'unicode_duplicate_character_map':True,'min_after_doc_mapping':True,'same_head_mapping_commutes':True,'missing_real_doc_not_padded':True,
        'hard_min_winner_gradient':True,'padded_doc_excluded':True,'tie_first_doc':True,'detached_cache_checkpoint_gradients':True,
        'front_parameters_unchanged':True,'original_window_geometry':True,'GPU_used':False,'real_training_performed':False})
    print('ALLDOC_CPU_TINY_TEST_PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tiny-test','freeze','verify','prepare','gpu-smoke','train-frozen','train-tail'])
    parser.add_argument('--backfill-dir',type=Path,help='Physical original793 cache route only; fixed by prepare, no change to expected identities or numeric gates')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.backfill_dir is not None:
        BACK=args.backfill_dir.resolve();assert BACK.is_relative_to(EXP.resolve()) and BACK not in (NEW.resolve(),EXTRA.resolve())
        MANIFESTS=(NEW/'encoder22_feature_manifest.json',BACK/'encoder22_feature_manifest.json',
            NEW/'claim_feature_manifest.json',OLD/'claim_feature_manifest.json',EXTRA/'feature_manifest.json')
    with threadpool_limits(limits=4):
        try:
            {'tiny-test':tiny_test,'freeze':freeze,'verify':check_protocol,'prepare':prepare,'gpu-smoke':gpu_smoke,
                'train-frozen':lambda:train('frozen0'),'train-tail':lambda:train('tail2')}[args.stage]()
        except Exception:
            qa.save(OUT/f'{args.stage}_failure_{time.time_ns()}.json',{'stage':args.stage,'exception':traceback.format_exc(),
                'no_parameter_adjustment':True,'test_opened':False,'GPU_initialized':torch.cuda.is_initialized()})
            raise
