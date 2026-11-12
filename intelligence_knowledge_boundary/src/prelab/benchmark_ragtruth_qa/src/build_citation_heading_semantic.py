"""Frozen387 inherited-heading claims x three sources. Prepare/check are CPU only.

infer/all require a new explicit GPU scheduler authorization, never automatic.
No model training or thresholds here; future matched LR protocol is recorded.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import os
import shutil
import time
import numpy as np
import build_cited_source_semantic as old

ROOT=old.ROOT
OUT=ROOT/'results/citation_heading_semantic_v1'
SCOPE=ROOT/'results/citation_heading_scope_v1'
LEGACY=ROOT/'results/cited_source_semantic_v1'
NAMES=['inherited_any_source_support_mean','inherited_source_support_gap_mean']
read,lines,save,jsonl,sha,digest=old.read,old.lines,old.save,old.jsonl,old.sha,old.digest


def protocol():
    return {'version':'citation-heading-source-semantic-v1','answers':793,'original_claims':8852,
        'inherited_claims':387,'scored_answers':37,'source_claim_pairs':1161,
        'selection':'Exact frozen citation_heading_scope_v1/inherited_claims.json; do not rerun/extend parser or source attribution. These body claims had old parser status none and were NOT in earlier6252 source-specific pairs.',
        'claim':'Unmodified original claim text, including bullet/numbering; do not prepend inherited heading or rewrite references.',
        'source':'Original three separate source header/body strings, same split_full_sources as previous6252 experiment; retain all text/whitespace, no chunking/reordering/truncation.',
        'input':'source_text + tokenizer.eos_token + original_claim, standard special tokens, <=512. Pair source order1/2/3; per answer original claim order.',
        'arithmetic':'Reuse pinned local MiniCheck RoBERTa-large, FP32 CUDA eager/eval, TF32 off, no autocast, batch4 with original dynamic padding; official softmax(logits)[1]. Save both logits and every source support, do not replace source axis by old mixed chunks.',
        'columns':NAMES,
        'claim_values':'For inherited source j: [max(s1,s2,s3),max(s1,s2,s3)-s_j]. For all noninherited claims [0,0]. Source ID only selects a support value, is not a numeric feature.',
        'window_values':'Original frozen lexical token->claim map, mean over original lexical tokens per original4rawBPE window. All210364 windows retained; punctuation geometry unchanged.',
        'oracle':'Before new GPU pairs, repeat original16023 anchor in its original mixed-doc batch geometry. Same predeclared maxabs logits2e-4/support2e-5; do not relax. CPU preparation only checks synthetic math, input IDs/lengths/hashes and frozen coordinate identities.',
        'execution':'CPU prepare/check only until root separately authorizes infer/all. No GPU polling/automatic start, no training.',
        'limits':['This adds semantic checking for previously unscored inherited body claims; prior6252 cache did not contain them.',
                  'Cannot resolve step numbering, time/order, detached trailing references or arbitrary narrative scope.',
                  'Extra semantic model plus original generation signals, not a pure native white-box probe.',
                  'Repeated original fit/cal development; no official test; no labels used for pair selection or feature construction.']}


def training_protocol():
    return {'version':'heading-semantic-matched-lr-v1','status':'design_only_not_trained',
        'peers':['lookback','harp_claim','semantic_claim'],'modes':['any_source_control','inherited_source_gap'],
        'C':[.001,.01,.1],'planned_fits':18,
        'common14':'Frozen peer/tail2 two probabilities + exact original8 citation lexical features + exact frozen3 heading-scope features + new max-any-source support mean.',
        'treatment15':'Identical14 common columns plus only new max-any-source minus inherited-source support mean.',
        'equal_inference_budget':'Both receive same1161 new MiniCheck pairs. No old6252 score substitution. No-scope windows have both new semantic values0.',
        'fit':'Original634 fit /168123 windows, exact original q.base_weights base/loss/class/group scheme; loss mass168123. StandardScaler fitted only on fit with base weights.',
        'LR':{'solver':'liblinear','penalty':'l2','max_iter':2000,'random_state':20261010},
        'thresholds':'Same original159 calibration /42241 windows; independent window and answer F1 thresholds, ties precision then higher threshold.',
        'candidate_selection':'Per peer/mode max(min(windowF1,answerF1),windowF1,windowprecision,-C), exactly q.selection_key. All18 candidates/model/scaler/scores retained.',
        'answer':'Max score over ALL original eligible windows; original answers/labels and order unchanged.',
        'CPU_only_training':True,'training_authorized_now':False,'official_test_opened':False,
        'limits':'Upstream fit probabilities not cross-fitted; same repeatedly exposed calibration. Never describe this as an independent test or semantically certified labels.'}


def claim_values(source_id,support):
    if source_id is None:
        assert support is None
        return np.zeros(2,np.float64)
    assert source_id in (1,2,3)
    s=np.asarray(support,np.float64)
    assert s.shape==(3,) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    return np.array([s.max(),s.max()-s[source_id-1]],np.float64)


def aggregate(values,indices):
    mask=indices>=0;assert mask.any(1).all()
    return (values[np.maximum(indices,0)]*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]


def tiny():
    assert np.array_equal(claim_values(None,None),[0,0])
    assert np.allclose(claim_values(2,[.9,.2,.5]),[.9,.7])
    assert np.array_equal(claim_values(1,[.9,.2,.5]),[.9,0])
    v=np.array([[0,0],[.9,.7],[.4,.1]])
    ix=np.array([[0,0,-1,-1],[1,1,2,-1]])
    assert np.allclose(aggregate(v,ix),[[0,0],[2.2/3,1.5/3]])
    s='passage 1: A.\n\npassage 2: B.\n\npassage 3: C.\n'
    assert ''.join(x['text'] for x in old.split_full_sources(s))==s
    return {'passed':True,'synthetic_only':True,'max_gap_and_zero_scope_checked':True,
        'lexical_padding_denominator_checked':True,'source_split_byte_text_identity':True,
        'actual_GPU_oracle_run':False,'model_loaded':False,'GPU_used':False}


def source_files():
    return [Path(__file__),Path(old.__file__),ROOT/'src/build_citation_alignment.py',
        ROOT/'semantic_baseline/run_semantic.py',ROOT/'semantic_baseline/run_semantic_cuda.py',
        old.OLD/'download_manifest.json',old.OLD/'plans.jsonl',old.OLD/'scores/16023.json',
        SCOPE/'features_complete.json',SCOPE/'inherited_claims.json',SCOPE/'window_features.npy',
        SCOPE/'geometry.json',SCOPE/'design_freeze.json',
        LEGACY/'preparation_freeze.json',LEGACY/'claims.jsonl',LEGACY/'pair_plans.jsonl',
        LEGACY/'window_geometry.npz',LEGACY/'geometry.json',
        ROOT/'data/fit.jsonl',ROOT/'data/calibration.jsonl']


def prepare():
    assert not (OUT/'preparation_freeze.json').exists() and not (OUT/'inference_started.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);os.environ['TOKENIZERS_PARALLELISM']='false'
    scope_done=read(SCOPE/'features_complete.json');assert scope_done['status']=='complete' and scope_done['no_test'] and not scope_done['trained']
    for n,h in scope_done['files_sha256'].items():assert sha(SCOPE/n)==h,n
    prior=read(LEGACY/'preparation_freeze.json')
    for n in ('claims.jsonl','pair_plans.jsonl','window_geometry.npz','geometry.json'):
        assert sha(LEGACY/n)==prior['files_sha256'][n],n
    selected=read(SCOPE/'inherited_claims.json');assert len(selected)==387
    selected_by={(r['response_id'],r['claim_index']):r for r in selected};assert len(selected_by)==387
    assert len({k[0] for k in selected_by})==37
    prior_scored={(p['response_id'],c['claim_index']) for p in lines(LEGACY/'pair_plans.jsonl') for c in p['claims']}
    assert len(prior_scored)==2084 and not (prior_scored & selected_by.keys())
    allowed=('response_id','partition','original_response','retrieved_passages','answer_sha256')
    rows={r['response_id']:{k:r[k] for k in allowed} for part in ('fit','calibration') for r in lines(ROOT/'data'/f'{part}.jsonl')}
    claims=lines(LEGACY/'claims.jsonl');assert len(claims)==8852 and len(rows)==793
    all_claims=[];by_answer=defaultdict(list);mask=[]
    for i,c in enumerate(claims):
        assert c['global_claim_index']==i and c['partition']==rows[c['response_id']]['partition']
        assert rows[c['response_id']]['original_response'][c['start']:c['end']]==c['text']
        key=(c['response_id'],c['claim_index']);r=selected_by.get(key)
        if r:
            assert all(r[k]==c[k] for k in ('start','end','text','partition'))
            assert c['parser']==r['old_parser'] and c['parser']['status']=='none'
            a,b=r['scope']['start'],r['scope']['end']
            assert rows[c['response_id']]['original_response'][a:b]==r['scope']['text']
            cid=r['scope']['source_id'];assert cid in (1,2,3)
            by_answer[c['response_id']].append({**c,'inherited_source_id':cid,'inherited_header':r['scope']})
        all_claims.append({**c,'inherited_source_id':r['scope']['source_id'] if r else None})
        mask.append(float(bool(r)))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(old.OLD/'model',local_files_only=True,use_fast=True)
    plans=[];lengths=[];counts=Counter();oversized=[]
    for rid,cc in by_answer.items():
        row=rows[rid];sources=old.split_full_sources(row['retrieved_passages']);chosen=[]
        for c in cc:
            pairs=[]
            for source in sources:
                text=source['text']+tok.eos_token+c['text'];ids=tok.encode(text,add_special_tokens=True,truncation=False)
                pairs.append({'source_id':source['source_id'],'input_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'input_ids_sha256':digest(ids),'token_count':len(ids)})
                lengths.append(len(ids));counts[row['partition']+'_pairs']+=1
                if len(ids)>512:oversized.append({'response_id':rid,'claim_index':c['claim_index'],'source_id':source['source_id'],'length':len(ids)})
            chosen.append({k:c[k] for k in ('claim_index','global_claim_index','text','start','end','inherited_source_id','inherited_header')})
            chosen[-1]['pairs']=pairs;counts[row['partition']+'_claims']+=1
        counts[row['partition']+'_answers']+=1
        plans.append({'response_id':rid,'partition':row['partition'],'answer_sha256':row['answer_sha256'],
            'document_sha256':hashlib.sha256(row['retrieved_passages'].encode()).hexdigest(),'sources':sources,'claims':chosen})
    assert len(plans)==37 and len(lengths)==1161
    statistics={'pairs':len(lengths),'scored_claims':387,'scored_answers':37,'partition_counts':dict(counts),
        'input_tokens':sum(lengths),'minimum_length':min(lengths),'maximum_length':max(lengths),
        'mean_length':float(np.mean(lengths)),'over512':oversized,'source_chunking_required':bool(oversized),
        'old6252_pair_overlap':0,'no_model_loaded':True,'GPU_used':False,'trained':False,'official_test_opened':False}
    save(OUT/'preparation_statistics.json',statistics)
    assert not oversized,'Oversized pair; report and stop before inference/protocol change.'
    # Reuse exact geometry; no parsing or new tokenization of the detector output.
    shutil.copyfile(LEGACY/'window_geometry.npz',OUT/'window_geometry.npz')
    assert sha(OUT/'window_geometry.npz')==sha(LEGACY/'window_geometry.npz')
    with np.load(OUT/'window_geometry.npz') as z:
        ix=z['window_claim_indices'];fraction=aggregate(np.asarray(mask)[:,None],ix)[:,0].astype(np.float32)
        assert np.array_equal(fraction,np.load(SCOPE/'window_features.npy')[:,0])
        geometry=read(LEGACY/'geometry.json');assert geometry['window_order_sha256']==digest(z['window_ids'].tolist())
        assert ix.shape==(210364,4)
    jsonl(OUT/'pair_plans.jsonl',plans);jsonl(OUT/'claims.jsonl',all_claims)
    save(OUT/'feature_names.json',NAMES);save(OUT/'geometry.json',{**geometry,'heading_scope_fraction_exact':True,
        'reused_geometry_sha256':sha(LEGACY/'window_geometry.npz'),'old_scope_features_sha256':sha(SCOPE/'window_features.npy')})
    save(OUT/'protocol.json',protocol());save(OUT/'training_protocol.json',training_protocol());save(OUT/'CPU_SELFCHECK.json',tiny())
    names=['pair_plans.jsonl','claims.jsonl','window_geometry.npz','feature_names.json','geometry.json',
        'protocol.json','training_protocol.json','CPU_SELFCHECK.json','preparation_statistics.json']
    save(OUT/'preparation_freeze.json',{'status':'prepared_waiting_explicit_GPU_authorization',
        'files_sha256':{n:sha(OUT/n) for n in names},'source_sha256':{str(p.resolve()):sha(p) for p in source_files()},
        'no_test':True,'trained':False,'GPU_used':False})
    print('HEADING_SEMANTIC_PREPARED_NO_GPU',statistics,flush=True)


def check():
    f=read(OUT/'preparation_freeze.json');assert read(OUT/'protocol.json')==protocol()
    assert read(OUT/'training_protocol.json')==training_protocol()
    for n,h in f['files_sha256'].items():assert sha(OUT/n)==h,n
    for n,h in f['source_sha256'].items():assert sha(n)==h,n
    assert f['no_test'] and not f['trained']
    return f


def infer():
    check();assert not (OUT/'inference_started.json').exists()
    for n,h in read(old.OLD/'download_manifest.json')['files_sha256'].items():assert sha(old.OLD/n)==h,n
    plans=lines(OUT/'pair_plans.jsonl');assert len(plans)==37 and sum(len(p['claims'])*3 for p in plans)==1161
    save(OUT/'inference_started.json',{'time':time.time(),'pid':os.getpid(),'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'device':'cuda:0'})
    tick=time.perf_counter();spec=importlib.util.spec_from_file_location('heading_semantic_original_cuda',ROOT/'semantic_baseline/run_semantic_cuda.py')
    gpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gpu);model,tok=gpu.load_model()
    anchor=lines(old.OLD/'plans.jsonl')[0];texts,_=gpu.pairs(anchor,tok);z,p,_=gpu.probabilities(model,tok,texts)
    original=read(old.OLD/'scores'/f'{anchor["response_id"]}.json')
    dz=float(np.max(np.abs(z-np.asarray(original['logits']).reshape(-1,2))))
    dp=float(np.max(np.abs(p-np.asarray(original['support_by_claim_document']).ravel())))
    save(OUT/'numeric_agreement.json',{'response_id':anchor['response_id'],'logit_max_abs_diff':dz,'support_max_abs_diff':dp,
        'passed':dz<=2e-4 and dp<=2e-5,'reference_sha256':sha(old.OLD/'scores'/f'{anchor["response_id"]}.json')})
    assert dz<=2e-4 and dp<=2e-5,'Original arithmetic oracle failed; do not relax.'
    directory=OUT/'scores';directory.mkdir(exist_ok=True);records=[]
    for plan in plans:
        texts=[];identities=[]
        for c in plan['claims']:
            for source,pair in zip(plan['sources'],c['pairs']):
                text=source['text']+tok.eos_token+c['text'];ids=tok.encode(text,add_special_tokens=True,truncation=False)
                assert hashlib.sha256(text.encode()).hexdigest()==pair['input_sha256']
                assert digest(ids)==pair['input_ids_sha256'] and len(ids)==pair['token_count']<=512
                texts.append(text);identities.append({'claim_index':c['claim_index'],'global_claim_index':c['global_claim_index'],**pair})
        logits,support,_=gpu.probabilities(model,tok,texts)
        assert logits.shape==(len(texts),2) and support.shape==(len(texts),) and np.isfinite(logits).all()
        assert np.isfinite(support).all() and ((support>=0)&(support<=1)).all()
        p=directory/f'{plan["response_id"]}.json'
        save(p,{'response_id':plan['response_id'],'partition':plan['partition'],'pair_plan_sha256':digest(plan),
            'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'pairs':identities,'logits':logits.tolist(),
            'support_by_claim_source':support.reshape(-1,3).tolist(),'source_ids':[1,2,3],
            'claim_indices':[c['claim_index'] for c in plan['claims']],'dtype':'float32','device':'cuda:0','trained':False})
        records.append({'response_id':plan['response_id'],'path':str(p.relative_to(OUT)),'sha256':sha(p),'pairs':len(texts)})
    check();save(OUT/'inference_complete.json',{'status':'complete','answers':37,'pairs':1161,'records':records,
        'seconds':time.perf_counter()-tick,'numeric_agreement_sha256':sha(OUT/'numeric_agreement.json'),
        'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'no_test':True,'trained':False})
    del model
    import torch
    torch.cuda.empty_cache()
    print('HEADING_SEMANTIC_INFERENCE_COMPLETE',flush=True)


def export():
    check();assert not (OUT/'features_complete.json').exists()
    done=read(OUT/'inference_complete.json');assert done['status']=='complete' and done['pairs']==1161
    assert done['preparation_freeze_sha256']==sha(OUT/'preparation_freeze.json')
    plans={p['response_id']:p for p in lines(OUT/'pair_plans.jsonl')};scores={}
    for rec in done['records']:
        p=OUT/rec['path'];assert sha(p)==rec['sha256'];r=read(p);plan=plans[r['response_id']]
        assert r['pair_plan_sha256']==digest(plan) and r['source_ids']==[1,2,3]
        assert r['claim_indices']==[c['claim_index'] for c in plan['claims']]
        for ci,s in zip(r['claim_indices'],r['support_by_claim_source']):scores[(r['response_id'],ci)]=s
    assert len(scores)==387
    claims=lines(OUT/'claims.jsonl');values=[]
    for c in claims:
        s=scores.get((c['response_id'],c['claim_index']));assert (s is not None)==(c['inherited_source_id'] is not None)
        values.append(claim_values(c['inherited_source_id'],s))
    with np.load(OUT/'window_geometry.npz') as z:
        x=aggregate(np.asarray(values,np.float64),z['window_claim_indices']).astype(np.float32)
        window_ids=z['window_ids'].copy();response_ids=z['response_ids'].copy()
    assert x.shape==(210364,2) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
    no_scope=np.load(SCOPE/'window_features.npy')[:,0]==0;assert not x[no_scope].any()
    np.save(OUT/'window_features.npy',x);np.savez_compressed(OUT/'features.npz',features=x,window_ids=window_ids,response_ids=response_ids)
    save(OUT/'export_statistics.json',{'nonzero_per_column':(x!=0).sum(0).tolist(),'zero_without_scope_exact':True,
        'original_window_count':len(x),'gold_fields_used':False,'thresholds_selected':False})
    names=['window_features.npy','features.npz','feature_names.json','geometry.json','export_statistics.json',
        'protocol.json','training_protocol.json','preparation_freeze.json','inference_complete.json','numeric_agreement.json']
    save(OUT/'features_complete.json',{'status':'complete','rows':210364,'columns':2,'dtype':'float32',
        'window_order_sha256':digest(window_ids.tolist()),'files_sha256':{n:sha(OUT/n) for n in names},'no_test':True,'trained':False})
    print('HEADING_SEMANTIC_FEATURES_COMPLETE',x.shape,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','check','infer','export','all'))
    stage=parser.parse_args().stage
    if stage=='all':infer();export()
    else:
        globals()[stage]()
        if stage=='check':print('HEADING_SEMANTIC_FROZEN_CHECK_PASSED_NO_GPU',flush=True)
