"""Independent cited-source MiniCheck signals; prepare is CPU/text only.

Only infer/all may initialize CUDA, and must be explicitly scheduled by root.
No detector training, annotation-value lookup, threshold selection or test access.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
import time
import numpy as np
import build_citation_alignment as citation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cited_source_semantic_v1'
OLD = ROOT / 'semantic_baseline/cuda_variant'
LEX = ROOT / 'results/citation_alignment_v1'
NAMES = ['valid_ref_claim_fraction', 'any_source_support_mean',
         'cited_support_gap_mean', 'citation_overlap_x_gap']


def read(p): return json.loads(Path(p).read_text('utf-8'))
def lines(p): return [json.loads(x) for x in Path(p).read_text('utf-8').splitlines() if x]
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')
def jsonl(p, rows):
    Path(p).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), 'utf-8')


def protocol():
    return {'version': 'qa-cited-source-semantic-v1', 'answers': 793, 'fit': 634, 'calibration': 159,
        'claims': 8852, 'recognized_reference_claims': 2084, 'source_claim_pairs': 6252,
        'source': 'Split only original line-start passage1/2/3 headers. Keep exact header/body and whitespace; no reordering or chunking.',
        'claim': 'Reuse original automatic claims verbatim, including reference text. Run each recognized-reference claim against all3 sources, including claims whose recognized IDs are all invalid. No syntax or label filtering beyond the pre-existing parser.',
        'model': 'Pinned existing MiniCheck RoBERTa-large, FP32, eager, CUDA:0, TF32 off, no autocast, no training, eval mode, local_files_only.',
        'input': 'Exact source header/body + existing tokenizer eos token + exact claim; single-string tokenizer, add specials, no truncation; assert length<=512 before GPU.',
        'batch': '4, per-answer claim-major then source1/2/3, final batch unpadded rows preserved; ordinary dynamic padding within each batch.',
        'support': 'Official softmax(logits)[1]; retain both logits and all three support probabilities, never reinterpret old mixed-source document_chunk scores.',
        'columns': NAMES,
        'valid_ref_claim_fraction': 'Mean over window lexical tokens of whether assigned claim has >=1 recognized citation ID in {1,2,3}.',
        'any_source_support_mean': 'Mean max(support1,support2,support3) over original window lexical tokens, with claim value0 if no valid citation.',
        'cited_support_gap_mean': 'Mean [max(all3)-max(valid-cited sources)] over original window lexical tokens, claim value0 if no valid citation.',
        'citation_overlap_x_gap': 'Original citation_overlap over raw4BPE window multiplied by the preceding window mean gap.',
        'geometry': 'Original full-string Llama BPE offsets, raw4 stride1, punctuation counts length; discard only windows with no alphanumeric character. Token->claim is pre-existing maximum alphanumeric overlap, ties earlier. Compare all210364 window IDs and overlap to frozen lexical producer.',
        'invalid_unknown': 'Never automatic risk labels; all original answers/windows retained. No valid citation gives all4 values0.',
        'limits': 'Additional semantic checker, not native generation probe. Multiple cited sources use maximum individual support, not proof of joint entailment. Reference wording remains in hypothesis. Existing fit/cal development only; no test or gold-driven construction.',
        'oracle': 'Replay first original plan using same original mixed chunks solely to confirm loaded checkpoint/arithmetic. Max abs logit2e-4/probability2e-5, unchanged existing limits; not new source-score equivalence.'}


def source_files():
    p = [Path(__file__), ROOT/'src/build_citation_alignment.py',
         ROOT/'semantic_baseline/run_semantic.py', ROOT/'semantic_baseline/run_semantic_cuda.py',
         OLD/'plans.jsonl', OLD/'download_manifest.json', OLD/'scores/16023.json',
         ROOT/'data/feature_preparation/plans.jsonl', LEX/'features.npz', LEX/'claims.jsonl', LEX/'geometry.json', LEX/'complete.json']
    p += [ROOT/'data'/f'{part}.jsonl' for part in ('fit', 'calibration')]
    return p


def split_full_sources(text):
    matches = list(citation.SOURCE.finditer(text))
    assert [int(m.group(1)) for m in matches] == [1, 2, 3]
    assert not text[:matches[0].start()].strip()
    return [{'source_id': int(m.group(1)), 'start': m.start(),
             'end': matches[i+1].start() if i+1<len(matches) else len(text),
             'text': text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]}
            for i,m in enumerate(matches)]


def claim_values(valid_ids, support):
    if not valid_ids: return np.zeros(3, np.float64)
    s = np.asarray(support, np.float64)
    assert s.shape == (3,) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    return np.asarray([1., s.max(), s.max()-max(s[i-1] for i in valid_ids)])


def selfcheck():
    text = 'passage 1: A.\npassage 2: B.\npassage 3: C.'
    s = split_full_sources(text); assert ''.join(x['text'] for x in s) == text
    assert np.array_equal(claim_values([], None), [0,0,0])
    assert np.allclose(claim_values([2], [.9,.2,.5]), [1,.9,.7])
    assert np.allclose(claim_values([1,2], [.9,.2,.5]), [1,.9,0])
    p = citation.parse_citations('It is B [2].')
    assert p['references'][0]['ids'] == [2]
    assert not ({i for r in citation.parse_citations('See passage 10.')['references'] for i in r['ids']} & {1,2,3})
    assert citation.parse_citations('In 2024, three people attended.')['status']=='none'
    return {'passed': True, 'source_text_unchanged': True, 'valid_invalid_multi_citation_checked': True,
            'no_model_loaded': True, 'GPU_used': False}


def prepare():
    assert not (OUT/'inference_started.json').exists()
    assert not (OUT/'preparation_freeze.json').exists(), 'Prepared already; use check instead of overwriting'
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OLD/'model', local_files_only=True, use_fast=True)
    plans = lines(OLD/'plans.jsonl')
    # Project input rows immediately; annotation fields in released development exports are never accessed.
    allowed = ('response_id','partition','group_id','original_response','retrieved_passages','quality','answer_sha256')
    rows = {r['response_id']: {k:r[k] for k in allowed}
            for part in ('fit','calibration') for r in lines(ROOT/'data'/f'{part}.jsonl') if r['quality']=='good'}
    native = {p['response_id']:p for p in lines(ROOT/'data/feature_preparation/plans.jsonl')}
    old_claims = {(r['response_id'],r['claim_index']):r for r in lines(LEX/'claims.jsonl')}
    assert len(rows)==len(plans)==len(native)==793 and len(old_claims)==8852
    old_complete=read(LEX/'complete.json')
    for name in ('features.npz','claims.jsonl','geometry.json'): assert sha(LEX/name)==old_complete['files_sha256'][name]
    batches=[]; claim_records=[]; window_claims=[]; overlap=[]; window_ids=[]; answer_ids=[]; answer_order=[]
    counts=Counter(); cursor=0; bounds={}; partition_start=0; previous=None
    for p in plans:
        rid=p['response_id'];r=rows[rid];n=native[rid];text=r['original_response'];part=r['partition']
        assert part in ('fit','calibration') and p['partition']==n['partition']==part
        assert n['original_response']==text and p['answer_sha256']==r['answer_sha256']==hashlib.sha256(text.encode()).hexdigest()
        assert p['document_sha256']==hashlib.sha256(r['retrieved_passages'].encode()).hexdigest()
        if previous is not None and part!=previous: bounds[previous]=[partition_start,len(window_ids)];partition_start=len(window_ids)
        previous=part;answer_order.append(rid)
        sources=split_full_sources(r['retrieved_passages']);claims=p['claims'];refs=[];selected=[]
        for ci,c in enumerate(claims):
            assert text[c['start']:c['end']]==c['text']
            parsed=citation.parse_citations(c['text']);assert parsed==old_claims[(rid,ci)]['parser']
            ids=sorted({i for ref in parsed['references'] for i in ref['ids']});valid=[i for i in ids if i in (1,2,3)]
            cr={'response_id':rid,'partition':part,'claim_index':ci,'global_claim_index':cursor+ci,
                'start':c['start'],'end':c['end'],'text':c['text'],'recognized_ids':ids,'valid_ids':valid,'parser':parsed}
            claim_records.append(cr)
            refs.extend({**ref,'start':c['start']+ref['start'],'end':c['start']+ref['end']} for ref in parsed['references'])
            if not parsed['references']: continue
            pairs=[]
            for src in sources:
                pair=src['text']+tok.eos_token+c['text'];length=len(tok.encode(pair,add_special_tokens=True,truncation=False))
                assert length<=512, (rid,ci,src['source_id'],length)
                pairs.append({'source_id':src['source_id'],'input_sha256':hashlib.sha256(pair.encode()).hexdigest(),'token_count':length})
                counts['pairs']+=1;counts['input_tokens']+=length;counts['max_input_tokens']=max(counts['max_input_tokens'],length)
            selected.append({'claim_index':ci,'global_claim_index':cursor+ci,'text':c['text'],'pairs':pairs})
            counts['recognized_claims']+=1;counts[f'{part}_recognized_claims']+=1
        if selected:
            batches.append({'response_id':rid,'partition':part,'answer_sha256':r['answer_sha256'],
                'document_sha256':p['document_sha256'],'original_plan_sha256':digest(p),
                'sources':sources,'claims':selected})
        offsets=n['original']['response_token_offsets']
        lexical=np.asarray([any(text[j].isalnum() for j in range(a,b)) for a,b in offsets],bool)
        groups=citation.token_claim_assignment(text,offsets,lexical,claims)
        for start in range(max(1,len(offsets)-3)):
            ix=list(range(start,min(start+4,len(offsets))));good=[i for i in ix if lexical[i]]
            if not good: continue
            window_ids.append(f'{rid}__k4_{start:05d}');answer_ids.append(rid)
            window_claims.append([int(groups[i])+cursor for i in good]+[-1]*(4-len(good)))
            overlap.append(citation.citation_overlap(text,offsets,ix,refs))
        cursor+=len(claims);counts[part]+=1
    bounds[previous]=[partition_start,len(window_ids)]
    assert counts['pairs']==6252 and counts['recognized_claims']==2084 and cursor==8852
    assert counts['fit']==634 and counts['calibration']==159 and len(window_ids)==210364
    assert bounds=={'fit':[0,168123],'calibration':[168123,210364]}
    with np.load(LEX/'features.npz',allow_pickle=False) as z:
        assert window_ids==z['window_ids'].tolist() and answer_ids==z['response_ids'].tolist()
        assert np.array_equal(np.asarray(overlap,np.float32),z['features'][:,5])
    np.savez_compressed(OUT/'window_geometry.npz', window_claim_indices=np.asarray(window_claims,np.int32),
        citation_overlap=np.asarray(overlap,np.float64),window_ids=np.asarray(window_ids),response_ids=np.asarray(answer_ids))
    jsonl(OUT/'pair_plans.jsonl',batches);jsonl(OUT/'claims.jsonl',claim_records)
    save(OUT/'feature_names.json',NAMES);save(OUT/'protocol.json',protocol());save(OUT/'CPU_SELFCHECK.json',selfcheck())
    save(OUT/'geometry.json',{'answers':793,'windows':210364,'claims':8852,'bounds':bounds,
        'answer_ids':answer_order,'answer_order_sha256':digest(answer_order),'window_order_sha256':digest(window_ids),
        'geometry_matches_original':True,'citation_overlap_exact':True,'annotation_fields_used':False})
    save(OUT/'preparation_statistics.json',{**dict(counts),'scored_answers':len(batches),'source_chunking_required':False,
        'official_test_opened':False,'GPU_used':False,'model_loaded':False,'trained':False})
    names=['pair_plans.jsonl','claims.jsonl','window_geometry.npz','feature_names.json','protocol.json','CPU_SELFCHECK.json','geometry.json','preparation_statistics.json']
    save(OUT/'preparation_freeze.json',{'status':'prepared_waiting_GPU_authorization','files_sha256':{x:sha(OUT/x) for x in names},
        'source_sha256':{str(x.resolve()):sha(x) for x in source_files()},'no_test':True,'trained':False})
    print('CITED_SOURCE_PREPARED',dict(counts),'GPU_NOT_USED',flush=True)


def check():
    f=read(OUT/'preparation_freeze.json');assert read(OUT/'protocol.json')==protocol()
    for name,expected in f['files_sha256'].items():assert sha(OUT/name)==expected,name
    for name,expected in f['source_sha256'].items():assert sha(name)==expected,name
    assert f['no_test'] and f['trained'] is False
    return f


def infer():
    check();assert not (OUT/'inference_started.json').exists(),'Do not overwrite a started run'
    # Verify pinned assets before loading the model or initializing CUDA.
    for name,expected in read(OLD/'download_manifest.json')['files_sha256'].items(): assert sha(OLD/name)==expected,name
    plans=lines(OUT/'pair_plans.jsonl');assert sum(len(p['claims'])*3 for p in plans)==6252
    save(OUT/'inference_started.json',{'time':time.time(),'pid':os.getpid(),'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'device':'cuda:0'})
    start=time.perf_counter()
    spec=importlib.util.spec_from_file_location('cited_source_original_cuda',ROOT/'semantic_baseline/run_semantic_cuda.py')
    gpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gpu)
    model,tok=gpu.load_model()
    anchor=lines(OLD/'plans.jsonl')[0];texts,_=gpu.pairs(anchor,tok)
    z,p,_=gpu.probabilities(model,tok,texts)
    original=read(OLD/'scores'/f'{anchor["response_id"]}.json')
    dz=float(np.max(np.abs(z-np.asarray(original['logits']).reshape(-1,2))))
    dp=float(np.max(np.abs(p-np.asarray(original['support_by_claim_document']).ravel())))
    save(OUT/'numeric_agreement.json',{'response_id':anchor['response_id'],'logit_max_abs_diff':dz,'support_max_abs_diff':dp,
         'passed':dz<=2e-4 and dp<=2e-5,'reference_score_sha256':sha(OLD/'scores'/f'{anchor["response_id"]}.json')})
    assert dz<=2e-4 and dp<=2e-5,'Original checkpoint/arithmetic gate failed; no threshold relaxation'
    records=[];directory=OUT/'scores';directory.mkdir(exist_ok=True)
    for i,plan in enumerate(plans):
        tick=time.perf_counter();texts=[];identities=[]
        for c in plan['claims']:
            for source,pair in zip(plan['sources'],c['pairs']):
                text=source['text']+tok.eos_token+c['text']
                assert hashlib.sha256(text.encode()).hexdigest()==pair['input_sha256']
                assert len(tok.encode(text,add_special_tokens=True,truncation=False))==pair['token_count']<=512
                texts.append(text);identities.append({'claim_index':c['claim_index'],'global_claim_index':c['global_claim_index'],**pair})
        logits,support,_=gpu.probabilities(model,tok,texts)
        assert logits.shape==(len(texts),2) and support.shape==(len(texts),) and np.isfinite(logits).all()
        assert ((support>=0)&(support<=1)).all()
        result={'response_id':plan['response_id'],'partition':plan['partition'],'pair_plan_sha256':digest(plan),
            'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'pairs':identities,'logits':logits.tolist(),
            'support_by_claim_source':support.reshape(-1,3).tolist(),'claim_indices':[c['claim_index'] for c in plan['claims']],
            'source_ids':[1,2,3],'seconds':time.perf_counter()-tick,'device':'cuda:0','dtype':'float32','trained':False}
        path=directory/f'{plan["response_id"]}.json';save(path,result)
        records.append({'response_id':plan['response_id'],'path':str(path.relative_to(OUT)),'sha256':sha(path),'pairs':len(texts)})
        if (i+1)%50==0:print('CITED_SOURCE_INFER',i+1,len(plans),'seconds',round(time.perf_counter()-start,1),flush=True)
    check()
    save(OUT/'inference_complete.json',{'status':'complete','answers':len(plans),'pairs':sum(x['pairs'] for x in records),
        'records':records,'seconds':time.perf_counter()-start,'numeric_agreement_sha256':sha(OUT/'numeric_agreement.json'),
        'preparation_freeze_sha256':sha(OUT/'preparation_freeze.json'),'no_test':True,'trained':False})
    del model
    import torch
    torch.cuda.empty_cache()
    print('CITED_SOURCE_INFERENCE_COMPLETE',flush=True)


def export():
    check();assert not (OUT/'features_complete.json').exists()
    done=read(OUT/'inference_complete.json');assert done['status']=='complete' and done['pairs']==6252
    assert done['preparation_freeze_sha256']==sha(OUT/'preparation_freeze.json')
    plans={p['response_id']:p for p in lines(OUT/'pair_plans.jsonl')};scores={}
    for rec in done['records']:
        path=OUT/rec['path'];assert sha(path)==rec['sha256'];r=read(path);plan=plans[r['response_id']]
        assert r['pair_plan_sha256']==digest(plan) and r['source_ids']==[1,2,3]
        assert r['claim_indices']==[c['claim_index'] for c in plan['claims']]
        for ci,s in zip(r['claim_indices'],r['support_by_claim_source']):scores[(r['response_id'],ci)]=s
    assert len(scores)==2084
    claims=lines(OUT/'claims.jsonl');values=[]
    for c in claims:
        s=scores.get((c['response_id'],c['claim_index']))
        assert (s is not None)==bool(c['parser']['references'])
        values.append(claim_values(c['valid_ids'],s))
    v=np.asarray(values,np.float64)
    with np.load(OUT/'window_geometry.npz',allow_pickle=False) as z:
        ix=z['window_claim_indices'];mask=ix>=0;assert mask.any(1).all()
        base=(v[np.maximum(ix,0)]*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        x=np.column_stack((base,base[:,2]*z['citation_overlap'])).astype(np.float32)
        window_ids=z['window_ids'].copy();response_ids=z['response_ids'].copy()
    assert x.shape==(210364,4) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
    np.save(OUT/'window_features.npy',x)
    np.savez_compressed(OUT/'features.npz',features=x,window_ids=window_ids,response_ids=response_ids)
    save(OUT/'export_statistics.json',{'nonzero_per_column':(x!=0).sum(0).tolist(),'window_count':len(x),
        'all_original_windows_retained':True,'no_annotation_fields_used':True,'thresholds_selected':False})
    names=['window_features.npy','features.npz','feature_names.json','geometry.json','export_statistics.json',
           'protocol.json','preparation_freeze.json','inference_complete.json','numeric_agreement.json']
    save(OUT/'features_complete.json',{'status':'complete','rows':210364,'columns':4,'dtype':'float32',
        'window_order_sha256':digest(window_ids.tolist()),'files_sha256':{n:sha(OUT/n) for n in names},'no_test':True,'trained':False})
    print('CITED_SOURCE_FEATURES_COMPLETE',x.shape,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','check','infer','export','all']);args=parser.parse_args()
    if args.stage=='all': infer();export()
    else: globals()[args.stage]()
