"""CPU-only PsiloQA auxiliary token/mapping preparation; no training command."""
from pathlib import Path
from collections import Counter
import argparse
import importlib.metadata
import json
import time
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer
import feature_qa as replay
import build_gold as gold
import tail_finetune as mapping
import fava_nfc_character_map as nfc_mapping
import prepare_psiloqa_candidates as candidates

ROOT=replay.ROOT
DATA=ROOT/'auxiliary_psiloqa_v1'
OUT=DATA/'tokenization_v1'
MODEL=ROOT.parent/'models/ModernBERT-base'
LIMIT=8192


def read(p):return replay.read(p)
def sha(p):return replay.sha(p)
def save(n,x):replay.save(OUT/n,x)
def rows(p):return candidates.rows(p)
def savel(n,data):candidates_path=OUT/n; candidates_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')


def protocol():return {
    'version':'PsiloQA-official-train-auxiliary-tokenization-v1',
    'cohort':'Exactly15847 staged candidates; retain635 unmarked answers and4 null complexity. No metadata-based filtering, refit, official validation/test read or new download.',
    'encoder_input':'Exact wiki_passage + ModernBERT sep_token + exact original question + sep_token + exact original llm_answer, add_special_tokens=True, truncation=False, no padding. No golden_answer, annotated markup or generator identity is read for input construction.',
    'raw_BPE':'Same local Llama tokenizer and encode_view full-string wrapper as original QA. Auxiliary prompt=exact wiki_passage + two newline characters + original question; reference_range=(0,len(wiki_passage)). Preserve clipped and raw answer boundary offsets, punctuation, byte fallback duplicates. This is only an auxiliary shared-coordinate tokenization, NOT the original generator trace or a Llama forward.',
    'labels':'Unchanged automatic spans through original gold.map_characters and independent interval oracle; lexical=any isalnum character, risk=any gold span intersecting an isalnum character. Answer label=bool(original labels); punctuation-only positive may have no lexical risk, recorded not relabeled. Unmarked positions are silver negatives, not verified truth.',
    'mapping':'Original nonwhitespace-character-average encoder-to-raw mapping; reuse frozen strict NFC helper for proven single-character compositions only, with full proof ledger. No text/offset/label repair and no old module global mutations.',
    'windows':'Original4rawBPE stride1; punctuation consumes slots, short answer gives one short window, all-punctuation windows excluded only from lexical window count.',
    'failure_policy':'Attempt every fixed candidate and retain full exception ledger; no silent truncation, filtering or fallback. Record over8192 and missing lexical coverage. Failed mapping cannot fabricate a successful row; expected/attempted/mapped identities audited. Any exception blocks training-ready status.',
    'output':'IndexedAuxiliary-compatible token_inputs.jsonl plus byte offsets and per-answer index. automatic_not_human_gold and unmarked_positions_silver_not_verified_negative are true. Separate original labels and provenance stay unchanged.',
    'operations':'CPU tokenizers only, local_files_only; no model weights, CUDA, training, data changes, global status edits or GPU queue.'}


def versions():return {k:importlib.metadata.version(k) for k in ['transformers','tokenizers','numpy','torch']}


def source_files():
    paths=[Path(__file__),Path(replay.__file__),Path(gold.__file__),Path(mapping.__file__),Path(nfc_mapping.__file__),Path(candidates.__file__),
        ROOT/'src/run_auxiliary_transfer.py',DATA/'manifest.json',DATA/'complete.json',DATA/'candidate_fit.jsonl',DATA/'model_inputs.jsonl',DATA/'design_freeze.json']
    for d in [MODEL,replay.MODEL]:
        for name in ['tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','config.json']:
            if (d/name).exists():paths.append(d/name)
    return {str(p.resolve()):sha(p) for p in paths}


def prepare():
    assert not torch.cuda.is_initialized();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'token_design_freeze.json').exists()
    complete=read(DATA/'complete.json')
    for n,h in complete['files_sha256'].items():assert sha(DATA/n)==h
    m=read(DATA/'manifest.json');assert m['candidate_answers']==15847 and m['trained'] is False
    save('token_protocol.json',protocol())
    save('token_design_freeze.json',{'source_sha256':source_files(),'protocol_sha256':sha(OUT/'token_protocol.json'),
        'package_versions':versions(),'trained':False,'GPU_used':False})
    print('PSILOQA_TOKEN_PROTOCOL_FROZEN_CPU_ONLY',flush=True)


def check_frozen():
    f=read(OUT/'token_design_freeze.json')
    assert f['source_sha256']==source_files() and f['package_versions']==versions()
    assert f['protocol_sha256']==sha(OUT/'token_protocol.json') and read(OUT/'token_protocol.json')==protocol()
    for n,h in read(DATA/'complete.json')['files_sha256'].items():assert sha(DATA/n)==h
    return f


def run():
    assert not torch.cuda.is_initialized();check_frozen()
    assert not (OUT/'started.json').exists()
    save('started.json',{'time':time.time(),'freeze_sha256':sha(OUT/'token_design_freeze.json')})
    llama=AutoTokenizer.from_pretrained(replay.MODEL,local_files_only=True)
    bert=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert llama.is_fast and bert.is_fast and bert.model_max_length==LIMIT
    counter=Counter();lengths=[];exceptions=[];repairs=[];attempts=[];index=[];byte_offsets=[]
    inputs=iter(rows(DATA/'model_inputs.jsonl'));started=time.perf_counter()
    with (OUT/'token_inputs.jsonl').open('wb') as stream:
        for i,row in enumerate(rows(DATA/'candidate_fit.jsonl')):
            inp=next(inputs);rid=row['response_id'];assert all(row[k]==inp[k] for k in inp)
            assert set(inp)=={'response_id','retrieved_passages','question','original_response'}
            assert row['official_split']=='train' and row['partition']=='auxiliary_candidate_fit'
            text=inp['original_response'];counter['attempted_answers']+=1
            attempt={'response_id':rid,'candidate_index':i,'mapped':False,'exception_stages':[]}
            stage='raw_tokenization';length=None
            try:
                auxiliary_prompt=inp['retrieved_passages']+'\n\n'+inp['question']
                view=replay.encode_view(llama,auxiliary_prompt,text,(0,len(inp['retrieved_passages'])))
                offsets=view['response_token_offsets'];n=len(offsets)
                stage='label_mapping';_,_,lexical,risk=gold.map_characters(text,row['labels'],offsets)
                gold.independent_label_check(text,row['labels'],offsets,lexical,risk)
                stage='encoder_tokenization'
                prefix=inp['retrieved_passages']+bert.sep_token+inp['question']+bert.sep_token
                whole=prefix+text
                encoded=bert(whole,add_special_tokens=True,truncation=False,padding=False,return_offsets_mapping=True)
                length=len(encoded['input_ids']);lengths.append(length)
                counter['encoder_input_tokens_attempted']+=length
                eo=np.asarray(encoded['offset_mapping'],np.int64)
                inside=(eo[:,1]>len(prefix))&(eo[:,0]<len(whole))&(eo[:,1]>eo[:,0])
                begin=np.where(inside,np.maximum(0,eo[:,0]-len(prefix)),-1)
                end=np.where(inside,np.minimum(len(text),eo[:,1]-len(prefix)),-1)
                stage='NFC_character_mapping'
                charmap,proof=nfc_mapping.character_map(text,offsets,begin,end,normalizer=bert.backend_tokenizer.normalizer,return_diagnostics=True)
                if proof['repaired_character_count']:
                    repairs.append({'response_id':rid,**proof});counter.update(nfc_repaired_answers=1,nfc_repaired_characters=proof['repaired_character_count'])
                stage='mapping_mass_check'
                sums=np.bincount(charmap[0],weights=charmap[2],minlength=n)
                nons=np.asarray([any(not c.isspace() for c in text[a:b]) for a,b in offsets])
                assert np.max(np.abs(sums[nons]-1))<2e-7 and not sums[~nons].any()
                assert len(charmap[0])==len(charmap[1])==len(charmap[2])
                assert np.all(charmap[1]>=0) and np.all(charmap[1]<length) and np.isfinite(charmap[2]).all()
                issues=[]
                if length>LIMIT:issues.append('encoder_over8192_no_truncation')
                if not any(lexical):issues.append('no_lexical_answer_tokens')
                for problem in issues:
                    exceptions.append({'response_id':rid,'stage':problem,'input_tokens':length,'kept_full_record':True})
                attempt['exception_stages']=issues
                if row['risk'] and not any(risk):counter['positive_answer_no_lexical_risk']+=1
                assert row['risk']==int(bool(row['labels']))
                eligible=[(a,b) for a,b in gold.windows_for_count(n) if any(lexical[a:b])]
                out={'response_id':rid,'source_id':row['source_id'],'group_id':row['group_id'],'task_type':'QA_automatic',
                    'answer_sha256':replay.digest(text),'input_ids':encoded['input_ids'],
                    'answer_encoder_start':begin.tolist(),'answer_encoder_end':end.tolist(),
                    'mapping':[x.tolist() for x in charmap],'response_token_ids':view['answer_token_ids'],
                    'response_token_offsets':offsets,'response_token_offsets_raw':view['response_token_offsets_raw'],
                    'lexical_mask':lexical,'risk_mask':risk,'answer_risk':row['risk'],
                    'automatic_not_human_gold':True,'unmarked_positions_silver_not_verified_negative':True,
                    'historical_native_generation_trace':False,'encoder_input_sha256':replay.digest(whole),
                    'encoder_ids_sha256':replay.digest(encoded['input_ids']),
                    'raw_auxiliary_rendered_text_sha256':view['rendered_text_sha256'],
                    'input_field_character_lengths':[len(inp[k]) for k in ['retrieved_passages','question','original_response']],
                    'encoder_answer_character_range':[len(prefix),len(whole)]}
                byte_offsets.append(stream.tell());stream.write((json.dumps(out,ensure_ascii=False)+'\n').encode('utf-8'))
                index.append({'response_id':rid,'source_id':row['source_id'],'group_id':row['group_id'],
                    'input_token_count':length,'raw_token_count':n,'answer_risk':row['risk'],'lexical_token_count':sum(lexical),
                    'risk_token_count':sum(risk),'candidate_index':i})
                counter.update(mapped_answers=1,encoder_input_tokens=length,raw_answer_tokens=n,
                    lexical_answer_tokens=sum(lexical),risk_answer_tokens=sum(risk),windows=len(eligible),
                    risk_windows=sum(any(risk[a:b]) for a,b in eligible),unmarked_answers=int(not row['risk']),risk_answers=row['risk'])
                attempt['mapped']=True
            except (ValueError,AssertionError,IndexError,TypeError) as exc:
                exceptions.append({'response_id':rid,'stage':stage,'error_type':type(exc).__name__,'error':str(exc),
                    'input_tokens':length,'original_text_labels_preserved':True,'kept_full_record':False})
                attempt['exception_stages'].append(stage)
            attempts.append(attempt)
            if (i+1)%1000==0:print('PSILOQA_TOKEN_INPUTS',i+1,'mapped',counter['mapped_answers'],'exceptions',len(exceptions),round(time.perf_counter()-started,1),flush=True)
    assert next(inputs,None) is None and counter['attempted_answers']==15847
    assert not torch.cuda.is_initialized()
    np.save(OUT/'token_byte_offsets.npy',np.asarray(byte_offsets,np.int64))
    savel('answer_index.jsonl',index);savel('attempts.jsonl',attempts);savel('exceptions.jsonl',exceptions);savel('nfc_repairs.jsonl',repairs)
    m=read(DATA/'manifest.json')
    report={'status':'prepared_not_trained' if not exceptions and counter['mapped_answers']==15847 else 'review_required_no_silent_filtering',
        'expected_answers':15847,'answers':counter['mapped_answers'],'stats':dict(counter),
        'max_input_tokens':max(lengths) if lengths else None,'median_input_tokens':float(np.median(lengths)) if lengths else None,
        'p95_input_tokens':float(np.percentile(lengths,95)) if lengths else None,
        'exceptions':exceptions,'no_truncation':True,'trained':False,'GPU_used':False,
        'automatic_not_human_gold':True,'unmarked_positions_silver_not_verified_negative':True,
        'official_validation_test_read':False,'source_manifest_sha256':sha(DATA/'manifest.json'),
        'token_design_sha256':sha(OUT/'token_design_freeze.json'),'token_inputs_sha256':sha(OUT/'token_inputs.jsonl'),
        'tokenizer_signature':replay.tokenizer_signature(llama),'modernbert_tokenizer_sha256':sha(MODEL/'tokenizer.json'),
        'normalizer_state':json.loads(bert.backend_tokenizer.normalizer.__getstate__()),
        'wall_seconds':time.perf_counter()-started,'existing_data_modified':False,
        'input_construction':'Allowlist exact three fields from model_inputs.jsonl; provenance never read by tokenizer run.'}
    save('token_input_preparation.json',report)
    verify()
    print(json.dumps({k:report[k] for k in ['status','answers','stats','max_input_tokens','p95_input_tokens','exceptions','wall_seconds']},ensure_ascii=False),flush=True)


def verify():
    check_frozen();report=read(OUT/'token_input_preparation.json')
    assert sha(OUT/'token_inputs.jsonl')==report['token_inputs_sha256']
    index=list(rows(OUT/'answer_index.jsonl'));attempts=list(rows(OUT/'attempts.jsonl'))
    assert len(attempts)==len({r['response_id'] for r in attempts})==15847
    assert sum(r['mapped'] for r in attempts)==len(index)==report['answers']
    offsets=np.load(OUT/'token_byte_offsets.npy');assert len(offsets)==len(index)
    # Real compatibility check; no model construction, training or global changes.
    from run_auxiliary_transfer import IndexedAuxiliary
    ds=IndexedAuxiliary(OUT/'token_inputs.jsonl',offsets)
    for i in sorted({0,len(index)//2,len(index)-1}):
        if i<0:continue
        r=ds[i];assert r['response_id']==index[i]['response_id'] and r['group_id']==index[i]['group_id']
        assert r['raw_token_count']==index[i]['raw_token_count'] and len(r['input_ids'])==index[i]['input_token_count']
        assert set(r)=={'response_id','group_id','input_ids','raw_token_count','mapping'}
    ds.close();assert not torch.cuda.is_initialized()
    save('TOKEN_CHECK.json',{'status':'passed' if not report['exceptions'] else 'exceptions_preserved_review_required',
        'all_expected_candidates_attempted':True,'no_silent_drops':True,'mapped_answers':len(index),
        'actual_IndexedAuxiliary_adapter_cpu_samples':min(3,len(index)),'trained':False,'GPU_used':False})
    names=['token_protocol.json','token_design_freeze.json','token_inputs.jsonl','token_byte_offsets.npy','answer_index.jsonl',
        'attempts.jsonl','exceptions.jsonl','nfc_repairs.jsonl','token_input_preparation.json','TOKEN_CHECK.json']
    save('complete.json',{'status':report['status'],'files_sha256':{n:sha(OUT/n) for n in names},
        'all_candidates_attempted':True,'ready_for_future_training':not report['exceptions'],'trained':False,'GPU_used':False})
    print('PSILOQA_TOKEN_CHECK',report['status'],len(index),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','check']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'check':verify}[args.stage]()
