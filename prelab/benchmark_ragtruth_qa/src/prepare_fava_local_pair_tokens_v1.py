"""Tokenize all FAVA local repair pairs on CPU; no training/GPU entry point.

Each side is a separate full-context input. Only the original local character
target is mapped; outside it is unknown, never a whole-answer negative label.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import importlib.metadata
import json
import time
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer

import feature_qa as replay
import tail_finetune as legacy_map
import fava_nfc_character_map as nfc

ROOT = replay.ROOT
DATA = ROOT/'auxiliary_fava_local_pairs_v1'
OUT = DATA/'tokenization_v1'
MODEL = ROOT.parent/'models/ModernBERT-base'
LIMIT = 8192
INPUT_KEYS = {'retrieved_passages','question','response'}


def readl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def savel(name, rows):
    with (OUT/name).open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def save(name, value):
    replay.save(OUT/name,value)


def protocol():
    return {'version':'fava-local-pair-tokenization-v1', 'pairs':10040, 'side_inputs':20080,
            'scope':'Only the reviewed official-train FAVA local-pair candidates, their whitelist inputs and inherited isolation metadata. No QA fit/cal/test text, labels, scores or weights consumed.',
            'encoder_input':'Exact retrieved_passages + ModernBERT SEP + exact empty question + SEP + one original/preferred response. Special tokens added, no padding or truncation. Never concatenate both versions or target/provenance fields.',
            'raw_axis':'Existing Llama encode_view over exact retrieved_passages + two newlines + empty question in its full chat wrapper, followed by this response. Preserve original and clipped answer offsets, punctuation and overlapping byte tokens. Auxiliary coordinates only, not native generation or a forward pass.',
            'target':'Keep each side original independent [start,end) Unicode-character range and exact text. target_raw_token_indices/target_encoder_token_indices use nonwhitespace intersection; *_lexical_indices require isalnum characters INSIDE the target. Do not select punctuation merely because its BPE also contains an outside-target letter.',
            'unknown_outside':'No risk_mask, answer_risk, whole-answer truth label, full corrected-answer negative, or outside-target binary supervision is created. original/preferred are noisy author-relative roles, not factual certificates.',
            'mapping':'All original nonwhitespace answer characters map to encoder owners; raw BPE receives equal original-character mass, overlap owners split mass. Reuse strict NFC helper only for proven single-character canonical compositions. Text and target coordinates never changed; repeated text resolved solely by original offsets, not find().',
            'eligibility':'Retain every10040 pair/20080 sides. Eligible pair requires both sides fully mapped, complete encoder input <=8192, local raw and encoder lexical targets, and upstream structural eligibility. Keep four punctuation-target pairs with no lexical supervision. All other mapping/length failures explicitly retained; no silent subset or truncation.',
            'weights_suggestion':'If later using only eligible pairs, recompute equal participating material groups -> original answers -> eligible pairs:1/(G*A_g*P_a). Both sides remain one pair unit. This is an explicit eligible-subset base-weight suggestion, not a loss or training choice.',
            'outputs':'One token_inputs.jsonl record per side including failed sides, byte offsets, full text inputs and complete token/character mapping when successful; input_index.jsonl and paired_index.jsonl for later paired readers. All lengths attempted; failed raw mapping does not suppress encoder tokenization.',
            'tokenizer_limit':LIMIT, 'operations':{'CPU_only':True,'GPU_used':False,'trained':False,'QA_data_modified':False,'new_downloads':False}}


def source_files():
    paths=[Path(__file__),Path(replay.__file__),Path(legacy_map.__file__),Path(nfc.__file__),
           DATA/'complete.json',DATA/'manifest.json',DATA/'candidate_pairs.jsonl',DATA/'model_inputs.jsonl',
           DATA/'material_group_index.jsonl',DATA/'original_answer_index.jsonl',
           DATA/'INDEPENDENT_QUALITY_CHECK.json',DATA/'STRUCTURAL_FLAGS.json',DATA/'protocol.json',
           ROOT/'auxiliary_fava_v2/SOURCE_ISOLATION_REPORT.json']
    for directory in (MODEL,replay.MODEL):
        for name in ('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','config.json'):
            if (directory/name).exists():
                paths.append(directory/name)
    return {str(path.resolve()):replay.sha(path) for path in paths}


def versions():
    return {name:importlib.metadata.version(name) for name in ('transformers','tokenizers','numpy','torch')}


def verify_candidates():
    complete=replay.read(DATA/'complete.json')
    assert complete['pairs']==10040 and complete['input_rows']==20080 and complete['all_pairs_retained']
    assert complete['original_answers']==5275 and complete['material_groups']==5238
    assert not complete['GPU_used'] and not complete['trained'] and not complete['QA_gold_cal_test_read']
    for name,digest in complete['files_sha256'].items():
        assert replay.sha(DATA/name)==digest,name
    manifest=replay.read(DATA/'manifest.json')
    assert replay.sha(DATA/'manifest.json')==complete['manifest_sha256']
    for name,digest in manifest['artifacts_sha256'].items():
        assert replay.sha(DATA/name)==digest,name
    assert replay.read(DATA/'INDEPENDENT_QUALITY_CHECK.json')['passed']


def prepare():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'token_design_freeze.json').exists()
    verify_candidates()
    save('token_protocol.json',protocol())
    save('token_design_freeze.json',{'source_sha256':source_files(),'package_versions':versions(),
                                    'protocol_sha256':replay.sha(OUT/'token_protocol.json'),
                                    'GPU_used':False,'trained':False})
    assert not torch.cuda.is_initialized()
    print('FAVA_LOCAL_PAIR_TOKEN_DESIGN_FROZEN_CPU_ONLY',flush=True)


def check_frozen():
    assert not torch.cuda.is_initialized()
    frozen=replay.read(OUT/'token_design_freeze.json')
    assert frozen['source_sha256']==source_files()
    assert frozen['package_versions']==versions()
    assert frozen['protocol_sha256']==replay.sha(OUT/'token_protocol.json')
    assert replay.read(OUT/'token_protocol.json')==protocol()
    verify_candidates()


def local_indices(text,offsets,target):
    left,right=target['start'],target['end']
    assert 0<=left<right<=len(text) and text[left:right]==target['text']
    nonspace=[];lexical=[]
    for i,(a,b) in enumerate(offsets):
        lo,hi=max(left,int(a)),min(right,int(b))
        if lo<hi and any(not c.isspace() for c in text[lo:hi]):
            nonspace.append(i)
        if lo<hi and any(c.isalnum() for c in text[lo:hi]):
            lexical.append(i)
    return nonspace,lexical


def encoder_local_indices(text,begin,end,target,proof):
    owners=[[] for _ in text]
    for i,(a,b) in enumerate(zip(begin,end)):
        if a<0:
            assert b==-1
            continue
        for c in range(int(a),int(b)):
            if not text[c].isspace():
                owners[c].append(i)
    for repair in proof['repairs']:
        at=repair['missing_character_index']
        assert not owners[at]
        owners[at]=repair['encoder_token_indices']
    assert all(owners[i] for i,c in enumerate(text) if not c.isspace())
    chars=range(target['start'],target['end'])
    covered=sorted({j for c in chars if not text[c].isspace() for j in owners[c]})
    lexical=sorted({j for c in chars if text[c].isalnum() for j in owners[c]})
    return covered,lexical


def convert(mi,target,llama,bert):
    assert set(mi)==INPUT_KEYS and mi['question']==''
    text=mi['response']
    assert text[target['start']:target['end']]==target['text']
    out={'model_inputs':mi,'target':target,'outside_target':'unknown_no_binary_supervision',
         'response_sha256':replay.digest(text),'model_inputs_sha256':replay.digest(mi)}
    errors=[];proof=None;view=None;encoded=None
    # Independently attempt encoder and raw tokenization so one failure never
    # hides the other side's full input length or original text.
    try:
        prefix=mi['retrieved_passages']+bert.sep_token+mi['question']+bert.sep_token
        whole=prefix+text
        encoded=bert(whole,add_special_tokens=True,truncation=False,padding=False,
                     return_offsets_mapping=True,return_attention_mask=True)
        eo=np.asarray(encoded['offset_mapping'],np.int64)
        inside=(eo[:,1]>len(prefix))&(eo[:,0]<len(whole))&(eo[:,1]>eo[:,0])
        begin=np.where(inside,np.maximum(0,eo[:,0]-len(prefix)),-1)
        end=np.where(inside,np.minimum(len(text),eo[:,1]-len(prefix)),-1)
        assert encoded['attention_mask']==[1]*len(encoded['input_ids'])
        out.update(input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],
                   encoder_offsets=eo.tolist(),answer_encoder_start=begin.tolist(),answer_encoder_end=end.tolist(),
                   answer_encoder_positions=np.flatnonzero(inside).tolist(),encoder_text_sha256=replay.digest(whole),
                   encoder_input_ids_sha256=replay.digest(encoded['input_ids']),input_tokens=len(encoded['input_ids']))
        if len(encoded['input_ids'])>LIMIT:
            errors.append({'stage':'encoder_over8192_no_truncation','length':len(encoded['input_ids'])})
    except Exception as exc:
        errors.append({'stage':'encoder_tokenization','error':repr(exc)})
    try:
        prompt=mi['retrieved_passages']+'\n\n'+mi['question']
        view=replay.encode_view(llama,prompt,text,reference_range=(0,len(mi['retrieved_passages'])))
        offsets=view['response_token_offsets']
        lexical=[int(any(c.isalnum() for c in text[a:b])) for a,b in offsets]
        raw_target,raw_lexical=local_indices(text,offsets,target)
        out.update(raw_full_input_ids=view['input_ids'],raw_full_input_ids_sha256=view['input_ids_sha256'],
                   raw_rendered_text_sha256=view['rendered_text_sha256'],answer_token_positions=view['answer_token_positions'],
                   response_token_ids=view['answer_token_ids'],response_token_offsets=offsets,
                   response_token_offsets_raw=view['response_token_offsets_raw'],lexical_mask=lexical,
                   target_raw_token_indices=raw_target,target_raw_lexical_indices=raw_lexical,
                   raw_answer_tokens=len(offsets),raw_full_input_tokens=len(view['input_ids']))
    except Exception as exc:
        errors.append({'stage':'raw_tokenization_or_local_target','error':repr(exc)})
    if encoded is not None and view is not None:
        try:
            charmap,proof=nfc.character_map(text,view['response_token_offsets'],begin,end,
                                           normalizer=bert.backend_tokenizer.normalizer,return_diagnostics=True)
            n=len(view['response_token_offsets'])
            total=np.bincount(charmap[0],weights=charmap[2],minlength=n)
            nonspace=np.asarray([any(not c.isspace() for c in text[a:b]) for a,b in view['response_token_offsets']],bool)
            assert len(charmap[0])==len(charmap[1])==len(charmap[2])
            assert np.all(charmap[0]>=0) and np.all(charmap[0]<n)
            assert np.all(charmap[1]>=0) and np.all(charmap[1]<len(encoded['input_ids']))
            assert np.isfinite(charmap[2]).all() and np.all(charmap[2]>0)
            assert np.max(np.abs(total[nonspace]-1),initial=0)<2e-7 and not total[~nonspace].any()
            enc_target,enc_lexical=encoder_local_indices(text,begin,end,target,proof)
            # Direct interval oracle: every target alnum character must have
            # an intersecting raw and encoder target; repeats are not searched.
            chars=[c for c in range(target['start'],target['end']) if text[c].isalnum()]
            for c in chars:
                assert any(view['response_token_offsets'][j][0]<=c<view['response_token_offsets'][j][1] for j in out['target_raw_lexical_indices'])
                assert any(begin[j]<=c<end[j] for j in enc_lexical)
            out.update(mapping=[x.tolist() for x in charmap],target_encoder_token_indices=enc_target,
                       target_encoder_lexical_indices=enc_lexical,
                       encoder_response_lexical_mask=[int(a>=0 and any(c.isalnum() for c in text[int(a):int(b)])) for a,b in zip(begin,end)],
                       mapping_row_mass_max_error=float(np.max(np.abs(total[nonspace]-1),initial=0)),
                       nfc_proof=proof)
        except Exception as exc:
            errors.append({'stage':'character_mapping_or_local_coverage','error':repr(exc)})
    mapped='mapping' in out
    has_local=mapped and bool(out['target_raw_lexical_indices']) and bool(out['target_encoder_lexical_indices'])
    out.update(mapping_complete=mapped,local_lexical_supervision_available=has_local,
               status='mapped' if mapped else 'mapping_failure',exceptions=errors,
               synthetic_not_human_gold=True,native_generation_trace=False,
               full_answer_labels_created=False)
    assert 'risk_mask' not in out and 'answer_risk' not in out
    return out


def cpu_selfcheck(llama,bert):
    # Repeated names: the provided second occurrence wins by coordinates.
    text='Anna met Anna. Cafe\u0301!'
    target={'start':9,'end':13,'text':'Anna','meaning':'synthetic_test_only'}
    out=convert({'retrieved_passages':'Reference1: Anna met another person.','question':'','response':text},target,llama,bert)
    assert out['mapping_complete'] and not out['exceptions']
    assert all(out['response_token_offsets'][j][1]>9 for j in out['target_raw_lexical_indices'])
    # A punctuation target shares a mixed token in this synthetic geometry;
    # its outside-target letters must not turn it into lexical supervision.
    assert local_indices('word.',[[0,5]],{'start':4,'end':5,'text':'.'})==([0],[])
    normal='Anna met Bea.'
    ordinary=convert({'retrieved_passages':'Reference1: Anna met Bea.','question':'','response':normal},
                     {'start':9,'end':12,'text':'Bea'},llama,bert)
    assert ordinary['mapping_complete'] and ordinary['nfc_proof']['used_legacy_path']
    old=legacy_map.character_map(normal,ordinary['response_token_offsets'],
                                 ordinary['answer_encoder_start'],ordinary['answer_encoder_end'])
    assert all(np.array_equal(v,np.asarray(w,dtype=v.dtype)) for v,w in zip(old,ordinary['mapping']))
    assert not torch.cuda.is_initialized()
    result={'status':'passed','second_repeated_target_by_original_coordinates':True,
            'punctuation_target_has_no_lexical_label_even_in_mixed_raw_token':True,
            'ordinary_mapping_legacy_exact':True,'NFC_synthetic_full_mapping':out['mapping_complete'],
            'outside_target_unknown':True,'GPU_used':False,'trained':False}
    save('CPU_SELFCHECK.json',result)


def run():
    check_frozen()
    assert not (OUT/'started.json').exists(),'Preserve any interrupted run; do not silently overwrite'
    save('started.json',{'freeze_sha256':replay.sha(OUT/'token_design_freeze.json'),'time':time.time()})
    llama=AutoTokenizer.from_pretrained(replay.MODEL,local_files_only=True)
    bert=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert llama.is_fast and bert.is_fast and bert.model_max_length==LIMIT
    cpu_selfcheck(llama,bert)
    start=time.perf_counter();counter=Counter();lengths=[];raw_lengths=[];index=[];pairs=[];errors=[];repairs=[];positions=[]
    inputs=iter(readl(DATA/'model_inputs.jsonl'))
    with (OUT/'token_inputs.jsonl').open('wb') as stream:
        for pi,pair in enumerate(readl(DATA/'candidate_pairs.jsonl')):
            assert pair['official_split']=='train' and pair['partition']=='auxiliary_candidate_fit'
            assert pair['human_gold'] is False and pair['outside_targets']=='unknown_no_new_supervision'
            sides={}
            for side in ('original','preferred'):
                inp=next(inputs)
                assert set(inp)=={'input_id','model_inputs'} and set(inp['model_inputs'])==INPUT_KEYS
                assert inp['input_id']==pair[side]['input_id']
                assert replay.digest(inp['model_inputs'])==pair[side]['model_inputs_sha256']
                text=inp['model_inputs']['response'];target=pair[side]['target']
                assert replay.digest(text)==pair[side]['response_sha256']
                assert text[target['start']:target['end']]==target['text']
                record=convert(inp['model_inputs'],target,llama,bert)
                record.update(input_id=inp['input_id'],pair_id=pair['pair_id'],side=side,
                              source_response_id=pair['source_response_id'],source_id=pair['source_id'],group_id=pair['group_id'],
                              candidate_pair_sha256=replay.digest(pair),partition='auxiliary_candidate_fit',official_split='train')
                pointer=stream.tell();positions.append(pointer)
                raw=(json.dumps(record,ensure_ascii=False)+'\n').encode('utf-8');stream.write(raw)
                desc={k:record[k] for k in ('input_id','pair_id','side','source_response_id','source_id','group_id','status','mapping_complete','local_lexical_supervision_available')}
                desc.update(record_index=len(index),byte_offset=pointer,byte_length=len(raw),
                            record_bytes_sha256=__import__('hashlib').sha256(raw).hexdigest(),
                            input_tokens=record.get('input_tokens'),raw_answer_tokens=record.get('raw_answer_tokens'),
                            target_raw_lexical_count=len(record.get('target_raw_lexical_indices',[])),
                            target_encoder_lexical_count=len(record.get('target_encoder_lexical_indices',[])),
                            target=target,exceptions=record['exceptions'])
                index.append(desc);sides[side]=desc
                counter['attempted_inputs']+=1;counter['mapped_inputs']+=int(record['mapping_complete'])
                counter['local_lexical_sides']+=int(record['local_lexical_supervision_available'])
                if 'input_tokens' in record:
                    lengths.append(record['input_tokens']);counter['encoder_input_tokens']+=record['input_tokens']
                if 'raw_answer_tokens' in record:
                    raw_lengths.append(record['raw_answer_tokens']);counter['raw_answer_tokens']+=record['raw_answer_tokens']
                    counter['raw_full_input_tokens']+=record['raw_full_input_tokens']
                    counter['lexical_answer_tokens']+=sum(record['lexical_mask'])
                    counter['local_raw_lexical_tokens']+=len(record['target_raw_lexical_indices'])
                if record.get('nfc_proof',{}).get('repaired_character_count',0):
                    repairs.append({'input_id':record['input_id'],'pair_id':pair['pair_id'],**record['nfc_proof']})
                    counter['NFC_repaired_inputs']+=1
                    counter['NFC_repaired_characters']+=record['nfc_proof']['repaired_character_count']
                for exc in record['exceptions']:
                    errors.append({'input_id':record['input_id'],'pair_id':pair['pair_id'],'full_record_retained':True,**exc})
            reasons=[]
            if not pair['eligibility']['structural_eligible']:
                reasons+=['upstream:'+r for r in pair['eligibility']['reasons']]
            for side in ('original','preferred'):
                desc=sides[side]
                if not desc['mapping_complete']:reasons.append(side+':mapping_incomplete')
                if not desc['local_lexical_supervision_available']:reasons.append(side+':no_local_lexical_supervision')
                reasons += [side+':'+e['stage'] for e in desc['exceptions']]
            pairs.append({'pair_id':pair['pair_id'],'source_response_id':pair['source_response_id'],
                          'source_id':pair['source_id'],'group_id':pair['group_id'],'type':pair['type'],
                          'candidate_index':pi,'candidate_pair_sha256':replay.digest(pair),
                          'original':sides['original'],'preferred':sides['preferred'],
                          'eligible_for_future_local_pair_training':not reasons,'ineligibility_reasons':reasons,
                          'both_sides_one_pair_unit':True,'outside_targets':'unknown',
                          'candidate_suggested_base_weight':pair['suggested_base_weight'],
                          'eligible_pair_base_weight':None})
            if (pi+1)%1000==0:
                print('FAVA_LOCAL_PAIR_TOKENS',pi+1,10040,round(time.perf_counter()-start,1),flush=True)
    assert next(inputs,None) is None
    assert len(pairs)==10040 and len(index)==counter['attempted_inputs']==20080
    assert len({r['input_id'] for r in index})==20080 and len({r['pair_id'] for r in pairs})==10040
    # Conditional eligible-only weights. No pair is deleted from any artifact.
    tree=defaultdict(lambda:defaultdict(list))
    for p in pairs:
        if p['eligible_for_future_local_pair_training']:
            tree[p['group_id']][p['source_response_id']].append(p)
    group_mass=Counter();answer_mass=Counter();eligible_pairs=0
    for group,answers in tree.items():
        for answer,entries in answers.items():
            denominator=len(tree)*len(answers)*len(entries)
            for p in entries:
                p['eligible_pair_base_weight']={'numerator':1,'denominator':denominator,'value':1/denominator,
                                               'participating_groups':len(tree),'original_answers_in_group':len(answers),
                                               'eligible_pairs_in_original_answer':len(entries),'scope':'eligible_subset_only_not_a_training_loss'}
                group_mass[group]+=1/denominator;answer_mass[answer]+=1/denominator;eligible_pairs+=1
    assert abs(sum(group_mass.values())-1)<1e-10
    assert max(abs(v-1/len(tree)) for v in group_mass.values())<1e-14
    np.save(OUT/'token_byte_offsets.npy',np.asarray(positions,np.int64))
    savel('input_index.jsonl',index);savel('paired_index.jsonl',pairs);savel('exceptions.jsonl',errors);savel('nfc_repairs.jsonl',repairs)
    savel('eligible_material_group_index.jsonl',[{'group_id':g,'original_answers':len(answers),
          'eligible_pairs':sum(len(p) for p in answers.values()),'original_answer_ids':sorted(answers),
          'base_mass':group_mass[g]} for g,answers in sorted(tree.items())])
    stats={'candidate_pairs':10040,'candidate_inputs':20080,'candidate_original_answers':len({p['source_response_id'] for p in pairs}),
           'candidate_material_groups':len({p['group_id'] for p in pairs}),
           'eligible_pairs':eligible_pairs,'eligible_original_answers':len(answer_mass),'eligible_material_groups':len(tree),
           'ineligible_but_retained_pairs':10040-eligible_pairs,'counters':dict(counter),
           'all_input_lengths_known':len(lengths)==20080,'max_encoder_input_tokens':max(lengths,default=0),
           'median_encoder_input_tokens':float(np.median(lengths)) if lengths else None,
           'max_raw_answer_tokens':max(raw_lengths,default=0),'mapping_or_length_exception_events':len(errors),
           'ineligible_pairs':[{'pair_id':p['pair_id'],'reasons':p['ineligibility_reasons']} for p in pairs if not p['eligible_for_future_local_pair_training']],
           'eligible_weight_sum':sum(group_mass.values()),'all_candidates_retained':True,
           'no_truncation':True,'outside_targets_unknown':True,'full_answer_labels_created':False,
           'GPU_used':False,'trained':False,'QA_fit_cal_test_read':False,'QA_inputs_weights_modified':False,
           'seconds':time.perf_counter()-start}
    save('TOKEN_REPORT.json',stats)
    check_frozen()
    names=('token_inputs.jsonl','token_byte_offsets.npy','input_index.jsonl','paired_index.jsonl','exceptions.jsonl',
           'nfc_repairs.jsonl','eligible_material_group_index.jsonl','TOKEN_REPORT.json','CPU_SELFCHECK.json',
           'token_protocol.json','token_design_freeze.json','started.json')
    manifest={'status':'all_candidates_tokenized_not_trained','candidate_pairs':10040,'input_records':20080,
              'mapped_inputs':counter['mapped_inputs'],'eligible_pairs':eligible_pairs,
              'eligible_original_answers':len(answer_mass),'eligible_material_groups':len(tree),
              'mapping_or_length_exceptions':len(errors),'exception_review_needed':bool(errors),
              'candidate_manifest_sha256':replay.sha(DATA/'manifest.json'),
              'files_sha256':{n:replay.sha(OUT/n) for n in names},
              'source_isolation_inherited_not_rescanned':True,'all_rows_retained':True,
              'outside_targets_unknown':True,'GPU_used':False,'trained':False,'QA_fit_cal_test_read':False}
    save('manifest.json',manifest)
    save('complete.json',{'status':'CPU_tokenization_complete_not_trained','manifest_sha256':replay.sha(OUT/'manifest.json'),
                          'candidate_pairs':10040,'input_records':20080,'eligible_pairs':eligible_pairs,
                          'all_rows_retained':True,'exceptions_require_review':bool(errors),'GPU_used':False,'trained':False})
    assert not torch.cuda.is_initialized()
    print(json.dumps({k:v for k,v in stats.items() if k!='ineligible_pairs'}),flush=True)


def check():
    check_frozen()
    complete=replay.read(OUT/'complete.json');manifest=replay.read(OUT/'manifest.json')
    assert replay.sha(OUT/'manifest.json')==complete['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert replay.sha(OUT/name)==digest,name
    offsets=np.load(OUT/'token_byte_offsets.npy',allow_pickle=False)
    inputs=iter(readl(DATA/'model_inputs.jsonl'))
    checked=0
    with (OUT/'token_inputs.jsonl').open('rb') as stream:
        for i,index in enumerate(readl(OUT/'input_index.jsonl')):
            assert stream.tell()==offsets[i]==index['byte_offset']
            raw=stream.read(index['byte_length']);r=json.loads(raw);original=next(inputs)
            assert __import__('hashlib').sha256(raw).hexdigest()==index['record_bytes_sha256']
            assert r['model_inputs']==original['model_inputs'] and r['input_id']==original['input_id']==index['input_id']
            text=r['model_inputs']['response'];target=r['target']
            assert text[target['start']:target['end']]==target['text']
            assert 'risk_mask' not in r and 'answer_risk' not in r and r['outside_target']=='unknown_no_binary_supervision'
            if r['mapping_complete']:
                a,b,w=(np.asarray(x) for x in r['mapping']);n=len(r['response_token_ids'])
                assert len(a)==len(b)==len(w) and len(r['lexical_mask'])==n
                total=np.bincount(a.astype(int),weights=w,minlength=n)
                nonspace=[any(not c.isspace() for c in text[l:h]) for l,h in r['response_token_offsets']]
                assert np.max(np.abs(total[np.asarray(nonspace)]-1),initial=0)<2e-7
                assert local_indices(text,r['response_token_offsets'],target)==(r['target_raw_token_indices'],r['target_raw_lexical_indices'])
                assert encoder_local_indices(text,r['answer_encoder_start'],r['answer_encoder_end'],target,r['nfc_proof'])==(r['target_encoder_token_indices'],r['target_encoder_lexical_indices'])
            checked+=1
        assert not stream.read(1)
    assert checked==20080 and next(inputs,None) is None and not torch.cuda.is_initialized()
    save('CPU_OUTPUT_CHECK.json',{'status':'passed','all_input_records':checked,'byte_index_and_line_hashes_exact':True,
                                 'original_whitelist_text_and_target_slices_exact':True,'all_local_indices_and_row_masses_checked':True,
                                 'outside_target_unknown':True,'GPU_used':False,'trained':False,
                                 'manifest_sha256':replay.sha(OUT/'manifest.json')})
    print('FAVA_LOCAL_PAIR_TOKEN_OUTPUT_CHECK_PASSED20080',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run','check'))
    args=parser.parse_args()
    globals()[args.stage]()
