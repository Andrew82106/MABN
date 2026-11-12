"""CPU-only disclosed tokenizer feasibility, not a frozen replay definition."""
from pathlib import Path
import json,hashlib,statistics
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT.parent/'models/Llama-2-7b-chat-hf'
tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True,trust_remote_code=False)
assert tok.is_fast
lengths=[];crossing=[];missing_span_coverage=[]
for part in ['fit','calibration']:
    for line in (ROOT/'data'/f'{part}.jsonl').read_text('utf-8').splitlines():
        r=json.loads(line)
        # One explicit separator after [/INST] is a reconstruction assumption.
        prefix='<s>[INST] '+r['released_prompt']+' [/INST] '
        text=prefix+r['original_response']
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        answer_tokens=[(i,a-len(prefix),b-len(prefix)) for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix) and b>a]
        cross=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<len(prefix)<b]
        if cross:crossing.append({'response_id':r['response_id'],'token_indices':cross})
        for j,l in enumerate(r['labels']):
            if not any(max(a,l['start'])<min(b,l['end']) for _,a,b in answer_tokens):missing_span_coverage.append({'response_id':r['response_id'],'label_index':j})
        lengths.append({'response_id':r['response_id'],'partition':part,'total_tokens':len(enc['input_ids']),'response_tokens_overlapping_answer':len(answer_tokens)})
out={'status':'feasibility_only_not_frozen_extraction','model_loaded':False,'gpu_used':False,'test_data_read':False,
 'tokenizer_class':type(tok).__name__,'tokenizer_is_fast':tok.is_fast,'vocab_size':tok.vocab_size,
 'template_assumption':'<s>[INST] {released_prompt} [/INST] {original_response}; one separator space; add_special_tokens=False. Original token trajectory unavailable.',
 'rows':len(lengths),'max_total_tokens':max(r['total_tokens'] for r in lengths),'median_total_tokens':statistics.median(r['total_tokens'] for r in lengths),
 'sum_response_tokens':sum(r['response_tokens_overlapping_answer'] for r in lengths),'median_response_tokens':statistics.median(r['response_tokens_overlapping_answer'] for r in lengths),
 'over_4096':sum(r['total_tokens']>4096 for r in lengths),'prefix_boundary_crossing_rows':crossing,'labels_without_overlapping_token':missing_span_coverage,
 'offset_policy_needed':'Use original response character coordinates; a token crossing the prefix boundary must not silently remove answer characters. Do not invent original token IDs.',
 'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL.glob('*') if p.name in ['tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','config.json']},
 'per_row':lengths}
(ROOT/'data/tokenizer_feasibility.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
brief={k:v for k,v in out.items() if k not in ['per_row','files_sha256','prefix_boundary_crossing_rows','labels_without_overlapping_token']}
brief['prefix_boundary_crossing_row_count']=len(crossing)
brief['labels_without_overlapping_token_count']=len(missing_span_coverage)
print(json.dumps(brief,ensure_ascii=False,indent=2))
