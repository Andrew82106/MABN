"""General final QA data export. Real raw reads require explicit root release."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,re,sys
import contracts as c
from gold_rows import map_row
sys.path.insert(0,str(c.ROOT/'src'))
import feature_qa as feature

TYPES={'Evident Baseless Info','Subtle Baseless Info','Evident Conflict','Subtle Conflict'}

def layout(tokenizer,row):
    # Original encode_view is independent of dataset partition. Do not falsify
    # test rows as fit to bypass the old development-only prepare_row guard.
    prompt=row['released_prompt'];passages=row['retrieved_passages'];text=row['original_response']
    assert passages and prompt.count(passages)==1
    start=prompt.index(passages)
    view=feature.encode_view(tokenizer,prompt,text,(start,start+len(passages)))
    assert len(view['input_ids'])<=4096,'Fixed canonical replay context exceeded; stop, do not truncate/drop row'
    return {'version':feature.VERSION,'response_id':row['response_id'],'source_id':row['source_id'],'group_id':row['group_id'],
        'partition':row['partition'],'official_split':row['official_split'],'released_prompt':prompt,'original_response':text,
        'prompt_sha256':c.text_sha(prompt),'answer_sha256':c.text_sha(text),'original':view,'no_context':None,
        'labels_used':False,'exact_original_generation_trace':False,
        'replay_definition':'Fixed wrapper/tokenizer/checkpoint/precision teacher-forced reconstruction of published text'}

def package(rows,tokenizer,partition):
    assert rows and len({r['response_id'] for r in rows})==len(rows)
    eligible=[];excluded=[];plans=[];tokens=[];windows=[];answers=[];window_excluded=[];edges=[]
    for row in rows:
        assert row['partition']==partition and row['model']=='llama-2-7b-chat'
        assert c.text_sha(row['original_response'])==row['answer_sha256'] and c.text_sha(row['released_prompt'])==row['prompt_sha256']
        if row['quality']!='good':
            excluded.append({**row,'official_quality_eligible':False,'exclusion_reason':'official_quality_not_good'});continue
        assert all(s['label_type'] in TYPES for s in row['labels']),'Unknown official type; do not omit it'
        eligible.append(row);plan=layout(tokenizer,row);plans.append(plan)
        token,w,ex,answer,e=map_row(row,plan);tokens.append(token);windows.extend(w);window_excluded.extend(ex);answers.append(answer);edges.extend(e)
    assert eligible,'No eligible answers; stop rather than invent a score'
    return {'original_records':rows,'quality_excluded':excluded,'rows':eligible,'plans':plans,'tokens':tokens,'windows_k4':windows,'windows_excluded':window_excluded,'answers':answers,'edges':edges}

def write_bundle(out,data,partition,binding,simulation=False):
    out=Path(out);assert not out.exists(),'Never overwrite a release or simulation bundle'
    out.mkdir(parents=True);(out/f'feature_preparation_{partition}').mkdir()
    files={}
    payloads={f'{partition}_original_records.jsonl':data['original_records'],f'{partition}_quality_excluded.jsonl':data['quality_excluded'],
        f'{partition}_inputs.jsonl':[{k:r[k] for k in feature.VISIBLE_FIELDS} for r in data['rows']],
        f'feature_preparation_{partition}/plans.jsonl':data['plans'],
        **{f'{stem}_{partition}.jsonl':data[stem] for stem in ('tokens','windows_k4','windows_excluded','answers')}}
    for name,value in payloads.items():c.save_lines(out/name,value);files[name]=c.sha(out/name)
    c.save(out/'gold_edge_cases.json',{'cases':data['edges'],'counts':dict(Counter(e['kind'] for e in data['edges'])),'policy':'Report only; never change labels or discard quality-good answers'})
    files['gold_edge_cases.json']=c.sha(out/'gold_edge_cases.json')
    counts={'official_candidate_answers':len(data['original_records']),'quality_good_answers':len(data['answers']),'quality_excluded_answers':len(data['quality_excluded']),
        'groups':len({a['group_id'] for a in data['answers']}),'risk_answers':sum(a['label'] for a in data['answers']),
        'raw_tokens':sum(t['token_count'] for t in data['tokens']),'lexical_tokens':sum(sum(t['lexical_mask']) for t in data['tokens']),
        'risk_tokens':sum(sum(t['risk_mask']) for t in data['tokens']),'eligible_windows':len(data['windows_k4']),'risk_windows':sum(w['label'] for w in data['windows_k4']),
        'windows_without_lexical_tokens':len(data['windows_excluded']),'official_spans':sum(len(a['original_labels']) for a in data['answers'])}
    manifest={'complete':True,'selfcheck_passed':True,'mode':'calibration_simulation' if simulation else 'authorized_official_test',
        'partition':partition,'binding':binding,'counts':counts,'outputs_sha256':files,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'labels_changed':False,'quality_rule':'official_quality_good_only','all_four_types_retained':True,'refusal_classifier_used':False,
        'models_run_or_fitted':False,'official_test_read':not simulation,'canonical_first_boundary_token_retained':True,
        'no_eligible_window_answer_ids':[a['response_id'] for a in data['answers'] if not a['eligible_window_count']],
        'stable_data_layer_only':'Model-specific feature/inference adapters are separate and must be frozen before release; test partition is preserved as test.'}
    c.save(out/'bundle_manifest.json',manifest);return manifest

def tokenizer():
    feature.assert_cpu_only()
    tok=feature.AutoTokenizer.from_pretrained(feature.MODEL,local_files_only=True,use_fast=True,trust_remote_code=False)
    assert feature.tokenizer_signature(tok)==c.read(c.ROOT/'data/feature_preparation/manifest.json')['signature']['tokenizer']
    return tok

def release(development_freeze_path,out):
    # This is the first operation. No raw hashes or record iteration above it.
    frozen,identity,binding=c.require_release(development_freeze_path)
    assert Path(out).resolve()==(c.ROOT/'data/final_test').resolve(),'Fixed real output root only'
    assert not Path(out).exists(),'Test was already released or started; no silent second release'
    rawroot=c.ROOT.parent;expected=c.read(c.ROOT/'data/development_manifest.json')['source_files_sha256']
    raw_source=rawroot/'data/raw/source_info.jsonl';raw_response=rawroot/'data/raw/response.jsonl'
    for p in (raw_source,raw_response):
        rel=str(p.relative_to(rawroot));assert c.sha(p)==expected[rel],('Raw snapshot changed',rel)
    identities={r['source_id']:r for r in identity['identities']};response_ids={r['response_id'] for r in identity['identities']}
    sid_re=re.compile(r'(?<!\\)"source_id"\s*:\s*"([^"\\]+)"');rid_re=re.compile(r'(?<!\\)"id"\s*:\s*"([^"\\]+)"')
    sources={};responses={}
    import json
    with raw_source.open(encoding='utf-8') as handle:
        for line in handle:
            match=sid_re.search(line)
            if not match or match.group(1) not in identities:continue
            s=json.loads(line);assert s['source_id'] not in sources and s['task_type']=='QA';sources[s['source_id']]=s
    with raw_response.open(encoding='utf-8') as handle:
        for line in handle:
            sm=sid_re.search(line);rm=rid_re.search(line)
            if not sm or not rm or sm.group(1) not in identities or rm.group(1) not in response_ids:continue
            r=json.loads(line);sid=r['source_id'];ident=identities[sid]
            assert sid not in responses and r['id']==ident['response_id'] and r['split']=='test' and r['model']=='llama-2-7b-chat'
            responses[sid]=r
    assert set(sources)==set(responses)==set(identities)
    index={r['source_id']:r for r in c.lines(c.ROOT/'data/source_index.jsonl')};rows=[]
    for ident in identity['identities']:
        sid=ident['source_id'];s=sources[sid];r=responses[sid]
        assert c.text_sha(s['prompt'])==index[sid]['source_prompt_sha256']
        assert c.text_sha(s['source_info']['passages'])==index[sid]['retrieved_passages_sha256']
        rows.append({'source_id':sid,'group_id':ident['group_id'],'partition':'test','official_split':'test','response_id':r['id'],
            'model':r['model'],'temperature':r['temperature'],'quality':r['quality'],'question':s['source_info']['question'],
            'retrieved_passages':s['source_info']['passages'],'released_prompt':s['prompt'],'original_response':r['response'],'labels':r['labels'],
            'annotation_origin':'RAGTruth released human span annotation','label_offsets':'original response character offsets, end exclusive; no text normalization',
            'answer_sha256':c.text_sha(r['response']),'prompt_sha256':c.text_sha(s['prompt']),
            'official_quality_eligible':r['quality']=='good','raw_source_record':s,'raw_response_record':r})
    assert len(rows)==150
    data=package(rows,tokenizer(),'test')
    _,_,binding_after=c.require_release(development_freeze_path);assert binding_after==binding
    manifest=write_bundle(out,data,'test',binding,False)
    print('AUTHORIZED_TEST_EXPORT_COMPLETE',manifest['counts'],flush=True)

def simulate(out):
    """Only existing admitted/rejected calibration exports, never raw files."""
    out=Path(out);assert out.resolve().is_relative_to(c.HERE.resolve())
    rows=c.lines(c.ROOT/'data/calibration.jsonl')
    rejected=[r for r in c.lines(c.ROOT/'data/quality_excluded_development.jsonl') if r['partition']=='calibration']
    assert len(rows)==159 and len(rejected)==3 and all(r['official_split']=='train' for r in rows+rejected)
    data=package(rows+rejected,tokenizer(),'calibration')
    oldplans={p['response_id']:p for p in c.lines(c.ROOT/'data/feature_preparation/plans.jsonl')}
    assert all(p['original']==oldplans[p['response_id']]['original'] for p in data['plans'])
    for stem in ('tokens','windows_k4','windows_excluded','answers'):
        assert data[stem]==c.lines(c.ROOT/f'data/{stem}_calibration.jsonl'),('Calibration gold drift',stem)
    binding={'calibration_jsonl_sha256':c.sha(c.ROOT/'data/calibration.jsonl'),'gold_manifest_sha256':c.sha(c.ROOT/'data/gold_manifest.json'),'tools_protocol_sha256':c.sha(c.HERE/'protocol.json')}
    manifest=write_bundle(out,data,'calibration',binding,True)
    c.save(out/'EXPORT_SELFTEST.json',{'passed':True,'all159_token_layouts_exact':True,'all_calibration_gold_records_exact':True,'official_quality_good159_excluded3':True,'official_test_content_read':False,'counts':manifest['counts'],'bundle_manifest_sha256':c.sha(out/'bundle_manifest.json')})
    print('CALIBRATION_EXPORT_SIMULATION_PASS',manifest['counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['simulate','release']);p.add_argument('--freeze');p.add_argument('--out',required=True);a=p.parse_args()
    if a.stage=='simulate':simulate(a.out)
    else:
        assert a.freeze,'Explicit frozen development manifest is required'
        release(a.freeze,a.out)
