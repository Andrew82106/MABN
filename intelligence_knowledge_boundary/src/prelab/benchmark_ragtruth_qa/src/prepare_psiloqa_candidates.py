"""Stage fixed official English TRAIN rows as automatic auxiliary candidates.

No tokenizer, model, network or fit. Golden answers and generator identities
exist only in a separate provenance file, never in candidate input records.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import re
import shutil
import time
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'auxiliary_psiloqa_review_v1'
OUT=ROOT/'auxiliary_psiloqa_v1'
REV='375c3321b8331a689d664f77eb2753c6cdb17159'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def savel(n,x):
    with (OUT/n).open('w',encoding='utf-8') as f:
        for row in x:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def dist(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'sum':int(x.sum()),'mean':float(x.mean()),
        **dict(zip(['min','p50','p95','p99','max'],map(float,np.percentile(x,[0,50,95,99,100]))))}


def sources():
    names=['train-00000-of-00001.parquet','download_manifest.json','review_complete.json','complete.json',
        'diagnostic_details.json','coordinate_status.jsonl','english_rows.jsonl','english_article_groups.jsonl',
        'alignment_failures.jsonl','same_material_question_answer_duplicates.jsonl','source_material_matches.jsonl','reported_quarantine.jsonl']
    return {str(p.resolve()):sha(p) for p in [Path(__file__),*[REVIEW/n for n in names]]}


def protocol():return {
    'scope':'Only previously downloaded pinned official PsiloQA TRAIN, lang=en. No new download, validation/test read, tokenizer, model, GPU or training.',
    'selection':'Use existing strict_coordinate_pass ledger, excluding263 character/markup failures. Retain4 null complexity if coordinates pass. Quarantine all6 rows in3 fixed same-material/question/answer conflicting-label groups; report intersection, never choose one label or repair. Reuse completed source-only quarantine without a new scan.',
    'input_fields':['retrieved_passages','question','original_response'],
    'input_mapping':{'retrieved_passages':'exact wiki_passage','question':'exact question','original_response':'exact llm_answer'},
    'never_inputs':'golden_answer, llm_checkpoint, original model-bearing id, annotated_span, complexity, wiki title/url and released markup are only in provenance, not in candidate records or model-input file.',
    'labels':'Original Python Unicode end-exclusive label pairs, order and answer untouched. candidate labels add text only by original answer[start:end]. One binary inconsistency label; do not invent a four-type human taxonomy.',
    'identity':'response_id=psilo_train_{zero_based_parquet_row_index}; source_id=psilo_material_{raw_passage_sha256}; group_id unchanged from reviewed original article/material group. These keys are bookkeeping, never numeric/text features.',
    'groups':'Preserve all6888 original groups and each member decision, including groups with no retained members. No new split; all retained candidates are auxiliary_candidate_fit.',
    'provenance':'Separate full original English records for all16115 rows, stable id mapping, dataset revision and original row/text/labels hashes. Copy existing267-row failure ledger and original article group ledger byte-for-byte; extra4 metadata-only entries remain kept.',
    'limits':'Automatic GPT annotation, no-context response generation, refusals prefiltered, language field not verified per sentence. High risk share is not deployment prevalence. Coordinate integrity is not semantic truth verification. No human gold and no current training inclusion.',
}


def design():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design_freeze.json').exists()
    assert read(REVIEW/'review_complete.json')['status']=='complete'
    save('DATA_PROTOCOL.json',protocol())
    save('design_freeze.json',{'sources_sha256':sources(),'protocol_sha256':sha(OUT/'DATA_PROTOCOL.json')})
    print('PSILOQA_CANDIDATE_PROTOCOL_FROZEN',flush=True)


def check_design():
    frozen=read(OUT/'design_freeze.json')
    assert frozen['sources_sha256']==sources()
    assert read(OUT/'DATA_PROTOCOL.json')==protocol() and frozen['protocol_sha256']==sha(OUT/'DATA_PROTOCOL.json')
    for n,h in read(REVIEW/'complete.json')['files_sha256'].items():assert sha(REVIEW/n)==h
    for n,h in read(REVIEW/'review_complete.json')['files_sha256'].items():assert sha(REVIEW/n)==h
    downloaded=read(REVIEW/'download_manifest.json');assert downloaded['dataset_revision']==REV
    for f in downloaded['files']:assert sha(REVIEW/f['path'])==f['sha256']
    return frozen


def run():
    frozen=check_design();assert not (OUT/'started.json').exists()
    save('started.json',{'time':time.time(),'design_freeze_sha256':sha(OUT/'design_freeze.json')});tick=time.perf_counter()
    table=pq.read_table(REVIEW/'train-00000-of-00001.parquet')
    original=[(i,r) for i,r in enumerate(table.to_pylist()) if r['lang']=='en']
    coord={r['id']:r for r in rows(REVIEW/'coordinate_status.jsonl')}
    checked={r['id']:r for r in rows(REVIEW/'english_rows.jsonl')}
    groups=list(rows(REVIEW/'english_article_groups.jsonl'));group_by_id={i:g['group_id'] for g in groups for i in g['ids']}
    conflicts=list(rows(REVIEW/'same_material_question_answer_duplicates.jsonl'))
    conflict_ids={i for d in conflicts if d['different_label_lists'] for i in d['ids']}
    quarantined={r['id'] for r in rows(REVIEW/'reported_quarantine.jsonl')}
    assert len(original)==len(coord)==len(checked)==16115 and len(groups)==6888
    assert len(conflicts)==3 and len(conflict_ids)==6 and not quarantined
    badcoord={i for i,c in coord.items() if not c['strict_coordinate_pass']};assert len(badcoord)==263
    idmap={r['id']:f'psilo_train_{i}' for i,r in original}
    decisions=[];candidate=[];provenance=[];inputonly=[]
    for rawindex,r in original:
        oid=r['id'];rid=idmap[oid];reasons=[]
        if oid in badcoord:reasons.append('character_or_markup_alignment_failure')
        if oid in conflict_ids:reasons.append('same_detector_input_conflicting_released_labels_entire_group')
        if oid in quarantined:reasons.append('existing_source_only_material_quarantine')
        decision={'response_id':rid,'group_id':group_by_id[oid],'train_row_index':rawindex,
            'kept':not reasons,'exclusion_reasons':reasons,'coordinate_pass':coord[oid]['strict_coordinate_pass'],
            'metadata_issues':coord[oid]['metadata_issues']}
        decisions.append(decision)
        provenance.append({'response_id':rid,'original_row_index':rawindex,'source_dataset':'s-nlp/PsiloQA',
            'source_revision':REV,'original_record_sha256':digest(canon(r)),
            'text_sha256':{k:digest(r[k]) for k in ['wiki_passage','question','llm_answer','golden_answer','annotated_span']},
            'original_labels_sha256':digest(canon(r['labels'])),'original_record':r,
            'candidate_status':'kept' if not reasons else 'excluded','exclusion_reasons':reasons,
            'provenance_only_not_detector_input':True})
        if reasons:continue
        answer=r['llm_answer'];clean=re.sub(r'\[/?HAL\]','',r['annotated_span'])
        assert clean==answer
        spans=[]
        for m in re.finditer(r'\[HAL\](.*?)\[/HAL\]',r['annotated_span'],re.S):
            start=len(re.sub(r'\[/?HAL\]','',r['annotated_span'][:m.start()]))
            spans.append([start,start+len(m.group(1))])
        assert spans==r['labels'] and all(0<=a<b<=len(answer) for a,b in spans)
        labels=[{'start':a,'end':b,'text':answer[a:b]} for a,b in r['labels']]
        modelinput={'response_id':rid,'retrieved_passages':r['wiki_passage'],'question':r['question'],'original_response':answer}
        inputonly.append(modelinput)
        candidate.append({**modelinput,'source_id':'psilo_material_'+digest(r['wiki_passage']),
            'group_id':group_by_id[oid],'partition':'auxiliary_candidate_fit','official_split':'train',
            'task_type':'QA_automatic','labels':labels,'risk':int(bool(labels))})
    savel('candidate_fit.jsonl',candidate);savel('model_inputs.jsonl',inputonly)
    savel('provenance.jsonl',provenance);savel('all_english_decisions.jsonl',decisions)
    savel('excluded_candidates.jsonl',[d for d in decisions if not d['kept']])
    savel('conflicting_input_groups.jsonl',[{'input_key_sha256':d['input_key_sha256'],
        'response_ids':[idmap[i] for i in d['ids']],'entire_group_quarantined':True,
        'also_character_failure':[idmap[i] for i in d['ids'] if i in badcoord]} for d in conflicts])
    keep={r['response_id'] for r in candidate};decision_by_id={r['response_id']:r for r in decisions}
    group_rows=[]
    for g in groups:
        ids=[idmap[i] for i in g['ids']]
        group_rows.append({'group_id':g['group_id'],'all_response_ids':ids,
            'kept_response_ids':[i for i in ids if i in keep],
            'excluded_members':[decision_by_id[i] for i in ids if i not in keep],
            'original_material_sha256':g['material_sha256']})
    savel('article_group_index.jsonl',group_rows)
    for source,name in [('alignment_failures.jsonl','original_review_failure_ledger.jsonl'),
                        ('english_article_groups.jsonl','original_article_groups.jsonl')]:
        shutil.copyfile(REVIEW/source,OUT/name);assert sha(REVIEW/source)==sha(OUT/name)
    lengths={field:dist([len(c[field]) for c in candidate]) for field in ['retrieved_passages','question','original_response']}
    lengths['concatenated_three_fields_without_separators']=dist([sum(len(c[f]) for f in ['retrieved_passages','question','original_response']) for c in candidate])
    spans=[s for c in candidate for s in c['labels']]
    result={'status':'staged_automatic_auxiliary_candidates_not_trained','english_train_rows':len(original),
        'coordinate_pass_before_duplicate_quarantine':len(original)-len(badcoord),'character_failures':len(badcoord),
        'conflicting_groups':len(conflicts),'conflict_rows':len(conflict_ids),'conflict_and_character_failure_intersection':len(conflict_ids&badcoord),
        'additional_conflict_exclusions_after_coordinate_filter':len(conflict_ids-badcoord),
        'total_excluded_rows':len(decisions)-len(candidate),'candidate_answers':len(candidate),
        'candidate_risk_answers':sum(c['risk'] for c in candidate),'candidate_unmarked_answers':sum(not c['risk'] for c in candidate),
        'risk_fraction_not_deployment_prevalence':sum(c['risk'] for c in candidate)/len(candidate),
        'original_article_groups_recorded':len(groups),'retained_article_groups':len({c['group_id'] for c in candidate}),
        'retained_material_sources':len({c['source_id'] for c in candidate}),
        'retained_null_complexity_answers':sum(p['response_id'] in keep and p['original_record']['complexity'] is None for p in provenance),
        'source_only_overlap_quarantine_reused_count':len(quarantined),'labels':len(spans),
        'label_characters':sum(s['end']-s['start'] for s in spans),'span_character_lengths':dist([s['end']-s['start'] for s in spans]),
        'character_lengths':lengths,'annotation_categories':'Only automatic binary inconsistency; no original four-way error taxonomy.',
        'source_revision':REV,'human_gold':False,'trained':False,'GPU_used':False,'tokenizer_loaded':False,
        'new_downloads':False,'validation_test_read':False,'existing_train_cal_test_or_status_modified':False,
        'design_sha256':sha(OUT/'design_freeze.json'),'sources_sha256':frozen['sources_sha256'],
        'elapsed_seconds':time.perf_counter()-tick}
    save('manifest.json',result)
    verify()
    artifacts=['candidate_fit.jsonl','model_inputs.jsonl','provenance.jsonl','all_english_decisions.jsonl','excluded_candidates.jsonl',
        'conflicting_input_groups.jsonl','article_group_index.jsonl','original_review_failure_ledger.jsonl','original_article_groups.jsonl',
        'DATA_PROTOCOL.json','design_freeze.json','manifest.json','DATA_CHECK.json']
    save('complete.json',{'status':'complete_candidates_not_training','files_sha256':{n:sha(OUT/n) for n in artifacts},
        'trained':False,'tokenization_started':False,'GPU_used':False})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def verify():
    check_design();c=list(rows(OUT/'candidate_fit.jsonl'));inputs=list(rows(OUT/'model_inputs.jsonl'))
    provenance={r['response_id']:r for r in rows(OUT/'provenance.jsonl')}
    decisions=list(rows(OUT/'all_english_decisions.jsonl'));groups=list(rows(OUT/'article_group_index.jsonl'))
    assert len(c)==len({r['response_id'] for r in c})==len(inputs)
    labelkeys={};required={'response_id','retrieved_passages','question','original_response'}
    for row,inp in zip(c,inputs):
        assert set(inp)==required and all(inp[k]==row[k] for k in required)
        assert not {'golden_answer','llm_checkpoint','annotated_span','original_id','complexity','wiki_title','wiki_url'}&set(row)
        orig=provenance[row['response_id']]['original_record']
        assert row['retrieved_passages']==orig['wiki_passage'] and row['question']==orig['question'] and row['original_response']==orig['llm_answer']
        assert [[s['start'],s['end']] for s in row['labels']]==orig['labels']
        assert all(s['text']==row['original_response'][s['start']:s['end']] for s in row['labels'])
        key=canon([row['retrieved_passages'],row['question'],row['original_response']])
        assert key not in labelkeys, 'Any repeated detector input must be reviewed, not silently merged'
        labelkeys[key]=row['labels']
    keep={r['response_id'] for r in c};assert keep=={r['response_id'] for r in decisions if r['kept']}
    all_members=[i for g in groups for i in g['all_response_ids']]
    assert len(all_members)==len(set(all_members))==len(provenance)==16115
    assert keep=={i for g in groups for i in g['kept_response_ids']}
    assert len(keep)==15847 and sum(r['risk'] for r in c)==15212
    assert sum(provenance[i]['original_record']['complexity'] is None for i in keep)==4
    save('DATA_CHECK.json',{'status':'passed','candidate_answers':len(c),'unique_ids_and_detector_inputs':True,
        'original_three_input_fields_exact':True,'original_label_pairs_and_span_text_exact':True,
        'separate_provenance_only_golden_answer_and_generator':True,'all_original_article_members_accounted':True,
        'null_complexity_four_kept':True,'tokenizer_or_model_loaded':False,'trained':False})
    print('PSILOQA_CANDIDATE_CHECK_PASSED',len(c),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['design','run','check']);args=parser.parse_args()
    {'design':design,'run':run,'check':verify}[args.command]()
