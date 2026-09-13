"""Source/ID-only planning inventory; never inspect new answers or risk labels.

JSON records are projected onto source identity / task / model / split fields.
No generation, feature extraction, quality/class filtering, or label counts.
All writes remain inside this research directory; indexes are not data freezes.
"""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter, defaultdict
import unicodedata

HERE=Path(__file__).resolve().parent
PRE=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def h(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def norm(s):return re.sub(r'[^\w]+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def readl(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def save(name,x):
    (HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def savel(name,rows):
    (HERE/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),'utf-8')

files={}; used_titles=set();used_subjects=set();used_question_ids=set();split_counts={}
input_paths=[PRE/'round6_evidence_grounding/data/inputs.jsonl',PRE/'round6_evidence_grounding/data/dev_inputs.jsonl',
 PRE/'round7_evidence_grounding/data/inputs.jsonl',PRE/'round7_evidence_grounding/data/dev_inputs.jsonl',
 PRE/'round9_evidence_binding/data/inputs.jsonl',PRE/'round10_dual_granularity/data/inputs.jsonl',
 PRE/'round16_dataset_expansion/data/inputs.jsonl']
for p in input_paths:
    if not p.exists():continue
    files[str(p.relative_to(PRE))]=sha(p);by=defaultdict(set);groups=defaultdict(set);n=Counter()
    for r in readl(p):
        split=r['split'];n[split]+=1;by[split].add(r['question_id']);groups[split].add(r.get('group_id',r['question_id']))
        used_question_ids.add(r['question_id'].removeprefix('r16_'))
        used_subjects.update(norm(s) for s in r.get('subjects',[]))
        used_titles.update(norm(v['title']) for v in r['passages'])
    split_counts[str(p.relative_to(PRE))]={'rows':dict(n),'questions':{s:len(v) for s,v in by.items()},'group_ids':{s:len(v) for s,v in groups.items()}}
legacy=PRE/'round10_dual_granularity/data/curation/legacy_isolation_inventory.json'
d=json.loads(legacy.read_text('utf-8'));used_titles.update(d['old_titles']);used_subjects.update(d['old_subjects']);files[str(legacy.relative_to(PRE))]=sha(legacy)

cur=PRE/'round16_dataset_expansion/data/curation'
ex=json.loads((cur/'donor_exclusions.json').read_text('utf-8'));known_bad=set(ex['blocking_donor_candidate_ids']);frags=[norm(s) for s in ex['blocked_title_fragments']]
rejection_reasons=defaultdict(list)
for p in cur.glob('*rejections.jsonl'):
    files[str(p.relative_to(PRE))]=sha(p)
    for r in readl(p):
        cid=r.get('candidate_id')
        if cid:rejection_reasons[cid].append(r.get('reason',r.get('rationale','previous source curation rejection')))
pool_path=cur/'available_source_pool.jsonl';files[str(pool_path.relative_to(PRE))]=sha(pool_path)
candidates=[];stages=Counter();types=defaultdict(Counter)
for lineno,r in enumerate(readl(pool_path),1):
    # Deliberately do not export original_question/original_answer_quote/content.
    title=norm(r['source_title']);reasons=[]
    if title in used_titles or r['candidate_id'] in used_question_ids:reasons.append('previously_used_source_or_target')
    if r['candidate_id'] in known_bad:reasons.append('recorded_donor_exclusion')
    matched=[f for f in frags if f in title]
    if matched:reasons.append('recorded_used_event_fragment')
    if r['candidate_id'] in rejection_reasons:reasons.append('prior_curation_rejection_requires_resolution')
    status='metadata_candidate_needs_semantic_review' if not reasons else 'excluded_or_quarantined'
    stages[status]+=1;types[status][r['official_information_type']]+=1
    candidates.append({'candidate_id':r['candidate_id'],'status':status,'source_title':r['source_title'],
      'source_url':r['source_url'],'revision_id':r['revision_id'],'source_content_sha256':r['source_content_sha256'],
      'original_information_type':r['official_information_type'],'original_domain':r['official_category'],
      'source_snapshot_path':str(pool_path.resolve()),'source_snapshot_line_1based':lineno,
      'exclusion_or_quarantine_reasons':reasons,'recorded_event_matches':matched,
      'prior_source_rejection_reasons':rejection_reasons.get(r['candidate_id'],[]),
      'not_reviewed':['full-text old subject/alias mentions','same specific event across different article titles','partial evidence indirect leakage','actual requested attribute type','new-pool mutual event links'],
      'answer_or_risk_label_inspected':False})
savel('ragognize_remaining_metadata_index.jsonl',candidates)

# Build the union of actually used RAGTruth source IDs from all news rounds.
old_paths=[PRE/f'data/processed/{s}.jsonl' for s in ('train','val','test')]
old_paths += [PRE/'round2/data/news_large/labeled.jsonl',PRE/'round2/data/news_confirmation/rows.jsonl',
 PRE/'round3/data/rows.jsonl',PRE/'round4/data/rows.jsonl',PRE/'round4/confirmation/data/rows.jsonl']
old_source_ids=set();old_counts={}
for p in old_paths:
    if not p.exists():continue
    # Only identifiers are retained; old answer and label values are unused.
    ids={r['source_id'] for r in readl(p)};old_source_ids.update(ids)
    old_counts[str(p.relative_to(PRE))]=len(ids);files[str(p.relative_to(PRE))]=sha(p)
raw=PRE/'data/raw';source_path=raw/'source_info.jsonl';response_path=raw/'response.jsonl'
sources={r['source_id']:r for r in readl(source_path)}
qa={sid:r for sid,r in sources.items() if r['task_type']=='QA'}
response_meta=defaultdict(list);response_total=Counter()
for r in readl(response_path):
    # No quality/response/labels access. ID/model/temperature/split metadata only.
    response_total[sources[r['source_id']]['task_type']]+=1
    if r['source_id'] in qa:
        response_meta[r['source_id']].append({k:r[k] for k in ('id','model','temperature','split')})
qa_indices=[];native_counts=Counter();question_hashes=defaultdict(list);passage_hashes=defaultdict(list)
for sid,s in qa.items():
    rm=response_meta[sid];assert len(rm)==6 and len({r['model'] for r in rm})==6
    official={r['split'] for r in rm};assert len(official)==1
    native=[r for r in rm if r['model']=='mistral-7B-instruct'];assert len(native)==1
    split=next(iter(official));native_counts[split]+=1
    info=s['source_info'];qh=h(norm(info['question']));ph=h(norm(info['passages']))
    question_hashes[qh].append(sid);passage_hashes[ph].append(sid)
    qa_indices.append({'source_id':sid,'task_type':'QA','source':s['source'],
      'question_text_sha256':h(info['question']),'normalized_question_sha256':qh,
      'retrieved_passages_sha256':h(info['passages']),'normalized_retrieved_passages_sha256':ph,
      'released_prompt_sha256':h(s['prompt']),'official_split':split,
      'previously_used_source_id':sid in old_source_ids,'model_response_metadata':rm,
      'native_mistral_response_id':native[0]['id'],
      'proposed_role':'sealed_official_source_test_pending_event_audit' if split=='test' else 'development_or_calibration_pending_group_audit',
      'checkpoint_version_status':'unresolved_in_release','original_token_ids_available':False,
      'risk_labels_or_answer_text_inspected':False})
savel('ragtruth_qa_metadata_index.jsonl',sorted(qa_indices,key=lambda x:str(x['source_id'])))
dups={'same_normalized_question':[v for v in question_hashes.values() if len(v)>1],
      'same_entire_retrieved_passages':[v for v in passage_hashes.values() if len(v)>1]}
save('ragtruth_qa_duplicate_metadata.json',dups)
for p in (source_path,response_path):files[str(p.relative_to(PRE))]=sha(p)
summary={'status':'source_only_feasibility_not_frozen_final_test','original_and_current_input_splits':split_counts,
 'ragognize':{'source_pool':len(candidates),'stages':dict(stages),'official_types_by_stage':{k:dict(v) for k,v in types.items()},
  'previously_used_title_or_target':sum('previously_used_source_or_target' in r['exclusion_or_quarantine_reasons'] for r in candidates),
  'note':'Remaining metadata candidates are not a certified count of independent questions; all full-text/alias/event and fact-removal reviews remain.'},
 'ragtruth':{'local_source_counts':dict(Counter(r['task_type'] for r in sources.values())),
  'local_response_counts_without_label_inspection':dict(response_total),'old_news_unique_source_ids':len(old_source_ids),
  'old_news_source_counts_by_input':old_counts,'used_source_tasks':dict(Counter(sources[s]['task_type'] for s in old_source_ids)),
  'qa_source_id_overlap_with_old_news':len(set(qa)&old_source_ids),'qa_sources':len(qa),
  'qa_responses_all_models':sum(len(v) for v in response_meta.values()),'native_mistral_official_split_counts':dict(native_counts),
  'same_normalized_question_components':len(dups['same_normalized_question']),
  'same_entire_passage_components':len(dups['same_entire_retrieved_passages']),
  'qualification':'QA task has no demonstrated previous feature/development use; original inspect_data.py counted all task label aggregates, so do not claim nobody ever touched the raw release. No candidate QA answer text or labels inspected during this plan.'},
 'no_new_generation':True,'no_gpu':True,'no_downloads':True,'new_answer_text_or_risk_labels_inspected':False,
 'source_files_sha256':files,'script_sha256':sha(Path(__file__))}
save('source_inventory_summary.json',summary)
print(json.dumps({k:summary[k] for k in ('ragognize','ragtruth')},ensure_ascii=False,indent=2))
