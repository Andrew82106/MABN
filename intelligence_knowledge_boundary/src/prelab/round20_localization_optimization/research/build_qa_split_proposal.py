"""Metadata-only source split proposal; no answer/quality/span label access."""
from pathlib import Path
import json, hashlib, re, unicodedata
from collections import Counter, defaultdict

HERE=Path(__file__).resolve().parent
PRE=HERE.parents[1]
SEED='20260911'
def readl(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def norm(s):return re.sub(r'[^\w]+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def sh(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest()

old_questions=defaultdict(list)
old_files=[PRE/'round2/data/trivia_questions.jsonl',PRE/'round2/data/rag_questions.jsonl']
for p in old_files:
    for r in readl(p):
        qs=[r['question']] if 'question' in r else [x['question'] for x in r['qas']]
        for q in qs:old_questions[sh(norm(q))].append({'file':str(p.relative_to(PRE)),'id':r['id']})
source_index=HERE/'ragtruth_qa_metadata_index.jsonl'
rows=list(readl(source_index))
overlaps=[{'source_id':r['source_id'],'old_question_references':old_questions[r['normalized_question_sha256']]} for r in rows if r['normalized_question_sha256'] in old_questions]
blocked={r['source_id'] for r in overlaps}
eligible_train=sorted([r for r in rows if r['official_split']=='train' and r['source_id'] not in blocked],key=lambda r:sh(SEED+'|qa-source|'+str(r['source_id'])))
ncal=round(len(eligible_train)*.2)
cal={r['source_id'] for r in eligible_train[:ncal]}
out=[]
for r in rows:
    sid=r['source_id']
    partition='quarantine_previous_exact_question' if sid in blocked else ('sealed_test' if r['official_split']=='test' else ('calibration' if sid in cal else 'fit'))
    out.append({'source_id':sid,'group_id':'ragtruth_qa_source_'+str(sid),'proposed_partition':partition,
      'official_split':r['official_split'],'native_mistral_response_id':r['native_mistral_response_id'],
      'all_response_ids':[x['id'] for x in r['model_response_metadata']],
      'all_models':[x['model'] for x in r['model_response_metadata']],
      'question_text_sha256':r['question_text_sha256'],'retrieved_passages_sha256':r['retrieved_passages_sha256'],
      'released_prompt_sha256':r['released_prompt_sha256'],'status':'proposal_pending_entity_event_audit_and_model_replay_decision',
      'risk_or_response_content_used_for_partition':False})
outpath=HERE/'ragtruth_qa_split_proposal.jsonl'
outpath.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),'utf-8')
manifest={'status':'proposal_not_final_data_freeze','seed':SEED,'rule':'Keep official test sealed; hash-ranked 20% official train for calibration; no labels or answer text used; all six model responses follow source partition.',
 'sources_by_proposed_partition':dict(Counter(r['proposed_partition'] for r in out)),
 'all_model_responses_by_proposed_partition':dict(Counter(r['proposed_partition'] for r in out for _ in r['all_response_ids'])),
 'native_mistral_responses_by_proposed_partition':dict(Counter(r['proposed_partition'] for r in out)),
 'old_exact_question_overlap':overlaps,'old_unique_question_hash_count':len(old_questions),
 'remaining_required_audits':['Entity/alias and specific-event links across QA sources and all old news/TriviaQA/SQuAD/Hotpot/RAGognize inputs; exact source IDs and whole-text hashes do not establish this.',
  'Repeated individual retrieval passages and partial-article overlap: the previous index only checked whole-context hashes.',
  'Confirm original Mistral checkpoint/tokenizer/template or explicitly preregister a reconstruction assumption.',
  'Fix label types/implicit_true/refusal/quality and character-to-token scoring policy using training material only, before any test-label analysis.'],
 'read_no_new_answers_or_labels':True,
 'inputs_sha256':{str(p.relative_to(PRE)):filehash(p) for p in old_files+[source_index]},
 'proposal_sha256':filehash(outpath),'script_sha256':filehash(Path(__file__))}
(HERE/'ragtruth_qa_split_proposal_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({k:manifest[k] for k in ['sources_by_proposed_partition','all_model_responses_by_proposed_partition','old_unique_question_hash_count','old_exact_question_overlap']},ensure_ascii=False,indent=2))
