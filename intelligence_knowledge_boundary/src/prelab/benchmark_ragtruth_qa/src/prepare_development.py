"""Export released Llama2-chat QA fit/calibration answers; keep test sealed.

No source selection uses quality, answer contents, spans, or detector scores.
Only admitted official-training source lines are parsed from response.jsonl.
"""
from pathlib import Path
import json,re,hashlib
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[1];PRE=ROOT.parent;OUT=ROOT/'data'
SEED='20260911';MODEL='llama-2-7b-chat'
def readl(p):
    with p.open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def save(name,r):(OUT/name).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')
def savel(name,rs):(OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),'utf-8')

src={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl') if r['task_type']=='QA'}
meta={r['source_id']:r for r in readl(PRE/'round20_localization_optimization/research/ragtruth_qa_metadata_index.jsonl')}
mat=json.loads((OUT/'material_overlap_audit.json').read_text('utf-8'))['edges']
sem=json.loads((OUT/'question_link_candidates.json').read_text('utf-8'))['pairs']

# Explicit assistant judgments after reading the 140 nominated question pairs.
# Broad procedural vocabulary alone is not an entity/event match.
reviewed=set([i for i,r in enumerate(sem) if len(r['shared_tokens'])>=2 and r['jaccard']>=.5])
names=set('apple itunes facebook android google snapchat roth outlook xbox supreme medicare germany france india texas gatos youtube'.split())
reviewed.update(i for i,r in enumerate(sem) if set(r['shared_tokens'])&names or (len(r['shared_tokens'])>=2 and .4<=r['jaccard']<.5))
accepted={18,59,90,91,100,101,102,104,196,202,204,216,252,253,299,305,311,312,339,348,354,369,
390,391,392,422,456,457,460,461,462,477,478,479,492,499,514,547,551,552,553,554,557,574,575,597,598,
599,600,602,604,606,615,633,634,636,640,648,686,692,694,700,723,727,729,731,750,751,757,758,759,781,787,
830,852,853,857,858}
assert accepted<=reviewed
semreviews=[]
for i in sorted(reviewed):
    r=sem[i]
    why='Same or plausibly related named platform/institution, narrow subject, or recipe family; conservatively keep together without inspecting answers.' if i in accepted else 'Shared words refer to a broad procedure, different target object, or different use of a name; this question-only edge does not establish a shared source/entity/event.'
    semreviews.append({'pair_index':i,'a':r['a'],'b':r['b'],'decision':'group_conservatively' if i in accepted else 'do_not_group_from_this_edge','rationale':why})
save('semantic_group_review.json',{'reviewer':'data_build source-question-only assistant review','reviewed_pair_count':len(reviewed),'candidate_pairs_total':len(sem),
 'scope':'Read all 20 material-link question pairs and selected lexical candidates with >=2 shared informative words at Jaccard>=.4, plus named-platform/institution/place keyword candidates. This is not exhaustive all-entity or researcher human verification.',
 'reviews':semreviews,'unreviewed_lexical_pairs_status':'Not certified as independent; mostly single generic word candidates. A separate audit remains necessary before any claim of strict new-entity/event generalization.'})

parent={sid:sid for sid in src}
def find(x):
    if parent[x]!=x:parent[x]=find(parent[x])
    return parent[x]
def union(a,b):
    a,b=find(a),find(b)
    if a!=b:parent[max(a,b)]=min(a,b)
old_connected=set()
for e in mat:
    if e['a'].startswith('qa:') and e['b'].startswith('qa:'):union(e['a'][3:],e['b'][3:])
    else:old_connected.add((e['a'] if e['a'].startswith('qa:') else e['b'])[3:])
for i in accepted:union(sem[i]['a'],sem[i]['b'])
groups=defaultdict(list)
for sid in src:groups[find(sid)].append(sid)
gids={sid:'rtqa_group_'+h('|'.join(sorted(members)))[:16] for members in groups.values() for sid in members}
roles={};sealed_groups=[];oldgroups=[];devgroups=[]
for leader,members in groups.items():
    members=sorted(members)
    if any(meta[s]['official_split']=='test' for s in members):
        sealed_groups.append(members)
        for s in members:roles[s]='sealed_official_test' if meta[s]['official_split']=='test' else 'withheld_train_linked_to_test'
    elif any(s in old_connected for s in members):
        oldgroups.append(members)
        for s in members:roles[s]='withheld_old_material_link'
    else:devgroups.append(members)
devgroups.sort(key=lambda ms:h(SEED+'|'+gids[ms[0]]))
target_cal=round(sum(map(len,devgroups))*.2);ncal=0
for ms in devgroups:
    role='calibration' if abs(ncal+len(ms)-target_cal)<abs(ncal-target_cal) else 'fit'
    if role=='calibration':ncal+=len(ms)
    for s in ms:roles[s]=role
index=[]
for sid in sorted(src):
    r=meta[sid];native=next(x for x in r['model_response_metadata'] if x['model']==MODEL)
    index.append({'source_id':sid,'group_id':gids[sid],'partition':roles[sid],'official_split':r['official_split'],
     'native_model':MODEL,'response_id':native['id'],'all_model_response_ids':[x['id'] for x in r['model_response_metadata']],
     'source_prompt_sha256':r['released_prompt_sha256'],'retrieved_passages_sha256':r['retrieved_passages_sha256']})
savel('source_index.jsonl',index)
savel('sealed_source_index.jsonl',[r for r in index if r['partition'].startswith('sealed') or r['partition'].startswith('withheld')])
allowed={s for s in src if roles[s] in ('fit','calibration')}

# Locked quality policy is official RAGTruth quality==good, retaining ALL labels.
# No label_type/implicit_true/due_to_null or class-balance filtering.
exported=[];excluded=[];quality_counts=Counter();span_counts=Counter();alignment_issues=[]
source_re=re.compile(r'"source_id"\s*:\s*"([^"\\]+)"')
native_model_re=re.compile(r'"model"\s*:\s*"llama-2-7b-chat"')
parsed_source_ids=set()
with (PRE/'data/raw/response.jsonl').open(encoding='utf-8') as f:
    for line in f:
        sid_match=source_re.search(line)
        if not sid_match or sid_match.group(1) not in allowed or not native_model_re.search(line):continue
        r=json.loads(line);sid=r['source_id']
        assert r['model']==MODEL and r['split']=='train' and sid not in parsed_source_ids
        parsed_source_ids.add(sid);s=src[sid]
        quality_counts[r['quality']]+=1
        row={'source_id':sid,'group_id':gids[sid],'partition':roles[sid],'official_split':r['split'],
         'response_id':r['id'],'model':r['model'],'temperature':r['temperature'],'quality':r['quality'],
         'question':s['source_info']['question'],'retrieved_passages':s['source_info']['passages'],
         'released_prompt':s['prompt'],'original_response':r['response'],'labels':r['labels'],
         'annotation_origin':'RAGTruth released human span annotation','label_offsets':'original response character offsets, end exclusive; no text normalization',
         'answer_sha256':h(r['response']),'prompt_sha256':h(s['prompt'])}
        issues=[]
        for k,lab in enumerate(r['labels']):
            span_counts[lab['label_type']]+=1
            if not (isinstance(lab['start'],int) and isinstance(lab['end'],int) and 0<=lab['start']<lab['end']<=len(r['response'])):issues.append({'label_index':k,'reason':'invalid_character_interval'})
            elif r['response'][lab['start']:lab['end']]!=lab['text']:issues.append({'label_index':k,'reason':'original_text_offset_mismatch'})
        row['span_integrity_issues']=issues
        if issues:alignment_issues.append({'source_id':sid,'response_id':r['id'],'issues':issues})
        row['official_quality_eligible']=r['quality']=='good'
        if r['quality']=='good':exported.append(row)
        else:
            row['exclusion_reason']='official_quality_not_good';excluded.append(row)
assert parsed_source_ids==allowed,(len(parsed_source_ids),len(allowed))
for part in ['fit','calibration']:
    savel(part+'.jsonl',sorted([r for r in exported if r['partition']==part],key=lambda r:r['source_id']))
savel('quality_excluded_development.jsonl',sorted(excluded,key=lambda r:r['source_id']))
save('span_integrity_review.json',{'issues':alignment_issues,'policy':'Preserve original labels and offsets unchanged; any invalid geometry must be resolved or marked unscorable before training, never silently relabeled.'})
paths=[OUT/n for n in ['source_index.jsonl','sealed_source_index.jsonl','fit.jsonl','calibration.jsonl','quality_excluded_development.jsonl','semantic_group_review.json','span_integrity_review.json']]
manifest={'status':'development_data_prepared_test_sealed_not_model_trace_freeze','seed':SEED,'native_response_model':MODEL,
 'source_counts_by_partition':dict(Counter(roles.values())),'group_counts_by_partition':{p:len({gids[s] for s in src if roles[s]==p}) for p in sorted(set(roles.values()))},
 'eligible_answers_by_partition':dict(Counter(r['partition'] for r in exported)),'quality_excluded_answers_by_partition':dict(Counter(r['partition'] for r in excluded)),
 'development_quality_counts':dict(quality_counts),'development_span_types_all_quality_rows':dict(span_counts),'span_integrity_issue_rows':len(alignment_issues),
 'quality_policy':'Official quality==good only. Preserve all released labels, including implicit_true/due_to_null flags, original start/end/text/meta. Do not filter by label type or prediction. Non-good rows retained separately for audit.',
 'grouping_policy':f'All {len(mat)} material overlap edges and {len(accepted)} explicitly reviewed narrow-question/platform/possible-recipe-family links grouped. Every group touching official test stays sealed: official train members are withheld, not moved to fitting. All six model answers follow source/group assignment.',
 'source_grouping_limitations':['Not an exhaustive entity-alias/event audit; no claim of new-entity/new-event test.','Weather-template overlaps are conservatively grouped even across cities.','Remaining generic lexical candidate links were not all manually reviewed.'],
 'exact_original_trace_claimed':False,'intended_replay':'Fixed disclosed Llama2-7B-chat family checkpoint, explicit original RAGTruth wrapper, NF4 teacher-forced reconstruction; original token trajectory unavailable.',
 'development_exporter_parsed_official_test_rows':False,'development_exporter_parsed_withheld_train_rows':False,
 'official_test_response_text_or_labels_inspected_or_exported':False,
 'source_files_sha256':{str(p.relative_to(PRE)):sha(p) for p in [PRE/'data/raw/source_info.jsonl',PRE/'data/raw/response.jsonl',PRE/'round20_localization_optimization/research/ragtruth_qa_metadata_index.jsonl']},
 'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'code_sha256':sha(Path(__file__))}
save('development_manifest.json',manifest)
print(json.dumps({k:manifest[k] for k in ['source_counts_by_partition','group_counts_by_partition','eligible_answers_by_partition','quality_excluded_answers_by_partition','development_quality_counts','development_span_types_all_quality_rows','span_integrity_issue_rows']},ensure_ascii=False,indent=2))
