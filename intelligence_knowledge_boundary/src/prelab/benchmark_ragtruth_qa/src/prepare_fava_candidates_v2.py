"""Factual-only silver FAVA staging with source-only conservative quarantine.

Never opens raw/response.jsonl, any QA response/label file, or a tokenizer.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import importlib.util
import json
import re
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'auxiliary_fava_v2'
FAVA=ROOT/'additional_data/fava_training'
CENSUS=ROOT/'additional_data/fava_training_full_alignment_v1'
HUMAN=ROOT/'auxiliary_human_v1'
RAW=ROOT.parent/'data/raw/source_info.jsonl'
spec=importlib.util.spec_from_file_location('fava_fixed_parser',FAVA/'inspect_training.py')
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
spec=importlib.util.spec_from_file_location('existing_material_rules',Path(__file__).with_name('prepare_auxiliary_human.py'))
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
FACT={'entity','relation','invented','contradictory'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save_rows(name,data):
    with (OUT/name).open('w',encoding='utf-8') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')


class Union:
    def __init__(self,ids):self.parent={x:x for x in ids}
    def find(self,x):
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def groups(self):
        out=defaultdict(list)
        for x in self.parent:out[self.find(x)].append(x)
        return {k:sorted(v) for k,v in out.items()}


def protocol():
    return {'version':'fava-factual-only-silver-candidate-data-v2',
        'revision_from_v1':'Only ignore empty/whitespace-only material parts for exact matching and shared-reference grouping. Preserve all original input sections, the same7483 fixed candidates, original spans, nonempty exact matches,20word rules and deduplication. Keep v1 artifacts unchanged and report actual membership/group differences.',
        'fixed_candidates':'Exactly7483 original raw indices classified factual_only by frozen full character census; no subjective/unverifiable mixed rows, no no-span rows, no parsing failures. No selection from model scores or QA calibration outcomes.',
        'source':'Existing fixed official FAVA training.json revisionf4e40415d525b18bcb49ba241f26fd8e12eeb606; no new download.',
        'input':'Preserve five original numbered Reference sections as one exact original substring, original corrupted answer, question empty. Never use edited projection, completion, type tags, or corrections as model input.',
        'released_prompt':'Metadata only: original checking-prompt prefix ending at Text/[Text] answer boundary. It is not a factual user question or native generator trajectory. Evidence range points to its original numbered references. Semantic encoder will read references+SEP+emptyquestion+SEP+answer.',
        'labels':'Preserve original exact start/end/text/type and explicit_delete flags for the four factual candidate types. These are synthetic silver positives; unmarked positions may be silver negatives under the released annotation, NOT verified true tokens or human gold. No new truth verification is claimed.',
        'identity_sources':'Do not open raw/response.jsonl or QA answer/label files. Use source_info2965 source documents, QA source_index989 identities, and source IDs only from frozen human auxiliary candidate/quarantine indexes.',
        'conservative_RT_block':'NonQA1976=confirmed_train1676(1614 auxiliary candidate sources+62 known-train quarantined sources)+unconfirmed300. Block all QA non-fit identities, every unconfirmed nonQA source and original62 quarantine sources, then propagate through frozen existing material/business overlap edges. Do not claim the300 are individually verified official-test identities.',
        'material_match':'Only evidence text. Existing Unicode-word norm re.findall(r"\\w+",text.casefold()); match any20 consecutive normalized words, separately within each evidence part. Also quarantine exact nonempty/non-whitespace raw evidence-part SHA matches, even short parts. Empty or whitespace-only parts convey no source identity and are excluded from exact matching. QA questions and schema boilerplate are excluded using unchanged existing evidence_parts.',
        'FAVA_groups':'Before filtering, connect all7483 fixed candidates sharing any exact original nonempty/non-whitespace reference-block SHA. Empty/whitespace-only blocks remain verbatim input but never form material links. Any component touching blocked RT material is wholly quarantined. Distractor blocks can create links; group IDs are conservative material components, not independent entities/events/articles.',
        'duplicates':'After quarantine, deduplicate only identical rendered references, original answer, and complete typed span list; retain smallest raw index, preserve duplicate index. Different reference orders or labels remain distinct.',
        'output_schema':'response_id=fava_train_{raw_index}; source_id=exact unordered reference-multiset SHA; group_id=shared-reference connected component; task_type=FAVA_synthetic; question empty; original hashes and silver provenance retained.',
        'operations':'Only candidate data/manifest/isolation/group/duplicate ledgers; no tokenizer, GPU, model or training. Preserve original census/human/QA files and root token-protocol files.',
        'limits':'Exact/hash/20word isolation cannot detect all paraphrases or recover article identity. Silver negatives are noisy auxiliary supervision. Not public-QA evaluation and not a new held-out test.'}


def source_hashes():
    paths=[Path(__file__),Path(parser.__file__),Path(rules.__file__),RAW,ROOT/'data/source_index.jsonl',
        FAVA/'training.json',CENSUS/'rows.jsonl',CENSUS/'complete.json',CENSUS/'AUDIT.json',
        HUMAN/'manifest.json',HUMAN/'candidate_fit.jsonl',HUMAN/'quarantined_source_index.jsonl',HUMAN/'overlap_edge_index.jsonl',
        Path(__file__).with_name('prepare_fava_candidates.py'),ROOT/'auxiliary_fava_v1/manifest.json',
        ROOT/'auxiliary_fava_v1/complete.json',ROOT/'auxiliary_fava_v1/candidate_fit.jsonl',
        ROOT/'auxiliary_fava_v1/fixed_candidate_index.jsonl',ROOT/'auxiliary_fava_v1/candidate_material_group_index.jsonl']
    return {str(p.resolve()):sha(p) for p in paths}


def design():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'data_design_freeze.json').exists()
    save('DATA_PROTOCOL.json',protocol())
    save('data_design_freeze.json',{'protocol_sha256':sha(OUT/'DATA_PROTOCOL.json'),
        'source_sha256':source_hashes(),'tokenization_started':False,'trained':False,'GPU_used':False})
    print('FAVA_CANDIDATE_DATA_PROTOCOL_FROZEN',flush=True)


def check():
    f=read(OUT/'data_design_freeze.json');assert read(OUT/'DATA_PROTOCOL.json')==protocol()
    assert f['protocol_sha256']==sha(OUT/'DATA_PROTOCOL.json') and f['source_sha256']==source_hashes()
    return f


def identity(sources):
    manifest=read(HUMAN/'manifest.json')
    for name in ('candidate_fit.jsonl','quarantined_source_index.jsonl','overlap_edge_index.jsonl'):
        assert sha(HUMAN/name)==manifest['artifacts_sha256'][name]
    assert manifest['sources_sha256'][str(RAW.resolve())]==sha(RAW)
    # This pre-existing file contains TRAIN-only human auxiliary candidates.
    # Extract only its first source_id field; do not parse any answer or label.
    prefix=re.compile(r'^\{"source_id": "([^"\\]+)"')
    train=set()
    with (HUMAN/'candidate_fit.jsonl').open(encoding='utf-8') as f:
        for line in f:
            m=prefix.match(line);assert m;train.add(m.group(1))
    quarantine={r['source_id'] for r in rows(HUMAN/'quarantined_source_index.jsonl')}
    assert len(train)==1614 and len(quarantine)==62 and not train&quarantine
    qa={r['source_id']:r for r in rows(ROOT/'data/source_index.jsonl')}
    nonqa={sid for sid,r in sources.items() if r['task_type']!='QA'}
    assert len(sources)==2965 and len(qa)==989 and len(nonqa)==1976
    assert set(qa)==set(sources)-nonqa and train|quarantine<=nonqa
    confirmed=train|quarantine;unconfirmed=nonqa-confirmed
    assert len(confirmed)==1676 and len(unconfirmed)==300
    reasons=defaultdict(set)
    for sid in qa:
        if qa[sid]['partition']!='fit':reasons[sid].add('QA_nonfit_source_identity')
    for sid in unconfirmed:reasons[sid].add('nonQA_training_identity_unconfirmed_conservatively_blocked')
    for sid in quarantine:reasons[sid].add('existing_known_train_material_quarantine')
    uf=Union(sources)
    for edge in rows(HUMAN/'overlap_edge_index.jsonl'):uf.join(edge['source_a'],edge['source_b'])
    groups=uf.groups();gid={sid:'rt_material_'+digest('|'.join(members)) for members in groups.values() for sid in members}
    blocked_groups={gid[sid] for sid in reasons}
    blocked={sid for sid in sources if gid[sid] in blocked_groups}
    identity_rows=[]
    for sid in sorted(sources):
        r={'source_id':sid,'task_type':sources[sid]['task_type'],'material_group_id':gid[sid],
           'blocked_after_existing_edge_propagation':sid in blocked,'direct_block_reasons':sorted(reasons[sid])}
        if sid in qa:r.update(identity_kind='existing_QA_index',partition=qa[sid]['partition'],official_split=qa[sid]['official_split'])
        else:r.update(identity_kind='confirmed_official_train' if sid in confirmed else 'unconfirmed_conservatively_blocked',
                      official_split='train' if sid in confirmed else None)
        identity_rows.append(r)
    save_rows('rt_source_identity_index.jsonl',identity_rows)
    save_rows('rt_material_group_index.jsonl',[{'group_id':gid[members[0]],'source_ids':members,
        'blocked':gid[members[0]] in blocked_groups,'directly_blocked_members':{sid:sorted(reasons[sid]) for sid in members if reasons[sid]}}
        for members in sorted(groups.values())])
    return blocked,gid,{'all_sources':2965,'QA_sources':989,'nonQA_sources':1976,
        'nonQA_confirmed_train_sources':1676,'nonQA_unconfirmed_conservatively_blocked':300,
        'original_nonQA_train_quarantine':62,'QA_nonfit_sources':sum(x['partition']!='fit' for x in qa.values()),
        'blocked_after_material_edge_propagation':len(blocked),'test_answers_or_gold_opened':False,
        'raw_response_file_opened':False,'all_test_source_identities_individually_recovered':False}


def run():
    frozen=check();assert not (OUT/'data_started.json').exists()
    save('data_started.json',{'time':time.time(),'design_freeze_sha256':sha(OUT/'data_design_freeze.json')})
    tick=time.perf_counter()
    done=read(CENSUS/'complete.json');assert read(CENSUS/'AUDIT.json')['passed']
    for n,h in done['files_sha256'].items():assert sha(CENSUS/n)==h
    selected=[r for r in rows(CENSUS/'rows.jsonl') if r['exact_character_alignment'] and r['alignment_class']=='factual_only']
    assert len(selected)==7483;raw=read(FAVA/'training.json')
    sources={r['source_id']:r for r in rows(RAW)}
    blocked,rt_group,identity_report=identity(sources)
    records={};ref_texts={};ref_members=defaultdict(list)
    marker=re.compile(r'\nPlease identify[^\n]*\n(?:Text: |\[Text\] )')
    for checked in selected:
        i=checked['raw_index'];original=raw[i]
        assert digest(original['prompt'])==checked['prompt_sha256'] and digest(original['completion'])==checked['completion_sha256']
        refs,answer=parser.prompt_parts(original['prompt']);assert len(refs)==5
        assert digest(answer)==checked['answer_sha256']
        labels=checked['aligned_synthetic_spans'];assert labels and {s['type'] for s in labels}<=FACT
        assert all(answer[s['start']:s['end']]==s['text'] for s in labels)
        mm=list(marker.finditer(original['prompt']));assert len(mm)==1;m=mm[0]
        first=re.search(r'(?:^|\n)Reference \[1\]: ',original['prompt'][:m.start()]);assert first
        begin=first.start()+int(original['prompt'][first.start():first.start()+1]=='\n')
        evidence=original['prompt'][begin:m.start()];released=original['prompt'][:m.end()]
        assert released[begin:m.start()]==evidence
        assert len(re.findall(r'(?:^|\n)Reference \[\d+\]: ',evidence))==5
        rid=f'fava_train_{i}'
        rec={'raw_index':i,'response_id':rid,'source_id':checked['reference_set_group_sha256'],
            'partition':'auxiliary_candidate_fit','official_split':'train','task_type':'FAVA_synthetic',
            'source_dataset':'fava-uw/fava-data','source_revision':'f4e40415d525b18bcb49ba241f26fd8e12eeb606',
            'question':'','retrieved_passages':evidence,'original_response':answer,'labels':labels,
            'answer_sha256':digest(answer),'prompt_sha256':checked['prompt_sha256'],
            'original_prompt_sha256':checked['prompt_sha256'],'completion_sha256':checked['completion_sha256'],
            'released_prompt':released,'released_prompt_sha256':digest(released),
            'evidence_original_prompt_range':[begin,m.start()],
            'released_prompt_semantics':'Original FAVA checking-prompt prefix, not original factual user question or native generation trace.',
            'reference_text_sha256':checked['reference_text_sha256'],
            'annotation_origin':'FAVA released synthetic markup; exact character-aligned silver supervision',
            'synthetic_not_human_gold':True,'unmarked_positions_silver_not_verified_negative':True,
            'edited_projection_used':False,'new_labels_generated':False,'currently_used_in_training':False}
        records[i]=rec
        for ordinal,text in enumerate(refs,1):
            h=digest(text);assert h==rec['reference_text_sha256'][ordinal-1]
            if h in ref_texts:assert ref_texts[h]==text
            else:ref_texts[h]=text
            ref_members[h].append((i,ordinal))
    uf=Union(records)
    for reference_hash,members in ref_members.items():
        if not ref_texts[reference_hash].strip():continue
        for i,_ in members[1:]:uf.join(members[0][0],i)
    groups=uf.groups();group_id={i:'fava_material_'+digest('|'.join(f'fava_train_{j}' for j in members))
        for members in groups.values() for i in members}
    # One witness per shingle is enough because the existing source-only graph
    # already connects all RT sources sharing that same20-word shingle. Verify it.
    grams={};exact={}
    for sid,source in sources.items():
        for part_index,part in enumerate(rules.evidence_parts(source)):
            h=digest(part)
            if part.strip():exact.setdefault(h,[]).append((sid,part_index))
            words=rules.norm(part)
            for pos in range(len(words)-19):
                key=digest(' '.join(words[pos:pos+20]));witness=grams.setdefault(key,(sid,part_index,pos))
                assert rt_group[witness[0]]==rt_group[sid]
    print('FAVA_RT_MATERIAL_INDEX_READY',len(grams),'unique_FAVA_reference_blocks',len(ref_texts),flush=True)
    matches=[];hits_by_ref=defaultdict(list)
    for number,(h,text) in enumerate(sorted(ref_texts.items()),1):
        matched={}
        for sid,parti in (exact.get(h,[]) if text.strip() else []):
            key=rt_group[sid];matched[key]={'rt_source_id_witness':sid,'rt_part_index':parti,
                'rt_material_group_id':key,'blocked':sid in blocked,'reason':'exact_evidence_part_SHA',
                'reference_text_sha256':h,'shared_20word_hashes':0}
        words=rules.norm(text);seen=set()
        for pos in range(len(words)-19):
            key=digest(' '.join(words[pos:pos+20]))
            if key in seen:continue
            seen.add(key);witness=grams.get(key)
            if witness is None:continue
            sid,parti,rtpos=witness;gid=rt_group[sid]
            if gid not in matched:
                matched[gid]={'rt_source_id_witness':sid,'rt_part_index':parti,'rt_word_start':rtpos,
                    'fava_word_start':pos,'shared20_sha256_witness':key,'rt_material_group_id':gid,
                    'blocked':sid in blocked,'reason':'shared_consecutive20_normalized_words',
                    'reference_text_sha256':h,'shared_20word_hashes':0}
            matched[gid]['shared_20word_hashes']+=1
        for match in sorted(matched.values(),key=lambda r:r['rt_material_group_id']):
            match['fava_members_raw_index_and_reference_ordinal']=ref_members[h]
            hits_by_ref[h].append(len(matches));matches.append(match)
        if number%5000==0:print('FAVA_MATERIAL_MATCH',number,len(ref_texts),flush=True)
    direct=set();witness_by_candidate=defaultdict(list)
    for match_index,m in enumerate(matches):
        if m['blocked']:
            for i,ordinal in m['fava_members_raw_index_and_reference_ordinal']:
                direct.add(i);witness_by_candidate[i].append({'match_index':match_index,'reference_ordinal':ordinal})
    bad_groups={group_id[i] for i in direct}
    quarantined={i for i in records if group_id[i] in bad_groups}
    group_rows=[]
    for members in sorted(groups.values()):
        gid=group_id[members[0]]
        group_rows.append({'group_id':gid,'raw_indices':members,'response_ids':[records[i]['response_id'] for i in members],
            'candidate_count':len(members),'quarantined':gid in bad_groups,
            'directly_blocked_raw_indices':sorted(set(members)&direct),'may_connect_via_distractor':True})
    save_rows('candidate_material_group_index.jsonl',group_rows)
    save_rows('shared_reference_block_index.jsonl',[{'reference_text_sha256':h,
        'members_raw_index_and_reference_ordinal':members,'may_be_shared_distractor':True}
        for h,members in sorted(ref_members.items()) if ref_texts[h].strip() and len({i for i,o in members})>1])
    save_rows('material_match_index.jsonl',matches)
    save_rows('quarantined_candidate_index.jsonl',[{'raw_index':i,'response_id':records[i]['response_id'],
        'source_id':records[i]['source_id'],'group_id':group_id[i],
        'reason':'shared_reference_component_touches_blocked_RT_material',
        'direct_match':i in direct,'direct_match_witnesses':witness_by_candidate[i],
        'component_direct_blocked_raw_indices':sorted(j for j in groups[uf.find(i)] if j in direct)} for i in sorted(quarantined)])
    retained=[];duplicates=[];seen={}
    for i,rec in sorted(records.items()):
        if i in quarantined:continue
        key=digest(canonical([rec['retrieved_passages'],rec['original_response'],rec['labels']]))
        if key in seen:
            duplicates.append({'raw_index':i,'response_id':rec['response_id'],'retained_response_id':seen[key],
                'group_id':group_id[i],'reason':'identical_rendered_evidence_answer_and_full_silver_spans','content_sha256':key})
            continue
        seen[key]=rec['response_id'];rec['group_id']=group_id[i];retained.append(rec)
    assert len(retained)+len(duplicates)+len(quarantined)==7483
    save_rows('candidate_fit.jsonl',retained);save_rows('duplicate_index.jsonl',duplicates)
    save_rows('fixed_candidate_index.jsonl',[{'raw_index':i,'response_id':records[i]['response_id'],
        'source_id':records[i]['source_id'],'group_id':group_id[i],'factual_only_exact_alignment':True,
        'quarantined':i in quarantined,'original_prompt_sha256':records[i]['original_prompt_sha256'],
        'completion_sha256':records[i]['completion_sha256']} for i in sorted(records)])
    retained_counts=Counter(r['group_id'] for r in retained)
    v1=ROOT/'auxiliary_fava_v1';old_manifest=read(v1/'manifest.json')
    assert read(v1/'complete.json')['manifest_sha256']==sha(v1/'manifest.json')
    for n,h in old_manifest['artifacts_sha256'].items():assert sha(v1/n)==h
    previous={r['raw_index']:r for r in rows(v1/'candidate_fit.jsonl')}
    previous_fixed={r['raw_index']:r for r in rows(v1/'fixed_candidate_index.jsonl')}
    assert set(previous_fixed)==set(records)
    for rec in retained:
        if rec['raw_index'] in previous:
            old=previous[rec['raw_index']]
            assert {k:v for k,v in rec.items() if k!='group_id'}=={k:v for k,v in old.items() if k!='group_id'}
    blank_parts=[{'raw_index':i,'reference_ordinal':ordinal,'characters':len(ref_texts[h]),
                  'reference_text_sha256':h,'exactly_empty':ref_texts[h]==''}
                 for h,members in ref_members.items() if not ref_texts[h].strip() for i,ordinal in members]
    restored=sorted(set(r['raw_index'] for r in retained)-set(previous))
    removed=sorted(set(previous)-set(r['raw_index'] for r in retained))
    changed_groups=[{'raw_index':i,'v1_group_id':previous_fixed[i]['group_id'],'v2_group_id':group_id[i]}
                    for i in sorted(records) if previous_fixed[i]['group_id']!=group_id[i]]
    save('V1_TO_V2_DIFF.json',{'fixed_candidates_unchanged':True,'fixed_candidates':len(records),
        'empty_or_whitespace_parts':sorted(blank_parts,key=lambda r:(r['raw_index'],r['reference_ordinal'])),
        'empty_or_whitespace_parts_count':len(blank_parts),'affected_raw_indices':sorted({r['raw_index'] for r in blank_parts}),
        'restored_raw_indices':restored,'removed_raw_indices':removed,
        'restoration_reason':'Empty/whitespace-only sections carry no source identity; ignoring these artificial matching/group links corrects overblocking, not substantive leakage.',
        'group_id_changes':changed_groups,'v1_all_candidate_components':sum(1 for _ in rows(v1/'candidate_material_group_index.jsonl')),
        'v2_all_candidate_components':len(groups),'common_candidate_text_labels_and_other_fields_exact':True,
        'v1_retained':len(previous),'v2_retained':len(retained),'v1_all_manifest_artifacts_unchanged':True,
        'v1_manifest_sha256':sha(v1/'manifest.json'),'GPU_used':False,'model_scores_used':False})
    save('SOURCE_ISOLATION_REPORT.json',{'identity':identity_report,'fixed_candidates':7483,
        'candidate_reference_components':len(groups),'largest_candidate_component':max(map(len,groups.values())),
        'RT_unique20word_index_size':len(grams),'unique_candidate_reference_blocks':len(ref_texts),
        'matched_reference_RT_component_pairs':len(matches),'directly_blocked_candidates':len(direct),
        'quarantined_candidates_after_component_propagation':len(quarantined),
        'retained_candidates':len(retained),'retained_reference_components':len(retained_counts),
        'largest_retained_component':max(retained_counts.values(),default=0),'exact_duplicates_removed':len(duplicates),
        'raw_response_file_opened':False,'test_answers_or_labels_read':False,'no_new_split':True,
        'not_entity_or_article_independence_proof':True})
    artifacts=['candidate_fit.jsonl','fixed_candidate_index.jsonl','duplicate_index.jsonl','quarantined_candidate_index.jsonl',
        'candidate_material_group_index.jsonl','shared_reference_block_index.jsonl','material_match_index.jsonl',
        'rt_source_identity_index.jsonl','rt_material_group_index.jsonl','SOURCE_ISOLATION_REPORT.json','DATA_PROTOCOL.json','V1_TO_V2_DIFF.json']
    assert frozen==check()
    manifest={'status':'staged_synthetic_auxiliary_fit_not_trained','answers':len(retained),'fixed_factual_only_candidates':7483,
        'quarantined_candidates':len(quarantined),'duplicates_removed':len(duplicates),
        'unique_sources':len({r['source_id'] for r in retained}),'unique_groups':len(retained_counts),
        'largest_material_group':max(retained_counts.values(),default=0),
        'span_types':dict(Counter(s['type'] for r in retained for s in r['labels'])),
        'question_empty':True,'five_original_reference_sections_preserved':True,
        'all_positives_synthetic_not_human':True,'unmarked_positions_silver_not_verified_negatives':True,
        'edited_projection_used':False,'model_scores_used_for_selection':False,
        'tokenizer_loaded':False,'GPU_used':False,'trained':False,'raw_response_file_opened':False,
        'official_test_answers_or_labels_read':False,'existing_QA_fit_cal_or_test_modified':False,
        'data_design_sha256':sha(OUT/'data_design_freeze.json'),'sources_sha256':source_hashes(),
        'artifacts_sha256':{n:sha(OUT/n) for n in artifacts},'seconds':time.perf_counter()-tick,
        'limits':protocol()['limits']}
    save('manifest.json',manifest)
    save('complete.json',{'status':'candidate_data_staging_complete','manifest_sha256':sha(OUT/'manifest.json'),
        'answers':len(retained),'tokenization_started_by_this_stage':False,'GPU_used':False,'trained':False})
    print('FAVA_SILVER_CANDIDATES_STAGED',json.dumps({k:manifest[k] for k in ('answers','quarantined_candidates','duplicates_removed','unique_groups','largest_material_group','span_types')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('design','check','run'));stage=p.parse_args().stage
    try:globals()[stage]()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True);save(f'DATA_FAILURE_{stage}_{time.time_ns()}.json',{'error':repr(exc),'GPU_used':False,'trained':False});raise
