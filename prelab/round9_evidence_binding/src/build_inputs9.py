"""Build reviewable Round9 source pairs. Never generates model text or freezes data.

Automatic donor ranking is a proposal only. Every displayed donor and pair still
requires source review before the parent process creates a data freeze.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
ROOT = ROUND.parents[1]
CUR = ROUND / 'data/curation'
SEED = 20260911
CATEGORIES = ['time', 'quantity', 'location', 'relation', 'action']
CURATION_FILES = ['time_quantity_candidates.json', 'geography_candidates.json', 'quantity_tail_candidates.json', 'root_relation_action.json', 'relation_additional.json']
TEMPLATE = 'Please answer the following questions using these search results. Write one short sentence for each numbered item.\n\nQuestions:\n{questions}\n\nSearch results:\n{search_results}'
SYSTEM = 'You are a helpful assistant.'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.findall(r'[a-z0-9]+', value))

def has_phrase(text, phrase):
    return bool(phrase) and ' ' + phrase + ' ' in ' ' + text + ' '

def wc(text):
    return len(text.split())

def source_span(source, quote):
    text = source['source_content']
    assert text.count(quote) == 1, (source['candidate_id'], quote)
    start = text.index(quote)
    return {'start': start, 'end': start + len(quote)}

def load_candidates(pool):
    candidates = []
    for name in CURATION_FILES:
        data = read_json(CUR / name)
        rows = data['decisions'] if isinstance(data, dict) else data
        for source_row in rows:
            row = dict(source_row)
            row['curation_file'] = name
            source = pool[row['candidate_id']]
            assert row['category'] in CATEGORIES
            spans = [source_span(source, row[k]) for k in ['common_quote', 'evidence_quote', 'partial_quote']]
            spans.sort(key=lambda r: r['start'])
            assert all(a['end'] <= b['start'] for a,b in zip(spans, spans[1:])), row['candidate_id']
            for key in ['source_title', 'source_url', 'revision_id', 'retrieval_date_utc', 'source_content_sha256', 'source_split', 'source_id', 'official_category', 'official_information_type']:
                row[key] = source[key]
            candidates.append(row)
    assert len({r['candidate_id'] for r in candidates}) == len(candidates), 'duplicate target candidate'
    return candidates

def old_material():
    old_rows = []
    for rnd in ['round6_evidence_grounding', 'round7_evidence_grounding']:
        for name in ['inputs.jsonl', 'dev_inputs.jsonl']:
            path = ROOT / 'prelab' / rnd / 'data' / name
            if path.exists():
                old_rows.extend(read_rows(path))
    subjects = sorted({norm(s) for r in old_rows for s in r.get('subjects', [])})
    material = [(r['row_id'], norm(' '.join(r['questions']) + ' ' + ' '.join(p['title'] + ' ' + p['text'] for p in r['passages']))) for r in old_rows]
    return subjects, material

def select_targets(candidates):
    excluded = {r['candidate_id']: r['reason'] for r in read_json(CUR/'assembly_exclusions.json')}
    selected, not_selected = [], []
    for category in CATEGORIES:
        available = [r for r in candidates if r['category'] == category and r['candidate_id'] not in excluded and 'excluded' not in r.get('status','')]
        assert len(available) >= 40, (category, len(available))
        seen_topics = collections.Counter()
        chosen = []
        while len(chosen) < 40:
            def score(row):
                complete = wc(row['common_quote']) + wc(row['evidence_quote'])
                partial = wc(row['common_quote']) + wc(row['partial_quote'])
                length_gap = abs(complete-partial) / max(complete,partial,1)
                reserve = 0.05 if row.get('selection_recommendation') in ['backup','reserve','alternate'] else 0
                return (length_gap + 0.015 * seen_topics[row['official_category']] + reserve, row['candidate_id'])
            winner = min(available, key=score)
            available.remove(winner)
            winner = dict(winner)
            winner['selection_score'] = score(winner)[0]
            chosen.append(winner)
            seen_topics[winner['official_category']] += 1
        selected.extend(chosen)
        not_selected.extend({'candidate_id':r['candidate_id'], 'category':category, 'reason':'source-only length/diversity ranking beyond 40; no model output viewed'} for r in available)
    not_selected.extend({'candidate_id':r['candidate_id'],'category':r['category'],'reason':excluded.get(r['candidate_id'],r.get('status'))} for r in candidates if r['candidate_id'] in excluded or 'excluded' in r.get('status',''))
    return selected, not_selected

def split_assign(selected):
    # Exact per-category counts; source/subject/event review is still required.
    assigned = {}
    rng = random.Random(SEED)
    for category in CATEGORIES:
        ids = sorted(r['candidate_id'] for r in selected if r['category'] == category)
        rng.shuffle(ids)
        for i,cid in enumerate(ids):
            assigned[cid] = 'train' if i < 24 else 'validation' if i < 32 else 'test'
    return assigned

def sentence_spans(text):
    """Conservative proposal segmentation; reviewers confirm complete sentences."""
    result = []
    abbreviations = {'mr','mrs','ms','dr','prof','st','jr','sr','inc','ltd','co','corp','no','vs','gen','rev','hon','maj','e.g','i.e'}
    for match in re.finditer(r'[^\n]+', text):
        line = match.group().strip()
        if not line or line.startswith('='):
            continue
        offset = match.start() + len(match.group()) - len(match.group().lstrip())
        start = 0
        for punct in re.finditer(r'[.!?](?=\s+|$)', line):
            end = punct.end()
            token = re.search(r'([A-Za-z.]+)\.$', line[:end])
            if token and (token.group(1).lower() in abbreviations or len(token.group(1)) == 1 or re.fullmatch(r'(?:[A-Z]\.)+[A-Z]?',token.group(1))):
                continue
            nxt = line[end:].lstrip()
            if nxt and nxt[0].islower():
                continue
            quote = line[start:end].strip()
            if quote:
                left = line.find(quote,start,end)
                result.append((offset+left, offset+left+len(quote)))
            start = end
            while start < len(line) and line[start].isspace():start += 1
        if start < len(line):
            quote = line[start:].strip()
            if quote and len(quote.split()) >= 7:
                left = line.find(quote,start)
                result.append((offset+left,offset+left+len(quote)))
    return result

def attribute_kind(target):
    q = norm(target['question']); answer = target['reference_answer']
    if target['category'] == 'quantity':
        if re.search(r'[$€£₹₦]|dollars|euros|baht|cash|investment|cost|revenue|assets|fee|money',answer+' '+q,re.I):return 'money'
        if re.search(r'%|percent|rate|efficiency',answer+' '+q,re.I):return 'percentage'
        return 'count'
    if target['category'] == 'time':
        if re.search(r'how many years|duration|how long',q):return 'duration'
        if 'month' in q:return 'month'
        if 'year' in q and 'date' not in q:return 'year'
        return 'date'
    return target['category']

def proposed_donor_quotes(source, category):
    text = source['source_content']; quote = source['original_answer_quote']; spans = sentence_spans(text)
    options = []
    if category == 'location':
        pattern = re.compile(r'\b(?:located|headquartered|based|born|resides|situated|moved|relocated|settled)\b.{0,100}\b(?:in|at|to|near)\b|\b(?:in|near)\s+[A-Z][a-z]+(?:,|\s+[A-Z])')
        options.extend((a,b) for a,b in spans if pattern.search(text[a:b]))
    elif text.count(quote) == 1:
        qa = text.index(quote); qb = qa + len(quote)
        covered = [(a,b) for a,b in spans if a < qb and b > qa]
        if covered:options.append((covered[0][0],covered[-1][1]))
    # One complete, short paragraph is preferred over a fragment when feasible.
    expanded = []
    for a,b in options:
        pa = text.rfind('\n',0,a)+1
        pb = text.find('\n',b)
        if pb < 0:pb = len(text)
        par = text[pa:pb].strip()
        if 12 <= wc(par) <= 85 and text.count(par) == 1:
            start = text.index(par); expanded.append((start,start+len(par)))
        expanded.append((a,b))
    unique = []
    for a,b in expanded:
        q = text[a:b].strip()
        if 12 <= wc(q) <= 100 and q[-1:] in '.!?' and text.count(q) == 1 and q not in unique and not q.startswith('='):
            unique.append(q)
    return unique

def donor_score(target, source, quote):
    category=target['category']; kind=attribute_kind(target)
    wanted={'time':'TEMPORAL','quantity':'NUMERICAL','relation':'IDENTITY','action':'OTHER','location':'SPATIAL'}[category]
    score=6 if source['official_information_type'] == wanted else 0
    if kind=='money':score += 5 if re.search(r'[$€£₹₦]|\b(?:dollars|euros|baht|birr)\b',quote,re.I) else -10
    if kind=='percentage':score += 5 if '%' in quote or 'percent' in quote.lower() else -10
    if category=='quantity' and re.search(r'\b(?:minutes|seconds|hours|years old|age of)\b',source['original_question'],re.I):score -= 12
    if category=='time' and kind != 'duration' and re.search(r'how long|how many years|duration',source['original_question'],re.I):score -= 8
    if category=='time' and kind == 'duration':score += 8 if re.search(r'how long|how many years|duration',source['original_question'],re.I) else -8
    ignore={'what','which','when','where','how','many','much','the','a','an','in','on','at','to','of','for','was','were','did','is','and','its','with','by','from'}
    terms=set(norm(target['question']).split())-ignore-set(' '.join(norm(s) for s in target['subjects']).split())
    other=set(norm(source['original_question']).split())
    score += min(6,len(terms & other))
    score -= abs(wc(quote)-35)/80
    return score

def build_donor_options(pool, selected, old_subjects):
    selected_ids={r['candidate_id'] for r in selected}
    selected_subjects=[norm(s) for r in selected for s in r['subjects']]
    target_evidence=[r['evidence_quote'] for r in selected if len(r['evidence_quote'])>25]
    cache={};excluded=collections.Counter()
    for source in pool.values():
        if source['candidate_id'] in selected_ids or source['source_title'].startswith('Draft:'):
            excluded['target_or_draft']+=1;continue
        full=norm(source['source_content'])
        if any(has_phrase(full,s) for s in selected_subjects):
            excluded['mentions_selected_target']+=1;continue
        if re.search(r'2025.{0,25}Asian Winter Games|Asian Winter Games.{0,25}2025',source['source_content'],re.I):
            excluded['old_2025_asian_winter_games_event']+=1;continue
        if any(q in source['source_content'] for q in target_evidence):
            excluded['duplicates_selected_fact_sentence']+=1;continue
        category_options={}
        for category in CATEGORIES:
            original={'time':'TEMPORAL','quantity':'NUMERICAL','relation':'IDENTITY','action':'OTHER'}.get(category)
            if original and source['official_information_type'] != original:continue
            quotes=proposed_donor_quotes(source,category)
            quotes=[q for q in quotes if not any(has_phrase(norm(q),s) for s in old_subjects)]
            if quotes:category_options[category]=quotes
        if category_options:cache[source['candidate_id']]=category_options
    return cache,dict(excluded)

def assign_donors(pool, selected, cache):
    override_path=CUR/'donor_overrides.json'
    overrides=read_json(override_path) if override_path.exists() else {}
    used=set(); assignments={}; packets=[]
    # Scarcer location options are assigned first; all choices are deterministic.
    ordered=sorted(selected,key=lambda r:(0 if r['category']=='location' else 1,CATEGORIES.index(r['category']),r['candidate_id']))
    for target in ordered:
        cid=target['candidate_id']; alternatives=[]
        if cid in overrides:
            donor_id=overrides[cid]['donor_id'];quote=overrides[cid]['quote']
            assert donor_id not in used and donor_id not in {r['candidate_id'] for r in selected}
            source_span(pool[donor_id],quote)
            winner=(donor_id,quote,'explicit_source_review_override')
        else:
            options=[]
            for donor_id,by_category in cache.items():
                if donor_id in used:continue
                for quote in by_category.get(target['category'],[]):
                    options.append((donor_score(target,pool[donor_id],quote),donor_id,quote))
            assert options,('no proposed donor',cid,target['category'])
            options.sort(key=lambda r:(-r[0],r[1],len(r[2])))
            first=options[0];winner=(first[1],first[2],first[0])
            seen=set()
            for score,donor_id,quote in options:
                if donor_id in seen:continue
                alternatives.append({'donor_id':donor_id,'quote':quote,'ranking_score':score})
                seen.add(donor_id)
                if len(alternatives)==3:break
        donor_id,quote,score=winner;used.add(donor_id)
        source=pool[donor_id]
        assignments[cid]={'donor_id':donor_id,'quote':quote,'ranking_score':score,'span':source_span(source,quote),'review_status':'unreviewed_programmatic_proposal'}
        packets.append({'question_id':cid,'category':target['category'],'question':target['question'],'subjects':target['subjects'],'reference_answer':target['reference_answer'],'target_title':target['source_title'],'common_quote':target['common_quote'],'evidence_quote':target['evidence_quote'],'partial_quote':target['partial_quote'],'target_source_hash':target['source_content_sha256'],'donor_id':donor_id,'donor_title':source['source_title'],'donor_quote':quote,'donor_source_url':source['source_url'],'donor_revision_id':source['revision_id'],'donor_source_hash':source['source_content_sha256'],'donor_original_question':source['original_question'],'donor_original_answer_quote':source['original_answer_quote'],'alternatives':alternatives,'review_status':'pending_manual_pair_and_donor_review'})
    return assignments,packets

def build_pairs(pool,selected,assignments,splits):
    inputs=[];refs=[]
    for target in sorted(selected,key=lambda r:(CATEGORIES.index(r['category']),r['candidate_id'])):
        cid=target['candidate_id'];source=pool[cid];donor=assignments[cid];other=pool[donor['donor_id']]
        order_rng=random.Random(f'{SEED}:{cid}:passages')
        donor_first=bool(order_rng.getrandbits(1))
        for condition,changed in [('complete','evidence_quote'),('partial','partial_quote')]:
            quotes=[target['common_quote'],target[changed]]
            quotes.sort(key=lambda q:source['source_content'].index(q))
            target_p={'title':source['source_title'],'text':' '.join(quotes)}
            donor_p={'title':other['source_title'],'text':donor['quote']}
            passages=[donor_p,target_p] if donor_first else [target_p,donor_p]
            prompt=TEMPLATE.format(questions='1. '+target['question'],search_results='\n\n'.join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(passages)))
            inputs.append({'row_id':cid+'__'+condition,'question_id':cid,'group_id':cid,'split':splits[cid],'condition':condition,'questions':[target['question']],'subjects':target['subjects'],'passages':passages,'prompt':prompt,'system':SYSTEM,'dataset':'RAGognize_reconstructed_round9','expected_items':1,'category':target['category']})
        refs.append({'question_id':cid,'group_id':cid,'split':splits[cid],'dataset':'RAGognize_reconstructed_round9','category':target['category'],'subjects':target['subjects'],'original_question':pool[cid]['original_question'],'original_answer_quote':pool[cid]['original_answer_quote'],'source_title':source['source_title'],'source_url':source['source_url'],'revision_id':source['revision_id'],'source_content_sha256':source['source_content_sha256'],'items':[{'item_index':1,'reference_answer':target['reference_answer'],'aliases':target['answer_aliases'],'evidence':[{'title':source['source_title'],'text':target['evidence_quote'],**source_span(source,target['evidence_quote'])}],'rationale':target['rationale']}],'coverage':{'complete':[True],'partial':[False]},'common':{'text':target['common_quote'],**source_span(source,target['common_quote'])},'partial_replacement':{'text':target['partial_quote'],**source_span(source,target['partial_quote'])},'donor':{'source_id':donor['donor_id'],'title':other['source_title'],'text':donor['quote'],'url':other['source_url'],'revision_id':other['revision_id'],'source_content_sha256':other['source_content_sha256'],**donor['span']},'review_status':'pending_donor_and_pair_review_not_frozen'})
    return inputs,refs

def checks(selected,inputs,assignments,old_material_rows):
    conflicts=[]
    for target in selected:
        for subject in target['subjects']:
            hits=sorted({rid for rid,text in old_material_rows if has_phrase(text,norm(subject))})
            if hits:conflicts.append({'kind':'new_target_mentioned_in_old_visible_material','question_id':target['candidate_id'],'subject':subject,'old_row_ids':hits})
    # Explicit target names in other new pair text are possible source leakage.
    by_q=collections.defaultdict(list)
    for row in inputs:by_q[row['question_id']].append(row)
    for target in selected:
        for other_id,rows in by_q.items():
            if other_id==target['candidate_id']:continue
            text=norm(' '.join(p['title']+' '+p['text'] for row in rows for p in row['passages']))
            mentions=[s for s in target['subjects'] if has_phrase(text,norm(s))]
            if mentions:conflicts.append({'kind':'cross_group_target_mention','target_id':target['candidate_id'],'other_id':other_id,'subjects':mentions})
    lengths=[]
    for qid,pair in by_q.items():
        a,b=pair
        assert a['questions']==b['questions'] and a['split']==b['split']
        assert [p['title'] for p in a['passages']]==[p['title'] for p in b['passages']]
        ac=sum(wc(p['text']) for p in a['passages']);bc=sum(wc(p['text']) for p in b['passages'])
        lengths.append({'question_id':qid,'complete_words':ac,'partial_words':bc,'relative_gap':abs(ac-bc)/max(ac,bc)})
    assert len(inputs)==400 and len(selected)==200 and len(assignments)==200
    assert len({r['donor_id'] for r in assignments.values()})==200
    assert not set(assignments) & {r['donor_id'] for r in assignments.values()}
    return conflicts,lengths

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--targets-only',action='store_true');args=parser.parse_args()
    pool={r['candidate_id']:r for r in read_rows(CUR/'source_pool.jsonl')}
    candidates=load_candidates(pool);selected,excluded=select_targets(candidates)
    write_json(CUR/'selected_target_candidates.json',selected)
    write_json(CUR/'target_selection_exclusions.json',excluded)
    if args.targets_only:
        print('Selected target candidates:',dict(collections.Counter(r['category'] for r in selected)));return
    old_subjects,old_rows=old_material()
    cache,donor_exclusion_counts=build_donor_options(pool,selected,old_subjects)
    assignments,packets=assign_donors(pool,selected,cache)
    splits=split_assign(selected);inputs,refs=build_pairs(pool,selected,assignments,splits)
    conflicts,lengths=checks(selected,inputs,assignments,old_rows)
    write_json(CUR/'donor_assignments_proposed.json',assignments)
    write_rows(ROUND/'data/donor_review_packets.jsonl',packets)
    write_rows(ROUND/'data/paired_inputs.jsonl',inputs)
    write_rows(ROUND/'data/paired_references.jsonl',refs)
    write_json(CUR/'pair_conflicts.json',conflicts)
    write_json(CUR/'pair_lengths.json',lengths)
    manifest={'status':'paired_inputs_for_review_not_frozen','seed':SEED,'groups':200,'rows':400,'expected_items':1,'counts':dict(collections.Counter(f"{r['split']}:{r['category']}" for r in refs)),'independent_group_claim':'provisional; literal names/source ownership checked, alias and shared-event review remains','donor_selection':'programmatic ranking only; every displayed donor and pair requires manual review','donor_exclusion_counts':donor_exclusion_counts,'conflicts':len(conflicts),'length_gap_max':max(r['relative_gap'] for r in lengths),'length_gap_over_20pct':sum(r['relative_gap']>.2 for r in lengths),'curation_hashes':{f:sha(CUR/f) for f in CURATION_FILES},'source_pool_sha256':sha(CUR/'source_pool.jsonl'),'files':{str(p.relative_to(ROUND)):sha(p) for p in [ROUND/'data/paired_inputs.jsonl',ROUND/'data/paired_references.jsonl',ROUND/'data/donor_review_packets.jsonl',CUR/'selected_target_candidates.json',CUR/'donor_assignments_proposed.json',CUR/'pair_conflicts.json',CUR/'pair_lengths.json']},'restrictions':'No model generations or detection scores read. No GPU, training, final input promotion or freeze performed.'}
    write_json(ROUND/'data/assembly_review_manifest.json',manifest)
    print(json.dumps({'groups':200,'rows':400,'conflicts':len(conflicts),'max_length_gap':manifest['length_gap_max'],'donor_candidates':len(cache)},ensure_ascii=False))

if __name__=='__main__':main()
