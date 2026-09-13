"""Propose localized repairs against the fixed reviewed assembly snapshot."""
import collections
import importlib.util
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('builder', Path(__file__).with_name('build_inputs9.py'))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
pool = {r['candidate_id']:r for r in B.read_rows(P/'data/curation/source_pool.jsonl')}
packets = B.read_rows(P/'data/donor_review_packets.jsonl')
selected = B.read_json(P/'data/curation/selected_target_candidates.json')
targets = {r['candidate_id']:r for r in selected}
all_candidates = {r['candidate_id']:r for r in B.load_candidates(pool)}
assignments = B.read_json(P/'data/curation/donor_assignments_proposed.json')
old_subjects, _ = B.old_material()
used = {r['donor_id'] for r in assignments.values()}
target_subjects = [B.norm(s) for r in selected for s in r['subjects']]
target_subjects += [B.norm(s) for s in all_candidates['ragognize_train_1396']['subjects']]
reasons = collections.defaultdict(list)
for i,reason in {
4:'Same Eurovision 2025 event and duplicate calendar passage',6:'Same Eurovision 2025 event and duplicate calendar passage',16:'No actual geographic location fact',21:'Eurovision 2025 donor event reused across groups',23:'Commentary about a person, not a geographic location fact',27:'Religious-office purpose, not a geographic location fact',34:'in a director’s film is not a geographic location',39:'Fictional plot setting, not a real subject location',42:'No determined actual calendar date',50:'Same Senedd final-report event directly supplies the target answer',52:'Known internally inconsistent embassy opening interval',62:'Senedd report event reused across groups',70:'Senedd report event reused across groups',75:'Senedd report event reused across groups',91:'Approved replacement of Draft FerryGoGo target by reviewed Adedoyin Oseni target',92:'Fictional family count',94:'Flintstone 84 percent metric already excluded as ambiguous',97:'Wrong event chronology and no area quantity',114:'Same Eurovision 2025 event and not a percentage',124:'Same Eurovision 2025 event reused across groups'
}.items():reasons[packets[i]['question_id']].append(reason)

money = re.compile(r'[$€£₹₦฿₱₺₽¢]|\b(?:dollars|euros|baht|birr|rupees|shillings|crore)\b',re.I)
area = re.compile(r'\b(?:hectares?|acres?|square (?:miles?|kilomet(?:er|re)s?|met(?:er|re)s?)|km2|km²|m²)\b',re.I)
speed = re.compile(r'\b(?:kilometers per hour|kilometres per hour|km/h|kph|mph|miles per hour)\b',re.I)
months = r'January|February|March|April|May|June|July|August|September|October|November|December'
geography = re.compile(r'\b(?:located|headquartered|based|born|resides|situated|moved|relocated|settled|died)\b[^.!?]{0,90}\b(?:in|at|to|near)\b',re.I)
calendar = re.compile(r'\b(?:in|on|by|from|during|until|since|between)\s+(?:\d{1,2}\s+)?(?:(?:'+months+r')|(?:late |early |mid-)?(?:19|20)\d\d|spring|summer|autumn|winter)',re.I)
fiction = re.compile(r'(?i)\b(?:fictional character|plot revolves|story follows|novel follows|film follows|story revolves)\b')

def kind(t):
    q=t['question'];a=t['reference_answer']
    if t['category']=='quantity':
        if area.search(q+' '+a):return 'area'
        if speed.search(q+' '+a):return 'speed'
        return B.attribute_kind(t)
    return t['category']

def appropriate(t,quote):
    k=kind(t)
    if k=='money':return bool(money.search(quote))
    if k=='percentage':return '%' in quote or 'percent' in quote.lower()
    if k=='area':return bool(area.search(quote))
    if k=='speed':return bool(speed.search(quote))
    if k=='count':return bool(re.search(r'\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million)\b',quote,re.I)) and not money.search(quote)
    if k=='location':return bool(geography.search(quote))
    if k=='time':return bool(calendar.search(quote))
    return True

for r in packets:
    t=targets[r['question_id']]
    if not appropriate(t,r['donor_quote']):reasons[r['question_id']].append('Fails minimum requested physical-unit/real-location/calendar-value check')
for row in B.read_json(P/'data/curation/pair_lengths.json'):
    if row['relative_gap']>.2:reasons[row['question_id']].append('Complete/partial visible-word difference exceeds 20 percent')

ban = set(r['candidate_id'] for r in B.read_json(P/'data/curation/assembly_exclusions.json'))
for fname,key in [('time_quantity_review.json','excluded'),('geography_review.json','excluded'),('quantity_tail_review.json','excluded_or_reframed')]:
    data=B.read_json(P/'data/curation'/fname)
    for row in data.get(key,[]):
        if isinstance(row,dict) and row.get('candidate_id'):ban.add(row['candidate_id'])
ban.update(['ragognize_test_0038','ragognize_test_0086','ragognize_test_1208','ragognize_train_1396'])

def acceptable_source(s):
    if s['candidate_id'] in ban or s['candidate_id'] in targets or s['source_title'].startswith('Draft:'):return False
    text=s['source_content']
    if 'Senedd constituency' in s['source_title']:return False
    if re.search(r'Eurovision.{0,35}2025|2025.{0,35}Eurovision|2025.{0,25}Asian Winter Games|Asian Winter Games.{0,25}2025',text,re.I):return False
    nt=B.norm(text)
    if any(B.has_phrase(nt,sub) for sub in target_subjects):return False
    return True

sources = {cid:s for cid,s in pool.items() if acceptable_source(s)}
all_quotes = {}
for cid,s in sources.items():
    text=s['source_content'];opts=[]
    for m in re.finditer(r'[^\n]+',text):
        q=m.group().strip()
        if 12<=B.wc(q)<=110 and q[-1:] in '.!?' and not q.startswith('=') and text.count(q)==1:opts.append(q)
    for a,b in B.sentence_spans(text):
        q=text[a:b].strip()
        if 12<=B.wc(q)<=95 and q[-1:] in '.!?' and text.count(q)==1:opts.append(q)
    filtered=[]
    for q in dict.fromkeys(opts):
        start=text.index(q);previous=text[:start]
        sections=re.findall(r'(?m)^=+\s*(.*?)\s*=+$',previous)
        if sections and re.search(r'plot|synopsis|gameplay|story',sections[-1],re.I):continue
        if fiction.search(q):continue
        nq=B.norm(q)
        if any(B.has_phrase(nq,sub) for sub in old_subjects):continue
        filtered.append(q)
    if filtered:all_quotes[cid]=filtered

proposals=[];reserved=set(used)
for index,packet in enumerate(packets):
    old_id=packet['question_id']
    if old_id not in reasons:continue
    target = all_candidates['ragognize_train_1396'] if index==91 else targets[old_id]
    current=assignments[old_id]
    length_only=all('20 percent' in r for r in reasons[old_id])
    options=[]
    for donor_id,quotes in all_quotes.items():
        if donor_id in reserved and not (length_only and donor_id==current['donor_id']):continue
        source=sources[donor_id]
        for q in quotes:
            if not appropriate(target,q):continue
            a=B.wc(target['common_quote'])+B.wc(target['evidence_quote'])+B.wc(q)
            b=B.wc(target['common_quote'])+B.wc(target['partial_quote'])+B.wc(q)
            gap=abs(a-b)/max(a,b)
            if gap>.2:continue
            score=B.donor_score(target,source,q)
            if donor_id==current['donor_id']:score+=20
            if target['category']=='location' and re.search(r'\b(?:city|town|village|district|province|county|state|region|country)\b',q,re.I):score+=1
            if target['category']=='quantity' and kind(target)=='count' and '%' in q:score-=7
            score-=gap*2
            options.append((score,donor_id,q,gap))
    options.sort(key=lambda x:(-x[0],x[1],len(x[2])))
    assert options,('no repair proposal',index,old_id,kind(target))
    score,donor_id,quote,gap=options[0];reserved.add(donor_id);source=pool[donor_id]
    alternatives=[];seen={donor_id}
    for sc,did,q,g in options[1:]:
        if did in seen:continue
        alternatives.append({'donor_id':did,'title':pool[did]['source_title'],'quote':q,'length_gap':g})
        seen.add(did)
        if len(alternatives)==2:break
    proposals.append({'snapshot_row_index':index,'old_question_id':old_id,'new_question_id':target['candidate_id'],'reasons':reasons[old_id],'target_replaced':index==91,'target_question':target['question'],'target_subjects':target['subjects'],'target_reference':target['reference_answer'],'target_common':target['common_quote'],'target_evidence':target['evidence_quote'],'target_partial':target['partial_quote'],'old_donor_id':current['donor_id'],'old_donor_title':pool[current['donor_id']]['source_title'],'old_donor_quote':current['quote'],'proposed_donor_id':donor_id,'proposed_donor_title':source['source_title'],'proposed_donor_quote':quote,'proposed_donor_span':B.source_span(source,quote),'source_url':source['source_url'],'revision_id':source['revision_id'],'source_content_sha256':source['source_content_sha256'],'new_relative_length_gap':gap,'alternatives':alternatives,'status':'proposal_requires_root_semantic_review'})

B.write_json(P/'data/curation/donor_repair_proposals.json',{'snapshot_inputs_sha256':B.sha(P/'data/paired_inputs.jsonl'),'snapshot_packets_sha256':B.sha(P/'data/donor_review_packets.jsonl'),'status':'proposals_only_original_files_unchanged','proposal_count':len(proposals),'proposals':proposals,'donor_banned_ids':sorted(ban),'source_rules':'New donors are unused by every original group, not targets, not Draft, not previously rejected candidates, not Senedd constituencies, not Eurovision 2025 or Asian Winter Games 2025. Proposed sentence must pass requested unit/geography/date checks; human source review still required.'})
print('Repair proposals:',len(proposals))
