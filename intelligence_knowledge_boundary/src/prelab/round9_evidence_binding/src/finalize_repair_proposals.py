"""Collect only source-review decisions; do not mutate the reviewed paired inputs."""
import copy
import importlib.util
from pathlib import Path

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('builder', Path(__file__).with_name('build_inputs9.py'))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
pool = {r['candidate_id']: r for r in B.read_rows(P/'data/curation/source_pool.jsonl')}
packets = B.read_rows(P/'data/donor_review_packets.jsonl')
targets = {r['candidate_id']: r for r in B.read_json(P/'data/curation/selected_target_candidates.json')}
all_targets = {r['candidate_id']: r for r in B.load_candidates(pool)}
initial = B.read_json(P/'data/curation/donor_repair_proposals.json')
proposed = {r['snapshot_row_index']: copy.deepcopy(r) for r in initial['proposals']}
changes = {}

def between(cid, begin, end):
    text = pool[cid]['source_content']; start = text.index(begin)
    return text[start:text.index(end, start)+len(end)]

def change(i, reason, donor_id=None, quote=None, target_updates=None):
    packet = packets[i]; target = copy.deepcopy(targets[packet['question_id']])
    if i == 91: target = copy.deepcopy(all_targets['ragognize_train_1396'])
    if target_updates: target.update(target_updates)
    current = changes.get(i)
    if current:
        current['reasons'].append(reason)
        if not target_updates: target = current.pop('_target')
    else:
        current = {'snapshot_row_index': i, 'old_question_id': packet['question_id'], 'reasons': [reason]}
    donor_id = donor_id or packet['donor_id']; quote = quote or packet['donor_quote']
    source = pool[donor_id]
    for k in ['common_quote','evidence_quote','partial_quote']:
        B.source_span(pool[target['candidate_id']],target[k])
    a=B.wc(target['common_quote'])+B.wc(target['evidence_quote'])+B.wc(quote)
    b=B.wc(target['common_quote'])+B.wc(target['partial_quote'])+B.wc(quote)
    current.update({'new_question_id': target['candidate_id'], 'target_replaced': i==91,
        'target_question':target['question'],'target_subjects':target['subjects'],
        'target_reference':target['reference_answer'],'target_common':target['common_quote'],
        'target_evidence':target['evidence_quote'],'target_partial':target['partial_quote'],
        'old_donor_id':packet['donor_id'],'old_donor_title':packet['donor_title'],
        'old_donor_quote':packet['donor_quote'],'proposed_donor_id':donor_id,
        'proposed_donor_title':source['source_title'],'proposed_donor_quote':quote,
        'proposed_donor_span':B.source_span(source,quote),'source_url':source['source_url'],
        'revision_id':source['revision_id'],'source_content_sha256':source['source_content_sha256'],
        'new_relative_length_gap':abs(a-b)/max(a,b),
        'status':'proposal_requires_root_semantic_review','_target':target})
    changes[i]=current

# Root already read these proposed replacements; the remaining physical units
# and six length problems are explicit substantive findings, not regex rejection.
for i in [50,52,62,75,94,109,112,113,114,115,116,118,157,164,175,181,190]:
    r=proposed[i];change(i,'; '.join(r['reasons']),r['proposed_donor_id'],r['proposed_donor_quote'])

head=B.read_json(P/'data/reviews/donor_rows_000_049.json')
for row in head['rows']:
    if row['decision']=='needs_change':
        alt=row['suggested_alternate']
        change(row['packet_row_index'],row['reason'],alt['donor_id'],alt['quote'])

middle=B.read_json(P/'data/reviews/donor_rows_100_149.json')
for row in middle['items']:
    if row['decision']!='needs_change' or row['index'] in [110,115]:continue
    alt=row['suggested_change'];i=row['index']
    if alt['kind']=='replace_target_partial_quote':
        change(i,row['reason'],target_updates={'partial_quote':alt['replacement_quote']})
    else:change(i,row['reason'],alt['donor_id'],alt['replacement_quote'])

change(70,'Remove repeated Senedd report donor; root preferred independent rugby squad announcement.',
    'ragognize_train_0243',between('ragognize_train_0243','Pritchard was named in the Brumbies squad','against the Drua.'))
change(84,'Original donor was transfer money, whereas target asks a circulation count.',
    'ragognize_train_1623','By December 2024, the series had over 200,000 copies in circulation.')
change(91,'Root approved replacing the Draft target by previously reviewed Adedoyin Oseni; retain original split and passage order.',
    'ragognize_train_0518',between('ragognize_train_0518','The fire was first reported at around 3:27 a.m.','before spreading upwards.'))
change(92,'Replace fictional daughters with an attributed real hospitalization count.',
    'ragognize_train_1797',between('ragognize_train_1797','Starting in mid-January 2025, approximately 93 people','intensive care treatment.'))
change(97,'Replace internally inconsistent Grand Prix donor by a genuine area fact.',
    'ragognize_test_0402',between('ragognize_test_0402','The complex emphasizes eco-friendly design','public gatherings.'))
change(124,'Root approved retaining the independent biographical identity statement and omitting the Eurovision 2025 sentence.',
    quote='Marty Zambotto (born 25 November 1995), known professionally as Go-Jo, is an Australian singer, songwriter, and record producer.')

# A full relevant sentence removes the unrelated Lobo preface without changing
# the requested relationship or its original-source support.
cid=packets[125]['question_id']
change(125,'Length repair by retaining the full relevant Those Not Afraid sentence; unrelated Lobo sentence omitted.',
    target_updates={'evidence_quote':between(cid,'In September 2024, it was announed','kills first.')})
change(160,'Original HIV donor lacked the antecedent for these drugs; replace by an explicit real player withdrawal. The initially proposed extended cancellation paragraph mentioned another formal target, Rose BC, and is not used.',
    'ragognize_train_1860',between('ragognize_train_1860','In late November 2024, Kelsey Plum announced','Laces with a wildcard spot.'))
change(132,'New correctly anchored target partial is longer; retain the donor source but use the complete biographical role/background opening to match length.',
    quote=between(packets[132]['donor_id'],'Brahma Prakash Singh (born 1980)','cultural enactments.'))
change(143,'Retain the complete developer/publisher relationship sentence; omit the platform list because it mentions the old target Microsoft.',
    quote='Pipistrello and the Cursed Yoyo is an upcoming platform-adventure developed by Pocket Trap and published by PM Studios.')
change(168,'Use the complete 2024 attributed Sonic development statement, excluding the truncated 2017 quotation.',
    quote='In 2024, Iizuka said that while Sonic Team had no plans for Adventure 3, they would "love to make it".')
cid=packets[172]['question_id']
change(172,'Restore asset declaration requirement and non-declarer antecedent; use genuine unrelated political-visit background to match length.',
    target_updates={
        'evidence_quote':between(cid,'All public officials are required by law','failing to folloow the law.'),
        'partial_quote':between(cid,'In 2024, parliament discussed','Malawians working there.')})
change(176,'Replace the ambiguous CEO appositive by the source’s unambiguous founder sentence.',
    target_updates={'common_quote':'Peak Design was founded in 2010 by Peter Dering.'})
change(193,'Replace the source’s known incorrect January date for the February 2025 private-enterprise symposium; no factual source editing.',
    'ragognize_train_0262',between('ragognize_train_0262','The Jonas Brothers announced that, in 2025,','and a soundtrack.'))
change(194,'Restore the critic and publication attribution rather than quote a sentence from inside an open quotation.',
    quote='Sridevi. S of The Times of India gave 2.5/5 stars and was critical about the lead pair chemistry.')
change(195,'Use the actual film production/author relationship instead of an unanchored fictional protagonist.',
    quote=between(packets[195]['donor_id'],'A.I. Heart U is an upcoming','and Jonathan Kite.'))

effective=copy.deepcopy(packets)
for i,r in changes.items():
    effective[i].update({'question_id':r['new_question_id'],'question':r['target_question'],
        'subjects':r['target_subjects'],'common_quote':r['target_common'],
        'evidence_quote':r['target_evidence'],'partial_quote':r['target_partial'],
        'donor_id':r['proposed_donor_id'],'donor_quote':r['proposed_donor_quote']})
assert len({r['question_id'] for r in effective})==200
assert len({r['donor_id'] for r in effective})==200
assert not {r['question_id'] for r in effective}&{r['donor_id'] for r in effective}
for r in changes.values():r.pop('_target',None)
rejected=[{'snapshot_row_index':r['snapshot_row_index'],'status':'rejected_unnecessary_programmatic_flag',
           'reason':'Existing source-review approval governs; no physical-unit, event, sentence-integrity, or length issue established.'}
          for r in initial['proposals'] if r['snapshot_row_index'] not in changes]
result={'status':'localized_proposals_only_no_original_pair_modified',
    'snapshot_inputs_sha256':B.sha(P/'data/paired_inputs.jsonl'),
    'snapshot_packets_sha256':B.sha(P/'data/donor_review_packets.jsonl'),
    'proposal_count':len(changes),'proposals':[changes[i] for i in sorted(changes)],
    'rejected_automatic_candidates':rejected,
    'resolved_without_edit':[{'row':110,'reason':'Shared Southland vote is removed from row115, which also needed a money donor.'},
                            {'rows':[180,187],'reason':'Root ruled London, Riyadh and Weimar meetings are distinct events, not one shared event merely because all concern Ukraine diplomacy.'}],
    'source_review_files':{f' Donor block {a:03d}-{b:03d}'.strip():B.sha(P/f'data/reviews/donor_rows_{a:03d}_{b:03d}.json') for a,b in [(0,49),(50,99),(100,149),(150,199)]},
    'length_over_20_after_proposals':[{'row':r['snapshot_row_index'],'gap':r['new_relative_length_gap']} for r in changes.values() if r['new_relative_length_gap']>.2]}
B.write_json(P/'data/curation/donor_repair_proposals_v2.json',result)
print({'proposals':len(changes),'remaining_length_exceptions':result['length_over_20_after_proposals']})
