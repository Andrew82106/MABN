"""Assemble and audit source-only heldout inputs; no generation or score access."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CUR = Path(__file__).resolve().parent
DATA = CUR.parent
WORK = CUR.parents[3]
CATEGORIES = ['time', 'quantity', 'location', 'relation', 'action']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]

def norm(s):
    return re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', s).casefold()).strip()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def write_rows(path, value):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in value), encoding='utf-8')

def windows(text, n=15):
    words = norm(text).split()
    return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}

def main():
    packets = [p for category in CATEGORIES for p in rows(CUR/f'paired_review_{category}.jsonl')]
    assert [p['group_index'] for p in packets] == list(range(60))
    assert Counter(p['category'] for p in packets) == {c:12 for c in CATEGORIES}
    pool = {p['candidate_id']: p for p in rows(CUR/'remaining_source_pool.jsonl')}
    inventory = read(CUR/'legacy_isolation_inventory.json')
    for name, expected in inventory['source_inputs'].items():
        assert sha(WORK/name) == expected, ('legacy_changed', name)
    old_texts = inventory['normalized_old_visible_passages']
    old_titles = set(inventory['old_titles'])
    old_subjects = set(inventory['old_subjects'])
    old_windows = set().union(*(windows(t) for t in old_texts))
    dev = rows(DATA/'dev_inputs.jsonl')
    inputs = [row for p in packets for row in p['paired_inputs']]
    assert len(inputs) == 120 and len({r['row_id'] for r in inputs}) == 120
    assert not {r['row_id'] for r in inputs} & {r['row_id'] for r in dev}
    assert not {r['question_id'] for r in inputs} & {r['question_id'] for r in dev}
    refs, ownership, all_quotes, findings = [], [], [], []
    for packet in packets:
        t, d, cid = packet['target'], packet['donor'], packet['question_id']
        s, ds = pool[cid], pool[d['donor_candidate_id']]
        for role, source in [('target',s),('donor',ds)]:
            assert hashlib.sha256(source['source_content'].encode()).hexdigest() == source['source_content_sha256']
            assert norm(source['source_title']) not in old_titles | old_subjects
            ownership.append({'question_id':cid, 'role':role, 'source_id':source['candidate_id'],
                              'title':source['source_title'],'url':source['source_url'],
                              'revision_id':source['revision_id'],'source_content_sha256':source['source_content_sha256']})
        spans = {}
        for key in ['common_quote','evidence_quote','partial_quote']:
            q = t[key]
            assert s['source_content'].count(q) == 1
            start = s['source_content'].index(q)
            spans[key] = {'text':q,'start':start,'end':start+len(q)}
            all_quotes.append({'question_id':cid,'role':key,'title':s['source_title'],'text':q})
        intervals = sorted((x['start'],x['end']) for x in spans.values())
        assert all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:])), cid
        dq = d['donor_quote']
        assert ds['source_content'].count(dq) == 1
        start = ds['source_content'].index(dq)
        all_quotes.append({'question_id':cid,'role':'donor','title':ds['source_title'],'text':dq})
        a,b = packet['paired_inputs']
        assert a['condition'] == 'complete' and b['condition'] == 'partial'
        assert a['questions'] == b['questions'] == [t['question']]
        assert [x['title'] for x in a['passages']] == [x['title'] for x in b['passages']]
        for row,key in [(a,'evidence_quote'),(b,'partial_quote')]:
            assert row['passages'][[p['title'] for p in row['passages']].index(ds['source_title'])]['text'] == dq
            tq = sorted([t['common_quote'],t[key]], key=s['source_content'].index)
            assert row['passages'][[p['title'] for p in row['passages']].index(s['source_title'])]['text'] == ' '.join(tq)
        assert max(packet['relative_char_gap'],packet['relative_word_gap']) <= .2
        rationale = t['rationale']
        if cid == 'ragognize_train_0348':
            rationale += ' Annotation caveat: the partial lead already supplies the film year 2024, so missing full release date does not make every date component unsupported.'
        refs.append({'question_id':cid,'group_id':cid,'split':'test','dataset':a['dataset'],
                     'category':t['category'],'subjects':t['subjects'],
                     'original_question':s['original_question'],'original_answer_quote':s['original_answer_quote'],
                     'source_title':s['source_title'],'source_url':s['source_url'],'revision_id':s['revision_id'],
                     'source_content_sha256':s['source_content_sha256'],
                     'items':[{'item_index':1,'reference_answer':t['reference_answer'],
                               'aliases':t.get('answer_aliases',[]),
                               'evidence':[{'title':s['source_title'],**spans['evidence_quote']}],
                               'rationale':rationale}],
                     'coverage':{'complete':[True],'partial':[False]},
                     'coverage_scope':'Whether the complete requested answer is supplied; never a generated-response or token-risk label.',
                     'common':spans['common_quote'],'partial_replacement':spans['partial_quote'],
                     'donor':{'source_id':ds['candidate_id'],'title':ds['source_title'],'text':dq,
                              'url':ds['source_url'],'revision_id':ds['revision_id'],
                              'source_content_sha256':ds['source_content_sha256'],'start':start,'end':start+len(dq)},
                     'review_status':'source_reviewed_pairs_independently_reviewed_or_pending_root_final_freeze'})
    for key in ['source_id','url','source_content_sha256']:
        assert len({s[key] for s in ownership}) == 120, ('ownership_collision',key)
    for q in all_quotes:
        ws = windows(q['text'])
        overlap = sorted(ws & old_windows)
        if overlap:
            findings.append({'type':'old_15word_window','question_id':q['question_id'],'role':q['role'],'matches':overlap})
        visible = ' '+norm(q['title']+' '+q['text'])+' '
        matched = [term for term in old_subjects if len(term.split()) > 1 and ' '+term+' ' in visible]
        if matched:
            findings.append({'type':'old_subject_in_new_visible','question_id':q['question_id'],'role':q['role'],'terms':sorted(matched),'text':q['text']})
    indexed = defaultdict(set)
    for q in all_quotes:
        for w in windows(q['text']):
            indexed[w].add(q['question_id'])
    cross = defaultdict(list)
    for w, owners in indexed.items():
        if len(owners)>1:
            cross[tuple(sorted(owners))].append(w)
    for owners, ws in cross.items():
        findings.append({'type':'new_cross_group_15word_window','question_ids':owners,'matches':sorted(ws)})
    for p in packets:
        for role,terms in [('target',p['target']['subjects']),('donor',p['donor'].get('donor_subjects',[]))]:
            for term in terms:
                nt = ' '+norm(term)+' '
                for q in all_quotes:
                    if q['question_id'] != p['question_id'] and nt in ' '+norm(q['title']+' '+q['text'])+' ':
                        findings.append({'type':'new_cross_group_subject','subject_owner':p['question_id'],'subject_role':role,
                                         'term':term,'visible_owner':q['question_id'],'visible_role':q['role'],'text':q['text']})
    from transformers import AutoTokenizer
    model = WORK/'prelab/models/Qwen2.5-7B-Instruct-bnb-4bit'
    tok = AutoTokenizer.from_pretrained(model,local_files_only=True)
    lengths = []
    for row in inputs:
        ids = tok.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True)
        assert len(ids) <= 3072
        lengths.append({'row_id':row['row_id'],'prompt_tokens':len(ids)})
    write_rows(DATA/'test_inputs.jsonl', inputs)
    write_rows(DATA/'test_references.jsonl', refs)
    write_rows(CUR/'paired_review_all.jsonl', packets)
    audit = {'status':'ready_for_root_final_review_and_freeze_no_model_outputs_used',
             'groups':60,'input_rows':120,'expected_response_items':120,'per_category':{c:12 for c in CATEGORIES},
             'conditions':{'complete':60,'partial':60},'split':'test',
             'source_owners':ownership,'unique_source_ids':120,'unique_source_urls':120,'unique_source_content_hashes':120,
             'legacy_input_hashes_verified':inventory['source_inputs'],'old_and_new_row_or_group_id_overlap':[],
             'principal_subject_old_visible_hits':read(CUR/'donor_subject_overlap_review.json'),
             'principal_subject_hit_adjudication':[{'question_id':'ragognize_train_0107','term':'Harvey',
                 'decision':'distinct_entities','reason':'New target is Harvey legal AI software. Old occurrences are actors Harvey Keitel and Paul Harvey and the pupil Harvey Willgoose; no shared target or election/event.'}],
             'additional_screen_findings':findings,
             'additional_screen_adjudications':[
                 {'type':'old_15word_window','question_id':'ragognize_test_1482','decision':'distinct_entities_and_events_shared_financial_boilerplate',
                  'reason':'The new quotation identifies RoboSense, 5 January 2024 and HK$990 million. The old passage identifies Innoscience, 30 December 2024 and HK$1.4 billion. The matching words are a generic description of a Hong Kong initial public offering, not the same offering or company.'},
                 {'type':'new_cross_group_15word_window','question_ids':['ragognize_train_0355','ragognize_train_0961'],'decision':'distinct_people_and_target_facts_shared_biographical_boilerplate',
                  'reason':'Tony Harbison represents district 144; Cecelie Williams represents district 111. Their common state and 2024 electoral cycle are background only. The requested and hidden facts are Harbison\'s number of grandchildren and Williams\'s marital-law proposal, with no shared answer evidence. Both groups belong to this fresh test split.'}],
             'screen_limit':'Substring and 15-word checks assist the independent semantic source and paired-input reviews; empty matches are not a proof against every implicit world-knowledge relation.',
             'max_relative_material_char_gap':max(p['relative_char_gap'] for p in packets),
             'max_relative_material_word_gap':max(p['relative_word_gap'] for p in packets),
             'min_prompt_tokens':min(x['prompt_tokens'] for x in lengths),'max_prompt_tokens':max(x['prompt_tokens'] for x in lengths),
             'prompt_token_counts':lengths,
             'files':{str(p.relative_to(DATA)):sha(p) for p in [DATA/'test_inputs.jsonl',DATA/'test_references.jsonl',CUR/'paired_review_all.jsonl',CUR/'remaining_source_pool.jsonl',CUR/'legacy_isolation_inventory.json']}}
    write(CUR/'final_source_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['groups','input_rows','additional_screen_findings','max_relative_material_char_gap','max_relative_material_word_gap','min_prompt_tokens','max_prompt_tokens','files']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
