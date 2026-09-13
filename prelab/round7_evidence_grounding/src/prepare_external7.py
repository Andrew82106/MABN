"""Build a reviewed, source-only RAGognize external subset; no model calls."""
from pathlib import Path
import collections
import hashlib
import json
import re
import sys
from urllib.parse import quote as urlquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'external_ragognize'
RAW = OUT / 'raw'
PREFIX = 'Please answer the following questions using these search results. Write one short sentence for each numbered item.'
SYSTEM = 'You are a helpful assistant.'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')

def sentences(text):
    # Text is never reconstructed from these derived spans; original text stays exact.
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+', text) if x.strip()]

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    review = read_json(OUT / 'selection_review.json')
    raw_rows = [json.loads(x) for x in (RAW / 'test_sources.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs = collections.defaultdict(dict)
    positions = {}
    for ix, row in enumerate(raw_rows):
        key = int(row['user_prompt_index'])
        assert row['answerable'] not in pairs[key], (key, 'duplicate condition')
        pairs[key][row['answerable']] = row
        positions[key, row['answerable']] = ix
    chosen = review['accepted']
    assert len(chosen) == len({x['id'] for x in chosen}) == 50
    assert [x['id'] for x in chosen] == sorted(x['id'] for x in chosen)

    p6 = ROOT.parent / 'round6_evidence_grounding'
    exclusions = set()
    for line in (p6 / 'data' / 'inputs.jsonl').read_text(encoding='utf-8').splitlines():
        x = json.loads(line)
        exclusions.update(x['subjects'])
        exclusions.update(p['title'] for p in x['passages'])
    for x in read_json(p6 / 'planning' / 'hotpotqa_preview.json')['rows']:
        exclusions.update(x['context']['title'])
    inputs, refs, originals, all_titles, entity_ids = [], [], [], {}, collections.defaultdict(list)
    for spec in chosen:
        k = spec['id']
        pair = pairs[k]
        assert set(pair) == {False, True}
        complete, partial = pair[True], pair[False]
        assert complete['user_prompt'] == partial['user_prompt']
        assert 'NON_CONFIRMATORY' in complete['tags']
        assert max(len(x['documents']) for x in pair.values()) <= 2
        assert len(complete['documents']) == len(partial['documents'])
        qid = f'ragognize_test_{k:04d}'
        question = spec.get('question', complete['user_prompt'])
        d = complete['details']['user_prompt']
        article = d['details']['suitable_article']
        target_title = d['article_title']
        quoted = spec.get('quote', d['passage_containing_answer']).strip()
        evidence = []
        for pi, passage in enumerate(complete['documents']):
            if passage['title'] != target_title:
                continue
            start = passage['text'].find(quoted)
            assert start >= 0, (k, 'reference quote is not in visible complete text')
            sentlist = sentences(passage['text'])
            cursor = 0
            for sid, sentence in enumerate(sentlist):
                ss = passage['text'].find(sentence, cursor)
                assert ss >= cursor
                cursor = ss + len(sentence)
                if ss < start + len(quoted) and cursor > start:
                    evidence.append({'title': target_title, 'passage_index': pi,
                                     'sent_id': sid, 'text': sentence,
                                     'char_start': ss, 'char_end': cursor})
        assert evidence, k
        titles_this = {p['title'] for x in pair.values() for p in x['documents']}
        assert not titles_this & exclusions, (k, titles_this & exclusions)
        assert not set(spec['entities']) & exclusions, (k, 'Round6 entity overlap')
        for title in titles_this:
            assert title not in all_titles, (k, title, all_titles.get(title))
            all_titles[title] = qid
        for entity in spec['entities']:
            entity_ids[entity].append(qid)
        for flag, condition in [(True, 'complete'), (False, 'partial')]:
            x = pair[flag]
            passages = [{'title': p['title'], 'text': p['text'], 'sentences': sentences(p['text'])} for p in x['documents']]
            prompt = PREFIX + '\n\nQuestions:\n1. ' + question + '\n\nSearch results:\n' + '\n\n'.join(
                f'[{i}] {p["title"]}\n{p["text"]}' for i, p in enumerate(passages, 1))
            inputs.append({'row_id': qid + '__' + condition, 'question_id': qid, 'group_id': qid,
                           'split': 'external_test', 'condition': condition,
                           'questions': [question], 'subjects': spec['entities'],
                           'passages': passages, 'prompt': prompt, 'system': SYSTEM,
                           'dataset': 'RAGognize', 'expected_items': 1,
                           'source_split': 'test', 'source_question_id': k,
                           'source_dataset_row_index': positions[k, flag]})
            originals.append({'question_id': qid, 'source_dataset_row_index': positions[k, flag],
                              'answerable': flag, 'user_prompt': x['user_prompt'],
                              'documents': x['documents'], 'category': x['category'],
                              'information_type': x['information_type'], 'tags': x['tags'],
                              'information_date': x['information_date']})
        refs.append({'question_id': qid, 'source_question_id': k, 'split': 'external_test',
                     'dataset': 'RAGognize', 'original_question': complete['user_prompt'],
                     'question': question, 'rewrite_rationale': spec.get('rewrite_reason'),
                     'subjects': spec['entities'], 'category': complete['category'],
                     'information_type': complete['information_type'],
                     'information_date': complete['information_date'],
                     'items': [{'item_index': 1, 'reference_answer': spec['answer'],
                                'aliases': spec['aliases'], 'evidence': evidence,
                                'rationale': spec['note']}],
                     'coverage': {'complete': [True], 'partial': [False]},
                     'evidence_review': spec['note'],
                     'source_quote': quoted,
                     'upstream_reference_answer': d['golden_answer'],
                     'upstream_reference_provenance': 'Dataset-generated reference, used only alongside checked original evidence; not new experimental output.',
                     'source_article': {'title': article['title'], 'url': article['url'],
                                        'revision_id': article['revision_id'],
                                        'retrieval_date_utc': article['retrieval_date_utc']},
                     'label_warning': 'Coverage is a property of the input, not an output risk label. New Qwen statements must be annotated from their actual claims. Refusal and supported facts in a partial-context answer are not automatically risks.'})
    assert len(inputs) == 100 and len(refs) == 50
    assert len({r['row_id'] for r in inputs}) == 100
    jsonl(OUT / 'external_inputs.jsonl', inputs)
    jsonl(OUT / 'external_references.jsonl', refs)
    jsonl(OUT / 'selected_official_source_rows.jsonl', originals)
    dump(OUT / 'isolation_reservations.json', {
        'status': 'reserve_against_all_main_splits_and_development',
        'question_ids': [x['question_id'] for x in refs],
        'source_titles': sorted(all_titles), 'source_title_owner': all_titles,
        'key_entities': sorted(entity_ids), 'key_entity_question_ids': dict(entity_ids),
        'round6_exclusion_source_hashes': {
            'data/inputs.jsonl': sha(p6 / 'data' / 'inputs.jsonl'),
            'planning/hotpotqa_preview.json': sha(p6 / 'planning' / 'hotpotqa_preview.json')},
        'source_title_overlap_with_round6_and_preview': [],
        'key_entity_exact_overlap_with_round6_and_preview': []})

    # Rejection records separate mechanical eligibility from qualitative screening.
    rejects = []
    specific = {
        13:'No precise date and private-family historical fact; outside prioritized public recent-event scope.',
        38:'Source says August 25 to September 3 is a week later; internally inconsistent date description.',
        143:'Forward-looking broad individual opinion rather than a concrete institutional event; also shares background source with selected 441.',
        657:'Question supplies the interface/workflow weaknesses being requested; answer leakage. Also shares Manus source with 188.',
        759:'Partial retains substantive reform demands from the same Bangladesh movement; cannot certify lack of support for this broad question.',
        794:'Partial retains final March 2025 naming recommendations for other constituencies in the same commission process; may imply the requested date.',
        820:'Question presupposes Democratic individuals although source states none; candidate source also overlaps 114.',
        829:'Question already states a major component of the requested AI-development goal.',
        831:'Question supplies award organization, year and category, leaking most of the requested recognition.',
        863:'Enactment/passsage versus coming into effect are ambiguous; complete mentions expected April passage but later-2025 effect.',
        980:'Incomplete question does not identify which comment/event sufficiently, and the extracted quote ends mid-sentence.',
        997:'Projection anchored only to that year in the reference; insufficient temporal specificity.',
        1191:'Source explicitly says responsible individuals remain unidentified; not a positive factual answer target.',
        1261:'Question already supplies the database examination purpose rather than withholding the requested answer.',
        1452:'Historical snapshot calls an introduced bill an Act; legal status/presupposition ambiguity.',
        1470:'Unspecified recognition and vague event description make a stable target difficult.',
        1482:'Unqualified Qiu and event reference are underspecified.',
        1566:'Question itself supplies successful AI synthesis and fluorescence, from which much of the requested functional-capability conclusion follows.',
        1620:'Question and pageant name already reveal United States origin; answer leakage.',
        1717:'Question lacks the exhumation event/entity identifier.',
        1758:'Leading confirmation wording is it meant to replace them despite NON_CONFIRMATORY tag.',
        1788:'Source states emissions decreased to 10,400,000 million tonnes, an internally inconsistent unit/magnitude.',
        1813:'Question has no identified blackout event, so a different blackout in partial could answer its generic wording.',
        1913:'The French initiative is unidentified in the question; contextual referent changes with the documents.',
        1949:'Asks about document organization rather than an independently specified substantive fact.',
        1979:'Residency in the neighborhood is a relative target without naming the neighborhood in the question.',
        2049:'Source nonconfirmation is itself the expected answer; exclude evidence-about-uncertainty targets from this initial subset.',
        2114:'Recent years does not identify a specific outbreak; source excerpt provides no stable outbreak date.',
        2193:'Historical religious art themes are outside the prioritized institutional/technical/event scope.'}
    chosen_ids = {x['id'] for x in chosen}
    for k, pair in sorted(pairs.items()):
        if k in chosen_ids:
            continue
        x = pair.get(True, next(iter(pair.values())))
        stage = 'mechanical_eligibility'
        if set(pair) != {True, False}: reason = 'No unique official complete/partial test pair.'
        elif 'NON_CONFIRMATORY' not in x['tags']: reason = 'Official confirmatory prompt tag excluded.'
        elif x['category'] not in {'POLITICS','TECHNOLOGY','SCIENCE','BUSINESS','EDUCATION','HEALTH','OTHER'}: reason = 'Category outside selected public/institution/technical scope.'
        elif max(len(z['documents']) for z in pair.values()) > 2: reason = 'More than two original documents; outside predeclared short-context external subset.'
        else:
            stage = 'question_reference_shortlist_or_pair_review'
            titles = {p['title'] for z in pair.values() for p in z['documents']}
            overlap = sorted(titles & set(all_titles))
            if k in specific: reason = specific[k]
            elif overlap: reason = 'Display-source overlap with selected independent question(s): ' + '; '.join(overlap)
            else: reason = 'Not selected in the manually curated thematic quota; older/general biography, leisure/fiction, broad underspecified target, or lower priority than the 50 chosen public-event/institution/technical questions. Not asserted invalid.'
        rejects.append({'source_question_id': k, 'question': x['user_prompt'],
                        'category': x['category'], 'stage': stage, 'reason': reason})
    jsonl(OUT / 'rejected_candidates.jsonl', rejects)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT.parent / 'models' / 'Qwen2.5-7B-Instruct-bnb-4bit'), local_files_only=True)
    counts = []
    for row in inputs:
        n = len(tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt']}], tokenize=True, add_generation_prompt=True))
        assert n <= 3072, (row['row_id'], n)
        counts.append({'row_id':row['row_id'], 'prompt_tokens': n})
    dump(OUT / 'prompt_token_counts.json', counts)
    manifest = {
        'status': 'draft_pending_independent_parent_review_not_frozen',
        'dataset': 'F4biian/RAGognize', 'source_split': 'test',
        'revision': read_json(RAW / 'dataset_api.json')['sha'],
        'raw_test_rows': len(raw_rows), 'raw_unique_questions': len(pairs),
        'selected_independent_questions': 50, 'paired_inputs': 100,
        'conditions': {'complete':50,'partial':50},
        'all_original_document_text_and_order_preserved': True,
        'mutually_disjoint_visible_source_titles': len(all_titles),
        'question_rewrites': [s['id'] for s in chosen if 'question' in s],
        'categories': dict(collections.Counter(x['category'] for x in refs)),
        'information_types': dict(collections.Counter(x['information_type'] for x in refs)),
        'prompt_token_min': min(x['prompt_tokens'] for x in counts),
        'prompt_token_max': max(x['prompt_tokens'] for x in counts),
        'original_document_count_per_input': dict(collections.Counter(len(x['passages']) for x in inputs)),
        'license': 'CC-BY-SA-4.0',
        'attribution': 'F4biian/RAGognize; Wikipedia contributors and Wikimedia Foundation.',
        'raw_source_hashes': {p.name:sha(p) for p in sorted(RAW.iterdir()) if p.is_file()},
        'output_hashes': {p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {'build_manifest.json'}},
        'experimental_model_calls':0, 'gpu_use':False,
        'limits': ['Assistant source review, not independent human gold.',
                   'Wikipedia source facts are evaluated at stored snapshots, not externally verified present-day truth.',
                   'Some historical facts and undated institutional descriptions occur in recent articles; do not describe every target as post-2024 knowledge.',
                   'Official missing contexts often replace the entire target entity with unrelated material; this is a short-context external transfer check, not the main multi-entity local-omission experiment.',
                   'Input coverage is not an output hallucination label; all newly generated claims need blind annotation.']}
    dump(OUT / 'build_manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'raw_source_hashes','output_hashes','limits'}}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
