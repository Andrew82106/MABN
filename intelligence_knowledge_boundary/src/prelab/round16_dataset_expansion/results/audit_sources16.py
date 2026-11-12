"""Independent source-only isolation review; never opens generations or scores."""
from pathlib import Path
import collections
import hashlib
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CUR = ROOT / 'data/curation'
R10 = REPO / 'prelab/round10_dual_granularity/data'
FILES = ['quantity_candidates', 'relation_candidates', 'action_time_candidates',
         'root_candidates', 'root_location_extension']
QUOTE_KEYS = ['common_quote', 'evidence_quote', 'partial_quote']


def readj(p):
    return json.loads(p.read_text(encoding='utf-8'))


def readl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s):
    return re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', s).casefold()).strip()


def shingles(s):
    w = norm(s).split()
    return {' '.join(w[i:i + 15]) for i in range(len(w) - 14)}


def event_key(r, source):
    # Restrict inference to the requested fact, not every competition in a biography.
    x = norm(' '.join([r['question'], r['subject'], r.get('event_group', '')]))
    overrides = {
        'ragognize_test_0560': 'Eurovision Song Contest 2025',
        'ragognize_test_1878': 'Eurovision Song Contest 2025',
        'ragognize_train_0242': 'Eurovision Song Contest 2025',
        'ragognize_test_0003': 'Eurovision Song Contest 2025',
        'ragognize_train_1930': 'Eurovision Song Contest 2025',
        'ragognize_train_2100': 'FIS Freestyle Ski and Snowboarding World Championships 2025',
        'ragognize_train_1936': 'Syrian coastal clashes March 2025',
        'ragognize_train_2053': 'Syrian coastal clashes March 2025',
        'ragognize_train_1157': '2025 FIM JuniorGP race-meeting calendar',
        'ragognize_train_1291': 'USL Championship 2025 schedule publication',
        'ragognize_train_2163': 'SRT Dark Red Line Northern Extension approval December 2024',
    }
    if r['candidate_id'] in overrides:
        return overrides[r['candidate_id']]
    if '2025' in x and 'asian winter games' in x:
        return '2025 Asian Winter Games'
    if '2025' in x and ('eurovision' in x or 'söngvakeppnin' in x):
        return 'Eurovision Song Contest 2025'
    if '2025' in x and ('world single distances' in x):
        return '2025 World Single Distances Speed Skating Championships'
    if '2025' in x and ('fis freestyle ski' in x or 'fis world championships' in x):
        return 'FIS Freestyle Ski and Snowboarding World Championships 2025'
    if r.get('event_group'):
        return r['event_group']
    return 'entity: ' + norm(source['source_title'])


OLD_EVENT_LINKS = {
    '2025 Asian Winter Games': {
        'old_source': 'Macau at the 2025 Asian Winter Games',
        'old_phase': 'Round7 external_test',
        'reason': '同届赛事已有旧留出材料，不能把不同国家/项目说成全新赛事；并不意味着新国家名额答案已出现。'},
    'Eurovision Song Contest 2025': {
        'old_source': 'Yuval Raphael', 'old_question_id': 'ragognize_train_0088',
        'old_phase': 'Round9/10 validation',
        'reason': '旧验证题已问其Eurovision 2025选拔公布时居所；新条目是同届参赛歌曲/选拔，不是全新事件。'},
    'Syrian coastal clashes March 2025': {
        'old_source': 'March 2025 Western Syria clashes', 'old_question_id': 'ragognize_test_1866',
        'old_phase': 'Round9 test donor; also Round7 external material',
        'reason': '同一轮叙利亚沿海冲突；Beit Ana平民数与Huweija被俘须按事件隔离，但后者事实未在旧材料出现。'},
    '2025 FIM JuniorGP race-meeting calendar': {
        'old_source': '2025 European Talent Cup', 'old_question_id': 'ragognize_test_1645',
        'old_phase': 'Round9 test donor',
        'reason': '新旧证据同为7站、5月4日Estoril至11月23日Valencia，长句近乎相同而赛数12/11不同。属于具体赛历材料重叠，不能当普通体育套话；不把两个级别的赛数说成同一答案。'},
}


BLOCKING = {
    'ragognize_train_1291': {
        'kind': 'same_target_fact_and_sentence', 'old_question_id': 'ragognize_train_1041',
        'old_source': '2025 Oakland Roots SC season',
        'reason': '同一个USL 2025全联盟赛程公告，December 19, 2024及原句已用于旧train问题；换俱乐部标题未换事实。'},
    'ragognize_train_2163': {
        'kind': 'same_target_fact_and_material', 'old_question_id': 'ragognize_test_0582',
        'old_source': 'Bangkok University station',
        'reason': '同一SRT Dark Red Line Northern Extension的December 2024批准日期及2028开通背景已展示在Round7 external_test partial资料。'},
    'ragognize_train_2244': {
        'kind': 'same_event_location_fact', 'old_source': 'Macau at the 2025 Asian Winter Games',
        'reason': '新题问Jordan参加同届冬亚会的举办城市Harbin，旧Macau材料已明示同届在Harbin举办；不是新的城市事实。'},
}


HIT_EXPLANATIONS = {
    'ragognize_test_0912': '通用Missouri议员当选介绍句；Overcast与旧Tony Harbison/Cecelie Williams为不同人物/选区，所问军种也不同，不因15词套话阻断。',
    'ragognize_test_2197': 'University of Alabama为真实共同机构，但新目标为2025棒球队教练；旧学校介绍/校友材料不含该教练事实，不把所有大学关联主体并成一题。',
    'ragognize_test_1878': 'the message是普通英文短语，旧The Message电影标题的字符串命中不表示同一作品。Eurovision事件关联另报。',
    'ragognize_train_1806': 'Joyce Chitsulo确为旧目标本人，且公共任命委员会关系重复出现在新partial背景。Francesca党籍是新事实，但该背景应替换为不含旧人物的原句，不能称全部材料全新。',
}


def run():
    inventory_path = R10 / 'curation/legacy_isolation_inventory.json'
    inv = readj(inventory_path)
    source_hashes = {str(inventory_path.relative_to(REPO)): sha(inventory_path)}
    old_titles, old_subjects = set(inv['old_titles']), set(inv['old_subjects'])
    old_texts = set(inv['normalized_old_visible_passages'])
    origins = collections.defaultdict(list)
    old_questions = []
    for relative, expected in inv['source_inputs'].items():
        f = REPO / relative
        assert sha(f) == expected, ('legacy input changed', str(f))
        source_hashes[str(f.relative_to(REPO))] = expected
        if f.suffix != '.jsonl':
            continue
        for row in readl(f):
            old_questions.append({'file': relative, 'row_id': row.get('row_id'),
                                  'split': row.get('split'), 'questions': row.get('questions', [])})
            for p in row.get('passages', []):
                if p.get('text'):
                    key = norm(p['text'])
                    origins[key].append({'file': relative, 'row_id': row.get('row_id'),
                                         'split': row.get('split'), 'title': p.get('title')})
    for name in ['inputs.jsonl', 'test_references.jsonl']:
        f = R10 / name
        source_hashes[str(f.relative_to(REPO))] = sha(f)
        for row in readl(f):
            old_subjects.update(norm(s) for s in row.get('subjects', []))
            if name == 'inputs.jsonl':
                old_questions.append({'file': str(f.relative_to(REPO)), 'row_id': row['row_id'],
                                      'split': row['split'], 'questions': row['questions']})
                for p in row['passages']:
                    old_titles.add(norm(p['title']))
                    key = norm(p['text'])
                    old_texts.add(key)
                    origins[key].append({'file': str(f.relative_to(REPO)), 'row_id': row['row_id'],
                                         'split': row['split'], 'title': p['title']})
            else:
                old_titles.add(norm(row['source_title']))
                old_titles.add(norm(row['donor']['title']))
    poolfile = CUR / 'available_source_pool.jsonl'
    source_hashes[str(poolfile.relative_to(REPO))] = sha(poolfile)
    pool = {r['candidate_id']: r for r in readl(poolfile)}
    rows, file_counts = [], {}
    for name in FILES:
        f = CUR / (name + '.jsonl')
        batch = readl(f) if f.exists() else []
        file_counts[name] = len(batch)
        if f.exists():
            source_hashes[str(f.relative_to(REPO))] = sha(f)
        rows.extend(batch)
    assert len(rows) == len({r['candidate_id'] for r in rows}), 'duplicate candidate IDs'
    old_index = collections.defaultdict(set)
    for text in old_texts:
        for s in shingles(text):
            old_index[s].add(text)
    findings, groups, raw_new_shingles = [], {}, collections.defaultdict(set)
    for r in rows:
        cid = r['candidate_id']
        src = pool[cid]
        text = src['source_content']
        assert hashlib.sha256(text.encode()).hexdigest() == src['source_content_sha256']
        spans = []
        for k in QUOTE_KEYS:
            q = r[k]
            assert text.count(q) == 1, (cid, k, 'nonunique/nonexact')
            st = text.index(q)
            spans.append((st, st + len(q)))
            for s in shingles(q):
                raw_new_shingles[s].add(cid)
        assert all(b <= c or d <= a for i, (a, b) in enumerate(spans) for c, d in spans[i+1:]), cid
        full = ' ' + norm(' '.join(r[k] for k in QUOTE_KEYS)) + ' '
        exact = sorted({x for x in [norm(r['subject']), norm(src['source_title'])]
                        if x in old_subjects | old_titles})
        mentions = sorted(x for x in old_subjects if len(x.split()) >= 2 and ' '+x+' ' in full)
        near = []
        for k in QUOTE_KEYS:
            matches = collections.defaultdict(list)
            for s in shingles(r[k]):
                for old in old_index.get(s, []):
                    matches[old].append(s)
            for old, shared in sorted(matches.items()):
                near.append({'field': k, 'shared_15word_shingles': len(shared),
                             'example': sorted(shared)[0], 'old_text': old,
                             'origins': origins.get(old, [])})
        groups[cid] = event_key(r, src)
        if exact or mentions or near:
            if cid in BLOCKING:
                decision = BLOCKING[cid]['reason']
            elif groups[cid] in OLD_EVENT_LINKS:
                decision = OLD_EVENT_LINKS[groups[cid]]['reason']
            else:
                decision = HIT_EXPLANATIONS.get(cid, 'NEEDS_SEMANTIC_REVIEW')
            findings.append({'candidate_id': cid, 'subject': r['subject'], 'title': src['source_title'],
                             'question': r['question'], 'exact_old_identity_hits': exact,
                             'old_subject_mentions': mentions, 'old_15word_matches': near,
                             'decision': decision})
    pairmatches = collections.defaultdict(list)
    for s, ids in raw_new_shingles.items():
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                pairmatches[(a, b)].append(s)
    newpairs = []
    for (a, b), shared in sorted(pairmatches.items()):
        same = groups[a] == groups[b]
        newpairs.append({'candidate_ids': [a, b], 'shared_15word_shingles': len(shared),
                         'example': sorted(shared)[0], 'same_event_group': same,
                         'review': '同届赛事，成组划分。' if same else
                         '不同主体/赛事：已读句子为选秀/赛制/选美继任或商业销售套话；不自动并组，实际共享实体联系仍由donor阶段复核。'})
    present = {r['candidate_id'] for r in rows}
    event_links = [{'candidate_id': cid, 'event_group': key, **OLD_EVENT_LINKS[key]}
                   for cid, key in groups.items() if key in OLD_EVENT_LINKS]
    unknown = [f['candidate_id'] for f in findings if f['decision'] == 'NEEDS_SEMANTIC_REVIEW']
    connected = collections.defaultdict(list)
    for cid, event in groups.items():
        connected[event].append(cid)
    remaining_background_issue = any(r['candidate_id'] == 'ragognize_train_1806'
                                    and 'Joyce Chitsulo' in r['partial_quote'] for r in rows)
    result = {
        'schema': 'round16-independent-source-isolation-v1',
        'status': 'interim' if len(rows) != 400 else ('needs_review' if unknown else 'reviewed'),
        'scope': 'Only source/question/quote curation; no model generations, labels, scores or training read.',
        'snapshot_files_sha256': source_hashes, 'auditor_source_sha256': sha(Path(__file__)),
        'candidate_counts': file_counts, 'candidate_count': len(rows), 'reviewed_candidate_count': len(rows),
        'quote_count': len(rows) * 3, 'unique_source_count': len({pool[r['candidate_id']]['source_content_sha256'] for r in rows}),
        'old_inventory_counts': {'titles': len(old_titles), 'subjects': len(old_subjects), 'visible_passages': len(old_texts)},
        'blocking_candidate_ids': sorted(present & BLOCKING.keys()),
        'blocking_reasons': {c: BLOCKING[c] for c in sorted(present & BLOCKING.keys())},
        'background_replacement_requested': [{'candidate_id': 'ragognize_train_1806', 'field': 'partial_quote',
                                              'reason': HIT_EXPLANATIONS['ragognize_train_1806']}] if remaining_background_issue else [],
        'resolved_replacements': {'ragognize_train_1291': 'ragognize_test_0531',
                                  'ragognize_train_2163': 'ragognize_test_1072',
                                  'ragognize_train_2244': 'ragognize_train_1736'},
        'resolved_background_edit': 'ragognize_train_1806 now uses birth-year/education original sentences, removing old target Joyce Chitsulo.',
        'old_event_connected_candidate_ids': sorted(r['candidate_id'] for r in event_links),
        'old_event_policy': 'All newly curated members of each old-connected event group must enter train, never new validation/test; different target facts may be retained but are not wholly unseen events.',
        'exclude_from_fresh_test_due_to_old_event_or_specific_material': sorted({r['candidate_id'] for r in event_links} | (present & BLOCKING.keys())),
        'old_event_links': event_links,
        'new_event_groups': groups, 'event_or_entity_group_count': len(set(groups.values())),
        'new_connected_candidates': [{'event_group': key, 'candidate_ids': sorted(ids)}
                                     for key, ids in sorted(connected.items()) if len(ids) > 1],
        'grouping_limit': 'Group count combines entity-only groups and specific event groups; it is not proof of statistical independence. No split has been assigned or audited here.',
        'old_matches': findings, 'unreviewed_match_candidate_ids': unknown,
        'new_shared_material_pairs': newpairs,
        'specific_semantic_checks': {
            'ragognize_test_0640': 'No old Lithuanian Military Commandant organization/date found. Bangladesh police commandant is only a shared generic role, not same entity.',
            'ragognize_train_1882': 'Goma ceasefire/negotiation appeal by Nduhungirehe is not the old Walikale/Pinga mineral-offensive fact. Same broad M23 conflict does not justify automatic grouping.',
            'ragognize_test_2158': 'Federal hiring freeze contractor restriction is not old January27 grant/loan pause.',
            'ragognize_test_2246': 'EO14148 NSA memorandum review is not old grant-pause policy.',
            'ragognize_train_1353': 'EO14183 military medical standards update is not old grant-pause policy.',
            'ragognize_train_1749': 'Jaramana hospital episode near Damascus is a distinct local confrontation from the March coastal clashes.',
            'athlete_biographies': 'Schooling, birthplace and childhood sports initiation remain person groups even if a source mentions later World Cups or Olympics.',
            'generic_templates': 'Different conference tournaments, regional pageants and different film rights deals are not automatically one event merely because 15 words match.'},
        'limitations': ['Semantic source truth/label quality is a separate review; this audit cannot prove absence of every implicit alias.',
                        'This review is bound to the listed candidate hashes; any replacement requires refresh before freezing.',
                        'Donor ownership, cross-split target contamination and displayed paired evidence are assessed separately in paired_source_review.json.',
                        'Old-event candidates may be useful training expansion but cannot substantiate a wholly unseen-event test.'],
    }
    out = CUR / 'isolation_review.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'candidates': len(rows), 'groups': len(set(groups.values())),
                      'blocking': result['blocking_candidate_ids'], 'old_event_or_material_test_exclusions': len(result['exclude_from_fresh_test_due_to_old_event_or_specific_material']),
                      'unreviewed': unknown, 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    run()
