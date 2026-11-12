"""Independent R16 input/source audit. Reads no generations, labels or scores."""
from pathlib import Path
import collections
import hashlib
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CUR = DATA / 'curation'
FILES = ['quantity_candidates', 'relation_candidates', 'action_time_candidates',
         'root_candidates', 'root_location_extension']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def readj(p):
    return json.loads(p.read_text(encoding='utf-8'))


def readl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def norm(s):
    return re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', s).casefold()).strip()


def shingles(s):
    w = norm(s).split()
    return {' '.join(w[i:i+15]) for i in range(len(w)-14)}


def review_records(doc):
    for key in ('reviews', 'items', 'records', 'decisions', 'entries'):
        if isinstance(doc.get(key), list):
            return doc[key]
    raise ValueError('Unknown manual review schema')


def run():
    input_hash, ref_hash = sha(DATA/'inputs.jsonl'), sha(DATA/'references.jsonl')
    inputs, refs = readl(DATA/'inputs.jsonl'), readl(DATA/'references.jsonl')
    candidates = {r['candidate_id']: r for f in FILES for r in readl(CUR/(f+'.jsonl'))}
    pool = {r['candidate_id']: r for r in readl(CUR/'available_source_pool.jsonl')}
    snapshots = {r['candidate_id']: r for r in readl(DATA/'source_snapshots.jsonl')}
    isolation, exclusions = readj(CUR/'isolation_review.json'), readj(CUR/'donor_exclusions.json')
    issues, observations = [], []
    def check(ok, message, cid=None):
        if not ok:
            issues.append({'candidate_id': cid, 'issue': message})
    check(len(inputs) == 800 and len(refs) == len(candidates) == 400, 'expected 400 paired questions')
    check(isolation['status'] == 'reviewed' and not isolation['blocking_candidate_ids'], 'source isolation incomplete')
    check(set(candidates) == {r['candidate_id'] for r in refs}, 'candidate/reference ID mismatch')
    for f in FILES:
        path = CUR/(f+'.jsonl')
        key = str(path.relative_to(ROOT.parents[1]))
        check(isolation['snapshot_files_sha256'].get(key) == sha(path), 'stale isolation candidate hash '+f)
    byqid = collections.defaultdict(list)
    for row in inputs:
        byqid[row['question_id']].append(row)
    check(len({r['row_id'] for r in inputs}) == len(inputs), 'duplicate input row IDs')
    check(set(byqid) == {r['question_id'] for r in refs}, 'input/reference question mismatch')
    gid_splits, event_splits, subject_splits = [collections.defaultdict(set) for _ in range(3)]
    donor_ids, donor_hashes, source_hashes, gaps = [], [], [], []
    approved = set()
    manual_files = []
    for path in sorted(CUR.glob('paired_donor_review_*.json')):
        doc = readj(path)
        manual_files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        for rr in review_records(doc):
            if rr.get('decision', rr.get('status')) in ('approve', 'approved', 'passed'):
                approved.add((rr['donor_candidate_id'], rr['donor_text_sha256'], rr['donor_source_content_sha256']))
    unseen = []
    old = readj(ROOT.parent/'round10_dual_granularity/data/curation/legacy_isolation_inventory.json')
    old_titles = set(old['old_titles'])
    old_text = set(old['normalized_old_visible_passages'])
    for r in readl(ROOT.parent/'round10_dual_granularity/data/inputs.jsonl'):
        old_titles.update(norm(p['title']) for p in r['passages'])
        old_text.update(norm(p['text']) for p in r['passages'])
    old_grams = set().union(*(shingles(s) for s in old_text))
    new_names = {norm(r['subject']): cid for cid, r in candidates.items() if len(norm(r['subject']).split()) > 1}
    for r in refs:
        cid, qid = r['candidate_id'], r['question_id']
        c, src, donor = candidates[cid], pool[cid], r['donor']
        check(qid == 'r16_'+cid, 'question ID convention', cid)
        check(r['question'] == c['question'] and r['reference_answer'] == c['reference_answer'], 'stale curated question/reference', cid)
        check(r['subject'] == c['subject'] and r['category'] == c['category'], 'curation fields mismatch', cid)
        check(src == snapshots.get(cid), 'source snapshot mismatch', cid)
        check(digest(src['source_content']) == src['source_content_sha256'] == r['source_content_sha256'], 'target source hash', cid)
        check(norm(src['source_title']) not in old_titles, 'old target title', cid)
        source_hashes.append(src['source_content_sha256'])
        for key in ('common_quote', 'evidence_quote', 'partial_quote'):
            span = r['source_spans'][key]
            check(span['text'] == c[key] == src['source_content'][span['start']:span['end']], 'target exact quote/span mismatch '+key, cid)
            check(src['source_content'].count(span['text']) == 1, 'target quote nonunique '+key, cid)
        did = donor['candidate_id']; ds = pool[did]
        donor_ids.append(did); donor_hashes.append(ds['source_content_sha256'])
        check(ds == snapshots.get(did), 'donor source snapshot mismatch', cid)
        check(digest(ds['source_content']) == ds['source_content_sha256'] == donor['source_content_sha256'], 'donor full source hash', cid)
        check(ds['source_content'][donor['start']:donor['end']] == donor['text'] and ds['source_content'].count(donor['text']) == 1, 'donor exact unique quote/span', cid)
        check(donor['title'] == ds['source_title'] and norm(donor['title']) not in old_titles, 'donor title mismatch/old', cid)
        check(did not in exclusions['blocking_donor_candidate_ids'], 'excluded donor ID remains', cid)
        check(not any(norm(f) in norm(donor['title']) for f in exclusions['blocked_title_fragments']), 'excluded donor event title remains', cid)
        check(not shingles(donor['text']) & old_grams, 'old 15-word donor material', cid)
        textnorm = ' '+norm(donor['title']+' '+donor['text'])+' '
        hits = [(name, other) for name, other in new_names.items() if ' '+name+' ' in textnorm]
        check(not hits, 'new target subject mentioned in donor: '+str(hits), cid)
        key = (did, digest(donor['text']), donor['source_content_sha256'])
        if key not in approved:
            unseen.append({'candidate_id': cid, 'donor_candidate_id': did, 'donor_title': donor['title'],
                           'donor_text_sha256': key[1], 'donor_source_content_sha256': key[2]})
        rows = byqid.get(qid, [])
        check(len(rows) == 2 and {x['condition'] for x in rows} == {'complete', 'partial'}, 'two conditions missing', cid)
        material_lengths = []
        target_positions = []
        for row in rows:
            cond = row['condition']
            check(row['split'] == r['split'] and row['group_id'] == r['group_id'], 'reference split/group mismatch', cid)
            check(row['questions'] == [c['question']] and row['subjects'] == [c['subject']], 'question/subject mismatch', cid)
            check(row['expected_items'] == 1 and row['system'] == 'You are a helpful assistant.', 'instruction schema changed', cid)
            expected_target = {'title': src['source_title'], 'text': c['common_quote']+'\n'+c['evidence_quote' if cond == 'complete' else 'partial_quote']}
            expected_donor = {'title': donor['title'], 'text': donor['text']}
            check(len(row['passages']) == 2 and expected_target in row['passages'] and expected_donor in row['passages'], 'displayed source material mismatch', cid)
            if expected_target in row['passages']:
                target_positions.append(row['passages'].index(expected_target))
            rendered = '\n\n'.join('['+str(i+1)+'] '+p['title']+'\n'+p['text'] for i, p in enumerate(row['passages']))
            prompt = 'Please answer the following questions using these search results. Write one short sentence for each numbered item.\n\nQuestions:\n1. '+c['question']+'\n\nSearch results:\n'+rendered
            check(row['prompt'] == prompt, 'neutral prompt changed or hint added', cid)
            material_lengths.append((len(rendered), len(rendered.split())))
            gid_splits[row['group_id']].add(row['split'])
            event_splits[isolation['new_event_groups'][cid]].add(row['split'])
            subject_splits[norm(c['subject'])].add(row['split'])
        check(len(set(target_positions)) == 1, 'source order changed with condition', cid)
        if len(material_lengths) == 2:
            gap = [abs(material_lengths[0][j]-material_lengths[1][j])/max(material_lengths[0][j],material_lengths[1][j]) for j in (0, 1)]
            gaps.append(gap)
            check(max(gap) <= .2+1e-12, 'paired length gap exceeds frozen design', cid)
        if cid in isolation['old_event_connected_candidate_ids']:
            check(r['split'] == 'train', 'old-connected event entered new holdout', cid)
    check(len(set(donor_ids)) == len(donor_ids), 'donor source reused')
    check(len(set(source_hashes)) == len(source_hashes), 'target content reused under another ID')
    check(not set(donor_ids) & set(candidates), 'donor is also target source')
    check(len(set(donor_hashes)) == len(donor_hashes), 'donor content reused under another ID')
    check(not set(donor_hashes) & set(source_hashes), 'donor/target content duplicate')
    for name, mapping in [('group',gid_splits),('event',event_splits),('subject',subject_splits)]:
        check(all(len(s) == 1 for s in mapping.values()), name+' crosses splits')
    check(sha(DATA/'inputs.jsonl') == input_hash and sha(DATA/'references.jsonl') == ref_hash, 'input changed during audit')
    result = {'status': 'passed' if not issues and not unseen else 'needs_review',
              'inputs_sha256': input_hash, 'references_sha256': ref_hash,
              'isolation_review_sha256': sha(CUR/'isolation_review.json'),
              'donor_exclusions_sha256': sha(CUR/'donor_exclusions.json'),
              'source_snapshots_sha256': sha(DATA/'source_snapshots.jsonl'),
              'auditor_source_sha256': sha(Path(__file__)), 'input_rows': len(inputs),
              'paired_questions': len(refs), 'event_or_subject_groups': len(gid_splits),
              'question_split_counts': dict(collections.Counter(r['split'] for r in refs)),
              'donor_semantics_read_count': len(refs)-len(unseen),
              'donor_semantics_reuse': 'Exact donor ID, full-source SHA and displayed-quote SHA must match a prior affirmative manual review; new quotes require fresh reading.',
              'manual_review_files': manual_files, 'unreviewed_current_donors': unseen,
              'issues': issues, 'max_pair_char_gap': max(g[0] for g in gaps),
              'max_pair_word_gap': max(g[1] for g in gaps),
              'scope': 'All raw pairs/sources/hashes/prompts/splits, plus donor semantics. No generation outputs or detector scores read.',
              'limitations': ['Grouping follows requested facts. Shared broad film-festival/league/industry background is not automatically the same question event.',
                             '373 or another computed group count is not proof of statistical independence.',
                             'Source statements are evidence for this experiment, not independently verified real-world truth.',
                             'Partial coverage is not a hallucination label; actual generated claims must still be reviewed.']}
    (CUR/'paired_source_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','input_rows','paired_questions','donor_semantics_read_count','question_split_counts')},ensure_ascii=False))
    print('issues',len(issues),'unread',len(unseen))


if __name__ == '__main__':
    run()
