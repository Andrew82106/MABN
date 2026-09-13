"""Build reviewed source pairs, grouped splits and a generation input freeze."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import unicodedata

from curation16 import POOL, validate
ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT/'data/curation'
FILES = ['quantity_candidates.jsonl', 'relation_candidates.jsonl', 'action_time_candidates.jsonl',
         'root_candidates.jsonl', 'root_location_extension.jsonl']
TEMPLATE = 'Please answer the following questions using these search results. Write one short sentence for each numbered item.\n\nQuestions:\n1. {question}\n\nSearch results:\n{sources}'
spec = importlib.util.spec_from_file_location('r16_source_sentences', ROOT.parent/'round9_evidence_binding/src/build_inputs9.py')
r9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r9)


def readl(path):
    return [json.loads(s) for s in path.read_text('utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')


def savel(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), 'utf-8')


def norm(text):
    return re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def phrases(text, n=15):
    words = norm(text).split()
    return {' '.join(words[j:j+n]) for j in range(len(words)-n+1)}


def present(term, text):
    return ' '+norm(term)+' ' in ' '+norm(text)+' '


def legacy():
    p = ROOT.parent/'round10_dual_granularity/data/curation/legacy_isolation_inventory.json'
    doc = json.loads(p.read_text('utf-8'))
    current = ROOT.parent/'round10_dual_granularity/data/inputs.jsonl'
    old = readl(current)
    titles = set(doc['old_titles']) | {norm(s['title']) for r in old for s in r['passages']}
    subjects = set(doc['old_subjects']) | {norm(s) for r in old for s in r.get('subjects', [])}
    texts = doc['normalized_old_visible_passages'] + [s['text'] for r in old for s in r['passages']]
    grams = set().union(*(phrases(t) for t in texts))
    return titles, subjects, grams


class Groups:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]; a = self.parent[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def grouped(rows, review):
    ids = [r['candidate_id'] for r in rows]; groups = Groups(ids)
    keys = defaultdict(list)
    overrides = review.get('new_event_groups', {})
    if isinstance(overrides, list):
        overrides = {r['candidate_id']: r['event_group'] for r in overrides}
    for row in rows:
        cid = row['candidate_id']
        keys['subject:'+norm(row['subject'])].append(cid)
        event = overrides.get(cid, row.get('event_group'))
        if event:
            keys['event:'+norm(event)].append(cid)
    for values in keys.values():
        for cid in values[1:]:
            groups.union(values[0], cid)
    # Curated cross-source entity/event connections from the source-only review.
    for relation in review.get('new_connected_candidates', []):
        values = relation.get('candidate_ids', []) if isinstance(relation, dict) else relation
        values = [i for i in values if i in groups.parent]
        for cid in values[1:]:
            groups.union(values[0], cid)
    return {cid: 'r16_group_'+digest(groups.find(cid))[:12] for cid in ids}


def split_groups(rows, group_ids, seed, forced_train_ids=()):
    members = defaultdict(list)
    for r in rows:
        members[group_ids[r['candidate_id']]].append(r)
    # Deterministic grouped greedy allocation, with no outcomes or labels.
    rng = random.Random(seed); order = sorted(members); rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    totals = Counter(); cat_counts = {s: Counter() for s in ('train', 'validation', 'test')}
    goal = {'train': .75*len(rows), 'validation': .125*len(rows), 'test': .125*len(rows)}
    full_cat = Counter(r['category'] for r in rows); assigned = {}
    forced = {group_ids[c] for c in forced_train_ids if c in group_ids}
    order.sort(key=lambda g: g not in forced)
    for gid in order:
        cc = Counter(r['category'] for r in members[gid]); n = len(members[gid])
        def cost(split):
            ratio = goal[split]/len(rows)
            # Change in normalized squared deviation from split and category goals.
            v = ((totals[split]+n-goal[split])**2-(totals[split]-goal[split])**2)/max(goal[split], 1)
            for cat, k in cc.items():
                target = full_cat[cat]*ratio
                v += ((cat_counts[split][cat]+k-target)**2-(cat_counts[split][cat]-target)**2)/max(target, 1)
            return v, ('train', 'validation', 'test').index(split)
        chosen = 'train' if gid in forced else min(goal, key=cost)
        assigned[gid] = chosen; totals[chosen] += n; cat_counts[chosen].update(cc)
    return assigned, {'question_counts': dict(totals), 'event_group_counts': dict(Counter(assigned.values())),
                       'categories': {s: dict(c) for s, c in cat_counts.items()},
                       'old_event_connected_train_groups': len(forced),
                       'old_event_connected_train_questions': sum(len(members[g]) for g in forced)}


def donor_options(source, old_subjects, old_grams, new_subjects):
    text = source['source_content']; spans = r9.sentence_spans(text)
    if not spans:
        return []
    left = spans[0][0]; options = []
    terms = [' '+norm(t)+' ' for t in old_subjects if len(t.split()) > 1]
    terms += [' '+norm(t)+' ' for t in new_subjects if len(norm(t)) >= 4]
    for _, right in spans:
        quote = text[left:right].strip(); words = len(quote.split())
        if words > 550:
            break
        if words < 25 or text.count(quote) != 1:
            continue
        visible = ' '+norm(source['source_title']+' '+quote)+' '
        if any(t in visible for t in terms):
            break
        if phrases(quote) & old_grams:
            break
        options.append(quote)
    return options


def context(row, source, donor, quote, condition):
    k = 'evidence_quote' if condition == 'complete' else 'partial_quote'
    a = {'title': source['source_title'], 'text': row['common_quote']+'\n'+row[k]}
    b = {'title': donor['source_title'], 'text': quote}
    # Same deterministic source order in the two conditions, independent of answer.
    return [a, b] if int(digest(row['candidate_id'])[:8], 16) % 2 == 0 else [b, a]


def material(passages):
    return '\n\n'.join(f"[{j+1}] {p['title']}\n{p['text']}" for j, p in enumerate(passages))


def run(freeze=False):
    assert not (ROOT/'data/input_freeze.json').exists(), 'Frozen input build is immutable'
    config = json.loads((ROOT/'protocol.json').read_text('utf-8'))
    review_path = CUR/'isolation_review.json'
    review = json.loads(review_path.read_text('utf-8')) if review_path.exists() else {}
    candidates = [validate(r) for name in FILES if (CUR/name).exists() for r in readl(CUR/name)]
    blocked = set(review.get('blocking_candidate_ids', []))
    candidates = [r for r in candidates if r['candidate_id'] not in blocked]
    candidates.sort(key=lambda r: (r['category'], r['candidate_id']))
    assert len({r['candidate_id'] for r in candidates}) == len(candidates)
    if freeze:
        assert len(candidates) == config['target_new_questions']
        assert review.get('status') in ('passed', 'reviewed', 'ready_for_assembly')
        assert review.get('reviewed_candidate_count') == len(candidates)
    old_titles, old_subjects, old_grams = legacy()
    for row in candidates:
        source = POOL[row['candidate_id']]
        assert norm(source['source_title']) not in old_titles
        assert digest(source['source_content']) == source['source_content_sha256']
    group_ids = grouped(candidates, review)
    split_map, split_stats = split_groups(candidates, group_ids, config['seed'], review.get('old_event_connected_candidate_ids', []))
    donors = readl(CUR/'donor_reserved_pool.jsonl')
    if donors and 'source_content' not in donors[0]:
        donors = [POOL[r['candidate_id']] for r in donors]
    exclusions_path = CUR/'donor_exclusions.json'
    exclusions = json.loads(exclusions_path.read_text('utf-8')) if exclusions_path.exists() else {}
    blocked_donors = set(exclusions.get('blocking_donor_candidate_ids', []))
    blocked_fragments = [norm(t) for t in exclusions.get('blocked_title_fragments', [])]
    donors = [d for d in donors if d['candidate_id'] not in blocked_donors and
              not any(t in norm(d['source_title']) for t in blocked_fragments)]
    rng = random.Random(config['seed']); rng.shuffle(donors)
    new_subjects = [r['subject'] for r in candidates]
    options = {d['candidate_id']: donor_options(d, old_subjects, old_grams, new_subjects) for d in donors
               if norm(d['source_title']) not in old_titles}
    preference_path = CUR/'pair_preferences.json'
    preferences = json.loads(preference_path.read_text('utf-8')).get('pairs', {}) if preference_path.exists() else {}
    by_donor = {d['candidate_id']:d for d in donors}; preferred = {}
    for row in candidates:
        cid = row['candidate_id']; old_pair = preferences.get(cid, {})
        did = old_pair.get('donor_candidate_id'); quote = old_pair.get('quote')
        if did not in by_donor or quote not in options.get(did, []): continue
        d=by_donor[did]; source=POOL[cid]
        aa,bb=[material(context(row,source,d,quote,c)) for c in ('complete','partial')]
        cg=abs(len(aa)-len(bb))/max(len(aa),len(bb))
        aw,bw=len(aa.split()),len(bb.split());wg=abs(aw-bw)/max(aw,bw)
        if max(cg,wg)<=.20: preferred[cid]=(d,quote,cg,wg)
    reserved={v[0]['candidate_id'] for v in preferred.values()}
    assert len(reserved)==len(preferred)
    used = set(); inputs, references, pair_stats, missing = [], [], [], []
    snapshots = {}
    # Hardest length-matching pairs first; no generated outputs exist at this stage.
    work = sorted(candidates, key=lambda r: -abs(len(r['evidence_quote'])-len(r['partial_quote'])))
    for row in work:
        cid = row['candidate_id']; source = POOL[cid]; choices = []
        for d in ([] if cid in preferred else donors):
            did = d['candidate_id']
            if did in used or did in reserved:
                continue
            for quote in options.get(did, []):
                a, b = [context(row, source, d, quote, cond) for cond in ('complete', 'partial')]
                aa, bb = material(a), material(b)
                cg = abs(len(aa)-len(bb))/max(len(aa), len(bb))
                aw, bw = len(aa.split()), len(bb.split()); wg = abs(aw-bw)/max(aw, bw)
                if max(cg, wg) <= .20:
                    choices.append((int(source['official_category'] != d['official_category']),
                                    len(quote.split()), digest(cid+did), d, quote, cg, wg))
                    break
        if cid in preferred:
            d,quote,cg,wg=preferred[cid]
        elif not choices:
            missing.append(cid); continue
        else:
            _, _, _, d, quote, cg, wg = min(choices, key=lambda x: x[:3])
        used.add(d['candidate_id'])
        gid = group_ids[cid]; split = split_map[gid]; qid = 'r16_'+cid
        for condition in ('complete', 'partial'):
            passages = context(row, source, d, quote, condition)
            inputs.append({'row_id': qid+'__'+condition, 'question_id': qid, 'group_id': gid,
                           'split': split, 'condition': condition, 'questions': [row['question']],
                           'subjects': [row['subject']], 'passages': passages,
                           'system': 'You are a helpful assistant.',
                           'prompt': TEMPLATE.format(question=row['question'], sources=material(passages)),
                           'dataset': 'RAGognize_Qwen_expansion_round16', 'expected_items': 1, 'category': row['category']})
        spans = {}
        for k in ('common_quote', 'evidence_quote', 'partial_quote'):
            pos = source['source_content'].index(row[k]); spans[k] = {'text': row[k], 'start': pos, 'end': pos+len(row[k])}
        dp = d['source_content'].index(quote)
        references.append({'question_id': qid, 'group_id': gid, 'split': split, 'category': row['category'],
                           'candidate_id': cid, 'subject': row['subject'], 'question': row['question'],
                           'reference_answer': row['reference_answer'], 'source_title': source['source_title'],
                           'source_url': source['source_url'], 'revision_id': source['revision_id'],
                           'source_content_sha256': source['source_content_sha256'], 'source_spans': spans,
                           'donor': {'candidate_id': d['candidate_id'], 'title': d['source_title'], 'text': quote,
                                     'url': d['source_url'], 'revision_id': d['revision_id'],
                                     'source_content_sha256': d['source_content_sha256'], 'start': dp, 'end': dp+len(quote)},
                           'coverage': {'complete': True, 'partial': False},
                           'coverage_is_not_generated_risk_label': True, 'review_note': row['review_note']})
        pair_stats.append({'candidate_id': cid, 'group_id': gid, 'split': split, 'relative_char_gap': cg, 'relative_word_gap': wg})
        snapshots[cid] = source; snapshots[d['candidate_id']] = d
    inputs.sort(key=lambda r: (('train', 'validation', 'test').index(r['split']), r['question_id'], r['condition']))
    references.sort(key=lambda r: r['question_id'])
    report = {'status': 'draft_not_frozen', 'candidate_count': len(candidates), 'blocked_candidates': sorted(blocked),
              'paired_questions': len(references), 'rows': len(inputs), 'event_or_subject_groups': len(set(group_ids.values())),
              'splits': split_stats, 'category_counts': dict(Counter(r['category'] for r in candidates)),
              'missing_donor_candidates': missing, 'preserved_reviewed_pairs':len(preferred),
              'eligible_donor_sources': sum(bool(v) for v in options.values()),
              'max_relative_char_gap': max((p['relative_char_gap'] for p in pair_stats), default=0),
              'max_relative_word_gap': max((p['relative_word_gap'] for p in pair_stats), default=0)}
    savel(ROOT/'data/inputs.jsonl', inputs); savel(ROOT/'data/references.jsonl', references)
    savel(ROOT/'data/source_snapshots.jsonl', list(snapshots.values()))
    savel(CUR/'pair_statistics.jsonl', pair_stats)
    save(CUR/'assembly_report.json', report)
    if freeze:
        assert not missing and len(inputs) == 2*config['target_new_questions']
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ROOT.parent/'models/Qwen2.5-7B-Instruct-bnb-4bit', local_files_only=True)
        lengths = [len(tok.apply_chat_template([{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': r['prompt']}], tokenize=True, add_generation_prompt=True)) for r in inputs]
        assert max(lengths) <= 3072
        paired = json.loads((CUR/'paired_source_review.json').read_text('utf-8'))
        assert paired['status'] == 'passed' and paired['inputs_sha256'] == sha(ROOT/'data/inputs.jsonl')
        paths = ['data/inputs.jsonl', 'data/references.jsonl', 'data/source_snapshots.jsonl', 'protocol.json',
                 'data/curation/isolation_review.json', 'data/curation/paired_source_review.json',
                 'data/curation/donor_exclusions.json', 'data/curation/pair_preferences.json', 'data/legacy_files_snapshot.json',
                 'src/generate16.py', 'src/assemble16.py', 'src/curation16.py']
        paths += [p.relative_to(ROOT).as_posix() for p in sorted(CUR.glob('source_semantic_*.json'))]
        paths += ['data/curation/'+n for n in FILES]
        save(ROOT/'data/input_freeze.json', {'status': 'frozen_before_generation', 'questions': len(references),
                                           'rows': len(inputs), 'groups': report['event_or_subject_groups'],
                                           'min_prompt_tokens': min(lengths), 'max_prompt_tokens': max(lengths),
                                           'files_sha256': {n: sha(ROOT/n) for n in paths}})
        report['status']='frozen_before_generation'; save(CUR/'assembly_report.json',report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true')
    a = p.parse_args(); run(a.freeze)
