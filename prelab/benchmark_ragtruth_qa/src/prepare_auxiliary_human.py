"""Stage official TRAIN-only non-QA human spans as optional auxiliary data.

Never parses an official test response or its quality/labels. Source material
and split identities are used only for conservative overlap quarantine. This
does not change any existing training list, calibration list, or test release.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / 'data/raw'
OUT = ROOT / 'auxiliary_human_v1'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]


def write_rows(name, rows):
    with (OUT / name).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def norm(text):
    return re.findall(r'\w+', text.casefold())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def evidence_parts(row):
    info = row['source_info']
    if row['task_type'] == 'QA':
        return [s for s in re.split(r'(?:^|\n\s*\n)passage\s+\d+:', info['passages'], flags=re.I) if s.strip()]
    return list(strings(info))


def model_input(row):
    prompt = row['prompt']
    if row['task_type'] == 'Summary':
        evidence = row['source_info']
        start = prompt.find(evidence)
        assert start >= 0 and prompt.count(evidence) == 1
        return prompt[:start].rstrip(), evidence, [start, start + len(evidence)]
    assert row['task_type'] == 'Data2txt'
    marker = 'Structured data:\n'
    assert prompt.count(marker) == 1 and prompt.endswith('\nOverview:')
    begin = prompt.index(marker) + len(marker)
    end = len(prompt) - len('\nOverview:')
    evidence = prompt[begin:end]
    # Preserve the actual published rendering, not a newly serialized dict.
    import ast
    assert ast.literal_eval(evidence) == row['source_info']
    return prompt[:begin].rstrip(), evidence, [begin, end]


def main():
    assert not (OUT / 'started.json').exists(), 'Do not overwrite an existing data staging run'
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [RAW / 'source_info.jsonl', RAW / 'response.jsonl', RAW / 'LICENSE',
              ROOT / 'data/source_index.jsonl', Path(__file__)]
    source_hashes = {str(p.resolve()): sha(p) for p in inputs}
    save('started.json', {'time': time.time(), 'sources_sha256': source_hashes})
    sources = {s['source_id']: s for s in lines(RAW / 'source_info.jsonl')}
    sid_rx = re.compile(r'"source_id"\s*:\s*"([^"\\]+)"')
    split_rx = re.compile(r'"split"\s*:\s*"([^"\\]+)"')
    split = {}
    # Identity-only pass: no JSON parsing of responses from either split.
    for line in (RAW / 'response.jsonl').open(encoding='utf-8'):
        sid, part = sid_rx.search(line).group(1), split_rx.search(line).group(1)
        assert part in ('train', 'test')
        assert split.setdefault(sid, part) == part
    qa_index = {s['source_id']: s for s in lines(ROOT / 'data/source_index.jsonl')}
    blocked = {sid for sid, part in split.items() if part == 'test'}
    blocked.update(sid for sid, s in qa_index.items() if s['partition'] != 'fit')
    # Link source documents using exact material and consecutive20-word reuse.
    # For structured data, compare values/reviews rather than schema boilerplate.
    parent = {sid: sid for sid in sources}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    business_seen = {}
    edge_reasons = defaultdict(set)
    for sid, source in sources.items():
        for part in evidence_parts(source):
            words = norm(part)
            if len(words) < 20:
                continue
            keys = {digest(' '.join(words[i:i+20])) for i in range(len(words) - 19)}
            for key in keys:
                previous = seen.setdefault(key, sid)
                if previous != sid:
                    union(previous, sid)
                    edge_reasons[tuple(sorted((previous, sid)))].add('shared_consecutive20_words')
        if source['task_type'] == 'Data2txt':
            info = source['source_info']
            key = digest(json.dumps([norm(str(info.get(k, ''))) for k in ('name', 'address', 'city', 'state')]))
            previous = business_seen.setdefault(key, sid)
            if previous != sid:
                union(previous, sid)
                edge_reasons[tuple(sorted((previous, sid)))].add('same_business_name_address_city_state')
    groups = defaultdict(list)
    for sid in sources:
        groups[find(sid)].append(sid)
    group_id = {sid: 'rtaux_group_' + digest('|'.join(sorted(members)))[:16]
                for members in groups.values() for sid in members}
    forbidden = {group_id[sid] for sid in blocked}
    allowed = {sid for sid, s in sources.items() if s['task_type'] in ('Summary', 'Data2txt')
               and split[sid] == 'train' and group_id[sid] not in forbidden}
    candidates = {sid for sid, s in sources.items() if s['task_type'] in ('Summary', 'Data2txt') and split[sid] == 'train'}
    rows = []
    exclusions = []
    issues = []
    dedup = {}
    duplicates = []
    parsed = Counter()
    for line in (RAW / 'response.jsonl').open(encoding='utf-8'):
        sid = sid_rx.search(line).group(1)
        # Exclude before parsing. Withheld answer contents and gold stay unopened.
        if sid not in allowed or split_rx.search(line).group(1) != 'train':
            continue
        response = json.loads(line)
        assert response['split'] == 'train' and response['source_id'] == sid
        source = sources[sid]
        parsed[source['task_type']] += 1
        if response['quality'] != 'good':
            exclusions.append({'response_id': response['id'], 'source_id': sid,
                               'quality': response['quality'], 'reason': 'official_quality_not_good'})
            continue
        question, evidence, evidence_range = model_input(source)
        answer = response['response']
        for i, label in enumerate(response['labels']):
            if not (isinstance(label['start'], int) and isinstance(label['end'], int)
                    and 0 <= label['start'] < label['end'] <= len(answer)
                    and answer[label['start']:label['end']] == label['text']):
                issues.append({'response_id': response['id'], 'label_index': i, 'reason': 'original_span_mismatch'})
        key = digest(json.dumps([sid, answer, response['labels']], ensure_ascii=False, sort_keys=True))
        if key in dedup:
            duplicates.append({'response_id': response['id'], 'retained_response_id': dedup[key], 'source_id': sid,
                               'reason': 'identical_answer_and_complete_labels_in_same_source'})
            continue
        dedup[key] = response['id']
        rows.append({'source_id': sid, 'group_id': group_id[sid], 'response_id': response['id'],
                     'partition': 'auxiliary_candidate_fit', 'official_split': 'train',
                     'task_type': source['task_type'], 'source_dataset': source['source'],
                     'model': response['model'], 'temperature': response['temperature'],
                     'quality': response['quality'], 'question': question, 'retrieved_passages': evidence,
                     'released_prompt': source['prompt'], 'evidence_original_prompt_range': evidence_range,
                     'original_response': answer, 'labels': response['labels'],
                     'answer_sha256': digest(answer), 'prompt_sha256': digest(source['prompt']),
                     'annotation_origin': 'RAGTruth released human span annotation',
                     'new_labels_generated': False, 'currently_used_in_training': False})
    # Preserve any malformed cases in staging; do not silently repair/filter them.
    rows.sort(key=lambda r: (r['source_id'], r['response_id']))
    write_rows('candidate_fit.jsonl', rows)
    write_rows('quality_excluded_index.jsonl', exclusions)
    write_rows('duplicate_index.jsonl', duplicates)
    write_rows('quarantined_source_index.jsonl', [
        {'source_id': sid, 'group_id': group_id[sid], 'task_type': sources[sid]['task_type'],
         'official_split': 'train', 'reason': 'material_group_touches_official_test_or_QA_nonfit_source'}
        for sid in sorted(candidates - allowed)])
    write_rows('overlap_edge_index.jsonl', [{'source_a': a, 'source_b': b, 'reasons': sorted(reasons)}
                                          for (a, b), reasons in sorted(edge_reasons.items())])
    counts = {}
    for task in ('Summary', 'Data2txt'):
        selected = [r for r in rows if r['task_type'] == task]
        counts[task] = {'answers': len(selected), 'sources': len({r['source_id'] for r in selected}),
                        'groups': len({r['group_id'] for r in selected}),
                        'risk_answers': sum(bool(r['labels']) for r in selected),
                        'span_types': dict(Counter(l['label_type'] for r in selected for l in r['labels']))}
    save('SPAN_ALIGNMENT_ISSUES.json', {'count': len(issues), 'issues': issues})
    artifacts = ['candidate_fit.jsonl', 'quality_excluded_index.jsonl', 'duplicate_index.jsonl',
                 'quarantined_source_index.jsonl', 'overlap_edge_index.jsonl', 'SPAN_ALIGNMENT_ISSUES.json']
    manifest = {
        'status': 'staged_not_used_in_training' if not issues else 'staged_alignment_review_required',
        'purpose': 'Optional source-expanding human-supervised auxiliary tasks; final task stays unchanged QA.',
        'counts': counts, 'candidate_train_sources_before_quarantine': len(candidates),
        'quarantined_train_sources': len(candidates - allowed), 'unique_sources': len({r['source_id'] for r in rows}),
        'unique_groups': len({r['group_id'] for r in rows}), 'answers': len(rows),
        'parsed_train_rows_by_task': dict(parsed), 'quality_excluded_rows': len(exclusions),
        'same_source_exact_answer_gold_duplicates_removed': len(duplicates), 'span_alignment_issues': len(issues),
        'labels': 'All original human types/flags/meta/offsets retained unchanged; no synthetic or new human labels.',
        'split_guard': 'Identity-only regex pass; official test and quarantined response JSON never parsed. All official test source materials and all QA non-fit source materials are blocked for auxiliary source-overlap grouping.',
        'overlap_rule': 'Source material consecutive20 normalized Unicode-word overlap, plus exact Data2txt business name/address/city/state. Existing QA non-fit identities include all linked/test/withheld sources.',
        'limits': ['Material overlap check is conservative and not an exhaustive entity/event paraphrase audit.',
                   'Summary and Data2txt are auxiliary detection tasks, not new QA questions or a new QA test.',
                   'Structured-source unknown fields are preserved; original human unsupported/conflict labels are not reinterpreted.',
                   'Repeated generators do not multiply independent source count.',
                   'No training budget, task weights, or improvement claim is established by this data staging step.'],
        'official_test_answers_or_gold_parsed': False, 'existing_fit_calibration_changed': False,
        'sources_sha256': source_hashes, 'artifacts_sha256': {name: sha(OUT / name) for name in artifacts}}
    assert source_hashes == {str(p.resolve()): sha(p) for p in inputs}
    save('manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'counts', 'answers', 'unique_sources', 'unique_groups',
                                             'quarantined_train_sources', 'span_alignment_issues')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
