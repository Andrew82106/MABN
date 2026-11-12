"""Blind packets and exact serialization of explicit annotator decisions only."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(x) for x in Path(path).read_text('utf-8-sig').splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows():
    inputs = readl(ROOT/'data/inputs.jsonl')
    rows = {}
    for row in inputs:
        p = ROOT/'data/generation_records'/(row['row_id']+'.json')
        if p.exists():
            g = json.loads(p.read_text('utf-8'))
            assert g['row_id'] == row['row_id']
            for item in g['items']:
                rows[item['item_id']] = row, g, item, sha(p)
    return inputs, rows


def packet(start=0, count=5, split=None):
    inputs, rows = load_rows()
    selected = [r for r in inputs if split is None or r['split'] == split][start:start+count]
    for index, row in enumerate(selected, start):
        iid = row['row_id']+'__1'
        if iid not in rows:
            print('NOT_YET_GENERATED', index, row['row_id'])
            continue
        _, g, item, _ = rows[iid]
        print('ROW', index, iid)
        print('QUESTION', row['questions'][0])
        for j, p in enumerate(row['passages'], 1):
            print('SOURCE', j, p['title'], '|', p['text'])
        print('ANSWER', json.dumps(item, ensure_ascii=False))
        print('FULL_RESPONSE', json.dumps(g['response'], ensure_ascii=False))


def locate(text, quote, occurrence=None):
    assert isinstance(quote, str) and quote.strip() and quote == quote.strip()
    found = []; at = 0
    while True:
        at = text.find(quote, at)
        if at < 0:
            break
        found.append(at); at += 1
    assert found, 'Missing explicit quote: '+quote
    if occurrence is None:
        assert len(found) == 1, 'Quote repeated; specify occurrence: '+quote
        return found[0]
    assert type(occurrence) is int and 0 <= occurrence < len(found)
    return found[occurrence]


def record(decision, annotator, rows):
    row, g, item, digest = rows[decision['item_id']]
    for key in ('text', 'start', 'end'):
        assert decision[key] == item[key], 'Decision no longer matches original answer: '+key
    assert decision['source_generation_sha256'] == digest, 'Original annotation generation changed'
    if item['start'] is not None:
        assert g['response'][item['start']:item['end']] == item['text']
    stance = decision['stance']; relation = decision['evidence_relation']; risk = decision['risk']
    assert stance in {'asserted', 'abstained', 'tentative', 'missing'}
    assert relation in {'supported', 'unsupported', 'contradicted', 'unresolved', 'not_applicable'}
    assert risk in (0, 1, None)
    status = decision.get('localization_status', 'resolved' if stance == 'asserted' and risk is not None else 'excluded')
    assert status in {'resolved', 'excluded', 'unresolved'}
    assert decision['rationale'].strip()
    if status == 'resolved':
        assert stance == 'asserted' and item['parse_ok'] and item['start'] is not None
        assert (risk == 0 and relation == 'supported') or (risk == 1 and relation in {'unsupported', 'contradicted'})
    if stance != 'asserted':
        assert risk is None and status == 'excluded'
    if status == 'unresolved':
        assert risk is None
    spans = []
    for n, raw in enumerate(decision.get('spans', [])):
        s = {'quote': raw} if isinstance(raw, str) else raw
        pos = locate(item['text'], s['quote'], s.get('occurrence'))
        start, end = item['start']+pos, item['start']+pos+len(s['quote'])
        assert g['response'][start:end] == s['quote']
        assert any(c.isalnum() for c in s['quote'])
        spans.append({'start': start, 'end': end, 'text': s['quote'],
                      'claim_id': item['item_id']+f'__claim_{n+1}',
                      'rationale': s.get('rationale', decision['rationale']),
                      'evidence_refs': s.get('evidence_refs', decision.get('evidence_refs', []))})
    spans.sort(key=lambda s: (s['start'], s['end']))
    assert all(a['end'] <= b['start'] for a, b in zip(spans, spans[1:]))
    assert bool(spans) == bool(risk == 1)
    result = {k: row[k] for k in ('row_id', 'question_id', 'group_id', 'split', 'condition')}
    result.update({k: item[k] for k in ('item_id', 'text', 'start', 'end')})
    result.update(source_generation_sha256=digest, original_risk=risk, original_stance=stance,
                  evidence_relation=relation, localization_status=status,
                  claim_scope=[{k: item[k] for k in ('start', 'end', 'text')}], risk_spans=spans,
                  rationale=decision['rationale'], annotator=annotator, token_scores_viewed=False,
                  human_gold=False)
    return result


def add(path, decisions, annotator='root_initial'):
    """Append explicit decisions after validating actual generation and coordinates."""
    path = Path(path); path = path if path.is_absolute() else ROOT/path
    _, rows = load_rows()
    old = json.loads(path.read_text('utf-8')) if path.exists() else {'annotator': annotator, 'decisions': []}
    assert old['annotator'] == annotator
    ids = {d['item_id'] for d in old['decisions']}
    for d in decisions:
        d = dict(d)
        assert d['item_id'] not in ids
        _, _, item, digest = rows[d['item_id']]
        for key in ('text', 'start', 'end'):
            assert d.get(key, item[key]) == item[key]
            d[key] = item[key]
        assert d.get('source_generation_sha256', digest) == digest
        d['source_generation_sha256'] = digest
        record(d, annotator, rows)
        ids.add(d['item_id']); old['decisions'].append(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print('EXPLICIT_ANNOTATION_DECISIONS', path.name, len(old['decisions']))


def serialize(files, output):
    inputs, rows = load_rows(); seen = {}; sources = {}
    assert len(rows) == len(inputs), 'Generation not complete'
    for name in files:
        p = Path(name); p = p if p.is_absolute() else ROOT/p
        doc = json.loads(p.read_text('utf-8'))
        sources[str(p.relative_to(ROOT))] = sha(p)
        for d in doc['decisions']:
            assert d['item_id'] not in seen, 'Overlapping annotator assignments'
            seen[d['item_id']] = record(d, doc['annotator'], rows)
    assert set(seen) == set(rows), ('Missing decisions', set(rows)-set(seen))
    path = Path(output); path = path if path.is_absolute() else ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [seen[r['row_id']+'__1'] for r in inputs]
    path.write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in ordered), 'utf-8')
    path.with_suffix('.manifest.json').write_text(json.dumps({'decisions_sha256': sources, 'output_sha256': sha(path), 'items':len(ordered)}, indent=2)+'\n', 'utf-8')
    print('SERIALIZED_INITIAL', len(ordered))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('packet'); q.add_argument('--start', type=int, default=0); q.add_argument('--count', type=int, default=5); q.add_argument('--split')
    q = sub.add_parser('serialize'); q.add_argument('--files', nargs='+', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.stage == 'packet': packet(a.start, a.count, a.split)
    else: serialize(a.files, a.output)
