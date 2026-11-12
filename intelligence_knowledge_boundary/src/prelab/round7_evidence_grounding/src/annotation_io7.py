"""Blind annotation I/O only: no score, feature, or classifier imports.

`packet` displays original evidence and actual outputs for a fixed group slice.
`serialize` copies explicit reviewer decisions into the common label schema.
There are no decisions inferred from the complete/partial condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(split):
    rows = [r for r in readl(ROOT/'data/inputs.jsonl') if r['split'] == split]
    refs = {r['question_id']: r for r in readl(ROOT/'data/references.jsonl')}
    by_group = {}
    for row in rows:
        by_group.setdefault(row['question_id'], []).append(row)
    return [(qid, rr, refs[qid]) for qid, rr in by_group.items()]


def packet(split, start, count):
    result = []
    for index, (qid, rows, ref) in enumerate(groups(split)[start:start+count], start=start):
        item = {'group_index_in_split': index, 'source_index': ref.get('source_index'),
                'question_id': qid, 'questions': rows[0]['questions'],
                'references': ref['items'], 'conditions': []}
        for row in rows:
            path = ROOT/'data/generation_records'/(row['row_id']+'.json')
            if not path.exists():
                raise FileNotFoundError('Generation is not ready: '+str(path))
            generated = json.loads(path.read_text(encoding='utf-8'))
            item['conditions'].append({'row_id': row['row_id'],
                'condition': row['condition'], 'visible_passages': [
                    {'title': p['title'], 'text': p['text']} for p in row['passages']],
                'actual_response': generated['response'], 'items': generated['items'],
                'source_generation_sha256': sha(path)})
        result.append(item)
    return result


def serialize(decisions, output, annotator):
    rows = {r['row_id']: r for r in readl(ROOT/'data/inputs.jsonl')}
    refs = {r['question_id']: r for r in readl(ROOT/'data/references.jsonl')}
    records, seen = [], set()
    for decision in decisions:
        item_id = decision['item_id']
        assert item_id not in seen, item_id
        seen.add(item_id)
        row_id = item_id.rsplit('__', 1)[0]
        row = rows[row_id]
        path = ROOT/'data/generation_records'/(row_id+'.json')
        generated = json.loads(path.read_text(encoding='utf-8'))
        matches = [i for i in generated['items'] if i['item_id'] == item_id]
        assert len(matches) == 1, item_id
        item = matches[0]
        if item['parse_ok']:
            assert generated['response'][item['start']:item['end']] == item['text']
        relation, stance = decision['evidence_relation'], decision['stance']
        assert relation in {'supported', 'unsupported', 'contradicted', 'not_applicable', 'unresolved'}
        assert stance in {'asserted', 'abstained', 'missing', 'tentative'}
        risk = int(relation != 'supported') if stance == 'asserted' and relation in {
            'supported', 'unsupported', 'contradicted'} else None
        assert decision['reference_correctness'] in {'correct', 'incorrect', 'unresolved', 'not_applicable'}
        assert decision['rationale'].strip(), 'Explicit claim review required'
        if 'source_generation_sha256' in decision:
            assert decision['source_generation_sha256'] == sha(path), 'Stale reviewer source'
        if 'text' in decision:
            assert decision['text'] == item['text'], 'Reviewer text mismatch'
        reference = refs[row['question_id']]['items'][item['item_index']-1]
        record = {**decision, **item, 'row_id': row_id, 'question_id': row['question_id'],
            'group_id': row.get('group_id', row['question_id']), 'split': row['split'],
            'dataset': row['dataset'], 'condition': row['condition'], 'risk': risk,
            'reference_evidence': reference['evidence'],
            'visible_source_titles': [p['title'] for p in row['passages']],
            'multi_claim': decision.get('multi_claim', False),
            'annotation_method': 'assistant_manual_source_and_output_review',
            'annotator': annotator, 'source_generation_sha256': sha(path),
            'detector_scores_used': False}
        for span in record.get('risk_spans', []):
            assert generated['response'][span['start']:span['end']] == span['text']
            assert item['start'] <= span['start'] <= span['end'] <= item['end']
        records.append(record)
    path = ROOT/output
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records), encoding='utf-8')
    temporary.replace(path)
    return {'file': str(path), 'items': len(records), 'sha256': sha(path), 'automatic_labeling': False}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    view = sub.add_parser('packet')
    view.add_argument('--split', required=True)
    view.add_argument('--start', type=int, default=0)
    view.add_argument('--count', type=int, default=5)
    write = sub.add_parser('serialize')
    write.add_argument('--decisions', required=True)
    write.add_argument('--output', required=True)
    write.add_argument('--annotator', required=True)
    args = ap.parse_args()
    if args.command == 'packet':
        result = packet(args.split, args.start, args.count)
    else:
        decision_path = ROOT/args.decisions
        decisions = json.loads(decision_path.read_text(encoding='utf-8-sig'))
        result = serialize(decisions, args.output, args.annotator)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
