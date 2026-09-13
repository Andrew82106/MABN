"""Save source-reviewed question candidates; validates quotes, not semantics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = {r['candidate_id']: r for r in map(json.loads, (ROOT/'data/curation/available_source_pool.jsonl').read_text('utf-8').splitlines())}


def validate(row):
    src = POOL[row['candidate_id']]['source_content']; ranges = []
    for k in ('common_quote', 'evidence_quote', 'partial_quote'):
        quote = row[k]
        assert quote and src.count(quote) == 1, (row['candidate_id'], k, 'quote not unique verbatim')
        start = src.index(quote); ranges.append((start, start+len(quote)))
    ranges.sort()
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), (row['candidate_id'], 'overlapping quotes')
    assert row['category'] in ('time', 'quantity', 'relation', 'action', 'location')
    assert all(row[k] for k in ('question', 'subject', 'reference_answer'))
    assert row['question'].endswith('?')
    return row


def save_candidates(filename, new_rows):
    path = ROOT/'data/curation'/filename
    old = [json.loads(s) for s in path.read_text('utf-8').splitlines()] if path.exists() else []
    known = {r['candidate_id'] for r in old}
    assert not known & {r['candidate_id'] for r in new_rows}
    for row in new_rows:
        validate(row)
    rows = old + new_rows
    assert len({r['candidate_id'] for r in rows}) == len(rows)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), 'utf-8')
    print(filename, 'total', len(rows))


def candidate(cid, category, subject, question, answer, common, evidence, partial, note='Partial keeps background but omits the requested fact.'):
    return validate(dict(candidate_id=cid, category=category, subject=subject, question=question,
                         reference_answer=answer, common_quote=common, evidence_quote=evidence,
                         partial_quote=partial, review_note=note, status='accepted'))
