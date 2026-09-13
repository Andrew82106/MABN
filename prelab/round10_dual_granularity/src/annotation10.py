"""Explicit blind annotation of fresh test outputs; reuse exact Round9 serializer.

No prediction reads or automatic correctness labels. Old canonical development
labels stay byte-identical; only fresh test decisions are serialized here.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT.parent/'round9_evidence_binding'
sys.path.insert(0, str(R9/'src'))
import annotation9 as base
base.ROOT = ROOT

readl, sha, packet, add, record, load_rows = base.readl, base.sha, base.packet, base.add, base.record, base.load_rows


def serialize_test(files, output='data/annotations_test.jsonl'):
    inputs, records = load_rows()
    expected = [r for r in inputs if r['split'] == 'test']
    assert len(expected) == 120 and all(r['row_id']+'__1' in records for r in expected)
    allowed = {r['row_id']+'__1' for r in expected}
    sources, seen = {}, {}
    for name in files:
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        doc = json.loads(path.read_text('utf-8'))
        sources[str(path.relative_to(ROOT))] = sha(path)
        for decision in doc['decisions']:
            iid = decision['item_id']
            assert iid in allowed and iid not in seen
            seen[iid] = record(decision, doc['annotator'], records)
    assert set(seen) == allowed, ('Missing test annotations', sorted(allowed-set(seen)))
    target = ROOT/output
    ordered = [seen[r['row_id']+'__1'] for r in expected]
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered), 'utf-8')
    target.with_suffix('.manifest.json').write_text(json.dumps({'decisions_sha256':sources,
        'output_sha256':sha(target),'items':len(ordered)},ensure_ascii=False,indent=2)+'\n','utf-8')
    print('SERIALIZED_TEST',len(ordered),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='stage',required=True)
    q=sub.add_parser('packet');q.add_argument('--start',type=int,default=0);q.add_argument('--count',type=int,default=5)
    s=sub.add_parser('serialize-test');s.add_argument('--files',nargs='+',required=True)
    args=p.parse_args()
    if args.stage=='packet':packet(args.start,args.count,'test')
    else:serialize_test(args.files)
