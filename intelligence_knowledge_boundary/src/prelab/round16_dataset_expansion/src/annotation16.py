"""Explicit answer/span annotation, reusing the validated original coordinate format."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'round9_evidence_binding/src'))
import annotation9 as base
base.ROOT = ROOT
readl, sha, record, load_rows = base.readl, base.sha, base.record, base.load_rows


def add(path, decisions, annotator='root:initial'):
    """Commit a complete annotation document so parallel readers see valid JSON."""
    path=Path(path); path=path if path.is_absolute() else ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_bytes(path.read_bytes() if path.exists() else json.dumps({'annotator':annotator,'decisions':[]}).encode('utf-8'))
    base.add(pending,decisions,annotator)
    pending.replace(path)


def explicit(path, decisions, annotator):
    """Serialize hand-written (row_index, kind, quotes, rationale) judgments.

    This is bookkeeping only: no text-based inference or default decision.
    S=supported; U=unsupported; C=contradicted; A=reviewed safe refusal;
    X=unresolved factual assertion; T=non-assertive/tentative response.
    """
    inputs, _ = load_rows(); out = []
    for index, kind, quotes, rationale in decisions:
        assert kind in ('S', 'U', 'C', 'A', 'X', 'T')
        assert isinstance(index, int) and 0 <= index < len(inputs)
        d = {'item_id': inputs[index]['row_id']+'__1', 'rationale': rationale, 'spans': quotes,
             'stance': 'abstained' if kind == 'A' else 'tentative' if kind == 'T' else 'asserted',
             'evidence_relation': {'S':'supported', 'U':'unsupported', 'C':'contradicted',
                                   'A':'not_applicable', 'X':'unresolved', 'T':'not_applicable'}[kind],
             'risk': 0 if kind == 'S' else 1 if kind in ('U', 'C') else None,
             'localization_status': 'unresolved' if kind == 'X' else 'excluded' if kind in ('A', 'T') else 'resolved',
             'reviewed_safe_refusal': kind == 'A'}
        out.append(d)
    add(path, out, annotator)


def packet(start=0, count=10):
    inputs, rows = load_rows()
    for index, row in enumerate(inputs[start:start+count], start):
        iid = row['row_id']+'__1'
        if iid not in rows:
            print('NOT_YET_GENERATED', index, iid); continue
        _, g, item, _ = rows[iid]
        print('ROW', index, iid)
        print('QUESTION', row['questions'][0])
        for j, p in enumerate(row['passages'], 1):
            print('SOURCE', j, p['title'], '|', p['text'])
        print('ANSWER', json.dumps(item['text'], ensure_ascii=False))
        if not item['parse_ok']:
            print('PARSE_WARNING', json.dumps(item, ensure_ascii=False))


def paired_packet(start=0, count=10):
    """Print repeated visible documents once; preserve each row's document set."""
    inputs, rows = load_rows()
    assert start % 2 == 0 and count % 2 == 0
    for index in range(start, min(start+count, len(inputs)), 2):
        pair = inputs[index:index+2]; assert len(pair)==2 and pair[0]['question_id']==pair[1]['question_id']
        docs=[]
        print('QUESTION', pair[0]['questions'][0])
        for row in pair:
            for d in row['passages']:
                if d not in docs: docs.append(d)
        for j,d in enumerate(docs): print('DOC',j+1,d['title'],'|',d['text'])
        for j,row in enumerate(pair,index):
            iid=row['row_id']+'__1'
            if iid not in rows: print('NOT_YET_GENERATED',j,iid);continue
            _,g,item,_=rows[iid]
            print('ROW',j,iid,'USES_DOCS',[docs.index(d)+1 for d in row['passages']])
            print('ANSWER',json.dumps(item['text'],ensure_ascii=False))
            if not item['parse_ok']: print('PARSE_WARNING',json.dumps(item,ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('packet'); q.add_argument('--start', type=int, default=0); q.add_argument('--count', type=int, default=10); q.add_argument('--paired',action='store_true')
    a = p.parse_args()
    (paired_packet if a.paired else packet)(a.start, a.count)
