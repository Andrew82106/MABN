"""Persist root's explicit independent item judgments, never infer labels."""
from pathlib import Path
import json
from datetime import datetime, timezone
from annotation_io7 import ROOT, groups, sha

CODES = {
    'S': ('asserted', 'supported', 'correct'),
    'U': ('asserted', 'unsupported', 'correct'),
    'I': ('asserted', 'unsupported', 'incorrect'),
    'N': ('asserted', 'unsupported', 'unresolved'),
    'C': ('asserted', 'contradicted', 'incorrect'),
    'A': ('abstained', 'not_applicable', 'not_applicable'),
    'X': ('asserted', 'unresolved', 'unresolved'),
}


def append(split, judgments):
    """Judgment: (group index, six explicit codes, six actual-claim rationales)."""
    path = ROOT/f'data/annotation_reviews/{split}_root_independent_review.json'
    content = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'reviewer': '/root', 'initial_labels_seen': False, 'detector_scores_seen': False,
        'human_gold': False, 'items': []}
    old = {i['item_id']: i for i in content['items']}
    gg = groups(split)
    for judgment in judgments:
        index, codes, notes = judgment[:3]
        multi_claim_indices = set(judgment[3]) if len(judgment) > 3 else set()
        qid, rows, ref = gg[index]
        assert len(codes) == len(notes) == 6
        n = 0
        for condition in ('complete', 'partial'):
            row = next(r for r in rows if r['condition'] == condition)
            gp = ROOT/'data/generation_records'/(row['row_id']+'.json')
            generated = json.loads(gp.read_text(encoding='utf-8'))
            for item in generated['items']:
                assert item['item_id'] not in old, 'Already independently reviewed'
                stance, relation, correctness = CODES[codes[n]]
                record = {**item, 'row_id': row['row_id'], 'question_id': qid,
                    'group_index_in_split': index, 'source_index': ref.get('source_index'),
                    'stance': stance, 'evidence_relation': relation,
                    'reference_correctness': correctness, 'rationale': notes[n],
                    'multi_claim': n in multi_claim_indices,
                    'source_generation_sha256': sha(gp)}
                content['items'].append(record)
                old[item['item_id']] = record
                n += 1
    content.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                   reviewed_items=len(content['items']),
                   reviewed_groups=len({r['question_id'] for r in content['items']}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(path.name, content['reviewed_groups'], content['reviewed_items'])
