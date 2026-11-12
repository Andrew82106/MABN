"""Fix LUMINA's unrelated context using development sources before generation."""
from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT.parent/'round7_evidence_grounding/data/dev_inputs.jsonl'


def readl(p):
    return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s.strip()]


def build():
    path = ROOT/'data/lumina_random_manifest.json'
    assert not path.exists(), 'Do not overwrite fixed donor assignments'
    assert not any((ROOT/'data/generation_records').glob('*.json')), 'Fix random context before generation'
    rows = readl(ROOT/'data/inputs.jsonl'); pool = readl(POOL)
    # One deterministic representative per old development group; do not sample
    # alternative old conditions as distinct semantic sources.
    donors = {r['question_id']: r for r in sorted(pool, key=lambda r:r['row_id'], reverse=True)}
    assert all(r['split']=='development' for r in pool)
    all_titles = {p['title'].casefold() for r in rows for p in r['passages']}
    assignments = {}
    for row in rows:
        gid = row['group_id']
        if gid in assignments:
            assert len(assignments[gid]['passages']) == len(row['passages'])
            continue
        seed = int(hashlib.sha256(('round9-lumina-20260911:'+gid).encode()).hexdigest()[:16],16)
        rng = random.Random(seed); choices = []
        for old in sorted(donors.values(), key=lambda r:r['row_id']):
            valid = []
            for p in old['passages']:
                if p['title'].casefold() in all_titles:
                    continue
                q = {'title':p['title'], 'text':p['text'], 'origin_row_id':old['row_id']}
                if q not in valid:
                    valid.append(q)
            if len(valid) >= len(row['passages']):
                choices.append((old,valid))
        assert choices, 'Independent development pool too small'
        old, valid = choices[rng.randrange(len(choices))]
        chosen = rng.sample(valid,len(row['passages']))
        assignments[gid] = {'donor_id':'r9_random_'+gid, 'origin_group_id':old['question_id'], 'passages':chosen}
    doc = {'schema':'round9-random-context-v1', 'pool_file':'../round7_evidence_grounding/data/dev_inputs.jsonl',
           'pool_sha256':hashlib.sha256(POOL.read_bytes()).hexdigest(),
           'selection':'SHA256(round9-lumina-20260911:group_id) seeded choice from old development groups, then sample distinct passages; same choice for both conditions',
           'created_before_generation':True, 'assignments':assignments}
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('FIXED_RANDOM_CONTEXT',len(assignments))


if __name__=='__main__':
    build()
