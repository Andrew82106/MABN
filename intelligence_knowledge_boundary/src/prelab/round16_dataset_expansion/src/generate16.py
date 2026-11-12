"""Resumable own-model generation after the source/input freeze, no gold reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
path = ROOT.parent/'round7_evidence_grounding/src/model7.py'
spec = importlib.util.spec_from_file_location('r16_model7', path)
model7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(model7)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    temp.replace(path)


def run(limit=None):
    freeze = json.loads((ROOT/'data/input_freeze.json').read_text('utf-8'))
    assert freeze['status'] == 'frozen_before_generation'
    for name, digest in freeze['files_sha256'].items():
        assert sha(ROOT/name) == digest, name
    rows = [json.loads(s) for s in (ROOT/'data/inputs.jsonl').read_text('utf-8').splitlines()]
    signature = {'model7_sha256': sha(Path(model7.__file__)), 'wrapper_sha256': sha(Path(__file__)),
                 'model_config_sha256': sha(model7.MODEL/'config.json'),
                 'tokenizer_config_sha256': sha(model7.MODEL/'tokenizer_config.json'),
                 'generation_config': model7.CONFIG,
                 'input_freeze_sha256': sha(ROOT/'data/input_freeze.json')}
    done = {}; pending = []
    for row in rows:
        p = ROOT/'data/generation_records'/(row['row_id']+'.json')
        if p.exists():
            r = json.loads(p.read_text('utf-8'))
            assert r['signature'] == signature and r['prompt_hash'] == model7.prompt_hash(row)
            done[row['row_id']] = sha(p)
        else:
            pending.append(row)
    if limit is not None:
        pending = pending[:limit]
    if pending:
        tok, model = model7.load_model()
        start = time.perf_counter()
        for i, row in enumerate(pending):
            # Only system and prompt strings enter the generator's chat template.
            result = model7.generate_answer(tok, model, row)
            result.update(group_id=row['group_id'], category=row['category'], signature=signature,
                          input_row_sha256=model7.digest(row))
            p = ROOT/'data/generation_records'/(row['row_id']+'.json')
            save(p, result); done[row['row_id']] = sha(p)
            if i == 0 or (i+1) % 25 == 0 or i+1 == len(pending):
                print(f'Generated {len(done)}/{len(rows)}; current run {time.perf_counter()-start:.1f}s', flush=True)
    save(ROOT/'data/generation_manifest.json', {'generated': len(done), 'expected': len(rows),
                                               'complete': len(done) == len(rows), 'signature': signature,
                                               'record_sha256': done})
    if len(done) == len(rows):
        text = ''.join(json.dumps(json.loads((ROOT/'data/generation_records'/(r['row_id']+'.json')).read_text('utf-8')), ensure_ascii=False)+'\n' for r in rows)
        (ROOT/'data/generated.jsonl').write_text(text, 'utf-8')
    print('GENERATION_STAGE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int)
    args = parser.parse_args(); run(args.limit)
