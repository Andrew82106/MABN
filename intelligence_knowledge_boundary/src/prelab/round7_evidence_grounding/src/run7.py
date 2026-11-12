"""Resumable GPU stages. No annotation or detector score is read here."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

import model7
from model7 import ROOT, CONFIG, digest, prompt_hash, load_model


def readl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def savel(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in rows), encoding='utf-8')
    temporary.replace(path)


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def generation_signature():
    funcs = [model7.generate_answer, model7.parse_items, model7.generation_config,
             model7.chat_ids, model7.token_offsets]
    return {'config': CONFIG, 'generator_functions_sha256': digest('\n'.join(inspect.getsource(f) for f in funcs)),
            'model_config_sha256': sha(model7.MODEL / 'config.json'),
            'tokenizer_config_sha256': sha(model7.MODEL / 'tokenizer_config.json')}


def stage_signature(stage):
    files = [Path(model7.__file__)]
    if stage == 'attention':
        import attention7
        files.append(Path(attention7.__file__))
    if stage == 'lumina':
        import lumina7
        files.append(Path(lumina7.__file__))
    return {'stage': stage, 'code_hashes': {p.name: sha(p) for p in files},
            'generator': generation_signature(), 'protocol_version': 'round7-v1'}


def checked(path, expected):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f'Stale record {path}: {key}; retain old outputs and diagnose before rerunning')
    return record


def random_source(row, pool):
    """Fixed independent development sources; paired conditions use the same set."""
    subjects = [s.casefold() for s in row.get('subjects', []) if len(s) > 3]
    candidates = [r for r in pool if r['question_id'] != row['question_id'] and
                  not any(s in '\n'.join(p['title']+' '+p['text'] for p in r['passages']).casefold() for s in subjects)]
    if not candidates:
        raise ValueError('No independent random context for ' + row['row_id'])
    candidates = sorted(candidates, key=lambda r: r['row_id'])
    start = int(digest(row['question_id'])[:12], 16) % len(candidates)
    candidates = candidates[start:] + candidates[:start]
    prohibited_titles = {p['title'] for p in row['passages']}
    prohibited_texts = {p['text'] for p in row['passages']}
    passages, origins, seen = [], [], set()
    for candidate in candidates:
        for passage in candidate['passages']:
            if passage['title'] in prohibited_titles or passage['text'] in prohibited_texts or passage['text'] in seen:
                continue
            passages.append(passage)
            origins.append(candidate['question_id'])
            seen.add(passage['text'])
            if len(passages) == len(row['passages']):
                return {'row_id': 'random_'+row['question_id'],
                        'question_id': 'independent_'+digest(origins)[:16],
                        'passages': passages, 'origin_question_ids': origins}
    raise ValueError('Too few distinct independent passages for ' + row['row_id'])


def pack(area, rows, gen_hash, stages):
    generations = []
    for row in rows:
        value = checked(area/'generation_records'/(row['row_id']+'.json'),
                        {'generation_signature_hash': gen_hash, 'prompt_hash': prompt_hash(row)})
        if value is not None:
            generations.append(value)
    savel(area/'generated.jsonl', generations)
    save(area/'generation_manifest.json', {
        'expected_rows': len(rows), 'completed_rows': len(generations),
        'complete': len(rows) == len(generations), 'generated_file_sha256': sha(area/'generated.jsonl'),
        'generation_signature_hash': gen_hash, 'generation_signature': generation_signature(),
        'external_generation_model_calls': 0, 'labels_read': False})
    for stage in stages:
        records = []
        for row in rows:
            p = area/stage/(row['row_id']+'.json')
            if p.exists():
                record = json.loads(p.read_text(encoding='utf-8'))
                if stage != 'baselines':
                    assert p.with_suffix('.npz').exists(), str(p)
                    assert record['arrays_sha256'] == sha(p.with_suffix('.npz'))
                records.append(record)
        save(area/stage/'manifest.json', {'stage': stage, 'expected_rows': len(rows),
             'completed_rows': len(records), 'complete': len(records) == len(rows),
             'seconds': sum(r.get('seconds', 0) for r in records),
             'peak_allocated_gib': max((r.get('peak_allocated_gib', 0) for r in records), default=0),
             'record_files': [r['row_id']+'.json' for r in records], 'labels_read': False})
        if stage == 'baselines':
            savel(area/'baselines.jsonl', [x for r in records for x in r['items']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['generate', 'features', 'attention', 'lumina', 'baselines', 'all', 'pack', 'smoke'])
    ap.add_argument('--data-file', default='data/inputs.jsonl')
    ap.add_argument('--area', default='data')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--row-ids')
    ap.add_argument('--random-pool', default='../round6_evidence_grounding/data/inputs.jsonl')
    args = ap.parse_args()
    path = ROOT/args.data_file
    area = ROOT/args.area
    rows = readl(path)
    assert rows and len({r['row_id'] for r in rows}) == len(rows)
    work = rows
    if args.row_ids:
        wanted = set(args.row_ids.split(','))
        work = [r for r in work if r['row_id'] in wanted]
        assert {r['row_id'] for r in work} == wanted
    if args.limit:
        work = work[:args.limit]
    if args.stage == 'smoke':
        assert all(r['split'] in ('train', 'development', 'dev') for r in work), 'Smoke must use development/training inputs'
        stages = ['generate', 'features', 'attention', 'lumina', 'baselines']
    elif args.stage == 'all':
        stages = ['generate', 'features', 'attention', 'lumina', 'baselines']
    else:
        stages = [args.stage]
    all_stages = ['features', 'attention', 'lumina', 'baselines']
    gen_sig = generation_signature()
    gen_hash = digest(gen_sig)
    if args.stage == 'pack':
        pack(area, rows, gen_hash, all_stages)
        return
    tok = model = None
    pool = None
    for stage in stages:
        signature = gen_sig if stage == 'generate' else stage_signature(stage)
        signature_hash = digest(signature)
        if stage == 'lumina':
            pool = readl(ROOT/args.random_pool)
        for count, row in enumerate(work, 1):
            generation_path = area/'generation_records'/(row['row_id']+'.json')
            generation_expected = {'generation_signature_hash': gen_hash, 'prompt_hash': prompt_hash(row)}
            generated = checked(generation_path, generation_expected)
            target = generation_path if stage == 'generate' else area/stage/(row['row_id']+'.json')
            expected = dict(generation_expected)
            if stage != 'generate':
                if generated is None:
                    raise RuntimeError('Generate this row first: ' + row['row_id'])
                expected.update(stage_signature_hash=signature_hash, source_generation_sha256=sha(generation_path))
            random_row = None
            if stage == 'lumina':
                random_row = random_source(row, pool)
                expected['random_context_sha256'] = digest(random_row['passages'])
            cached = checked(target, expected)
            if cached is not None:
                print('CACHE', stage, count, '/', len(work), row['row_id'], flush=True)
                continue
            if model is None:
                tok, model = load_model()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if stage == 'generate':
                value = model7.generate_answer(tok, model, row)
                value.update(expected, generation_signature=gen_sig)
            elif stage == 'baselines':
                records = []
                for item in generated['items']:
                    if item['parse_ok'] and item['text']:
                        records.append(model7.baseline_item(tok, model, row, generated, item))
                value = {'row_id': row['row_id'], 'items': records, **expected}
            else:
                if stage == 'features':
                    arrays, value = model7.extract_features(tok, model, row, generated, audit=args.stage == 'smoke')
                    for check in value['prefix_checks']:
                        assert check['same_next_argmax'] and check['post_token_is_exact_generated_id']
                        assert min(check['layer_cosine'].values()) >= .999
                        assert check['next_logits_max_abs_difference'] <= .5
                elif stage == 'attention':
                    from attention7 import extract_attention
                    arrays, value = extract_attention(tok, model, row, generated)
                else:
                    from lumina7 import extract_lumina
                    arrays, value = extract_lumina(tok, model, row, generated, random_row)
                ids = [str(x) for x in arrays['item_ids']]
                assert len(ids) == len(set(ids))
                assert set(ids) <= {i['item_id'] for i in generated['items']}
                save_npz(target.with_suffix('.npz'), arrays)
                value.update(expected, row_id=row['row_id'], arrays_sha256=sha(target.with_suffix('.npz')))
            torch.cuda.synchronize()
            value.update(seconds=time.perf_counter()-start,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         stage_signature=signature, dataset=row.get('dataset', 'hotpotqa'),
                         runner_sha256=sha(__file__))
            save(target, value)
            print('DONE', stage, count, '/', len(work), row['row_id'],
                  'seconds', round(value['seconds'], 2), 'peak_GiB', round(value['peak_allocated_gib'], 3), flush=True)
        pack(area, rows, gen_hash, [s for s in stages if s in all_stages])


if __name__ == '__main__':
    main()
