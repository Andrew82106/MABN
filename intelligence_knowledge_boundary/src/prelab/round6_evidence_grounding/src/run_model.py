"""Resumable, label-free model stages for the Round 6 pilot."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from model6 import (ROOT, CONFIG, LAYERS, SURFACE_NAMES, digest, prompt_hash,
                    runtime_signature, load_model, generate_answer,
                    extract_features, baseline_item)


def readl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    tmp.replace(path)


def savel(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf8')
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(rows, args):
    if args.row_ids:
        wanted = set(args.row_ids.split(','))
        rows = [row for row in rows if row['row_id'] in wanted]
        assert {row['row_id'] for row in rows} == wanted, 'Unknown row IDs'
    if args.limit:
        rows = rows[:args.limit]
    return rows


def checked_record(path, expected):
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding='utf8'))
    for key, item in expected.items():
        assert value.get(key) == item, ('Refusing stale cache reuse', str(path), key)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'generate', 'features', 'baselines'], required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--row-ids')
    args = parser.parse_args()
    data = ROOT/'data'
    signature = runtime_signature()
    run_hash = digest(signature)
    smoke_external = args.stage == 'smoke' and (data/'dev_inputs.jsonl').exists()
    assert (data/'inputs.jsonl').exists() or smoke_external, 'Main inputs absent; only a dev-input smoke is permitted'
    all_rows = readl(data/'inputs.jsonl') if (data/'inputs.jsonl').exists() else []
    assert len({r['row_id'] for r in all_rows}) == len(all_rows)
    rows = select(all_rows, args)
    if args.stage == 'smoke':
        if smoke_external:
            rows = select(readl(data/'dev_inputs.jsonl'), args)[:2]
        elif not args.row_ids:
            rows = [r for r in all_rows if r['split'] == 'train'][:2]
        assert len(rows) >= 2, 'Smoke requires two development examples'
    area = data/'smoke' if smoke_external else data
    generation_dir = area/'generation_records'
    generation_dir.mkdir(parents=True, exist_ok=True)
    model = tok = None

    def ensure_model():
        nonlocal tok, model
        if model is None:
            tok, model = load_model()

    started = time.perf_counter()
    if args.stage in ('generate', 'smoke'):
        for i, row in enumerate(rows):
            path = generation_dir/(row['row_id']+'.json')
            expected = {'runtime_hash': run_hash, 'prompt_hash': prompt_hash(row)}
            generated = checked_record(path, expected)
            smoke_cache = data/'smoke/generation_records'/(row['row_id']+'.json')
            if generated is None and args.stage == 'generate' and smoke_cache.exists():
                generated = checked_record(smoke_cache, expected)
                assert all(generated[key] == row[key] for key in ('row_id', 'question_id', 'split', 'condition'))
                assert row['split'] == 'train', 'Only prespecified train development rows may reuse smoke generations'
                generated['generation_origin'] = 'exact_frozen_training_development_smoke_reuse'
                generated['smoke_record_sha256'] = file_hash(smoke_cache)
                save(path, generated)
            if generated is None:
                ensure_model()
                generated = generate_answer(tok, model, row)
                generated['runtime_hash'] = run_hash
                generated['generation_origin'] = 'development_smoke' if args.stage == 'smoke' else 'fresh_primary_generation'
                save(path, generated)
            if args.stage == 'smoke':
                ensure_model()
                features, metadata = extract_features(tok, model, row, generated, audit=True)
                np.savez_compressed(area/f'smoke_{row["row_id"]}.npz', **features)
                save(area/f'smoke_{row["row_id"]}.json', metadata)
                checks = metadata['prefix_checks']
                assert checks, 'No parseable item to audit'
                for check in checks:
                    assert check['post_token_is_exact_generated_id'] and check['same_next_argmax']
                    assert min(check['layer_cosine'].values()) >= .999
                    assert check['next_logits_max_abs_difference'] <= .5
            print(args.stage.upper(), i+1, '/', len(rows), row['row_id'],
                  'tokens', generated['generated_tokens'], 'parse', generated['parser']['all_items_parse_ok'],
                  'seconds', round(generated['seconds'], 2), flush=True)
        collection = []
        expected_rows = rows if smoke_external else all_rows
        for row in expected_rows:
            record = checked_record(generation_dir/(row['row_id']+'.json'),
                                    {'runtime_hash': run_hash, 'prompt_hash': prompt_hash(row)})
            if record is not None:
                collection.append(record)
        savel(area/'generated.jsonl', collection)
        complete = len(collection) == len(expected_rows)
        save(area/'generation_manifest.json', {'runtime_hash': run_hash, 'runtime': signature,
             'input_file': str(data/'dev_inputs.jsonl' if smoke_external else data/'inputs.jsonl'),
             'input_file_sha256': file_hash(data/'dev_inputs.jsonl' if smoke_external else data/'inputs.jsonl'),
             'expected_rows': len(expected_rows), 'completed_rows': len(collection), 'complete': complete,
             'generation_origin_counts': {origin: sum(r.get('generation_origin') == origin for r in collection)
                                          for origin in sorted({r.get('generation_origin', 'unspecified') for r in collection})},
             'generated_file_sha256': file_hash(area/'generated.jsonl'), 'stage': args.stage,
             'primary_resampling': False, 'primary_labels_used': False,
             'prefix_audit_tolerance': {'minimum_layer_cosine': .999, 'maximum_next_logit_absolute_difference': .5,
                                        'same_next_argmax_required': True},
             'wall_seconds_this_stage': time.perf_counter()-started})
        return

    frozen = json.loads((data/'generation_manifest.json').read_text(encoding='utf8'))
    assert frozen['complete'] and frozen['runtime_hash'] == run_hash, 'Freeze all primary generations first'
    assert frozen['generated_file_sha256'] == file_hash(data/'generated.jsonl')
    assert frozen['input_file_sha256'] == file_hash(data/'inputs.jsonl')
    generated_by_id = {x['row_id']: x for x in readl(data/'generated.jsonl')}
    assert set(generated_by_id) == {x['row_id'] for x in all_rows}
    if args.stage == 'features':
        directory = data/'features'
        directory.mkdir(exist_ok=True)
        for i, row in enumerate(rows):
            generated = generated_by_id[row['row_id']]
            path = directory/(row['row_id']+'.json')
            expected = {'runtime_hash': run_hash, 'prompt_hash': prompt_hash(row), 'response_hash': digest(generated['response'])}
            metadata = checked_record(path, expected)
            if metadata is None:
                ensure_model()
                values, metadata = extract_features(tok, model, row, generated)
                metadata['runtime_hash'] = run_hash
                np.savez_compressed(directory/(row['row_id']+'.npz'), **values)
                save(path, metadata)
            assert (directory/(row['row_id']+'.npz')).exists()
            print('FEATURES', i+1, '/', len(rows), row['row_id'], 'items', len(metadata['items']), flush=True)
        metadata_rows, bundles = [], []
        for row in all_rows:
            path = directory/(row['row_id']+'.json')
            if not path.exists():
                continue
            metadata = checked_record(path, {'runtime_hash': run_hash, 'prompt_hash': prompt_hash(row),
                                             'response_hash': digest(generated_by_id[row['row_id']]['response'])})
            with np.load(directory/(row['row_id']+'.npz'), allow_pickle=False) as loaded:
                bundles.append({k: loaded[k] for k in ['item_ids', 'mean_nll', 'mean_entropy', 'surface']+
                                [f'hidden_{l}' for l in LAYERS]})
            metadata_rows.append(metadata)
        if bundles:
            aggregate = {key: np.concatenate([bundle[key] for bundle in bundles], axis=0) for key in bundles[0]}
            aggregate['surface_names'] = np.asarray(SURFACE_NAMES)
            for key, value in aggregate.items():
                if value.dtype.kind in 'fc':
                    assert np.isfinite(value).all(), ('Nonfinite features', key)
            assert len(set(aggregate['item_ids'].tolist())) == len(aggregate['item_ids'])
            np.savez_compressed(directory/'items.npz', **aggregate)
        savel(directory/'records.jsonl', metadata_rows)
        save(directory/'manifest.json', {'runtime_hash': run_hash, 'runtime': signature,
             'rows': len(metadata_rows), 'expected_rows': len(all_rows), 'complete': len(metadata_rows) == len(all_rows),
             'item_count': sum(len(m['items']) for m in metadata_rows),
             'source_generation_hash': frozen['generated_file_sha256'],
             'items_file_sha256': file_hash(directory/'items.npz') if bundles else None,
             'mean_uncertainty_tokens': 'All exact generated tokens overlapping item text; each token probability/entropy read at preceding causal position',
             'hidden_state_tokens': 'Last alphanumeric character identifies lexical content token; hidden read at that token, not one earlier',
             'surface_names': SURFACE_NAMES, 'wall_seconds_this_stage': time.perf_counter()-started})
        return

    directory = data/'baseline_records'
    directory.mkdir(exist_ok=True)
    expected_items = [(row, item) for row in all_rows for item in generated_by_id[row['row_id']]['items']
                      if item['parse_ok'] and item['text']]
    wanted_rows = {row['row_id'] for row in rows}
    work = [(row, item) for row, item in expected_items if row['row_id'] in wanted_rows]
    for i, (row, item) in enumerate(work):
        path = directory/(item['item_id']+'.json')
        expected = {'runtime_hash': run_hash, 'source_generation_hash': frozen['generated_file_sha256']}
        value = checked_record(path, expected)
        if value is None:
            ensure_model()
            value = baseline_item(tok, model, row, generated_by_id[row['row_id']], item)
            value.update(expected)
            save(path, value)
        print('BASELINES', i+1, '/', len(work), item['item_id'], 'self_parse', value['self_confidence'] is not None, flush=True)
    collection = []
    for row, item in expected_items:
        value = checked_record(directory/(item['item_id']+'.json'),
                               {'runtime_hash': run_hash, 'source_generation_hash': frozen['generated_file_sha256']})
        if value is not None:
            collection.append(value)
    savel(data/'baselines.jsonl', collection)
    save(data/'baseline_manifest.json', {'runtime_hash': run_hash, 'runtime': signature,
         'source_generation_hash': frozen['generated_file_sha256'], 'items': len(collection),
         'expected_items': len(expected_items), 'complete': len(collection) == len(expected_items),
         'self_confidence_parse_failures': sum(x['self_confidence'] is None for x in collection),
         'baselines_file_sha256': file_hash(data/'baselines.jsonl'), 'external_model_calls': 0,
         'primary_generation_modified': False, 'labels_read': False,
         'wall_seconds_this_stage': time.perf_counter()-started})


if __name__ == '__main__':
    main()
