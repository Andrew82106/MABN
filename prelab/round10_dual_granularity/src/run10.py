"""Round10 runner: exact Round9 development reuse and fresh heldout generation.

Never loads labels. Fresh test generation requires a source/input freeze.
Every newly extracted cache records its actual code, input and generation hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT.parent / 'round9_evidence_binding'
R7 = ROOT.parent / 'round7_evidence_grounding'
for path in (R7/'src', R9/'src'):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
import model7
import binding9
import run9

VERSION = 'round10-runner-v1'
readl, sha, save, savel, save_npz = run9.readl, run9.sha, run9.save, run9.savel, run9.save_npz


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == sha(source), f'Existing copy changed: {target}'
    else:
        shutil.copyfile(source, target)
    assert sha(target) == sha(source)


def prepare_dev():
    rows = [r for r in readl(R9/'data/inputs.jsonl') if r['split'] in ('train', 'validation')]
    assert len(rows) == 320 and len({r['group_id'] for r in rows}) == 160
    manifest = {'schema': VERSION, 'source_root': str(R9.resolve()),
                'source_data_freeze_sha256': sha(R9/'data/freeze.json'),
                'source_annotation_freeze_sha256': sha(R9/'data/annotation_freeze.json'),
                'rows': {}, 'annotation_files': {}, 'excludes_round9_test': True}
    for r in rows:
        rid = r['row_id']; files = {}
        for folder, exts in [('generation_records', ['json']), ('features', ['json', 'npz'])]:
            for ext in exts:
                name = f'data/{folder}/{rid}.{ext}'
                copy_exact(R9/name, ROOT/name); files[name] = sha(R9/name)
        manifest['rows'][rid] = {'input_row_sha256': model7.digest(r), 'files_sha256': files}
    for split in ('train', 'validation'):
        name = f'data/annotations_{split}.jsonl'
        copy_exact(R9/name, ROOT/name); manifest['annotation_files'][name] = sha(R9/name)
    savel(ROOT/'data/dev_inputs.jsonl', rows)
    manifest['dev_inputs_sha256'] = sha(ROOT/'data/dev_inputs.jsonl')
    existing = ROOT/'data/reuse_manifest.json'
    if existing.exists(): assert json.loads(existing.read_text('utf-8')) == manifest
    else: save(existing, manifest)
    pack(rows)
    return {'reused_rows': len(rows), 'old_test_used': False}


def all_rows():
    path = ROOT/'data/inputs.jsonl'
    return readl(path if path.exists() else ROOT/'data/dev_inputs.jsonl')


def verify_inputs(rows):
    reuse = json.loads((ROOT/'data/reuse_manifest.json').read_text('utf-8'))
    assert sha(ROOT/'data/dev_inputs.jsonl') == reuse['dev_inputs_sha256']
    assert sha(R9/'data/freeze.json') == reuse['source_data_freeze_sha256']
    assert sha(R9/'data/annotation_freeze.json') == reuse['source_annotation_freeze_sha256']
    known = reuse['rows']
    for row in rows:
        run9.safe_identity(row); run9.validate_visible_strings(row)
        assert len(row['questions']) == row['expected_items'] == 1
        if row['row_id'] in known:
            assert row['split'] in ('train', 'validation')
            assert model7.digest(row) == known[row['row_id']]['input_row_sha256']
        else:
            assert row['split'] == 'test'
    if any(r['split'] == 'test' for r in rows):
        freeze = json.loads((ROOT/'data/freeze.json').read_text('utf-8'))
        assert freeze['status'] == 'frozen' and freeze['created_before_test_generation']
        for name, expected in freeze['files_sha256'].items(): assert sha(ROOT/name) == expected, name
    return reuse


def signature(stage):
    files = [Path(__file__), Path(model7.__file__), Path(binding9.__file__),
             Path(run9.__file__), Path(run9.attention7.__file__)]
    if stage == 'soft': files.append(ROOT/'src/feature10.py')
    return {'schema': VERSION, 'stage': stage,
            'code_sha256': {str(p.resolve()): sha(p) for p in files},
            'model_config_sha256': sha(model7.MODEL/'config.json'),
            'tokenizer_config_sha256': sha(model7.MODEL/'tokenizer_config.json'),
            'generation_config': model7.CONFIG}


def pack(rows=None):
    rows = rows or all_rows()
    records = []
    for r in rows:
        p = ROOT/'data/generation_records'/(r['row_id']+'.json')
        if p.exists(): records.append(json.loads(p.read_text('utf-8')))
    savel(ROOT/'data/generated.jsonl', records)
    save(ROOT/'data/generation_manifest.json', {'expected_rows': len(rows), 'completed_rows': len(records),
        'generated_sha256': sha(ROOT/'data/generated.jsonl'), 'complete': len(records) == len(rows)})


def validate_new(arrays, meta, generated):
    n = len(generated['response_token_ids'])
    for name, dimensions, names_key in [('new_features', 16, 'new_feature_names'),
                                        ('surface_features', 8, 'surface_feature_names')]:
        a = arrays[name]
        assert a.shape == (n, dimensions) and a.dtype == np.float32 and np.isfinite(a).all(), name
        assert len(meta[names_key]) == len(set(meta[names_key])) == dimensions, names_key
    assert arrays['token_ids'].tolist() == generated['response_token_ids']
    offsets = np.asarray(generated['response_token_offsets'])
    assert np.array_equal(arrays['token_start'], offsets[:, 0])
    assert np.array_equal(arrays['token_end'], offsets[:, 1])


def run(stages, splits=None, limit=None):
    rows = all_rows(); reuse = verify_inputs(rows)
    work = [r for r in rows if splits is None or r['split'] in splits]
    if limit is not None: work = work[:limit]
    tok = model = None
    for stage in stages:
        sig = signature(stage); code_digest = model7.digest(sig)
        for ix, row in enumerate(work, 1):
            rid = row['row_id']; gp = ROOT/'data/generation_records'/(rid+'.json')
            folder = {'generate': 'generation_records', 'core': 'features', 'soft': 'soft_features'}[stage]
            target = ROOT/'data'/folder/(rid+'.json')
            source_reuse = reuse['rows'].get(rid, {}).get('files_sha256', {})
            name = f'data/{folder}/{rid}.json'
            if stage in ('generate', 'core') and name in source_reuse:
                for suffix in (['json'] if stage == 'generate' else ['json', 'npz']):
                    relative = f'data/{folder}/{rid}.{suffix}'
                    assert sha(ROOT/relative) == source_reuse[relative]
                    assert sha(R9/relative) == source_reuse[relative]
                continue
            expected = {'input_row_sha256': model7.digest(row), 'stage_signature_sha256': code_digest}
            if stage != 'generate': expected['source_generation_sha256'] = sha(gp)
            if target.exists():
                cached = json.loads(target.read_text('utf-8'))
                for k, v in expected.items(): assert cached[k] == v, (target, k)
                if stage != 'generate': assert sha(target.with_suffix('.npz')) == cached['arrays_sha256']
                continue
            if model is None: tok, model = model7.load_model()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            if stage == 'generate':
                value = model7.generate_answer(tok, model, run9.generation_row(row))
                value.update(run9.safe_identity(row), expected_items=1)
                run9.validate_generated(tok, row, value)
            else:
                generated = json.loads(gp.read_text('utf-8'))
                run9.validate_generated(tok, row, generated)
                if stage == 'core':
                    arrays, value = binding9.extract_binding_features(tok, model, run9.visible(row), run9.extractor_generation(generated))
                    run9.validate_arrays('core', arrays, generated, value)
                else:
                    feature10 = importlib.import_module('feature10')
                    arrays, value = feature10.extract_features(tok, model, run9.visible(row), run9.extractor_generation(generated))
                    validate_new(arrays, value, generated)
                save_npz(target.with_suffix('.npz'), arrays)
                value['arrays_sha256'] = sha(target.with_suffix('.npz'))
            torch.cuda.synchronize()
            value.update(expected, stage_signature=sig, row_id=rid, labels_read=False,
                         seconds=time.perf_counter()-started,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            assert signature(stage) == sig, 'Feature code changed during extraction'
            verify_inputs(rows)
            save(target, value)
            if stage == 'generate': pack(rows)
            print('DONE', stage, ix, len(work), rid, round(value['seconds'], 3), flush=True)
        pack(rows)
    return {'stages': stages, 'selected_rows': len(work)}


def audit():
    rows = all_rows(); reuse = verify_inputs(rows)
    tok = run9.load_tokenizer(); completed = {'generate': 0, 'core': 0, 'soft': 0}
    missing = []; records = []
    for row in rows:
        rid = row['row_id']; gp = ROOT/'data/generation_records'/(rid+'.json')
        if not gp.exists(): missing.append(['generate', rid]); continue
        g = json.loads(gp.read_text('utf-8')); run9.validate_generated(tok, row, g)
        records.append(g); completed['generate'] += 1
        if rid in reuse['rows']:
            assert sha(gp) == reuse['rows'][rid]['files_sha256'][f'data/generation_records/{rid}.json']
        else:
            assert g['input_row_sha256'] == model7.digest(row)
            assert g['stage_signature_sha256'] == model7.digest(signature('generate'))
        for stage, folder in [('core', 'features'), ('soft', 'soft_features')]:
            path = ROOT/'data'/folder/(rid+'.json')
            if not path.exists(): missing.append([stage, rid]); continue
            meta = json.loads(path.read_text('utf-8'))
            assert meta['source_generation_sha256'] == sha(gp)
            assert meta['arrays_sha256'] == sha(path.with_suffix('.npz'))
            if stage == 'core' and rid in reuse['rows']:
                assert sha(path) == reuse['rows'][rid]['files_sha256'][f'data/features/{rid}.json']
            else:
                assert meta['stage_signature_sha256'] == model7.digest(signature(stage))
                assert meta['input_row_sha256'] == model7.digest(row)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as handle:
                arrays = {k: handle[k] for k in handle.files}
            if stage == 'core': run9.validate_arrays('core', arrays, g, meta)
            else: validate_new(arrays, meta, g)
            completed[stage] += 1
    assert readl(ROOT/'data/generated.jsonl') == records
    report = {'expected_rows': len(rows), 'completed': completed, 'missing': missing,
              'complete': not missing, 'labels_read': False, 'model_loaded': False}
    save(ROOT/'results/extraction_audit10.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare-dev', 'generate', 'core', 'soft', 'all', 'audit'])
    p.add_argument('--split', nargs='+', choices=['train', 'validation', 'test']); p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.stage == 'prepare-dev': out = prepare_dev()
    elif args.stage == 'audit': out = audit()
    else: out = run(['generate', 'core', 'soft'] if args.stage == 'all' else [args.stage], args.split, args.limit)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
