"""Frozen 793-development-answer NF4 replay: LB1024, NLL and hidden4096 only.

Requires the separate two-example GPU selfcheck to have passed. Resumes only
hash-verified complete rows; never opens test data or annotation values.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import feature_qa as feature
import run_feature_qa as checked

ROOT = feature.ROOT
OUT = ROOT / 'data/features'
VERSION = 'ragtruth-qa-llama2-nf4-development-base-v1'
BASE_KEYS = ('lb', 'nll', 'hidden_last', 'token_ids', 'answer_token_positions',
             'response_token_offsets', 'response_token_offsets_raw', 'token_start', 'token_end')


def validate_arrays(arrays, plan):
    view = plan['original']; count = len(view['answer_token_ids'])
    assert set(arrays) == set(BASE_KEYS)
    assert arrays['lb'].shape == (count, 1024)
    assert arrays['nll'].shape == (count,)
    assert arrays['hidden_last'].shape == (count, 4096)
    for key in ('lb', 'nll', 'hidden_last'):
        assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
    assert np.all((arrays['lb'] >= 0) & (arrays['lb'] <= 1))
    assert np.all(arrays['nll'] >= 0)
    for key, target in (('token_ids', 'answer_token_ids'),
                        ('answer_token_positions', 'answer_token_positions'),
                        ('response_token_offsets', 'response_token_offsets'),
                        ('response_token_offsets_raw', 'response_token_offsets_raw')):
        assert np.array_equal(arrays[key], np.asarray(view[target]))
    assert np.array_equal(arrays['token_start'], arrays['response_token_offsets'][:, 0])
    assert np.array_equal(arrays['token_end'], arrays['response_token_offsets'][:, 1])
    assert all(arrays[key].dtype == np.int64 for key in ('token_ids', 'answer_token_positions'))
    assert all(arrays[key].dtype == np.int32 for key in BASE_KEYS[5:])
    return count


def prepare():
    plans, _, _, selfcheck_signature = checked.prepare_signature()
    path = ROOT / 'data/replay_selfcheck/manifest.json'
    passed = feature.read(path)
    assert passed['passed'] is True and passed['gpu_rows_selected'] == 2
    assert passed['signature_sha256'] == feature.digest(selfcheck_signature)
    assert [row['response_id'] for row in passed['checks']] == selfcheck_signature['selected_response_ids']
    assert all(row['passed'] and row['no_context_hidden_repeat_exact'] for row in passed['checks'])
    assert len({plan['response_id'] for plan in plans}) == len(plans) == 793
    assert {partition: sum(p['partition'] == partition for p in plans) for partition in feature.PARTITIONS} == feature.EXPECTED
    for plan in plans:
        assert plan['labels_used'] is False and plan['official_split'] == 'train'
        assert str(plan['response_id']).isalnum(), 'Only safe frozen response IDs may become filenames'
    signature = {'version': VERSION, 'runner_sha256': feature.sha(Path(__file__)),
        'selfcheck_signature_sha256': feature.digest(selfcheck_signature),
        'passed_selfcheck_manifest_sha256': feature.sha(path),
        'plans_sha256': selfcheck_signature['plans_sha256'],
        'source_development_files_sha256': selfcheck_signature['source_development_files_sha256'],
        'feature_code_sha256': feature.sha(Path(feature.__file__)),
        'loader_code_sha256': feature.sha(Path(checked.__file__)),
        'load_config': checked.LOAD_CONFIG, 'repo_id': checked.REPO, 'revision': checked.REVISION,
        'response_ids_in_order': [p['response_id'] for p in plans],
        'partitions': feature.EXPECTED, 'feature_keys': list(BASE_KEYS),
        'include_delta': False, 'include_harp': False, 'test_or_withheld_content_read': False,
        'annotation_values_accessed': False, 'exact_original_generation_trace': False,
        'eligibility': 'Frozen good-quality official-train development exports; 634 fit and 159 calibration'}
    checked.frozen_json(ROOT / 'data/feature_signature.json', signature)
    return plans, signature


def cached(plan, signature_sha):
    path = OUT / (str(plan['response_id']) + '.npz')
    meta_path = path.with_suffix('.json')
    if not meta_path.exists():
        assert not path.exists(), ('Uncommitted row requires inspection', str(path))
        return None
    meta = feature.read(meta_path)
    assert meta['complete'] and meta['signature_sha256'] == signature_sha
    assert meta['plan_sha256'] == feature.digest(plan)
    assert meta['response_id'] == plan['response_id'] and meta['partition'] == plan['partition']
    assert feature.sha(path) == meta['npz_sha256'] and path.stat().st_size == meta['npz_bytes']
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key] for key in opened.files}
    assert validate_arrays(arrays, plan) == meta['response_tokens']
    return meta


def progress(plans, signature, records, start, load_meta, resumed, verified=False):
    complete = verified and len(records) == len(plans)
    feature.save(ROOT / 'data/feature_manifest.json', {
        'version': VERSION, 'complete': complete, 'signature_sha256': feature.digest(signature),
        'planned_records': len(plans), 'completed_records': len(records),
        'completed_partitions': {part: sum(r['partition'] == part for r in records) for part in feature.PARTITIONS},
        'completed_response_tokens': sum(r['response_tokens'] for r in records),
        'completed_npz_bytes': sum(r['npz_bytes'] for r in records),
        'raw_array_bytes': sum(r['raw_array_bytes'] for r in records),
        'model_feature_seconds': sum(r['feature_metadata']['seconds'] for r in records),
        'row_serialize_and_hash_seconds': sum(r['serialization_hash_seconds'] for r in records),
        'max_peak_allocated_gpu_gib': max((r['feature_metadata']['peak_allocated_gpu_gib'] for r in records), default=None),
        'current_invocation_wall_seconds': time.perf_counter() - start,
        'current_invocation_resumed_rows': resumed, 'model_loading': load_meta,
        'records': [{'response_id': r['response_id'], 'partition': r['partition'],
                     'response_tokens': r['response_tokens'], 'plan_sha256': r['plan_sha256'],
                     'npz_sha256': r['npz_sha256'], 'npz_bytes': r['npz_bytes'],
                     'metadata_sha256': feature.sha(OUT / (str(r['response_id']) + '.json'))} for r in records],
        'features': {'lb': 'float32[T,1024]', 'nll': 'float32[T]', 'hidden_last': 'float32[T,4096]'},
        'quantized_replay': True, 'original_published_generation_trace': False,
        'no_truncation_or_answer_regeneration': True, 'test_or_withheld_content_read': False,
        'annotation_values_accessed': False, 'delta_or_harp_extracted': False})


def main():
    started = time.perf_counter(); model = None; records = []; resumed = 0; loading = None
    plans, signature = prepare(); signature_sha = feature.digest(signature)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / '.runner.lock'
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode('ascii')); os.close(descriptor)
    try:
        for plan in plans:
            entry = cached(plan, signature_sha)
            if entry is None:
                break
            records.append(entry)
        resumed = len(records)
        print('QA_BASE_START', json.dumps({'records': len(plans), 'resumed': resumed,
              'signature_sha256': signature_sha, 'pid': os.getpid()}), flush=True)
        progress(plans, signature, records, started, loading, resumed)
        if len(records) < len(plans):
            assert shutil.disk_usage(OUT).free > 6 * 2**30, 'Need 6 GiB free workspace disk before replay'
            model, loading = checked.load_nf4()
        for index, plan in enumerate(plans[resumed:], start=resumed):
            # Also verifies isolated later cached rows on resume, without redoing them.
            existing = cached(plan, signature_sha)
            if existing is not None:
                records.append(existing); resumed += 1
                continue
            arrays, meta = feature.extract_features(model, plan, include_delta=False)
            count = validate_arrays(arrays, plan)
            serialize_started = time.perf_counter()
            path = OUT / (str(plan['response_id']) + '.npz')
            checked.save_npz(path, arrays)
            entry = {'complete': True, 'response_id': plan['response_id'], 'source_id': plan['source_id'],
                'group_id': plan['group_id'], 'partition': plan['partition'], 'official_split': 'train',
                'signature_sha256': signature_sha, 'plan_sha256': feature.digest(plan),
                'response_tokens': count, 'raw_array_bytes': sum(a.nbytes for a in arrays.values()),
                'npz_sha256': feature.sha(path), 'npz_bytes': path.stat().st_size,
                'feature_metadata': meta, 'serialization_hash_seconds': time.perf_counter() - serialize_started,
                'labels_used': False, 'test_content_read': False}
            feature.save(path.with_suffix('.json'), entry); records.append(entry)
            if len(records) % 10 == 0 or len(records) == len(plans):
                progress(plans, signature, records, started, loading, resumed)
                elapsed = time.perf_counter() - started
                print('QA_BASE_PROGRESS', len(records), '/', len(plans), 'elapsed_s', round(elapsed, 2),
                      'last_feature_s', round(meta['seconds'], 3), flush=True)
        assert [r['response_id'] for r in records] == signature['response_ids_in_order']
        assert feature.sha(Path(__file__)) == signature['runner_sha256']
        assert feature.sha(Path(feature.__file__)) == signature['feature_code_sha256']
        assert feature.sha(Path(checked.__file__)) == signature['loader_code_sha256']
        assert feature.sha(ROOT / 'data/feature_preparation/plans.jsonl') == signature['plans_sha256']
        # Read-back validates full output coordinates and finite values, not merely file count.
        audit_started = time.perf_counter()
        audited = [cached(plan, signature_sha) for plan in plans]
        assert all(record is not None for record in audited)
        progress(plans, signature, records, started, loading, resumed, verified=True)
        feature.save(ROOT / 'data/feature_audit.json', {'passed': True, 'records': len(audited),
            'response_tokens': sum(r['response_tokens'] for r in audited),
            'signature_sha256': signature_sha, 'feature_manifest_sha256': feature.sha(ROOT / 'data/feature_manifest.json'),
            'all_npz_hashes_shapes_coordinates_finite_values_checked': True,
            'audit_seconds': time.perf_counter() - audit_started, 'test_content_read': False,
            'labels_used': False})
        print('QA_BASE_COMPLETE', len(records), 'wall_s', round(time.perf_counter() - started, 2), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, action='store_true')
    parser.parse_args(); main()
