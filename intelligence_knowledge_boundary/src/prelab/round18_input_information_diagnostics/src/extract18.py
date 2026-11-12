"""Train-only exact R16 replay using the unchanged Round10 feature definition.

No labels, new generation, fitted transforms, or held-out replay. Only the
original final-RMSNorm layer-28 state is exported; layer 21 is discarded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R17 = ROOT.parent / 'round17_expanded_retraining'
sys.path.insert(0, str(R17/'src'))
import extract17 as base
import feature10
import binding9

SOURCE = base.SOURCE
VERSION = 'round18-train-only-original-feature10-v1'
EXPECTED_ROWS = 602
SCHEMA = {
    'lb': 'float32[N,784], layer-major/head-minor, original Lookback ratios',
    'nll': 'float32[N], selected-token negative natural log probability',
    'new_features': 'float32[N,16], unchanged feature10.NEW_FEATURE_NAMES order',
    'surface_features': 'float32[N,8], unchanged feature10.SURFACE_FEATURE_NAMES order',
    'hidden_28': 'float32[N,3584], unchanged final RMSNorm output at post-read P+j',
    'token_ids': 'int64[N]',
    'response_token_offsets': 'int32[N,2], original full-response character offsets',
    'token_start': 'int32[N]',
    'token_end': 'int32[N]',
}


def selected_rows():
    rows, source_manifest = base.source_rows()
    selected = [row for row in rows if row['split'] == 'train']
    assert len(selected) == EXPECTED_ROWS
    assert len({row['question_id'] for row in selected}) == 301
    assert all(row['split'] == 'train' for row in selected)
    # Reject accidental held-out cache files rather than silently ignoring them.
    allowed = {row['row_id'] for row in selected}
    for suffix in ('*.json', '*.npz'):
        for path in (ROOT/'data/features').glob(suffix):
            assert path.stem in allowed, ('Non-training feature file', str(path))
    return selected, source_manifest


def signature():
    return {
        'version': VERSION,
        'code_sha256': {str(p.resolve()): base.sha(p) for p in
                        (Path(__file__), Path(feature10.__file__), Path(binding9.__file__))},
        'base_replay_signature': base.signature(),
        'r17_feature_manifest_sha256': base.sha(R17/'data/feature_manifest.json'),
        'selector': "R16 inputs.jsonl actual split == 'train'; no label/answer filtering",
        'expected_rows': EXPECTED_ROWS,
        'feature10_version': feature10.VERSION,
        'schema': SCHEMA,
        'lb_nll_cache_comparison': 'Exact equality to every original R17 response-token value',
    }


def reference_manifest():
    manifest = base.read(R17/'data/feature_manifest.json')
    assert manifest['complete'] and manifest['completed_count'] == 800
    return manifest


def validate(arrays, generated):
    assert set(arrays) == set(SCHEMA), sorted(arrays)
    base.validate(arrays, generated)
    n = len(generated['response_token_ids'])
    for name, shape in [('new_features', (n, 16)), ('surface_features', (n, 8)),
                        ('hidden_28', (n, 3584))]:
        assert arrays[name].shape == shape and arrays[name].dtype == np.float32, name
        assert np.isfinite(arrays[name]).all(), name
    assert arrays['token_ids'].dtype == np.int64
    for name in ('response_token_offsets', 'token_start', 'token_end'):
        assert arrays[name].dtype == np.int32, name


def compare_r17(row, generated, generation_sha, arrays, r17_manifest):
    """Read only the current training row's frozen reference feature cache."""
    assert row['split'] == 'train'
    rid = row['row_id']; record = r17_manifest['records'][rid]
    assert record['source_generation_sha256'] == generation_sha
    metadata_path, path = R17/record['json'], R17/record['npz']
    assert base.sha(metadata_path) == record['json_sha256']
    assert base.sha(path) == record['npz_sha256']
    meta = base.read(metadata_path)
    assert meta['source_generation_sha256'] == generation_sha
    assert meta['input_row_sha256'] == base.model7.digest(row)
    assert meta['arrays_sha256'] == record['npz_sha256']
    delta = {}
    with np.load(path, allow_pickle=False) as old:
        for name in ('token_ids', 'response_token_offsets', 'token_start', 'token_end'):
            assert np.array_equal(arrays[name], old[name]), (rid, name)
        for name in ('lb', 'nll'):
            assert arrays[name].shape == old[name].shape
            delta[name] = float(np.abs(arrays[name]-old[name]).max())
            assert np.array_equal(arrays[name], old[name]), ('R17 mismatch; do not relax silently', rid, delta)
    return {'passed': True, 'all_response_tokens_compared': len(generated['response_token_ids']),
            'max_absolute_differences': delta, 'r17_npz_sha256': record['npz_sha256'],
            'r17_json_sha256': record['json_sha256']}


def expected(row, generation_sha, sig):
    return {'row_id': row['row_id'], 'split': 'train',
            'input_row_sha256': base.model7.digest(row),
            'source_generation_sha256': generation_sha,
            'extraction_signature_sha256': base.model7.digest(sig)}


def cached(row, generated, generation_sha, sig, r17_manifest):
    path = ROOT/'data/features'/(row['row_id']+'.json')
    if not path.exists():
        return None
    meta = base.read(path)
    for key, value in expected(row, generation_sha, sig).items():
        assert meta[key] == value, (row['row_id'], key)
    assert base.sha(path.with_suffix('.npz')) == meta['arrays_sha256']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate(arrays, generated)
    assert compare_r17(row, generated, generation_sha, arrays, r17_manifest) == meta['r17_cache_check']
    return arrays, meta


def extract(tokenizer, model, row, generated, generation_sha, r17_manifest):
    assert row['split'] == 'train'
    original, original_meta = feature10.extract_features(tokenizer, model, base.visible(row), generated)
    # Preserve original dtypes/axes. These are references, without recalculation.
    arrays = {key: original[key] for key in
              ('new_features', 'surface_features', 'hidden_28', 'token_ids',
               'response_token_offsets', 'token_start', 'token_end')}
    arrays.update(lb=original['lookback_features'], nll=original['token_nll'])
    validate(arrays, generated)
    comparison = compare_r17(row, generated, generation_sha, arrays, r17_manifest)
    meta = {'version': VERSION, 'row_id': row['row_id'],
            'input_tokens': len(generated['input_token_ids']),
            'response_tokens': len(generated['response_token_ids']),
            'feature10_metadata': original_meta, 'r17_cache_check': comparison,
            'schema': SCHEMA, 'new_feature_names': feature10.NEW_FEATURE_NAMES,
            'surface_feature_names': feature10.SURFACE_FEATURE_NAMES,
            'hidden_export': 'Only original hidden_28: post-final-RMSNorm float32[N,3584]; hidden_21 not saved',
            'labels_or_detector_scores_read': False, 'response_regenerated': False,
            'seconds': original_meta['seconds'], 'peak_allocated_gib': original_meta['peak_allocated_gib']}
    return arrays, meta


def store_row(row, generated, generation_sha, arrays, meta, sig):
    assert signature() == sig, 'Source or extraction signature changed during replay'
    assert base.sha(SOURCE/'data/generation_records'/(row['row_id']+'.json')) == generation_sha
    path = ROOT/'data/features'/(row['row_id']+'.npz')
    base.save_arrays(path, arrays)
    meta.update(expected(row, generation_sha, sig), arrays_sha256=base.sha(path), extraction_signature=sig)
    base.save(path.with_suffix('.json'), meta)


def manifest(rows, source_manifest, sig, r17_manifest):
    records, missing = {}, []
    for row in rows:
        generated, digest = base.generation(row, source_manifest)
        value = cached(row, generated, digest, sig, r17_manifest)
        if value is None:
            missing.append(row['row_id']); continue
        _, meta = value; relative = 'data/features/'+row['row_id']
        records[row['row_id']] = {'source_generation_sha256': digest,
            'json': relative+'.json', 'json_sha256': base.sha(ROOT/(relative+'.json')),
            'npz': relative+'.npz', 'npz_sha256': meta['arrays_sha256'],
            'response_tokens': meta['response_tokens'], 'split': 'train',
            'r17_max_absolute_differences': meta['r17_cache_check']['max_absolute_differences']}
    result = {'version': VERSION, 'expected_rows': EXPECTED_ROWS, 'completed_count': len(records),
              'completed_rows': len(records), 'generated': len(records), 'complete': not missing,
              'missing_rows': missing, 'records': records, 'schema': SCHEMA,
              'total_response_tokens_completed': sum(r['response_tokens'] for r in records.values()),
              'actual_split_filter': 'train', 'selected_question_groups': 301,
              'validation_or_test_rows_extracted': 0,
              'all_training_responses_including_refusals_and_parse_failures': True,
              'every_lb_nll_token_exactly_compared_to_r17': True,
              'source_root': str(SOURCE.resolve()), 'reference_cache_root': str(R17.resolve()),
              'extraction_signature': sig, 'extraction_signature_sha256': base.model7.digest(sig),
              'new_feature_names': feature10.NEW_FEATURE_NAMES, 'surface_feature_names': feature10.SURFACE_FEATURE_NAMES,
              'labels_or_detector_scores_read': False, 'response_regenerated': False,
              'limitation': 'Original soft embedding/lexical alignment is a heuristic, not entailment. Full saved-sequence replay preserves prior feature definitions; strict online prefix numerical identity is not claimed.'}
    base.save(ROOT/'data/feature_manifest.json', result)
    return result


def selfcheck(tokenizer, model, rows, source_manifest, sig, r17_manifest):
    """Two training records, cache equality, plus repeat replay of the first."""
    # Selection uses input split and IDs, never annotations or detector outputs.
    chosen = [rows[0], rows[-1]]
    checks = []
    for index, row in enumerate(chosen):
        generated, digest = base.generation(row, source_manifest)
        arrays, meta = extract(tokenizer, model, row, generated, digest, r17_manifest)
        item = {'row_id': row['row_id'], 'source_generation_sha256': digest,
                'r17_cache_check': meta['r17_cache_check'],
                'shapes': {k: list(v.shape) for k, v in arrays.items()},
                'dtypes': {k: str(v.dtype) for k, v in arrays.items()}}
        if index == 0:
            repeat, _ = extract(tokenizer, model, row, generated, digest, r17_manifest)
            item['repeat_max_absolute_differences'] = {
                key: float(np.abs(arrays[key].astype(np.float64)-repeat[key].astype(np.float64)).max())
                for key in arrays}
            assert all(np.array_equal(arrays[k], repeat[k]) for k in arrays), item
        store_row(row, generated, digest, arrays, meta, sig)
        checks.append(item)
    base.save(ROOT/'data/extraction_selfcheck.json', {
        'passed': True, 'extraction_signature_sha256': base.model7.digest(sig), 'checks': checks,
        'selected_only_actual_training_split': True, 'labels_or_detector_scores_read': False,
        'response_regenerated': False, 'original_feature10_called_without_modification': True,
        'numerical_limitation': 'Full-shape cached replay equality is tested. Different-length truncation can alter BF16 rounding; no strict online prefix numerical identity is asserted.'})
    print('SELFCHECK_PASSED', json.dumps(checks, ensure_ascii=False), flush=True)


def run(selfcheck_only=False, audit_only=False):
    rows, source_manifest = selected_rows()
    sig, r17_manifest = signature(), reference_manifest()
    if audit_only:
        result = manifest(rows, source_manifest, sig, r17_manifest)
        print('AUDIT', result['completed_count'], EXPECTED_ROWS, result['complete'], flush=True)
        return
    check_path = ROOT/'data/extraction_selfcheck.json'
    tokenizer = model = None
    if not check_path.exists():
        tokenizer, model = base.model7.load_model()
        assert getattr(model, 'is_loaded_in_4bit', False)
        selfcheck(tokenizer, model, rows, source_manifest, sig, r17_manifest)
    else:
        check = base.read(check_path)
        assert check['passed'] and check['extraction_signature_sha256'] == base.model7.digest(sig)
    if selfcheck_only:
        manifest(rows, source_manifest, sig, r17_manifest)
        return
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        generated, digest = base.generation(row, source_manifest)
        if cached(row, generated, digest, sig, r17_manifest) is not None:
            continue
        if model is None:
            tokenizer, model = base.model7.load_model()
            assert getattr(model, 'is_loaded_in_4bit', False)
        torch.cuda.reset_peak_memory_stats()
        arrays, meta = extract(tokenizer, model, row, generated, digest, r17_manifest)
        store_row(row, generated, digest, arrays, meta, sig)
        print('DONE', i, EXPECTED_ROWS, row['row_id'], meta['response_tokens'], round(meta['seconds'], 3), flush=True)
        if i % 100 == 0:
            manifest(rows, source_manifest, sig, r17_manifest)
    selected_rows()
    result = manifest(rows, source_manifest, sig, r17_manifest)
    print('COMPLETE', result['completed_count'], EXPECTED_ROWS, result['complete'],
          'elapsed_seconds', round(time.perf_counter()-started, 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck-only', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    run(selfcheck_only=args.selfcheck_only, audit_only=args.audit)
