"""Evaluation-only coordinate-validation repair for frozen CUDA NLI outputs.

The completed extraction signature, probabilities and CPU source are unchanged.
Only the validator now respects the gold file's merged character_intervals.
Readout/weights/OOF/threshold/model code is reused without modifications.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_development as q
import run_frozen_nli_local_signal_cuda_v1 as gpu

ROOT = q.ROOT
OUT = ROOT / 'results/nli_local_signal_cuda_scoring_v1'
GPU_OUT = gpu.OUT
GPU_SOURCE = Path(gpu.__file__)
LEARNING_FUNCTIONS = ('build_window_design', 'nested_weights', 'one_readout', 'fit_thresholds', 'result_record', 'score')


def protocol():
    value = copy.deepcopy(gpu.protocol())
    value['version'] = 'native-qa-frozen-local-nli-cuda-evaluation-validation-fix-v1'
    value['status'] = 'evaluation_only_repair_frozen_before_fitting'
    value['evaluation_repair'] = {
        'reason': 'Original score failed before any fit: character_intervals are merged, not a per-token offset list.',
        'authoritative_definition': 'src/build_gold.py merge_intervals and window construction: sort intervals, omit empty intervals, merge when next.start <= previous.end.',
        'change': 'Only the equality check of existing window.character_intervals uses that existing merged definition.',
        'unchanged': 'Every input, segment, four-view probability, raw BPE, feature projection, gold label, readout, class/group weight, fold, C, threshold rule and answermax.',
        'cache_root': str(GPU_OUT.resolve()), 'cache_identity': 'Original CUDA signature and all793 NPZ hashes checked without rebinding or rewriting them.',
        'execution_signature_sha256': q.digest(q.read(GPU_OUT / 'execution_signature.json')),
        'GPU_extraction_complete_sha256': q.sha(GPU_OUT / 'extraction_complete.json'),
        'outputs': 'Independent evaluation directory; original CPU/GPU directories remain unchanged.',
        'readout_source': str(gpu.CPU_SOURCE.resolve()), 'readout_source_sha256': q.sha(gpu.CPU_SOURCE),
        'GPU_runner_sha256': q.sha(GPU_SOURCE), 'learning_functions': list(LEARNING_FUNCTIONS),
        'stage_order': 'prepare/check then explicit score then verify; no GPU execution commands.',
    }
    return value


def merge_intervals(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def coordinate_identity(meta, prepared, save=True):
    plans = {r['response_id']: r for r in q.lines(gpu.cpu.DATA / 'feature_preparation/plans.jsonl')}
    lookup = {r['response_id']: r for r in prepared}; merged_different = 0
    assert set(plans) == set(lookup) == {a['response_id'] for a in meta['answers']}
    for answer in meta['answers']:
        rid = answer['response_id']; row = lookup[rid]; original = plans[rid]['original']
        token = meta['by_response'][rid]['tokens']
        assert row['answer_sha256'] == token['answer_sha256'] == plans[rid]['answer_sha256']
        assert row['raw_token_count'] == token['token_count'] == len(original['answer_token_ids'])
        assert row['response_token_offsets_sha256'] == gpu.cpu.digest(original['response_token_offsets'])
        assert row['original_token_ids_sha256'] == gpu.cpu.digest(original['answer_token_ids'])
        assert token['token_ids'] == original['answer_token_ids']
        assert token['answer_token_positions'] == original['answer_token_positions']
        assert token['response_token_offsets'] == original['response_token_offsets']
        assert token['response_token_offsets_raw'] == original['response_token_offsets_raw']
        assert (np.asarray(row['lexical_token_segment']) >= 0).tolist() == token['lexical_mask']
        for wi in meta['answer_windows'][rid]:
            window = meta['windows'][wi]; indices = window['token_indices']
            assert window['token_ids'] == [token['token_ids'][j] for j in indices]
            assert window['answer_token_positions'] == [token['answer_token_positions'][j] for j in indices]
            offsets = [token['response_token_offsets'][j] for j in indices]
            assert window['character_intervals'] == merge_intervals(offsets)
            merged_different += int(window['character_intervals'] != offsets)
    result = {'role': gpu.ROLE, 'answers_exact': len(meta['answers']), 'windows_exact': len(meta['windows']),
              'prepared_offset_and_token_ID_hashes_replayed': True, 'plan_to_gold_token_metadata_exact': True,
              'window_token_ID_position_and_merged_character_intervals_exact': True,
              'windows_whose_merged_intervals_differ_from_individual_BPE_offsets': merged_different,
              'lexical_segment_mask_exact': True, 'gold_or_mapping_changed': False, 'official_test_opened': False}
    if save:
        q.save(OUT / 'TOKEN_WINDOW_IDENTITY.json', result)
    return result


def source_paths():
    paths = [Path(__file__), GPU_SOURCE, gpu.CPU_SOURCE, Path(q.__file__), ROOT / 'src/build_gold.py']
    paths += [GPU_OUT / name for name in ('inputs.jsonl', 'protocol.json', 'execution_signature.json',
              'preparation_complete.json', 'source_snapshot.json', 'GPU_SMOKE.json', 'extraction_complete.json')]
    return paths


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'prepare_started.json').exists()
    rows, _ = gpu.check_prepared(); gpu.check_extracted(rows)
    assert not (GPU_OUT / 'nli_only_fixed_lr.pkl').exists() and not (GPU_OUT / 'complete.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'prepare_started.json', {'status': 'evaluation_only_preparation'})
    q.save(OUT / 'FAILURE_01_UNMERGED_ASSERT.json', {
        'stage': 'original_CUDA_score', 'session_id': 66406, 'actual_exit_code': 1,
        'source': str(gpu.CPU_SOURCE.resolve()), 'function': 'replay_token_window_identity',
        'assertion': "window['character_intervals'] == [token['response_token_offsets'][j] for j in indices]",
        'failure_line': 718, 'new_fits_before_failure': 0, 'GPU_extraction_unchanged': True,
        'resolution': 'Existing field stores merged intervals. New evaluation-only validator; no input/label/probability change.'})
    for name in ('inputs.jsonl', 'extraction_complete.json'):
        pending = OUT / (name + '.pending'); pending.write_bytes((GPU_OUT / name).read_bytes()); pending.replace(OUT / name)
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'source_snapshot.json', {'files_sha256': {str(p.resolve()): q.sha(p) for p in source_paths()}})
    assert merge_intervals([[5, 8], [0, 2], [2, 5], [10, 10], [12, 15], [13, 16]]) == [[0, 8], [12, 16]]
    identity = coordinate_identity(q.metadata(), rows, save=False)
    bound = readout()
    same = {name: inspect.getsource(getattr(bound, name)) == inspect.getsource(getattr(gpu.cpu, name)) for name in LEARNING_FUNCTIONS}
    assert all(same.values())
    q.save(OUT / 'CPU_CHECK.json', {'status': 'passed', 'identity': identity,
           'unchanged_learning_functions': same, 'original_extraction_signature_validated': True,
           'all793_original_cache_hashes_validated': True, 'GPU_used': False, 'new_fits': 0, 'official_test_opened': False})
    (OUT / 'PLAN.md').write_text(
        '# NLI CUDA：仅修正评测坐标校验\n\n'
        '原GPU提取源码、协议、执行签名和793答概率缓存均不改。旧score在任何拟合前失败：'
        'character_intervals本来就是合并区间，旧检查误与逐BPE列表比较。新入口只按build_gold.py原定义修正该断言。\n\n'
        '原输入/标注/12维映射、两组5折LR及全fit模型、权重和阈值规则全部复用冻结源码。'
        '读取概率时仍严格校验原GPU签名与每文件哈希，没有将新评测源码塞回提取签名。'
        '独立结果目录，保留首次失败。仍为我们的方法候选，不是正式基线。\n', encoding='utf-8')
    names = ['inputs.jsonl', 'extraction_complete.json', 'protocol.json', 'source_snapshot.json',
             'CPU_CHECK.json', 'PLAN.md', 'FAILURE_01_UNMERGED_ASSERT.json']
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_evaluation_repair_not_fitted',
           'files_sha256': {name: q.sha(OUT / name) for name in names}, 'GPU_used': False, 'new_fits': 0})
    print('NLI_CUDA_EVALUATION_REPAIR_PREPARED_EXACT_CACHE', flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['status'] == 'prepared_evaluation_repair_not_fitted'
    for name, expected in p['files_sha256'].items():
        assert q.sha(OUT / name) == expected, name
    for name, expected in q.read(OUT / 'source_snapshot.json')['files_sha256'].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / 'protocol.json') == protocol()
    assert q.sha(OUT / 'inputs.jsonl') == q.sha(GPU_OUT / 'inputs.jsonl')
    assert q.sha(OUT / 'extraction_complete.json') == q.sha(GPU_OUT / 'extraction_complete.json')
    rows, _ = gpu.check_prepared()
    return rows, p


def check_extracted(rows):
    assert q.sha(OUT / 'extraction_complete.json') == q.sha(GPU_OUT / 'extraction_complete.json')
    return gpu.check_extracted(rows)


def check():
    assert not torch.cuda.is_initialized()
    rows, _ = check_prepared(); check_extracted(rows)
    assert q.read(OUT / 'CPU_CHECK.json')['status'] == 'passed'
    print('NLI_CUDA_EVALUATION_REPAIR_CHECKED', flush=True)


def source_cache(path, row):
    # Explicit read-only location adapter. Validate the original execution
    # identity and bytes, rather than pretending the cache was re-extracted.
    assert Path(path).resolve() == (OUT / 'segment_scores' / f"{row['response_id']}.npz").resolve()
    return gpu.validate_cache(GPU_OUT / 'segment_scores' / Path(path).name, row)


def segment_features(row):
    cube = gpu.validate_cache(GPU_OUT / 'segment_scores' / f"{row['response_id']}.npz", row)
    return cube.reshape(len(row['segments']), -1), cube[:, :, 2]


def readout():
    bound = gpu.private_source('_nli_readout_coordinate_check_repair')
    bound.OUT = OUT; bound.protocol = protocol; bound.check_prepared = check_prepared
    bound.check_extracted = check_extracted; bound.load_segment_features = segment_features
    bound.replay_token_window_identity = coordinate_identity
    assert gpu.OUT == GPU_OUT and gpu.cpu.OUT == gpu.CPU_OUT
    return bound


def score():
    assert not torch.cuda.is_initialized()
    readout().score()


def verify():
    assert not torch.cuda.is_initialized()
    # Reuse the already frozen independent probability/threshold audit in a
    # private namespace. Only output paths and the explicit source-cache reader
    # are rebound; its mathematics and fitting-free behavior remain unchanged.
    spec = importlib.util.spec_from_file_location('_nli_cuda_readonly_verify_for_evaluation_fix', GPU_SOURCE)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    verifier.OUT = OUT; verifier.check_prepared = check_prepared
    verifier.check_extracted = check_extracted; verifier.validate_cache = source_cache
    verifier.verify()
    assert gpu.OUT == GPU_OUT and gpu.cpu.OUT == gpu.CPU_OUT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'check', 'score', 'verify'))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'check': check, 'score': score, 'verify': verify}[arguments.stage]()
