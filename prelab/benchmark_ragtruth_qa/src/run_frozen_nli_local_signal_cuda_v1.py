"""Device-only CUDA execution of the frozen local NLI candidate (ours).

The CPU source and its outputs are read-only. Exact prepared text/segmentation,
four premise views, checkpoint and readout functions are reused. All commands
are explicit; prepare/check never load a model or initialize CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import run_development as q

ROOT = q.ROOT
CPU_SOURCE = ROOT / 'src/run_frozen_nli_local_signal_v1.py'
CPU_OUT = ROOT / 'results/nli_local_signal_v1'
OUT = ROOT / 'results/nli_local_signal_cuda_v1'
BATCH, THREADS, SEED = 8, 4, 20261012
ROLE = 'ours_method_candidate; never a baseline'
BACKEND = 'cuda_fp32_sdpa_math_batch8_v1'
CPU_GPU_ATOL = 3e-5
PEAK_LIMIT = int(6.5 * 1024 ** 3)
MIN_FREE = 3 * 1024 ** 3


def private_source(name):
    spec = importlib.util.spec_from_file_location(name, CPU_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = private_source('_frozen_nli_cpu_readonly_for_cuda')
CPU_SEMANTICS = copy.deepcopy(cpu.protocol())


def protocol():
    value = copy.deepcopy(CPU_SEMANTICS)
    value['version'] = 'native-qa-frozen-local-nli-cuda-v1'
    value['status'] = 'prepared_only_waiting_for_GPU_authorization'
    value['role'] = ROLE
    value['nli']['device'] = 'CUDA0 FP32, eval/inference_mode; no autocast or TF32; SDPA math; reference_compile=False'
    value['nli']['batch_pairs'] = BATCH
    value['nli']['independent_replay'] = 'Original CPU independent-replay report is retained as upstream evidence. CUDA smoke compares its same108 pairs at fixed3e-5 probability tolerance; same-path CUDA repeats must be exact. No CPU/GPU bit-identity claim.'
    value['execution_gate'] = 'Explicit prepare/check/gpu-smoke/extract/score/verify. No automatic queue or stage chaining.'
    value['device_transfer'] = {
        'CPU_source': str(CPU_SOURCE.resolve()), 'CPU_result_directory': str(CPU_OUT.resolve()),
        'input_policy': 'Copy original793 inputs.jsonl byte-for-byte. Do not regenerate segments, sources or token ownership.',
        'only_execution_changes': ['device CPU->CUDA0', 'FP32 math SDPA selection, TF32 disabled',
                                   'reference_compile=False to use the same uncompiled math on Windows'],
        'weights_and_config': 'Original checkpoint and all learned weights unchanged. No architecture/attention-mask/pooling/position configuration change.',
        'batch_policy': 'Same original pair order and per-answer batch8, including actual final short batch. No length sorting, merging, padding to fixed cap or truncation.',
        'backend': BACKEND, 'same_path_repeat_exact': True,
        'CPU_GPU_probability_atol': CPU_GPU_ATOL, 'CPU_GPU_probability_rtol': 0,
        'memory_limit_bytes': PEAK_LIMIT, 'minimum_free_before_load_bytes': MIN_FREE,
        'smoke': 'Original first/middle/last108 pairs with original CPU joint-batch geometry; also each anchor production-row geometry and the first longest prepared pair batch. No labels or score-based sample choice.',
        'failure': 'Record failure and stop. No automatic tolerance change, batch reduction, FP16/BF16 fallback or retry.',
        'cache': 'New directory and execution signature in every NPZ; CPU segment caches are never imported or accepted.',
        'readout': 'Private copy of the frozen CPU module reuses unchanged fit/weights/scaler/OOF/threshold/answermax functions. Only its output/check bindings differ; no CPU module globals or files are modified.',
        'numerical_limit': 'Device arithmetic may cause small probability differences. Passing smoke is tolerance agreement, never bit-exact CPU/GPU equivalence.',
    }
    return value


def atomic_npz(path, **arrays):
    assert not path.exists(), f'No overwrite: {path}'
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def source_paths():
    return [Path(__file__), CPU_SOURCE, Path(q.__file__), ROOT / 'src/feature_qa.py',
            CPU_OUT / 'protocol.json', CPU_OUT / 'inputs.jsonl', CPU_OUT / 'preparation_complete.json',
            CPU_OUT / 'source_snapshot.json', CPU_OUT / 'CPU_WIRING_CHECK.json',
            CPU_OUT / 'CPU_WIRING_PROBABILITIES.npz', CPU_OUT / 'INDEPENDENT_VERIFY_REPLAY.json',
            cpu.MODEL / 'config.json', cpu.MODEL / 'tokenizer.json', cpu.MODEL / 'tokenizer_config.json',
            cpu.MODEL / 'special_tokens_map.json', cpu.MODEL / 'download_manifest.json',
            Path(importlib.util.find_spec('transformers.models.modernbert.modeling_modernbert').origin)]


def semantic_equivalence():
    original, device = copy.deepcopy(CPU_SEMANTICS), protocol()
    assert original['nli']['batch_pairs'] == device['nli']['batch_pairs'] == BATCH
    device.pop('device_transfer')
    for key in ('version', 'status', 'execution_gate'):
        device[key] = original[key]
    device['nli']['device'] = original['nli']['device']
    device['nli']['independent_replay'] = original['nli']['independent_replay']
    assert device == original, 'A change other than declared device execution was introduced'


def upstream_check():
    assert not torch.cuda.is_initialized()
    rows, prepared = cpu.check_prepared()
    assert q.read(CPU_OUT / 'protocol.json') == CPU_SEMANTICS
    replay = q.read(CPU_OUT / 'INDEPENDENT_VERIFY_REPLAY.json')
    assert replay['status'] == 'independent_preparation_and_real3_replay_passed'
    assert replay['runner_sha256'] == q.sha(CPU_SOURCE)
    assert replay['input_sha256'] == q.sha(CPU_OUT / 'inputs.jsonl')
    assert replay['protocol_sha256'] == q.sha(CPU_OUT / 'protocol.json')
    wiring = q.read(CPU_OUT / 'CPU_WIRING_CHECK.json')
    assert wiring['input_file_sha256'] == q.sha(CPU_OUT / 'inputs.jsonl')
    assert wiring['preparation_sha256'] == q.sha(CPU_OUT / 'preparation_complete.json')
    assert wiring['probability_cache_sha256'] == q.sha(CPU_OUT / 'CPU_WIRING_PROBABILITIES.npz')
    assert len(rows) == 793 and prepared['segments'] == 8829 and prepared['NLI_pairs'] == 35316
    return rows, prepared


def runtime_signature():
    return {'backend': BACKEND, 'source_sha256': q.sha(__file__),
            'CPU_source_sha256': q.sha(CPU_SOURCE), 'protocol_sha256': q.sha(OUT / 'protocol.json'),
            'input_sha256': q.sha(OUT / 'inputs.jsonl'), 'model_sha256': cpu.MODEL_SHA256,
            'revision': cpu.MODEL_REVISION, 'batch': BATCH, 'precision': 'float32',
            'software': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'tokenizers')},
            'cuda_runtime': torch.version.cuda, 'role': ROLE}


def infer_pairs(tokenizer, model, pairs, device):
    """The production pair order/padding/softmax path; CPU also supports tiny checks."""
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    outputs = []; batches = []
    for left in range(0, len(pairs), BATCH):
        batch = pairs[left:left + BATCH]
        encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], add_special_tokens=True,
                            padding=True, truncation=False, return_tensors='pt')
        assert encoded['input_ids'].shape[1] <= cpu.PAIR_LIMIT
        batches.append([len(batch), int(encoded['input_ids'].shape[1])])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                   if device.type == 'cuda' else nullcontext())
        with torch.inference_mode(), torch.autocast(device.type, enabled=False), context:
            logits = model(**encoded).logits
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            outputs.append(logits.softmax(-1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    assert result.shape == (len(pairs), 3) and np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all() and np.allclose(result.sum(1), 1, rtol=0, atol=2e-6)
    return result, batches


def cpu_selfcheck():
    # Test the actual batching path without loading pretrained parameters.
    from types import SimpleNamespace
    class Tokenizer:
        def __call__(self, premise, hypothesis, **kwargs):
            assert kwargs == dict(add_special_tokens=True, padding=True, truncation=False, return_tensors='pt')
            lengths = [len(a) + len(b) + 2 for a, b in zip(premise, hypothesis)]
            ids = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, length in enumerate(lengths):
                ids[i, :length] = torch.arange(1, length + 1); mask[i, :length] = 1
            return {'input_ids': ids, 'attention_mask': mask}
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            values = (input_ids * attention_mask).sum(1).float() / 100
            return SimpleNamespace(logits=torch.stack([values, -values, torch.zeros_like(values)], -1))
    model = Model().eval().requires_grad_(False)
    pairs = [('x' * (i % 3 + 1), 'y') for i in range(11)]
    a, batches = infer_pairs(Tokenizer(), model, pairs, torch.device('cpu'))
    b, _ = infer_pairs(Tokenizer(), model, pairs, torch.device('cpu'))
    assert batches == [[8, 6], [3, 6]] and np.array_equal(a, b)
    assert np.array_equal(a[0], a[3])
    # Exercise the actual record validator, including rejection of a CPU-style
    # cache with identical numerical values but no CUDA execution provenance.
    row = {'response_id': 'cpu_toy_only', 'full_premise': 'premise',
           'passage_premises': ['one', 'two', 'three'],
           'segments': [{'segment_id': 0, 'hypothesis': 'claim'}]}
    signature = cache_signature(row)
    arrays = {**{k: np.asarray(v) for k, v in signature.items()},
              'segment_ids': np.asarray([0], np.int32), 'view_names': np.asarray(cpu.VIEWS),
              **{name: np.full((1, 4), 1 / 3, np.float32) for name in cpu.CLASSES}}
    with tempfile.TemporaryDirectory() as temp:
        good = Path(temp) / 'good.npz'; atomic_npz(good, **arrays); validate_cache(good, row)
        bad = dict(arrays); bad.pop('execution_backend'); bad.pop('execution_signature_sha256')
        wrong = Path(temp) / 'cpu.npz'; atomic_npz(wrong, **bad)
        try:
            validate_cache(wrong, row)
            raise RuntimeError('CPU-style cache was accepted')
        except AssertionError:
            pass
    semantic_equivalence()
    return {'status': 'passed', 'same_semantics_except_declared_execution': True,
            'actual_inference_batch_loop_toy_8_plus3': True, 'padding_and_pair_order': True,
            'E_N_C_three_columns_and_finite_FP32': True, 'same_path_repeat_exact': True,
            'CPU_cache_rejected_by_actual_validator': True, 'pretrained_model_loaded': False,
            'GPU_initialized': torch.cuda.is_initialized(), 'new_fits': 0, 'official_test_opened': False}


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'prepare_started.json').exists(), 'No preparation overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'prepare_started.json', {'status': 'started_CPU_only'})
    rows, source_prepared = upstream_check(); semantic_equivalence()
    pending = OUT / 'inputs.jsonl.pending'
    pending.write_bytes((CPU_OUT / 'inputs.jsonl').read_bytes()); pending.replace(OUT / 'inputs.jsonl')
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'source_snapshot.json', {'files_sha256': {str(p.resolve()): q.sha(p) for p in source_paths()}})
    q.save(OUT / 'execution_signature.json', runtime_signature())
    q.save(OUT / 'CPU_SELFCHECK.json', cpu_selfcheck())
    longest = max(range(len(rows)), key=lambda i: (max(rows[i]['pair_token_lengths']), -i))
    q.save(OUT / 'resource_plan.json', {
        'answers': len(rows), 'segments': source_prepared['segments'], 'pairs': source_prepared['NLI_pairs'],
        'pair_batch': BATCH, 'maximum_pair_tokens': source_prepared['maximum_pair_tokens'],
        'longest_answer_index': longest, 'longest_response_id': rows[longest]['response_id'],
        'model_parameters': 149607171, 'FP32_parameter_bytes': 149607171 * 4,
        'one_dense_attention_matrix_bytes_at_max8x12': BATCH * 12 * source_prepared['maximum_pair_tokens'] ** 2 * 4,
        'expected_gpu_peak_GiB_before_measurement': [1.5, 3.5], 'hard_peak_limit_bytes': PEAK_LIMIT,
        'disk_reserve_GiB': 1, 'three_probability_values_bytes': source_prepared['NLI_pairs'] * 3 * 4,
        'runtime': 'GPU time unknown until smoke; no CPU-to-GPU speedup promise. No automatic batch changes.',
        'GPU_measured': False})
    text = ('# 冻结 NLI CUDA 执行版（我们的方法候选）\n\n'
            '同793输入/8829片段/四视图/35316对、同ModernBERT参数与E/N/C；CPU产物只读。'
            'CUDA FP32 batch8，禁混精度、TF32及编译，使用SDPA math。原读出源码通过私有模块复用，新输出目录独立。\n\n'
            'prepare/check无模型和GPU；gpu-smoke比较原CPU108对（最大概率差≤3e-5）、生产逐答批次和最长683词元批。'
            '同路径重复exact，峰值allocated/reserved均≤6.5GiB；失败保留，不降批或放宽。extract须独立显式授权，'
            '只接受带新CUDA签名的逐答缓存。score另起CPU进程；verify只复算，不重新拟合。\n\n'
            '5折只针对读出层，current上游仍in-sample且曾用cal选型；本分支不是正式baseline或独立最终测试。\n')
    (OUT / 'PLAN.md').write_text(text, encoding='utf-8')
    names = ['inputs.jsonl', 'protocol.json', 'source_snapshot.json', 'execution_signature.json',
             'CPU_SELFCHECK.json', 'resource_plan.json', 'PLAN.md']
    q.save(OUT / 'preparation_complete.json', {'status': 'prepared_CPU_only_not_extracted', 'role': ROLE,
           'CPU_preparation_sha256': q.sha(CPU_OUT / 'preparation_complete.json'),
           'CPU_inputs_byte_exact': q.sha(OUT / 'inputs.jsonl') == q.sha(CPU_OUT / 'inputs.jsonl'),
           'files_sha256': {name: q.sha(OUT / name) for name in names},
           'GPU_used': False, 'new_fits': 0, 'official_test_opened': False})
    assert not torch.cuda.is_initialized()
    print('FROZEN_NLI_CUDA_PREPARED_CPU_ONLY', flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['status'] == 'prepared_CPU_only_not_extracted' and p['CPU_inputs_byte_exact']
    assert p['CPU_preparation_sha256'] == q.sha(CPU_OUT / 'preparation_complete.json')
    for name, expected in p['files_sha256'].items():
        assert q.sha(OUT / name) == expected, name
    for name, expected in q.read(OUT / 'source_snapshot.json')['files_sha256'].items():
        assert q.sha(name) == expected, name
    assert q.read(OUT / 'protocol.json') == protocol()
    assert q.read(OUT / 'execution_signature.json') == runtime_signature()
    assert q.sha(OUT / 'inputs.jsonl') == q.sha(CPU_OUT / 'inputs.jsonl')
    rows = q.lines(OUT / 'inputs.jsonl'); assert len(rows) == 793
    return rows, p


def check():
    assert not torch.cuda.is_initialized()
    rows, _ = check_prepared(); upstream_check(); semantic_equivalence()
    assert sum(len(row['segments']) for row in rows) == 8829
    assert q.read(OUT / 'CPU_SELFCHECK.json')['status'] == 'passed'
    assert not torch.cuda.is_initialized()
    print('FROZEN_NLI_CUDA_CHECKED_CPU_ONLY', flush=True)


def load_cuda():
    # Called only by explicitly selected GPU commands, after CPU binding checks.
    assert q.sha(cpu.MODEL / 'model.safetensors') == cpu.MODEL_SHA256
    assert q.read(cpu.MODEL / 'download_manifest.json')['revision'] == cpu.MODEL_REVISION
    torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info(0)
    assert free >= MIN_FREE, f'Insufficient free GPU memory: {free}'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(cpu.MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(cpu.MODEL, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa', reference_compile=False).eval().requires_grad_(False).to('cuda:0')
    assert model.config.id2label == {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    assert model.config.max_position_embeddings == cpu.PAIR_LIMIT
    assert model.config._attn_implementation == 'sdpa' and model.config.reference_compile is False
    assert all(p.dtype == torch.float32 and p.device.type == 'cuda' and not p.requires_grad for p in model.parameters())
    metadata = {'device': torch.cuda.get_device_name(0), 'total_memory_bytes': total,
                'free_before_load_bytes': free, 'parameters': sum(p.numel() for p in model.parameters()),
                'parameter_dtype': 'float32', 'model_sha256': cpu.MODEL_SHA256, 'backend': BACKEND}
    return tokenizer, model, metadata


def memory_gate():
    values = {'allocated_peak_bytes': torch.cuda.max_memory_allocated(0),
              'reserved_peak_bytes': torch.cuda.max_memory_reserved(0)}
    assert max(values.values()) <= PEAK_LIMIT, values
    return values


def clean_gpu():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def failure(stage, error):
    path = OUT / f'{stage}_FAILURE.json'
    if path.exists():
        return
    q.save(path, {'stage': stage, 'status': 'failed_stop_no_fallback', 'error': repr(error),
                 'execution_signature_sha256': q.digest(runtime_signature()), 'GPU_used': torch.cuda.is_initialized(),
                 'role': ROLE, 'official_test_opened': False})


def gpu_smoke():
    rows, _ = check_prepared()
    assert not (OUT / 'GPU_SMOKE.json').exists() and not (OUT / 'gpu-smoke_FAILURE.json').exists()
    model = tokenizer = None; started = time.perf_counter()
    try:
        tokenizer, model, loaded = load_cuda()
        chosen = [0, len(rows) // 2, len(rows) - 1]
        pairs = []; owners = []
        for i in chosen:
            one, indexes = cpu.row_pairs(rows[i]); pairs.extend(one)
            owners.extend((rows[i]['response_id'], *key) for key in indexes)
        a, shapes = infer_pairs(tokenizer, model, pairs, torch.device('cuda:0'))
        b, shapes2 = infer_pairs(tokenizer, model, pairs, torch.device('cuda:0'))
        assert shapes == shapes2 and np.array_equal(a, b)
        with np.load(CPU_OUT / 'CPU_WIRING_PROBABILITIES.npz', allow_pickle=False) as old:
            assert old['owner_response_ids'].tolist() == [x[0] for x in owners]
            assert old['owner_segment_ids'].tolist() == [x[1] for x in owners]
            assert old['owner_view_indices'].tolist() == [x[2] for x in owners]
            assert old['selected_input_row_sha256'].tolist() == [cpu.digest(rows[i]) for i in chosen]
            old_values = old['probabilities'].copy()
        difference = float(np.max(np.abs(a - old_values)))
        assert difference <= CPU_GPU_ATOL, ('CPU_vs_GPU', difference)
        row_outputs = []
        for i in chosen:
            one, _ = cpu.row_pairs(rows[i]); values, _ = infer_pairs(tokenizer, model, one, torch.device('cuda:0'))
            row_outputs.append(values)
        production = np.concatenate(row_outputs)
        row_difference = float(np.max(np.abs(production - a)))
        assert row_difference <= CPU_GPU_ATOL
        longest = q.read(OUT / 'resource_plan.json')['longest_answer_index']
        longest_pairs, _ = cpu.row_pairs(rows[longest])
        long_a, long_shapes = infer_pairs(tokenizer, model, longest_pairs, torch.device('cuda:0'))
        long_b, _ = infer_pairs(tokenizer, model, longest_pairs, torch.device('cuda:0'))
        assert np.array_equal(long_a, long_b) and max(x[1] for x in long_shapes) == 683
        memory = memory_gate(); torch.cuda.synchronize()
        atomic_npz(OUT / 'GPU_SMOKE_PROBABILITIES.npz', anchor_probabilities=a,
                   anchor_production_probabilities=production, longest_probabilities=long_a,
                   response_ids=np.asarray([rows[i]['response_id'] for i in chosen]),
                   execution_signature_sha256=np.asarray(q.digest(runtime_signature())))
        q.save(OUT / 'GPU_SMOKE.json', {'status': 'passed', 'role': ROLE,
               'execution_signature_sha256': q.digest(runtime_signature()), 'loaded': loaded,
               'CPU_anchor_pairs': len(pairs), 'CPU_GPU_max_abs_error': difference,
               'joint_vs_production_row_max_abs_error': row_difference, 'same_path_repeat_exact': True,
               'anchor_batch_shapes': shapes, 'longest_batch_shapes': long_shapes, **memory,
               'seconds': time.perf_counter() - started, 'probabilities_sha256': q.sha(OUT / 'GPU_SMOKE_PROBABILITIES.npz'),
               'GPU_used': True, 'trained': False, 'official_test_opened': False})
        print('FROZEN_NLI_CUDA_SMOKE_PASSED_NO_AUTOMATIC_EXTRACT', flush=True)
    except Exception as error:
        failure('gpu-smoke', error); raise
    finally:
        del model, tokenizer; clean_gpu()


def cache_signature(row):
    pairs, owners = cpu.row_pairs(row)
    return {'response_id': row['response_id'], 'input_row_sha256': cpu.digest(row),
            'pair_definition_sha256': cpu.digest({'pairs': pairs, 'owners': owners}),
            'input_file_sha256': q.sha(OUT / 'inputs.jsonl'), 'protocol_sha256': q.sha(OUT / 'protocol.json'),
            'model_sha256': cpu.MODEL_SHA256, 'execution_signature_sha256': q.digest(runtime_signature()),
            'execution_backend': BACKEND, 'method_role': ROLE}


def validate_cache(path, row):
    signature = cache_signature(row)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(signature) | {'segment_ids', 'view_names', *cpu.CLASSES}
        for key, value in signature.items():
            assert data[key].item() == value, (path.name, key)
        assert data['segment_ids'].tolist() == list(range(len(row['segments'])))
        assert data['view_names'].tolist() == list(cpu.VIEWS)
        cube = np.stack([data[name] for name in cpu.CLASSES], axis=-1)
        assert cube.dtype == np.float32 and cube.shape == (len(row['segments']), 4, 3)
        assert np.isfinite(cube).all() and ((cube >= 0) & (cube <= 1)).all()
        assert np.allclose(cube.sum(-1), 1, rtol=0, atol=2e-6)
    return cube


def extract():
    rows, _ = check_prepared(); smoke = q.read(OUT / 'GPU_SMOKE.json')
    assert smoke['status'] == 'passed' and smoke['execution_signature_sha256'] == q.digest(runtime_signature())
    assert q.sha(OUT / 'GPU_SMOKE_PROBABILITIES.npz') == smoke['probabilities_sha256']
    assert not (OUT / 'extract_FAILURE.json').exists(), 'Failure retained; no automatic retry'
    if (OUT / 'extraction_complete.json').exists():
        check_extracted(rows); print('FROZEN_NLI_CUDA_ALREADY_COMPLETE_NO_GPU', flush=True); return
    folder = OUT / 'segment_scores'; folder.mkdir(exist_ok=True)
    missing = []
    for i, row in enumerate(rows):
        path = folder / f"{row['response_id']}.npz"
        if path.exists():
            validate_cache(path, row)
        else:
            missing.append(i)
    tokenizer = model = None; started = time.perf_counter(); loaded = None
    try:
        if missing:
            tokenizer, model, loaded = load_cuda()
            assert loaded['device'] == smoke['loaded']['device']
            for count, i in enumerate(missing, 1):
                row = rows[i]; pairs, owners = cpu.row_pairs(row)
                values, shapes = infer_pairs(tokenizer, model, pairs, torch.device('cuda:0'))
                cube = np.empty((len(row['segments']), 4, 3), np.float32)
                for value, (si, vi) in zip(values, owners):
                    cube[si, vi] = value
                memory_gate()
                path = folder / f"{row['response_id']}.npz"
                atomic_npz(path, **{k: np.asarray(v) for k, v in cache_signature(row).items()},
                           segment_ids=np.arange(len(row['segments']), dtype=np.int32), view_names=np.asarray(cpu.VIEWS),
                           **{name: cube[:, :, j] for j, name in enumerate(cpu.CLASSES)})
                validate_cache(path, row)
                if count == 1 or count % 25 == 0 or count == len(missing):
                    progress = {'status': 'running', 'pid': os.getpid(), 'complete': len(rows) - len(missing) + count,
                                'total': len(rows), 'seconds_this_run': time.perf_counter() - started}
                    q.save(OUT / 'progress.json', progress); print(progress, flush=True)
        files = {}
        for row in rows:
            path = folder / f"{row['response_id']}.npz"; validate_cache(path, row); files[path.name] = q.sha(path)
        q.save(OUT / 'extraction_complete.json', {'status': 'complete_frozen_probabilities_not_scored', 'role': ROLE,
               'answers': len(rows), 'segments': 8829, 'pairs': 35316, 'files_sha256': files,
               'input_sha256': q.sha(OUT / 'inputs.jsonl'), 'protocol_sha256': q.sha(OUT / 'protocol.json'),
               'model_sha256': cpu.MODEL_SHA256, 'execution_signature_sha256': q.digest(runtime_signature()),
               'class_order': list(cpu.CLASSES), 'view_order': list(cpu.VIEWS),
               'GPU_SMOKE_sha256': q.sha(OUT / 'GPU_SMOKE.json'), 'new_rows_this_run': len(missing),
               'seconds_this_run': time.perf_counter() - started, 'loaded': loaded,
               'annotation_values_accessed': False, 'GPU_used': bool(missing), 'trained': False,
               'official_test_opened': False})
        print('FROZEN_NLI_CUDA_EXTRACTED_ALL793_NO_AUTOMATIC_SCORE', flush=True)
    except Exception as error:
        failure('extract', error); raise
    finally:
        del model, tokenizer; clean_gpu()


def check_extracted(rows):
    done = q.read(OUT / 'extraction_complete.json')
    assert done['status'] == 'complete_frozen_probabilities_not_scored'
    assert done['answers'] == 793 and done['segments'] == 8829 and done['pairs'] == 35316
    assert done['execution_signature_sha256'] == q.digest(runtime_signature())
    assert done['input_sha256'] == q.sha(OUT / 'inputs.jsonl')
    assert done['protocol_sha256'] == q.sha(OUT / 'protocol.json') and done['model_sha256'] == cpu.MODEL_SHA256
    assert done['GPU_SMOKE_sha256'] == q.sha(OUT / 'GPU_SMOKE.json')
    assert set(done['files_sha256']) == {row['response_id'] + '.npz' for row in rows}
    for row in rows:
        path = OUT / 'segment_scores' / f"{row['response_id']}.npz"
        assert q.sha(path) == done['files_sha256'][path.name]; validate_cache(path, row)
    return done


def bound_readout():
    # A new module namespace, never mutate cpu or the canonical imported module.
    bound = private_source('_frozen_nli_readout_private_cuda_output')
    bound.OUT = OUT
    bound.protocol = protocol
    bound.check_prepared = check_prepared
    bound.check_extracted = check_extracted
    bound.validate_score_file = validate_cache
    return bound


def score():
    assert not torch.cuda.is_initialized(), 'Scoring is a separate CPU process'
    bound = bound_readout()
    assert cpu.OUT == CPU_OUT and bound.OUT == OUT
    bound.score()  # Original fit/threshold/metrics code, without algorithm changes.


def manual_threshold(y, s):
    values, inverse = np.unique(s, return_inverse=True)
    counts = np.bincount(inverse); positives = np.bincount(inverse, weights=y)
    tp = np.r_[0, np.cumsum(positives[::-1])]; predicted = np.r_[0, np.cumsum(counts[::-1])]
    thresholds = np.r_[np.nextafter(values[-1], np.inf), values[::-1]]
    f1 = 2 * tp / (predicted + y.sum())
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    best = max(range(len(f1)), key=lambda i: (f1[i], precision[i], thresholds[i]))
    return {'threshold': float(thresholds[best]), 'f1': float(f1[best]), 'precision': float(precision[best]),
            'rows': len(y), 'positive': int(y.sum())}


def replay_probability(x, mean, scale, coefficient, intercept):
    z = x.copy(); z -= np.asarray(mean); z /= np.asarray(scale)
    return expit(z @ np.asarray(coefficient)[0] + np.asarray(intercept)[0])


def verify():
    assert not torch.cuda.is_initialized()
    rows, _ = check_prepared(); check_extracted(rows)
    complete = q.read(OUT / 'complete.json')
    assert complete['status'] == 'complete_development_only' and complete['role'] == ROLE
    for name, expected in complete['files_sha256'].items():
        assert q.sha(OUT / name) == expected, name
    assert complete['protocol_sha256'] == q.sha(OUT / 'protocol.json')
    assert complete['input_sha256'] == q.sha(OUT / 'inputs.jsonl')
    assert complete['extraction_sha256'] == q.sha(OUT / 'extraction_complete.json')
    meta = q.metadata(); features = np.load(OUT / 'window_nli_features.npy')
    assert [r['response_id'] for r in rows] == [a['response_id'] for a in meta['answers']]
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as data:
        predictions = {k: data[k].copy() for k in ('raw_max_contradiction', 'nli_only_fixed_lr', 'current_plus_nli_fixed_lr', 'current_score')}
        assert data['method_role'].item() == ROLE
    independent = np.empty_like(features); raw = np.empty(len(features), np.float64)
    for row in rows:
        cube = validate_cache(OUT / 'segment_scores' / f"{row['response_id']}.npz", row)
        vector = cube.reshape(len(cube), 12); owner = row['lexical_token_segment']
        for wi in meta['answer_windows'][row['response_id']]:
            ids = [owner[k] for k in meta['windows'][wi]['token_indices'] if owner[k] >= 0]
            independent[wi] = np.mean(vector[ids].astype(np.float64), axis=0)
            raw[wi] = float(cube[ids, :, 2].max())
    assert np.allclose(features, independent, rtol=0, atol=1e-7)
    assert np.array_equal(raw, predictions['raw_max_contradiction'])
    with np.load(cpu.CURRENT, allow_pickle=False) as original:
        assert np.array_equal(original['window_scores'], predictions['current_score'])
    fit_end = cpu.EXPECTED_WINDOWS['fit']; errors = {}
    groups = np.asarray([w['group_id'] for w in meta['windows'][:fit_end]])
    expected_folds = np.full(fit_end, -1, np.int8)
    for fold, (_, held) in enumerate(cpu.GroupKFold(n_splits=5).split(features[:fit_end], groups=groups)):
        expected_folds[held] = fold
    for name, x in [('nli_only_fixed_lr', features), ('current_plus_nli_fixed_lr',
                     np.column_stack((predictions['current_score'], features)).astype(np.float32))]:
        with (OUT / f'{name}.pkl').open('rb') as handle:
            artifact = pickle.load(handle)
        assert np.array_equal(artifact['fold_id'], expected_folds)
        replay = np.empty(len(x), np.float64)
        for record in artifact['fold_records']:
            held = np.flatnonzero(expected_folds == record['fold'])
            train = np.flatnonzero(expected_folds != record['fold'])
            assert not set(groups[held]) & set(groups[train])
            replay[held] = replay_probability(x[held], record['scaler_mean'], record['scaler_scale'],
                                              record['coefficient'], record['intercept'])
        scaler = artifact['full_scaler']; classifier = artifact['full_classifier']
        replay[fit_end:] = replay_probability(x[fit_end:], scaler.mean_, scaler.scale_, classifier.coef_, classifier.intercept_)
        errors[name] = float(np.max(np.abs(replay - predictions[name])))
        assert errors[name] <= 1e-10
    freeze = q.read(OUT / 'fit_freeze_before_calibration_report.json'); summary = q.read(OUT / 'summary.json')
    wy = np.asarray([w['label'] for w in meta['windows']], np.int64)
    ay = np.asarray([a['label'] for a in meta['answers']], np.int64)
    for name in ('raw_max_contradiction', 'nli_only_fixed_lr', 'current_plus_nli_fixed_lr'):
        ws = predictions[name]
        ans = np.asarray([max(ws[meta['answer_windows'][a['response_id']]]) for a in meta['answers']])
        thresholds = {'window': manual_threshold(wy[:fit_end], ws[:fit_end]),
                      'answer': manual_threshold(ay[:634], ans[:634])}
        assert thresholds == freeze['thresholds'][name] == summary['results'][name]['thresholds']
        assert q.metrics(meta, ws, thresholds) == summary['results'][name]['metrics']
    report = {'status': 'passed', 'role': ROLE, 'complete_sha256': q.sha(OUT / 'complete.json'),
              'all_GPU_cache_identities': True, 'all_answers': 793, 'all_windows': len(features),
              'independent_projection_max_abs_error': float(np.max(np.abs(features - independent))),
              'ten_fold_and_two_full_LR_probability_errors': errors,
              'fit_only_threshold_independent_counts_exact': True, 'unchanged_current_exact': True,
              'all_window_and_answer_metrics_replayed': True, 'new_fits': 0, 'GPU_used': False, 'official_test_opened': False}
    path = OUT / 'INDEPENDENT_VERIFY.json'
    if path.exists():
        assert q.read(path) == report
    else:
        q.save(path, report)
    print('FROZEN_NLI_CUDA_CPU_VERIFY_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'check', 'gpu-smoke', 'extract', 'score', 'verify'))
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        {'prepare': prepare, 'check': check, 'gpu-smoke': gpu_smoke, 'extract': extract,
         'score': score, 'verify': verify}[args.stage]()
