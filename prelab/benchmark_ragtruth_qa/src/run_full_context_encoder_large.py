"""Generic large control: reuse frozen v2 geometry and training, no test access.

Only training changes: generic ModernBERT-large dimensions/initial checkpoint
and AdamW foreach=False. The imported v2 file and its result directory are never
written. The local facade alters only that module's torch reference, not torch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
import torch
from threadpoolctl import threadpool_limits
import run_full_context_encoder_v2 as base

ROOT = base.ROOT
OUT = ROOT / 'results/full_context_encoder_large_v1'
MODEL = ROOT.parent / 'models/ModernBERT-large'
REVISION = '45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'
BASE_OUT = ROOT / 'results/full_context_encoder_v2'
BASE_SOURCE = Path(base.__file__)
q = base.q
_original_protocol = base.protocol
_original_source_files = base.source_files
_original_fp32_check = base.check_fp32_state


def adamw_scalar_loop(*args, **kwargs):
    assert 'foreach' not in kwargs or kwargs['foreach'] is False
    kwargs['foreach'] = False
    optimizer = torch.optim.AdamW(*args, **kwargs)
    assert all(g['foreach'] is False for g in optimizer.param_groups)
    return optimizer


class _OptimFacade:
    AdamW = staticmethod(adamw_scalar_loop)

    def __getattr__(self, name):
        return getattr(torch.optim, name)


class _TorchFacade:
    optim = _OptimFacade()

    def __getattr__(self, name):
        return getattr(torch, name)


def protocol():
    p = _original_protocol()
    p.update({
        'version': 'qa-full-context-modernbert-large-v1-bf16-forward-foreach-false',
        'architecture': 'Generic ModernBERT-large, 28 layers / 1024 hidden / 16 heads, two token logits; all395833346 classifier parameters trainable',
        'initialization': f'answerdotai/ModernBERT-large revision{REVISION}; generic masked-LM pretraining, not LettuceDetect/RAGTruth finetuned weights',
        'base_control': 'Full frozen ModernBERT-base v2: same3839 rendered inputs and original BPE character mapping, exact training weights, all6 epoch orders, seed20261005, all hyperparameters and calibration selection.',
        'implementation_differences': ['ModernBERT-large generic checkpoint and configuration', 'AdamW foreach=False; FP32 parameters, gradients and optimizer states unchanged'],
        'comparison_limits': 'Size and AdamW foreach implementation change together. QA only, full offline original answers and retrieved passages, same source-group loss and BPE evaluation. Not an exact LettuceDetect reproduction or generator-state probe.',
        'precision_limits': 'BF16 forward/checkpoint recomputation and FP32 optimization match base v2. foreach=False reduces temporary optimizer memory; neither numeric equality with foreach nor a localization improvement is assumed.',
        'smoke_comparison': 'Same longest979-token response15303 and synthetic zero objective as base v2, no gold objective. Complete backward/clip/AdamW step; record allocated/reserved memory, FP32 state and time. Failure is retained, no automatic fallback.',
        'v1_control': 'Compatibility filenames V1_REUSE_AGREEMENT and v1_reference fields in the reused runner refer to base v2 in this run. The base v2 source is directly reused, with output/model/protocol/optimizer bindings isolated in this wrapper.',
        'resource_policy': 'GPU smoke and actual process exit must be reported to root before any train command. No optimizer, quantization, precision, sequence truncation, batch or six-epoch budget fallback.',
    })
    p['training']['optimizer'] = 'AdamW(foreach=False)'
    p['training']['foreach'] = False
    return p


def source_files():
    d = q.read(MODEL / 'download_manifest.json')
    assert d['repo_id'] == 'answerdotai/ModernBERT-large' and d['revision'] == REVISION
    assert d['status'] == 'complete' and d['all_files_source_hash_matched']
    assert d['ragtruth_finetuned_checkpoint'] is False
    for rec in d['files']:
        assert q.sha(MODEL / rec['filename']) == rec['actual_sha256']
        assert rec['source_hash_match'] is True
    paths = _original_source_files()
    extra = [Path(__file__), Path(__file__).with_name('download_modernbert_large.py'),
             Path(__file__).with_name('download_checkpoint.py'),
             BASE_OUT / 'CPU_SELFCHECK.json', BASE_OUT / 'protocol.json',
             Path(torch.optim.AdamW.__module__.replace('.', '/'))]
    # Bind the actual local optimizer implementation, not a guessed relative path.
    import inspect
    extra[-1] = Path(inspect.getfile(torch.optim.AdamW))
    paths.update({str(p.resolve()): q.sha(p) for p in extra})
    return paths


def check_fp32_state(model, optimizer=None):
    report = _original_fp32_check(model, optimizer)
    if optimizer is not None:
        assert all(g['foreach'] is False for g in optimizer.param_groups)
    report['optimizer_foreach'] = False if optimizer is not None else None
    return report


def bind():
    base.OUT = OUT
    base.MODEL = MODEL
    base.V1_OUT = BASE_OUT
    base.V1_SOURCE = BASE_SOURCE
    base.protocol = protocol
    base.source_files = source_files
    base.check_fp32_state = check_fp32_state
    base.torch = _TorchFacade()


def cpu_test():
    assert not torch.cuda.is_initialized()
    base.cpu_test()
    config = base.ModernBertConfig.from_pretrained(MODEL, local_files_only=True)
    config.num_labels = 2
    config.reference_compile = False
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = base.ModernBertForTokenClassification(config)
    assert sum(p.numel() for p in model.parameters()) == 395833346
    assert (config.num_hidden_layers, config.hidden_size, config.num_attention_heads) == (28, 1024, 16)
    del model
    report = q.read(OUT / 'CPU_SELFCHECK.json')
    report.update(preparation_sha256=q.sha(OUT / 'preparation_complete.json'),
                  large_meta_parameters=395833346, large_dimensions=[28, 1024, 16],
                  optimizer_foreach_false_all_entrypoints=True,
                  original_torch_optimizer_unmodified=torch.optim.AdamW is not adamw_scalar_loop)
    assert report['passed'] and report['original_torch_optimizer_unmodified']
    assert not torch.cuda.is_initialized()
    q.save(OUT / 'CPU_SELFCHECK.json', report)


def gpu_smoke():
    cpu = q.read(OUT / 'CPU_SELFCHECK.json')
    assert cpu['passed'] and cpu['preparation_sha256'] == q.sha(OUT / 'preparation_complete.json')
    start = time.perf_counter()
    base.gpu_smoke()
    smoke = q.read(OUT / 'GPU_SELFCHECK.json')
    assert smoke['input_tokens'] == 979 and smoke['parameters'] == 395833346
    assert smoke['dtype_check']['optimizer_foreach'] is False
    q.save(OUT / 'GPU_RESOURCE_DETAIL.json', {
        'passed': True, 'smoke_sha256': q.sha(OUT / 'GPU_SELFCHECK.json'),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'cuda_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'cuda_device': torch.cuda.get_device_name(0),
        'whole_smoke_seconds_including_load_and_checks': time.perf_counter()-start,
        'optimizer_foreach': False, 'official_test_opened': False,
        'training_started': False, 'model_and_optimizer_deleted': True,
        'note': 'Actual process exit is reported separately; manifest creation does not release the GPU.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cpu-test', 'gpu-smoke', 'train'])
    args = parser.parse_args()
    bind()
    with threadpool_limits(limits=4):
        try:
            {'prepare': base.prepare, 'cpu-test': cpu_test,
             'gpu-smoke': gpu_smoke, 'train': base.train}[args.stage]()
        except Exception:
            OUT.mkdir(parents=True, exist_ok=True)
            report = {'traceback': traceback.format_exc(), 'stage': args.stage,
                      'no_automatic_fallback': True}
            if torch.cuda.is_initialized():
                report.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json', report)
            raise
